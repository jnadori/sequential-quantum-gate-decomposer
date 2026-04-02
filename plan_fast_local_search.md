# Plan: Diagonal Variance Approximation for Fast Local Search

## Problem

`local_search_acq` dominates candidate generation time. With default settings
(`local_search_fraction=0.5`, `candidates_per_iter=100`, `max_local_steps=10`),
up to 50 local searches run per iteration, each taking up to 10 greedy steps.

Each step's cost breaks down as:

| Phase | Per-neighbor cost | Count | Total per step |
|-------|------------------|-------|----------------|
| Neighbor generation + canonicalize | O(D^2) | D * n_tokens | ~O(D^3 * n_tokens) |
| Cross-kernel (SSK) | O(D^2 * order) per pair | n_nb * n_train | **dominant** |
| mu computation | O(n_train) | n_nb | O(n_nb * n_train) |
| Forward solve L v = Ks[:,j] | **O(n_train^2)** | n_nb_surviving | **expensive** |

With n_train=300, each forward solve costs ~90,000 ops. The mu-pruning (line 1642)
skips neighbors where `mu >= best_lcb`, but many survivors still need the full
O(n_train^2) solve to compute variance. At D=15, n_tokens=6, a typical step has
~50-80 unique neighbors after dedup, many of which pass mu-pruning.

## Goal

Replace the per-neighbor O(n_train^2) forward solve with a two-stage approach:

1. **Cheap diagonal approximation** — O(n_train) per neighbor, computed alongside mu
2. **Exact solve for top-k only** — O(k * n_train^2) with k small (e.g., 5)

This reduces the forward-solve cost from O(n_nb * n_train^2) to
O(n_nb * n_train + k * n_train^2), which at n_nb=70, n_train=300, k=5 is roughly
a **10x speedup** on the forward-solve phase.

## Files to Modify

- `squander/src-cpp/decomposition/N_Qubit_Decomposition_Surrogate.cpp`
  (only `local_search_acq`, around lines 1526-1673)
- `squander/src-cpp/decomposition/include/N_Qubit_Decomposition_Surrogate.h`
  (add `inv_L_diag` to GPRegressor, optionally)

## Background: The Diagonal Approximation

The exact posterior variance at a test point x is:

```
var(x) = scale - v^T v,    where L v = Ks[:,x]
```

The forward solve `L v = Ks[:,x]` computes:

```
v[0] = Ks[0,x] / L[0,0]
v[1] = (Ks[1,x] - L[1,0]*v[0]) / L[1,1]
v[2] = (Ks[2,x] - L[2,0]*v[0] - L[2,1]*v[1]) / L[2,2]
...
```

The diagonal approximation drops the off-diagonal terms of L:

```
v_approx[i] = Ks[i,x] / L[i,i]
var_approx(x) = scale - sum_i (Ks[i,x] / L[i,i])^2
```

This is equivalent to approximating K_train^{-1} with `diag(1/K_train_diag)` after
Cholesky rescaling. It captures the "how much does each training point individually
constrain this prediction" structure while ignoring inter-training-point correlations.

**Why this is good enough for ranking**: We only need the approximate variance to rank
neighbors for a top-k shortlist. The exact LCB is computed for the final k candidates.
Even if the diagonal approximation has ~30% error on absolute variance, it tends to
preserve the relative ordering well because the dominant variance contribution comes
from the diagonal terms (training points close to x in kernel space).

**Why this is conservative**: The diagonal approximation **overestimates** the variance
reduction (because it ignores positive off-diagonal covariance between training
points), which means it **underestimates** the posterior variance. This gives a
**lower** (more optimistic) LCB, so the top-k by approximate LCB is a **superset**
of what the exact top-k would be. We're unlikely to miss the true best neighbor.

## Detailed Implementation

### Step 1: Precompute inverse Cholesky diagonal

After GP fitting (in `GPRegressor::fit` and `GPRegressor::fit_incremental`), store the
inverse diagonal of L as a member vector. This avoids recomputing it per local search.

In `N_Qubit_Decomposition_Surrogate.h`, add to GPRegressor (around line 162):

```cpp
std::vector<double> inv_L_diag;  // 1.0 / L[i,i] for diagonal approximation
```

In `GPRegressor::fit` (line 602, after Cholesky succeeds), add:

```cpp
// Cache inverse Cholesky diagonal for fast variance approximation
inv_L_diag.resize(n);
for (int i = 0; i < n; ++i)
    inv_L_diag[i] = 1.0 / L_data[i * n + i];
```

In `GPRegressor::fit_incremental` (after the incremental Cholesky extension completes),
add the same block but for the full new size `n`:

```cpp
inv_L_diag.resize(n);
for (int i = 0; i < n; ++i)
    inv_L_diag[i] = 1.0 / L_data[i * n + i];
```

### Step 2: Rewrite the LCB evaluation in `local_search_acq`

Replace lines 1610-1662 (the current "Batch-evaluate LCB" block). The new logic:

```cpp
// Batch-evaluate LCB — compute cross-kernel only for training subset
int n_nb = static_cast<int>(neighbors.size());
std::vector<double> Ks(gp.n_train * n_nb);
cache.compute_cross_kernel_subset(neighbors, scale, gp.train_indices_, Ks.data());

// === Stage 1: Compute mu AND approximate variance for all neighbors ===
// Cost: O(n_nb * n_train) — same order as mu alone

int n_t = gp.n_train;
std::vector<double> mu(n_nb);
std::vector<double> var_approx(n_nb);

for (int j = 0; j < n_nb; ++j) {
    double mu_val = 0.0;
    double v2_sum = 0.0;
    for (int i = 0; i < n_t; ++i) {
        double ks_ij = Ks[i * n_nb + j];
        mu_val += ks_ij * gp.alpha_data[i];
        double v_approx_i = ks_ij * gp.inv_L_diag[i];
        v2_sum += v_approx_i * v_approx_i;
    }
    mu[j] = mu_val;
    var_approx[j] = std::max(scale - v2_sum, 0.0);
}

// Compute approximate LCB for all neighbors
std::vector<double> lcb_approx(n_nb);
for (int j = 0; j < n_nb; ++j)
    lcb_approx[j] = mu[j] - kappa * std::sqrt(var_approx[j]);

// === Stage 2: Select top-k by approximate LCB for exact evaluation ===
constexpr int k_exact = 5;
int k = std::min(k_exact, n_nb);

// Partial sort to find top-k (don't need full sort)
std::vector<int> order(n_nb);
std::iota(order.begin(), order.end(), 0);
std::partial_sort(order.begin(), order.begin() + k, order.end(),
                  [&](int a, int b) { return lcb_approx[a] < lcb_approx[b]; });

// === Stage 3: Exact forward solve for top-k only ===
// Cost: O(k * n_train^2)

double best_nb_lcb = std::numeric_limits<double>::infinity();
int best_nb_idx = -1;
std::vector<double> v_col(n_t);

for (int rank = 0; rank < k; ++rank) {
    int j = order[rank];

    // Prune: if mu[j] >= best_nb_lcb (exact), no variance can help
    if (mu[j] >= best_nb_lcb) continue;

    // Exact forward solve L v = Ks[:,j]
    for (int i = 0; i < n_t; ++i)
        v_col[i] = Ks[i * n_nb + j];
    for (int i = 0; i < n_t; ++i) {
        for (int ii = 0; ii < i; ++ii)
            v_col[i] -= gp.L_data[i * n_t + ii] * v_col[ii];
        v_col[i] /= gp.L_data[i * n_t + i];
    }

    double sv2 = 0;
    for (int i = 0; i < n_t; ++i)
        sv2 += v_col[i] * v_col[i];
    double std_val = std::sqrt(std::max(scale - sv2, 0.0));
    double lcb_val = lcb(mu[j], std_val);
    if (lcb_val < best_nb_lcb) {
        best_nb_lcb = lcb_val;
        best_nb_idx = j;
    }
}

if (best_nb_lcb >= best_lcb) break;  // local optimum

current = neighbors[best_nb_idx].copy();
best_lcb = best_nb_lcb;
steps_taken++;
```

### Step 3: Also update the initial LCB evaluation

Lines 1533-1551 compute the exact LCB at the starting point. This is fine — it's a
single forward solve and only happens once per local search. No change needed here.

### Step 4: Choose k_exact

The constant `k_exact = 5` is a tuning parameter. It should be:
- Large enough that the true best neighbor is almost always in the top-k approximate
- Small enough that the O(k * n_train^2) cost is negligible

**Recommendation**: Use `k_exact = 5` as default. With ~70 neighbors, this means we
do exact solves for the top 7% only. The diagonal approximation is conservative
(overestimates variance reduction), so the true best is very likely in the top-5
approximate.

If you want a config parameter for tuning, add `local_search_topk` to the config map
with default 5. But a hardcoded constant is fine to start — this is an internal
optimization detail, not a user-facing knob.

### Step 5: Fuse mu and variance-proxy computation into a single loop

The implementation in Step 2 already does this — the inner loop computes both mu and
the diagonal variance proxy in a single pass over `Ks[:,j]`. This is important for
cache efficiency: `Ks` is accessed column-by-column (stride n_nb), so reading each
element once for both computations is much better than two separate passes.

The fused loop body is:

```cpp
double ks_ij = Ks[i * n_nb + j];
mu_val += ks_ij * gp.alpha_data[i];
double v_approx_i = ks_ij * gp.inv_L_diag[i];
v2_sum += v_approx_i * v_approx_i;
```

Four multiply-adds per element, all from contiguous (or stride-n_nb) memory.
The compiler should vectorize this well.

### Step 6 (Optional): Batch the top-k forward solves via LAPACKE

Instead of k manual forward-substitution loops, pack the top-k columns of Ks into a
contiguous matrix and call `LAPACKE_dtrtrs` once:

```cpp
// Pack top-k columns into contiguous buffer (n_t x k, column-major for LAPACK)
std::vector<double> Ks_topk(n_t * k);
for (int rank = 0; rank < k; ++rank) {
    int j = order[rank];
    for (int i = 0; i < n_t; ++i)
        Ks_topk[rank * n_t + i] = Ks[i * n_nb + j];  // column-major
}

// Single batched triangular solve: L V = Ks_topk
// Note: LAPACK column-major, our L is stored row-major.
// Use transpose: L_row_major treated as U^T in column-major.
LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N',
               n_t, k, gp.L_data.data(), n_t,
               Ks_topk.data(), n_t);

// Extract variance from solved columns
for (int rank = 0; rank < k; ++rank) {
    int j = order[rank];
    if (mu[j] >= best_nb_lcb) continue;
    double sv2 = 0;
    for (int i = 0; i < n_t; ++i)
        sv2 += Ks_topk[rank * n_t + i] * Ks_topk[rank * n_t + i];
    double std_val = std::sqrt(std::max(scale - sv2, 0.0));
    double lcb_val = lcb(mu[j], std_val);
    if (lcb_val < best_nb_lcb) {
        best_nb_lcb = lcb_val;
        best_nb_idx = j;
    }
}
```

**Important caveat**: `L_data` is stored row-major (LAPACK_ROW_MAJOR was used in
`GPRegressor::fit`). A row-major lower-triangular matrix L is the same as a
column-major upper-triangular matrix L^T. So in column-major mode, call with
`uplo='U', trans='T'` to solve L v = b. Verify this is correct by checking a
small example.

This step is optional — the manual loop is already fast for k=5 and n_t=300. The
LAPACKE call mainly helps if k is larger or if you want cleaner code. Skip this
if you want to keep the implementation simple.

## Cost Analysis

Per local search step with n_nb=70 neighbors and n_train=300:

| Phase | Before | After |
|-------|--------|-------|
| Cross-kernel (SSK) | 70 * 300 pairs | unchanged |
| mu computation | 70 * 300 = 21K ops | fused into approx (free) |
| Forward solve | ~40 * 300^2 = 3.6M ops | 5 * 300^2 = 450K ops |
| Approximate variance | — | 70 * 300 = 21K ops (fused with mu) |
| partial_sort | — | O(n_nb * log(k)) ≈ negligible |
| **Total solve phase** | **~3.6M** | **~0.5M** |

The forward solve phase gets ~7x faster. The cross-kernel (SSK) computation is
unchanged and may still dominate at large D — this plan only targets the solve phase.
But the solve phase is the part that scales as n_train^2, so it becomes increasingly
important as gp_max_train grows.

With 50 local searches × 5 steps average × 7x speedup on the solve phase, the total
wall-clock savings depends on what fraction of local search time is solve vs SSK. A
rough estimate: if solve is 40% of local search time, overall candidate generation
speeds up by ~25%.

## Testing

1. **Correctness test**: Run local_search_acq with both the old (exact for all) and new
   (approximate + top-k exact) implementations on the same inputs. Verify that:
   - The returned circuit is identical in >90% of cases
   - When different, the exact LCB of the new result is within 5% of the old result's
     exact LCB
   - The step counts are similar (within ±1 on average)

2. **Approximation quality test**: For a set of ~100 random neighbors, compute both
   exact and approximate variance. Report the rank correlation (Spearman's rho).
   Expect rho > 0.85. Also check that the true best neighbor (by exact LCB) is in the
   top-5 by approximate LCB in >95% of cases.

3. **Performance test**: Time `generate_candidates` with and without the change on a
   realistic problem (3-4 qubits, D=10-15, n_train=200-300). Measure wall-clock
   speedup of the candidate generation phase specifically (use the existing timing
   code that reports `t_cand`).

4. **Integration test**: Full `search_over_D_range` run to verify that final solution
   quality is unchanged. The search is stochastic, so run 5 trials with each
   implementation on the same unitary and compare distributions of final best_score.

## Notes

- The `inv_L_diag` vector must be recomputed whenever L changes (i.e., after every
  `fit` or `fit_incremental` call). This is O(n_train) and negligible.

- The diagonal approximation is related to the Nyström approximation with diagonal
  correction. For GP practitioners: this is the "FITC-like" variance but using only
  the Cholesky diagonal instead of inducing points.

- If the SSK cross-kernel computation turns out to be the true bottleneck (not the
  forward solve), this plan won't help much. In that case, consider the incremental
  SSK idea (exploiting single-position differences between neighbors) as a separate
  optimization.

- The `k_exact` parameter trades off speed vs approximation quality. At k=1 you're
  fully trusting the diagonal approximation. At k=n_nb you're back to the original
  behavior. k=5 is a pragmatic middle ground.

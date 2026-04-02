# Plan: Rank-Based Target Transform for GP Training

## Problem

At large D, most evaluated circuits have similar (bad) scores. The current transform
pipeline is:

```
raw score → log10(max(score, dynamic_floor)) → zero-mean unit-variance normalization
```

This compresses the training targets into a narrow band. When the GP fits to these
nearly-identical values, it learns an almost constant function (the posterior mean is
flat, the posterior variance is uniform). Thompson sampling then degenerates to random
selection because all posterior draws look the same.

The collapse detection at line 2659 (`scale/noise < 0.1`) confirms this: the GP is
signaling that the kernel can't explain any variance in the targets.

## Goal

Replace the log10 + z-score transform with a **rank-based copula transform** that
maximally spreads the training signal regardless of the raw score distribution:

```
raw score → rank among training set → uniform quantile → inverse normal CDF (probit)
```

This guarantees the transformed targets are standard-normal with maximal entropy,
giving the GP the best possible signal to work with even when raw scores are clustered.

## Files to Modify

- `squander/src-cpp/decomposition/N_Qubit_Decomposition_Surrogate.cpp`

No header changes needed — this is purely an internal change to the training data
preprocessing inside `run_window_search`.

## Detailed Implementation

### Step 1: Add inverse normal CDF utility

Add a static helper function near the top of the file (before `run_window_search`,
around line 870 or in the anonymous namespace area). This computes the probit function
(inverse of the standard normal CDF) using the rational approximation from
Abramowitz & Stegun (formula 26.2.23), which is accurate to ~4.5e-4:

```cpp
/// Inverse standard normal CDF (probit function).
/// Input p must be in (0, 1). Uses Abramowitz & Stegun 26.2.23.
static double probit(double p) {
    // Symmetry: probit(p) = -probit(1-p)
    bool flip = (p > 0.5);
    if (flip) p = 1.0 - p;

    // Rational approximation for probit(p) when 0 < p <= 0.5
    double t = std::sqrt(-2.0 * std::log(p));
    // Coefficients from A&S 26.2.23
    constexpr double c0 = 2.515517;
    constexpr double c1 = 0.802853;
    constexpr double c2 = 0.010328;
    constexpr double d1 = 1.432788;
    constexpr double d2 = 0.189269;
    constexpr double d3 = 0.001308;

    double val = t - (c0 + c1 * t + c2 * t * t) /
                     (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    return flip ? val : -val;
}
```

This is a well-known approximation. No external library needed.

### Step 2: Replace the target transform block

In `run_window_search`, find the current transform block (lines 2600-2621):

```cpp
// Log10 transform + normalize
std::vector<double> log_y(n_x);
double min_nonzero = std::numeric_limits<double>::infinity();
for (int i = 0; i < n_x; ++i)
    if (train_y[i] > 0) min_nonzero = std::min(min_nonzero, train_y[i]);
double dynamic_floor = (min_nonzero < std::numeric_limits<double>::infinity()) ?
    min_nonzero * 0.01 : tolerance * 0.01;

double lmu = 0, lsig = 0;
for (int i = 0; i < n_x; ++i) {
    log_y[i] = std::log10(std::max(train_y[i], dynamic_floor));
    lmu += log_y[i];
}
lmu /= n_x;
for (int i = 0; i < n_x; ++i)
    lsig += (log_y[i] - lmu) * (log_y[i] - lmu);
lsig = std::sqrt(lsig / n_x);
if (lsig < 1e-6) lsig = 1e-6;

std::vector<double> log_y_norm(n_x);
for (int i = 0; i < n_x; ++i)
    log_y_norm[i] = (log_y[i] - lmu) / lsig;
```

Replace with:

```cpp
// Rank-based copula transform: rank → uniform quantile → probit
// This maximally spreads the training signal regardless of raw score distribution.
std::vector<double> log_y_norm(n_x);
{
    // Step A: Compute ranks (lower score = lower rank = better).
    // Use average ranks for ties to avoid arbitrary ordering.
    std::vector<int> order(n_x);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&train_y](int a, int b) { return train_y[a] < train_y[b]; });

    std::vector<double> ranks(n_x);
    int i = 0;
    while (i < n_x) {
        // Find run of tied values
        int j = i + 1;
        while (j < n_x && train_y[order[j]] == train_y[order[i]])
            ++j;
        // Assign average rank to all tied values
        double avg_rank = 0.5 * (i + j - 1);  // 0-based average
        for (int k = i; k < j; ++k)
            ranks[order[k]] = avg_rank;
        i = j;
    }

    // Step B: Map ranks to uniform quantiles in (0, 1).
    // Use (rank + 0.5) / n to avoid 0 and 1 (which would give ±inf in probit).
    // Step C: Apply probit (inverse normal CDF).
    for (int i = 0; i < n_x; ++i) {
        double u = (ranks[i] + 0.5) / n_x;
        log_y_norm[i] = probit(u);
    }
}
```

**Important**: The variable is still called `log_y_norm` to minimize downstream changes.
Everything after this block (GP fitting, hyperparameter optimization, prediction)
consumes `log_y_norm` and does not need to change.

### Step 3: Remove the inverse transform (if any)

Check whether `lmu` and `lsig` are used later in `run_window_search` to invert the
transform (e.g., to convert GP predictions back to raw scores for logging). Search for
all uses of `lmu` and `lsig` in the function.

If they are used only for the transform itself (which is the case based on the code
read — the GP predictions are compared in normalized space, and raw `best_score` is
tracked separately from the GP), then removing them is safe.

If there is any inverse-transform code like `raw = 10^(pred * lsig + lmu)`, remove it
and replace with a comment noting that the rank transform is not invertible in
closed form and raw scores should be used directly.

### Step 4: Update collapse detection

The collapse detection block (lines 2659-2667):

```cpp
double opt_scale = gp.get_scale();
if (opt_scale / std::max(gp.noise, 1e-12) < 0.1) {
    gp.log_scale = 0.0;
    gp.noise = 1e-2;
    gp.fit(ssk_cache, train_indices.data(), n_x, log_y_norm.data());
    prev_log_scale = gp.log_scale;
    prev_noise = gp.noise;
}
```

With the rank transform, the targets are guaranteed to have unit variance and zero mean
(approximately, by construction). The collapse scenario (scale << noise) should be
much rarer. However, **keep this block as-is** — it's a safety net that costs nothing
and the condition may still trigger if the SSK kernel is truly degenerate at very
large D. No change needed here.

### Step 5: Adjust hyperparameter bounds (optional but recommended)

The rank transform produces targets with unit variance by construction, so the GP's
`log_scale` should settle near 0 (scale ≈ 1.0). The current bounds of [-3, 3] are
fine but could be tightened to [-2, 2] for faster convergence. This is in
`run_window_search` where `scale_bounds` is defined:

```cpp
// Current:
std::pair<double,double> scale_bounds(-3.0, 3.0);
// Could tighten to (optional):
std::pair<double,double> scale_bounds(-2.0, 2.0);
```

This is optional — the algorithm is correct either way, the tighter bounds just reduce
hyperparameter search time slightly. If you're unsure, leave the bounds as-is.

### Step 6: Handle n_x == 1 edge case

When there's only one training point, the rank transform produces `probit(0.5) = 0.0`,
which is a single zero target. This is fine — the GP will predict the mean everywhere
with maximum uncertainty. The current log-transform has the same behavior (single point
→ zero after normalization). No special handling needed.

### Step 7: Handle incremental fit compatibility

The incremental Cholesky update (`fit_incremental`) requires that the first `old_n`
training targets are unchanged. With the rank transform, adding new training points
changes the ranks (and therefore the transformed values) of ALL existing points. This
means **incremental fit will almost never trigger** — the prefix check will fail
because the training targets change.

This is acceptable. The rank transform improves GP signal quality, which is more
valuable than incremental Cholesky savings. The full refit is O(n^3) with n ≤ 300
(gp_max_train), which takes ~1ms.

If you want to preserve incremental fit compatibility in the future, you could use
a two-stage approach: rank-transform a fixed "anchor" set, then incrementally add
new points with their ranks computed relative to the anchor. But this adds complexity
for minimal gain — skip it for now.

## Why This Works

Consider a typical large-D scenario where 300 training circuits have scores like:

```
[0.85, 0.86, 0.86, 0.87, 0.87, 0.87, 0.88, ...]  (clustered around 0.87)
```

**Current log10 + z-score:**
```
log10 → [-0.071, -0.066, -0.066, -0.060, ...]  (range: 0.011)
z-score → [-1.2, -0.8, -0.8, -0.3, ...]        (some spread, but noisy)
```
The log transform barely helps because the scores are already in a narrow band.
The z-score amplifies noise.

**Rank copula transform:**
```
ranks → [0, 1, 2, 3, 4, 5, 6, ...]
uniform → [0.002, 0.005, 0.008, 0.012, ...]
probit → [-2.88, -2.58, -2.41, -2.26, ...]     (well-separated, Gaussian)
```
Every circuit gets a distinct, well-separated target value. The GP can now learn which
structural features correlate with rank, even when the raw scores are nearly identical.

## Testing

1. **Unit test for `probit`**: Verify `probit(0.5) == 0.0`, `probit(0.025) ≈ -1.96`,
   `probit(0.975) ≈ 1.96`, `probit(0.001) ≈ -3.09`. Compare against scipy's
   `norm.ppf` if available.

2. **Transform correctness**: Generate 100 random scores in [0.8, 0.9], apply the rank
   transform, verify the output is approximately N(0,1): mean ≈ 0, std ≈ 1, sorted in
   the same order as the input scores.

3. **Tie handling**: Generate scores with many ties (e.g., 50 copies of 0.85, 50 copies
   of 0.90). Verify tied scores get the same transformed value and the output is still
   approximately N(0,1).

4. **Integration test**: Run `search_over_D_range` on a known 3-qubit unitary at D=8-12.
   Compare score trajectories (best_score vs iteration) between the old log10 transform
   and the new rank transform. The rank transform should show faster improvement at
   larger D values (D ≥ 10) where score clustering is worst.

5. **Regression test**: Verify small-D performance (D=3-5) is not degraded. At small D,
   scores are well-separated so both transforms should perform similarly.

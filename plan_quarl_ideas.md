# Plan: Quarl-Inspired Improvements to SurSearch

Based on the Quarl paper (Li et al., 2307.10120) — a learning-based quantum circuit optimizer
that uses RL with GNNs to apply equivalence-preserving circuit transformations.

Target file: `squander/src-cpp/decomposition/N_Qubit_Decomposition_Surrogate.cpp`
Header: `squander/src-cpp/decomposition/include/N_Qubit_Decomposition_Surrogate.h`

---

## 1. Trajectory-Based Exploration (Cost-Increasing Steps)

**Paper reference:** Sec. 3 (pp. 2-3), Fig. 2b, Algorithm 1 (line 18, α threshold)

**Problem:** SurSearch generates and evaluates each candidate independently. It cannot traverse
"valleys" in the optimization landscape where a sequence of mutations must pass through worse
intermediate solutions to reach a better one. Quarl shows that >50% of circuits need 4+
transformations to improve, and allowing cost-increasing steps reduces gate count from 46 to 36
on barenco_tof_3.

**Design:**
- Add a `trajectory_explore` method that starts from a good skeleton and applies a chain of
  mutations (configurable horizon T, e.g. 3-8 steps)
- Accept intermediate steps even when score worsens, up to `alpha * initial_score` (Quarl uses
  α=1.2)
- Register all intermediate circuits into the GP training set — even "bad" circuits provide
  valuable landscape information
- Add a NOP action (Quarl Algorithm 1, line 18): if no mutation improves or stays within
  threshold, terminate the trajectory early
- Integrate into `run_window_search`: after Thompson sampling selects and evaluates candidates,
  run short trajectories from the top-k evaluated circuits

**New config keys:**
- `trajectory_horizon` (int, default 5): max steps per trajectory
- `trajectory_alpha` (double, default 1.2): cost-increase threshold
- `n_trajectories_per_iter` (int, default 3): trajectories launched per search iteration
- `trajectory_fraction` (double, default 0.3): fraction of iteration budget spent on trajectories

**New methods:**
```cpp
struct TrajectoryStep {
    GrayCode circuit;
    double score;
    Matrix_real params;
};

std::vector<TrajectoryStep> trajectory_explore(
    const GrayCode& start, double start_score,
    double alpha, int horizon,
    SSKCache& cache, GrayCodeSet& seen,
    std::mt19937& local_gen);
```

**Integration point:** `run_window_search`, after the decompose phase (line ~2893), before
patience check. Run trajectories from the best newly-evaluated circuits.

**Key insight from Quarl:** The trajectory return Rt captures multi-step value that single-step
evaluation misses. Even without RL, applying sequential mutations and accepting temporary
regressions lets the search escape local optima.

---

## 2. Position-Aware / Locality-Guided Mutations

**Paper reference:** Sec. 4.1-4.2 (pp. 8-9), Eq. 4-5, Fig. 4

**Problem:** `mutate_point` (line 1384) picks positions uniformly at random. Many mutations are
wasted on positions where the current token is already near-optimal. Quarl's GNN computes
per-gate value estimates and uses temperature softmax to focus on high-value (high-improvement-
potential) positions.

**Design:**
- Add a `position_sensitivity` method that, for a given circuit, estimates which positions are
  most improvable using the GP
- For each position p, compute the best predicted score (GP mu) across all alternative tokens
  at that position. The position with the largest predicted improvement is most promising.
- Use temperature softmax (Quarl Eq. 4) to convert position values into a sampling distribution
- Temperature adapts to circuit size via Quarl Eq. 5: `t = 1/ln(λ(D-1)/(1-λ))` where λ∈(0,1)
  controls exploration
- Use this distribution in `mutate_point` and `mutate_block` instead of uniform position
  sampling

**New config keys:**
- `position_guided_fraction` (double, default 0.5): fraction of point mutations that use
  position guidance (rest remain uniform for diversity)
- `position_lambda` (double, default 0.9): λ parameter for temperature computation

**New methods:**
```cpp
// Compute position improvement potential using GP predictions
std::vector<double> position_sensitivity(
    const GrayCode& seq, SSKCache& cache, GPRegressor& gp, double scale);

// Position-guided mutation: sample position from softmax distribution
GrayCode mutate_point_guided(
    const GrayCode& seq, SSKCache& cache, GPRegressor& gp, double scale);
```

**Integration point:** `generate_candidates` (line 1680). When generating evolutionary
candidates, use `mutate_point_guided` for a fraction of point mutations. Pass GP reference
through to mutation operators.

**Performance note:** Computing GP mu for D*n_tokens trial circuits per call could be expensive.
Mitigate by:
- Only computing mu (skip variance — O(n_train) per candidate vs O(n_train^2) for variance)
- Caching the cross-kernel computation across positions that share most of their sequence
- Only doing position-guided mutations for the top tournament-selected parents, not all

---

## 3. Adaptive Kappa (Exploration-Exploitation Scheduling)

**Paper reference:** Eq. 5 (p. 9), Sec. 5.2 stop conditions

**Problem:** `kappa` is fixed throughout the window search (line 954). The only adaptation is
swapping to `rb_kappa` during narrowing. This means the exploration-exploitation tradeoff is
static, but the optimal balance changes as the search progresses.

**Design:**
- Start with high kappa (exploration) and decay toward low kappa (exploitation) over the
  window's iteration budget
- On stagnation detection, temporarily boost kappa to escape local optima
- Schedule: `effective_kappa = kappa * (base_decay + stagnation_boost)`
  - `base_decay = 1.0 - decay_rate * (itr / window_max_iters)` — linear decay
  - `stagnation_boost = boost_factor * max(0, iters_since_improvement - patience/2) / (patience/2)`
- Pass `effective_kappa` to `local_search_acq` and use it in Thompson sampling (scale the
  posterior samples)

**New config keys:**
- `kappa_decay_rate` (double, default 0.5): fraction of kappa to decay over full window
- `kappa_stagnation_boost` (double, default 0.5): max boost multiplier on stagnation
- `adaptive_kappa` (bool, default true): enable/disable

**Implementation:** Modify `run_window_search` at line ~2470 (iteration loop start):
```cpp
double effective_kappa = kappa;
if (adaptive_kappa) {
    double progress = static_cast<double>(itr) / window_max_iters;
    double base = 1.0 - kappa_decay_rate * progress;
    double stagnation = 0.0;
    if (iters_since_improvement > window_patience / 2) {
        stagnation = kappa_stagnation_boost *
            std::min(1.0, static_cast<double>(iters_since_improvement - window_patience/2)
                         / (window_patience / 2));
    }
    effective_kappa = kappa * (base + stagnation);
}
// Temporarily set this->kappa = effective_kappa for LCB calls
```

**Minimal code change:** ~15 lines in `run_window_search`, plus config parsing.

---

## 4. Stratified Circuit Buffer with Cost-Biased Sampling

**Paper reference:** Sec. 5.2, "Initial circuit buffer"

**Problem:** Tournament selection treats the entire population equally. As the population grows
(easily 1000+ circuits), tournament selection becomes noisy and doesn't prioritize the most
informative parents. Quarl maintains a buffer keyed by cost and samples with a distribution
biased toward lower cost.

**Design:**
- Maintain a `CircuitBuffer` that organizes circuits by D-value, sorted by score within each D
- Cap each D-bucket at `buffer_max_per_D` (default 50) — prune worst circuits when full
- For parent selection in `generate_candidates`, use cost-biased sampling:
  - First sample a D value (uniform within window, or biased toward D values with better scores)
  - Then sample a circuit from that D's bucket with probability proportional to `exp(-score/tau)`
    where tau is a temperature parameter
- This replaces tournament selection for evolutionary candidate generation

**New data structure:**
```cpp
struct CircuitBuffer {
    // D -> vector of (index_into_X, score), sorted by score ascending
    std::map<int, std::vector<std::pair<int, double>>> by_D;
    int max_per_D;

    void add(int idx, int D, double score);
    int sample_parent(std::mt19937& gen, int win_lo, int win_hi,
                      const std::vector<double>& y);
};
```

**Integration point:** `generate_candidates` (line 1680), replace `tournament_select` lambda
with `buffer.sample_parent(...)`. Update the buffer in the decompose phase of
`run_window_search` when new circuits are registered.

---

## 5. Position Coverage Masks (Hard/Soft)

**Paper reference:** Sec. 5.4, "Policy-guided search" — hard masks, soft masks

**Problem:** `local_search_acq` (line 1526) and `generate_candidates` have no mechanism to
ensure all positions in a circuit are explored. The search can fixate on mutating the same
positions repeatedly while ignoring others.

**Design:**
- Track a `position_visit_count` vector per parent circuit during local search
- **Hard mask:** After mutating position p, mark it as visited. Don't revisit it until all
  positions have been visited at least once.
- **Soft mask:** After all positions have one visit, clear the counts and start over. This
  ensures uniform coverage over time without being too restrictive.
- Apply masks in `local_search_acq` when generating neighbors: skip positions that are
  hard-masked

**Implementation in `local_search_acq`:**
```cpp
std::vector<int> visit_count(D, 0);

for (int step = 0; step < max_local_steps; ++step) {
    // Generate substitution neighbors only at unvisited positions
    for (int pos = 0; pos < D; ++pos) {
        if (visit_count[pos] > 0 && *std::min_element(...) == 0)
            continue;  // hard mask: skip visited positions while unvisited remain
        // ... generate neighbors at pos
    }
    // Mark the position that was selected
    visit_count[best_pos]++;
    // Soft mask reset: if all visited, clear
    if (*std::min_element(visit_count.begin(), visit_count.end()) > 0)
        std::fill(visit_count.begin(), visit_count.end(), 0);
}
```

**Benefit:** Guarantees O(D) coverage per D local search steps, preventing the search from
repeatedly optimizing the same region of the circuit.

---

## 6. Parallel Exploitation Bursts

**Paper reference:** Sec. 5.4, Fig. 6 — parallel fine-tuning + policy-guided search

**Problem:** The search loop in `run_window_search` alternates between GP fitting, candidate
generation, and decomposition. There's no mechanism for intensive local exploitation around the
current best solution running in parallel with the broader GP-guided exploration.

**Design:**
- Every `exploitation_interval` iterations (e.g. every 5), launch an "exploitation burst"
  that runs in parallel with the next GP-guided iteration
- The burst does exhaustive 1-neighborhood search around `best_circuit`:
  - All single-position substitutions (D * (n_tokens-1) candidates)
  - All single-position grow/shrink variants
  - Pre-screen with GP mu, only decompose top-m
- If the burst finds an improvement, feed it back into the main search state (X, y, cache)
- Use TBB to run the burst concurrently with the main search iteration

**New config keys:**
- `exploitation_interval` (int, default 5): iterations between bursts
- `exploitation_top_m` (int, default 10): max candidates to decompose per burst

**Architecture:**
```
Main thread:           GP fit -> candidates -> Thompson -> decompose
                       |                                      |
Exploitation thread:   exhaustive_local(best) -----> merge results
```

**Integration point:** `run_window_search`, wrap the main iteration body and exploitation burst
in a TBB task group. After both complete, merge results.

**Complexity note:** This is the highest-effort change. Can be deferred or simplified to a
non-parallel version that just runs exploitation bursts sequentially every N iterations.

---

## Implementation Order

1. **Adaptive kappa** (item 3) — lowest effort, immediate impact, ~15 lines
2. **Position coverage masks** (item 5) — low effort, improves local search, ~30 lines
3. **Stratified circuit buffer** (item 4) — medium effort, better parent selection, ~80 lines
4. **Trajectory exploration** (item 1) — medium effort, highest potential impact, ~120 lines
5. **Position-guided mutations** (item 2) — medium effort, focuses evaluations, ~100 lines
6. **Parallel exploitation bursts** (item 6) — highest effort, moderate impact, ~150 lines

Items 1-3 can be implemented independently. Item 4 (trajectories) benefits from item 3
(adaptive kappa) being in place. Item 5 (position-guided) benefits from item 2 (coverage masks)
being in place. Item 6 is independent but depends on having a stable search loop.

---

## Testing Strategy

- Use small benchmark circuits (barenco_tof_3, tof_3, mod5_4) for fast iteration
- Compare against current SurSearch baseline with identical config except the new features
- Metrics: best score found, number of evaluations to solution, wall-clock time
- Each feature should be independently toggleable via config keys for A/B comparison

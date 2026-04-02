# Surrogate-Assisted Search Optimization Plan
**Inspiration:** GPyTorch & BoTorch (Advanced Bayesian Optimization & Exact GPs)

This document outlines a phased plan to integrate advanced Gaussian Process (GP) techniques into the C++ `N_Qubit_Decomposition_Surrogate` search algorithm. The goal is to scale the surrogate model, reduce \( O(N^3) \) bottlenecks, and improve batch candidate acquisition.

---

## Phase 1: Performance & Bottleneck Elimination
*Focus: Replace \( O(N^3) \) bottlenecks in the critical loop with \( O(N^2) \) or better alternatives.*

### 1.1 Analytical Gradients for Hyperparameter Optimization
*   **Current State:** `gp_hyper_opt_combined` uses finite differences for gradients, requiring two full Cholesky decompositions \( O(N^3) \) per iteration.
*   **Action Item:** Implement exact analytical gradients for the Marginal Log Likelihood (MLL).
    *   Compute \( K^{-1} = L^{-T} L^{-1} \).
    *   Scale gradient: \( \frac{\partial K}{\partial \sigma_{scale}^2} = K_{norm} \)
    *   Noise gradient: \( \frac{\partial K}{\partial \sigma_{noise}^2} = I \)
    *   Apply trace: \( \nabla_\theta LML = \frac{1}{2} \text{Tr}\left( (\alpha \alpha^T - K^{-1}) \frac{\partial K}{\partial \theta} \right) \).
*   **Expected Impact:** Massively speeds up GP fitting (`GPRegressor::optimize_hyperparameters`), allowing for larger `gp_max_train` limits.

### 1.2 LOVE (Lanczos Variance Estimates) for Fast Predictions
*   **Current State:** Exact predictive variance in `local_search_acq` and `GPRegressor::predict` calls `LAPACKE_dtrtrs` to solve \( L v = K_s \) for every candidate.
*   **Action Item:** Precompute a low-rank Lanczos decomposition \( R \) such that \( K_{XX}^{-1} \approx R R^T \).
*   **Expected Impact:** Reduces variance prediction to a fast vector-matrix multiplication \( \text{Var}(f^*) \approx \sigma_{scale}^2 - ||R^T K_s||^2 \), massively accelerating the acquisition phase during LCB/EI evaluations.

---

## Phase 2: Acquisition & Search Strategy Improvements
*Focus: Replace hardcoded heuristics with native probabilistic approaches.*

### 2.1 Fantasization for Batch Acquisition (Replacing Diversity Threshold)
*   **Current State:** Batch candidate selection relies on a hard SSK distance threshold (`diversity_thresh`) to reject structurally similar circuits.
*   **Action Item:** Implement BoTorch-style Fantasization.
    *   Select the best candidate via the acquisition function (LCB/EI).
    *   "Fantasize" its score by assuming it matches the GP's predicted mean (`mu`).
    *   Use the existing `fit_incremental` method to temporally add this fantasy point to the GP.
    *   The posterior variance naturally shrinks around the selected point, organically driving the next LCB/EI selection toward a diverse region.
*   **Expected Impact:** Removes arbitrary diversity thresholds and creates a mathematically principled approach to batch generation.

---

## Phase 3: Multi-Fidelity Capabilities
*Focus: Stop discarding valuable learned structures across depth searches.*

### 3.1 Multi-Fidelity GP across Circuit Depths
*   **Current State:** In `search_over_D_range` and `compress_over_D_range`, a fresh `SSKCache` and `GPRegressor` are instantiated for every Depth level (`D`). Information learned at `D+1` is wiped.
*   **Action Item:** Treat circuit depth as a continuous fidelity parameter.
    *   Modify the kernel to include a depth component: \( K_{total}(x, y) = K_{SSK}(x, y) \times K_{RBF}(\text{depth}_x, \text{depth}_y) \).
    *   Preserve the `SSKCache` and GP state across iterations.
*   **Expected Impact:** The surrogate can predict promising structures at `D-1` based on patterns that were highly successful at `D` and `D+1`.

---

## Phase 4: Extreme Scaling (Future / Optional)
*Focus: Scaling the GP to thousands of evaluated circuits without memory or CPU explosions.*

### 4.1 Sparse GPs (SVGP) / Inducing Points
*   **Current State:** Pivoted Cholesky is used to hard-truncate the training set to `gp_max_train = 300`. Excluded points are entirely forgotten.
*   **Action Item:** Implement a Fully Independent Training Conditional (FITC) or Nyström approximation. Treat the 300 points as "Inducing Points" \( U \) that summarize the whole dataset \( X \).
*   **Expected Impact:** The GP retains global knowledge of *all* searched circuits while keeping the computational footprint constrained to the size of the inducing points.

### 4.2 LinearOperator / Matrix-Free Inference (PCG)
*   **Current State:** Explicit \( N \times N \) covariance matrix instantiation and `LAPACKE_dposv` (Cholesky) factorization.
*   **Action Item:** Abstract the kernel matrix into a `LinearOperator` class that only defines a matrix-vector multiplication function (`matvec(v)`). Replace Cholesky with Preconditioned Conjugate Gradients (PCG).
*   **Expected Impact:** Lowers memory footprint to \( O(N) \) and avoids the \( O(N^3) \) Cholesky entirely, allowing scaling to 10k+ circuits.

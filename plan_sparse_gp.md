# Plan: Sparse GP with Inducing Points (Idea D)

## Problem
Current GP uses full Cholesky on up to 300 training points (gp_max_train).
Pivoted Cholesky caps the training set but throws away information from remaining data.
GP prediction requires cross-kernel against all 300 training points.

## Approach
Replace full GP with FITC or VFE sparse GP using m=50-100 inducing points:
- Inducing points selected via existing pivoted Cholesky strategy
- GP prediction becomes O(m) per candidate instead of O(300): 3-6x faster
- All n data points contribute to inducing point selection (better surrogate quality)
- `compute_cross_kernel_subset` calls need kernel only against m inducing points

## Key changes
- New `SparseGPRegressor` class (or mode in existing GPRegressor)
- `fit()`: Compute Qff = Kfu @ Kuu^-1 @ Kuf, then FITC likelihood
- `predict()`: mu = Ksu @ Kuu^-1 @ alpha, var = Kss - Qss + sigma_n^2
- Inducing point selection: reuse pivoted Cholesky from current capping code
- Wire into `run_window_search` as a drop-in replacement

## Complexity
Medium — the Woodbury inversion formulas are well-known but need careful implementation
for numerical stability. The inducing point kernel matrices are small (m x m),
so the Cholesky is cheap.

## Status: Not started

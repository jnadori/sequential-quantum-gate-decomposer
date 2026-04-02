# Plan: Lightweight GP Proxy for Local Search (Idea B)

## Problem
Local search uses the full GP (300 training points) for cross-kernel computation at every step.
`compute_cross_kernel_subset` with 300 training points dominates candidate generation time.

## Approach
Use a small random subset of training points (30-50) for GP predictions inside local search:
- At the start of `generate_candidates`, subsample `train_indices_` to create a lightweight GP
- Pass the lightweight GP into `local_search_acq`
- Only the final Thompson sampling (selecting circuits for decomposition) uses the full GP

The mu prediction with 30 training points is noisy but well-correlated with 300-point prediction.
Local search just needs a rough gradient direction, not precise LCB values.

## Key changes
- Create a fast `GPRegressor` fitted on the subsampled training set before local search
- Pass it to `local_search_acq` instead of the full GP
- Add config parameter `local_search_gp_subset` (default 50)

## Status: DONE (implemented as part of A+B)

# Plan: Stochastic Neighborhood in Local Search (Idea A)

## Problem
`local_search_acq` enumerates ALL D*(n_tokens-1) substitution neighbors per step (~200 for D=20, 10 edges).
Combined with 10 steps per call and 50 calls per iteration, this causes 30M SSK evaluations per iteration.

## Approach
Instead of evaluating all neighbors per step, sample a small random subset:
- Pick k random positions (e.g., k=3-5) instead of all D
- For each position, try all token alternatives
- Cuts neighbors from ~200 to ~30 per step: ~7x speedup

The existing position-visit-count logic can be adapted: instead of hard-masking visited positions,
use it to bias the sampling toward unvisited positions.

## Key changes
- `local_search_acq`: replace full position loop with sampled subset
- Add config parameter `local_search_positions` (default 5) to control sample size
- Keep grow/shrink neighbors as-is (they're already small)

## Status: DONE (implemented as part of A+B)

# Plan: Adaptive Operator Selection (Idea E)

## Problem
Evolutionary operator probabilities in `generate_candidates()` are hardcoded
(0.25 point mutation, 0.15 block regen, etc.). Different problems and search phases
benefit from different operators, but the allocation never adapts.

## Approach
Track which operators produce circuits that improve the score.
Use a Thompson bandit (Beta distributions) to adapt probabilities:

```cpp
struct OperatorStats {
    double alpha, beta;  // Beta distribution parameters
    int total_calls;
};
```

Each iteration:
1. Sample operator probabilities from Beta posteriors
2. Tag each candidate with its source operator
3. After decomposition, if a candidate improved best_score, update its operator's Beta params
4. Decay old observations (sliding window or exponential decay)

## Key changes
- Add `OperatorStats` array (one per operator type) to class
- Tag candidates with operator ID in `generate_candidates`
- Update stats after decompose phase in `run_window_search`
- Replace hardcoded probability thresholds with sampled probabilities

## Complexity
Low — small bookkeeping struct, minimal code changes. The operator tagging is the
main structural change (need to pass operator ID alongside each candidate).

## Status: Not started

# Plan: Incremental Cross-Kernel for 1-Position Mutations (Idea C)

## Problem
When local search steps from circuit C to neighbor N (differing at one position),
`compute_cross_kernel_subset` recomputes the full SSK from scratch for all neighbors.
Most SSK values between N and training points are nearly identical to those of C.

## Approach
Cache the cross-kernel vector K(training[:], C) from the current circuit.
For each neighbor N (differing at position p), compute only the delta:
- For SSK of order k, only subsequences containing position p are affected
- This is O(D^(k-1)) terms out of O(D^k) total, so ~1/D of the work
- For D=20: ~20x speedup per local search step after the first

## Key changes
- Implement `incremental_ssk_delta()` in SSKCache that computes only the change
  from modifying one position in a sequence
- Modify `local_search_acq` to cache the current circuit's kernel vector and
  use incremental updates for substitution neighbors
- Grow/shrink neighbors still use full computation (topology changes)

## Complexity
Medium — requires understanding the SSK DP recurrence to factor out position-dependent terms.
The batch_ssk_raw DP uses Kp matrices and gap-decay recurrences that are position-coupled,
so the incremental formula needs careful derivation.

## Status: Not started

# Plan: Contiguous Block Mutations for Large-D Effectiveness

## Problem

At large circuit depth D, the current evolutionary operators make changes that are too
small relative to the total sequence length. `mutate_point` flips one gate out of D,
`mutate_swap` swaps two positions, and `mutate_block` replaces a contiguous block with
**independently random** tokens (each position drawn uniformly). These produce either
trivially small perturbations or structurally incoherent noise. The search stagnates
because the GP surrogate can't distinguish the resulting candidates.

## Goal

Add two new mutation operators that make **structurally meaningful** large-scale changes:

1. **`mutate_transplant`** — copy a contiguous block from a donor circuit (a different
   good circuit from the population) into the recipient, replacing a block of the same
   length. This transfers known-good subcircuit patterns between individuals.

2. **`mutate_block_regenerate`** — replace a contiguous block by running
   `generate_valid_sequence` logic *only within that block*, respecting the context of
   the surrounding gates. This is a "controlled explosion" that randomizes a region
   while maintaining validity.

Also improve the existing `mutate_block` to bias toward the donor/neighbor-aware
strategy instead of pure random tokens.

## Files to Modify

- `squander/src-cpp/decomposition/include/N_Qubit_Decomposition_Surrogate.h`
- `squander/src-cpp/decomposition/N_Qubit_Decomposition_Surrogate.cpp`

## Detailed Implementation

### Step 1: Add method declarations to the header

In `N_Qubit_Decomposition_Surrogate.h`, in the "Evolutionary operators" section
(around line 399-405), add:

```cpp
/// Transplant a contiguous block from a donor circuit into the recipient.
/// donor and recipient must have the same length (or block is sized to min).
GrayCode mutate_transplant(const GrayCode& recipient, const GrayCode& donor);

/// Replace a contiguous block with a freshly generated valid sub-sequence,
/// respecting context from surrounding gates.
GrayCode mutate_block_regenerate(const GrayCode& seq, int blk_size);
```

### Step 2: Implement `mutate_transplant`

Location: `N_Qubit_Decomposition_Surrogate.cpp`, after `mutate_block` (around line 1456).

**Algorithm:**

```
mutate_transplant(recipient, donor):
    D_r = recipient.size()
    D_d = donor.size()
    blk = random int in [2, min(block_size+1, min(D_r, D_d))]

    for attempt in 0..49:
        # Pick random block start in donor
        d_start = random int in [0, D_d - blk)
        # Pick random insertion point in recipient
        r_start = random int in [0, D_r - blk)

        # Copy recipient, overwrite [r_start, r_start+blk) from donor[d_start..]
        new_seq = recipient.copy()
        for i in 0..blk-1:
            new_seq[r_start + i] = donor[d_start + i]

        result = canonicalize_and_validate(new_seq)
        if result.size() > 0:
            return result

    return GrayCode()  // failed
```

Key points:
- The block size is randomized between 2 and `block_size` (the existing config param,
  default 3). This controls mutation radius.
- Both donor and recipient come from the population via tournament selection (the caller
  in `generate_candidates` handles this).
- When D_r != D_d (mixed-D mode), clamp blk to `min(D_r, D_d)` so the block fits both.

### Step 3: Implement `mutate_block_regenerate`

Location: right after `mutate_transplant`.

**Algorithm:**

```
mutate_block_regenerate(seq, blk_size):
    D = seq.size()
    bs = min(blk_size, D)
    # Use a larger block than mutate_block: [bs, min(2*bs, D)]
    actual_bs = random int in [bs, min(2 * bs, D)]

    for attempt in 0..49:
        start = random int in [0, D - actual_bs]

        new_seq = seq.copy()

        # Build mask context from positions outside the block
        # (positions 0..start-1 and start+actual_bs..D-1 are fixed)
        path_masks[i] = token_masks[new_seq[i]]  for all i

        # Regenerate positions [start, start+actual_bs) one by one
        valid_block = true
        for pos in start .. start+actual_bs-1:
            # Collect valid tokens at this position
            candidates = []
            for e in 0..n_tokens-1:
                new_seq[pos] = e
                path_masks[pos] = token_masks[e]
                if not check_new_position(path_masks, pos):
                    candidates.append(e)

            if candidates is empty:
                valid_block = false
                break

            # Pick uniformly from valid tokens
            new_seq[pos] = candidates[random index]
            path_masks[pos] = token_masks[new_seq[pos]]

        if not valid_block:
            continue

        result = canonicalize_and_validate(new_seq)
        if result.size() > 0:
            return result

    return GrayCode()  // failed
```

Key points:
- The block size is **larger** than `mutate_block` uses — up to `2 * block_size`. This
  is the whole point: make a bigger structural change.
- The regeneration respects `check_new_position` at each step, so the block is locally
  valid. The final `canonicalize_and_validate` handles global validity (canonical form,
  OSR).
- Unlike `mutate_block` which picks tokens independently (ignoring subspace constraints
  until the end), this builds the block incrementally so it has a much higher acceptance
  rate.

### Step 4: Wire into `generate_candidates`

In `generate_candidates` (line ~1768), modify the operator probability tables.

**Mixed-D mode** (the `if (mixed_d)` branch, line 1776):

Replace the current distribution:
```
0.30  mutate_point
0.15  mutate_swap
0.15  mutate_block         <-- replace
0.10  crossover_uniform
0.05  mutate_grow
0.15  mutate_shrink
0.10  generate_valid_sequence
```

With:
```
0.20  mutate_point
0.10  mutate_swap
0.10  mutate_block_regenerate   (NEW)
0.15  mutate_transplant         (NEW)
0.10  crossover_uniform
0.05  mutate_grow
0.15  mutate_shrink
0.05  mutate_block              (reduced, kept for small perturbations)
0.10  generate_valid_sequence
```

**Fixed-D mode** (the `else` branch, line 1800):

Replace:
```
0.40  mutate_point
0.20  mutate_swap
0.15  mutate_block         <-- replace
0.15  crossover_uniform
0.10  generate_valid_sequence
```

With:
```
0.25  mutate_point
0.10  mutate_swap
0.15  mutate_block_regenerate   (NEW)
0.20  mutate_transplant         (NEW)
0.10  crossover_uniform
0.05  mutate_block              (reduced)
0.15  generate_valid_sequence
```

For `mutate_transplant`, the caller needs **two** parents. Use `tournament_select()`
for both (same pattern as `crossover_uniform`):
```cpp
} else if (r < threshold_transplant) {
    const GrayCode& p1 = tournament_select();
    const GrayCode& p2 = tournament_select();
    result = mutate_transplant(p1, p2);
}
```

### Step 5: Adaptive block size based on D

Currently `block_size` is a fixed config parameter (default 3). At large D, even
`2 * block_size = 6` may be too small. Add adaptive scaling in `generate_candidates`
before calling the block operators:

```cpp
int adaptive_blk = std::max(block_size, D / 4);  // at D=20, blk=5; at D=40, blk=10
```

Pass `adaptive_blk` to `mutate_block_regenerate` and `mutate_transplant` instead of
the raw `block_size`. This ensures the mutation radius scales with circuit depth.

This should NOT be a new config parameter — just compute it locally.

## Testing

1. **Unit test**: For a small topology (e.g. 3 qubits, linear), verify that
   `mutate_transplant` and `mutate_block_regenerate` produce valid canonical circuits
   that differ from the input by at least `blk_size` positions.

2. **Acceptance rate test**: Run each operator 1000 times on random D=15 circuits.
   `mutate_block_regenerate` should have >50% acceptance (vs the current `mutate_block`
   which has low acceptance at large D because random tokens usually violate subspace
   constraints). `mutate_transplant` should have similar acceptance to `mutate_block`
   since the donor block was already valid in its original context.

3. **Integration test**: Run `search_over_D_range` on a known 3-qubit unitary at D=8-12
   and verify the search completes. Compare iteration counts and best_score trajectories
   with and without the new operators to confirm the new operators don't regress small-D
   performance.

## Notes

- The `mutate_transplant` operator is analogous to "headless chicken crossover" in
  genetic programming — it uses a second individual as a source of building blocks
  rather than doing true recombination. This is more effective than uniform crossover
  when the building blocks are positional (contiguous subcircuits).
- The `mutate_block_regenerate` operator fills the gap between `mutate_point` (too
  small) and `generate_valid_sequence` (too random). It preserves the circuit's global
  structure while randomizing a local region.
- Both operators interact correctly with `canonicalize_and_validate` — they produce
  raw sequences that get canonicalized and checked, same as all other operators.

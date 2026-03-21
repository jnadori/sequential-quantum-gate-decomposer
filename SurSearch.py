from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from squander import N_Qubit_Decomposition_custom, Circuit

from math import ceil
import heapq
import time



# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def precompute_edge_masks(edges, vertices):
    vertex_to_bit = {v: i for i, v in enumerate(sorted(vertices))}
    return [
        (1 << vertex_to_bit[e[0]]) | (1 << vertex_to_bit[e[1]])
        for e in edges
    ]


def precompute_thresholds(num_vertices):
    return [
        (n, ceil((4**n - 3 * n - 1) / 4) + 1)
        for n in range(2, num_vertices)
    ]


# ---------------------------------------------------------------------------
# Canonical form via dependency DAG (bitmask version)
# ---------------------------------------------------------------------------

def canonical_form(seq, masks):
    if not seq:
        return ()

    n = len(seq)
    adj = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        for j in range(i):
            if masks[j] & masks[i]:  # shared vertex = dependent
                adj[j].append(i)
                in_degree[i] += 1

    heap = [(seq[i], i) for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        edge, idx = heapq.heappop(heap)
        result.append(edge)
        for neighbor in adj[idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, (seq[neighbor], neighbor))

    return tuple(result)


# ---------------------------------------------------------------------------
# Subspace check (bitmask version)
# ---------------------------------------------------------------------------

def check_new_position(window_masks, pos, thresholds):
    for n, j in thresholds:
        if j > pos + 1:
            continue

        combined = 0
        for i in range(pos - j + 1, pos + 1):
            combined |= window_masks[i]

        if combined.bit_count() <= n:
            return True

    return False


# ---------------------------------------------------------------------------
# Canonical prefix check (bitmask version)
# ---------------------------------------------------------------------------

def is_canonical_prefix(path, path_masks, depth):
    i = depth
    while i > 0:
        if path_masks[i] & path_masks[i - 1]:
            break  # dependent — order fixed
        if path[i] < path[i - 1]:
            return False  # independent and out of order
        i -= 1
    return True


# ---------------------------------------------------------------------------
# DFS generators
# ---------------------------------------------------------------------------

def unique_k_sequences(topology, k):
    edges = sorted(set(tuple(e) for e in topology))
    vertices = {v for edge in edges for v in edge}
    num_vertices = len(vertices)
    edge_masks = precompute_edge_masks(edges, vertices)
    thresholds = precompute_thresholds(num_vertices)

    seen = set()
    results = []
    path = [None] * k
    path_masks = [0] * k

    def dfs(depth):
        if depth == k:
            seq = tuple(path)
            masks = path_masks[:k]
            canon = canonical_form(seq, masks)
            if canon not in seen:
                seen.add(canon)
                results.append(canon)
            return

        for edge, mask in zip(edges, edge_masks):
            path[depth] = edge
            path_masks[depth] = mask

            if check_new_position(path_masks, depth, thresholds):
                continue
            if not is_canonical_prefix(path, path_masks, depth):
                continue

            dfs(depth + 1)

    dfs(0)
    return results


# ---------------------------------------------------------------------------
# Random sequence generation & mutation operators (for large-D evolutionary search)
# ---------------------------------------------------------------------------

def _canonicalize_and_validate(seq, edge_to_mask, thresholds):
    masks = [edge_to_mask[e] for e in seq]
    canon = canonical_form(seq, masks)
    canon_masks = [edge_to_mask[e] for e in canon]
    for depth in range(len(canon)):
        if check_new_position(canon_masks, depth, thresholds):
            return None
    return canon


def generate_valid_sequence(D, edges, edge_masks, thresholds, edge_to_mask):
    E = len(edges)
    path = [None] * D
    path_masks = [0] * D

    for depth in range(D):
        valid = []
        for idx in range(E):
            path[depth] = edges[idx]
            path_masks[depth] = edge_masks[idx]
            if not check_new_position(path_masks, depth, thresholds):
                valid.append(idx)
        if not valid:
            return None
        chosen = valid[np.random.randint(len(valid))]
        path[depth] = edges[chosen]
        path_masks[depth] = edge_masks[chosen]

    return _canonicalize_and_validate(tuple(path), edge_to_mask, thresholds)


def mutate_point(seq, edges, edge_to_mask, thresholds, max_attempts=50):
    D = len(seq)
    E = len(edges)
    for _ in range(max_attempts):
        pos = np.random.randint(D)
        new_seq = list(seq)
        new_seq[pos] = edges[np.random.randint(E)]
        result = _canonicalize_and_validate(tuple(new_seq), edge_to_mask, thresholds)
        if result is not None:
            return result
    return None


def mutate_swap(seq, edge_to_mask, thresholds, max_attempts=50):
    D = len(seq)
    if D < 2:
        return None
    for _ in range(max_attempts):
        i, j = np.random.choice(D, size=2, replace=False)
        new_seq = list(seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        result = _canonicalize_and_validate(tuple(new_seq), edge_to_mask, thresholds)
        if result is not None:
            return result
    return None


def mutate_block(seq, edges, edge_to_map, thresholds, block_size=3, max_attempts=50):
    D = len(seq)
    E = len(edges)
    bs = min(block_size, D)
    for _ in range(max_attempts):
        start = np.random.randint(D - bs + 1)
        new_seq = list(seq)
        for pos in range(start, start + bs):
            new_seq[pos] = edges[np.random.randint(E)]
        result = _canonicalize_and_validate(tuple(new_seq), edge_to_map, thresholds)
        if result is not None:
            return result
    return None


def crossover_uniform(seq1, seq2, edge_to_mask, thresholds, max_attempts=50):
    D = len(seq1)
    for _ in range(max_attempts):
        new_seq = tuple(seq1[i] if np.random.random() < 0.5 else seq2[i]
                        for i in range(D))
        result = _canonicalize_and_validate(new_seq, edge_to_mask, thresholds)
        if result is not None:
            return result
    return None


def mutate_grow(seq, edges, edge_to_mask, thresholds, D_max, max_attempts=50):
    """Insert one random edge at a random position (D -> D+1). Respects D_max."""
    D = len(seq)
    if D >= D_max:
        return None
    E = len(edges)
    for _ in range(max_attempts):
        pos = np.random.randint(D + 1)
        edge = edges[np.random.randint(E)]
        new_seq = list(seq)
        new_seq.insert(pos, edge)
        result = _canonicalize_and_validate(tuple(new_seq), edge_to_mask, thresholds)
        if result is not None:
            return result
    return None


def mutate_shrink(seq, edge_to_mask, thresholds, D_min, max_attempts=50):
    """Remove one edge at a random position (D -> D-1). Respects D_min."""
    D = len(seq)
    if D <= D_min:
        return None
    for _ in range(max_attempts):
        pos = np.random.randint(D)
        new_seq = list(seq)
        new_seq.pop(pos)
        result = _canonicalize_and_validate(tuple(new_seq), edge_to_mask, thresholds)
        if result is not None:
            return result
    return None


def _local_search_acq(start_seq, edges, edge_to_mask, thresholds,
                      ssk_cache, gp, scale, kappa,
                      max_steps=10, seen=None,
                      D_min=None, D_max=None):
    """Greedy local search on LCB using discrete gradient (BOSS-style).

    At each step, generates all valid single-position substitution neighbors
    (and optionally grow/shrink neighbors when D_min/D_max are provided),
    evaluates their LCB via the GP surrogate (no decomposition calls), and
    moves to the neighbor with the best (lowest) LCB value.
    """
    current = start_seq

    # Evaluate LCB for starting point
    Ks_cur = ssk_cache.compute_cross_kernel([current], scale)
    mu_cur = float(Ks_cur.T @ gp.alpha)
    v_cur = np.linalg.solve(gp.L, Ks_cur)
    std_cur = np.sqrt(max(scale - float(np.sum(v_cur * v_cur)), 0.0))
    best_lcb = mu_cur - kappa * std_cur

    steps_taken = 0
    for step in range(max_steps):
        D = len(current)
        neighbors = set()

        # Single-position substitution neighbors
        for pos in range(D):
            for edge in edges:
                if edge == current[pos]:
                    continue
                new_seq = list(current)
                new_seq[pos] = edge
                candidate = _canonicalize_and_validate(
                    tuple(new_seq), edge_to_mask, thresholds)
                if candidate is not None and (seen is None or candidate not in seen):
                    neighbors.add(candidate)

        # Grow neighbors (D -> D+1)
        if D_max is not None and D < D_max:
            for pos in range(D + 1):
                for edge in edges:
                    new_seq = list(current)
                    new_seq.insert(pos, edge)
                    candidate = _canonicalize_and_validate(
                        tuple(new_seq), edge_to_mask, thresholds)
                    if candidate is not None and (seen is None or candidate not in seen):
                        neighbors.add(candidate)

        # Shrink neighbors (D -> D-1)
        if D_min is not None and D > D_min:
            for pos in range(D):
                new_seq = list(current)
                new_seq.pop(pos)
                candidate = _canonicalize_and_validate(
                    tuple(new_seq), edge_to_mask, thresholds)
                if candidate is not None and (seen is None or candidate not in seen):
                    neighbors.add(candidate)

        neighbors = list(neighbors)
        if not neighbors:
            break

        # Batch-evaluate LCB for all neighbors via surrogate
        Ks = ssk_cache.compute_cross_kernel(neighbors, scale)
        mu = Ks.T @ gp.alpha
        v = np.linalg.solve(gp.L, Ks)
        var = np.maximum(scale - np.sum(v * v, axis=0), 0.0)
        std = np.sqrt(var)
        lcb_vals = mu.ravel() - kappa * std

        best_idx = int(np.argmin(lcb_vals))
        if lcb_vals[best_idx] >= best_lcb:
            break  # local optimum reached

        current = neighbors[best_idx]
        best_lcb = lcb_vals[best_idx]
        steps_taken += 1

    return current, best_lcb, steps_taken


def _generate_candidates(population, scores, n_candidates, edges, edge_masks,
                         thresholds, edge_to_mask, seen,
                         ssk_cache=None, gp=None, scale=None, kappa=None,
                         tournament_size=3, block_size=3,
                         local_search_fraction=0.5, max_local_steps=10,
                         D_min=None, D_max=None):
    candidates = []
    n_pop = len(population)
    t_size = min(tournament_size, n_pop)

    # Pre-generate random numbers in bulk
    max_total_attempts = n_candidates * 10
    op_probs = np.random.random(max_total_attempts)
    tour_indices = np.random.randint(0, n_pop, size=(max_total_attempts * 2, t_size))
    tour_idx = 0

    def tournament_select():
        nonlocal tour_idx
        indices = tour_indices[tour_idx]
        tour_idx += 1
        best_idx = indices[np.argmin(scores[indices])]
        return population[best_idx]

    # GP-directed local search candidates
    n_local_found = 0
    total_local_steps = 0
    if ssk_cache is not None and gp is not None:
        n_local_target = int(n_candidates * local_search_fraction)
        for _ in range(n_local_target):
            parent = tournament_select()
            result, lcb, steps = _local_search_acq(
                parent, edges, edge_to_mask, thresholds,
                ssk_cache, gp, scale, kappa,
                max_steps=max_local_steps, seen=seen,
                D_min=D_min, D_max=D_max)
            total_local_steps += steps
            if result is not None and result not in seen:
                seen.add(result)
                candidates.append(result)
                n_local_found += 1

    mixed_d = D_min is not None and D_max is not None

    # Random candidates via operators
    attempts = 0
    while len(candidates) < n_candidates and attempts < max_total_attempts:
        r = op_probs[attempts]
        attempts += 1

        if mixed_d:
            if r < 0.30:
                result = mutate_point(tournament_select(), edges, edge_to_mask, thresholds)
            elif r < 0.45:
                result = mutate_swap(tournament_select(), edge_to_mask, thresholds)
            elif r < 0.60:
                result = mutate_block(tournament_select(), edges, edge_to_mask,
                                      thresholds, block_size=block_size)
            elif r < 0.70:
                p1 = tournament_select()
                p2 = tournament_select()
                if len(p1) == len(p2):
                    result = crossover_uniform(p1, p2, edge_to_mask, thresholds)
                else:
                    result = mutate_point(p1, edges, edge_to_mask, thresholds)
            elif r < 0.80:
                result = mutate_grow(tournament_select(), edges, edge_to_mask,
                                     thresholds, D_max)
            elif r < 0.90:
                result = mutate_shrink(tournament_select(), edge_to_mask,
                                       thresholds, D_min)
            else:
                rand_D = np.random.randint(D_min, D_max + 1)
                result = generate_valid_sequence(rand_D, edges, edge_masks,
                                                  thresholds, edge_to_mask)
        else:
            D = len(population[0])
            if r < 0.40:
                result = mutate_point(tournament_select(), edges, edge_to_mask, thresholds)
            elif r < 0.60:
                result = mutate_swap(tournament_select(), edge_to_mask, thresholds)
            elif r < 0.75:
                result = mutate_block(tournament_select(), edges, edge_to_mask,
                                      thresholds, block_size=block_size)
            elif r < 0.85:
                result = crossover_uniform(tournament_select(), tournament_select(),
                                           edge_to_mask, thresholds)
            else:
                result = generate_valid_sequence(D, edges, edge_masks, thresholds,
                                                 edge_to_mask)

        if result is not None and result not in seen:
            seen.add(result)
            candidates.append(result)

    avg_steps = total_local_steps / max(n_local_found, 1)
    return candidates, n_local_found, avg_steps


Array = np.ndarray
def create_circuit_from_edges(x,N):
    circ = Circuit(N)
    for edge in x:
        circ.add_U3(edge[0])
        circ.add_U3(edge[1])
        circ.add_CNOT(edge[0],edge[1])
    for qbit_idx in range(N):
        circ.add_U3(qbit_idx)
    return circ


@dataclass
class GPRegressor:
    kernel_params: Array
    noise: float = 1e-3
    jitter: float = 1e-8
    kernel_fn: Optional[Any] = None

    # Learned during fit:
    F_train: Optional[Array] = None
    y_train: Optional[Array] = None
    L: Optional[Array] = None
    alpha: Optional[Array] = None

    def _kernel(self, F1: Array, F2: Array) -> Array:
        return self.kernel_fn(F1, F2, self.kernel_params)

    def fit(self, F: Array, y: Array) -> "GPRegressor":
        """Fit GP on precomputed feature matrix F (n, d) and targets y."""
        y = np.asarray(y, dtype=float).reshape(-1)
        F = np.asarray(F, dtype=float)
        if F.shape[0] != y.shape[0]:
            raise ValueError("F.shape[0] must equal len(y).")

        self.F_train = F
        self.y_train = y

        K = self._kernel(F, F)
        K = self._add_diagonal_noise(K, self.noise, self.jitter)

        jitter = self.jitter
        n = K.shape[0]
        for _ in range(10):
            try:
                self.L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                K[np.arange(n), np.arange(n)] += jitter
                jitter *= 10
        else:
            raise np.linalg.LinAlgError("Kernel matrix not positive definite even after jitter augmentation.")
        self.alpha = self._solve_cholesky(self.L, y)
        return self

    def predict(
        self,
        F_star: Array,
        return_std: bool = True,
        include_noise: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        if self.F_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before predict().")
        if self.L is None or self.alpha is None:
            raise RuntimeError("Model is not properly fit (missing L/alpha).")

        F_star = np.asarray(F_star, dtype=float)
        Ks = self._kernel(self.F_train, F_star)
        mean = Ks.T @ self.alpha  # (n_star,)

        if not return_std:
            return mean, None

        # Self-kernel is always scale (SSK normalized diagonal = 1)
        scale = float(np.exp(self.kernel_params[0]))
        kss_diag = np.full(F_star.shape[0], scale)

        # Posterior variance: diag(Kss) - diag(Ks^T K^{-1} Ks)
        v = np.linalg.solve(self.L, Ks)  # (n_train, n_star)
        var = kss_diag - np.sum(v * v, axis=0)

        var = np.maximum(var, 0.0)

        if include_noise:
            var = var + self.noise

        std = np.sqrt(var)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        if self.F_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before log_marginal_likelihood().")

        F = self.F_train
        y = self.y_train

        K = self._kernel(F, F)
        K = self._add_diagonal_noise(K, self.noise, self.jitter)

        L = np.linalg.cholesky(K)
        alpha = self._solve_cholesky(L, y)

        n = y.shape[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        quad = float(y.T @ alpha)
        return -0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)

    def optimize_hyperparameters(self, n_restarts: int = 1,
                                  kernel_bounds: Optional[list] = None,
                                  noise_bounds: Tuple[float, float] = (-8.0, 2.0)) -> "GPRegressor":
        if self.F_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before optimize_hyperparameters().")

        n_kernel = len(self.kernel_params)

        def objective(theta):
            self.kernel_params = theta[:n_kernel]
            self.noise = float(np.exp(theta[n_kernel]))  # log-space for positivity
            try:
                self.fit(self.F_train, self.y_train)
                return -self.log_marginal_likelihood()
            except np.linalg.LinAlgError:
                return 1e10

        # Initial point: current params + log(noise)
        theta0 = np.concatenate([self.kernel_params, [np.log(self.noise)]])
        best_theta = theta0.copy()
        best_obj = objective(theta0)

        # Bounds: use caller-provided kernel bounds or default [-3,3] per kernel param
        if kernel_bounds is not None:
            bounds = list(kernel_bounds) + [noise_bounds]
        else:
            bounds = [(-3.0, 3.0)] * n_kernel + [noise_bounds]

        for _ in range(n_restarts):
            theta_init = theta0 + 0.5 * np.random.randn(len(theta0))
            theta_init = np.clip(theta_init, [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(objective, theta_init, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 100, "ftol": 1e-4})
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.copy()

        # Apply best parameters and refit
        self.kernel_params = best_theta[:n_kernel]
        self.noise = float(np.exp(best_theta[n_kernel]))
        self.fit(self.F_train, self.y_train)
        return self

    @staticmethod
    def _add_diagonal_noise(K: Array, noise: float, jitter: float) -> Array:
        if noise < 0:
            raise ValueError("noise must be >= 0.")
        K2 = np.array(K, dtype=float, copy=True)
        n = K2.shape[0]
        K2[np.arange(n), np.arange(n)] += (noise + jitter)
        return K2

    @staticmethod
    def _solve_cholesky(L: Array, y: Array) -> Array:
        tmp = np.linalg.solve(L, y)
        return np.linalg.solve(L.T, tmp)


def _circuits_to_int_array(circuits):
    """Convert edge-tuple sequences to a numpy int32 array with integer edge IDs."""
    edge_to_id = {}
    next_id = 0
    n_circuits = len(circuits)
    seq_len = len(circuits[0])
    arr = np.empty((n_circuits, seq_len), dtype=np.int32)
    for i, circ in enumerate(circuits):
        for j, edge in enumerate(circ):
            if edge not in edge_to_id:
                edge_to_id[edge] = next_id
                next_id += 1
            arr[i, j] = edge_to_id[edge]
    return arr, edge_to_id


def ssk_kernel_value(s1, s2, D, match_sq, order):
    """Compute SSK kernel between two equal-length sequences.

    Uses the matrix DP from Moss et al. (BOSS). D is the pre-computed gap decay
    matrix (strictly upper triangular) and match_sq = match_decay ** 2.

    The kernel sums contributions from subsequence orders 1 through order,
    with equal weights (order_coefs = 1).
    """
    n = len(s1)

    # Match indicator matrix
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if s1[i] == s2[j]:
                S[i, j] = 1.0

    Kp = np.ones((n, n))
    k = match_sq * float(np.sum(S * Kp))

    for _ in range(order - 1):
        Kpp = match_sq * ((S * Kp) @ D)
        Kp = (Kpp.T @ D).T
        k += match_sq * float(np.sum(S * Kp))

    return k


def _get_gap_decay_matrix(n, gap_decay, cache={}):
    """Get or compute gap decay matrix for length n (module-level cache)."""
    key = (n, gap_decay)
    if key not in cache:
        idx = np.arange(n)
        cache[key] = np.where(
            idx[None, :] > idx[:, None],
            gap_decay ** (idx[None, :] - idx[:, None] - 1), 0.0)
    return cache[key]


def ssk_gram_matrix(circuits, gap_decay, match_decay, order):
    """Compute the full SSK Gram matrix for a list of circuits (variable-length).

    Returns an (n_circuits, n_circuits) normalized kernel matrix.
    Groups pairs by (len_i, len_j) for batched computation.
    """
    n_circuits = len(circuits)
    match_sq = match_decay * match_decay
    lens = [len(c) for c in circuits]

    # Encode all circuits as int arrays
    edge_to_id = {}
    next_id = 0
    circuit_ints = []
    for circ in circuits:
        int_arr = np.empty(len(circ), dtype=np.int32)
        for j, edge in enumerate(circ):
            if edge not in edge_to_id:
                edge_to_id[edge] = next_id
                next_id += 1
            int_arr[j] = edge_to_id[edge]
        circuit_ints.append(int_arr)

    lens_arr = np.array(lens, dtype=np.int32)

    # Fast path: all same length
    if len(set(lens)) == 1:
        n = lens[0]
        D = _get_gap_decay_matrix(n, gap_decay)
        ci_arr = np.stack(circuit_ints)
        ii, jj = np.triu_indices(n_circuits)
        P = len(ii)

        ci = ci_arr[ii]
        cj = ci_arr[jj]
        S = (ci[:, :, None] == cj[:, None, :]).astype(np.float64)

        Kp = np.ones((P, n, n))
        k = match_sq * np.sum(S * Kp, axis=(1, 2))

        for _ in range(order - 1):
            Kpp = match_sq * ((S * Kp) @ D)
            Kp = (Kpp.transpose(0, 2, 1) @ D).transpose(0, 2, 1)
            k += match_sq * np.sum(S * Kp, axis=(1, 2))

        K = np.zeros((n_circuits, n_circuits))
        K[ii, jj] = k
        K[jj, ii] = k

        diag = np.sqrt(np.diag(K))
        diag[diag < 1e-12] = 1.0
        K_norm = K / (diag[:, None] * diag[None, :])
        return K_norm

    # Variable-length path: group upper-triangle pairs by (len_i, len_j)
    ii, jj = np.triu_indices(n_circuits)
    pair_lens_i = lens_arr[ii]
    pair_lens_j = lens_arr[jj]

    K_raw = np.zeros((n_circuits, n_circuits))

    unique_len_pairs = set(zip(pair_lens_i.tolist(), pair_lens_j.tolist()))
    for len_i, len_j in unique_len_pairs:
        mask = (pair_lens_i == len_i) & (pair_lens_j == len_j)
        pair_indices = np.where(mask)[0]
        P = len(pair_indices)

        ii_sel = ii[pair_indices]
        jj_sel = jj[pair_indices]

        ci = np.stack([circuit_ints[i] for i in ii_sel])
        cj = np.stack([circuit_ints[j] for j in jj_sel])

        S = (ci[:, :, None] == cj[:, None, :]).astype(np.float64)

        D_i = _get_gap_decay_matrix(len_i, gap_decay)
        D_j = _get_gap_decay_matrix(len_j, gap_decay)

        Kp = np.ones((P, len_i, len_j))
        k = match_sq * np.sum(S * Kp, axis=(1, 2))

        for _ in range(order - 1):
            Kpp = match_sq * ((S * Kp) @ D_j)
            Kp = (Kpp.transpose(0, 2, 1) @ D_i).transpose(0, 2, 1)
            k += match_sq * np.sum(S * Kp, axis=(1, 2))

        K_raw[ii_sel, jj_sel] = k
        K_raw[jj_sel, ii_sel] = k

    diag = np.sqrt(np.diag(K_raw))
    diag[diag < 1e-12] = 1.0
    K_norm = K_raw / (diag[:, None] * diag[None, :])
    return K_norm


class SSKCache:
    """Incremental SSK kernel cache supporting variable-length circuits.

    Computes and caches pairwise SSK kernel values on demand, avoiding
    the need to precompute the full Gram matrix over all circuits.
    Maintains a dense normalized kernel matrix for fast slicing.
    """

    def __init__(self, gap_decay, match_decay, order):
        self.gap_decay = gap_decay
        self.match_sq = match_decay * match_decay
        self.order = order
        self.circuits = []
        self.circuit_to_idx = {}
        self._D_matrices = {}
        self._capacity = 64
        self._size = 0
        self._K_norm = np.empty((64, 64))
        self._raw_diags = np.empty(64)
        self._circuits_int = []  # list of variable-length int arrays
        self._circuit_lens = np.empty(64, dtype=np.int32)
        self._edge_to_id = {}
        self._next_id = 0

    def _get_D_matrix(self, n):
        """Get or compute gap decay matrix for length n."""
        if n not in self._D_matrices:
            idx = np.arange(n)
            self._D_matrices[n] = np.where(
                idx[None, :] > idx[:, None],
                self.gap_decay ** (idx[None, :] - idx[:, None] - 1), 0.0)
        return self._D_matrices[n]

    def _encode_circuit(self, circuit):
        """Encode a circuit (tuple of edges) as an int array using shared edge IDs."""
        n = len(circuit)
        new_int = np.empty(n, dtype=np.int32)
        for j, edge in enumerate(circuit):
            if edge not in self._edge_to_id:
                self._edge_to_id[edge] = self._next_id
                self._next_id += 1
            new_int[j] = self._edge_to_id[edge]
        return new_int

    def _grow_capacity(self):
        """Double the capacity of pre-allocated arrays."""
        new_cap = self._capacity * 2
        sz = self._size

        new_lens = np.empty(new_cap, dtype=np.int32)
        new_lens[:sz] = self._circuit_lens[:sz]
        self._circuit_lens = new_lens

        new_raw_diags = np.empty(new_cap)
        new_raw_diags[:sz] = self._raw_diags[:sz]
        self._raw_diags = new_raw_diags

        new_K_norm = np.empty((new_cap, new_cap))
        new_K_norm[:sz, :sz] = self._K_norm[:sz, :sz]
        self._K_norm = new_K_norm

        self._capacity = new_cap

    def _batch_ssk_raw(self, ci_list, cj_list, n1, n2):
        """Compute raw SSK values for a batch of (n1, n2) length pairs.

        ci_list: np.array of shape (P, n1) — int-encoded first sequences
        cj_list: np.array of shape (P, n2) — int-encoded second sequences
        Returns: np.array of shape (P,) — raw (unnormalized) SSK values
        """
        P = ci_list.shape[0]
        match_sq = self.match_sq
        D1 = self._get_D_matrix(n1)
        D2 = self._get_D_matrix(n2)

        S = (ci_list[:, :, None] == cj_list[:, None, :]).astype(np.float64)
        Kp = np.ones((P, n1, n2))
        raw = match_sq * np.sum(S * Kp, axis=(1, 2))

        for _ in range(self.order - 1):
            Kpp = match_sq * ((S * Kp) @ D2)
            Kp = (Kpp.transpose(0, 2, 1) @ D1).transpose(0, 2, 1)
            raw += match_sq * np.sum(S * Kp, axis=(1, 2))

        return raw

    def _single_ssk_raw(self, int_a, int_b):
        """Compute raw SSK value for a single pair of sequences."""
        n1, n2 = len(int_a), len(int_b)
        match_sq = self.match_sq
        D1 = self._get_D_matrix(n1)
        D2 = self._get_D_matrix(n2)

        S = (int_a[:, None] == int_b[None, :]).astype(np.float64)
        Kp = np.ones((n1, n2))
        k = match_sq * float(np.sum(S * Kp))

        for _ in range(self.order - 1):
            Kpp = match_sq * ((S * Kp) @ D2)
            Kp = (Kpp.T @ D1).T
            k += match_sq * float(np.sum(S * Kp))

        return k

    def register(self, circuit):
        key = tuple(circuit)
        if key in self.circuit_to_idx:
            return self.circuit_to_idx[key]

        new_idx = self._size
        self.circuits.append(key)
        self.circuit_to_idx[key] = new_idx

        new_int = self._encode_circuit(key)
        n_new = len(new_int)

        if new_idx >= self._capacity:
            self._grow_capacity()

        self._circuits_int.append(new_int)
        self._circuit_lens[new_idx] = n_new

        # Compute raw kernel values: new vs all existing + self
        raw_new = np.empty(new_idx + 1)

        # Self-kernel
        raw_new[new_idx] = self._single_ssk_raw(new_int, new_int)

        # Group existing circuits by length and batch-compute
        if new_idx > 0:
            lens = self._circuit_lens[:new_idx]
            unique_lens = np.unique(lens)

            for ulen in unique_lens:
                group_indices = np.where(lens == ulen)[0]
                P = len(group_indices)
                ulen = int(ulen)

                cj = np.stack([self._circuits_int[i] for i in group_indices])
                ci = np.broadcast_to(new_int[:n_new][None, :], (P, n_new))

                raw_new[group_indices] = self._batch_ssk_raw(ci, cj, n_new, ulen)

        self._raw_diags[new_idx] = raw_new[new_idx]

        diag_new = np.sqrt(max(raw_new[new_idx], 1e-24))
        diag_all = np.sqrt(np.maximum(self._raw_diags[:new_idx + 1], 1e-24))
        norm_row = raw_new / (diag_new * diag_all)
        self._K_norm[new_idx, :new_idx + 1] = norm_row
        self._K_norm[:new_idx + 1, new_idx] = norm_row

        self._size += 1
        return new_idx

    def compute_cross_kernel(self, candidate_circuits, scale):
        """Compute cross-kernel between registered and candidate circuits.

        Returns scale * K_norm_cross of shape (n_registered, n_candidates).
        Handles variable-length circuits by grouping by length.
        """
        n_reg = self._size
        n_cand = len(candidate_circuits)
        if n_reg == 0 or n_cand == 0:
            return np.empty((n_reg, n_cand))

        match_sq = self.match_sq

        # Encode all candidates and track lengths
        cand_ints = []
        cand_lens = np.empty(n_cand, dtype=np.int32)
        for i, circ in enumerate(candidate_circuits):
            enc = self._encode_circuit(tuple(circ))
            cand_ints.append(enc)
            cand_lens[i] = len(enc)

        raw_cross = np.empty((n_reg, n_cand))
        reg_lens = self._circuit_lens[:n_reg]
        batch_limit = 10000

        # Group candidates by length, then registered by length
        unique_cand_lens = np.unique(cand_lens)
        for cand_len in unique_cand_lens:
            cand_len = int(cand_len)
            cand_indices = np.where(cand_lens == cand_len)[0]
            n_cand_group = len(cand_indices)
            cand_group_int = np.stack([cand_ints[i] for i in cand_indices])

            unique_reg_lens = np.unique(reg_lens)
            for reg_len in unique_reg_lens:
                reg_len = int(reg_len)
                reg_indices = np.where(reg_lens == reg_len)[0]
                n_reg_group = len(reg_indices)
                reg_group_int = np.stack(
                    [self._circuits_int[i] for i in reg_indices])

                cand_batch = max(1, batch_limit // max(n_reg_group, 1))
                for b_start in range(0, n_cand_group, cand_batch):
                    b_end = min(b_start + cand_batch, n_cand_group)
                    bs = b_end - b_start

                    ci = np.repeat(reg_group_int, bs, axis=0)
                    cj = np.tile(cand_group_int[b_start:b_end],
                                 (n_reg_group, 1))

                    raw = self._batch_ssk_raw(ci, cj, reg_len, cand_len)
                    raw_cross[np.ix_(
                        reg_indices,
                        cand_indices[b_start:b_end])] = raw.reshape(
                            n_reg_group, bs)

        # Compute candidate self-kernels for normalization
        cand_self_raw = np.empty(n_cand)
        for cand_len in unique_cand_lens:
            cand_len = int(cand_len)
            cand_indices = np.where(cand_lens == cand_len)[0]
            b_cand = np.stack([cand_ints[i] for i in cand_indices])

            cand_self_raw[cand_indices] = self._batch_ssk_raw(
                b_cand, b_cand, cand_len, cand_len)

        # Normalize
        reg_diags = np.sqrt(np.maximum(self._raw_diags[:n_reg], 1e-24))
        cand_diags = np.sqrt(np.maximum(cand_self_raw, 1e-24))
        K_norm_cross = raw_cross / (reg_diags[:, None] * cand_diags[None, :])

        return scale * K_norm_cross

    def kernel_between(self, circuit_a, circuit_b):
        """Compute normalized SSK kernel between two arbitrary circuits."""
        int_a = self._encode_circuit(tuple(circuit_a))
        int_b = self._encode_circuit(tuple(circuit_b))
        n1, n2 = len(int_a), len(int_b)

        if n1 == n2:
            # Same length: batch all three (ab, aa, bb) together
            D = self._get_D_matrix(n1)
            ci = np.array([int_a, int_a, int_b])
            cj = np.array([int_b, int_a, int_b])
            S = (ci[:, :, None] == cj[:, None, :]).astype(np.float64)
            Kp = np.ones((3, n1, n1))
            raw = self.match_sq * np.sum(S * Kp, axis=(1, 2))
            for _ in range(self.order - 1):
                Kpp = self.match_sq * ((S * Kp) @ D)
                Kp = (Kpp.transpose(0, 2, 1) @ D).transpose(0, 2, 1)
                raw += self.match_sq * np.sum(S * Kp, axis=(1, 2))
            raw_ab, raw_aa, raw_bb = float(raw[0]), float(raw[1]), float(raw[2])
        else:
            # Different lengths: compute each separately
            raw_ab = self._single_ssk_raw(int_a, int_b)
            raw_aa = self._single_ssk_raw(int_a, int_a)
            raw_bb = self._single_ssk_raw(int_b, int_b)

        denom = np.sqrt(max(raw_aa, 1e-24) * max(raw_bb, 1e-24))
        return raw_ab / denom

    def kernel_matrix(self, idx1, idx2, scale):
        return scale * self._K_norm[np.ix_(idx1, idx2)]


class qgd_SurSearch:
    def __init__(self,Umtx,config, D,topology=None):
        self.Umtx = Umtx
        self.N = int(np.log2(Umtx.shape[0]))
        self.config = config
        self.kappa = config.get('kappa',2)
        self.D = D
        if topology == None:
            topology = []
            for idx in range(self.N-1):
                for jdx in range(idx+1,self.N):
                    topology.append((idx,jdx))
        self.topology = topology
        self.best_score = float('inf')
        self.best_params = None
        self.best_circ = None
        config.setdefault('radius_base', config.get('rconstant', 0.5))


    def search_over_D(self, log_file="sursearch_diagnostics.txt"):

        # Auto-dispatch to evolutionary search when space is too large
        edges_est = sorted(set(tuple(e) for e in self.topology))
        enum_threshold = self.config.get('enum_threshold', 10000)
        if len(edges_est) ** self.D > enum_threshold:
            return self.search_over_D_evolve(log_file=log_file)

        # Timing and call counters (before precomputation so total includes it)
        _decompose_time = 0.0
        _decompose_count = 0
        _search_t0 = time.time()

        circuits = unique_k_sequences(self.topology, self.D)
        edges = sorted(set(tuple(e) for e in self.topology))
        acquisition = self.config.get('acquisition', 'LCB')
        tolerance = self.config.get('tolerance', 1e-8)
        max_iters = self.config.get('max_sur_iters', min(len(circuits) // 3, 450))
        patience = self.config.get('patience', max(30, max_iters // 3))
        X0_size = self.config.get('X0_size', 5)

        # Compute SSK kernel for LCB surrogate
        feature_cache = None
        gp_kernel_params = None
        gp_noise = None
        gp_bounds = None
        feat_dim = None
        gp_kernel_fn = None

        if acquisition == 'LCB':
            gap_decay = self.config.get('ssk_gap_decay', 0.8)
            match_decay = self.config.get('ssk_match_decay', 0.8)
            ssk_order = self.config.get('ssk_order', 3)
            K_ssk = ssk_gram_matrix(circuits, gap_decay, match_decay, ssk_order)
            feature_cache = {tuple(c): np.array([float(i)]) for i, c in enumerate(circuits)}
            feat_dim = K_ssk.shape[1]
            gp_kernel_params = np.array([0.0])  # log_scale only
            gp_noise = 1e-2
            gp_bounds = [(-3.0, 3.0)]
            gp_noise_bounds = (-8.0, -1.0)  # max noise=0.37 forces kernel to explain data

            def _ssk_kernel_fn(F1, F2, params):
                scale = float(np.exp(params[0]))
                idx1 = F1[:, 0].astype(int)
                idx2 = F2[:, 0].astype(int)
                return scale * K_ssk[np.ix_(idx1, idx2)]

            gp_kernel_fn = _ssk_kernel_fn

            def _feature_matrix(circs):
                return np.array([feature_cache[tuple(c)] for c in circs])

        # Initial random sample
        np.random.shuffle(circuits)
        X = list(circuits[:X0_size])
        remaining_circs = list(circuits[X0_size:])
        _t = time.time()
        _init_results = [self.decompose(x) for x in X]
        _decompose_time += time.time() - _t
        _decompose_count += len(X)
        y = np.array([r[0] for r in _init_results])
        _init_params = [r[1] for r in _init_results]
        best_idx = int(np.argmin(y))
        best_score = float(y[best_idx])
        best_circ = X[best_idx]
        best_params = _init_params[best_idx]

        with open(log_file, 'w') as f:
            f.write(f"SurSearch diagnostics: N={self.N}, D={self.D}, "
                    f"acquisition={acquisition}, kappa={self.kappa}, topology={self.topology}\n")
            feat_info = f"feature dim: {feat_dim}" if feat_dim is not None else "no surrogate"
            f.write(f"Total circuits: {len(circuits)}, initial sample: {X0_size}, {feat_info}\n")
            f.write(f"Initial y stats: min={np.min(y):.6f}, "
                    f"mean={np.mean(y):.4f}, std={np.std(y):.4f}\n")
            f.write("-" * 70 + "\n")

        if best_score < tolerance:
            print("Success at random selection")
            with open(log_file, 'a') as f:
                f.write("Success at random selection\n")
            self.best_score = best_score
            self.best_params = best_params
            self.best_circ = best_circ
            self._decompose_time = _decompose_time
            self._decompose_count = _decompose_count
            self._total_search_time = time.time() - _search_t0
            return best_circ

        iters_since_improvement = 0
        for itr in range(max_iters):
            if not remaining_circs:
                break

            if acquisition == 'random':
                selected_idx = int(np.random.randint(len(remaining_circs)))
                acq_disp = float('nan')
            else:
                F_train = _feature_matrix(X)
                F_remaining = _feature_matrix(remaining_circs)

                # Dynamic floor: clamp successes to avoid -inf in log10
                nonzero_y = y[y > 0]
                dynamic_floor = float(nonzero_y.min()) * 0.01 if len(nonzero_y) > 0 else tolerance * 0.01
                y_clamped = np.maximum(y, dynamic_floor)
                log_y = np.log10(y_clamped)
                lmu = float(log_y.mean())
                lsig = float(max(log_y.std(), 1e-6))
                log_y_norm = (log_y - lmu) / lsig

                gp = GPRegressor(
                    kernel_params=gp_kernel_params.copy(),
                    noise=gp_noise,
                    kernel_fn=gp_kernel_fn,
                ).fit(F_train, log_y_norm)
                n_restarts = max(1, 3 - len(X) // 10)
                try:
                    gp.optimize_hyperparameters(n_restarts=n_restarts, kernel_bounds=gp_bounds,
                                                noise_bounds=gp_noise_bounds)
                except Exception:
                    pass
                # Collapse detection: if scale/noise ratio is tiny, reset warm-start
                opt_scale = float(np.exp(gp.kernel_params[0]))
                opt_noise = float(gp.noise)
                if opt_scale / max(opt_noise, 1e-12) < 0.1:
                    gp_kernel_params = np.array([0.0])
                    gp_noise = 1e-2
                else:
                    gp_kernel_params = gp.kernel_params.copy()
                    gp_noise = gp.noise
                gp_scale_disp = float(np.exp(gp_kernel_params[0]))
                gp_noise_disp = gp_noise

                log_mu_norm, log_std_norm = gp.predict(F_remaining, return_std=True,
                                                       include_noise=False)

                acq_vals = self.lcb(log_mu_norm, log_std_norm)

                selected_idx = int(np.argmin(acq_vals))
                acq_disp = float(acq_vals[selected_idx])

            _t = time.time()
            new_score, new_params = self.decompose(remaining_circs[selected_idx])
            _decompose_time += time.time() - _t
            _decompose_count += 1

            if new_score < best_score:
                best_score = new_score
                best_circ = remaining_circs[selected_idx]
                best_params = new_params
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            lines = [
                f"Iteration {itr}:",
                f"  acquisition={acquisition}, acq_val={acq_disp:.6f}",
            ]
            if acquisition == 'LCB':
                lines.append(f"  GP: scale={gp_scale_disp:.4f}, noise={gp_noise_disp:.6f}")
            lines += [
                f"  Actual score: {new_score:.6f}",
                f"  Best score so far: {best_score:.6f} "
                f"({len(X)+1} evaluated, {len(remaining_circs)-1} remaining)",
                f"  y stats: min={np.min(y):.4f}, mean={np.mean(y):.4f}, "
                f"std={np.std(y):.4f}",
            ]
            for line in lines:
                print(line)
            with open(log_file, 'a') as f:
                f.write("\n".join(lines) + "\n")

            if new_score < tolerance:
                msg = f"Success at iteration: {itr}"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
                self.best_score = new_score
                self.best_params = new_params
                self.best_circ = remaining_circs[selected_idx]
                self._decompose_time = _decompose_time
                self._decompose_count = _decompose_count
                self._total_search_time = time.time() - _search_t0
                return remaining_circs[selected_idx]

            if iters_since_improvement >= patience:
                msg = f"Stagnation exit at iteration {itr} (no improvement for {patience} iters)"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
                break

            y = np.append(y, new_score)
            X.append(remaining_circs[selected_idx])
            remaining_circs.pop(selected_idx)
        self.best_score = best_score
        self.best_params = best_params
        self.best_circ = best_circ
        self._decompose_time = _decompose_time
        self._decompose_count = _decompose_count
        self._total_search_time = time.time() - _search_t0
        print("Solution not found")
        return best_circ

    def search_over_D_evolve(self, log_file="sursearch_diagnostics.txt"):
        """Surrogate-assisted evolutionary search for large-D spaces.

        Uses evolutionary operators (mutation, crossover, random generation)
        to propose candidate circuits, then screens them with a GP surrogate
        using the SSK kernel. Follows the BOSS approach (Moss et al., 2020).
        """
        edges = sorted(set(tuple(e) for e in self.topology))
        vertices = {v for edge in edges for v in edge}
        edge_masks = precompute_edge_masks(edges, vertices)
        thresholds = precompute_thresholds(len(vertices))
        edge_to_mask = {e: m for e, m in zip(edges, edge_masks)}

        tolerance = self.config.get('tolerance', 1e-8)
        max_iters = self.config.get('max_sur_iters', 450)
        patience = self.config.get('patience', max(30, max_iters // 3))
        X0_size = self.config.get('X0_size', 10)
        candidates_per_iter = self.config.get('candidates_per_iter', 100)
        tournament_size = self.config.get('tournament_size', 3)
        block_size = self.config.get('block_mutation_size', 3)

        gap_decay = self.config.get('ssk_gap_decay', 0.8)
        match_decay = self.config.get('ssk_match_decay', 0.8)
        ssk_order = self.config.get('ssk_order', 3)

        ssk_cache = SSKCache(gap_decay, match_decay, ssk_order)
        seen = set()

        # Timing and call counters
        _decompose_time = 0.0
        _decompose_count = 0
        _search_t0 = time.time()

        # Initial random sample
        X = []
        while len(X) < X0_size:
            seq = generate_valid_sequence(self.D, edges, edge_masks, thresholds,
                                          edge_to_mask)
            if seq is not None and seq not in seen:
                seen.add(seq)
                ssk_cache.register(seq)
                X.append(seq)

        _t = time.time()
        _init_results = [self.decompose(x) for x in X]
        _decompose_time += time.time() - _t
        _decompose_count += len(X)
        y = np.array([r[0] for r in _init_results])
        _init_params = [r[1] for r in _init_results]
        best_idx = int(np.argmin(y))
        best_score = float(y[best_idx])
        best_circ = X[best_idx]
        best_params = _init_params[best_idx]

        with open(log_file, 'w') as f:
            f.write(f"SurSearch (evolutionary) diagnostics: N={self.N}, D={self.D}, "
                    f"kappa={self.kappa}, "
                    f"topology={self.topology}\n")
            f.write(f"Mode: evolutionary, initial sample: {X0_size}, "
                    f"candidates_per_iter: {candidates_per_iter}\n")
            f.write(f"Initial y stats: min={np.min(y):.6f}, "
                    f"mean={np.mean(y):.4f}, std={np.std(y):.4f}\n")
            f.write("-" * 70 + "\n")

        if best_score < tolerance:
            print("Success at random selection")
            with open(log_file, 'a') as f:
                f.write("Success at random selection\n")
            self.best_score = best_score
            self.best_params = best_params
            self.best_circ = best_circ
            self._decompose_time = _decompose_time
            self._decompose_count = _decompose_count
            self._total_search_time = time.time() - _search_t0
            return best_circ

        # GP setup (SSK kernel — single hyperparameter: log_scale)
        gp_kernel_params = np.array([0.0])
        gp_noise = 1e-2
        gp_bounds = [(-3.0, 3.0)]
        gp_noise_bounds = (-8.0, -1.0)

        def _ssk_kernel_fn(F1, F2, params):
            scale = float(np.exp(params[0]))
            idx1 = F1[:, 0].astype(int)
            idx2 = F2[:, 0].astype(int)
            return ssk_cache.kernel_matrix(idx1, idx2, scale)

        local_search_fraction = self.config.get('local_search_fraction', 0.5)
        max_local_steps = self.config.get('max_local_steps', 10)

        iters_since_improvement = 0
        for itr in range(max_iters):
            # 1. Fit GP first (so local search can use it for directed generation)
            F_train = np.array([[float(ssk_cache.circuit_to_idx[tuple(x)])]
                                for x in X])

            nonzero_y = y[y > 0]
            dynamic_floor = (float(nonzero_y.min()) * 0.01
                             if len(nonzero_y) > 0 else tolerance * 0.01)
            y_clamped = np.maximum(y, dynamic_floor)
            log_y = np.log10(y_clamped)
            lmu = float(log_y.mean())
            lsig = float(max(log_y.std(), 1e-6))
            log_y_norm = (log_y - lmu) / lsig

            gp = GPRegressor(
                kernel_params=gp_kernel_params.copy(),
                noise=gp_noise,
                kernel_fn=_ssk_kernel_fn,
            ).fit(F_train, log_y_norm)

            n_restarts = max(1, 3 - len(X) // 10)
            try:
                gp.optimize_hyperparameters(n_restarts=n_restarts,
                                            kernel_bounds=gp_bounds,
                                            noise_bounds=gp_noise_bounds)
            except Exception:
                pass

            # Collapse detection
            opt_scale = float(np.exp(gp.kernel_params[0]))
            opt_noise = float(gp.noise)
            if opt_scale / max(opt_noise, 1e-12) < 0.1:
                gp_kernel_params = np.array([0.0])
                gp_noise = 1e-2
            else:
                gp_kernel_params = gp.kernel_params.copy()
                gp_noise = gp.noise
            gp_scale_disp = float(np.exp(gp_kernel_params[0]))
            gp_noise_disp = gp_noise

            scale = float(np.exp(gp.kernel_params[0]))

            # 2. Generate candidates via hybrid approach (GP-directed + random)
            candidates, n_local, avg_local_steps = _generate_candidates(
                X, y, candidates_per_iter, edges, edge_masks, thresholds,
                edge_to_mask, seen,
                ssk_cache=ssk_cache, gp=gp, scale=scale, kappa=self.kappa,
                tournament_size=tournament_size, block_size=block_size,
                local_search_fraction=local_search_fraction,
                max_local_steps=max_local_steps)

            if not candidates:
                print("Cannot generate new candidates")
                break

            # 3. Screen all candidates with GP
            Ks = ssk_cache.compute_cross_kernel(candidates, scale)
            log_mu_norm = Ks.T @ gp.alpha
            v = np.linalg.solve(gp.L, Ks)
            var = np.maximum(scale - np.sum(v * v, axis=0), 0.0)
            log_std_norm = np.sqrt(var)

            # Thompson Sampling batch selection
            n_ts_samples = self.config.get('n_thompson_samples', 10)
            diversity_thresh = self.config.get('topk_diversity_threshold', 0.95)

            # Posterior covariance at candidate points
            n_cand = len(candidates)
            K_cand = scale * ssk_gram_matrix(
                candidates, gap_decay, match_decay, ssk_order)
            cov_post = K_cand - v.T @ v

            # Cholesky with jitter for numerical stability
            jitter = 1e-6
            L_post = None
            for _ in range(10):
                try:
                    L_post = np.linalg.cholesky(
                        cov_post + jitter * np.eye(n_cand))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            if L_post is None:
                L_post = np.diag(log_std_norm)

            # Draw posterior samples and pick minimizers
            mu_cand = log_mu_norm.ravel()
            selected_indices = []
            selected_set = set()
            for _ in range(n_ts_samples):
                z = np.random.randn(n_cand)
                f_sample = mu_cand + L_post @ z
                for sidx in np.argsort(f_sample):
                    sidx = int(sidx)
                    if sidx in selected_set:
                        continue
                    too_similar = (
                        any(ssk_cache.kernel_between(
                                candidates[sidx], candidates[p])
                            > diversity_thresh for p in selected_indices)
                        if selected_indices and diversity_thresh < 1.0
                        else False
                    )
                    if not too_similar:
                        selected_indices.append(sidx)
                        selected_set.add(sidx)
                        break

            acq_disp = float(
                mu_cand[selected_indices[0]]
                - self.kappa * log_std_norm[selected_indices[0]]
            ) if selected_indices else float('nan')
            selected_from_local = sum(
                1 for si in selected_indices if si < n_local)

            # 4. Evaluate selected candidates
            improved_this_iter = False
            for sel_idx in selected_indices:
                _t = time.time()
                new_score, new_params = self.decompose(candidates[sel_idx])
                _decompose_time += time.time() - _t
                _decompose_count += 1

                ssk_cache.register(candidates[sel_idx])
                y = np.append(y, new_score)
                X.append(candidates[sel_idx])

                if new_score < best_score:
                    best_score = new_score
                    best_circ = candidates[sel_idx]
                    best_params = new_params
                    improved_this_iter = True

                if new_score < tolerance:
                    msg = f"Success at iteration: {itr}"
                    print(msg)
                    with open(log_file, 'a') as f:
                        f.write(msg + "\n")
                    self.best_score = new_score
                    self.best_params = new_params
                    self.best_circ = candidates[sel_idx]
                    self._decompose_time = _decompose_time
                    self._decompose_count = _decompose_count
                    self._total_search_time = time.time() - _search_t0
                    return candidates[sel_idx]

            if improved_this_iter:
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            lines = [
                f"Iteration {itr}:",
                f"  acq_val={acq_disp:.6f}",
                f"  GP: scale={gp_scale_disp:.4f}, noise={gp_noise_disp:.6f}",
                f"  Evaluated {len(selected_indices)} candidate(s) "
                f"(Thompson Sampling, {n_ts_samples} draws)",
                f"  Best score so far: {best_score:.6f} "
                f"({len(X)} evaluated, {len(candidates)} candidates generated)",
                f"  y stats: min={np.min(y):.4f}, mean={np.mean(y):.4f}, "
                f"std={np.std(y):.4f}",
                f"  Local search: {n_local} candidates, "
                f"avg {avg_local_steps:.1f} steps, "
                f"{selected_from_local}/{len(selected_indices)} selected",
            ]
            for line in lines:
                print(line)
            with open(log_file, 'a') as f:
                f.write("\n".join(lines) + "\n")

            if iters_since_improvement >= patience:
                msg = f"Stagnation exit at iteration {itr} (no improvement for {patience} iters)"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
                break

        self.best_score = best_score
        self.best_params = best_params
        self.best_circ = best_circ
        self._decompose_time = _decompose_time
        self._decompose_count = _decompose_count
        self._total_search_time = time.time() - _search_t0
        print("Solution not found")
        return best_circ

    def search_over_D_range(self, D_min, D_max, log_file="sursearch_cross_d.txt"):
        """Unified cross-D surrogate search from D_min to D_max.

        Searches over all circuit depths simultaneously using a single GP
        with variable-length SSK kernel. Biased toward lower D via penalty
        term in acquisition to find the most compact solution.
        """
        edges = sorted(set(tuple(e) for e in self.topology))
        vertices = {v for edge in edges for v in edge}
        edge_masks = precompute_edge_masks(edges, vertices)
        thresholds = precompute_thresholds(len(vertices))
        edge_to_mask = {e: m for e, m in zip(edges, edge_masks)}

        tolerance = self.config.get('tolerance', 1e-8)
        max_iters = self.config.get('max_sur_iters', 450)
        patience = self.config.get('patience', max(30, max_iters // 3))
        X0_size = self.config.get('X0_size', 10)
        candidates_per_iter = self.config.get('candidates_per_iter', 100)
        tournament_size = self.config.get('tournament_size', 3)
        block_size = self.config.get('block_mutation_size', 3)

        gap_decay = self.config.get('ssk_gap_decay', 0.8)
        match_decay = self.config.get('ssk_match_decay', 0.8)
        ssk_order = self.config.get('ssk_order', 3)
        d_penalty = self.config.get('d_penalty', 0.1)

        ssk_cache = SSKCache(gap_decay, match_decay, ssk_order)
        seen = set()

        _decompose_time = 0.0
        _decompose_count = 0
        _search_t0 = time.time()

        # Initial random sample — round-robin across D values
        X = []
        D_range = list(range(D_min, D_max + 1))
        attempts = 0
        while len(X) < X0_size and attempts < X0_size * 100:
            D_val = D_range[len(X) % len(D_range)]
            seq = generate_valid_sequence(D_val, edges, edge_masks, thresholds,
                                          edge_to_mask)
            attempts += 1
            if seq is not None and seq not in seen:
                seen.add(seq)
                ssk_cache.register(seq)
                X.append(seq)

        _t = time.time()
        _init_results = [self.decompose(x) for x in X]
        _decompose_time += time.time() - _t
        _decompose_count += len(X)
        y = np.array([r[0] for r in _init_results])
        _init_params = [r[1] for r in _init_results]
        best_idx = int(np.argmin(y))
        best_score = float(y[best_idx])
        best_circ = X[best_idx]
        best_params = _init_params[best_idx]

        with open(log_file, 'w') as f:
            f.write(f"SurSearch (cross-D) diagnostics: N={self.N}, "
                    f"D={D_min}-{D_max}, kappa={self.kappa}, "
                    f"d_penalty={d_penalty}, topology={self.topology}\n")
            f.write(f"Mode: cross-D evolutionary, initial sample: {X0_size}, "
                    f"candidates_per_iter: {candidates_per_iter}\n")
            f.write(f"Initial y stats: min={np.min(y):.6f}, "
                    f"mean={np.mean(y):.4f}, std={np.std(y):.4f}\n")
            f.write("-" * 70 + "\n")

        if best_score < tolerance:
            print(f"Success at random selection (D={len(best_circ)})")
            with open(log_file, 'a') as f:
                f.write(f"Success at random selection (D={len(best_circ)})\n")
            self.best_score = best_score
            self.best_params = best_params
            self.best_circ = best_circ
            self._decompose_time = _decompose_time
            self._decompose_count = _decompose_count
            self._total_search_time = time.time() - _search_t0
            return best_circ

        # GP setup (SSK kernel — single hyperparameter: log_scale)
        gp_kernel_params = np.array([0.0])
        gp_noise = 1e-2
        gp_bounds = [(-3.0, 3.0)]
        gp_noise_bounds = (-8.0, -1.0)

        def _ssk_kernel_fn(F1, F2, params):
            scale = float(np.exp(params[0]))
            idx1 = F1[:, 0].astype(int)
            idx2 = F2[:, 0].astype(int)
            return ssk_cache.kernel_matrix(idx1, idx2, scale)

        local_search_fraction = self.config.get('local_search_fraction', 0.5)
        max_local_steps = self.config.get('max_local_steps', 10)

        iters_since_improvement = 0
        for itr in range(max_iters):
            # 1. Fit GP
            F_train = np.array([[float(ssk_cache.circuit_to_idx[tuple(x)])]
                                for x in X])

            nonzero_y = y[y > 0]
            dynamic_floor = (float(nonzero_y.min()) * 0.01
                             if len(nonzero_y) > 0 else tolerance * 0.01)
            y_clamped = np.maximum(y, dynamic_floor)
            log_y = np.log10(y_clamped)
            lmu = float(log_y.mean())
            lsig = float(max(log_y.std(), 1e-6))
            log_y_norm = (log_y - lmu) / lsig

            gp = GPRegressor(
                kernel_params=gp_kernel_params.copy(),
                noise=gp_noise,
                kernel_fn=_ssk_kernel_fn,
            ).fit(F_train, log_y_norm)

            n_restarts = max(1, 3 - len(X) // 10)
            try:
                gp.optimize_hyperparameters(n_restarts=n_restarts,
                                            kernel_bounds=gp_bounds,
                                            noise_bounds=gp_noise_bounds)
            except Exception:
                pass

            opt_scale = float(np.exp(gp.kernel_params[0]))
            opt_noise = float(gp.noise)
            if opt_scale / max(opt_noise, 1e-12) < 0.1:
                gp_kernel_params = np.array([0.0])
                gp_noise = 1e-2
            else:
                gp_kernel_params = gp.kernel_params.copy()
                gp_noise = gp.noise
            gp_scale_disp = float(np.exp(gp_kernel_params[0]))
            gp_noise_disp = gp_noise

            scale = float(np.exp(gp.kernel_params[0]))

            # 2. Generate candidates with grow/shrink operators
            candidates, n_local, avg_local_steps = _generate_candidates(
                X, y, candidates_per_iter, edges, edge_masks, thresholds,
                edge_to_mask, seen,
                ssk_cache=ssk_cache, gp=gp, scale=scale, kappa=self.kappa,
                tournament_size=tournament_size, block_size=block_size,
                local_search_fraction=local_search_fraction,
                max_local_steps=max_local_steps,
                D_min=D_min, D_max=D_max)

            if not candidates:
                print("Cannot generate new candidates")
                break

            # 3. Screen candidates with GP
            Ks = ssk_cache.compute_cross_kernel(candidates, scale)
            log_mu_norm = Ks.T @ gp.alpha
            v = np.linalg.solve(gp.L, Ks)
            var = np.maximum(scale - np.sum(v * v, axis=0), 0.0)
            log_std_norm = np.sqrt(var)

            # Thompson Sampling batch selection with D-penalty
            n_ts_samples = self.config.get('n_thompson_samples', 10)
            diversity_thresh = self.config.get('topk_diversity_threshold', 0.95)

            n_cand = len(candidates)
            K_cand = scale * ssk_gram_matrix(
                candidates, gap_decay, match_decay, ssk_order)
            cov_post = K_cand - v.T @ v

            jitter = 1e-6
            L_post = None
            for _ in range(10):
                try:
                    L_post = np.linalg.cholesky(
                        cov_post + jitter * np.eye(n_cand))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            if L_post is None:
                L_post = np.diag(log_std_norm)

            # Apply D-penalty to posterior mean to bias toward lower D
            mu_cand = log_mu_norm.ravel().copy()
            if d_penalty > 0:
                cand_D_vals = np.array([len(c) for c in candidates],
                                       dtype=np.float64)
                mu_cand += d_penalty * (cand_D_vals / D_max)

            selected_indices = []
            selected_set = set()
            for _ in range(n_ts_samples):
                z = np.random.randn(n_cand)
                f_sample = mu_cand + L_post @ z
                for sidx in np.argsort(f_sample):
                    sidx = int(sidx)
                    if sidx in selected_set:
                        continue
                    too_similar = (
                        any(ssk_cache.kernel_between(
                                candidates[sidx], candidates[p])
                            > diversity_thresh for p in selected_indices)
                        if selected_indices and diversity_thresh < 1.0
                        else False
                    )
                    if not too_similar:
                        selected_indices.append(sidx)
                        selected_set.add(sidx)
                        break

            acq_disp = float(
                log_mu_norm.ravel()[selected_indices[0]]
                - self.kappa * log_std_norm[selected_indices[0]]
            ) if selected_indices else float('nan')
            selected_from_local = sum(
                1 for si in selected_indices if si < n_local)

            # 4. Evaluate selected candidates
            improved_this_iter = False
            for sel_idx in selected_indices:
                _t = time.time()
                new_score, new_params = self.decompose(candidates[sel_idx])
                _decompose_time += time.time() - _t
                _decompose_count += 1

                ssk_cache.register(candidates[sel_idx])
                y = np.append(y, new_score)
                X.append(candidates[sel_idx])

                if new_score < best_score:
                    best_score = new_score
                    best_circ = candidates[sel_idx]
                    best_params = new_params
                    improved_this_iter = True

                if new_score < tolerance:
                    D_found = len(candidates[sel_idx])
                    msg = f"Success at iteration: {itr} (D={D_found})"
                    print(msg)
                    with open(log_file, 'a') as f:
                        f.write(msg + "\n")
                    self.best_score = new_score
                    self.best_params = new_params
                    self.best_circ = candidates[sel_idx]
                    self._decompose_time = _decompose_time
                    self._decompose_count = _decompose_count
                    self._total_search_time = time.time() - _search_t0
                    return candidates[sel_idx]

            if improved_this_iter:
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            # D distribution stats
            cand_d_counts = {}
            for c in candidates:
                d = len(c)
                cand_d_counts[d] = cand_d_counts.get(d, 0) + 1
            d_dist_str = ", ".join(
                f"D{d}:{n}" for d, n in sorted(cand_d_counts.items()))

            lines = [
                f"Iteration {itr}:",
                f"  acq_val={acq_disp:.6f}",
                f"  GP: scale={gp_scale_disp:.4f}, noise={gp_noise_disp:.6f}",
                f"  Evaluated {len(selected_indices)} candidate(s) "
                f"(Thompson Sampling, {n_ts_samples} draws)",
                f"  Best score so far: {best_score:.6f} (D={len(best_circ)})"
                f" ({len(X)} evaluated, {len(candidates)} candidates)",
                f"  y stats: min={np.min(y):.4f}, mean={np.mean(y):.4f}, "
                f"std={np.std(y):.4f}",
                f"  Local search: {n_local} candidates, "
                f"avg {avg_local_steps:.1f} steps, "
                f"{selected_from_local}/{len(selected_indices)} selected",
                f"  D distribution: {d_dist_str}",
            ]
            for line in lines:
                print(line)
            with open(log_file, 'a') as f:
                f.write("\n".join(lines) + "\n")

            if iters_since_improvement >= patience:
                msg = (f"Stagnation exit at iteration {itr} "
                       f"(no improvement for {patience} iters)")
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
                break

        self.best_score = best_score
        self.best_params = best_params
        self.best_circ = best_circ
        self._decompose_time = _decompose_time
        self._decompose_count = _decompose_count
        self._total_search_time = time.time() - _search_t0
        print("Solution not found")
        return best_circ

    def Start_Decomposition(self, D_start, D_end, log_file_prefix="sursearch"):
        return self.search_over_D_range(
            D_start, D_end,
            log_file=f"{log_file_prefix}_D{D_start}-{D_end}.txt")

    def decompose(self, x):
        optimizer = self.config.get('optimizer', 'BFGS')
        ansatz = create_circuit_from_edges(x, self.N)
        cDecomp = N_Qubit_Decomposition_custom(self.Umtx.conj().T, config=self.config)
        cDecomp.set_Verbose(0)
        cDecomp.set_Cost_Function_Variant(3)
        cDecomp.set_Gate_Structure(ansatz)
        cDecomp.set_Optimized_Parameters(np.random.rand(ansatz.get_Parameter_Num()) * 2 * np.pi)
        cDecomp.set_Optimizer(optimizer)
        cDecomp.Start_Decomposition()
        params = cDecomp.get_Optimized_Parameters()
        return cDecomp.Optimization_Problem(params), params

    def lcb(self, mu, std):
        return mu - self.kappa * std

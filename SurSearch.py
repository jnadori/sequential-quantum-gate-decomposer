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


def _generate_candidates(population, scores, n_candidates, edges, edge_masks,
                         thresholds, edge_to_mask, seen, tournament_size=3,
                         block_size=3):
    candidates = []
    n_pop = len(population)
    D = len(population[0])

    def tournament_select():
        indices = np.random.choice(n_pop, size=min(tournament_size, n_pop), replace=False)
        best_idx = indices[np.argmin(scores[indices])]
        return population[best_idx]

    max_total_attempts = n_candidates * 10
    attempts = 0

    while len(candidates) < n_candidates and attempts < max_total_attempts:
        attempts += 1
        r = np.random.random()

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

    return candidates


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


def ssk_gram_matrix(circuits, gap_decay, match_decay, order):
    """Compute the full SSK Gram matrix for a list of equal-length circuits.

    Returns an (n_circuits, n_circuits) symmetric positive-semidefinite matrix
    and its row-normalized version (kernel values divided by sqrt of diagonals).

    Uses batched numpy operations over all upper-triangle pairs to avoid
    Python-level nested loops.
    """
    n_circuits = len(circuits)
    n = len(circuits[0])
    match_sq = match_decay * match_decay

    # Gap decay matrix (strictly upper triangular)
    idx = np.arange(n)
    D = np.where(idx[None, :] > idx[:, None],
                 gap_decay ** (idx[None, :] - idx[:, None] - 1), 0.0)

    # Integer-encode circuits for vectorized comparison
    ci_arr, _ = _circuits_to_int_array(circuits)

    # All upper-triangle pairs (including diagonal)
    ii, jj = np.triu_indices(n_circuits)
    P = len(ii)

    # Batched S matrices: S[p, a, b] = 1 if circuit_ii[p][a] == circuit_jj[p][b]
    ci = ci_arr[ii]  # (P, n)
    cj = ci_arr[jj]  # (P, n)
    S = (ci[:, :, None] == cj[:, None, :]).astype(np.float64)  # (P, n, n)

    # Batched DP over subsequence orders
    Kp = np.ones((P, n, n))
    k = match_sq * np.sum(S * Kp, axis=(1, 2))  # (P,)

    for _ in range(order - 1):
        Kpp = match_sq * ((S * Kp) @ D)              # (P, n, n)
        Kp = (Kpp.transpose(0, 2, 1) @ D).transpose(0, 2, 1)
        k += match_sq * np.sum(S * Kp, axis=(1, 2))

    # Assemble symmetric Gram matrix
    K = np.zeros((n_circuits, n_circuits))
    K[ii, jj] = k
    K[jj, ii] = k

    # Normalise: K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])
    diag = np.sqrt(np.diag(K))
    diag[diag < 1e-12] = 1.0
    K_norm = K / (diag[:, None] * diag[None, :])
    return K_norm


class SSKCache:
    """Incremental SSK kernel cache for evolutionary search.

    Computes and caches pairwise SSK kernel values on demand, avoiding
    the need to precompute the full Gram matrix over all circuits.
    Maintains a dense normalized kernel matrix for fast slicing.
    """

    def __init__(self, gap_decay, match_decay, order, seq_len):
        self.match_sq = match_decay * match_decay
        self.order = order
        self.seq_len = seq_len
        self.circuits = []
        self.circuit_to_idx = {}
        idx = np.arange(seq_len)
        self.D_matrix = np.where(
            idx[None, :] > idx[:, None],
            gap_decay ** (idx[None, :] - idx[:, None] - 1), 0.0)
        self._K_norm = np.empty((0, 0))
        self._raw_diags = np.empty(0)
        # For vectorized kernel computation
        self._circuits_int = np.empty((0, seq_len), dtype=np.int32)
        self._edge_to_id = {}
        self._next_id = 0

    def register(self, circuit):
        key = tuple(circuit)
        if key in self.circuit_to_idx:
            return self.circuit_to_idx[key]
        new_idx = len(self.circuits)
        self.circuits.append(key)
        self.circuit_to_idx[key] = new_idx

        # Encode new circuit as int array
        n = self.seq_len
        new_int = np.empty(n, dtype=np.int32)
        for j, edge in enumerate(key):
            if edge not in self._edge_to_id:
                self._edge_to_id[edge] = self._next_id
                self._next_id += 1
            new_int[j] = self._edge_to_id[edge]

        # Grow the int array
        new_circuits_int = np.empty((new_idx + 1, n), dtype=np.int32)
        if new_idx > 0:
            new_circuits_int[:new_idx] = self._circuits_int
        new_circuits_int[new_idx] = new_int
        self._circuits_int = new_circuits_int

        # Batch-compute SSK kernel between new circuit and all existing
        P = new_idx + 1
        D = self.D_matrix
        match_sq = self.match_sq

        ci = np.broadcast_to(new_int[None, :], (P, n))   # new circuit repeated
        cj = self._circuits_int                            # all circuits
        S = (ci[:, :, None] == cj[:, None, :]).astype(np.float64)  # (P, n, n)

        Kp = np.ones((P, n, n))
        raw_new = match_sq * np.sum(S * Kp, axis=(1, 2))

        for _ in range(self.order - 1):
            Kpp = match_sq * ((S * Kp) @ D)
            Kp = (Kpp.transpose(0, 2, 1) @ D).transpose(0, 2, 1)
            raw_new += match_sq * np.sum(S * Kp, axis=(1, 2))

        # Grow raw diagonals array
        raw_diags = np.empty(new_idx + 1)
        if new_idx > 0:
            raw_diags[:new_idx] = self._raw_diags
        raw_diags[new_idx] = raw_new[new_idx]
        self._raw_diags = raw_diags

        # Expand the normalized kernel matrix by one row/column
        K_new = np.empty((new_idx + 1, new_idx + 1))
        if new_idx > 0:
            K_new[:new_idx, :new_idx] = self._K_norm

        diag_new = np.sqrt(max(raw_new[new_idx], 1e-24))
        diag_all = np.sqrt(np.maximum(self._raw_diags, 1e-24))
        norm_row = raw_new / (diag_new * diag_all)
        K_new[new_idx, :] = norm_row
        K_new[:, new_idx] = norm_row

        self._K_norm = K_new
        return new_idx

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

        ssk_cache = SSKCache(gap_decay, match_decay, ssk_order, self.D)
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

        iters_since_improvement = 0
        for itr in range(max_iters):
            # Generate candidates via evolutionary operators
            candidates = _generate_candidates(
                X, y, candidates_per_iter, edges, edge_masks, thresholds,
                edge_to_mask, seen, tournament_size=tournament_size,
                block_size=block_size)

            if not candidates:
                print("Cannot generate new candidates")
                break

            # Register candidates in SSK cache
            for c in candidates:
                ssk_cache.register(c)

            # Build feature matrices (indices into SSK cache)
            F_train = np.array([[float(ssk_cache.circuit_to_idx[tuple(x)])]
                                for x in X])
            F_cand = np.array([[float(ssk_cache.circuit_to_idx[tuple(c)])]
                               for c in candidates])

            # Dynamic floor + log transform (same as search_over_D)
            nonzero_y = y[y > 0]
            dynamic_floor = (float(nonzero_y.min()) * 0.01
                             if len(nonzero_y) > 0 else tolerance * 0.01)
            y_clamped = np.maximum(y, dynamic_floor)
            log_y = np.log10(y_clamped)
            lmu = float(log_y.mean())
            lsig = float(max(log_y.std(), 1e-6))
            log_y_norm = (log_y - lmu) / lsig

            # Fit GP
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

            # Acquisition function
            log_mu_norm, log_std_norm = gp.predict(F_cand, return_std=True,
                                                    include_noise=False)

            acq_vals = self.lcb(log_mu_norm, log_std_norm)

            selected_idx = int(np.argmin(acq_vals))
            acq_disp = float(acq_vals[selected_idx])

            # Evaluate best candidate
            _t = time.time()
            new_score, new_params = self.decompose(candidates[selected_idx])
            _decompose_time += time.time() - _t
            _decompose_count += 1

            if new_score < best_score:
                best_score = new_score
                best_circ = candidates[selected_idx]
                best_params = new_params
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            lines = [
                f"Iteration {itr}:",
                f"  acq_val={acq_disp:.6f}",
                f"  GP: scale={gp_scale_disp:.4f}, noise={gp_noise_disp:.6f}",
                f"  Actual score: {new_score:.6f}",
                f"  Best score so far: {best_score:.6f} "
                f"({len(X)+1} evaluated, {len(candidates)} candidates generated)",
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
                self.best_circ = candidates[selected_idx]
                self._decompose_time = _decompose_time
                self._decompose_count = _decompose_count
                self._total_search_time = time.time() - _search_t0
                return candidates[selected_idx]

            if iters_since_improvement >= patience:
                msg = f"Stagnation exit at iteration {itr} (no improvement for {patience} iters)"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + "\n")
                break

            y = np.append(y, new_score)
            X.append(candidates[selected_idx])

        self.best_score = best_score
        self.best_params = best_params
        self.best_circ = best_circ
        self._decompose_time = _decompose_time
        self._decompose_count = _decompose_count
        self._total_search_time = time.time() - _search_t0
        print("Solution not found")
        return best_circ

    def Start_Decomposition(self, D_start, D_end, log_file_prefix="sursearch"):
        edges = sorted(set(tuple(e) for e in self.topology))
        enum_threshold = self.config.get('enum_threshold', 10000)
        tolerance = self.config.get('tolerance', 1e-8)

        for D in range(D_start, D_end + 1):
            self.D = D

            # Estimate circuit count for iteration budget
            if len(edges) ** D <= enum_threshold:
                n_circuits = len(unique_k_sequences(self.topology, D))
            else:
                n_circuits = len(edges) ** D

            max_iters = min(n_circuits // 3, 450)
            self.config['max_sur_iters'] = max(max_iters, 1)

            log_file = f"{log_file_prefix}_D{D}.txt"
            print(f"--- D={D}: ~{n_circuits} circuits, max_iters={self.config['max_sur_iters']} ---")
            self.search_over_D(log_file=log_file)

            if self.best_score < tolerance:
                print(f"Solution found at D={D}")
                return self.best_circ

        print(f"Solution not found up to D={D_end}")
        return self.best_circ

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

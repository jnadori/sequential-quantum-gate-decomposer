from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool
from squander import N_Qubit_Decomposition_custom, Circuit
from squander.synthesis.qgd_POSMM import qgd_POSMM_Decomposition

from math import ceil
import heapq



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

def pairwise_kernel_matrix(
    F1: Array,
    F2: Array,
    params: Array,
) -> Array:
    """RBF kernel matrix from precomputed feature matrices.

    Args:
        F1: (n1, d) feature matrix
        F2: (n2, d) feature matrix
        params: [log_scale, log_lengthscale]
    """
    from scipy.spatial.distance import cdist
    scale = float(np.exp(params[0]))
    lengthscale = float(np.exp(params[1]))
    sq_dists = cdist(F1, F2, 'sqeuclidean')
    return scale * np.exp(-sq_dists / (2 * lengthscale ** 2))


@dataclass
class GPRegressor:
    kernel_params: Array
    noise: float = 1e-3
    jitter: float = 1e-8

    # Learned during fit:
    F_train: Optional[Array] = None
    y_train: Optional[Array] = None
    L: Optional[Array] = None
    alpha: Optional[Array] = None

    def fit(self, F: Array, y: Array) -> "GPRegressor":
        """Fit GP on precomputed feature matrix F (n, d) and targets y."""
        y = np.asarray(y, dtype=float).reshape(-1)
        F = np.asarray(F, dtype=float)
        if F.shape[0] != y.shape[0]:
            raise ValueError("F.shape[0] must equal len(y).")

        self.F_train = F
        self.y_train = y

        K = pairwise_kernel_matrix(F, F, self.kernel_params)
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
        Ks = pairwise_kernel_matrix(self.F_train, F_star, self.kernel_params)
        mean = Ks.T @ self.alpha  # (n_star,)

        if not return_std:
            return mean, None

        # RBF self-kernel is always scale
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

        K = pairwise_kernel_matrix(F, F, self.kernel_params)
        K = self._add_diagonal_noise(K, self.noise, self.jitter)

        L = np.linalg.cholesky(K)
        alpha = self._solve_cholesky(L, y)

        n = y.shape[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        quad = float(y.T @ alpha)
        return -0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)

    def optimize_hyperparameters(self, n_restarts: int = 3,
                                  kernel_bounds: Optional[list] = None) -> "GPRegressor":
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
            bounds = list(kernel_bounds) + [(-8.0, 2.0)]
        else:
            bounds = [(-3.0, 3.0)] * n_kernel + [(-8.0, 2.0)]

        for _ in range(n_restarts):
            theta_init = theta0 + 0.5 * np.random.randn(len(theta0))
            theta_init = np.clip(theta_init, [b[0] for b in bounds], [b[1] for b in bounds])
            res = minimize(objective, theta_init, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-4})
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



def circuit_features(x, edges, n_qubits):
    """Hand-crafted sequence-level features for a circuit.

    Returns: edge frequency (E), bigram counts (E²), qubit activity (N),
    consecutive overlap (1), unique edges fraction (1).
    """
    E = len(edges)
    D = len(x)
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    edge_freq = np.zeros(E)
    for e in x:
        edge_freq[edge_to_idx[e]] += 1
    edge_freq /= D

    bigram = np.zeros(E * E)
    if D > 1:
        for k in range(D - 1):
            i = edge_to_idx[x[k]]
            j = edge_to_idx[x[k + 1]]
            bigram[i * E + j] += 1
        bigram /= (D - 1)

    qubit_act = np.zeros(n_qubits)
    for e in x:
        qubit_act[e[0]] += 1
        qubit_act[e[1]] += 1
    qubit_act /= (2 * D)

    consec = 0.0
    if D > 1:
        for k in range(D - 1):
            if set(x[k]) & set(x[k + 1]):
                consec += 1
        consec /= (D - 1)

    unique_frac = len(set(x)) / E if E > 0 else 0.0
    return np.concatenate([edge_freq, bigram, qubit_act, [consec, unique_frac]])



def decompose_old(Umtx, x):
    config = {'use_basin_hopping': 1, 'bh_T': 1.1375279022671254, 'bh_stepsize': 0.9200273804590016, 'bh_interval': 94, 'bh_target_accept_rate': 0.5661497388955112, 'bh_stepwise_factor': 0.5557762288919466}
    N = int(np.log2(Umtx.shape[0]))
    ansatz = create_circuit_from_edges(x,N)
    cDecomp = N_Qubit_Decomposition_custom(Umtx.conj().T,config=config)
    cDecomp.set_Verbose(0)
    cDecomp.set_Cost_Function_Variant(3)
    cDecomp.set_Gate_Structure(ansatz)
    cDecomp.set_Optimized_Parameters(np.random.rand(ansatz.get_Parameter_Num())*2*np.pi)
    cDecomp.set_Optimizer("BFGS2")
    cDecomp.Start_Decomposition()
    params = cDecomp.get_Optimized_Parameters()
    return cDecomp.Optimization_Problem(params)

def generate_goal_unitary(N, edges):
    goal = create_circuit_from_edges(edges,N)
    return goal.get_Matrix(np.random.rand(goal.get_Parameter_Num())*2*np.pi)

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

    def search_over_D(self, log_file="sursearch_diagnostics.txt", pool=None):
        from scipy.spatial.distance import pdist

        circuits = unique_k_sequences(self.topology, self.D)
        edges = sorted(set(tuple(e) for e in self.topology))

        # Hand-crafted features only (no WL) + z-score standardization
        raw = {tuple(c): circuit_features(c, edges, self.N) for c in circuits}
        all_feats = np.array(list(raw.values()))
        feat_mean = all_feats.mean(axis=0)
        feat_std = all_feats.std(axis=0)
        feat_std[feat_std < 1e-12] = 1.0
        feature_cache = {k: (v - feat_mean) / feat_std for k, v in raw.items()}

        np.random.shuffle(circuits)
        X = circuits[:self.config.get("X0_size", 20)]
        remaining_circs = circuits[self.config.get("X0_size", 20):]
        y = np.array([self.decompose(x, pool=pool) for x in X])
        max_iters = self.config.get('max_sur_iters', 1000)
        best_circ = X[np.argmin(y)]
        if np.min(y) < self.config.get('tolerance', 1e-8):
            print(f"Success at random selection")

        # Data-driven lengthscale initialization
        all_standardized = np.array(list(feature_cache.values()))
        median_dist = float(np.median(pdist(all_standardized, 'euclidean')))
        log_median = np.log(max(median_dist, 1e-6))

        feat_dim = all_feats.shape[1]
        with open(log_file, 'w') as f:
            f.write(f"SurSearch diagnostics: N={self.N}, D={self.D}, "
                    f"kappa={self.kappa}, topology={self.topology}\n")
            f.write(f"Total circuits: {len(circuits)}, "
                    f"initial sample: {len(X)}, "
                    f"feature dim: {feat_dim} (hand-crafted)\n")
            f.write(f"Initial y stats: min={np.min(y):.6f}, "
                    f"mean={np.mean(y):.4f}, std={np.std(y):.4f}\n")
            f.write(f"Median pairwise distance: {median_dist:.4f}, "
                    f"log_median: {log_median:.4f}\n")
            f.write("-" * 70 + "\n")

        best_score = np.min(y)
        _LOG_FLOOR = 1e-12
        # [log_scale, log_lengthscale] — RBF kernel, lengthscale from data
        gp_kernel_params = np.array([0.0, log_median])
        gp_noise = 1e-2
        # Adaptive bounds: lengthscale upper = log_median + 3
        gp_bounds = [(-3.0, 3.0), (log_median - 3.0, log_median + 3.0)]

        def _feature_matrix(circs):
            return np.array([feature_cache[tuple(c)] for c in circs])

        for itr in range(max_iters):
            F_train = _feature_matrix(X)
            F_remaining = _feature_matrix(remaining_circs)

            log_y = np.log10(np.maximum(y, _LOG_FLOOR))
            gp = GPRegressor(
                kernel_params=gp_kernel_params.copy(),
                noise=gp_noise,
            ).fit(F_train, log_y)
            gp.optimize_hyperparameters(n_restarts=2, kernel_bounds=gp_bounds)
            gp_kernel_params = gp.kernel_params.copy()
            gp_noise = gp.noise
            log_mu, log_std = gp.predict(F_remaining, return_std=True, include_noise=False)

            log_lcb_vals = self.lcb(log_mu, log_std)
            lcb_idx = np.argmin(log_lcb_vals)
            lcb_score = self.decompose(remaining_circs[lcb_idx], pool=pool)

            pred_score = 10 ** log_mu[lcb_idx]

            if lcb_score < best_score:
                best_score = lcb_score
                best_circ = remaining_circs[lcb_idx]

            # --- Diagnostics ---
            lines = [
                f"Iteration {itr}:",
                f"  GP hyperparams: scale={np.exp(gp.kernel_params[0]):.4f}, "
                f"lengthscale={np.exp(gp.kernel_params[1]):.4f}, noise={gp.noise:.6f}",
                f"  Log10-space: mu={log_mu[lcb_idx]:.4f}, std={log_std[lcb_idx]:.4f}, "
                f"LCB={log_lcb_vals[lcb_idx]:.4f}",
                f"  Predicted score: {pred_score:.6f}, actual score: {lcb_score:.6f}, "
                f"ratio: {pred_score / max(lcb_score, _LOG_FLOOR):.2f}x",
                f"  Best score so far: {best_score:.6f} "
                f"({len(X)+1} evaluated, {len(remaining_circs)-1} remaining)",
                f"  y stats: min={np.min(y):.4f}, mean={np.mean(y):.4f}, "
                f"std={np.std(y):.4f}",
            ]
            for line in lines:
                print(line)
            with open(log_file, 'a') as f:
                f.write("\n".join(lines) + "\n")
            # -------------------

            if lcb_score < self.config.get('tolerance', 1e-8):
                print(f"Success at iteration: {itr}")
                with open(log_file, 'a') as f:
                    f.write(f"Success at iteration: {itr}\n")
                return remaining_circs[lcb_idx]
            y = np.append(y, lcb_score)
            X.append(remaining_circs[lcb_idx])
            remaining_circs.pop(lcb_idx)
        print("Solution not found")
        return best_circ


    def decompose(self, x, pool=None):
        ansatz = create_circuit_from_edges(x, self.N)
        if self.config.get('optimizer') == 'POSMM':
            posmm = qgd_POSMM_Decomposition(
                self.Umtx.conj().T,
                ansatz,
                self.config.get('rconstant', 0.5),
                self.config,
                pool=pool,
            )
            _, score = posmm.Start_decomposition()
            return score
        n_restarts = self.config.get('decompose_restarts', 3)
        best = float('inf')
        for _ in range(n_restarts):
            cDecomp = N_Qubit_Decomposition_custom(self.Umtx.conj().T, config=self.config)
            cDecomp.set_Verbose(0)
            cDecomp.set_Cost_Function_Variant(3)
            cDecomp.set_Gate_Structure(ansatz)
            cDecomp.set_Optimized_Parameters(np.random.rand(ansatz.get_Parameter_Num()) * 2 * np.pi)
            cDecomp.set_Optimizer(self.config.get('optimizer', 'BFGS'))
            cDecomp.Start_Decomposition()
            params = cDecomp.get_Optimized_Parameters()
            score = cDecomp.Optimization_Problem(params)
            best = min(best, score)
        return best

    def lcb(self,mu,std):
        return mu - self.kappa*std


if __name__ == "__main__":
    N=3
    goal_edges = [(0,1),(1,2),(0,1),(0,2),(0,2),(1,2),(0,1)]
    Umtx = generate_goal_unitary(N, edges=goal_edges)
    config = {'use_basin_hopping': 1, 'bh_T': 1.1375279022671254, 'bh_stepsize': 0.9200273804590016, 'bh_interval': 94, 'bh_target_accept_rate': 0.5661497388955112,
              'bh_stepwise_factor': 0.5557762288919466,'optimizer':'BFGS2','parallel':2,'X0_size':50,'max_sur_iters':500, 'kappa':0.5}
    sursearch = qgd_SurSearch(Umtx,config,len(goal_edges))
    sursearch.search_over_D()

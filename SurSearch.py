from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from squander import N_Qubit_Decomposition_custom, Circuit

Array = np.ndarray
KernelFn = Callable[[Any, Any, Array], float]
# KernelFn signature:
#   k(x1, x2, params) -> scalar
# You can make x1/x2 be strings, tuples, numpy arrays, etc.
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
    X1: Sequence[Any],
    X2: Sequence[Any],
    kernel_fn: KernelFn,
    params: Array,
) -> Array:
    n1 = len(X1)
    n2 = len(X2)
    K = np.empty((n1, n2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            K[i, j] = float(kernel_fn(X1[i], X2[j], params))
    return K


@dataclass
class GPRegressor:
    kernel_fn: KernelFn
    kernel_params: Array
    noise: float = 1e-3
    jitter: float = 1e-8

    # Learned during fit:
    X_train: Optional[Sequence[Any]] = None
    y_train: Optional[Array] = None
    L: Optional[Array] = None
    alpha: Optional[Array] = None

    def fit(self, X: Sequence[Any], y: Array) -> "GPRegressor":
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != y.shape[0]:
            raise ValueError("len(X) must equal len(y).")

        self.X_train = X
        self.y_train = y

        K = pairwise_kernel_matrix(X, X, self.kernel_fn, self.kernel_params)
        K = self._add_diagonal_noise(K, self.noise, self.jitter)

        self.L = np.linalg.cholesky(K)
        self.alpha = self._solve_cholesky(self.L, y)
        return self

    def predict(
        self,
        X_star: Sequence[Any],
        return_std: bool = True,
        include_noise: bool = False,
    ) -> Tuple[Array, Optional[Array]]:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before predict().")
        if self.L is None or self.alpha is None:
            raise RuntimeError("Model is not properly fit (missing L/alpha).")

        Ks = pairwise_kernel_matrix(
            self.X_train, X_star, self.kernel_fn, self.kernel_params
        )
        mean = Ks.T @ self.alpha  # (n_star,)

        if not return_std:
            return mean, None

        Kss = pairwise_kernel_matrix(
            X_star, X_star, self.kernel_fn, self.kernel_params
        )
        # Posterior variance: diag(Kss - Ks^T K^{-1} Ks)
        v = np.linalg.solve(self.L, Ks)  # (n_train, n_star)
        var = np.diag(Kss) - np.sum(v * v, axis=0)

        # Numerical safety:
        var = np.maximum(var, 0.0)

        if include_noise:
            var = var + self.noise

        std = np.sqrt(var)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        """
        Log p(y | X, kernel_params, noise).
        Useful for hyperparameter tuning (optimize kernel_params + noise).
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before log_marginal_likelihood().")

        X = self.X_train
        y = self.y_train

        K = pairwise_kernel_matrix(X, X, self.kernel_fn, self.kernel_params)
        K = self._add_diagonal_noise(K, self.noise, self.jitter)

        L = np.linalg.cholesky(K)
        alpha = self._solve_cholesky(L, y)

        n = y.shape[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        quad = float(y.T @ alpha)
        return -0.5 * quad - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)

    def optimize_hyperparameters(self, n_restarts: int = 3) -> "GPRegressor":
        """
        Optimize kernel_params and noise by maximizing log marginal likelihood.
        Uses Nelder-Mead with random restarts to avoid local optima.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before optimize_hyperparameters().")

        n_kernel = len(self.kernel_params)

        def objective(theta):
            self.kernel_params = theta[:n_kernel]
            self.noise = float(np.exp(theta[n_kernel]))  # log-space for positivity
            try:
                self.fit(self.X_train, self.y_train)
                return -self.log_marginal_likelihood()
            except np.linalg.LinAlgError:
                return 1e10

        # Initial point: current params + log(noise)
        theta0 = np.concatenate([self.kernel_params, [np.log(self.noise)]])
        best_theta = theta0.copy()
        best_obj = objective(theta0)

        for _ in range(n_restarts):
            theta_init = theta0 + 0.5 * np.random.randn(len(theta0))
            res = minimize(objective, theta_init, method="Nelder-Mead",
                           options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4})
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.copy()

        # Apply best parameters and refit
        self.kernel_params = best_theta[:n_kernel]
        self.noise = float(np.exp(best_theta[n_kernel]))
        self.fit(self.X_train, self.y_train)
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
        # Solves (L L^T)^{-1} y
        tmp = np.linalg.solve(L, y)
        return np.linalg.solve(L.T, tmp)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _edge_match(e1: Tuple, e2: Tuple) -> int:
    """Count shared qubit indices between two edges (0, 1, or 2)."""
    return (e1[0] == e2[0]) + (e1[1] == e2[1]) + (e1[0] == e2[1]) + (e1[1] == e2[0])

def _lcs_length(x1: Sequence[Tuple], x2: Sequence[Tuple]) -> int:
    """Longest common subsequence length using exact edge equality."""
    n, m = len(x1), len(x2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x1[i-1] == x2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def _raw_kernel(x1: Any, x2: Any, params: Array) -> float:
    """Unnormalized blended kernel for circuit edge sequences."""
    order_weight = _sigmoid(float(params[1]))

    # Set similarity: all-pairs, permutation invariant
    set_score = 0.0
    for e1 in x1:
        for e2 in x2:
            set_score += _edge_match(e1, e2)
    set_score /= (2.0 * len(x1) * len(x2))

    # Order similarity: longest common subsequence ratio
    lcs = _lcs_length(x1, x2)
    order_score = lcs / max(len(x1), len(x2))

    return (1.0 - order_weight) * set_score + order_weight * order_score

def kernel(x1: Any, x2: Any, params: Array) -> float:
    """
    Cosine-normalized blended kernel for circuit edge sequences.

    params[0]: log output scale
    params[1]: order_weight (pre-sigmoid) — controls how much canonical
               ordering matters vs set membership
    """
    scale = float(np.exp(params[0]))
    k12 = _raw_kernel(x1, x2, params)
    k11 = _raw_kernel(x1, x1, params)
    k22 = _raw_kernel(x2, x2, params)
    return scale * k12 / np.sqrt(k11 * k22)

def decompose(Umtx, x):
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
    print(cDecomp.Optimization_Problem(params))
    return cDecomp.Optimization_Problem(params)

def generate_goal_unitary(N, edges):
    goal = create_circuit_from_edges(edges,N)
    return goal.get_Matrix(np.random.rand(goal.get_Parameter_Num())*2*np.pi)


if __name__ == "__main__":
    N=3
    goal_edges = [(0,1),(1,2),(0,1)]
    Umtx = generate_goal_unitary(N, edges=goal_edges)
    X = [[(2,1),(0,2),(2,1)],[(0,2),(1,2),(1,2)],[(0,1),(0,1),(1,2)],[(0,1),(0,1),(0,1)],[(1,2),(1,2),(1,2)]]
    y = np.array([decompose(Umtx, x) for x in X])

    gp = GPRegressor(
        kernel_fn=kernel,
        kernel_params=np.array([0.0, 1.0]),  # [log-scale, order_weight (pre-sigmoid)]
        noise=1e-2,
    ).fit(X, y)

    print(f"Before optimization: log_scale={gp.kernel_params[0]:.4f}, order_weight={_sigmoid(gp.kernel_params[1]):.4f}, noise={gp.noise:.6f}")
    gp.optimize_hyperparameters(n_restarts=5)
    print(f"After optimization:  log_scale={gp.kernel_params[0]:.4f}, order_weight={_sigmoid(gp.kernel_params[1]):.4f}, noise={gp.noise:.6f}")

    X_star = [[(0,1),(1,2),(0,1)],[(0,2),(1,2),(1,2)]]
    mu, std = gp.predict(X_star, return_std=True, include_noise=False)
    print("mu:", mu)
    print("std:", std)
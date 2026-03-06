# gp_custom.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
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


# --------------------------
# Example usage (replace kernel)
# --------------------------

def kernel_placeholder(x1: Any, x2: Any, params: Array) -> float:
    """
    Replace this with your Hamming-based kernel (or any kernel).

    params is a 1D array you can interpret however you like.
    """
    # Example: linear kernel on numeric vectors (just a placeholder)
    
    a = np.asarray(x1, dtype=float).reshape(-1)
    b = np.asarray(x2, dtype=float).reshape(-1)
    scale = float(np.exp(params[0]))  # keep positive via exp
    return scale * float(a @ b)


if __name__ == "__main__":
    # X can be strings if your kernel supports it.
    X = [np.array([0, 1, 0]), np.array([1, 1, 0]), np.array([1, 0, 0])]
    y = np.array([0.2, 0.9, 0.4])

    gp = GPRegressor(
        kernel_fn=kernel_placeholder,
        kernel_params=np.array([0.0]),  # log-scale in this placeholder
        noise=1e-2,
    ).fit(X, y)

    X_star = [np.array([0, 1, 1]), np.array([1, 0, 1])]
    mu, std = gp.predict(X_star, return_std=True, include_noise=False)
    print("mu:", mu)
    print("std:", std)
    print("lml:", gp.log_marginal_likelihood())
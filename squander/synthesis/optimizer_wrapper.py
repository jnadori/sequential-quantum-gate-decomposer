# -*- coding: utf-8 -*-
"""
Copyright 2024 Budapest Quantum Computing Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
from squander import N_Qubit_Decomposition_custom
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


class OptimizationResult:
    """Container for optimization results."""

    def __init__(
        self,
        parameters: np.ndarray,
        cost: float,
        success: bool = True,
        message: str = "",
        nfev: int = 0,
        nit: int = 0
    ):
        self.parameters = parameters
        self.cost = cost
        self.success = success
        self.message = message
        self.nfev = nfev  # Number of function evaluations
        self.nit = nit    # Number of iterations


def optimize_circuit_scipy(
    Umtx: np.ndarray,
    circuit: Circuit,
    method: str = 'L-BFGS-B',
    initial_params: Optional[np.ndarray] = None,
    config: Dict[str, Any] = {},
    verbose: int = 0
) -> OptimizationResult:
    """
    Optimize circuit parameters using scipy.optimize.

    Args:
        Umtx: Target unitary matrix (conjugate transposed)
        circuit: Circuit structure to optimize
        method: scipy optimization method ('L-BFGS-B', 'BFGS', 'CG', 'Powell',
                'basinhopping', 'CMA-ES', etc.)
        initial_params: Initial parameter values (random if None)
        config: Configuration dictionary with keys:
            - max_iterations: Maximum number of iterations
            - tolerance: Optimization tolerance
            - cost_function_variant: Cost function type (0, 1, 2, or 3)
        verbose: Verbosity level (0=silent, 1=minimal, 2=detailed)

    Returns:
        OptimizationResult with optimized parameters and cost
    """
    from scipy.optimize import minimize

    # Create decomposition object
    accelerator_num = config.get('accelerator_num', 0)
    decomposer = N_Qubit_Decomposition_custom(Umtx, config=config, accelerator_num=accelerator_num)
    decomposer.set_Gate_Structure(circuit)
    decomposer.set_Verbose(verbose)

    # Set cost function variant
    cost_variant = config.get('cost_function_variant', 3)
    decomposer.set_Cost_Function_Variant(cost_variant)

    # Get parameter count
    param_num = decomposer.get_Parameter_Num()

    # Initialize parameters
    if initial_params is None:
        initial_params = np.random.rand(param_num) * 2 * np.pi

    # Define cost function and gradient
    def cost_fn(params):
        return decomposer.Optimization_Problem(params)

    def grad_fn(params):
        return decomposer.Optimization_Problem_Grad(params)

    def cost_and_grad(params):
        cost, grad = decomposer.Optimization_Problem_Combined(params)
        return cost, grad

    # Optimization options (different methods accept different options)
    options = {'maxiter': config.get('max_iterations', 1000)}

    # Add tolerance parameters based on method
    tol = config.get('tolerance', 1e-8)
    if method in ['L-BFGS-B', 'TNC']:
        options['ftol'] = tol
        options['gtol'] = tol
    elif method in ['BFGS', 'CG', 'Newton-CG']:
        options['gtol'] = tol
    elif method in ['Nelder-Mead', 'Powell']:
        options['ftol'] = tol
        options['xtol'] = tol


    if verbose >= 2:
        options['disp'] = True

    # Run optimization based on method type
    if method == 'basinhopping':
        # Basin-hopping global optimization
        from scipy.optimize import basinhopping

        # Minimizer kwargs for local optimization
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'jac': True,
            'options': {'maxiter': 50}  # Iterations per basin
        }

        # Basin-hopping iterations (use fewer iterations for speed)
        # max_iterations in config refers to total local minimization iterations,
        # so divide by iterations per basin to get number of basin hops
        niter = config.get('basinhopping_niter', 10)  # Default: 10 basin hops

        result = basinhopping(
            cost_and_grad,
            initial_params,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            disp=verbose >= 2
        )

    elif method == 'CMA-ES':
        # CMA-ES optimization (requires cmaes package)
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES optimizer requires the 'cma' package. "
                "Install it with: pip install cma"
            )

        # CMA-ES options
        sigma0 = config.get('cma_sigma0', 0.5)  # Initial step size
        max_iter = config.get('max_iterations', 1000)

        cma_options = {
            'maxiter': max_iter,
            'tolfun': tol,
            'verbose': -1 if verbose < 2 else 1
        }

        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(initial_params, sigma0, cma_options)
        es.optimize(cost_fn)

        # Get result
        result_x = es.result.xbest
        result_fun = es.result.fbest

        # Create result object compatible with scipy format
        class CMAResult:
            def __init__(self, x, fun, nfev):
                self.x = x
                self.fun = fun
                self.success = True
                self.message = "CMA-ES optimization completed"
                self.nfev = nfev
                self.nit = es.result.iterations

        result = CMAResult(result_x, result_fun, es.result.evaluations)

    elif method in ['L-BFGS-B', 'BFGS', 'CG', 'Newton-CG', 'TNC']:
        # Methods that support gradients
        result = minimize(
            cost_and_grad,
            initial_params,
            method=method,
            jac=True,
            options=options
        )
    else:
        # Methods without gradients
        result = minimize(
            cost_fn,
            initial_params,
            method=method,
            options=options
        )

    return OptimizationResult(
        parameters=result.x,
        cost=result.fun,
        success=result.success,
        message=result.message,
        nfev=result.nfev,
        nit=result.nit
    )


def optimize_circuit_builtin(
    Umtx: np.ndarray,
    circuit: Circuit,
    optimizer: str = 'BFGS',
    initial_params: Optional[np.ndarray] = None,
    config: Dict[str, Any] = {},
    verbose: int = 0
) -> OptimizationResult:
    """
    Optimize circuit using built-in SQUANDER optimizers.

    Args:
        Umtx: Target unitary matrix (conjugate transposed)
        circuit: Circuit structure to optimize
        optimizer: SQUANDER optimizer ('BFGS', 'ADAM', 'AGENTS', etc.)
        initial_params: Initial parameter values (random if None)
        config: Configuration dictionary
        verbose: Verbosity level

    Returns:
        OptimizationResult with optimized parameters and cost
    """
    # Create decomposition object
    accelerator_num = config.get('accelerator_num', 0)
    decomposer = N_Qubit_Decomposition_custom(Umtx, config=config, accelerator_num=accelerator_num)
    decomposer.set_Gate_Structure(circuit)
    decomposer.set_Verbose(verbose)

    # Set optimizer
    decomposer.set_Optimizer(optimizer)

    # Set cost function variant
    cost_variant = config.get('cost_function_variant', 3)
    decomposer.set_Cost_Function_Variant(cost_variant)

    # Set initial parameters
    param_num = decomposer.get_Parameter_Num()
    if initial_params is None:
        initial_params = np.random.rand(param_num) * 2 * np.pi

    decomposer.set_Optimized_Parameters(initial_params)

    # Run optimization
    decomposer.Start_Decomposition()

    # Get results
    final_params = decomposer.get_Optimized_Parameters()
    final_cost = decomposer.Optimization_Problem(final_params)

    return OptimizationResult(
        parameters=final_params,
        cost=final_cost,
        success=(final_cost < config.get('tolerance', 1e-8)),
        message=f"Optimizer {optimizer} completed",
        nfev=0,  # Not tracked by built-in optimizers
        nit=0
    )


def optimize_circuit_custom(
    Umtx: np.ndarray,
    circuit: Circuit,
    optimizer_fn: Callable,
    initial_params: Optional[np.ndarray] = None,
    config: Dict[str, Any] = {},
    verbose: int = 0
) -> OptimizationResult:
    """
    Optimize circuit using a custom optimizer function.

    The custom optimizer function should have signature:
        optimizer_fn(cost_fn, grad_fn, x0, **kwargs) -> (x_opt, cost_final, info_dict)

    Args:
        Umtx: Target unitary matrix (conjugate transposed)
        circuit: Circuit structure to optimize
        optimizer_fn: Custom optimization function
        initial_params: Initial parameter values (random if None)
        config: Configuration dictionary passed to optimizer_fn as kwargs
        verbose: Verbosity level

    Returns:
        OptimizationResult with optimized parameters and cost
    """
    # Create decomposition object
    accelerator_num = config.get('accelerator_num', 0)
    decomposer = N_Qubit_Decomposition_custom(Umtx, config=config, accelerator_num=accelerator_num)
    decomposer.set_Gate_Structure(circuit)
    decomposer.set_Verbose(verbose)

    # Set cost function variant
    cost_variant = config.get('cost_function_variant', 3)
    decomposer.set_Cost_Function_Variant(cost_variant)

    # Get parameter count
    param_num = decomposer.get_Parameter_Num()

    # Initialize parameters
    if initial_params is None:
        initial_params = np.random.rand(param_num) * 2 * np.pi

    # Define cost function and gradient
    def cost_fn(params):
        return decomposer.Optimization_Problem(params)

    def grad_fn(params):
        return decomposer.Optimization_Problem_Grad(params)

    # Run custom optimizer
    result = optimizer_fn(cost_fn, grad_fn, initial_params, **config)

    # Parse result
    if isinstance(result, tuple):
        if len(result) == 2:
            x_opt, cost_final = result
            info = {}
        elif len(result) == 3:
            x_opt, cost_final, info = result
        else:
            raise ValueError("Custom optimizer must return (x_opt, cost) or (x_opt, cost, info)")
    else:
        raise ValueError("Custom optimizer must return tuple")

    return OptimizationResult(
        parameters=x_opt,
        cost=cost_final,
        success=info.get('success', cost_final < config.get('tolerance', 1e-8)),
        message=info.get('message', 'Custom optimizer completed'),
        nfev=info.get('nfev', 0),
        nit=info.get('nit', 0)
    )


def optimize_circuit(
    Umtx: np.ndarray,
    circuit: Circuit,
    optimizer: str = 'scipy:L-BFGS-B',
    initial_params: Optional[np.ndarray] = None,
    config: Dict[str, Any] = {},
    verbose: int = 0,
    custom_optimizer: Optional[Callable] = None
) -> OptimizationResult:
    """
    Universal interface to optimize circuit parameters.

    Args:
        Umtx: Target unitary matrix (conjugate transposed)
        circuit: Circuit structure to optimize
        optimizer: Optimizer specification:
            - 'scipy:<method>' for scipy.optimize (e.g., 'scipy:L-BFGS-B')
            - 'BFGS', 'ADAM', 'AGENTS' for built-in SQUANDER optimizers
            - 'custom' to use custom_optimizer function
        initial_params: Initial parameter values (random if None)
        config: Configuration dictionary
        verbose: Verbosity level
        custom_optimizer: Custom optimization function (required if optimizer='custom')

    Returns:
        OptimizationResult with optimized parameters and cost
    """
    if optimizer.startswith('scipy:'):
        method = optimizer.split(':', 1)[1]
        return optimize_circuit_scipy(Umtx, circuit, method, initial_params, config, verbose)
    elif optimizer == 'custom':
        if custom_optimizer is None:
            raise ValueError("custom_optimizer function must be provided when optimizer='custom'")
        return optimize_circuit_custom(Umtx, circuit, custom_optimizer, initial_params, config, verbose)
    else:
        # Use built-in SQUANDER optimizer
        return optimize_circuit_builtin(Umtx, circuit, optimizer, initial_params, config, verbose)

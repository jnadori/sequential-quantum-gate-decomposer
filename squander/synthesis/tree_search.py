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
from typing import List, Tuple, Optional, Dict, Any, Callable
import time

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.synthesis.gray_code import NaryGrayCodeCounter
from squander.synthesis.circuit_builder import (
    build_topology,
    construct_circuit_from_gray_code,
    visualize_circuit_structure
)
from squander.synthesis.optimizer_wrapper import optimize_circuit, OptimizationResult


class TreeSearchDecomposition:
    """
    Pure Python implementation of tree search based quantum circuit synthesis.

    This class performs tree search over gate structures using n-ary Gray codes
    to enumerate different CNOT configurations. For each configuration, it
    optimizes the rotation angles using scipy or custom optimizers.

    Reference: squander/src-cpp/decomposition/N_Qubit_Decomposition_Tree_Search.cpp

    Attributes:
        Umtx: Target unitary matrix (conjugate transposed)
        qbit_num: Number of qubits
        topology: List of allowed (target, control) qubit pairs
        config: Configuration dictionary
        verbose: Verbosity level
        best_circuit: Best circuit found so far
        best_parameters: Best parameters found so far
        best_cost: Best cost achieved so far
    """

    def __init__(
        self,
        Umtx: np.ndarray,
        topology: Optional[List[Tuple[int, int]]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        """
        Initialize the tree search decomposition.

        Args:
            Umtx: Target unitary matrix (should be conjugate transposed)
            topology: Optional list of (target, control) qubit pairs.
                     If None, all-to-all connectivity is assumed.
            config: Configuration dictionary with keys:
                - tree_level_max: Maximum tree depth (default 10)
                - tree_level_min: Minimum tree depth (default 0)
                - optimization_tolerance: Target cost (default 1e-8)
                - optimizer: Optimizer specification (default 'scipy:L-BFGS-B')
                - max_iterations: Max iterations per optimization (default 1000)
                - cost_function_variant: Cost function type (default 3)
                - num_trials_per_level: Number of random restarts (default 1)
                - parallel: Enable parallel evaluation (default False)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug)
        """
        self.Umtx = Umtx
        self.qbit_num = int(np.round(np.log2(len(Umtx))))

        # Build topology
        self.topology = build_topology(self.qbit_num, topology)
        self.n_ary_limit = len(self.topology)

        # Configuration
        self.config = config if config is not None else {}
        self.tree_level_max = self.config.get('tree_level_max', 10)
        self.tree_level_min = self.config.get('tree_level_min', 0)
        self.optimization_tolerance = self.config.get('optimization_tolerance', 1e-8)
        self.optimizer = self.config.get('optimizer', 'scipy:L-BFGS-B')
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.cost_function_variant = self.config.get('cost_function_variant', 3)
        self.num_trials = self.config.get('num_trials_per_level', 1)
        self.parallel = self.config.get('parallel', False)

        self.verbose = verbose

        # Results
        self.best_circuit = None
        self.best_parameters = None
        self.best_cost = float('inf')
        self.best_gray_code = None
        self.best_level = None

        # Statistics
        self.search_history = []

    def _log(self, level: int, message: str):
        """Print log message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(message, flush=True)

    def optimize_gate_structure(
        self,
        circuit: Circuit,
        gray_code: Optional[np.ndarray] = None,
        trial_idx: int = 0
    ) -> OptimizationResult:
        """
        Optimize a single gate structure.

        Args:
            circuit: Circuit to optimize
            gray_code: Gray code that generated this circuit (for logging)
            trial_idx: Trial number (for logging)

        Returns:
            OptimizationResult with optimized parameters and cost
        """
        # Create optimization config
        opt_config = {
            'max_iterations': self.max_iterations,
            'tolerance': self.optimization_tolerance,
            'cost_function_variant': self.cost_function_variant,
        }

        # Run optimization
        verbose_opt = max(0, self.verbose - 2)  # Reduce verbosity for optimizer
        result = optimize_circuit(
            self.Umtx,
            circuit,
            optimizer=self.optimizer,
            config=opt_config,
            verbose=verbose_opt
        )

        return result

    def search_single_level(self, level: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Search all gate configurations at a specific tree level (depth).

        Args:
            level: Tree level (number of CNOT layers)

        Returns:
            Tuple of (best_gray_code, best_cost) at this level
        """
        if level == 0:
            # Special case: no CNOTs, just U3 gates
            self._log(2, f"  Level {level}: No CNOT gates, skipping...")
            return None, float('inf')

        # Generate n-ary limits for this level
        n_ary_limits = [self.n_ary_limit] * level

        # Create Gray code counter
        gray_counter = NaryGrayCodeCounter(n_ary_limits)
        total_configs = gray_counter.offset_max + 1

        self._log(1, f"  Level {level}: Searching {total_configs} configurations...")

        # Track best at this level
        level_best_gray = None
        level_best_cost = float('inf')
        level_best_circuit = None
        level_best_params = None

        configs_evaluated = 0
        start_time = time.time()

        # Iterate over all Gray codes at this level
        for gray_code in gray_counter:
            # Build circuit from Gray code
            circuit = construct_circuit_from_gray_code(
                gray_code,
                self.qbit_num,
                self.topology,
                use_u3=True
            )

            # Optimize with multiple random restarts
            best_trial_cost = float('inf')
            best_trial_params = None

            for trial in range(self.num_trials):
                result = self.optimize_gate_structure(circuit, gray_code, trial)

                if result.cost < best_trial_cost:
                    best_trial_cost = result.cost
                    best_trial_params = result.parameters

            # Update level best
            if best_trial_cost < level_best_cost:
                level_best_cost = best_trial_cost
                level_best_gray = gray_code.copy()
                level_best_circuit = circuit
                level_best_params = best_trial_params

                self._log(2, f"    Config {configs_evaluated}: New best cost = {best_trial_cost:.6e}")

            configs_evaluated += 1

            # Early stopping if we found a good solution
            if level_best_cost < self.optimization_tolerance:
                self._log(2, f"    Early stopping: Found solution with cost {level_best_cost:.6e}")
                break

            # Progress update
            if self.verbose >= 2 and configs_evaluated % 100 == 0:
                elapsed = time.time() - start_time
                rate = configs_evaluated / elapsed if elapsed > 0 else 0
                self._log(2, f"    Evaluated {configs_evaluated}/{total_configs} configs ({rate:.1f}/s)")

        elapsed = time.time() - start_time
        self._log(1, f"  Level {level}: Best cost = {level_best_cost:.6e} (evaluated {configs_evaluated} in {elapsed:.1f}s)")

        # Update global best
        if level_best_cost < self.best_cost:
            self.best_cost = level_best_cost
            self.best_gray_code = level_best_gray
            self.best_circuit = level_best_circuit
            self.best_parameters = level_best_params
            self.best_level = level

        # Save to history
        self.search_history.append({
            'level': level,
            'cost': level_best_cost,
            'gray_code': level_best_gray,
            'configs_evaluated': configs_evaluated,
            'elapsed_time': elapsed
        })

        return level_best_gray, level_best_cost

    def start_decomposition(self) -> Dict[str, Any]:
        """
        Run the full tree search decomposition.

        Returns:
            Dictionary with decomposition results:
                - success: Whether target tolerance was achieved
                - cost: Final cost
                - level: Best tree level found
                - gray_code: Best Gray code
                - circuit: Best circuit structure
                - parameters: Optimized parameters
                - total_time: Total search time
        """
        self._log(1, "="*70)
        self._log(1, f"Tree Search Decomposition for {self.qbit_num}-qubit unitary")
        self._log(1, f"Topology: {len(self.topology)} allowed CNOT connections")
        self._log(1, f"Optimizer: {self.optimizer}")
        self._log(1, f"Target tolerance: {self.optimization_tolerance:.2e}")
        self._log(1, "="*70)

        start_time = time.time()

        # Search over tree levels
        for level in range(self.tree_level_min, self.tree_level_max + 1):
            self._log(1, f"\nSearching level {level}...")

            best_gray, best_cost = self.search_single_level(level)

            # Early stopping if we achieved tolerance
            if best_cost < self.optimization_tolerance:
                self._log(1, f"\n*** Target tolerance achieved at level {level}! ***")
                break

        total_time = time.time() - start_time

        # Final summary
        self._log(1, "\n" + "="*70)
        self._log(1, "Tree Search Complete")
        self._log(1, "="*70)
        self._log(1, f"Best cost: {self.best_cost:.6e}")
        self._log(1, f"Best level: {self.best_level}")
        self._log(1, f"Total time: {total_time:.1f}s")
        self._log(1, f"Success: {self.best_cost < self.optimization_tolerance}")
        self._log(1, "="*70)

        # Visualize best circuit
        if self.verbose >= 2 and self.best_gray_code is not None:
            viz = visualize_circuit_structure(self.best_gray_code, self.topology, self.qbit_num)
            self._log(2, "\n" + viz)

        return {
            'success': self.best_cost < self.optimization_tolerance,
            'cost': self.best_cost,
            'level': self.best_level,
            'gray_code': self.best_gray_code,
            'circuit': self.best_circuit,
            'parameters': self.best_parameters,
            'total_time': total_time,
            'search_history': self.search_history
        }

    def get_circuit(self) -> Optional[Circuit]:
        """Get the best circuit found."""
        return self.best_circuit

    def get_optimized_parameters(self) -> Optional[np.ndarray]:
        """Get the optimized parameters."""
        return self.best_parameters

    def get_decomposition_error(self) -> float:
        """Get the final decomposition error."""
        return self.best_cost

    def get_qiskit_circuit(self):
        """Export best circuit to Qiskit format."""
        if self.best_circuit is None or self.best_parameters is None:
            raise ValueError("No circuit found yet. Run start_decomposition() first.")

        from squander import Qiskit_IO
        return Qiskit_IO.get_Qiskit_Circuit(self.best_circuit, self.best_parameters)

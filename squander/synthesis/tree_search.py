# -*- coding: utf-8 -*-
"""
Tree Search Decomposition

Python implementation of tree search decomposition for quantum circuit synthesis.
Replicates the functionality of N_Qubit_Decomposition_Tree_Search from C++.
"""

import heapq
import itertools
import time
from typing import List, Tuple, Optional, Dict, Any

import multiprocessing as mp
import numpy as np

from squander.synthesis.gray_code import NaryGrayCodeCounter
from squander.synthesis.circuit_builder import construct_gate_structure_from_gray_code


class TreeSearchDecomposition:
    """
    Tree search decomposition for quantum circuit synthesis.
    
    This class implements the same algorithm as the C++ N_Qubit_Decomposition_Tree_Search,
    but with tree search logic in Python and cost evaluation in C++.
    """
    
    def __init__(
        self,
        Umtx: np.ndarray,
        topology: Optional[List[Tuple[int, int]]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        search_name: str = "Tree Search Decomposition"
    ):
        """
        Initialize tree search decomposer.
        
        Args:
            Umtx: Unitary matrix to decompose (2^N x 2^N complex array)
            topology: List of (target, control) qubit pairs. If None, all-to-all.
            config: Configuration dictionary with parameters
            verbose: Verbosity level (0=silent, 1=normal, 2=verbose)
        """
        self.Umtx = Umtx.copy()
        self.qbit_num = int(np.log2(Umtx.shape[0]))
        
        if Umtx.shape[0] != Umtx.shape[1] or 2**self.qbit_num != Umtx.shape[0]:
            raise ValueError(f"Umtx must be square with dimension 2^N, got {Umtx.shape}")
        
        # Set up topology
        if topology is None:
            # All-to-all connectivity (exclude self-loops; CNOT needs distinct qubits)
            self.topology = []
            for target in range(self.qbit_num):
                for control in range(self.qbit_num):
                    if target == control:
                        continue
                    self.topology.append((target, control))
        else:
            # Trust caller but guard against invalid entries
            self.topology = [
                (t, c) for (t, c) in topology if t != c
            ]
        
        # Configuration with defaults
        self.config = config.copy() if config else {}
        self.verbose = verbose
        self.search_name = search_name
        
        # Ensure parallel is set to 0 for circuit simulation (single thread)
        if 'parallel' not in self.config:
            self.config['parallel'] = 0
        
        # Extract configuration parameters
        self.tree_level_max = self.config.get('tree_level_max', 14)
        self.optimization_tolerance = self.config.get('optimization_tolerance', 1e-8)
        self.optimizer = self.config.get('optimizer', 'BFGS')  # 'BFGS', 'ADAM', or scipy optimizers
        self.cost_function_variant = self.config.get('cost_function_variant', 3)  # Hilbert-Schmidt test
        self.max_outer_iterations = self.config.get('max_outer_iterations', None)
        self.max_inner_iterations = self.config.get('max_inner_iterations', None)
        self.trace_offset = self.config.get('trace_offset', 0)
        self.project_name = self.config.get('project_name', '')
        
        # Best solution tracking
        self.current_minimum = float('inf')
        self.best_gray_code = []
        self.best_parameters = None
        self.best_decomposition = None  # Store the best decomposition instance
        self.best_final_circuit = None  # Store the final (flattened) circuit structure
        self.best_gate_structure = None  # Store the original hierarchical gate structure
        self.number_of_iters = 0
        self.nodes_evaluated = 0
        
        # Multiprocessing
        self.n_processes = self.config.get('n_processes', mp.cpu_count())
        
        # Print initialization info
        if self.verbose >= 1:
            print("="*70)
            print(f"{self.search_name} initialized")
            print(f"Qubits: {self.qbit_num}")
            print(f"Topology edges: {len(self.topology)}")
            print(f"Max tree level: {self.tree_level_max}")
            print(f"Optimizer: {self.optimizer}")
            print(f"Processes: {self.n_processes}")
            print("="*70)
    
    def start_decomposition(self) -> Dict[str, Any]:
        """
        Start tree search decomposition.
        
        Returns:
            Dictionary with results:
            - success: bool
            - cost: float (final cost)
            - level: int (best level found)
            - gray_code: List[int] (best Gray code)
            - parameters: np.ndarray (optimized parameters)
            - total_time: float
            - number_of_iters: int
            - nodes_evaluated: int
        """
        start_time = time.time()
        self.nodes_evaluated = 0  # Reset node counter
        
        if self.verbose >= 1:
            print(f"\nStarting tree search decomposition for {self.qbit_num}-qubit matrix")
        
        best_solution = None
        minimum_best_solution = float('inf')
        
        # Iterate over levels from 0 to tree_level_max
        for level in range(0, self.tree_level_max + 1):
            if self.verbose >= 1:
                print(f"\n{'='*70}")
                print(f"Exploring level {level}/{self.tree_level_max}")
                print(f"{'='*70}")
            
            # Perform tree search at this level
            gray_code, cost, parameters, nodes_at_level = self.tree_search_over_gate_structures(level)
            self.nodes_evaluated += nodes_at_level
            
            # Update best solution
            if cost < minimum_best_solution:
                minimum_best_solution = cost
                best_solution = {
                    'gray_code': gray_code,
                    'cost': cost,
                    'parameters': parameters,
                    'level': level
                }
                self.current_minimum = cost
                self.best_gray_code = gray_code
                self.best_parameters = parameters
                # Note: best_decomposition is set in _optimize_cpp
            
            if self.verbose >= 1:
                print(f"Level {level} best cost: {cost:.6e}")
                print(f"Overall best cost: {minimum_best_solution:.6e}")
            
            # Early stopping if tolerance met
            if self.current_minimum < self.optimization_tolerance:
                if self.verbose >= 1:
                    print(f"\nTolerance {self.optimization_tolerance:.6e} reached at level {level}")
                break
        
        total_time = time.time() - start_time
        
        # Prepare results
        if best_solution is None:
            raise RuntimeError("No solution found in tree search")
        
        result = {
            'success': self.current_minimum < self.optimization_tolerance,
            'cost': self.current_minimum,
            'level': best_solution['level'],
            'gray_code': self.best_gray_code,
            'parameters': self.best_parameters,
            'total_time': total_time,
            'number_of_iters': self.number_of_iters,
            'nodes_evaluated': self.nodes_evaluated
        }
        
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("Tree Search Results:")
            print(f"  Success: {result['success']}")
            print(f"  Final cost: {result['cost']:.6e}")
            print(f"  Best level: {result['level']}")
            print(f"  Nodes evaluated: {result['nodes_evaluated']}")
            print(f"  Total time: {result['total_time']:.1f}s")
            print(f"  Total iterations: {result['number_of_iters']}")
            print(f"{'='*70}\n")
        
        return result
    
    def tree_search_over_gate_structures(
        self,
        level_num: int
    ) -> Tuple[List[int], float, np.ndarray, int]:
        """
        Perform tree search over gate structures for a given level.
        
        Args:
            level_num: Number of CNOT layers (tree depth)
        
        Returns:
            Tuple of (best_gray_code, best_cost, best_parameters, nodes_evaluated)
        """
        # Handle level 0 (no two-qubit gates)
        if level_num == 0:
            gray_code, cost, params = self._optimize_empty_gate_structure()
            return (gray_code, cost, params, 1)  # 1 node evaluated (empty structure)
        
        # Set up parallel search
        n_ary_limit_max = len(self.topology)
        iteration_max = n_ary_limit_max ** level_num
        
        if self.verbose >= 2:
            print(f"  Total combinations to explore: {iteration_max}")
        
        # Divide work among processes
        n_workers = min(self.n_processes, iteration_max)
        work_per_worker = iteration_max // n_workers
        
        # Create work ranges
        work_ranges = []
        for i in range(n_workers):
            start_idx = i * work_per_worker
            if i == n_workers - 1:
                end_idx = iteration_max
            else:
                end_idx = (i + 1) * work_per_worker
            work_ranges.append((start_idx, end_idx))
        
        # Create worker function arguments (ensure parallel=0 in worker configs too)
        worker_config = self.config.copy()
        worker_config['parallel'] = 0
        
        worker_args = [
            (
                start_idx,
                end_idx,
                self.Umtx.copy(),
                self.qbit_num,
                self.topology,
                level_num,
                n_ary_limit_max,
                worker_config,
                self.optimization_tolerance,
                self.verbose
            )
            for start_idx, end_idx in work_ranges
        ]
        
        # Execute parallel search
        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = pool.starmap(_tree_search_worker, worker_args)
        else:
            # Single process - no overhead
            results = [_tree_search_worker(*args) for args in worker_args]
        
        # Collect results and find best
        best_cost = float('inf')
        best_gray_code = None
        best_params = None
        total_iters = 0
        total_nodes = 0
        
        for result in results:
            if result is None:
                continue
            iters, cost, gray_code, params, nodes = result
            total_iters += iters
            total_nodes += nodes
            if cost < best_cost:
                best_cost = cost
                best_gray_code = gray_code
                best_params = params
        
        self.number_of_iters += total_iters
        
        # Check if we found optimal solution
        if best_cost < self.optimization_tolerance:
            if self.verbose >= 1:
                print(f"  Optimal solution found: cost = {best_cost:.6e}")
        
        # IMPORTANT: If we found a better solution from workers, we need to sync
        # best_decomposition in the main process. Workers can't update self.best_decomposition
        # because multiprocessing doesn't share state. We need to re-create the decomposition
        # with the correct gate structure and parameters.
        if best_gray_code is not None and best_cost < self.current_minimum:
            self._sync_best_decomposition(best_gray_code, best_params, best_cost)
        
        if best_gray_code is not None:
            return (best_gray_code, best_cost, best_params, total_nodes)
        else:
            # Fallback if no solution found
            return ([], float('inf'), np.array([]), total_nodes)
    
    def _optimize_empty_gate_structure(self) -> Tuple[List[int], float, np.ndarray]:
        """Optimize empty gate structure (level 0 - no two-qubit gates)."""
        from squander.synthesis.circuit_builder import add_finalizing_layer
        from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
        
        # Create circuit with only finalizing layer
        gate_structure = Circuit(self.qbit_num)
        add_finalizing_layer(gate_structure, self.qbit_num)
        
        # Optimize
        cost, parameters = self._perform_optimization(gate_structure)
        
        if cost < self.current_minimum:
            self.current_minimum = cost
            self.best_parameters = parameters
        
        return ([], cost, parameters)
    
    def _sync_best_decomposition(
        self,
        gray_code: List[int],
        parameters: np.ndarray,
        cost: float
    ):
        """
        Synchronize best solution state with results from worker processes.
        
        Workers can't pass back circuit objects (they can't be pickled), so
        we must re-run the optimization in the main process to get the correct
        circuit-parameter pairing. We run multiple random restarts since the
        optimization is non-convex and can get stuck in local minima.
        
        Args:
            gray_code: Best Gray code from workers
            parameters: Optimized parameters from workers (not used directly)
            cost: Cost achieved by workers (used as target)
        """
        from squander import N_Qubit_Decomposition_custom
        from squander.synthesis.circuit_builder import add_finalizing_layer
        from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
        
        # Reconstruct gate structure from Gray code
        if len(gray_code) == 0:
            gate_structure = Circuit(self.qbit_num)
            add_finalizing_layer(gate_structure, self.qbit_num)
        else:
            gate_structure = construct_gate_structure_from_gray_code(
                gray_code,
                self.topology,
                self.qbit_num
            )
        
        # Re-run optimization in main process with multiple random restarts
        # This is necessary because:
        # 1. Circuit objects can't be pickled/transferred from workers
        # 2. Parameter indexing depends on internal C++ state
        # 3. Optimization is non-convex, multiple restarts improve chance of convergence
        
        best_decomp = None
        best_cost_found = float('inf')
        num_restarts = self.config.get('sync_restarts', 5)  # Number of random restarts
        
        for restart in range(num_restarts):
            # Create fresh gate structure for each restart (parameter indices are reset)
            if len(gray_code) == 0:
                gate_structure_trial = Circuit(self.qbit_num)
                add_finalizing_layer(gate_structure_trial, self.qbit_num)
            else:
                gate_structure_trial = construct_gate_structure_from_gray_code(
                    gray_code,
                    self.topology,
                    self.qbit_num
                )
            
            decomp = N_Qubit_Decomposition_custom(
                self.Umtx.conj().T,
                qbit_num=self.qbit_num,
                initial_guess="random",
                config=self.config
            )
            
            decomp.set_Gate_Structure(gate_structure_trial)
            
            # Count layers for optimization blocks
            gates = gate_structure_trial.get_Gates()
            num_layers = sum(1 for gate in gates if hasattr(gate, 'get_Gates'))
            if num_layers == 0:
                num_layers = 1
            decomp.set_Optimization_Blocks(num_layers)
            
            decomp.set_Verbose(0)
            decomp.set_Cost_Function_Variant(costfnc=self.cost_function_variant)
            decomp.set_Optimization_Tolerance(self.optimization_tolerance)
            decomp.set_Optimizer(optimizer=self.optimizer)
            
            # Run optimization
            decomp.Start_Decomposition()
            
            # Get results for this trial
            trial_params = decomp.get_Optimized_Parameters()
            trial_cost = decomp.Optimization_Problem(trial_params)
            
            if trial_cost < best_cost_found:
                best_cost_found = trial_cost
                best_decomp = decomp
                
                # Early stopping if we reached tolerance
                if trial_cost < self.optimization_tolerance:
                    break
        
        if best_decomp is None:
            # Fallback - shouldn't happen
            best_decomp = decomp
            best_cost_found = trial_cost
        
        # Get the final circuit and parameters from best restart
        final_circuit = best_decomp.get_Circuit()
        final_params = best_decomp.get_Optimized_Parameters()
        
        # Store the results
        self.best_decomposition = best_decomp
        self.best_gate_structure = final_circuit
        self.best_parameters = final_params.copy()
        self.best_gray_code = gray_code.copy() if gray_code else []
        self.current_minimum = best_cost_found
    
    def _perform_optimization(
        self,
        gate_structure
    ) -> Tuple[float, np.ndarray]:
        """
        Perform optimization on a gate structure.
        
        Args:
            gate_structure: Circuit structure to optimize
        
        Returns:
            Tuple of (cost, optimized_parameters)
        """
        # Check if using C++ or Python optimizer
        if self.optimizer in ['BFGS', 'ADAM', 'BFGS2', 'ADAM_BATCHED']:
            return self._optimize_cpp(gate_structure)
        else:
            return self._optimize_python(gate_structure)
    
    def _optimize_cpp(self, gate_structure) -> Tuple[float, np.ndarray]:
        """Optimize using C++ optimizer."""
        from squander import N_Qubit_Decomposition_custom
        
        # Create decomposition instance
        decomp = N_Qubit_Decomposition_custom(
            self.Umtx.conj().T,
            qbit_num=self.qbit_num,
            initial_guess="random",
            config=self.config
        )
        
        # Set custom gate structure
        decomp.set_Gate_Structure(gate_structure)
        
        # Configure optimization
        # get_gate_num() in C++ returns number of layers (sub-circuits), not total gates
        # Count the number of layers (sub-circuits) in the gate structure
        gates = gate_structure.get_Gates()
        num_layers = sum(1 for gate in gates if hasattr(gate, 'get_Gates'))  # Count Circuit objects (layers)
        if num_layers == 0:
            num_layers = 1  # At least one layer
        decomp.set_Optimization_Blocks(num_layers)
        
        # Note: set_Max_Iterations() sets INNER iterations, not outer iterations
        # Outer iterations default to 4 for BFGS (qbit_num <= 5) or 1 for ADAM (qbit_num > 5)
        # and can be set via config if needed, but there's no direct Python method for it
        
        decomp.set_Verbose(self.verbose if self.verbose <= 1 else 1)
        decomp.set_Cost_Function_Variant(costfnc=self.cost_function_variant)
        decomp.set_Optimization_Tolerance(self.optimization_tolerance)
        decomp.set_Trace_Offset(trace_offset=self.trace_offset)
        decomp.set_Optimizer(optimizer=self.optimizer)
        
        if self.project_name:
            decomp.set_Project_Name(self.project_name)
        
        # Set optimizer-specific INNER iteration parameters
        # set_Max_Iterations() sets max_inner_iterations, not max_outer_iterations
        if self.optimizer in ['ADAM', 'BFGS2']:
            param_num = gate_structure.get_Parameter_Num()
            if self.max_inner_iterations is None:
                max_inner = int(param_num / 852 * 1e7)
            else:
                max_inner = self.max_inner_iterations
            decomp.set_Max_Iterations(max_inner)
            # Note: set_random_shift_count_max is not exposed in Python wrapper
        elif self.optimizer == 'ADAM_BATCHED':
            if self.max_inner_iterations is None:
                decomp.set_Max_Iterations(2000)
            else:
                decomp.set_Max_Iterations(self.max_inner_iterations)
        elif self.optimizer == 'BFGS':
            if self.max_inner_iterations is None:
                decomp.set_Max_Iterations(10000)
            else:
                decomp.set_Max_Iterations(self.max_inner_iterations)
        
        # Start optimization
        decomp.Start_Decomposition()
        
        # Get results
        # IMPORTANT: After Start_Decomposition(), get_Circuit() returns the FINAL circuit
        # structure. get_Optimized_Parameters() returns parameters that match this final
        # structure. However, if the circuit was modified (e.g., flattened), the parameters
        # might not match.
        final_circuit = decomp.get_Circuit()
        parameters = decomp.get_Optimized_Parameters()
        num_iters = decomp.get_Num_of_Iters()
        
        # Verify parameter count matches final circuit
        final_param_num = final_circuit.get_Parameter_Num()
        original_param_num = gate_structure.get_Parameter_Num()
        
        # Calculate cost using Optimization_Problem()
        # CRITICAL: The C++ code uses get_current_minimum() which is not exposed in Python.
        # We need to calculate the cost with the correct structure and parameters.
        #
        # If parameters match original structure but final circuit was modified, we need to
        # calculate cost with the original structure and full parameters (not truncated).
        if len(parameters) != final_param_num and len(parameters) == original_param_num:
            # Parameters match original structure, but final circuit was modified
            # Calculate cost with original structure and full parameters
            # This matches what the optimization actually optimized
            try:
                temp_decomp = N_Qubit_Decomposition_custom(
                    self.Umtx.conj().T,
                    qbit_num=self.qbit_num,
                    initial_guess="random",
                    config=self.config
                )
                temp_decomp.set_Gate_Structure(gate_structure)
                # Count layers for optimization blocks
                gates = gate_structure.get_Gates()
                num_layers = sum(1 for gate in gates if hasattr(gate, 'get_Gates'))
                if num_layers == 0:
                    num_layers = 1
                temp_decomp.set_Optimization_Blocks(num_layers)
                temp_decomp.set_Cost_Function_Variant(costfnc=self.cost_function_variant)
                # Use full parameters with original structure for cost calculation
                # This gives us the actual optimized cost
                cost = temp_decomp.Optimization_Problem(parameters)
                if not np.isfinite(cost):
                    # If cost is inf/nan, fall back to using final circuit
                    if self.verbose >= 2:
                        print(f"  Warning: Cost with original structure is {cost}, using final circuit")
                    cost = decomp.Optimization_Problem(parameters[:final_param_num] if len(parameters) > final_param_num else parameters)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  Warning: Error calculating cost with original structure: {e}")
                # Fall back to using final circuit
                cost = decomp.Optimization_Problem(parameters[:final_param_num] if len(parameters) > final_param_num else parameters)
        else:
            # Parameters should match final circuit - verify and use
            if len(parameters) != final_param_num:
                # Mismatch - adjust parameters to match final circuit
                if len(parameters) > final_param_num:
                    parameters = parameters[:final_param_num]
                elif len(parameters) < final_param_num:
                    parameters = np.pad(parameters, (0, final_param_num - len(parameters)), 'constant')
            cost = decomp.Optimization_Problem(parameters)
        
        self.number_of_iters += num_iters
        
        # IMPORTANT: Store and return the FULL parameters (45 for depth 6: 6*6 + 3*3)
        # that match the ORIGINAL hierarchical structure. The optimization was done with
        # the hierarchical structure, so those are the correct parameters to store.
        # The final flattened circuit (9 params) is just for export - we should use
        # the hierarchical structure with full parameters.
        
        # Store this decomposition instance if it's the best so far
        if cost < self.current_minimum:
            self.best_decomposition = decomp
            # Store the final circuit structure (flattened, for Qiskit export via get_Qiskit_Circuit())
            self.best_final_circuit = final_circuit
            # Store the FULL parameters (45) matching the original hierarchical structure
            # These are the parameters that were actually optimized
            self.best_parameters = parameters.copy()
            # Store the original hierarchical gate structure for reconstruction/verification
            self.best_gate_structure = gate_structure
        
        # Return the FULL parameters (45) matching the original hierarchical structure
        return (cost, parameters)
    
    def _optimize_python(self, gate_structure) -> Tuple[float, np.ndarray]:
        """Optimize using Python optimizer (scipy)."""
        # This will be implemented to support scipy optimizers
        # For now, fall back to C++ optimizer
        if self.verbose >= 1:
            print(f"Warning: Python optimizer '{self.optimizer}' not yet implemented, using BFGS")
        self.optimizer = 'BFGS'
        return self._optimize_cpp(gate_structure)
    
    def get_qiskit_circuit(self):
        """
        Get Qiskit circuit from best solution.
        
        Returns:
            Qiskit QuantumCircuit
        """
        # Primary method: Use the decomposition instance's get_Qiskit_Circuit
        # This is the most reliable since it uses the exact circuit that was
        # optimized with the stored parameters
        if self.best_decomposition is not None:
            try:
                return self.best_decomposition.get_Qiskit_Circuit()
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Warning: Could not get Qiskit circuit from decomposition: {e}")
        
        # Fallback: Use Qiskit_IO with stored gate structure and parameters
        from squander import Qiskit_IO
        
        if self.best_gate_structure is not None and self.best_parameters is not None:
            param_num = self.best_gate_structure.get_Parameter_Num()
            if len(self.best_parameters) == param_num:
                flat_circuit = self.best_gate_structure.get_Flat_Circuit()
                return Qiskit_IO.get_Qiskit_Circuit(flat_circuit, self.best_parameters)
            else:
                if self.verbose >= 1:
                    print(f"Warning: Parameter count mismatch: "
                          f"gate structure expects {param_num}, "
                          f"got {len(self.best_parameters)}")
        
        # Last resort: Reconstruct from Gray code
        if self.best_gray_code is None or len(self.best_gray_code) == 0:
            from squander.synthesis.circuit_builder import add_finalizing_layer
            from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
            
            gate_structure = Circuit(self.qbit_num)
            add_finalizing_layer(gate_structure, self.qbit_num)
        else:
            gate_structure = construct_gate_structure_from_gray_code(
                self.best_gray_code,
                self.topology,
                self.qbit_num
            )
        
        flat_circuit = gate_structure.get_Flat_Circuit()
        return Qiskit_IO.get_Qiskit_Circuit(flat_circuit, self.best_parameters)


class AStarSearchDecomposition(TreeSearchDecomposition):
    """
    A* search based decomposition that explores gate structures using a
    heuristic score: Gray-code length (#expanded counters) + cost_weight * cost_function.
    """

    def __init__(
        self,
        Umtx: np.ndarray,
        topology: Optional[List[Tuple[int, int]]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        super().__init__(
            Umtx=Umtx,
            topology=topology,
            config=config,
            verbose=verbose,
            search_name="A* Search Decomposition"
        )
        self.astar_cost_weight = float(self.config.get('astar_cost_weight', 100.0))
        max_expansions = self.config.get('astar_max_expansions', None)
        self.astar_max_expansions = (
            int(max_expansions) if max_expansions is not None else None
        )
        self.astar_cost_epsilon = float(self.config.get('astar_cost_epsilon', 1e-12))

    def start_decomposition(self) -> Dict[str, Any]:
        """
        Run A* search over gate structures using a priority queue.
        """
        start_time = time.time()
        self.current_minimum = float('inf')
        self.best_gray_code = []
        self.best_parameters = None
        self.best_decomposition = None
        self.best_final_circuit = None
        self.number_of_iters = 0

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Starting {self.search_name} for {self.qbit_num}-qubit matrix")
            print(f"Heuristic: (#Gray counters) + {self.astar_cost_weight:.1f} * cost")
            print(f"{'='*70}")

        topology_size = len(self.topology)
        if topology_size == 0:
            raise RuntimeError("Topology must contain at least one edge for A* search")

        visited_costs: Dict[Tuple[int, ...], float] = {}
        entry_counter = itertools.count()
        priority_queue: List[Tuple[float, int, Tuple[int, ...]]] = []
        nodes_evaluated = 0
        budget_exhausted = False

        previous_best = self.current_minimum
        try:
            root_cost, _ = self._evaluate_gray_code([])
        except ValueError as exc:
            raise RuntimeError(f"Failed to evaluate root gate structure: {exc}") from exc
        nodes_evaluated += 1
        root_key: Tuple[int, ...] = tuple()
        visited_costs[root_key] = root_cost
        self.best_gray_code = []

        if root_cost + self.astar_cost_epsilon < previous_best:
            # Explicitly record that the empty structure is the current best
            self.current_minimum = root_cost

        heapq.heappush(
            priority_queue,
            (self._compute_heuristic(0, root_cost), next(entry_counter), root_key)
        )

        if self.verbose >= 2:
            print(f"  Seed node: level=0 cost={root_cost:.6e}")

        while priority_queue:
            _, _, current_key = heapq.heappop(priority_queue)
            current_level = len(current_key)

            if self.current_minimum < self.optimization_tolerance:
                break

            if (
                self.astar_max_expansions is not None
                and nodes_evaluated >= self.astar_max_expansions
            ):
                budget_exhausted = True
                break

            if current_level >= self.tree_level_max:
                continue

            for edge_idx in range(topology_size):
                child_code = list(current_key) + [edge_idx]
                previous_best = self.current_minimum
                try:
                    cost, _ = self._evaluate_gray_code(child_code)
                except ValueError as exc:
                    if self.verbose >= 2:
                        print(
                            f"  Skipping invalid gate structure for edge {edge_idx}: {exc}"
                        )
                    continue
                nodes_evaluated += 1

                if cost + self.astar_cost_epsilon < previous_best:
                    self.current_minimum = cost
                    self.best_gray_code = child_code.copy()

                    if self.verbose >= 2:
                        print(
                            f"  New best: level={len(child_code)} cost={cost:.6e} "
                            f"heap_size={len(priority_queue)}"
                        )

                child_key = tuple(child_code)

                if not self._should_enqueue(child_key, cost, visited_costs):
                    if (
                        self.astar_max_expansions is not None
                        and nodes_evaluated >= self.astar_max_expansions
                    ):
                        budget_exhausted = True
                        break
                    continue

                heuristic_value = self._compute_heuristic(len(child_code), cost)
                heapq.heappush(
                    priority_queue,
                    (heuristic_value, next(entry_counter), child_key)
                )

                if (
                    self.astar_max_expansions is not None
                    and nodes_evaluated >= self.astar_max_expansions
                ):
                    budget_exhausted = True
                    break

            if budget_exhausted:
                break

        total_time = time.time() - start_time

        result = {
            'success': self.current_minimum < self.optimization_tolerance,
            'cost': self.current_minimum,
            'level': len(self.best_gray_code),
            'gray_code': self.best_gray_code,
            'parameters': self.best_parameters,
            'total_time': total_time,
            'number_of_iters': self.number_of_iters,
            'nodes_evaluated': nodes_evaluated
        }

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print("A* Search Results:")
            print(f"  Success: {result['success']}")
            print(f"  Final cost: {result['cost']:.6e}")
            print(f"  Best level: {result['level']}")
            print(f"  Nodes evaluated: {nodes_evaluated}")
            if budget_exhausted:
                print("  Note: Max A* expansions reached before convergence")
            print(f"  Total time: {result['total_time']:.1f}s")
            print(f"  Total iterations: {result['number_of_iters']}")
            print(f"{'='*70}\n")

        return result

    def _compute_heuristic(self, counter_length: int, cost: float) -> float:
        """Heuristic score = Gray-code length + weight * optimization cost."""
        return float(counter_length) + self.astar_cost_weight * float(cost)

    def _evaluate_gray_code(
        self,
        gray_code: List[int]
    ) -> Tuple[float, np.ndarray]:
        """Optimize a gate structure defined by a Gray code prefix."""
        if len(gray_code) == 0:
            _, cost, params = self._optimize_empty_gate_structure()
            return cost, params

        gate_structure = construct_gate_structure_from_gray_code(
            gray_code,
            self.topology,
            self.qbit_num
        )
        return self._perform_optimization(gate_structure)

    def _should_enqueue(
        self,
        key: Tuple[int, ...],
        cost: float,
        visited_costs: Dict[Tuple[int, ...], float]
    ) -> bool:
        """Check whether a state should be enqueued based on previous visits."""
        previous_cost = visited_costs.get(key, None)
        if previous_cost is None or cost + self.astar_cost_epsilon < previous_cost:
            visited_costs[key] = cost
            return True
        return False


def _tree_search_worker(
    start_idx: int,
    end_idx: int,
    Umtx: np.ndarray,
    qbit_num: int,
    topology: List[Tuple[int, int]],
    level_num: int,
    n_ary_limit_max: int,
    config: Dict,
    optimization_tolerance: float,
    verbose: int
) -> Optional[Tuple[int, float, List[int], np.ndarray, int]]:
    """
    Worker function for parallel tree search.
    
    Args:
        start_idx: Starting Gray code index
        end_idx: Ending Gray code index (exclusive)
        Umtx: Unitary matrix to decompose
        qbit_num: Number of qubits
        topology: List of (target, control) pairs
        level_num: Number of CNOT layers
        n_ary_limit_max: Number of topology edges
        config: Configuration dictionary
        optimization_tolerance: Tolerance for early stopping
        verbose: Verbosity level
    
    Returns:
        Tuple of (num_iterations, best_cost, best_gray_code, best_params, nodes_evaluated) or None if no solution
    """
    import numpy as np
    
    # Create Gray code counter
    n_ary_limits = [n_ary_limit_max] * level_num
    gcode_counter = NaryGrayCodeCounter(n_ary_limits, start_idx)
    gcode_counter.set_offset_max(end_idx - 1)
    
    local_best_cost = float('inf')
    local_best_gray_code = None
    local_best_params = None
    total_iters = 0
    nodes_evaluated = 0
    
    # Import here to avoid pickling issues
    from squander.synthesis.circuit_builder import construct_gate_structure_from_gray_code
    
    # Iterate over assigned range
    iter_idx = start_idx
    while iter_idx < end_idx:
        # Get current Gray code
        gray_code = gcode_counter.get()
        
        # Construct gate structure
        try:
            gate_structure = construct_gate_structure_from_gray_code(
                gray_code,
                topology,
                qbit_num
            )
        except Exception as e:
            if verbose >= 2:
                print(f"Error constructing gate structure: {e}")
            iter_idx += 1
            if iter_idx < end_idx:
                done, _, _, _ = gcode_counter.next()
                if done:
                    break
            continue
        
        # Optimize (create fresh optimizer for each structure)
        try:
            cost, params = _optimize_structure(
                gate_structure,
                Umtx,
                qbit_num,
                config,
                verbose
            )
            
            total_iters += 1  # Approximate iteration count
            nodes_evaluated += 1  # Count this node as evaluated
            
            # Update local best
            if cost < local_best_cost:
                local_best_cost = cost
                local_best_gray_code = gray_code.copy()
                local_best_params = params.copy()
                
                # Early stopping if tolerance met
                if cost < optimization_tolerance:
                    if verbose >= 2:
                        print(f"  Worker found optimal solution: cost = {cost:.6e}")
                    break
                        
        except Exception as e:
            if verbose >= 2:
                print(f"Error optimizing structure: {e}")
        
        # Move to next Gray code
        iter_idx += 1
        if iter_idx < end_idx:
            done, _, _, _ = gcode_counter.next()
            if done:
                break
    
    if local_best_gray_code is not None:
        return (total_iters, local_best_cost, local_best_gray_code, local_best_params, nodes_evaluated)
    else:
        return None


def _optimize_structure(
    gate_structure,
    Umtx: np.ndarray,
    qbit_num: int,
    config: Dict,
    verbose: int
) -> Tuple[float, np.ndarray]:
    """Optimize a single gate structure."""
    from squander import N_Qubit_Decomposition_custom
    
    optimizer_name = config.get('optimizer', 'BFGS')
    
    # Create decomposition instance
    decomp = N_Qubit_Decomposition_custom(
        Umtx.conj().T,
        qbit_num=qbit_num,
        initial_guess="random",
        config=config
    )
    
    decomp.set_Gate_Structure(gate_structure)
    # Count the number of layers (sub-circuits) in the gate structure
    gates = gate_structure.get_Gates()
    num_layers = sum(1 for gate in gates if hasattr(gate, 'get_Gates'))  # Count Circuit objects (layers)
    if num_layers == 0:
        num_layers = 1  # At least one layer
    decomp.set_Optimization_Blocks(num_layers)
    decomp.set_Verbose(0)  # Reduce verbosity in workers
    decomp.set_Cost_Function_Variant(costfnc=config.get('cost_function_variant', 3))
    decomp.set_Optimization_Tolerance(config.get('optimization_tolerance', 1e-8))
    decomp.set_Optimizer(optimizer=optimizer_name)
    
    # Set optimizer-specific INNER iteration parameters
    # set_Max_Iterations() sets max_inner_iterations, not max_outer_iterations
    max_inner_iters = config.get('max_inner_iterations', None)
    if optimizer_name in ['ADAM', 'BFGS2']:
        param_num = gate_structure.get_Parameter_Num()
        if max_inner_iters is None:
            max_inner = int(param_num / 852 * 1e7)
        else:
            max_inner = max_inner_iters
        decomp.set_Max_Iterations(max_inner)
        # Note: set_random_shift_count_max is not exposed in Python wrapper
    elif optimizer_name == 'ADAM_BATCHED':
        if max_inner_iters is None:
            decomp.set_Max_Iterations(2000)
        else:
            decomp.set_Max_Iterations(max_inner_iters)
    elif optimizer_name == 'BFGS':
        if max_inner_iters is None:
            decomp.set_Max_Iterations(10000)
        else:
            decomp.set_Max_Iterations(max_inner_iters)
    
    # Optimize
    decomp.Start_Decomposition()
    
    # Get results
    # IMPORTANT: After Start_Decomposition(), get_Circuit() returns the FINAL circuit structure.
    # get_Optimized_Parameters() should return parameters matching this final structure.
    final_circuit = decomp.get_Circuit()
    params = decomp.get_Optimized_Parameters()
    
    # Verify parameter count matches final circuit
    final_param_num = final_circuit.get_Parameter_Num()
    original_param_num = gate_structure.get_Parameter_Num()
    
    # Calculate cost - use same logic as _optimize_cpp
    if len(params) != final_param_num and len(params) == original_param_num:
        # Parameters match original structure, but final circuit was modified
        # Calculate cost with original structure and full parameters
        try:
            temp_decomp = N_Qubit_Decomposition_custom(
                Umtx.conj().T,
                qbit_num=qbit_num,
                initial_guess="random",
                config=config
            )
            temp_decomp.set_Gate_Structure(gate_structure)
            gates = gate_structure.get_Gates()
            num_layers = sum(1 for gate in gates if hasattr(gate, 'get_Gates'))
            if num_layers == 0:
                num_layers = 1
            temp_decomp.set_Optimization_Blocks(num_layers)
            temp_decomp.set_Cost_Function_Variant(costfnc=config.get('cost_function_variant', 3))
            # Use full parameters with original structure for cost calculation
            cost = temp_decomp.Optimization_Problem(params)
            if not np.isfinite(cost):
                # Fall back to using final circuit
                cost = decomp.Optimization_Problem(params[:final_param_num] if len(params) > final_param_num else params)
        except Exception as e:
            if verbose >= 2:
                print(f"  Error in worker cost calculation: {e}")
            # Fall back to using final circuit
            cost = decomp.Optimization_Problem(params[:final_param_num] if len(params) > final_param_num else params)
    else:
        # Parameters match final circuit - use directly
        if len(params) != final_param_num:
            # Adjust to match
            if len(params) > final_param_num:
                params = params[:final_param_num]
            elif len(params) < final_param_num:
                params = np.pad(params, (0, final_param_num - len(params)), 'constant')
        cost = decomp.Optimization_Problem(params)
    
    return (cost, params)

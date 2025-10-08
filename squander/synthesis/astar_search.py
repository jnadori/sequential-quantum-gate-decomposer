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
import heapq
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.synthesis.circuit_builder import (
    construct_circuit_from_gray_code,
)
from squander.synthesis.optimizer_wrapper import optimize_circuit, OptimizationResult


@dataclass(order=True)
class SearchNode:
    """
    Node in A* search tree for quantum circuit synthesis.

    Attributes:
        f_cost: Total estimated cost (g + h), used for priority ordering
        g_cost: Actual optimization cost achieved so far
        h_cost: Heuristic estimate of remaining cost
        level: Tree depth (number of CNOT layers)
        gray_code: Configuration array indexing into topology
        circuit: Gate structure (not used in comparison)
        parameters: Optimized circuit parameters (not used in comparison)
        node_id: Unique identifier for this node
    """
    f_cost: float
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    level: int = field(compare=False)
    gray_code: np.ndarray = field(compare=False, default=None)
    circuit: Circuit = field(compare=False, default=None)
    parameters: np.ndarray = field(compare=False, default=None)
    node_id: int = field(compare=False, default=0)


class HeuristicEvaluator:
    """
    Fast heuristic evaluation for A* search.

    Provides quick cost estimates without full optimization to guide search.
    """

    def __init__(self, Umtx: np.ndarray, config: Dict[str, Any] = None):
        """
        Initialize heuristic evaluator.

        Args:
            Umtx: Target unitary matrix (conjugate transposed)
            config: Configuration dictionary
        """
        self.Umtx = Umtx
        self.config = config if config is not None else {}
        self.qbit_num = int(np.round(np.log2(len(Umtx))))

        # Heuristic configuration
        self.heuristic_type = self.config.get('heuristic_type', 'quick_opt')
        self.quick_iterations = self.config.get('heuristic_iterations', 10)

    def evaluate(self, circuit: Circuit, verbose: int = 0) -> float:
        """
        Evaluate heuristic cost estimate for a circuit.

        Args:
            circuit: Circuit structure to evaluate
            verbose: Verbosity level

        Returns:
            Heuristic cost estimate (lower bound on achievable cost)
        """
        if self.heuristic_type == 'quick_opt':
            return self._quick_optimization_heuristic(circuit, verbose)
        elif self.heuristic_type == 'gradient_norm':
            return self._gradient_norm_heuristic(circuit, verbose)
        elif self.heuristic_type == 'random_sample':
            return self._random_sample_heuristic(circuit, verbose)
        else:
            # Default: no heuristic (admissible but not informative)
            return 0.0

    def _quick_optimization_heuristic(self, circuit: Circuit, verbose: int = 0) -> float:
        """
        Heuristic based on quick partial optimization (few iterations).

        This gives a reasonable estimate of achievable cost with limited computation.
        """
        # Quick optimization config
        quick_config = self.config.copy()
        quick_config['max_iterations'] = self.quick_iterations
        quick_config['tolerance'] = 1e-4  # Less strict for heuristic

        # Use fast optimizer
        result = optimize_circuit(
            self.Umtx,
            circuit,
            optimizer=self.config.get('optimizer', 'scipy:L-BFGS-B'),
            config=quick_config,
            verbose=max(0, verbose - 2)
        )

        return result.cost

    def _gradient_norm_heuristic(self, circuit: Circuit, verbose: int = 0) -> float:
        """
        Heuristic based on gradient norm at random initialization.

        Large gradient suggests parameters far from optimum.
        """
        from squander import N_Qubit_Decomposition_custom

        # Create decomposer
        decomposer = N_Qubit_Decomposition_custom(
            self.Umtx,
            config=self.config,
            accelerator_num=0
        )
        decomposer.set_Gate_Structure(circuit)

        # Random initialization
        param_num = decomposer.get_Parameter_Num()
        params = np.random.rand(param_num) * 2 * np.pi

        # Get cost and gradient
        cost = decomposer.Optimization_Problem(params)
        grad = decomposer.Optimization_Problem_Grad(params)
        grad_norm = np.linalg.norm(grad)

        # Heuristic: combine cost with scaled gradient norm
        # Higher gradient suggests more room for improvement
        return cost * 0.5 + grad_norm * 0.1

    def _random_sample_heuristic(self, circuit: Circuit, verbose: int = 0) -> float:
        """
        Heuristic based on minimum cost from random parameter samples.

        Sample multiple random initializations and return best cost.
        """
        from squander import N_Qubit_Decomposition_custom

        # Create decomposer
        decomposer = N_Qubit_Decomposition_custom(
            self.Umtx,
            config=self.config,
            accelerator_num=0
        )
        decomposer.set_Gate_Structure(circuit)

        # Sample random parameters
        param_num = decomposer.get_Parameter_Num()
        num_samples = self.config.get('heuristic_samples', 5)

        min_cost = float('inf')
        for _ in range(num_samples):
            params = np.random.rand(param_num) * 2 * np.pi
            cost = decomposer.Optimization_Problem(params)
            min_cost = min(min_cost, cost)

        return min_cost


class AStarTreeSearch:
    """
    A* search based quantum circuit synthesis.

    Uses informed search with heuristics to find optimal gate structures
    more efficiently than exhaustive enumeration.
    """

    def __init__(
        self,
        Umtx: np.ndarray,
        topology: List[Tuple[int, int]],
        config: Optional[Dict[str, Any]] = None,
        verbose: int = 1
    ):
        """
        Initialize A* tree search.

        Args:
            Umtx: Target unitary matrix (conjugate transposed)
            topology: List of (target, control) qubit pairs
            config: Configuration dictionary with keys:
                - tree_level_max: Maximum tree depth (default 10)
                - optimization_tolerance: Target cost (default 1e-8)
                - optimizer: Optimizer specification (default 'scipy:L-BFGS-B')
                - max_iterations: Max iterations per optimization (default 1000)
                - beam_width: Beam search width (0 = no beam search, default 0)
                - heuristic_type: Type of heuristic ('quick_opt', 'gradient_norm', 'random_sample')
                - heuristic_iterations: Iterations for quick_opt heuristic (default 10)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug)
        """
        self.Umtx = Umtx
        self.qbit_num = int(np.round(np.log2(len(Umtx))))
        self.topology = topology
        self.n_ary_limit = len(topology)

        # Configuration
        self.config = config if config is not None else {}
        self.tree_level_max = self.config.get('tree_level_max', 10)
        self.optimization_tolerance = self.config.get('optimization_tolerance', 1e-8)
        self.optimizer = self.config.get('optimizer', 'scipy:L-BFGS-B')
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.beam_width = self.config.get('beam_width', 0)  # 0 = no beam search

        self.verbose = verbose

        # Heuristic evaluator
        self.heuristic = HeuristicEvaluator(Umtx, self.config)

        # Results
        self.best_node: Optional[SearchNode] = None
        self.best_cost = float('inf')

        # Statistics
        self.nodes_evaluated = 0
        self.nodes_expanded = 0
        self.nodes_pruned = 0
        self.node_counter = 0  # For unique IDs

    def _log(self, level: int, message: str):
        """Print log message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(message, flush=True)

    def _create_initial_nodes(self) -> List[SearchNode]:
        """
        Create initial nodes at level 1 (one CNOT layer).

        We skip level 0 (empty circuit) as it doesn't have CNOTs.
        """
        initial_nodes = []

        # Create one node for each topology option
        for topo_idx in range(self.n_ary_limit):
            gray_code = np.array([topo_idx], dtype=np.int32)

            # Build circuit
            circuit = construct_circuit_from_gray_code(
                gray_code,
                self.qbit_num,
                self.topology,
                use_u3=True
            )

            # Optimize to get g_cost
            opt_config = {
                'max_iterations': self.max_iterations,
                'tolerance': self.optimization_tolerance,
                'cost_function_variant': self.config.get('cost_function_variant', 3),
            }

            result = optimize_circuit(
                self.Umtx,
                circuit,
                optimizer=self.optimizer,
                config=opt_config,
                verbose=max(0, self.verbose - 2)
            )

            g_cost = result.cost
            h_cost = 0.0  # Admissible heuristic for now

            node = SearchNode(
                f_cost=g_cost + h_cost,
                g_cost=g_cost,
                h_cost=h_cost,
                level=1,
                gray_code=gray_code,
                circuit=circuit,
                parameters=result.parameters,
                node_id=self._get_next_node_id()
            )

            initial_nodes.append(node)
            self.nodes_evaluated += 1

            self._log(3, f"  Initial node {topo_idx}: topo={self.topology[topo_idx]}, "
                        f"g={g_cost:.6e}, f={node.f_cost:.6e}")

        return initial_nodes

    def _get_next_node_id(self) -> int:
        """Get unique node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _expand_node(self, node: SearchNode) -> List[SearchNode]:
        """
        Expand node by adding one CNOT layer with all topology options.

        Args:
            node: Node to expand

        Returns:
            List of child nodes
        """
        if node.level >= self.tree_level_max:
            return []

        children = []

        # Try each topology option for next layer
        for topo_idx in range(self.n_ary_limit):
            # Extend gray code
            new_gray_code = np.append(node.gray_code, topo_idx)

            # Build circuit
            circuit = construct_circuit_from_gray_code(
                new_gray_code,
                self.qbit_num,
                self.topology,
                use_u3=True
            )

            # Full optimization for g_cost
            opt_config = {
                'max_iterations': self.max_iterations,
                'tolerance': self.optimization_tolerance,
                'cost_function_variant': self.config.get('cost_function_variant', 3),
            }

            result = optimize_circuit(
                self.Umtx,
                circuit,
                optimizer=self.optimizer,
                config=opt_config,
                verbose=max(0, self.verbose - 2)
            )

            g_cost = result.cost

            # Evaluate heuristic for remaining cost (h_cost)
            # For now, use 0 (admissible) - can be improved with better heuristic
            h_cost = 0.0

            # Create child node
            child = SearchNode(
                f_cost=g_cost + h_cost,
                g_cost=g_cost,
                h_cost=h_cost,
                level=node.level + 1,
                gray_code=new_gray_code,
                circuit=circuit,
                parameters=result.parameters,
                node_id=self._get_next_node_id()
            )

            children.append(child)
            self.nodes_evaluated += 1

            # Log child creation
            self._log(3, f"      Child {topo_idx}: topo={self.topology[topo_idx]}, "
                        f"g={g_cost:.6e}, h={h_cost:.6e}, f={child.f_cost:.6e}")

        self.nodes_expanded += 1
        return children

    def _should_prune(self, node: SearchNode) -> bool:
        """
        Determine if node should be pruned.

        Args:
            node: Node to check

        Returns:
            True if node should be pruned
        """
        # Prune if f_cost exceeds best found solution
        if node.f_cost > self.best_cost:
            return True

        # Prune if we already found a solution meeting tolerance
        if self.best_cost < self.optimization_tolerance:
            return True

        return False

    def search(self) -> Dict[str, Any]:
        """
        Run A* search for circuit decomposition.

        Returns:
            Dictionary with search results:
                - success: Whether target tolerance was achieved
                - cost: Best cost found
                - level: Best tree level
                - gray_code: Best configuration
                - circuit: Best circuit
                - parameters: Optimized parameters
                - total_time: Total search time
                - nodes_evaluated: Number of nodes evaluated
                - nodes_expanded: Number of nodes expanded
                - nodes_pruned: Number of nodes pruned
        """
        self._log(1, "="*70)
        self._log(1, f"A* Tree Search Decomposition for {self.qbit_num}-qubit unitary")
        self._log(1, f"Topology: {len(self.topology)} allowed CNOT connections")
        self._log(1, f"Optimizer: {self.optimizer}")
        self._log(1, f"Heuristic: {self.config.get('heuristic_type', 'quick_opt')}")
        self._log(1, f"Target tolerance: {self.optimization_tolerance:.2e}")
        if self.beam_width > 0:
            self._log(1, f"Beam width: {self.beam_width}")
        self._log(1, "="*70)

        start_time = time.time()

        # Initialize priority queue with level 1 nodes
        self._log(2, "\nCreating initial nodes at level 1...")
        priority_queue = []
        initial_nodes = self._create_initial_nodes()

        for node in initial_nodes:
            heapq.heappush(priority_queue, node)

            # Update best if needed
            if node.g_cost < self.best_cost:
                self.best_cost = node.g_cost
                self.best_node = node
                self._log(2, f"  Initial best: cost={self.best_cost:.6e}, level={node.level}")

        self._log(1, f"\nStarting A* search with {len(initial_nodes)} initial nodes...")

        # A* search loop
        while priority_queue:
            # Pop node with lowest f_cost
            current = heapq.heappop(priority_queue)

            self._log(2, f"\nExploring node {current.node_id}: "
                        f"level={current.level}, f={current.f_cost:.6e}, "
                        f"g={current.g_cost:.6e}, h={current.h_cost:.6e}")

            # Check if this is best so far
            if current.g_cost < self.best_cost:
                self.best_cost = current.g_cost
                self.best_node = current
                self._log(1, f"  *** New best: cost={self.best_cost:.6e}, level={current.level}")

            # Check for solution
            if current.g_cost < self.optimization_tolerance:
                self._log(1, f"\n*** Solution found! Cost={current.g_cost:.6e}, Level={current.level}")
                break

            # Don't expand beyond max level
            if current.level >= self.tree_level_max:
                self._log(2, f"  Max level reached, not expanding")
                continue

            # Expand node
            self._log(2, f"  Expanding node {current.node_id}...")
            children = self._expand_node(current)

            # Add children to queue with pruning
            for child in children:
                if self._should_prune(child):
                    self._log(3, f"      Pruning child {child.node_id}: "
                                f"f={child.f_cost:.6e} > best={self.best_cost:.6e}")
                    self.nodes_pruned += 1
                    continue

                heapq.heappush(priority_queue, child)
                self._log(3, f"      Added child {child.node_id} to queue")

            # Beam search: keep only top-k nodes at each level
            if self.beam_width > 0 and len(priority_queue) > self.beam_width:
                # Keep only beam_width best nodes
                priority_queue = heapq.nsmallest(self.beam_width, priority_queue)
                heapq.heapify(priority_queue)
                self._log(2, f"  Beam search: pruned queue to {self.beam_width} nodes")

        total_time = time.time() - start_time

        # Final summary
        self._log(1, "\n" + "="*70)
        self._log(1, "A* Search Complete")
        self._log(1, "="*70)
        self._log(1, f"Best cost: {self.best_cost:.6e}")
        if self.best_node:
            self._log(1, f"Best level: {self.best_node.level}")
        self._log(1, f"Nodes evaluated: {self.nodes_evaluated}")
        self._log(1, f"Nodes expanded: {self.nodes_expanded}")
        self._log(1, f"Nodes pruned: {self.nodes_pruned}")
        self._log(1, f"Total time: {total_time:.1f}s")
        self._log(1, f"Success: {self.best_cost < self.optimization_tolerance}")
        self._log(1, "="*70)

        return {
            'success': self.best_cost < self.optimization_tolerance,
            'cost': self.best_cost,
            'level': self.best_node.level if self.best_node else None,
            'gray_code': self.best_node.gray_code if self.best_node else None,
            'circuit': self.best_node.circuit if self.best_node else None,
            'parameters': self.best_node.parameters if self.best_node else None,
            'total_time': total_time,
            'nodes_evaluated': self.nodes_evaluated,
            'nodes_expanded': self.nodes_expanded,
            'nodes_pruned': self.nodes_pruned,
        }

    def get_circuit(self) -> Optional[Circuit]:
        """Get the best circuit found."""
        return self.best_node.circuit if self.best_node else None

    def get_optimized_parameters(self) -> Optional[np.ndarray]:
        """Get the optimized parameters."""
        return self.best_node.parameters if self.best_node else None

    def get_decomposition_error(self) -> float:
        """Get the final decomposition error."""
        return self.best_cost

    def get_qiskit_circuit(self):
        """Export best circuit to Qiskit format."""
        if self.best_node is None:
            raise ValueError("No circuit found yet. Run search() first.")

        from squander import Qiskit_IO
        return Qiskit_IO.get_Qiskit_Circuit(
            self.best_node.circuit,
            self.best_node.parameters
        )

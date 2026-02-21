## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""


import heapq
import itertools
import time
from typing import List, Tuple, Optional, Dict, Any
import math
import copy
import multiprocessing as mp
import numpy as np

from squander.synthesis.qgd_CircuitNode import qgd_CircuitNode as CircuitNode
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import qgd_N_Qubit_Decomposition_custom as N_Qubit_Decomposition_custom
from squander.synthesis.qgd_POSMM import qgd_POSMM_Decomposition


# Module-level globals populated once per worker by _worker_init
_worker_Umtx = None
_worker_config = None
_worker_two_qbit_basis = None
_worker_qbit_num = None

# Additional globals for approximate search workers
_worker_layer_fidelity = None
_worker_minimal_error = None

# Additional globals for POSMM search workers
_worker_rconstant = None


def _worker_init(Umtx, config, two_qbit_basis, qbit_num):
    """Initializer for pool workers. Called once per process."""
    global _worker_Umtx, _worker_config, _worker_two_qbit_basis, _worker_qbit_num
    _worker_Umtx = Umtx
    _worker_config = config.copy()
    _worker_config['parallel'] = 0          # avoid nested C++ parallelism
    _worker_two_qbit_basis = two_qbit_basis
    _worker_qbit_num = qbit_num


def _decompose_node_worker(node):
    """Worker function for parallel node decomposition.

    Top-level function required by multiprocessing.Pool (bound methods
    are not picklable).  Shared state (Umtx, config, etc.) is read from
    module globals set by _worker_init.

    Returns only (score, params) — no C++ objects — to keep IPC cheap.
    """
    # Build the ansatz circuit (same logic as construct_circuit_from_node)
    node_circuit = Circuit(_worker_qbit_num)
    for trgt_qbit, ctrl_qbit in node.raw_gates:
        layer = _worker_two_qbit_basis.Remap_Qbits(
            {0: trgt_qbit, 1: ctrl_qbit}, _worker_qbit_num
        )
        node_circuit.add_Circuit(layer)
    final_layer = Circuit(_worker_qbit_num)
    for qbit in range(_worker_qbit_num):
        final_layer.add_U3(qbit)
    node_circuit.add_Circuit(final_layer)
    ansatz = node_circuit

    # Create a fresh decomposition instance and run
    cDecompose = N_Qubit_Decomposition_custom(_worker_Umtx, config=_worker_config)
    cDecompose.set_Verbose(0)
    cDecompose.set_Optimization_Tolerance(1e-6)
    cDecompose.set_Cost_Function_Variant(3)
    cDecompose.set_Optimizer("BFGS")
    cDecompose.set_Gate_Structure(ansatz)
    cDecompose.set_Optimized_Parameters(
        np.random.rand(ansatz.get_Parameter_Num()) * 2 * np.pi / 50
    )
    cDecompose.Start_Decomposition()
    params = cDecompose.get_Optimized_Parameters()
    score = cDecompose.Optimization_Problem(params)
    return (score, params)


def _approx_worker_init(Umtx, config, two_qbit_basis, qbit_num, layer_fidelity, minimal_error):
    """Initializer for approximate search pool workers. Called once per process."""
    global _worker_Umtx, _worker_config, _worker_two_qbit_basis, _worker_qbit_num
    global _worker_layer_fidelity, _worker_minimal_error
    _worker_Umtx = Umtx
    _worker_config = config.copy()
    _worker_config['parallel'] = 0          # avoid nested C++ parallelism
    _worker_two_qbit_basis = two_qbit_basis
    _worker_qbit_num = qbit_num
    _worker_layer_fidelity = layer_fidelity
    _worker_minimal_error = minimal_error


def _decompose_node_approx_worker(node):
    """Worker function for parallel approximate node decomposition.

    Like _decompose_node_worker but uses adaptive tolerance based on node depth
    and the layer_fidelity / minimal_error parameters.

    Returns only (score, params) — no C++ objects — to keep IPC cheap.
    """
    # Compute adaptive tolerance from node depth
    level_num = node.depth
    tolerance = max(level_num, 1) * (1 - _worker_layer_fidelity) * _worker_minimal_error

    # Build the ansatz circuit (same logic as construct_circuit_from_node)
    node_circuit = Circuit(_worker_qbit_num)
    for trgt_qbit, ctrl_qbit in node.raw_gates:
        layer = _worker_two_qbit_basis.Remap_Qbits(
            {0: trgt_qbit, 1: ctrl_qbit}, _worker_qbit_num
        )
        node_circuit.add_Circuit(layer)
    final_layer = Circuit(_worker_qbit_num)
    for qbit in range(_worker_qbit_num):
        final_layer.add_U3(qbit)
    node_circuit.add_Circuit(final_layer)
    ansatz = node_circuit

    # Create a fresh decomposition instance and run
    cDecompose = N_Qubit_Decomposition_custom(_worker_Umtx, config=_worker_config)
    cDecompose.set_Verbose(0)
    cDecompose.set_Optimization_Tolerance(tolerance)
    cDecompose.set_Cost_Function_Variant(3)
    cDecompose.set_Optimizer("BFGS")
    cDecompose.set_Gate_Structure(ansatz)
    cDecompose.set_Optimized_Parameters(
        np.random.rand(ansatz.get_Parameter_Num()) * 2 * np.pi / 50
    )
    cDecompose.Start_Decomposition()
    params = cDecompose.get_Optimized_Parameters()
    score = cDecompose.Optimization_Problem(params)
    return (score, params)


def _posmm_worker_init(Umtx, config, two_qbit_basis, qbit_num, rconstant):
    """Initializer for POSMM search pool workers. Called once per process."""
    global _worker_Umtx, _worker_config, _worker_two_qbit_basis, _worker_qbit_num
    global _worker_rconstant
    _worker_Umtx = Umtx
    _worker_config = config.copy()
    _worker_config['parallel'] = 0          # avoid nested C++ parallelism
    _worker_two_qbit_basis = two_qbit_basis
    _worker_qbit_num = qbit_num
    _worker_rconstant = rconstant


def _decompose_node_posmm_worker(node):
    """Worker function for parallel POSMM node decomposition.

    Top-level function required by multiprocessing.Pool.  Shared state is
    read from module globals set by _posmm_worker_init.

    Returns only (score, params) — no C++ objects — to keep IPC cheap.
    """
    node_circuit = Circuit(_worker_qbit_num)
    for trgt_qbit, ctrl_qbit in node.raw_gates:
        layer = _worker_two_qbit_basis.Remap_Qbits(
            {0: trgt_qbit, 1: ctrl_qbit}, _worker_qbit_num
        )
        node_circuit.add_Circuit(layer)
    final_layer = Circuit(_worker_qbit_num)
    for qbit in range(_worker_qbit_num):
        final_layer.add_U3(qbit)
    node_circuit.add_Circuit(final_layer)
    ansatz = node_circuit

    posmm = qgd_POSMM_Decomposition(_worker_Umtx, ansatz, _worker_rconstant, _worker_config)
    params, score = posmm.Start_decomposition()
    return (score, params)


class qgd_TreeSearch:
    def __init__(
        self,
        Umtx,
        topology=None,
        config={},
        two_qbit_basis = None
    ):
        self.Umtx = Umtx
        self.qbit_num = int(np.log2(Umtx.shape[0]))
        if topology is None: 
            topology = []
            for control_qbit in range(self.qbit_num-1):
                for target_qbit in range(control_qbit+1,self.qbit_num):
                    topology.append((target_qbit,control_qbit))
        self.topology = topology
        self.custom_basis = True
        if two_qbit_basis is None:
            self.custom_basis = False
            two_qbit_basis = Circuit(2)
            two_qbit_basis.add_U3(0)
            two_qbit_basis.add_U3(1)
            two_qbit_basis.add_CNOT(0,1)
        self.two_qbit_basis = two_qbit_basis
        config.setdefault('parallel', 0 )
        config.setdefault('parallel_search', 0)
        config.setdefault('verbosity', 0 )
        config.setdefault('tolerance', 1e-8 )
        config.setdefault('optimizer',"BFGS")
        config.setdefault('tree_level_max',math.floor((4**self.qbit_num-3*self.qbit_num-1)/4))
        self.config = config

    def construct_circuit_from_node(self, node):
        node_circuit = Circuit(self.qbit_num)
        connections = node.raw_gates
        for trgt_qbit, ctrl_qbit in connections:
            layer = self.two_qbit_basis.Remap_Qbits({0:trgt_qbit,1:ctrl_qbit},self.qbit_num)
            node_circuit.add_Circuit(layer)
        final_layer = Circuit(self.qbit_num)
        for qbit in range(self.qbit_num):
            final_layer.add_U3(qbit)
        node_circuit.add_Circuit(final_layer)
        return node_circuit
    
    def start_decomposition(self):
        best_circuit = None
        best_score = np.inf
        best_parameters = np.array([])
        start = time.time()
        nodes_evaluated = 0
        previous_nodes = []
        level_nodes = [CircuitNode(self.qbit_num,self.topology)]

        # Create a persistent pool once (if parallel)
        pool = None
        if self.config['parallel_search'] > 0:
            pool = mp.Pool(
                processes=self.config['parallel_search'],
                initializer=_worker_init,
                initargs=(self.Umtx, self.config, self.two_qbit_basis, self.qbit_num),
            )
        try:
            for level in range(0,self.config['tree_level_max']+1):
                previous_nodes += level_nodes
                nodes_evaluated += len(level_nodes)
                level_results = self.search_over_level(level_nodes, pool)
                for result in level_results:
                    if result[0]<best_score:
                        best_score=result[0]
                        best_circuit=result[1]
                        best_parameters=result[2]
                if best_score < self.config['tolerance']:
                    break
                new_level_nodes = []
                for node in level_nodes:
                    new_level_nodes += node.expand(not self.custom_basis)
                level_nodes = new_level_nodes
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if self.config['verbosity']>0:
            print(f"Compiled circuit found at level {level}, compilation time {time.time()-start}")
        return best_circuit,best_parameters,best_score
    
    def search_over_level(self, level_nodes, pool=None):
        if pool is not None:
            worker_results = pool.map(_decompose_node_worker, level_nodes)
            # Reconstruct circuits in main process (cheap) instead of
            # pickling C++ Circuit objects back from workers (expensive).
            results = []
            for node, (score, params) in zip(level_nodes, worker_results):
                circ = self.construct_circuit_from_node(node)
                results.append([score, circ, params])
            return results
        else:
            results = []
            for node in level_nodes:
                cost_function,circ,params = self.decompose_single_node(node)
                results.append([cost_function,circ,params])
            return results

    def decompose_single_node(self,node):
        ansatz = self.construct_circuit_from_node(node)
        cDecompose = N_Qubit_Decomposition_custom(self.Umtx,  config=self.config)
        cDecompose.set_Verbose(0)
        cDecompose.set_Optimization_Tolerance( 1e-6 )
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimizer(self.config["optimizer"])
        cDecompose.set_Gate_Structure(ansatz)
        cDecompose.set_Optimized_Parameters(np.random.rand(ansatz.get_Parameter_Num())*2*np.pi/50)
        cDecompose.Start_Decomposition()
        params = cDecompose.get_Optimized_Parameters()
        score = cDecompose.Optimization_Problem(params)
        return score,ansatz,params

class qgd_ApproximateSearch(qgd_TreeSearch):
    def __init__(
        self,
        Umtx,
        topology=None,
        config={},
        two_qbit_basis=None,
        layer_fidelity=0.995,
        minimal_error=1
    ):
        super().__init__(
            Umtx=Umtx,
            topology=topology,
            config=config,
            two_qbit_basis=two_qbit_basis
        )
        self.layer_fidelity = layer_fidelity
        self.minimal_error = minimal_error

    def calculate_tolerance(self, level_num):
        """Calculate adaptive tolerance based on accumulated layer errors.

        For N layers each with fidelity F, the total error is approximately:
        error ~ N * (1 - F) for small (1-F)

        minimal_error acts as a scaling factor.

        Returns max(level_num, 1) * (1 - layer_fidelity) * minimal_error
        """
        return max(level_num, 1) * (1 - self.layer_fidelity) * self.minimal_error

    def start_decomposition(self):
        best_circuit = None
        best_score = np.inf
        best_parameters = np.array([])
        start = time.time()
        nodes_evaluated = 0
        previous_nodes = []
        level_nodes = [CircuitNode(self.qbit_num, self.topology)]

        # Create a persistent pool once (if parallel)
        pool = None
        if self.config['parallel_search'] > 0:
            pool = mp.Pool(
                processes=self.config['parallel_search'],
                initializer=_approx_worker_init,
                initargs=(self.Umtx, self.config, self.two_qbit_basis,
                          self.qbit_num, self.layer_fidelity, self.minimal_error),
            )
        try:
            for level in range(0, self.config['tree_level_max'] + 1):
                adaptive_tolerance = self.calculate_tolerance(level)
                previous_nodes += level_nodes
                nodes_evaluated += len(level_nodes)
                level_results = self.search_over_level(level_nodes, pool)
                for result in level_results:
                    if result[0] < best_score:
                        best_score = result[0]
                        best_circuit = result[1]
                        best_parameters = result[2]
                if best_score < adaptive_tolerance:
                    break
                new_level_nodes = []
                for node in level_nodes:
                    new_level_nodes += node.expand(not self.custom_basis)
                level_nodes = new_level_nodes
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if self.config['verbosity'] > 0:
            print(f"Compiled circuit found at level {level}, compilation time {time.time()-start}")
        return best_circuit, best_parameters, best_score

    def search_over_level(self, level_nodes, pool=None):
        if pool is not None:
            worker_results = pool.map(_decompose_node_approx_worker, level_nodes)
            results = []
            for node, (score, params) in zip(level_nodes, worker_results):
                circ = self.construct_circuit_from_node(node)
                results.append([score, circ, params])
            return results
        else:
            results = []
            for node in level_nodes:
                cost_function, circ, params = self.decompose_single_node(node)
                results.append([cost_function, circ, params])
            return results

    def decompose_single_node(self, node):
        tolerance = self.calculate_tolerance(node.depth)
        ansatz = self.construct_circuit_from_node(node)
        cDecompose = N_Qubit_Decomposition_custom(self.Umtx, config=self.config)
        cDecompose.set_Verbose(0)
        cDecompose.set_Optimization_Tolerance(tolerance)
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimizer(self.config["optimizer"])
        cDecompose.set_Gate_Structure(ansatz)
        cDecompose.set_Optimized_Parameters(np.random.rand(ansatz.get_Parameter_Num()) * 2 * np.pi / 50)
        cDecompose.Start_Decomposition()
        params = cDecompose.get_Optimized_Parameters()
        score = cDecompose.Optimization_Problem(params)
        return score, ansatz, params


class qgd_POSMMSearch(qgd_TreeSearch):
    """Tree search that uses the POSMM optimizer for each node decomposition.

    POSMM (Parallel Optimization with Stochastic Multi-start Method) samples
    multiple random starting points in parameter space and filters them by a
    distance radius that shrinks with iteration count, keeping optimization
    attempts spread across the landscape.

    Additional config keys (on top of qgd_TreeSearch defaults):
        optimization_loops (int): number of POSMM outer iterations (default 100)
        worker_num        (int): simultaneous starting points per iteration (default 1)

    Args:
        rconstant: radius scaling constant for POSMM distance filtering.
                   Defaults to pi, half the width of the periodic parameter domain.
    """

    def __init__(
        self,
        Umtx,
        topology=None,
        config={},
        two_qbit_basis=None,
        rconstant=np.pi,
    ):
        super().__init__(
            Umtx=Umtx,
            topology=topology,
            config=config,
            two_qbit_basis=two_qbit_basis,
        )
        self.config.setdefault('optimization_loops', 100)
        self.config.setdefault('worker_num', 1)
        self.config.setdefault('screen_iterations', 100)
        self.config.setdefault('refine_count', max(1, self.config['worker_num'] // 4))
        self.config.setdefault('max_stagnation', 5)
        self.config.setdefault('perturbation_scale', 0.5)
        self.config.setdefault('refinement_attempts', 20)
        self.config.setdefault('refinement_scales', [0.1, 0.3, 1.0])
        self.rconstant = rconstant

    def decompose_single_node(self, node):
        ansatz = self.construct_circuit_from_node(node)
        posmm = qgd_POSMM_Decomposition(self.Umtx, ansatz, self.rconstant, self.config)
        params, score = posmm.Start_decomposition()
        return score, ansatz, params

    def search_over_level(self, level_nodes, pool=None):
        if pool is not None:
            worker_results = pool.map(_decompose_node_posmm_worker, level_nodes)
            results = []
            for node, (score, params) in zip(level_nodes, worker_results):
                circ = self.construct_circuit_from_node(node)
                results.append([score, circ, params])
            return results
        else:
            results = []
            for node in level_nodes:
                cost_function, circ, params = self.decompose_single_node(node)
                results.append([cost_function, circ, params])
            return results

    def start_decomposition(self):
        best_circuit = None
        best_score = np.inf
        best_parameters = np.array([])
        start = time.time()
        nodes_evaluated = 0
        previous_nodes = []
        level_nodes = [CircuitNode(self.qbit_num, self.topology)]

        pool = None
        if self.config['parallel_search'] > 0:
            pool = mp.Pool(
                processes=self.config['parallel_search'],
                initializer=_posmm_worker_init,
                initargs=(self.Umtx, self.config, self.two_qbit_basis,
                          self.qbit_num, self.rconstant),
            )
        try:
            for level in range(0, self.config['tree_level_max'] + 1):
                previous_nodes += level_nodes
                nodes_evaluated += len(level_nodes)
                level_results = self.search_over_level(level_nodes, pool)
                for result in level_results:
                    if result[0] < best_score:
                        best_score = result[0]
                        best_circuit = result[1]
                        best_parameters = result[2]
                if best_score < self.config['tolerance']:
                    break
                new_level_nodes = []
                for node in level_nodes:
                    new_level_nodes += node.expand(not self.custom_basis)
                level_nodes = new_level_nodes
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if self.config['verbosity'] > 0:
            print(f"Compiled circuit found at level {level}, compilation time {time.time()-start}")
        return best_circuit, best_parameters, best_score

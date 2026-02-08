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

import multiprocessing as mp
import numpy as np

from squander.synthesis.qgd_CircuitNode import qgd_CircuitNode as CircuitNode
from squander import Circuit

class TreeSearch:
    def __init__(
        self,
        Umtx,
        topology=None,
        config={},
        two_qbit_basis = None
    ):
        self.Umtx = Umtx
        self.qbit_num = np.log2(Umtx.shape[0][0])
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
        config.setdefault()
        self.config = config
        config.setdefault('parallel', 0 )
        config.setdefault('verbosity', 0 )
        config.setdefault('tolerance', 1e-8 )
        config.setdefault('tree_level_max',math.floor((4**self.qbit_num-3*self.qbit_num-1)/4))

    def construct_circuit_from_node(self, node):
        node_circuit = Circuit(self.qbit_num)
        connections = node.raw_gates
        for trgt_qbit, ctrl_qbit in connections:
            layer = self.two_qbit_basis.copy()
            layer.Remap_Qbits({0:trgt_qbit,1:ctrl_qbit},self.qbit_num)
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
        for level in range(0,self.config['tree_level_max']+1):
            previous_nodes += level_nodes
            nodes_evaluated += len(level_nodes)
            
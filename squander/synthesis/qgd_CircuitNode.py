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
import math
   
from typing import List, Tuple, Optional, Set

class qgd_CircuitNode:
    """
    A Node representing a quantum circuit structure for Tree Search optimization.
    
    Features:
    - Stores the physical gate history (raw_gates).
    - Automatically computes a unique Canonical Form (canonical_gates) based on commutation rules.
    - Implements smart hashing and equality for efficient use in Python Sets.
    - specific 'expand' logic to prune redundant branches immediately.
    """

    def __init__(self, num_qubits: int, 
                 topology: List[Tuple[int, int]], 
                 parent: Optional['qgd_CircuitNode'] = None, 
                 new_gate: Optional[Tuple[int, int]] = None):
        """
        Args:
            num_qubits: Total number of qubits.
            topology: List of allowed gate connections (e.g., [(0,1), (1,2)]).
            parent: The existing circuit node to extend.
            new_gate: The specific gate (control, target) to add to the parent's sequence.
        """
        self.num_qubits = num_qubits
        self.topology = topology
        
        # 1. Build Physical History
        # We assume new_gate is (control, target). Direction matters.
        if parent:
            self.raw_gates = parent.raw_gates + [new_gate]
            self.depth = parent.depth + 1
            
            # 2. Incremental Canonicalization (The "Bubble" Optimization)
            # Start with parent's already-sorted form and insert the new gate
            initial_list = list(parent.canonical_gates)
            initial_list.append(new_gate)
            self.canonical_gates = self._insert_and_canonicalize(initial_list)
        else:
            self.raw_gates = []
            self.canonical_gates = tuple()
            self.depth = 0

        # 3. Pre-calculate Hash
        # This is critical for performance in Sets/Dicts
        self._hash_cache = hash(self.canonical_gates)

    def _gates_commute(self, g1: Tuple[int, int], g2: Tuple[int, int]) -> bool:
        """
        Returns True if two CNOT gates commute.
        Standard Rule: They commute if they operate on disjoint sets of qubits.
        Example: (0, 1) commutes with (2, 3), but NOT with (1, 2).
        """
        return set(g1).isdisjoint(set(g2))

    def _insert_and_canonicalize(self, gate_list: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        """
        Takes a list where only the LAST element is out of order.
        Bubbles the last element backwards (left) as far as commutation allows.
        """
        sequence = list(gate_list)
        idx = len(sequence) - 1
        
        while idx > 0:
            current = sequence[idx]
            neighbor = sequence[idx-1]
            
            # SWAP CONDITION:
            # 1. They must physically commute (order doesn't affect quantum state)
            # 2. Neighbor is 'larger' than current (we want small -> large sorting)
            if self._gates_commute(current, neighbor) and neighbor > current:
                sequence[idx], sequence[idx-1] = sequence[idx-1], sequence[idx]
                idx -= 1
            else:
                # Blocked by non-commuting gate or already in order
                break
                
        return tuple(sequence)

    def expand(self, CNOT_basis = True) -> List['qgd_CircuitNode']:
        """
        Generates the next layer of child nodes.
        Includes LOCAL pruning to prevent generating obviously redundant branches.
        """
        children = []
        last_gate = self.raw_gates[-1] if self.raw_gates else None

        for candidate_gate in self.topology:
            # candidate_gate is (control, target)
            qbit_subspaces = []
            if CNOT_basis:
                for qbit_subspace in range(2,self.num_qubits):
                    tlb = self.tlb(qbit_subspace)
                    unique_qbits = frozenset(x for tup in self.gates[-tlb+1:] for x in tup)
                    if len(unique_qbits)<= qbit_subspace:
                        qbit_subspaces.append((qbit_subspace,unique_qbits))


            skip = False
            for qbit_subspace,unique_qbits in qbit_subspaces:
                if (len(unique_qbits) + (candidate_gate[0] not in unique_qbits) + (candidate_gate[1] not in unique_qbits))<= qbit_subspace:
                    skip = True

            if last_gate is not None:
                if self._gates_commute(candidate_gate, last_gate):
                    if candidate_gate < last_gate:
                        skip = True
            if skip:
                continue
            new_node = qgd_CircuitNode(
                num_qubits=self.num_qubits, 
                topology=self.topology, 
                parent=self, 
                new_gate=candidate_gate
            )
            children.append(new_node)
            
        return children
    
    def tlb(self,qbit_space):
        return math.floor((4**qbit_space-3*qbit_space-1)/4)
    # --- PYTHON PROTOCOL METHODS ---

    def __hash__(self):
        # Hashes the canonical form, so [A, B] and [B, A] land in the same bucket
        return self._hash_cache

    def __eq__(self, other):
        # Checks if two nodes represent the same quantum operations
        if not isinstance(other, qgd_CircuitNode):
            return False
        # Fast fail with hash, then deep check
        if self._hash_cache != other._hash_cache:
            return False
        return self.canonical_gates == other.canonical_gates

    def __repr__(self):
        return f"CircuitNode(Depth={self.depth}, Gates={self.raw_gates})"

    def __lt__(self, other):
        # Helper for priority queues (if you use A* search)
        return self.depth < other.depth


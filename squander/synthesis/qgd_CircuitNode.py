from typing import List, Tuple, Optional
import math

class CircuitNode:
    def __init__(self, num_qubits: int, 
                 topology: List[Tuple[int, int]], 
                 parent: Optional['CircuitNode'] = None, 
                 new_gate: Optional[Tuple[int, int]] = None):
        
        self.num_qubits = num_qubits
        self.topology = topology
        
        # 1. Store Raw History (Good for debugging/backtracking)
        if parent:
            self.raw_gates = parent.raw_gates + [new_gate]
            # Optimization: Start with parent's sorted list
            initial_list = list(parent.canonical_gates)
            initial_list.append(new_gate)
            self.canonical_gates = self._insert_and_canonicalize(initial_list)
        else:
            self.raw_gates = []
            self.canonical_gates = tuple()

        # 2. Pre-calculate Hash
        # We cache this because hashing a tuple is O(N), and we don't want to 
        # recompute it every time the Set checks it.
        self._hash_cache = hash(self.canonical_gates)

    def _gates_commute(self, g1: Tuple[int, int], g2: Tuple[int, int]) -> bool:
        """Returns True if gates operate on disjoint qubits."""
        return set(g1).isdisjoint(set(g2))

    def _insert_and_canonicalize(self, gate_list: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        """
        Takes a list that is SORTED except for the very last element.
        Bubbles the last element backwards until it hits a non-commuting gate.
        """
        sequence = list(gate_list)
        idx = len(sequence) - 1
        
        # Bubble the new gate 'left' as long as allowed
        while idx > 0:
            current = sequence[idx]
            neighbor = sequence[idx-1]
            
            # Can we swap? 
            # 1. Must commute
            # 2. Neighbor must be 'larger' (lexicographically)
            if self._gates_commute(current, neighbor) and neighbor > current:
                # SWAP
                sequence[idx], sequence[idx-1] = sequence[idx-1], sequence[idx]
                idx -= 1
            else:
                # If we can't swap, we found the final position. Stop.
                break
                
        return tuple(sequence)
    
    def __hash__(self):
        """
        Returns the pre-calculated hash of the CANONICAL form.
        This ensures Node([A, B]) and Node([B, A]) hash to the same bucket.
        """
        return self._hash_cache

    def __eq__(self, other):
        """
        Checks if two nodes represent the same physical circuit.
        """
        if not isinstance(other, CircuitNode):
            return False
        # Fast integer comparison first (hash), then full tuple comparison
        if self._hash_cache != other._hash_cache:
            return False
        return self.canonical_gates == other.canonical_gates

    def __repr__(self):
        return f"Circuit(Gates={len(self.raw_gates)})"
    
    def expand(self) -> List['CircuitNode']:
        """
        Generates all valid children (next layer of the tree).
        Applies basic pruning rules to avoid redundant circuits.
        """
        children = []
        qbit_subspaces = []
        for qbit_subspace in range(2,self.num_qubits):
            tlb = self.tlb(qbit_subspace)
            unique_qbits = frozenset(x for tup in self.gates[-tlb+1:] for x in tup)
            if len(unique_qbits)<= qbit_subspace:
                qbit_subspaces.append((qbit_subspace,unique_qbits))


        for connection in self.topology:
            # connection is (q1, q2)
            skip = False
            for qbit_subspace,unique_qbits in qbit_subspaces:
                if (len(unique_qbits) + (connection[0] not in unique_qbits) + (connection[1] not in unique_qbits))<= qbit_subspace:
                    skip = True
            if skip:
                continue
            # Create the child node
            child = CircuitNode(
                num_qubits=self.num_qubits,
                topology=self.topology,
                parent=self,
                new_gate=connection
            )
            children.append(child)
            
        return children

    def get_depth(self):
        return len(self.gates)

    def __repr__(self):
        return f"CircuitNode(depth={self.get_depth()}, gates={self.gates})"

    # Making the node hashable allows you to put it in a 'visited' set
    def __hash__(self):
        # We hash the sequence of gates. 
        # We assume topology/qubit count doesn't change during search.
        return hash(tuple(self.gates))

    def __eq__(self, other):
        return self.gates == other.gates
    
    def tlb(self,qbit_space):
        return math.floor((4**qbit_space-3*qbit_space-1)/4)
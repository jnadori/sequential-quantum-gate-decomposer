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
from typing import List, Tuple, Optional
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


def build_topology(qbit_num: int, topology: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    """
    Build topology as list of (target, control) qubit pairs.

    Args:
        qbit_num: Number of qubits
        topology: Optional list of qubit pairs. If None, all-to-all connectivity is assumed.

    Returns:
        List of (target_qbit, control_qbit) tuples representing allowed CNOT connections
    """
    if topology is None:
        # All-to-all connectivity
        topology_list = []
        for target in range(qbit_num):
            for control in range(qbit_num):
                if target != control:
                    topology_list.append((target, control))
        return topology_list
    else:
        # Use provided topology
        return list(topology)


def gray_code_to_qubit_pairs(
    gray_code: np.ndarray,
    topology: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Convert Gray code to list of (target, control) qubit pairs.

    Each element of the Gray code is an index into the topology list,
    selecting which qubit pair to use at that layer.

    Args:
        gray_code: Array of indices into topology
        topology: List of (target, control) qubit pairs

    Returns:
        List of (target, control) tuples for each layer
    """
    qubit_pairs = []
    for gcode_val in gray_code:
        if gcode_val >= len(topology):
            raise ValueError(f"Gray code value {gcode_val} exceeds topology size {len(topology)}")
        qubit_pairs.append(topology[gcode_val])

    return qubit_pairs


def construct_circuit_from_gray_code(
    gray_code: np.ndarray,
    qbit_num: int,
    topology: List[Tuple[int, int]],
    use_u3: bool = True
) -> Circuit:
    """
    Construct a quantum circuit from a Gray code.

    The circuit structure is:
    - For each Gray code element:
        - U3 gate on target qubit
        - U3 gate on control qubit
        - CNOT(target, control)
    - Final layer of U3 gates on all qubits

    Args:
        gray_code: Array of topology indices
        qbit_num: Number of qubits
        topology: List of allowed (target, control) qubit pairs
        use_u3: If True, use U3 gates; otherwise use RZ-RY-RZ decomposition

    Returns:
        Circuit object representing the gate structure

    Reference: N_Qubit_Decomposition_Tree_Search.cpp:635-672
    """
    circuit = Circuit(qbit_num)

    # Convert Gray code to qubit pairs
    qubit_pairs = gray_code_to_qubit_pairs(gray_code, topology)

    # Add two-qubit blocks for each layer
    for target_qbit, control_qbit in qubit_pairs:
        if use_u3:
            circuit.add_U3(target_qbit)
            circuit.add_U3(control_qbit)
        else:
            # RZ-RY-RZ decomposition
            circuit.add_RZ(target_qbit)
            circuit.add_RY(target_qbit)
            circuit.add_RZ(target_qbit)
            circuit.add_RZ(control_qbit)
            circuit.add_RY(control_qbit)
            circuit.add_RZ(control_qbit)

        circuit.add_CNOT(target_qbit, control_qbit)

    # Add finalizing layer (U3 on all qubits)
    for qbit in range(qbit_num):
        if use_u3:
            circuit.add_U3(qbit)
        else:
            circuit.add_RZ(qbit)
            circuit.add_RY(qbit)
            circuit.add_RZ(qbit)

    return circuit


def construct_circuit_from_qubit_pairs(
    qubit_pairs: List[Tuple[int, int]],
    qbit_num: int,
    use_u3: bool = True
) -> Circuit:
    """
    Construct a quantum circuit directly from qubit pairs.

    Args:
        qubit_pairs: List of (target, control) qubit pairs for each layer
        qbit_num: Number of qubits
        use_u3: If True, use U3 gates; otherwise use RZ-RY-RZ decomposition

    Returns:
        Circuit object representing the gate structure
    """
    circuit = Circuit(qbit_num)

    # Add two-qubit blocks for each layer
    for target_qbit, control_qbit in qubit_pairs:
        if target_qbit >= qbit_num or control_qbit >= qbit_num:
            raise ValueError(f"Qubit indices ({target_qbit}, {control_qbit}) exceed qbit_num={qbit_num}")
        if target_qbit == control_qbit:
            raise ValueError(f"Target and control qubits must be different: ({target_qbit}, {control_qbit})")

        if use_u3:
            circuit.add_U3(target_qbit)
            circuit.add_U3(control_qbit)
        else:
            circuit.add_RZ(target_qbit)
            circuit.add_RY(target_qbit)
            circuit.add_RZ(target_qbit)
            circuit.add_RZ(control_qbit)
            circuit.add_RY(control_qbit)
            circuit.add_RZ(control_qbit)

        circuit.add_CNOT(target_qbit, control_qbit)

    # Add finalizing layer
    for qbit in range(qbit_num):
        if use_u3:
            circuit.add_U3(qbit)
        else:
            circuit.add_RZ(qbit)
            circuit.add_RY(qbit)
            circuit.add_RZ(qbit)

    return circuit


def count_cnot_gates(gray_code: np.ndarray) -> int:
    """
    Count the number of CNOT gates in a circuit defined by a Gray code.

    Args:
        gray_code: Array of topology indices

    Returns:
        Number of CNOT gates (equal to length of Gray code)
    """
    return len(gray_code)


def visualize_circuit_structure(
    gray_code: np.ndarray,
    topology: List[Tuple[int, int]],
    qbit_num: int,
    output: str = 'text'
) -> str:
    """
    Visualize the circuit structure using Qiskit's drawing capabilities.

    Args:
        gray_code: Array of topology indices
        topology: List of allowed qubit pairs
        qbit_num: Number of qubits
        output: Output format ('text', 'mpl', 'latex'). Default is 'text'.

    Returns:
        String representation of the circuit (for 'text' and 'latex'),
        or displays matplotlib figure (for 'mpl')
    """
    from squander import Qiskit_IO

    # Build circuit from Gray code
    circuit = construct_circuit_from_gray_code(gray_code, qbit_num, topology, use_u3=True)

    # Initialize parameters with zeros (just for visualization)
    param_num = circuit.get_Parameter_Num()
    parameters = np.zeros(param_num, dtype=np.float64)

    # Convert to Qiskit circuit
    qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)

    # Draw the circuit
    if output == 'text':
        return str(qiskit_circuit.draw(output='text'))
    elif output == 'mpl':
        # Returns matplotlib figure
        return qiskit_circuit.draw(output='mpl')
    elif output == 'latex':
        return qiskit_circuit.draw(output='latex_source')
    else:
        raise ValueError(f"Unknown output format: {output}. Use 'text', 'mpl', or 'latex'.")

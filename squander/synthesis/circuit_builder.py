# -*- coding: utf-8 -*-
"""
Circuit Builder for Tree Search

Builds gate structures from Gray codes for tree search decomposition.
"""

from typing import List, Tuple, Optional
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit


def construct_gate_structure_from_gray_code(
    gray_code: List[int],
    topology: List[Tuple[int, int]],
    qbit_num: int,
    custom_two_qubit_block: Optional[Circuit] = None
) -> Circuit:
    """
    Construct gate structure from Gray code.
    
    This replicates the C++ construct_gate_structure_from_Gray_code function.
    
    Args:
        gray_code: List of topology edge indices (length = number of CNOT layers)
        topology: List of (target_qubit, control_qubit) pairs
        qbit_num: Number of qubits
        custom_two_qubit_block: Optional custom two-qubit block template
    
    Returns:
        Circuit structure with gates
    """
    gate_structure = Circuit(qbit_num)
    
    # Add two-qubit blocks for each Gray code element
    for gcode_idx in range(len(gray_code)):
        edge_idx = gray_code[gcode_idx]
        
        if edge_idx < 0 or edge_idx >= len(topology):
            continue  # Skip invalid indices
        
        target_qbit = topology[edge_idx][0]
        control_qbit = topology[edge_idx][1]
        
        # Add two-qubit block
        add_two_qubit_block(
            gate_structure,
            target_qbit,
            control_qbit,
            qbit_num,
            custom_two_qubit_block
        )
    
    # Add finalizing layer (U3 on all qubits)
    add_finalizing_layer(gate_structure, qbit_num)
    
    return gate_structure


def add_two_qubit_block(
    gate_structure: Circuit,
    target_qbit: int,
    control_qbit: int,
    qbit_num: int,
    custom_block: Optional[Circuit] = None
):
    """
    Add two-qubit building block to circuit.
    
    Replicates C++ add_two_qubit_block functionality.
    Creates a layer with: U3(target), U3(control), CNOT(target, control)
    
    Args:
        gate_structure: Circuit to add block to
        target_qbit: Target qubit index
        control_qbit: Control qubit index
        qbit_num: Total number of qubits
        custom_block: Optional custom two-qubit block template
    """
    # Validate qubit indices
    if control_qbit >= qbit_num or target_qbit >= qbit_num:
        raise ValueError(
            f"Qubit indices out of range: target={target_qbit}, "
            f"control={control_qbit}, qbit_num={qbit_num}"
        )
    
    if control_qbit == target_qbit:
        raise ValueError(
            f"Target and control qubits must be different: {target_qbit}"
        )
    
    if custom_block is not None:
        # Clone custom block and remap qubits
        layer = custom_block.copy()
        
        # Create qubit mapping: map template qubits to actual qubits
        # The C++ code:
        #   1. Sets qbit_num on the layer
        #   2. Creates qbit_order vector with -2 (unmapped) for all positions
        #   3. Sets qbit_order[target_qbit_idx] = 0, qbit_order[control_qbit_idx] = 1
        #   4. Calls reorder_qubits which maps qubit at position i to position qbit_order[i]
        #
        # For Remap_Qbits: {old_qubit_index: new_qubit_index}
        # Template has 2 qubits (0 and 1), map them to actual qubits
        # C++ uses reverse indexing: target_qbit_idx = qbit_num - 1 - target_qbit
        # But for Remap_Qbits, we use direct mapping to match the target/control qubit positions
        
        # Create mapping: template qubit 0 -> target_qbit, template qubit 1 -> control_qbit
        qbit_map = {0: target_qbit, 1: control_qbit}
        
        # Remap the circuit to full qubit register
        remapped_layer = layer.Remap_Qbits(qbit_map, qbit_num)
        
        gate_structure.add_Circuit(remapped_layer)
    else:
        # Standard block: Create a layer with U3(target), U3(control), CNOT(target, control)
        layer = Circuit(qbit_num)
        layer.add_U3(target_qbit)
        layer.add_U3(control_qbit)
        layer.add_CNOT(target_qbit, control_qbit)
        gate_structure.add_Circuit(layer)


def add_finalizing_layer(gate_structure: Circuit, qbit_num: int):
    """
    Add finalizing layer (single qubit rotations on all qubits).
    
    Replicates C++ add_finalyzing_layer functionality.
    Creates a layer with U3 gate on each qubit.
    
    Args:
        gate_structure: Circuit to add layer to
        qbit_num: Number of qubits
    """
    if gate_structure is None:
        raise ValueError("gate_structure cannot be None")
    
    # Create a layer block
    block = Circuit(qbit_num)
    for idx in range(qbit_num):
        block.add_U3(idx)
    
    gate_structure.add_Circuit(block)

# -*- coding: utf-8 -*-
"""
Example demonstrating A* search for quantum circuit synthesis.

Shows how to use A* search with different configurations and compares
it to exhaustive search.
"""

import numpy as np
from squander import TreeSearch, ApproximateSearch, Circuit


def generate_random_circuit(N: int, d: int, seed: int = None) -> Circuit:
    """
    Generate a random quantum circuit with N qubits and d 2-qubit blocks.

    Each 2-qubit block consists of U3 gates on both qubits followed by a CNOT.
    A final layer of U3 gates is added on all qubits.

    Args:
        N: Number of qubits
        d: Number of 2-qubit blocks (depth)
        seed: Random seed for reproducibility

    Returns:
        Circuit: A random quantum circuit
    """
    if seed is not None:
        np.random.seed(seed)

    # All-to-all connectivity topology
    topology = []
    for control_qbit in range(N - 1):
        for target_qbit in range(control_qbit + 1, N):
            topology.append((target_qbit, control_qbit))

    # 2-qubit basis: U3-U3-CNOT
    two_qbit_basis = Circuit(2)
    two_qbit_basis.add_U3(0)
    two_qbit_basis.add_U3(1)
    two_qbit_basis.add_CNOT(0, 1)

    # Build the circuit
    circuit = Circuit(N)

    # Add d random 2-qubit blocks
    for _ in range(d):
        # Randomly select a qubit pair from the topology
        target_qbit, control_qbit = topology[np.random.randint(len(topology))]

        # Copy and remap the 2-qubit basis to the selected qubits
        two_qbit_basis = Circuit(N)
        two_qbit_basis.add_U3(target_qbit)
        two_qbit_basis.add_U3(control_qbit)
        two_qbit_basis.add_CNOT(target_qbit, control_qbit)
        circuit.add_Circuit(two_qbit_basis)

    # Add final layer of U3 gates on all qubits
    final_layer = Circuit(N)
    for qbit in range(N):
        final_layer.add_U3(qbit)
    circuit.add_Circuit(final_layer)

    return circuit


def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate N-qubit QFT matrix."""
    dim = 2**N
    Fn = np.zeros([dim, dim], dtype=np.complex128)
    for idx in range(dim):
        for jdx in range(dim):
            Fn[idx][jdx] = 1/np.sqrt(dim) * (np.exp(2j*np.pi/dim)**(idx*jdx))
    return Fn


def example_basic():
    """Basic A* search example."""
    print("="*70)
    print("Example 1: Basic A* Search")
    print("="*70)

    # Generate a random circuit and get its unitary
    N = 3
    d = 4  # Number of 2-qubit blocks

    random_circuit = generate_random_circuit(N, d)
    random_params = np.random.rand(random_circuit.get_Parameter_Num()) * 2 * np.pi
    Umtx = random_circuit.get_Matrix(random_params)

    print(f"Generated random circuit with {N} qubits and {d} 2-qubit blocks")
    print(f"Circuit has {random_circuit.get_Parameter_Num()} parameters")

    # A* configuration
    config = {
        'tree_level_max': 4,
        'optimization_tolerance': 1e-6,
        'optimizer': 'BFGS',
        'parallel_search':6
    }

    # Run decomposition with A* search
    decomposer = TreeSearch(
        Umtx.conj().T,
        topology=None,
        config=config    )

    best_circuit, best_parameters, best_score = decomposer.start_decomposition()

    # Print results
    print(f"\nResult:")
    print(f"  Reported score: {best_score}")
    print(f"  Gate counts: {best_circuit.get_Gate_Nums()}")

    # Verify the decomposition by computing the actual error
    decomposed_Umtx = best_circuit.get_Matrix(best_parameters)

    # Compute trace fidelity: |Tr(U† V)|² / d²
    d_dim = Umtx.shape[0]
    trace_fidelity = np.abs(np.trace(Umtx.conj().T @ decomposed_Umtx))**2 / d_dim**2
    hilbert_schmidt_cost = 1 - trace_fidelity

    print(f"\nVerification (independent calculation):")
    print(f"  Hilbert-Schmidt cost: {hilbert_schmidt_cost:.6f}")
    print(f"  Trace fidelity: {trace_fidelity:.6f}")

    if hilbert_schmidt_cost < 1e-4:
        print("  SUCCESS: Decomposition is accurate!")
    else:
        print("  FAILURE: Decomposition does NOT match target unitary!")

def example_approximate():
    """Approximate search example with adaptive tolerance."""
    print("="*70)
    print("Example 2: Approximate Search")
    print("="*70)

    # Generate a random circuit and get its unitary
    N = 3
    d = 4  # Number of 2-qubit blocks

    random_circuit = generate_random_circuit(N, d, seed=42)
    random_params = np.random.rand(random_circuit.get_Parameter_Num()) * 2 * np.pi
    Umtx = random_circuit.get_Matrix(random_params)

    print(f"Generated random circuit with {N} qubits and {d} 2-qubit blocks")
    print(f"Circuit has {random_circuit.get_Parameter_Num()} parameters")

    config = {
        'tree_level_max': 4,
        'optimizer': 'BFGS2',
        'parallel_search': 6
    }

    # Run decomposition with approximate search
    # layer_fidelity: expected fidelity per CNOT layer (e.g. 0.995 for noisy hardware)
    # minimal_error: scaling factor for the adaptive tolerance
    decomposer = ApproximateSearch(
        Umtx.conj().T,
        topology=None,
        config=config,
        layer_fidelity=0.995,
        minimal_error=1e-1
    )

    best_circuit, best_parameters, best_score = decomposer.start_decomposition()

    # Print results
    print(f"\nResult:")
    print(f"  Reported score: {best_score}")
    print(f"  Gate counts: {best_circuit.get_Gate_Nums()}")

    # Verify the decomposition by computing the actual error
    decomposed_Umtx = best_circuit.get_Matrix(best_parameters)

    # Compute trace fidelity: |Tr(U† V)|² / d²
    d_dim = Umtx.shape[0]
    trace_fidelity = np.abs(np.trace(Umtx.conj().T @ decomposed_Umtx))**2 / d_dim**2
    hilbert_schmidt_cost = 1 - trace_fidelity

    print(f"\nVerification (independent calculation):")
    print(f"  Hilbert-Schmidt cost: {hilbert_schmidt_cost:.6f}")
    print(f"  Trace fidelity: {trace_fidelity:.6f}")

    # For approximate search, the tolerance is adaptive:
    # at the level found, tolerance = max(level, 1) * (1 - 0.995) * 0.1
    best_level = len(best_circuit.get_Gate_Nums()) if best_circuit else 0
    print(f"\n  Adaptive tolerance at acceptance: {decomposer.calculate_tolerance(best_level):.6e}")

    if hilbert_schmidt_cost < 1e-2:
        print("  SUCCESS: Approximate decomposition is within acceptable error!")
    else:
        print("  NOTE: Decomposition has larger error (expected for approximate search)")


def main():
    """Run all examples."""
    example_basic()
    example_approximate()

    print("="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

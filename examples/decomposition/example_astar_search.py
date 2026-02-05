# -*- coding: utf-8 -*-
"""
Example demonstrating A* search for quantum circuit synthesis.

Shows how to use A* search with different configurations and compares
it to exhaustive search.
"""

import numpy as np
from squander.synthesis.tree_search_old.tree_search import (
    TreeSearchDecomposition,
    AStarSearchDecomposition,
    ApproximateTreeSearchDecomposition
)


def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate N-qubit QFT matrix."""
    dim = 2**N
    Fn = np.zeros([dim, dim], dtype=np.complex128)
    for idx in range(dim):
        for jdx in range(dim):
            Fn[idx][jdx] = 1/np.sqrt(dim) * (np.exp(2j*np.pi/dim)**(idx*jdx))
    return Fn


def example_basic_astar():
    """Basic A* search example."""
    print("="*70)
    print("Example 1: Basic A* Search")
    print("="*70)

    # Generate 3-qubit QFT
    N = 3
    Umtx = generate_qft_matrix(N)

    # A* configuration
    config = {
        'tree_level_max': 5,
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'optimizer': 'BFGS',
        'astar_cost_weight': 100.0,  # Heuristic weight for A* search
        'astar_max_expansions': 5000,  # Maximum nodes to explore
    }

    # Run decomposition with A* search
    decomposer = AStarSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config,
        verbose=0
    )

    result = decomposer.start_decomposition()

    # Print results
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Cost: {result['cost']:.6e}")
    print(f"  Level: {result['level']}")
    print(f"  Nodes evaluated: {result['nodes_evaluated']}")
    print(f"  Time: {result['total_time']:.2f}s")

    # Get Qiskit circuit
    qc = decomposer.get_qiskit_circuit()
    print(f"  CNOT count: {qc.count_ops().get('cx', 0)}")
    print()


def example_exhaustive_vs_astar():
    """Compare exhaustive and A* search."""
    print("="*70)
    print("Example 2: Exhaustive vs A* Comparison")
    print("="*70)

    # Generate 2-qubit unitary
    N = 2
    Umtx = generate_qft_matrix(N)

    # Base config
    base_config = {
        'tree_level_max': 3,
        'optimization_tolerance': 1e-6,
        'max_iterations': 200,
        'optimizer': 'BFGS',
    }

    # Exhaustive search
    print("\n1. Exhaustive Search:")
    config_exhaustive = base_config.copy()

    decomposer_ex = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config_exhaustive,
        verbose=0
    )
    result_ex = decomposer_ex.start_decomposition()

    print(f"   Cost: {result_ex['cost']:.6e}")
    print(f"   Time: {result_ex['total_time']:.2f}s")
    print(f"   Nodes evaluated: {result_ex['nodes_evaluated']}")

    # A* search
    print("\n2. A* Search:")
    config_astar = base_config.copy()
    config_astar['astar_cost_weight'] = 100.0

    decomposer_as = AStarSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config_astar,
        verbose=0
    )
    result_as = decomposer_as.start_decomposition()

    print(f"   Cost: {result_as['cost']:.6e}")
    print(f"   Time: {result_as['total_time']:.2f}s")
    print(f"   Nodes evaluated: {result_as['nodes_evaluated']}")

    # Speedup
    speedup = result_ex['total_time'] / result_as['total_time']
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"Nodes reduction: {result_ex['nodes_evaluated'] / result_as['nodes_evaluated']:.1f}x")
    print()


def example_astar_limited_budget():
    """A* search with limited node expansion budget."""
    print("="*70)
    print("Example 3: A* Search with Limited Budget")
    print("="*70)

    # Generate 3-qubit QFT
    N = 3
    Umtx = generate_qft_matrix(N)

    # A* with limited expansion budget
    config = {
        'tree_level_max': 6,
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'optimizer': 'BFGS',
        'astar_cost_weight': 100.0,
        'astar_max_expansions': 100,  # Limit search to 100 node evaluations
    }

    decomposer = AStarSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config,
        verbose=0
    )

    result = decomposer.start_decomposition()

    print(f"\nResult:")
    print(f"  Cost: {result['cost']:.6e}")
    print(f"  Level: {result['level']}")
    print(f"  Nodes evaluated: {result['nodes_evaluated']}")
    print(f"  Time: {result['total_time']:.2f}s")

    # Get Qiskit circuit
    qc = decomposer.get_qiskit_circuit()
    print(f"  CNOT count: {qc.count_ops().get('cx', 0)}")
    print()


def example_approximate_search():
    """Approximate search with adaptive tolerance."""
    print("="*70)
    print("Example 4: Approximate Search with Adaptive Tolerance")
    print("="*70)

    # Generate 3-qubit QFT
    N = 3
    Umtx = generate_qft_matrix(N)

    # Approximate search configuration
    config = {
        'tree_level_max': 4,  # Reduced from 6 - approximate search does exhaustive at each level
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'optimizer': 'BFGS',
    }

    decomposer = ApproximateTreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        layer_fidelity=0.99,   # Expected fidelity per layer (1% error per layer)
        minimal_error=1e-3,     # Minimum error threshold (safety margin)
        config=config,
        verbose=0
    )

    result = decomposer.start_decomposition()

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Cost: {result['cost']:.6e}")
    print(f"  Level: {result['level']}")
    print(f"  Nodes evaluated: {result['nodes_evaluated']}")
    print(f"  Time: {result['total_time']:.2f}s")

    # Get Qiskit circuit
    qc = decomposer.get_qiskit_circuit()
    print(f"  CNOT count: {qc.count_ops().get('cx', 0)}")
    print()


def main():
    """Run all examples."""
    example_basic_astar()
    example_exhaustive_vs_astar()
    example_astar_limited_budget()
    example_approximate_search()

    print("="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

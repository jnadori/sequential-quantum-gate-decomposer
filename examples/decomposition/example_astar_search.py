# -*- coding: utf-8 -*-
"""
Example demonstrating A* search for quantum circuit synthesis.

Shows how to use A* search with different configurations and compares
it to exhaustive search.
"""

import numpy as np
from squander.synthesis.tree_search import TreeSearchDecomposition


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
        'search_mode': 'astar',
        'tree_level_max': 5,
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'optimizer': 'scipy:L-BFGS-B',
        'heuristic_type': 'quick_opt',
        'heuristic_iterations': 10,
    }

    # Run decomposition
    decomposer = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config,
        verbose=1
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
        'optimizer': 'scipy:L-BFGS-B',
    }

    # Exhaustive search
    print("\n1. Exhaustive Search:")
    config_exhaustive = base_config.copy()
    config_exhaustive['search_mode'] = 'exhaustive'

    decomposer_ex = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config_exhaustive,
        verbose=0
    )
    result_ex = decomposer_ex.start_decomposition()

    print(f"   Cost: {result_ex['cost']:.6e}")
    print(f"   Time: {result_ex['total_time']:.2f}s")

    # A* search
    print("\n2. A* Search:")
    config_astar = base_config.copy()
    config_astar.update({
        'search_mode': 'astar',
        'heuristic_type': 'quick_opt',
        'heuristic_iterations': 10,
    })

    decomposer_as = TreeSearchDecomposition(
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
    print()


def example_beam_search():
    """A* with beam search."""
    print("="*70)
    print("Example 3: A* with Beam Search")
    print("="*70)

    # Generate 3-qubit QFT
    N = 3
    Umtx = generate_qft_matrix(N)

    # A* with beam search
    config = {
        'search_mode': 'astar',
        'tree_level_max': 6,
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'optimizer': 'scipy:L-BFGS-B',
        'heuristic_type': 'quick_opt',
        'heuristic_iterations': 10,
        'beam_width': 10,  # Keep only top 10 nodes
    }

    decomposer = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config,
        verbose=1
    )

    result = decomposer.start_decomposition()

    print(f"\nResult:")
    print(f"  Cost: {result['cost']:.6e}")
    print(f"  Level: {result['level']}")
    print(f"  Nodes evaluated: {result['nodes_evaluated']}")
    print(f"  Nodes pruned: {result['nodes_pruned']}")
    print(f"  Time: {result['total_time']:.2f}s")
    print()


def main():
    """Run all examples."""
    example_basic_astar()
    example_exhaustive_vs_astar()
    example_beam_search()

    print("="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

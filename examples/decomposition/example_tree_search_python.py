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
from squander.synthesis.tree_search_old.tree_search import (
    TreeSearchDecomposition,
    AStarSearchDecomposition
)
from squander import utils


def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate N-qubit QFT matrix."""
    dim = 2**N
    Fn = np.zeros([dim, dim], dtype=np.complex128)
    for idx in range(dim):
        for jdx in range(dim):
            Fn[idx][jdx] = 1/np.sqrt(dim) * (np.exp(2j*np.pi/dim)**(idx*jdx))
    return Fn


def example_basic_usage():
    """Basic example: 3-qubit QFT with tree search."""
    print("="*70)
    print("Example 1: Basic Tree Search for 3-qubit QFT")
    print("="*70)

    # Generate target unitary
    N = 3
    Umtx = generate_qft_matrix(N)

    # Configure tree search
    config = {
        'tree_level_max': 5,          # Search up to 5 CNOT layers
        'tree_level_min': 1,           # Start from 1 CNOT layer
        'optimization_tolerance': 1e-8,
        'optimizer': 'scipy:L-BFGS-B', # Use scipy L-BFGS-B
        'max_iterations': 1000,
        'cost_function_variant': 3,    # Hilbert-Schmidt test
    }

    # Create decomposer
    decomposer = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=None,  # All-to-all connectivity
        config=config,
        verbose=2
    )

    # Run tree search
    result = decomposer.start_decomposition()

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Success: {result['success']}")
    print(f"Final cost: {result['cost']:.6e}")
    print(f"Best level: {result['level']} CNOT layers")
    print(f"Total time: {result['total_time']:.1f}s")

    # Get Qiskit circuit
    qc = decomposer.get_qiskit_circuit()
    print(f"\nCircuit depth: {qc.depth()}")
    print(f"CNOT count: {qc.count_ops().get('cx', 0)}")

    # Verify decomposition
    decomposed_matrix = utils.get_unitary_from_qiskit_circuit(qc)
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    phase = np.angle(product_matrix[0, 0])
    product_matrix = product_matrix * np.exp(-1j*phase)
    product_matrix = np.eye(len(product_matrix))*2 - product_matrix - product_matrix.conj().T
    error = (np.real(np.trace(product_matrix)))/2
    print(f"Verification error: {error:.6e}")


def example_astar_usage():
    """Example: 3-qubit QFT solved via A* heuristic search."""
    print("\n\n" + "="*70)
    print("Example 2: A* Search with Heuristic Priority Queue")
    print("="*70)

    N = 3
    Umtx = generate_qft_matrix(N)

    config = {
        'tree_level_max': 6,
        'optimization_tolerance': 1e-6,
        'optimizer': 'BFGS',
        'astar_cost_weight': 100.0,
        'astar_max_expansions': 250,
    }

    decomposer = AStarSearchDecomposition(
        Umtx.conj().T,
        topology=None,
        config=config,
        verbose=1
    )

    result = decomposer.start_decomposition()

    print(f"\nFinal cost: {result['cost']:.6e}")
    print(f"Best level: {result['level']} CNOT layers")
    print(f"Optimizer iterations: {result['number_of_iters']}")


def example_limited_topology():
    """Example with limited topology (nearest-neighbor only)."""
    print("\n\n" + "="*70)
    print("Example 3: Tree Search with Limited Topology (Nearest-Neighbor)")
    print("="*70)

    # Generate target unitary
    N = 3
    Umtx = generate_qft_matrix(N)

    # Define nearest-neighbor topology
    # For 3 qubits: 0-1, 1-0, 1-2, 2-1
    topology = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
    ]

    print(f"\nTopology: {topology}")

    # Configure tree search
    config = {
        'tree_level_max': 8,  # May need more layers with limited topology
        'optimization_tolerance': 1e-6,
        'optimizer': 'scipy:L-BFGS-B',
        'max_iterations': 500,
    }

    # Create decomposer
    decomposer = TreeSearchDecomposition(
        Umtx.conj().T,
        topology=topology,
        config=config,
        verbose=1
    )

    # Run tree search
    result = decomposer.start_decomposition()

    print(f"\nFinal cost: {result['cost']:.6e}")
    print(f"Best level: {result['level']} CNOT layers")


def example_custom_optimizer():
    """Example using a custom optimizer (simulated annealing)."""
    print("\n\n" + "="*70)
    print("Example 4: Tree Search with Custom Optimizer (Simulated Annealing)")
    print("="*70)

    from scipy.optimize import dual_annealing

    def custom_sa_optimizer(cost_fn, grad_fn, x0, **kwargs):
        """Custom simulated annealing optimizer."""
        # Note: dual_annealing doesn't use gradients
        bounds = [(0, 2*np.pi)] * len(x0)

        result = dual_annealing(
            cost_fn,
            bounds,
            maxiter=kwargs.get('max_iterations', 100),
            seed=42
        )

        info = {
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'nit': result.nit
        }

        return result.x, result.fun, info

    # Generate target unitary
    N = 2
    Umtx = generate_qft_matrix(N)

    # Configure tree search with custom optimizer
    config = {
        'tree_level_max': 4,
        'optimization_tolerance': 1e-6,
        'optimizer': 'custom',
        'max_iterations': 100,
    }

    # Note: This would require modifying tree_search.py to accept custom_optimizer
    # For now, this is just a demonstration of the API design
    print("Custom optimizer example (API demonstration)")
    print("To use custom optimizers, pass custom_optimizer parameter to optimize_circuit()")


def example_builtin_optimizer():
    """Example using built-in SQUANDER optimizer."""
    print("\n\n" + "="*70)
    print("Example 5: Tree Search with Built-in SQUANDER Optimizer")
    print("="*70)

    # Generate target unitary
    N = 2
    Umtx = generate_qft_matrix(N)

    # Configure tree search with built-in optimizer
    config = {
        'tree_level_max': 3,
        'optimization_tolerance': 1e-8,
        'optimizer': 'BFGS',  # Built-in SQUANDER BFGS
        'max_iterations': 1000,
    }

    # Create decomposer
    decomposer = TreeSearchDecomposition(
        Umtx.conj().T,
        config=config,
        verbose=2
    )

    # Run tree search
    result = decomposer.start_decomposition()

    print(f"\nFinal cost: {result['cost']:.6e}")
    print(f"Best level: {result['level']} CNOT layers")


def main():
    """Run all examples."""
    print("Tree Search Decomposition Examples")
    print("="*70)
    print("\nThese examples demonstrate the Python implementation of")
    print("tree search based quantum circuit synthesis.\n")

    # Run examples
    example_basic_usage()
    example_astar_usage()
    example_limited_topology()
    example_custom_optimizer()
    example_builtin_optimizer()

    print("\n\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()

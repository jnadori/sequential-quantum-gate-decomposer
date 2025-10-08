# -*- coding: utf-8 -*-
"""
Benchmark script for tree search with different optimizers.

Tests various scipy and SQUANDER optimizers on quantum circuit synthesis,
reporting average CNOT count, standard deviation, and compilation time.
"""

import numpy as np
import time
from squander.synthesis.tree_search import TreeSearchDecomposition
from typing import List, Dict, Tuple
import sys


def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate N-qubit QFT matrix."""
    dim = 2**N
    Fn = np.zeros([dim, dim], dtype=np.complex128)
    for idx in range(dim):
        for jdx in range(dim):
            Fn[idx][jdx] = 1/np.sqrt(dim) * (np.exp(2j*np.pi/dim)**(idx*jdx))
    return Fn


def benchmark_optimizer(
    Umtx: np.ndarray,
    optimizer: str,
    num_trials: int = 10,
    config: Dict = None
) -> Tuple[List[int], List[float], List[bool]]:
    """
    Benchmark a single optimizer over multiple trials.

    Args:
        Umtx: Target unitary matrix
        optimizer: Optimizer name
        num_trials: Number of independent trials
        config: Configuration dictionary

    Returns:
        Tuple of (cnot_counts, compile_times, successes)
    """
    if config is None:
        config = {}

    config['optimizer'] = optimizer

    cnot_counts = []
    compile_times = []
    successes = []

    for trial in range(num_trials):
        try:
            # Create decomposer
            decomposer = TreeSearchDecomposition(
                Umtx.conj().T,
                topology=None,
                config=config,
                verbose=0  # Silent
            )

            # Run decomposition
            start_time = time.time()
            result = decomposer.start_decomposition()
            elapsed = time.time() - start_time

            # Get Qiskit circuit to count CNOTs
            qc = decomposer.get_qiskit_circuit()
            cnot_count = qc.count_ops().get('cx', 0)

            cnot_counts.append(cnot_count)
            compile_times.append(elapsed)
            successes.append(result['success'])

        except Exception as e:
            print(f"    Trial {trial+1} failed: {e}")
            continue

    return cnot_counts, compile_times, successes


def print_statistics(
    optimizer_name: str,
    cnot_counts: List[int],
    compile_times: List[float],
    successes: List[bool]
):
    """Print benchmark statistics for an optimizer."""
    if len(cnot_counts) == 0:
        print(f"{optimizer_name:25s} | FAILED - No successful trials")
        return

    avg_cnot = np.mean(cnot_counts)
    std_cnot = np.std(cnot_counts)
    avg_time = np.mean(compile_times)
    std_time = np.std(compile_times)
    success_rate = np.mean(successes) * 100

    print(f"{optimizer_name:25s} | "
          f"CNOT: {avg_cnot:5.1f} ± {std_cnot:4.1f} | "
          f"Time: {avg_time:6.2f}s ± {std_time:5.2f}s | "
          f"Success: {success_rate:5.1f}%")


def main():
    """Run benchmark comparing different optimizers."""

    # Configuration
    N = 3  # Number of qubits
    num_trials = 10  # Trials per optimizer

    print("="*80)
    print("Tree Search Optimizer Benchmark")
    print("="*80)
    print(f"Target: {N}-qubit QFT")
    print(f"Trials per optimizer: {num_trials}")
    print(f"Topology: All-to-all")
    print("="*80)
    print()

    # Generate target unitary
    Umtx = generate_qft_matrix(N)

    # Base configuration
    base_config = {
        'tree_level_max': 7,
        'tree_level_min': 1,
        'optimization_tolerance': 1e-8,
        'max_iterations': 1000,
        'cost_function_variant': 3,
        'parallel' : 0
    }

    # Optimizers to benchmark
    optimizers = [
        # Scipy optimizers
        'scipy:L-BFGS-B',
        'scipy:basinhopping',
        'scipy:CMA-ES',
        # SQUANDER built-in optimizers
        'BFGS',
    ]

    print("Optimizer                 | CNOT Count     | Compile Time   | Success Rate")
    print("-" * 80)

    results = {}

    for optimizer in optimizers:
        print(f"Running {optimizer}...", end=' ', flush=True)

        cnot_counts, compile_times, successes = benchmark_optimizer(
            Umtx,
            optimizer,
            num_trials=num_trials,
            config=base_config.copy()
        )

        results[optimizer] = {
            'cnot_counts': cnot_counts,
            'compile_times': compile_times,
            'successes': successes
        }

        # Clear the "Running..." message
        print("\r" + " " * 40 + "\r", end='')

        print_statistics(optimizer, cnot_counts, compile_times, successes)

    print("="*80)

    # Find best optimizer by average CNOT count
    best_optimizer = None
    best_cnot = float('inf')

    for optimizer, data in results.items():
        if len(data['cnot_counts']) > 0:
            avg_cnot = np.mean(data['cnot_counts'])
            if avg_cnot < best_cnot:
                best_cnot = avg_cnot
                best_optimizer = optimizer

    if best_optimizer:
        print(f"\nBest optimizer (by CNOT count): {best_optimizer}")
        print(f"Average CNOT count: {best_cnot:.1f}")

    # Find fastest optimizer
    fastest_optimizer = None
    fastest_time = float('inf')

    for optimizer, data in results.items():
        if len(data['compile_times']) > 0:
            avg_time = np.mean(data['compile_times'])
            if avg_time < fastest_time:
                fastest_time = avg_time
                fastest_optimizer = optimizer

    if fastest_optimizer:
        print(f"\nFastest optimizer: {fastest_optimizer}")
        print(f"Average compile time: {fastest_time:.2f}s")

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)


if __name__ == "__main__":
    main()
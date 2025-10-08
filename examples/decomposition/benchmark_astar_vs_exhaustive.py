# -*- coding: utf-8 -*-
"""
Benchmark script comparing A* search vs exhaustive tree search.

Compares performance metrics including:
- Number of circuit optimizations performed
- Total compilation time
- Final CNOT count
- Success rate
"""

import numpy as np
import time
from squander.synthesis.tree_search import TreeSearchDecomposition
from typing import Dict, List
import sys


def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate N-qubit QFT matrix."""
    dim = 2**N
    Fn = np.zeros([dim, dim], dtype=np.complex128)
    for idx in range(dim):
        for jdx in range(dim):
            Fn[idx][jdx] = 1/np.sqrt(dim) * (np.exp(2j*np.pi/dim)**(idx*jdx))
    return Fn


def generate_random_unitary(N: int) -> np.ndarray:
    """Generate random N-qubit unitary matrix."""
    from scipy.stats import unitary_group
    return unitary_group.rvs(2**N)


def benchmark_search_mode(
    Umtx: np.ndarray,
    search_mode: str,
    config: Dict,
    num_trials: int = 1
) -> Dict:
    """
    Benchmark a single search mode over multiple trials.

    Args:
        Umtx: Target unitary matrix
        search_mode: 'exhaustive' or 'astar'
        config: Configuration dictionary
        num_trials: Number of trials

    Returns:
        Dictionary with statistics
    """
    cnot_counts = []
    compile_times = []
    successes = []
    nodes_evaluated = []

    for trial in range(num_trials):
        try:
            # Create decomposer
            config['search_mode'] = search_mode
            decomposer = TreeSearchDecomposition(
                Umtx.conj().T,
                topology=None,
                config=config,
                verbose=0
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

            # Track nodes evaluated (for A*)
            if 'nodes_evaluated' in result:
                nodes_evaluated.append(result['nodes_evaluated'])

        except Exception as e:
            print(f"    Trial {trial+1} failed: {e}")
            continue

    return {
        'cnot_counts': cnot_counts,
        'compile_times': compile_times,
        'successes': successes,
        'nodes_evaluated': nodes_evaluated
    }


def print_comparison(exhaustive_results: Dict, astar_results: Dict, config: Dict):
    """Print comparison statistics."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # CNOT counts
    if exhaustive_results['cnot_counts'] and astar_results['cnot_counts']:
        ex_cnot = np.mean(exhaustive_results['cnot_counts'])
        as_cnot = np.mean(astar_results['cnot_counts'])
        print(f"\nCNOT Count:")
        print(f"  Exhaustive: {ex_cnot:.1f} ± {np.std(exhaustive_results['cnot_counts']):.1f}")
        print(f"  A*:         {as_cnot:.1f} ± {np.std(astar_results['cnot_counts']):.1f}")

    # Compile time
    if exhaustive_results['compile_times'] and astar_results['compile_times']:
        ex_time = np.mean(exhaustive_results['compile_times'])
        as_time = np.mean(astar_results['compile_times'])
        speedup = ex_time / as_time if as_time > 0 else 0
        print(f"\nCompile Time:")
        print(f"  Exhaustive: {ex_time:.2f}s ± {np.std(exhaustive_results['compile_times']):.2f}s")
        print(f"  A*:         {as_time:.2f}s ± {np.std(astar_results['compile_times']):.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")

    # Nodes evaluated (A* only)
    if astar_results['nodes_evaluated']:
        print(f"\nNodes Evaluated (A*):")
        print(f"  Average:    {np.mean(astar_results['nodes_evaluated']):.0f}")

        # Estimate exhaustive nodes (depends on tree_level_max)
        level_max = config.get('tree_level_max', 7)
        topology_size = 6  # 3-qubit all-to-all = 6 pairs
        # Total nodes for exhaustive = sum of N^L for L=1 to level_max
        exhaustive_nodes = sum(topology_size**L for L in range(1, level_max + 1))
        print(f"  Exhaustive (est): {exhaustive_nodes}")
        if astar_results['nodes_evaluated']:
            reduction = exhaustive_nodes / np.mean(astar_results['nodes_evaluated'])
            print(f"  Reduction:  {reduction:.1f}x fewer evaluations")

    # Success rate
    if exhaustive_results['successes'] and astar_results['successes']:
        ex_success = np.mean(exhaustive_results['successes']) * 100
        as_success = np.mean(astar_results['successes']) * 100
        print(f"\nSuccess Rate:")
        print(f"  Exhaustive: {ex_success:.1f}%")
        print(f"  A*:         {as_success:.1f}%")

    print("\n" + "="*80)


def main():
    """Run benchmark comparing exhaustive vs A* search."""

    # Configuration
    N = 3  # Number of qubits
    num_trials = 3  # Trials per mode

    print("="*80)
    print("A* vs Exhaustive Search Benchmark")
    print("="*80)
    print(f"Target: {N}-qubit QFT")
    print(f"Trials per mode: {num_trials}")
    print(f"Topology: All-to-all")
    print("="*80)

    # Generate target unitary
    Umtx = generate_qft_matrix(N)

    # Base configuration
    base_config = {
        'tree_level_max': 5,  # Lower for faster benchmarking
        'tree_level_min': 1,
        'optimization_tolerance': 1e-6,
        'max_iterations': 500,
        'cost_function_variant': 3,
        'optimizer': 'scipy:L-BFGS-B',
    }

    # A* specific config
    astar_config = base_config.copy()
    astar_config.update({
        'heuristic_type': 'quick_opt',  # or 'gradient_norm', 'random_sample'
        'heuristic_iterations': 10,
        'beam_width': 0,  # 0 = no beam search
    })

    # Run exhaustive search
    print("\nRunning Exhaustive Search...")
    exhaustive_results = benchmark_search_mode(
        Umtx,
        'exhaustive',
        base_config,
        num_trials=num_trials
    )
    print(f"  Completed {len(exhaustive_results['compile_times'])} trials")

    # Run A* search
    print("\nRunning A* Search...")
    astar_results = benchmark_search_mode(
        Umtx,
        'astar',
        astar_config,
        num_trials=num_trials
    )
    print(f"  Completed {len(astar_results['compile_times'])} trials")

    # Print comparison
    print_comparison(exhaustive_results, astar_results, astar_config)

    # Test different heuristics
    print("\n" + "="*80)
    print("HEURISTIC COMPARISON (A* only)")
    print("="*80)

    heuristics = ['quick_opt', 'gradient_norm', 'random_sample']
    heuristic_results = {}

    for heuristic in heuristics:
        print(f"\nTesting heuristic: {heuristic}")
        config = astar_config.copy()
        config['heuristic_type'] = heuristic

        results = benchmark_search_mode(
            Umtx,
            'astar',
            config,
            num_trials=2
        )
        heuristic_results[heuristic] = results

        if results['compile_times']:
            avg_time = np.mean(results['compile_times'])
            avg_nodes = np.mean(results['nodes_evaluated']) if results['nodes_evaluated'] else 0
            print(f"  Time: {avg_time:.2f}s, Nodes: {avg_nodes:.0f}")

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

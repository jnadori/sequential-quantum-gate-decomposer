# -*- coding: utf-8 -*-
"""
Comprehensive benchmark comparing A* search vs regular tree search decomposition.

This script:
- Generates random circuits for testing (for depth d: d random edges with U3+CNOT, then N U3s)
- Uses star topology (qubit 0 as center connected to all others)
- Compares TreeSearchDecomposition vs AStarSearchDecomposition
- Tests on qubit_num = 3, 4
- Tests circuit depth up to 15
- Runs on single core
- Visualizes results and saves raw data
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Import tree search classes
try:
    from squander.synthesis.tree_search_old.tree_search import TreeSearchDecomposition, AStarSearchDecomposition
except ImportError:
    try:
        # Fallback: try importing from squander.synthesis
        from squander.synthesis import TreeSearchDecomposition, AStarSearchDecomposition
    except ImportError:
        # If still failing, try adding current directory to path
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from squander.synthesis.tree_search_old.tree_search import TreeSearchDecomposition, AStarSearchDecomposition


def generate_star_topology(N: int) -> List[Tuple[int, int]]:
    """
    Generate star topology for N qubits.
    
    In a star topology, qubit 0 is the central qubit connected to all others.
    Returns bidirectional edges: (center, leaf) and (leaf, center) for each leaf.
    
    Args:
        N: Number of qubits
        
    Returns:
        List of (target, control) qubit pairs representing star topology
    """
    topology = []
    center = 0
    for leaf in range(1, N):
        # Add both directions: (center, leaf) and (leaf, center)
        topology.append((center, leaf))
        topology.append((leaf, center))
    return topology


def generate_random_circuit_unitary(
    N: int,
    depth: int,
    topology: List[Tuple[int, int]],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random N-qubit unitary matrix from a random circuit.
    
    For depth d:
    - Select d random edges from topology
    - For each edge: add U3(target), U3(control), CNOT(target, control)
    - Add N U3 gates at the end (one per qubit)
    
    Args:
        N: Number of qubits
        depth: Circuit depth (number of two-qubit blocks)
        topology: List of (target, control) qubit pairs
        seed: Random seed for reproducibility
        
    Returns:
        Unitary matrix (2^N x 2^N) from the random circuit
    """
    if seed is not None:
        np.random.seed(seed)
    
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
    
    # Create circuit
    circuit = Circuit(N)
    
    # Select d random edges
    if len(topology) == 0:
        raise ValueError("Topology must contain at least one edge")
    
    # Sample d edges with replacement (allowing same edge multiple times)
    selected_edges = np.random.choice(len(topology), size=depth, replace=True)
    
    # Add two-qubit blocks for each selected edge
    for edge_idx in selected_edges:
        target_qbit, control_qbit = topology[edge_idx]
        
        # Create a layer with U3(target), U3(control), CNOT(target, control)
        layer = Circuit(N)
        layer.add_U3(target_qbit)
        layer.add_U3(control_qbit)
        layer.add_CNOT(target_qbit, control_qbit)
        circuit.add_Circuit(layer)
    
    # Add finalizing layer: N U3 gates (one per qubit)
    final_layer = Circuit(N)
    for idx in range(N):
        final_layer.add_U3(idx)
    circuit.add_Circuit(final_layer)
    
    # Generate random parameters for the circuit
    num_params = circuit.get_Parameter_Num()
    parameters = np.random.uniform(0, 2 * np.pi, size=num_params)
    
    # Get the unitary matrix from the circuit
    unitary = circuit.get_Matrix(parameters)
    
    return unitary


def run_single_trial(
    trial: int,
    qbit_num: int,
    level: int,
    topology: List[Tuple[int, int]],
    config: Dict,
    astar_config: Dict,
    benchmark_tree_search: bool = True,
    benchmark_astar: bool = True
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Run a single trial: generate circuit and run enabled methods sequentially.
    
    Args:
        trial: Trial number
        qbit_num: Number of qubits
        level: Circuit depth level
        topology: List of (target, control) qubit pairs
        config: Configuration dictionary for tree_search
        astar_config: Configuration dictionary for astar
        benchmark_tree_search: If False, skip tree_search
        benchmark_astar: If False, skip astar
        
    Returns:
        Tuple of (tree_result, astar_result) - either can be None if disabled
    """
    seed = trial + 1000 * qbit_num + 100 * level
    # Generate random circuit unitary with depth = level
    Umtx = generate_random_circuit_unitary(
        qbit_num, level, topology, seed=seed
    )
    
    # Run enabled methods sequentially on the same circuit
    tree_result = None
    astar_result = None
    
    if benchmark_tree_search:
        tree_result = run_single_benchmark(
            Umtx, 'tree_search', qbit_num, level, config, topology=topology, seed=seed
        )
    
    if benchmark_astar:
        astar_result = run_single_benchmark(
            Umtx, 'astar', qbit_num, level, astar_config, topology=topology, seed=seed
        )
    
    return tree_result, astar_result


def run_single_benchmark(
    Umtx: np.ndarray,
    method: str,
    qbit_num: int,
    tree_level_max: int,
    config: Dict,
    topology: Optional[List[Tuple[int, int]]] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run a single benchmark trial.
    
    Args:
        Umtx: Target unitary matrix
        method: 'tree_search' or 'astar'
        qbit_num: Number of qubits
        tree_level_max: Maximum tree level to explore
        config: Configuration dictionary
        topology: List of (target, control) qubit pairs for connectivity constraints
        seed: Random seed
        
    Returns:
        Dictionary with benchmark results
    """
    result = {
        'method': method,
        'qbit_num': qbit_num,
        'tree_level_max': tree_level_max,
        'seed': seed,
        'success': False,
        'cost': float('inf'),
        'level': -1,
        'total_time': -1.0,
        'number_of_iters': 0,
        'nodes_evaluated': 0,
        'error': None
    }
    
    try:
        # Create decomposer
        if method == 'tree_search':
            decomposer = TreeSearchDecomposition(
                Umtx.copy(),
                topology=topology,
                config=config,
                verbose=0
            )
        elif method == 'astar':
            decomposer = AStarSearchDecomposition(
                Umtx.copy(),
                topology=topology,
                config=config,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run decomposition
        start_time = time.time()
        decomp_result = decomposer.start_decomposition()
        elapsed = time.time() - start_time
        
        # Extract results
        result['success'] = decomp_result['success']
        result['cost'] = decomp_result['cost']
        result['level'] = decomp_result['level']
        result['total_time'] = elapsed
        result['number_of_iters'] = decomp_result.get('number_of_iters', 0)
        
        # Get nodes evaluated
        if 'nodes_evaluated' in decomp_result:
            result['nodes_evaluated'] = decomp_result['nodes_evaluated']
        elif method == 'tree_search':
            # Estimate nodes evaluated for tree search (sum of all combinations up to level)
            # This is an approximation - actual nodes depend on early stopping
            topology_size = len(decomposer.topology)
            if decomp_result['level'] > 0:
                nodes_est = sum(topology_size**L for L in range(1, decomp_result['level'] + 1))
            else:
                nodes_est = 0
            result['nodes_evaluated'] = nodes_est
        else:
            result['nodes_evaluated'] = 0
        
        # Try to get Qiskit circuit for CNOT count
        try:
            qc = decomposer.get_qiskit_circuit()
            cnot_count = qc.count_ops().get('cx', 0)
            result['cnot_count'] = cnot_count
        except Exception as e:
            result['cnot_count'] = -1
            result['error'] = str(e)
            
    except Exception as e:
        result['error'] = str(e)
        # Suppress error output during benchmark - errors are tracked in results
        # print(f"  Error in {method} (q={qbit_num}, level={tree_level_max}, seed={seed}): {e}")
    
    return result


def run_benchmark_suite(
    qbit_nums: List[int],
    tree_level_max: int,
    num_trials: int = 5,
    base_config: Optional[Dict] = None,
    benchmark_tree_search: bool = True,
    benchmark_astar: bool = True
) -> List[Dict]:
    """
    Run full benchmark suite.
    
    Args:
        qbit_nums: List of qubit numbers to test
        tree_level_max: Maximum tree level
        num_trials: Number of random unitaries per configuration
        base_config: Base configuration dictionary
        benchmark_tree_search: If False, skip tree_search benchmarking
        benchmark_astar: If False, skip astar benchmarking
        
    Returns:
        List of all benchmark results
    """
    if base_config is None:
        base_config = {}
    
    all_results = []
    
    # Suppress intermediate output - only show final results
    # print("="*80)
    # print("A* vs Tree Search Benchmark Suite")
    # print("="*80)
    # print(f"Qubit numbers: {qbit_nums}")
    # print(f"Max tree level: {tree_level_max}")
    # print(f"Trials per configuration: {num_trials}")
    # print(f"Single core: True")
    # print(f"Test circuits: Random circuits (depth d = d edges + N U3s)")
    # print("="*80)
    # print()
    
    for qbit_num in qbit_nums:
        # print(f"\n{'='*80}")
        # print(f"Testing {qbit_num}-qubit circuits")
        # print(f"{'='*80}")
        
        # Create progress bar for this qubit number
        levels = range(1, min(tree_level_max + 1, 16))  # Cap at 15
        pbar = tqdm(levels, desc=f"{qbit_num}-qubit circuits", unit="level")
        
        for level in pbar:
            # print(f"\n  Level {level}/{tree_level_max}:")
            pbar.set_postfix({'level': level})
            
            # Configure for this level
            config = base_config.copy()
            config['tree_level_max'] = level
            config['n_processes'] = 1  # Single core
            config['parallel'] = 0  # Single thread
            
            # A* specific config
            astar_config = config.copy()
            astar_config['astar_cost_weight'] = astar_config.get('astar_cost_weight', 100.0)
            astar_config['astar_max_expansions'] = astar_config.get('astar_max_expansions', None)
            
            # Generate star topology (qubit 0 is center, connected to all others)
            topology = generate_star_topology(qbit_num)
            
            # Parallelize across trials
            # Use ThreadPoolExecutor since C++ code releases GIL
            import multiprocessing as mp
            max_workers = min(mp.cpu_count(), num_trials)
            
            # Prepare arguments for parallel execution
            trial_args = [
                (trial, qbit_num, level, topology, config, astar_config, benchmark_tree_search, benchmark_astar)
                for trial in range(num_trials)
            ]
            
            # Run trials in parallel using ThreadPoolExecutor
            # (C++ code releases GIL, so threads work well here)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all trials
                future_to_trial = {
                    executor.submit(run_single_trial, *args): args[0]
                    for args in trial_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_trial):
                    try:
                        tree_result, astar_result = future.result()
                        if tree_result is not None:
                            all_results.append(tree_result)
                        if astar_result is not None:
                            all_results.append(astar_result)
                    except Exception as e:
                        trial_num = future_to_trial[future]
                        # Create error results
                        error_result = {
                            'method': 'error',
                            'qbit_num': qbit_num,
                            'tree_level_max': level,
                            'seed': trial_num + 1000 * qbit_num + 100 * level,
                            'success': False,
                            'cost': float('inf'),
                            'level': -1,
                            'total_time': -1.0,
                            'number_of_iters': 0,
                            'nodes_evaluated': 0,
                            'error': str(e)
                        }
                        all_results.append(error_result)
    
    return all_results


def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze benchmark results and compute statistics.
    
    Args:
        results: List of benchmark result dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    analysis = {}
    
    # Group by method, qubit_num, and level
    for result in results:
        key = (result['method'], result['qbit_num'], result['tree_level_max'])
        if key not in analysis:
            analysis[key] = {
                'times': [],
                'costs': [],
                'successes': [],
                'nodes_evaluated': [],
                'levels': [],
                'cnot_counts': []
            }
        
        if result['total_time'] > 0:
            analysis[key]['times'].append(result['total_time'])
        if result['cost'] < float('inf'):
            analysis[key]['costs'].append(result['cost'])
        if result['success'] is not None:
            analysis[key]['successes'].append(result['success'])
        if result['nodes_evaluated'] > 0:
            analysis[key]['nodes_evaluated'].append(result['nodes_evaluated'])
        if result['level'] >= 0:
            analysis[key]['levels'].append(result['level'])
        if 'cnot_count' in result and result['cnot_count'] >= 0:
            analysis[key]['cnot_counts'].append(result['cnot_count'])
    
    # Compute statistics
    stats = {}
    for key, data in analysis.items():
        method, qbit_num, level = key
        stats[key] = {
            'method': method,
            'qbit_num': qbit_num,
            'level': level,
            'num_trials': len(data['times']),
            'mean_time': np.mean(data['times']) if data['times'] else 0,
            'std_time': np.std(data['times']) if data['times'] else 0,
            'mean_cost': np.mean(data['costs']) if data['costs'] else float('inf'),
            'std_cost': np.std(data['costs']) if data['costs'] else 0,
            'success_rate': np.mean(data['successes']) * 100 if data['successes'] else 0,
            'mean_nodes': np.mean(data['nodes_evaluated']) if data['nodes_evaluated'] else 0,
            'mean_level': np.mean(data['levels']) if data['levels'] else 0,
            'mean_cnot': np.mean(data['cnot_counts']) if data['cnot_counts'] else 0,
        }
    
    return stats


def create_visualizations(stats: Dict, output_dir: str):
    """
    Create visualization plots from statistics.
    
    Args:
        stats: Statistics dictionary from analyze_results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    if sns is not None:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract data by qubit number
    qbit_nums = sorted(set(key[1] for key in stats.keys()))
    
    for qbit_num in qbit_nums:
        # Filter stats for this qubit number
        tree_stats = {k: v for k, v in stats.items() 
                     if k[1] == qbit_num and k[0] == 'tree_search'}
        astar_stats = {k: v for k, v in stats.items() 
                      if k[1] == qbit_num and k[0] == 'astar'}
        
        # Extract levels and times
        tree_levels = sorted([k[2] for k in tree_stats.keys()])
        astar_levels = sorted([k[2] for k in astar_stats.keys()])
        
        # Plot 1: Compilation Time vs Level
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tree_levels:
            tree_times = [tree_stats[('tree_search', qbit_num, l)]['mean_time'] 
                         for l in tree_levels]
            tree_stds = [tree_stats[('tree_search', qbit_num, l)]['std_time'] 
                        for l in tree_levels]
            ax.errorbar(tree_levels, tree_times, yerr=tree_stds, 
                       marker='o', label='Tree Search', linewidth=2, markersize=8)
        
        if astar_levels:
            astar_times = [astar_stats[('astar', qbit_num, l)]['mean_time'] 
                          for l in astar_levels]
            astar_stds = [astar_stats[('astar', qbit_num, l)]['std_time'] 
                         for l in astar_levels]
            ax.errorbar(astar_levels, astar_times, yerr=astar_stds, 
                       marker='s', label='A* Search', linewidth=2, markersize=8)
        
        ax.set_xlabel('Tree Level (Circuit Depth)', fontsize=12)
        ax.set_ylabel('Compilation Time (seconds)', fontsize=12)
        ax.set_title(f'Compilation Time Comparison ({qbit_num} qubits)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_comparison_{qbit_num}q.png'), dpi=300)
        plt.close()
        
        # Plot 2: Nodes Evaluated vs Level
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tree_levels:
            tree_nodes = [tree_stats[('tree_search', qbit_num, l)]['mean_nodes'] 
                         for l in tree_levels]
            ax.plot(tree_levels, tree_nodes, marker='o', label='Tree Search', 
                   linewidth=2, markersize=8)
        
        if astar_levels:
            astar_nodes = [astar_stats[('astar', qbit_num, l)]['mean_nodes'] 
                          for l in astar_levels]
            ax.plot(astar_levels, astar_nodes, marker='s', label='A* Search', 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Tree Level (Circuit Depth)', fontsize=12)
        ax.set_ylabel('Nodes Evaluated', fontsize=12)
        ax.set_title(f'Search Space Exploration ({qbit_num} qubits)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nodes_comparison_{qbit_num}q.png'), dpi=300)
        plt.close()
        
        # Plot 3: Success Rate vs Level
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tree_levels:
            tree_success = [tree_stats[('tree_search', qbit_num, l)]['success_rate'] 
                           for l in tree_levels]
            ax.plot(tree_levels, tree_success, marker='o', label='Tree Search', 
                   linewidth=2, markersize=8)
        
        if astar_levels:
            astar_success = [astar_stats[('astar', qbit_num, l)]['success_rate'] 
                            for l in astar_levels]
            ax.plot(astar_levels, astar_success, marker='s', label='A* Search', 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Tree Level (Circuit Depth)', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(f'Success Rate Comparison ({qbit_num} qubits)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'success_rate_{qbit_num}q.png'), dpi=300)
        plt.close()
        
        # Plot 4: Speedup (Tree Search / A* Time)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        common_levels = sorted(set(tree_levels) & set(astar_levels))
        if common_levels:
            speedups = []
            for l in common_levels:
                tree_time = tree_stats[('tree_search', qbit_num, l)]['mean_time']
                astar_time = astar_stats[('astar', qbit_num, l)]['mean_time']
                if astar_time > 0:
                    speedup = tree_time / astar_time
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            ax.plot(common_levels, speedups, marker='o', linewidth=2, markersize=8, color='green')
            ax.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
            ax.set_xlabel('Tree Level (Circuit Depth)', fontsize=12)
            ax.set_ylabel('Speedup (Tree Search / A*)', fontsize=12)
            ax.set_title(f'A* Speedup Over Tree Search ({qbit_num} qubits)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'speedup_{qbit_num}q.png'), dpi=300)
        plt.close()
    
    # Suppress visualization output
    # print(f"\nVisualizations saved to {output_dir}/")


def save_raw_data(results: List[Dict], stats: Dict, output_dir: str):
    """
    Save raw benchmark data to files.
    
    Args:
        results: List of all benchmark results
        stats: Statistics dictionary
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types to native Python types for JSON
    def convert_to_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # Convert tuple keys to strings for JSON compatibility
            result = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    # Convert tuple key to string representation
                    key_str = str(k)
                elif isinstance(k, (int, float, bool)) or k is None:
                    key_str = k
                else:
                    key_str = str(k)
                result[key_str] = convert_to_json(v)
            return result
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(convert_to_json(item) for item in obj)
        return obj
    
    results_json = convert_to_json(results)
    stats_json = convert_to_json(stats)
    
    results_file = os.path.join(output_dir, f'raw_results_{timestamp}.json')
    stats_file = os.path.join(output_dir, f'statistics_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    with open(stats_file, 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    # Suppress raw data output - only show in final summary
    # print(f"Raw data saved to:")
    # print(f"  {results_file}")
    # print(f"  {stats_file}")


def print_summary(stats: Dict):
    """
    Print summary statistics to console.
    
    Args:
        stats: Statistics dictionary
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    qbit_nums = sorted(set(key[1] for key in stats.keys()))
    
    for qbit_num in qbit_nums:
        print(f"\n{qbit_num}-qubit circuits:")
        print("-" * 80)
        
        # Get all levels for this qubit number
        levels = sorted(set(key[2] for key in stats.keys() if key[1] == qbit_num))
        
        print(f"{'Level':<8} {'Method':<15} {'Time (s)':<15} {'Nodes':<15} {'Success %':<12}")
        print("-" * 80)
        
        for level in levels:
            for method in ['tree_search', 'astar']:
                key = (method, qbit_num, level)
                if key in stats:
                    s = stats[key]
                    method_name = 'Tree Search' if method == 'tree_search' else 'A* Search'
                    print(f"{level:<8} {method_name:<15} "
                          f"{s['mean_time']:.2f} ± {s['std_time']:.2f}  "
                          f"{s['mean_nodes']:.0f}        "
                          f"{s['success_rate']:.1f}%")
                # If method not in stats, it was disabled - skip silently


def main():
    """Main benchmark execution."""
    
    # Configuration
    qbit_nums = [3]
    tree_level_max = 5
    num_trials = 50 # Number of random unitaries per configuration
    
    # Enable/disable specific methods
    benchmark_tree_search = False  # Set to False to skip tree_search
    benchmark_astar = True  # Set to False to skip astar
    
    # Base configuration
    base_config = {
        'optimization_tolerance': 1e-8,
        'optimizer': 'BFGS',
        'cost_function_variant': 3,
        'max_inner_iterations': 5000,
        'n_processes': 1,  # Single core
        'parallel': 0,  # Single thread
        'astar_cost_weight': 100.0,
        'astar_max_expansions': None,  # No limit
    }
    
    # Run benchmarks (suppress output)
    # print("Starting benchmark suite...")
    results = run_benchmark_suite(
        qbit_nums=qbit_nums,
        tree_level_max=tree_level_max,
        num_trials=num_trials,
        base_config=base_config,
        benchmark_tree_search=benchmark_tree_search,
        benchmark_astar=benchmark_astar
    )
    
    # Analyze results
    # print("\n" + "="*80)
    # print("Analyzing results...")
    # print("="*80)
    stats = analyze_results(results)
    
    # Print summary (this is the only output we want)
    print_summary(stats)
    
    # Save data and create visualizations (suppress output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmark_results_{timestamp}"
    
    # print(f"\nSaving results to {output_dir}/...")
    save_raw_data(results, stats, output_dir)
    create_visualizations(stats, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()


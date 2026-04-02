import numpy as np
from squander import Circuit, Qiskit_IO
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import qgd_N_Qubit_Decomposition_custom as N_Qubit_Decomposition_custom
from time import time
from math import ceil
import tqdm
import multiprocessing as mp
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt

Tree_Search = True
Adaptive = False
tag = "zero_start"
N = 3
max_depth = 13  # Focus on depths 1-13 for now
samples = 1000  # Increased to 50000 samples for better statistics
start = 1
step_size = 1
decomp_tol = 1e-8

def construct_unitary(N, depth):
    # Store gate assignments so we can rebuild the circuit
    gates = []
    for idx in range(depth):
        trgt, ctrl = np.random.choice(np.arange(N), 2, replace=False)
        gates.append((trgt, ctrl))

    def build_circuit():
        circ = Circuit(N)
        for trgt, ctrl in gates:
            circo = Circuit(N)
            circo.add_U3(trgt)
            circo.add_U3(ctrl)
            circo.add_CNOT(trgt, ctrl)
            circ.add_Circuit(circo)
        for idx in range(N):
            circo = Circuit(N)
            circo.add_U3(idx)
            circ.add_Circuit(circo)
        return circ

    circ = build_circuit()
    Umtx = circ.get_Matrix(np.random.rand(circ.get_Parameter_Num()) * 2 * np.pi)
    return build_circuit, Umtx

def synthesis(N, depth):
    build_circuit, Umtx = construct_unitary(N, depth)

    # Phase 1: AGENTS
    circ1 = build_circuit()
    config_agents = {
        'parallel': 0,
        'agent_num': 64,
        'max_inner_iterations_agent': 10000,
        'optimization_tolerance_agent':1e-2,
        'linesearch_points':5,
        'agent_lifetime_agent': 200
    }
    cDecomp = N_Qubit_Decomposition_custom(Umtx.conj().T, config=config_agents)
    cDecomp.set_Verbose(0)
    cDecomp.set_Cost_Function_Variant(3)
    cDecomp.set_Optimization_Tolerance(decomp_tol)
    cDecomp.set_Gate_Structure(circ1)
    cDecomp.set_Optimized_Parameters(np.zeros(circ1.get_Parameter_Num()))
    cDecomp.set_Optimizer("AGENTS")
    t0 = time()
    cDecomp.Start_Decomposition()
    agents_params = cDecomp.get_Optimized_Parameters()
    agents_score = cDecomp.Optimization_Problem(agents_params)

    if agents_score < decomp_tol:
        return time() - t0, True

    # Phase 2: BFGS polish from AGENTS result
    circ2 = build_circuit()
    config_bfgs = {'parallel': 0}
    cDecomp2 = N_Qubit_Decomposition_custom(Umtx.conj().T, config=config_bfgs)
    cDecomp2.set_Verbose(0)
    cDecomp2.set_Cost_Function_Variant(3)
    cDecomp2.set_Optimization_Tolerance(decomp_tol)
    cDecomp2.set_Gate_Structure(circ2)
    cDecomp2.set_Optimized_Parameters(agents_params)
    cDecomp2.set_Optimizer("BFGS")
    cDecomp2.Start_Decomposition()
    final_params = cDecomp2.get_Optimized_Parameters()
    score = cDecomp2.Optimization_Problem(final_params)
    return time() - t0, score < decomp_tol

def synthesis_with_metadata(args):
    """Wrapper that includes metadata for result organization"""
    N, depth, sample_idx = args
    comp_time, success = synthesis(N, depth)
    return comp_time, success, depth, sample_idx

shape = len(range(start, max_depth + 1, step_size))
compilation_time = np.zeros(shape=(shape, samples))
success = np.zeros(shape=(shape, samples))

# Create all tasks upfront - more efficient parallelization
print(f"Preparing {shape * samples} total tasks (depths {start}-{max_depth}, {samples} samples each)...")
tasks = [(N, depth, n) for depth in range(start, max_depth + 1, step_size)
         for n in range(samples)]

print(f"Running benchmark with {mp.cpu_count() // 2} parallel processes...")
# Single pool for all work - much more efficient!
with Pool(processes=mp.cpu_count()) as pool:
    # imap_unordered for better performance with progress bar
    results_tree = list(tqdm.tqdm(
        pool.imap_unordered(synthesis_with_metadata, tasks, chunksize=5),
        total=len(tasks),
        desc="Processing AGENTS+BFGS circuits",
        smoothing=0.25
    ))

# Process results and store in arrays
for comp_time, success_placeholder, depth_idx, sample_idx in results_tree:
    depth_array_idx = depth_idx - start  # Convert depth to array index
    compilation_time[depth_array_idx, sample_idx] = comp_time
    success[depth_array_idx, sample_idx] = success_placeholder

# Calculate statistics
depths = np.array(range(start, max_depth + 1, step_size))
failure_counts = samples - success.sum(axis=1)  # Number of failures at each depth
failure_rates = failure_counts / samples * 100  # Percentage of failures
success_rates = success.sum(axis=1) / samples * 100  # Percentage of successes

mean_time = np.zeros(shape)
min_time = np.zeros(shape)
max_time = np.zeros(shape)

for idx in range(shape):
    mean_time[idx] = compilation_time[idx].mean()
    min_time[idx] = compilation_time[idx].min()
    max_time[idx] = compilation_time[idx].max()

if Tree_Search:
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Failure Rate at Each Depth
    ax1 = axes[0]
    ax1.bar(depths, failure_counts, alpha=0.7, color='red', label='Failures')
    ax1.bar(depths, success.sum(axis=1), bottom=failure_counts, alpha=0.7, color='green', label='Successes')
    ax1.set_xlabel('Circuit Depth', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'AGENTS+BFGS Compilation Success/Failure at Each Depth (N={N} qubits, {samples} samples)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (d, f_count, f_rate) in enumerate(zip(depths, failure_counts, failure_rates)):
        if f_count > 0:
            ax1.text(d, f_count / 2, f'{f_rate:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Plot 2: Mean Compilation Time (for successful compilations only)
    ax4 = axes[1]
    valid_indices = ~np.isnan(mean_time)
    yerr_low = mean_time[valid_indices] - min_time[valid_indices]
    yerr_high = max_time[valid_indices] - mean_time[valid_indices]
    ax4.errorbar(depths[valid_indices], mean_time[valid_indices], yerr=[yerr_low, yerr_high],
                 fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8, color='purple', alpha=0.7)
    ax4.set_xlabel('Circuit Depth', fontsize=12)
    ax4.set_yscale('log')
    ax4.set_ylabel('Compilation Time (seconds)', fontsize=12)
    ax4.set_title('Mean Compilation Time for Successful AGENTS+BFGS Compilations', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_name_tree = f'Optimization_benchmark_N{N}_AGENTS_BFGS_{tag}.png' if tag != "" else f'Optimization_benchmark_N{N}_AGENTS_BFGS.png'
    plt.savefig(plot_name_tree, dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "=" * 70)
print(f"AGENTS+BFGS OPTIMIZATION BENCHMARK SUMMARY (N={N} qubits, {samples} samples)")
print("=" * 70)
print(f"{'Depth':<8} {'Success':<10} {'Failure':<10} {'Fail %':<10} {'Mean Time (s)':<15}")
print("-" * 70)
for i, d in enumerate(depths):
    succ = int(success[i].sum())
    fail = int(failure_counts[i])
    fail_pct = failure_rates[i]
    mt = mean_time[i] if not np.isnan(mean_time[i]) else 'N/A'
    if isinstance(mt, float):
        mt_str = f"{mt:.3f}"
    else:
        mt_str = str(mt)

    print(f"{d:<8} {succ:<10} {fail:<10} {fail_pct:<10.1f} {mt_str:<15}")
print("\n" + "=" * 70)

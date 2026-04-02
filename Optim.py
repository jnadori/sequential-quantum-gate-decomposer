import numpy as np
from squander import Circuit, N_Qubit_Decomposition_custom, Qiskit_IO
from time import time
from math import ceil, log
import tqdm
import multiprocessing as mp
from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
Tree_Search = True
Adaptive = False
optimizer = "BFGS2"
tag="BH"
N = 3
max_depth = 13  # Focus on depths 1-5 for now
samples = 10000  # Increased to 1000 samples for better statistics
start=1
step_size=1
decomp_tol = 1e-8
baseline_fails = [0.01,0.02,0.14,0.25,0.41,0.51,0.65,0.75,0.8,0.88,0.92]
target_fail = 0.05
def construct_unitary(N,depth):
    circ = Circuit(N)
    for idx in range(depth):
        circo = Circuit(N)
        trgt,ctrl = np.random.choice(np.arange(N),2,replace=False)
        circo.add_U3(trgt)
        circo.add_U3(ctrl)
        circo.add_CNOT(trgt,ctrl)
        circ.add_Circuit(circo)
    for idx in range(N):
        circo = Circuit(N)
        circo.add_U3(idx)
        circ.add_Circuit(circo)

    return circ,circ.get_Matrix(np.random.rand(circ.get_Parameter_Num())*2*np.pi)

def synthesis(N,depth):
    Circ, Umtx = construct_unitary(N,depth)
    config = {'parallel': 0, 'use_basin_hopping': 1, 'bh_T': 1.1375279022671254, 'bh_stepsize': 0.9200273804590016, 'bh_interval': 94, 'bh_target_accept_rate': 0.5661497388955112, 'bh_stepwise_factor': 0.5557762288919466}
    NVDecompose = N_Qubit_Decomposition_custom(Umtx.conj().T,config = config)
    NVDecompose.set_Verbose(0)
    NVDecompose.set_Optimizer(optimizer)
    NVDecompose.set_Cost_Function_Variant( 3 )
    NVDecompose.set_Optimization_Tolerance( decomp_tol )
    NVDecompose.set_Gate_Structure(Circ)
    NVDecompose.set_Optimized_Parameters(np.random.rand(Circ.get_Parameter_Num())*2*np.pi)
    start = time()
    NVDecompose.Start_Decomposition()
    compilation_time_tree = time()-start
    parameters=NVDecompose.get_Optimized_Parameters()
    sucess_tree = NVDecompose.Optimization_Problem(parameters)<decomp_tol
    return compilation_time_tree, sucess_tree

def synthesis_with_metadata(args):
    """Wrapper that includes metadata for result organization"""
    N, depth, sample_idx = args
    comp_time, success = synthesis(N, depth)
    return comp_time, success, depth, sample_idx

shape = len(range(start,max_depth+1,step_size))
compilation_time=np.zeros(shape=(shape,samples))
success = np.zeros(shape=(shape,samples))
# Create all tasks upfront - more efficient parallelization
print(f"Preparing {shape * samples} total tasks (depths {start}-{max_depth}, {samples} samples each)...")
tasks = [(N, depth, n) for depth in range(start, max_depth+1, step_size)
         for n in range(samples)]

print(f"Running benchmark with {mp.cpu_count()//2} parallel processes...")
# Single pool for all work - much more efficient!
with Pool(processes=mp.cpu_count()//2) as pool:
    # imap_unordered for better performance with progress bar
    results_tree = list(tqdm.tqdm(
    pool.imap_unordered(synthesis_with_metadata, tasks, chunksize=5),
    total=len(tasks),
    desc="Processing tree circuits",
    smoothing=0.25
))   

# Process results and store in arrays
for comp_time, success_placeholder, depth_idx, sample_idx in results_tree:
    depth_array_idx = depth_idx - start  # Convert depth to array index
    compilation_time[depth_array_idx, sample_idx] = comp_time
    success[depth_array_idx, sample_idx] = success_placeholder


# Calculate statistics
depths = np.array(range(start, max_depth+1, step_size))
failure_counts = samples - success.sum(axis=1)  # Number of failures at each depth
failure_rates = failure_counts / samples * 100  # Percentage of failures
success_rates = success.sum(axis=1) / samples * 100  # Percentage of successes

mean_time = np.zeros(shape)
std_time = np.zeros(shape)

for idx in range(shape):
        mean_time[idx] = compilation_time[idx].mean()
        std_time[idx] = compilation_time[idx].std()
if Tree_Search:
# Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Failure Rate at Each Depth
    ax1 = axes[0]
    ax1.bar(depths, failure_counts, alpha=0.7, color='red', label='Failures')
    ax1.bar(depths, success.sum(axis=1), bottom=failure_counts, alpha=0.7, color='green', label='Successes')
    ax1.set_xlabel('Circuit Depth', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Compilation Success/Failure at Each Depth (N={N} qubits, {samples} samples)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (d, f_count, f_rate) in enumerate(zip(depths, failure_counts, failure_rates)):
        if f_count > 0:
            ax1.text(d, f_count/2, f'{f_rate:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')


    # Plot 4: Mean Compilation Time (for successful compilations only)
    ax4 = axes[1]
    valid_indices = ~np.isnan(mean_time)
    ax4.errorbar(depths[valid_indices], mean_time[valid_indices], yerr=std_time[valid_indices],
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8, color='purple', alpha=0.7)
    ax4.set_xlabel('Circuit Depth', fontsize=12)
    ax4.set_ylabel('Compilation Time (seconds)', fontsize=12)
    ax4.set_title('Mean Compilation Time for Successful Compilations', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_name_tree = f'Optimization_benchmark_N{N}_{optimizer}_{tag}.png' if tag != "" else f'Optimization_benchmark_N{N}_{optimizer}.png'
    plt.savefig(plot_name_tree, dpi=300, bbox_inches='tight')
    plt.close()


print("\n" + "="*70)
print(f"OPTIMIZATION BENCHMARK SUMMARY (N={N} qubits, {samples} samples) (Optimizer: {optimizer})")
print("="*70)
print(f"{'Depth':<8} {'Success':<10} {'Failure':<10} {'Fail %':<10} {'Mean Time (s)':<15}")
print("-"*70)
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
print("\n" + "="*70)

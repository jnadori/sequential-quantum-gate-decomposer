"""Benchmark Python vs C++ POSMM implementation: time and failure rate."""

import time
import numpy as np
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import qgd_N_Qubit_Decomposition_custom as N_Qubit_Decomposition_custom
from squander.synthesis.qgd_POSMM import qgd_POSMM_Decomposition

QUBITS = [3]
DEPTHS = [3, 4, 5]
N_TRIALS = 20
TOLERANCE = 1e-8

# Shared POSMM parameters
OPTIMIZATION_LOOPS = 200
WORKER_NUM = 32
RCONSTANT = 0.5
MAX_INNER_ITERATIONS = 10000

POSMM_CONFIG = {
    'optimization_loops': OPTIMIZATION_LOOPS,
    'worker_num': WORKER_NUM,
    'tolerance': TOLERANCE,
    'parallel_worker': 0,
    'max_inner_iterations': MAX_INNER_ITERATIONS,
}

CPP_CONFIG = {
    'max_iteration_loops_posmm': OPTIMIZATION_LOOPS,
    'worker_num_posmm': WORKER_NUM,
    'max_inner_iterations_bfgs': MAX_INNER_ITERATIONS,
    'max_outer_iterations': 1,
    'parallel': 0,
}


def build_topology(N):
    return [(i, j) for i in range(N - 1) for j in range(i + 1, N)]


def random_circuit_edges(N, D):
    return [tuple(np.random.choice(np.arange(N), 2, replace=False)) for _ in range(D)]


def create_circuit(edges, N):
    circ = Circuit(N)
    for edge in edges:
        circ.add_U3(edge[0])
        circ.add_U3(edge[1])
        circ.add_CNOT(edge[0], edge[1])
    for qbit_idx in range(N):
        circ.add_U3(qbit_idx)
    return circ


def generate_unitary(N, edges):
    circ = create_circuit(edges, N)
    return circ.get_Matrix(np.random.rand(circ.get_Parameter_Num()) * 2 * np.pi)


def run_python_posmm(Umtx, gate_structure, config):
    posmm = qgd_POSMM_Decomposition(Umtx, gate_structure, RCONSTANT, config)
    t0 = time.time()
    best_params, best_score = posmm.Start_decomposition()
    elapsed = time.time() - t0
    return best_score, elapsed


def run_cpp_posmm(Umtx, gate_structure, config):
    cDecompose = N_Qubit_Decomposition_custom(Umtx, config=config)
    cDecompose.set_Verbose(0)
    cDecompose.set_Cost_Function_Variant(3)
    cDecompose.set_Optimization_Tolerance(TOLERANCE)
    cDecompose.set_Optimizer("POSMM")
    cDecompose.set_Gate_Structure(gate_structure)
    params = np.random.rand(gate_structure.get_Parameter_Num()) * 2 * np.pi
    cDecompose.set_Optimized_Parameters(params)
    t0 = time.time()
    cDecompose.Start_Decomposition()
    elapsed = time.time() - t0
    end_params = cDecompose.get_Optimized_Parameters()
    score = cDecompose.Optimization_Problem(end_params)
    return score, elapsed


if __name__ == "__main__":
    results = {}

    for N in QUBITS:
        for D in DEPTHS:
            key = (N, D)
            results[key] = {'python': [], 'cpp': []}

            for trial in range(N_TRIALS):
                seed = 1000 * N + 100 * D + trial
                np.random.seed(seed)
                edges = random_circuit_edges(N, D)
                Umtx = generate_unitary(N, edges)
                gate_structure = create_circuit(edges, N)

                # Python POSMM
                np.random.seed(seed + 50000)
                py_score, py_time = run_python_posmm(Umtx.conj().T, gate_structure, POSMM_CONFIG)

                # C++ POSMM
                np.random.seed(seed + 50000)
                cpp_score, cpp_time = run_cpp_posmm(Umtx.conj().T, gate_structure, CPP_CONFIG)

                results[key]['python'].append({'score': py_score, 'time': py_time})
                results[key]['cpp'].append({'score': cpp_score, 'time': cpp_time})

                py_ok = "OK" if py_score < TOLERANCE else "FAIL"
                cpp_ok = "OK" if cpp_score < TOLERANCE else "FAIL"
                print(f"  N={N} D={D} trial {trial+1:>2}/{N_TRIALS}  "
                      f"Py: {py_score:.2e} {py_time:6.2f}s [{py_ok}]  "
                      f"C++: {cpp_score:.2e} {cpp_time:6.2f}s [{cpp_ok}]")

    # Summary
    print(f"\n{'='*100}")
    print(f"{'BENCHMARK SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'N':>3} {'D':>3} | {'':^40s} | {'':^40s}")
    print(f"{'':>3} {'':>3} | {'Python POSMM':^40s} | {'C++ POSMM':^40s}")
    print(f"{'':>3} {'':>3} | {'success':>9} {'mean_t':>8} {'med_t':>8} {'med_score':>12} "
          f"| {'success':>9} {'mean_t':>8} {'med_t':>8} {'med_score':>12}")
    print("-" * 100)

    for (N, D), data in sorted(results.items()):
        for label in ['python', 'cpp']:
            trials = data[label]
            scores = [t['score'] for t in trials]
            times = [t['time'] for t in trials]
            n_success = sum(1 for s in scores if s < TOLERANCE)

            if label == 'python':
                print(f"{N:>3} {D:>3} | {n_success:>4}/{len(trials):<4} "
                      f"{np.mean(times):>7.2f}s {np.median(times):>7.2f}s "
                      f"{np.median(scores):>12.2e}", end=" ")
            else:
                print(f"| {n_success:>4}/{len(trials):<4} "
                      f"{np.mean(times):>7.2f}s {np.median(times):>7.2f}s "
                      f"{np.median(scores):>12.2e}")

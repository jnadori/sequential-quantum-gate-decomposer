"""Compile QASM circuits from circs/ using C++ N_Qubit_Decomposition_Surrogate."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import os
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Surrogate, utils

CIRCS_DIR = "circs"
RESULTS_CSV = "compilation_results.csv"
TOLERANCE = 1e-8


OPTIMIZER_CONFIG_BH = {
    # Local optimizer + basin hopping
    'optimizer': 'BFGS2',
    'parallel': 0,
    'tolerance': TOLERANCE,
    'use_basin_hopping': 1,
    'bh_T': 1.1375279022671254,
    'bh_stepsize': 0.9200273804590016,
    'bh_interval': 94,
    'bh_target_accept_rate': 0.5661497388955112,
    'bh_stepwise_factor': 0.5557762288919466,
    # Surrogate search params
    'kappa': 0.5,                   # LCB exploration weight
    'X0_size': 40,                  # initial random samples per D level
    'candidates_per_iter': 250,     # candidates generated per iteration
    'n_thompson_samples': 15,       # candidates evaluated per iteration
    'local_search_fraction': 0.5,   # fraction of candidates refined by local search
    'max_local_steps': 30,          # local search steps per candidate
    'window_patience': 20,          # iterations without >1% improvement before moving to next D
    'window_max_iters': 80,         # hard cap on iterations per window
    'd_window_width': 2,            # search 2 adjacent D values per window
    'max_consecutive_stagnations': 3, # skip ahead after 3 fruitless windows
    'gp_max_train': 500,            # max GP training points (sparse subset selection)
    'topk_diversity_threshold': 0.95, # Thompson sampling diversity filter
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 3,
    # Rollback config (more patient, fine-grained)
    'rb_kappa': 0.3,
    'rb_window_patience': 50,
    'rb_window_max_iters': 160,
    'rb_candidates_per_iter': 250,
    'rb_n_thompson_samples': 20,
    'rb_local_search_fraction': 0.85,
    'rb_max_local_steps': 75,
}

OPTIMIZER_CONFIG_POSMM = {
    # Local optimizer + basin hopping
    'optimizer': 'POSMM',
    'parallel': 0,
    'tolerance': TOLERANCE,
    'worker_num': 5,
    'max_iteration_loops_posmm':5,
    # Surrogate search params
    'kappa': 0.5,                   # LCB exploration weight
    'X0_size': 15,                  # initial random samples per D level
    'candidates_per_iter': 250,     # candidates generated per iteration
    'n_thompson_samples': 20,       # candidates evaluated per iteration
    'local_search_fraction': 0.75,   # fraction of candidates refined by local search
    'max_local_steps': 50,          # local search steps per candidate
    'window_patience': 25,          # iterations without >1% improvement before moving to next D
    'window_max_iters': 80,         # hard cap on iterations per window
    'd_window_width': 2,            # search 2 adjacent D values per window
    'max_consecutive_stagnations': 3, # skip ahead after 3 fruitless windows
    'gp_max_train': 1000,            # max GP training points (sparse subset selection)
    'topk_diversity_threshold': 0.95, # Thompson sampling diversity filter
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 3,
    # Rollback config (more patient, fine-grained)
    'rb_kappa': 0.3,
    'rb_window_patience': 50,
    'rb_window_max_iters': 160,
    'rb_candidates_per_iter': 250,
    'rb_n_thompson_samples': 20,
    'rb_local_search_fraction': 0.85,
    'rb_max_local_steps': 75,
}

OPTIMIZER_CONFIG = OPTIMIZER_CONFIG_BH


TWO_QUBIT_GATES = {'CNOT', 'CZ', 'CH', 'CU', 'CP', 'CR', 'CRX', 'CRY', 'CRZ',
                    'CROT', 'SYC', 'RXX', 'RYY', 'RZZ', 'SWAP'}


def count_two_qubit_gates(filename):
    circ, params = utils.qasm_to_squander_circuit(filename)
    gate_counts = circ.get_Gate_Nums()
    return sum(gate_counts.get(g, 0) for g in TWO_QUBIT_GATES)


def get_unitary(filename):
    qc = QuantumCircuit.from_qasm_file(filename)
    return utils.get_unitary_from_qiskit_circuit_operator(qc)


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCS_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No QASM files found in {CIRCS_DIR}/")
        return

    # Write CSV header
    write_header = not os.path.exists(RESULTS_CSV)
    if write_header:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['circuit', 'qubits', 'two_qbit_original', 'two_qbit_compiled',
                             'compilation_time_s', 'error'])

    for qasm_file in qasm_files:
        name = os.path.basename(qasm_file)
        print(f"\n{'='*60}")
        print(f"Circuit: {name}")
        print(f"{'='*60}")

        two_qbit_original = count_two_qubit_gates(qasm_file)
        Umtx = get_unitary(qasm_file)
        N = int(np.log2(Umtx.shape[0]))

        print(f"  Qubits: {N}, Original 2-qubit gates: {two_qbit_original}")

        config = dict(OPTIMIZER_CONFIG)
        config['level_limit'] = two_qbit_original
        config['parallel'] = 0
        decomp = N_Qubit_Decomposition_Surrogate(Umtx.conj().T, config=config)
        decomp.set_Optimizer(config['optimizer'])
        decomp.set_Cost_Function_Variant(3)
        decomp.set_Project_Name(os.path.splitext(name)[0]+'_'+config['optimizer'])

        t0 = time.time()
        decomp.Start_Decomposition()
        elapsed = time.time() - t0

        best_score = decomp.get_Decomposition_Error()
        two_qbit_compiled = decomp.get_Gate_Num() - 1  # blocks minus finalizing layer

        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}, Error: {best_score:.2e}, Time: {elapsed:.1f}s")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled, f"{elapsed:.2f}", f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

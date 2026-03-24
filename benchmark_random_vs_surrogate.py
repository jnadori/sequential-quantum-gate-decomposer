"""Benchmark surrogate-guided vs random candidate selection on QASM circuits."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Surrogate, utils

CIRCS_DIR = "circs"
RESULTS_CSV = "benchmark_random_vs_surrogate.csv"
TOLERANCE = 1e-8

# Shared config (same optimizer for both modes so only candidate selection differs)
BASE_CONFIG = {
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
    'kappa': 0.5,
    'X0_size': 40,
    'candidates_per_iter': 250,
    'n_thompson_samples': 15,
    'local_search_fraction': 0.5,
    'max_local_steps': 30,
    'window_patience': 20,
    'window_max_iters': 80,
    'd_window_width': 2,
    'max_consecutive_stagnations': 3,
    'gp_max_train': 500,
    'topk_diversity_threshold': 0.95,
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 3,
    # Rollback config
    'rb_kappa': 0.3,
    'rb_window_patience': 50,
    'rb_window_max_iters': 160,
    'rb_candidates_per_iter': 250,
    'rb_n_thompson_samples': 20,
    'rb_local_search_fraction': 0.85,
    'rb_max_local_steps': 75,
}


TWO_QUBIT_GATES = {'CNOT', 'CZ', 'CH', 'CU', 'CP', 'CR', 'CRX', 'CRY', 'CRZ',
                    'CROT', 'SYC', 'RXX', 'RYY', 'RZZ', 'SWAP'}


def count_two_qubit_gates(filename):
    circ, params = utils.qasm_to_squander_circuit(filename)
    gate_counts = circ.get_Gate_Nums()
    return sum(gate_counts.get(g, 0) for g in TWO_QUBIT_GATES)


def get_unitary(filename):
    qc = QuantumCircuit.from_qasm_file(filename)
    return utils.get_unitary_from_qiskit_circuit_operator(qc)


def run_decomposition(Umtx, config, name_suffix):
    """Run a single decomposition and return (two_qbit_compiled, error, elapsed)."""
    decomp = N_Qubit_Decomposition_Surrogate(Umtx.conj().T, config=config)
    decomp.set_Optimizer(config['optimizer'])
    decomp.set_Cost_Function_Variant(3)
    decomp.set_Project_Name(name_suffix)

    t0 = time.time()
    decomp.Start_Decomposition()
    elapsed = time.time() - t0

    best_score = decomp.get_Decomposition_Error()
    two_qbit_compiled = decomp.get_Gate_Num() - 1  # blocks minus finalizing layer

    return two_qbit_compiled, best_score, elapsed


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
            writer.writerow([
                'circuit', 'qubits', 'two_qbit_original',
                'two_qbit_surrogate', 'error_surrogate', 'time_surrogate',
                'two_qbit_random', 'error_random', 'time_random',
            ])

    for qasm_file in qasm_files:
        name = os.path.basename(qasm_file)
        print(f"\n{'='*60}")
        print(f"Circuit: {name}")
        print(f"{'='*60}")

        two_qbit_original = count_two_qubit_gates(qasm_file)
        Umtx = get_unitary(qasm_file)
        N = int(np.log2(Umtx.shape[0]))

        print(f"  Qubits: {N}, Original 2-qubit gates: {two_qbit_original}")

        # --- Surrogate-guided run ---
        print(f"  Running SURROGATE mode...")
        config_sur = dict(BASE_CONFIG)
        config_sur['level_limit'] = two_qbit_original
        config_sur['use_random_candidates'] = 0
        tq_sur, err_sur, time_sur = run_decomposition(
            Umtx, config_sur, os.path.splitext(name)[0] + '_surrogate')
        print(f"    2q={tq_sur}, error={err_sur:.2e}, time={time_sur:.1f}s")

        # --- Random baseline run ---
        print(f"  Running RANDOM mode...")
        config_rand = dict(BASE_CONFIG)
        config_rand['level_limit'] = two_qbit_original
        config_rand['use_random_candidates'] = 1
        tq_rand, err_rand, time_rand = run_decomposition(
            Umtx, config_rand, os.path.splitext(name)[0] + '_random')
        print(f"    2q={tq_rand}, error={err_rand:.2e}, time={time_rand:.1f}s")

        # --- Write results ---
        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                name, N, two_qbit_original,
                tq_sur, f"{err_sur:.2e}", f"{time_sur:.2f}",
                tq_rand, f"{err_rand:.2e}", f"{time_rand:.2f}",
            ])

        print(f"  Surrogate: {tq_sur} gates ({err_sur:.2e}) in {time_sur:.1f}s")
        print(f"  Random:    {tq_rand} gates ({err_rand:.2e}) in {time_rand:.1f}s")

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

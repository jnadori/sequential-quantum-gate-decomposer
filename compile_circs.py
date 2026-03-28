"""Compile QASM circuits from circs/ using C++ N_Qubit_Decomposition_Surrogate."""

import csv
import glob
import os
import tempfile
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Surrogate_GateLevel, utils

CIRCS_DIR = "circs"
RESULTS_CSV = "compilation_results_qmill.csv"
TOLERANCE = 1e-8


OPTIMIZER_CONFIG_BH = {
    # Local optimizer + basin hopping
    'optimizer': 'BFGS2',
    'parallel': 2,
    'tolerance': TOLERANCE,
    'use_basin_hopping': 1,
    'bh_T': 1.1375279022671254,
    'bh_stepsize': 0.9200273804590016,
    'bh_interval': 94,
    'bh_target_accept_rate': 0.5661497388955112,
    'bh_stepwise_factor': 0.5557762288919466,
    # IBM Eagle native gateset: {RZ, SX, X, CX}
    # Bitmask: RZ=bit3, X=bit5, SX=bit12 → (1<<3)|(1<<5)|(1<<12) = 4136
    'gate_1q_gateset': 4136,
    # Surrogate search params
    'kappa': 0.1,                   # LCB exploration weight
    'X0_size': 150,                  # initial random samples per D level
    'candidates_per_iter': 250,     # candidates generated per iteration
    'n_thompson_samples': 50,       # candidates evaluated per iteration
    'local_search_fraction': 0.5,   # fraction of candidates refined by local search
    'max_local_steps': 30,          # local search steps per candidate
    'window_patience': 25,          # iterations without >1% improvement before moving to next D
    'window_max_iters': 500,         # hard cap on iterations per window
    'd_window_width': 2,            # search 2 adjacent D values per window
    'max_consecutive_stagnations': 5, # skip ahead after 3 fruitless windows
    'gp_max_train': 1000,            # max GP training points (sparse subset selection)
    'topk_diversity_threshold': 0.95, # Thompson sampling diversity filter
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 3,
    # Rollback config (more patient, fine-grained)
    'rb_kappa': 0.03,
    'rb_window_patience': 250,
    'rb_window_max_iters': 2000,
    'rb_candidates_per_iter': 500,
    'rb_n_thompson_samples': 50,
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
        config['parallel'] = 2
        decomp = N_Qubit_Decomposition_Surrogate_GateLevel(np.ascontiguousarray(Umtx.conj().T), config=config)
        decomp.set_Verbose(0)
        decomp.set_Optimizer(config['optimizer'])
        decomp.set_Cost_Function_Variant(3)
        decomp.set_Project_Name(os.path.splitext(name)[0]+'_'+config['optimizer'])

        t0 = time.time()
        decomp.Start_Decomposition()
        elapsed = time.time() - t0

        best_score = decomp.get_Decomposition_Error()
        # Count CNOTs by capturing List_Gates C++ stdout
        saved_fd = os.dup(1)
        with tempfile.TemporaryFile(mode='w+b') as tmp:
            os.dup2(tmp.fileno(), 1)
            decomp.List_Gates()
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
            tmp.seek(0)
            listing = tmp.read().decode('utf-8', errors='replace')
        two_qbit_compiled = listing.count(': CNOT ')

        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}, Error: {best_score:.2e}, Time: {elapsed:.1f}s")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled, f"{elapsed:.2f}", f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

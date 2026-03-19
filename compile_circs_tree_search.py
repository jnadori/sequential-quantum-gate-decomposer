"""Compile QASM circuits from circs/ using squander's Tree Search decomposition."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import glob
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Tree_Search, utils

CIRCS_DIR = "circs"
RESULTS_CSV = "compilation_results_tree_search.csv"
TOLERANCE = 1e-8

CONFIG = {
    'optimization_tolerance': TOLERANCE,
    'use_osr': 1,
    'use_graph_search': 1,
    'parallel': 0,
    'tree_level_max': 14,
    'stop_first_solution': 1,
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


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCS_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No QASM files found in {CIRCS_DIR}/")
        return

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

        config = dict(CONFIG)
        config['tree_level_max'] = int(np.floor((4**N - 3*N - 1) / 4))

        cDecomp = N_Qubit_Decomposition_Tree_Search(Umtx.conj().T, config=config)
        cDecomp.set_Verbose(0)

        t0 = time.time()
        cDecomp.Start_Decomposition()
        elapsed = time.time() - t0

        error = cDecomp.get_Decomposition_Error()
        circ = cDecomp.get_Circuit()
        gate_counts = circ.get_Gate_Nums()
        two_qbit_compiled = sum(gate_counts.get(g, 0) for g in TWO_QUBIT_GATES)

        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}, Error: {error:.2e}, Time: {elapsed:.1f}s")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled,
                             f"{elapsed:.2f}", f"{error:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

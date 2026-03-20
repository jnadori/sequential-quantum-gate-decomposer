"""Compile QASM circuits from circs/ using SurSearch Start_Decomposition."""

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
from squander import utils
from SurSearch import qgd_SurSearch

CIRCS_DIR = "circs"
RESULTS_CSV = "compilation_results.csv"
TOLERANCE = 1e-8


OPTIMIZER_CONFIG = {
    'optimizer': 'BFGS2',
    'parallel': 0,
    'tolerance': TOLERANCE,
    'acquisition': 'LCB',
    'use_basin_hopping': 1, 'bh_T': 1.1375279022671254, 'bh_stepsize': 0.9200273804590016, 'bh_interval': 94, 'bh_target_accept_rate': 0.5661497388955112, 'bh_stepwise_factor': 0.5557762288919466
}


def x0_size_for_d(D):
    return 20 if D > 7 else 5


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
        D_START = max(5,two_qbit_original//2+1)
        Umtx = get_unitary(qasm_file)
        N = int(np.log2(Umtx.shape[0]))

        print(f"  Qubits: {N}, Original 2-qubit gates: {two_qbit_original}")

        if two_qbit_original < D_START:
            print(f"  Skipping: original 2-qubit gate count ({two_qbit_original}) < D_START ({D_START})")
            with open(RESULTS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, N, two_qbit_original, two_qbit_original, 0.0, 0.0])
            continue

        config = dict(OPTIMIZER_CONFIG)
        searcher = qgd_SurSearch(Umtx, config, D_START)

        t0 = time.time()
        real_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            for D in range(D_START, two_qbit_original + 1):
                config['X0_size'] = x0_size_for_d(D)
                searcher.D = D
                searcher.search_over_D(log_file=f"sursearch_logs/{name}_D{D}.txt")
                dec_t = getattr(searcher, '_decompose_time', 0.0)
                dec_n = getattr(searcher, '_decompose_count', 0)
                tot_t = getattr(searcher, '_total_search_time', 0.0)
                py_t = tot_t - dec_t
                print(f"  D={D}: best_score={searcher.best_score:.6e}  "
                      f"decompositions={dec_n}  "
                      f"decompose_time={dec_t:.1f}s  "
                      f"python/GP_time={py_t:.1f}s  "
                      f"total={tot_t:.1f}s",
                      file=real_stdout, flush=True)
                if searcher.best_score < TOLERANCE:
                    break
        finally:
            sys.stdout.close()
            sys.stdout = real_stdout
        elapsed = time.time() - t0

        best_score = searcher.best_score
        two_qbit_compiled = len(searcher.best_circ) if searcher.best_circ is not None else two_qbit_original

        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}, Error: {best_score:.2e}, Time: {elapsed:.1f}s")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled, f"{elapsed:.2f}", f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

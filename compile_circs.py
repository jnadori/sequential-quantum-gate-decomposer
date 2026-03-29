"""Compile QASM circuits from circs/ using C++ N_Qubit_Decomposition_Surrogate."""

import csv
import glob
import os
import tempfile
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Surrogate, Qiskit_IO, utils

CIRCS_DIR = "circs_qasm2"
OUTPUT_DIR = "compiled_circs"
RESULTS_CSV = "compilation_results_qmill.csv"
TOLERANCE = 1e-8


OPTIMIZER_CONFIG = {
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
    'kappa': 0.5,
    'X0_size': 150,
    'candidates_per_iter': 1500,
    'n_thompson_samples': 500,  # top-k candidates selected via LCB + diversity
    'local_search_fraction': 0.5,
    'max_local_steps': 30,
    'local_search_positions': 5,       # sampled positions per local search step
    'local_search_gp_subset': 50,      # lightweight GP training subset for local search
    'window_max_iters': 500,
    'stagnation_window': 10,
    'stagnation_improvement_frac': 0.01,
    'window_patience': 75,             # hard patience cap (fallback)
    'gp_max_train': 300,
    'topk_diversity_threshold': 0.95,
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 3,
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write CSV header
    write_header = not os.path.exists(RESULTS_CSV)
    if write_header:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['circuit', 'qubits', 'two_qbit_original', 'two_qbit_compiled', 'gate_num_compiled',
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
        decomp = N_Qubit_Decomposition_Surrogate(np.ascontiguousarray(Umtx.conj().T), config=config)
        decomp.set_Verbose(0)
        decomp.set_Optimizer(config['optimizer'])
        decomp.set_Cost_Function_Variant(3)
        decomp.set_Project_Name(os.path.splitext(name)[0]+'_'+config['optimizer'])

        t0 = time.time()
        decomp.Start_Decomposition()
        elapsed = time.time() - t0

        best_score = decomp.get_Decomposition_Error()
        circ = decomp.get_Circuit()
        params = decomp.get_Optimized_Parameters()
        gate_num = circ.get_Gate_Nums()
        two_qbit_compiled = gate_num.get('CNOT',0)
        gate_num_compiled = sum([x for x in gate_num.values()])
        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}, Total gate count: {gate_num_compiled}, Error: {best_score:.2e}, Time: {elapsed:.1f}s")

        # Save compiled circuit as QASM
        qc = Qiskit_IO.get_Qiskit_Circuit(circ, params)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + "_compiled.qasm")
        from qiskit.qasm2 import dump as qasm2_dump
        with open(out_path, 'w') as qf:
            qasm2_dump(qc, qf)
        print(f"  Saved: {out_path}")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled, gate_num_compiled, f"{elapsed:.2f}", f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

"""Compile QASM circuits using C++ N_Qubit_Decomposition_Surrogate.
Compression-only: loads target circuit and top-down compresses it."""

import csv
import glob
import os
import time

import numpy as np
from qiskit import QuantumCircuit
from squander import N_Qubit_Decomposition_Surrogate, Qiskit_IO, utils

CIRCS_DIR = "circs"
OUTPUT_DIR = "compiled_circs"
RESULTS_CSV = "compilation_results_MQBENCH_WL_EI.csv"
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
    'kappa': 2.,
    'X0_size': 150,
    'candidates_per_iter': 250,
    'n_thompson_samples': 500,
    'topk_diversity_threshold': 1.,
    'local_search_fraction': 0.5,
    'max_local_steps': 30,
    'local_search_positions': 10,
    'local_search_gp_subset': 2000,
    'window_max_iters': 500,
    'stagnation_window': 75,
    'stagnation_improvement_frac': 0.01,
    'window_patience': 50,
    'gp_max_train': 2000,
    # SSK kernel params
    'ssk_gap_decay': 0.8,
    'ssk_match_decay': 0.8,
    'ssk_order': 4,
    'D_start': 1,
    # BOSS acquisition-guided GA (set use_boss_ga=1 to enable)
    'use_boss_ga': 1,
    'boss_pop_size': 200,
    'boss_generations': 10,
    'boss_offspring_ratio': 2.0,
    'acquisition_function': 0,  # 0=LCB, 1=EI,
    'kernel_type':1,
    'wl_iterations': 5,
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
            writer.writerow(['circuit', 'qubits', 'two_qbit_original',
                             'two_qbit_compressed', 'gate_num_compiled',
                             'time_s', 'error'])

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
        config['level_limit'] = two_qbit_original * 3
        config['parallel'] = 2
        project = os.path.splitext(name)[0] + '_' + config['optimizer']

        # Load target circuit and transpile to CNOT+U3 basis via Qiskit (optimization_level=0)
        from qiskit import transpile
        qc = QuantumCircuit.from_qasm_file(qasm_file)
        qc_basis = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=0)
        orig_circ, orig_params = Qiskit_IO.convert_Qiskit_to_Squander(qc_basis)

        # Compression with loaded circuit
        decomp = N_Qubit_Decomposition_Surrogate(np.ascontiguousarray(Umtx.conj().T), config=config)
        decomp.set_Verbose(0)
        decomp.set_Optimizer(config['optimizer'])
        decomp.set_Cost_Function_Variant(3)
        decomp.set_Project_Name(project + '_compress')
        decomp.set_Gate_Structure(orig_circ)
        decomp.set_Optimized_Parameters(orig_params)

        t0 = time.time()
        decomp.Start_Decomposition()
        elapsed = time.time() - t0

        best_score = decomp.get_Decomposition_Error()
        circ = decomp.get_Circuit()
        params = decomp.get_Optimized_Parameters()
        gate_num = circ.get_Gate_Nums()
        two_qbit_compiled = gate_num.get('CNOT', 0)
        gate_num_compiled = sum(gate_num.values())
        print(f"  Result: {two_qbit_compiled} CNOTs, Total: {gate_num_compiled}, Error: {best_score:.2e}, Time: {elapsed:.1f}s")

        # Save compiled circuit as QASM
        qc = Qiskit_IO.get_Qiskit_Circuit(circ, params)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + "_compiled.qasm")
        from qiskit.qasm2 import dump as qasm2_dump
        with open(out_path, 'w') as qf:
            qasm2_dump(qc, qf)
        print(f"  Saved: {out_path}")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_compiled,
                             gate_num_compiled, f"{elapsed:.2f}",
                             f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

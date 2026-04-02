"""Compile QASM circuits from circs/ using C++ N_Qubit_Decomposition_Surrogate_GateLevel.
Two-phase: bottom-up search with dense R+Rz+CROT, then top-down compression."""

import csv
import glob
import os
import time

import numpy as np
from qiskit import QuantumCircuit
import quasar
from squander import N_Qubit_Decomposition_Surrogate_GateLevel, Qiskit_IO, utils

CIRCS_DIR = "circs_qasm2"
OUTPUT_DIR = "compiled_circs_gl"
RESULTS_CSV = "compilation_results_gl.csv"
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
    # Surrogate search params
    'kappa': 0.5,
    'X0_size': 150,
    'candidates_per_iter': 1500,
    'n_thompson_samples': 125,
    'topk_diversity_threshold': 0.95,
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
    'D_start': 5,
    # BOSS acquisition-guided GA (set use_boss_ga=1 to enable)
    'use_boss_ga': 0,
    'boss_pop_size': 200,
    'boss_generations': 10,
    'boss_offspring_ratio': 2.0,
    'acquisition_function': 0,  # 0=LCB, 1=EI
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

    write_header = not os.path.exists(RESULTS_CSV)
    if write_header:
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['circuit', 'qubits', 'two_qbit_original',
                             'two_qbit_search', 'two_qbit_compressed',
                             'gate_num_compiled', 'search_time_s', 'compress_time_s',
                             'error'])

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

        # Star topology: qubit 0 is hub, connected to all others
        star_topology = [(i, 0) for i in range(1, N)]

        # Phase 1: Bottom-up search
        print("  Phase 1: Bottom-up search...")
        decomp = N_Qubit_Decomposition_Surrogate_GateLevel(
            np.ascontiguousarray(Umtx.conj().T), topology=star_topology, config=config)
        decomp.set_Verbose(0)
        decomp.set_Optimizer(config['optimizer'])
        decomp.set_Cost_Function_Variant(3)
        decomp.set_Project_Name(project)

        t0 = time.time()
        decomp.Start_Decomposition()
        search_time = time.time() - t0

        search_score = decomp.get_Decomposition_Error()
        search_circ = decomp.get_Circuit()
        search_params = decomp.get_Optimized_Parameters()
        search_gate_num = search_circ.get_Gate_Nums()
        two_qbit_search = search_gate_num.get('CROT', 0)
        print(f"  Search result: {two_qbit_search} CROTs, Error: {search_score:.2e}, Time: {search_time:.1f}s")

        # Phase 2: Top-down compression
        print("  Phase 2: Top-down compression...")
        decomp2 = N_Qubit_Decomposition_Surrogate_GateLevel(
            np.ascontiguousarray(Umtx.conj().T), topology=star_topology, config=config)
        decomp2.set_Verbose(0)
        decomp2.set_Optimizer(config['optimizer'])
        decomp2.set_Cost_Function_Variant(3)
        decomp2.set_Project_Name(project + '_compress')
        decomp2.set_Gate_Structure(search_circ)
        decomp2.set_Optimized_Parameters(search_params)

        t0 = time.time()
        decomp2.Start_Decomposition()
        compress_time = time.time() - t0

        best_score = decomp2.get_Decomposition_Error()
        circ = decomp2.get_Circuit()
        params = decomp2.get_Optimized_Parameters()
        gate_num = circ.get_Gate_Nums()
        two_qbit_compiled = gate_num.get('CROT', 0)
        gate_num_compiled = sum(gate_num.values())
        print(f"  Compressed: {two_qbit_compiled} CROTs, Total: {gate_num_compiled}, Error: {best_score:.2e}, Time: {compress_time:.1f}s")

        # Save compiled circuit as QASM
        qc = Qiskit_IO.get_Qiskit_Circuit(circ, params)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + "_compiled.qasm")
        from qiskit.qasm2 import dump as qasm2_dump
        with open(out_path, 'w') as qf:
            qasm2_dump(qc, qf)
        print(f"  Saved: {out_path}")

        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, N, two_qbit_original, two_qbit_search, two_qbit_compiled,
                             gate_num_compiled, f"{search_time:.2f}", f"{compress_time:.2f}",
                             f"{best_score:.2e}"])

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()

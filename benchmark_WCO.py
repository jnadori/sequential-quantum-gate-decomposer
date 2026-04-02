"""
Benchmark PartAM cleanup phase per circuit.

Runs each circuit 5 times with PartAM (cleanup=True) and records:
  - qubit count
  - initial CNOT count (original QASM circuit)
  - CNOT count before cleanup (post-synthesis, pre-cleanup)
  - CNOT count after cleanup (final)
  - decomposition error
  - compilation time (seconds)

Results are exported to benchmark_cleanup.csv.

Usage:
    conda activate qgd
    python examples/decomposition/benchmark_cleanup.py
"""

import numpy as np
import time
import os
import glob
import csv
import random

from squander import Wide_Circuit_Optimization
from squander import utils
from squander import Circuit

N_RUNS = 3
OUTPUT_CSV = "benchmark_WCO.csv"


def run_once(circ_orig, parameters_orig):
    config = {
        'strategy': "SurrSearch",
        'test_subcircuits': True,
        'test_final_circuit': False,
        'max_partition_size': 3,
        'verbosity': 0,
        'optimizer':'BFGS2',
        'parallel':0,
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
        'level_limit':14,
        # Rollback config (more patient, fine-grained)
        'rb_kappa': 0.3,
        'rb_window_patience': 50,
        'rb_window_max_iters': 160,
        'rb_candidates_per_iter': 250,
        'rb_n_thompson_samples': 20,
        'rb_local_search_fraction': 0.85,
        'rb_max_local_steps': 75,
        'use_basin_hopping': 1, 'bh_T': 1.1375279022671254, 'bh_stepsize': 0.9200273804590016, 'bh_interval': 94, 'bh_target_accept_rate': 0.5661497388955112, 'bh_stepwise_factor': 0.5557762288919466
    }

    start = time.time()
    wide_circuit_optimizer = Wide_Circuit_Optimization.qgd_Wide_Circuit_Optimization( config )
    circ, parameters = wide_circuit_optimizer.OptimizeWideCircuit( circ_orig, parameters_orig)

    elapsed = time.time() - start
    cnot_after_cleanup = circ.get_Gate_Nums().get('CNOT', 0)

    return cnot_after_cleanup, elapsed


if __name__ == '__main__':
    circs_dir = "WCO_circs"
    qasm_files = sorted(glob.glob(os.path.join(circs_dir, "*.qasm")))

    if not qasm_files:
        print(f"No .qasm files found in {circs_dir}/")
        exit(1)

    print(f"Found {len(qasm_files)} circuits in {circs_dir}/")
    print(f"Running {N_RUNS} times per circuit (cleanup=True)\n")

    fieldnames = [
        'circuit', 'n_qubits', 'run',
        'initial_cnot', 'cnot_pre_cleanup', 'cnot_post_cleanup',
        'error', 'time_s','routing_time_s'
    ]

    # Open CSV once and flush after each circuit so partial results are never lost
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in qasm_files:
            name = os.path.basename(filepath)
            print(f"{'='*70}")
            print(f"Circuit: {name}")

            circ_orig, parameters_orig = utils.qasm_to_squander_circuit(filepath)
            n_qubits = circ_orig.get_Qbit_Num()
            initial_cnot = circ_orig.get_Gate_Nums().get('CNOT', 0)
            print(f"Qubits: {n_qubits}, Initial CNOTs: {initial_cnot}")
            print(f"{'Run':>4} {'Pre-cleanup':>12} {'Post-cleanup':>12} {'Error':>12} {'Time(s)':>10} {'Routing time(s)':>10}")

            for run_idx in range(N_RUNS):
                cnot_post, elapsed = run_once(circ_orig, parameters_orig)
                print(f"{run_idx:>4} {cnot_post:>12} {elapsed:>10.1f} ")
                writer.writerow({
                    'circuit': name,
                    'n_qubits': n_qubits,
                    'run': run_idx,
                    'initial_cnot': initial_cnot,
                    'cnot_post_cleanup': cnot_post,
                    'time_s': round(elapsed, 3),
                })
                f.flush()

            print()

    print(f"Results saved to {OUTPUT_CSV}")
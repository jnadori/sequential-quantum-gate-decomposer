"""Benchmark SurSearch across qubit counts and circuit depths."""

import sys
import time
import numpy as np
from SurSearch import qgd_SurSearch, generate_goal_unitary, unique_k_sequences

QUBITS = [3]
DEPTHS = list(range(4, 6))
N_TRIALS = 25
ACQUISITIONS = ['LCB']
KERNELS = ['ssk']
TOLERANCE = 1e-8

OPTIMIZER_CONFIGS = {
    'AGENTS_BFGS': {
        'optimizer': 'AGENTS_BFGS',
        'parallel': 2,
        'tolerance': TOLERANCE,
        'agent_num': 64,
        'agent_lifetime': 1000,
        'radius_base_agent': 0.5,
        'rediversify_keep_top_agent': 3,
    },
    'POSMM': {
        'optimizer': 'POSMM',
        'parallel': 2,
        'kappa': 0.5,
        'max_iteration_loops': 10,
        'worker_num': 8,
        'radius_base': 0.5,
        'tolerance': TOLERANCE,
        'max_inner_iterations_bfgs': 10000,
    },
}

def build_topology(N):
    return [(i, j) for i in range(N - 1) for j in range(i + 1, N)]


def random_goal_edges(N, D):
    topo = build_topology(N)
    return [topo[np.random.randint(len(topo))] for _ in range(D)]


def parse_log(log_file):
    """Extract iteration count and success iteration from a log file."""
    n_iters = 0
    success_iter = None
    try:
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('Iteration '):
                    n_iters += 1
                if line.startswith('Success at iteration:'):
                    success_iter = int(line.split(':')[-1].strip())
    except FileNotFoundError:
        pass
    return n_iters, success_iter


def run_one(N, D, seed, acquisition, kernel='handcrafted', opt_name='POSMM'):
    np.random.seed(seed)
    goal_edges = random_goal_edges(N, D)
    Umtx = generate_goal_unitary(N, edges=goal_edges)

    topo = build_topology(N)
    n_circuits = len(unique_k_sequences(topo, D))

    config = dict(OPTIMIZER_CONFIGS[opt_name])
    config['acquisition'] = acquisition
    config['kernel'] = kernel
    config['X0_size'] = 5
    config['max_sur_iters'] = max(1, n_circuits - 5)

    log_file = f"bench_N{N}_D{D}_s{seed}_{acquisition}_{kernel}_{opt_name}.txt"
    searcher = qgd_SurSearch(Umtx, config, D, topology=topo)

    old_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    try:
        t0 = time.time()
        best_circ = searcher.search_over_D(log_file=log_file)
        elapsed = time.time() - t0
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    best_score = searcher.best_score
    n_iters, success_iter = parse_log(log_file)

    # Compute total evaluations to first success
    if success_iter is not None:
        evals_to_success = config['X0_size'] + success_iter + 1
    elif best_score < TOLERANCE:
        evals_to_success = config['X0_size']  # succeeded in initial sample
    else:
        evals_to_success = None

    return {
        'n_circuits': n_circuits,
        'best_score': best_score,
        'time_s': elapsed,
        'success': best_score < TOLERANCE,
        'n_iters': n_iters,
        'success_iter': success_iter,
        'evals_to_success': evals_to_success,
        'X0_size': config['X0_size'],
    }


def survival_at(trials, k):
    """P(not yet succeeded by evaluation k)."""
    n = len(trials)
    not_yet = sum(1 for t in trials
                  if t['evals_to_success'] is None or t['evals_to_success'] > k)
    return not_yet / n


if __name__ == "__main__":
    results = {}
    for N in QUBITS:
        for D in DEPTHS:
            topo = build_topology(N)
            n_circ = len(unique_k_sequences(topo, D))
            if n_circ < 2:
                print(f"Skipping N={N}, D={D}: only {n_circ} circuit(s)")
                continue

            for opt_name in OPTIMIZER_CONFIGS:
                for acquisition in ACQUISITIONS:
                    for kernel in KERNELS:
                        trials = []
                        for trial in range(N_TRIALS):
                            seed = 1000 * N + 100 * D + trial
                            print(f"  N={N} D={D} opt={opt_name} acq={acquisition} kern={kernel} "
                                  f"trial {trial+1}/{N_TRIALS} ...",
                                  end=" ", flush=True)
                            res = run_one(N, D, seed, acquisition, kernel=kernel, opt_name=opt_name)
                            trials.append(res)
                            ok = "OK" if res['success'] else "FAIL"
                            print(f"score={res['best_score']:.6f}  {res['time_s']:.1f}s  "
                                  f"iters={res['n_iters']}  [{ok}]")
                        results[(N, D, opt_name, acquisition, kernel)] = trials

    # Summary table
    print(f"\n{'='*160}")
    print(f"{'BENCHMARK SUMMARY':^160}")
    print(f"{'='*160}")
    print(f"{'N':>3} {'D':>3} {'optimizer':>10} {'acq':>6} {'kernel':>13} {'#circ':>7} "
          f"{'best':>10} {'median':>10} {'worst':>10} "
          f"{'mean_t':>8} {'success':>9} {'mean_evals':>10} "
          f"{'surv10%':>8} {'surv25%':>8} {'surv50%':>8}")
    print("-" * 160)

    for (N, D, opt_name, acquisition, kernel), trials in sorted(results.items()):
        scores = [t['best_score'] for t in trials]
        times = [t['time_s'] for t in trials]
        n_success = sum(t['success'] for t in trials)
        n_circ = trials[0]['n_circuits']

        succ_evals = [t['evals_to_success'] for t in trials if t['evals_to_success'] is not None]
        mean_evals = np.mean(succ_evals) if succ_evals else float('nan')

        k10 = max(1, int(n_circ * 0.10))
        k25 = max(1, int(n_circ * 0.25))
        k50 = max(1, int(n_circ * 0.50))
        s10 = survival_at(trials, k10)
        s25 = survival_at(trials, k25)
        s50 = survival_at(trials, k50)

        print(f"{N:>3} {D:>3} {opt_name:>10} {acquisition:>6} {kernel:>13} {n_circ:>7} "
              f"{min(scores):>10.6f} {float(np.median(scores)):>10.6f} "
              f"{max(scores):>10.6f} "
              f"{np.mean(times):>7.1f}s "
              f"{n_success:>4}/{len(trials):<4} "
              f"{mean_evals:>10.1f} "
              f"{s10:>8.2f} {s25:>8.2f} {s50:>8.2f}")

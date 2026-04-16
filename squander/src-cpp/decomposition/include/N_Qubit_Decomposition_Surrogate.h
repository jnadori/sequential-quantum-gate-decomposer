/*
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*! \file N_Qubit_Decomposition_Surrogate.h
    \brief Header file for surrogate-assisted evolutionary search for circuit decomposition.
    Ports the Python SurSearch algorithm to C++ for performance.
*/

#ifndef N_Qubit_Decomposition_Surrogate_H
#define N_Qubit_Decomposition_Surrogate_H

#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "N_Qubit_Decomposition_Tree_Search.h"
#include "GrayCode.h"
#include "GrayCodeHash.h"
#include "n_aryGrayCodeCounter.h"
#include "GPRegressor.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// LAPACKE declarations and GrayCodeSet/Map aliases live in GPRegressor.h,
// which is included above.


// ---------------------------------------------------------------------------
// N_Qubit_Decomposition_Surrogate
// ---------------------------------------------------------------------------

class N_Qubit_Decomposition_Surrogate : public Optimization_Interface {

protected:
    /// The maximal number of two-qubit gates (circuit depth)
    int level_limit;

    /// Qubit connectivity topology
    std::vector<matrix_base<int>> topology;

    /// Target and control qubit indices for each topology edge
    matrix_base<int> possible_target_qbits;
    matrix_base<int> possible_control_qbits;

    // Precomputed topology data for validation
    std::vector<int> edge_masks;                   // bitmask per topology edge
    std::vector<std::pair<int,int>> thresholds;    // subspace check thresholds (n, j)

    // Precomputed edge neighbor map: edge_neighbors[e] = edges sharing at least one qubit with e
    std::vector<std::vector<int>> edge_neighbors;

    // Gate-based tokenization
    // Token layout: [0, n_1q_tokens) = 1q gates, [n_1q_tokens, n_tokens) = directed CNOT gates
    // For 1q token t: qubit = t % qbit_num, gate_type = gate_1q_types[t / qbit_num]
    // For CNOT token t: directed_idx = t - n_1q_tokens
    //   target = cnot_target_qbits[directed_idx], control = cnot_control_qbits[directed_idx]
    bool gate_based_mode;                          // config flag, default false
    int n_tokens;                                  // gate_based: n_1q_tokens + n_directed_cnots; else: n_edges
    int n_1q_tokens;                               // gate_based: n_1q_types * qbit_num; else: 0
    int n_directed_cnots;                          // gate_based: 2 * n_edges (both directions); else: 0
    int n_1q_types;                                // number of distinct 1q gate types in alphabet
    std::vector<gate_type> gate_1q_types;          // the 1q gate types in the alphabet
    std::vector<int> gate_1q_param_counts;         // parameter count for each 1q gate type
    std::vector<int> cnot_target_qbits;             // target qubit per directed CNOT
    std::vector<int> cnot_control_qbits;            // control qubit per directed CNOT
    std::vector<int> cnot_undirected_edge;          // undirected edge index per directed CNOT (for OSR)
    std::vector<int> token_masks;                  // bitmask per token (1 bit for 1q, 2 for CNOT)
    std::vector<std::vector<int>> token_neighbors; // tokens sharing at least one qubit

    // OSR data (using existing C++ infrastructure) — disabled when has_custom_topology
    bool has_custom_topology;
    int config_D_start;  // user-specified D_start (-1 = use OSR)
    std::vector<std::vector<int>> osr_cuts;
    std::vector<int> osr_cut_bounds;
    std::vector<std::vector<int>> osr_cut_crossing_edges;
    int osr_D_min;

    // Search state
    double kappa;
    double best_score;
    GrayCode best_circuit;
    Matrix_real best_params;

    // Search config
    double tolerance;
    int max_iters;
    int patience;
    int X0_size;
    int candidates_per_iter;
    int tournament_size;
    int block_size;
    double local_search_fraction;
    int max_local_steps;
    int local_search_positions;   // max positions to sample per local search step (0 = all)
    int local_search_gp_subset;   // training subset size for lightweight GP in local search (0 = full)
    int local_search_max_neighbors;  // max neighbors evaluated per local search step (0 = unlimited)
    int local_search_patience;       // stop local search after this many steps with marginal improvement
    double local_search_min_improvement; // relative improvement threshold for patience check
    int eval_budget_base;
    double d_penalty;
    int enum_threshold;
    int gp_max_train;  // max training points for GP (0 = unlimited)
    double gp_score_ratio;  // fraction of gp_max_train filled by best-by-score (rest: diversity)
    int d_seed_budget;  // max D-1 circuits seeded into GP at D transition (default 50)

    // Per-D search config
    int window_patience;
    int window_max_iters;

    // Best-so-far plateau stagnation detection
    int stagnation_window;              // lookback window for best-so-far plateau check (default 5)
    double stagnation_improvement_frac; // required relative improvement in best (default 0.01)

    // Adaptive kappa (exploration-exploitation scheduling)
    bool adaptive_kappa;
    double kappa_decay_rate;
    double kappa_stagnation_boost;

    // Position-guided mutations
    double position_guided_fraction;
    double position_lambda;
    double position_temperature;  // softmax temperature override; -1.0 = use Quarl formula

    // Weisfeiler-Lehman kernel config
    int wl_iterations;    // WL refinement rounds (default 3)


    // Timing
    double decompose_time;
    int decompose_count;
    double total_search_time;

public:
    /// Nullary constructor
    N_Qubit_Decomposition_Surrogate();

    /// Constructor with topology
    N_Qubit_Decomposition_Surrogate(Matrix Umtx_in, int qbit_num_in,
        std::vector<matrix_base<int>> topology_in,
        std::map<std::string, Config_Element>& config,
        int accelerator_num = 0);

    /// Constructor without topology (full connectivity)
    N_Qubit_Decomposition_Surrogate(Matrix Umtx_in, int qbit_num_in,
        std::map<std::string, Config_Element>& config,
        int accelerator_num = 0);

    virtual ~N_Qubit_Decomposition_Surrogate();

    /// Main entry point — called from Python via Start_Decomposition
    virtual void start_decomposition();

    /// Determine gate structure (required by base class)
    virtual Gates_block* determine_gate_structure(Matrix_real& optimized_parameters_mtx);

    /// Set the unitary matrix for decomposition
    void set_unitary(Matrix& Umtx_new);

    // ---- Search methods ----

    /// Surrogate-assisted evolutionary search for a single D
    void search_over_D_evolve(int D, const std::string& log_file);

    /// Cross-D surrogate search from D_min to D_max (bottom-up)
    void search_over_D_range(int D_min, int D_max, const std::string& log_file);

    /// Top-down compression from D_start down to D_min
    void compress_over_D_range(int D_start, int D_min, const std::string& log_file,
                               const GrayCode& initial_skeleton,
                               Gates_block* imported_gate_structure,
                               const Matrix_real& imported_params);

    /// Extract CNOT/CROT skeleton from imported gate structure as a GrayCode
    GrayCode extract_skeleton_from_gates();

    /// Result codes for per-window search
    enum WindowResult { WINDOW_SUCCESS, WINDOW_STAGNATION, WINDOW_BUDGET };

    /// Run focused surrogate search within a D window [win_lo, win_hi].
    /// Uses shared WLKernelCache/seen/X/y/all_params, fresh GP per window.
    WindowResult run_window_search(int win_lo, int win_hi,
        WLKernelCache& cache, GrayCodeSet& seen,
        std::vector<GrayCode>& X, std::vector<double>& y,
        std::vector<Matrix_real>& all_params,
        const std::string& log_file,
        int patience_override = -1);

    // ---- Circuit evaluation ----

    /// Result of a single decompose() call, used for parallel batching
    struct DecompResult {
        double score;
        Matrix_real params;
        double elapsed;
    };

    /// Decompose a circuit: build gate structure, optimize with random init, return (score, params)
    std::pair<double, Matrix_real> decompose(const GrayCode& circuit);

    /// Thread-safe decompose variant that takes an external RNG
    virtual std::pair<double, Matrix_real> decompose_with_rng(const GrayCode& circuit, std::mt19937& local_gen);

    /// Thread-safe decompose using provided initial parameters instead of random
    std::pair<double, Matrix_real> decompose_with_initial_params(const GrayCode& circuit, const Matrix_real& initial_params);

    /// Parallel decompose a batch of circuits using TBB (respects 'parallel' config)
    void parallel_decompose_batch(const std::vector<GrayCode>& circuits, std::vector<DecompResult>& results);

    /// Parallel decompose with warm-start initial params (one per circuit)
    void parallel_decompose_batch_with_init(
        const std::vector<GrayCode>& circuits,
        const std::vector<Matrix_real>& init_params,
        std::vector<DecompResult>& results);

    /// Compute warm-start params for a D-length circuit obtained by removing gate at
    /// position 'pos' from a D+1 circuit with optimized params 'parent_params'.
    /// Each gate contributes 6 params; finalizing layer contributes 3*qbit_num params.
    Matrix_real excise_gate_params(const Matrix_real& parent_params, int D_plus1, int pos);

    // ---- Gate structure building (following Tree_Search pattern) ----

    virtual Gates_block* construct_gate_structure(const GrayCode& gcode, bool finalize = true);
    void add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit);
    void add_single_qubit_gate(Gates_block* gate_structure, int target_qbit, gate_type gtype = U3_OPERATION);
    // Bring base class add_finalyzing_layer into scope to avoid hiding
    using Optimization_Interface::add_finalyzing_layer;
    void add_finalyzing_layer(Gates_block* gate_structure);

    // ---- Token helpers (gate-based mode) ----

    /// Sort key for canonical ordering: (type, qubit1, qubit2)
    /// U3 tokens: (0, qubit, -1). CNOT tokens: (1, target, control).
    virtual std::tuple<int,int,int> token_sort_key(int token) const;

    // ---- Incremental canonical DAG for point mutations ----

    /// Cached DAG structure for incremental canonical form updates
    struct CanonicalDAG {
        std::vector<std::vector<int>> adj;   // adjacency list (n elements)
        std::vector<int> in_degree;          // in-degree per position
        std::vector<int> masks;              // bitmask per position
        int n;                               // sequence length
    };

    /// Build a CanonicalDAG from a sequence. O(D^2).
    CanonicalDAG build_canonical_dag(const GrayCode& seq);

    /// Run topological sort on an existing DAG. O(D log D).
    /// The seq must reflect the current token at each position.
    GrayCode canonical_form_from_dag(const CanonicalDAG& dag, const GrayCode& seq);

    /// Update DAG in-place for a single position change. O(D).
    /// Modifies adj, in_degree, and masks for the new token at pos.
    void update_dag_point_mutation(CanonicalDAG& dag, int pos, int old_token, int new_token);

    /// Canonicalize and validate a point mutation using cached DAG.
    /// Applies mutation, computes canonical form, restores DAG, validates.
    GrayCode canonicalize_and_validate_from_dag(
        CanonicalDAG& dag, const GrayCode& seq, int pos, int new_token);

    // ---- Validation and canonicalization ----

    /// Canonical form via dependency DAG topological sort (bitmask version)
    GrayCode canonical_form(const GrayCode& seq);

    /// Canonicalize and validate: canonical form + subspace check + OSR feasibility
    /// Returns empty GrayCode on failure.
    GrayCode canonicalize_and_validate(const GrayCode& seq);

    /// Check if position creates a subspace violation
    bool check_new_position(const int* window_masks, int pos);

    /// Check OSR feasibility of a circuit
    virtual bool check_osr_feasibility(const GrayCode& circuit);

    // ---- Enumeration ----

    /// Enumerate all valid D-length circuits (for small spaces)
    std::vector<GrayCode> enumerate_circuits(int D);

    // ---- Evolutionary operators ----

    GrayCode generate_valid_sequence(int D);
    GrayCode mutate_point(const GrayCode& seq);
    GrayCode mutate_swap(const GrayCode& seq);
    GrayCode mutate_block(const GrayCode& seq, int blk_size);
    GrayCode mutate_transplant(const GrayCode& recipient, const GrayCode& donor, int blk_size);
    GrayCode mutate_block_regenerate(const GrayCode& seq, int blk_size);
    GrayCode crossover_uniform(const GrayCode& seq1, const GrayCode& seq2);
    GrayCode mutate_grow(const GrayCode& seq, int D_max);
    GrayCode mutate_shrink(const GrayCode& seq, int D_min);
    GrayCode mutate_point_guided(const GrayCode& seq, WLKernelCache& cache,
                                 GPRegressor& gp, double scale);

    // P2: Quantum-aware gate-rewriting mutations
    /// Commute adjacent gates that act on disjoint qubit sets
    GrayCode mutate_commute(const GrayCode& seq);
    /// Cancel adjacent identical CNOT pairs (CX·CX = I)
    GrayCode mutate_cancel_pairs(const GrayCode& seq);
    /// Insert identity pair (g·g) at random position for structural escape
    GrayCode mutate_insert_identity(const GrayCode& seq);

    /// Greedy local search on LCB acquisition
    GrayCode local_search_acq(const GrayCode& start, WLKernelCache& cache,
                              GPRegressor& gp, double scale,
                              int D_min_local = -1, int D_max_local = -1,
                              int* steps_out = nullptr);

    /// Generate hybrid candidate set (local search + evolutionary)
    /// train_y_norm: normalized training targets (for lightweight GP proxy), may be nullptr
    void generate_candidates(const std::vector<GrayCode>& population,
                             const double* scores, int n_pop,
                             int n_candidates,
                             WLKernelCache& cache, GPRegressor& gp, double scale,
                             const double* train_y_norm,
                             GrayCodeSet& seen,
                             std::vector<GrayCode>& candidates_out,
                             int& n_local_out, double& avg_steps_out,
                             int D_min_gen = -1, int D_max_gen = -1,
                             double force_random_fraction = 0.0);

    // ---- Acquisition ----

    double lcb(double mu, double std_val) const { return mu - kappa * std_val; }
};


#endif // N_Qubit_Decomposition_Surrogate_H

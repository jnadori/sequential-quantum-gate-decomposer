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

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif

#ifdef __cplusplus
extern "C"
{
#endif
int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs,
                  double* A, int LDA, double* B, int LDB);
int LAPACKE_dpotrf(int matrix_layout, char uplo, int n,
                   double* A, int LDA);
int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag,
                   int n, int nrhs, const double* A, int LDA,
                   double* B, int LDB);
void cblas_dgemm(int Order, int TransA, int TransB,
                 int M, int N, int K,
                 double alpha, const double* A, int lda,
                 const double* B, int ldb,
                 double beta, double* C, int ldc);
#ifdef __cplusplus
}
#endif


// ---------------------------------------------------------------------------
// GrayCode hashing for use in unordered containers
// (GrayCodeHash is defined in GrayCodeHash.h as GrayCodeHash_base<int>)
// ---------------------------------------------------------------------------

using GrayCodeSet = std::unordered_set<GrayCode, GrayCodeHash>;
using GrayCodeMap = std::unordered_map<GrayCode, int, GrayCodeHash>;


// ---------------------------------------------------------------------------
// SSKCache — Subsequence String Kernel cache with incremental registration
// ---------------------------------------------------------------------------

class SSKCache {

public:
    double gap_decay;
    double match_sq;
    int order;

    // Registered circuits
    std::vector<GrayCode> circuits;
    GrayCodeMap circuit_to_idx;

    // Gap decay matrices keyed by sequence length
    std::map<int, std::vector<double>> D_matrices;

    // Dense normalized kernel matrix and raw diagonal values
    std::vector<double> K_norm;   // capacity * capacity, row-major
    std::vector<double> raw_diags;
    int size_;
    int capacity_;

    SSKCache();
    SSKCache(double gap_decay_in, double match_decay_in, int order_in);

    /// Get or compute the gap decay matrix for length n (n x n, row-major)
    const std::vector<double>& get_D_matrix(int n);

    /// Register a circuit, computing kernel values against all existing circuits.
    /// Returns the index of the circuit.
    int register_circuit(const GrayCode& circuit);

    /// Compute raw SSK values for P pairs of sequences.
    /// ci: P*n1 ints (row-major), cj: P*n2 ints (row-major)
    /// Writes P values into raw_out.
    void batch_ssk_raw(const int* ci, const int* cj, int P, int n1, int n2,
                       double* raw_out);
    /// Compute raw SSK for a single pair of sequences.
    double single_ssk_raw(const int* a, int n1, const int* b, int n2);

    /// Compute cross-kernel: scale * K_norm_cross of shape (n_registered, n_candidates).
    /// Result is row-major in result_out (must be pre-allocated).
    void compute_cross_kernel(const std::vector<GrayCode>& candidates,
                              double scale, double* result_out);

    /// Compute cross-kernel for a subset of registered circuits only.
    /// reg_subset: indices into the registered circuits array.
    /// Result is row-major of shape (n_subset, n_candidates).
    void compute_cross_kernel_subset(const std::vector<GrayCode>& candidates,
                                     double scale,
                                     const std::vector<int>& reg_subset,
                                     double* result_out);

    /// Compute normalized SSK between two arbitrary circuits.
    double kernel_between(const GrayCode& a, const GrayCode& b);

    /// Extract sub-matrix: scale * K_norm[idx1, idx2]
    void kernel_matrix(const int* idx1, int n1, const int* idx2, int n2,
                       double scale, double* result_out);

private:
    void grow_capacity();
};


// ---------------------------------------------------------------------------
// GPRegressor — Gaussian Process with SSK kernel and Cholesky inference
// ---------------------------------------------------------------------------

class GPRegressor {

public:
    double log_scale;    // kernel hyperparameter (log of scale)
    double noise;
    double jitter;

    // Learned during fit:
    std::vector<double> L_data;      // Cholesky factor, n_train x n_train row-major
    std::vector<double> alpha_data;  // K^{-1} y, length n_train
    std::vector<int> train_indices_; // indices into SSKCache for each training point
    int n_train;

    GPRegressor();

    /// Fit GP on training data. train_indices index into ssk_cache.
    void fit(SSKCache& cache, const int* train_indices, int n,
             const double* y);

    /// Incremental Cholesky update: add new training points without full refactorization.
    /// old_n is the number of points already in the Cholesky factor L_data.
    /// The new points are at indices train_indices[old_n..n-1].
    /// y contains ALL n training targets (old + new).
    void fit_incremental(SSKCache& cache, const int* train_indices, int n,
                         int old_n, const double* y);

    /// Predict mean and std at candidate circuits.
    /// mu_out and std_out must be pre-allocated with n_candidates elements.
    void predict(SSKCache& cache, const std::vector<GrayCode>& candidates,
                 double* mu_out, double* std_out);

    /// Compute log marginal likelihood for given hyperparameters.
    double log_marginal_likelihood(SSKCache& cache, const int* train_indices,
                                   int n, const double* y,
                                   double test_log_scale, double test_noise);

    /// Optimize hyperparameters (log_scale, noise) via grid search.
    void optimize_hyperparameters(SSKCache& cache, const int* train_indices,
                                  int n, const double* y,
                                  int n_restarts,
                                  const std::pair<double,double>& scale_bounds,
                                  const std::pair<double,double>& noise_bounds);

    double get_scale() const { return std::exp(log_scale); }
};


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

    // OSR data (using existing C++ infrastructure)
    std::vector<std::vector<int>> osr_cuts;
    std::vector<int> osr_cut_bounds;
    std::vector<std::vector<int>> osr_cut_crossing_edges;
    int osr_D_min;

    // Search state
    double kappa;
    double best_score;
    GrayCode best_circuit;
    Matrix_real best_params;

    // SSK kernel config
    double sur_gap_decay;
    double sur_match_decay;
    int sur_ssk_order;

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
    int n_thompson_samples;
    double diversity_thresh;
    double d_penalty;
    int enum_threshold;
    int gp_max_train;  // max training points for GP (0 = unlimited)
    int d_seed_budget;  // max D-1 circuits seeded into GP at D transition (default 50)

    // D-window config
    int d_window_width;
    int base_window_width;
    int window_patience;
    int window_max_iters;
    int max_consecutive_stagnations;

    // Rollback-phase overrides (used during narrowing after solution found)
    double rb_kappa;
    int rb_window_patience;
    int rb_window_max_iters;
    int rb_candidates_per_iter;
    int rb_n_thompson_samples;
    double rb_local_search_fraction;
    int rb_max_local_steps;

    // Random baseline mode (bypasses GP/Thompson sampling)
    bool use_random_candidates;

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

    /// Cross-D surrogate search from D_min to D_max (uses sliding D window)
    void search_over_D_range(int D_min, int D_max, const std::string& log_file);

    /// Result codes for per-window search
    enum WindowResult { WINDOW_SUCCESS, WINDOW_STAGNATION, WINDOW_BUDGET };

    /// Run focused surrogate search within a D window [win_lo, win_hi].
    /// Uses shared SSKCache/seen/X/y/all_params, fresh GP per window.
    WindowResult run_window_search(int win_lo, int win_hi,
        SSKCache& cache, GrayCodeSet& seen,
        std::vector<GrayCode>& X, std::vector<double>& y,
        std::vector<Matrix_real>& all_params,
        const std::string& log_file);

    // ---- Circuit evaluation ----

    /// Decompose a circuit: build gate structure, optimize with random init, return (score, params)
    std::pair<double, Matrix_real> decompose(const GrayCode& circuit);

    // ---- Gate structure building (following Tree_Search pattern) ----

    Gates_block* construct_gate_structure(const GrayCode& gcode, bool finalize = true);
    void add_two_qubit_block(Gates_block* gate_structure, int target_qbit, int control_qbit);
    // Bring base class add_finalyzing_layer into scope to avoid hiding
    using Optimization_Interface::add_finalyzing_layer;
    void add_finalyzing_layer(Gates_block* gate_structure);

    // ---- Validation and canonicalization ----

    /// Canonical form via dependency DAG topological sort (bitmask version)
    GrayCode canonical_form(const GrayCode& seq);

    /// Canonicalize and validate: canonical form + subspace check + OSR feasibility
    /// Returns empty GrayCode on failure.
    GrayCode canonicalize_and_validate(const GrayCode& seq);

    /// Check if position creates a subspace violation
    bool check_new_position(const int* window_masks, int pos);

    /// Check OSR feasibility of a circuit
    bool check_osr_feasibility(const GrayCode& circuit);

    // ---- Enumeration ----

    /// Enumerate all valid D-length circuits (for small spaces)
    std::vector<GrayCode> enumerate_circuits(int D);

    // ---- Evolutionary operators ----

    GrayCode generate_valid_sequence(int D);
    GrayCode mutate_point(const GrayCode& seq);
    GrayCode mutate_swap(const GrayCode& seq);
    GrayCode mutate_block(const GrayCode& seq, int blk_size);
    GrayCode crossover_uniform(const GrayCode& seq1, const GrayCode& seq2);
    GrayCode mutate_grow(const GrayCode& seq, int D_max);
    GrayCode mutate_shrink(const GrayCode& seq, int D_min);

    /// Greedy local search on LCB acquisition
    GrayCode local_search_acq(const GrayCode& start, SSKCache& cache,
                              GPRegressor& gp, double scale,
                              GrayCodeSet& seen,
                              int D_min_local = -1, int D_max_local = -1,
                              int* steps_out = nullptr);

    /// Generate hybrid candidate set (local search + evolutionary)
    void generate_candidates(const std::vector<GrayCode>& population,
                             const double* scores, int n_pop,
                             int n_candidates,
                             SSKCache& cache, GPRegressor& gp, double scale,
                             GrayCodeSet& seen,
                             std::vector<GrayCode>& candidates_out,
                             int& n_local_out, double& avg_steps_out,
                             int D_min_gen = -1, int D_max_gen = -1);

    // ---- Acquisition ----

    double lcb(double mu, double std_val) const { return mu - kappa * std_val; }

    // ---- Standalone SSK Gram matrix ----

    /// Compute full SSK Gram matrix for a list of circuits (variable-length)
    static void ssk_gram_matrix(const std::vector<GrayCode>& circuits,
                                double gap_decay, double match_decay, int order,
                                double* K_out);
};


#endif // N_Qubit_Decomposition_Surrogate_H

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

/*! \file GPRegressor.h
    \brief Header file for surrogate-assisted evolutionary search for circuit decomposition.
    Ports the Python SurSearch algorithm to C++ for performance.
*/

#ifndef GPRegressor_H
#define GPRegressor_H

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
#include <tuple>
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
int LAPACKE_dtrtri(int matrix_layout, char uplo, char diag, int n,
                   double* A, int LDA);
int LAPACKE_dsteqr(int matrix_layout, char compz, int n,
                   double* d, double* e, double* z, int ldz);
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
    std::vector<double> inv_L_diag;  // 1.0 / L[i,i] for diagonal variance approximation
    int n_train;

    // LOVE (Lanczos Variance Estimates) precomputed state
    std::vector<double> R_love;  // n_train x love_rank, row-major. K^{-1} ~ R*R^T
    int love_rank;               // Lanczos iterations (0 = disabled)
    bool love_valid;             // true if R_love is consistent with current L_data

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

    /// Compute negative LML and analytical gradient jointly.
    /// Returns NLL; fills grad_out[2] with {dNLL/d(log_scale), dNLL/d(log_noise)}.
    double log_marginal_likelihood_with_grad(
        SSKCache& cache, const int* train_indices, int n, const double* y,
        double test_log_scale, double test_noise, double* grad_out);

    /// Optimize hyperparameters (log_scale, noise) via grid search.
    void optimize_hyperparameters(SSKCache& cache, const int* train_indices,
                                  int n, const double* y,
                                  int n_restarts,
                                  const std::pair<double,double>& scale_bounds,
                                  const std::pair<double,double>& noise_bounds);

    /// Precompute LOVE (Lanczos Variance Estimates) for fast variance predictions.
    /// Must be called after fit() or fit_incremental(). Builds a low-rank factor R
    /// such that K^{-1} ~ R*R^T. Variance prediction becomes O(n*r) per candidate.
    void compute_love(int rank);

    double get_scale() const { return std::exp(log_scale); }
};
#endif // N_Qubit_Decomposition_Surrogate_H

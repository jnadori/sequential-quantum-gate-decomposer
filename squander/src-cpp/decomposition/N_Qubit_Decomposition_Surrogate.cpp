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

/*! \file N_Qubit_Decomposition_Surrogate.cpp
    \brief Surrogate-assisted evolutionary search for circuit decomposition.
    C++ port of the Python SurSearch algorithm.
*/

#include "N_Qubit_Decomposition_Surrogate.h"
#include "BFGS_Powell.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>
#include "tbb/tbb.h"


#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#if BLAS == 1
extern "C" void MKL_Set_Num_Threads(int);
#elif BLAS == 2
extern "C" void openblas_set_num_threads(int);
#endif

static void force_single_thread_blas_lapack() {
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);
    setenv("MKL_NUM_THREADS", "1", 1);
    setenv("MKL_DYNAMIC", "FALSE", 1);
    setenv("OPENBLAS_NUM_THREADS", "1", 1);

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif

#if BLAS == 1
    MKL_Set_Num_Threads(1);
#elif BLAS == 2
    openblas_set_num_threads(1);
#endif
}
// ============================================================================
// Canonical prefix check — adapted from Tree_Search.cpp
// ============================================================================

static int canonical_prefix_ok_surrogate(const GrayCode& path,
                                         const std::vector<matrix_base<int>>& topology) {
    const int m = static_cast<int>(path.size());
    if (m <= 1) return -1;

    std::vector<std::vector<int>> succ(m);
    std::vector<int> indeg(m, 0);
    std::unordered_map<int, int> last_on;
    last_on.reserve(m * 2);

    for (int k = 0; k < m; ++k) {
        const int a = topology[path[k]][0];
        const int b = topology[path[k]][1];
        for (int q : {a, b}) {
            auto it = last_on.find(q);
            if (it != last_on.end()) {
                succ[it->second].push_back(k);
                ++indeg[k];
                it->second = k;
            } else {
                last_on.emplace(q, k);
            }
        }
    }

    struct Node {
        std::pair<int, int> p;
        int idx;
    };
    struct Cmp {
        bool operator()(const Node& a, const Node& b) const {
            if (a.p != b.p) return a.p > b.p;
            return a.idx > b.idx;
        }
    };
    std::priority_queue<Node, std::vector<Node>, Cmp> pq;
    for (int k = 0; k < m; ++k)
        if (indeg[k] == 0)
            pq.push(Node{std::make_pair(topology[path[k]][0], topology[path[k]][1]), k});

    for (int pos = 0; pos < m; ++pos) {
        if (pq.empty()) return pos;
        Node u = pq.top();
        pq.pop();
        if (u.idx != pos) return pos;
        for (int v : succ[u.idx]) {
            if (--indeg[v] == 0)
                pq.push(Node{std::make_pair(topology[path[v]][0], topology[path[v]][1]), v});
        }
    }
    return -1;
}


// ============================================================================
// SSKCache implementation
// ============================================================================

SSKCache::SSKCache() : gap_decay(0.8), match_sq(0.64), order(3),
                       size_(0), capacity_(64) {
    K_norm.resize(capacity_ * capacity_, 0.0);
    raw_diags.resize(capacity_, 0.0);
}

SSKCache::SSKCache(double gap_decay_in, double match_decay_in, int order_in)
    : gap_decay(gap_decay_in), match_sq(match_decay_in * match_decay_in),
      order(order_in), size_(0), capacity_(64) {
    K_norm.resize(capacity_ * capacity_, 0.0);
    raw_diags.resize(capacity_, 0.0);
}

void SSKCache::grow_capacity() {
    int new_cap = capacity_ * 2;
    std::vector<double> new_K(new_cap * new_cap, 0.0);
    for (int i = 0; i < size_; ++i)
        std::memcpy(&new_K[i * new_cap], &K_norm[i * capacity_],
                     size_ * sizeof(double));
    K_norm = std::move(new_K);
    raw_diags.resize(new_cap, 0.0);
    capacity_ = new_cap;
}

const std::vector<double>& SSKCache::get_D_matrix(int n) {
    auto it = D_matrices.find(n);
    if (it != D_matrices.end()) return it->second;

    std::vector<double> D(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            D[i * n + j] = std::pow(gap_decay, j - i - 1);
    D_matrices[n] = std::move(D);
    return D_matrices[n];
}

void SSKCache::batch_ssk_raw(const int* ci, const int* cj, int P, int n1,
                             int n2, double* raw_out) {
    const double g = gap_decay;
    const double msq = match_sq;

    std::vector<double> kp(n1 * n2);
    std::vector<double> tmp(n1 * n2);

    // Working arrays for loop-interchanged column recurrence
    std::vector<double> run_vec(n1);
    std::vector<double> mask(n1);       // branchless match mask per column step
    std::vector<double> kp_col(n1);     // extracted column of kp

    // Precompute match-position lists for b values
    std::vector<std::vector<int>> b_positions;

    for (int p = 0; p < P; ++p) {
        const int* a = ci + p * n1;
        const int* b = cj + p * n2;

        // Build match-position index for b
        int max_b = 0;
        for (int j = 0; j < n2; ++j)
            if (b[j] > max_b) max_b = b[j];
        b_positions.assign(max_b + 1, std::vector<int>());
        for (int j = 0; j < n2; ++j)
            b_positions[b[j]].push_back(j);

        // Kp = ones(n1, n2)
        std::fill(kp.begin(), kp.end(), 1.0);

        // First-order term: count matches via histogram
        // O(n1 + n2 + n_edges) instead of O(n1 * n2)
        double raw = 0.0;
        for (int i = 0; i < n1; ++i) {
            const int ai = a[i];
            if (ai <= max_b)
                raw += static_cast<double>(b_positions[ai].size());
        }
        raw *= msq;

        for (int iter = 1; iter < order; ++iter) {
            // tmp = match_sq * (S .* kp) @ D2 via column recurrence
            // Loop interchange: j-outer, i-inner for SIMD vectorization over i.
            // For each column j, run[i] is independent across rows, so the
            // compiler can vectorize the i-loop (contiguous run_vec access).

            // Initialize: run[i] = 0, tmp[:,0] = 0
            std::fill(run_vec.begin(), run_vec.end(), 0.0);
            for (int i = 0; i < n1; ++i)
                tmp[i * n2] = 0.0;

            for (int j = 1; j < n2; ++j) {
                const int bj_prev = b[j - 1];

                // Precompute branchless match mask: mask[i] = (a[i] == b[j-1]) ? 1.0 : 0.0
                // and extract column j-1 of kp into contiguous array.
                // Both are O(n1) with stride-n2 reads but enable vectorized core.
                for (int i = 0; i < n1; ++i) {
                    mask[i] = static_cast<double>(a[i] == bj_prev);
                    kp_col[i] = kp[i * n2 + j - 1];
                }

                // Core: vectorizable over i (all arrays contiguous)
                for (int i = 0; i < n1; ++i) {
                    run_vec[i] = kp_col[i] * mask[i] + g * run_vec[i];
                }

                // Scatter results to tmp[:,j]
                for (int i = 0; i < n1; ++i)
                    tmp[i * n2 + j] = msq * run_vec[i];
            }

            // kp = D1^T @ tmp  (row recurrence)
            // First row is zero.
            std::fill_n(kp.data(), n2, 0.0);

            double add = 0.0;
            for (int i = 1; i < n1; ++i) {
                const int ai = a[i];
                const double* tmp_prev = tmp.data() + (i - 1) * n2;
                const double* kp_prev = kp.data() + (i - 1) * n2;
                double* kp_row = kp.data() + i * n2;

                // Branch-free kp update (vectorizable over j, contiguous)
                for (int j = 0; j < n2; ++j)
                    kp_row[j] = tmp_prev[j] + g * kp_prev[j];

                // Sparse accumulation: only sum at matching positions
                if (ai <= max_b) {
                    for (int jp : b_positions[ai])
                        add += kp_row[jp];
                }
            }

            raw += msq * add;
        }

        raw_out[p] = raw;
    }
}
double SSKCache::single_ssk_raw(const int* a, int n1, const int* b, int n2) {
    double result;
    batch_ssk_raw(a, b, 1, n1, n2, &result);
    return result;
}

int SSKCache::register_circuit(const GrayCode& circuit) {
    auto it = circuit_to_idx.find(circuit);
    if (it != circuit_to_idx.end()) return it->second;

    int new_idx = size_;
    circuits.push_back(circuit.copy());
    circuit_to_idx[circuits.back()] = new_idx;

    int n_new = static_cast<int>(circuit.size());

    if (new_idx >= capacity_) grow_capacity();

    // Self-kernel
    double self_raw = single_ssk_raw(circuit.get_data(), n_new,
                                     circuit.get_data(), n_new);
    raw_diags[new_idx] = self_raw;

    // Kernel vs all existing circuits, grouped by length
    std::vector<double> raw_new(new_idx + 1, 0.0);
    raw_new[new_idx] = self_raw;

    if (new_idx > 0) {
        // Group existing by length
        std::map<int, std::vector<int>> len_groups;
        for (int i = 0; i < new_idx; ++i)
            len_groups[static_cast<int>(circuits[i].size())].push_back(i);

        for (std::map<int, std::vector<int>>::iterator it = len_groups.begin(); it != len_groups.end(); ++it) {
            int ulen = it->first;
            std::vector<int>& indices = it->second;
            int P = static_cast<int>(indices.size());
            std::vector<int> ci_flat(P * n_new);
            std::vector<int> cj_flat(P * ulen);

            for (int p = 0; p < P; ++p) {
                std::memcpy(&ci_flat[p * n_new], circuit.get_data(),
                            n_new * sizeof(int));
                std::memcpy(&cj_flat[p * ulen], circuits[indices[p]].get_data(),
                            ulen * sizeof(int));
            }

            std::vector<double> raw_batch(P);
            batch_ssk_raw(ci_flat.data(), cj_flat.data(), P, n_new, ulen,
                          raw_batch.data());

            for (int p = 0; p < P; ++p)
                raw_new[indices[p]] = raw_batch[p];
        }
    }

    // Normalize and store in K_norm
    double inv_diag_new = 1.0 / std::sqrt(std::max(raw_new[new_idx], 1e-24));
    for (int i = 0; i <= new_idx; ++i) {
        double inv_diag_i = 1.0 / std::sqrt(std::max(raw_diags[i], 1e-24));
        double norm_val = raw_new[i] * inv_diag_new * inv_diag_i;
        K_norm[new_idx * capacity_ + i] = norm_val;
        K_norm[i * capacity_ + new_idx] = norm_val;
    }

    size_++;
    return new_idx;
}

void SSKCache::compute_cross_kernel(const std::vector<GrayCode>& candidates,
                                    double scale, double* result_out) {
    int n_reg = size_;
    int n_cand = static_cast<int>(candidates.size());
    if (n_reg == 0 || n_cand == 0) return;

    // Group candidates by length
    std::map<int, std::vector<int>> cand_by_len;
    for (int i = 0; i < n_cand; ++i)
        cand_by_len[static_cast<int>(candidates[i].size())].push_back(i);

    // Group registered by length
    std::map<int, std::vector<int>> reg_by_len;
    for (int i = 0; i < n_reg; ++i)
        reg_by_len[static_cast<int>(circuits[i].size())].push_back(i);

    // Raw cross-kernel (n_reg x n_cand)
    std::vector<double> raw_cross(n_reg * n_cand, 0.0);

    for (std::map<int, std::vector<int>>::iterator cit = cand_by_len.begin(); cit != cand_by_len.end(); ++cit) {
        int cand_len = cit->first;
        std::vector<int>& cand_indices = cit->second;
        int n_cg = static_cast<int>(cand_indices.size());

        for (std::map<int, std::vector<int>>::iterator rit = reg_by_len.begin(); rit != reg_by_len.end(); ++rit) {
            int reg_len = rit->first;
            std::vector<int>& reg_indices = rit->second;
            int n_rg = static_cast<int>(reg_indices.size());

            // For each candidate, pair with all registered of this length
            int batch_limit = 10000;
            int cand_batch = std::max(1, batch_limit / std::max(n_rg, 1));

            for (int b_start = 0; b_start < n_cg; b_start += cand_batch) {
                int b_end = std::min(b_start + cand_batch, n_cg);
                int bs = b_end - b_start;
                int P = n_rg * bs;

                std::vector<int> ci_flat(P * reg_len);
                std::vector<int> cj_flat(P * cand_len);

                for (int r = 0; r < n_rg; ++r) {
                    for (int c = 0; c < bs; ++c) {
                        int idx = r * bs + c;
                        std::memcpy(&ci_flat[idx * reg_len],
                                    circuits[reg_indices[r]].get_data(),
                                    reg_len * sizeof(int));
                        std::memcpy(&cj_flat[idx * cand_len],
                                    candidates[cand_indices[b_start + c]].get_data(),
                                    cand_len * sizeof(int));
                    }
                }

                std::vector<double> raw_batch(P);
                batch_ssk_raw(ci_flat.data(), cj_flat.data(), P,
                              reg_len, cand_len, raw_batch.data());

                for (int r = 0; r < n_rg; ++r)
                    for (int c = 0; c < bs; ++c)
                        raw_cross[reg_indices[r] * n_cand + cand_indices[b_start + c]] =
                            raw_batch[r * bs + c];
            }
        }
    }

    // Compute candidate self-kernels
    std::vector<double> cand_self_raw(n_cand, 0.0);
    for (std::map<int, std::vector<int>>::iterator sit = cand_by_len.begin(); sit != cand_by_len.end(); ++sit) {
        int cand_len_s = sit->first;
        std::vector<int>& cand_indices_s = sit->second;
        int n_cg = static_cast<int>(cand_indices_s.size());
        std::vector<int> ci_flat(n_cg * cand_len_s);
        for (int i = 0; i < n_cg; ++i)
            std::memcpy(&ci_flat[i * cand_len_s],
                        candidates[cand_indices_s[i]].get_data(),
                        cand_len_s * sizeof(int));
        std::vector<double> self_batch(n_cg);
        batch_ssk_raw(ci_flat.data(), ci_flat.data(), n_cg,
                      cand_len_s, cand_len_s, self_batch.data());
        for (int i = 0; i < n_cg; ++i)
            cand_self_raw[cand_indices_s[i]] = self_batch[i];
    }

    // Normalize: result = scale * raw / (sqrt(diag_reg) * sqrt(diag_cand))
    // Precompute inverse-sqrt factors to avoid repeated sqrt+division
    std::vector<double> inv_dr(n_reg);
    for (int r = 0; r < n_reg; ++r)
        inv_dr[r] = 1.0 / std::sqrt(std::max(raw_diags[r], 1e-24));
    std::vector<double> inv_dc(n_cand);
    for (int c = 0; c < n_cand; ++c)
        inv_dc[c] = 1.0 / std::sqrt(std::max(cand_self_raw[c], 1e-24));

    for (int r = 0; r < n_reg; ++r) {
        for (int c = 0; c < n_cand; ++c) {
            result_out[r * n_cand + c] = scale * raw_cross[r * n_cand + c] * inv_dr[r] * inv_dc[c];
        }
    }
}

double SSKCache::kernel_between(const GrayCode& a, const GrayCode& b) {
    int n1 = static_cast<int>(a.size());
    int n2 = static_cast<int>(b.size());
    double raw_ab = single_ssk_raw(a.get_data(), n1, b.get_data(), n2);
    double raw_aa = single_ssk_raw(a.get_data(), n1, a.get_data(), n1);
    double raw_bb = single_ssk_raw(b.get_data(), n2, b.get_data(), n2);
    double denom = std::sqrt(std::max(raw_aa, 1e-24) * std::max(raw_bb, 1e-24));
    return raw_ab / denom;
}

void SSKCache::compute_cross_kernel_subset(
    const std::vector<GrayCode>& candidates, double scale,
    const std::vector<int>& reg_subset, double* result_out) {

    int n_sub = static_cast<int>(reg_subset.size());
    int n_cand = static_cast<int>(candidates.size());
    if (n_sub == 0 || n_cand == 0) return;

    // Map subset position (0..n_sub-1) -> registered index
    // Group subset positions by their circuit length
    std::map<int, std::vector<int>> sub_by_len; // len -> list of subset positions
    for (int s = 0; s < n_sub; ++s) {
        int reg_idx = reg_subset[s];
        int len = static_cast<int>(circuits[reg_idx].size());
        sub_by_len[len].push_back(s);
    }

    // Group candidates by length
    std::map<int, std::vector<int>> cand_by_len;
    for (int i = 0; i < n_cand; ++i)
        cand_by_len[static_cast<int>(candidates[i].size())].push_back(i);

    // Raw cross-kernel (n_sub x n_cand)
    std::vector<double> raw_cross(n_sub * n_cand, 0.0);

    for (std::map<int, std::vector<int>>::iterator cit = cand_by_len.begin();
         cit != cand_by_len.end(); ++cit) {
        int cand_len = cit->first;
        std::vector<int>& cand_indices = cit->second;
        int n_cg = static_cast<int>(cand_indices.size());

        for (std::map<int, std::vector<int>>::iterator rit = sub_by_len.begin();
             rit != sub_by_len.end(); ++rit) {
            int reg_len = rit->first;
            std::vector<int>& sub_positions = rit->second;
            int n_rg = static_cast<int>(sub_positions.size());

            int batch_limit = 10000;
            int cand_batch = std::max(1, batch_limit / std::max(n_rg, 1));

            for (int b_start = 0; b_start < n_cg; b_start += cand_batch) {
                int b_end = std::min(b_start + cand_batch, n_cg);
                int bs = b_end - b_start;
                int P = n_rg * bs;

                std::vector<int> ci_flat(P * reg_len);
                std::vector<int> cj_flat(P * cand_len);

                for (int r = 0; r < n_rg; ++r) {
                    int reg_idx = reg_subset[sub_positions[r]];
                    for (int c = 0; c < bs; ++c) {
                        int idx = r * bs + c;
                        std::memcpy(&ci_flat[idx * reg_len],
                                    circuits[reg_idx].get_data(),
                                    reg_len * sizeof(int));
                        std::memcpy(&cj_flat[idx * cand_len],
                                    candidates[cand_indices[b_start + c]].get_data(),
                                    cand_len * sizeof(int));
                    }
                }

                std::vector<double> raw_batch(P);
                batch_ssk_raw(ci_flat.data(), cj_flat.data(), P,
                              reg_len, cand_len, raw_batch.data());

                for (int r = 0; r < n_rg; ++r)
                    for (int c = 0; c < bs; ++c)
                        raw_cross[sub_positions[r] * n_cand + cand_indices[b_start + c]] =
                            raw_batch[r * bs + c];
            }
        }
    }

    // Compute candidate self-kernels
    std::vector<double> cand_self_raw(n_cand, 0.0);
    for (std::map<int, std::vector<int>>::iterator sit = cand_by_len.begin();
         sit != cand_by_len.end(); ++sit) {
        int cand_len_s = sit->first;
        std::vector<int>& cand_indices_s = sit->second;
        int n_cg = static_cast<int>(cand_indices_s.size());
        std::vector<int> ci_flat(n_cg * cand_len_s);
        for (int i = 0; i < n_cg; ++i)
            std::memcpy(&ci_flat[i * cand_len_s],
                        candidates[cand_indices_s[i]].get_data(),
                        cand_len_s * sizeof(int));
        std::vector<double> self_batch(n_cg);
        batch_ssk_raw(ci_flat.data(), ci_flat.data(), n_cg,
                      cand_len_s, cand_len_s, self_batch.data());
        for (int i = 0; i < n_cg; ++i)
            cand_self_raw[cand_indices_s[i]] = self_batch[i];
    }

    // Normalize: result = scale * raw / (sqrt(diag_reg) * sqrt(diag_cand))
    // Precompute inverse-sqrt factors to avoid repeated sqrt+division
    std::vector<double> inv_dr(n_sub);
    for (int s = 0; s < n_sub; ++s)
        inv_dr[s] = 1.0 / std::sqrt(std::max(raw_diags[reg_subset[s]], 1e-24));
    std::vector<double> inv_dc(n_cand);
    for (int c = 0; c < n_cand; ++c)
        inv_dc[c] = 1.0 / std::sqrt(std::max(cand_self_raw[c], 1e-24));

    for (int s = 0; s < n_sub; ++s) {
        for (int c = 0; c < n_cand; ++c) {
            result_out[s * n_cand + c] = scale * raw_cross[s * n_cand + c] * inv_dr[s] * inv_dc[c];
        }
    }
}

void SSKCache::kernel_matrix(const int* idx1, int n1, const int* idx2, int n2,
                             double scale, double* result_out) {
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            result_out[i * n2 + j] = scale * K_norm[idx1[i] * capacity_ + idx2[j]];
}


// ============================================================================
// GPRegressor implementation
// ============================================================================

GPRegressor::GPRegressor()
    : log_scale(0.0), noise(1e-2), jitter(1e-8), n_train(0) {}

void GPRegressor::fit(SSKCache& cache, const int* train_indices, int n,
                      const double* y) {
    n_train = n;
    train_indices_.assign(train_indices, train_indices + n);
    double scale = std::exp(log_scale);

    // Build kernel matrix K = scale * K_norm[train, train] + (noise + jitter) * I
    L_data.resize(n * n);
    cache.kernel_matrix(train_indices, n, train_indices, n, scale, L_data.data());
    for (int i = 0; i < n; ++i)
        L_data[i * n + i] += noise + jitter;

    // alpha = K^{-1} y via Cholesky
    alpha_data.assign(y, y + n);

    // LAPACKE_dposv solves A*X = B where A is symmetric positive definite
    // It overwrites A with Cholesky factor L and B with solution X
    int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', n, 1,
                             L_data.data(), n, alpha_data.data(), 1);
    if (info != 0) {
        // Fallback: add more jitter
        cache.kernel_matrix(train_indices, n, train_indices, n, scale, L_data.data());
        double extra_jitter = jitter;
        for (int attempt = 0; attempt < 10; ++attempt) {
            extra_jitter *= 10;
            for (int i = 0; i < n; ++i)
                L_data[i * n + i] += extra_jitter;
            alpha_data.assign(y, y + n);
            info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'L', n, 1,
                                 L_data.data(), n, alpha_data.data(), 1);
            if (info == 0) break;
            cache.kernel_matrix(train_indices, n, train_indices, n, scale, L_data.data());
            for (int i = 0; i < n; ++i)
                L_data[i * n + i] += noise;
        }
    }
    // After dposv, L_data contains the lower Cholesky factor
    inv_L_diag.resize(n);
    for (int i = 0; i < n; ++i)
        inv_L_diag[i] = 1.0 / L_data[i * n + i];
}

void GPRegressor::fit_incremental(SSKCache& cache, const int* train_indices,
                                  int n, int old_n, const double* y) {
    // Incremental Cholesky update: extend L from old_n to n points.
    // L_data already has the old_n x old_n Cholesky factor from previous fit.
    // For each new point k (old_n <= k < n), we compute:
    //   L[k, 0:k] by forward-solving L[0:k,0:k] @ l = k_new[0:k]
    //   L[k, k] = sqrt(k_new[k,k] + noise + jitter - ||l||^2)
    // Then recompute alpha = L^T \ (L \ y) for all n points.

    if (old_n <= 0 || old_n >= n) {
        // Fall back to full fit
        fit(cache, train_indices, n, y);
        return;
    }

    double scale = std::exp(log_scale);

    // Expand L_data from old_n x old_n to n x n (row-major)
    std::vector<double> L_new(n * n, 0.0);
    for (int i = 0; i < old_n; ++i)
        for (int j = 0; j <= i; ++j)
            L_new[i * n + j] = L_data[i * old_n + j];

    // For each new point k
    for (int k = old_n; k < n; ++k) {
        // Compute cross-kernel row: K[k, 0:k] = scale * K_norm[train[k], train[0:k]]
        std::vector<double> k_row(k);
        int idx_k = train_indices[k];
        for (int j = 0; j < k; ++j) {
            int idx_j = train_indices[j];
            // Access from K_norm in cache
            k_row[j] = scale * cache.K_norm[idx_k * cache.capacity_ + idx_j];
        }

        // Forward solve: L[0:k, 0:k] @ l = k_row  (lower triangular solve)
        // Do manual forward substitution (small, in-place)
        std::vector<double> l(k);
        for (int i = 0; i < k; ++i) {
            double s = k_row[i];
            for (int j = 0; j < i; ++j)
                s -= L_new[i * n + j] * l[j];
            l[i] = s / L_new[i * n + i];
        }

        // Store L[k, 0:k]
        for (int j = 0; j < k; ++j)
            L_new[k * n + j] = l[j];

        // Diagonal: K[k,k] + noise + jitter - ||l||^2
        double k_kk = scale * cache.K_norm[idx_k * cache.capacity_ + idx_k]
                      + noise + jitter;
        double l_norm_sq = 0.0;
        for (int j = 0; j < k; ++j)
            l_norm_sq += l[j] * l[j];

        double diag = k_kk - l_norm_sq;
        if (diag <= 0) {
            // Numerical issue — fall back to full fit
            fit(cache, train_indices, n, y);
            return;
        }
        L_new[k * n + k] = std::sqrt(diag);
    }

    // Store the expanded Cholesky factor
    L_data = std::move(L_new);
    n_train = n;
    train_indices_.assign(train_indices, train_indices + n);

    inv_L_diag.resize(n);
    for (int i = 0; i < n; ++i)
        inv_L_diag[i] = 1.0 / L_data[i * n + i];

    // Recompute alpha = L^T \ (L \ y)
    alpha_data.assign(y, y + n);

    // Forward solve: L @ z = y
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j)
            alpha_data[i] -= L_data[i * n + j] * alpha_data[j];
        alpha_data[i] /= L_data[i * n + i];
    }

    // Back solve: L^T @ alpha = z
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            alpha_data[i] -= L_data[j * n + i] * alpha_data[j];
        alpha_data[i] /= L_data[i * n + i];
    }
}

void GPRegressor::predict(SSKCache& cache, const std::vector<GrayCode>& candidates,
                          double* mu_out, double* std_out) {
    int n_cand = static_cast<int>(candidates.size());
    double scale = std::exp(log_scale);

    // Compute cross-kernel only for training subset rows
    std::vector<double> Ks(n_train * n_cand);
    cache.compute_cross_kernel_subset(candidates, scale, train_indices_, Ks.data());

    // mu = Ks^T @ alpha
    for (int j = 0; j < n_cand; ++j) {
        double val = 0.0;
        for (int i = 0; i < n_train; ++i)
            val += Ks[i * n_cand + j] * alpha_data[i];
        mu_out[j] = val;
    }

    if (std_out == nullptr) return;

    // v = L^{-1} @ Ks  (solve L v = Ks for v)
    // L is lower triangular, stored in L_data after dposv
    std::vector<double> v(n_train * n_cand);
    std::memcpy(v.data(), Ks.data(), n_train * n_cand * sizeof(double));
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                   n_train, n_cand, L_data.data(), n_train,
                   v.data(), n_cand);

    // var = scale - sum(v^2, axis=0)
    for (int j = 0; j < n_cand; ++j) {
        double sum_v2 = 0.0;
        for (int i = 0; i < n_train; ++i)
            sum_v2 += v[i * n_cand + j] * v[i * n_cand + j];
        double var = std::max(scale - sum_v2, 0.0);
        std_out[j] = std::sqrt(var);
    }
}

double GPRegressor::log_marginal_likelihood(SSKCache& cache, const int* train_indices,
                                            int n, const double* y,
                                            double test_log_scale, double test_noise) {
    double scale = std::exp(test_log_scale);

    std::vector<double> K(n * n);
    cache.kernel_matrix(train_indices, n, train_indices, n, scale, K.data());
    for (int i = 0; i < n; ++i)
        K[i * n + i] += test_noise + jitter;

    // Cholesky factorization
    std::vector<double> L(K);
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, L.data(), n);
    if (info != 0) return -1e30;  // not positive definite

    // alpha = K^{-1} y
    std::vector<double> alpha(y, y + n);
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n, 1, L.data(), n, alpha.data(), 1);
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'T', 'N', n, 1, L.data(), n, alpha.data(), 1);

    // log det = 2 * sum(log(diag(L)))
    double log_det = 0.0;
    for (int i = 0; i < n; ++i)
        log_det += 2.0 * std::log(L[i * n + i]);

    // quad = y^T alpha
    double quad = 0.0;
    for (int i = 0; i < n; ++i)
        quad += y[i] * alpha[i];

    return -0.5 * quad - 0.5 * log_det - 0.5 * n * std::log(2.0 * M_PI);
}

// Context for BFGS_Powell callback during GP hyperparameter optimization
struct GPHyperOptContext {
    GPRegressor* gp;
    SSKCache* cache;
    const int* train_indices;
    int n;
    const double* y;
    std::pair<double,double> scale_bounds;
    std::pair<double,double> noise_bounds;
};

// BFGS_Powell callback: computes negative log marginal likelihood and gradient
// theta = [log_scale, log_noise], gradient via finite differences
static void gp_hyper_opt_combined(Matrix_real theta, void* void_ctx,
                                   double* f_out, Matrix_real& grad) {
    GPHyperOptContext* ctx = reinterpret_cast<GPHyperOptContext*>(void_ctx);

    double ls = std::max(ctx->scale_bounds.first,
                 std::min(ctx->scale_bounds.second, (double)theta[0]));
    double log_ns = std::max(ctx->noise_bounds.first,
                     std::min(ctx->noise_bounds.second, (double)theta[1]));
    double ns = std::exp(log_ns);

    double lml = ctx->gp->log_marginal_likelihood(
        *ctx->cache, ctx->train_indices, ctx->n, ctx->y, ls, ns);
    *f_out = -lml;

    // Finite-difference gradient
    double eps = 1e-5;
    for (int d = 0; d < 2; ++d) {
        double theta_p[2] = {theta[0], theta[1]};
        theta_p[d] += eps;
        double ls_p = std::max(ctx->scale_bounds.first,
                       std::min(ctx->scale_bounds.second, theta_p[0]));
        double log_ns_p = std::max(ctx->noise_bounds.first,
                           std::min(ctx->noise_bounds.second, theta_p[1]));
        double lml_p = ctx->gp->log_marginal_likelihood(
            *ctx->cache, ctx->train_indices, ctx->n, ctx->y,
            ls_p, std::exp(log_ns_p));
        grad[d] = (-lml_p - *f_out) / eps;
    }
}

void GPRegressor::optimize_hyperparameters(SSKCache& cache, const int* train_indices,
                                           int n, const double* y,
                                           int n_restarts,
                                           const std::pair<double,double>& scale_bounds,
                                           const std::pair<double,double>& noise_bounds) {
    // Optimize theta = [log_scale, log(noise)] using squander's BFGS_Powell,
    // matching the Python scipy.optimize.minimize(method='L-BFGS-B') approach.

    GPHyperOptContext ctx;
    ctx.gp = this;
    ctx.cache = &cache;
    ctx.train_indices = train_indices;
    ctx.n = n;
    ctx.y = y;
    ctx.scale_bounds = scale_bounds;
    ctx.noise_bounds = noise_bounds;

    double best_obj = 1e30;
    double best_theta[2] = {log_scale, std::log(noise)};

    // Evaluate initial point
    {
        double ns = std::exp(best_theta[1]);
        double lml = log_marginal_likelihood(cache, train_indices, n, y,
                                             best_theta[0], ns);
        best_obj = -lml;
    }

    std::mt19937 rng(42);
    std::normal_distribution<double> randn(0.0, 0.5);

    for (int restart = 0; restart < n_restarts; ++restart) {
        Matrix_real theta(1, 2);
        if (restart == 0) {
            theta[0] = best_theta[0];
            theta[1] = best_theta[1];
        } else {
            theta[0] = best_theta[0] + randn(rng);
            theta[1] = best_theta[1] + randn(rng);
        }
        // Clamp to bounds
        theta[0] = std::max(scale_bounds.first, std::min(scale_bounds.second, (double)theta[0]));
        theta[1] = std::max(noise_bounds.first, std::min(noise_bounds.second, (double)theta[1]));

        BFGS_Powell optimizer(gp_hyper_opt_combined, &ctx);
        double f = optimizer.Start_Optimization(theta, 100);

        if (f < best_obj) {
            best_obj = f;
            best_theta[0] = theta[0];
            best_theta[1] = theta[1];
        }
    }

    log_scale = std::max(scale_bounds.first, std::min(scale_bounds.second, best_theta[0]));
    noise = std::exp(std::max(noise_bounds.first, std::min(noise_bounds.second, best_theta[1])));
}


// ============================================================================
// N_Qubit_Decomposition_Surrogate — constructors and destructor
// ============================================================================

N_Qubit_Decomposition_Surrogate::N_Qubit_Decomposition_Surrogate() {}

N_Qubit_Decomposition_Surrogate::N_Qubit_Decomposition_Surrogate(
    Matrix Umtx_in, int qbit_num_in,
    std::vector<matrix_base<int>> topology_in,
    std::map<std::string, Config_Element>& config_in,
    int accelerator_num_in)
    : Optimization_Interface(Umtx_in, qbit_num_in, false, config_in, RANDOM, accelerator_num_in) {

    // Set topology (same pattern as Tree_Search)
    topology = topology_in;
    if (topology.size() == 0) {
        for (int qbit1 = 0; qbit1 < qbit_num; qbit1++) {
            for (int qbit2 = qbit1 + 1; qbit2 < qbit_num; qbit2++) {
                matrix_base<int> edge(2, 1);
                edge[0] = qbit1;
                edge[1] = qbit2;
                topology.push_back(edge);
            }
        }
    } else {
        for (size_t idx = 0; idx < topology.size(); idx++) {
            if (topology[idx][0] > topology[idx][1])
                std::swap(topology[idx][0], topology[idx][1]);
        }
    }

    int n_edges = static_cast<int>(topology.size());
    possible_target_qbits = matrix_base<int>(1, n_edges);
    possible_control_qbits = matrix_base<int>(1, n_edges);
    for (int i = 0; i < n_edges; i++) {
        possible_target_qbits[i] = topology[i][0];
        possible_control_qbits[i] = topology[i][1];
    }

    // Precompute edge bitmasks
    // Map vertices to bit positions
    std::set<int> vertex_set;
    for (auto& e : topology) {
        vertex_set.insert(e[0]);
        vertex_set.insert(e[1]);
    }
    std::vector<int> sorted_verts(vertex_set.begin(), vertex_set.end());
    std::map<int, int> vertex_to_bit;
    for (int i = 0; i < static_cast<int>(sorted_verts.size()); ++i)
        vertex_to_bit[sorted_verts[i]] = i;

    edge_masks.resize(n_edges);
    for (int i = 0; i < n_edges; ++i)
        edge_masks[i] = (1 << vertex_to_bit[topology[i][0]]) |
                         (1 << vertex_to_bit[topology[i][1]]);

    // Precompute subspace thresholds
    int num_vertices = static_cast<int>(vertex_set.size());
    for (int n = 2; n < num_vertices; ++n) {
        int j = static_cast<int>(std::ceil(
            (std::pow(4.0, n) - 3.0 * n - 1.0) / 4.0)) + 1;
        thresholds.push_back({n, j});
    }

    // Precompute edge neighbor map: edges sharing at least one qubit
    edge_neighbors.resize(n_edges);
    for (int i = 0; i < n_edges; ++i) {
        for (int j = 0; j < n_edges; ++j) {
            if (i != j && (edge_masks[i] & edge_masks[j]))
                edge_neighbors[i].push_back(j);
        }
    }

    // Default: edge-only mode (gate-based mode is in the GateLevel subclass)
    gate_based_mode = false;

    // Parse config
    level_limit = 0;
    if (config.count("level_limit") > 0) {
        long long ll;
        config["level_limit"].get_property(ll);
        level_limit = static_cast<int>(ll);
    }

    // SSK config
    sur_gap_decay = 0.8;
    sur_match_decay = 0.8;
    sur_ssk_order = 3;

    // Search config
    kappa = 2.0;
    tolerance = 1e-8;
    max_iters = 450;
    patience = 150;
    X0_size = 10;
    candidates_per_iter = 100;
    tournament_size = 3;
    block_size = 3;
    local_search_fraction = 0.5;
    max_local_steps = 10;
    local_search_positions = 5;
    local_search_gp_subset = 50;
    n_thompson_samples = 10;
    diversity_thresh = 0.95;
    d_penalty = 0.0;
    enum_threshold = 10000;
    gp_max_train = 300;
    d_seed_budget = 50;
    window_patience = 50;
    window_max_iters = 100;
    stagnation_window = 5;
    stagnation_improvement_frac = 0.25;
    use_random_candidates = false;
    adaptive_kappa = true;
    kappa_decay_rate = 0.5;
    kappa_stagnation_boost = 0.5;
    position_guided_fraction = 0.5;
    position_lambda = 0.9;

    // Override from config
    if (config.count("kappa") > 0) {
        double v; config["kappa"].get_property(v); kappa = v;
    }
    if (config.count("tolerance") > 0) {
        double v; config["tolerance"].get_property(v); tolerance = v;
    }
    if (config.count("max_sur_iters") > 0) {
        long long v; config["max_sur_iters"].get_property(v); max_iters = static_cast<int>(v);
    }
    if (config.count("patience") > 0) {
        long long v; config["patience"].get_property(v); patience = static_cast<int>(v);
    }
    if (config.count("X0_size") > 0) {
        long long v; config["X0_size"].get_property(v); X0_size = static_cast<int>(v);
    }
    if (config.count("candidates_per_iter") > 0) {
        long long v; config["candidates_per_iter"].get_property(v);
        candidates_per_iter = static_cast<int>(v);
    }
    if (config.count("tournament_size") > 0) {
        long long v; config["tournament_size"].get_property(v);
        tournament_size = static_cast<int>(v);
    }
    if (config.count("block_mutation_size") > 0) {
        long long v; config["block_mutation_size"].get_property(v);
        block_size = static_cast<int>(v);
    }
    if (config.count("local_search_fraction") > 0) {
        double v; config["local_search_fraction"].get_property(v);
        local_search_fraction = v;
    }
    if (config.count("max_local_steps") > 0) {
        long long v; config["max_local_steps"].get_property(v);
        max_local_steps = static_cast<int>(v);
    }
    if (config.count("local_search_positions") > 0) {
        long long v; config["local_search_positions"].get_property(v);
        local_search_positions = static_cast<int>(v);
    }
    if (config.count("local_search_gp_subset") > 0) {
        long long v; config["local_search_gp_subset"].get_property(v);
        local_search_gp_subset = static_cast<int>(v);
    }
    if (config.count("n_thompson_samples") > 0) {
        long long v; config["n_thompson_samples"].get_property(v);
        n_thompson_samples = static_cast<int>(v);
    }
    if (config.count("topk_diversity_threshold") > 0) {
        double v; config["topk_diversity_threshold"].get_property(v);
        diversity_thresh = v;
    }
    if (config.count("d_penalty") > 0) {
        double v; config["d_penalty"].get_property(v); d_penalty = v;
    }
    if (config.count("enum_threshold") > 0) {
        long long v; config["enum_threshold"].get_property(v);
        enum_threshold = static_cast<int>(v);
    }
    if (config.count("window_patience") > 0) {
        long long v; config["window_patience"].get_property(v);
        window_patience = static_cast<int>(v);
    }
    if (config.count("window_max_iters") > 0) {
        long long v; config["window_max_iters"].get_property(v);
        window_max_iters = static_cast<int>(v);
    }
    if (config.count("gp_max_train") > 0) {
        long long v; config["gp_max_train"].get_property(v);
        gp_max_train = static_cast<int>(v);
    }
    if (config.count("d_seed_budget") > 0) {
        long long v; config["d_seed_budget"].get_property(v);
        d_seed_budget = static_cast<int>(v);
    }
    if (config.count("stagnation_window") > 0) {
        long long v; config["stagnation_window"].get_property(v);
        stagnation_window = static_cast<int>(v);
    }
    if (config.count("stagnation_improvement_frac") > 0) {
        double v; config["stagnation_improvement_frac"].get_property(v);
        stagnation_improvement_frac = v;
    }
    if (config.count("ssk_gap_decay") > 0) {
        double v; config["ssk_gap_decay"].get_property(v); sur_gap_decay = v;
    }
    if (config.count("ssk_match_decay") > 0) {
        double v; config["ssk_match_decay"].get_property(v); sur_match_decay = v;
    }
    if (config.count("ssk_order") > 0) {
        long long v; config["ssk_order"].get_property(v); sur_ssk_order = static_cast<int>(v);
    }
    if (config.count("use_random_candidates") > 0) {
        long long v; config["use_random_candidates"].get_property(v);
        use_random_candidates = static_cast<bool>(v);
    }
    if (config.count("adaptive_kappa") > 0) {
        long long v; config["adaptive_kappa"].get_property(v);
        adaptive_kappa = static_cast<bool>(v);
    }
    if (config.count("kappa_decay_rate") > 0) {
        double v; config["kappa_decay_rate"].get_property(v);
        kappa_decay_rate = v;
    }
    if (config.count("kappa_stagnation_boost") > 0) {
        double v; config["kappa_stagnation_boost"].get_property(v);
        kappa_stagnation_boost = v;
    }
    if (config.count("position_guided_fraction") > 0) {
        double v; config["position_guided_fraction"].get_property(v);
        position_guided_fraction = v;
    }
    if (config.count("position_lambda") > 0) {
        double v; config["position_lambda"].get_property(v);
        position_lambda = v;
    }

    // Edge-only tokenization (gate-based mode is in the GateLevel subclass)
    n_1q_types = 0;
    n_1q_tokens = 0;
    n_directed_cnots = 0;
    gate_1q_types.clear();
    gate_1q_param_counts.clear();
    cnot_target_qbits.clear();
    cnot_control_qbits.clear();
    cnot_undirected_edge.clear();

    n_tokens = n_edges;
    token_masks = edge_masks;
    token_neighbors = edge_neighbors;

    // Compute OSR cut bounds using existing C++ infrastructure
    osr_cuts = unique_cuts(qbit_num);
    double Fnorm = std::sqrt(static_cast<double>(Umtx.rows));

    osr_cut_bounds.resize(osr_cuts.size());
    osr_cut_crossing_edges.resize(osr_cuts.size());

    for (size_t c = 0; c < osr_cuts.size(); ++c) {
        std::pair<int, double> osr_result = operator_schmidt_rank(Umtx, qbit_num, osr_cuts[c], Fnorm, 1e-3);
        int min_cnots = osr_result.first;
        osr_cut_bounds[c] = min_cnots;

        // Find edges crossing this cut
        std::set<int> in_cut(osr_cuts[c].begin(), osr_cuts[c].end());
        for (int i = 0; i < n_edges; ++i) {
            bool a_in = in_cut.count(topology[i][0]) > 0;
            bool b_in = in_cut.count(topology[i][1]) > 0;
            if (a_in != b_in)
                osr_cut_crossing_edges[c].push_back(i);
        }
    }

    // Compute osr_D_min using MinCnotBoundSolver
    std::vector<std::pair<int, double>> cut_bounds_pairs;
    for (size_t c = 0; c < osr_cuts.size(); ++c)
        cut_bounds_pairs.push_back({osr_cut_bounds[c], 0.0});

    MinCnotBoundSolver solver(qbit_num, osr_cuts, topology);
    osr_D_min = solver.solve_min_cnots(cut_bounds_pairs, 100);
    if (osr_D_min < 0) osr_D_min = 0;

    // Default optimizer (overridden by set_Optimizer from Python)
    alg = BFGS;

    // Override optimizer from config
    if (config.count("optimizer") > 0) {
        // Read as string from config element
        long long alg_val = -1;
        config["optimizer"].get_property(alg_val);
        if (alg_val >= 0) {
            alg = static_cast<optimization_aglorithms>(alg_val);
        }
    }

    best_score = std::numeric_limits<double>::infinity();
    decompose_time = 0.0;
    decompose_count = 0;
    total_search_time = 0.0;
}

N_Qubit_Decomposition_Surrogate::N_Qubit_Decomposition_Surrogate(
    Matrix Umtx_in, int qbit_num_in,
    std::map<std::string, Config_Element>& config_in,
    int accelerator_num_in)
    : N_Qubit_Decomposition_Surrogate(Umtx_in, qbit_num_in,
          std::vector<matrix_base<int>>(), config_in, accelerator_num_in) {}

N_Qubit_Decomposition_Surrogate::~N_Qubit_Decomposition_Surrogate() {}

void N_Qubit_Decomposition_Surrogate::set_unitary(Matrix& Umtx_new) {
    Umtx = Umtx_new;
}


// ============================================================================
// Validation and canonicalization
// ============================================================================

bool N_Qubit_Decomposition_Surrogate::check_new_position(const int* window_masks, int pos) {
    for (size_t t = 0; t < thresholds.size(); ++t) {
        int n = thresholds[t].first;
        int j = thresholds[t].second;
        if (j > pos + 1) continue;
        int combined = 0;
        for (int i = pos - j + 1; i <= pos; ++i)
            combined |= window_masks[i];
        if (__builtin_popcount(combined) <= n)
            return true;  // subspace violation
    }
    return false;
}

GrayCode N_Qubit_Decomposition_Surrogate::canonical_form(const GrayCode& seq) {
    int n = static_cast<int>(seq.size());
    if (n == 0) return GrayCode();

    // Build dependency DAG using token bitmasks
    std::vector<std::vector<int>> adj(n);
    std::vector<int> in_degree(n, 0);

    // Get masks for each position
    std::vector<int> masks(n);
    for (int i = 0; i < n; ++i)
        masks[i] = token_masks[seq[i]];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (masks[j] & masks[i]) {
                adj[j].push_back(i);
                in_degree[i]++;
            }
        }
    }

    // Min-heap topological sort using token_sort_key: (type, qubit1, qubit2)
    struct HeapNode {
        std::tuple<int,int,int> key;
        int idx;
        bool operator>(const HeapNode& o) const {
            if (key != o.key) return key > o.key;
            return idx > o.idx;
        }
    };
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> heap;
    for (int i = 0; i < n; ++i) {
        if (in_degree[i] == 0) {
            heap.push(HeapNode{token_sort_key(seq[i]), i});
        }
    }

    matrix_base<int> limits(1, n);
    for (int i = 0; i < n; ++i)
        limits[i] = n_tokens;
    GrayCode result(limits);

    int pos = 0;
    while (!heap.empty()) {
        HeapNode top = heap.top();
        heap.pop();
        result[pos++] = seq[top.idx];
        for (int neighbor : adj[top.idx]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                heap.push(HeapNode{token_sort_key(seq[neighbor]), neighbor});
            }
        }
    }

    return result;
}

GrayCode N_Qubit_Decomposition_Surrogate::canonicalize_and_validate(const GrayCode& seq) {
    GrayCode canon = canonical_form(seq);
    int D = static_cast<int>(canon.size());

    // Check subspace constraints
    std::vector<int> masks(D);
    for (int i = 0; i < D; ++i)
        masks[i] = token_masks[canon[i]];

    for (int depth = 0; depth < D; ++depth) {
        if (check_new_position(masks.data(), depth))
            return GrayCode();  // invalid — subspace violation
    }

    // Check OSR feasibility
    if (!check_osr_feasibility(canon))
        return GrayCode();

    return canon;
}

bool N_Qubit_Decomposition_Surrogate::check_osr_feasibility(const GrayCode& circuit) {
    int n_edges = static_cast<int>(topology.size());
    std::vector<int> edge_counts(n_edges, 0);
    for (int i = 0; i < static_cast<int>(circuit.size()); ++i) {
        edge_counts[circuit[i]]++;
    }

    for (size_t c = 0; c < osr_cuts.size(); ++c) {
        if (osr_cut_bounds[c] <= 0) continue;
        int sum = 0;
        for (int eidx : osr_cut_crossing_edges[c])
            sum += edge_counts[eidx];
        if (sum < osr_cut_bounds[c])
            return false;
    }
    return true;
}


// ============================================================================
// Enumeration
// ============================================================================

std::vector<GrayCode> N_Qubit_Decomposition_Surrogate::enumerate_circuits(int D) {
    std::vector<GrayCode> results;
    GrayCodeSet seen;

    // DFS enumeration with pruning
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_tokens;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    std::function<void(int)> dfs = [&](int depth) {
        if (depth == D) {
            GrayCode canon = canonical_form(path);
            if (seen.find(canon) == seen.end()) {
                if (check_osr_feasibility(canon)) {
                    seen.insert(canon.copy());
                    results.push_back(canon.copy());
                }
            }
            return;
        }

        for (int e = 0; e < n_tokens; ++e) {
            path[depth] = e;
            path_masks[depth] = token_masks[e];

            if (check_new_position(path_masks.data(), depth))
                continue;

            // Canonical prefix check (simplified: compare with previous independent ops)
            bool canonical = true;
            int i = depth;
            while (i > 0) {
                if (path_masks[i] & path_masks[i - 1]) break;  // dependent
                if (path[i] < path[i - 1]) { canonical = false; break; }
                i--;
            }
            if (!canonical) continue;

            dfs(depth + 1);
        }
    };

    dfs(0);
    return results;
}


// ============================================================================
// Evolutionary operators
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::generate_valid_sequence(int D) {
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_tokens;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    for (int depth = 0; depth < D; ++depth) {
        std::vector<int> valid;
        for (int e = 0; e < n_tokens; ++e) {
            path[depth] = e;
            path_masks[depth] = token_masks[e];
            if (!check_new_position(path_masks.data(), depth))
                valid.push_back(e);
        }
        if (valid.empty()) return GrayCode();  // failed

        std::uniform_int_distribution<int> pick(0, static_cast<int>(valid.size()) - 1);
        int chosen = valid[pick(gen)];
        path[depth] = chosen;
        path_masks[depth] = token_masks[chosen];
    }

    return canonicalize_and_validate(path);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_point(const GrayCode& seq) {
    int D = static_cast<int>(seq.size());
    if (n_tokens < 2) return GrayCode();  // cannot substitute with only one token
    std::uniform_int_distribution<int> pos_dist(0, D - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int pos = pos_dist(gen);
        int cur_token = seq[pos];
        int new_token;

        // 70% chance: pick from neighboring tokens (share a qubit) for higher acceptance
        // 30% chance: pick any other token for exploration
        const std::vector<int>& nbrs = token_neighbors[cur_token];
        if (!nbrs.empty() && coin(gen) < 0.7) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(gen)];
        } else {
            std::uniform_int_distribution<int> token_dist(0, n_tokens - 2);
            new_token = token_dist(gen);
            if (new_token >= cur_token) ++new_token;  // skip current token
        }

        GrayCode new_seq = seq.copy();
        new_seq[pos] = new_token;
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_point_guided(
    const GrayCode& seq, SSKCache& cache, GPRegressor& gp, double scale) {

    int D = static_cast<int>(seq.size());
    if (D < 1 || n_tokens < 2 || gp.n_train == 0) return mutate_point(seq);

    // For each position, find the best GP mu across all alternative tokens.
    // Position improvement potential = current_mu - best_alternative_mu (lower is better).
    // We only compute mu (skip variance) for efficiency.

    // First, compute mu for the current circuit
    std::vector<GrayCode> cur_vec = {seq};
    std::vector<double> Ks_cur(gp.n_train);
    cache.compute_cross_kernel_subset(cur_vec, scale, gp.train_indices_, Ks_cur.data());
    double mu_cur = 0;
    for (int i = 0; i < gp.n_train; ++i)
        mu_cur += Ks_cur[i] * gp.alpha_data[i];

    // For each position, generate all single-token substitutions and batch-predict mu
    std::vector<double> pos_improvement(D, 0.0);
    std::vector<GrayCode> all_variants;
    std::vector<int> variant_pos;    // which position each variant belongs to
    all_variants.reserve(D * (n_tokens - 1));
    variant_pos.reserve(D * (n_tokens - 1));

    for (int pos = 0; pos < D; ++pos) {
        for (int e = 0; e < n_tokens; ++e) {
            if (e == seq[pos]) continue;
            GrayCode new_seq = seq.copy();
            new_seq[pos] = e;
            all_variants.push_back(std::move(new_seq));
            variant_pos.push_back(pos);
        }
    }

    if (all_variants.empty()) return mutate_point(seq);

    int n_var = static_cast<int>(all_variants.size());
    std::vector<double> Ks_var(gp.n_train * n_var);
    cache.compute_cross_kernel_subset(all_variants, scale, gp.train_indices_, Ks_var.data());

    // Compute mu for each variant and track best improvement per position
    std::vector<double> best_mu_per_pos(D, mu_cur);
    for (int j = 0; j < n_var; ++j) {
        double mu_val = 0.0;
        for (int i = 0; i < gp.n_train; ++i)
            mu_val += Ks_var[i * n_var + j] * gp.alpha_data[i];
        int pos = variant_pos[j];
        if (mu_val < best_mu_per_pos[pos])
            best_mu_per_pos[pos] = mu_val;
    }

    // Position improvement = how much the best alternative improves over current
    // Higher value = more improvable position (we want to mutate these more)
    for (int pos = 0; pos < D; ++pos)
        pos_improvement[pos] = mu_cur - best_mu_per_pos[pos];

    // Temperature softmax (Quarl Eq. 5): t = 1 / ln(λ(D-1) / (1-λ))
    double lam = position_lambda;
    double temp_arg = lam * (D - 1) / (1.0 - lam);
    double temperature = (temp_arg > 1.0) ? 1.0 / std::log(temp_arg) : 1.0;

    // Softmax over position improvements
    std::vector<double> weights(D);
    double max_imp = *std::max_element(pos_improvement.begin(), pos_improvement.end());
    double w_sum = 0;
    for (int pos = 0; pos < D; ++pos) {
        weights[pos] = std::exp((pos_improvement[pos] - max_imp) / std::max(temperature, 1e-8));
        w_sum += weights[pos];
    }
    for (int pos = 0; pos < D; ++pos)
        weights[pos] /= w_sum;

    // Build CDF and sample position
    std::vector<double> cdf(D);
    cdf[0] = weights[0];
    for (int pos = 1; pos < D; ++pos)
        cdf[pos] = cdf[pos - 1] + weights[pos];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        double r = uni(gen);
        int pos = D - 1;
        for (int p = 0; p < D; ++p) {
            if (r <= cdf[p]) { pos = p; break; }
        }

        int cur_token = seq[pos];
        int new_token;

        // Same neighbor-biased token selection as mutate_point
        const std::vector<int>& nbrs = token_neighbors[cur_token];
        if (!nbrs.empty() && coin(gen) < 0.7) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(gen)];
        } else {
            std::uniform_int_distribution<int> token_dist(0, n_tokens - 2);
            new_token = token_dist(gen);
            if (new_token >= cur_token) ++new_token;
        }

        GrayCode new_seq = seq.copy();
        new_seq[pos] = new_token;
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_swap(const GrayCode& seq) {
    int D = static_cast<int>(seq.size());
    if (D < 2) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    for (int attempt = 0; attempt < 50; ++attempt) {
        GrayCode new_seq = seq.copy();
        int i = pos_dist(gen);
        int j = pos_dist(gen);
        while (j == i) j = pos_dist(gen);
        std::swap(new_seq[i], new_seq[j]);
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block(const GrayCode& seq, int blk_size) {
    int D = static_cast<int>(seq.size());
    int bs = std::min(blk_size, D);
    std::uniform_int_distribution<int> start_dist(0, D - bs);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        GrayCode new_seq = seq.copy();
        int start = start_dist(gen);
        for (int p = start; p < start + bs; ++p) {
            int cur_token = seq[p];
            const std::vector<int>& nbrs = token_neighbors[cur_token];
            if (!nbrs.empty() && coin(gen) < 0.5) {
                std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
                new_seq[p] = nbrs[nbr_dist(gen)];
            } else {
                new_seq[p] = token_dist(gen);
            }
        }
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_transplant(
        const GrayCode& recipient, const GrayCode& donor, int blk_size) {
    int D_r = static_cast<int>(recipient.size());
    int D_d = static_cast<int>(donor.size());
    int max_blk = std::min(blk_size, std::min(D_r, D_d));
    if (max_blk < 2) return GrayCode();
    std::uniform_int_distribution<int> blk_dist(2, max_blk);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int blk = blk_dist(gen);
        std::uniform_int_distribution<int> r_start_dist(0, D_r - blk);
        std::uniform_int_distribution<int> d_start_dist(0, D_d - blk);
        int r_start = r_start_dist(gen);
        int d_start = d_start_dist(gen);

        GrayCode new_seq = recipient.copy();
        for (int i = 0; i < blk; ++i)
            new_seq[r_start + i] = donor[d_start + i];

        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block_regenerate(
        const GrayCode& seq, int blk_size) {
    int D = static_cast<int>(seq.size());
    int bs = std::min(blk_size, D);
    int max_bs = std::min(2 * bs, D);
    if (bs < 1) return GrayCode();
    std::uniform_int_distribution<int> bs_dist(bs, max_bs);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int actual_bs = bs_dist(gen);
        std::uniform_int_distribution<int> start_dist(0, D - actual_bs);
        int start = start_dist(gen);

        GrayCode new_seq = seq.copy();
        std::vector<int> path_masks(D);
        for (int i = 0; i < D; ++i)
            path_masks[i] = token_masks[new_seq[i]];

        bool valid_block = true;
        for (int pos = start; pos < start + actual_bs; ++pos) {
            std::vector<int> candidates;
            for (int e = 0; e < n_tokens; ++e) {
                new_seq[pos] = e;
                path_masks[pos] = token_masks[e];
                if (!check_new_position(path_masks.data(), pos))
                    candidates.push_back(e);
            }
            if (candidates.empty()) {
                valid_block = false;
                break;
            }
            std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
            int chosen = candidates[pick(gen)];
            new_seq[pos] = chosen;
            path_masks[pos] = token_masks[chosen];
        }
        if (!valid_block) continue;

        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::crossover_uniform(const GrayCode& seq1,
                                                             const GrayCode& seq2) {
    int D = static_cast<int>(seq1.size());
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        GrayCode new_seq = seq1.copy();
        for (int i = 0; i < D; ++i)
            new_seq[i] = (coin(gen) < 0.5) ? seq1[i] : seq2[i];
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_grow(const GrayCode& seq, int D_max) {
    int D = static_cast<int>(seq.size());
    if (D >= D_max) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int pos = pos_dist(gen);
        int new_token;
        // Bias toward tokens neighboring the adjacent positions
        int adj_token = (pos < D) ? seq[pos] : seq[D - 1];
        const std::vector<int>& nbrs = token_neighbors[adj_token];
        if (!nbrs.empty() && coin(gen) < 0.5) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(gen)];
        } else {
            new_token = token_dist(gen);
        }

        // Build new sequence with insertion
        matrix_base<int> limits(1, D + 1);
        for (int i = 0; i < D + 1; ++i) limits[i] = n_tokens;
        GrayCode new_seq(limits);
        for (int i = 0; i < pos; ++i) new_seq[i] = seq[i];
        new_seq[pos] = new_token;
        for (int i = pos; i < D; ++i) new_seq[i + 1] = seq[i];

        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_shrink(const GrayCode& seq, int D_min) {
    int D = static_cast<int>(seq.size());
    if (D <= D_min) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int pos = pos_dist(gen);
        GrayCode new_seq = seq.remove_Digit(pos);
        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}


// ============================================================================
// Local search on LCB acquisition
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::local_search_acq(
    const GrayCode& start, SSKCache& cache, GPRegressor& gp,
    double scale, GrayCodeSet& seen,
    int D_min_local, int D_max_local, int* steps_out) {

    // Thread-local RNG for stochastic position sampling (called from TBB parallel_for)
    thread_local std::mt19937 local_rng(
        std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
        static_cast<size_t>(std::chrono::steady_clock::now().time_since_epoch().count()));

    GrayCode current = start.copy();

    // Evaluate LCB at start
    std::vector<GrayCode> start_vec = {current};
    std::vector<double> Ks_cur(gp.n_train);
    cache.compute_cross_kernel_subset(start_vec, scale, gp.train_indices_, Ks_cur.data());
    double mu_cur = 0;
    for (int i = 0; i < gp.n_train; ++i)
        mu_cur += Ks_cur[i] * gp.alpha_data[i];

    // Solve L v = Ks for variance
    std::vector<double> v_cur(gp.n_train);
    std::memcpy(v_cur.data(), Ks_cur.data(), gp.n_train * sizeof(double));
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                   gp.n_train, 1, gp.L_data.data(), gp.n_train,
                   v_cur.data(), 1);
    double sum_v2 = 0;
    for (int i = 0; i < gp.n_train; ++i)
        sum_v2 += v_cur[i] * v_cur[i];
    double std_cur = std::sqrt(std::max(scale - sum_v2, 0.0));
    double best_lcb = lcb(mu_cur, std_cur);

    int steps_taken = 0;
    std::vector<int> pos_visit_count(static_cast<int>(current.size()), 0);

    for (int step = 0; step < max_local_steps; ++step) {
        int D = static_cast<int>(current.size());

        // Resize visit counts if D changed (grow/shrink)
        if (static_cast<int>(pos_visit_count.size()) != D)
            pos_visit_count.assign(D, 0);

        int min_visits = *std::min_element(pos_visit_count.begin(), pos_visit_count.end());

        std::vector<GrayCode> neighbors;
        std::vector<int> neighbor_pos;  // source position for each neighbor (-1 for grow/shrink)

        // Single-position substitution neighbors (stochastic sampling)
        // Sample local_search_positions positions per step (0 = all)
        int n_pos_to_try = (local_search_positions > 0 && local_search_positions < D)
                           ? local_search_positions : D;

        // Build candidate positions: prioritize unvisited, then sample
        std::vector<int> candidate_positions;
        candidate_positions.reserve(D);
        for (int pos = 0; pos < D; ++pos) {
            if (pos_visit_count[pos] <= min_visits)
                candidate_positions.push_back(pos);
        }
        // Shuffle and truncate to n_pos_to_try
        if (static_cast<int>(candidate_positions.size()) > n_pos_to_try) {
            // Fisher-Yates partial shuffle for first n_pos_to_try elements
            for (int i = 0; i < n_pos_to_try; ++i) {
                std::uniform_int_distribution<int> swap_dist(i, static_cast<int>(candidate_positions.size()) - 1);
                std::swap(candidate_positions[i], candidate_positions[swap_dist(local_rng)]);
            }
            candidate_positions.resize(n_pos_to_try);
        }

        for (int pi = 0; pi < static_cast<int>(candidate_positions.size()); ++pi) {
            int pos = candidate_positions[pi];
            for (int e = 0; e < n_tokens; ++e) {
                if (e == current[pos]) continue;
                GrayCode new_seq = current.copy();
                new_seq[pos] = e;
                GrayCode cand = canonicalize_and_validate(new_seq);
                if (cand.size() > 0 && seen.find(cand) == seen.end()) {
                    neighbors.push_back(std::move(cand));
                    neighbor_pos.push_back(pos);
                }
            }
        }

        // Grow neighbors (sample positions too)
        if (D_max_local > 0 && D < D_max_local) {
            int n_grow_pos = std::min(n_pos_to_try, D + 1);
            std::vector<int> grow_positions(D + 1);
            std::iota(grow_positions.begin(), grow_positions.end(), 0);
            if (n_grow_pos < D + 1) {
                for (int i = 0; i < n_grow_pos; ++i) {
                    std::uniform_int_distribution<int> swap_dist(i, D);
                    std::swap(grow_positions[i], grow_positions[swap_dist(local_rng)]);
                }
                grow_positions.resize(n_grow_pos);
            }
            for (int pi = 0; pi < n_grow_pos; ++pi) {
                int pos = grow_positions[pi];
                for (int e = 0; e < n_tokens; ++e) {
                    matrix_base<int> limits(1, D + 1);
                    for (int i = 0; i < D + 1; ++i) limits[i] = n_tokens;
                    GrayCode new_seq(limits);
                    for (int i = 0; i < pos; ++i) new_seq[i] = current[i];
                    new_seq[pos] = e;
                    for (int i = pos; i < D; ++i) new_seq[i + 1] = current[i];
                    GrayCode cand = canonicalize_and_validate(new_seq);
                    if (cand.size() > 0 && seen.find(cand) == seen.end()) {
                        neighbors.push_back(std::move(cand));
                        neighbor_pos.push_back(-1);
                    }
                }
            }
        }

        // Shrink neighbors (sample positions too)
        if (D_min_local >= 0 && D > D_min_local) {
            int n_shrink_pos = std::min(n_pos_to_try, D);
            std::vector<int> shrink_positions(D);
            std::iota(shrink_positions.begin(), shrink_positions.end(), 0);
            if (n_shrink_pos < D) {
                for (int i = 0; i < n_shrink_pos; ++i) {
                    std::uniform_int_distribution<int> swap_dist(i, D - 1);
                    std::swap(shrink_positions[i], shrink_positions[swap_dist(local_rng)]);
                }
                shrink_positions.resize(n_shrink_pos);
            }
            for (int pi = 0; pi < n_shrink_pos; ++pi) {
                int pos = shrink_positions[pi];
                GrayCode new_seq = current.remove_Digit(pos);
                GrayCode cand = canonicalize_and_validate(new_seq);
                if (cand.size() > 0 && seen.find(cand) == seen.end()) {
                    neighbors.push_back(std::move(cand));
                    neighbor_pos.push_back(-1);
                }
            }
        }

        // Deduplicate neighbors
        GrayCodeSet neighbor_set;
        std::vector<GrayCode> unique_neighbors;
        for (auto& nb : neighbors) {
            if (neighbor_set.find(nb) == neighbor_set.end()) {
                neighbor_set.insert(nb.copy());
                unique_neighbors.push_back(std::move(nb));
            }
        }
        neighbors = std::move(unique_neighbors);

        if (neighbors.empty()) break;

        // Batch-evaluate LCB — compute cross-kernel only for training subset
        int n_nb = static_cast<int>(neighbors.size());
        std::vector<double> Ks(gp.n_train * n_nb);
        cache.compute_cross_kernel_subset(neighbors, scale, gp.train_indices_, Ks.data());

        // Stage 1: Compute mu AND approximate variance for all neighbors
        // Fused loop: O(n_train * n_nb) — same cost as mu alone
        int n_t = gp.n_train;
        std::vector<double> mu(n_nb);
        std::vector<double> lcb_approx(n_nb);

        for (int j = 0; j < n_nb; ++j) {
            double mu_val = 0.0;
            double v2_sum = 0.0;
            for (int i = 0; i < n_t; ++i) {
                double ks_ij = Ks[i * n_nb + j];
                mu_val += ks_ij * gp.alpha_data[i];
                double v_approx_i = ks_ij * gp.inv_L_diag[i];
                v2_sum += v_approx_i * v_approx_i;
            }
            mu[j] = mu_val;
            double var_approx = std::max(scale - v2_sum, 0.0);
            lcb_approx[j] = mu_val - kappa * std::sqrt(var_approx);
        }

        // Stage 2: Select top-k by approximate LCB for exact evaluation
        constexpr int k_exact = 5;
        int k = std::min(k_exact, n_nb);

        std::vector<int> order(n_nb);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int a, int b) { return lcb_approx[a] < lcb_approx[b]; });

        // Stage 3: Exact forward solve for top-k only — O(k * n_train^2)
        double best_nb_lcb = std::numeric_limits<double>::infinity();
        int best_nb_idx = -1;
        std::vector<double> v_col(n_t);

        for (int rank = 0; rank < k; ++rank) {
            int j = order[rank];

            if (mu[j] >= best_nb_lcb) continue;

            // Forward solve L v = Ks[:,j]
            for (int i = 0; i < n_t; ++i)
                v_col[i] = Ks[i * n_nb + j];
            for (int i = 0; i < n_t; ++i) {
                for (int ii = 0; ii < i; ++ii)
                    v_col[i] -= gp.L_data[i * n_t + ii] * v_col[ii];
                v_col[i] /= gp.L_data[i * n_t + i];
            }

            double sv2 = 0;
            for (int i = 0; i < n_t; ++i)
                sv2 += v_col[i] * v_col[i];
            double std_val = std::sqrt(std::max(scale - sv2, 0.0));
            double lcb_val = lcb(mu[j], std_val);
            if (lcb_val < best_nb_lcb) {
                best_nb_lcb = lcb_val;
                best_nb_idx = j;
            }
        }

        if (best_nb_lcb >= best_lcb) break;  // local optimum

        // Update position visit count
        int selected_pos = neighbor_pos[best_nb_idx];
        if (selected_pos >= 0 && selected_pos < static_cast<int>(pos_visit_count.size())) {
            pos_visit_count[selected_pos]++;
            // Soft mask reset: once all positions visited, clear counts
            if (*std::min_element(pos_visit_count.begin(), pos_visit_count.end()) > 0)
                std::fill(pos_visit_count.begin(), pos_visit_count.end(), 0);
        }

        current = neighbors[best_nb_idx].copy();
        best_lcb = best_nb_lcb;
        steps_taken++;
    }

    if (steps_out) *steps_out = steps_taken;
    return current;
}


// ============================================================================
// Candidate generation
// ============================================================================

void N_Qubit_Decomposition_Surrogate::generate_candidates(
    const std::vector<GrayCode>& population, const double* scores, int n_pop,
    int n_candidates, SSKCache& cache, GPRegressor& gp, double scale,
    const double* train_y_norm,
    GrayCodeSet& seen, std::vector<GrayCode>& candidates_out,
    int& n_local_out, double& avg_steps_out,
    int D_min_gen, int D_max_gen) {

    candidates_out.clear();
    bool mixed_d = (D_min_gen >= 0 && D_max_gen >= 0);

    // Build valid population indices (respect D_max)
    std::vector<int> pop_indices;
    for (int i = 0; i < n_pop; ++i) {
        if (D_max_gen < 0 || static_cast<int>(population[i].size()) <= D_max_gen)
            pop_indices.push_back(i);
    }
    if (pop_indices.empty())
        pop_indices.resize(n_pop);
    for (int i = 0; i < n_pop && pop_indices.empty(); ++i)
        pop_indices.push_back(i);

    int n_valid = static_cast<int>(pop_indices.size());
    int t_size = std::min(tournament_size, n_valid);

    std::uniform_int_distribution<int> valid_dist(0, n_valid - 1);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);

    auto tournament_select = [&]() -> const GrayCode& {
        int best_idx = pop_indices[valid_dist(gen)];
        for (int i = 1; i < t_size; ++i) {
            int idx = pop_indices[valid_dist(gen)];
            if (scores[idx] < scores[best_idx])
                best_idx = idx;
        }
        return population[best_idx];
    };

    // GP-directed local search candidates (parallelized)
    int n_local_found = 0;
    int total_local_steps = 0;
    int n_local_target = static_cast<int>(n_candidates * local_search_fraction);

    if (n_local_target > 0) {
        // Phase A: pre-select parents sequentially (uses gen)
        std::vector<GrayCode> local_parents(n_local_target);
        for (int i = 0; i < n_local_target; ++i)
            local_parents[i] = tournament_select().copy();

        // Build lightweight GP proxy for local search (Idea B)
        // Subsample training points for cheaper cross-kernel computation
        GPRegressor* local_gp_ptr = &gp;
        GPRegressor local_gp;
        double local_scale = scale;

        int n_full = gp.n_train;
        int n_sub = local_search_gp_subset;
        if (n_sub > 0 && n_sub < n_full && train_y_norm != nullptr) {
            // Select subset: top half by score (exploitation), rest random (coverage)
            int n_best = n_sub / 2;
            int n_rand = n_sub - n_best;

            std::vector<int> sub_indices;
            sub_indices.reserve(n_sub);
            for (int i = 0; i < n_best && i < n_full; ++i)
                sub_indices.push_back(i);

            // Random sample from remaining
            std::vector<int> remaining;
            remaining.reserve(n_full - n_best);
            for (int i = n_best; i < n_full; ++i)
                remaining.push_back(i);
            for (int i = 0; i < n_rand && !remaining.empty(); ++i) {
                std::uniform_int_distribution<int> rdist(0, static_cast<int>(remaining.size()) - 1);
                int pick = rdist(gen);
                sub_indices.push_back(remaining[pick]);
                remaining[pick] = remaining.back();
                remaining.pop_back();
            }

            std::sort(sub_indices.begin(), sub_indices.end());
            int n_actual = static_cast<int>(sub_indices.size());

            // Build sub-training set — directly subset the normalized targets
            std::vector<int> sub_train(n_actual);
            std::vector<double> sub_y(n_actual);
            for (int i = 0; i < n_actual; ++i) {
                sub_train[i] = gp.train_indices_[sub_indices[i]];
                sub_y[i] = train_y_norm[sub_indices[i]];
            }

            local_gp.log_scale = gp.log_scale;
            local_gp.noise = gp.noise;
            local_gp.jitter = gp.jitter;
            local_gp.fit(cache, sub_train.data(), n_actual, sub_y.data());
            local_scale = local_gp.get_scale();
            local_gp_ptr = &local_gp;
        }

        // Phase B: run local searches in parallel
        // local_search_acq is read-only on cache, gp, and seen
        std::vector<GrayCode> local_results(n_local_target);
        std::vector<int> local_steps(n_local_target, 0);

        unsigned int nthreads = std::thread::hardware_concurrency();
        int64_t concurrency = std::min(static_cast<int64_t>(nthreads),
                                       static_cast<int64_t>(n_local_target));
        int parallel = get_parallel_configuration();
        int64_t work_batch = (parallel == 0) ? concurrency : 1;

        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, concurrency, work_batch),
            [&](tbb::blocked_range<int64_t> r) {
                for (int64_t job_idx = r.begin(); job_idx < r.end(); ++job_idx) {
                    int64_t batch_sz = n_local_target / concurrency;
                    int64_t start = job_idx * batch_sz;
                    int64_t end = (job_idx == concurrency - 1) ? n_local_target : start + batch_sz;

                    for (int64_t i = start; i < end; ++i) {
                        local_results[i] = local_search_acq(
                            local_parents[i], cache, *local_gp_ptr, local_scale, seen,
                            D_min_gen, D_max_gen, &local_steps[i]);
                    }
                }
            });

        // Phase C: sequential dedup and insert
        for (int i = 0; i < n_local_target; ++i) {
            total_local_steps += local_steps[i];
            if (local_results[i].size() > 0 && seen.find(local_results[i]) == seen.end() &&
                (D_max_gen < 0 || static_cast<int>(local_results[i].size()) <= D_max_gen) &&
                (D_min_gen < 0 || static_cast<int>(local_results[i].size()) >= D_min_gen)) {
                seen.insert(local_results[i].copy());
                candidates_out.push_back(std::move(local_results[i]));
                n_local_found++;
            }
        }
    }

    // Helper: position-guided or uniform point mutation
    bool use_guided = (position_guided_fraction > 0 && gp.n_train > 0);
    auto maybe_guided_point = [&](const GrayCode& parent) -> GrayCode {
        if (use_guided && op_dist(gen) < position_guided_fraction)
            return mutate_point_guided(parent, cache, gp, scale);
        return mutate_point(parent);
    };

    // Random candidates via evolutionary operators
    int max_attempts = n_candidates * 10;
    for (int attempt = 0;
         static_cast<int>(candidates_out.size()) < n_candidates && attempt < max_attempts;
         ++attempt) {
        double r = op_dist(gen);
        GrayCode result;

        // Adaptive block size: scale with circuit depth
        int ref_D = mixed_d ? (D_min_gen + D_max_gen) / 2
                            : static_cast<int>(population[0].size());
        int adaptive_blk = std::max(block_size, ref_D / 4);

        if (mixed_d) {
            if (r < 0.20)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.30)
                result = mutate_swap(tournament_select());
            else if (r < 0.40)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk);
            else if (r < 0.55) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk);
            }
            else if (r < 0.65) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                if (p1.size() == p2.size())
                    result = crossover_uniform(p1, p2);
                else
                    result = maybe_guided_point(p1);
            }
            else if (r < 0.70)
                result = mutate_grow(tournament_select(), D_max_gen);
            else if (r < 0.85)
                result = mutate_shrink(tournament_select(), D_min_gen);
            else if (r < 0.90)
                result = mutate_block(tournament_select(), block_size);
            else {
                std::geometric_distribution<int> geo(0.4);
                int rand_D = D_min_gen + std::min(geo(gen), D_max_gen - D_min_gen);
                result = generate_valid_sequence(rand_D);
            }
        } else {
            if (r < 0.25)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.35)
                result = mutate_swap(tournament_select());
            else if (r < 0.50)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk);
            else if (r < 0.70) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk);
            }
            else if (r < 0.80)
                result = crossover_uniform(tournament_select(), tournament_select());
            else if (r < 0.85)
                result = mutate_block(tournament_select(), block_size);
            else {
                int D = static_cast<int>(population[0].size());
                result = generate_valid_sequence(D);
            }
        }

        if (result.size() > 0 && seen.find(result) == seen.end() &&
            (!mixed_d || (static_cast<int>(result.size()) >= D_min_gen &&
                          static_cast<int>(result.size()) <= D_max_gen))) {
            seen.insert(result.copy());
            candidates_out.push_back(std::move(result));
        }
    }

    n_local_out = n_local_found;
    avg_steps_out = (n_local_found > 0) ?
        static_cast<double>(total_local_steps) / n_local_found : 0.0;
}


// ============================================================================
// Gate structure building (following Tree_Search pattern)
// ============================================================================

Gates_block* N_Qubit_Decomposition_Surrogate::construct_gate_structure(
    const GrayCode& gcode, bool finalize) {
    Gates_block* gate_structure = new Gates_block(qbit_num);

    for (int i = 0; i < static_cast<int>(gcode.size()); ++i) {
        int token = gcode[i];
        // 2-qubit block in edge-based mode (U3+U3+CNOT)
        int target = possible_target_qbits[token];
        int control = possible_control_qbits[token];
        add_two_qubit_block(gate_structure, target, control);
    }

    if (finalize)
        add_finalyzing_layer(gate_structure);

    return gate_structure;
}

void N_Qubit_Decomposition_Surrogate::add_two_qubit_block(
    Gates_block* gate_structure, int target_qbit, int control_qbit) {
    Gates_block* layer = new Gates_block(qbit_num);
    layer->add_u3(target_qbit);
    layer->add_u3(control_qbit);
    layer->add_cnot(target_qbit, control_qbit);
    gate_structure->add_gate(layer);
}

void N_Qubit_Decomposition_Surrogate::add_single_qubit_gate(
    Gates_block* gate_structure, int target_qbit, gate_type gtype) {
    Gates_block* layer = new Gates_block(qbit_num);
    switch (gtype) {
        case U3_OPERATION:   layer->add_u3(target_qbit);   break;
        case RY_OPERATION:   layer->add_ry(target_qbit);   break;
        case RX_OPERATION:   layer->add_rx(target_qbit);   break;
        case RZ_OPERATION:   layer->add_rz(target_qbit);   break;
        case H_OPERATION:    layer->add_h(target_qbit);    break;
        case X_OPERATION:    layer->add_x(target_qbit);    break;
        case Y_OPERATION:    layer->add_y(target_qbit);    break;
        case Z_OPERATION:    layer->add_z(target_qbit);    break;
        case S_OPERATION:    layer->add_s(target_qbit);    break;
        case SDG_OPERATION:  layer->add_sdg(target_qbit);  break;
        case T_OPERATION:    layer->add_t(target_qbit);    break;
        case TDG_OPERATION:  layer->add_tdg(target_qbit);  break;
        case SX_OPERATION:   layer->add_sx(target_qbit);   break;
        case SXDG_OPERATION: layer->add_sxdg(target_qbit); break;
        case U1_OPERATION:   layer->add_u1(target_qbit);   break;
        case U2_OPERATION:   layer->add_u2(target_qbit);   break;
        case R_OPERATION:    layer->add_r(target_qbit);    break;
        default:             layer->add_u3(target_qbit);   break;
    }
    gate_structure->add_gate(layer);
}

std::tuple<int,int,int> N_Qubit_Decomposition_Surrogate::token_sort_key(int token) const {
    // Edge-mode: (type=0, target, control)
    return std::make_tuple(0, topology[token][0], topology[token][1]);
}

void N_Qubit_Decomposition_Surrogate::add_finalyzing_layer(Gates_block* gate_structure) {
    Gates_block* block = new Gates_block(qbit_num);
    for (int idx = 0; idx < qbit_num; idx++)
        block->add_u3(idx);
    gate_structure->add_gate(block);
}


// ============================================================================
// Circuit decomposition
// ============================================================================

std::pair<double, Matrix_real> N_Qubit_Decomposition_Surrogate::decompose(
    const GrayCode& circuit) {
    return decompose_with_rng(circuit, gen);
}

std::pair<double, Matrix_real> N_Qubit_Decomposition_Surrogate::decompose_with_rng(
    const GrayCode& circuit, std::mt19937& local_gen) {

    Gates_block* gate_structure = construct_gate_structure(circuit);
    int param_num = gate_structure->get_parameter_num();

    N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp.set_custom_gate_structure(gate_structure);
    delete gate_structure;  // set_custom_gate_structure clones the gates; free the original
    cDecomp.set_verbose(0);
    cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
    cDecomp.set_optimization_tolerance(tolerance);
    cDecomp.set_optimizer(alg);

    // Random initial parameters
    Matrix_real random_params(1, param_num);
    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);
    for (int i = 0; i < param_num; ++i)
        random_params[i] = param_dist(local_gen);
    cDecomp.set_optimized_parameters(random_params.get_data(), param_num);

    cDecomp.start_decomposition();

    Matrix_real params = cDecomp.get_optimized_parameters();
    double score = cDecomp.optimization_problem(params);

    return {score, params};
}

void N_Qubit_Decomposition_Surrogate::parallel_decompose_batch(
    const std::vector<GrayCode>& circuits,
    std::vector<DecompResult>& results) {

    int n = static_cast<int>(circuits.size());
    results.resize(n);
    if (n == 0) return;

    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = std::min(static_cast<int64_t>(nthreads), static_cast<int64_t>(n));
    int parallel = get_parallel_configuration();
    int64_t work_batch = (parallel == 0) ? concurrency : 1;

    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, concurrency, work_batch),
        [&](tbb::blocked_range<int64_t> r) {
            std::mt19937 local_gen(std::random_device{}());

            for (int64_t job_idx = r.begin(); job_idx < r.end(); ++job_idx) {
                int64_t batch_sz = n / concurrency;
                int64_t start = job_idx * batch_sz;
                int64_t end = (job_idx == concurrency - 1) ? n : start + batch_sz;

                for (int64_t i = start; i < end; ++i) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto res = decompose_with_rng(circuits[i], local_gen);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    results[i].score = res.first;
                    results[i].params = res.second;
                    results[i].elapsed = std::chrono::duration<double>(t1 - t0).count();
                }
            }
        });
}


// ============================================================================
// Standalone SSK Gram matrix
// ============================================================================

void N_Qubit_Decomposition_Surrogate::ssk_gram_matrix(
    const std::vector<GrayCode>& circuits, double gap_decay_in,
    double match_decay_in, int order_in, double* K_out) {

    SSKCache temp_cache(gap_decay_in, match_decay_in, order_in);
    int n = static_cast<int>(circuits.size());

    // Register all circuits
    for (int i = 0; i < n; ++i)
        temp_cache.register_circuit(circuits[i]);

    // Extract normalized kernel matrix
    double scale = 1.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            K_out[i * n + j] = temp_cache.K_norm[i * temp_cache.capacity_ + j];
}


// ============================================================================
// Main search methods
// ============================================================================

void N_Qubit_Decomposition_Surrogate::search_over_D_range(
    int D_min_search, int D_max_search, const std::string& log_file) {

    auto search_start = std::chrono::high_resolution_clock::now();
    decompose_time = 0.0;
    decompose_count = 0;

    // Clamp D_min up to OSR lower bound
    if (D_min_search < osr_D_min)
        D_min_search = osr_D_min;

    best_score = std::numeric_limits<double>::infinity();

    // Log header
    {
        std::ofstream flog(log_file);
        flog << "SurSearch (C++ sequential-D): N=" << qbit_num
             << ", D=" << D_min_search << "-" << D_max_search
             << ", kappa=" << kappa << std::endl;
        flog << "---" << std::endl;
    }

    // Persistent state across all D values (but SSKCache is rebuilt per D)
    GrayCodeSet seen;
    std::vector<GrayCode> X;
    std::vector<double> y;
    std::vector<Matrix_real> all_params;

    for (int D = D_min_search; D <= D_max_search; ++D) {

        // --- Grow-seed D from D-1 circuits ---
        bool early_solution = false;
        int n_grown = 0;

        // Count existing samples at this D
        int existing_at_D = 0;
        for (size_t i = 0; i < X.size(); ++i)
            if (static_cast<int>(X[i].size()) == D)
                existing_at_D++;

        int needed = std::max(0, X0_size - existing_at_D);

        if (needed > 0) {
            // Gather D-1 circuits sorted by score (best first)
            std::vector<int> prev_indices;
            for (size_t i = 0; i < X.size(); ++i)
                if (static_cast<int>(X[i].size()) == D - 1)
                    prev_indices.push_back(static_cast<int>(i));

            if (!prev_indices.empty()) {
                std::sort(prev_indices.begin(), prev_indices.end(),
                          [&](int a, int b) { return y[a] < y[b]; });

                // Phase A: batch-generate grown circuits
                std::vector<GrayCode> grow_batch;
                for (int pi : prev_indices) {
                    if (static_cast<int>(grow_batch.size()) >= needed) break;
                    for (int attempt = 0; attempt < 10; ++attempt) {
                        GrayCode grown = mutate_grow(X[pi], D);
                        if (grown.size() == 0) continue;
                        if (static_cast<int>(grown.size()) != D) continue;
                        if (seen.find(grown) != seen.end()) continue;

                        seen.insert(grown.copy());
                        grow_batch.push_back(std::move(grown));
                        break;
                    }
                }

                // Phase B: parallel decompose
                std::vector<DecompResult> grow_results;
                parallel_decompose_batch(grow_batch, grow_results);

                // Phase C: sequential state update
                for (int gi = 0; gi < static_cast<int>(grow_batch.size()); ++gi) {
                    decompose_time += grow_results[gi].elapsed;
                    decompose_count++;

                    y.push_back(grow_results[gi].score);
                    all_params.push_back(grow_results[gi].params);
                    X.push_back(std::move(grow_batch[gi]));
                    n_grown++;

                    if (y.back() < best_score) {
                        best_score = y.back();
                        best_circuit = X.back().copy();
                        best_params = all_params.back();
                    }
                    if (y.back() < tolerance) {
                        early_solution = true;
                    }
                    if (early_solution) break;
                }
            }
        }

        // Log D header
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "\n=== D=" << D << " (grow-seeded: " << n_grown << ") ===" << std::endl;
        }

        if (early_solution) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "SOLUTION found during grow-seeding at D="
                 << best_circuit.size() << ", score=" << best_score << std::endl;
            break;
        }

        // --- Build fresh SSKCache for this D ---
        // Register only the top 2*gp_max_train circuits by score (bounded cache)
        int d_filter = std::max(1, D - 1);
        std::vector<int> cache_candidates;
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            if (static_cast<int>(X[i].size()) >= d_filter)
                cache_candidates.push_back(i);
        }
        std::sort(cache_candidates.begin(), cache_candidates.end(),
                  [&y](int a, int b) { return y[a] < y[b]; });
        int cache_limit = std::min(static_cast<int>(cache_candidates.size()),
                                   2 * gp_max_train);

        SSKCache ssk_cache(sur_gap_decay, sur_match_decay, sur_ssk_order);
        for (int ci = 0; ci < cache_limit; ++ci)
            ssk_cache.register_circuit(X[cache_candidates[ci]]);

        // --- Run search at this D ---
        WindowResult result = run_window_search(D, D,
            ssk_cache, seen, X, y, all_params, log_file);

        if (result == WINDOW_SUCCESS) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "SOLUTION at D=" << best_circuit.size() << std::endl;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_search_time = std::chrono::duration<double>(end - search_start).count();

    {
        std::ofstream flog(log_file, std::ios::app);
        flog << "\nSearch complete: best=" << best_score
             << " (D=" << best_circuit.size() << ")"
             << " total_evals=" << decompose_count
             << " time=" << total_search_time << "s" << std::endl;
    }

    // Store results in base class
    optimized_parameters_mtx = best_params;
    current_minimum = best_score;
    decomposition_error = best_score;
}


// ============================================================================
// Per-window surrogate search
// ============================================================================

N_Qubit_Decomposition_Surrogate::WindowResult
N_Qubit_Decomposition_Surrogate::run_window_search(
    int win_lo, int win_hi,
    SSKCache& ssk_cache, GrayCodeSet& seen,
    std::vector<GrayCode>& X, std::vector<double>& y,
    std::vector<Matrix_real>& all_params,
    const std::string& log_file) {

    // Count existing data points within this window
    int existing_in_window = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        int d = static_cast<int>(X[i].size());
        if (d >= win_lo && d <= win_hi)
            existing_in_window++;
    }

    // Generate initial samples for this window if needed
    int needed = std::max(0, X0_size - existing_in_window);
    if (needed > 0) {
        std::vector<int> D_range;
        for (int d = win_lo; d <= win_hi; ++d)
            D_range.push_back(d);

        std::vector<double> d_weights(D_range.size());
        double w_sum = 0;
        for (size_t i = 0; i < D_range.size(); ++i) {
            d_weights[i] = 1.0 / D_range[i];
            w_sum += d_weights[i];
        }
        for (size_t i = 0; i < d_weights.size(); ++i) d_weights[i] /= w_sum;

        std::vector<double> d_cdf(D_range.size());
        d_cdf[0] = d_weights[0];
        for (size_t i = 1; i < D_range.size(); ++i)
            d_cdf[i] = d_cdf[i - 1] + d_weights[i];

        std::uniform_real_distribution<double> uni(0.0, 1.0);
        int generated = 0;
        int attempts = 0;
        while (generated < needed && attempts < needed * 100) {
            // Phase A: batch-generate circuits sequentially
            int batch_target = std::min(needed - generated,
                static_cast<int>(std::thread::hardware_concurrency()));
            std::vector<GrayCode> batch_circuits;
            batch_circuits.reserve(batch_target);

            while (static_cast<int>(batch_circuits.size()) < batch_target &&
                   attempts < needed * 100) {
                double r = uni(gen);
                int D_val = D_range.back();
                for (size_t i = 0; i < d_cdf.size(); ++i) {
                    if (r <= d_cdf[i]) { D_val = D_range[i]; break; }
                }
                GrayCode seq = generate_valid_sequence(D_val);
                attempts++;
                if (seq.size() > 0 && seen.find(seq) == seen.end()) {
                    seen.insert(seq.copy());
                    ssk_cache.register_circuit(seq);
                    batch_circuits.push_back(std::move(seq));
                }
            }

            if (batch_circuits.empty()) continue;

            // Phase B: parallel decompose
            std::vector<DecompResult> batch_results;
            parallel_decompose_batch(batch_circuits, batch_results);

            // Phase C: sequential state update
            for (int bi = 0; bi < static_cast<int>(batch_circuits.size()); ++bi) {
                decompose_time += batch_results[bi].elapsed;
                decompose_count++;
                y.push_back(batch_results[bi].score);
                all_params.push_back(batch_results[bi].params);
                X.push_back(std::move(batch_circuits[bi]));
                generated++;

                double score = y.back();
                if (score < best_score) {
                    best_score = score;
                    best_circuit = X.back().copy();
                    best_params = all_params.back();
                }

                if (score < tolerance) {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "Solution found during initial sampling (D="
                         << X.back().size() << ")" << std::endl;
                    return WINDOW_SUCCESS;
                }
            }
        }
    }

    if (X.empty()) return WINDOW_BUDGET;

    {
        std::ofstream flog(log_file, std::ios::app);
        flog << "Window [" << win_lo << "," << win_hi << "]: "
             << existing_in_window << " existing + " << needed << " new samples, "
             << "total data=" << X.size() << std::endl;
    }

    // Fresh GP for this window
    GPRegressor gp;
    gp.log_scale = 0.0;
    gp.noise = 1e-2;
    gp.jitter = 1e-8;

    std::pair<double,double> scale_bounds = {-3.0, 3.0};
    std::pair<double,double> noise_bounds = {-8.0, -1.0};

    int iters_since_improvement = 0;
    double prev_best_score = best_score;

    // Best-so-far plateau stagnation tracking
    std::vector<double> iter_bests;  // window-best score at end of each iteration
    iter_bests.reserve(window_max_iters);

    // Track best score within window D range for logging
    double window_best_score = std::numeric_limits<double>::infinity();
    int window_best_D = -1;
    for (size_t i = 0; i < X.size(); ++i) {
        int d = static_cast<int>(X[i].size());
        if (d >= win_lo && d <= win_hi && y[i] < window_best_score) {
            window_best_score = y[i];
            window_best_D = d;
        }
    }
    double prev_log_scale = gp.log_scale;
    double prev_noise = gp.noise;

    int d_filter_min = std::max(1, win_lo - 1);

    double saved_kappa = kappa;

    for (int itr = 0; itr < window_max_iters; ++itr) {

        // Adaptive kappa: decay over iterations, boost on stagnation
        if (adaptive_kappa && window_max_iters > 0) {
            double progress = static_cast<double>(itr) / window_max_iters;
            double base = 1.0 - kappa_decay_rate * progress;
            double stagnation = 0.0;
            if (iters_since_improvement > window_patience / 2) {
                stagnation = kappa_stagnation_boost *
                    std::min(1.0, static_cast<double>(iters_since_improvement - window_patience / 2)
                                  / std::max(1, window_patience / 2));
            }
            kappa = saved_kappa * (base + stagnation);
        }

        double t_gp = 0, t_cand = 0, t_acq = 0, t_dec = 0;

        std::vector<GrayCode> candidates;
        std::vector<int> selected_indices;

        if (use_random_candidates) {
            // ============================================================
            // Random baseline: evolutionary candidates + random selection
            // (no GP fitting, no local search, no Thompson sampling)
            // ============================================================
            auto t_phase0 = std::chrono::high_resolution_clock::now();

            // Temporarily disable GP-directed local search
            double saved_lsf = local_search_fraction;
            local_search_fraction = 0.0;

            GPRegressor dummy_gp;
            int n_local;
            double avg_steps;
            generate_candidates(X, y.data(), static_cast<int>(X.size()),
                                candidates_per_iter,
                                ssk_cache, dummy_gp, 1.0, nullptr,
                                seen, candidates, n_local, avg_steps,
                                win_lo, win_hi);

            local_search_fraction = saved_lsf;

            auto t_phase1 = std::chrono::high_resolution_clock::now();
            t_cand = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            if (candidates.empty()) break;

            int n_cand = static_cast<int>(candidates.size());

            // Random selection: uniformly pick n_thompson_samples candidates
            std::vector<int> perm(n_cand);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), gen);
            int n_pick = std::min(n_thompson_samples, n_cand);
            for (int i = 0; i < n_pick; ++i)
                selected_indices.push_back(perm[i]);

        } else {
            // ============================================================
            // Surrogate-assisted search: GP + Thompson sampling
            // ============================================================

            // Build filtered pool sorted by score, register top 2*gp_max_train
            std::vector<int> filtered_x_indices;
            for (int i = 0; i < static_cast<int>(X.size()); ++i) {
                if (static_cast<int>(X[i].size()) >= d_filter_min)
                    filtered_x_indices.push_back(i);
            }
            int n_filtered = static_cast<int>(filtered_x_indices.size());
            if (n_filtered == 0) break;

            std::sort(filtered_x_indices.begin(), filtered_x_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });

            // Register top 2*gp_max_train by score (lazy, bounded)
            int reg_limit = std::min(n_filtered, 2 * gp_max_train);
            for (int i = 0; i < reg_limit; ++i) {
                int xi = filtered_x_indices[i];
                if (ssk_cache.circuit_to_idx.find(X[xi]) == ssk_cache.circuit_to_idx.end())
                    ssk_cache.register_circuit(X[xi]);
            }

            // Cap GP training set at gp_max_train using hybrid selection:
            // 50% best-by-score + 50% pivoted Cholesky (diversity)
            // Only use the registered subset for pivoted Cholesky
            int n_pool = reg_limit;  // work within registered circuits

            std::vector<int> selected_fi;  // selected indices into filtered_x_indices
            if (n_pool <= gp_max_train) {
                selected_fi.resize(n_pool);
                for (int i = 0; i < n_pool; ++i) selected_fi[i] = i;
            } else {
                int n_best = gp_max_train / 2;
                int n_total = gp_max_train;
                selected_fi.reserve(n_total);
                std::vector<bool> taken(n_pool, false);

                // Stage 1: top n_best by score (already sorted)
                for (int i = 0; i < n_best; ++i) {
                    selected_fi.push_back(i);
                    taken[i] = true;
                }

                // Stage 2: pivoted Cholesky for kernel-diverse points
                std::vector<int> cidx(n_pool);
                for (int i = 0; i < n_pool; ++i)
                    cidx[i] = ssk_cache.circuit_to_idx[X[filtered_x_indices[i]]];
                int cap = ssk_cache.capacity_;

                // L(i,j) Cholesky factor, d[i] residual diagonal
                std::vector<double> L_data(static_cast<size_t>(n_pool) * n_total, 0.0);
                std::vector<double> d(n_pool, 1.0);

                for (int j = 0; j < n_total; ++j) {
                    int pj;
                    if (j < n_best) {
                        pj = selected_fi[j];
                    } else {
                        pj = -1;
                        double max_d = -1;
                        for (int i = 0; i < n_pool; ++i)
                            if (!taken[i] && d[i] > max_d) { max_d = d[i]; pj = i; }
                        if (pj < 0 || max_d < 1e-12) break;
                        selected_fi.push_back(pj);
                        taken[pj] = true;
                    }

                    double L_diag = std::sqrt(std::max(d[pj], 1e-12));
                    L_data[pj * n_total + j] = L_diag;

                    for (int i = 0; i < n_pool; ++i) {
                        if (i == pj) continue;
                        if (j >= n_best && taken[i]) continue;
                        double K_val = ssk_cache.K_norm[cidx[i] * cap + cidx[pj]];
                        double dot = 0;
                        for (int k = 0; k < j; ++k)
                            dot += L_data[i * n_total + k] * L_data[pj * n_total + k];
                        double L_ij = (K_val - dot) / L_diag;
                        L_data[i * n_total + j] = L_ij;
                        if (!taken[i]) d[i] = std::max(d[i] - L_ij * L_ij, 0.0);
                    }
                    d[pj] = 0;
                }
            }

            int n_x = static_cast<int>(selected_fi.size());
            std::vector<int> train_indices(n_x);
            std::vector<double> train_y(n_x);
            for (int i = 0; i < n_x; ++i) {
                int xi = filtered_x_indices[selected_fi[i]];
                train_indices[i] = ssk_cache.circuit_to_idx[X[xi]];
                train_y[i] = y[xi];
            }

            // Log10 transform + normalize
            std::vector<double> log_y(n_x);
            double min_nonzero = std::numeric_limits<double>::infinity();
            for (int i = 0; i < n_x; ++i)
                if (train_y[i] > 0) min_nonzero = std::min(min_nonzero, train_y[i]);
            double dynamic_floor = (min_nonzero < std::numeric_limits<double>::infinity()) ?
                min_nonzero * 0.25 : tolerance * 0.25;

            double lmu = 0, lsig = 0;
            for (int i = 0; i < n_x; ++i) {
                log_y[i] = std::log10(std::max(train_y[i], dynamic_floor));
                lmu += log_y[i];
            }
            lmu /= n_x;
            for (int i = 0; i < n_x; ++i)
                lsig += (log_y[i] - lmu) * (log_y[i] - lmu);
            lsig = std::sqrt(lsig / n_x);
            if (lsig < 1e-6) lsig = 1e-6;

            std::vector<double> log_y_norm(n_x);
            for (int i = 0; i < n_x; ++i)
                log_y_norm[i] = (log_y[i] - lmu) / lsig;

            // --- GP phase ---
            auto t_phase0 = std::chrono::high_resolution_clock::now();

            // Optimize GP hyperparameters
            int n_restarts = std::max(1, 3 - n_x / 10);
            gp.optimize_hyperparameters(ssk_cache, train_indices.data(), n_x,
                                        log_y_norm.data(), n_restarts,
                                        scale_bounds, noise_bounds);

            // Fit GP — use incremental Cholesky only when hyperparameters are stable
            // AND the first old_n training indices are unchanged (the training set is
            // rebuilt from scratch each iteration via score-sort + pivoted Cholesky,
            // so the prefix can change even when n_x grows).
            int old_n_train = gp.n_train;
            bool hp_changed = (gp.log_scale != prev_log_scale || gp.noise != prev_noise);

            bool prefix_matches = false;
            if (!hp_changed && old_n_train > 0 && old_n_train < n_x) {
                prefix_matches = true;
                for (int i = 0; i < old_n_train; ++i) {
                    if (train_indices[i] != gp.train_indices_[i]) {
                        prefix_matches = false;
                        break;
                    }
                }
            }

            if (prefix_matches) {
                gp.fit_incremental(ssk_cache, train_indices.data(), n_x,
                                   old_n_train, log_y_norm.data());
            } else {
                gp.fit(ssk_cache, train_indices.data(), n_x, log_y_norm.data());
            }
            prev_log_scale = gp.log_scale;
            prev_noise = gp.noise;

            // Collapse detection
            double opt_scale = gp.get_scale();
            if (opt_scale / std::max(gp.noise, 1e-12) < 0.1) {
                gp.log_scale = 0.0;
                gp.noise = 1e-2;
                gp.fit(ssk_cache, train_indices.data(), n_x, log_y_norm.data());
                prev_log_scale = gp.log_scale;
                prev_noise = gp.noise;
            }

            double scale = gp.get_scale();

            auto t_phase1 = std::chrono::high_resolution_clock::now();
            t_gp = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            // --- Candidate generation phase ---
            t_phase0 = std::chrono::high_resolution_clock::now();

            // Generate candidates restricted to this window's D range
            int n_local;
            double avg_steps;
            generate_candidates(X, y.data(), static_cast<int>(X.size()),
                                candidates_per_iter,
                                ssk_cache, gp, scale, log_y_norm.data(),
                                seen, candidates, n_local, avg_steps,
                                win_lo, win_hi);

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_cand = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            if (candidates.empty()) break;

            int n_cand = static_cast<int>(candidates.size());

            // --- Acquisition phase ---
            t_phase0 = std::chrono::high_resolution_clock::now();

            int n_t = gp.n_train;

            // Screen with GP: compute cross-kernel only for training subset
            std::vector<int> train_indices_vec(train_indices.begin(),
                                               train_indices.begin() + n_x);
            std::vector<double> Ks(n_t * n_cand);
            ssk_cache.compute_cross_kernel_subset(candidates, scale,
                                                  train_indices_vec, Ks.data());

            // Compute mu and approximate std for all candidates (cheap diagonal approx)
            std::vector<double> log_mu_norm(n_cand);
            std::vector<double> approx_lcb(n_cand);
            for (int j = 0; j < n_cand; ++j) {
                double mu_val = 0;
                double v2_sum = 0;
                for (int i = 0; i < n_t; ++i) {
                    double ks_ij = Ks[i * n_cand + j];
                    mu_val += ks_ij * gp.alpha_data[i];
                    double v_approx = ks_ij * gp.inv_L_diag[i];
                    v2_sum += v_approx * v_approx;
                }
                log_mu_norm[j] = mu_val;
                double mu_j = mu_val;
                if (d_penalty > 0 && win_hi > 0) {
                    double D_val = static_cast<double>(candidates[j].size());
                    mu_j += d_penalty * (D_val / win_hi);
                }
                double std_approx = std::sqrt(std::max(scale - v2_sum, 0.0));
                approx_lcb[j] = mu_j - kappa * std_approx;
            }

            // Pre-filter: keep top n_shortlist by approximate LCB
            // Use 4x the pick count to leave room for diversity filtering
            int n_pick = std::min(n_thompson_samples, n_cand);
            int n_shortlist = std::min(n_cand, std::max(n_pick * 4, 200));

            std::vector<int> approx_order(n_cand);
            std::iota(approx_order.begin(), approx_order.end(), 0);
            std::partial_sort(approx_order.begin(),
                              approx_order.begin() + n_shortlist,
                              approx_order.end(),
                              [&](int a, int b) { return approx_lcb[a] < approx_lcb[b]; });

            // Build shortlist mapping: shortlist position -> original candidate index
            std::vector<int> shortlist(approx_order.begin(),
                                       approx_order.begin() + n_shortlist);

            // Exact forward solve only for shortlisted candidates
            // Extract Ks columns for shortlist into compact array
            std::vector<double> Ks_short(n_t * n_shortlist);
            for (int s = 0; s < n_shortlist; ++s) {
                int j = shortlist[s];
                for (int i = 0; i < n_t; ++i)
                    Ks_short[i * n_shortlist + s] = Ks[i * n_cand + j];
            }

            // Free full Ks — no longer needed
            { std::vector<double>().swap(Ks); }

            // Solve L v = Ks_short for exact variance
            std::vector<double> v(n_t * n_shortlist);
            std::memcpy(v.data(), Ks_short.data(), n_t * n_shortlist * sizeof(double));
            LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                           n_t, n_shortlist, gp.L_data.data(), n_t,
                           v.data(), n_shortlist);

            // Compute exact LCB for shortlist
            std::vector<double> lcb_scores(n_shortlist);
            for (int s = 0; s < n_shortlist; ++s) {
                double sv2 = 0;
                for (int i = 0; i < n_t; ++i)
                    sv2 += v[i * n_shortlist + s] * v[i * n_shortlist + s];
                double std_val = std::sqrt(std::max(scale - sv2, 0.0));
                double mu_j = log_mu_norm[shortlist[s]];
                if (d_penalty > 0 && win_hi > 0) {
                    double D_val = static_cast<double>(candidates[shortlist[s]].size());
                    mu_j += d_penalty * (D_val / win_hi);
                }
                lcb_scores[s] = mu_j - kappa * std_val;
            }

            // Sort shortlist by exact LCB ascending
            std::vector<int> lcb_order(n_shortlist);
            std::iota(lcb_order.begin(), lcb_order.end(), 0);
            std::sort(lcb_order.begin(), lcb_order.end(),
                      [&](int a, int b) { return lcb_scores[a] < lcb_scores[b]; });

            // Precompute Ks_short column norms for cosine diversity
            std::vector<double> ks_col_norm(n_shortlist, 0.0);
            for (int s = 0; s < n_shortlist; ++s) {
                double sq = 0;
                for (int i = 0; i < n_t; ++i) {
                    double val = Ks_short[i * n_shortlist + s];
                    sq += val * val;
                }
                ks_col_norm[s] = std::sqrt(sq);
            }

            // Build reverse map: candidate index -> shortlist position
            std::unordered_map<int,int> cand_to_short;
            cand_to_short.reserve(n_shortlist);
            for (int s = 0; s < n_shortlist; ++s)
                cand_to_short[shortlist[s]] = s;

            // Greedy top-k selection with cosine diversity filter
            for (int rank = 0; rank < n_shortlist && static_cast<int>(selected_indices.size()) < n_pick; ++rank) {
                int sidx = lcb_order[rank];  // index into shortlist
                int cidx = shortlist[sidx];  // index into candidates
                bool too_similar = false;
                if (!selected_indices.empty() && diversity_thresh < 1.0) {
                    double norm_c = ks_col_norm[sidx];
                    if (norm_c < 1e-12) {
                        // Zero-norm candidate: accept (maximally different)
                    } else {
                        for (int prev_cidx : selected_indices) {
                            auto it = cand_to_short.find(prev_cidx);
                            if (it == cand_to_short.end()) continue;
                            int prev_sidx = it->second;
                            double norm_p = ks_col_norm[prev_sidx];
                            if (norm_p < 1e-12) continue;
                            double dot = 0;
                            for (int i = 0; i < n_t; ++i)
                                dot += Ks_short[i * n_shortlist + sidx] * Ks_short[i * n_shortlist + prev_sidx];
                            double cosine = dot / (norm_c * norm_p);
                            if (cosine > diversity_thresh) {
                                too_similar = true;
                                break;
                            }
                        }
                    }
                }
                if (!too_similar) {
                    selected_indices.push_back(cidx);
                }
            }

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_acq = std::chrono::duration<double>(t_phase1 - t_phase0).count();
        } // end if/else use_random_candidates

        // --- Decompose phase (shared, parallelized) ---
        auto t_phase0 = std::chrono::high_resolution_clock::now();

        int n_sel = static_cast<int>(selected_indices.size());
        bool improved_this_iter = false;
        bool found_solution = false;

        // Phase A: parallel decompose
        int n_cand_total = static_cast<int>(candidates.size());
        std::vector<GrayCode> sel_circuits(n_sel);
        for (int si = 0; si < n_sel; ++si) {
            int idx = selected_indices[si];
            if (idx < 0 || idx >= n_cand_total) {
                std::cerr << "BUG: selected_indices[" << si << "]=" << idx
                          << " out of range [0," << n_cand_total << ")" << std::endl;
                std::abort();
            }
            sel_circuits[si] = candidates[idx].copy();
        }

        std::vector<DecompResult> dec_results;
        parallel_decompose_batch(sel_circuits, dec_results);

        // Phase B: sequential state update
        for (int si = 0; si < n_sel; ++si) {
            int sel_idx = selected_indices[si];
            double new_score = dec_results[si].score;
            Matrix_real new_params = dec_results[si].params;
            decompose_time += dec_results[si].elapsed;
            decompose_count++;

            y.push_back(new_score);
            all_params.push_back(new_params);
            X.push_back(candidates[sel_idx].copy());

            // Update window-local best
            int cand_size = static_cast<int>(candidates[sel_idx].size());
            if (cand_size >= win_lo && cand_size <= win_hi && new_score < window_best_score) {
                window_best_score = new_score;
                window_best_D = cand_size;
            }

            if (new_score < best_score ||
                (new_score < tolerance &&
                 static_cast<int>(candidates[sel_idx].size()) < static_cast<int>(best_circuit.size()))) {
                best_score = new_score;
                best_circuit = candidates[sel_idx].copy();
                best_params = new_params;
                improved_this_iter = true;
            }

            if (new_score < tolerance) {
                found_solution = true;
                break;
            }
        }

        auto t_phase1 = std::chrono::high_resolution_clock::now();
        t_dec = std::chrono::duration<double>(t_phase1 - t_phase0).count();

        if (found_solution) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Iter " << itr << ": SOLUTION at D=" << best_circuit.size()
                 << ", score=" << best_score
                 << (use_random_candidates ? " [RANDOM]" : "")
                 << " [cand=" << std::fixed << std::setprecision(2) << t_cand
                 << "s dec=" << t_dec << "s]"
                 << std::defaultfloat << std::endl;
            kappa = saved_kappa;
            return WINDOW_SUCCESS;
        }

        // Record window-best at end of this iteration for plateau detection
        iter_bests.push_back(window_best_score);

        // Patience fallback: reset on meaningful best improvement
        if (improved_this_iter && (prev_best_score - best_score) > prev_best_score * 0.25) {
            iters_since_improvement = 0;
            prev_best_score = best_score;
        } else {
            iters_since_improvement++;
        }

        // Log — show window-local best score
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Iter " << itr
                 << ": best=" << window_best_score
                 << " (D=" << window_best_D << ")"
                 << " evals=" << X.size()
                 << (use_random_candidates ? " [RANDOM]" : "")
                 << " kappa=" << std::setprecision(3) << kappa
                 << " [gp=" << std::setprecision(2) << t_gp
                 << "s cand=" << t_cand
                 << "s acq=" << t_acq
                 << "s dec=" << t_dec << "s]"
                 << std::defaultfloat
                 << std::endl;
        }

        // Best-so-far plateau stagnation check
        int n_iters_done = static_cast<int>(iter_bests.size());
        if (n_iters_done >= stagnation_window) {
            double old_best = iter_bests[n_iters_done - stagnation_window];
            double cur_best = iter_bests.back();
            // Stagnation: best hasn't improved by at least improvement_frac over last stagnation_window iters
            if (cur_best >= old_best * (1.0 - stagnation_improvement_frac)) {
                std::ofstream flog(log_file, std::ios::app);
                flog << "  Best-so-far plateau after " << itr + 1 << " iters"
                     << " (best " << stagnation_window << " iters ago: " << old_best
                     << ", now: " << cur_best << ", required "
                     << std::fixed << std::setprecision(1) << (stagnation_improvement_frac * 100)
                     << "% improvement)" << std::defaultfloat << std::endl;
                kappa = saved_kappa;
                return WINDOW_STAGNATION;
            }
        }

        // Hard patience cap as fallback
        if (iters_since_improvement >= window_patience) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Patience stagnation after " << itr + 1 << " iters" << std::endl;
            kappa = saved_kappa;
            return WINDOW_STAGNATION;
        }
    }

    kappa = saved_kappa;
    return WINDOW_BUDGET;
}

void N_Qubit_Decomposition_Surrogate::search_over_D_evolve(
    int D, const std::string& log_file) {
    // Single-D search is a special case of range search
    search_over_D_range(D, D, log_file);
}

void N_Qubit_Decomposition_Surrogate::start_decomposition() {
    std::stringstream sstream;
    sstream << "Starting " << (use_random_candidates ? "RANDOM baseline" : "surrogate")
            << " search for " << qbit_num << "-qubit matrix" << std::endl;
    print(sstream, 1);

    // Temporarily turn off OpenMP parallelism

	force_single_thread_blas_lapack();

    int D_start = osr_D_min;
    int D_end = level_limit;
    if (D_end <= 0) D_end = D_start + 10;  // default range

    std::string log_prefix = project_name.empty() ? "sursearch" : "sursearch_" + project_name;
    std::string log_file = log_prefix + "_D" + std::to_string(D_start) + "-" +
                            std::to_string(D_end) + ".txt";

    search_over_D_range(D_start, D_end, log_file);

    // Construct the final gate structure from best circuit
    if (best_circuit.size() > 0) {
        Gates_block* gate_structure = construct_gate_structure(best_circuit);
        // Store in base class
        release_gates();
        combine(gate_structure);
        delete gate_structure;  // combine clones the gates; free the original
        optimized_parameters_mtx = best_params;
        decomposition_error = best_score;
    }


}

Gates_block* N_Qubit_Decomposition_Surrogate::determine_gate_structure(
    Matrix_real& optimized_parameters_mtx_loc) {
    // This is called by start_decomposition in the base class pattern
    // For surrogate search, the structure is determined during search
    if (best_circuit.size() > 0) {
        return construct_gate_structure(best_circuit);
    }
    return nullptr;
}

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
    n_thompson_samples = 10;
    diversity_thresh = 0.95;
    d_penalty = 0.0;
    enum_threshold = 10000;
    gp_max_train = 300;
    d_seed_budget = 50;
    d_window_width = 2;
    base_window_width = 1;
    window_patience = 50;
    window_max_iters = 100;
    max_consecutive_stagnations = 3;
    use_random_candidates = false;

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
    if (config.count("d_window_width") > 0) {
        long long v; config["d_window_width"].get_property(v);
        d_window_width = static_cast<int>(v);
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
    if (config.count("max_consecutive_stagnations") > 0) {
        long long v; config["max_consecutive_stagnations"].get_property(v);
        max_consecutive_stagnations = static_cast<int>(v);
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

    // Rollback-phase defaults (inherit from main config, with higher patience)
    rb_kappa = kappa;
    rb_window_patience = window_patience * 2;
    rb_window_max_iters = window_max_iters * 2;
    rb_candidates_per_iter = candidates_per_iter;
    rb_n_thompson_samples = n_thompson_samples;
    rb_local_search_fraction = local_search_fraction;
    rb_max_local_steps = max_local_steps;

    // Override rollback config from user
    if (config.count("rb_kappa") > 0) {
        double v; config["rb_kappa"].get_property(v); rb_kappa = v;
    }
    if (config.count("rb_window_patience") > 0) {
        long long v; config["rb_window_patience"].get_property(v);
        rb_window_patience = static_cast<int>(v);
    }
    if (config.count("rb_window_max_iters") > 0) {
        long long v; config["rb_window_max_iters"].get_property(v);
        rb_window_max_iters = static_cast<int>(v);
    }
    if (config.count("rb_candidates_per_iter") > 0) {
        long long v; config["rb_candidates_per_iter"].get_property(v);
        rb_candidates_per_iter = static_cast<int>(v);
    }
    if (config.count("rb_n_thompson_samples") > 0) {
        long long v; config["rb_n_thompson_samples"].get_property(v);
        rb_n_thompson_samples = static_cast<int>(v);
    }
    if (config.count("rb_local_search_fraction") > 0) {
        double v; config["rb_local_search_fraction"].get_property(v);
        rb_local_search_fraction = v;
    }
    if (config.count("rb_max_local_steps") > 0) {
        long long v; config["rb_max_local_steps"].get_property(v);
        rb_max_local_steps = static_cast<int>(v);
    }

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

    // Build dependency DAG using edge bitmasks
    std::vector<std::vector<int>> adj(n);
    std::vector<int> in_degree(n, 0);

    // Get masks for each position
    std::vector<int> masks(n);
    for (int i = 0; i < n; ++i)
        masks[i] = edge_masks[seq[i]];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (masks[j] & masks[i]) {
                adj[j].push_back(i);
                in_degree[i]++;
            }
        }
    }

    // Min-heap topological sort: (edge_pair, index)
    struct HeapNode {
        std::pair<int,int> edge;
        int idx;
        bool operator>(const HeapNode& o) const {
            if (edge != o.edge) return edge > o.edge;
            return idx > o.idx;
        }
    };
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> heap;
    for (int i = 0; i < n; ++i) {
        if (in_degree[i] == 0) {
            heap.push(HeapNode{
                std::make_pair(topology[seq[i]][0], topology[seq[i]][1]), i});
        }
    }

    matrix_base<int> limits(1, n);
    for (int i = 0; i < n; ++i)
        limits[i] = static_cast<int>(topology.size());
    GrayCode result(limits);

    int pos = 0;
    while (!heap.empty()) {
        HeapNode top = heap.top();
        heap.pop();
        result[pos++] = seq[top.idx];
        for (int neighbor : adj[top.idx]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                heap.push(HeapNode{
                    std::make_pair(topology[seq[neighbor]][0], topology[seq[neighbor]][1]),
                    neighbor});
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
        masks[i] = edge_masks[canon[i]];

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
    for (int i = 0; i < static_cast<int>(circuit.size()); ++i)
        edge_counts[circuit[i]]++;

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
    int n_edges = static_cast<int>(topology.size());
    std::vector<GrayCode> results;
    GrayCodeSet seen;

    // DFS enumeration with pruning
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_edges;

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

        for (int e = 0; e < n_edges; ++e) {
            path[depth] = e;
            path_masks[depth] = edge_masks[e];

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
    int n_edges = static_cast<int>(topology.size());

    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_edges;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    std::uniform_int_distribution<int> edge_dist(0, n_edges - 1);

    for (int depth = 0; depth < D; ++depth) {
        std::vector<int> valid;
        for (int e = 0; e < n_edges; ++e) {
            path[depth] = e;
            path_masks[depth] = edge_masks[e];
            if (!check_new_position(path_masks.data(), depth))
                valid.push_back(e);
        }
        if (valid.empty()) return GrayCode();  // failed

        std::uniform_int_distribution<int> pick(0, static_cast<int>(valid.size()) - 1);
        int chosen = valid[pick(gen)];
        path[depth] = chosen;
        path_masks[depth] = edge_masks[chosen];
    }

    return canonicalize_and_validate(path);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_point(const GrayCode& seq) {
    int D = static_cast<int>(seq.size());
    int n_edges = static_cast<int>(topology.size());
    if (n_edges < 2) return GrayCode();  // cannot substitute with only one edge
    std::uniform_int_distribution<int> pos_dist(0, D - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int pos = pos_dist(gen);
        int cur_edge = seq[pos];
        int new_edge;

        // 70% chance: pick from neighboring edges (share a qubit) for higher acceptance
        // 30% chance: pick any other edge for exploration
        const std::vector<int>& nbrs = edge_neighbors[cur_edge];
        if (!nbrs.empty() && coin(gen) < 0.7) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_edge = nbrs[nbr_dist(gen)];
        } else {
            std::uniform_int_distribution<int> edge_dist(0, n_edges - 2);
            new_edge = edge_dist(gen);
            if (new_edge >= cur_edge) ++new_edge;  // skip current edge
        }

        GrayCode new_seq = seq.copy();
        new_seq[pos] = new_edge;
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
    int n_edges = static_cast<int>(topology.size());
    int bs = std::min(blk_size, D);
    std::uniform_int_distribution<int> start_dist(0, D - bs);
    std::uniform_int_distribution<int> edge_dist(0, n_edges - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        GrayCode new_seq = seq.copy();
        int start = start_dist(gen);
        for (int p = start; p < start + bs; ++p) {
            int cur_edge = seq[p];
            const std::vector<int>& nbrs = edge_neighbors[cur_edge];
            if (!nbrs.empty() && coin(gen) < 0.5) {
                std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
                new_seq[p] = nbrs[nbr_dist(gen)];
            } else {
                new_seq[p] = edge_dist(gen);
            }
        }
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
    int n_edges = static_cast<int>(topology.size());
    std::uniform_int_distribution<int> pos_dist(0, D);
    std::uniform_int_distribution<int> edge_dist(0, n_edges - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int attempt = 0; attempt < 50; ++attempt) {
        int pos = pos_dist(gen);
        int edge;
        // Bias toward edges neighboring the adjacent positions
        int adj_edge = (pos < D) ? seq[pos] : seq[D - 1];
        const std::vector<int>& nbrs = edge_neighbors[adj_edge];
        if (!nbrs.empty() && coin(gen) < 0.5) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            edge = nbrs[nbr_dist(gen)];
        } else {
            edge = edge_dist(gen);
        }

        // Build new sequence with insertion
        matrix_base<int> limits(1, D + 1);
        for (int i = 0; i < D + 1; ++i) limits[i] = n_edges;
        GrayCode new_seq(limits);
        for (int i = 0; i < pos; ++i) new_seq[i] = seq[i];
        new_seq[pos] = edge;
        for (int i = pos; i < D; ++i) new_seq[i + 1] = seq[i];

        GrayCode result = canonicalize_and_validate(new_seq);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_shrink(const GrayCode& seq, int D_min) {
    int D = static_cast<int>(seq.size());
    if (D <= D_min) return GrayCode();
    int n_edges = static_cast<int>(topology.size());
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

    int n_edges = static_cast<int>(topology.size());
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
    for (int step = 0; step < max_local_steps; ++step) {
        int D = static_cast<int>(current.size());
        std::vector<GrayCode> neighbors;

        // Single-position substitution neighbors
        for (int pos = 0; pos < D; ++pos) {
            for (int e = 0; e < n_edges; ++e) {
                if (e == current[pos]) continue;
                GrayCode new_seq = current.copy();
                new_seq[pos] = e;
                GrayCode cand = canonicalize_and_validate(new_seq);
                if (cand.size() > 0 && seen.find(cand) == seen.end())
                    neighbors.push_back(std::move(cand));
            }
        }

        // Grow neighbors
        if (D_max_local > 0 && D < D_max_local) {
            for (int pos = 0; pos <= D; ++pos) {
                for (int e = 0; e < n_edges; ++e) {
                    matrix_base<int> limits(1, D + 1);
                    for (int i = 0; i < D + 1; ++i) limits[i] = n_edges;
                    GrayCode new_seq(limits);
                    for (int i = 0; i < pos; ++i) new_seq[i] = current[i];
                    new_seq[pos] = e;
                    for (int i = pos; i < D; ++i) new_seq[i + 1] = current[i];
                    GrayCode cand = canonicalize_and_validate(new_seq);
                    if (cand.size() > 0 && seen.find(cand) == seen.end())
                        neighbors.push_back(std::move(cand));
                }
            }
        }

        // Shrink neighbors
        if (D_min_local >= 0 && D > D_min_local) {
            for (int pos = 0; pos < D; ++pos) {
                GrayCode new_seq = current.remove_Digit(pos);
                GrayCode cand = canonicalize_and_validate(new_seq);
                if (cand.size() > 0 && seen.find(cand) == seen.end())
                    neighbors.push_back(std::move(cand));
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

        // Compute mu for all neighbors (cheap: O(n_train * n_nb))
        std::vector<double> mu(n_nb);
        for (int j = 0; j < n_nb; ++j) {
            double val = 0;
            for (int i = 0; i < gp.n_train; ++i)
                val += Ks[i * n_nb + j] * gp.alpha_data[i];
            mu[j] = val;
        }

        // Sort neighbors by mu ascending for early termination.
        // Since lcb = mu - kappa*std <= mu, any neighbor with
        // mu[j] >= best_lcb can be skipped without solving.
        std::vector<int> mu_order(n_nb);
        std::iota(mu_order.begin(), mu_order.end(), 0);
        std::sort(mu_order.begin(), mu_order.end(),
                  [&](int a, int b) { return mu[a] < mu[b]; });

        // Column-wise forward substitution with early termination
        int n_t = gp.n_train;
        std::vector<double> v_col(n_t);
        double best_nb_lcb = std::numeric_limits<double>::infinity();
        int best_nb_idx = -1;

        for (int rank = 0; rank < n_nb; ++rank) {
            int j = mu_order[rank];

            // Prune: if mu[j] >= best_lcb, no variance can help
            if (mu[j] >= best_nb_lcb) break;

            // Forward solve L v = Ks[:,j] for this single column
            for (int i = 0; i < n_t; ++i)
                v_col[i] = Ks[i * n_nb + j];
            for (int i = 0; i < n_t; ++i) {
                for (int k = 0; k < i; ++k)
                    v_col[i] -= gp.L_data[i * n_t + k] * v_col[k];
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
    GrayCodeSet& seen, std::vector<GrayCode>& candidates_out,
    int& n_local_out, double& avg_steps_out,
    int D_min_gen, int D_max_gen) {

    candidates_out.clear();
    int n_edges = static_cast<int>(topology.size());
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

    // GP-directed local search candidates
    int n_local_found = 0;
    int total_local_steps = 0;
    int n_local_target = static_cast<int>(n_candidates * local_search_fraction);
    for (int i = 0; i < n_local_target; ++i) {
        const GrayCode& parent = tournament_select();
        int steps = 0;
        GrayCode result = local_search_acq(parent, cache, gp, scale, seen,
                                           D_min_gen, D_max_gen, &steps);
        total_local_steps += steps;
        if (result.size() > 0 && seen.find(result) == seen.end() &&
            (D_max_gen < 0 || static_cast<int>(result.size()) <= D_max_gen)) {
            seen.insert(result.copy());
            candidates_out.push_back(std::move(result));
            n_local_found++;
        }
    }

    // Random candidates via evolutionary operators
    int max_attempts = n_candidates * 10;
    for (int attempt = 0;
         static_cast<int>(candidates_out.size()) < n_candidates && attempt < max_attempts;
         ++attempt) {
        double r = op_dist(gen);
        GrayCode result;

        if (mixed_d) {
            if (r < 0.30)
                result = mutate_point(tournament_select());
            else if (r < 0.45)
                result = mutate_swap(tournament_select());
            else if (r < 0.60)
                result = mutate_block(tournament_select(), block_size);
            else if (r < 0.70) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                if (p1.size() == p2.size())
                    result = crossover_uniform(p1, p2);
                else
                    result = mutate_point(p1);
            }
            else if (r < 0.75)
                result = mutate_grow(tournament_select(), D_max_gen);
            else if (r < 0.90)
                result = mutate_shrink(tournament_select(), D_min_gen);
            else {
                std::geometric_distribution<int> geo(0.4);
                int rand_D = D_min_gen + std::min(geo(gen), D_max_gen - D_min_gen);
                result = generate_valid_sequence(rand_D);
            }
        } else {
            if (r < 0.40)
                result = mutate_point(tournament_select());
            else if (r < 0.60)
                result = mutate_swap(tournament_select());
            else if (r < 0.75)
                result = mutate_block(tournament_select(), block_size);
            else if (r < 0.85)
                result = crossover_uniform(tournament_select(), tournament_select());
            else {
                int D = static_cast<int>(population[0].size());
                result = generate_valid_sequence(D);
            }
        }

        if (result.size() > 0 && seen.find(result) == seen.end() &&
            (!mixed_d || static_cast<int>(result.size()) <= D_max_gen)) {
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
        int target = possible_target_qbits[gcode[i]];
        int control = possible_control_qbits[gcode[i]];
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

    Gates_block* gate_structure = construct_gate_structure(circuit);

    N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp.set_custom_gate_structure(gate_structure);
    delete gate_structure;  // set_custom_gate_structure clones the gates; free the original
    cDecomp.set_verbose(0);
    cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
    cDecomp.set_optimization_tolerance(tolerance);
    cDecomp.set_optimizer(alg);

    int param_num = gate_structure->get_parameter_num();
    // Random initial parameters
    Matrix_real random_params(1, param_num);
    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);
    for (int i = 0; i < param_num; ++i)
        random_params[i] = param_dist(gen);
    cDecomp.set_optimized_parameters(random_params.get_data(), param_num);

    cDecomp.start_decomposition();

    Matrix_real params = cDecomp.get_optimized_parameters();
    double score = cDecomp.optimization_problem(params);

    return {score, params};
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
        flog << "SurSearch (C++ rolling-window): N=" << qbit_num
             << ", D=" << D_min_search << "-" << D_max_search
             << ", W=" << std::max(1, d_window_width)
             << ", kappa=" << kappa << std::endl;
        flog << "---" << std::endl;
    }

    // Rolling-window search: slide [win_lo, win_hi] across D range
    // Persistent state across all windows
    // Adaptive window width: sqrt(range) so wider ranges get wider windows
    int D_range = D_max_search - D_min_search;
    int W = std::max(d_window_width, static_cast<int>(
        std::ceil(std::sqrt(std::max(1, D_range)))));
    int win_lo = D_min_search;
    int win_hi = std::min(win_lo + W - 1, D_max_search);
    int consecutive_stagnations = 0;

    SSKCache ssk_cache(sur_gap_decay, sur_match_decay, sur_ssk_order);
    GrayCodeSet seen;
    std::vector<GrayCode> X;
    std::vector<double> y;
    std::vector<Matrix_real> all_params;

    while (win_lo <= D_max_search) {
        win_hi = std::min(win_lo + W - 1, D_max_search);

        // --- Grow-seed each D in [win_lo, win_hi] that lacks samples ---
        bool early_solution = false;
        std::vector<int> n_grown_per_D(win_hi - win_lo + 1, 0);

        for (int D = win_lo; D <= win_hi && !early_solution; ++D) {
            // Count existing samples at this D
            int existing_at_D = 0;
            for (size_t i = 0; i < X.size(); ++i)
                if (static_cast<int>(X[i].size()) == D)
                    existing_at_D++;

            int needed = std::max(0, X0_size - existing_at_D);
            if (needed <= 0) continue;

            // Gather D-1 circuits sorted by score (best first)
            std::vector<int> prev_indices;
            for (size_t i = 0; i < X.size(); ++i)
                if (static_cast<int>(X[i].size()) == D - 1)
                    prev_indices.push_back(static_cast<int>(i));

            if (prev_indices.empty()) continue;

            std::sort(prev_indices.begin(), prev_indices.end(),
                      [&](int a, int b) { return y[a] < y[b]; });

            int n_grown = 0;
            for (int pi : prev_indices) {
                if (n_grown >= needed) break;
                for (int attempt = 0; attempt < 10; ++attempt) {
                    GrayCode grown = mutate_grow(X[pi], D);
                    if (grown.size() == 0) continue;
                    if (static_cast<int>(grown.size()) != D) continue;
                    if (seen.find(grown) != seen.end()) continue;

                    seen.insert(grown.copy());
                    ssk_cache.register_circuit(grown);

                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto dec_result = decompose(grown);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    decompose_time += std::chrono::duration<double>(t1 - t0).count();
                    decompose_count++;

                    y.push_back(dec_result.first);
                    all_params.push_back(dec_result.second);
                    X.push_back(std::move(grown));
                    n_grown++;

                    if (y.back() < best_score) {
                        best_score = y.back();
                        best_circuit = X.back().copy();
                        best_params = all_params.back();
                    }
                    if (y.back() < tolerance) {
                        early_solution = true;
                    }
                    break;  // success for this parent, move to next
                }
                if (early_solution) break;
            }
            n_grown_per_D[D - win_lo] = n_grown;
        }

        // Log window header with grow-seed counts
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "\n=== Window [" << win_lo << "," << win_hi << "] (grow-seeded:";
            for (int D = win_lo; D <= win_hi; ++D)
                flog << " D" << D << "=" << n_grown_per_D[D - win_lo];
            flog << ") ===" << std::endl;
        }

        if (early_solution) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "SOLUTION found during grow-seeding at D="
                 << best_circuit.size() << ", score=" << best_score << std::endl;
            flog << "SOLUTION at D=" << best_circuit.size() << std::endl;
        }

        // --- Run window search (skip if already solved during seeding) ---
        double best_at_entry = best_score;
        bool found_in_window = early_solution;
        if (!early_solution) {
            WindowResult result = run_window_search(win_lo, win_hi,
                ssk_cache, seen, X, y, all_params, log_file);

            found_in_window = (result == WINDOW_SUCCESS);
        }

        if (found_in_window) {
            int D_found = static_cast<int>(best_circuit.size());
            if (!early_solution) {
                std::ofstream flog(log_file, std::ios::app);
                flog << "SOLUTION at D=" << D_found << std::endl;
            }

            // Swap to rollback config for narrowing phase
            double save_kappa = kappa;
            int save_wp = window_patience;
            int save_wmi = window_max_iters;
            int save_cpi = candidates_per_iter;
            int save_nts = n_thompson_samples;
            double save_lsf = local_search_fraction;
            int save_mls = max_local_steps;

            kappa = rb_kappa;
            window_patience = rb_window_patience;
            window_max_iters = rb_window_max_iters;
            candidates_per_iter = rb_candidates_per_iter;
            n_thompson_samples = rb_n_thompson_samples;
            local_search_fraction = rb_local_search_fraction;
            max_local_steps = rb_max_local_steps;

            // Try to find shorter circuits by searching D_found-1 down to D_min + 1 (since it wouldve found a solution at D_min in the last iter probably.), one D at a time
            for (int D_try = D_found - 1; D_try >= D_min_search + 1; --D_try) {
                {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "Narrowing: solution at D=" << D_found
                         << ", trying D=" << D_try << std::endl;
                }

                // Grow-seed into D_try from D_try-1 circuits if needed
                int existing_at_D = 0;
                for (size_t i = 0; i < X.size(); ++i)
                    if (static_cast<int>(X[i].size()) == D_try)
                        existing_at_D++;

                if (existing_at_D < X0_size) {
                    std::vector<int> prev_indices;
                    for (size_t i = 0; i < X.size(); ++i)
                        if (static_cast<int>(X[i].size()) == D_try - 1)
                            prev_indices.push_back(static_cast<int>(i));
                    std::sort(prev_indices.begin(), prev_indices.end(),
                              [&](int a, int b) { return y[a] < y[b]; });

                    int needed = X0_size - existing_at_D;
                    int n_grown = 0;
                    for (int pi : prev_indices) {
                        if (n_grown >= needed) break;
                        for (int attempt = 0; attempt < 10; ++attempt) {
                            GrayCode grown = mutate_grow(X[pi], D_try);
                            if (grown.size() == 0 || static_cast<int>(grown.size()) != D_try) continue;
                            if (seen.find(grown) != seen.end()) continue;

                            seen.insert(grown.copy());
                            ssk_cache.register_circuit(grown);
                            auto t0 = std::chrono::high_resolution_clock::now();
                            auto dec_result = decompose(grown);
                            auto t1 = std::chrono::high_resolution_clock::now();
                            decompose_time += std::chrono::duration<double>(t1 - t0).count();
                            decompose_count++;

                            y.push_back(dec_result.first);
                            all_params.push_back(dec_result.second);
                            X.push_back(std::move(grown));
                            n_grown++;

                            if (y.back() < best_score) {
                                best_score = y.back();
                                best_circuit = X.back().copy();
                                best_params = all_params.back();
                            }
                            break;
                        }
                    }
                }

                WindowResult narrow_result = run_window_search(D_try, D_try,
                    ssk_cache, seen, X, y, all_params, log_file);

                if (narrow_result == WINDOW_SUCCESS) {
                    D_found = static_cast<int>(best_circuit.size());
                    {
                        std::ofstream flog(log_file, std::ios::app);
                        flog << "SOLUTION at D=" << D_found << std::endl;
                    }
                    // Continue narrowing from this new D_found
                    continue;
                } else {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "D=" << D_try << " stagnated, minimum depth is D="
                         << D_found << std::endl;
                    break;
                }
            }

            // Restore rolling-window config
            kappa = save_kappa;
            window_patience = save_wp;
            window_max_iters = save_wmi;
            candidates_per_iter = save_cpi;
            n_thompson_samples = save_nts;
            local_search_fraction = save_lsf;
            max_local_steps = save_mls;

            break;  // Done — narrowing complete
        }

        // --- Find which D had the best result in this window ---
        int window_best_D = win_lo;
        double window_best_score = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < X.size(); ++i) {
            int d = static_cast<int>(X[i].size());
            if (d >= win_lo && d <= win_hi && y[i] < window_best_score) {
                window_best_score = y[i];
                window_best_D = d;
            }
        }

        // --- Slide the window forward ---
        // Improvement = >1% relative gain AND score is actually good (below 0.5).
        // Without the threshold, first window always "improves" from infinity.
        bool improved = (best_score < best_at_entry * 0.99) &&
                        (best_score < 0.5);

        // Next window must not include D values below the window's best D
        int min_next_lo = window_best_D;

        if (improved) {
            consecutive_stagnations = 0;
            win_lo = std::max(win_lo + 1, min_next_lo);
            {
                std::ofstream flog(log_file, std::ios::app);
                flog << "Window [" << win_lo - 1 << "," << win_hi
                     << "] improved (best at D=" << window_best_D
                     << "), sliding to [" << win_lo << ","
                     << std::min(win_lo + W - 1, D_max_search) << "]" << std::endl;
            }
        } else {
            consecutive_stagnations++;
            if (consecutive_stagnations >= max_consecutive_stagnations) {
                // Adaptive jump: sqrt(remaining range) — aggressive leaps
                // early when OSR underestimates, tapering near D_max
                int remaining = D_max_search - win_lo;
                int jump = std::max(W, static_cast<int>(
                    std::ceil(std::sqrt(std::max(1, remaining)))));
                int old_lo = win_lo;
                win_lo = std::max(win_lo + jump, min_next_lo);
                consecutive_stagnations = 0;
                {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "Window [" << old_lo << "," << win_hi
                         << "] stagnated (" << max_consecutive_stagnations
                         << "/" << max_consecutive_stagnations
                         << "), jumping +" << (win_lo - old_lo) << " to [" << win_lo << ","
                         << std::min(win_lo + W - 1, D_max_search) << "]" << std::endl;
                }
            } else {
                int old_lo = win_lo;
                win_lo = std::max(win_lo + 1, min_next_lo);
                {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "Window [" << old_lo << "," << win_hi
                         << "] stagnated (" << consecutive_stagnations
                         << "/" << max_consecutive_stagnations
                         << "), sliding to [" << win_lo << ","
                         << std::min(win_lo + W - 1, D_max_search) << "]" << std::endl;
                }
            }
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

                auto t0 = std::chrono::high_resolution_clock::now();
                std::pair<double, Matrix_real> dec_result = decompose(seq);
                auto t1 = std::chrono::high_resolution_clock::now();
                decompose_time += std::chrono::duration<double>(t1 - t0).count();
                decompose_count++;

                y.push_back(dec_result.first);
                all_params.push_back(dec_result.second);
                X.push_back(std::move(seq));
                generated++;

                // Update global best
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

    for (int itr = 0; itr < window_max_iters; ++itr) {

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
                                ssk_cache, dummy_gp, 1.0, seen,
                                candidates, n_local, avg_steps,
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

            // Build filtered pool (indices into X) for circuits with D >= d_filter_min
            std::vector<int> filtered_x_indices;
            for (int i = 0; i < static_cast<int>(X.size()); ++i) {
                if (static_cast<int>(X[i].size()) >= d_filter_min)
                    filtered_x_indices.push_back(i);
            }
            int n_filtered = static_cast<int>(filtered_x_indices.size());
            if (n_filtered == 0) break;

            // Cap GP training set at gp_max_train using hybrid selection:
            // 50% best-by-score (exploitation) + 50% pivoted Cholesky (diversity)
            std::sort(filtered_x_indices.begin(), filtered_x_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });

            std::vector<int> selected_fi;  // selected indices into filtered_x_indices
            if (n_filtered <= gp_max_train) {
                selected_fi.resize(n_filtered);
                for (int i = 0; i < n_filtered; ++i) selected_fi[i] = i;
            } else {
                int n_best = gp_max_train / 2;
                int n_total = gp_max_train;
                selected_fi.reserve(n_total);
                std::vector<bool> taken(n_filtered, false);

                // Stage 1: top n_best by score (already sorted)
                for (int i = 0; i < n_best; ++i) {
                    selected_fi.push_back(i);
                    taken[i] = true;
                }

                // Stage 2: pivoted Cholesky for kernel-diverse points
                std::vector<int> cidx(n_filtered);
                for (int i = 0; i < n_filtered; ++i)
                    cidx[i] = ssk_cache.circuit_to_idx[X[filtered_x_indices[i]]];
                int cap = ssk_cache.capacity_;

                // L(i,j) Cholesky factor, d[i] residual diagonal
                std::vector<double> L_data(static_cast<size_t>(n_filtered) * n_total, 0.0);
                std::vector<double> d(n_filtered, 1.0);

                for (int j = 0; j < n_total; ++j) {
                    int pj;
                    if (j < n_best) {
                        pj = selected_fi[j];
                    } else {
                        pj = -1;
                        double max_d = -1;
                        for (int i = 0; i < n_filtered; ++i)
                            if (!taken[i] && d[i] > max_d) { max_d = d[i]; pj = i; }
                        if (pj < 0 || max_d < 1e-12) break;
                        selected_fi.push_back(pj);
                        taken[pj] = true;
                    }

                    double L_diag = std::sqrt(std::max(d[pj], 1e-12));
                    L_data[pj * n_total + j] = L_diag;

                    for (int i = 0; i < n_filtered; ++i) {
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
                min_nonzero * 0.01 : tolerance * 0.01;

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
                                ssk_cache, gp, scale, seen,
                                candidates, n_local, avg_steps,
                                win_lo, win_hi);

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_cand = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            if (candidates.empty()) break;

            int n_cand = static_cast<int>(candidates.size());

            // --- Acquisition phase ---
            t_phase0 = std::chrono::high_resolution_clock::now();

            // Screen with GP: compute cross-kernel only for training subset
            std::vector<int> train_indices_vec(train_indices.begin(),
                                               train_indices.begin() + n_x);
            std::vector<double> Ks(gp.n_train * n_cand);
            ssk_cache.compute_cross_kernel_subset(candidates, scale,
                                                  train_indices_vec, Ks.data());

            std::vector<double> log_mu_norm(n_cand);
            for (int j = 0; j < n_cand; ++j) {
                double val = 0;
                for (int i = 0; i < gp.n_train; ++i)
                    val += Ks[i * n_cand + j] * gp.alpha_data[i];
                log_mu_norm[j] = val;
            }

            // Solve L v = Ks for variance
            std::vector<double> v(gp.n_train * n_cand);
            std::memcpy(v.data(), Ks.data(), gp.n_train * n_cand * sizeof(double));
            LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                           gp.n_train, n_cand, gp.L_data.data(), gp.n_train,
                           v.data(), n_cand);

            std::vector<double> log_std_norm(n_cand);
            for (int j = 0; j < n_cand; ++j) {
                double sv2 = 0;
                for (int i = 0; i < gp.n_train; ++i)
                    sv2 += v[i * n_cand + j] * v[i * n_cand + j];
                log_std_norm[j] = std::sqrt(std::max(scale - sv2, 0.0));
            }

            // Thompson Sampling with posterior covariance
            // Register candidates in the main cache to reuse existing kernel values
            // instead of recomputing O(n_cand^2) SSK pairs from scratch
            std::vector<int> cand_cache_idx(n_cand);
            for (int i = 0; i < n_cand; ++i)
                cand_cache_idx[i] = ssk_cache.register_circuit(candidates[i]);

            std::vector<double> K_cand(n_cand * n_cand);
            ssk_cache.kernel_matrix(cand_cache_idx.data(), n_cand,
                                    cand_cache_idx.data(), n_cand,
                                    scale, K_cand.data());

            for (int i = 0; i < n_cand; ++i) {
                for (int j = 0; j < n_cand; ++j) {
                    double viv = 0;
                    for (int k = 0; k < gp.n_train; ++k)
                        viv += v[k * n_cand + i] * v[k * n_cand + j];
                    K_cand[i * n_cand + j] -= viv;
                }
            }

            // Cholesky of posterior covariance
            double cov_jitter = 1e-6;
            std::vector<double> L_post(K_cand);
            int info = -1;
            for (int attempt = 0; attempt < 10; ++attempt) {
                L_post = K_cand;
                for (int i = 0; i < n_cand; ++i)
                    L_post[i * n_cand + i] += cov_jitter;
                info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_cand, L_post.data(), n_cand);
                if (info == 0) break;
                cov_jitter *= 10;
            }
            if (info != 0) {
                std::fill(L_post.begin(), L_post.end(), 0.0);
                for (int i = 0; i < n_cand; ++i)
                    L_post[i * n_cand + i] = log_std_norm[i];
            }

            // D-penalty (normalize by win_hi, not global D_max)
            std::vector<double> mu_cand(log_mu_norm);
            if (d_penalty > 0 && win_hi > 0) {
                for (int i = 0; i < n_cand; ++i) {
                    double D_val = static_cast<double>(candidates[i].size());
                    mu_cand[i] += d_penalty * (D_val / win_hi);
                }
            }

            // Thompson Sampling: draw samples, pick minimizers with diversity
            std::normal_distribution<double> normal(0.0, 1.0);
            std::set<int> selected_set;

            for (int ts = 0; ts < n_thompson_samples; ++ts) {
                std::vector<double> z(n_cand);
                for (int i = 0; i < n_cand; ++i) z[i] = normal(gen);

                std::vector<double> f_sample(n_cand);
                for (int i = 0; i < n_cand; ++i) {
                    double Lz = 0;
                    for (int j = 0; j <= i; ++j)
                        Lz += L_post[i * n_cand + j] * z[j];
                    f_sample[i] = mu_cand[i] + Lz;
                }

                std::vector<int> order(n_cand);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](int a, int b) { return f_sample[a] < f_sample[b]; });

                for (int sidx : order) {
                    if (selected_set.count(sidx)) continue;
                    bool too_similar = false;
                    if (!selected_indices.empty() && diversity_thresh < 1.0) {
                        for (int prev : selected_indices) {
                            // Use cached normalized kernel instead of recomputing SSK
                            double k_val = ssk_cache.K_norm[
                                cand_cache_idx[sidx] * ssk_cache.capacity_ +
                                cand_cache_idx[prev]];
                            if (k_val > diversity_thresh) {
                                too_similar = true;
                                break;
                            }
                        }
                    }
                    if (!too_similar) {
                        selected_indices.push_back(sidx);
                        selected_set.insert(sidx);
                        break;
                    }
                }
            }

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_acq = std::chrono::duration<double>(t_phase1 - t_phase0).count();
        } // end if/else use_random_candidates

        // --- Decompose phase (shared) ---
        auto t_phase0 = std::chrono::high_resolution_clock::now();

        // Evaluate selected candidates
        int n_sel = static_cast<int>(selected_indices.size());

        bool improved_this_iter = false;
        bool found_solution = false;

        for (int si = 0; si < n_sel; ++si) {
            int sel_idx = selected_indices[si];

            auto t0 = std::chrono::high_resolution_clock::now();
            std::pair<double, Matrix_real> new_result = decompose(candidates[sel_idx]);
            double new_score = new_result.first;
            Matrix_real new_params = new_result.second;
            auto t1 = std::chrono::high_resolution_clock::now();
            decompose_time += std::chrono::duration<double>(t1 - t0).count();
            decompose_count++;

            ssk_cache.register_circuit(candidates[sel_idx]);
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
            return WINDOW_SUCCESS;
        }

        // Only reset patience on meaningful improvement (>1% relative or solution found)
        if (improved_this_iter && (prev_best_score - best_score) > prev_best_score * 0.01) {
            iters_since_improvement = 0;
            prev_best_score = best_score;
        } else {
            iters_since_improvement++;
        }

        // Log — show window-local best score (not global best)
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Iter " << itr
                 << ": best=" << window_best_score
                 << " (D=" << window_best_D << ")"
                 << " evals=" << X.size()
                 << (use_random_candidates ? " [RANDOM]" : "")
                 << " [gp=" << std::fixed << std::setprecision(2) << t_gp
                 << "s cand=" << t_cand
                 << "s acq=" << t_acq
                 << "s dec=" << t_dec << "s]"
                 << std::defaultfloat
                 << std::endl;
        }

        if (iters_since_improvement >= window_patience) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Stagnation after " << itr + 1 << " iters" << std::endl;
            return WINDOW_STAGNATION;
        }
    }

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
#if BLAS == 0
    int num_threads_saved = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS == 1
    int num_threads_saved = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS == 2
    int num_threads_saved = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

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

    // Restore thread count
#if BLAS == 0
    omp_set_num_threads(num_threads_saved);
#elif BLAS == 1
    MKL_Set_Num_Threads(num_threads_saved);
#elif BLAS == 2
    openblas_set_num_threads(num_threads_saved);
#endif
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

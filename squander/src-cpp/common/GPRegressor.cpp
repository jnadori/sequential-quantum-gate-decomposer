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
#include "GPRegressor.h"
#include "common.h"
#include "BFGS_Powell.h"
#include <chrono>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include <cstdlib>


// ============================================================================
// SSKCache implementation
// ============================================================================

SSKCache::SSKCache() : gap_decay(0.8), match_sq(0.64), order(3),
                       kernel_type_(0), wl_iterations_(3),
                       size_(0), capacity_(64) {
    K_norm.resize(capacity_ * capacity_, 0.0);
    raw_diags.resize(capacity_, 0.0);
}

SSKCache::SSKCache(double gap_decay_in, double match_decay_in, int order_in)
    : gap_decay(gap_decay_in), match_sq(match_decay_in * match_decay_in),
      order(order_in), kernel_type_(0), wl_iterations_(3),
      size_(0), capacity_(64) {
    K_norm.resize(capacity_ * capacity_, 0.0);
    raw_diags.resize(capacity_, 0.0);
}

SSKCache::SSKCache(int kernel_type, int wl_iterations,
                   const std::vector<int>& token_masks,
                   double gap_decay_in, double match_decay_in, int order_in)
    : gap_decay(gap_decay_in), match_sq(match_decay_in * match_decay_in),
      order(order_in), kernel_type_(kernel_type),
      wl_iterations_(wl_iterations), token_masks_(token_masks),
      size_(0), capacity_(64) {
    K_norm.resize(capacity_ * capacity_, 0.0);
    raw_diags.resize(capacity_, 0.0);
}

std::unordered_map<size_t, int> SSKCache::compute_wl_features(
    const int* tokens, int D) const {

    // Build undirected adjacency: edge (i,j) if gates share a qubit
    std::vector<std::vector<int>> adj(D);
    for (int i = 0; i < D; ++i) {
        int mi = token_masks_[tokens[i]];
        for (int j = i + 1; j < D; ++j) {
            if (mi & token_masks_[tokens[j]]) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // Initialize labels from token values
    std::vector<size_t> labels(D);
    std::unordered_map<size_t, int> features;
    for (int i = 0; i < D; ++i) {
        labels[i] = std::hash<int>{}(tokens[i]);
        features[labels[i]]++;
    }

    // WL iterations: refine labels based on neighborhood
    for (int h = 0; h < wl_iterations_; ++h) {
        std::vector<size_t> new_labels(D);
        for (int i = 0; i < D; ++i) {
            // Collect and sort neighbor labels
            std::vector<size_t> nbr;
            nbr.reserve(adj[i].size());
            for (int j : adj[i])
                nbr.push_back(labels[j]);
            std::sort(nbr.begin(), nbr.end());

            // Hash combine: (current_label, sorted neighbor labels)
            size_t combined = labels[i];
            combined ^= std::hash<size_t>{}(nbr.size()) + 0x9e3779b9
                         + (combined << 6) + (combined >> 2);
            for (size_t nl : nbr)
                combined ^= std::hash<size_t>{}(nl) + 0x9e3779b9
                             + (combined << 6) + (combined >> 2);
            new_labels[i] = combined;
            features[combined]++;
        }
        labels = new_labels;
    }

    return features;
}

double SSKCache::wl_dot(const std::unordered_map<size_t, int>& f1,
                        const std::unordered_map<size_t, int>& f2) {
    double dot = 0.0;
    const auto& smaller = (f1.size() <= f2.size()) ? f1 : f2;
    const auto& larger  = (f1.size() <= f2.size()) ? f2 : f1;
    for (const auto& kv : smaller) {
        auto it = larger.find(kv.first);
        if (it != larger.end())
            dot += static_cast<double>(kv.second) * it->second;
    }
    return dot;
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
    const int ord = order;

    std::vector<double> kp(n1 * n2);
    std::vector<double> tmp(n1 * n2);
    std::vector<double> run_vec(n1);
    std::vector<double> mask(n1);
    std::vector<double> kp_col(n1);

    // Pre-allocate b_positions once to avoid per-pair heap alloc/dealloc under
    // concurrent TBB workers (which causes malloc lock contention).
    int max_token = 0;
    for (int p = 0; p < P; ++p)
        for (int j = 0; j < n2; ++j)
            if (cj[p * n2 + j] > max_token) max_token = cj[p * n2 + j];
    std::vector<std::vector<int>> b_positions(max_token + 1);

    for (int p = 0; p < P; ++p) {
        const int* a = ci + p * n1;
        const int* b = cj + p * n2;

        // Build match-position index for b — clear without deallocating
        for (int t = 0; t <= max_token; ++t) b_positions[t].clear();
        int max_b = 0;
        for (int j = 0; j < n2; ++j) {
            b_positions[b[j]].push_back(j);
            if (b[j] > max_b) max_b = b[j];
        }

        // Kp = ones(n1, n2)
        std::fill(kp.begin(), kp.end(), 1.0);

        // First-order term: count matches via histogram
        double raw = 0.0;
        for (int i = 0; i < n1; ++i) {
            const int ai = a[i];
            if (ai <= max_b)
                raw += static_cast<double>(b_positions[ai].size());
        }
        raw *= msq;

        for (int iter = 1; iter < ord; ++iter) {
            // tmp = match_sq * (S .* kp) @ D2 via column recurrence
            std::fill(run_vec.begin(), run_vec.end(), 0.0);
            for (int i = 0; i < n1; ++i)
                tmp[i * n2] = 0.0;

            for (int j = 1; j < n2; ++j) {
                const int bj_prev = b[j - 1];

                for (int i = 0; i < n1; ++i) {
                    mask[i] = static_cast<double>(a[i] == bj_prev);
                    kp_col[i] = kp[i * n2 + j - 1];
                }

                for (int i = 0; i < n1; ++i) {
                    run_vec[i] = kp_col[i] * mask[i] + g * run_vec[i];
                }

                for (int i = 0; i < n1; ++i)
                    tmp[i * n2 + j] = msq * run_vec[i];
            }

            // kp = D1^T @ tmp  (row recurrence)
            std::fill_n(kp.data(), n2, 0.0);

            double add = 0.0;
            for (int i = 1; i < n1; ++i) {
                const int ai = a[i];
                const double* tmp_prev = tmp.data() + (i - 1) * n2;
                const double* kp_prev = kp.data() + (i - 1) * n2;
                double* kp_row = kp.data() + i * n2;

                for (int j = 0; j < n2; ++j)
                    kp_row[j] = tmp_prev[j] + g * kp_prev[j];

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

    if (new_idx >= capacity_) {
        grow_capacity();
        wl_features_.reserve(capacity_);
    }

    if (kernel_type_ == 1) {
        // --- Weisfeiler-Lehman kernel ---
        auto feat = compute_wl_features(circuit.get_data(), n_new);
        double self_raw = wl_dot(feat, feat);
        raw_diags[new_idx] = self_raw;

        std::vector<double> raw_new(new_idx + 1, 0.0);
        raw_new[new_idx] = self_raw;
        for (int i = 0; i < new_idx; ++i)
            raw_new[i] = wl_dot(feat, wl_features_[i]);

        wl_features_.push_back(std::move(feat));

        double inv_diag_new = 1.0 / std::sqrt(std::max(self_raw, 1e-24));
        for (int i = 0; i <= new_idx; ++i) {
            double inv_diag_i = 1.0 / std::sqrt(std::max(raw_diags[i], 1e-24));
            double norm_val = raw_new[i] * inv_diag_new * inv_diag_i;
            K_norm[new_idx * capacity_ + i] = norm_val;
            K_norm[i * capacity_ + new_idx] = norm_val;
        }
    } else {
        // --- SSK kernel (existing) ---
        double self_raw = single_ssk_raw(circuit.get_data(), n_new,
                                         circuit.get_data(), n_new);
        raw_diags[new_idx] = self_raw;

        std::vector<double> raw_new(new_idx + 1, 0.0);
        raw_new[new_idx] = self_raw;

        if (new_idx > 0) {
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

        double inv_diag_new = 1.0 / std::sqrt(std::max(raw_new[new_idx], 1e-24));
        for (int i = 0; i <= new_idx; ++i) {
            double inv_diag_i = 1.0 / std::sqrt(std::max(raw_diags[i], 1e-24));
            double norm_val = raw_new[i] * inv_diag_new * inv_diag_i;
            K_norm[new_idx * capacity_ + i] = norm_val;
            K_norm[i * capacity_ + new_idx] = norm_val;
        }
    }

    size_++;
    return new_idx;
}

void SSKCache::compute_cross_kernel(const std::vector<GrayCode>& candidates,
                                    double scale, double* result_out) {
    int n_reg = size_;
    int n_cand = static_cast<int>(candidates.size());
    if (n_reg == 0 || n_cand == 0) return;

    if (kernel_type_ == 1) {
        // --- WL cross-kernel ---
        std::vector<std::unordered_map<size_t, int>> cand_feats(n_cand);
        std::vector<double> cand_self_raw(n_cand);

        for (int c = 0; c < n_cand; ++c) {
            cand_feats[c] = compute_wl_features(candidates[c].get_data(),
                                                static_cast<int>(candidates[c].size()));
            cand_self_raw[c] = wl_dot(cand_feats[c], cand_feats[c]);
        }

        std::vector<double> inv_dr(n_reg);
        for (int r = 0; r < n_reg; ++r)
            inv_dr[r] = 1.0 / std::sqrt(std::max(raw_diags[r], 1e-24));
        std::vector<double> inv_dc(n_cand);
        for (int c = 0; c < n_cand; ++c)
            inv_dc[c] = 1.0 / std::sqrt(std::max(cand_self_raw[c], 1e-24));

        for (int c = 0; c < n_cand; ++c) {
            for (int r = 0; r < n_reg; ++r) {
                double raw = wl_dot(wl_features_[r], cand_feats[c]);
                result_out[r * n_cand + c] = scale * raw * inv_dr[r] * inv_dc[c];
            }
        }
        return;
    }

    // --- SSK cross-kernel (existing) ---

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
    if (kernel_type_ == 1) {
        auto fa = compute_wl_features(a.get_data(), static_cast<int>(a.size()));
        auto fb = compute_wl_features(b.get_data(), static_cast<int>(b.size()));
        double raw_ab = wl_dot(fa, fb);
        double raw_aa = wl_dot(fa, fa);
        double raw_bb = wl_dot(fb, fb);
        double denom = std::sqrt(std::max(raw_aa, 1e-24) * std::max(raw_bb, 1e-24));
        return raw_ab / denom;
    }
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

    std::vector<double> inv_dr(n_sub);
    for (int s = 0; s < n_sub; ++s)
        inv_dr[s] = 1.0 / std::sqrt(std::max(raw_diags[reg_subset[s]], 1e-24));

    if (kernel_type_ == 1) {
        // --- WL cross-kernel ---
        std::vector<std::unordered_map<size_t, int>> cand_feats(n_cand);
        std::vector<double> cand_self_raw(n_cand, 0.0);
        for (int c = 0; c < n_cand; ++c) {
            cand_feats[c] = compute_wl_features(candidates[c].get_data(),
                                                static_cast<int>(candidates[c].size()));
            cand_self_raw[c] = wl_dot(cand_feats[c], cand_feats[c]);
        }
        std::vector<double> inv_dc(n_cand);
        for (int c = 0; c < n_cand; ++c)
            inv_dc[c] = 1.0 / std::sqrt(std::max(cand_self_raw[c], 1e-24));
        for (int s = 0; s < n_sub; ++s)
            for (int c = 0; c < n_cand; ++c)
                result_out[s * n_cand + c] = scale *
                    wl_dot(wl_features_[reg_subset[s]], cand_feats[c]) * inv_dr[s] * inv_dc[c];
        return;
    }

    // --- SSK cross-kernel ---

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
    std::vector<double> inv_dc(n_cand);
    for (int c = 0; c < n_cand; ++c)
        inv_dc[c] = 1.0 / std::sqrt(std::max(cand_self_raw[c], 1e-24));

    for (int s = 0; s < n_sub; ++s)
        for (int c = 0; c < n_cand; ++c)
            result_out[s * n_cand + c] = scale * raw_cross[s * n_cand + c] * inv_dr[s] * inv_dc[c];
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
    : log_scale(0.0), noise(1e-2), jitter(1e-8), n_train(0),
      love_rank(50), love_valid(false) {}

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

    // Auto-compute LOVE if rank is configured
    if (love_rank > 0 && n > love_rank)
        compute_love(love_rank);
    else
        love_valid = false;
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

    // Auto-compute LOVE if rank is configured
    if (love_rank > 0 && n > love_rank)
        compute_love(love_rank);
    else
        love_valid = false;
}

void GPRegressor::compute_love(int rank) {
    int n = n_train;
    if (rank <= 0 || n <= 0) { love_valid = false; return; }
    int r = std::min(rank, n);

    // Lanczos iteration on K^{-1} to produce Q (n x r) and tridiagonal T
    // Q stored row-major: Q[i*r + k] = k-th Lanczos vector at component i
    std::vector<double> Q(n * r, 0.0);
    std::vector<double> alpha_lanc(r);   // diagonal of T
    std::vector<double> beta_lanc(r);    // off-diagonal of T (beta[0] unused)

    // Random starting vector (deterministic seed for reproducibility)
    std::mt19937 rng(42 + static_cast<unsigned>(n));
    std::normal_distribution<double> randn(0.0, 1.0);

    // q_0 = random, normalized
    double nrm = 0.0;
    for (int i = 0; i < n; ++i) {
        Q[i * r] = randn(rng);
        nrm += Q[i * r] * Q[i * r];
    }
    nrm = std::sqrt(nrm);
    for (int i = 0; i < n; ++i)
        Q[i * r] /= nrm;

    // Lanczos workspace: w = K^{-1} * q_j
    std::vector<double> w(n);
    std::vector<double> q_prev(n);  // q_{j-1}

    int actual_r = r;
    for (int j = 0; j < r; ++j) {
        // w = K^{-1} * q_j via triangular solves: L z = q_j, L^T w = z
        for (int i = 0; i < n; ++i)
            w[i] = Q[i * r + j];

        // Forward solve: L z = q_j
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < i; ++k)
                w[i] -= L_data[i * n + k] * w[k];
            w[i] /= L_data[i * n + i];
        }
        // Back solve: L^T w = z
        for (int i = n - 1; i >= 0; --i) {
            for (int k = i + 1; k < n; ++k)
                w[i] -= L_data[k * n + i] * w[k];
            w[i] /= L_data[i * n + i];
        }

        // alpha[j] = q_j^T w
        double a = 0.0;
        for (int i = 0; i < n; ++i)
            a += Q[i * r + j] * w[i];
        alpha_lanc[j] = a;

        // w = w - alpha[j] * q_j
        for (int i = 0; i < n; ++i)
            w[i] -= a * Q[i * r + j];

        // w = w - beta[j-1] * q_{j-1}
        if (j > 0) {
            for (int i = 0; i < n; ++i)
                w[i] -= beta_lanc[j - 1] * q_prev[i];
        }

        // Full reorthogonalization: w = w - Q * (Q^T w)
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k <= j; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i)
                    dot += Q[i * r + k] * w[i];
                for (int i = 0; i < n; ++i)
                    w[i] -= dot * Q[i * r + k];
            }
        }

        // beta[j] = ||w||
        double beta = 0.0;
        for (int i = 0; i < n; ++i)
            beta += w[i] * w[i];
        beta = std::sqrt(beta);

        if (j + 1 < r) {
            // Save q_j for next iteration's reorthogonalization
            for (int i = 0; i < n; ++i)
                q_prev[i] = Q[i * r + j];

            if (beta < 1e-12) {
                // Krylov subspace exhausted
                actual_r = j + 1;
                break;
            }

            beta_lanc[j] = beta;
            // q_{j+1} = w / beta
            for (int i = 0; i < n; ++i)
                Q[i * r + (j + 1)] = w[i] / beta;
        } else {
            beta_lanc[j] = 0.0;  // last iteration, no next vector
        }
    }
    r = actual_r;

    if (r <= 0) { love_valid = false; return; }

    // Eigendecompose tridiagonal T (r x r)
    // Pack into d (diagonal) and e (off-diagonal) format for LAPACKE_dsteqr
    std::vector<double> d(r);
    std::vector<double> e(r - 1);
    std::vector<double> V(r * r, 0.0);  // eigenvectors, row-major

    for (int i = 0; i < r; ++i)
        d[i] = alpha_lanc[i];
    for (int i = 0; i < r - 1; ++i)
        e[i] = beta_lanc[i];
    // Initialize V = I (dsteqr needs identity for eigenvectors)
    for (int i = 0; i < r; ++i)
        V[i * r + i] = 1.0;

    int info = LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', r,
                               d.data(), e.data(), V.data(), r);
    if (info != 0) {
        love_valid = false;
        return;
    }

    // Compute R = Q * V * diag(1/sqrt(eigenvalues))
    // Q is n x r (row-major, but only first r columns used), V is r x r
    R_love.resize(n * r);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < r; ++j) {
            double s = (d[j] > 1e-12) ? 1.0 / std::sqrt(d[j]) : 0.0;
            double val = 0.0;
            for (int k = 0; k < r; ++k)
                val += Q[i * rank + k] * V[k * r + j];
            R_love[i * r + j] = val * s;
        }
    }

    // Store actual rank used (may be < love_rank if Lanczos converged early)
    love_rank = r;
    love_valid = true;
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

    if (love_valid && love_rank > 0) {
        // LOVE fast variance: Var(f*) = scale - ||R^T k_s||^2
        for (int j = 0; j < n_cand; ++j) {
            double sum_rt_ks_sq = 0.0;
            for (int k = 0; k < love_rank; ++k) {
                double rt_ks_k = 0.0;
                for (int i = 0; i < n_train; ++i)
                    rt_ks_k += R_love[i * love_rank + k] * Ks[i * n_cand + j];
                sum_rt_ks_sq += rt_ks_k * rt_ks_k;
            }
            double var = std::max(scale - sum_rt_ks_sq, 0.0);
            std_out[j] = std::sqrt(var);
        }
    } else {
        // Exact fallback: v = L^{-1} @ Ks  (solve L v = Ks for v)
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

double GPRegressor::log_marginal_likelihood_with_grad(
    SSKCache& cache, const int* train_indices, int n, const double* y,
    double test_log_scale, double test_noise, double* grad_out) {

    double scale = std::exp(test_log_scale);

    // Build K = scale * K_norm + (noise + jitter) * I
    std::vector<double> K(n * n);
    cache.kernel_matrix(train_indices, n, train_indices, n, scale, K.data());
    for (int i = 0; i < n; ++i)
        K[i * n + i] += test_noise + jitter;

    // Cholesky factorization
    std::vector<double> L(K);
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, L.data(), n);
    if (info != 0) {
        grad_out[0] = 0.0;
        grad_out[1] = 0.0;
        return 1e30;
    }

    // alpha = K^{-1} y via triangular solves
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

    // NLL = -LML
    double nll = 0.5 * quad + 0.5 * log_det + 0.5 * n * std::log(2.0 * M_PI);

    // Compute L^{-1} in-place for trace computation
    info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', n, L.data(), n);
    if (info != 0) {
        grad_out[0] = 0.0;
        grad_out[1] = 0.0;
        return nll;
    }

    // Tr(K^{-1}) = ||L^{-1}||_F^2 (only lower triangle has data)
    double tr_Kinv = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            tr_Kinv += L[i * n + j] * L[i * n + j];

    // ||alpha||^2
    double alpha_sq = 0.0;
    for (int i = 0; i < n; ++i)
        alpha_sq += alpha[i] * alpha[i];

    // Analytical gradients of NLL
    // dK/d(log_scale) = scale * K_norm = K - (noise+jitter)*I
    // dNLL/d(log_scale) = 0.5*(quad - n + (noise+jitter)*(TrKinv - alpha_sq))
    grad_out[0] = 0.5 * (quad - n + (test_noise + jitter) * (tr_Kinv - alpha_sq));

    // dK/d(log_noise) = noise * I
    // dNLL/d(log_noise) = 0.5 * noise * (alpha_sq - TrKinv)
    grad_out[1] = 0.5 * test_noise * (alpha_sq - tr_Kinv);

    return nll;
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

// BFGS_Powell callback: computes negative log marginal likelihood and analytical gradient
// theta = [log_scale, log_noise]
static void gp_hyper_opt_combined(Matrix_real theta, void* void_ctx,
                                   double* f_out, Matrix_real& grad) {
    GPHyperOptContext* ctx = reinterpret_cast<GPHyperOptContext*>(void_ctx);

    double ls = std::max(ctx->scale_bounds.first,
                 std::min(ctx->scale_bounds.second, (double)theta[0]));
    double log_ns = std::max(ctx->noise_bounds.first,
                     std::min(ctx->noise_bounds.second, (double)theta[1]));
    double ns = std::exp(log_ns);

    double grad_arr[2];
    double nll = ctx->gp->log_marginal_likelihood_with_grad(
        *ctx->cache, ctx->train_indices, ctx->n, ctx->y, ls, ns, grad_arr);

    *f_out = nll;
    grad[0] = grad_arr[0];
    grad[1] = grad_arr[1];
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

    std::mt19937 rng(std::random_device{}());
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

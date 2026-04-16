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

/*! \file SurrogateModel.cpp
    \brief GP surrogate model — training, acquisition, and rank correlation.
*/

#include "SurrogateModel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>


// ============================================================================
// Constructor
// ============================================================================

SurrogateModel::SurrogateModel(int gp_max_train, double gp_score_ratio,
                               int wl_iterations, const std::vector<int>& token_masks)
    : cache_(wl_iterations, token_masks),
      scale_(1.0),
      gp_max_train_(gp_max_train),
      gp_score_ratio_(gp_score_ratio),
      prev_log_scale_(0.0),
      prev_noise_(1e-2) {}


// ============================================================================
// Reset
// ============================================================================

void SurrogateModel::reset() {
    gp_ = GPRegressor();
    gp_.log_scale = 0.0;
    gp_.noise = 1e-2;
    gp_.jitter = 1e-8;
    prev_log_scale_ = 0.0;
    prev_noise_ = 1e-2;
}


// ============================================================================
// Training
// ============================================================================

SurrogateModel::TrainResult SurrogateModel::train_window(
    const std::vector<GrayCode>& X, const std::vector<double>& y,
    int d_filter_lo, int d_filter_hi) {

    TrainResult result;

    // Build filtered pool sorted by score
    std::vector<int> filtered_x_indices;
    for (int i = 0; i < static_cast<int>(X.size()); ++i) {
        int d = static_cast<int>(X[i].size());
        if (d >= d_filter_lo && d <= d_filter_hi)
            filtered_x_indices.push_back(i);
    }
    int n_filtered = static_cast<int>(filtered_x_indices.size());
    if (n_filtered == 0) {
        result.n_train = 0;
        result.scale = 1.0;
        return result;
    }

    std::sort(filtered_x_indices.begin(), filtered_x_indices.end(),
              [&y](int a, int b) { return y[a] < y[b]; });

    // Fill remaining GP budget with lower-D results
    if (d_filter_lo > 1) {
        int n_fill_target = std::max(0, gp_max_train_ - n_filtered);
        if (n_fill_target > 0) {
            std::vector<int> lower_indices;
            for (int i = 0; i < static_cast<int>(X.size()); ++i) {
                int d = static_cast<int>(X[i].size());
                if (d < d_filter_lo)
                    lower_indices.push_back(i);
            }
            std::sort(lower_indices.begin(), lower_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });
            int n_fill = std::min(static_cast<int>(lower_indices.size()), n_fill_target);
            for (int i = 0; i < n_fill; ++i)
                filtered_x_indices.push_back(lower_indices[i]);
            std::sort(filtered_x_indices.begin(), filtered_x_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });
        }
    }

    // Register top circuits in cache
    int reg_limit = std::min(static_cast<int>(filtered_x_indices.size()), 2 * gp_max_train_);
    for (int i = 0; i < reg_limit; ++i) {
        int xi = filtered_x_indices[i];
        if (cache_.circuit_to_idx.find(X[xi]) == cache_.circuit_to_idx.end())
            cache_.register_circuit(X[xi]);
    }

    // Hybrid selection: score-top + pivoted Cholesky diversity
    int n_pool = reg_limit;
    int effective_gp_max_train = gp_max_train_;

    std::vector<int> selected_fi;
    if (n_pool <= effective_gp_max_train) {
        selected_fi.resize(n_pool);
        for (int i = 0; i < n_pool; ++i) selected_fi[i] = i;
    } else {
        int n_best = static_cast<int>(effective_gp_max_train * gp_score_ratio_);
        int n_total = effective_gp_max_train;
        selected_fi.reserve(n_total);
        std::vector<bool> taken(n_pool, false);

        for (int i = 0; i < n_best; ++i) {
            selected_fi.push_back(i);
            taken[i] = true;
        }

        // Pivoted Cholesky diversity
        std::vector<int> cidx(n_pool);
        for (int i = 0; i < n_pool; ++i)
            cidx[i] = cache_.circuit_to_idx[X[filtered_x_indices[i]]];
        int cap = cache_.capacity_;

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
                double K_val = cache_.K_norm[cidx[i] * cap + cidx[pj]];
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
        train_indices[i] = cache_.circuit_to_idx[X[xi]];
        train_y[i] = y[xi];
    }

    // Log10 transform + normalize
    std::vector<double> log_y(n_x);
    double min_nonzero = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_x; ++i)
        if (train_y[i] > 0) min_nonzero = std::min(min_nonzero, train_y[i]);
    double dynamic_floor = (min_nonzero < std::numeric_limits<double>::infinity()) ?
        min_nonzero * 0.01 : 1e-10;

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

    // Optimize GP hyperparameters
    std::pair<double,double> scale_bounds = {-3.0, 3.0};
    std::pair<double,double> noise_bounds = {-8.0, -1.0};
    int n_restarts = std::max(1, 3 - n_x / 10);
    gp_.optimize_hyperparameters(cache_, train_indices.data(), n_x,
                                  log_y_norm.data(), n_restarts,
                                  scale_bounds, noise_bounds);

    // Fit GP (full fit for one-shot)
    gp_.fit(cache_, train_indices.data(), n_x, log_y_norm.data());
    prev_log_scale_ = gp_.log_scale;
    prev_noise_ = gp_.noise;

    // Collapse detection
    double opt_scale = gp_.get_scale();
    if (opt_scale / std::max(gp_.noise, 1e-12) < 0.1 || opt_scale < 1e-6) {
        gp_.log_scale = 0.0;
        gp_.noise = 1e-2;
        gp_.fit(cache_, train_indices.data(), n_x, log_y_norm.data());
        prev_log_scale_ = gp_.log_scale;
        prev_noise_ = gp_.noise;
    }

    scale_ = gp_.get_scale();

    result.n_train = gp_.n_train;
    result.scale = scale_;
    result.log_y_norm = std::move(log_y_norm);
    result.filtered_x_indices = std::move(filtered_x_indices);
    return result;
}


// ============================================================================
// Acquisition
// ============================================================================

SurrogateModel::AcquisitionResult SurrogateModel::select_candidates_lcb(
    const std::vector<GrayCode>& candidates,
    const GrayCodeSet& seen,
    double kappa, double d_penalty, int win_hi, int n_pick) {

    AcquisitionResult result;
    int n_cand = static_cast<int>(candidates.size());
    if (n_cand == 0 || gp_.n_train == 0) return result;

    int n_t = gp_.n_train;

    // Cross-kernel Ks (n_t x n_cand)
    std::vector<int> train_indices_vec(gp_.train_indices_.begin(),
                                       gp_.train_indices_.begin() + gp_.n_train);
    std::vector<double> Ks(n_t * n_cand);
    cache_.compute_cross_kernel_subset(candidates, scale_,
                                        train_indices_vec, Ks.data());

    // mu = Ks^T @ alpha
    std::vector<double> log_mu_norm(n_cand);
    for (int j = 0; j < n_cand; ++j) {
        double val = 0;
        for (int i = 0; i < n_t; ++i)
            val += Ks[i * n_cand + j] * gp_.alpha_data[i];
        log_mu_norm[j] = val;
    }

    // v = L^{-1} Ks
    std::vector<double> v(n_t * n_cand);
    std::memcpy(v.data(), Ks.data(), n_t * n_cand * sizeof(double));
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                   n_t, n_cand, gp_.L_data.data(), n_t,
                   v.data(), n_cand);

    // Diagonal posterior variance
    std::vector<double> var_diag(n_cand);
    for (int i = 0; i < n_cand; ++i) {
        double sum_v2 = 0.0;
        for (int k = 0; k < n_t; ++k)
            sum_v2 += v[k * n_cand + i] * v[k * n_cand + i];
        var_diag[i] = std::max(scale_ - sum_v2, 1e-10);
    }

    // D-penalty on mu
    std::vector<double> mu_cand(log_mu_norm);
    if (d_penalty > 0 && win_hi > 0) {
        for (int i = 0; i < n_cand; ++i) {
            double D_val = static_cast<double>(candidates[i].size());
            mu_cand[i] += d_penalty * (D_val / win_hi);
        }
    }

    // LCB
    std::vector<double> lcb_val(n_cand);
    for (int i = 0; i < n_cand; ++i)
        lcb_val[i] = mu_cand[i] - kappa * std::sqrt(var_diag[i]);

    // Sort by LCB ascending
    std::vector<int> lcb_order(n_cand);
    std::iota(lcb_order.begin(), lcb_order.end(), 0);
    std::sort(lcb_order.begin(), lcb_order.end(),
              [&](int a, int b) { return lcb_val[a] < lcb_val[b]; });

    // Greedy selection with dedup
    int actual_pick = std::min(n_pick, n_cand);
    std::set<int> selected_set;
    for (int sidx : lcb_order) {
        if (static_cast<int>(result.selected_indices.size()) >= actual_pick) break;
        if (selected_set.count(sidx)) continue;
        if (seen.find(candidates[sidx]) != seen.end()) continue;
        result.selected_indices.push_back(sidx);
        result.gp_mu.push_back(log_mu_norm[sidx]);
        selected_set.insert(sidx);
    }

    return result;
}


// ============================================================================
// Rank correlation
// ============================================================================

double SurrogateModel::compute_rho(const std::vector<double>& predicted,
                                    const std::vector<double>& actual) {
    int n = static_cast<int>(predicted.size());
    if (n < 5) return 0.0;

    auto rank_of = [](const std::vector<double>& vals) {
        int sz = static_cast<int>(vals.size());
        std::vector<int> order(sz);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return vals[a] < vals[b]; });
        std::vector<double> ranks(sz);
        for (int i = 0; i < sz; ++i)
            ranks[order[i]] = static_cast<double>(i);
        return ranks;
    };

    auto pred_ranks = rank_of(predicted);
    auto actual_ranks = rank_of(actual);

    double sum_d2 = 0;
    for (int i = 0; i < n; ++i) {
        double d = pred_ranks[i] - actual_ranks[i];
        sum_d2 += d * d;
    }
    return 1.0 - 6.0 * sum_d2 / (static_cast<double>(n) * (static_cast<double>(n) * n - 1.0));
}

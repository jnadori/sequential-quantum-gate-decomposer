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

/*! \file SurrogateModel.h
    \brief GP surrogate model wrapping GPRegressor + WLKernelCache.
*/

#ifndef SurrogateModel_H
#define SurrogateModel_H

#include "GPRegressor.h"
#include "GrayCode.h"
#include "GrayCodeHash.h"

#include <vector>

class SurrogateModel {
public:
    SurrogateModel(int gp_max_train, double gp_score_ratio,
                   int wl_iterations, const std::vector<int>& token_masks);

    /// Reset GP for a new D-window
    void reset();

    struct TrainResult {
        int n_train;
        double scale;
        std::vector<double> log_y_norm;
        std::vector<int> filtered_x_indices;
    };

    /// Build training set from X/y filtered to [d_lo, d_hi] and train GP
    TrainResult train_window(
        const std::vector<GrayCode>& X, const std::vector<double>& y,
        int d_filter_lo, int d_filter_hi);

    struct AcquisitionResult {
        std::vector<int> selected_indices;
        std::vector<double> gp_mu;
    };

    /// LCB acquisition: select top n_pick candidates
    AcquisitionResult select_candidates_lcb(
        const std::vector<GrayCode>& candidates,
        const GrayCodeSet& seen,
        double kappa, double d_penalty, int win_hi, int n_pick);

    /// Spearman rank correlation
    static double compute_rho(const std::vector<double>& predicted,
                               const std::vector<double>& actual);

    // Access for external users (local_search_acq, mutate_point_guided, etc.)
    GPRegressor& gp() { return gp_; }
    WLKernelCache& cache() { return cache_; }
    double scale() const { return scale_; }
    int n_train() const { return gp_.n_train; }

private:
    GPRegressor gp_;
    WLKernelCache cache_;
    double scale_;
    int gp_max_train_;
    double gp_score_ratio_;

    double prev_log_scale_;
    double prev_noise_;
};

#endif // SurrogateModel_H

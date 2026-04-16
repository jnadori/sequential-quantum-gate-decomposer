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

/*! \file CircuitMutators.h
    \brief Evolutionary mutation operators, local search, and candidate generation.
*/

#ifndef CircuitMutators_H
#define CircuitMutators_H

#include "CircuitCanonicalizer.h"
#include "GPRegressor.h"

#include <random>
#include <vector>

struct MutatorConfig {
    int block_size = 3;
    int tournament_size = 3;
    double local_search_fraction = 0.5;
    int max_local_steps = 10;
    int local_search_positions = 5;
    int local_search_gp_subset = 50;
    int local_search_max_neighbors = 50;
    int local_search_patience = 3;
    double local_search_min_improvement = 1e-4;
    double position_guided_fraction = 0.5;
    double position_lambda = 0.9;
    double position_temperature = -1.0;
};

class CircuitMutators {
public:
    CircuitMutators(CircuitCanonicalizer* canon, const MutatorConfig& config);

    // Mutation operators — take std::mt19937& for RNG
    GrayCode mutate_point(const GrayCode& seq, std::mt19937& rng);
    GrayCode mutate_point_guided(const GrayCode& seq, WLKernelCache& cache,
                                  GPRegressor& gp, double scale, std::mt19937& rng);
    GrayCode mutate_swap(const GrayCode& seq, std::mt19937& rng);
    GrayCode mutate_block(const GrayCode& seq, int blk_size, std::mt19937& rng);
    GrayCode mutate_transplant(const GrayCode& recipient, const GrayCode& donor,
                                int blk_size, std::mt19937& rng);
    GrayCode mutate_block_regenerate(const GrayCode& seq, int blk_size, std::mt19937& rng);
    GrayCode crossover_uniform(const GrayCode& seq1, const GrayCode& seq2, std::mt19937& rng);
    GrayCode mutate_grow(const GrayCode& seq, int D_max, std::mt19937& rng);
    GrayCode mutate_shrink(const GrayCode& seq, int D_min, std::mt19937& rng);
    GrayCode mutate_commute(const GrayCode& seq, std::mt19937& rng);
    GrayCode mutate_cancel_pairs(const GrayCode& seq, std::mt19937& rng);
    GrayCode mutate_insert_identity(const GrayCode& seq, std::mt19937& rng);

    // Local search on LCB acquisition
    GrayCode local_search_acq(const GrayCode& start, WLKernelCache& cache,
                               GPRegressor& gp, double scale,
                               double kappa,
                               int D_min_local = -1, int D_max_local = -1,
                               int* steps_out = nullptr);

    // Hybrid candidate generation
    void generate_candidates(const std::vector<GrayCode>& population,
                              const double* scores, int n_pop,
                              int n_candidates,
                              WLKernelCache& cache, GPRegressor& gp, double scale,
                              const double* train_y_norm,
                              GrayCodeSet& seen,
                              std::vector<GrayCode>& candidates_out,
                              int& n_local_out, double& avg_steps_out,
                              std::mt19937& rng,
                              int D_min_gen = -1, int D_max_gen = -1,
                              double force_random_fraction = 0.0);

private:
    CircuitCanonicalizer* canon_;
    MutatorConfig config_;
};

#endif // CircuitMutators_H

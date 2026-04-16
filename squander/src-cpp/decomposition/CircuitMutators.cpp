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

/*! \file CircuitMutators.cpp
    \brief Evolutionary mutation operators, local search, and candidate generation.
*/

#include "CircuitMutators.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

static constexpr int MUTATION_MAX_ATTEMPTS = 50;

template <typename F>
GrayCode retry_mutate(F&& body) {
    for (int attempt = 0; attempt < MUTATION_MAX_ATTEMPTS; ++attempt) {
        GrayCode result = body(attempt);
        if (result.size() > 0) return result;
    }
    return GrayCode();
}

} // anonymous namespace


// ============================================================================
// Constructor
// ============================================================================

CircuitMutators::CircuitMutators(CircuitCanonicalizer* canon, const MutatorConfig& config)
    : canon_(canon), config_(config) {}


// ============================================================================
// Mutation operators
// ============================================================================

GrayCode CircuitMutators::mutate_point(const GrayCode& seq, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    if (n_tokens < 2) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    CircuitCanonicalizer::CanonicalDAG dag = canon_->build_canonical_dag(seq);

    return retry_mutate([&](int) -> GrayCode {
        int pos = pos_dist(rng);
        int cur_token = seq[pos];
        int new_token;

        const std::vector<int>& nbrs = canon_->token_neighbors()[cur_token];
        if (!nbrs.empty() && coin(rng) < 0.7) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(rng)];
        } else {
            std::uniform_int_distribution<int> token_dist(0, n_tokens - 2);
            new_token = token_dist(rng);
            if (new_token >= cur_token) ++new_token;
        }

        return canon_->canonicalize_and_validate_from_dag(dag, seq, pos, new_token);
    });
}

GrayCode CircuitMutators::mutate_point_guided(
    const GrayCode& seq, WLKernelCache& cache, GPRegressor& gp,
    double scale, std::mt19937& rng) {

    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    if (D < 1 || n_tokens < 2 || gp.n_train == 0) return mutate_point(seq, rng);

    // Compute mu for current circuit
    std::vector<GrayCode> cur_vec = {seq};
    std::vector<double> Ks_cur(gp.n_train);
    cache.compute_cross_kernel_subset(cur_vec, scale, gp.train_indices_, Ks_cur.data());
    double mu_cur = 0;
    for (int i = 0; i < gp.n_train; ++i)
        mu_cur += Ks_cur[i] * gp.alpha_data[i];

    // Generate all single-token substitutions
    std::vector<double> pos_improvement(D, 0.0);
    std::vector<GrayCode> all_variants;
    std::vector<int> variant_pos;
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

    if (all_variants.empty()) return mutate_point(seq, rng);

    int n_var = static_cast<int>(all_variants.size());
    std::vector<double> Ks_var(gp.n_train * n_var);
    cache.compute_cross_kernel_subset(all_variants, scale, gp.train_indices_, Ks_var.data());

    std::vector<double> best_mu_per_pos(D, mu_cur);
    for (int j = 0; j < n_var; ++j) {
        double mu_val = 0.0;
        for (int i = 0; i < gp.n_train; ++i)
            mu_val += Ks_var[i * n_var + j] * gp.alpha_data[i];
        int pos = variant_pos[j];
        if (mu_val < best_mu_per_pos[pos])
            best_mu_per_pos[pos] = mu_val;
    }

    for (int pos = 0; pos < D; ++pos)
        pos_improvement[pos] = mu_cur - best_mu_per_pos[pos];

    // Softmax temperature
    double temperature;
    if (config_.position_temperature > 0.0) {
        temperature = config_.position_temperature;
    } else {
        double lam = config_.position_lambda;
        double temp_arg = lam * (D - 1) / (1.0 - lam);
        temperature = (temp_arg > 1.0) ? 1.0 / std::log(temp_arg) : 1.0;
    }

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

    std::vector<double> cdf(D);
    cdf[0] = weights[0];
    for (int pos = 1; pos < D; ++pos)
        cdf[pos] = cdf[pos - 1] + weights[pos];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    CircuitCanonicalizer::CanonicalDAG dag = canon_->build_canonical_dag(seq);

    return retry_mutate([&](int) -> GrayCode {
        double r = uni(rng);
        int pos = D - 1;
        for (int p = 0; p < D; ++p) {
            if (r <= cdf[p]) { pos = p; break; }
        }

        int cur_token = seq[pos];
        int new_token;

        const std::vector<int>& nbrs = canon_->token_neighbors()[cur_token];
        if (!nbrs.empty() && coin(rng) < 0.7) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(rng)];
        } else {
            std::uniform_int_distribution<int> token_dist(0, n_tokens - 2);
            new_token = token_dist(rng);
            if (new_token >= cur_token) ++new_token;
        }

        return canon_->canonicalize_and_validate_from_dag(dag, seq, pos, new_token);
    });
}

GrayCode CircuitMutators::mutate_swap(const GrayCode& seq, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    if (D < 2) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq.copy();
        int i = pos_dist(rng);
        int j = pos_dist(rng);
        while (j == i) j = pos_dist(rng);
        std::swap(new_seq[i], new_seq[j]);
        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_block(const GrayCode& seq, int blk_size, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    int bs = std::min(blk_size, D);
    std::uniform_int_distribution<int> start_dist(0, D - bs);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq.copy();
        int start = start_dist(rng);
        for (int p = start; p < start + bs; ++p) {
            int cur_token = seq[p];
            const std::vector<int>& nbrs = canon_->token_neighbors()[cur_token];
            if (!nbrs.empty() && coin(rng) < 0.5) {
                std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
                new_seq[p] = nbrs[nbr_dist(rng)];
            } else {
                new_seq[p] = token_dist(rng);
            }
        }
        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_transplant(
        const GrayCode& recipient, const GrayCode& donor, int blk_size, std::mt19937& rng) {
    int D_r = static_cast<int>(recipient.size());
    int D_d = static_cast<int>(donor.size());
    int max_blk = std::min(blk_size, std::min(D_r, D_d));
    if (max_blk < 2) return GrayCode();
    std::uniform_int_distribution<int> blk_dist(2, max_blk);

    return retry_mutate([&](int) -> GrayCode {
        int blk = blk_dist(rng);
        std::uniform_int_distribution<int> r_start_dist(0, D_r - blk);
        std::uniform_int_distribution<int> d_start_dist(0, D_d - blk);
        int r_start = r_start_dist(rng);
        int d_start = d_start_dist(rng);

        GrayCode new_seq = recipient.copy();
        for (int i = 0; i < blk; ++i)
            new_seq[r_start + i] = donor[d_start + i];

        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_block_regenerate(
        const GrayCode& seq, int blk_size, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    int bs = std::min(blk_size, D);
    int max_bs = std::min(2 * bs, D);
    if (bs < 1) return GrayCode();
    std::uniform_int_distribution<int> bs_dist(bs, max_bs);

    return retry_mutate([&](int) -> GrayCode {
        int actual_bs = bs_dist(rng);
        std::uniform_int_distribution<int> start_dist(0, D - actual_bs);
        int start = start_dist(rng);

        GrayCode new_seq = seq.copy();
        std::uniform_int_distribution<int> pick(0, n_tokens - 1);
        for (int pos = start; pos < start + actual_bs; ++pos)
            new_seq[pos] = pick(rng);

        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::crossover_uniform(const GrayCode& seq1, const GrayCode& seq2, std::mt19937& rng) {
    int D = static_cast<int>(seq1.size());
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq1.copy();
        for (int i = 0; i < D; ++i)
            new_seq[i] = (coin(rng) < 0.5) ? seq1[i] : seq2[i];
        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_grow(const GrayCode& seq, int D_max, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    if (D >= D_max) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
        int pos = pos_dist(rng);
        int new_token;
        int adj_token = (pos < D) ? seq[pos] : seq[D - 1];
        const std::vector<int>& nbrs = canon_->token_neighbors()[adj_token];
        if (!nbrs.empty() && coin(rng) < 0.5) {
            std::uniform_int_distribution<int> nbr_dist(0, static_cast<int>(nbrs.size()) - 1);
            new_token = nbrs[nbr_dist(rng)];
        } else {
            new_token = token_dist(rng);
        }

        matrix_base<int> limits(1, D + 1);
        for (int i = 0; i < D + 1; ++i) limits[i] = n_tokens;
        GrayCode new_seq(limits);
        for (int i = 0; i < pos; ++i) new_seq[i] = seq[i];
        new_seq[pos] = new_token;
        for (int i = pos; i < D; ++i) new_seq[i + 1] = seq[i];

        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_shrink(const GrayCode& seq, int D_min, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    if (D <= D_min) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    return retry_mutate([&](int) -> GrayCode {
        int pos = pos_dist(rng);
        GrayCode new_seq = seq.remove_Digit(pos);
        return canon_->canonicalize_and_validate(new_seq);
    });
}


// ============================================================================
// P2: Quantum-aware gate-rewriting mutations
// ============================================================================

GrayCode CircuitMutators::mutate_commute(const GrayCode& seq, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    if (D < 2) return GrayCode();

    const std::vector<int>& tmasks = canon_->token_masks();
    std::vector<int> commutable_positions;
    for (int i = 0; i < D - 1; ++i) {
        if ((tmasks[seq[i]] & tmasks[seq[i + 1]]) == 0)
            commutable_positions.push_back(i);
    }

    if (commutable_positions.empty()) return GrayCode();

    std::uniform_int_distribution<int> pair_dist(0, static_cast<int>(commutable_positions.size()) - 1);
    int pos = commutable_positions[pair_dist(rng)];

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq.copy();
        std::swap(new_seq[pos], new_seq[pos + 1]);
        return canon_->canonicalize_and_validate(new_seq);
    });
}

GrayCode CircuitMutators::mutate_cancel_pairs(const GrayCode& seq, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    if (D < 2) return GrayCode();

    const std::vector<int>& tmasks = canon_->token_masks();

    std::vector<int> cancel_positions;
    for (int i = 0; i < D - 1; ++i) {
        if (seq[i] == seq[i + 1])
            cancel_positions.push_back(i);
    }

    if (cancel_positions.empty()) {
        for (int i = 0; i < D - 1; ++i) {
            for (int j = i + 1; j < D; ++j) {
                if (seq[i] != seq[j]) continue;
                int mask_ij = tmasks[seq[i]];
                bool all_disjoint = true;
                for (int k = i + 1; k < j; ++k) {
                    if (tmasks[seq[k]] & mask_ij) {
                        all_disjoint = false;
                        break;
                    }
                }
                if (all_disjoint) {
                    GrayCode new_seq = seq.copy();
                    for (int k = j; k > i + 1; --k)
                        std::swap(new_seq[k], new_seq[k - 1]);
                    GrayCode result = new_seq.remove_Digit(i);
                    result = result.remove_Digit(i);
                    return canon_->canonicalize_and_validate(result);
                }
            }
        }
        return GrayCode();
    }

    std::uniform_int_distribution<int> pair_dist(0, static_cast<int>(cancel_positions.size()) - 1);
    int pos = cancel_positions[pair_dist(rng)];

    GrayCode new_seq = seq.remove_Digit(pos);
    new_seq = new_seq.remove_Digit(pos);
    return canon_->canonicalize_and_validate(new_seq);
}

GrayCode CircuitMutators::mutate_insert_identity(const GrayCode& seq, std::mt19937& rng) {
    int D = static_cast<int>(seq.size());
    int n_tokens = canon_->n_tokens();
    if (D < 2 || n_tokens < 1) return GrayCode();

    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_int_distribution<int> pos_dist(0, D);

    return retry_mutate([&](int) -> GrayCode {
        int token = token_dist(rng);
        int pos = pos_dist(rng);

        matrix_base<int> limits(1, D + 2);
        for (int i = 0; i < D + 2; ++i) limits[i] = n_tokens;
        GrayCode new_seq(0, limits);
        for (int i = 0; i < pos; ++i) new_seq[i] = seq[i];
        new_seq[pos] = token;
        new_seq[pos + 1] = token;
        for (int i = pos; i < D; ++i) new_seq[i + 2] = seq[i];

        return canon_->canonicalize_and_validate(new_seq);
    });
}


// ============================================================================
// Local search on LCB acquisition
// ============================================================================

GrayCode CircuitMutators::local_search_acq(
    const GrayCode& start, WLKernelCache& cache, GPRegressor& gp,
    double scale, double kappa,
    int D_min_local, int D_max_local, int* steps_out) {

    thread_local std::mt19937 local_rng(std::random_device{}());

    if (gp.n_train == 0) {
        if (steps_out) *steps_out = 0;
        return start.copy();
    }

    int n_tokens = canon_->n_tokens();

    GrayCode current = start.copy();

    // Evaluate LCB at start
    std::vector<GrayCode> start_vec = {current};
    std::vector<double> Ks_cur(gp.n_train);
    cache.compute_cross_kernel_subset(start_vec, scale, gp.train_indices_, Ks_cur.data());
    double mu_cur = 0;
    for (int i = 0; i < gp.n_train; ++i)
        mu_cur += Ks_cur[i] * gp.alpha_data[i];

    std::vector<double> v_cur(gp.n_train);
    std::memcpy(v_cur.data(), Ks_cur.data(), gp.n_train * sizeof(double));
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                   gp.n_train, 1, gp.L_data.data(), gp.n_train,
                   v_cur.data(), 1);
    double sum_v2 = 0;
    for (int i = 0; i < gp.n_train; ++i)
        sum_v2 += v_cur[i] * v_cur[i];
    double std_cur = std::sqrt(std::max(scale - sum_v2, 1e-10));
    double best_lcb = mu_cur - kappa * std_cur;

    int steps_taken = 0;
    int no_improve_streak = 0;
    std::vector<int> pos_visit_count(static_cast<int>(current.size()), 0);

    for (int step = 0; step < config_.max_local_steps; ++step) {
        int D = static_cast<int>(current.size());

        if (static_cast<int>(pos_visit_count.size()) != D)
            pos_visit_count.assign(D, 0);

        int min_visits = *std::min_element(pos_visit_count.begin(), pos_visit_count.end());

        std::vector<GrayCode> neighbors;
        std::vector<int> neighbor_pos;

        CircuitCanonicalizer::CanonicalDAG dag = canon_->build_canonical_dag(current);

        int n_pos_to_try = (config_.local_search_positions > 0 && config_.local_search_positions < D)
                           ? config_.local_search_positions : D;

        std::vector<int> candidate_positions;
        candidate_positions.reserve(D);
        for (int pos = 0; pos < D; ++pos) {
            if (pos_visit_count[pos] <= min_visits)
                candidate_positions.push_back(pos);
        }
        if (static_cast<int>(candidate_positions.size()) > n_pos_to_try) {
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
                GrayCode cand = canon_->canonicalize_and_validate_from_dag(dag, current, pos, e);
                if (cand.size() > 0) {
                    neighbors.push_back(std::move(cand));
                    neighbor_pos.push_back(pos);
                }
            }
        }

        // Grow neighbors
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
                    GrayCode cand = canon_->canonicalize_and_validate(new_seq);
                    if (cand.size() > 0) {
                        neighbors.push_back(std::move(cand));
                        neighbor_pos.push_back(-1);
                    }
                }
            }
        }

        // Shrink neighbors
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
                GrayCode cand = canon_->canonicalize_and_validate(new_seq);
                if (cand.size() > 0) {
                    neighbors.push_back(std::move(cand));
                    neighbor_pos.push_back(-1);
                }
            }
        }

        // Cap total neighbors
        if (config_.local_search_max_neighbors > 0 &&
            static_cast<int>(neighbors.size()) > config_.local_search_max_neighbors) {
            for (int i = 0; i < config_.local_search_max_neighbors; ++i) {
                std::uniform_int_distribution<int> sd(i, static_cast<int>(neighbors.size()) - 1);
                int j = sd(local_rng);
                if (i != j) {
                    GrayCode tmp = neighbors[i].copy();
                    neighbors[i] = neighbors[j];
                    neighbors[j] = tmp;
                    std::swap(neighbor_pos[i], neighbor_pos[j]);
                }
            }
            neighbors.resize(config_.local_search_max_neighbors);
            neighbor_pos.resize(config_.local_search_max_neighbors);
        }

        // Deduplicate
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

        // Batch-evaluate LCB
        int n_nb = static_cast<int>(neighbors.size());
        std::vector<double> Ks(gp.n_train * n_nb);
        cache.compute_cross_kernel_subset(neighbors, scale, gp.train_indices_, Ks.data());

        int n_t = gp.n_train;
        std::vector<double> mu(n_nb);
        std::vector<double> lcb_approx(n_nb);

        if (gp.love_valid && gp.love_rank > 0) {
            for (int j = 0; j < n_nb; ++j) {
                double mu_val = 0.0;
                for (int i = 0; i < n_t; ++i)
                    mu_val += Ks[i * n_nb + j] * gp.alpha_data[i];
                mu[j] = mu_val;

                double sum_rt_ks_sq = 0.0;
                for (int k = 0; k < gp.love_rank; ++k) {
                    double rt_ks_k = 0.0;
                    for (int i = 0; i < n_t; ++i)
                        rt_ks_k += gp.R_love[i * gp.love_rank + k] * Ks[i * n_nb + j];
                    sum_rt_ks_sq += rt_ks_k * rt_ks_k;
                }
                double var = std::max(scale - sum_rt_ks_sq, 1e-10);
                lcb_approx[j] = mu_val - kappa * std::sqrt(var);
            }
        } else {
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
                double var_approx = std::max(scale - v2_sum, 1e-10);
                lcb_approx[j] = mu_val - kappa * std::sqrt(var_approx);
            }
        }

        // Select top-k for exact evaluation
        constexpr int k_exact = 5;
        int k = std::min(k_exact, n_nb);

        std::vector<int> order(n_nb);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int a, int b) { return lcb_approx[a] < lcb_approx[b]; });

        double best_nb_lcb = std::numeric_limits<double>::infinity();
        int best_nb_idx = -1;
        std::vector<double> v_col(n_t);

        for (int rank = 0; rank < k; ++rank) {
            int j = order[rank];
            if (mu[j] >= best_nb_lcb) continue;

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
            double std_val = std::sqrt(std::max(scale - sv2, 1e-10));
            double lcb_val = mu[j] - kappa * std_val;
            if (lcb_val < best_nb_lcb) {
                best_nb_lcb = lcb_val;
                best_nb_idx = j;
            }
        }

        if (best_nb_lcb >= best_lcb) break;

        double improvement = best_lcb - best_nb_lcb;
        if (improvement < config_.local_search_min_improvement * std::abs(best_lcb) + 1e-12)
            no_improve_streak++;
        else
            no_improve_streak = 0;
        if (config_.local_search_patience > 0 && no_improve_streak >= config_.local_search_patience) break;

        int selected_pos = neighbor_pos[best_nb_idx];
        if (selected_pos >= 0 && selected_pos < static_cast<int>(pos_visit_count.size())) {
            pos_visit_count[selected_pos]++;
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

void CircuitMutators::generate_candidates(
    const std::vector<GrayCode>& population, const double* scores, int n_pop,
    int n_candidates, WLKernelCache& cache, GPRegressor& gp, double scale,
    const double* train_y_norm,
    GrayCodeSet& seen, std::vector<GrayCode>& candidates_out,
    int& n_local_out, double& avg_steps_out,
    std::mt19937& rng,
    int D_min_gen, int D_max_gen, double force_random_fraction) {

    candidates_out.clear();
    bool mixed_d = (D_min_gen >= 0 && D_max_gen >= 0);
    int n_tokens = canon_->n_tokens();

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
    int t_size = std::min(config_.tournament_size, n_valid);

    std::uniform_int_distribution<int> valid_dist(0, n_valid - 1);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);

    auto tournament_select = [&]() -> const GrayCode& {
        int best_idx = pop_indices[valid_dist(rng)];
        for (int i = 1; i < t_size; ++i) {
            int idx = pop_indices[valid_dist(rng)];
            if (scores[idx] < scores[best_idx])
                best_idx = idx;
        }
        return population[best_idx];
    };

    // GP-directed local search candidates
    int n_local_found = 0;
    int total_local_steps = 0;
    int n_local_target = static_cast<int>(n_candidates * config_.local_search_fraction);

    if (n_local_target > 0) {
        std::vector<GrayCode> local_parents(n_local_target);
        std::uniform_real_distribution<double> frac_dist(0.0, 1.0);
        int D_rand_lo = (D_min_gen >= 0) ? D_min_gen : 1;
        int D_rand_hi = (D_max_gen >= 0) ? D_max_gen : (n_pop > 0 ? static_cast<int>(population[0].size()) : 1);
        std::uniform_int_distribution<int> D_rand_dist(D_rand_lo, std::max(D_rand_lo, D_rand_hi));
        for (int i = 0; i < n_local_target; ++i) {
            if (force_random_fraction > 0.0 && frac_dist(rng) < force_random_fraction) {
                int D_val = D_rand_dist(rng);
                GrayCode rand_parent = canon_->generate_valid_sequence(D_val, rng);
                local_parents[i] = (rand_parent.size() > 0) ? rand_parent : tournament_select().copy();
            } else {
                local_parents[i] = tournament_select().copy();
            }
        }

        // Lightweight GP proxy
        GPRegressor* local_gp_ptr = &gp;
        GPRegressor local_gp;
        double local_scale = scale;

        int n_full = gp.n_train;
        int n_sub = config_.local_search_gp_subset;
        if (n_sub > 0 && n_sub < n_full && train_y_norm != nullptr) {
            int n_best = n_sub / 2;
            int n_rand = n_sub - n_best;

            std::vector<int> sub_indices;
            sub_indices.reserve(n_sub);
            for (int i = 0; i < n_best && i < n_full; ++i)
                sub_indices.push_back(i);

            std::vector<int> remaining;
            remaining.reserve(n_full - n_best);
            for (int i = n_best; i < n_full; ++i)
                remaining.push_back(i);
            for (int i = 0; i < n_rand && !remaining.empty(); ++i) {
                std::uniform_int_distribution<int> rdist(0, static_cast<int>(remaining.size()) - 1);
                int pick = rdist(rng);
                sub_indices.push_back(remaining[pick]);
                remaining[pick] = remaining.back();
                remaining.pop_back();
            }

            std::sort(sub_indices.begin(), sub_indices.end());
            int n_actual = static_cast<int>(sub_indices.size());

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

        std::vector<GrayCode> local_results(n_local_target);
        std::vector<int> local_steps(n_local_target, 0);

        tbb::parallel_for(
            tbb::blocked_range<int>(0, n_local_target, 1),
            [&](tbb::blocked_range<int> r) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    local_results[i] = local_search_acq(
                        local_parents[i], cache, *local_gp_ptr, local_scale,
                        0.0,  // kappa not used in local_search_acq for the inner loop
                        D_min_gen, D_max_gen, &local_steps[i]);
                }
            }
        );

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

    // Position-guided or uniform point mutation
    bool use_guided = (config_.position_guided_fraction > 0 && gp.n_train > 0);
    auto maybe_guided_point = [&](const GrayCode& parent) -> GrayCode {
        if (use_guided && op_dist(rng) < config_.position_guided_fraction)
            return mutate_point_guided(parent, cache, gp, scale, rng);
        return mutate_point(parent, rng);
    };

    GrayCodeSet batch_seen;
    for (auto& c : candidates_out)
        batch_seen.insert(c.copy());

    int max_attempts = n_candidates * 10;
    for (int attempt = 0;
         static_cast<int>(candidates_out.size()) < n_candidates && attempt < max_attempts;
         ++attempt) {
        double r = op_dist(rng);
        GrayCode result;

        int ref_D = mixed_d ? (D_min_gen + D_max_gen) / 2
                            : static_cast<int>(population[0].size());
        int adaptive_blk = std::max(config_.block_size, ref_D / 3);

        if (mixed_d) {
            if (r < 0.10)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.15)
                result = mutate_swap(tournament_select(), rng);
            else if (r < 0.30)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk, rng);
            else if (r < 0.50) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk, rng);
            }
            else if (r < 0.60) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                if (p1.size() == p2.size())
                    result = crossover_uniform(p1, p2, rng);
                else
                    result = maybe_guided_point(p1);
            }
            else if (r < 0.65)
                result = mutate_grow(tournament_select(), D_max_gen, rng);
            else if (r < 0.70)
                result = mutate_commute(tournament_select(), rng);
            else if (r < 0.75)
                result = mutate_cancel_pairs(tournament_select(), rng);
            else if (r < 0.80)
                result = mutate_shrink(tournament_select(), D_min_gen, rng);
            else if (r < 0.85)
                result = mutate_block(tournament_select(), config_.block_size, rng);
            else if (r < 0.90)
                result = mutate_insert_identity(tournament_select(), rng);
            else {
                std::geometric_distribution<int> geo(0.4);
                int rand_D = D_min_gen + std::min(geo(rng), D_max_gen - D_min_gen);
                result = canon_->generate_valid_sequence(rand_D, rng);
            }
        } else {
            if (r < 0.13)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.18)
                result = mutate_swap(tournament_select(), rng);
            else if (r < 0.33)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk, rng);
            else if (r < 0.53) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk, rng);
            }
            else if (r < 0.63)
                result = crossover_uniform(tournament_select(), tournament_select(), rng);
            else if (r < 0.68)
                result = mutate_commute(tournament_select(), rng);
            else if (r < 0.73)
                result = mutate_cancel_pairs(tournament_select(), rng);
            else if (r < 0.78)
                result = mutate_block(tournament_select(), config_.block_size, rng);
            else if (r < 0.83)
                result = mutate_insert_identity(tournament_select(), rng);
            else {
                int D = static_cast<int>(population[0].size());
                result = canon_->generate_valid_sequence(D, rng);
            }
        }

        if (result.size() > 0 && batch_seen.find(result) == batch_seen.end() &&
            seen.find(result) == seen.end() &&
            (!mixed_d || (static_cast<int>(result.size()) >= D_min_gen &&
                          static_cast<int>(result.size()) <= D_max_gen))) {
            batch_seen.insert(result.copy());
            candidates_out.push_back(std::move(result));
        }
    }

    n_local_out = n_local_found;
    avg_steps_out = (n_local_found > 0) ?
        static_cast<double>(total_local_steps) / n_local_found : 0.0;
}

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

/*! \file GateRemover.cpp
    \brief Phase 1 single-gate removal + Phase 1.5 multi-gate pair removal.
*/

#include "GateRemover.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>


// ============================================================================
// Constructor
// ============================================================================

GateRemover::GateRemover(CircuitCanonicalizer* canon, int qbit_num)
    : canon_(canon), qbit_num_(qbit_num) {}


// ============================================================================
// Triviality score
// ============================================================================

double GateRemover::triviality_score(const Matrix_real& params, int pos) {
    double total = 0.0;
    for (int p = 0; p < 6; ++p) {
        double theta = params[pos * 6 + p];
        double k = std::round(theta / (M_PI / 2.0));
        total += std::abs(theta - k * M_PI / 2.0);
    }
    return total;
}


// ============================================================================
// Parameter extraction
// ============================================================================

Matrix_real GateRemover::excise_gate_params(
    const Matrix_real& parent_params, int D_plus1, int pos, int qbit_num) {

    int gate_params = (D_plus1 - 1) * 6;
    int final_params = 3 * qbit_num;
    int child_param_num = gate_params + final_params;
    int parent_gate_params = D_plus1 * 6;

    Matrix_real child_params(1, child_param_num);

    int dst = 0;
    for (int i = 0; i < pos * 6; ++i)
        child_params[dst++] = parent_params[i];
    for (int i = (pos + 1) * 6; i < parent_gate_params; ++i)
        child_params[dst++] = parent_params[i];
    for (int i = 0; i < final_params; ++i)
        child_params[dst++] = parent_params[parent_gate_params + i];

    return child_params;
}

Matrix_real GateRemover::excise_multi_gate_params(
    const Matrix_real& parent_params, int D_source,
    const std::vector<int>& positions_sorted_desc, int qbit_num) {

    int n_remove = static_cast<int>(positions_sorted_desc.size());
    int D_child = D_source - n_remove;
    int final_params = 3 * qbit_num;
    int child_param_num = D_child * 6 + final_params;
    int parent_gate_params = D_source * 6;

    std::vector<bool> remove(D_source, false);
    for (int pos : positions_sorted_desc)
        remove[pos] = true;

    Matrix_real child_params(1, child_param_num);
    int dst = 0;

    for (int g = 0; g < D_source; ++g) {
        if (!remove[g]) {
            for (int p = 0; p < 6; ++p)
                child_params[dst++] = parent_params[g * 6 + p];
        }
    }

    for (int i = 0; i < final_params; ++i)
        child_params[dst++] = parent_params[parent_gate_params + i];

    return child_params;
}


// ============================================================================
// Phase 1: single-gate removal
// ============================================================================

GateRemover::RemovalResult GateRemover::try_single_removals(
    const std::vector<GrayCode>& X,
    const std::vector<double>& y,
    const std::vector<Matrix_real>& all_params,
    GrayCodeSet& seen,
    int D,
    Phase1Data& phase1_data_out,
    int phase1_topk,
    int max_removals_factor) {

    RemovalResult result;
    phase1_data_out.entries.clear();

    // Collect all D+1 circuits sorted by score
    std::vector<int> dplus1_indices;
    for (size_t i = 0; i < X.size(); ++i) {
        if (static_cast<int>(X[i].size()) == D + 1)
            dplus1_indices.push_back(static_cast<int>(i));
    }
    std::sort(dplus1_indices.begin(), dplus1_indices.end(),
        [&y](int a, int b) { return y[a] < y[b]; });

    int n_parents = std::min(static_cast<int>(dplus1_indices.size()), phase1_topk);
    if (n_parents == 0) return result;

    const int max_removals = max_removals_factor * (D + 1);
    struct RemovalEntry {
        GrayCode circuit;
        int position;
        double triviality;
        int parent_pi;  // index into dplus1_indices
    };
    std::vector<RemovalEntry> removal_entries;

    for (int pi = 0; pi < n_parents && static_cast<int>(removal_entries.size()) < max_removals; ++pi) {
        int parent_x_idx = dplus1_indices[pi];
        const GrayCode& parent_circ = X[parent_x_idx];
        int D_plus1 = static_cast<int>(parent_circ.size());
        const Matrix_real& parent_params = all_params[parent_x_idx];

        for (int pos = 0; pos < D_plus1 && static_cast<int>(removal_entries.size()) < max_removals; ++pos) {
            GrayCode shortened = parent_circ.remove_Digit(pos);
            GrayCode canon = canon_->canonicalize_and_validate(shortened);
            if (canon.size() > 0 && seen.find(canon) == seen.end()) {
                seen.insert(canon.copy());
                removal_entries.push_back({std::move(canon), pos,
                                           triviality_score(parent_params, pos), pi});
            }
        }
    }

    // Sort: by parent rank, then by triviality
    std::stable_sort(removal_entries.begin(), removal_entries.end(),
        [](const RemovalEntry& a, const RemovalEntry& b) {
            if (a.parent_pi != b.parent_pi) return a.parent_pi < b.parent_pi;
            return a.triviality < b.triviality;
        });

    // Build output vectors
    result.candidates.reserve(removal_entries.size());
    result.warmstart_params.reserve(removal_entries.size());

    for (auto& e : removal_entries) {
        int parent_x_idx = dplus1_indices[e.parent_pi];
        int D_plus1 = static_cast<int>(X[parent_x_idx].size());

        // Store Phase1 data
        phase1_data_out.entries.push_back({
            parent_x_idx, e.position,
            std::numeric_limits<double>::infinity(),  // score filled in after evaluation
            e.triviality
        });

        result.candidates.push_back(std::move(e.circuit));
        result.warmstart_params.push_back(
            excise_gate_params(all_params[parent_x_idx], D_plus1, e.position, qbit_num_));
    }

    return result;
}


// ============================================================================
// Phase 1.5: multi-gate pair removal
// ============================================================================

GateRemover::RemovalResult GateRemover::try_pair_removals(
    const std::vector<GrayCode>& X,
    const std::vector<double>& y,
    const std::vector<Matrix_real>& all_params,
    const Phase1Data& phase1_data,
    GrayCodeSet& seen,
    int D,
    int max_pairs,
    int top_k_positions) {

    RemovalResult result;

    // Group Phase1Data entries by parent
    std::unordered_map<int, std::vector<int>> parent_to_entries;
    for (int i = 0; i < static_cast<int>(phase1_data.entries.size()); ++i) {
        parent_to_entries[phase1_data.entries[i].parent_x_idx].push_back(i);
    }

    // For each parent, sort entries by score and generate pairs
    struct PairCandidate {
        int parent_x_idx;
        int pos_a;
        int pos_b;
        double pair_priority;
    };
    std::vector<PairCandidate> all_pairs;

    for (auto& [parent_x_idx, entry_indices] : parent_to_entries) {
        const GrayCode& parent_circ = X[parent_x_idx];
        int D_plus1 = static_cast<int>(parent_circ.size());

        // Sort by triviality (lower = more removable)
        std::sort(entry_indices.begin(), entry_indices.end(),
            [&](int a, int b) {
                return phase1_data.entries[a].triviality < phase1_data.entries[b].triviality;
            });

        int K = std::min(top_k_positions, static_cast<int>(entry_indices.size()));
        K = std::min(K, D_plus1 / 2);

        // Generate all pairs from top-K positions
        for (int i = 0; i < K; ++i) {
            for (int j = i + 1; j < K; ++j) {
                int pos_i = phase1_data.entries[entry_indices[i]].position;
                int pos_j = phase1_data.entries[entry_indices[j]].position;
                double priority = static_cast<double>(i) + static_cast<double>(j);
                all_pairs.push_back({parent_x_idx, pos_i, pos_j, priority});
            }
        }
    }

    // Sort pairs by parent score rank, then pair_priority
    // Build parent rank map
    std::unordered_map<int, double> parent_score;
    for (auto& [px_idx, _] : parent_to_entries)
        parent_score[px_idx] = y[px_idx];

    std::sort(all_pairs.begin(), all_pairs.end(),
        [&](const PairCandidate& a, const PairCandidate& b) {
            double sa = parent_score[a.parent_x_idx];
            double sb = parent_score[b.parent_x_idx];
            if (sa != sb) return sa < sb;
            return a.pair_priority < b.pair_priority;
        });

    // Generate candidates from top max_pairs pairs
    int generated = 0;
    for (auto& pc : all_pairs) {
        if (generated >= max_pairs) break;

        const GrayCode& parent_circ = X[pc.parent_x_idx];
        int D_plus1 = static_cast<int>(parent_circ.size());

        // Remove higher index first to preserve lower index
        int hi = std::max(pc.pos_a, pc.pos_b);
        int lo = std::min(pc.pos_a, pc.pos_b);

        GrayCode shortened = parent_circ.remove_Digit(hi);
        shortened = shortened.remove_Digit(lo);  // lo index still valid after hi removed

        GrayCode canon = canon_->canonicalize_and_validate(shortened);
        if (canon.size() > 0 && seen.find(canon) == seen.end()) {
            seen.insert(canon.copy());
            result.candidates.push_back(std::move(canon));
            result.warmstart_params.push_back(
                excise_multi_gate_params(
                    all_params[pc.parent_x_idx], D_plus1,
                    {hi, lo}, qbit_num_));
            generated++;
        }
    }

    return result;
}

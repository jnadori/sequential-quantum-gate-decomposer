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

/*! \file CircuitCanonicalizer.h
    \brief Circuit validation, canonical form computation, and sequence generation.
*/

#ifndef CircuitCanonicalizer_H
#define CircuitCanonicalizer_H

#include "GrayCode.h"
#include "GrayCodeHash.h"
#include "GPRegressor.h"

#include <functional>
#include <map>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

class CircuitCanonicalizer {
public:
    using TokenSortKeyFn = std::function<std::tuple<int,int,int>(int)>;

    /// Cached DAG structure for incremental canonical form updates
    struct CanonicalDAG {
        std::vector<std::vector<int>> adj;
        std::vector<int> in_degree;
        std::vector<int> masks;
        int n;
    };

    CircuitCanonicalizer(
        const std::vector<int>& token_masks,
        const std::vector<std::vector<int>>& token_neighbors,
        const std::vector<std::pair<int,int>>& thresholds,
        int n_tokens,
        bool has_custom_topology,
        const std::vector<matrix_base<int>>& topology,
        const std::vector<std::vector<int>>& osr_cuts,
        const std::vector<int>& osr_cut_bounds,
        const std::vector<std::vector<int>>& osr_cut_crossing_edges,
        TokenSortKeyFn token_sort_key_fn);

    // Canonical form and validation
    GrayCode canonical_form(const GrayCode& seq);
    GrayCode canonicalize_and_validate(const GrayCode& seq);
    bool check_osr_feasibility(const GrayCode& circuit);

    // Incremental canonical DAG
    CanonicalDAG build_canonical_dag(const GrayCode& seq);
    GrayCode canonical_form_from_dag(const CanonicalDAG& dag, const GrayCode& seq);
    void update_dag_point_mutation(CanonicalDAG& dag, int pos, int old_token, int new_token);
    GrayCode canonicalize_and_validate_from_dag(CanonicalDAG& dag, const GrayCode& seq, int pos, int new_token);

    // Subspace violation check
    bool check_new_position(const int* window_masks, int pos);

    // Enumeration and random generation
    std::vector<GrayCode> enumerate_circuits(int D);
    GrayCode generate_valid_sequence(int D, std::mt19937& rng);

    // Accessors
    int n_tokens() const { return n_tokens_; }
    const std::vector<int>& token_masks() const { return token_masks_; }
    const std::vector<std::vector<int>>& token_neighbors() const { return token_neighbors_; }

private:
    // Topology data
    std::vector<int> token_masks_;
    std::vector<std::vector<int>> token_neighbors_;
    std::vector<std::pair<int,int>> thresholds_;
    int n_tokens_;

    // OSR data
    bool has_custom_topology_;
    std::vector<matrix_base<int>> topology_;
    std::vector<std::vector<int>> osr_cuts_;
    std::vector<int> osr_cut_bounds_;
    std::vector<std::vector<int>> osr_cut_crossing_edges_;

    // Virtual method callback
    TokenSortKeyFn token_sort_key_fn_;
};

#endif // CircuitCanonicalizer_H

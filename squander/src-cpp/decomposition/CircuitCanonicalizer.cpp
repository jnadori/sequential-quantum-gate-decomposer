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

/*! \file CircuitCanonicalizer.cpp
    \brief Circuit validation, canonical form computation, and sequence generation.
*/

#include "CircuitCanonicalizer.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>


// ============================================================================
// Constructor
// ============================================================================

CircuitCanonicalizer::CircuitCanonicalizer(
    const std::vector<int>& token_masks,
    const std::vector<std::vector<int>>& token_neighbors,
    const std::vector<std::pair<int,int>>& thresholds,
    int n_tokens,
    bool has_custom_topology,
    const std::vector<matrix_base<int>>& topology,
    const std::vector<std::vector<int>>& osr_cuts,
    const std::vector<int>& osr_cut_bounds,
    const std::vector<std::vector<int>>& osr_cut_crossing_edges,
    TokenSortKeyFn token_sort_key_fn)
    : token_masks_(token_masks),
      token_neighbors_(token_neighbors),
      thresholds_(thresholds),
      n_tokens_(n_tokens),
      has_custom_topology_(has_custom_topology),
      topology_(topology),
      osr_cuts_(osr_cuts),
      osr_cut_bounds_(osr_cut_bounds),
      osr_cut_crossing_edges_(osr_cut_crossing_edges),
      token_sort_key_fn_(std::move(token_sort_key_fn)) {}


// ============================================================================
// Validation
// ============================================================================

bool CircuitCanonicalizer::check_new_position(const int* window_masks, int pos) {
    for (size_t t = 0; t < thresholds_.size(); ++t) {
        int n = thresholds_[t].first;
        int j = thresholds_[t].second;
        if (j > pos + 1) continue;
        int combined = 0;
        for (int i = pos - j + 1; i <= pos; ++i)
            combined |= window_masks[i];
        if (__builtin_popcount(combined) <= n)
            return true;  // subspace violation
    }
    return false;
}


// ============================================================================
// Canonical form computation
// ============================================================================

GrayCode CircuitCanonicalizer::canonical_form(const GrayCode& seq) {
    int n = static_cast<int>(seq.size());
    if (n == 0) return GrayCode();

    // Build dependency DAG using token bitmasks
    std::vector<std::vector<int>> adj(n);
    std::vector<int> in_degree(n, 0);

    std::vector<int> masks(n);
    for (int i = 0; i < n; ++i)
        masks[i] = token_masks_[seq[i]];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (masks[j] & masks[i]) {
                adj[j].push_back(i);
                in_degree[i]++;
            }
        }
    }

    // Min-heap topological sort using token_sort_key
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
        if (in_degree[i] == 0)
            heap.push(HeapNode{token_sort_key_fn_(seq[i]), i});
    }

    matrix_base<int> limits(1, n);
    for (int i = 0; i < n; ++i)
        limits[i] = n_tokens_;
    GrayCode result(limits);

    int pos = 0;
    while (!heap.empty()) {
        HeapNode top = heap.top();
        heap.pop();
        result[pos++] = seq[top.idx];
        for (int neighbor : adj[top.idx]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0)
                heap.push(HeapNode{token_sort_key_fn_(seq[neighbor]), neighbor});
        }
    }

    return result;
}


// ============================================================================
// Incremental canonical DAG methods
// ============================================================================

CircuitCanonicalizer::CanonicalDAG
CircuitCanonicalizer::build_canonical_dag(const GrayCode& seq) {
    int n = static_cast<int>(seq.size());
    CanonicalDAG dag;
    dag.n = n;
    dag.adj.assign(n, std::vector<int>());
    dag.in_degree.assign(n, 0);
    dag.masks.resize(n);
    for (int i = 0; i < n; ++i)
        dag.masks[i] = token_masks_[seq[i]];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dag.masks[j] & dag.masks[i]) {
                dag.adj[j].push_back(i);
                dag.in_degree[i]++;
            }
        }
    }
    return dag;
}

GrayCode CircuitCanonicalizer::canonical_form_from_dag(
    const CanonicalDAG& dag, const GrayCode& seq) {

    int n = dag.n;
    if (n == 0) return GrayCode();

    std::vector<int> in_deg = dag.in_degree;

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
        if (in_deg[i] == 0)
            heap.push(HeapNode{token_sort_key_fn_(seq[i]), i});
    }

    matrix_base<int> limits(1, n);
    for (int i = 0; i < n; ++i)
        limits[i] = n_tokens_;
    GrayCode result(limits);

    int pos = 0;
    while (!heap.empty()) {
        HeapNode top = heap.top();
        heap.pop();
        result[pos++] = seq[top.idx];
        for (int neighbor : dag.adj[top.idx]) {
            in_deg[neighbor]--;
            if (in_deg[neighbor] == 0)
                heap.push(HeapNode{token_sort_key_fn_(seq[neighbor]), neighbor});
        }
    }

    return result;
}

void CircuitCanonicalizer::update_dag_point_mutation(
    CanonicalDAG& dag, int pos, int old_token, int new_token) {

    int old_mask = dag.masks[pos];
    int new_mask = token_masks_[new_token];

    if (old_mask == new_mask) return;

    dag.masks[pos] = new_mask;

    for (int q = 0; q < dag.n; ++q) {
        if (q == pos) continue;

        bool old_dep = (old_mask & dag.masks[q]) != 0;
        bool new_dep = (new_mask & dag.masks[q]) != 0;

        if (old_dep == new_dep) continue;

        if (q < pos) {
            if (old_dep && !new_dep) {
                auto& adj_q = dag.adj[q];
                adj_q.erase(std::remove(adj_q.begin(), adj_q.end(), pos), adj_q.end());
                dag.in_degree[pos]--;
            } else {
                dag.adj[q].push_back(pos);
                dag.in_degree[pos]++;
            }
        } else {
            if (old_dep && !new_dep) {
                auto& adj_pos = dag.adj[pos];
                adj_pos.erase(std::remove(adj_pos.begin(), adj_pos.end(), q), adj_pos.end());
                dag.in_degree[q]--;
            } else {
                dag.adj[pos].push_back(q);
                dag.in_degree[q]++;
            }
        }
    }
}

GrayCode CircuitCanonicalizer::canonicalize_and_validate_from_dag(
    CanonicalDAG& dag, const GrayCode& seq, int pos, int new_token) {

    int old_token = seq[pos];

    update_dag_point_mutation(dag, pos, old_token, new_token);

    GrayCode mutated = seq.copy();
    mutated[pos] = new_token;

    GrayCode canon = canonical_form_from_dag(dag, mutated);

    // Restore DAG to original state
    update_dag_point_mutation(dag, pos, new_token, old_token);

    if (canon.size() == 0) return GrayCode();
    if (!has_custom_topology_ && !check_osr_feasibility(canon))
        return GrayCode();
    return canon;
}

GrayCode CircuitCanonicalizer::canonicalize_and_validate(const GrayCode& seq) {
    GrayCode canon = canonical_form(seq);

    if (!has_custom_topology_ && !check_osr_feasibility(canon))
        return GrayCode();

    return canon;
}

bool CircuitCanonicalizer::check_osr_feasibility(const GrayCode& circuit) {
    int n_edges = static_cast<int>(topology_.size());
    std::vector<int> edge_counts(n_edges, 0);
    for (int i = 0; i < static_cast<int>(circuit.size()); ++i)
        edge_counts[circuit[i]]++;

    for (size_t c = 0; c < osr_cuts_.size(); ++c) {
        if (osr_cut_bounds_[c] <= 0) continue;
        int sum = 0;
        for (int eidx : osr_cut_crossing_edges_[c])
            sum += edge_counts[eidx];
        if (sum < osr_cut_bounds_[c])
            return false;
    }
    return true;
}


// ============================================================================
// Enumeration
// ============================================================================

std::vector<GrayCode> CircuitCanonicalizer::enumerate_circuits(int D) {
    std::vector<GrayCode> results;
    GrayCodeSet seen;

    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_tokens_;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    std::function<void(int)> dfs = [&](int depth) {
        if (depth == D) {
            GrayCode canon = canonical_form(path);
            if (seen.find(canon) == seen.end()) {
                if (has_custom_topology_ || check_osr_feasibility(canon)) {
                    seen.insert(canon.copy());
                    results.push_back(canon.copy());
                }
            }
            return;
        }

        for (int e = 0; e < n_tokens_; ++e) {
            path[depth] = e;
            path_masks[depth] = token_masks_[e];

            bool canonical = true;
            int i = depth;
            while (i > 0) {
                if (path_masks[i] & path_masks[i - 1]) break;
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
// Random valid sequence generation
// ============================================================================

GrayCode CircuitCanonicalizer::generate_valid_sequence(int D, std::mt19937& rng) {
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_tokens_;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    std::uniform_int_distribution<int> pick(0, n_tokens_ - 1);
    for (int depth = 0; depth < D; ++depth) {
        int chosen = pick(rng);
        path[depth] = chosen;
        path_masks[depth] = token_masks_[chosen];
    }

    return canonicalize_and_validate(path);
}

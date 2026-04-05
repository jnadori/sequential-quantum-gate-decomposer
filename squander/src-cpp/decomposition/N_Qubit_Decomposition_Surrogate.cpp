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
#include "GPRegressor.h"

#include <chrono>
#include <random>
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
    has_custom_topology = (topology_in.size() > 0);
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
    d_penalty = 0.0;
    enum_threshold = 10000;
    gp_max_train = 300;
    gp_score_ratio = 0.5;
    diversity_thresh = 0.95;
    d_seed_budget = 50;
    window_patience = 50;
    window_max_iters = 100;
    stagnation_window = 5;
    stagnation_improvement_frac = 0.25;
    config_D_start = -1;  // -1 = use OSR lower bound
    use_random_candidates = false;
    adaptive_kappa = true;
    kappa_decay_rate = 0.5;
    kappa_stagnation_boost = 0.5;
    position_guided_fraction = 0.5;
    position_lambda = 0.9;
    position_temperature = -1.0;

    // BOSS acquisition-guided GA
    use_boss_ga = false;
    boss_pop_size = 200;
    boss_generations = 10;
    boss_offspring_ratio = 2.0;
    acquisition_function_type = 0;  // 0=LCB, 1=EI

    // Kernel type
    kernel_type = 0;      // 0=SSK, 1=WL
    wl_iterations = 3;

    // Helper lambdas to extract config values safely
    auto get_dbl = [&](const std::string& key, double& var) {
        if (config.count(key)) config[key].get_property(var);
    };
    auto get_int = [&](const std::string& key, auto& var) {
        if (config.count(key)) {
            long long v; config[key].get_property(v);
            var = static_cast<std::remove_reference_t<decltype(var)>>(v);
        }
    };

    // Override from config
    get_dbl("kappa",kappa);
    get_dbl("tolerance",tolerance);
    get_int("max_sure_iters",max_iters);
    get_int("patience",patience);
    get_int("candidates_per_iter",candidates_per_iter);
    get_int("tournament_size",tournament_size);
    get_int("block_mutation_size",block_size);
    get_dbl("local_search_fraction",local_search_fraction);
    get_int("max_local_steps",max_local_steps);
    get_int("local_search_positions",local_search_positions);
    get_int("local_search_gp_subset",local_search_gp_subset);
    get_int("n_thompson_samples",n_thompson_samples);
    get_dbl("d_penalty",d_penalty);
    get_int("enum_threshold",enum_threshold);
    get_int("window_patience",window_patience);
    get_int("window_max_iters",window_max_iters);
    get_int("gp_max_train",gp_max_train);
    get_dbl("gp_score_ratio",gp_score_ratio);
    get_dbl("topk_diversity_threshold",diversity_thresh);
    get_int("d_seed_budget",d_seed_budget);
    get_int("stagnation_window",stagnation_window);
    get_dbl("stagnation_improvement_frac",stagnation_improvement_frac);
    get_dbl("ssk_gap_decay",sur_gap_decay);
    get_dbl("ssk_match_decay",sur_match_decay);
    get_int("ssk_order",sur_ssk_order);
    get_int("D_start",config_D_start);



    if (config.count("use_random_candidates") > 0) {
        long long v; config["use_random_candidates"].get_property(v);
        use_random_candidates = static_cast<bool>(v);
    }

    if (config.count("adaptive_kappa") > 0) {
        long long v; config["adaptive_kappa"].get_property(v);
        adaptive_kappa = static_cast<bool>(v);
    }
    get_dbl("kappa_decay_rate",kappa_decay_rate);
    get_dbl("kappa_stagnation_boost",kappa_stagnation_boost);
    get_dbl("position_guided_fraction",position_guided_fraction);
    get_dbl("position_lambda",position_lambda);
    get_dbl("position_temperature",position_temperature);
    if (config.count("use_boss_ga") > 0) {
        long long v; config["use_boss_ga"].get_property(v);
        use_boss_ga = static_cast<bool>(v);
    }
    get_int("boss_pop_size",boss_pop_size);
    get_int("boss_generations",boss_generations);
    get_int("acquisition_function",acquisition_function_type);
    get_dbl("boss_offspring_ratio",boss_offspring_ratio);
    get_int("kernel_type",kernel_type);
    get_int("wl_iterations",wl_iterations);
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

    // Compute OSR cut bounds — skip when custom topology is provided
    osr_D_min = 0;
    if (!has_custom_topology) {
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
    }

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

// ============================================================================
// Incremental canonical DAG methods
// ============================================================================

N_Qubit_Decomposition_Surrogate::CanonicalDAG
N_Qubit_Decomposition_Surrogate::build_canonical_dag(const GrayCode& seq) {
    int n = static_cast<int>(seq.size());
    CanonicalDAG dag;
    dag.n = n;
    dag.adj.assign(n, std::vector<int>());
    dag.in_degree.assign(n, 0);
    dag.masks.resize(n);
    for (int i = 0; i < n; ++i)
        dag.masks[i] = token_masks[seq[i]];

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

GrayCode N_Qubit_Decomposition_Surrogate::canonical_form_from_dag(
    const CanonicalDAG& dag, const GrayCode& seq) {

    int n = dag.n;
    if (n == 0) return GrayCode();

    // Deep copy in_degree since topological sort mutates it
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
            heap.push(HeapNode{token_sort_key(seq[i]), i});
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
        for (int neighbor : dag.adj[top.idx]) {
            in_deg[neighbor]--;
            if (in_deg[neighbor] == 0)
                heap.push(HeapNode{token_sort_key(seq[neighbor]), neighbor});
        }
    }

    return result;
}

void N_Qubit_Decomposition_Surrogate::update_dag_point_mutation(
    CanonicalDAG& dag, int pos, int old_token, int new_token) {

    int old_mask = dag.masks[pos];
    int new_mask = token_masks[new_token];

    if (old_mask == new_mask) return;  // no structural change

    dag.masks[pos] = new_mask;

    for (int q = 0; q < dag.n; ++q) {
        if (q == pos) continue;

        bool old_dep = (old_mask & dag.masks[q]) != 0;
        bool new_dep = (new_mask & dag.masks[q]) != 0;

        if (old_dep == new_dep) continue;

        if (q < pos) {
            // Edge direction: q -> pos
            if (old_dep && !new_dep) {
                // Remove edge q -> pos
                auto& adj_q = dag.adj[q];
                adj_q.erase(std::remove(adj_q.begin(), adj_q.end(), pos), adj_q.end());
                dag.in_degree[pos]--;
            } else {
                // Add edge q -> pos
                dag.adj[q].push_back(pos);
                dag.in_degree[pos]++;
            }
        } else {
            // Edge direction: pos -> q
            if (old_dep && !new_dep) {
                // Remove edge pos -> q
                auto& adj_pos = dag.adj[pos];
                adj_pos.erase(std::remove(adj_pos.begin(), adj_pos.end(), q), adj_pos.end());
                dag.in_degree[q]--;
            } else {
                // Add edge pos -> q
                dag.adj[pos].push_back(q);
                dag.in_degree[q]++;
            }
        }
    }
}

GrayCode N_Qubit_Decomposition_Surrogate::canonicalize_and_validate_from_dag(
    CanonicalDAG& dag, const GrayCode& seq, int pos, int new_token) {

    int old_token = seq[pos];

    // Apply mutation to DAG
    update_dag_point_mutation(dag, pos, old_token, new_token);

    // Build mutated sequence
    GrayCode mutated = seq.copy();
    mutated[pos] = new_token;

    // Compute canonical form from updated DAG
    GrayCode canon = canonical_form_from_dag(dag, mutated);

    // Restore DAG to original state
    update_dag_point_mutation(dag, pos, new_token, old_token);

    // Validate
    if (canon.size() == 0) return GrayCode();
    if (!has_custom_topology && !check_osr_feasibility(canon))
        return GrayCode();
    return canon;
}

GrayCode N_Qubit_Decomposition_Surrogate::canonicalize_and_validate(const GrayCode& seq) {
    GrayCode canon = canonical_form(seq);

    // Check OSR feasibility (skip with custom topology)
    if (!has_custom_topology && !check_osr_feasibility(canon))
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
                if (has_custom_topology || check_osr_feasibility(canon)) {
                    seen.insert(canon.copy());
                    results.push_back(canon.copy());
                }
            }
            return;
        }

        for (int e = 0; e < n_tokens; ++e) {
            path[depth] = e;
            path_masks[depth] = token_masks[e];

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


namespace {

static constexpr int MUTATION_MAX_ATTEMPTS = 50;

// retry_mutate: calls body(attempt) up to MUTATION_MAX_ATTEMPTS times,
// returning the first non-empty GrayCode result. F must return GrayCode(int).
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
// Evolutionary operators
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::generate_valid_sequence(int D) {
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i)
        limits[i] = n_tokens;

    GrayCode path(0, limits);
    std::vector<int> path_masks(D, 0);

    std::uniform_int_distribution<int> pick(0, n_tokens - 1);
    for (int depth = 0; depth < D; ++depth) {
        int chosen = pick(gen);
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

    // Build DAG once, reuse across attempts
    CanonicalDAG dag = build_canonical_dag(seq);

    return retry_mutate([&](int) -> GrayCode {
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

        return canonicalize_and_validate_from_dag(dag, seq, pos, new_token);
    });
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

    // Softmax temperature: use configured override or Quarl Eq. 5 formula
    double temperature;
    if (position_temperature > 0.0) {
        temperature = position_temperature;
    } else {
        double lam = position_lambda;
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

    // Build CDF and sample position
    std::vector<double> cdf(D);
    cdf[0] = weights[0];
    for (int pos = 1; pos < D; ++pos)
        cdf[pos] = cdf[pos - 1] + weights[pos];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    // Build DAG once, reuse across attempts
    CanonicalDAG dag = build_canonical_dag(seq);

    return retry_mutate([&](int) -> GrayCode {
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

        return canonicalize_and_validate_from_dag(dag, seq, pos, new_token);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_swap(const GrayCode& seq) {
    int D = static_cast<int>(seq.size());
    if (D < 2) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq.copy();
        int i = pos_dist(gen);
        int j = pos_dist(gen);
        while (j == i) j = pos_dist(gen);
        std::swap(new_seq[i], new_seq[j]);
        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block(const GrayCode& seq, int blk_size) {
    int D = static_cast<int>(seq.size());
    int bs = std::min(blk_size, D);
    std::uniform_int_distribution<int> start_dist(0, D - bs);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
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
        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_transplant(
        const GrayCode& recipient, const GrayCode& donor, int blk_size) {
    int D_r = static_cast<int>(recipient.size());
    int D_d = static_cast<int>(donor.size());
    int max_blk = std::min(blk_size, std::min(D_r, D_d));
    if (max_blk < 2) return GrayCode();
    std::uniform_int_distribution<int> blk_dist(2, max_blk);

    return retry_mutate([&](int) -> GrayCode {
        int blk = blk_dist(gen);
        std::uniform_int_distribution<int> r_start_dist(0, D_r - blk);
        std::uniform_int_distribution<int> d_start_dist(0, D_d - blk);
        int r_start = r_start_dist(gen);
        int d_start = d_start_dist(gen);

        GrayCode new_seq = recipient.copy();
        for (int i = 0; i < blk; ++i)
            new_seq[r_start + i] = donor[d_start + i];

        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block_regenerate(
        const GrayCode& seq, int blk_size) {
    int D = static_cast<int>(seq.size());
    int bs = std::min(blk_size, D);
    int max_bs = std::min(2 * bs, D);
    if (bs < 1) return GrayCode();
    std::uniform_int_distribution<int> bs_dist(bs, max_bs);

    return retry_mutate([&](int) -> GrayCode {
        int actual_bs = bs_dist(gen);
        std::uniform_int_distribution<int> start_dist(0, D - actual_bs);
        int start = start_dist(gen);

        GrayCode new_seq = seq.copy();
        std::uniform_int_distribution<int> pick(0, n_tokens - 1);
        for (int pos = start; pos < start + actual_bs; ++pos) {
            new_seq[pos] = pick(gen);
        }

        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::crossover_uniform(const GrayCode& seq1,
                                                             const GrayCode& seq2) {
    int D = static_cast<int>(seq1.size());
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
        GrayCode new_seq = seq1.copy();
        for (int i = 0; i < D; ++i)
            new_seq[i] = (coin(gen) < 0.5) ? seq1[i] : seq2[i];
        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_grow(const GrayCode& seq, int D_max) {
    int D = static_cast<int>(seq.size());
    if (D >= D_max) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D);
    std::uniform_int_distribution<int> token_dist(0, n_tokens - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    return retry_mutate([&](int) -> GrayCode {
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

        return canonicalize_and_validate(new_seq);
    });
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_shrink(const GrayCode& seq, int D_min) {
    int D = static_cast<int>(seq.size());
    if (D <= D_min) return GrayCode();
    std::uniform_int_distribution<int> pos_dist(0, D - 1);

    return retry_mutate([&](int) -> GrayCode {
        int pos = pos_dist(gen);
        GrayCode new_seq = seq.remove_Digit(pos);
        return canonicalize_and_validate(new_seq);
    });
}


// ============================================================================
// Local search on LCB acquisition
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::local_search_acq(
    const GrayCode& start, SSKCache& cache, GPRegressor& gp,
    double scale, GrayCodeSet& seen,
    int D_min_local, int D_max_local, int* steps_out) {

    // Thread-local RNG: seeded with thread_id XOR a monotonic counter to guarantee
    // distinct seeds even when threads initialize simultaneously (avoids time-based collision).
    thread_local std::mt19937 local_rng = []() {
        static std::atomic<uint32_t> ctr{0};
        uint32_t tid = static_cast<uint32_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        uint32_t count = ctr.fetch_add(1, std::memory_order_relaxed);
        return std::mt19937(tid ^ (count * 0x9e3779b9u));
    }();

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
    double std_cur = std::sqrt(std::max(scale - sum_v2, 1e-10));
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

        // Build DAG once per step for incremental canonicalization
        CanonicalDAG dag = build_canonical_dag(current);

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
                GrayCode cand = canonicalize_and_validate_from_dag(dag, current, pos, e);
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
        int n_t = gp.n_train;
        std::vector<double> mu(n_nb);
        std::vector<double> lcb_approx(n_nb);

        if (gp.love_valid && gp.love_rank > 0) {
            // LOVE-based variance: O(n * r * n_nb) — more accurate than diagonal
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
            // Diagonal approximation fallback: O(n_train * n_nb)
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
            double std_val = std::sqrt(std::max(scale - sv2, 1e-10));
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
// Normal distribution helpers and Expected Improvement
// ============================================================================

static double normal_cdf(double x) {
    // Rational approximation (Abramowitz & Stegun 26.2.17)
    // Maximum error: 1.5e-7
    static const double b1 =  0.319381530;
    static const double b2 = -0.356563782;
    static const double b3 =  1.781477937;
    static const double b4 = -1.821255978;
    static const double b5 =  1.330274429;
    static const double p  =  0.2316419;
    double sign = (x < 0) ? -1.0 : 1.0;
    x = std::fabs(x) / std::sqrt(2.0);
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((b5 * t + b4) * t) + b3) * t + b2) * t + b1) * t *
               std::exp(-x * x);
    return 0.5 * (1.0 + sign * y);
}

static double normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double N_Qubit_Decomposition_Surrogate::expected_improvement(
    double mu, double std_val, double y_best) const {
    // EI = (y_best - mu) * Phi(Z) + sigma * phi(Z),  Z = (y_best - mu) / sigma
    // Negated so lower = better (consistent with LCB convention)
    if (std_val < 1e-12) return 0.0;
    double z = (y_best - mu) / std_val;
    double ei = (y_best - mu) * normal_cdf(z) + std_val * normal_pdf(z);
    return -ei;  // negate: lower is better
}


// ============================================================================
// BOSS acquisition-guided GA
// ============================================================================

void N_Qubit_Decomposition_Surrogate::boss_acquisition_ga(
    const std::vector<GrayCode>& seed_circuits,
    const double* scores, int n_seeds,
    int pop_size, int n_generations,
    SSKCache& cache, GPRegressor& gp, double scale,
    const std::vector<int>& train_indices, int n_train,
    double y_best_norm,
    GrayCodeSet& seen,
    std::vector<GrayCode>& candidates_out,
    int D_min_gen, int D_max_gen) {

    bool mixed_d = (D_min_gen >= 0 && D_max_gen >= 0);
    int n_t = gp.n_train;

    // Helper: compute acquisition for a batch of circuits
    auto batch_acquisition = [&](const std::vector<GrayCode>& circs,
                                 std::vector<double>& acq_out,
                                 std::vector<double>& mu_out) {
        int nc = static_cast<int>(circs.size());
        if (nc == 0) return;
        acq_out.resize(nc);
        mu_out.resize(nc);

        // Cross-kernel
        std::vector<double> Ks(n_t * nc);
        cache.compute_cross_kernel_subset(circs, scale, train_indices, Ks.data());

        // mu = Ks^T @ alpha
        for (int j = 0; j < nc; ++j) {
            double mu_val = 0.0;
            for (int i = 0; i < n_t; ++i)
                mu_val += Ks[i * nc + j] * gp.alpha_data[i];
            mu_out[j] = mu_val;
        }

        // Variance + acquisition
        if (gp.love_valid && gp.love_rank > 0) {
            // LOVE fast variance: O(n * r * nc)
            for (int j = 0; j < nc; ++j) {
                double sum_rt_ks_sq = 0.0;
                for (int k = 0; k < gp.love_rank; ++k) {
                    double rt_ks_k = 0.0;
                    for (int i = 0; i < n_t; ++i)
                        rt_ks_k += gp.R_love[i * gp.love_rank + k] * Ks[i * nc + j];
                    sum_rt_ks_sq += rt_ks_k * rt_ks_k;
                }
                double std_val = std::sqrt(std::max(scale - sum_rt_ks_sq, 1e-10));
                if (acquisition_function_type == 1) {
                    acq_out[j] = expected_improvement(mu_out[j], std_val, y_best_norm);
                } else {
                    acq_out[j] = lcb(mu_out[j], std_val);
                }
            }
        } else {
            // Exact fallback: forward solve v = L^{-1} Ks
            std::vector<double> v(Ks);
            LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                           n_t, nc, gp.L_data.data(), n_t,
                           v.data(), nc);
            for (int j = 0; j < nc; ++j) {
                double sum_v2 = 0.0;
                for (int i = 0; i < n_t; ++i)
                    sum_v2 += v[i * nc + j] * v[i * nc + j];
                double std_val = std::sqrt(std::max(scale - sum_v2, 1e-10));
                if (acquisition_function_type == 1) {
                    acq_out[j] = expected_improvement(mu_out[j], std_val, y_best_norm);
                } else {
                    acq_out[j] = lcb(mu_out[j], std_val);
                }
            }
        }
    };

    // ---- Population initialization ----
    struct Individual {
        GrayCode circuit;
        double acq_value;
        double mu;
    };

    std::vector<Individual> population;
    population.reserve(pop_size);
    GrayCodeSet pop_seen;  // dedup within population

    // Phase A: seed from observed circuits (top 50% by score)
    int n_seed_target = pop_size / 2;

    // Sort seed indices by score (best first)
    std::vector<int> seed_order(n_seeds);
    std::iota(seed_order.begin(), seed_order.end(), 0);
    std::sort(seed_order.begin(), seed_order.end(),
              [&](int a, int b) { return scores[a] < scores[b]; });

    for (int si = 0; si < n_seeds && static_cast<int>(population.size()) < n_seed_target; ++si) {
        int idx = seed_order[si];
        const GrayCode& seed = seed_circuits[idx];

        // Mutate once for diversity
        GrayCode mutated = mutate_point(seed);
        GrayCode cand = (mutated.size() > 0) ? mutated : seed.copy();

        if (mixed_d && (static_cast<int>(cand.size()) < D_min_gen ||
                        static_cast<int>(cand.size()) > D_max_gen)) continue;
        if (pop_seen.find(cand) != pop_seen.end()) continue;
        if (seen.find(cand) != seen.end()) {
            // Use the seed directly if mutated version is already seen
            cand = seed.copy();
            if (mixed_d && (static_cast<int>(cand.size()) < D_min_gen ||
                            static_cast<int>(cand.size()) > D_max_gen)) continue;
            if (pop_seen.find(cand) != pop_seen.end()) continue;
        }

        pop_seen.insert(cand.copy());
        population.push_back({cand.copy(), 0.0, 0.0});
    }

    // Phase B: fill rest with random circuits
    int n_random = pop_size - static_cast<int>(population.size());
    if (mixed_d) {
        // Build D distribution weighted by 1/D
        int n_D = D_max_gen - D_min_gen + 1;
        std::vector<double> d_weights(n_D);
        double w_sum = 0;
        for (int i = 0; i < n_D; ++i) {
            d_weights[i] = 1.0 / (D_min_gen + i);
            w_sum += d_weights[i];
        }
        for (int i = 0; i < n_D; ++i) d_weights[i] /= w_sum;
        std::vector<double> d_cdf(n_D);
        d_cdf[0] = d_weights[0];
        for (int i = 1; i < n_D; ++i) d_cdf[i] = d_cdf[i - 1] + d_weights[i];

        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int attempt = 0;
             static_cast<int>(population.size()) < pop_size && attempt < n_random * 50;
             ++attempt) {
            double r = uni(gen);
            int D_val = D_max_gen;
            for (int i = 0; i < n_D; ++i) {
                if (r <= d_cdf[i]) { D_val = D_min_gen + i; break; }
            }
            GrayCode seq = generate_valid_sequence(D_val);
            if (seq.size() > 0 && pop_seen.find(seq) == pop_seen.end()) {
                pop_seen.insert(seq.copy());
                population.push_back({seq.copy(), 0.0, 0.0});
            }
        }
    } else {
        // Fixed D: use D from first seed or default
        int D = (n_seeds > 0) ? static_cast<int>(seed_circuits[0].size()) : 3;
        for (int attempt = 0;
             static_cast<int>(population.size()) < pop_size && attempt < n_random * 50;
             ++attempt) {
            GrayCode seq = generate_valid_sequence(D);
            if (seq.size() > 0 && pop_seen.find(seq) == pop_seen.end()) {
                pop_seen.insert(seq.copy());
                population.push_back({seq.copy(), 0.0, 0.0});
            }
        }
    }

    if (population.empty()) return;

    int actual_pop = static_cast<int>(population.size());

    // Batch-evaluate initial population
    {
        std::vector<GrayCode> pop_circuits;
        pop_circuits.reserve(actual_pop);
        for (auto& ind : population) pop_circuits.push_back(ind.circuit.copy());

        std::vector<double> acq_vals, mu_vals;
        batch_acquisition(pop_circuits, acq_vals, mu_vals);

        for (int i = 0; i < actual_pop; ++i) {
            population[i].acq_value = acq_vals[i];
            population[i].mu = mu_vals[i];
        }
    }

    // ---- GA loop ----
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    std::uniform_int_distribution<int> pop_dist(0, actual_pop - 1);
    int t_size = std::min(tournament_size, actual_pop);

    // Acquisition-guided tournament selection
    auto acq_tournament = [&]() -> int {
        int best = pop_dist(gen);
        for (int i = 1; i < t_size; ++i) {
            int idx = pop_dist(gen);
            if (population[idx].acq_value < population[best].acq_value)
                best = idx;
        }
        return best;
    };

    for (int ga_gen = 0; ga_gen < n_generations; ++ga_gen) {
        int n_offspring = static_cast<int>(actual_pop * boss_offspring_ratio);

        // Generate offspring via acquisition-guided selection + evolutionary operators
        GrayCodeSet offspring_seen;
        std::vector<GrayCode> offspring_circuits;
        offspring_circuits.reserve(n_offspring);

        int max_attempts = n_offspring * 10;
        for (int attempt = 0;
             static_cast<int>(offspring_circuits.size()) < n_offspring && attempt < max_attempts;
             ++attempt) {
            double r = op_dist(gen);
            GrayCode result;

            if (r < 0.25) {
                // Point mutation
                int p = acq_tournament();
                result = mutate_point(population[p].circuit);
            } else if (r < 0.40) {
                // Block mutation
                int p = acq_tournament();
                int adaptive_blk = std::max(block_size,
                    static_cast<int>(population[p].circuit.size()) / 4);
                result = mutate_block_regenerate(population[p].circuit, adaptive_blk);
            } else if (r < 0.60) {
                // Crossover or transplant
                int p1 = acq_tournament();
                int p2 = acq_tournament();
                if (population[p1].circuit.size() == population[p2].circuit.size()) {
                    result = crossover_uniform(population[p1].circuit, population[p2].circuit);
                } else {
                    int adaptive_blk = std::max(block_size,
                        std::min(static_cast<int>(population[p1].circuit.size()),
                                 static_cast<int>(population[p2].circuit.size())) / 4);
                    result = mutate_transplant(population[p1].circuit, population[p2].circuit,
                                               adaptive_blk);
                }
            } else if (r < 0.75) {
                // Transplant
                int p1 = acq_tournament();
                int p2 = acq_tournament();
                int adaptive_blk = std::max(block_size,
                    std::min(static_cast<int>(population[p1].circuit.size()),
                             static_cast<int>(population[p2].circuit.size())) / 4);
                result = mutate_transplant(population[p1].circuit, population[p2].circuit,
                                           adaptive_blk);
            } else if (r < 0.85 && mixed_d) {
                // Grow
                int p = acq_tournament();
                result = mutate_grow(population[p].circuit, D_max_gen);
            } else if (r < 0.95 && mixed_d) {
                // Shrink
                int p = acq_tournament();
                result = mutate_shrink(population[p].circuit, D_min_gen);
            } else {
                // Random injection
                if (mixed_d) {
                    std::geometric_distribution<int> geo(0.4);
                    int rand_D = D_min_gen + std::min(geo(gen), D_max_gen - D_min_gen);
                    result = generate_valid_sequence(rand_D);
                } else {
                    int D = (n_seeds > 0) ? static_cast<int>(seed_circuits[0].size()) : 3;
                    result = generate_valid_sequence(D);
                }
            }

            if (result.size() > 0 &&
                offspring_seen.find(result) == offspring_seen.end() &&
                (!mixed_d || (static_cast<int>(result.size()) >= D_min_gen &&
                              static_cast<int>(result.size()) <= D_max_gen))) {
                offspring_seen.insert(result.copy());
                offspring_circuits.push_back(result.copy());
            }
        }

        if (offspring_circuits.empty()) continue;

        // Batch-evaluate offspring acquisition
        int n_off = static_cast<int>(offspring_circuits.size());
        std::vector<double> off_acq, off_mu;
        batch_acquisition(offspring_circuits, off_acq, off_mu);

        // (mu + lambda) replacement: merge parents + offspring, keep top pop_size
        // Sort indices only — avoids sorting a container of GrayCodes directly
        int n_combined = actual_pop + n_off;
        std::vector<int> order(n_combined);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            double av = (a < actual_pop) ? population[a].acq_value : off_acq[a - actual_pop];
            double bv = (b < actual_pop) ? population[b].acq_value : off_acq[b - actual_pop];
            return av < bv;
        });

        // Build new population in sorted order, then swap
        int new_pop = std::min(pop_size, n_combined);
        std::vector<Individual> new_population;
        new_population.reserve(new_pop);
        GrayCodeSet new_pop_seen;
        for (int i = 0; i < new_pop; ++i) {
            int idx = order[i];
            if (idx < actual_pop) {
                new_pop_seen.insert(population[idx].circuit.copy());
                new_population.push_back({population[idx].circuit.copy(),
                                          population[idx].acq_value, population[idx].mu});
            } else {
                int oi = idx - actual_pop;
                new_pop_seen.insert(offspring_circuits[oi].copy());
                new_population.push_back({offspring_circuits[oi].copy(),
                                          off_acq[oi], off_mu[oi]});
            }
        }
        population = std::move(new_population);
        pop_seen = std::move(new_pop_seen);
        actual_pop = new_pop;

        // Update pop_dist range
        pop_dist = std::uniform_int_distribution<int>(0, actual_pop - 1);
    }

    // ---- Final output: top candidates with diversity filtering ----
    int n_pick = std::min(n_thompson_samples, actual_pop);

    for (int pick = 0; pick < n_pick; ++pick) {
        // Walk through population sorted by acquisition (already sorted)
        bool found = false;
        for (int sidx = 0; sidx < actual_pop; ++sidx) {
            if (seen.find(population[sidx].circuit) != seen.end()) continue;

            // On-demand SSK diversity check
            bool too_similar = false;
            if (!candidates_out.empty() && diversity_thresh < 1.0) {
                for (int prev = 0; prev < static_cast<int>(candidates_out.size()); ++prev) {
                    double k_val = cache.kernel_between(population[sidx].circuit, candidates_out[prev]);
                    if (k_val > diversity_thresh) { too_similar = true; break; }
                }
            }

            if (!too_similar) {
                candidates_out.push_back(population[sidx].circuit.copy());
                found = true;
                break;
            }
        }
        if (!found) break;
    }
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
    int D_min_gen, int D_max_gen, double force_random_fraction) {

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
        std::uniform_real_distribution<double> frac_dist(0.0, 1.0);
        // Determine D range for random parents (Fix 3)
        int D_rand_lo = (D_min_gen >= 0) ? D_min_gen : 1;
        int D_rand_hi = (D_max_gen >= 0) ? D_max_gen : (n_pop > 0 ? static_cast<int>(population[0].size()) : 1);
        std::uniform_int_distribution<int> D_rand_dist(D_rand_lo, std::max(D_rand_lo, D_rand_hi));
        for (int i = 0; i < n_local_target; ++i) {
            // Fix 3: inject random-region parents when stagnating
            if (force_random_fraction > 0.0 && frac_dist(gen) < force_random_fraction) {
                int D_val = D_rand_dist(gen);
                GrayCode rand_parent = generate_valid_sequence(D_val);
                local_parents[i] = (rand_parent.size() > 0) ? rand_parent : tournament_select().copy();
            } else {
                local_parents[i] = tournament_select().copy();
            }
        }

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
    // Use local batch_seen for dedup (not global seen) so mutations aren't
    // blocked by the ever-growing seen set. Global seen is checked before
    // decomposing in run_window_search.
    GrayCodeSet batch_seen;
    for (auto& c : candidates_out)
        batch_seen.insert(c.copy());

    int max_attempts = n_candidates * 10 + static_cast<int>(seen.size());
    for (int attempt = 0;
         static_cast<int>(candidates_out.size()) < n_candidates && attempt < max_attempts;
         ++attempt) {
        double r = op_dist(gen);
        GrayCode result;

        // Adaptive block size: scale with circuit depth (larger = more aggressive)
        int ref_D = mixed_d ? (D_min_gen + D_max_gen) / 2
                            : static_cast<int>(population[0].size());
        int adaptive_blk = std::max(block_size, ref_D / 3);

        if (mixed_d) {
            if (r < 0.10)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.15)
                result = mutate_swap(tournament_select());
            else if (r < 0.30)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk);
            else if (r < 0.50) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk);
            }
            else if (r < 0.60) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                if (p1.size() == p2.size())
                    result = crossover_uniform(p1, p2);
                else
                    result = maybe_guided_point(p1);
            }
            else if (r < 0.65)
                result = mutate_grow(tournament_select(), D_max_gen);
            else if (r < 0.80)
                result = mutate_shrink(tournament_select(), D_min_gen);
            else if (r < 0.85)
                result = mutate_block(tournament_select(), block_size);
            else {
                std::geometric_distribution<int> geo(0.4);
                int rand_D = D_min_gen + std::min(geo(gen), D_max_gen - D_min_gen);
                result = generate_valid_sequence(rand_D);
            }
        } else {
            if (r < 0.15)
                result = maybe_guided_point(tournament_select());
            else if (r < 0.20)
                result = mutate_swap(tournament_select());
            else if (r < 0.38)
                result = mutate_block_regenerate(tournament_select(), adaptive_blk);
            else if (r < 0.60) {
                const GrayCode& p1 = tournament_select();
                const GrayCode& p2 = tournament_select();
                result = mutate_transplant(p1, p2, adaptive_blk);
            }
            else if (r < 0.70)
                result = crossover_uniform(tournament_select(), tournament_select());
            else if (r < 0.75)
                result = mutate_block(tournament_select(), block_size);
            else {
                int D = static_cast<int>(population[0].size());
                result = generate_valid_sequence(D);
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

std::pair<double, Matrix_real> N_Qubit_Decomposition_Surrogate::decompose_with_initial_params(
    const GrayCode& circuit, const Matrix_real& initial_params) {

    Gates_block* gate_structure = construct_gate_structure(circuit);
    int param_num = gate_structure->get_parameter_num();

    N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp.set_custom_gate_structure(gate_structure);
    delete gate_structure;
    cDecomp.set_verbose(0);
    cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
    cDecomp.set_optimization_tolerance(tolerance);
    cDecomp.set_optimizer(alg);

    // Use imported parameters (clip to param_num if sizes differ)
    int n_use = std::min(static_cast<int>(initial_params.size()), param_num);
    Matrix_real start_params(1, param_num);
    for (int i = 0; i < n_use; ++i)
        start_params[i] = initial_params[i];
    // Fill remaining with random if imported params are shorter
    if (n_use < param_num) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);
        for (int i = n_use; i < param_num; ++i)
            start_params[i] = param_dist(gen);
    }
    cDecomp.set_optimized_parameters(start_params.get_data(), param_num);

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

    // Clamp D_min up to OSR lower bound (skip with custom topology)
    if (!has_custom_topology && D_min_search < osr_D_min)
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

        SSKCache ssk_cache(kernel_type, wl_iterations, token_masks,
                           sur_gap_decay, sur_match_decay, sur_ssk_order);
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
// Extract CNOT/CROT skeleton from imported gate structure
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::extract_skeleton_from_gates() {
    // Build edge lookup: (min_qbit, max_qbit) -> token index
    std::map<std::pair<int,int>, int> edge_to_token;
    int n_edges = static_cast<int>(topology.size());
    for (int i = 0; i < n_edges; ++i) {
        int a = topology[i][0], b = topology[i][1];
        edge_to_token[{std::min(a,b), std::max(a,b)}] = i;
    }

    // Recursively walk gate structure to find CNOT/CROT gates
    std::vector<int> tokens;
    std::function<void(Gate*)> walk = [&](Gate* g) {
        gate_type t = g->get_type();
        if (t == BLOCK_OPERATION) {
            Gates_block* blk = static_cast<Gates_block*>(g);
            std::vector<Gate*> sub = blk->get_gates();
            for (Gate* sg : sub) walk(sg);
        } else if (t == CNOT_OPERATION || t == CROT_OPERATION) {
            int tgt = g->get_target_qbit();
            int ctrl = g->get_control_qbit();
            auto key = std::make_pair(std::min(tgt, ctrl), std::max(tgt, ctrl));
            auto it = edge_to_token.find(key);
            if (it != edge_to_token.end()) {
                tokens.push_back(it->second);
            }
        }
    };

    for (Gate* g : gates) walk(g);

    int D = static_cast<int>(tokens.size());
    matrix_base<int> limits(1, D);
    for (int i = 0; i < D; ++i) limits[i] = n_tokens;
    GrayCode gcode(0, limits);
    for (int i = 0; i < D; ++i) gcode[i] = tokens[i];

    return gcode;
}


// ============================================================================
// Top-down compression search
// ============================================================================

void N_Qubit_Decomposition_Surrogate::compress_over_D_range(
    int D_start, int D_min, const std::string& log_file,
    const GrayCode& initial_skeleton,
    Gates_block* imported_gate_structure,
    const Matrix_real& imported_params) {

    auto search_start = std::chrono::high_resolution_clock::now();
    decompose_time = 0.0;
    decompose_count = 0;

    best_score = std::numeric_limits<double>::infinity();

    // Log header
    {
        std::ofstream flog(log_file);
        flog << "SurCompress (C++ top-down): N=" << qbit_num
             << ", D=" << D_start << " down to " << D_min
             << ", kappa=" << kappa << std::endl;
        flog << "---" << std::endl;
    }

    // Persistent state across all D values
    GrayCodeSet seen;
    std::vector<GrayCode> X;
    std::vector<double> y;
    std::vector<Matrix_real> all_params;

    // Seed with the initial circuit
    GrayCode initial = initial_skeleton.copy();
    int D_initial = static_cast<int>(initial.size());

    {
        std::ofstream flog(log_file, std::ios::app);
        flog << "\nInitial circuit: D=" << D_initial << std::endl;
    }

    // Evaluate initial circuit using the imported gate structure directly
    // (imported params match this structure exactly, so score should be near zero)
    {
        N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
        cDecomp.set_custom_gate_structure(imported_gate_structure);
        // set_custom_gate_structure clones the gates; free the original
        delete imported_gate_structure;
        cDecomp.set_verbose(0);
        cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
        cDecomp.set_optimization_tolerance(tolerance);
        cDecomp.set_optimizer(alg);

        int param_num = static_cast<int>(imported_params.size());
        cDecomp.set_optimized_parameters(imported_params.get_data(), param_num);
        cDecomp.start_decomposition();

        Matrix_real params = cDecomp.get_optimized_parameters();
        double score = cDecomp.optimization_problem(params);

        X.push_back(initial.copy());
        y.push_back(score);
        all_params.push_back(params);
        seen.insert(initial.copy());
        decompose_count++;

        if (score < best_score) {
            best_score = score;
            best_circuit = initial.copy();
            best_params = params;
        }

        std::ofstream flog(log_file, std::ios::app);
        flog << "Initial score: " << score << " (D=" << D_initial << ")" << std::endl;
    }


    for (int D = D_start; D >= D_min; --D) {

        // Skip D levels where we already have a solution at this depth or shorter
        if (best_score < tolerance && static_cast<int>(best_circuit.size()) <= D) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "\n=== D=" << D << " skipped (solution at D="
                 << best_circuit.size() << ", score=" << best_score << ") ===" << std::endl;
            continue;
        }

        // --- Phase 1: Systematic single-gate removal from best D+1 circuit ---
        {
            bool early_solution = false;
            int best_dplus1_idx = -1;
            double best_dplus1_score = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < X.size(); ++i) {
                if (static_cast<int>(X[i].size()) == D + 1 && y[i] < best_dplus1_score) {
                    best_dplus1_score = y[i];
                    best_dplus1_idx = static_cast<int>(i);
                }
            }

            if (best_dplus1_idx >= 0) {
                const GrayCode& best_dplus1 = X[best_dplus1_idx];
                int D_plus1 = static_cast<int>(best_dplus1.size());

                std::vector<GrayCode> removal_candidates;
                for (int pos = 0; pos < D_plus1; ++pos) {
                    GrayCode shortened = best_dplus1.remove_Digit(pos);
                    GrayCode result = canonicalize_and_validate(shortened);
                    if (result.size() > 0 && seen.find(result) == seen.end()) {
                        seen.insert(result.copy());
                        removal_candidates.push_back(std::move(result));
                    }
                }

                {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "\n=== D=" << D << " Phase 1: trying "
                         << removal_candidates.size() << " single-gate removals (from D="
                         << D_plus1 << " best, score=" << best_dplus1_score << ") ===" << std::endl;
                }

                if (!removal_candidates.empty()) {
                    std::vector<DecompResult> dec_results;
                    parallel_decompose_batch(removal_candidates, dec_results);

                    for (int si = 0; si < static_cast<int>(removal_candidates.size()); ++si) {
                        X.push_back(removal_candidates[si].copy());
                        y.push_back(dec_results[si].score);
                        all_params.push_back(dec_results[si].params);
                        decompose_time += dec_results[si].elapsed;
                        decompose_count++;

                        if (dec_results[si].score < best_score ||
                            (dec_results[si].score < tolerance && best_score < tolerance &&
                             removal_candidates[si].size() < best_circuit.size())) {
                            best_score = dec_results[si].score;
                            best_circuit = removal_candidates[si].copy();
                            best_params = dec_results[si].params;
                        }
                        if (dec_results[si].score < tolerance) {
                            early_solution = true;
                            break;
                        }
                    }
                }

                if (early_solution) {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "SOLUTION found by single-gate removal at D=" << D
                         << ", score=" << best_score << std::endl;
                    continue;  // try compressing further
                }
            }
        }

        // --- Phase 2: Shrink-seed D from D+1 circuits + random seeding ---
        bool early_solution = false;
        int n_shrunk = 0;
        int n_random = 0;

        // Gather D+1 circuits sorted by score (best first)
        std::vector<int> prev_indices;
        for (size_t i = 0; i < X.size(); ++i)
            if (static_cast<int>(X[i].size()) == D + 1)
                prev_indices.push_back(static_cast<int>(i));

        std::vector<GrayCode> seed_circuits;
        std::mt19937 gen(42 + D);

        // Shrink-seed from D+1 solutions
        if (!prev_indices.empty()) {
            std::sort(prev_indices.begin(), prev_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });

            int needed = X0_size;
            for (int pi : prev_indices) {
                if (n_shrunk >= needed) break;
                for (int attempt = 0; attempt < 10 && n_shrunk < needed; ++attempt) {
                    GrayCode shrunk = mutate_shrink(X[pi], D);
                    if (shrunk.size() > 0 && seen.find(shrunk) == seen.end()) {
                        seen.insert(shrunk.copy());
                        seed_circuits.push_back(shrunk.copy());
                        n_shrunk++;
                    }
                }
            }
        }

        // Random seeding: generate purely random circuits at this D
        // to cover structures not adjacent to D+1 solutions
        int random_budget = 3 * X0_size;
        for (int attempt = 0; n_random < random_budget && attempt < random_budget * 20; ++attempt) {
            GrayCode rand_circ = generate_valid_sequence(D);
            if (rand_circ.size() > 0 && seen.find(rand_circ) == seen.end()) {
                seen.insert(rand_circ.copy());
                seed_circuits.push_back(rand_circ.copy());
                n_random++;
            }
        }

        // Parallel decompose all seeds
        if (!seed_circuits.empty()) {
            std::vector<DecompResult> dec_results;
            parallel_decompose_batch(seed_circuits, dec_results);

            for (int si = 0; si < static_cast<int>(seed_circuits.size()); ++si) {
                X.push_back(seed_circuits[si].copy());
                y.push_back(dec_results[si].score);
                all_params.push_back(dec_results[si].params);
                decompose_time += dec_results[si].elapsed;
                decompose_count++;

                // Always prefer shorter circuit if both are below tolerance
                if (dec_results[si].score < best_score ||
                    (dec_results[si].score < tolerance && best_score < tolerance &&
                     seed_circuits[si].size() < best_circuit.size())) {
                    best_score = dec_results[si].score;
                    best_circuit = seed_circuits[si].copy();
                    best_params = dec_results[si].params;
                }
                if (dec_results[si].score < tolerance) {
                    early_solution = true;
                    break;
                }
            }
        }

        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "\n=== D=" << D << " (shrink-seeded: " << n_shrunk
                 << ", random-seeded: " << n_random << ") ===" << std::endl;
        }

        if (early_solution) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "SOLUTION found during seeding at D=" << D
                 << ", score=" << best_score << std::endl;
            continue;  // try compressing further
        }

        // Build fresh SSKCache for this D (same-D only)
        std::vector<int> cache_candidates;
        for (size_t i = 0; i < X.size(); ++i)
            if (static_cast<int>(X[i].size()) == D)
                cache_candidates.push_back(static_cast<int>(i));

        std::sort(cache_candidates.begin(), cache_candidates.end(),
                  [&y](int a, int b) { return y[a] < y[b]; });

        int cache_limit = std::min(static_cast<int>(cache_candidates.size()), 2 * gp_max_train);

        SSKCache ssk_cache(kernel_type, wl_iterations, token_masks,
                           sur_gap_decay, sur_match_decay, sur_ssk_order);
        for (int ci = 0; ci < cache_limit; ++ci)
            ssk_cache.register_circuit(X[cache_candidates[ci]]);

        // Run window search at this D
        WindowResult result = run_window_search(D, D,
            ssk_cache, seen, X, y, all_params, log_file);

        if (result == WINDOW_SUCCESS) {
            // Solution found at this D, keep compressing
            continue;
        }

        // Stagnation — retry once with fresh random seeding
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "Stagnation at D=" << D << ", retrying with fresh random seeds..." << std::endl;
        }

        int n_retry = 0;
        int retry_budget = 3 * X0_size;
        std::vector<GrayCode> retry_circuits;
        for (int attempt = 0; n_retry < retry_budget && attempt < retry_budget * 20; ++attempt) {
            GrayCode rand_circ = generate_valid_sequence(D);
            if (rand_circ.size() > 0 && seen.find(rand_circ) == seen.end()) {
                seen.insert(rand_circ.copy());
                retry_circuits.push_back(rand_circ.copy());
                n_retry++;
            }
        }

        if (!retry_circuits.empty()) {
            std::vector<DecompResult> dec_results;
            parallel_decompose_batch(retry_circuits, dec_results);

            bool retry_found = false;
            for (int si = 0; si < static_cast<int>(retry_circuits.size()); ++si) {
                X.push_back(retry_circuits[si].copy());
                y.push_back(dec_results[si].score);
                all_params.push_back(dec_results[si].params);
                decompose_time += dec_results[si].elapsed;
                decompose_count++;

                if (dec_results[si].score < best_score ||
                    (dec_results[si].score < tolerance && best_score < tolerance &&
                     retry_circuits[si].size() < best_circuit.size())) {
                    best_score = dec_results[si].score;
                    best_circuit = retry_circuits[si].copy();
                    best_params = dec_results[si].params;
                }
                if (dec_results[si].score < tolerance) {
                    retry_found = true;
                    break;
                }
            }

            if (retry_found) {
                std::ofstream flog(log_file, std::ios::app);
                flog << "SOLUTION found in retry seeding at D=" << D
                     << ", score=" << best_score << std::endl;
                continue;  // try compressing further
            }

            // Rebuild cache with updated data and retry window search
            std::vector<int> cache_candidates2;
            for (size_t i = 0; i < X.size(); ++i)
                if (static_cast<int>(X[i].size()) == D)
                    cache_candidates2.push_back(static_cast<int>(i));
            std::sort(cache_candidates2.begin(), cache_candidates2.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });
            int cache_limit2 = std::min(static_cast<int>(cache_candidates2.size()), 2 * gp_max_train);

            SSKCache ssk_cache2(kernel_type, wl_iterations, token_masks,
                               sur_gap_decay, sur_match_decay, sur_ssk_order);
            for (int ci = 0; ci < cache_limit2; ++ci)
                ssk_cache2.register_circuit(X[cache_candidates2[ci]]);

            WindowResult result2 = run_window_search(D, D,
                ssk_cache2, seen, X, y, all_params, log_file,
                window_patience / 3);

            if (result2 == WINDOW_SUCCESS) {
                std::ofstream flog(log_file, std::ios::app);
                flog << "SOLUTION found in retry window search at D=" << D << std::endl;
                continue;
            }
        }

        // Stagnation at this D — stop compression entirely
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "Stopping compression: stagnation at D=" << D << std::endl;
        }
        break;
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double total_search_time = std::chrono::duration<double>(search_end - search_start).count();

    {
        std::ofstream flog(log_file, std::ios::app);
        flog << "\nCompression complete: best=" << best_score
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
    const std::string& log_file,
    int patience_override) {

    int effective_patience = (patience_override > 0) ? patience_override : window_patience;

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

    // Fix 1: early termination when n_sel=0 streak
    int zero_sel_streak = 0;
    // Fix 2: reduce decompose budget when GP is unreliable
    double prev_rho = 0.0;
    int low_rho_streak = 0;
    // Fix 4: adaptive diversity threshold
    int consecutive_low_sel = 0;
    double effective_diversity_thresh = diversity_thresh;

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

    // GP training data: widen ±2 D to give GP score variation to learn from
    int d_filter_lo = std::max(1, win_lo - 2);
    int d_filter_hi = win_hi + 2;

    double saved_kappa = kappa;

    for (int itr = 0; itr < window_max_iters; ++itr) {

        // Adaptive kappa: decay over iterations, boost on stagnation
        if (adaptive_kappa && window_max_iters > 0 && acquisition_function_type != 1) {
            double progress = static_cast<double>(itr) / window_max_iters;
            double base = 1.0 - kappa_decay_rate * progress;
            double stagnation = 0.0;
            if (iters_since_improvement > effective_patience / 2) {
                stagnation = kappa_stagnation_boost *
                    std::min(1.0, static_cast<double>(iters_since_improvement - effective_patience / 2)
                                  / std::max(1, effective_patience / 2));
            }
            kappa = saved_kappa * (base + stagnation);
        }

        double t_gp = 0, t_cand = 0, t_acq = 0, t_dec = 0;
        int n_filtered_log = 0;

        std::vector<GrayCode> candidates;
        std::vector<int> selected_indices;
        std::vector<double> gp_mu_selected;  // GP predicted mu for rank correlation

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
                int d = static_cast<int>(X[i].size());
                if (d >= d_filter_lo && d <= d_filter_hi)
                    filtered_x_indices.push_back(i);
            }
            int n_filtered = static_cast<int>(filtered_x_indices.size());
            n_filtered_log = n_filtered;
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
                int n_best = static_cast<int>(gp_max_train * gp_score_ratio);
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
            // AND the first old_n training indices are unchanged. The prefix is
            // stabilised by the reordering block above before this check.
            int old_n_train = gp.n_train;

            // Stabilise prefix for incremental Cholesky: reorder train_indices /
            // train_y / log_y_norm so previously-fitted points come first (in their
            // stored order). The training set is rebuilt from scratch each iteration
            // via score-sort + pivoted Cholesky, so without this reordering the
            // prefix would change even when the set only grows.
            {
                std::unordered_map<int, int> new_pos;
                new_pos.reserve(n_x);
                for (int i = 0; i < n_x; ++i)
                    new_pos[train_indices[i]] = i;

                bool all_old_present = (old_n_train > 0 && old_n_train < n_x);
                for (int i = 0; i < old_n_train && all_old_present; ++i)
                    if (new_pos.find(gp.train_indices_[i]) == new_pos.end())
                        all_old_present = false;

                if (all_old_present) {
                    std::vector<int>    ri(n_x);
                    std::vector<double> ry(n_x), rl(n_x);

                    std::unordered_set<int> old_set(gp.train_indices_.begin(),
                                                    gp.train_indices_.begin() + old_n_train);
                    int pos = 0;
                    for (int i = 0; i < old_n_train; ++i) {
                        int src = new_pos[gp.train_indices_[i]];
                        ri[pos] = train_indices[src];
                        ry[pos] = train_y[src];
                        rl[pos] = log_y_norm[src];
                        ++pos;
                    }
                    for (int i = 0; i < n_x; ++i) {
                        if (!old_set.count(train_indices[i])) {
                            ri[pos] = train_indices[i];
                            ry[pos] = train_y[i];
                            rl[pos] = log_y_norm[i];
                            ++pos;
                        }
                    }
                    train_indices = std::move(ri);
                    train_y       = std::move(ry);
                    log_y_norm    = std::move(rl);
                }
            }

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
            if (opt_scale / std::max(gp.noise, 1e-12) < 0.1 || opt_scale < 1e-6) {
                gp.log_scale = 0.0;
                gp.noise = 1e-2;
                gp.fit(ssk_cache, train_indices.data(), n_x, log_y_norm.data());
                prev_log_scale = gp.log_scale;
                prev_noise = gp.noise;
            }

            double scale = gp.get_scale();

            auto t_phase1 = std::chrono::high_resolution_clock::now();
            t_gp = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            // --- Candidate generation + acquisition phase ---
            t_phase0 = std::chrono::high_resolution_clock::now();

            if (use_boss_ga) {
                // ============================================================
                // BOSS acquisition-guided GA
                // ============================================================

                // Build seed data from GP training pool
                std::vector<GrayCode> seed_circuits;
                std::vector<double> seed_scores;
                seed_circuits.reserve(n_x);
                seed_scores.reserve(n_x);
                for (int i = 0; i < n_x; ++i) {
                    int xi = filtered_x_indices[selected_fi[i]];
                    seed_circuits.push_back(X[xi].copy());
                    seed_scores.push_back(y[xi]);
                }

                // y_best in normalized space for EI
                double y_best_norm = *std::min_element(log_y_norm.begin(), log_y_norm.end());

                std::vector<int> train_vec(train_indices.begin(), train_indices.begin() + n_x);

                boss_acquisition_ga(
                    seed_circuits, seed_scores.data(), n_x,
                    boss_pop_size, boss_generations,
                    ssk_cache, gp, scale,
                    train_vec, n_x,
                    y_best_norm,
                    seen,
                    candidates,
                    win_lo, win_hi);

                // candidates are already diversity-filtered by boss_acquisition_ga
                // selected_indices = all candidates
                int n_boss = static_cast<int>(candidates.size());
                selected_indices.resize(n_boss);
                std::iota(selected_indices.begin(), selected_indices.end(), 0);

                // Save GP predicted mu for rank correlation (not available from BOSS GA directly)
                // We skip Spearman computation for BOSS mode

                t_phase1 = std::chrono::high_resolution_clock::now();
                t_cand = std::chrono::duration<double>(t_phase1 - t_phase0).count();
                t_acq = 0;  // acquisition is integrated into GA

                if (candidates.empty()) break;

            } else {
                // ============================================================
                // Original: generate candidates + LCB selection
                // ============================================================

            // Generate candidates restricted to this window's D range
            int n_local;
            double avg_steps;
            // Fix 3: inject random-region parents when GP is stagnating
            double force_random = (iters_since_improvement > effective_patience / 2) ? 0.5 : 0.0;
            generate_candidates(X, y.data(), static_cast<int>(X.size()),
                                candidates_per_iter,
                                ssk_cache, gp, scale, log_y_norm.data(),
                                seen, candidates, n_local, avg_steps,
                                win_lo, win_hi, force_random);

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_cand = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            if (candidates.empty()) break;

            int n_cand = static_cast<int>(candidates.size());

            // --- Acquisition phase ---
            t_phase0 = std::chrono::high_resolution_clock::now();

            int n_t = gp.n_train;

            // Compute cross-kernel Ks (n_t x n_cand)
            std::vector<int> train_indices_vec(train_indices.begin(),
                                               train_indices.begin() + n_x);
            std::vector<double> Ks(n_t * n_cand);
            ssk_cache.compute_cross_kernel_subset(candidates, scale,
                                                  train_indices_vec, Ks.data());

            // mu = Ks^T @ alpha
            std::vector<double> log_mu_norm(n_cand);
            for (int j = 0; j < n_cand; ++j) {
                double val = 0;
                for (int i = 0; i < n_t; ++i)
                    val += Ks[i * n_cand + j] * gp.alpha_data[i];
                log_mu_norm[j] = val;
            }

            // v = L^{-1} Ks  (forward solve for all candidates)
            std::vector<double> v(n_t * n_cand);
            std::memcpy(v.data(), Ks.data(), n_t * n_cand * sizeof(double));
            LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                           n_t, n_cand, gp.L_data.data(), n_t,
                           v.data(), n_cand);

            // Free Ks — no longer needed
            { std::vector<double>().swap(Ks); }

            // --- Batch LCB with on-demand diversity ---
            // Replaces Thompson sampling; eliminates O(n_cand^2) SSK kernel + O(n_cand^3) Cholesky

            // 1. Diagonal posterior variance from v: var_i = scale - sum_k v[k,i]^2
            std::vector<double> var_diag(n_cand);
            for (int i = 0; i < n_cand; ++i) {
                double sum_v2 = 0.0;
                for (int k = 0; k < n_t; ++k)
                    sum_v2 += v[k * n_cand + i] * v[k * n_cand + i];
                var_diag[i] = std::max(scale - sum_v2, 1e-10);
            }

            // Free v — no longer needed
            { std::vector<double>().swap(v); }

            // 2. D-penalty on mu (normalize by win_hi, not global D_max)
            std::vector<double> mu_cand(log_mu_norm);
            if (d_penalty > 0 && win_hi > 0) {
                for (int i = 0; i < n_cand; ++i) {
                    double D_val = static_cast<double>(candidates[i].size());
                    mu_cand[i] += d_penalty * (D_val / win_hi);
                }
            }

            // 3. Compute LCB for each candidate
            std::vector<double> lcb_val(n_cand);
            for (int i = 0; i < n_cand; ++i)
                lcb_val[i] = mu_cand[i] - kappa * std::sqrt(var_diag[i]);

            // 4. Sort candidates by LCB ascending (best first)
            std::vector<int> lcb_order(n_cand);
            std::iota(lcb_order.begin(), lcb_order.end(), 0);
            std::sort(lcb_order.begin(), lcb_order.end(),
                      [&](int a, int b) { return lcb_val[a] < lcb_val[b]; });

            // 5. Greedy LCB selection with on-demand SSK diversity
            //    Only computes O(n_pick^2) SSK evaluations instead of O(n_cand^2)
            // Fix 2: reduce decompose budget when GP is unreliable
            int effective_n_pick = n_thompson_samples;
            if (prev_rho < 0.15 && low_rho_streak >= 2 && iters_since_improvement > 5)
                effective_n_pick = std::max(10, n_thompson_samples / 4);
            int n_pick = std::min(effective_n_pick, n_cand);
            std::set<int> selected_set;
            std::vector<GrayCode> selected_circuits;

            for (int pick = 0; pick < n_pick; ++pick) {
                bool found = false;
                for (int sidx : lcb_order) {
                    if (selected_set.count(sidx)) continue;
                    if (seen.find(candidates[sidx]) != seen.end()) continue;

                    // On-demand diversity check against already-selected candidates
                    bool too_similar = false;
                    if (!selected_circuits.empty() && effective_diversity_thresh < 1.0) {
                        for (int prev_idx = 0; prev_idx < static_cast<int>(selected_circuits.size()); ++prev_idx) {
                            double k_val = ssk_cache.kernel_between(candidates[sidx], selected_circuits[prev_idx]);
                            if (k_val > effective_diversity_thresh) {
                                too_similar = true;
                                break;
                            }
                        }
                    }

                    if (!too_similar) {
                        selected_indices.push_back(sidx);
                        selected_set.insert(sidx);
                        selected_circuits.push_back(candidates[sidx].copy());
                        found = true;
                        break;
                    }
                }
                if (!found) break;
            }

            // Save GP predicted mu for selected candidates (for rank correlation)
            for (int cidx : selected_indices)
                gp_mu_selected.push_back(log_mu_norm[cidx]);

            t_phase1 = std::chrono::high_resolution_clock::now();
            t_acq = std::chrono::duration<double>(t_phase1 - t_phase0).count();

            } // end else (original candidate gen)
            } // end if use_boss_ga / original

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
            ssk_cache.register_circuit(candidates[sel_idx]);

            // Update window-local best
            int cand_size = static_cast<int>(candidates[sel_idx].size());
            if (cand_size >= win_lo && cand_size <= win_hi && new_score < window_best_score) {
                window_best_score = new_score;
                window_best_D = cand_size;
            }

            if (new_score < best_score ||
                (new_score < tolerance && best_score < tolerance &&
                 static_cast<int>(candidates[sel_idx].size()) < static_cast<int>(best_circuit.size()))) {
                best_score = new_score;
                best_circuit = candidates[sel_idx].copy();
                best_params = new_params;
                improved_this_iter = true;
            }

            if (new_score < tolerance &&
                static_cast<int>(candidates[sel_idx].size()) >= win_lo &&
                static_cast<int>(candidates[sel_idx].size()) <= win_hi) {
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

        // Compute Spearman rank correlation between GP predicted mu and actual scores
        double spearman_rho = 0.0;
        if (!use_random_candidates && n_sel >= 5 && static_cast<int>(gp_mu_selected.size()) == n_sel) {
            // Collect actual scores for this batch
            std::vector<double> actual_scores(n_sel);
            for (int si = 0; si < n_sel; ++si)
                actual_scores[si] = dec_results[si].score;

            // Compute ranks for predicted and actual
            auto rank_of = [](const std::vector<double>& vals) {
                int n = static_cast<int>(vals.size());
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](int a, int b) { return vals[a] < vals[b]; });
                std::vector<double> ranks(n);
                for (int i = 0; i < n; ++i)
                    ranks[order[i]] = static_cast<double>(i);
                return ranks;
            };

            auto pred_ranks = rank_of(gp_mu_selected);
            auto actual_ranks = rank_of(actual_scores);

            // Spearman rho = 1 - 6*sum(d^2) / (n*(n^2-1))
            double sum_d2 = 0;
            for (int i = 0; i < n_sel; ++i) {
                double d = pred_ranks[i] - actual_ranks[i];
                sum_d2 += d * d;
            }
            spearman_rho = 1.0 - 6.0 * sum_d2 / (static_cast<double>(n_sel) * (static_cast<double>(n_sel) * n_sel - 1.0));
        }

        // Fix 2: update rho tracking for next iter's budget decision
        if (spearman_rho < 0.15) low_rho_streak++;
        else low_rho_streak = 0;
        prev_rho = spearman_rho;

        // Log — show window-local best score
        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  Iter " << itr
                 << ": best=" << window_best_score
                 << " (D=" << window_best_D << ")"
                 << " evals=" << X.size()
                 << (use_random_candidates ? " [RANDOM]" : "")
                 << " kappa=" << std::setprecision(3) << kappa
                 << " rho=" << std::setprecision(3) << spearman_rho
                 << " n_train=" << gp.n_train
                 << " n_filtered=" << n_filtered_log
                 << " n_sel=" << static_cast<int>(selected_indices.size())
                 << " [gp=" << std::setprecision(2) << t_gp
                 << "s cand=" << t_cand
                 << "s acq=" << t_acq
                 << "s dec=" << t_dec << "s]"
                 << std::defaultfloat
                 << std::endl;
        }

        // Fix 1: early termination when n_sel=0 for 3 consecutive iters
        if (n_sel == 0) {
            if (++zero_sel_streak >= 3) {
                std::ofstream flog(log_file, std::ios::app);
                flog << "  Early termination: n_sel=0 for 3 consecutive iters" << std::endl;
                kappa = saved_kappa;
                return WINDOW_STAGNATION;
            }
        } else {
            zero_sel_streak = 0;
        }

        // Fix 4: update adaptive diversity threshold based on n_sel trend
        if (n_sel < n_thompson_samples / 2)
            consecutive_low_sel++;
        else
            consecutive_low_sel = 0;
        if (consecutive_low_sel > 5)
            effective_diversity_thresh = std::min(1.0, diversity_thresh * 2.0);
        else
            effective_diversity_thresh = diversity_thresh;

        // Best-so-far plateau stagnation check
        int n_iters_done = static_cast<int>(iter_bests.size());
        if (n_iters_done >= stagnation_window) {
            double old_best = iter_bests[n_iters_done - stagnation_window];
            double cur_best = iter_bests.back();
            // Stagnation: relative improvement over last window < required fraction
            double rel_improvement = (old_best > 1e-15)
                ? (old_best - cur_best) / old_best
                : 0.0;
            if (rel_improvement < stagnation_improvement_frac) {
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
        if (iters_since_improvement >= effective_patience) {
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

    // Temporarily turn off OpenMP parallelism
    force_single_thread_blas_lapack();

    if (gates.size() > 0) {
        // Compression mode: imported gate structure detected
        std::stringstream sstream;
        sstream << "Starting " << (use_random_candidates ? "RANDOM baseline" : "surrogate")
                << " COMPRESSION for " << qbit_num << "-qubit matrix" << std::endl;
        print(sstream, 1);

        GrayCode skeleton = extract_skeleton_from_gates();
        int D_initial = static_cast<int>(skeleton.size());

        // Clone the imported gate structure for compress_over_D_range
        // (compress_over_D_range takes ownership and will delete it)
        Gates_block* gate_structure_for_compress = new Gates_block(qbit_num);
        for (Gate* g : gates) {
            gate_structure_for_compress->add_gate(g->clone());
        }

        // Optimize imported circuit directly (before releasing gates)
        // to get good initial parameters that match the gate structure
        Matrix_real initial_params;

        {
            Gates_block* imported_structure = new Gates_block(qbit_num);
            for (Gate* g : gates) {
                imported_structure->add_gate(g->clone());
            }

            N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
            cDecomp.set_custom_gate_structure(imported_structure);
            delete imported_structure;
            cDecomp.set_verbose(0);
            cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
            cDecomp.set_optimization_tolerance(tolerance);
            cDecomp.set_optimizer(alg);

            int param_num = get_parameter_num();
            cDecomp.set_optimized_parameters(optimized_parameters_mtx.get_data(), param_num);
            cDecomp.start_decomposition();

            initial_params = cDecomp.get_optimized_parameters();
        }

        // Release imported gates — we only need the skeleton + cloned structure
        release_gates();

        int D_min = (config_D_start >= 0) ? config_D_start : 1;

        std::string log_prefix = project_name.empty() ? "surcompress" : "surcompress_" + project_name;
        std::string log_file = log_prefix + "_D" + std::to_string(D_initial) + "-" +
                                std::to_string(D_min) + ".txt";

        compress_over_D_range(D_initial, D_min, log_file, skeleton,
                              gate_structure_for_compress, initial_params);
    } else {
        // Bottom-up search mode
        std::stringstream sstream;
        sstream << "Starting " << (use_random_candidates ? "RANDOM baseline" : "surrogate")
                << " search for " << qbit_num << "-qubit matrix" << std::endl;
        print(sstream, 1);

        int D_start = (config_D_start >= 0) ? config_D_start : osr_D_min;
        int D_end = level_limit;
        if (D_end <= 0) D_end = D_start + 20;  // default range

        std::string log_prefix = project_name.empty() ? "sursearch" : "sursearch_" + project_name;
        std::string log_file = log_prefix + "_D" + std::to_string(D_start) + "-" +
                                std::to_string(D_end) + ".txt";

        search_over_D_range(D_start, D_end, log_file);
    }

    // Construct the final gate structure from best circuit
    if (best_circuit.size() > 0) {
        Gates_block* gate_structure = construct_gate_structure(best_circuit);
        int expected_param_num = gate_structure->get_parameter_num();

        // best_params may have been set from the imported gate structure (compression mode),
        // which has a different U3-gate count than construct_gate_structure.
        // Re-decompose to get compatible params in that case.
        if (static_cast<int>(best_params.size()) != expected_param_num) {
            delete gate_structure;
            std::pair<double, Matrix_real> result = decompose(best_circuit);
            best_score = result.first;
            best_params = result.second;
            decomposition_error = best_score;
            gate_structure = construct_gate_structure(best_circuit);
        }

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

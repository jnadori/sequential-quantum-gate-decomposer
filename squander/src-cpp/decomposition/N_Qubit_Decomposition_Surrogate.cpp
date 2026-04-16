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

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <chrono>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

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
    local_search_max_neighbors = 50;
    local_search_patience = 3;
    local_search_min_improvement = 1e-4;
    eval_budget_base = 10;
    d_penalty = 0.0;
    enum_threshold = 10000;
    gp_max_train = 300;
    gp_score_ratio = 0.5;
    d_seed_budget = 50;
    window_patience = 50;
    window_max_iters = 100;
    stagnation_window = 5;
    stagnation_improvement_frac = 0.25;
    config_D_start = -1;  // -1 = use OSR lower bound
    adaptive_kappa = true;
    kappa_decay_rate = 0.5;
    kappa_stagnation_boost = 0.5;
    position_guided_fraction = 0.5;
    position_lambda = 0.9;
    position_temperature = -1.0;

    // Weisfeiler-Lehman kernel refinement rounds
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
    get_int("local_search_max_neighbors",local_search_max_neighbors);
    get_int("local_search_patience",local_search_patience);
    get_dbl("local_search_min_improvement",local_search_min_improvement);
    get_int("eval_budget_base",eval_budget_base);
    get_int("n_thompson_samples",eval_budget_base);  // legacy alias
    get_dbl("d_penalty",d_penalty);
    get_int("enum_threshold",enum_threshold);
    get_int("window_patience",window_patience);
    get_int("window_max_iters",window_max_iters);
    get_int("gp_max_train",gp_max_train);
    get_dbl("gp_score_ratio",gp_score_ratio);
    get_int("d_seed_budget",d_seed_budget);
    get_int("stagnation_window",stagnation_window);
    get_dbl("stagnation_improvement_frac",stagnation_improvement_frac);
    // P3: clamp stagnation_window so the plateau detector can fire before patience cap
    if (stagnation_window >= window_patience)
        stagnation_window = std::max(3, window_patience / 3);
    get_int("D_start",config_D_start);

    if (config.count("adaptive_kappa") > 0) {
        long long v; config["adaptive_kappa"].get_property(v);
        adaptive_kappa = static_cast<bool>(v);
    }
    get_dbl("kappa_decay_rate",kappa_decay_rate);
    get_dbl("kappa_stagnation_boost",kappa_stagnation_boost);
    get_dbl("position_guided_fraction",position_guided_fraction);
    get_dbl("position_lambda",position_lambda);
    get_dbl("position_temperature",position_temperature);
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

    // Initialize CircuitCanonicalizer
    canon_ = std::make_unique<CircuitCanonicalizer>(
        token_masks, token_neighbors, thresholds, n_tokens,
        has_custom_topology, topology,
        osr_cuts, osr_cut_bounds, osr_cut_crossing_edges,
        [this](int token) { return this->token_sort_key(token); });

    // Initialize CircuitMutators
    MutatorConfig mcfg;
    mcfg.block_size = block_size;
    mcfg.tournament_size = tournament_size;
    mcfg.local_search_fraction = local_search_fraction;
    mcfg.max_local_steps = max_local_steps;
    mcfg.local_search_positions = local_search_positions;
    mcfg.local_search_gp_subset = local_search_gp_subset;
    mcfg.local_search_max_neighbors = local_search_max_neighbors;
    mcfg.local_search_patience = local_search_patience;
    mcfg.local_search_min_improvement = local_search_min_improvement;
    mcfg.position_guided_fraction = position_guided_fraction;
    mcfg.position_lambda = position_lambda;
    mcfg.position_temperature = position_temperature;
    mutators_ = std::make_unique<CircuitMutators>(canon_.get(), mcfg);

    // Initialize SurrogateModel
    surrogate_ = std::make_unique<SurrogateModel>(
        gp_max_train, gp_score_ratio, wl_iterations, token_masks);

    // Initialize GateRemover
    gate_remover_ = std::make_unique<GateRemover>(canon_.get(), qbit_num);

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
// Validation and canonicalization — delegated to CircuitCanonicalizer
// ============================================================================

bool N_Qubit_Decomposition_Surrogate::check_new_position(const int* window_masks, int pos) {
    return canon_->check_new_position(window_masks, pos);
}

GrayCode N_Qubit_Decomposition_Surrogate::canonical_form(const GrayCode& seq) {
    return canon_->canonical_form(seq);
}

N_Qubit_Decomposition_Surrogate::CanonicalDAG
N_Qubit_Decomposition_Surrogate::build_canonical_dag(const GrayCode& seq) {
    return canon_->build_canonical_dag(seq);
}

GrayCode N_Qubit_Decomposition_Surrogate::canonical_form_from_dag(
    const CanonicalDAG& dag, const GrayCode& seq) {
    return canon_->canonical_form_from_dag(dag, seq);
}

void N_Qubit_Decomposition_Surrogate::update_dag_point_mutation(
    CanonicalDAG& dag, int pos, int old_token, int new_token) {
    canon_->update_dag_point_mutation(dag, pos, old_token, new_token);
}

GrayCode N_Qubit_Decomposition_Surrogate::canonicalize_and_validate_from_dag(
    CanonicalDAG& dag, const GrayCode& seq, int pos, int new_token) {
    return canon_->canonicalize_and_validate_from_dag(dag, seq, pos, new_token);
}

GrayCode N_Qubit_Decomposition_Surrogate::canonicalize_and_validate(const GrayCode& seq) {
    return canon_->canonicalize_and_validate(seq);
}

bool N_Qubit_Decomposition_Surrogate::check_osr_feasibility(const GrayCode& circuit) {
    return canon_->check_osr_feasibility(circuit);
}

std::vector<GrayCode> N_Qubit_Decomposition_Surrogate::enumerate_circuits(int D) {
    return canon_->enumerate_circuits(D);
}


// ============================================================================
// Evolutionary operators — delegated to CircuitMutators
// ============================================================================

GrayCode N_Qubit_Decomposition_Surrogate::generate_valid_sequence(int D) {
    return canon_->generate_valid_sequence(D, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_point(const GrayCode& seq) {
    return mutators_->mutate_point(seq, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_point_guided(
    const GrayCode& seq, WLKernelCache& cache, GPRegressor& gp, double scale) {
    return mutators_->mutate_point_guided(seq, cache, gp, scale, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_swap(const GrayCode& seq) {
    return mutators_->mutate_swap(seq, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block(const GrayCode& seq, int blk_size) {
    return mutators_->mutate_block(seq, blk_size, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_transplant(
    const GrayCode& recipient, const GrayCode& donor, int blk_size) {
    return mutators_->mutate_transplant(recipient, donor, blk_size, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_block_regenerate(
    const GrayCode& seq, int blk_size) {
    return mutators_->mutate_block_regenerate(seq, blk_size, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::crossover_uniform(
    const GrayCode& seq1, const GrayCode& seq2) {
    return mutators_->crossover_uniform(seq1, seq2, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_grow(const GrayCode& seq, int D_max) {
    return mutators_->mutate_grow(seq, D_max, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_shrink(const GrayCode& seq, int D_min) {
    return mutators_->mutate_shrink(seq, D_min, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_commute(const GrayCode& seq) {
    return mutators_->mutate_commute(seq, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_cancel_pairs(const GrayCode& seq) {
    return mutators_->mutate_cancel_pairs(seq, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::mutate_insert_identity(const GrayCode& seq) {
    return mutators_->mutate_insert_identity(seq, gen);
}

GrayCode N_Qubit_Decomposition_Surrogate::local_search_acq(
    const GrayCode& start, WLKernelCache& cache, GPRegressor& gp,
    double scale, int D_min_local, int D_max_local, int* steps_out) {
    return mutators_->local_search_acq(start, cache, gp, scale, kappa,
                                        D_min_local, D_max_local, steps_out);
}

void N_Qubit_Decomposition_Surrogate::generate_candidates(
    const std::vector<GrayCode>& population, const double* scores, int n_pop,
    int n_candidates, WLKernelCache& cache, GPRegressor& gp, double scale,
    const double* train_y_norm,
    GrayCodeSet& seen, std::vector<GrayCode>& candidates_out,
    int& n_local_out, double& avg_steps_out,
    int D_min_gen, int D_max_gen, double force_random_fraction) {
    mutators_->generate_candidates(population, scores, n_pop, n_candidates,
                                   cache, gp, scale, train_y_norm,
                                   seen, candidates_out,
                                   n_local_out, avg_steps_out,
                                   gen, D_min_gen, D_max_gen,
                                   force_random_fraction);
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

    constexpr int n_retries = 5;
    double best_score = std::numeric_limits<double>::max();
    Matrix_real best_params;

    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);

    for (int retry = 0; retry < n_retries; ++retry) {
        Gates_block* gate_structure = construct_gate_structure(circuit);
        int param_num = gate_structure->get_parameter_num();

        N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
        cDecomp.set_custom_gate_structure(gate_structure);
        delete gate_structure;
        cDecomp.set_verbose(0);
        cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
        cDecomp.set_optimization_tolerance(tolerance);
        cDecomp.set_optimizer(alg);

        Matrix_real random_params(1, param_num);
        for (int i = 0; i < param_num; ++i)
            random_params[i] = param_dist(local_gen);
        cDecomp.set_optimized_parameters(random_params.get_data(), param_num);

        cDecomp.start_decomposition();

        Matrix_real params = cDecomp.get_optimized_parameters();
        double score = cDecomp.optimization_problem(params);

        if (score < best_score) {
            best_score = score;
            best_params = std::move(params);
        }

        if (best_score < tolerance)
            break;
    }

    return {best_score, std::move(best_params)};
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

    tbb::parallel_for(
        tbb::blocked_range<int>(0, n, 1),
        [&](tbb::blocked_range<int> r) {
            thread_local std::mt19937 local_gen(std::random_device{}());
            for (int i = r.begin(); i < r.end(); ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto res = decompose_with_rng(circuits[i], local_gen);
                auto t1 = std::chrono::high_resolution_clock::now();
                results[i].score = res.first;
                results[i].params = res.second;
                results[i].elapsed = std::chrono::duration<double>(t1 - t0).count();
            }
        }
    );
}


Matrix_real N_Qubit_Decomposition_Surrogate::excise_gate_params(
    const Matrix_real& parent_params, int D_plus1, int pos) {

    // Parent: D_plus1 * 6 gate params + 3 * qbit_num finalizing params
    // Child (D = D_plus1 - 1): (D_plus1 - 1) * 6 gate params + 3 * qbit_num finalizing params
    // Remove the 6 params at [pos*6, pos*6+5] from the gate params portion
    int gate_params = (D_plus1 - 1) * 6;
    int final_params = 3 * qbit_num;
    int child_param_num = gate_params + final_params;
    int parent_gate_params = D_plus1 * 6;

    Matrix_real child_params(1, child_param_num);

    // Copy gate params before the removed position
    int dst = 0;
    for (int i = 0; i < pos * 6; ++i)
        child_params[dst++] = parent_params[i];
    // Copy gate params after the removed position
    for (int i = (pos + 1) * 6; i < parent_gate_params; ++i)
        child_params[dst++] = parent_params[i];
    // Copy finalizing layer params
    for (int i = 0; i < final_params; ++i)
        child_params[dst++] = parent_params[parent_gate_params + i];

    return child_params;
}


void N_Qubit_Decomposition_Surrogate::parallel_decompose_batch_with_init(
    const std::vector<GrayCode>& circuits,
    const std::vector<Matrix_real>& init_params,
    std::vector<DecompResult>& results) {

    int n = static_cast<int>(circuits.size());
    results.resize(n);
    if (n == 0) return;

    tbb::parallel_for(
        tbb::blocked_range<int>(0, n, 1),
        [&](tbb::blocked_range<int> r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto res = decompose_with_initial_params(circuits[i], init_params[i]);
                auto t1 = std::chrono::high_resolution_clock::now();
                results[i].score = res.first;
                results[i].params = res.second;
                results[i].elapsed = std::chrono::duration<double>(t1 - t0).count();
            }
        }
    );
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

    // Persistent state across all D values (but WLKernelCache is rebuilt per D)
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

        // --- Build fresh WLKernelCache for this D ---
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

        WLKernelCache wl_cache(wl_iterations, token_masks);
        for (int ci = 0; ci < cache_limit; ++ci)
            wl_cache.register_circuit(X[cache_candidates[ci]]);

        // --- Run search at this D ---
        WindowResult result = run_window_search(D, D,
            wl_cache, seen, X, y, all_params, log_file);

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

        // --- Phase 1: Single-gate removal via GateRemover ---
        {
            GateRemover::Phase1Data phase1_data;
            auto removal = gate_remover_->try_single_removals(
                X, y, all_params, seen, D, phase1_data);

            {
                std::ofstream flog(log_file, std::ios::app);
                flog << "\n=== D=" << D << " Phase 1: trying "
                     << removal.candidates.size() << " single-gate removals ===" << std::endl;
            }

            bool early_solution = false;

            if (!removal.candidates.empty()) {
                std::vector<DecompResult> dec_results;
                parallel_decompose_batch_with_init(removal.candidates, removal.warmstart_params, dec_results);

                for (int si = 0; si < static_cast<int>(removal.candidates.size()); ++si) {
                    X.push_back(removal.candidates[si].copy());
                    y.push_back(dec_results[si].score);
                    all_params.push_back(dec_results[si].params);
                    decompose_time += dec_results[si].elapsed;
                    decompose_count++;

                    if (dec_results[si].score < best_score ||
                        (dec_results[si].score < tolerance && best_score < tolerance &&
                         removal.candidates[si].size() < best_circuit.size())) {
                        best_score = dec_results[si].score;
                        best_circuit = removal.candidates[si].copy();
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
                continue;
            }

            // --- Phase 1.5: Multi-gate pair removal ---
            if (!phase1_data.entries.empty()) {
                auto pair_removal = gate_remover_->try_pair_removals(
                    X, y, all_params, phase1_data, seen, D);

                {
                    std::ofstream flog(log_file, std::ios::app);
                    flog << "Phase 1.5: trying " << pair_removal.candidates.size()
                         << " pair removals" << std::endl;
                }

                if (!pair_removal.candidates.empty()) {
                    std::vector<DecompResult> pair_results;
                    parallel_decompose_batch_with_init(
                        pair_removal.candidates, pair_removal.warmstart_params, pair_results);

                    bool pair_solution = false;
                    for (int si = 0; si < static_cast<int>(pair_removal.candidates.size()); ++si) {
                        X.push_back(pair_removal.candidates[si].copy());
                        y.push_back(pair_results[si].score);
                        all_params.push_back(pair_results[si].params);
                        decompose_time += pair_results[si].elapsed;
                        decompose_count++;

                        if (pair_results[si].score < best_score ||
                            (pair_results[si].score < tolerance && best_score < tolerance &&
                             pair_removal.candidates[si].size() < best_circuit.size())) {
                            best_score = pair_results[si].score;
                            best_circuit = pair_removal.candidates[si].copy();
                            best_params = pair_results[si].params;
                        }
                        if (pair_results[si].score < tolerance) {
                            pair_solution = true;
                            break;
                        }
                    }

                    if (pair_solution) {
                        std::ofstream flog(log_file, std::ios::app);
                        flog << "SOLUTION found by pair removal at D=" << D
                             << ", score=" << best_score << std::endl;
                        continue;
                    }
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
        std::vector<Matrix_real> seed_init_params;  // P5: warm-start params (empty = random)
        std::mt19937 shrink_gen(42 + D);

        // Shrink-seed from D+1 solutions with parameter transfer (P5)
        if (!prev_indices.empty()) {
            std::sort(prev_indices.begin(), prev_indices.end(),
                      [&y](int a, int b) { return y[a] < y[b]; });

            int needed = X0_size;
            for (int pi : prev_indices) {
                if (n_shrunk >= needed) break;
                int D_plus1 = static_cast<int>(X[pi].size());
                std::uniform_int_distribution<int> pos_dist(0, D_plus1 - 1);
                for (int attempt = 0; attempt < 10 && n_shrunk < needed; ++attempt) {
                    int pos = pos_dist(shrink_gen);
                    GrayCode shrunk_raw = X[pi].remove_Digit(pos);
                    GrayCode shrunk = canonicalize_and_validate(shrunk_raw);
                    if (shrunk.size() > 0 && seen.find(shrunk) == seen.end()) {
                        seen.insert(shrunk.copy());
                        seed_circuits.push_back(shrunk.copy());
                        seed_init_params.push_back(
                            excise_gate_params(all_params[pi], D_plus1, pos));
                        n_shrunk++;
                    }
                }
            }
        }

        // Find best D-length params so far for warm-starting random seeds
        Matrix_real best_D_params;
        double best_D_score = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < X.size(); ++i) {
            if (static_cast<int>(X[i].size()) == D && y[i] < best_D_score) {
                best_D_score = y[i];
                best_D_params = all_params[i];
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
                seed_init_params.push_back(best_D_params.size() > 0 ? best_D_params.copy() : Matrix_real(1, 0));
                n_random++;
            }
        }

        // Parallel decompose all seeds with warm-start where available
        if (!seed_circuits.empty()) {
            std::vector<DecompResult> dec_results;
            parallel_decompose_batch_with_init(seed_circuits, seed_init_params, dec_results);

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

        // Build fresh WLKernelCache for this D (same-D only)
        std::vector<int> cache_candidates;
        for (size_t i = 0; i < X.size(); ++i)
            if (static_cast<int>(X[i].size()) == D)
                cache_candidates.push_back(static_cast<int>(i));

        std::sort(cache_candidates.begin(), cache_candidates.end(),
                  [&y](int a, int b) { return y[a] < y[b]; });

        int cache_limit = std::min(static_cast<int>(cache_candidates.size()), 2 * gp_max_train);

        WLKernelCache wl_cache(wl_iterations, token_masks);
        for (int ci = 0; ci < cache_limit; ++ci)
            wl_cache.register_circuit(X[cache_candidates[ci]]);

        // Run window search at this D
        WindowResult result = run_window_search(D, D,
            wl_cache, seen, X, y, all_params, log_file);

        if (result == WINDOW_SUCCESS) {
            // Solution found at this D, keep compressing
            continue;
        }

        // Stagnation — one retry with fresh random seeding (no second GP round)
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
            for (size_t i = 0; i < X.size(); ++i) {
                if (static_cast<int>(X[i].size()) == D && y[i] < best_D_score) {
                    best_D_score = y[i];
                    best_D_params = all_params[i];
                }
            }
            std::vector<Matrix_real> retry_init(retry_circuits.size());
            for (size_t i = 0; i < retry_circuits.size(); ++i)
                retry_init[i] = best_D_params.size() > 0 ? best_D_params.copy() : Matrix_real(1, 0);

            std::vector<DecompResult> dec_results;
            parallel_decompose_batch_with_init(retry_circuits, retry_init, dec_results);

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
// Per-window surrogate search — simplified single-pass
// ============================================================================

N_Qubit_Decomposition_Surrogate::WindowResult
N_Qubit_Decomposition_Surrogate::run_window_search(
    int win_lo, int win_hi,
    WLKernelCache& wl_cache, GrayCodeSet& seen,
    std::vector<GrayCode>& X, std::vector<double>& y,
    std::vector<Matrix_real>& all_params,
    const std::string& log_file,
    int patience_override) {

    (void)patience_override;  // unused in single-pass mode

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
            int batch_target = needed - generated;
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
                    wl_cache.register_circuit(seq);
                    batch_circuits.push_back(std::move(seq));
                }
            }

            if (batch_circuits.empty()) continue;

            std::vector<DecompResult> batch_results;
            parallel_decompose_batch(batch_circuits, batch_results);

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

    // --- 3-round GP loop ---
    constexpr int n_gp_rounds = 3;
    int d_filter_lo = std::max(1, win_lo - 2);
    int d_filter_hi = win_hi + 2;
    bool improved = false;
    bool found_solution = false;

    for (int round = 0; round < n_gp_rounds && !found_solution; ++round) {
        auto t_round_start = std::chrono::high_resolution_clock::now();

        // Train GP (retrain each round with growing dataset)
        surrogate_->reset();
        auto train_result = surrogate_->train_window(X, y, d_filter_lo, d_filter_hi);
        if (train_result.n_train == 0) break;

        double scale = train_result.scale;

        // Generate candidates
        std::vector<GrayCode> candidates;
        int n_local;
        double avg_steps;
        mutators_->generate_candidates(X, y.data(), static_cast<int>(X.size()),
                                        candidates_per_iter,
                                        surrogate_->cache(), surrogate_->gp(), scale,
                                        train_result.log_y_norm.data(),
                                        seen, candidates, n_local, avg_steps,
                                        gen, win_lo, win_hi, 0.0);

        if (candidates.empty()) break;

        // LCB acquisition
        int depth_adaptive_budget = std::max(60, static_cast<int>(eval_budget_base * 30.0 / std::sqrt(std::max(win_hi, 1))));
        auto acq_result = surrogate_->select_candidates_lcb(
            candidates, seen, kappa, d_penalty, win_hi, depth_adaptive_budget);

        int n_sel = static_cast<int>(acq_result.selected_indices.size());
        if (n_sel == 0) break;

        // Decompose selected candidates
        std::vector<GrayCode> sel_circuits(n_sel);
        for (int si = 0; si < n_sel; ++si)
            sel_circuits[si] = candidates[acq_result.selected_indices[si]].copy();

        std::vector<DecompResult> dec_results;
        parallel_decompose_batch(sel_circuits, dec_results);

        // Update search state
        for (int si = 0; si < n_sel; ++si) {
            int sel_idx = acq_result.selected_indices[si];
            double new_score = dec_results[si].score;

            y.push_back(new_score);
            all_params.push_back(dec_results[si].params);
            X.push_back(candidates[sel_idx].copy());
            wl_cache.register_circuit(candidates[sel_idx]);
            decompose_time += dec_results[si].elapsed;
            decompose_count++;

            if (new_score < best_score ||
                (new_score < tolerance && best_score < tolerance &&
                 candidates[sel_idx].size() < best_circuit.size())) {
                best_score = new_score;
                best_circuit = candidates[sel_idx].copy();
                best_params = dec_results[si].params;
                improved = true;
            }

            if (new_score < tolerance &&
                static_cast<int>(candidates[sel_idx].size()) >= win_lo &&
                static_cast<int>(candidates[sel_idx].size()) <= win_hi) {
                found_solution = true;
                break;
            }
        }

        // P3: Multi-restart BFGS for near-threshold candidates
        if (!found_solution) {
            std::vector<int> retry_indices;
            for (int si = 0; si < n_sel; ++si) {
                double s = dec_results[si].score;
                if (s >= 1e-6 && s < 1e-2)
                    retry_indices.push_back(si);
            }
            if (!retry_indices.empty()) {
                const int n_restarts = 5;
                std::ofstream flog(log_file, std::ios::app);
                flog << "  P3 multi-restart BFGS: retrying " << retry_indices.size()
                     << " near-threshold candidates" << std::endl;

                for (int ri : retry_indices) {
                    int sel_idx = acq_result.selected_indices[ri];
                    const GrayCode& circ = candidates[sel_idx];
                    double orig_score = dec_results[ri].score;

                    for (int rs = 0; rs < n_restarts; ++rs) {
                        auto retry_res = decompose(circ);
                        decompose_count++;

                        if (retry_res.first < orig_score) {
                            for (int xi = static_cast<int>(X.size()) - 1; xi >= 0; --xi) {
                                if (X[xi] == circ) {
                                    y[xi] = retry_res.first;
                                    all_params[xi] = retry_res.second;
                                    break;
                                }
                            }
                            orig_score = retry_res.first;

                            if (retry_res.first < best_score ||
                                (retry_res.first < tolerance && best_score < tolerance &&
                                 circ.size() < best_circuit.size())) {
                                best_score = retry_res.first;
                                best_circuit = circ.copy();
                                best_params = retry_res.second;
                                improved = true;
                            }
                            if (retry_res.first < tolerance) {
                                found_solution = true;
                                break;
                            }
                        }
                    }
                    if (found_solution) break;
                }
            }
        }

        // Log round
        double spearman_rho = 0.0;
        if (n_sel >= 5) {
            std::vector<double> actual_scores(n_sel);
            for (int si = 0; si < n_sel; ++si)
                actual_scores[si] = dec_results[si].score;
            spearman_rho = SurrogateModel::compute_rho(acq_result.gp_mu, actual_scores);
        }
        auto t_round_end = std::chrono::high_resolution_clock::now();
        double round_time = std::chrono::duration<double>(t_round_end - t_round_start).count();

        {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  GP round " << round << "/" << n_gp_rounds
                 << ": best=" << best_score
                 << " n_train=" << train_result.n_train
                 << " n_sel=" << n_sel
                 << " rho=" << std::setprecision(3) << spearman_rho
                 << " [time=" << std::fixed << std::setprecision(2) << round_time << "s]"
                 << std::defaultfloat << std::endl;
        }

        if (found_solution) {
            std::ofstream flog(log_file, std::ios::app);
            flog << "  SOLUTION at D=" << best_circuit.size()
                 << ", score=" << best_score << std::endl;
            return WINDOW_SUCCESS;
        }
    }

    // If we already have a solution at or below this window, that's success
    if (best_score < tolerance && static_cast<int>(best_circuit.size()) <= win_hi
        && static_cast<int>(best_circuit.size()) >= win_lo)
        return WINDOW_SUCCESS;

    return improved ? WINDOW_SUCCESS : WINDOW_STAGNATION;
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
        sstream << "Starting surrogate"
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
        sstream << "Starting surrogate"
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
        // Re-decompose to get compatible params — use existing best_params as warm-start
        // (decompose_with_initial_params clips/pads on mismatch) instead of random init,
        // which can lose a good solution when the initial circuit already scored well.
        if (static_cast<int>(best_params.size()) != expected_param_num) {
            delete gate_structure;
            std::pair<double, Matrix_real> result = decompose_with_initial_params(best_circuit, best_params);
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

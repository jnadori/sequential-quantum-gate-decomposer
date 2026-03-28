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

/*! \file N_Qubit_Decomposition_Surrogate_GateLevel.cpp
    \brief Skeleton-based surrogate search with per-CNOT 1q gate pattern trials.
    Searches over CNOT skeletons (edge tokens, fast) and tries multiple 1q gate
    arrangements during each skeleton evaluation.
*/

#include "N_Qubit_Decomposition_Surrogate_GateLevel.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#if BLAS == 1
extern "C" void MKL_Set_Num_Threads(int);
#elif BLAS == 2
extern "C" void openblas_set_num_threads(int);
#endif

static void force_single_thread_blas_lapack_gl() {
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
// Constructors and destructor
// ============================================================================

N_Qubit_Decomposition_Surrogate_GateLevel::N_Qubit_Decomposition_Surrogate_GateLevel(
    Matrix Umtx_in, int qbit_num_in,
    std::vector<matrix_base<int>> topology_in,
    std::map<std::string, Config_Element>& config_in,
    int accelerator_num_in)
    : N_Qubit_Decomposition_Surrogate(Umtx_in, qbit_num_in, topology_in, config_in, accelerator_num_in)
{
    // Parse gate_1q_gateset bitmask (reuse parent's member arrays)
    const std::vector<std::tuple<int, gate_type, int>> gateset_table = {
        { 0, U3_OPERATION,  3},
        { 1, RY_OPERATION,  1},
        { 2, RX_OPERATION,  1},
        { 3, RZ_OPERATION,  1},
        { 4, H_OPERATION,   0},
        { 5, X_OPERATION,   0},
        { 6, Y_OPERATION,   0},
        { 7, Z_OPERATION,   0},
        { 8, S_OPERATION,   0},
        { 9, SDG_OPERATION, 0},
        {10, T_OPERATION,   0},
        {11, TDG_OPERATION, 0},
        {12, SX_OPERATION,  0},
        {13, SXDG_OPERATION,0},
        {14, U1_OPERATION,  1},
        {15, U2_OPERATION,  2},
        {16, R_OPERATION,   2},
    };

    unsigned long long gateset_mask = 1;  // default: U3 only
    if (config.count("gate_1q_gateset") > 0) {
        long long v; config["gate_1q_gateset"].get_property(v);
        gateset_mask = static_cast<unsigned long long>(v);
    }

    gate_1q_types.clear();
    gate_1q_param_counts.clear();
    for (size_t gi = 0; gi < gateset_table.size(); ++gi) {
        int bit = std::get<0>(gateset_table[gi]);
        if (gateset_mask & (1ULL << bit)) {
            gate_1q_types.push_back(std::get<1>(gateset_table[gi]));
            gate_1q_param_counts.push_back(std::get<2>(gateset_table[gi]));
        }
    }
    n_1q_types = static_cast<int>(gate_1q_types.size());

    build_pattern_library();
}

N_Qubit_Decomposition_Surrogate_GateLevel::N_Qubit_Decomposition_Surrogate_GateLevel(
    Matrix Umtx_in, int qbit_num_in,
    std::map<std::string, Config_Element>& config_in,
    int accelerator_num_in)
    : N_Qubit_Decomposition_Surrogate_GateLevel(Umtx_in, qbit_num_in,
          std::vector<matrix_base<int>>(), config_in, accelerator_num_in) {}

N_Qubit_Decomposition_Surrogate_GateLevel::~N_Qubit_Decomposition_Surrogate_GateLevel() {}


// ============================================================================
// Pattern library construction
// ============================================================================

void N_Qubit_Decomposition_Surrogate_GateLevel::build_pattern_library() {
    pattern_library.clear();

    // Pattern 0: default U3+U3 (most general, always present)
    pattern_library.push_back({{U3_OPERATION}, {U3_OPERATION}});

    // Single-gate symmetric patterns: {g, g} for each gate type
    for (int i = 0; i < n_1q_types; ++i) {
        gate_type g = gate_1q_types[i];
        if (g == U3_OPERATION) continue;  // already added as pattern 0
        pattern_library.push_back({{g}, {g}});
    }

    // Asymmetric single-gate patterns: {g1, g2} for each pair
    for (int i = 0; i < n_1q_types; ++i) {
        for (int j = 0; j < n_1q_types; ++j) {
            if (i == j) continue;
            pattern_library.push_back({{gate_1q_types[i]}, {gate_1q_types[j]}});
        }
    }

    // Multi-gate patterns (if constituent gates are available)
    auto has_gate = [&](gate_type g) -> bool {
        for (int i = 0; i < n_1q_types; ++i)
            if (gate_1q_types[i] == g) return true;
        return false;
    };

    // ZYZ decomposition
    if (has_gate(RZ_OPERATION) && has_gate(RY_OPERATION)) {
        pattern_library.push_back(
            {{RZ_OPERATION, RY_OPERATION, RZ_OPERATION},
             {RZ_OPERATION, RY_OPERATION, RZ_OPERATION}});
    }

    // IBM basis: RZ-SX-RZ
    if (has_gate(RZ_OPERATION) && has_gate(SX_OPERATION)) {
        pattern_library.push_back(
            {{RZ_OPERATION, SX_OPERATION, RZ_OPERATION},
             {RZ_OPERATION, SX_OPERATION, RZ_OPERATION}});
    }

    // Bare CNOT (no 1q gates)
    pattern_library.push_back({{}, {}});

    std::stringstream sstream;
    sstream << "GateLevel: built pattern library with "
            << pattern_library.size() << " patterns from "
            << n_1q_types << " gate types" << std::endl;
    print(sstream, 1);
}


// ============================================================================
// Gate structure construction with specific pattern assignment
// ============================================================================

Gates_block* N_Qubit_Decomposition_Surrogate_GateLevel::construct_gate_structure_with_patterns(
    const GrayCode& gcode, const std::vector<int>& pattern_indices, bool finalize) {

    Gates_block* gate_structure = new Gates_block(qbit_num);
    int n_patterns = static_cast<int>(pattern_library.size());

    for (int i = 0; i < static_cast<int>(gcode.size()); ++i) {
        int token = gcode[i];
        int target = possible_target_qbits[token];
        int control = possible_control_qbits[token];

        int pidx = pattern_indices[i] % n_patterns;
        const GatePattern& pat = pattern_library[pidx];

        Gates_block* layer = new Gates_block(qbit_num);

        // Add 1q gates on target qubit
        for (gate_type g : pat.target_gates) {
            switch (g) {
                case U3_OPERATION:   layer->add_u3(target);   break;
                case RY_OPERATION:   layer->add_ry(target);   break;
                case RX_OPERATION:   layer->add_rx(target);   break;
                case RZ_OPERATION:   layer->add_rz(target);   break;
                case H_OPERATION:    layer->add_h(target);    break;
                case X_OPERATION:    layer->add_x(target);    break;
                case Y_OPERATION:    layer->add_y(target);    break;
                case Z_OPERATION:    layer->add_z(target);    break;
                case S_OPERATION:    layer->add_s(target);    break;
                case SDG_OPERATION:  layer->add_sdg(target);  break;
                case T_OPERATION:    layer->add_t(target);    break;
                case TDG_OPERATION:  layer->add_tdg(target);  break;
                case SX_OPERATION:   layer->add_sx(target);   break;
                case SXDG_OPERATION: layer->add_sxdg(target); break;
                case U1_OPERATION:   layer->add_u1(target);   break;
                case U2_OPERATION:   layer->add_u2(target);   break;
                case R_OPERATION:    layer->add_r(target);    break;
                default:             layer->add_u3(target);   break;
            }
        }

        // Add 1q gates on control qubit
        for (gate_type g : pat.control_gates) {
            switch (g) {
                case U3_OPERATION:   layer->add_u3(control);   break;
                case RY_OPERATION:   layer->add_ry(control);   break;
                case RX_OPERATION:   layer->add_rx(control);   break;
                case RZ_OPERATION:   layer->add_rz(control);   break;
                case H_OPERATION:    layer->add_h(control);    break;
                case X_OPERATION:    layer->add_x(control);    break;
                case Y_OPERATION:    layer->add_y(control);    break;
                case Z_OPERATION:    layer->add_z(control);    break;
                case S_OPERATION:    layer->add_s(control);    break;
                case SDG_OPERATION:  layer->add_sdg(control);  break;
                case T_OPERATION:    layer->add_t(control);    break;
                case TDG_OPERATION:  layer->add_tdg(control);  break;
                case SX_OPERATION:   layer->add_sx(control);   break;
                case SXDG_OPERATION: layer->add_sxdg(control); break;
                case U1_OPERATION:   layer->add_u1(control);   break;
                case U2_OPERATION:   layer->add_u2(control);   break;
                case R_OPERATION:    layer->add_r(control);    break;
                default:             layer->add_u3(control);   break;
            }
        }

        // Add CNOT
        layer->add_cnot(target, control);
        gate_structure->add_gate(layer);
    }

    if (finalize)
        add_finalyzing_layer(gate_structure);

    return gate_structure;
}


// ============================================================================
// Decompose override: try multiple 1q gate patterns per skeleton
// ============================================================================

std::pair<double, Matrix_real> N_Qubit_Decomposition_Surrogate_GateLevel::decompose_with_rng(
    const GrayCode& circuit, std::mt19937& local_gen) {

    int D = static_cast<int>(circuit.size());
    int n_patterns = static_cast<int>(pattern_library.size());

    double best_score = std::numeric_limits<double>::infinity();
    Matrix_real best_params;
    std::vector<int> best_pattern(D, 0);

    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);

    // Try every pattern uniformly (same pattern applied to all layers)
    for (int trial = 0; trial < n_patterns; ++trial) {
        std::vector<int> pattern_assignment(D, trial);

        // Build gate structure and optimize
        Gates_block* gate_structure = construct_gate_structure_with_patterns(
            circuit, pattern_assignment);
        int param_num = gate_structure->get_parameter_num();

        double score;
        Matrix_real params;

        N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false,
                                              config, RANDOM, accelerator_num);
        cDecomp.set_custom_gate_structure(gate_structure);
        delete gate_structure;
        cDecomp.set_verbose(0);
        cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
        cDecomp.set_optimization_tolerance(tolerance);
        cDecomp.set_optimizer(alg);

        if (param_num == 0) {
            // No free parameters — just evaluate the fixed circuit
            Matrix_real empty_params(1, 0);
            score = cDecomp.optimization_problem(empty_params);
        } else {
            Matrix_real random_params(1, param_num);
            for (int i = 0; i < param_num; ++i)
                random_params[i] = param_dist(local_gen);
            cDecomp.set_optimized_parameters(random_params.get_data(), param_num);

            cDecomp.start_decomposition();

            params = cDecomp.get_optimized_parameters();
            score = cDecomp.optimization_problem(params);
        }

        if (score < best_score) {
            best_score = score;
            best_params = params;
            best_pattern = pattern_assignment;
        }

        // Early exit if we found a solution
        if (best_score < tolerance) break;
    }

    // Store the winning pattern (racy during parallel search, but read
    // reliably after the single-threaded final evaluation in start_decomposition)
    last_best_pattern = best_pattern;

    return {best_score, best_params};
}


// ============================================================================
// Main entry point
// ============================================================================

void N_Qubit_Decomposition_Surrogate_GateLevel::start_decomposition() {
    std::stringstream sstream;
    sstream << "Starting " << (use_random_candidates ? "RANDOM baseline" : "surrogate")
            << " gate-level search for " << qbit_num << "-qubit matrix"
            << " (" << pattern_library.size() << " patterns per skeleton)" << std::endl;
    print(sstream, 1);

    force_single_thread_blas_lapack_gl();

    // Edge-based search: D counts 2q blocks (same as parent)
    int D_start = osr_D_min;
    int D_end = level_limit;
    if (D_end <= 0) D_end = D_start + 10;

    std::string log_prefix = project_name.empty() ? "sursearch_gl" : "sursearch_gl_" + project_name;
    std::string log_file = log_prefix + "_D" + std::to_string(D_start) + "-" +
                            std::to_string(D_end) + ".txt";

    // Run the edge-based search (inherited).
    // decompose_with_rng is virtual, so our override is called automatically.
    search_over_D_range(D_start, D_end, log_file);

    // Re-evaluate the best skeleton to find the optimal pattern assignment
    // and build the final gate structure.
    if (best_circuit.size() > 0) {
        auto [final_score, final_params] = decompose_with_rng(best_circuit, gen);
        // last_best_pattern now holds the winning pattern assignment

        Gates_block* gate_structure = construct_gate_structure_with_patterns(
            best_circuit, last_best_pattern);
        release_gates();
        combine(gate_structure);
        delete gate_structure;
        optimized_parameters_mtx = final_params;
        decomposition_error = final_score;
    }
}

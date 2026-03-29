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
    \brief Skeleton-based surrogate search with per-CROT 1q gate pattern trials.
    Searches over CROT skeletons (edge tokens) and tries two fixed 1q gate
    patterns during each skeleton evaluation:
      Pattern A: R(target)-Rz(target)-R(control)-Rz(control)-CROT
      Pattern B: Rz(target)-R(control)-Rz(control)-CROT
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

    // Pattern 0: R(target)-Rz(target)-R(control)-Rz(control)-CROT
    pattern_library.push_back(
        {{R_OPERATION, RZ_OPERATION},
         {R_OPERATION, RZ_OPERATION}});

    // Pattern 1: Rz(target)-R(control)-Rz(control)-CROT
    pattern_library.push_back(
        {{RZ_OPERATION},
         {R_OPERATION, RZ_OPERATION}});

    std::stringstream sstream;
    sstream << "GateLevel: built pattern library with "
            << pattern_library.size() << " patterns (CROT-based)" << std::endl;
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
            if (g == R_OPERATION)  layer->add_r(target);
            else                   layer->add_rz(target);
        }

        // Add 1q gates on control qubit
        for (gate_type g : pat.control_gates) {
            if (g == R_OPERATION)  layer->add_r(control);
            else                   layer->add_rz(control);
        }

        // Add CROT
        layer->add_crot(target, control);
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
            << " gate-level CROT search for " << qbit_num << "-qubit matrix"
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

    // Build the final gate structure using the best result from the search.
    // best_params and best_score are already set correctly by search_over_D_range.
    // We find the matching pattern by evaluating best_params (no re-optimization)
    // across each uniform pattern option — O(n_patterns) cheap evaluations.
    if (best_circuit.size() > 0) {
        int D = static_cast<int>(best_circuit.size());
        int n_pat = static_cast<int>(pattern_library.size());
        int best_param_size = static_cast<int>(best_params.size());

        std::vector<int> winning_pattern(D, 0);
        double best_eval = std::numeric_limits<double>::infinity();

        for (int trial = 0; trial < n_pat; ++trial) {
            std::vector<int> pattern_assignment(D, trial);
            Gates_block* gs = construct_gate_structure_with_patterns(
                best_circuit, pattern_assignment);

            if (gs->get_parameter_num() == best_param_size) {
                N_Qubit_Decomposition_custom cDecomp(
                    Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
                cDecomp.set_custom_gate_structure(gs);
                delete gs;
                cDecomp.set_verbose(0);
                cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);

                double s;
                if (best_param_size == 0) {
                    Matrix_real empty_params(1, 0);
                    s = cDecomp.optimization_problem(empty_params);
                } else {
                    s = cDecomp.optimization_problem(best_params);
                }

                if (s < best_eval) {
                    best_eval = s;
                    winning_pattern = pattern_assignment;
                }
                if (s < tolerance) break;
            } else {
                delete gs;
            }
        }

        Gates_block* gate_structure = construct_gate_structure_with_patterns(
            best_circuit, winning_pattern);
        release_gates();
        combine(gate_structure);
        delete gate_structure;
        optimized_parameters_mtx = best_params;
        decomposition_error = best_score;
    }
}

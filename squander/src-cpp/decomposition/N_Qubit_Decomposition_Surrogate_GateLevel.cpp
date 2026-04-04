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
    \brief Skeleton-based surrogate search with dense 1q gates.
    Searches over CROT skeletons (edge tokens) with a fixed dense gate pattern:
      R(target)-Rz(target)-R(control)-Rz(control)-CROT per layer.
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
}

N_Qubit_Decomposition_Surrogate_GateLevel::N_Qubit_Decomposition_Surrogate_GateLevel(
    Matrix Umtx_in, int qbit_num_in,
    std::map<std::string, Config_Element>& config_in,
    int accelerator_num_in)
    : N_Qubit_Decomposition_Surrogate_GateLevel(Umtx_in, qbit_num_in,
          std::vector<matrix_base<int>>(), config_in, accelerator_num_in) {}

N_Qubit_Decomposition_Surrogate_GateLevel::~N_Qubit_Decomposition_Surrogate_GateLevel() {}


// ============================================================================
// Gate structure construction: dense R+Rz on both qubits per CROT
// ============================================================================

Gates_block* N_Qubit_Decomposition_Surrogate_GateLevel::construct_gate_structure(
    const GrayCode& gcode, bool finalize) {

    Gates_block* gate_structure = new Gates_block(qbit_num);

    for (int i = 0; i < static_cast<int>(gcode.size()); ++i) {
        int token = gcode[i];
        int target = possible_target_qbits[token];
        int control = possible_control_qbits[token];

        Gates_block* layer = new Gates_block(qbit_num);

        // Dense 1q gates: R + Rz on target, R + Rz on control
        layer->add_r(target);
        layer->add_rz(target);
        layer->add_r(control);
        layer->add_rz(control);

        // CROT
        layer->add_crot(target, control);
        gate_structure->add_gate(layer);
    }

    if (finalize)
        add_finalyzing_layer(gate_structure);

    return gate_structure;
}


// ============================================================================
// Decompose override: single dense pattern, no trials
// ============================================================================

std::pair<double, Matrix_real> N_Qubit_Decomposition_Surrogate_GateLevel::decompose_with_rng(
    const GrayCode& circuit, std::mt19937& local_gen) {

    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);

    Gates_block* gate_structure = construct_gate_structure(circuit);
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

    return {score, params};
}


// ============================================================================
// Main entry point
// ============================================================================

void N_Qubit_Decomposition_Surrogate_GateLevel::start_decomposition() {

    force_single_thread_blas_lapack_gl();

    if (gates.size() > 0) {
        // Compression mode: imported gate structure detected
        std::stringstream sstream;
        sstream << "Starting surrogate gate-level COMPRESSION for " << qbit_num
                << "-qubit matrix (dense R+Rz pattern)" << std::endl;
        print(sstream, 1);

        GrayCode skeleton = extract_skeleton_from_gates();
        int D_initial = static_cast<int>(skeleton.size());

        // Save imported parameters before releasing gates
        Matrix_real imported_params = optimized_parameters_mtx.copy();

        // Clone the imported gate structure for compress_over_D_range
        Gates_block* gate_structure_for_compress = new Gates_block(qbit_num);
        for (Gate* g : gates) {
            gate_structure_for_compress->add_gate(g->clone());
        }

        release_gates();

        int D_min = (config_D_start >= 0) ? config_D_start : 1;

        std::string log_prefix = project_name.empty() ? "surcompress_gl" : "surcompress_gl_" + project_name;
        std::string log_file = log_prefix + "_D" + std::to_string(D_initial) + "-" +
                                std::to_string(D_min) + ".txt";

        compress_over_D_range(D_initial, D_min, log_file, skeleton,
                              gate_structure_for_compress, imported_params);

        if (best_circuit.size() > 0) {
            Gates_block* gate_structure = construct_gate_structure(best_circuit);
            release_gates();
            combine(gate_structure);
            delete gate_structure;
            optimized_parameters_mtx = best_params;
            decomposition_error = best_score;
        }
    } else {
        // Bottom-up adaptive-style search: fully dense all-edges layers
        std::stringstream sstream;
        sstream << "Starting adaptive gate-level search for " << qbit_num
                << "-qubit matrix (dense R+Rz+CROT, all edges per layer)" << std::endl;
        print(sstream, 1);

        int n_edges = static_cast<int>(topology.size());
        int level_start = (config_D_start >= 0) ? config_D_start : 1;
        int level_end = level_limit;
        if (level_end <= 0) level_end = level_start + 10;

        std::uniform_real_distribution<double> param_dist(0.0, 2.0 * M_PI);

        for (int level = level_start; level <= level_end; ++level) {
            // Build gate structure: `level` layers, each with all topology edges
            Gates_block* gate_structure = new Gates_block(qbit_num);
            for (int l = 0; l < level; ++l) {
                for (int e = 0; e < n_edges; ++e) {
                    int target = possible_target_qbits[e];
                    int control = possible_control_qbits[e];
                    Gates_block* layer = new Gates_block(qbit_num);
                    layer->add_r(target);
                    layer->add_rz(target);
                    layer->add_r(control);
                    layer->add_rz(control);
                    layer->add_crot(target, control);
                    gate_structure->add_gate(layer);
                }
            }
            add_finalyzing_layer(gate_structure);

            int param_num = gate_structure->get_parameter_num();

            std::stringstream ss;
            ss << "Level " << level << ": " << level * n_edges << " CROTs, "
               << param_num << " params" << std::endl;
            print(ss, 1);

            // Optimize
            N_Qubit_Decomposition_custom cDecomp(Umtx.copy(), qbit_num, false,
                                                  config, RANDOM, accelerator_num);
            cDecomp.set_custom_gate_structure(gate_structure);
            delete gate_structure;
            cDecomp.set_verbose(0);
            cDecomp.set_cost_function_variant(HILBERT_SCHMIDT_TEST);
            cDecomp.set_optimization_tolerance(tolerance);
            cDecomp.set_optimizer(alg);

            std::mt19937 gen(42 + level);
            Matrix_real random_params(1, param_num);
            for (int i = 0; i < param_num; ++i)
                random_params[i] = param_dist(gen);
            cDecomp.set_optimized_parameters(random_params.get_data(), param_num);

            cDecomp.start_decomposition();

            Matrix_real opt_params = cDecomp.get_optimized_parameters();
            double score = cDecomp.optimization_problem(opt_params);

            std::stringstream ss2;
            ss2 << "Level " << level << ": score=" << score << std::endl;
            print(ss2, 1);

            if (score < decomposition_error || decomposition_error < 0) {
                decomposition_error = score;
                optimized_parameters_mtx = opt_params;

                // Rebuild the gate structure to store as the result
                Gates_block* result_gs = new Gates_block(qbit_num);
                for (int l = 0; l < level; ++l) {
                    for (int e = 0; e < n_edges; ++e) {
                        int target = possible_target_qbits[e];
                        int control = possible_control_qbits[e];
                        Gates_block* layer = new Gates_block(qbit_num);
                        layer->add_r(target);
                        layer->add_rz(target);
                        layer->add_r(control);
                        layer->add_rz(control);
                        layer->add_crot(target, control);
                        result_gs->add_gate(layer);
                    }
                }
                add_finalyzing_layer(result_gs);

                release_gates();
                combine(result_gs);
                delete result_gs;
            }

            if (score < tolerance) {
                std::stringstream ss3;
                ss3 << "Solution found at level " << level << " with "
                    << level * n_edges << " CROTs" << std::endl;
                print(ss3, 1);
                break;
            }
        }
    }
}

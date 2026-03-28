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

/*! \file N_Qubit_Decomposition_Surrogate_GateLevel.h
    \brief Header file for gate-level surrogate search.
    Searches over CNOT skeletons (edge tokens) and tries multiple 1q gate
    arrangements per skeleton during evaluation.
*/

#ifndef N_Qubit_Decomposition_Surrogate_GateLevel_H
#define N_Qubit_Decomposition_Surrogate_GateLevel_H

#include "N_Qubit_Decomposition_Surrogate.h"


class N_Qubit_Decomposition_Surrogate_GateLevel
    : public N_Qubit_Decomposition_Surrogate {

public:
    /// Per-CNOT-layer pattern: 1q gates on target and control qubits
    struct GatePattern {
        std::vector<gate_type> target_gates;
        std::vector<gate_type> control_gates;
    };

    /// Constructor with topology
    N_Qubit_Decomposition_Surrogate_GateLevel(Matrix Umtx_in, int qbit_num_in,
        std::vector<matrix_base<int>> topology_in,
        std::map<std::string, Config_Element>& config,
        int accelerator_num = 0);

    /// Constructor without topology (full connectivity)
    N_Qubit_Decomposition_Surrogate_GateLevel(Matrix Umtx_in, int qbit_num_in,
        std::map<std::string, Config_Element>& config,
        int accelerator_num = 0);

    virtual ~N_Qubit_Decomposition_Surrogate_GateLevel();

    /// Main entry point — edge-based skeleton search + pattern trial evaluation
    void start_decomposition() override;

    /// Override: tries multiple 1q gate patterns per skeleton, returns best
    std::pair<double, Matrix_real> decompose_with_rng(
        const GrayCode& circuit, std::mt19937& local_gen) override;

    /// Build pattern library from gate_1q_gateset config
    void build_pattern_library();

    /// Build gate structure using a specific pattern assignment (one pattern index per edge)
    Gates_block* construct_gate_structure_with_patterns(
        const GrayCode& gcode, const std::vector<int>& pattern_indices,
        bool finalize = true);

protected:
    /// Library of per-CNOT-layer gate patterns
    std::vector<GatePattern> pattern_library;

    /// Pattern assignment from the last decompose_with_rng call (used by start_decomposition)
    std::vector<int> last_best_pattern;
};


#endif // N_Qubit_Decomposition_Surrogate_GateLevel_H

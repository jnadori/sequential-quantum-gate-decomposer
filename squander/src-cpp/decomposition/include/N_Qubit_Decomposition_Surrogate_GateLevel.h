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
    Searches over CROT skeletons (edge tokens) with dense 1q gates:
      R(target)-Rz(target)-R(control)-Rz(control)-CROT per layer.
*/

#ifndef N_Qubit_Decomposition_Surrogate_GateLevel_H
#define N_Qubit_Decomposition_Surrogate_GateLevel_H

#include "N_Qubit_Decomposition_Surrogate.h"


class N_Qubit_Decomposition_Surrogate_GateLevel
    : public N_Qubit_Decomposition_Surrogate {

public:
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

    /// Main entry point — edge-based skeleton search
    void start_decomposition() override;

    /// Override: builds dense R+Rz gate structure and optimizes
    std::pair<double, Matrix_real> decompose_with_rng(
        const GrayCode& circuit, std::mt19937& local_gen) override;

    /// Build gate structure with dense 1q gates (R+Rz on target and control per CROT)
    Gates_block* construct_gate_structure(const GrayCode& gcode, bool finalize = true);
};


#endif // N_Qubit_Decomposition_Surrogate_GateLevel_H

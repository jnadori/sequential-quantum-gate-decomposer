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

/*! \file GateRemover.h
    \brief Phase 1 single-gate removal + Phase 1.5 multi-gate pair removal.
*/

#ifndef GateRemover_H
#define GateRemover_H

#include "CircuitCanonicalizer.h"
#include "GrayCode.h"
#include "GrayCodeHash.h"
#include "matrix_real.h"

#include <vector>

class GateRemover {
public:
    GateRemover(CircuitCanonicalizer* canon, int qbit_num);

    struct RemovalResult {
        std::vector<GrayCode> candidates;
        std::vector<Matrix_real> warmstart_params;
    };

    struct Phase1Data {
        struct Entry {
            int parent_x_idx;
            int position;
            double score;
            double triviality;
        };
        std::vector<Entry> entries;
    };

    /// Phase 1: single-gate removal from top-K D+1 parents
    RemovalResult try_single_removals(
        const std::vector<GrayCode>& X,
        const std::vector<double>& y,
        const std::vector<Matrix_real>& all_params,
        GrayCodeSet& seen,
        int D,
        Phase1Data& phase1_data_out,
        int phase1_topk = 10,
        int max_removals_factor = 2);

    /// Phase 1.5: multi-gate pair removal using Phase 1 data
    RemovalResult try_pair_removals(
        const std::vector<GrayCode>& X,
        const std::vector<double>& y,
        const std::vector<Matrix_real>& all_params,
        const Phase1Data& phase1_data,
        GrayCodeSet& seen,
        int D,
        int max_pairs = 100,
        int top_k_positions = 15);

    /// Parameter extraction for single-gate removal
    static Matrix_real excise_gate_params(
        const Matrix_real& parent_params, int D_source, int pos, int qbit_num);

    /// Parameter extraction for multi-gate removal
    static Matrix_real excise_multi_gate_params(
        const Matrix_real& parent_params, int D_source,
        const std::vector<int>& positions_sorted_desc, int qbit_num);

    /// Triviality score: how close gate params are to nearest pi/2 multiples
    static double triviality_score(const Matrix_real& params, int pos);

private:
    CircuitCanonicalizer* canon_;
    int qbit_num_;
};

#endif // GateRemover_H

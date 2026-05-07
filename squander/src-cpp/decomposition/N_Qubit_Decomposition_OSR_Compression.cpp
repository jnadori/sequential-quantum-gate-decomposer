/*
Created on Sat May 02 2026
Copyright 2026

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
/*! \file N_Qubit_Decomposition_OSR_Compression.cpp
    \brief OSR-guided top-down compression for an existing gate structure.
*/

#include "N_Qubit_Decomposition_OSR_Compression.h"
#include "N_Qubit_Decomposition_Cost_Function.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <utility>

namespace {

struct CompressionCandidate {
    std::vector<int> removed_ids;
    Matrix_real initial_parameters;
    N_Qubit_Decomposition_OSR_Compression_Score score;
    int entangling_gate_num;
    std::shared_ptr<Gates_block> gate_structure;
    std::string key;
};

struct Phase1RemovalState {
    std::shared_ptr<Gates_block> gate_structure;
    std::vector<int> sequence;
    Matrix_real parameters;
    double current_minimum;
    double decomposition_error;
    int entangling_gate_num;
};

static bool is_entangling_gate_type(gate_type type) {
    switch (type) {
    case CNOT_OPERATION:
    case CZ_OPERATION:
    case CH_OPERATION:
    case SYC_OPERATION:
    case CRY_OPERATION:
    case CRX_OPERATION:
    case CRZ_OPERATION:
    case CP_OPERATION:
    case CR_OPERATION:
    case CROT_OPERATION:
    case CZ_NU_OPERATION:
    case CU_OPERATION:
    case ADAPTIVE_OPERATION:
    case RXX_OPERATION:
    case RYY_OPERATION:
    case RZZ_OPERATION:
    case SWAP_OPERATION:
    case CSWAP_OPERATION:
    case CCX_OPERATION:
        return true;
    default:
        return false;
    }
}

static bool is_entangling_gate(Gate* gate) {
    if (gate == NULL || gate->get_type() == BLOCK_OPERATION) {
        return false;
    }
    if (is_entangling_gate_type(gate->get_type())) {
        return true;
    }
    return gate->get_involved_qubits().size() > 1;
}

static void collect_entangling_gate_paths(Gates_block* block,
                                          std::vector<int>& prefix,
                                          std::vector<OSRGatePath>& out) {
    if (block == NULL) {
        return;
    }

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        prefix.push_back(idx);

        if (is_entangling_gate(gate)) {
            OSRGatePath path;
            path.indices = prefix;
            out.push_back(path);
        }

        if (gate != NULL && gate->get_type() == BLOCK_OPERATION) {
            collect_entangling_gate_paths(static_cast<Gates_block*>(gate), prefix, out);
        }

        prefix.pop_back();
    }
}

static std::vector<OSRGatePath> collect_entangling_gate_paths(Gates_block* block) {
    std::vector<OSRGatePath> ret;
    std::vector<int> prefix;
    collect_entangling_gate_paths(block, prefix, ret);
    return ret;
}

static void append_int_vector_signature(std::stringstream& sstream,
                                        const std::vector<int>& values) {
    sstream << "[";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        if (idx > 0) {
            sstream << ",";
        }
        sstream << values[idx];
    }
    sstream << "]";
}

static void append_gate_structure_signature(Gates_block* block,
                                            std::stringstream& sstream) {
    if (block == NULL) {
        sstream << "NULL";
        return;
    }

    sstream << "B" << block->get_gate_num() << "(";
    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            sstream << "NULL;";
            continue;
        }

        sstream << static_cast<int>(gate->get_type()) << ":T";
        std::vector<int> targets = gate->get_target_qbits();
        if (targets.empty() && gate->get_target_qbit() >= 0) {
            targets.push_back(gate->get_target_qbit());
        }
        append_int_vector_signature(sstream, targets);

        sstream << ":C";
        std::vector<int> controls = gate->get_control_qbits();
        if (controls.empty() && gate->get_control_qbit() >= 0) {
            controls.push_back(gate->get_control_qbit());
        }
        append_int_vector_signature(sstream, controls);

        sstream << ":P" << gate->get_parameter_num();
        if (gate->get_type() == BLOCK_OPERATION) {
            sstream << "{";
            append_gate_structure_signature(static_cast<Gates_block*>(gate), sstream);
            sstream << "}";
        }
        sstream << ";";
    }
    sstream << ")";
}

static std::string gate_structure_signature(Gates_block* block) {
    std::stringstream sstream;
    append_gate_structure_signature(block, sstream);
    return sstream.str();
}

static Gate* gate_at_path(Gates_block* block, const OSRGatePath& path) {
    Gates_block* current_block = block;
    for (size_t depth = 0; depth < path.indices.size(); ++depth) {
        if (current_block == NULL) {
            return NULL;
        }

        int gate_idx = path.indices[depth];
        if (gate_idx < 0 || gate_idx >= current_block->get_gate_num()) {
            return NULL;
        }

        Gate* gate = current_block->get_gate(gate_idx);
        if (depth == path.indices.size() - 1) {
            return gate;
        }
        if (gate == NULL || gate->get_type() != BLOCK_OPERATION) {
            return NULL;
        }
        current_block = static_cast<Gates_block*>(gate);
    }
    return NULL;
}

static bool get_two_qubit_endpoint_pair(Gate* gate, int& q0, int& q1) {
    if (gate == NULL) {
        return false;
    }

    std::vector<int> involved = gate->get_involved_qubits();
    std::sort(involved.begin(), involved.end());
    involved.erase(std::unique(involved.begin(), involved.end()), involved.end());
    if (involved.size() != 2) {
        return false;
    }

    q0 = involved[0];
    q1 = involved[1];
    return true;
}

static bool gate_endpoint_sets_are_disjoint(Gate* lhs, Gate* rhs) {
    int lhs_q0 = 0;
    int lhs_q1 = 0;
    int rhs_q0 = 0;
    int rhs_q1 = 0;
    if (!get_two_qubit_endpoint_pair(lhs, lhs_q0, lhs_q1) ||
        !get_two_qubit_endpoint_pair(rhs, rhs_q0, rhs_q1)) {
        return false;
    }

    return lhs_q0 != rhs_q0 && lhs_q0 != rhs_q1 &&
           lhs_q1 != rhs_q0 && lhs_q1 != rhs_q1;
}

static bool gate_type_is_directional(gate_type type) {
    switch (type) {
    case CZ_OPERATION:
    case SWAP_OPERATION:
    case RXX_OPERATION:
    case RYY_OPERATION:
    case RZZ_OPERATION:
        return false;
    default:
        return true;
    }
}

static bool rewire_two_qubit_gate(Gate* gate, int new_target, int new_control) {
    if (gate == NULL || new_target == new_control) {
        return false;
    }

    int old_q0 = 0;
    int old_q1 = 0;
    if (!get_two_qubit_endpoint_pair(gate, old_q0, old_q1)) {
        return false;
    }

    std::vector<int> controls = gate->get_control_qbits();
    if (!controls.empty() || gate->get_control_qbit() >= 0) {
        gate->set_target_qbit(new_target);
        gate->set_control_qbit(new_control);
        return true;
    }

    std::vector<int> targets = gate->get_target_qbits();
    if (targets.size() >= 2) {
        std::vector<int> new_targets;
        new_targets.push_back(new_target);
        new_targets.push_back(new_control);
        gate->set_target_qbits(new_targets);
        return true;
    }

    return false;
}

static bool path_has_prefix(const OSRGatePath& path, const std::vector<int>& prefix) {
    if (path.indices.size() < prefix.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), path.indices.begin());
}

static bool path_equals_prefix(const OSRGatePath& path, const std::vector<int>& prefix) {
    return path.indices.size() == prefix.size() && path_has_prefix(path, prefix);
}

static bool subtree_contains_removed_path(const std::set<OSRGatePath>& removed_paths,
                                          const std::vector<int>& prefix) {
    for (std::set<OSRGatePath>::const_iterator it = removed_paths.begin(); it != removed_paths.end(); ++it) {
        if (path_has_prefix(*it, prefix)) {
            return true;
        }
    }
    return false;
}

static Gates_block* clone_without_removed_paths(Gates_block* block,
                                                const std::set<OSRGatePath>& removed_paths,
                                                std::vector<int>& prefix) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        prefix.push_back(idx);

        bool remove_gate = false;
        for (std::set<OSRGatePath>::const_iterator it = removed_paths.begin(); it != removed_paths.end(); ++it) {
            if (path_equals_prefix(*it, prefix)) {
                remove_gate = true;
                break;
            }
        }

        if (!remove_gate) {
            if (gate->get_type() == BLOCK_OPERATION && subtree_contains_removed_path(removed_paths, prefix)) {
                Gates_block* cloned_block = clone_without_removed_paths(
                    static_cast<Gates_block*>(gate), removed_paths, prefix);
                ret->add_gate(cloned_block);
            } else {
                ret->add_gate(gate->clone());
            }
        }

        prefix.pop_back();
    }

    return ret;
}

static Gates_block* clone_without_removed_paths(Gates_block* block,
                                                const std::vector<OSRGatePath>& all_paths,
                                                const std::vector<int>& removed_ids) {
    std::set<OSRGatePath> removed_paths;
    for (size_t idx = 0; idx < removed_ids.size(); ++idx) {
        removed_paths.insert(all_paths[removed_ids[idx]]);
    }

    std::vector<int> prefix;
    return clone_without_removed_paths(block, removed_paths, prefix);
}

static Gates_block* clone_with_rewired_gate_path(Gates_block* block,
                                                 const OSRGatePath& path,
                                                 int depth,
                                                 int new_target,
                                                 int new_control) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (depth < static_cast<int>(path.indices.size()) &&
            idx == path.indices[depth]) {
            if (depth == static_cast<int>(path.indices.size()) - 1) {
                Gate* cloned_gate = gate->clone();
                if (!rewire_two_qubit_gate(cloned_gate, new_target, new_control)) {
                    delete cloned_gate;
                    delete ret;
                    return NULL;
                }
                ret->add_gate(cloned_gate);
            } else {
                if (gate->get_type() != BLOCK_OPERATION) {
                    delete ret;
                    return NULL;
                }

                Gates_block* rewired_block = clone_with_rewired_gate_path(
                    static_cast<Gates_block*>(gate), path, depth + 1,
                    new_target, new_control);
                if (rewired_block == NULL) {
                    delete ret;
                    return NULL;
                }
                ret->add_gate(rewired_block);
            }
        } else {
            ret->add_gate(gate->clone());
        }
    }

    return ret;
}

static Gates_block* clone_with_rewired_gate_path(Gates_block* block,
                                                 const OSRGatePath& path,
                                                 int new_target,
                                                 int new_control) {
    return clone_with_rewired_gate_path(block, path, 0, new_target, new_control);
}

static Gates_block* clone_with_swapped_sibling_gates(Gates_block* block,
                                                     const std::vector<int>& parent_path,
                                                     int depth,
                                                     int first_idx,
                                                     int second_idx) {
    Gates_block* ret = new Gates_block(block->get_qbit_num());

    if (depth == static_cast<int>(parent_path.size())) {
        for (int idx = 0; idx < block->get_gate_num(); ++idx) {
            int source_idx = idx;
            if (idx == first_idx) {
                source_idx = second_idx;
            } else if (idx == second_idx) {
                source_idx = first_idx;
            }
            Gate* source_gate = block->get_gate(source_idx);
            if (source_gate != NULL) {
                ret->add_gate(source_gate->clone());
            }
        }
        return ret;
    }

    int selected_idx = parent_path[depth];
    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (idx == selected_idx) {
            if (gate->get_type() != BLOCK_OPERATION) {
                delete ret;
                return NULL;
            }
            Gates_block* swapped_block = clone_with_swapped_sibling_gates(
                static_cast<Gates_block*>(gate), parent_path, depth + 1,
                first_idx, second_idx);
            if (swapped_block == NULL) {
                delete ret;
                return NULL;
            }
            ret->add_gate(swapped_block);
        } else {
            ret->add_gate(gate->clone());
        }
    }

    return ret;
}

static Gates_block* clone_with_swapped_sibling_gates(Gates_block* block,
                                                     const OSRGatePath& first_path,
                                                     const OSRGatePath& second_path) {
    if (first_path.indices.size() != second_path.indices.size() ||
        first_path.indices.empty()) {
        return NULL;
    }

    std::vector<int> first_parent(
        first_path.indices.begin(), first_path.indices.end() - 1);
    std::vector<int> second_parent(
        second_path.indices.begin(), second_path.indices.end() - 1);
    if (first_parent != second_parent) {
        return NULL;
    }

    int first_idx = first_path.indices.back();
    int second_idx = second_path.indices.back();
    if (first_idx == second_idx) {
        return NULL;
    }
    if (second_idx < first_idx) {
        std::swap(first_idx, second_idx);
    }

    return clone_with_swapped_sibling_gates(
        block, first_parent, 0, first_idx, second_idx);
}

static bool parameter_interval_for_path(Gates_block* block,
                                        const OSRGatePath& path,
                                        int depth,
                                        int offset,
                                        int& start,
                                        int& length) {
    if (block == NULL || depth >= static_cast<int>(path.indices.size())) {
        return false;
    }

    int gate_idx = path.indices[depth];
    Gate* gate = block->get_gate(gate_idx);
    if (gate == NULL) {
        return false;
    }

    int gate_offset = offset + gate->get_parameter_start_idx();
    if (depth == static_cast<int>(path.indices.size()) - 1) {
        start = gate_offset;
        length = gate->get_parameter_num();
        return true;
    }

    if (gate->get_type() != BLOCK_OPERATION) {
        return false;
    }

    return parameter_interval_for_path(
        static_cast<Gates_block*>(gate), path, depth + 1, gate_offset, start, length);
}

static Matrix_real reduced_parameters_without_paths(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& removed_paths,
    const Matrix_real& original_parameters) {
    if (original_parameters.size() == 0 ||
        original_parameters.size() != original_gate_structure->get_parameter_num()) {
        return Matrix_real(0, 0);
    }

    std::vector<std::pair<int, int>> intervals;
    intervals.reserve(removed_paths.size());
    for (size_t idx = 0; idx < removed_paths.size(); ++idx) {
        int start = 0;
        int length = 0;
        if (parameter_interval_for_path(
                original_gate_structure, removed_paths[idx], 0, 0, start, length) &&
            length > 0) {
            intervals.push_back(std::make_pair(start, start + length));
        }
    }

    if (intervals.empty()) {
        return original_parameters.copy();
    }

    std::sort(intervals.begin(), intervals.end());
    std::vector<std::pair<int, int>> merged;
    for (size_t idx = 0; idx < intervals.size(); ++idx) {
        if (merged.empty() || intervals[idx].first > merged.back().second) {
            merged.push_back(intervals[idx]);
        } else {
            merged.back().second = std::max(merged.back().second, intervals[idx].second);
        }
    }

    int removed_parameter_num = 0;
    for (size_t idx = 0; idx < merged.size(); ++idx) {
        removed_parameter_num += merged[idx].second - merged[idx].first;
    }

    Matrix_real reduced_parameters(1, original_parameters.size() - removed_parameter_num);
    int src = 0;
    int dst = 0;
    for (size_t idx = 0; idx < merged.size(); ++idx) {
        int keep_num = merged[idx].first - src;
        if (keep_num > 0) {
            std::memcpy(reduced_parameters.get_data() + dst,
                        original_parameters.get_data() + src,
                        keep_num * sizeof(double));
            dst += keep_num;
        }
        src = merged[idx].second;
    }

    if (src < original_parameters.size()) {
        int keep_num = original_parameters.size() - src;
        std::memcpy(reduced_parameters.get_data() + dst,
                    original_parameters.get_data() + src,
                    keep_num * sizeof(double));
    }

    return reduced_parameters;
}

static Matrix_real reduced_parameters_without_removed_paths(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& all_paths,
    const std::vector<int>& removed_ids,
    const Matrix_real& original_parameters) {
    std::vector<OSRGatePath> removed_paths;
    removed_paths.reserve(removed_ids.size());
    for (size_t idx = 0; idx < removed_ids.size(); ++idx) {
        removed_paths.push_back(all_paths[removed_ids[idx]]);
    }
    return reduced_parameters_without_paths(
        original_gate_structure, removed_paths, original_parameters);
}

static void add_topology_edge(std::set<std::pair<int, int>>& edges, int q0, int q1) {
    if (q0 == q1) {
        return;
    }
    if (q1 < q0) {
        std::swap(q0, q1);
    }
    edges.insert(std::make_pair(q0, q1));
}

static void collect_topology_edges(Gates_block* block, std::set<std::pair<int, int>>& edges) {
    if (block == NULL) {
        return;
    }

    for (int idx = 0; idx < block->get_gate_num(); ++idx) {
        Gate* gate = block->get_gate(idx);
        if (gate == NULL) {
            continue;
        }

        if (gate->get_type() == BLOCK_OPERATION) {
            collect_topology_edges(static_cast<Gates_block*>(gate), edges);
            continue;
        }

        if (!is_entangling_gate(gate)) {
            continue;
        }

        std::vector<int> involved = gate->get_involved_qubits();
        for (size_t q0_idx = 0; q0_idx < involved.size(); ++q0_idx) {
            for (size_t q1_idx = q0_idx + 1; q1_idx < involved.size(); ++q1_idx) {
                add_topology_edge(edges, involved[q0_idx], involved[q1_idx]);
            }
        }
    }
}

static std::vector<matrix_base<int>> topology_from_gate_structure(Gates_block* gate_structure, int qbit_num) {
    std::set<std::pair<int, int>> edges;
    collect_topology_edges(gate_structure, edges);

    if (edges.empty() && qbit_num > 1) {
        for (int q0 = 0; q0 < qbit_num; ++q0) {
            for (int q1 = q0 + 1; q1 < qbit_num; ++q1) {
                edges.insert(std::make_pair(q0, q1));
            }
        }
    }

    std::vector<matrix_base<int>> topology;
    topology.reserve(edges.size());
    for (std::set<std::pair<int, int>>::const_iterator it = edges.begin(); it != edges.end(); ++it) {
        matrix_base<int> edge(2, 1);
        edge[0] = it->first;
        edge[1] = it->second;
        topology.push_back(edge);
    }
    return topology;
}

static std::vector<std::pair<int, int>> topology_pairs_from_matrices(
    const std::vector<matrix_base<int>>& topology) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(topology.size());
    for (size_t idx = 0; idx < topology.size(); ++idx) {
        int q0 = topology[idx][0];
        int q1 = topology[idx][1];
        if (q0 == q1) {
            continue;
        }
        if (q1 < q0) {
            std::swap(q0, q1);
        }
        std::pair<int, int> edge(q0, q1);
        if (std::find(pairs.begin(), pairs.end(), edge) == pairs.end()) {
            pairs.push_back(edge);
        }
    }
    return pairs;
}

static std::vector<std::pair<int, int>> complete_topology_pairs(int qbit_num) {
    std::vector<std::pair<int, int>> pairs;
    for (int q0 = 0; q0 < qbit_num; ++q0) {
        for (int q1 = q0 + 1; q1 < qbit_num; ++q1) {
            pairs.push_back(std::make_pair(q0, q1));
        }
    }
    return pairs;
}

static Gates_block* construct_cnot_skeleton_gate_structure(
    int qbit_num,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<int>& sequence) {
    Gates_block* gate_structure = new Gates_block(qbit_num);

    for (size_t idx = 0; idx < sequence.size(); ++idx) {
        int edge_idx = sequence[idx];
        if (edge_idx < 0 || edge_idx >= static_cast<int>(edges.size())) {
            delete gate_structure;
            return NULL;
        }

        Gates_block* layer = new Gates_block(qbit_num);
        int target = edges[edge_idx].first;
        int control = edges[edge_idx].second;
        layer->add_u3(target);
        layer->add_u3(control);
        layer->add_cnot(target, control);
        gate_structure->add_gate(layer);
    }

    Gates_block* final_layer = new Gates_block(qbit_num);
    for (int qbit = 0; qbit < qbit_num; ++qbit) {
        final_layer->add_u3(qbit);
    }
    gate_structure->add_gate(final_layer);

    return gate_structure;
}

static Matrix_real excise_skeleton_gate_params(
    const Matrix_real& parent_params,
    int parent_cnot_num,
    int removed_position,
    int qbit_num) {
    int child_cnot_num = parent_cnot_num - 1;
    int child_param_num = child_cnot_num * 6 + 3 * qbit_num;
    int parent_gate_param_num = parent_cnot_num * 6;
    Matrix_real child_params(1, child_param_num);

    int dst = 0;
    for (int idx = 0; idx < removed_position * 6; ++idx) {
        child_params[dst++] = parent_params[idx];
    }
    for (int idx = (removed_position + 1) * 6; idx < parent_gate_param_num; ++idx) {
        child_params[dst++] = parent_params[idx];
    }
    for (int idx = 0; idx < 3 * qbit_num; ++idx) {
        child_params[dst++] = parent_params[parent_gate_param_num + idx];
    }

    return child_params;
}

static Matrix_real excise_skeleton_gate_params_multi(
    const Matrix_real& parent_params,
    int parent_cnot_num,
    const std::vector<int>& removed_positions,
    int qbit_num) {
    int n_remove = static_cast<int>(removed_positions.size());
    int child_cnot_num = parent_cnot_num - n_remove;
    int child_param_num = child_cnot_num * 6 + 3 * qbit_num;
    int parent_gate_param_num = parent_cnot_num * 6;

    std::vector<bool> remove_mask(parent_cnot_num, false);
    for (int pos : removed_positions) remove_mask[pos] = true;

    Matrix_real child_params(1, child_param_num);
    int dst = 0;
    for (int g = 0; g < parent_cnot_num; ++g) {
        if (remove_mask[g]) continue;
        for (int p = 0; p < 6; ++p) child_params[dst++] = parent_params[g * 6 + p];
    }
    for (int idx = 0; idx < 3 * qbit_num; ++idx) {
        child_params[dst++] = parent_params[parent_gate_param_num + idx];
    }
    return child_params;
}

static bool extract_cnot_skeleton_sequence(
    Gates_block* gate_structure,
    const std::vector<OSRGatePath>& paths,
    const std::vector<std::pair<int, int>>& edges,
    std::vector<int>& sequence) {
    sequence.clear();
    sequence.reserve(paths.size());

    for (size_t path_idx = 0; path_idx < paths.size(); ++path_idx) {
        Gate* gate = gate_at_path(gate_structure, paths[path_idx]);
        if (gate == NULL || gate->get_type() != CNOT_OPERATION) {
            return false;
        }

        int q0 = 0;
        int q1 = 0;
        if (!get_two_qubit_endpoint_pair(gate, q0, q1)) {
            return false;
        }

        std::pair<int, int> edge(q0, q1);
        if (edge.second < edge.first) {
            std::swap(edge.first, edge.second);
        }

        std::vector<std::pair<int, int>>::const_iterator it =
            std::find(edges.begin(), edges.end(), edge);
        if (it == edges.end()) {
            return false;
        }
        sequence.push_back(static_cast<int>(it - edges.begin()));
    }

    return true;
}

static int64_t limited_integer_power(int base, int exponent, int64_t limit) {
    int64_t value = 1;
    for (int idx = 0; idx < exponent; ++idx) {
        if (base <= 0 || value > limit / base) {
            return limit + 1;
        }
        value *= base;
    }
    return value;
}

static std::vector<CompressionCandidate> generate_cnot_skeleton_candidates(
    int qbit_num,
    int original_entangling_gate_num,
    const std::vector<std::pair<int, int>>& edges,
    const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    std::vector<CompressionCandidate> candidates;
    if (!options.enable_skeleton_search || edges.empty() ||
        options.skeleton_max_candidates <= 0 || original_entangling_gate_num <= 0) {
        return candidates;
    }

    int min_target;
    int max_target;
    if (options.skeleton_target_cnots >= 0) {
        min_target = options.skeleton_target_cnots;
        max_target = options.skeleton_target_cnots;
    } else {
        min_target = std::max(1, qbit_num - 1);
        max_target = original_entangling_gate_num - 1;
    }
    if (max_target < min_target || max_target < 0 ||
        max_target >= original_entangling_gate_num) {
        return candidates;
    }

    int edge_num = static_cast<int>(edges.size());
    std::set<std::string> seen;

    for (int target_depth = min_target; target_depth <= max_target; ++target_depth) {
        int remaining_budget =
            options.skeleton_max_candidates - static_cast<int>(candidates.size());
        if (remaining_budget <= 0) {
            break;
        }

        int64_t combination_num = limited_integer_power(
            edge_num, target_depth, static_cast<int64_t>(remaining_budget));
        if (combination_num > remaining_budget) {
            continue;
        }

        for (int64_t state = 0; state < combination_num; ++state) {
            int64_t value = state;
            std::vector<int> sequence(target_depth, 0);
            for (int depth = target_depth - 1; depth >= 0; --depth) {
                sequence[depth] = static_cast<int>(value % edge_num);
                value /= edge_num;
            }

            std::shared_ptr<Gates_block> gate_structure(
                construct_cnot_skeleton_gate_structure(qbit_num, edges, sequence));
            if (!gate_structure) {
                continue;
            }

            std::string key = gate_structure_signature(gate_structure.get());
            if (!seen.insert(key).second) {
                continue;
            }

            CompressionCandidate candidate;
            candidate.entangling_gate_num = target_depth;
            candidate.gate_structure = gate_structure;
            candidate.key = key;
            candidate.initial_parameters = Matrix_real(0, 0);
            candidate.score.min_remaining_cnots = std::numeric_limits<int>::max();
            candidate.score.kappa = std::numeric_limits<double>::infinity();
            candidate.score.residual = std::numeric_limits<double>::infinity();
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

// Generate replacement skeletons that shorten 2q (or 3q) spans in the base
// edge sequence whose CNOT count exceeds the OSR-bound allocation.
//
// Strategy: walk the linearized CNOT edge sequence of base_structure. Find
// maximal contiguous runs whose union of carrier qubits has size <= the
// configured limit (2 or 3). If a run's length exceeds the OSR-bound target
// for that span, generate variant skeletons by replacing the run with shorter
// length-T sequences over the carrier's edges and emit them as candidates.
// Candidates carry no warm-start params and rely on the existing validation
// pipeline (BFGS with random init + retries) to either confirm or reject.
static std::vector<CompressionCandidate> generate_local_resynth_candidates(
    Gates_block* base_structure,
    const std::vector<std::pair<int, int>>& topology_edges,
    const N_Qubit_Decomposition_OSR_Compression_Score& score,
    int qbit_num,
    const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    std::vector<CompressionCandidate> candidates;
    if (!options.enable_local_resynth || base_structure == NULL ||
        topology_edges.empty()) {
        return candidates;
    }

    // Linearize the base into an edge-index sequence. If the structure isn't
    // a clean U3+CNOT skeleton extraction will fail; bail out cleanly.
    std::vector<OSRGatePath> paths = collect_entangling_gate_paths(base_structure);
    if (paths.empty()) {
        return candidates;
    }
    std::vector<int> base_seq;
    if (!extract_cnot_skeleton_sequence(base_structure, paths, topology_edges, base_seq)) {
        return candidates;
    }
    int original_cnot_num = static_cast<int>(base_seq.size());
    if (original_cnot_num <= 0) {
        return candidates;
    }

    int max_carrier = std::max(options.local_resynth_max_span_qubits, 2);
    int budget = options.local_resynth_max_candidates;
    if (budget <= 0) {
        return candidates;
    }

    std::set<std::string> seen;

    // Walk all spans [i..j] (inclusive) and check the carrier-size constraint.
    for (int i = 0; i < original_cnot_num; ++i) {
        std::set<int> carrier;
        carrier.insert(topology_edges[base_seq[i]].first);
        carrier.insert(topology_edges[base_seq[i]].second);
        if (static_cast<int>(carrier.size()) > max_carrier) {
            continue;
        }

        // Indices of edges within the carrier (used to enumerate replacements).
        std::vector<int> carrier_edge_indices;
        for (size_t e = 0; e < topology_edges.size(); ++e) {
            int a = topology_edges[e].first;
            int b = topology_edges[e].second;
            if (carrier.count(a) > 0 && carrier.count(b) > 0) {
                carrier_edge_indices.push_back(static_cast<int>(e));
            }
        }
        if (carrier_edge_indices.empty()) {
            continue;
        }

        for (int j = i; j < original_cnot_num; ++j) {
            int a = topology_edges[base_seq[j]].first;
            int b = topology_edges[base_seq[j]].second;
            std::set<int> new_carrier = carrier;
            new_carrier.insert(a);
            new_carrier.insert(b);
            if (static_cast<int>(new_carrier.size()) > max_carrier) {
                break;
            }
            carrier = new_carrier;

            int span_len = j - i + 1;
            if (span_len < 2) {
                continue;
            }

            // Recompute carrier_edge_indices with the (possibly grown) carrier.
            std::vector<int> span_carrier_edges;
            for (size_t e = 0; e < topology_edges.size(); ++e) {
                int ea = topology_edges[e].first;
                int eb = topology_edges[e].second;
                if (carrier.count(ea) > 0 && carrier.count(eb) > 0) {
                    span_carrier_edges.push_back(static_cast<int>(e));
                }
            }
            if (span_carrier_edges.empty()) {
                continue;
            }

            // Derive a target length from the OSR bound: sum edge_counts over
            // edges inside the carrier, capped by the per-class depth limit.
            int bound_sum = 0;
            for (int e_idx : span_carrier_edges) {
                if (e_idx >= 0 && e_idx < static_cast<int>(score.edge_counts.size())) {
                    bound_sum += score.edge_counts[e_idx];
                }
            }
            int hard_cap = (static_cast<int>(carrier.size()) <= 2)
                ? options.local_resynth_max_depth_2q
                : options.local_resynth_max_depth_3q;
            int target_len = std::min(bound_sum, hard_cap);
            target_len = std::max(target_len, 0);
            if (target_len >= span_len) {
                continue;
            }

            // Build replacement sequences of length target_len over the carrier's edges.
            int n_carrier = static_cast<int>(span_carrier_edges.size());
            int64_t combinations = limited_integer_power(
                n_carrier, target_len, static_cast<int64_t>(budget));
            if (combinations > budget) {
                // Skip this span if enumeration would blow the budget; a deeper/narrower
                // span may still fit.
                continue;
            }

            for (int64_t state = 0; state < combinations; ++state) {
                int64_t value = state;
                std::vector<int> replacement(target_len, 0);
                for (int d = target_len - 1; d >= 0; --d) {
                    replacement[d] = span_carrier_edges[value % n_carrier];
                    value /= n_carrier;
                }

                std::vector<int> new_seq;
                new_seq.reserve(original_cnot_num - span_len + target_len);
                new_seq.insert(new_seq.end(), base_seq.begin(), base_seq.begin() + i);
                new_seq.insert(new_seq.end(), replacement.begin(), replacement.end());
                new_seq.insert(new_seq.end(), base_seq.begin() + j + 1, base_seq.end());

                std::shared_ptr<Gates_block> gate_structure(
                    construct_cnot_skeleton_gate_structure(qbit_num, topology_edges, new_seq));
                if (!gate_structure) {
                    continue;
                }

                std::string key = gate_structure_signature(gate_structure.get());
                if (!seen.insert(key).second) {
                    continue;
                }

                CompressionCandidate candidate;
                candidate.entangling_gate_num = static_cast<int>(new_seq.size());
                candidate.gate_structure = gate_structure;
                candidate.key = key;
                candidate.initial_parameters = Matrix_real(0, 0);
                candidate.score.min_remaining_cnots = std::numeric_limits<int>::max();
                candidate.score.kappa = std::numeric_limits<double>::infinity();
                candidate.score.residual = std::numeric_limits<double>::infinity();
                candidates.push_back(candidate);

                if (static_cast<int>(candidates.size()) >= budget) {
                    return candidates;
                }
            }
        }
    }

    return candidates;
}

static double residual_sum(const std::vector<std::pair<int, double>>& cut_bounds) {
    return std::accumulate(cut_bounds.begin(), cut_bounds.end(), 0.0,
        [&cut_bounds](double acc, const std::pair<int, double>& item) {
            return acc + item.first * cut_bounds.size() + item.second;
        });
}

static bool score_less(const N_Qubit_Decomposition_OSR_Compression_Score& lhs,
                       const N_Qubit_Decomposition_OSR_Compression_Score& rhs) {
    if (lhs.min_remaining_cnots != rhs.min_remaining_cnots) {
        return lhs.min_remaining_cnots < rhs.min_remaining_cnots;
    }
    if (lhs.kappa != rhs.kappa) {
        return lhs.kappa < rhs.kappa;
    }
    return lhs.residual < rhs.residual;
}

static bool beam_candidate_less(const CompressionCandidate& lhs,
                                const CompressionCandidate& rhs) {
    if (score_less(lhs.score, rhs.score)) {
        return true;
    }
    if (score_less(rhs.score, lhs.score)) {
        return false;
    }
    return lhs.key < rhs.key;
}

static bool final_candidate_less(const CompressionCandidate& lhs,
                                 const CompressionCandidate& rhs) {
    if (lhs.entangling_gate_num != rhs.entangling_gate_num) {
        return lhs.entangling_gate_num < rhs.entangling_gate_num;
    }
    return beam_candidate_less(lhs, rhs);
}

static std::string compression_candidate_key(const CompressionCandidate& candidate) {
    if (!candidate.key.empty()) {
        return candidate.key;
    }

    std::stringstream sstream;
    sstream << "removed:";
    for (size_t idx = 0; idx < candidate.removed_ids.size(); ++idx) {
        if (idx > 0) {
            sstream << ",";
        }
        sstream << candidate.removed_ids[idx];
    }
    return sstream.str();
}

static void sort_unique_candidates(std::vector<CompressionCandidate>& candidates, bool final_sort) {
    std::sort(candidates.begin(), candidates.end(),
              final_sort ? final_candidate_less : beam_candidate_less);

    std::set<std::string> seen;
    std::vector<CompressionCandidate> unique_candidates;
    unique_candidates.reserve(candidates.size());
    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        std::string key = compression_candidate_key(candidates[idx]);
        if (seen.insert(key).second) {
            unique_candidates.push_back(candidates[idx]);
        }
    }
    candidates.swap(unique_candidates);
}

static Gates_block* clone_gate_structure_for_candidate(
    Gates_block* original_gate_structure,
    const std::vector<OSRGatePath>& original_paths,
    const CompressionCandidate& candidate) {
    if (candidate.gate_structure) {
        return candidate.gate_structure->clone();
    }
    return clone_without_removed_paths(
        original_gate_structure, original_paths, candidate.removed_ids);
}

static bool edge_shares_endpoint(const std::pair<int, int>& edge, int q0, int q1) {
    return edge.first == q0 || edge.first == q1 ||
           edge.second == q0 || edge.second == q1;
}

static bool same_undirected_edge(const std::pair<int, int>& edge, int q0, int q1) {
    int a = q0;
    int b = q1;
    if (b < a) {
        std::swap(a, b);
    }
    return edge.first == a && edge.second == b;
}

static bool same_parent_and_adjacent(const OSRGatePath& lhs,
                                     const OSRGatePath& rhs) {
    if (lhs.indices.size() != rhs.indices.size() || lhs.indices.empty()) {
        return false;
    }

    for (size_t idx = 0; idx + 1 < lhs.indices.size(); ++idx) {
        if (lhs.indices[idx] != rhs.indices[idx]) {
            return false;
        }
    }

    return std::abs(lhs.indices.back() - rhs.indices.back()) == 1;
}

static void append_candidate_if_new(std::vector<CompressionCandidate>& out,
                                    std::set<std::string>& seen,
                                    CompressionCandidate& candidate) {
    if (!candidate.gate_structure) {
        return;
    }

    candidate.key = gate_structure_signature(candidate.gate_structure.get());
    if (seen.insert(candidate.key).second) {
        candidate.entangling_gate_num =
            static_cast<int>(collect_entangling_gate_paths(candidate.gate_structure.get()).size());
        out.push_back(candidate);
    }
}

static std::vector<CompressionCandidate> generate_local_mutation_candidates(
    Gates_block* base_structure,
    const Matrix_real& base_parameters,
    const CompressionCandidate& parent,
    const std::vector<std::pair<int, int>>& mutation_edges,
    const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    std::vector<CompressionCandidate> candidates;
    if (!options.enable_mutations || options.mutation_candidates <= 0 ||
        base_structure == NULL) {
        return candidates;
    }

    std::vector<OSRGatePath> paths = collect_entangling_gate_paths(base_structure);
    if (paths.empty()) {
        return candidates;
    }

    std::set<std::string> seen;
    seen.insert(gate_structure_signature(base_structure));

    for (size_t idx = 0; idx + 1 < paths.size() &&
                         static_cast<int>(candidates.size()) < options.mutation_candidates; ++idx) {
        const OSRGatePath& lhs_path = paths[idx];
        const OSRGatePath& rhs_path = paths[idx + 1];
        if (!same_parent_and_adjacent(lhs_path, rhs_path)) {
            continue;
        }

        Gate* lhs_gate = gate_at_path(base_structure, lhs_path);
        Gate* rhs_gate = gate_at_path(base_structure, rhs_path);
        if (!gate_endpoint_sets_are_disjoint(lhs_gate, rhs_gate)) {
            continue;
        }

        std::shared_ptr<Gates_block> swapped(
            clone_with_swapped_sibling_gates(base_structure, lhs_path, rhs_path));
        if (!swapped) {
            continue;
        }

        CompressionCandidate child = parent;
        child.gate_structure = swapped;
        if (base_parameters.size() == base_structure->get_parameter_num() &&
            swapped->get_parameter_num() == base_parameters.size()) {
            child.initial_parameters = base_parameters.copy();
        } else {
            child.initial_parameters = Matrix_real(0, 0);
        }
        append_candidate_if_new(candidates, seen, child);
    }

    for (size_t path_idx = 0; path_idx < paths.size() &&
                              static_cast<int>(candidates.size()) < options.mutation_candidates; ++path_idx) {
        Gate* gate = gate_at_path(base_structure, paths[path_idx]);
        if (gate == NULL) {
            continue;
        }

        int old_q0 = 0;
        int old_q1 = 0;
        if (!get_two_qubit_endpoint_pair(gate, old_q0, old_q1)) {
            continue;
        }

        bool directional = gate_type_is_directional(gate->get_type()) &&
            (!gate->get_control_qbits().empty() || gate->get_control_qbit() >= 0);

        for (int pass = 0; pass < 2 &&
                           static_cast<int>(candidates.size()) < options.mutation_candidates; ++pass) {
            for (size_t edge_idx = 0; edge_idx < mutation_edges.size() &&
                                      static_cast<int>(candidates.size()) < options.mutation_candidates; ++edge_idx) {
                const std::pair<int, int>& edge = mutation_edges[edge_idx];
                if (same_undirected_edge(edge, old_q0, old_q1)) {
                    continue;
                }

                bool shares_endpoint = edge_shares_endpoint(edge, old_q0, old_q1);
                if ((pass == 0 && !shares_endpoint) || (pass == 1 && shares_endpoint)) {
                    continue;
                }

                int orientation_count = directional ? 2 : 1;
                for (int orientation = 0;
                     orientation < orientation_count &&
                     static_cast<int>(candidates.size()) < options.mutation_candidates;
                     ++orientation) {
                    int new_target = (orientation == 0) ? edge.first : edge.second;
                    int new_control = (orientation == 0) ? edge.second : edge.first;

                    std::shared_ptr<Gates_block> rewired(
                        clone_with_rewired_gate_path(
                            base_structure, paths[path_idx], new_target, new_control));
                    if (!rewired) {
                        continue;
                    }

                    CompressionCandidate child = parent;
                    child.gate_structure = rewired;
                    if (base_parameters.size() == base_structure->get_parameter_num() &&
                        rewired->get_parameter_num() == base_parameters.size()) {
                        child.initial_parameters = base_parameters.copy();
                    } else {
                        child.initial_parameters = Matrix_real(0, 0);
                    }
                    append_candidate_if_new(candidates, seen, child);
                }
            }
        }
    }

    return candidates;
}

static bool candidate_is_osr_admissible(const CompressionCandidate& candidate,
                                        const N_Qubit_Decomposition_OSR_Compression_Options& options) {
    return candidate.score.min_remaining_cnots <= options.osr_bound_limit;
}

} // namespace

N_Qubit_Decomposition_OSR_Compression_Result::N_Qubit_Decomposition_OSR_Compression_Result()
    : current_minimum(std::numeric_limits<double>::infinity()),
      original_entangling_gate_num(0),
      compressed_entangling_gate_num(0),
      validated(false),
      reached_tolerance(false),
      decomposition_error(std::numeric_limits<double>::infinity()) {}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression()
    : N_Qubit_Decomposition_custom() {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression(
    Matrix Umtx_in,
    int qbit_num_in,
    std::map<std::string, Config_Element>& config,
    int accelerator_num)
    : N_Qubit_Decomposition_custom(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::N_Qubit_Decomposition_OSR_Compression(
    Matrix Umtx_in,
    int qbit_num_in,
    std::vector<matrix_base<int>> topology_in,
    std::map<std::string, Config_Element>& config,
    int accelerator_num)
    : N_Qubit_Decomposition_custom(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num),
      topology(std::move(topology_in)) {
    name = "OSR_Compression";
}

N_Qubit_Decomposition_OSR_Compression::~N_Qubit_Decomposition_OSR_Compression() {}

void N_Qubit_Decomposition_OSR_Compression::start_decomposition() {
    std::unique_ptr<Gates_block> original_gate_structure(clone());
    Matrix_real original_parameters = optimized_parameters_mtx.size() > 0
        ? optimized_parameters_mtx.copy()
        : Matrix_real(0, 0);

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    N_Qubit_Decomposition_OSR_Compression_Result result =
        compress_gate_structure(original_gate_structure.get());

    if (!result.reached_tolerance &&
        result.compressed_entangling_gate_num >= result.original_entangling_gate_num &&
        original_parameters.size() == original_gate_structure->get_parameter_num()) {
        double original_cost = optimization_problem(original_parameters);
        if (original_cost < optimization_tolerance_loc ||
            original_cost < result.current_minimum) {
            result.gate_structure.reset(original_gate_structure->clone());
            result.optimized_parameters = original_parameters.copy();
            result.current_minimum = original_cost;
            result.decomposition_error = original_cost;
            result.compressed_entangling_gate_num =
                result.original_entangling_gate_num;
            result.removed_gate_paths.clear();
            result.validated = true;
            result.reached_tolerance = original_cost < optimization_tolerance_loc;
        }
    }

    release_gates();
    combine(result.gate_structure.get());

    if (result.validated && result.optimized_parameters.size() == get_parameter_num()) {
        optimized_parameters_mtx = result.optimized_parameters.copy();
        current_minimum = result.current_minimum;
        decomposition_error = result.decomposition_error;
    } else {
        if (optimized_parameters_mtx.size() != get_parameter_num()) {
            optimized_parameters_mtx = Matrix_real(0, 0);
        }
        N_Qubit_Decomposition_custom::start_decomposition();
    }
}

N_Qubit_Decomposition_OSR_Compression_Options
N_Qubit_Decomposition_OSR_Compression::get_osr_compression_options() {
    N_Qubit_Decomposition_OSR_Compression_Options options;

    long long int_value;
    double double_value;

    if (config.count("osr_compression_beam") > 0) {
        config["osr_compression_beam"].get_property(int_value);
        options.beam_width = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_max_removed") > 0) {
        config["osr_compression_max_removed"].get_property(int_value);
        options.max_removed_gates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_bound_limit") > 0) {
        config["osr_compression_bound_limit"].get_property(int_value);
        options.osr_bound_limit = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_validation_trials") > 0) {
        config["osr_compression_validation_trials"].get_property(int_value);
        options.validation_trials = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_validate") > 0) {
        config["osr_compression_validate"].get_property(int_value);
        options.validate_final = int_value != 0;
    }
    if (config.count("osr_compression_osr_tolerance") > 0) {
        config["osr_compression_osr_tolerance"].get_property(double_value);
        options.osr_tolerance = double_value;
    }
    if (config.count("osr_compression_enable_mutations") > 0) {
        config["osr_compression_enable_mutations"].get_property(int_value);
        options.enable_mutations = int_value != 0;
    }
    if (config.count("osr_compression_mutation_rounds") > 0) {
        config["osr_compression_mutation_rounds"].get_property(int_value);
        options.mutation_rounds = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_mutation_candidates") > 0) {
        config["osr_compression_mutation_candidates"].get_property(int_value);
        options.mutation_candidates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_mutate_full_topology") > 0) {
        config["osr_compression_mutate_full_topology"].get_property(int_value);
        options.mutate_full_topology = int_value != 0;
    }
    if (config.count("osr_compression_enable_skeleton_search") > 0) {
        config["osr_compression_enable_skeleton_search"].get_property(int_value);
        options.enable_skeleton_search = int_value != 0;
    }
    if (config.count("osr_compression_skeleton_target_cnots") > 0) {
        config["osr_compression_skeleton_target_cnots"].get_property(int_value);
        options.skeleton_target_cnots = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_skeleton_max_candidates") > 0) {
        config["osr_compression_skeleton_max_candidates"].get_property(int_value);
        options.skeleton_max_candidates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_enable_triple_removal") > 0) {
        config["osr_compression_enable_triple_removal"].get_property(int_value);
        options.enable_triple_removal = int_value != 0;
    }
    if (config.count("osr_compression_triple_top_k") > 0) {
        config["osr_compression_triple_top_k"].get_property(int_value);
        options.triple_top_k = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_max_pairs_per_parent") > 0) {
        config["osr_compression_max_pairs_per_parent"].get_property(int_value);
        options.max_pairs_per_parent = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_max_triples_per_parent") > 0) {
        config["osr_compression_max_triples_per_parent"].get_property(int_value);
        options.max_triples_per_parent = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_handoff_to_additive") > 0) {
        config["osr_compression_handoff_to_additive"].get_property(int_value);
        options.handoff_to_additive = int_value != 0;
    }
    if (config.count("osr_compression_eval_restarts") > 0) {
        config["osr_compression_eval_restarts"].get_property(int_value);
        options.osr_eval_restarts = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_eval_polish_full") > 0) {
        config["osr_compression_eval_polish_full"].get_property(int_value);
        options.osr_eval_polish_full = int_value != 0;
    }
    if (config.count("osr_compression_eval_polish_iters") > 0) {
        config["osr_compression_eval_polish_iters"].get_property(int_value);
        options.osr_eval_polish_iters = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_enable_local_resynth") > 0) {
        config["osr_compression_enable_local_resynth"].get_property(int_value);
        options.enable_local_resynth = int_value != 0;
    }
    if (config.count("osr_compression_local_resynth_max_span_qubits") > 0) {
        config["osr_compression_local_resynth_max_span_qubits"].get_property(int_value);
        options.local_resynth_max_span_qubits = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_local_resynth_max_depth_2q") > 0) {
        config["osr_compression_local_resynth_max_depth_2q"].get_property(int_value);
        options.local_resynth_max_depth_2q = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_local_resynth_max_depth_3q") > 0) {
        config["osr_compression_local_resynth_max_depth_3q"].get_property(int_value);
        options.local_resynth_max_depth_3q = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_local_resynth_max_candidates") > 0) {
        config["osr_compression_local_resynth_max_candidates"].get_property(int_value);
        options.local_resynth_max_candidates = static_cast<int>(int_value);
    }
    if (config.count("osr_compression_validation_inner_iters") > 0) {
        config["osr_compression_validation_inner_iters"].get_property(int_value);
        options.validation_inner_iters = static_cast<int>(int_value);
    }

    options.osr_eval_restarts = std::max(options.osr_eval_restarts, 1);
    options.osr_eval_polish_iters = std::max(options.osr_eval_polish_iters, 0);
    options.local_resynth_max_span_qubits = std::max(options.local_resynth_max_span_qubits, 2);
    options.local_resynth_max_depth_2q = std::max(options.local_resynth_max_depth_2q, 0);
    options.local_resynth_max_depth_3q = std::max(options.local_resynth_max_depth_3q, 0);
    options.local_resynth_max_candidates = std::max(options.local_resynth_max_candidates, 0);
    options.validation_inner_iters = std::max(options.validation_inner_iters, 0);

    options.beam_width = std::max(options.beam_width, 1);
    options.validation_trials = std::max(options.validation_trials, 1);
    options.osr_bound_limit = std::max(options.osr_bound_limit, 0);
    options.mutation_rounds = std::max(options.mutation_rounds, 0);
    options.mutation_candidates = std::max(options.mutation_candidates, 0);
    options.skeleton_max_candidates = std::max(options.skeleton_max_candidates, 0);
    options.triple_top_k = std::max(options.triple_top_k, 0);
    options.max_pairs_per_parent = std::max(options.max_pairs_per_parent, 0);
    options.max_triples_per_parent = std::max(options.max_triples_per_parent, 0);

    return options;
}

N_Qubit_Decomposition_custom
N_Qubit_Decomposition_OSR_Compression::prepare_custom_optimizer(
    Gates_block* gate_structure_in,
    cost_function_type cost_function_variant) {
    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    N_Qubit_Decomposition_custom cDecomp_custom_random =
        N_Qubit_Decomposition_custom(Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
    cDecomp_custom_random.set_custom_gate_structure(gate_structure_in);
    cDecomp_custom_random.set_optimization_blocks(gate_structure_in->get_gate_num());
    cDecomp_custom_random.set_max_iteration(max_outer_iterations);
#ifndef __DFE__
    cDecomp_custom_random.set_verbose(verbose);
#else
    cDecomp_custom_random.set_verbose(0);
#endif
    cDecomp_custom_random.set_cost_function_variant(cost_function_variant);
    cDecomp_custom_random.set_debugfile("");
    cDecomp_custom_random.set_optimization_tolerance(optimization_tolerance_loc);
    cDecomp_custom_random.set_trace_offset(trace_offset);
    cDecomp_custom_random.set_optimizer(alg);

    if (alg == ADAM || alg == BFGS2) {
        int max_inner_iterations_loc = 10000;
        int param_num_loc = gate_structure_in->get_parameter_num();
        max_inner_iterations_loc = static_cast<int>((double)param_num_loc / 852 * 10000000.0);
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == ADAM_BATCHED) {
        int max_inner_iterations_loc = 2000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
        cDecomp_custom_random.set_random_shift_count_max(5);
    } else if (alg == BFGS) {
        int max_inner_iterations_loc = 10000;
        cDecomp_custom_random.set_max_inner_iterations(max_inner_iterations_loc);
    }

    return cDecomp_custom_random;
}

N_Qubit_Decomposition_OSR_Compression_Score
N_Qubit_Decomposition_OSR_Compression::evaluate_gate_structure_osr(
    Gates_block* gate_structure_in,
    const Matrix_real& initial_parameters,
    MinCnotBoundSolver& osr_bound_solver,
    std::vector<std::vector<int>>& all_cuts) {
    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();
    return evaluate_gate_structure_osr(
        gate_structure_in, initial_parameters, osr_bound_solver, all_cuts,
        options.osr_eval_restarts);
}

N_Qubit_Decomposition_OSR_Compression_Score
N_Qubit_Decomposition_OSR_Compression::evaluate_gate_structure_osr(
    Gates_block* gate_structure_in,
    const Matrix_real& initial_parameters,
    MinCnotBoundSolver& osr_bound_solver,
    std::vector<std::vector<int>>& all_cuts,
    int restarts) {
    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();
    N_Qubit_Decomposition_OSR_Compression_Score best_score;
    best_score.min_remaining_cnots = std::numeric_limits<int>::max();
    best_score.kappa = std::numeric_limits<double>::infinity();
    best_score.residual = std::numeric_limits<double>::infinity();

    if (qbit_num <= 1 || all_cuts.empty()) {
        best_score.min_remaining_cnots = 0;
        best_score.kappa = 0.0;
        best_score.residual = 0.0;
        return best_score;
    }

    int K = std::max(restarts, 1);
    double Fnorm = std::sqrt(static_cast<double>(1 << qbit_num));
    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);

    N_Qubit_Decomposition_OSR_Compression_Options opts_loc = options;

    N_Qubit_Decomposition_custom cDecomp_custom_random =
        prepare_custom_optimizer(gate_structure_in, OSR_ENTANGLEMENT);
    int n_params = cDecomp_custom_random.get_parameter_num();
    std::vector<double> warm_start_parameters(n_params);
    if (initial_parameters.size() == n_params) {
        std::copy(initial_parameters.get_data(),
                  initial_parameters.get_data() + initial_parameters.size(),
                  warm_start_parameters.begin());
    } else if (optimized_parameters_mtx.size() == n_params) {
        std::copy(optimized_parameters_mtx.get_data(),
                  optimized_parameters_mtx.get_data() + optimized_parameters_mtx.size(),
                  warm_start_parameters.begin());
    } else {
        for (int idx = 0; idx < n_params; ++idx) {
            warm_start_parameters[idx] = distrib_real(gen);
        }
    }

    // Optional one-shot full-circuit polish: refine warm_start_parameters at a
    // real local minimum of the standard cost before judging OSR rank, so a
    // bad random init doesn't make a removable gate look essential.
    if (opts_loc.osr_eval_polish_full && n_params > 0) {
        N_Qubit_Decomposition_custom cDecomp_polish =
            prepare_custom_optimizer(gate_structure_in, cost_fnc);
        cDecomp_polish.set_max_inner_iterations(opts_loc.osr_eval_polish_iters);
        cDecomp_polish.set_optimized_parameters(
            warm_start_parameters.data(), n_params);
        cDecomp_polish.start_decomposition();
        Matrix_real polished = cDecomp_polish.get_optimized_parameters();
        if (polished.size() == n_params) {
            std::copy(polished.get_data(), polished.get_data() + n_params,
                      warm_start_parameters.begin());
        }
    }

    // Carry forward the best params from one (cut, rank) iteration to the next
    // as a free warm start (mirrors the original code's implicit behavior).
    std::vector<double> running_parameters = warm_start_parameters;

    for (const std::vector<int>& cut : all_cuts) {
        if (cut.size() != 1) {
            continue;
        }

        int cut_size = static_cast<int>(cut.size());
        int max_rank = 2 * std::min(cut_size, qbit_num - cut_size);
        max_rank = std::max(max_rank, 1);

        for (int rank = max_rank - 1; rank >= 0; --rank) {
            // K-restart loop: pick the lowest-loss optimizer outcome among
            // K trials before reading the OSR rank. Trial 0 inherits the
            // running warm start; trials 1..K-1 fully randomize.
            double best_loss_for_rank = std::numeric_limits<double>::infinity();
            std::vector<double> best_params_for_rank(n_params);
            for (int k_trial = 0; k_trial < K; ++k_trial) {
                std::vector<double> trial_params(n_params);
                if (k_trial == 0) {
                    trial_params = running_parameters;
                } else {
                    for (int idx = 0; idx < n_params; ++idx) {
                        trial_params[idx] = distrib_real(gen);
                    }
                }
                if (n_params > 0) {
                    cDecomp_custom_random.set_optimized_parameters(
                        trial_params.data(), n_params);
                }
                cDecomp_custom_random.set_osr_params({cut}, rank, false);
                cDecomp_custom_random.start_decomposition();

                double trial_loss = cDecomp_custom_random.get_current_minimum();
                if (trial_loss < best_loss_for_rank) {
                    best_loss_for_rank = trial_loss;
                    Matrix_real outp = cDecomp_custom_random.get_optimized_parameters();
                    if (outp.size() == n_params) {
                        std::copy(outp.get_data(), outp.get_data() + n_params,
                                  best_params_for_rank.begin());
                    }
                }
            }

            if (n_params > 0) {
                running_parameters = best_params_for_rank;
                cDecomp_custom_random.set_optimized_parameters(
                    best_params_for_rank.data(), n_params);
            }

            Matrix U = Umtx.copy();
            Matrix_real params = cDecomp_custom_random.get_optimized_parameters();
            cDecomp_custom_random.apply_to(params, U);

            std::vector<std::pair<int, double>> osr_result;
            osr_result.reserve(all_cuts.size());
            int newrank = rank;
            for (const std::vector<int>& eval_cut : all_cuts) {
                osr_result.emplace_back(
                    operator_schmidt_rank(U, qbit_num, eval_cut, Fnorm, options.osr_tolerance));
                if (cut == eval_cut) {
                    newrank = osr_result.back().first;
                }
            }

            double kappa = std::numeric_limits<double>::infinity();
            std::vector<int> edge_counts;
            int min_cnots = osr_bound_solver.solve_min_cnots(osr_result, kappa, edge_counts);

            N_Qubit_Decomposition_OSR_Compression_Score score;
            score.min_remaining_cnots = min_cnots;
            score.kappa = kappa;
            score.residual = residual_sum(osr_result);
            score.edge_counts = edge_counts;
            score.cut_bounds = osr_result;

            if (score_less(score, best_score)) {
                best_score = score;
            }

            if (newrank > rank) {
                break;
            }
            rank = std::min(rank, newrank);
        }
    }

    if (best_score.min_remaining_cnots == std::numeric_limits<int>::max()) {
        std::vector<std::pair<int, double>> osr_result(all_cuts.size(), std::make_pair(0, 0.0));
        double kappa = std::numeric_limits<double>::infinity();
        std::vector<int> edge_counts;
        best_score.min_remaining_cnots = osr_bound_solver.solve_min_cnots(osr_result, kappa, edge_counts);
        best_score.kappa = kappa;
        best_score.residual = 0.0;
        best_score.edge_counts = edge_counts;
        best_score.cut_bounds = osr_result;
    }

    return best_score;
}

void N_Qubit_Decomposition_OSR_Compression::validate_compressed_gate_structure(
    Gates_block* gate_structure_in,
    const Matrix_real& initial_parameters,
    N_Qubit_Decomposition_OSR_Compression_Result& result) {
    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();

    double optimization_tolerance_loc;
    if (config.count("optimization_tolerance") > 0) {
        config["optimization_tolerance"].get_property(optimization_tolerance_loc);
    } else {
        optimization_tolerance_loc = optimization_tolerance;
    }

    result.validated = true;
    result.current_minimum = std::numeric_limits<double>::infinity();
    result.decomposition_error = std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<> distrib_real(0.0, 2 * M_PI);
    for (int iter = 0; iter < options.validation_trials; ++iter) {
        N_Qubit_Decomposition_custom cDecomp_custom_random =
            prepare_custom_optimizer(gate_structure_in, cost_fnc);
        if (options.validation_inner_iters > 0) {
            cDecomp_custom_random.set_max_inner_iterations(options.validation_inner_iters);
        }

        std::vector<double> optimized_parameters(cDecomp_custom_random.get_parameter_num());
        if (iter == 0 && initial_parameters.size() > 0) {
            int n_use = std::min(
                static_cast<int>(initial_parameters.size()),
                cDecomp_custom_random.get_parameter_num());
            std::copy(initial_parameters.get_data(),
                      initial_parameters.get_data() + n_use,
                      optimized_parameters.begin());
            for (int idx = n_use; idx < cDecomp_custom_random.get_parameter_num(); ++idx) {
                optimized_parameters[idx] = distrib_real(gen);
            }
        } else if (iter == 0 && optimized_parameters_mtx.size() > 0) {
            int n_use = std::min(
                static_cast<int>(optimized_parameters_mtx.size()),
                cDecomp_custom_random.get_parameter_num());
            std::copy(optimized_parameters_mtx.get_data(),
                      optimized_parameters_mtx.get_data() + n_use,
                      optimized_parameters.begin());
            for (int idx = n_use; idx < cDecomp_custom_random.get_parameter_num(); ++idx) {
                optimized_parameters[idx] = distrib_real(gen);
            }
        } else {
            for (size_t idx = 0; idx < optimized_parameters.size(); ++idx) {
                optimized_parameters[idx] = distrib_real(gen);
            }
        }
        if (!optimized_parameters.empty()) {
            cDecomp_custom_random.set_optimized_parameters(
                optimized_parameters.data(), static_cast<int>(optimized_parameters.size()));
        }

        cDecomp_custom_random.start_decomposition();
        Matrix_real optimized_parameters_tmp = cDecomp_custom_random.get_optimized_parameters();
        double current_minimum_tmp = cDecomp_custom_random.optimization_problem(optimized_parameters_tmp);
        if (current_minimum_tmp < result.current_minimum) {
            result.current_minimum = current_minimum_tmp;
            result.optimized_parameters = optimized_parameters_tmp.copy();
            result.decomposition_error = cDecomp_custom_random.get_decomposition_error();
        }
        if (current_minimum_tmp < optimization_tolerance_loc &&
            cDecomp_custom_random.get_decomposition_error() < optimization_tolerance_loc) {
            result.reached_tolerance = true;
            break;
        }
    }
}

N_Qubit_Decomposition_OSR_Compression_Result
N_Qubit_Decomposition_OSR_Compression::compress_gate_structure(
    Gates_block* gate_structure_in) {
    if (gate_structure_in == NULL) {
        std::string err("N_Qubit_Decomposition_OSR_Compression::compress_gate_structure: gate_structure is null");
        throw err;
    }

    N_Qubit_Decomposition_OSR_Compression_Options options = get_osr_compression_options();
    int original_entangling_gate_num =
        static_cast<int>(collect_entangling_gate_paths(gate_structure_in).size());

    int max_removed = options.max_removed_gates < 0
        ? original_entangling_gate_num
        : std::min(options.max_removed_gates, original_entangling_gate_num);

    Matrix_real search_initial_parameters = optimized_parameters_mtx.size() == gate_structure_in->get_parameter_num()
        ? optimized_parameters_mtx.copy()
        : Matrix_real(0, 0);
    std::unique_ptr<Gates_block> phase1_gate_structure;
    Gates_block* search_gate_structure = gate_structure_in;
    int phase1_removed_num = 0;
    N_Qubit_Decomposition_OSR_Compression_Result best_phase1_result;

    if (options.validate_final && max_removed >= 1) {
        std::vector<OSRGatePath> original_paths =
            collect_entangling_gate_paths(gate_structure_in);
        bool phase1_topology_user_set = !this->topology.empty();
        std::vector<matrix_base<int>> phase1_active_topology = phase1_topology_user_set
            ? this->topology
            : topology_from_gate_structure(gate_structure_in, qbit_num);
        std::vector<std::pair<int, int>> phase1_edges =
            topology_pairs_from_matrices(phase1_active_topology);

        std::vector<int> root_sequence;
        if (!phase1_edges.empty() &&
            extract_cnot_skeleton_sequence(
                gate_structure_in, original_paths, phase1_edges, root_sequence)) {
                std::vector<Phase1RemovalState> phase1_beam;
                std::unique_ptr<Gates_block> root_skeleton(
                    construct_cnot_skeleton_gate_structure(qbit_num, phase1_edges, root_sequence));
                if (root_skeleton) {
                    N_Qubit_Decomposition_OSR_Compression_Result root_skeleton_result;
                    validate_compressed_gate_structure(
                        root_skeleton.get(), search_initial_parameters, root_skeleton_result);
                    if (root_skeleton_result.reached_tolerance) {
                        Phase1RemovalState skeleton_root_state;
                        skeleton_root_state.gate_structure.reset(root_skeleton.release());
                        skeleton_root_state.sequence = root_sequence;
                        skeleton_root_state.parameters = root_skeleton_result.optimized_parameters.copy();
                        skeleton_root_state.current_minimum = root_skeleton_result.current_minimum;
                        skeleton_root_state.decomposition_error = root_skeleton_result.decomposition_error;
                        skeleton_root_state.entangling_gate_num =
                            static_cast<int>(root_sequence.size());
                        phase1_beam.push_back(skeleton_root_state);
                    }
                }

                Phase1RemovalState imported_root_state;
                imported_root_state.gate_structure.reset(gate_structure_in->clone());
                imported_root_state.sequence = root_sequence;
                imported_root_state.parameters = search_initial_parameters.copy();
                imported_root_state.current_minimum = 0.0;
                imported_root_state.decomposition_error = 0.0;
                imported_root_state.entangling_gate_num = static_cast<int>(root_sequence.size());
                phase1_beam.push_back(imported_root_state);

                std::set<std::string> phase1_seen;
                {
                    std::stringstream key_stream;
                    append_int_vector_signature(key_stream, root_sequence);
                    phase1_seen.insert(key_stream.str());
                }

                bool found_any_phase1_removal = false;
                Phase1RemovalState best_phase1_state;

                for (int depth = 0; depth < max_removed; ++depth) {
                    std::vector<Phase1RemovalState> next_beam;
                    int max_removals_this_depth =
                        qbit_num <= 3
                            ? std::numeric_limits<int>::max()
                            : 2 * static_cast<int>(phase1_beam[0].sequence.size());
                    int generated_this_depth = 0;

                    for (size_t state_idx = 0; state_idx < phase1_beam.size(); ++state_idx) {
                        if (generated_this_depth >= max_removals_this_depth) {
                            break;
                        }
                        Phase1RemovalState& state = phase1_beam[state_idx];
                        int parent_cnot_num = static_cast<int>(state.sequence.size());
                        if (parent_cnot_num <= 0) {
                            continue;
                        }

                        bool parent_has_skeleton_params =
                            state.parameters.size() == parent_cnot_num * 6 + 3 * qbit_num;
                        std::vector<std::pair<double, int>> removal_positions;
                        removal_positions.reserve(parent_cnot_num);
                        for (int pos = 0; pos < parent_cnot_num; ++pos) {
                            double triviality = 0.0;
                            if (parent_has_skeleton_params) {
                                for (int param = 0; param < 6; ++param) {
                                    double theta = state.parameters[pos * 6 + param];
                                    double k = std::round(theta / (M_PI / 2.0));
                                    triviality += std::abs(theta - k * M_PI / 2.0);
                                }
                            } else {
                                triviality = static_cast<double>(pos);
                            }
                            removal_positions.push_back(std::make_pair(triviality, pos));
                        }
                        std::sort(removal_positions.begin(), removal_positions.end());
                        int max_positions = std::min(
                            static_cast<int>(removal_positions.size()),
                            2 * parent_cnot_num);

                        auto try_multi_removal = [&](const std::vector<int>& positions_desc) -> bool {
                            std::vector<int> child_sequence = state.sequence;
                            for (int p : positions_desc) {
                                child_sequence.erase(child_sequence.begin() + p);
                            }
                            std::stringstream key_stream;
                            append_int_vector_signature(key_stream, child_sequence);
                            if (!phase1_seen.insert(key_stream.str()).second) return false;

                            std::unique_ptr<Gates_block> child_gate_structure(
                                construct_cnot_skeleton_gate_structure(
                                    qbit_num, phase1_edges, child_sequence));
                            if (!child_gate_structure) return false;

                            Matrix_real child_initial_parameters;
                            if (parent_has_skeleton_params) {
                                if (positions_desc.size() == 1) {
                                    child_initial_parameters = excise_skeleton_gate_params(
                                        state.parameters, parent_cnot_num,
                                        positions_desc[0], qbit_num);
                                } else {
                                    child_initial_parameters = excise_skeleton_gate_params_multi(
                                        state.parameters, parent_cnot_num,
                                        positions_desc, qbit_num);
                                }
                            } else {
                                child_initial_parameters = state.parameters.copy();
                            }

                            N_Qubit_Decomposition_OSR_Compression_Result child_result;
                            child_result.gate_structure.reset(child_gate_structure.release());
                            child_result.original_entangling_gate_num = original_entangling_gate_num;
                            child_result.compressed_entangling_gate_num =
                                static_cast<int>(child_sequence.size());

                            validate_compressed_gate_structure(
                                child_result.gate_structure.get(),
                                child_initial_parameters,
                                child_result);

                            if (!child_result.reached_tolerance) return false;

                            Phase1RemovalState child;
                            child.gate_structure.reset(child_result.gate_structure.release());
                            child.sequence = child_sequence;
                            child.parameters = child_result.optimized_parameters.copy();
                            child.current_minimum = child_result.current_minimum;
                            child.decomposition_error = child_result.decomposition_error;
                            child.entangling_gate_num =
                                static_cast<int>(child_sequence.size());
                            next_beam.push_back(child);

                            if (!found_any_phase1_removal ||
                                child.entangling_gate_num < best_phase1_state.entangling_gate_num ||
                                (child.entangling_gate_num == best_phase1_state.entangling_gate_num &&
                                 child.current_minimum < best_phase1_state.current_minimum)) {
                                best_phase1_state = child;
                                found_any_phase1_removal = true;
                            }
                            return true;
                        };

                        // k=1: existing single-position removals.
                        for (int position_idx = 0; position_idx < max_positions; ++position_idx) {
                            if (generated_this_depth >= max_removals_this_depth) break;
                            int remove_pos = removal_positions[position_idx].second;
                            generated_this_depth++;
                            try_multi_removal({remove_pos});
                        }

                        // k=2 and k=3: direct multi-position removals from the same parent.
                        // Only run on parents with rotation params, since multi-position excision
                        // assumes the skeleton-param layout (6 params per CNOT slot).
                        if (options.enable_triple_removal && parent_has_skeleton_params) {
                            int top_k = std::min({options.triple_top_k,
                                                  static_cast<int>(removal_positions.size()),
                                                  parent_cnot_num});

                            // k=2
                            int pairs_emitted = 0;
                            for (int i = 0; i < top_k && pairs_emitted < options.max_pairs_per_parent; ++i) {
                                for (int j = i + 1; j < top_k && pairs_emitted < options.max_pairs_per_parent; ++j) {
                                    if (generated_this_depth >= max_removals_this_depth) break;
                                    int pa = removal_positions[i].second;
                                    int pb = removal_positions[j].second;
                                    std::vector<int> positions_desc = {std::max(pa, pb), std::min(pa, pb)};
                                    generated_this_depth++;
                                    if (try_multi_removal(positions_desc)) pairs_emitted++;
                                }
                            }

                            // k=3
                            int triples_emitted = 0;
                            int top_k_triples = std::min(top_k, parent_cnot_num);
                            for (int i = 0; i < top_k_triples && triples_emitted < options.max_triples_per_parent; ++i) {
                                for (int j = i + 1; j < top_k_triples && triples_emitted < options.max_triples_per_parent; ++j) {
                                    for (int k = j + 1; k < top_k_triples && triples_emitted < options.max_triples_per_parent; ++k) {
                                        if (generated_this_depth >= max_removals_this_depth) break;
                                        int pa = removal_positions[i].second;
                                        int pb = removal_positions[j].second;
                                        int pc = removal_positions[k].second;
                                        std::vector<int> positions_desc = {pa, pb, pc};
                                        std::sort(positions_desc.begin(), positions_desc.end(), std::greater<int>());
                                        generated_this_depth++;
                                        if (try_multi_removal(positions_desc)) triples_emitted++;
                                    }
                                }
                            }
                        }
                    }

                    if (next_beam.empty()) {
                        break;
                    }

                    std::sort(next_beam.begin(), next_beam.end(),
                        [](const Phase1RemovalState& lhs, const Phase1RemovalState& rhs) {
                            if (lhs.entangling_gate_num != rhs.entangling_gate_num) {
                                return lhs.entangling_gate_num < rhs.entangling_gate_num;
                            }
                            return lhs.current_minimum < rhs.current_minimum;
                        });

                    // Diversity-preserving beam pruning: cap any single CNOT-count
                    // class at half the beam to prevent multi-removal children from
                    // starving the single-removal frontier. Without this, k=3 children
                    // (which sort first by entangling_gate_num) can fill all beam slots
                    // and evict the k=1 children that would have reached a better
                    // global minimum on the next depth step.
                    if (static_cast<int>(next_beam.size()) > options.beam_width) {
                        std::vector<Phase1RemovalState> diversified;
                        diversified.reserve(options.beam_width);
                        int per_class_cap = std::max(1, options.beam_width / 2);
                        std::map<int, int> class_counts;
                        for (auto& s : next_beam) {
                            if (static_cast<int>(diversified.size()) >= options.beam_width) break;
                            int& c = class_counts[s.entangling_gate_num];
                            if (c >= per_class_cap) continue;
                            c++;
                            diversified.push_back(std::move(s));
                        }
                        // If quota left some slots empty (e.g., only one class present),
                        // fill remaining slots with the best leftovers ignoring caps.
                        if (static_cast<int>(diversified.size()) < options.beam_width) {
                            for (auto& s : next_beam) {
                                if (static_cast<int>(diversified.size()) >= options.beam_width) break;
                                if (s.gate_structure) diversified.push_back(std::move(s));
                            }
                        }
                        next_beam.swap(diversified);
                    }
                    phase1_beam.swap(next_beam);
                }

                if (found_any_phase1_removal) {
                    phase1_removed_num =
                        original_entangling_gate_num - best_phase1_state.entangling_gate_num;
                    search_initial_parameters = best_phase1_state.parameters.copy();
                    phase1_gate_structure.reset(best_phase1_state.gate_structure->clone());
                    search_gate_structure = phase1_gate_structure.get();

                    best_phase1_result.gate_structure.reset(best_phase1_state.gate_structure->clone());
                    best_phase1_result.optimized_parameters = best_phase1_state.parameters.copy();
                    best_phase1_result.current_minimum = best_phase1_state.current_minimum;
                    best_phase1_result.decomposition_error = best_phase1_state.decomposition_error;
                    best_phase1_result.original_entangling_gate_num = original_entangling_gate_num;
                    best_phase1_result.compressed_entangling_gate_num =
                        best_phase1_state.entangling_gate_num;
                    best_phase1_result.validated = true;
                    best_phase1_result.reached_tolerance = true;
                }

                if (verbose > 0) {
                    std::stringstream sstream;
                    sstream << "OSR compression skeleton Phase 1 removed " << phase1_removed_num
                            << " CNOTs; best skeleton CNOTs: "
                            << (found_any_phase1_removal
                                    ? best_phase1_state.entangling_gate_num
                                    : original_entangling_gate_num)
                            << std::endl;
                    print(sstream, 1);
                }

                if (best_phase1_result.gate_structure) {
                    return best_phase1_result;
                }

                // Additive-search handoff (opt-in via osr_compression_handoff_to_additive).
                // When the deletion beam stalled (no CNOT reduction), the lowest-OSR
                // skeleton would be handed to N_Qubit_Decomposition_Surrogate as a warm
                // start. Deferred: Surrogate is not currently linked into the squander
                // build target (the .cpp lives at the repo root and is absent from
                // CMakeLists.txt / setup.py). Re-add Surrogate to the build, or retarget
                // this handoff to N_Qubit_Decomposition_adaptive, before wiring the call.
                if (options.handoff_to_additive && !found_any_phase1_removal && verbose > 0) {
                    std::stringstream sstream;
                    sstream << "OSR compression handoff_to_additive requested but deferred:"
                            << " additive class not currently in build.\n";
                    print(sstream, 1);
                }
        }

        if (!best_phase1_result.gate_structure) {
            Phase1RemovalState root_state;
            root_state.gate_structure.reset(gate_structure_in->clone());
            root_state.parameters = search_initial_parameters.copy();
            root_state.current_minimum = std::numeric_limits<double>::infinity();
            root_state.decomposition_error = std::numeric_limits<double>::infinity();
            root_state.entangling_gate_num = original_entangling_gate_num;

            std::vector<Phase1RemovalState> phase1_beam(1, root_state);
            std::set<std::string> phase1_seen;
            phase1_seen.insert(gate_structure_signature(gate_structure_in));
            bool found_any_phase1_removal = false;
            Phase1RemovalState best_phase1_state;

            for (int depth = 0; depth < max_removed; ++depth) {
                std::vector<Phase1RemovalState> next_beam;

                for (size_t state_idx = 0; state_idx < phase1_beam.size(); ++state_idx) {
                    Phase1RemovalState& state = phase1_beam[state_idx];
                    std::vector<OSRGatePath> current_paths =
                        collect_entangling_gate_paths(state.gate_structure.get());
                    if (current_paths.empty()) {
                        continue;
                    }

                    for (int remove_id = 0; remove_id < static_cast<int>(current_paths.size()); ++remove_id) {
                        Gate* gate = gate_at_path(state.gate_structure.get(), current_paths[remove_id]);
                        if (gate == NULL || gate->get_type() != CNOT_OPERATION) {
                            continue;
                        }

                        std::vector<int> removed_ids(1, remove_id);
                        std::unique_ptr<Gates_block> candidate_gate_structure(
                            clone_without_removed_paths(
                                state.gate_structure.get(), current_paths, removed_ids));
                        std::string candidate_key =
                            gate_structure_signature(candidate_gate_structure.get());
                        if (!phase1_seen.insert(candidate_key).second) {
                            continue;
                        }

                        Matrix_real initial_parameters = reduced_parameters_without_removed_paths(
                            state.gate_structure.get(), current_paths, removed_ids, state.parameters);

                        N_Qubit_Decomposition_OSR_Compression_Result candidate_result;
                        candidate_result.gate_structure.reset(candidate_gate_structure.release());
                        candidate_result.original_entangling_gate_num = original_entangling_gate_num;
                        candidate_result.compressed_entangling_gate_num =
                            static_cast<int>(current_paths.size()) - 1;

                        validate_compressed_gate_structure(
                            candidate_result.gate_structure.get(), initial_parameters, candidate_result);

                        if (!candidate_result.reached_tolerance) {
                            continue;
                        }

                        Phase1RemovalState child;
                        child.gate_structure.reset(candidate_result.gate_structure.release());
                        child.parameters = candidate_result.optimized_parameters.copy();
                        child.current_minimum = candidate_result.current_minimum;
                        child.decomposition_error = candidate_result.decomposition_error;
                        child.entangling_gate_num =
                            static_cast<int>(current_paths.size()) - 1;
                        next_beam.push_back(child);

                        if (!found_any_phase1_removal ||
                            child.entangling_gate_num < best_phase1_state.entangling_gate_num ||
                            (child.entangling_gate_num == best_phase1_state.entangling_gate_num &&
                             child.current_minimum < best_phase1_state.current_minimum)) {
                            best_phase1_state = child;
                            found_any_phase1_removal = true;
                        }
                    }
                }

                if (next_beam.empty()) {
                    break;
                }

                std::sort(next_beam.begin(), next_beam.end(),
                    [](const Phase1RemovalState& lhs, const Phase1RemovalState& rhs) {
                        if (lhs.entangling_gate_num != rhs.entangling_gate_num) {
                            return lhs.entangling_gate_num < rhs.entangling_gate_num;
                        }
                        return lhs.current_minimum < rhs.current_minimum;
                    });
                if (static_cast<int>(next_beam.size()) > options.beam_width) {
                    next_beam.resize(options.beam_width);
                }
                phase1_beam.swap(next_beam);
            }

            if (found_any_phase1_removal) {
                phase1_removed_num =
                    original_entangling_gate_num - best_phase1_state.entangling_gate_num;

                best_phase1_result.gate_structure.reset(best_phase1_state.gate_structure->clone());
                best_phase1_result.optimized_parameters = best_phase1_state.parameters.copy();
                best_phase1_result.current_minimum = best_phase1_state.current_minimum;
                best_phase1_result.decomposition_error = best_phase1_state.decomposition_error;
                best_phase1_result.original_entangling_gate_num = original_entangling_gate_num;
                best_phase1_result.compressed_entangling_gate_num =
                    best_phase1_state.entangling_gate_num;
                best_phase1_result.validated = true;
                best_phase1_result.reached_tolerance = true;
            }

            if (verbose > 0) {
                std::stringstream sstream;
                sstream << "OSR compression direct Phase 1 removed " << phase1_removed_num
                        << " CNOTs; best CNOTs: "
                        << (found_any_phase1_removal
                                ? best_phase1_state.entangling_gate_num
                                : original_entangling_gate_num)
                        << std::endl;
                print(sstream, 1);
            }

            if (best_phase1_result.gate_structure) {
                return best_phase1_result;
            }
        }
    }

    std::vector<OSRGatePath> removable_paths = collect_entangling_gate_paths(search_gate_structure);
    max_removed = options.max_removed_gates < 0
        ? static_cast<int>(removable_paths.size())
        : std::min(options.max_removed_gates - phase1_removed_num,
                   static_cast<int>(removable_paths.size()));

    if (max_removed <= 0 && best_phase1_result.gate_structure) {
        return best_phase1_result;
    }

    std::vector<std::vector<int>> all_cuts = unique_cuts(qbit_num);
    std::sort(all_cuts.begin(), all_cuts.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    });

    bool topology_user_set = !this->topology.empty();
    std::vector<matrix_base<int>> active_topology = topology_user_set
        ? this->topology
        : topology_from_gate_structure(search_gate_structure, qbit_num);
    std::vector<std::pair<int, int>> mutation_edges =
        (options.mutate_full_topology && !topology_user_set)
            ? complete_topology_pairs(qbit_num)
            : topology_pairs_from_matrices(active_topology);
    MinCnotBoundSolver osr_bound_solver(qbit_num, all_cuts, active_topology);

    CompressionCandidate root;
    root.entangling_gate_num = static_cast<int>(removable_paths.size());
    root.key = gate_structure_signature(search_gate_structure);
    root.initial_parameters = search_initial_parameters.copy();
    root.score = evaluate_gate_structure_osr(
        search_gate_structure, root.initial_parameters, osr_bound_solver, all_cuts);

    CompressionCandidate best = root;
    std::vector<CompressionCandidate> beam(1, root);

    for (int depth = 1; depth <= max_removed; ++depth) {
        std::vector<CompressionCandidate> next_candidates;

        for (size_t beam_idx = 0; beam_idx < beam.size(); ++beam_idx) {
            const CompressionCandidate& parent = beam[beam_idx];
            int start_id = parent.removed_ids.empty() ? 0 : parent.removed_ids.back() + 1;

            for (int remove_id = start_id; remove_id < static_cast<int>(removable_paths.size()); ++remove_id) {
                CompressionCandidate child;
                child.removed_ids = parent.removed_ids;
                child.removed_ids.push_back(remove_id);
                child.entangling_gate_num =
                    static_cast<int>(removable_paths.size()) - static_cast<int>(child.removed_ids.size());

                std::unique_ptr<Gates_block> candidate_gate_structure(
                    clone_without_removed_paths(search_gate_structure, removable_paths, child.removed_ids));
                child.initial_parameters = reduced_parameters_without_removed_paths(
                    search_gate_structure, removable_paths, child.removed_ids, search_initial_parameters);
                child.key = gate_structure_signature(candidate_gate_structure.get());
                child.score = evaluate_gate_structure_osr(
                    candidate_gate_structure.get(), child.initial_parameters, osr_bound_solver, all_cuts);

                if (candidate_is_osr_admissible(child, options)) {
                    next_candidates.push_back(child);
                }
            }
        }

        if (next_candidates.empty()) {
            break;
        }

        sort_unique_candidates(next_candidates, false);
        if (static_cast<int>(next_candidates.size()) > options.beam_width) {
            next_candidates.resize(options.beam_width);
        }

        for (size_t idx = 0; idx < next_candidates.size(); ++idx) {
            if (candidate_is_osr_admissible(next_candidates[idx], options) &&
                final_candidate_less(next_candidates[idx], best)) {
                best = next_candidates[idx];
            }
        }

        beam.swap(next_candidates);
    }

    std::vector<CompressionCandidate> validation_pool = beam;
    validation_pool.push_back(root);
    validation_pool.push_back(best);

    if (options.enable_mutations && options.mutation_rounds > 0 &&
        options.mutation_candidates > 0 && !mutation_edges.empty()) {
        std::vector<CompressionCandidate> mutation_seeds = validation_pool;
        sort_unique_candidates(mutation_seeds, true);
        if (static_cast<int>(mutation_seeds.size()) > options.beam_width) {
            mutation_seeds.resize(options.beam_width);
        }

        for (int round = 0; round < options.mutation_rounds; ++round) {
            std::vector<CompressionCandidate> round_mutations;

            for (size_t seed_idx = 0; seed_idx < mutation_seeds.size(); ++seed_idx) {
                const CompressionCandidate& seed = mutation_seeds[seed_idx];
                std::unique_ptr<Gates_block> seed_gate_structure(
                    clone_gate_structure_for_candidate(
                        search_gate_structure, removable_paths, seed));

                std::vector<CompressionCandidate> local_mutations =
                    generate_local_mutation_candidates(
                        seed_gate_structure.get(), seed.initial_parameters, seed,
                        mutation_edges, options);

                for (size_t mut_idx = 0; mut_idx < local_mutations.size(); ++mut_idx) {
                    CompressionCandidate& mutation = local_mutations[mut_idx];
                    mutation.score = evaluate_gate_structure_osr(
                        mutation.gate_structure.get(), mutation.initial_parameters,
                        osr_bound_solver, all_cuts);
                    if (candidate_is_osr_admissible(mutation, options)) {
                        round_mutations.push_back(mutation);
                    }
                }
            }

            if (round_mutations.empty()) {
                break;
            }

            sort_unique_candidates(round_mutations, false);
            if (static_cast<int>(round_mutations.size()) > options.mutation_candidates) {
                round_mutations.resize(options.mutation_candidates);
            }

            validation_pool.insert(
                validation_pool.end(), round_mutations.begin(), round_mutations.end());
            mutation_seeds.swap(round_mutations);
        }
    }

    if (options.enable_skeleton_search) {
        std::vector<std::pair<int, int>> skeleton_edges = topology_user_set
            ? topology_pairs_from_matrices(active_topology)
            : ((options.mutate_full_topology || qbit_num <= 3)
                   ? complete_topology_pairs(qbit_num)
                   : mutation_edges);
        std::vector<CompressionCandidate> skeleton_candidates =
            generate_cnot_skeleton_candidates(
                qbit_num, static_cast<int>(removable_paths.size()),
                skeleton_edges, options);
        validation_pool.insert(
            validation_pool.end(), skeleton_candidates.begin(), skeleton_candidates.end());
    }

    if (options.enable_local_resynth) {
        std::vector<std::pair<int, int>> resynth_edges = topology_user_set
            ? topology_pairs_from_matrices(active_topology)
            : ((options.mutate_full_topology || qbit_num <= 3)
                   ? complete_topology_pairs(qbit_num)
                   : mutation_edges);
        std::vector<CompressionCandidate> resynth_candidates =
            generate_local_resynth_candidates(
                search_gate_structure, resynth_edges, best.score, qbit_num, options);
        validation_pool.insert(
            validation_pool.end(), resynth_candidates.begin(), resynth_candidates.end());
    }

    sort_unique_candidates(validation_pool, true);
    int validation_pool_limit = options.beam_width;
    if (options.enable_mutations) {
        validation_pool_limit += options.mutation_candidates;
    }
    if (options.enable_skeleton_search) {
        validation_pool_limit += options.skeleton_max_candidates;
    }
    if (options.enable_local_resynth) {
        validation_pool_limit += options.local_resynth_max_candidates;
    }
    validation_pool_limit = std::max(validation_pool_limit, options.beam_width);
    if (static_cast<int>(validation_pool.size()) > validation_pool_limit) {
        validation_pool.resize(validation_pool_limit);
    }
    bool has_root_candidate = false;
    for (size_t idx = 0; idx < validation_pool.size(); ++idx) {
        if (compression_candidate_key(validation_pool[idx]) == root.key) {
            has_root_candidate = true;
            break;
        }
    }
    if (!has_root_candidate) {
        validation_pool.push_back(root);
    }

    N_Qubit_Decomposition_OSR_Compression_Result result;
    result.original_entangling_gate_num = static_cast<int>(removable_paths.size());

    if (!options.validate_final) {
        result.gate_structure.reset(
            clone_gate_structure_for_candidate(search_gate_structure, removable_paths, best));
        result.osr_score = best.score;
        for (size_t idx = 0; idx < best.removed_ids.size(); ++idx) {
            result.removed_gate_paths.push_back(removable_paths[best.removed_ids[idx]]);
        }
        result.compressed_entangling_gate_num = best.entangling_gate_num;
        return result;
    }

    bool selected_validated_candidate = false;
    N_Qubit_Decomposition_OSR_Compression_Result best_validated_result;

    best_validated_result.gate_structure.reset(
        clone_gate_structure_for_candidate(search_gate_structure, removable_paths, root));
    best_validated_result.osr_score = root.score;
    best_validated_result.original_entangling_gate_num = static_cast<int>(removable_paths.size());
    best_validated_result.compressed_entangling_gate_num = root.entangling_gate_num;
    validate_compressed_gate_structure(
        best_validated_result.gate_structure.get(), root.initial_parameters, best_validated_result);

    for (size_t idx = 0; idx < validation_pool.size(); ++idx) {
        const CompressionCandidate& candidate = validation_pool[idx];
        if (compression_candidate_key(candidate) == root.key) {
            continue;
        }
        N_Qubit_Decomposition_OSR_Compression_Result candidate_result;
        candidate_result.gate_structure.reset(
            clone_gate_structure_for_candidate(search_gate_structure, removable_paths, candidate));
        candidate_result.osr_score = candidate.score;
        candidate_result.original_entangling_gate_num = static_cast<int>(removable_paths.size());
        candidate_result.compressed_entangling_gate_num = candidate.entangling_gate_num;
        for (size_t removed_idx = 0; removed_idx < candidate.removed_ids.size(); ++removed_idx) {
            candidate_result.removed_gate_paths.push_back(removable_paths[candidate.removed_ids[removed_idx]]);
        }

        validate_compressed_gate_structure(
            candidate_result.gate_structure.get(), candidate.initial_parameters, candidate_result);

        if ((candidate_result.reached_tolerance && !best_validated_result.reached_tolerance) ||
            (candidate_result.reached_tolerance && best_validated_result.reached_tolerance &&
             candidate_result.compressed_entangling_gate_num < best_validated_result.compressed_entangling_gate_num) ||
            (candidate_result.reached_tolerance && best_validated_result.reached_tolerance &&
             candidate_result.compressed_entangling_gate_num == best_validated_result.compressed_entangling_gate_num &&
             candidate_result.current_minimum < best_validated_result.current_minimum) ||
            (!candidate_result.reached_tolerance && !best_validated_result.reached_tolerance &&
             candidate_result.current_minimum < best_validated_result.current_minimum)) {
            best_validated_result = std::move(candidate_result);
            selected_validated_candidate = true;
        }

        if (best_validated_result.reached_tolerance &&
            best_validated_result.compressed_entangling_gate_num == validation_pool.front().entangling_gate_num) {
            break;
        }
    }

    if (selected_validated_candidate) {
        return best_validated_result;
    }

    if (phase1_removed_num > 0 && best_validated_result.reached_tolerance) {
        return best_validated_result;
    }

    const CompressionCandidate* fallback = &best;
    for (size_t idx = 0; idx < validation_pool.size(); ++idx) {
        const CompressionCandidate& cand = validation_pool[idx];
        if (compression_candidate_key(cand) == root.key) {
            continue;
        }
        if (!candidate_is_osr_admissible(cand, options)) {
            continue;
        }
        bool fallback_is_root = compression_candidate_key(*fallback) == root.key;
        if (fallback_is_root || cand.entangling_gate_num < fallback->entangling_gate_num) {
            fallback = &cand;
        }
    }

    result.gate_structure.reset(
        clone_gate_structure_for_candidate(search_gate_structure, removable_paths, *fallback));
    result.osr_score = fallback->score;
    for (size_t idx = 0; idx < fallback->removed_ids.size(); ++idx) {
        result.removed_gate_paths.push_back(removable_paths[fallback->removed_ids[idx]]);
    }
    result.compressed_entangling_gate_num = fallback->entangling_gate_num;
    return result;
}

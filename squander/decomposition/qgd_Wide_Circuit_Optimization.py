"""
Implementation to optimize wide circuits (i.e. circuits with many qubits) by    partitioning the circuit into smaller partitions and redecompose the smaller partitions
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
    qgd_N_Qubit_Decomposition_Surrogate as N_Qubit_Decomposition_Surrogate
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits
from qgd_N_Qubit_Decomposition_Guided_tree import N_Qubit_Decomposition_Guided_Tree
import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology

def CNOTGateCount( circ: Circuit, max_gates: int = 0 ) -> int :
    """
    Call to get the number of CNOT gates in the circuit

    
    Args:

        circ (Circuit) A squander circuit representation


    Return:

        Returns with the CNOT gate count

    
    """ 

    if not isinstance(circ, Circuit ):
        raise Exception("The input parameters should be an instance of Squander Circuit")

    gate_counts = circ.get_Gate_Nums()
    num_cnots = gate_counts.get('CNOT', 0)
    
    if max_gates > 0: return num_cnots*max_gates + sum(y for x, y in gate_counts.items() if x !='CNOT')
    return num_cnots  #+  3*gate_counts.get('SWAP', 0)



class qgd_Wide_Circuit_Optimization:
    """
    Class implementing the optimization of wide circuits (i.e. circuits with many qubits) by
    partitioning the circuit into smaller partitions and redecompose the smaller partitions

    """
    
    def __init__( self, config ):

        config.setdefault('strategy', 'TreeSearch')
        config.setdefault('parallel', 0 )
        config.setdefault('verbosity', 0 )
        config.setdefault('tolerance', 1e-8 )
        config.setdefault('test_subcircuits', False )
        config.setdefault('test_final_circuit', True )
        config.setdefault('max_partition_size', 3 )
        config.setdefault('topology', None)
        config.setdefault('routed', False)
        config.setdefault('partition_strategy','ilp')
        config.setdefault('optimizer','BFGS')
        #testing the fields of config 
        strategy = config[ 'strategy' ]
        allowed_startegies = ['TreeSearch', 'TabuSearch', 'Adaptive', 'TreeGuided','SurrSearch', 'HybridSearch' ]
        if not strategy in allowed_startegies :
            raise Exception(f"The decomposition startegy should be either of {allowed_startegies}, got {strategy}.")


        parallel = config[  'parallel' ]
        allowed_parallel = [0, 1, 2 ]
        if not parallel in allowed_parallel :
            raise Exception(f"The parallel configuration should be either of {allowed_parallel}, got {parallel}.")


        verbosity = config[ 'verbosity' ]
        if not isinstance( verbosity, int) :
            raise Exception(f"The verbosity parameter should be an integer.")


        tolerance = config[ 'tolerance' ]
        if not isinstance( tolerance, float) :
            raise Exception(f"The tolerance parameter should be a float.")


        test_subcircuits = config[ 'test_subcircuits' ]
        if not isinstance( test_subcircuits, bool) :
            raise Exception(f"The test_subcircuits parameter should be a bool.")


        test_final_circuit = config[ 'test_final_circuit' ]
        if not isinstance( test_final_circuit, bool) :
            raise Exception(f"The test_final_circuit parameter should be a bool.")

 

        max_partition_size = config[ 'max_partition_size' ]
        if not isinstance( max_partition_size, int) :
            raise Exception(f"The max_partition_size parameter should be an integer.")

        self.config = config


        self.max_partition_size = max_partition_size



    def ConstructCircuitFromPartitions( self, circs: List[Circuit], parameter_arrs: List[List[np.ndarray]] ) -> Tuple[Circuit, np.ndarray]:
        """
        Call to construct the wide quantum circuit from the partitions.

    
        Args:

            circs ( List[Circuit] ) A list of Squander circuits to be compared

            parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

        Return:

            Returns with the constructed circuit and the corresponding parameter array

    
        """ 

        if not isinstance( circs, list ):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance( parameter_arrs, list ):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs) :
            raise Exception("The first two arguments should be of the same length")


        qbit_num = circs[0].get_Qbit_Num()



        wide_parameters = np.concatenate( parameter_arrs, axis=0 ) 


        wide_circuit = Circuit( qbit_num )

        for circ in circs:
            wide_circuit.add_Circuit( circ )


        assert wide_circuit.get_Parameter_Num() == wide_parameters.size, \
                f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"



        return wide_circuit, wide_parameters


    @staticmethod
    def DecomposePartition( Umtx: np.ndarray, config: dict, mini_topology = None, structure = None ) -> Circuit:
        """
        Call to run the decomposition of a given unitary Umtx, typically associated with the circuit 
        partition to be optimized

    
        Args:

            Umtx (np.ndarray) A complex typed unitary to be decomposed


        Return:

            Returns with the the decoposed circuit structure and with the corresponding gate parameters

    
        """ 
        strategy = config["strategy"]
        if strategy == "TreeSearch":
            cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
        elif strategy == "TabuSearch":
            cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology )
        elif strategy == "Adaptive":
            cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=1, topology=mini_topology )
        elif strategy == "TreeGuided":
            cDecompose = N_Qubit_Decomposition_Guided_Tree( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology )
        elif strategy == "SurrSearch":
            cDecompose = N_Qubit_Decomposition_Surrogate( Umtx.conj().T, config=config, accelerator_num=0)
        elif strategy == "Custom":
            cDecompose = N_Qubit_Decomposition_custom( Umtx.conj().T, config=config, accelerator_num=0 )
            assert structure is not None, "Custom decomposition strategy requires a gate structure to be provided."
            cDecompose.set_Gate_Structure( structure )
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")


        tolerance = config["tolerance"]
        cDecompose.set_Verbose( config["verbosity"] )
        cDecompose.set_Cost_Function_Variant( 3 )	
        cDecompose.set_Optimization_Tolerance( tolerance )
    

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer( config['optimizer'] )

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e: 
            #print(e)
            raise e
            #return []
        if not config.get("stop_first_solution", True): return cDecompose.all_solutions
        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
        assert parameters is not None


        if strategy == "Custom":
            err = cDecompose.Optimization_Problem(parameters)
            it = 0
            while err > tolerance and it < 10:
                cDecompose.set_Optimized_Parameters(np.random.rand(cDecompose.get_Parameter_Num())*(2*np.pi))
                cDecompose.Start_Decomposition()
                parameters       = cDecompose.get_Optimized_Parameters()
                err = cDecompose.Optimization_Problem(parameters)
                it += 1
            if err > tolerance or it != 0: print( "Decomposition error: ", err, it )
        else: err = cDecompose.get_Decomposition_Error()
        #print( "Decomposition error: ", err )
        if tolerance < err:
            #raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []


        return [(squander_circuit, parameters)]



    @staticmethod
    def CompareAndPickCircuits( circs: List[Circuit], parameter_arrs: List[List[np.ndarray]], metric : Callable[ [Circuit], int ] = CNOTGateCount ) -> Circuit:
        """
        Call to pick the most optimal circuit corresponding a specific metric. Looks for the circuit
        with the minimal metric value.

    
        Args:

            circs ( List[Circuit] ) A list of Squander circuits to be compared

            parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

            metric (optional) The metric function to decide which input circuit is better.


        Return:

            Returns with the chosen circuit and the corresponding parameter array

    
        """ 

        if not isinstance( circs, list ):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance( parameter_arrs, list ):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs) :
            raise Exception("The first two arguments should be of the same length")

        metrics = [metric( circ ) for circ in circs]

        metrics = np.array( metrics )

        min_idx = np.argmin( metrics )

        return circs[ min_idx ], parameter_arrs[ min_idx ]



    @staticmethod
    def PartitionDecompositionProcess( subcircuit: Circuit, subcircuit_parameters: np.ndarray, config: dict, structure=None ) -> Tuple[Circuit, np.ndarray]:
        """
        Implements an asynchronous process to decompose a unitary associated with a partition in a large 
        quantum circuit

    
        Args:

            circ ( Circuit ) A subcircuit representing a partition

            parameters ( np.ndarray ) A parameter array associated with the input circuit

        
        """             

        qbit_num_orig_circuit = subcircuit.get_Qbit_Num()

        involved_qbits = subcircuit.get_Qbits()

        qbit_num = len( involved_qbits )

        # create qbit map:
        qbit_map = {}
        for idx in range( len(involved_qbits) ):
            qbit_map[ involved_qbits[idx] ] = idx
        mini_topology = None 
        if config["topology"] != None:
            mini_topology = extract_subtopology(involved_qbits, qbit_map, config)
        # remap the subcircuit to a smaller qubit register
        remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )

        if qbit_num > 3 and structure is None and config.get("strategy", "") == "TreeGuided":
            circo = Circuit(qbit_num)
            for gate in remapped_subcircuit.get_Gates(): circo.add_Gate(gate)
            remapped_subcircuit = circo
            partitioned_circuit, params, recombine_info = qgd_Wide_Circuit_Optimization.make_all_partition_circuit(remapped_subcircuit, subcircuit_parameters, 3)
            optimized_circuits = []
            subcircs = partitioned_circuit.get_Gates()
            #first find the optimal CNOT decomposition
            for innercirc in subcircs:
                start_idx = innercirc.get_Parameter_Start_Index()
                innercirc_parameters = params[ start_idx:start_idx+innercirc.get_Parameter_Num() ]
                callback_fnc = lambda  x : qgd_Wide_Circuit_Optimization.CompareAndPickCircuits( [innercirc, *(z[0] for z in x)], [innercirc_parameters, *(z[1] for z in x)] )
                optimized_circuits.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": True, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}, structure=None)))
            parts, struct_idxs = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(remapped_subcircuit, 3, [x[0] for x in optimized_circuits], recombine_info)
            #enumerate all solutions for each subcircuit in the optimal
            all_sol_for_idx = []
            for idx in struct_idxs:
                innercirc = subcircs[idx]
                start_idx = innercirc.get_Parameter_Start_Index()
                innercirc_parameters = params[ start_idx:start_idx+innercirc.get_Parameter_Num() ]
                callback_fnc = lambda  x : x + [(innercirc, innercirc_parameters)]
                all_sol_for_idx.append(callback_fnc(qgd_Wide_Circuit_Optimization.PartitionDecompositionProcess(innercirc, innercirc_parameters, {**config, "stop_first_solution": False, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0))}, structure=None)))
            all_decomposed = []
            import itertools
            opt = qgd_Wide_Circuit_Optimization({**config, "max_partition_size": 3})
            if np.prod([len(x) for x in all_sol_for_idx]) > 32:
                import random
                trycombs = [[random.choice(x) for x in all_sol_for_idx] for _ in range(32)]
            else: trycombs = itertools.product(*all_sol_for_idx)
            for combination in trycombs:
                structures = [qgd_Wide_Circuit_Optimization.copy_circuit_structure(x[0]) for x in combination]
                optcirc, optparams = opt._OptimizeWideCircuit(remapped_subcircuit, subcircuit_parameters, False, parts, structures)
                reoptcirc, reoptparams = opt._OptimizeWideCircuit(optcirc.get_Flat_Circuit(), optparams)
                all_decomposed.append((reoptcirc.get_Flat_Circuit(), reoptparams))
        else:
            if not structure is None:
                structure = structure.Remap_Qbits( qbit_map, qbit_num )

            # get the unitary representing the circuit
            unitary = remapped_subcircuit.get_Matrix( subcircuit_parameters )
            config_mini = config.copy()
            config_mini['level_limit'] = max(0, remapped_subcircuit.get_Gate_Nums().get('CNOT', 0))
            config_mini['tree_level_max'] = max(0, remapped_subcircuit.get_Gate_Nums().get('CNOT', 0))
            if config['strategy']=='HybridSearch':
                if qbit_num==2 or remapped_subcircuit.get_Gate_Nums().get('CNOT', 0)<5:
                    config_mini['strategy']='TreeSearch'
                else:
                    config_mini['strategy']='SurrSearch'

            # decompose a small unitary into a new circuit
            all_decomposed = qgd_Wide_Circuit_Optimization.DecomposePartition( unitary, config_mini, mini_topology, structure=structure )
        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[ value ] = key
        result = []
        for decomposed_circuit, decomposed_parameters in all_decomposed:

            # remap the decomposed circuit in order to insert it into a large circuit
            new_subcircuit = decomposed_circuit.Remap_Qbits( inverse_qbit_map, qbit_num_orig_circuit )


            if config["test_subcircuits"]:
                CompareCircuits( subcircuit, subcircuit_parameters, new_subcircuit, decomposed_parameters, parallel=config["parallel"] )
            


            new_subcircuit = new_subcircuit.get_Flat_Circuit()
            result.append((new_subcircuit, decomposed_parameters))
        return result

    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, max_partition_size)
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}
        partitined_circuit = Circuit( qbit_num_orig_circuit )
        params = []
        for part in allparts:
            surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded_chains))
            #topo sort part + surrounded chains
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates}, {x: rgo[x] & gates for x in gates}):
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitined_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in chain:
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitined_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return partitined_circuit, parameters, (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit)
    def copy_circuit_structure(structure):
        from squander.gates.qgd_Circuit import CNOT
        newcirc = Circuit(structure.get_Qbit_Num())
        for gate in structure.get_Gates():
            if isinstance(gate, CNOT):
                newcirc.add_U3(gate.get_Target_Qbit())
                newcirc.add_U3(gate.get_Control_Qbit())
                newcirc.add_Gate(gate)
        for qbit in structure.get_Qbits():
            newcirc.add_U3(qbit)
        return newcirc
    def recombine_all_partition_circuit(circ, max_partition_size, optimized_subcircuits, recombine_info ):
        from squander.partitioning.ilp import topo_sort_partitions, ilp_global_optimal, recombine_single_qubit_chains
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = recombine_info
        max_gates = sum(sum(y for x, y in c.get_Gate_Nums().items() if x !='CNOT') for c in optimized_subcircuits[:len(allparts)])        
        weights = [CNOTGateCount(circ, max_gates) for circ in optimized_subcircuits[:len(allparts)]]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L], fusion_info)
        single_qubit_chain_idx = {frozenset(chain): idx + len(allparts) for idx, chain in enumerate(single_qubit_chains)}
        for extrapart in parts[len(struct_idxs):]:
            struct_idxs.append(single_qubit_chain_idx[extrapart])
        L = topo_sort_partitions(circ, max_partition_size, parts)
        return [parts[i] for i in L], [struct_idxs[i] for i in L]

    def OptimizeWideCircuit( self, circ: Circuit, parameters: np.ndarray, global_min=True ) -> Tuple[Circuit, np.ndarray]:
        part_size_start = self.max_partition_size
        part_size_end = part_size_start
        count = CNOTGateCount(circ, 0)
        fingerprint_dict = {}
        for max_part_size in range(part_size_start, part_size_end + 1):
            # instantiate the object for optimizing wide circuits
            wide_circuit_optimizer = qgd_Wide_Circuit_Optimization( {**self.config, 'max_partition_size': max_part_size} )
            while True:
                # run circuit optimization
                circ_flat, parameters = wide_circuit_optimizer._OptimizeWideCircuit( circ, parameters, global_min=global_min, fingerprint_dict=fingerprint_dict )
                circ = circ_flat.get_Flat_Circuit()
                newcount = CNOTGateCount(circ, 0)
                no_improve = newcount >= count
                count = newcount
                if no_improve: break
        return circ, parameters
    def _OptimizeWideCircuit( self, circ: Circuit, orig_parameters: np.ndarray, global_min=True, prepartitioning=None, structures=None, fingerprint_dict=None ) -> Tuple[Circuit, np.ndarray]:
        """
        Call to optimize a wide circuit (i.e. circuits with many qubits) by
        partitioning the circuit into smaller partitions and redecompose the smaller partitions


        Args: 

            circ ( Circuit ) A circuit to be partitioned

            orig_parameters ( np.ndarray ) A parameter array associated with the input circuit

        Return:

            Returns with the optimized circuit and the corresponding parameter array

        """
        from squander.utils import circuit_to_CNOT_basis
        circ, orig_parameters = circuit_to_CNOT_basis(circ, orig_parameters)
        max_gates = sum(y for x, y in circ.get_Gate_Nums().items() if x !='CNOT')
        if self.config["topology"] != None and self.config["routed"]==False:
            circ, orig_parameters = self.route_circuit(circ,orig_parameters)

        if global_min:
            partitined_circuit, parameters, recombine_info = qgd_Wide_Circuit_Optimization.make_all_partition_circuit(circ, orig_parameters, self.max_partition_size)

        elif prepartitioning is not None:
            from squander.partitioning.kahn import kahn_partition_preparts
            from squander.partitioning.tools import translate_param_order
            partitined_circuit, param_order, _ = kahn_partition_preparts(circ, self.max_partition_size, prepartitioning)
            parameters = translate_param_order(orig_parameters, param_order)
        else:
            partitined_circuit, parameters, _ = PartitionCircuit( circ, orig_parameters, self.max_partition_size, strategy=self.config['partition_strategy'] )

        qbit_num_orig_circuit = circ.get_Qbit_Num()


        subcircuits = partitined_circuit.get_Gates()

        #subcircuits = subcircuits[9:10]

        if parent_process() is None: print(len(subcircuits), "partitions found to optimize")


        # the list of optimized subcircuits
        optimized_subcircuits = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list = [None] * len(subcircuits)

        def get_fingerprint(circ, params):
            return tuple((gate.get_Name(), tuple(gate.get_Involved_Qbits())) for gate in circ.get_Gates()) + tuple(params)

        if parent_process() is not None:
            #  code for iterate over partitions and optimize them
            for partition_idx, subcircuit in enumerate( subcircuits ):
        

                # isolate the parameters corresponding to the given sub-circuit
                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = start_idx + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
            
                # callback function done on the master process to compare the new decomposed and the original suncircuit
                callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, *(z[0] for z in x)], [subcircuit_parameters, *(z[1] for z in x)], lambda c: CNOTGateCount(c, max_gates) )

                # call a process to decompose a subcircuit
                fingerprint = None if fingerprint_dict is None else get_fingerprint(subcircuit, subcircuit_parameters)
                if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                    new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
                else:
                    config = {**self.config, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}
                    config = config if structures is None or partition_idx >= len(structures) else {**config, 'strategy': 'Custom', 'max_inner_iterations': 10000, 'max_iteration_loops': 4}                
                    new_subcircuit, new_parameters = callback_fnc(self.PartitionDecompositionProcess( subcircuit, subcircuit_parameters, config,
                                                                                        None if structures is None or partition_idx >= len(structures) else structures[partition_idx] ))
                    if subcircuit != new_subcircuit:

                        print( "original subcircuit:    ", subcircuit.get_Gate_Nums(), partition_idx) 
                        print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums())
                    if fingerprint_dict is not None:
                        fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                        fingerprint_dict[get_fingerprint(new_subcircuit, new_parameters)] = (new_subcircuit, new_parameters)

                if partition_idx % 100 == 99: print(partition_idx+1, "partitions optimized")
                optimized_subcircuits[ partition_idx ] = new_subcircuit
                optimized_parameter_list[ partition_idx ] = new_parameters
        else:
            # list of AsyncResult objects
            async_results = [None] * len(subcircuits)
            with Pool(processes=mp.cpu_count()//2) as pool:

                #  code for iterate over partitions and optimize them
                for partition_idx, subcircuit in enumerate( subcircuits ):
            

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx   = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[ start_idx:end_idx ]
    
        
                    fingerprint = None if fingerprint_dict is None else get_fingerprint(subcircuit, subcircuit_parameters)
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict: continue
                    # call a process to decompose a subcircuit
                    config = {**self.config, 'tree_level_max': max(0, subcircuit.get_Gate_Nums().get('CNOT', 0)-1)}
                    config = config if structures is None or partition_idx >= len(structures) else {**config, 'strategy': 'Custom', 'max_inner_iterations': 10000, 'max_iteration_loops': 4}
                    async_results[partition_idx]  = pool.apply_async( self.PartitionDecompositionProcess, (subcircuit, subcircuit_parameters, config,
                                                                                                        None if structures is None or partition_idx >= len(structures) else structures[partition_idx]))
                #  code for iterate over async results and retrieve the new subcircuits
                for partition_idx, subcircuit in enumerate( subcircuits ):
                    # callback function done on the master process to compare the new decomposed and the original suncircuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    subcircuit_parameters = parameters[ start_idx:start_idx + subcircuit.get_Parameter_Num() ]
                    fingerprint = None if fingerprint_dict is None else get_fingerprint(subcircuit, subcircuit_parameters)
                    callback_fnc = lambda  x : self.CompareAndPickCircuits( [subcircuit, *(z[0] for z in x)], [subcircuit_parameters, *(z[1] for z in x)], lambda c: CNOTGateCount(c, max_gates) )
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                        new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
                    else:
                        new_subcircuit, new_parameters = callback_fnc(async_results[partition_idx].get( timeout = None ))

                        if subcircuit != new_subcircuit:

                            print( "original subcircuit:    ", subcircuit.get_Gate_Nums(), partition_idx) 
                            print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
                        if fingerprint_dict is not None:
                            fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                            fingerprint_dict[get_fingerprint(new_subcircuit, new_parameters)] = (new_subcircuit, new_parameters)
                    if partition_idx % 100 == 99: print(partition_idx+1, "partitions optimized")
                    optimized_subcircuits[ partition_idx ] = new_subcircuit
                    optimized_parameter_list[ partition_idx ] = new_parameters

        # construct the wide circuit from the optimized suncircuits
        if global_min:
            parts, struct_idxs = qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(circ, self.max_partition_size, optimized_subcircuits, recombine_info)
            structures = [qgd_Wide_Circuit_Optimization.copy_circuit_structure(optimized_subcircuits[x]) for x in struct_idxs]
            return self._OptimizeWideCircuit(circ, orig_parameters, global_min=False, prepartitioning=parts, structures=structures, fingerprint_dict=fingerprint_dict)
        
        wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )

        if parent_process() is None:
            print( "original circuit:    ", circ.get_Gate_Nums()) 
            print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 


        if self.config["test_final_circuit"]:
            CompareCircuits( partitined_circuit, parameters, wide_circuit, wide_parameters )
            #print("Test final circuit passed")

        
        return wide_circuit, wide_parameters

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):

        sabre = SABRE(circ, self.config["topology"])
        Squander_remapped_circuit, parameters_remapped_circuit, pi, final_pi, swap_count = sabre.map_circuit(orig_parameters)
        self.config.setdefault("initial_mapping",pi)
        self.config.setdefault("final_mapping",final_pi)
        self.config["routed"] = True
        return Squander_remapped_circuit, parameters_remapped_circuit

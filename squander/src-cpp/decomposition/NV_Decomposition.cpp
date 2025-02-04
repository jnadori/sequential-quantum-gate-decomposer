/*
Created on Fri Jun 26 14:13:26 2020
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

@author: Peter Rakyta, Ph.D.
*/
/*! \file NV_Decomposition.cpp
    \brief Class implementing the adaptive gate decomposition algorithm of arXiv:2203.04426
*/

#include "NV_Decomposition.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Random_Orthogonal.h"
#include "Random_Unitary.h"

#include "X.h"

#include <time.h>
#include <stdlib.h>






/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
NV_Decomposition::NV_Decomposition() : Optimization_Interface() {


    // set the level limit
    level_limit = 0;



    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }
    

    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;


}

/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
NV_Decomposition::NV_Decomposition( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {


    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        set_optimizer( BFGS );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        
        max_inner_iterations = 10000;

    }
    else {
        set_optimizer( ADAM );

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;

    }

    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;


}



/**
@brief Constructor of the class.
@param Umtx_in The unitary matrix to be decomposed
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param optimize_layer_num_in Optional logical value. If true, then the optimization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimization is performed for the maximal number of layers.
@param initial_guess_in Enumeration element indicating the method to guess initial values for the optimization. Possible values: 'zeros=0' ,'random=1', 'close_to_zero=2'
@param compression_enabled_in Optional logical value. If True(1) begin decomposition function will compress the circuit. If False(0) it will not. Compression can still be called in seperate wrapper function. 
@return An instance of the class
*/
NV_Decomposition::NV_Decomposition( Matrix Umtx_in, int qbit_num_in, int level_limit_in, int level_limit_min_in, std::vector<matrix_base<int>> topology_in, std::map<std::string, Config_Element>& config, int accelerator_num ) : Optimization_Interface(Umtx_in, qbit_num_in, false, config, RANDOM, accelerator_num) {



    // set the level limit
    level_limit = level_limit_in;
    level_limit_min = level_limit_min_in;

    // Maximal number of iteartions in the optimization process
    max_outer_iterations = 1;

    
    // setting the topology
    topology = topology_in;




    // BFGS is better for smaller problems, while ADAM for larger ones
    if ( qbit_num <= 5 ) {
        alg = BFGS;

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 4;
        max_inner_iterations = 10000;
    }
    else {
        alg = ADAM;

        // Maximal number of iteartions in the optimization process
        max_outer_iterations = 1;
    }

    // Boolean variable to determine whether randomized adaptive layers are used or not
    randomized_adaptive_layers = false;

}

/**
@brief Destructor of the class
*/
NV_Decomposition::~NV_Decomposition() {

}



/**
@brief Start the disentanglig process of the unitary
@param finalize_decomp Optional logical parameter. If true (default), the decoupled qubits are rotated into state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
@param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
*/
void
NV_Decomposition::start_decomposition(bool prepare_export) {


    //The stringstream input to store the output messages.
    std::stringstream sstream;
    sstream << "***************************************************************" << std::endl;
    sstream << "Starting to disentangle " << qbit_num << "-qubit matrix" << std::endl;
    sstream << "***************************************************************" << std::endl << std::endl << std::endl;

    print(sstream, 1);   



    // get the initial circuit including redundand 2-qbit blocks.
    get_initial_circuit();

}





/**
@brief Call to determine the initial gate structure (still conatining adaptive gates)
*/
void NV_Decomposition::get_initial_circuit() {
// temporarily turn off OpenMP parallelism
#if BLAS==0 // undefined BLAS
    num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    num_threads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    num_threads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif
    //measure the time for the decompositin
    tbb::tick_count start_time = tbb::tick_count::now();

    if (level_limit == 0 ) {
        std::stringstream sstream;
	sstream << "please increase level limit" << std::endl;
        print(sstream, 0);	
        return;
    }




    Gates_block* gate_structure_loc = NULL;
    if ( gates.size() > 0 ) {
        std::stringstream sstream;
        sstream << "Using imported gate structure for the decomposition." << std::endl;
        print(sstream, 1);	
        gate_structure_loc = optimize_imported_gate_structure(optimized_parameters_mtx);
    }
    else {
        std::stringstream sstream;
        sstream << "Construct initial gate structure for the decomposition." << std::endl;
        print(sstream, 1);
        gate_structure_loc = determine_initial_gate_structure(optimized_parameters_mtx);
    }


    long long export_circuit_2_binary_loc;
    if ( config.count("export_circuit_2_binary") > 0 ) {
        config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );  
    }
    else {
        export_circuit_2_binary_loc = 0;
    }     
        
        
    if ( export_circuit_2_binary_loc > 0 ) {
        std::string filename("circuit_squander.binary");
        if (project_name != "") {
            filename = project_name+ "_" +filename;
        }
        export_gate_list_to_binary(optimized_parameters_mtx, gate_structure_loc, filename, verbose);
        
        std::string unitaryname("unitary_squander.binary");
        if (project_name != "") {
            filename = project_name+ "_" +unitaryname;
        }
        export_unitary(unitaryname);
        
    }
    
    // store the created gate structure
    release_gates();
	combine( gate_structure_loc );
	delete( gate_structure_loc );
	
	
#if BLAS==0 // undefined BLAS
    omp_set_num_threads(num_threads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(num_threads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(num_threads);
#endif
}


/**
@brief Call to optimize an imported gate structure
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
NV_Decomposition::optimize_imported_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    Gates_block* gate_structure_loc = (static_cast<Gates_block*>(this))->clone();
        
    //measure the time for the decompositin
    tbb::tick_count start_time_loc = tbb::tick_count::now();

    std::stringstream sstream;
    sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
         
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }      

    // solve the optimization problem
    N_Qubit_Decomposition_custom cDecomp_custom;
    // solve the optimization problem in isolated optimization process
    cDecomp_custom = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, initial_guess, accelerator_num);
    cDecomp_custom.set_custom_gate_structure( gate_structure_loc );
    cDecomp_custom.set_optimized_parameters( optimized_parameters_mtx_loc.get_data(), optimized_parameters_mtx_loc.size() );
    cDecomp_custom.set_optimization_blocks( gate_structure_loc->get_gate_num() );
    cDecomp_custom.set_max_iteration( max_outer_iterations );
    cDecomp_custom.set_verbose(verbose);
    cDecomp_custom.set_cost_function_variant( cost_fnc );
    cDecomp_custom.set_debugfile("");
    cDecomp_custom.set_iteration_loops( iteration_loops );
    cDecomp_custom.set_optimization_tolerance( optimization_tolerance_loc ); 
    cDecomp_custom.set_trace_offset( trace_offset ); 
    cDecomp_custom.set_optimizer( alg );  
    cDecomp_custom.set_project_name( project_name );
    if (alg==ADAM || alg==BFGS2) { 
        int param_num_loc = gate_structure_loc->get_parameter_num();
        int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 10000 );          
    }
    else if ( alg==ADAM_BATCHED ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 2500;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );  
        cDecomp_custom.set_random_shift_count_max( 5 );  
    }
    else if ( alg==BFGS ) {
        cDecomp_custom.set_optimizer( alg );  
        int max_inner_iterations_loc = 10000;
        cDecomp_custom.set_max_inner_iterations( max_inner_iterations_loc );    
    }
    cDecomp_custom.start_decomposition(true);
    number_of_iters += cDecomp_custom.get_num_iters();
    //cDecomp_custom.list_gates(0);

    tbb::tick_count end_time_loc = tbb::tick_count::now();

    current_minimum = cDecomp_custom.get_current_minimum();
    optimized_parameters_mtx_loc = cDecomp_custom.get_optimized_parameters();



    if ( cDecomp_custom.get_current_minimum() < optimization_tolerance_loc ) {
        std::stringstream sstream;
	sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);	
    }   
    else {
        std::stringstream sstream;
	sstream << "Optimization problem converged to " << cDecomp_custom.get_current_minimum() << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
        print(sstream, 1);       
    }

    if (current_minimum > optimization_tolerance_loc) {
        std::stringstream sstream;
	sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl; 
        print(sstream, 1);             
        optimization_tolerance_loc = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }

    sstream.str("");
    sstream << "Continue with the compression of gate structure consisting of " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
    print(sstream, 1);	
    return gate_structure_loc;



}

/**
@brief Call determine the gate structrue of the decomposing circuit. (quantum circuit with CRY gates)
@param optimized_parameters_mtx_loc A matrix containing the initial parameters
*/
Gates_block* 
NV_Decomposition::determine_initial_gate_structure(Matrix_real& optimized_parameters_mtx_loc) {

    // strages to store the optimized minimums in case of different cirquit depths
    std::vector<double> minimum_vec;
    std::vector<Gates_block*> gate_structure_vec;
    std::vector<Matrix_real> optimized_parameters_vec;
    
    double optimization_tolerance_loc;
    if ( config.count("optimization_tolerance") > 0 ) {
        config["optimization_tolerance"].get_property( optimization_tolerance_loc );  
    }
    else {
        optimization_tolerance_loc = optimization_tolerance;
    }         
        
    


    int level = level_limit_min;
    while ( current_minimum > optimization_tolerance_loc && level <= level_limit) {

        // create gate structure to be optimized
        Gates_block* gate_structure_loc = new Gates_block(qbit_num);  
        
        optimized_parameters_mtx_loc = Matrix_real(0,0);
                   
        for (int idx=0; idx<level; idx++) {

            // create the new decomposing layer and add to the gate staructure
            add_nv_layers( gate_structure_loc );

        }
           
        // add finalyzing layer to the top of the gate structure
        add_finalyzing_layer( gate_structure_loc );
            

        //measure the time for the decompositin
        tbb::tick_count start_time_loc = tbb::tick_count::now();


        N_Qubit_Decomposition_custom cDecomp_custom_random, cDecomp_custom_close_to_zero;

        std::stringstream sstream;
        sstream << "Starting optimization with " << gate_structure_loc->get_gate_num() << " decomposing layers." << std::endl;
        print(sstream, 1);

	
                // solve the optimization problem in isolated optimization process
                cDecomp_custom_random = N_Qubit_Decomposition_custom( Umtx.copy(), qbit_num, false, config, RANDOM, accelerator_num);
                cDecomp_custom_random.set_custom_gate_structure( gate_structure_loc );
                cDecomp_custom_random.set_optimization_blocks( gate_structure_loc->get_gate_num() );
                cDecomp_custom_random.set_max_iteration( max_outer_iterations );
                cDecomp_custom_random.set_verbose(0);
                cDecomp_custom_random.set_cost_function_variant( cost_fnc );
                cDecomp_custom_random.set_debugfile("");
                cDecomp_custom_random.set_optimization_tolerance( optimization_tolerance_loc );
                cDecomp_custom_random.set_trace_offset( trace_offset ); 
                cDecomp_custom_random.set_optimizer( alg );
                cDecomp_custom_random.set_project_name( project_name );
                if ( alg == ADAM || alg == BFGS2 ) {
                    int param_num_loc = gate_structure_loc->get_parameter_num();
                    int max_inner_iterations_loc = (double)param_num_loc/852 * 1e7;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 10000 ); 
                }
                else if ( alg==ADAM_BATCHED ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 2000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                    cDecomp_custom_random.set_random_shift_count_max( 5 );   
                }
                else if ( alg==BFGS ) {
                    cDecomp_custom_random.set_optimizer( alg );  
                    int max_inner_iterations_loc = 10000;
                    cDecomp_custom_random.set_max_inner_iterations( max_inner_iterations_loc );  
                }
                
            
                cDecomp_custom_random.start_decomposition(true);
                

                
                number_of_iters += cDecomp_custom_random.get_num_iters(); // retrive the number of iterations spent on optimization
/*

*/
         tbb::tick_count end_time_loc = tbb::tick_count::now();
/*

*/
         double current_minimum_random         = cDecomp_custom_random.get_current_minimum();
         double current_minimum_close_to_zero = cDecomp_custom_close_to_zero.get_current_minimum();
         double current_minimum_loc;


         // select between the results obtained for different initial value strategy
         if ( current_minimum_random < optimization_tolerance_loc && current_minimum_close_to_zero > optimization_tolerance_loc ) {
             current_minimum_loc = current_minimum_random;
             optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
             initial_guess = RANDOM;
         }
         else if ( current_minimum_random > optimization_tolerance_loc && current_minimum_close_to_zero < optimization_tolerance_loc ) {
             current_minimum_loc = current_minimum_close_to_zero;
             optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
             initial_guess = CLOSE_TO_ZERO;
         }
         else if ( current_minimum_random < optimization_tolerance_loc && current_minimum_close_to_zero < optimization_tolerance_loc ) {
             Matrix_real optimized_parameters_mtx_random = cDecomp_custom_random.get_optimized_parameters();
             Matrix_real optimized_parameters_mtx_close_to_zero = cDecomp_custom_close_to_zero.get_optimized_parameters();

             int panelty_random         = get_panelty(gate_structure_loc, optimized_parameters_mtx_random);
             int panelty_close_to_zero = get_panelty(gate_structure_loc, optimized_parameters_mtx_close_to_zero );

             if ( panelty_random < panelty_close_to_zero ) {
                 current_minimum_loc = current_minimum_random;
                 optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                 initial_guess = RANDOM;
             }
             else {
                 current_minimum_loc = current_minimum_close_to_zero;
                 optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                 initial_guess = CLOSE_TO_ZERO;
             }

        }
        else {
           if ( current_minimum_random < current_minimum_close_to_zero ) {
                current_minimum_loc = current_minimum_random;
                optimized_parameters_mtx_loc = cDecomp_custom_random.get_optimized_parameters();
                initial_guess = RANDOM;
           }
           else {
                current_minimum_loc = current_minimum_close_to_zero;
                optimized_parameters_mtx_loc = cDecomp_custom_close_to_zero.get_optimized_parameters();
                initial_guess = CLOSE_TO_ZERO;
           }

        }

        minimum_vec.push_back(current_minimum_loc);
        gate_structure_vec.push_back(gate_structure_loc);
        optimized_parameters_vec.push_back(optimized_parameters_mtx_loc);



        if ( current_minimum_loc < optimization_tolerance_loc ) {
	    std::stringstream sstream;
            sstream << "Optimization problem solved with " << gate_structure_loc->get_gate_num() << " decomposing layers in " << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);	       
            break;
        }   
        else {
            std::stringstream sstream;
            sstream << "Optimization problem converged to " << current_minimum_loc << " with " <<  gate_structure_loc->get_gate_num() << " decomposing layers in "   << (end_time_loc-start_time_loc).seconds() << " seconds." << std::endl;
            print(sstream, 1);  
        }

        level++;
    }

//exit(-1);

    // find the best decomposition
    int idx_min = 0;
    double current_minimum = minimum_vec[0];
    for (int idx=1; idx<(int)minimum_vec.size(); idx++) {
        if( current_minimum > minimum_vec[idx] ) {
            idx_min = idx;
            current_minimum = minimum_vec[idx];
        }
    }
     

    Gates_block* gate_structure_loc = gate_structure_vec[idx_min];
    optimized_parameters_mtx_loc = optimized_parameters_vec[idx_min];

    // release unnecesarry data
    for (int idx=0; idx<(int)minimum_vec.size(); idx++) {
        if( idx == idx_min ) {
            continue;
        }
        delete( gate_structure_vec[idx] );
    }    
    minimum_vec.clear();
    gate_structure_vec.clear();
    optimized_parameters_vec.clear();
    


    if (current_minimum > optimization_tolerance_loc) {
       std::stringstream sstream;
       sstream << "Decomposition did not reached prescribed high numerical precision." << std::endl;
       print(sstream, 1);              
       optimization_tolerance_loc = 1.5*current_minimum < 1e-2 ? 1.5*current_minimum : 1e-2;
    }
    

    return gate_structure_loc;
       
}


/**
@brief Call to get the panelty derived from the number of CRY and CNOT gates in the circuit
@param gate_structure The gate structure to be optimized
@param optimized_parameters A matrix containing the initial parameters
*/
unsigned int 
NV_Decomposition::get_panelty( Gates_block* gate_structure, Matrix_real& optimized_parameters ) {


    int panelty = 0;

    // iterate over the elements of the parameter array
    int parameter_idx = 0;
    int layer_num = gate_structure->get_gate_num();
    //for ( int layer_idx=layer_num-1; layer_idx>=0; layer_idx--) {
    for ( int layer_idx=0; layer_idx<layer_num; layer_idx++) {
    
        Gates_block* layer = static_cast<Gates_block*>( gate_structure->get_gate( layer_idx ) );
    
        int gate_num = layer->get_gate_num();
        //for( int gate_idx=gate_num-1; gate_idx>=0; gate_idx-- ) {
        for( int gate_idx=0; gate_idx<gate_num; gate_idx++ ) {
        
            Gate* gate = layer->get_gate( gate_idx );

            double parameter = optimized_parameters[parameter_idx];
            parameter_idx = parameter_idx + gate->get_parameter_num(); 
 
            if ( gate->get_type() != ADAPTIVE_OPERATION ) {
               continue;
            }        
            
        
            if ( std::abs(std::sin(parameter)) < 0.999 && std::abs(std::cos(parameter)) < 1e-3 ) {
                // Condition of pure CNOT gate
                panelty += 1;
            }
            else if ( std::abs(std::sin(parameter)) < 1e-3 && std::abs(1-std::cos(parameter)) < 1e-3 ) {
                // Condition of pure Identity gate
                //panelty++;
            }
            else {
                // Condition of controlled rotation gate
                panelty += 2;
            }
        
        }

    }


    return panelty;


}




/**
@brief Call to add adaptive layers to the gate structure stored by the class.
*/
void 
NV_Decomposition::add_nv_layers() {

    add_nv_layers( this );

}

/**
@brief Call to add adaptive layers to the gate structure.
*/
void 
NV_Decomposition::add_nv_layers( Gates_block* gate_structure ) {


    // create the new decomposing layer and add to the gate staructure
    Gates_block* layer = construct_nv_gate_layers();
    gate_structure->combine( layer );


}




/**
@brief Call to construct adaptive layers.
*/
Gates_block* 
NV_Decomposition::construct_nv_gate_layers() {


    //The stringstream input to store the output messages.
    std::stringstream sstream;

    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );

    std::vector<Gates_block* > layers;


    if ( topology.size() > 0 ) {
        for ( std::vector<matrix_base<int>>::iterator it=topology.begin(); it!=topology.end(); it++) {

            if ( it->size() != 2 ) {
                std::stringstream sstream;
	        sstream << "The connectivity data should contains two qubits" << std::endl;
	        print(sstream, 0);	
                it->print_matrix();
                exit(-1);
            }

            int control_qbit_loc = (*it)[0];
            int target_qbit_loc = (*it)[1];

            if ( control_qbit_loc >= qbit_num || target_qbit_loc >= qbit_num ) {
                std::stringstream sstream;
	        sstream << "Label of control/target qubit should be less than the number of qubits in the register." << std::endl;	        
                print(sstream, 0);
                exit(-1);            
            }

            Gates_block* layer = new Gates_block( qbit_num );

            /*layer->add_rz(target_qbit_loc);
            layer->add_ry(target_qbit_loc);
            layer->add_rz(target_qbit_loc);
            layer->add_rz(control_qbit_loc);
            layer->add_ry(control_qbit_loc);
            layer->add_rz(control_qbit_loc);*/
            layer->add_u3(target_qbit_loc,true,true,true);
            layer->add_u3(control_qbit_loc,true,true,true);
            layer->add_crot(target_qbit_loc, control_qbit_loc);

            layers.push_back(layer);


        }
    }
    else {  

    }

/*
    for (int idx=0; idx<layers.size(); idx++) {
        Gates_block* layer = (Gates_block*)layers[idx];
        block->add_gate( layers[idx] );

    }
*/

    bool randomized_adaptive_layers_loc;
    if ( config.count("randomized_adaptive_layers") > 0 ) {
        config["randomized_adaptive_layers"].get_property( randomized_adaptive_layers_loc );  
    }
    else {
        randomized_adaptive_layers_loc = randomized_adaptive_layers;
    }


    // make difference between randomized adaptive layers and deterministic one
    if (randomized_adaptive_layers_loc) {

        std::uniform_int_distribution<> distrib_int(0, 5000);

        while (layers.size()>0) { 
            int idx = distrib_int(gen) % layers.size();

#ifdef __MPI__        
            MPI_Bcast( &idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
            block->add_gate( layers[idx] );
            layers.erase( layers.begin() + idx );
        }

    }
    else {
        while (layers.size()>0) { 
            block->add_gate( layers[0] );
            layers.erase( layers.begin() );
        }

    }


    return block;


}

/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure stored by the class.
*/
void 
NV_Decomposition::add_finalyzing_layer() {

    add_finalyzing_layer( this );

}

/**
@brief Call to add finalyzing layer (single qubit rotations on all of the qubits) to the gate structure.
*/
void 
NV_Decomposition::add_finalyzing_layer( Gates_block* gate_structure ) {


    // creating block of gates
    Gates_block* block = new Gates_block( qbit_num );
/*
    block->add_un();
    block->add_ry(qbit_num-1);
*/
    for (int idx=0; idx<qbit_num; idx++) {
            bool Theta = true;
            bool Phi = true;
            bool Lambda = true;
            block->add_u3(idx, Theta, Phi, Lambda);
//        block->add_ry(idx);
    }


    // adding the opeartion block to the gates
    if ( gate_structure == NULL ) {
        throw ("NV_Decomposition::add_finalyzing_layer: gate_structure is null pointer");
    }
    else {
        gate_structure->add_gate( block );
    }


}




/**
@brief Call to set custom layers to the gate structure that are intended to be used in the subdecomposition.
@param filename
*/
void 
NV_Decomposition::set_adaptive_gate_structure( std::string filename ) {

    if ( gates.size() > 0  ) {
        release_gates();
        optimized_parameters_mtx = Matrix_real(0,0);
    }

    Gates_block* gate_structure = import_gate_list_from_binary(optimized_parameters_mtx, filename, verbose);
    combine( gate_structure );
    delete gate_structure;

}

/**
@brief set unitary matrix from binary file
@param filename .binary file to import unitary from
*/
void 
NV_Decomposition::set_unitary_from_file( std::string filename ) {

    Umtx = import_unitary_from_binary(filename);

}
/**
@brief call to set Unitary from mtx
@param matrix to set over
*/
void 
NV_Decomposition::set_unitary( Matrix& Umtx_new ) {

    Umtx = Umtx_new;


}

/**
@brief Call to append custom layers to the gate structure that are intended to be used in the decomposition.
@param filename
*/
void 
NV_Decomposition::add_adaptive_gate_structure( std::string filename ) { 



    Matrix_real optimized_parameters_mtx_tmp;
    Gates_block* gate_structure_tmp = import_gate_list_from_binary(optimized_parameters_mtx_tmp, filename, verbose);

    if ( gates.size() > 0 ) {
        gate_structure_tmp->combine( static_cast<Gates_block*>(this) );

        release_gates();
        combine( gate_structure_tmp );
      

        Matrix_real optimized_parameters_mtx_tmp2( 1, optimized_parameters_mtx_tmp.size() + optimized_parameters_mtx.size() );
        
        memcpy( optimized_parameters_mtx_tmp2.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );
        memcpy( optimized_parameters_mtx_tmp2.get_data()+optimized_parameters_mtx.size(), optimized_parameters_mtx_tmp.get_data(), optimized_parameters_mtx_tmp.size()*sizeof(double) );
        
        optimized_parameters_mtx = optimized_parameters_mtx_tmp2;
    }
    else {
        combine( gate_structure_tmp );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp;
    }

}



/**
@brief Call to apply the imported gate structure on the unitary. The transformed unitary is to be decomposed in the calculations, and the imported gfate structure is released.
*/
void 
NV_Decomposition::apply_imported_gate_structure() {

    if ( gates.size() == 0 ) {
        return;
    }

    
    std::stringstream sstream;
    sstream << "The cost function before applying the imported gate structure is:" << optimization_problem( optimized_parameters_mtx )  << std::endl;   
    
    apply_to(  optimized_parameters_mtx, Umtx );
    release_gates();
    optimized_parameters_mtx = Matrix_real(0,0);
    

    sstream << "The cost function after applying the imported gate structure is:" << optimization_problem( optimized_parameters_mtx )  << std::endl;
    print(sstream, 3);	



}


/**
@brief Call to add an adaptive layer to the gate structure previously imported
@param filename
*/
void 
NV_Decomposition::add_layer_to_imported_gate_structure() {


    std::stringstream sstream;
    sstream << "Add new layer to the adaptive gate structure." << std::endl;	        
    print(sstream, 2);

    Gates_block* layer = construct_nv_gate_layers();


    combine( layer );

    Matrix_real tmp( 1, optimized_parameters_mtx.size() + layer->get_parameter_num() );
    memset( tmp.get_data(), 0, tmp.size()*sizeof(double) );
    memcpy( tmp.get_data(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );

    optimized_parameters_mtx = tmp;    

}








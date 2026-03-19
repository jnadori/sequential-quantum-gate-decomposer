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
/*! \file AGENTS_BFGS.cpp
    \brief Implementation of the AGENTS_BFGS optimization strategy:
           AGENTS with radius-excluded initialization and periodic BFGS polish.
*/

#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "Bayes_Opt.h"
#include "BFGS_Powell.h"

#include "RL_experience.h"

#include <fstream>
#include <algorithm>
#include <random>


#ifdef __DFE__
#include "common_DFE.h"
#endif



/**
@brief Call to solve layer by layer the optimization problem via the AGENTS_BFGS algorithm.
       This is AGENTS with two modifications:
       (A) Radius-excluded agent initialization (agents spread apart like POSMM)
       (B) Periodic BFGS polish of the best agent at each lifetime boundary
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_AGENTS_BFGS( int num_of_parameters, Matrix_real& solution_guess) {



        if ( (((cost_fnc != FROBENIUS_NORM) && (cost_fnc != HILBERT_SCHMIDT_TEST)) && cost_fnc != VQE  ) && (cost_fnc != INFIDELITY)) {
            std::string err("Optimization_Interface::solve_layer_optimization_problem_AGENTS_BFGS: Only cost functions 0 and 3 are implemented");
            throw err;
        }

#ifdef __DFE__
        if ( qbit_num >= 5 && get_accelerator_num() > 0 ) {
            upload_Umtx_to_DFE();
        }
#endif


        if (gates.size() == 0 ) {
            return;
        }


        double M_PI_quarter = M_PI/4;
        double M_PI_half = M_PI/2;
        double M_PI_double = M_PI*2;

        if (solution_guess.size() == 0 ) {
            solution_guess = Matrix_real(num_of_parameters,1);
            std::uniform_real_distribution<> distrib_real(0, M_PI_double);
            for ( int idx=0; idx<num_of_parameters; idx++) {
                solution_guess[idx] = distrib_real(gen);
            }

        }


#ifdef __MPI__
        MPI_Bcast( (void*)solution_guess.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif





        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
        }



        tbb::tick_count optimization_start = tbb::tick_count::now();
        double optimization_time = 0.0;





        current_minimum =   optimization_problem( optimized_parameters_mtx );

        int max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_agent") > 0 ) {
             long long value;
             config["max_inner_iterations_agent"].get_property( value );
             max_inner_iterations_loc = (int) value;
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
             long long value;
             config["max_inner_iterations"].get_property( value );
             max_inner_iterations_loc = (int) value;
        }
        else {
            max_inner_iterations_loc = max_inner_iterations;
        }


        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_agent") > 0 ) {
             config["optimization_tolerance_agent"].get_property( optimization_tolerance_loc );
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }


        long long export_circuit_2_binary_loc;
        if ( config.count("export_circuit_2_binary_agent") > 0 ) {
             config["export_circuit_2_binary_agent"].get_property( export_circuit_2_binary_loc );
        }
        else if ( config.count("export_circuit_2_binary") > 0 ) {
             config["export_circuit_2_binary"].get_property( export_circuit_2_binary_loc );
        }
        else {
            export_circuit_2_binary_loc = 0;
        }


        int agent_lifetime_loc;
        if ( config.count("agent_lifetime_agent") > 0 ) {
             long long agent_lifetime_loc_tmp;
             config["agent_lifetime_agent"].get_property( agent_lifetime_loc_tmp );
             agent_lifetime_loc = (int)agent_lifetime_loc_tmp;
        }
        else if ( config.count("agent_lifetime") > 0 ) {
             long long agent_lifetime_loc_tmp;
             config["agent_lifetime"].get_property( agent_lifetime_loc_tmp );
             agent_lifetime_loc = (int)agent_lifetime_loc_tmp;
        }
        else {
            agent_lifetime_loc = 1000;
        }



        std::stringstream sstream;
        sstream << "max_inner_iterations: " << max_inner_iterations_loc << std::endl;
        sstream << "agent_lifetime_loc: " << agent_lifetime_loc << std::endl;
        print(sstream, 2);

        double agent_randomization_rate_loc = 0.2;
        if ( config.count("aagent_randomization_rate_agent") > 0 ) {

             config["agent_randomization_rate_agent"].get_property( agent_randomization_rate_loc );
        }
        else if ( config.count("agent_randomization_rate") > 0 ) {
             config["agent_randomization_rate"].get_property( agent_randomization_rate_loc );
        }



        int agent_num;
        if ( config.count("agent_num_agent") > 0 ) {
             long long value;
             config["agent_num_agent"].get_property( value );
             agent_num = (int) value;
        }
        else if ( config.count("agent_num") > 0 ) {
             long long value;
             config["agent_num"].get_property( value );
             agent_num = (int) value;
        }
        else {
            agent_num = 64;
        }


        double agent_exploration_rate = 0.2;
        if ( config.count("agent_exploration_rate_agent") > 0 ) {

             config["agent_exploration_rate_agent"].get_property( agent_exploration_rate );
        }
        else if ( config.count("agent_exploration_rate") > 0 ) {
             config["agent_exploration_rate"].get_property( agent_exploration_rate );
        }

        int convergence_length = 20;
        if ( config.count("convergence_length_agent") > 0 ) {
             long long value;
             config["convergence_length_agent"].get_property( value );
             convergence_length = (int) value;
        }
        else if ( config.count("convergence_length") > 0 ) {
             long long value;
             config["convergence_length"].get_property( value );
             convergence_length = (int) value;
        }

        int linesearch_points = 3;
        if ( config.count("linesearch_points_agent") > 0 ) {
             long long value;
             config["linesearch_points_agent"].get_property( value );
             linesearch_points = (int) value;
        }
        else if ( config.count("linesearch_points") > 0 ) {
             long long value;
             config["linesearch_points"].get_property( value );
             linesearch_points = (int) value;
        }



        // The number if iterations after which the current results are displed/exported
        int output_periodicity;
        if ( config.count("output_periodicity_cosine") > 0 ) {
             long long value = 1;
             config["output_periodicity_cosine"].get_property( value );
             output_periodicity = (int) value;
        }
        if ( config.count("output_periodicity") > 0 ) {
             long long value = 1;
             config["output_periodicity"].get_property( value );
             output_periodicity = (int) value;
        }
        else {
            output_periodicity = 0;
        }

        // AGENTS_BFGS-specific config: radius_base for exclusion zone
        double radius_base_agent = 0.5;
        if ( config.count("radius_base_agent") > 0 ) {
             config["radius_base_agent"].get_property( radius_base_agent );
        }

        // AGENTS_BFGS-specific config: BFGS polish iterations
        long long bfgs_polish_iters_agent = 2000;
        if ( config.count("bfgs_polish_iters_agent") > 0 ) {
             config["bfgs_polish_iters_agent"].get_property( bfgs_polish_iters_agent );
        }

        // AGENTS_BFGS-specific config: number of top agents to BFGS-polish
        int bfgs_polish_top_k = 8;
        if ( config.count("bfgs_polish_top_k_agent") > 0 ) {
             long long value;
             config["bfgs_polish_top_k_agent"].get_property( value );
             bfgs_polish_top_k = (int) value;
        }
        if ( bfgs_polish_top_k > agent_num ) bfgs_polish_top_k = agent_num;

        // AGENTS_BFGS-specific config: how often to run BFGS polish (default: quarter the agent lifetime)
        int bfgs_polish_period = agent_lifetime_loc / 4;
        if ( config.count("bfgs_polish_period_agent") > 0 ) {
             long long value;
             config["bfgs_polish_period_agent"].get_property( value );
             bfgs_polish_period = (int) value;
        }
        if ( bfgs_polish_period < 1 ) bfgs_polish_period = 1;

        // Re-diversification: how many top agents to keep at lifetime boundaries
        int rediversify_keep_top_agent = 3;
        if ( config.count("rediversify_keep_top_agent") > 0 ) {
             long long value;
             config["rediversify_keep_top_agent"].get_property( value );
             rediversify_keep_top_agent = (int) value;
        }
        if ( rediversify_keep_top_agent > agent_num ) rediversify_keep_top_agent = agent_num;


        // BFGS polish: minimum improvement to avoid re-polishing stale agents
        double bfgs_polish_min_improvement_agent = 1e-4;
        if ( config.count("bfgs_polish_min_improvement_agent") > 0 ) {
             config["bfgs_polish_min_improvement_agent"].get_property( bfgs_polish_min_improvement_agent );
        }

        // Two-phase design: cosine scout iterations (Phase 1 cap)
        // Default: 3 agent lifetimes or max_inner_iterations, whichever is smaller
        int cosine_scout_iters_agent = std::min(max_inner_iterations_loc, agent_lifetime_loc * 3);
        if ( config.count("cosine_scout_iters_agent") > 0 ) {
             long long value;
             config["cosine_scout_iters_agent"].get_property( value );
             cosine_scout_iters_agent = (int) value;
        }

        // Stagnation detector: check period (default: quarter the agent lifetime)
        int stagnation_check_period_agent = agent_lifetime_loc / 4;
        if ( config.count("stagnation_check_period_agent") > 0 ) {
             long long value;
             config["stagnation_check_period_agent"].get_property( value );
             stagnation_check_period_agent = (int) value;
        }
        if ( stagnation_check_period_agent < 1 ) stagnation_check_period_agent = 1;

        // Stagnation detector: relative improvement threshold (default: 0.1%)
        double stagnation_threshold_agent = 1e-3;
        if ( config.count("stagnation_threshold_agent") > 0 ) {
             config["stagnation_threshold_agent"].get_property( stagnation_threshold_agent );
        }

        // Two-phase design: number of top diverse candidates for Phase 2 BFGS exploit
        int bfgs_exploit_top_k_agent = 16;
        if ( config.count("bfgs_exploit_top_k_agent") > 0 ) {
             long long value;
             config["bfgs_exploit_top_k_agent"].get_property( value );
             bfgs_exploit_top_k_agent = (int) value;
        }
        if ( bfgs_exploit_top_k_agent > agent_num ) bfgs_exploit_top_k_agent = agent_num;

        // Two-phase design: BFGS iterations per candidate in Phase 2
        long long bfgs_exploit_iters_agent = 5000;
        if ( config.count("bfgs_exploit_iters_agent") > 0 ) {
             config["bfgs_exploit_iters_agent"].get_property( bfgs_exploit_iters_agent );
        }

        // Visited basins: max size before pruning
        int max_visited_basins_agent = 200;
        if ( config.count("max_visited_basins_agent") > 0 ) {
             long long value;
             config["max_visited_basins_agent"].get_property( value );
             max_visited_basins_agent = (int) value;
        }

        // Basin-hopping: number of hops per candidate in Phase 2
        int bh_hops_agent = 10;
        if ( config.count("bh_hops_agent") > 0 ) {
             long long value;
             config["bh_hops_agent"].get_property( value );
             bh_hops_agent = (int) value;
        }

        // Basin-hopping: Metropolis temperature
        double bh_T_agent = 1.0;
        if ( config.count("bh_T_agent") > 0 ) {
             config["bh_T_agent"].get_property( bh_T_agent );
        }

        // Basin-hopping: initial perturbation stepsize (fraction of pi)
        double bh_stepsize_agent = 0.5;
        if ( config.count("bh_stepsize_agent") > 0 ) {
             config["bh_stepsize_agent"].get_property( bh_stepsize_agent );
        }

        // Basin-hopping: stepsize adaptation interval
        int bh_interval_agent = 5;
        if ( config.count("bh_interval_agent") > 0 ) {
             long long value;
             config["bh_interval_agent"].get_property( value );
             bh_interval_agent = std::max(1, (int) value);
        }

        // Basin-hopping: target acceptance rate
        double bh_target_accept_agent = 0.5;
        if ( config.count("bh_target_accept_agent") > 0 ) {
             config["bh_target_accept_agent"].get_property( bh_target_accept_agent );
        }
        bh_target_accept_agent = std::min(0.999, std::max(0.001, bh_target_accept_agent));

        // Basin-hopping: stepsize adaptation factor
        double bh_stepwise_factor_agent = 0.9;
        if ( config.count("bh_stepwise_factor_agent") > 0 ) {
             config["bh_stepwise_factor_agent"].get_property( bh_stepwise_factor_agent );
        }
        if (!(bh_stepwise_factor_agent > 0.0 && bh_stepwise_factor_agent < 1.0)) bh_stepwise_factor_agent = 0.9;

        sstream.str("");
        sstream << "AGENTS_BFGS: number of agents " << agent_num << std::endl;
        sstream << "AGENTS_BFGS: exploration_rate " << agent_exploration_rate << std::endl;
        sstream << "AGENTS_BFGS: lifetime " << agent_lifetime_loc << std::endl;
        sstream << "AGENTS_BFGS: radius_base_agent " << radius_base_agent << std::endl;
        sstream << "AGENTS_BFGS: bfgs_polish_iters " << bfgs_polish_iters_agent << std::endl;
        sstream << "AGENTS_BFGS: bfgs_polish_top_k " << bfgs_polish_top_k << std::endl;
        sstream << "AGENTS_BFGS: bfgs_polish_period " << bfgs_polish_period << std::endl;
        sstream << "AGENTS_BFGS: rediversify_keep_top " << rediversify_keep_top_agent << std::endl;
        sstream << "AGENTS_BFGS: cosine_scout_iters " << cosine_scout_iters_agent << std::endl;
        sstream << "AGENTS_BFGS: stagnation_check_period " << stagnation_check_period_agent << std::endl;
        sstream << "AGENTS_BFGS: stagnation_threshold " << stagnation_threshold_agent << std::endl;
        sstream << "AGENTS_BFGS: bfgs_exploit_top_k " << bfgs_exploit_top_k_agent << std::endl;
        sstream << "AGENTS_BFGS: bfgs_exploit_iters " << bfgs_exploit_iters_agent << std::endl;
        sstream << "AGENTS_BFGS: max_visited_basins " << max_visited_basins_agent << std::endl;
        sstream << "AGENTS_BFGS: bh_hops " << bh_hops_agent << std::endl;
        sstream << "AGENTS_BFGS: bh_T " << bh_T_agent << std::endl;
        sstream << "AGENTS_BFGS: bh_stepsize " << bh_stepsize_agent << std::endl;
        print(sstream, 2);


        bool terminate_optimization = false;

        // vector stroing the lates values of current minimums to identify convergence
        Matrix_real current_minimum_vec(1, convergence_length);
        memset( current_minimum_vec.get_data(), 0, current_minimum_vec.size()*sizeof(double) );
        double current_minimum_mean = 0.0;
        int current_minimum_idx = 0;

        double var_current_minimum = DBL_MAX;

        // Stagnation detector state
        double best_cost_at_last_check = DBL_MAX;
        bool phase1_stagnated = false;

        matrix_base<int> param_idx_agents( agent_num, 1 );

        // random generator of integers
        std::uniform_int_distribution<> distrib_int(0, num_of_parameters-1);

        for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
            // initital paraneter index of the agents
            param_idx_agents[ agent_idx ] = distrib_int(gen);
        }

#ifdef __MPI__
        MPI_Bcast( (void*)param_idx_agents.get_data(), agent_num, MPI_INT, 0, MPI_COMM_WORLD);
#endif


        int most_successfull_agent = 0;



tbb::tick_count t0_CPU = tbb::tick_count::now();

        // vector storing the parameter set used by the individual agents.

        std::vector<Matrix_real> solution_guess_mtx_agents( agent_num );
        solution_guess_mtx_agents.reserve( agent_num );

        std::uniform_real_distribution<> distrib_real(0.0, M_PI_double);

        //
        // MODIFICATION A: Radius-excluded agent initialization
        //
        // Agent 0 keeps the solution guess.
        // Agents 1..N: generate random points in [0, 2pi)^N,
        // rejecting any within radius of previously accepted agent starting points.
        //
        double init_radius = radius_base_agent * std::pow(std::log(2.0) / 1.0, 1.0 / (double)num_of_parameters);

        // Store accepted starting points for exclusion checks (position + cost)
        std::vector<std::pair<Matrix_real, double>> visited_basins;
        visited_basins.reserve(agent_num);

        // Dedup threshold for visited_basins insertion
        double visited_basins_dedup_radius = init_radius * 0.5;

        for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {


            // initialize random parameters for the agent
            Matrix_real solution_guess_mtx_agent = Matrix_real( num_of_parameters, 1 );
            memset( solution_guess_mtx_agent.get_data(), 0, solution_guess_mtx_agent.size()*sizeof(double) );

#ifdef __MPI__
            if ( current_rank == 0 ) {
#endif

                if ( agent_idx == 0 ) {
                    memcpy( solution_guess_mtx_agent.get_data(), solution_guess.get_data(), solution_guess.size()*sizeof(double) );
                }
                else {
                    // Radius-excluded random initialization
                    bool found = false;
                    for (int attempt = 0; attempt < 500; attempt++) {
                        Matrix_real candidate(1, num_of_parameters);
                        for (int idx = 0; idx < num_of_parameters; idx++) {
                            candidate[idx] = distrib_real(gen);
                        }

                        bool too_close = false;
                        for (int pdx = 0; pdx < (int)visited_basins.size(); pdx++) {
                            if (calculate_distance(candidate, visited_basins[pdx].first) < init_radius) {
                                too_close = true;
                                break;
                            }
                        }

                        if (!too_close) {
                            memcpy(solution_guess_mtx_agent.get_data(), candidate.get_data(), num_of_parameters * sizeof(double));
                            found = true;
                            break;
                        }
                    }

                    // Fallback: fully random if rejection sampling exhausted
                    if (!found) {
                        for (int idx = 0; idx < num_of_parameters; idx++) {
                            solution_guess_mtx_agent[idx] = distrib_real(gen);
                        }
                    }
                }


#ifdef __MPI__
            }

            MPI_Bcast( solution_guess_mtx_agent.get_data(), num_of_parameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

            // Track accepted starting point for exclusion (cost computed later)
            Matrix_real accepted_copy(1, num_of_parameters);
            memcpy(accepted_copy.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters * sizeof(double));
            visited_basins.push_back(std::make_pair(accepted_copy, DBL_MAX));

            solution_guess_mtx_agents[ agent_idx ] = solution_guess_mtx_agent;

        }




        // array storing the current minimum of th eindividual agents
        Matrix_real current_minimum_agents;

        // intitial cost function for each of the agents
        current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents );

        // Track last polish cost per agent for skip-stale check (Change 3)
        Matrix_real last_polish_cost_agents(agent_num, 1);
        for (int i = 0; i < agent_num; i++) last_polish_cost_agents[i] = DBL_MAX;

        // arrays to store some parameter values needed to be restored later
        Matrix_real parameter_value_save_agents( agent_num, 1 );

        // arrays to store the cost functions at shifted parameters
        Matrix_real f0_shifted_pi2_agents;
        Matrix_real f0_shifted_pi_agents;
        Matrix_real f0_shifted_pi4_agents;
        Matrix_real f0_shifted_3pi2_agents;


        bool three_point_line_search               =    cost_fnc == FROBENIUS_NORM;
        bool three_point_line_search_double_period =    cost_fnc == VQE && linesearch_points == 3;
        bool five_point_line_search                =    cost_fnc == HILBERT_SCHMIDT_TEST || ( cost_fnc == VQE && linesearch_points == 5 );



        // CPU time
        CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

        // Phase 1 cap: min of max_inner_iterations and cosine_scout_iters
        long long phase1_iters = std::min((long long)max_inner_iterations_loc, (long long)cosine_scout_iters_agent);

        ///////////////////////////////////////////////////////////////////////////
        for (long long iter_idx=0; iter_idx<phase1_iters; iter_idx++) {


            // CPU time
            t0_CPU = tbb::tick_count::now();


#ifdef __MPI__

            memset( param_idx_agents.get_data(), 0, param_idx_agents.size()*sizeof(int) );
            memset( parameter_value_save_agents.get_data(), 0.0, parameter_value_save_agents.size()*sizeof(double) );

            if ( current_rank == 0 ) {
#endif

                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {

                    // agent local parameter set
                    Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];

                    // determine parameter indices to be altered
                    int param_idx       = distrib_int(gen);
                    param_idx_agents[agent_idx] = param_idx;

                    // save the parameters to  be restored later
                    parameter_value_save_agents[agent_idx] = solution_guess_mtx_agent[param_idx];


                }

#ifdef __MPI__
            }

            MPI_Bcast( (void*)param_idx_agents.get_data(), agent_num, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast( (void*)parameter_value_save_agents.get_data(), agent_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


            if ( three_point_line_search ) {

                // calsulate the cist functions at shifted parameter values
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );

                // CPU time
                t0_CPU = tbb::tick_count::now();


                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi_agents = optimization_problem_batched( solution_guess_mtx_agents );


                // CPU time
                t0_CPU = tbb::tick_count::now();


                // determine the parameters of the cosine function and determine the parameter shift at the minimum
                for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {

                    double current_minimum_agent = current_minimum_agents[agent_idx];
                    double f0_shifted_pi         = f0_shifted_pi_agents[agent_idx];
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[agent_idx];


                    double A_times_cos = (current_minimum_agent-f0_shifted_pi)/2;
                    double offset      = (current_minimum_agent+f0_shifted_pi)/2;

                    double A_times_sin = offset - f0_shifted_pi2;

                    double phi0 = atan2( A_times_sin, A_times_cos);


                    double parameter_shift = phi0 > 0 ? M_PI-phi0 : -phi0-M_PI;
                    double amplitude = sqrt(A_times_sin*A_times_sin + A_times_cos*A_times_cos);


                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift;

                    current_minimum_agents[agent_idx] = offset - amplitude;

                }
            }
            else if ( three_point_line_search_double_period ) {


                // calsulate the cist functions at shifted parameter values
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );

                // CPU time
                t0_CPU = tbb::tick_count::now();


                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi_agents = optimization_problem_batched( solution_guess_mtx_agents );


                // CPU time
                t0_CPU = tbb::tick_count::now();


                // determine the parameters of the cosine function and determine the parameter shift at the minimum
                for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {

                    double current_minimum_agent = current_minimum_agents[agent_idx];
                    double f0_shifted_pi         = f0_shifted_pi_agents[agent_idx];
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[agent_idx];


                    double A_times_cos = (current_minimum_agent-f0_shifted_pi)/2;
                    double offset      = (current_minimum_agent+f0_shifted_pi)/2;

                    double A_times_sin = offset - f0_shifted_pi2;

                    double phi0 = atan2( A_times_sin, A_times_cos);


                    double parameter_shift = phi0 > 0 ? M_PI_half-phi0/2 : -phi0/2-M_PI_half;
                    double amplitude = sqrt(A_times_sin*A_times_sin + A_times_cos*A_times_cos);


                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                    = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift;

                    current_minimum_agents[agent_idx] = offset - amplitude;

                }
            }
            else if ( five_point_line_search ){


                // calsulate the cist functions at shifted parameter values
                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi4_agents = optimization_problem_batched( solution_guess_mtx_agents );

                // CPU time
                t0_CPU = tbb::tick_count::now();


                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_quarter;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );



                // CPU time
                t0_CPU = tbb::tick_count::now();


                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_pi_agents = optimization_problem_batched( solution_guess_mtx_agents );


                // CPU time
                t0_CPU = tbb::tick_count::now();

                for(int agent_idx=0; agent_idx<agent_num; agent_idx++) {
                    Matrix_real solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[agent_idx]] += M_PI_half;
                }

                // CPU time
                CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();

                // calculate batched cost function
                f0_shifted_3pi2_agents = optimization_problem_batched( solution_guess_mtx_agents );


                // CPU time
                t0_CPU = tbb::tick_count::now();


                // determine the parameters of the cosine function and determine the parameter shift at the minimum
                for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {

                    double current_minimum_agent = current_minimum_agents[agent_idx];
                    double f0_shifted_pi4        = f0_shifted_pi4_agents[agent_idx];
                    double f0_shifted_pi2        = f0_shifted_pi2_agents[agent_idx];
                    double f0_shifted_pi         = f0_shifted_pi_agents[agent_idx];
                    double f0_shifted_3pi2       = f0_shifted_3pi2_agents[agent_idx];
/*
                    f(p)          = kappa*sin(2p+xi) + gamma*sin(p+phi) + offset
                    f(p + pi/4)   = kappa*cos(2p+xi) + gamma*sin(p+pi/4+phi) + offset
                    f(p + pi/2)   = -kappa*sin(2p+xi) + gamma*cos(p+phi) + offset
                    f(p + pi)     = kappa*sin(2p+xi) - gamma*sin(p+phi) + offset
                    f(p + 3*pi/2) = -kappa*sin(2p+xi) - gamma*cos(p+phi) + offset
*/

                    double f1 = current_minimum_agent -  f0_shifted_pi;
                    double f2 = f0_shifted_pi2 - f0_shifted_3pi2;

                    double gamma = sqrt( f1*f1 + f2*f2 )*0.5;

                    double varphi = atan2( f1, f2) - parameter_value_save_agents[ agent_idx ];

                    double offset = 0.25*(current_minimum_agent +  f0_shifted_pi + f0_shifted_pi2 + f0_shifted_3pi2);
                    double f3     = 0.5*(current_minimum_agent +  f0_shifted_pi - 2*offset);
                    double f4     = f0_shifted_pi4 - offset - gamma*sin(parameter_value_save_agents[ agent_idx ]+M_PI_quarter+varphi);


                    double kappa = sqrt( f3*f3 + f4*f4);

                    double xi = atan2( f3, f4) - 2*parameter_value_save_agents[ agent_idx ];


                    double f;
                    double params[5];
                    params[0] = kappa;
                    params[1] = xi + 2*parameter_value_save_agents[ agent_idx ];
                    params[2] = gamma;
                    params[3] = varphi + parameter_value_save_agents[ agent_idx ];
                    params[4] = offset;


                    Matrix_real parameter_shift(1,1);
                    if ( abs(gamma) > abs(kappa) ) {
                        parameter_shift[0] = 3*M_PI/2 - varphi - parameter_value_save_agents[ agent_idx ];
                    }
                    else {
                        parameter_shift[0] = 3*M_PI/4 - xi/2 - parameter_value_save_agents[ agent_idx ]/2;
                    }
                    parameter_shift[0] = std::fmod( parameter_shift[0], M_PI_double);

                    BFGS_Powell cBFGS_Powell(HS_partial_optimization_problem_combined,(void*)&params);
                    f = cBFGS_Powell.Start_Optimization(parameter_shift, 10);

                    //update  the parameter vector
                    Matrix_real& solution_guess_mtx_agent                   = solution_guess_mtx_agents[ agent_idx ];
                    solution_guess_mtx_agent[param_idx_agents[ agent_idx ]] = parameter_value_save_agents[ agent_idx ] + parameter_shift[0];

                    current_minimum_agents[agent_idx] = f;

                }

            }





            // CPU time
            CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();


            // CPU time
            t0_CPU = tbb::tick_count::now();


            // generate random numbers to manage the behavior of the agents
            Matrix_real random_numbers(   agent_num, 2 );
            memset( random_numbers.get_data(), 0, 2*agent_num*sizeof(double) );

#ifdef __MPI__
            if ( current_rank == 0 ) {
#endif

                std::uniform_real_distribution<> distrib_to_choose(0.0, 1.0);

                for ( int agent_idx=0; agent_idx<2*agent_num; agent_idx++ ) {
                    random_numbers[agent_idx] = distrib_to_choose( gen );
                }

#ifdef __MPI__
            }
            MPI_Bcast( random_numbers.get_data(), 2*agent_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


            // ocassionaly recalculate teh current cost functions of the agents
            if ( iter_idx % agent_lifetime_loc == 0 )
            {
                // recalculate the current cost functions
                current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents );

                // Re-diversification at lifetime boundaries (skip the first one at iter 0)
                if ( iter_idx > 0 ) {
                    // Sort agents by cost (ascending)
                    std::vector<int> rediv_sorted(agent_num);
                    for (int i = 0; i < agent_num; i++) rediv_sorted[i] = i;
                    std::sort(rediv_sorted.begin(), rediv_sorted.end(),
                        [&](int a, int b) { return current_minimum_agents[a] < current_minimum_agents[b]; });

                    // Update global best from top agent
                    int best_agent = rediv_sorted[0];
                    if (current_minimum_agents[best_agent] <= current_minimum) {
                        current_minimum = current_minimum_agents[best_agent];
                        most_successfull_agent = best_agent;
                        memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agents[best_agent].get_data(), num_of_parameters * sizeof(double));
                    }

                    // Push agent positions into visited_basins with deduplication
                    for (int i = 0; i < agent_num; i++) {
                        Matrix_real basin_copy(1, num_of_parameters);
                        memcpy(basin_copy.get_data(), solution_guess_mtx_agents[i].get_data(), num_of_parameters * sizeof(double));
                        double agent_cost = current_minimum_agents[i];

                        // Check for near-duplicate
                        bool is_dup = false;
                        for (int vb = 0; vb < (int)visited_basins.size(); vb++) {
                            if (calculate_distance(basin_copy, visited_basins[vb].first) < visited_basins_dedup_radius) {
                                // Update cost if better
                                if (agent_cost < visited_basins[vb].second) {
                                    visited_basins[vb].second = agent_cost;
                                    memcpy(visited_basins[vb].first.get_data(), basin_copy.get_data(), num_of_parameters * sizeof(double));
                                }
                                is_dup = true;
                                break;
                            }
                        }
                        if (!is_dup) {
                            visited_basins.push_back(std::make_pair(basin_copy, agent_cost));
                        }
                    }

                    // Cap and prune visited_basins
                    if ((int)visited_basins.size() > max_visited_basins_agent) {
                        std::sort(visited_basins.begin(), visited_basins.end(),
                            [](const std::pair<Matrix_real, double>& a, const std::pair<Matrix_real, double>& b) {
                                return a.second < b.second;
                            });
                        visited_basins.resize(max_visited_basins_agent);
                    }

                    // Dynamic exclusion radius
                    int k = (int)visited_basins.size();
                    double rediv_radius = radius_base_agent * std::pow(std::log((double)(k + 1)) / (double)(k + 1), 1.0 / (double)num_of_parameters);

                    // Re-seed agents outside top-K with radius-excluded random points
                    for (int rank = rediversify_keep_top_agent; rank < agent_num; rank++) {
                        int ridx = rediv_sorted[rank];
                        Matrix_real& agent_params = solution_guess_mtx_agents[ridx];

                        bool found = false;
                        for (int attempt = 0; attempt < 500; attempt++) {
                            Matrix_real candidate(1, num_of_parameters);
                            for (int idx = 0; idx < num_of_parameters; idx++) {
                                candidate[idx] = distrib_real(gen);
                            }

                            bool too_close = false;
                            for (int pdx = 0; pdx < (int)visited_basins.size(); pdx++) {
                                if (calculate_distance(candidate, visited_basins[pdx].first) < rediv_radius) {
                                    too_close = true;
                                    break;
                                }
                            }

                            if (!too_close) {
                                memcpy(agent_params.get_data(), candidate.get_data(), num_of_parameters * sizeof(double));
                                found = true;
                                break;
                            }
                        }

                        // Fallback: fully random if rejection sampling exhausted
                        if (!found) {
                            for (int idx = 0; idx < num_of_parameters; idx++) {
                                agent_params[idx] = distrib_real(gen);
                            }
                        }

                        current_minimum_agents[ridx] = optimization_problem(agent_params);
                    }

                    std::stringstream sstream;
                    sstream << "AGENTS_BFGS: re-diversified " << (agent_num - rediversify_keep_top_agent) << " agents, kept top " << rediversify_keep_top_agent;
                    sstream << ", visited_basins: " << visited_basins.size() << ", rediv_radius: " << rediv_radius << std::endl;
                    print(sstream, 3);
                }
            }



            // govern the behavior of the agents
            for ( int agent_idx=0; agent_idx<agent_num; agent_idx++ ) {
                double& current_minimum_agent = current_minimum_agents[ agent_idx ];



                if (current_minimum_agent < optimization_tolerance_loc ) {
                    terminate_optimization = true;
                    current_minimum = current_minimum_agent;

                    most_successfull_agent = agent_idx;

                   Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];

                    // export the parameters of the current, most successful agent
                    memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters*sizeof(double) );
                }



                Matrix_real& solution_guess_mtx_agent = solution_guess_mtx_agents[ agent_idx ];

                // look for the best agent periodicaly
                if ( iter_idx % agent_lifetime_loc == 0 )
                {



                    if ( current_minimum_agent <= current_minimum ) {

                        most_successfull_agent = agent_idx;

                        // export the parameters of the current, most successful agent
                        memcpy(optimized_parameters_mtx.get_data(), solution_guess_mtx_agent.get_data(), num_of_parameters*sizeof(double) );

                        if ( export_circuit_2_binary_loc > 0 ) {
                            std::string filename("initial_circuit_iteration.binary");
                            if (project_name != "") {
                                filename=project_name+ "_"  +filename;
                            }
                            export_gate_list_to_binary(optimized_parameters_mtx, this, filename, verbose);
                        }


                        current_minimum = current_minimum_agent;



                    }
                    else {
                        // Re-diversification handles agent re-seeding above; keep current state here
                    }


                    // test global convergence
                    if ( agent_idx == 0 ) {

                        if ( output_periodicity>0 && iter_idx % output_periodicity == 0 ) {
                            export_current_cost_fnc(current_minimum);
                        }

                        current_minimum_mean = current_minimum_mean + (current_minimum - current_minimum_vec[ current_minimum_idx ])/current_minimum_vec.size();
                        current_minimum_vec[ current_minimum_idx ] = current_minimum;
                        current_minimum_idx = (current_minimum_idx + 1) % current_minimum_vec.size();

                        var_current_minimum = 0.0;
                        for (int idx=0; idx<current_minimum_vec.size(); idx++) {
                            var_current_minimum = var_current_minimum + (current_minimum_vec[idx]-current_minimum_mean)*(current_minimum_vec[idx]-current_minimum_mean);
                        }
                        var_current_minimum = std::sqrt(var_current_minimum)/current_minimum_vec.size();


                        if ( std::abs( current_minimum_mean - current_minimum) < 1e-7  && var_current_minimum < 1e-7 ) {
                            std::stringstream sstream;
                            sstream << "AGENTS_BFGS, iterations converged to "<< current_minimum << std::endl;
                            print(sstream, 3);
                            terminate_optimization = true;
                        }

                        // Stagnation detector: if best cost hasn't improved by relative threshold, exit Phase 1
                        if ( !terminate_optimization && iter_idx > 0 && iter_idx % stagnation_check_period_agent == 0 ) {
                            double relative_improvement = (best_cost_at_last_check - current_minimum) / (std::abs(best_cost_at_last_check) + 1e-30);
                            if ( relative_improvement < stagnation_threshold_agent && iter_idx >= stagnation_check_period_agent ) {
                                std::stringstream sstream;
                                sstream << "AGENTS_BFGS: Phase 1 stagnation detected at iter " << iter_idx
                                        << ", best cost " << current_minimum
                                        << ", improvement " << relative_improvement << std::endl;
                                print(sstream, 3);
                                phase1_stagnated = true;
                            }
                            best_cost_at_last_check = current_minimum;
                        }

                    }


		}

                if ( iter_idx % agent_lifetime_loc == 0 && agent_idx == 0) {
                    std::stringstream sstream;
                    sstream << "AGENTS_BFGS, agent " << agent_idx << ": processed iterations " << (double)iter_idx/max_inner_iterations_loc*100 << "%";
                    sstream << ", current minimum of agent 0: " << current_minimum_agents[ 0 ] << " global current minimum: " << current_minimum  << " CPU time: " << CPU_time;
                    sstream << " circuit simulation time: " << circuit_simulation_time  << std::endl;
                    print(sstream, 3);
                }




#ifdef __MPI__
                MPI_Barrier(MPI_COMM_WORLD);
#endif

            }  // for agent_idx
CPU_time += (tbb::tick_count::now() - t0_CPU).seconds();


            //
            // MODIFICATION B: Periodic BFGS polish of top-K agents
            //
            if ( iter_idx % bfgs_polish_period == 0 && !terminate_optimization )
            {
                // Build sorted index of agents by cost (ascending)
                std::vector<int> sorted_agent_indices(agent_num);
                for (int i = 0; i < agent_num; i++) sorted_agent_indices[i] = i;
                std::sort(sorted_agent_indices.begin(), sorted_agent_indices.end(),
                    [&](int a, int b) { return current_minimum_agents[a] < current_minimum_agents[b]; });

                for (int k_idx = 0; k_idx < bfgs_polish_top_k; k_idx++) {
                    if (terminate_optimization) break;

                    int polish_agent = sorted_agent_indices[k_idx];

                    // Skip stale agents that haven't improved since last polish
                    if (std::abs(current_minimum_agents[polish_agent] - last_polish_cost_agents[polish_agent]) < bfgs_polish_min_improvement_agent) {
                        continue;
                    }

                    Matrix_real bfgs_params(1, num_of_parameters);
                    memcpy(bfgs_params.get_data(), solution_guess_mtx_agents[polish_agent].get_data(), num_of_parameters * sizeof(double));

                    BFGS_Powell cBFGS_Powell(optimization_problem_combined, this);
                    double bfgs_result = cBFGS_Powell.Start_Optimization(bfgs_params, (long)bfgs_polish_iters_agent);

                    // Record polish cost for skip-stale tracking
                    last_polish_cost_agents[polish_agent] = bfgs_result;

                    // Push polished position into visited_basins with dedup
                    Matrix_real polished_copy(1, num_of_parameters);
                    memcpy(polished_copy.get_data(), bfgs_params.get_data(), num_of_parameters * sizeof(double));
                    bool is_dup = false;
                    for (int vb = 0; vb < (int)visited_basins.size(); vb++) {
                        if (calculate_distance(polished_copy, visited_basins[vb].first) < visited_basins_dedup_radius) {
                            if (bfgs_result < visited_basins[vb].second) {
                                visited_basins[vb].second = bfgs_result;
                                memcpy(visited_basins[vb].first.get_data(), polished_copy.get_data(), num_of_parameters * sizeof(double));
                            }
                            is_dup = true;
                            break;
                        }
                    }
                    if (!is_dup) {
                        visited_basins.push_back(std::make_pair(polished_copy, bfgs_result));
                    }

                    if (bfgs_result < current_minimum_agents[polish_agent]) {
                        // Update agent state with improved parameters
                        memcpy(solution_guess_mtx_agents[polish_agent].get_data(), bfgs_params.get_data(), num_of_parameters * sizeof(double));
                        current_minimum_agents[polish_agent] = bfgs_result;

                        // Update global best if improved
                        if (bfgs_result < current_minimum) {
                            current_minimum = bfgs_result;
                            memcpy(optimized_parameters_mtx.get_data(), bfgs_params.get_data(), num_of_parameters * sizeof(double));
                            most_successfull_agent = polish_agent;

                            // Reset convergence tracker — BFGS is actively improving
                            memset(current_minimum_vec.get_data(), 0, current_minimum_vec.size() * sizeof(double));
                            current_minimum_mean = 0.0;
                        }

                        if (bfgs_result < optimization_tolerance_loc) {
                            terminate_optimization = true;
                        }
                    }
                }
            }


            // terminate the agent if the whole optimization problem was solved
            if ( terminate_optimization || phase1_stagnated ) {
                break;
            }

        }


        //
        // Phase 2: Multi-start basin-hopping on top diverse candidates
        //
        if ( current_minimum >= optimization_tolerance_loc && !terminate_optimization ) {
            // Recalculate costs
            current_minimum_agents = optimization_problem_batched( solution_guess_mtx_agents );

            // Sort agents by cost (ascending)
            std::vector<int> bh_sorted(agent_num);
            for (int i = 0; i < agent_num; i++) bh_sorted[i] = i;
            std::sort(bh_sorted.begin(), bh_sorted.end(),
                [&](int a, int b) { return current_minimum_agents[a] < current_minimum_agents[b]; });

            // Select top-K diverse candidates
            double bh_diversity_threshold = radius_base_agent * 0.5;
            std::vector<int> bh_candidates;
            for (int i = 0; i < agent_num && (int)bh_candidates.size() < bfgs_exploit_top_k_agent; i++) {
                int cidx = bh_sorted[i];
                bool diverse = true;
                for (int j = 0; j < (int)bh_candidates.size(); j++) {
                    if (calculate_distance(solution_guess_mtx_agents[cidx], solution_guess_mtx_agents[bh_candidates[j]]) < bh_diversity_threshold) {
                        diverse = false;
                        break;
                    }
                }
                if (diverse) {
                    bh_candidates.push_back(cidx);
                }
            }

            int64_t num_bh = (int64_t)bh_candidates.size();

            sstream.str("");
            sstream << "AGENTS_BFGS: Phase 2 basin-hopping with " << num_bh << " diverse candidates, "
                    << bh_hops_agent << " hops, " << bfgs_exploit_iters_agent << " BFGS iters each" << std::endl;
            print(sstream, 3);

            // Pre-copy starting points and allocate per-candidate results
            std::vector<Matrix_real> bh_start_params(num_bh);
            std::vector<Matrix_real> bh_best_params(num_bh);
            std::vector<double> bh_best_scores(num_bh);
            for (int64_t i = 0; i < num_bh; i++) {
                int cidx = bh_candidates[i];
                bh_start_params[i] = Matrix_real(1, num_of_parameters);
                memcpy(bh_start_params[i].get_data(), solution_guess_mtx_agents[cidx].get_data(), num_of_parameters * sizeof(double));
                bh_best_params[i] = Matrix_real(1, num_of_parameters);
                memcpy(bh_best_params[i].get_data(), bh_start_params[i].get_data(), num_of_parameters * sizeof(double));
                bh_best_scores[i] = current_minimum_agents[cidx];
            }

            int parallel_bh = get_parallel_configuration();
            int64_t work_batch_bh = 1;
            if (parallel_bh == 0) {
                work_batch_bh = num_bh;
            }

            volatile bool bh_solution_found = false;

            tbb::parallel_for(
                tbb::blocked_range<int64_t>(0, num_bh, work_batch_bh),
                [&](tbb::blocked_range<int64_t> r) {
                    std::mt19937 local_gen(gen() + r.begin());
                    std::uniform_real_distribution<> local_uniform(0.0, 1.0);

                    for (int64_t i = r.begin(); i < r.end(); i++) {
                        if (bh_solution_found) break;

                        // Basin-hopping state for this candidate
                        Matrix_real x_current(1, num_of_parameters);
                        memcpy(x_current.get_data(), bh_start_params[i].get_data(), num_of_parameters * sizeof(double));

                        // Initial BFGS polish of starting point
                        BFGS_Powell cBFGS_init(optimization_problem_combined, this);
                        double f_current = cBFGS_init.Start_Optimization(x_current, (long)bfgs_exploit_iters_agent);

                        if (f_current < bh_best_scores[i]) {
                            bh_best_scores[i] = f_current;
                            memcpy(bh_best_params[i].get_data(), x_current.get_data(), num_of_parameters * sizeof(double));
                        }

                        if (bh_best_scores[i] < optimization_tolerance_loc) {
                            bh_solution_found = true;
                            continue;
                        }

                        double stepsize_now = bh_stepsize_agent;
                        int accept_count_window = 0;
                        int window_len = 0;

                        // Basin-hopping loop
                        for (int hop = 0; hop < bh_hops_agent; hop++) {
                            if (bh_solution_found) break;

                            // Perturb: uniform random displacement scaled by stepsize
                            Matrix_real x_trial(1, num_of_parameters);
                            for (int idx = 0; idx < num_of_parameters; idx++) {
                                double delta = (local_uniform(local_gen) * 2.0 - 1.0) * stepsize_now * M_PI;
                                x_trial[idx] = std::fmod(x_current[idx] + delta, 2.0 * M_PI);
                            }

                            // BFGS polish the perturbed point
                            BFGS_Powell cBFGS_hop(optimization_problem_combined, this);
                            double f_trial = cBFGS_hop.Start_Optimization(x_trial, (long)bfgs_exploit_iters_agent);

                            // Metropolis acceptance
                            bool accept = false;
                            if (f_trial <= f_current) {
                                accept = true;
                            } else {
                                double dE = f_trial - f_current;
                                double prob = std::exp(-dE / std::max(1e-300, bh_T_agent));
                                accept = (local_uniform(local_gen) < prob);
                            }

                            if (accept) {
                                memcpy(x_current.get_data(), x_trial.get_data(), num_of_parameters * sizeof(double));
                                f_current = f_trial;
                                accept_count_window++;
                            }

                            // Track best for this candidate
                            if (f_trial < bh_best_scores[i]) {
                                bh_best_scores[i] = f_trial;
                                memcpy(bh_best_params[i].get_data(), x_trial.get_data(), num_of_parameters * sizeof(double));
                            }

                            if (bh_best_scores[i] < optimization_tolerance_loc) {
                                bh_solution_found = true;
                                break;
                            }

                            // Adaptive stepsize
                            window_len++;
                            if (bh_interval_agent > 0 && (window_len % bh_interval_agent) == 0) {
                                double accept_rate = (double)accept_count_window / (double)bh_interval_agent;
                                if (accept_rate > bh_target_accept_agent) {
                                    stepsize_now /= bh_stepwise_factor_agent;
                                } else {
                                    stepsize_now *= bh_stepwise_factor_agent;
                                }
                                accept_count_window = 0;
                                window_len = 0;
                            }
                        }
                    }
                });

            // Collect results
            for (int64_t i = 0; i < num_bh; i++) {
                if (bh_best_scores[i] < current_minimum) {
                    current_minimum = bh_best_scores[i];
                    memcpy(optimized_parameters_mtx.get_data(), bh_best_params[i].get_data(), num_of_parameters * sizeof(double));

                    sstream.str("");
                    sstream << "AGENTS_BFGS: Phase 2 BH improved to " << current_minimum << " from candidate " << i << std::endl;
                    print(sstream, 3);
                }

                if (current_minimum < optimization_tolerance_loc) {
                    break;
                }
            }
        }

        tbb::tick_count optimization_end = tbb::tick_count::now();
        optimization_time  = optimization_time + (optimization_end-optimization_start).seconds();
        sstream.str("");
        sstream << "AGENTS_BFGS time: " << CPU_time << " " << current_minimum << std::endl;

        print(sstream, 1);
}

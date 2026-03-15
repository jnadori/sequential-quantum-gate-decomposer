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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file RL_experience.cpp
    \brief A class for RL_experience 
*/


#include "Optimization_Interface.h"
#include "N_Qubit_Decomposition_Cost_Function.h"
#include "BFGS_Powell.h"
#include "RL_experience.h"
#include "Bayes_Opt.h"

#include <fstream>
#include "tbb/tbb.h"
/**
@brief Call to solve layer by layer the optimization problem via the AGENT algorithm. The optimalized parameters are stored in attribute optimized_parameters.
@param num_of_parameters Number of parameters to be optimized
@param solution_guess Array containing the solution guess.
*/
void Optimization_Interface::solve_layer_optimization_problem_POSMM( int num_of_parameters, Matrix_real& solution_guess) {


        if (gates.size() == 0 ) {
            return;
        }


        if (solution_guess.size() == 0 ) {
            solution_guess = Matrix_real(num_of_parameters,1);
        }


        if (optimized_parameters_mtx.size() == 0) {
            optimized_parameters_mtx = Matrix_real(1, num_of_parameters);
            memcpy(optimized_parameters_mtx.get_data(), solution_guess.get_data(), num_of_parameters*sizeof(double) );
        }

        // maximal number of iteration loops
        int iteration_loops_max = 50;
        long long value;
        if ( config.count("max_iteration_loops_posmm") > 0 ) {
            config["max_iteration_loops_posmm"].get_property( value );
            iteration_loops_max = (int) value;
        }
        else if( config.count("max_iteration_loops") > 0 ) {
            config["max_iteration_loops"].get_property( value );       
            iteration_loops_max = (int) value;  
        }


        // maximal number of iteration loops
        int worker_num = 8;
        long long value1;
        if ( config.count("worker_num_posmm") > 0 ) {
            config["worker_num_posmm"].get_property( value1 );
            worker_num = (int) value1;
        }
        else if ( config.count("worker_num") > 0 ) {
            config["worker_num"].get_property( value1 );       
            worker_num = (int) value1;  
        }


        // random generator of real numbers   
        std::uniform_real_distribution<> distrib_real(0.0, 2*M_PI);

        long long max_inner_iterations_loc;
        if ( config.count("max_inner_iterations_bfgs") > 0 ) {
            config["max_inner_iterations_bfgs"].get_property( max_inner_iterations_loc );         
        }
        else if ( config.count("max_inner_iterations") > 0 ) {
            config["max_inner_iterations"].get_property( max_inner_iterations_loc );         
        }
        else {
            max_inner_iterations_loc =max_inner_iterations;
        }

        double radius_base=0.5;
        if ( config.count("radius_base_posmm") > 0 ) {
             config["radius_base_posmm"].get_property( radius_base );
        }
        if ( config.count("radius_base") > 0 ) {
             config["radius_base"].get_property( radius_base );
        }

        double optimization_tolerance_loc;
        if ( config.count("optimization_tolerance_posmm") > 0 ) {
             config["optimization_tolerance_posmm"].get_property( optimization_tolerance_loc );
        }
        else if ( config.count("optimization_tolerance") > 0 ) {
             config["optimization_tolerance"].get_property( optimization_tolerance_loc );
        }
        else {
            optimization_tolerance_loc = optimization_tolerance;
        }

        std::vector<Matrix_real> previous_points;

        double best_score=1.;

        std::vector<Matrix_real> points_to_optimize;

        for (int iteration_idx=0; iteration_idx<iteration_loops_max; iteration_idx++){
            double k = (double) iteration_idx +1.;
            double radius = radius_base*std::pow(std::log(k)/k, 1./(double)parameter_num);
            while (points_to_optimize.size()<worker_num){
                bool found = false;
                for (int tdx=0; tdx<501;tdx++){
                    bool should_continue=false;
                    Matrix_real new_random_point(1,parameter_num);
                    for (int idx=0;idx<parameter_num; idx++){
                        new_random_point[idx] = distrib_real(gen);
                    }
                    if (previous_points.size()>0){
                        for (int pdx=0; pdx<previous_points.size(); pdx++){
                            if (calculate_distance(new_random_point, previous_points[pdx])<radius){
                                should_continue=true;
                                break;
                            }
                        }

                    }
                    if (!should_continue && points_to_optimize.size()>0){
                        for (int pdx=0; pdx<points_to_optimize.size(); pdx++){
                            if (calculate_distance(new_random_point, points_to_optimize[pdx])<radius){
                                should_continue=true;
                                break;
                            }
                        }
                    }
                    if (should_continue){
                        continue;
                    }
                    points_to_optimize.push_back(new_random_point);
                    found = true;
                    break;
                }
                if (!found) return;
            }

        int64_t num_points = static_cast<int64_t>(points_to_optimize.size());

        // Save starting points before parallel optimization
        std::vector<Matrix_real> starting_points(num_points);
        for (int64_t i = 0; i < num_points; i++) {
            starting_points[i] = Matrix_real(1, parameter_num);
            memcpy(starting_points[i].get_data(), points_to_optimize[i].get_data(), num_of_parameters * sizeof(double));
        }

        // Pre-allocate result scores
        std::vector<double> opt_scores(num_points, std::numeric_limits<double>::max());

        // Determine concurrency
        int parallel = get_parallel_configuration();
        int64_t work_batch = 1;
        if (parallel == 0) {
            work_batch = num_points;
        }

        // Shared flag for early abort across workers
        volatile bool solution_found = false;

        // Parallel BFGS optimization of each point
        tbb::parallel_for(
            tbb::blocked_range<int64_t>(0, num_points, work_batch),
            [&](tbb::blocked_range<int64_t> r) {
                for (int64_t i = r.begin(); i < r.end(); i++) {
                    if (solution_found) break;
                    BFGS_Powell cBFGS_Powell(optimization_problem_combined, this);
                    opt_scores[i] = cBFGS_Powell.Start_Optimization(points_to_optimize[i], static_cast<long>(max_inner_iterations_loc));
                    if (opt_scores[i] < optimization_tolerance_loc) {
                        solution_found = true;
                    }
                }
            });

        // Post-processing: add starting points to previous_points
        for (int64_t i = 0; i < num_points; i++) {
            previous_points.push_back(starting_points[i]);
        }

        // Update best score
        for (int64_t i = 0; i < num_points; i++) {
            if (current_minimum > opt_scores[i]) {
                current_minimum = opt_scores[i];
                memcpy(optimized_parameters_mtx.get_data(),
                    points_to_optimize[i].get_data(),
                    num_of_parameters * sizeof(double));
            }
            if (current_minimum < optimization_tolerance_loc) {
                return;
            }
        }

        // Filter points_to_optimize: keep only points far from previous_points
        for (auto it = points_to_optimize.begin(); it != points_to_optimize.end(); ) {
            bool add_new_point = true;
            for (const auto& prev : previous_points) {
                if (calculate_distance(*it, prev) < radius) {
                    add_new_point = false;
                    break;
                }
            }
            if (!add_new_point) {
                it = points_to_optimize.erase(it);
            } else {
                ++it;
            }
        }
            
        }

        return;
}
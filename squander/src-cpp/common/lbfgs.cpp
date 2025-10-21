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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lbfgs.h"

LBFGS::LBFGS(void (* f_pointer) (Matrix_real , void *, double *, Matrix_real&), void* meta_data_in) : Grad_Descend(f_pointer, meta_data_in) {


}

void LBFGS::Initialize(){
    m = 50;  // Large enough for deep circuits

    s_history.resize(m);

    y_history.resize(m);

    // Initialize history matrices
    for (int i = 0; i < m; i++) {
        s_history[i] = Matrix_real(variable_num, 1);
        y_history[i] = Matrix_real(variable_num, 1);
    }

    current_size = 0;

    head_idx = 0;

    status = VARIABLES_INITIALIZED;

    return;
}

void LBFGS::Optimize(Matrix_real& x, double& f){

    // The calculated graient of the cost function at position x
    Matrix_real g( variable_num, 1 );       

    // the current search direction
    Matrix_real search_direction( variable_num, 1 );

    // the initial gradient during the line search
    Matrix_real g0_search( variable_num, 1 ); 

    // the initial point during the line search
    Matrix_real x0_search( variable_num, 1 ); 
    

    Initialize();

    double* rho = new double[m];

    for (int i = 0; i < m; i++) {
        rho[i] = 0.0;
    }


    status = VARIABLES_INITIALIZED; 

    
    double maximal_step; // The upper bound of the allowed step in the search direction
    
    if (function_call_count == 0 || status == VARIABLES_INITIALIZED) {
        get_f_ang_gradient(x, f, g);
    }
    
    double fprev = fabs(f + f + 1.0);

    while(true){

        double gradient_norm = 0.0;
        for (long idx = 0; idx < variable_num; idx++) {
            gradient_norm += g[idx] * g[idx];
        }

        double acc=1e-19;
        // Test for convergence by measuring the magnitude of the gradient.
        if (gradient_norm <= acc * acc) {
            status = MINIMUM_REACHED;
            delete[] rho;
            return;
        }

        // 3. Compute search direction
        if (current_size == 0) {
            // First iteration: steepest descent
            for (int i = 0; i < variable_num; i++) {
                search_direction[i] = -g[i];
            }
        } else {
            // Use two-loop recursion
            Two_Loop_Update(g, search_direction, rho);
        }
        double d__dot__g0 = 0.0;
        for (int i = 0; i < variable_num; i++) {
            d__dot__g0 += search_direction[i] * g[i];
        }
        // in case the cost fuction does not show decrement in the direction of the line search
        // Reset to steepest descent instead of giving up - this is critical for LBFGS robustness
        if (d__dot__g0 >= 0.0) {
            // Reset history and use steepest descent
            current_size = 0;
            head_idx = 0;
            d__dot__g0 = 0.0;
            for (int i = 0; i < variable_num; i++) {
                search_direction[i] = -g[i];
                d__dot__g0 += search_direction[i] * g[i];
            }
        }
        get_Maximal_Line_Search_Step(search_direction, maximal_step, d__dot__g0);

        // terminate if maximal number of iteration reached
        if (function_call_count >= maximal_iterations) {
            status = MAXIMAL_ITERATIONS_REACHED;
            delete[] rho;
            return;
        }

        line_search(x, g, search_direction, x0_search, g0_search, maximal_step, d__dot__g0, f);

        if (status == ZERO_STEP_SIZE_OCCURED) {
            delete[] rho;
            return;
        }

        Matrix_real s_k(variable_num, 1);
        Matrix_real y_k(variable_num, 1);
        for (int i = 0; i < variable_num; i++) {
            s_k[i] = x[i] - x0_search[i];      // x is already updated by line_search
            y_k[i] = g[i] - g0_search[i];      // g is already updated by line_search
        }

        update_history(s_k, y_k, rho);

    }
    delete[] rho;

    return;
}

void LBFGS::Two_Loop_Update(Matrix_real& g, Matrix_real& search_direction, double* rho){
    Matrix_real q(variable_num, 1);
    Matrix_real alpha(current_size, 1);

    memcpy(q.get_data(), g.get_data(), variable_num*sizeof(double));
    // backward loop
    for (int idx=current_size-1; idx>=0; idx--){
        int current_idx = (head_idx + idx) % m;
        double s__dot__q = 0.0;
        for (int kdx=0; kdx<variable_num; kdx++){
            s__dot__q += s_history[current_idx][kdx]*q[kdx];
        }
        alpha[idx] = rho[current_idx] * s__dot__q;
        for (int jdx=0; jdx<variable_num; jdx++){
            q[jdx] = q[jdx] - alpha[idx] * y_history[current_idx][jdx];  // FIX: use y_history not s_history!
        }
    }
    //scaling
    int newest_idx = (head_idx + current_size - 1) % m;
    double gamma = M3_scaling(y_history[newest_idx], s_history[newest_idx]);
        for (int idx=0; idx<variable_num; idx++){
        search_direction[idx] =  gamma * q[idx];
    }
    //forward loop
    for (int idx=0; idx<current_size; idx++){
        int current_idx = (head_idx + idx) % m;
        double y__dot__search_direction = 0.0;
        for (int kdx=0; kdx<variable_num; kdx++){
            y__dot__search_direction += y_history[current_idx][kdx]*search_direction[kdx];
        }
        double beta = rho[current_idx] * y__dot__search_direction;
        for (int jdx=0; jdx<variable_num; jdx++){
            search_direction[jdx] = search_direction[jdx] + (alpha[idx] - beta) * s_history[current_idx][jdx];  
        }
    }
    //negate the search direction
    for (int idx=0; idx<variable_num; idx++){
        search_direction[idx] = -search_direction[idx];
    }
    return;
}

double LBFGS::M3_scaling(Matrix_real y, Matrix_real s){
    if (current_size == 0){
        return 1.0;
    }
    double y__dot__s = 0.0;
    double y__dot__y = 0.0;
    for (int idx=0; idx<variable_num; idx++){
            y__dot__s += y[idx]*s[idx];
            y__dot__y += y[idx]*y[idx];
    }
    return y__dot__s / y__dot__y;
}

void LBFGS::update_history(Matrix_real s_k, Matrix_real y_k, double* rho){
    double y__dot__s = 0.0;
    double y_norm = 0.0;
    double s_norm = 0.0;
    for (int idx=0; idx<variable_num; idx++){
        y__dot__s += y_k[idx]*s_k[idx];
        y_norm += y_k[idx]*y_k[idx];
        s_norm += s_k[idx]*s_k[idx];
    }
    y_norm = sqrt(y_norm);
    s_norm = sqrt(s_norm);

    // Curvature condition: ensure y^T s > 0 for positive definiteness
    // Be very permissive to build history - only skip clearly bad updates
    if (y__dot__s <= 0.0) {
        return;
    }
    if (current_size<m){
        s_history[current_size]=s_k;
        y_history[current_size]=y_k;
        rho[current_size] = 1/y__dot__s;
        current_size ++;
    }
    else {
        //shift head leftwards 
        head_idx = (head_idx + 1) % m;
        rho[(head_idx + m -1) % m] = 1/y__dot__s;
        s_history[(head_idx + m -1) % m] = s_k;
        y_history[(head_idx + m -1) % m] = y_k;
    }

    return;
}

/**
@brief LBFGS-specific line search with stronger Wolfe curvature condition (c2=0.9)
This provides better curvature information for the limited-memory approximation
*/
void LBFGS::line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double& maximal_step, double& d__dot__g0, double& f)
{
    memcpy( x0_search.get_data(), x.get_data(), x.size()*sizeof(double) );
    memcpy( g0_search.get_data(), g.get_data(), g.size()*sizeof(double) );

    long max_loops = 50;

    Matrix_real g_best = g.copy();

    double step = std::min(1.0, maximal_step);

    double f_lowest  = f;
    double f_highest = f;

    double step_highest = maximal_step;
    double step_lowest = 0.0;

    double step_best = 0.0;
    double f_best         = f;
    double d__dot__g_abs_best = fabs(d__dot__g0);

    double d__dot__g_lowest = d__dot__g0;
    double d__dot__g_highest = 0.0;


    for( long iter_idx=0; iter_idx<max_loops; iter_idx++) {

        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }

        get_f_ang_gradient(x, f, g);

        // overlap between the search direction and the gradient
        double d__dot__g_current = 0.0;
        for (long idx = 0; idx < variable_num; idx++) {
            d__dot__g_current += search_direction[idx] * g[idx];
        }

        // update best solution
        if (f < f_best || fabs(d__dot__g_current) < d__dot__g_abs_best) {
            step_best      = step;
            f_best         = f;
            d__dot__g_abs_best = fabs(d__dot__g_current);
            memcpy( g_best.get_data(), g.get_data(), g.size()*sizeof(double) );
        }

        // exit the loop if maximal function calls reached
        if (function_call_count == maximal_iterations) {
            break;
        }

        // modify the upper and lower bound of the step interval
        if (f < f_lowest + step * 0.1 * d__dot__g0) {  // Armijo test (Wolfe 1st condition)

            // LBFGS-specific: Use c2=0.9 for stronger curvature condition
            if (d__dot__g_current >= d__dot__g0 * 0.9) { // Wolfe 2nd (curvature) condition
                break;
            }

            step_lowest      = step;
            f_lowest         = f;
            d__dot__g_lowest = d__dot__g_current;
        }
        else {
            step_highest      = step;
            f_highest         = f;
            d__dot__g_highest = d__dot__g_current;
        }

        // Calculate the next step length
        if (iter_idx == 0 || step_lowest > 0.0) {
            // fit a parabola
            double d__dot__g_lowest_fit = (f_highest - f_lowest) / (step_highest - step_lowest)*2.0 - d__dot__g_highest;
            double scale = -d__dot__g_lowest_fit*0.5 / (d__dot__g_highest - d__dot__g_lowest_fit);
            scale = std::max( 0.1, scale);
            step = step_lowest + scale * (step_highest - step_lowest);
        }
        else {
            step = step * 0.1;
            if (step < num_precision) {
                break;
            }
        }
    } // for loop

    // copy the best solution in place of the current solution
    if (step != step_best) {
        step = step_best;
        f = f_best;
        g = g_best;
        for (long idx = 0; idx < variable_num; idx++) {
            x[idx] = x0_search[idx] + step * search_direction[idx];
        }
    }

    if (step == 0.0 ) {
        status = ZERO_STEP_SIZE_OCCURED;
    }

    return;
}

/*
The destructor of the class
*/
LBFGS::~LBFGS(){

}
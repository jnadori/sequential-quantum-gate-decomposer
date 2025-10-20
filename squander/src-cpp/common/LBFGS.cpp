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

#include <LBFGS.h>

LBFGS::LBFGS(void (* f_pointer) (Matrix_real , void *, double *, Matrix_real&), void* meta_data_in) : Grad_Descend(f_pointer, meta_data_in) {

    m = 5;
    
    s_history.reserve(m);
    
    y_history.reserve(m);

    current_size = 0;

    head_idx = 0;
}


void LBFGS::Optimize(){

    // the initial gradient during the line search
    Matrix_real g0_search( variable_num, 1 ); 

    // the initial point during the line search
    Matrix_real x0_search( variable_num, 1 ); 

    // The calculated graient of the cost function at position x
    Matrix_real g( variable_num, 1 );       

    // the current search direction
    Matrix_real search_direction( variable_num, 1 );  

    int k = 0;
    
    double* rho[variable_num];
    
    Matrix_real H__dot__G; 

}

void LBFGS::update_history(Matrix_real s_k, Matrix_real y_k, double* rho){
    double y__dot__s = 0.0;
    for (int idx=0; idx<variable_num; idx++){
        y__dot__s += y_k[idx]*s_k[idx];
    }
    if (y__dot__s <= 0) {
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

/*
The destructor of the class
*/
LBFGS::~LBFGS(){

}
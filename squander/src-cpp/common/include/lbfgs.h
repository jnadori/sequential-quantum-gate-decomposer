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

#ifndef __LBFGS__H
#define __LBFGS__H

#include "grad_descend.h"


/**
@brief A class implementing the BFGS optimizer based on conjugate gradient direction method of M. J. D. Powell: A tolerant algorithm for linearly constrained optimization calculations, Mathematical Programming volume 45, pages 547–566 (1989)
*/
class LBFGS : public Grad_Descend 
{
protected:

    Matrix_real H;

    std::vector<Matrix_real> s_history;

    std::vector<Matrix_real> y_history;

    int m;

    int head_idx;

    int current_size;

public:

LBFGS(void (* f_pointer) (Matrix_real, void *, double *, Matrix_real&), void* meta_data_in);

void Initialize();

void Optimize(Matrix_real& x, double& f);

void Two_Loop_Update(Matrix_real& g, Matrix_real& search_direction, double* rho);

double M3_scaling(Matrix_real y, Matrix_real s);

void update_history(Matrix_real s_k, Matrix_real y_k, double* rho);

void line_search(Matrix_real& x, Matrix_real& g, Matrix_real& search_direction, Matrix_real& x0_search, Matrix_real& g0_search, double& maximal_step, double& d__dot__g0, double& f);

~LBFGS();

};

#endif  // __LBFGS__H
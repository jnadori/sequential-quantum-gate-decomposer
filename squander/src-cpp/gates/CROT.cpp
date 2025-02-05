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
/*! \file CROT.cpp
    \brief Class representing a CROT gate.
*/

#include "CROT.h"
#include "apply_large_kernel_to_input_AVX.h"
#include "apply_large_kernel_to_input.h"


using namespace std;


/**
@brief Nullary constructor of the class.
*/
CROT::CROT() {

        // number of qubits spanning the matrix of the gate
        qbit_num = -1;
        // the size of the matrix
        matrix_size = -1;
        // A string describing the type of the gate
        type = CROT_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = -1;

        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
        control_qbit = -1;


}


/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits in the unitaries
@param target_qbit_in The identification number of the target qubit. (0 <= target_qbit <= qbit_num-1)
@param control_qbit_in The identification number of the control qubit. (0 <= target_qbit <= qbit_num-1)
*/
CROT::CROT(int qbit_num_in,  int target_qbit_in, int control_qbit_in) {

 
        // number of qubits spanning the matrix of the gate
        qbit_num = qbit_num_in;
        // the size of the matrix
        matrix_size = Power_of_2(qbit_num);
        // A string describing the type of the gate
        type = CROT_OPERATION;
        // The number of free parameters
        parameter_num = 0;

        if (target_qbit_in >= qbit_num) {
            std::stringstream sstream;	   
            sstream << "The index of the target qubit is larger than the number of qubits" << std::endl;
            print(sstream, 0);	    	
            throw sstream.str();
        }
        // The index of the qubit on which the gate acts (target_qbit >= 0)
        target_qbit = target_qbit_in;


        if (control_qbit_in >= qbit_num) {
	    std::stringstream sstream;
            sstream << "The index of the control qubit is larger than the number of qubits" << std::endl;
            print(sstream, 0);	    	
            throw sstream.str();
        }
        // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled gate
        control_qbit = control_qbit_in;


}

/**
@brief Destructor of the class
*/
CROT::~CROT() {
}


/**
@brief Call to apply the gate on the input array/matrix CNOT*input
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
CROT::apply_to( Matrix& input, int parallel ) {
 



    if (input.rows != matrix_size ) {
	std::stringstream sstream;
	sstream << "Wrong matrix size in CROT gate apply" << std::endl;
        print(sstream, 0);	
        exit(-1);
    }

/*
    ThetaOver2 = theta0;
    Phi = parameters[0];
    Lambda = lambda0;
*/
/*  
Phi = Phi + M_PI;
Phi = (1.0-std::cos(Phi/2))*M_PI;
Phi = Phi - M_PI;
*/
//Phi = 0.5*(1.0-std::cos(Phi))*M_PI;

    Matrix U_2qbit(4,4);
    memset(U_2qbit.get_data(),0.0,(U_2qbit.size()*2)*sizeof(double));
    double invroottwo = 1./std::sqrt(2);
    U_2qbit[0].real = invroottwo;
    U_2qbit[1].real = invroottwo;
    U_2qbit[1*4].real = -1.*invroottwo;
    U_2qbit[1*4+1].real = invroottwo;
    U_2qbit[2*4+2].real = invroottwo;
    U_2qbit[2*4+3].real = -1*invroottwo;
    U_2qbit[3*4+3].real = invroottwo;
    U_2qbit[3*4+2].real = invroottwo;
    // apply the computing kernel on the matrix
    std::vector<int> involved_qbits = {control_qbit,target_qbit};
    //apply_large_kernel_to_input(U_2qbit,input,involved_qbits,input.size());
    if (input.cols<=1){
        apply_2qbit_kernel_to_state_vector_input(U_2qbit,input,control_qbit,target_qbit,input.size());
    }
    else{
        apply_2qbit_kernel_to_matrix_input(U_2qbit,input,control_qbit,target_qbit,input.size());
    }


}



/**
@brief Call to apply the gate on the input array/matrix by input*CNOT
@param input The input array on which the gate is applied
*/
void 
CROT::apply_from_right( Matrix& input ) {



}




/**
@brief Call to set the number of qubits spanning the matrix of the gate
@param qbit_num The number of qubits
*/
void CROT::set_qbit_num(int qbit_num) {
        // setting the number of qubits
        Gate::set_qbit_num(qbit_num);

}



/**
@brief Call to reorder the qubits in the matrix of the gate
@param qbit_list The reordered list of qubits spanning the matrix
*/
void CROT::reorder_qubits( vector<int> qbit_list) {

        Gate::reorder_qubits(qbit_list);

}

/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
CROT* CROT::clone() {

    CROT* ret = new CROT( qbit_num, target_qbit, control_qbit );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );

    return ret;

}




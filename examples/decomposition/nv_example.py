# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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
"""
## \file example.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package

## [import]
from squander import N_Qubit_Decomposition 
## [import]
## [import adaptive]
from squander import NV_Decomposition
from squander import N_Qubit_Decomposition_adaptive            
## [import adaptive]


#******************************
## [load Umtx]
from scipy.io import loadmat
import numpy as np
from scipy.stats import unitary_group

from squander import utils

from scipy.optimize import minimize
    
## load the unitary from file
## The unitary to be decomposed  
qbit_num=3
matrix_size=2**qbit_num
Umtx = unitary_group.rvs(matrix_size)
import time 
## [create decomposition class]
## creating a class to decompose the unitary
config = {      'agent_lifetime':200,
                'max_inner_iterations_agent': 100000,
                'max_inner_iterations_compression': 10000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000,
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-8,
                'optimization_tolerance_': 1e-12}
topology = [(0,1),(0,2),(1,2),(0,1),(0,2),(1,2),(0,1),(0,2)]
print("COMPILING GHZ CIRCUIT")
NVDecompose = NV_Decomposition( Umtx,config=config )
NVDecompose.set_Optimizer("BFGS")
NVDecompose.set_Verbose(3)
NVDecompose.set_Cost_Function_Variant( 3 )
NVDecompose.get_Initial_Circuit(subtype="CONTROL_INDEPENDENT",number_of_layers=1,topology=topology,final_layer=True)
NVDecompose.set_Optimization_Tolerance( 1e-12 )
NVDecompose.Start_Decomposition()
parameters=NVDecompose.get_Optimized_Parameters()
decomposition_error = NVDecompose.Optimization_Problem(parameters)
print(1-decomposition_error)
def fun(x):
    return NVDecompose.Optimization_Problem(x)
res = minimize(fun,parameters,method="Powell",tol=1e-10)
print(NVDecompose.Optimization_Problem(res.x))
"""
cDecompose.set_Optimizer("BFGS")
cDecompose.set_Verbose(3)
cDecompose.set_Cost_Function_Variant( 3 )
## [create decomposition class]


## [start decomposition]
# starting the decomposition
cDecompose.Start_Decomposition()
# list the decomposing operations
#print(cDecompose.get_Decomposition_Error())
quantum_circuit = cDecompose.get_Qiskit_Circuit()

print(quantum_circuit)
print(cDecompose.get_Optimized_Parameters())
"""

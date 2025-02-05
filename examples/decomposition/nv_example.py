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
    
## load the unitary from file
data = loadmat('Umtx.mat')
## The unitary to be decomposed  
matrix_size=8
Umtx = unitary_group.rvs(matrix_size)
## [load Umtx]
config = {      'agent_lifetime':200,
                'max_inner_iterations_agent': 100000,
                'max_inner_iterations_compression': 100000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000, 		
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-3,
                'optimization_tolerance_': 1e-8}

# determine the size of the unitary to be decomposed
matrix_size = len(Umtx)
topology=[(0,1),(0,2)]

## [create decomposition class]
## creating a class to decompose the unitary
NVDecompose = NV_Decomposition( Umtx, level_limit_max=20, topology=topology  )
NVDecompose.set_Optimizer("BFGS")
NVDecompose.set_Verbose(3)
NVDecompose.set_Cost_Function_Variant( 3 )
## [create decomposition class]
print(NVDecompose.get_Decomposition_Error())

## [start decomposition]
# starting the decomposition
NVDecompose.get_Initial_Circuit()
# list the decomposing operations
## [start decomposition]

cDecompose = N_Qubit_Decomposition_adaptive( Umtx, level_limit_max=20, topology=topology )
cDecompose.set_Optimizer("BFGS")
cDecompose.set_Verbose(3)
cDecompose.set_Cost_Function_Variant( 3 )
## [create decomposition class]


## [start decomposition]
# starting the decomposition
#cDecompose.get_Initial_Circuit()
# list the decomposing operations
print(cDecompose.get_Decomposition_Error())

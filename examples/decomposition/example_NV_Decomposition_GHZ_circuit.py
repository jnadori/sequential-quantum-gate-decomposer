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

import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
    
if qiskit_version[0] == '1':
    from qiskit import transpile
else :
    from qiskit import execute
Umtx=np.array([])
qbit_num=4
circuit_qiskit = QuantumCircuit(qbit_num)
circuit_qiskit.h(0)
circuit_qiskit.cx(0,1)
circuit_qiskit.cx(0,2)
circuit_qiskit.cx(0,3)

circuit_qiskit.save_unitary()
# Transpile for simulator
simulator = AerSimulator(method = 'unitary')
circ = transpile(circuit_qiskit, simulator)

# Run and get unitary
result = simulator.run(circ).result()
Umtx = result.get_unitary(circ).to_matrix()

print(circuit_qiskit)
config = {      'agent_lifetime':200,
                'max_inner_iterations_agent': 100000,
                'max_inner_iterations_compression': 10000,
                'max_inner_iterations' : 10000,
                'max_inner_iterations_final': 10000,
                'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
                'optimization_tolerance_agent': 1e-8,
                'optimization_tolerance_': 1e-8}
topology = [(0,1),(0,2),(0,3)]
print("COMPILING GHZ CIRCUIT")
NVDecompose = NV_Decomposition( Umtx,config=config )
NVDecompose.set_Optimizer("BFGS")
NVDecompose.set_Verbose(0)
NVDecompose.set_Cost_Function_Variant( 8 )
NVDecompose.get_Initial_Circuit(subtype="CONTROL_OPPOSITE",number_of_layers=3,topology=topology,final_layer=True)
NVDecompose.Start_Decomposition()
parameters=NVDecompose.get_Optimized_Parameters()
decomposition_error = NVDecompose.Optimization_Problem(parameters)
print(1-decomposition_error)

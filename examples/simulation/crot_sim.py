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

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file QC_sim_benchmark.py
## \brief Simple example python code demonstrating how to use state vector simulation with the SQUANDER package and compare the perfromance to QISKIT and Qulacs

from squander import Circuit       


import numpy as np
import random
import scipy.linalg
import time
import copy


np.set_printoptions(linewidth=200) 


# number of qubits
qbit_num_min = 2
qbit_num_max = 8

# number of levels
levels = 10

random_initial_state = False

##########################################################################################################
################################ SQUANDER #################################################################

execution_times_squander = {}
transformed_states_squander = {}
parameters_squander = {}
initial_state_squander      = {}

for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):
	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	if (random_initial_state ) :
		initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
		initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
		initial_state = initial_state_real + initial_state_imag*1j
		initial_state = initial_state/np.linalg.norm(initial_state)
	else:
		initial_state = np.zeros( (matrix_size,matrix_size), dtype=np.complex128 )
		#initial_state = np.zeros( (matrix_size,), dtype=np.complex128 )
		#initial_state[0] = 1.0 + 0j
		initial_state += (1.0+0.j)*np.eye(matrix_size,matrix_size)

	initial_state_squander[ qbit_num ] = copy.deepcopy(initial_state)

	# prepare circuit

	circuit_squander = Circuit( qbit_num )

	gates_num = 0
	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_squander.add_U3(target_qbit, True, True, True )
				circuit_squander.add_U3(control_qbit, True, True, True )
				circuit_squander.add_CNOT( target_qbit=target_qbit, control_qbit=control_qbit )
				#circuit_squander.ad_CRY( target_qbit=target_qbit, control_qinitial_state.sizebit=control_qbit )
				#gates_num = gates_num + 3

	for target_qbit in range(qbit_num):
		circuit_squander.add_U3(target_qbit, True, True, True )
		gates_num = gates_num + 1
		break		



	num_of_parameters = circuit_squander.get_Parameter_Num()
	#print("The number of free parameters at qubit_num= ", qbit_num, ": ", num_of_parameters )


	parameters = np.zeros(num_of_parameters)+3*np.pi/4#np.random.rand(num_of_parameters)*2*np.pi

	t0 = time.time()
	circuit_squander.apply_to( parameters, initial_state )
	t_SQUANDER = time.time() - t0
	print( "Time elapsed SQUANDER: ", t_SQUANDER, " seconds at qbit_num = ", qbit_num, ' number of gates: ', gates_num )

	execution_times_squander[ qbit_num ] = t_SQUANDER
	transformed_states_squander[ qbit_num ] = np.reshape(initial_state, (matrix_size,matrix_size) )
	#print(initial_state)
	parameters_squander[ qbit_num ] = parameters


print("SQUANDER execution times [s]:")
print( execution_times_squander )

from qulacs import Observable, QuantumCircuit, QuantumState
import qulacs

execution_times_qulacs = {}
transformed_states_qulacs = {}


for qbit_num in range(qbit_num_min, qbit_num_max+1, 1):

	# matrix size of the unitary
	matrix_size = 1 << qbit_num #pow(2, qbit_num )

	transformed_state = initial_state_squander[ qbit_num ]

	parameters = parameters_squander[ qbit_num ]
	parameter_idx = 0
	circuit_squander = Circuit( qbit_num )

	gates_num = 0
	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_squander.add_U3(target_qbit, True, True, True )
				circuit_squander.add_U3(control_qbit, True, True, True )
				circuit_squander.add_CROT( target_qbit=target_qbit, control_qbit=control_qbit )
				#circuit_squander.ad_CRY( target_qbit=target_qbit, control_qinitial_state.sizebit=control_qbit )
				#gates_num = gates_num + 3

	for target_qbit in range(qbit_num):
		circuit_squander.add_U3(target_qbit, True, True, True )
		gates_num = gates_num + 1
		break
	# prepare circuit
	#print( "Time elapsed qulacs: ", t_qulacs, " at qbit_num = ", qbit_num )
	num_of_parameters = circuit_squander.get_Parameter_Num()
	#print("The number of free parameters at qubit_num= ", qbit_num, ": ", num_of_parameters )


	parameters = np.zeros(num_of_parameters)+3*np.pi/4#np.random.rand(num_of_parameters)*2*np.pi

	t0 = time.time()
	circuit_squander.apply_to( parameters, transformed_state )
	t_qulacs = time.time()-t0
	execution_times_qulacs[ qbit_num ] = t_qulacs
	transformed_states_qulacs[ qbit_num ] = np.array(transformed_state)

print("Qulacs execution times [s]:")
print( execution_times_qulacs )
# check errors

print(' ')
print("Difference between the transformed state vectors:")
# SQUANDER-QISKIT-Qulacs comparision
keys = transformed_states_qulacs.keys()
for qbit_num in keys:
	state_squander = transformed_states_squander[ qbit_num ]
	
	#state_qiskit   = transformed_states_qiskit[ qbit_num ]
	state_qulacs   = transformed_states_qulacs[ qbit_num ]
	print(np.trace(np.dot(state_squander,state_qulacs.conj().T)))
	#print( "Squander vs QISKIT: ", np.linalg.norm( state_squander-state_qulacs ) )
	#print( "QISKIT vs Qulacs: ", np.linalg.norm( state_qiskit-state_qulacs ) )
	#print(state_qulacs)
	

"""
	# creating qulacs quantum circuit
	state = QuantumState(qbit_num)
	state.load( initial_state )

	circuit_qulacs = QuantumCircuit(qbit_num)

	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):

				circuit_qulacs.add_U3_gate(target_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
				parameter_idx = parameter_idx+3
				circuit_qulacs.add_U3_gate( control_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
				parameter_idx = parameter_idx+3

				circuit_qulacs.add_CNOT_gate(control_qbit,target_qbit)
				parameter_idx = parameter_idx+1
				#RY_gate = qulacs.gate.RotY( target_qbit, parameters[parameter_idx]*2 )
				#RY_gate = qulacs.gate.to_matrix_gate( RY_gate )
				#RY_gate.add_control_qubit( control_qbit, 1)
				#circuit_qulacs.add_gate( RY_gate )
				#circuit_qulacs.add_RotY_gate( target_qbit, parameters[parameter_idx]*2 )
				#parameter_idx = parameter_idx+1
				

	for target_qbit in range(qbit_num):
		circuit_qulacs.add_U3_gate( target_qbit, parameters[parameter_idx]*2, parameters[parameter_idx+1], parameters[parameter_idx+2] )
		parameter_idx = parameter_idx+3
		break
		

	t0 = time.time()
	# Execute and get the state vector
	circuit_qulacs.update_quantum_state( state )
	transformed_state = state.get_vector()
	t_qulacs = time.time() - t0"""

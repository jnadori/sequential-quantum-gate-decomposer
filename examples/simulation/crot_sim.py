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


np.set_printoptions(linewidth=200) 


# number of qubits
qbit_num_min = 2
qbit_num_max = 2

# number of levels
levels = 1

random_initial_state = False

##########################################################################################################
################################ SQUANER #################################################################

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
		initial_state[0,0] = 1.0 + 0j
		initial_state[2,0] = 2.0 + 0j
		initial_state[3,0] = 3.0 + 0j
		initial_state[1,1] = 1.0 + 0j
		initial_state[0,2] = 1.0 + 0j
		initial_state[2,2] = 3.0 + 0j
		initial_state[3,2] = 2.0 + 0j

	initial_state_squander[ qbit_num ] = initial_state.copy()

	# prepare circuit

	circuit_squander = Circuit( qbit_num )

	gates_num = 0
	for level in range(levels):

		# preparing circuit
		for control_qbit in range(qbit_num-1):
			for target_qbit in range(control_qbit+1, qbit_num):
				#circuit_squander.add_CNOT( target_qbit=target_qbit, control_qbit=control_qbit )
				circuit_squander.add_CROT( target_qbit=target_qbit, control_qbit=control_qbit )
				gates_num = gates_num + 1


	num_of_parameters = circuit_squander.get_Parameter_Num()
	#print("The number of free parameters at qubit_num= ", qbit_num, ": ", num_of_parameters )


	parameters = np.zeros(num_of_parameters) + 3*np.pi/4

	t0 = time.time()
	circuit_squander.apply_to( parameters, initial_state )
	t_SQUANDER = time.time() - t0
	print( "Time elapsed SQUANDER: ", t_SQUANDER, " seconds at qbit_num = ", qbit_num, ' number of gates: ', gates_num )

	execution_times_squander[ qbit_num ] = t_SQUANDER
	parameters_squander[ qbit_num ] = parameters


print( initial_state)


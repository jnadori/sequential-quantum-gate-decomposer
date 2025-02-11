import numpy as np
from os import path
from squander.decomposition.qgd_N_Qubit_Decomposition_custom_Wrapper import qgd_N_Qubit_Decomposition_custom_Wrapper
from squander.gates.qgd_Circuit import qgd_Circuit


##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_NV_Decomposition(qgd_N_Qubit_Decomposition_custom_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed.
# @param optimize_layer_num Set true to optimize the minimum number of operation layers required in the decomposition, or false when the predefined maximal number of layer gates is used (ideal for general unitaries).
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: "zeros" ,"random", "close_to_zero".
# @return An instance of the class
    def __init__( self, Umtx, initial_guess="RANDOM", config={}, accelerator_num=0 ):
        
        ## the number of qubits
        self.qbit_num = int(round( np.log2( len(Umtx) ) ))
        
        # config
        if not( type(config) is dict):
            print("Input parameter config should be a dictionary describing the following hyperparameters:") #TODO
            return

        # call the constructor of the wrapper class
        super(qgd_NV_Decomposition, self).__init__(Umtx, self.qbit_num, initial_guess, config=config, accelerator_num=accelerator_num)


    def get_Initial_Circuit(self, number_of_layers=1, topology=None):
        circuit_squander = qgd_Circuit( self.qbit_num )
        if topology is not None:
            #add nv layers
            for layer in range(number_of_layers):
                for involved_qbits in topology:
                    control_qbit,target_qbit = involved_qbits
                    circuit_squander.add_RZ(target_qbit)
                    circuit_squander.add_RX(target_qbit)                
                    circuit_squander.add_RZ(target_qbit)
                    circuit_squander.add_RZ(control_qbit)
                    circuit_squander.add_RX(control_qbit)                
                    circuit_squander.add_RZ(control_qbit)
                    circuit_squander.add_CROT(target_qbit=target_qbit,control_qbit=control_qbit)
        else:
            for layer in number_of_layers:
                for control_qbit in range(self.qbit_num-1):
                    for target_qbit in range(control_qbit+1,self.qbit_num):
                        circuit_squander.add_RZ(target_qbit)
                        circuit_squander.add_RX(target_qbit)                
                        circuit_squander.add_RZ(target_qbit)
                        circuit_squander.add_RZ(control_qbit)
                        circuit_squander.add_RX(control_qbit)                
                        circuit_squander.add_RZ(control_qbit)
                        circuit_squander.add_CROT(target_qbit=target_qbit,control_qbit=control_qbit)
        #add finalizing layer
        for qbit in range(self.qbit_num):
            circuit_squander.add_RZ(qbit)
            circuit_squander.add_RX(qbit)                
            circuit_squander.add_RZ(qbit)
        self.set_Gate_Structure(circuit_squander)

##
# @brief Wrapper function to call the start_decomposition method of C++ class N_Qubit_Decomposition
# @param prepare_export Logical parameter. Set true to prepare the list of gates to be exported, or false otherwise.
    def Start_Decomposition(self,prepare_export=True):

	# call the C wrapper function
        super(qgd_NV_Decomposition, self).Start_Decomposition(prepare_export=prepare_export)


##
# @brief Call to reorder the qubits in the matrix of the gate
# @param qbit_list The reordered list of qubits spanning the matrix
    def Reorder_Qubits( self, qbit_list ):

	# call the C wrapper function
        super(qgd_NV_Decomposition, self).Reorder_Qubits(qbit_list)


##
# @brief @brief Call to print the gates decomposing the initial unitary. These gates brings the intial matrix into unity.
    def List_Gates(self):

	# call the C wrapper function
        super(qgd_NV_Decomposition, self).List_Gates()


##
# @brief Call to retrieve the incorporated quantum circuit (Squander format)
# @return Return with a Qiskit compatible quantum circuit.
    def get_Circuit( self ):
        
        # call the C wrapper function
        return super().get_Circuit()

##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Qiskit_Circuit( self ):
    
        from squander import Qiskit_IO
        
        squander_circuit = self.get_Circuit()
        parameters       = self.get_Optimized_Parameters()
        
        return Qiskit_IO.get_Qiskit_Circuit( squander_circuit, parameters )





##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Cirq_Circuit( self ):

        import cirq


        # creating Cirq quantum circuit
        circuit = cirq.Circuit()

        # creating qubit register
        q = cirq.LineQubit.range(self.qbit_num)

        # retrive the list of decomposing gate structure
        gates = self.get_Gates()

        # constructing quantum circuit
        for idx in range(len(gates)-1, -1, -1):

            gate = gates[idx]

            if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
                circuit.append(cirq.CNOT(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            if gate.get("type") == "CRY":
                # adding CRY gate to the quantum circuit
                print("CRY gate needs to be implemented")

            elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
                circuit.append(cirq.CZ(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
                circuit.append(cirq.CH(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "SYC":
                # Sycamore gate
                circuit.append(cirq.google.SYC(q[self.qbit_num-1-gate.get("control_qbit")], q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "U3":
                print("Unsupported gate in the Cirq export: U3 gate")
                return None;

            elif gate.get("type") == "RX":
                # RX gate
                circuit.append(cirq.rx(gate.get("Theta")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "RY":
                # RY gate
                circuit.append(cirq.ry(gate.get("Theta")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "RZ":
                # RZ gate
                circuit.append(cirq.rz(gate.get("Phi")).on(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "H":
                # Hadamard gate
                circuit.append(cirq.h(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "X":
                # X gate
                circuit.append(cirq.x(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "Y":
                # Y gate
                circuit.append(cirq.y(q[self.qbit_num-1-gate.get("target_qbit")]))

            elif gate.get("type") == "Z":
                # Z gate
                circuit.append(cirq.z(q[self.qbit_num-1-gate.get("target_qbit")]))


            elif gate.get("type") == "SX":
                # RZ gate
                circuit.append(cirq.sx(q[self.qbit_num-1-gate.get("target_qbit")]))


        return circuit

        
##
# @brief Call to import initial quantum circuit in QISKIT format to be further comporessed
# @param qc_in The quantum circuit to be imported
    def import_Qiskit_Circuit( self, qc_in ):  

        from squander import Qiskit_IO
        
        
        Circuit_Squander, circuit_parameters = Qiskit_IO.convert_Qiskit_to_Squander( qc_in )
        
        # setting gate structure and optimized initial parameters
        self.set_Gate_Structure( Circuit_Squander )   
        self.set_Optimized_Parameters( circuit_parameters )      
        


## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param optimizer String indicating the optimizer. Possible values: "BFGS" ,"ADAM", "BFGS2", "AGENTS", "COSINE", "AGENTS_COMBINED".
# @return An instance of the class
    def set_Optimizer( self, optimizer="BFGS" ):

        # Set the optimizer
        super(qgd_NV_Decomposition, self).set_Optimizer(optimizer)  



## 
# @brief Call to prepare the circuit to be exported into Qiskit format. (parameters and gates gets bound together, gate block structure is converted to plain structure).
    def Prepare_Gates_To_Export(self):

        # Set the optimizer
        super(qgd_NV_Decomposition, self).Prepare_Gates_To_Export()  
        
        
##
# @brief Call to set custom gate structure to used in the decomposition
# @param Gate_structure An instance of Gates_Block
    def set_Gate_Structure( self, Gate_structure ):  

        if not isinstance(Gate_structure, qgd_Circuit) :
            raise Exception("Input parameter Gate_structure should be a an instance of Gates_Block")
               
                          
        super(qgd_NV_Decomposition, self).set_Gate_Structure( Gate_structure )   



## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param costfnc Variant of the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2, 3 for HILBERT_SCHMIDT_TEST
    def set_Cost_Function_Variant( self, costfnc=0 ):

        # Set the optimizer
        super(qgd_NV_Decomposition, self).set_Cost_Function_Variant(costfnc=costfnc)         


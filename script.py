
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
import matplotlib.pyplot as plt
from tqdm import tqdm
decomposition_error=np.logspace(-1,-6,11)
samples=50
for N in range(5,6):
    print("############################################################")
    print("##########starting decomposition for "+str(N)+" qubits###############")
    print("############################################################")
    CROT_gates=np.zeros(shape=(decomposition_error.shape[0],samples))
    layer_step=1
    if N>3:
        minimum_layer=7*(N-3)**2
        layer_step=3*(N-3)
    else:
        minimum_layer = 0
    matrix_size = 2**N
    topology=[]
    for i in range(1,N):
        topology.append((0,i))
    for n in tqdm(range(samples)):
        Umtx = unitary_group.rvs(matrix_size)
        NVDecompose = NV_Decomposition( Umtx )
        NVDecompose.set_Optimizer("BFGS")
        NVDecompose.set_Verbose(0)
        NVDecompose.set_Cost_Function_Variant( 8 )
        for decomp_tol_idx in range(decomposition_error.shape[0]):
            decomp_tol = decomposition_error[decomp_tol_idx]
            NVDecompose.set_Optimization_Tolerance( decomp_tol )
            levels = minimum_layer
            if decomp_tol_idx !=0:
                parameters=NVDecompose.get_Optimized_Parameters()
                decomposition_error_current = NVDecompose.Optimization_Problem(parameters)
                if decomposition_error_current<decomp_tol:
                    quantum_circuit = NVDecompose.get_Circuit()
                    count = quantum_circuit.get_Gate_Nums()
                    CROT_gates[decomp_tol_idx][n] = 0
                    if "CROT" in count.keys():
                        CROT_gates[decomp_tol_idx][n] += count['CROT']
                    np.savetxt(str("Decomp_"+str(N)+"_qbits_CROT.txt"),CROT_gates)
                    continue                
            while (levels<N*100):
                NVDecompose.get_Initial_Circuit("CONTROL_OPPOSITE",levels,topology,True)
                circuit = NVDecompose.get_Circuit()
                NVDecompose.set_Optimized_Parameters(np.zeros(circuit.get_Parameter_Num()))
                NVDecompose.Start_Decomposition()
                parameters=NVDecompose.get_Optimized_Parameters()
                decomposition_error_current = NVDecompose.Optimization_Problem(parameters)
                if decomposition_error_current<decomp_tol:
                    quantum_circuit = NVDecompose.get_Circuit()
                    count = quantum_circuit.get_Gate_Nums()
                    CROT_gates[decomp_tol_idx][n] = 0
                    if "CROT" in count.keys():
                        CROT_gates[decomp_tol_idx][n] += count['CROT']
                    np.savetxt(str("Decomp_"+str(N)+"_qbits_CROT.txt"),CROT_gates)
                    break
                levels += layer_step

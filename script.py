
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
decomposition_error=np.logspace(-1,-11,10)
samples=100
for N in range(2,6):
    CROT_gates=np.zeros(shape=(decomposition_error.shape[0],samples))
    minimum_layer = 0
    matrix_size = 2**N
    topology=[]
    for i in range(1,N):
        topology.append((0,i))
    for decomp_tol_idx in range(decomposition_error.shape[0]):
        decomp_tol = decomposition_error[decomp_tol_idx]
        for n in tqdm(range(samples)):
            Umtx = unitary_group.rvs(matrix_size)
            NVDecompose = NV_Decomposition( Umtx, level_limit_max=N*100,level_limit_min=minimum_layer,topology=topology  )
            NVDecompose.set_Optimizer("BFGS")
            NVDecompose.set_Verbose(0)
            NVDecompose.set_Cost_Function_Variant( 3 )
            NVDecompose.set_Optimization_Tolerance( decomp_tol )
            NVDecompose.get_Initial_Circuit()
            quantum_circuit = NVDecompose.get_Qiskit_Circuit()
            count = dict(quantum_circuit.count_ops())
            if "cry" not in count.keys():
                CROT_gates[decomp_tol_idx][n] = 0
            else:
                CROT_gates[decomp_tol_idx][n] = dict(quantum_circuit.count_ops())['cry']
        minimum_layer = int(np.min(CROT_gates[decomp_tol_idx]))
    np.savetxt(str("Decomp_"+str(N)+"_qbits.txt"),CROT_gates)
    y=np.mean(CROT_gates,axis=1)
    yerr=np.std(CROT_gates,axis=1)
    print(y,yerr)
    print(CROT_gates)
    plt.errorbar(decomposition_error,y,yerr=yerr)
    plt.xlabel("Decomposition Tolerance")
    plt.ylabel("Number of CROT gates")
    plt.title(str(N)+" qubit decomposition")
    plt.xscale("log")
    plt.savefig("Decomp_"+str(N)+"_qbits.png")

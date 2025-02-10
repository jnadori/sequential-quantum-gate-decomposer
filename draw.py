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
N=2
decomposition_error=np.logspace(-1,-10,20)
samples=100
plt.figure(figsize=(10,6))
CROT_gates=np.loadtxt(str("Decomp_"+str(N)+"_qbits_NV.txt"))
y_crot=np.mean(CROT_gates,axis=1)
yerr_crot=np.std(CROT_gates,axis=1)#np.array([np.min(CROT_gates,axis=1),np.max(CROT_gates,axis=1)])
plt.errorbar(decomposition_error,y_crot,yerr=yerr_crot,fmt='o-',label="CROT", capsize=5,color="blue",alpha=0.75)

CNOT_gates=np.loadtxt(str("Decomp_"+str(N)+"_qbits_CNOT.txt"))
y_cnot=np.mean(CNOT_gates,axis=1)
yerr_cnot=np.std(CNOT_gates,axis=1)#np.array([np.min(CNOT_gates,axis=1),np.max(CNOT_gates,axis=1)])
plt.errorbar(decomposition_error,y_cnot,yerr=yerr_cnot,fmt='x-',label="CNOT", capsize=5,color="green",alpha=0.75)

plt.xlabel("Decomposition Tolerance")
plt.ylabel("Number of control gates")
plt.title(str(N)+" qubit decomposition")
plt.xscale("log")
plt.legend()
plt.savefig("Decomp_"+str(N)+"_qbits_std.png")

"""
CROT_gates_nt=np.loadtxt(str("Decomp_"+str(N)+"_qbits_NV_no_topology.txt"))
y_crot_nt=np.mean(CROT_gates_nt,axis=1)
yerr_crot_nt=np.std(CROT_gates_nt,axis=1)#np.array([np.min(CROT_gates_nt,axis=1),np.max(CROT_gates_nt,axis=1)])
plt.errorbar(decomposition_error,y_crot_nt,yerr=yerr_crot_nt,fmt='*-',label="CROT no topology", capsize=5,color="red",alpha=0.75)

CNOT_gates_nt=np.loadtxt(str("Decomp_"+str(N)+"_qbits_CNOT_no_topology.txt"))
y_cnot_nt=np.mean(CNOT_gates_nt,axis=1)
yerr_cnot_nt=np.std(CNOT_gates_nt,axis=1)#np.array([np.min(CNOT_gates_nt,axis=1),np.max(CNOT_gates_nt,axis=1)])
plt.errorbar(decomposition_error,y_cnot_nt,yerr=yerr_cnot_nt,fmt='D-',label="CNOT no topology", capsize=5,color="purple",alpha=0.75)
"""

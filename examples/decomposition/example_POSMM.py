from squander import POSMM_Decomposition, Circuit, Qiskit_IO
import numpy as np
import time 
def construct_unitary(N,depth):
    circ = Circuit(N)
    for idx in range(depth):
        circo = Circuit(N)
        trgt,ctrl = np.random.choice(np.arange(N),2,replace=False)
        circo.add_U3(trgt)
        circo.add_U3(ctrl)
        circo.add_CNOT(trgt,ctrl)
        circ.add_Circuit(circo)
    for idx in range(N):
        circo = Circuit(N)
        circo.add_U3(idx)
        circ.add_Circuit(circo)

    return circ,circ.get_Matrix(np.random.randn(circ.get_Parameter_Num())*2*np.pi)
N = 3
d = 7
Circ, Umtx = construct_unitary(N,d)
config = {'parallel':0,'optimization_loops':100,'tolerance':1e-7,'worker_num':8}
NVDecompose = POSMM_Decomposition(Umtx.conj().T,Circ, 0.5, config = config)
start = time.time()
params,score = NVDecompose.Start_decomposition()
compilation_time = time.time()-start
print(score,compilation_time)

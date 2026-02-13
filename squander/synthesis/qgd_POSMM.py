import numpy as np 
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import qgd_N_Qubit_Decomposition_custom as N_Qubit_Decomposition_custom
from time import time

class qgd_POSMM_Decomposition:
    def __init__(self, Umtx, gate_structure, rconstant, config):
        self.Umtx = Umtx
        self.gate_structure = gate_structure
        self.qbit_num = gate_structure.get_Qbit_Num()
        self.d = gate_structure.get_Parameter_Num()
        self.config = config
        self.rconstant = rconstant
        self.optimization_loops = self.config['optimization_loops']
        self.worker_num = self.config['worker_num']
        self.tolerance = self.config['tolerance']

    def Start_decomposition(self):
        previous_points = np.empty((0, self.d))
        promising_points = []
        best_parameters = np.zeros(self.d)
        best_score = np.inf
        true_optim_time = 0
        for k in range(self.optimization_loops):
            r = self.rconstant*(np.log(k+1)/(k+1))**(1./self.d)
            while len(promising_points)<self.worker_num:
                if not self.generate_new_random_point(promising_points,previous_points,r):
                    return best_parameters, best_score
            points_to_optimize = promising_points[:self.worker_num]
            promising_points.clear()
            for point in points_to_optimize:
                worker_point, worker_score, worker_time = self.optimize_point(point)
                true_optim_time += worker_time
                if worker_score < self.tolerance:
                    return worker_point, worker_score
                if worker_score<best_score:
                    best_parameters = worker_point
                    best_score = worker_score
                previous_points = np.vstack([previous_points, point])
                Deltax = np.abs(previous_points - worker_point)
                deltax = np.minimum(Deltax, 2*np.pi - Deltax)
                distances = np.sqrt(np.sum(deltax**2, axis=1))
                if np.any(distances < r):
                    continue
                promising_points.append(worker_point)
        return best_parameters,best_score

    def optimize_point(self, start_point):
        wDecompose = N_Qubit_Decomposition_custom(self.Umtx,config = self.config)
        wDecompose.set_Verbose(0)
        wDecompose.set_Cost_Function_Variant( 3 )
        wDecompose.set_Optimization_Tolerance( self.tolerance )
        wDecompose.set_Gate_Structure(self.gate_structure)
        wDecompose.set_Optimized_Parameters(start_point)
        start = time()
        wDecompose.Start_Decomposition()
        worker_time = time()-start
        end_point=wDecompose.get_Optimized_Parameters()
        score = wDecompose.Optimization_Problem(end_point)
        return end_point,score, worker_time

    def generate_new_random_point(self, promising_points, previous_points, r):
        for trial in range(500):
            new_point = np.random.rand(self.d)*2*np.pi
            if len(previous_points) > 0 or len(promising_points) > 0:
                parts = [previous_points] if len(previous_points) > 0 else []
                if len(promising_points) > 0:
                    parts.append(np.array(promising_points))
                all_points = np.vstack(parts)
                Deltax = np.abs(all_points - new_point)
                deltax = np.minimum(Deltax, 2*np.pi - Deltax)
                distances = np.sqrt(np.sum(deltax**2, axis=1))
                if np.any(distances < r):
                    continue

            promising_points.append(new_point)
            return True
        return False
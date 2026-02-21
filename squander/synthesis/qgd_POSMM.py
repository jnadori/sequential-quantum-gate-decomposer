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
        self.screen_iterations = self.config.get('screen_iterations', 100)
        self.refine_count = self.config.get('refine_count', max(1, self.worker_num // 4))
        self.max_stagnation = self.config.get('max_stagnation', 5)
        self.perturbation_scale = self.config.get('perturbation_scale', 0.5)
        self.refinement_attempts = self.config.get('refinement_attempts', 20)
        self.refinement_scales = self.config.get('refinement_scales', [0.1, 0.3, 1.0])

    def Start_decomposition(self):
        previous_points = np.empty((0, self.d))
        promising_points = []
        best_parameters = np.zeros(self.d)
        best_score = np.inf
        stagnation_counter = 0
        prev_best_score = np.inf

        for k in range(self.optimization_loops):
            r = self.rconstant*(np.log(k+1)/(k+1))**(1./self.d)

            # Fill promising_points up to worker_num
            generation_failed = False
            while len(promising_points) < self.worker_num:
                if not self.generate_new_random_point(promising_points, previous_points, r):
                    # Fall back to perturbation from best known point
                    if best_score < np.inf:
                        if not self.generate_perturbed_point(best_parameters, promising_points, previous_points, r):
                            generation_failed = True
                            break
                    else:
                        generation_failed = True
                        break

            # Graceful degradation: optimize whatever points we have
            if len(promising_points) == 0:
                break

            batch_size = min(len(promising_points), self.worker_num)
            points_to_optimize = promising_points[:batch_size]
            promising_points = promising_points[batch_size:]  # preserve excess

            # Two-phase: screen then refine
            results = self.screen_and_optimize(points_to_optimize)

            for start_point, end_point, score in results:
                if score < self.tolerance:
                    return end_point, score
                if score < best_score:
                    best_parameters = end_point
                    best_score = score
                previous_points = np.vstack([previous_points, start_point])
                # Recycle optimized endpoint if far enough from known points
                Deltax = np.abs(previous_points - end_point)
                deltax = np.minimum(Deltax, 2*np.pi - Deltax)
                distances = np.sqrt(np.sum(deltax**2, axis=1))
                if not np.any(distances < r):
                    promising_points.append(end_point)

            # Stagnation detection
            if best_score < prev_best_score * 0.999:
                stagnation_counter = 0
                prev_best_score = best_score
            else:
                stagnation_counter += 1

            if stagnation_counter >= self.max_stagnation:
                break

            if generation_failed and len(promising_points) == 0:
                break

        # Local refinement phase: try perturbations around the best found point
        if best_score > self.tolerance and best_score < np.inf:
            best_parameters, best_score = self.local_refinement(best_parameters, best_score)

        return best_parameters, best_score

    def local_refinement(self, best_parameters, best_score):
        """Try small perturbations around best point at multiple scales."""
        for scale in self.refinement_scales:
            for _ in range(self.refinement_attempts):
                perturbed = (best_parameters + np.random.randn(self.d) * scale) % (2 * np.pi)
                end_point, score, _ = self.optimize_point(perturbed)
                if score < best_score:
                    best_parameters = end_point
                    best_score = score
                if score < self.tolerance:
                    return best_parameters, best_score
        return best_parameters, best_score

    def screen_and_optimize(self, points_to_optimize):
        """Screen candidates with short BFGS, then fully optimize the best ones."""
        # Phase 1: Quick screen with limited iterations
        screened = []
        for point in points_to_optimize:
            wDecompose = N_Qubit_Decomposition_custom(self.Umtx, config=self.config)
            wDecompose.set_Verbose(0)
            wDecompose.set_Cost_Function_Variant(3)
            wDecompose.set_Optimizer(self.config.get('optimizer', 'BFGS'))
            wDecompose.set_Optimization_Tolerance(self.tolerance)
            wDecompose.set_Gate_Structure(self.gate_structure)
            wDecompose.set_Optimized_Parameters(point)
            wDecompose.set_Max_Iterations(self.screen_iterations)
            wDecompose.Start_Decomposition()
            end_point = wDecompose.get_Optimized_Parameters()
            score = wDecompose.Optimization_Problem(end_point)
            if score < self.tolerance:
                return [(point, end_point, score)]
            screened.append((score, point, end_point))

        # Phase 2: Fully optimize best candidates, warm-started from screen result
        screened.sort(key=lambda x: x[0])
        results = []
        for _, start_point, screened_point in screened[:self.refine_count]:
            end_point, score, _ = self.optimize_point(screened_point)
            results.append((start_point, end_point, score))
        return results

    def optimize_point(self, start_point):
        wDecompose = N_Qubit_Decomposition_custom(self.Umtx, config=self.config)
        wDecompose.set_Verbose(0)
        wDecompose.set_Cost_Function_Variant(3)
        wDecompose.set_Optimizer(self.config.get('optimizer', 'BFGS'))
        wDecompose.set_Optimization_Tolerance(self.tolerance)
        wDecompose.set_Gate_Structure(self.gate_structure)
        wDecompose.set_Optimized_Parameters(start_point)
        start = time()
        wDecompose.Start_Decomposition()
        worker_time = time()-start
        end_point = wDecompose.get_Optimized_Parameters()
        score = wDecompose.Optimization_Problem(end_point)
        return end_point, score, worker_time

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

    def generate_perturbed_point(self, best_parameters, promising_points, previous_points, r):
        """Generate a new point by perturbing the best known solution."""
        for trial in range(100):
            perturbation = np.random.randn(self.d) * self.perturbation_scale
            new_point = (best_parameters + perturbation) % (2 * np.pi)
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

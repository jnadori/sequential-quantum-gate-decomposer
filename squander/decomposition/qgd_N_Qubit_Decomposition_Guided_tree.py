"""
Implementation to optimize wide circuits (i.e. circuits with many qubits) by    partitioning the circuit into smaller partitions and redecompose the smaller partitions
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
    qgd_N_Qubit_Decomposition_Surrogate as N_Qubit_Decomposition_Surrogate
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology

def CNOTGateCount( circ: Circuit, max_gates: int = 0 ) -> int :
    """
    Call to get the number of CNOT gates in the circuit

    
    Args:

        circ (Circuit) A squander circuit representation


    Return:

        Returns with the CNOT gate count

    
    """ 

    if not isinstance(circ, Circuit ):
        raise Exception("The input parameters should be an instance of Squander Circuit")

    gate_counts = circ.get_Gate_Nums()
    num_cnots = gate_counts.get('CNOT', 0)
    
    if max_gates > 0: return num_cnots*max_gates + sum(y for x, y in gate_counts.items() if x !='CNOT')
    return num_cnots  #+  3*gate_counts.get('SWAP', 0)


class N_Qubit_Decomposition_Guided_Tree(N_Qubit_Decomposition_custom):
    def __init__(self, Umtx, config, accelerator_num, topology, paramspace=None, paramscale=None):
        super().__init__(Umtx[0] if isinstance(Umtx, list) else Umtx, config=config, accelerator_num=accelerator_num)
        self.Umtx = Umtx if isinstance(Umtx, list) else [Umtx] #already conjugate transposed
        self.qbit_num = self.Umtx[0].shape[0].bit_length() -1
        self.config = config
        self.accelerator_num = accelerator_num
        self.paramspace = paramspace
        self.paramscale = () if paramscale is None else paramscale
        #self.set_Cost_Function_Variant( 0 )	 #0 is Frobenius, 3 is HS, 10 is OSR
        if topology is None:
            topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        self.topology = topology
    def enumerate_unordered_cnot_BFS(n: int, topology=None, use_gl=True):
        # Precompute unordered pairs
        topology = [(i, j) for i in range(n) for j in range(i+1, n)] if topology is None else topology
        prior_level_info = None
        while True:
            visited, seq_pairs_of, seq_dir_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(n, topology, prior_level_info, use_gl=use_gl)
            if not res: break
            yield res
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res)))
    def canonical_prefix_ok(seq):
        m = len(seq)
        if m <= 1: return -1
        succ = {}
        indeg = {}
        last_on = {}
        for k in range(m):
            for q in seq[k]:
                if q in last_on:
                    p = last_on[q]
                    succ.setdefault(p, []).append(k)
                    indeg[k] = indeg.get(k, 0) + 1
                last_on[q] = k
        import heapq
        pq = [(seq[x], x) for x in range(m) if indeg.get(x, 0) == 0]
        heapq.heapify(pq)
        for pos in range(m):
            # Kahn's algorithm
            if len(pq) == 0: return pos #malformed (shouldn't happen)
            u = heapq.heappop(pq)
            if u[1] != pos: return pos  #deviation: not canonical
            for v in succ.get(u[1], ()):
                indeg[v] -= 1
                if indeg[v] == 0: heapq.heappush(pq, (seq[v], v))
        return -1
    def enumerate_unordered_cnot_BFS_level(n: int, topology=None, prior_level_info=None, use_gl=True):
        """
        Enumerate GL(n,2) states in increasing CNOT depth.
        Moves are *recorded* as unordered pairs (for your "structure" view)
        but each expansion tries both directions internally.

        Yields: (depth, A, seq_pairs, seq_directed)
        - depth: minimal number of CNOTs
        - A: packed matrix (tuple of n bit-rows)
        - seq_pairs: minimal-length sequence of unordered pairs that reaches A
        - seq_directed: a matching directed-move realization of seq_pairs
        """
        if prior_level_info is None:
            # Initial state
            start_key = tuple(1 << i for i in range(n))

            # Visited: we only need to mark states once (minimal depth)
            visited = {start_key}

            # We also keep *one* representative sequence per state (unordered + directed)
            seq_pairs_of = {start_key: []}
            seq_dir_of = {start_key: []}

            # Yield the root
            return visited, seq_pairs_of, seq_dir_of, [(start_key, [], [])]
        else:
            visited, seq_pairs_of, seq_dir_of, q = prior_level_info
        res = []
        new_seq_pairs_of = {}
        new_seq_dir_of = {}

        while q:
            A = q.pop()
            last_pairs = seq_pairs_of[A]
            last_dirs = seq_dir_of[A]
            for p in topology:
                if not use_gl:
                    if len(last_pairs) >= 3 and all(p==x for x in last_pairs[-3:]): continue # avoid more than 3 repeated CNOTs
                    if N_Qubit_Decomposition_Guided_Tree.canonical_prefix_ok(last_pairs + [p]) >= 0: continue  # not canonical prefix
                # Try both directions, but record the *same* unordered step 'p'
                for mv in (p, (p[1], p[0])) if use_gl else (p,):
                    #CNOT left
                    if use_gl:
                        if mv[0] == mv[1]: B = A
                        else: B = list(A); B[mv[1]] ^= B[mv[0]]; B = tuple(B)
        
                        if B in visited: continue  # already discovered at minimal depth
                    else: B = tuple(last_dirs + [p])

                    visited.add(B)
                    new_seq_pairs_of[B] = last_pairs + [p]
                    new_seq_dir_of[B] = last_dirs + [mv]

                    # Emit as soon as we discover the state (BFS → minimal depth)
                    res.append((B, new_seq_pairs_of[B], new_seq_dir_of[B]))
        return visited, new_seq_pairs_of, new_seq_dir_of, res
    def build_sequence(stop=5, ordered=True, use_gl=True):
        #https://oeis.org/A002884
        #unordered sequence: 1, 1, 4, 88, 9556, 4526605
        #unordered at 5 qubits: {0: 1, 1: 10, 2: 85, 3: 650, 4: 4475, 5: 27375, 6: 142499, 7: 580482, 8: 1501297, 9: 1738232, 10: 517884, 11: 13591, 12: 24} 
        for i in range(2, stop+1):
            d = {}
            for z in N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(i, use_gl=use_gl):
                for x in (list if ordered else set)(tuple(x[1]) for x in z):
                    d[len(x)] = d.get(len(x), 0) + 1
                if not use_gl and len(d) > 5: break
            print({x: d[x] for x in sorted(d)}, sum(d.values()))
    def extract_bits(x, pos):
        return sum(((x >> p) & 1) << i for i, p in enumerate(pos))
    def build_osr_matrix(U, n, A):
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        dA = 1 << len(A)
        dB = 1 << len(B)
        return U.reshape([2]*(2*n)).transpose(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)).reshape(dA*dA, dB*dB)
    def accumulate_grad_for_cut(U, G, Umat, VTmat, n, A): # qubits on A
        A = list(reversed(A))
        B = list(sorted(set(range(n)) - set(A), reverse=True))
        A, B = [n-1-q for q in A], [n-1-q for q in B]
        mat = np.array(G) * Umat @ VTmat  # reconstruct U from its dyadic decomposition
        revmap = [None]*(2*n)
        for i, x in enumerate(tuple(A) + tuple(t+n for t in A) + tuple(B) + tuple(t+n for t in B)):
            revmap[x] = i
        U += mat.reshape([2]*(2*n)).transpose(tuple(revmap)).reshape(*U.shape)
        return U
    def trace_out_qubits(U, n, A):
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A)
        M = np.linalg.svd(M, compute_uv=True, full_matrices=False)[0][:,0].reshape(1<<len(A), 1<<len(A))
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(M)
    def numerical_rank_osr(M, Fnorm, tol=1e-10):
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False) / Fnorm
        #print(s)
        return int(np.sum(s >= s[0]*tol)), s
    def operator_schmidt_rank(U, n, A, Fnorm, tol=1e-10):
        return N_Qubit_Decomposition_Guided_Tree.numerical_rank_osr(N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, n, A), Fnorm, tol)
    def unique_cuts(n):
        import itertools
        """All nontrivial unordered bipartitions (no complements)."""
        qubits = tuple(range(n))
        for r in range(1, n//2 + 1):  # only up to half
            for S in itertools.combinations(qubits, r):
                if r < n - r:
                    yield S
                else:  # r == n-r (only possible when n even): tie-break
                    comp = tuple(q for q in qubits if q not in S)
                    if S < comp:      # lexicographically smaller tuple wins
                        yield S
    def get_circuit_from_pairs(self, pairs, finalizing=True):
        circ = Circuit(self.qbit_num)
        for pair in pairs:
            circ.add_U3(pair[0])
            circ.add_U3(pair[1])
            circ.add_CNOT(pair[0], pair[1])
        if finalizing:
            for qbit in range(self.qbit_num):
                circ.add_U3(qbit)
        return circ
    def ceil_log2(x): return 0 if x == 0 else (x-1).bit_length()
    def logsumexp_smoothmax(Lc, tau=1e-2):
        if not Lc: return 0.0
        if tau <= 0.0: raise RuntimeError("tau must be > 0")
        m = max(Lc)
        acc = 0.0
        for v in Lc: acc += np.exp((v - m)/tau)
        return tau * np.log(acc) + m
    def loss_for_rank(S, rank):
        start = 1 << rank
        if start >= len(S): return 0.0
        return sum(x*x for x in S[start:])
    def avg_loss(cuts_S, rank):
        if not cuts_S: return 0.0
        total_loss = 0.0
        for S in cuts_S:
            total_loss += N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank)
        return total_loss / len(cuts_S)
    # Aggregated cost over cuts: softmax (log-sum-exp) of per-cut dyadic losses
    def cuts_softmax_rank_cost(cuts_S, rank, tau=1e-2):
        Lc = []
        for S in cuts_S:
            Lc.append(N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank))
        return N_Qubit_Decomposition_Guided_Tree.logsumexp_smoothmax(Lc, tau)

    # Gradient w.r.t. the singular values (diagonal of dL/dΣ):
    def loss_for_rank_grad_diag(S, rank, Fnorm):
        """
        Gradient of a single-cut tail loss with respect to the RAW singular values,
        assuming S is already normalized and Fnorm is treated as constant.

        If S = sigma / Fnorm, then d/dsigma_i sum_{j>=r} S_j^2 = 2*S_i/Fnorm on tail.
        """
        n = len(S)
        start = 1 << rank
        grad = [0.0] * n
        if start >= n:
            return grad
        invF = 1.0 / Fnorm
        for i in range(start, n):
            grad[i] = 2.0 * S[i] * invF
        return grad
    def cuts_avg_rank_grad(cuts_S, rank, Fnorm):
        """
        Gradient of average tail loss across cuts.
        Returns one gradient vector per cut, same length as that cut's S.
        """
        C = len(cuts_S)
        if C == 0:
            return []
        scale = 1.0 / C
        out = []
        for S in cuts_S:
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(S, rank, Fnorm)
            out.append([scale * v for v in g])
        return out
    # Gradient w.r.t. singular values (same length as S).
    def cuts_softmax_rank_grad(cuts_S, rank, Fnorm, tau=1e-2):
        """
        Gradient of smooth-max across cuts:
            L = tau * log(sum_c exp(L_c / tau))
        so
            dL = sum_c softmax_c * dL_c
        """
        C = len(cuts_S)
        if C == 0:
            return []
        if tau <= 0.0:
            raise RuntimeError("tau must be > 0")

        Lc = [
            N_Qubit_Decomposition_Guided_Tree.loss_for_rank(S, rank)
            for S in cuts_S
        ]

        m = max(Lc)
        w = [np.exp((v - m) / tau) for v in Lc]
        Z = np.sum(w)
        if Z <= 0.0:
            Z = 1.0
        w = [x / Z for x in w]

        out = []
        for c, S in enumerate(cuts_S):
            g = N_Qubit_Decomposition_Guided_Tree.loss_for_rank_grad_diag(S, rank, Fnorm)
            out.append([w[c] * v for v in g])
        return out
    # Build M with build_osr_matrix, then SVD (econ) and grab top triplet.
    def top_k_triplet_for_cut(U, # (N x N), row-major, N = 1<<q
        q,                  # number of qubits
        A,  # qubits on side A
        Fnorm            # e.g., sqrt(N)
        ):
        # 1) Build M for this cut    
        M = N_Qubit_Decomposition_Guided_Tree.build_osr_matrix(U, q, A)
        k = min(M.shape)

        # 2) SVD: M = U * diag(S) * VT  (VT = V^H)
        # Row-major API handles leading dims as col counts.
        res = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        return res.S / Fnorm, res.U, res.Vh # normalized singular value
    def get_deriv_osr_entanglement(matrix, use_cuts, rank, use_softmax):
        qbit_num = len(matrix).bit_length()-1
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(qbit_num)) if len(use_cuts) == 0 else use_cuts
        Fnorm = np.sqrt(len(matrix))
        deriv = np.zeros(matrix.shape, dtype=complex)
        # Compute the derivative of the OSR entanglement cost function
        triplets = []
        allS = []
        for cut in cuts:
            # 1) top k triplet on the normalized reshape M_c
            S, Umat, VTmat = N_Qubit_Decomposition_Guided_Tree.top_k_triplet_for_cut(matrix, qbit_num, cut, Fnorm)
            triplets.append(([], Umat, VTmat))
            allS.append(S)
        if use_softmax: allS = N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_grad(allS, rank, Fnorm)
        else: allS = N_Qubit_Decomposition_Guided_Tree.cuts_avg_rank_grad(allS, rank, Fnorm)
        for i in range(len(cuts)):
            triplets[i] = (allS[i], triplets[i][1], triplets[i][2])
        for i in range(len(cuts)):
            G, Umat, VTmat = triplets[i]
            N_Qubit_Decomposition_Guided_Tree.accumulate_grad_for_cut(deriv, G, Umat, VTmat, qbit_num, cuts[i])
        return deriv
    # Compute grad component = Re Tr( A^† B ) for A = dL/dU, B = dU/dθ
    # A and B are (rows x cols) with row-major leading dimension.
    def real_trace_conj_dot(A, B):
        return np.sum(A.real * B.real + A.imag * B.imag) # Re Tr(A^† B)
    def param_derivs(circ, Umtx, x):
        n = len(x)
        derivs = [None]*n
        for i in range(n):
            kind = i % 3
            if kind == 0: # d/dt:  ∂U/∂t = U(t+π/2, φ, λ)
                x_shift = x.copy()
                x_shift[i] += np.pi/2
                Ui = Umtx.copy()
                circ.apply_to(x_shift, Ui)
                derivs[i] = Ui
            else: # d/dφ or d/dλ: ∂U/∂p = 0.5*(U(p+π/2) - U(p-π/2))
                xp = x.copy(); xp[i] += np.pi/2
                xm = x.copy(); xm[i] -= np.pi/2
                Up = Umtx.copy()
                Um = Umtx.copy()
                circ.apply_to(xp, Up)
                circ.apply_to(xm, Um)
                derivs[i] = 0.5 * (Up - Um)
        return derivs
    
    def _global_phase_fix(U):
        return U / (np.linalg.det(U)**(1/len(U)))
    def _polar_unitary(X):
        U,_,Vh = np.linalg.svd(X, full_matrices=False)
        return U @ Vh

    def su2_to_u3_zyz(U):
        """
        Decompose a 2x2 unitary (det=1) into Qiskit U3: Rz(phi) @ Ry(theta) @ Rz(lam).
        Returns (theta, phi, lam) in radians.
        """
        U = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U)
        # Handle numeric edge cases robustly
        a = U[0,0]; b = U[0,1]; c = U[1,0]; d = U[1,1]
        # Prefer arccos for theta; it's stable when |a| is not tiny
        ca = np.clip(np.abs(a), 0.0, 1.0)
        theta = 2.0 * np.arccos(ca)
        # If sin(theta/2) ~ 0, collapse to Z rotations
        eps = 1e-12
        if abs(np.sin(theta/2)) < eps:
            # Then c≈0, b≈0; only Z phases matter: U ≈ e^{iα} Rz(phi+lam)
            # Choose phi=0, lam = arg(d) - arg(a)
            phi = 0.0
            lam = (np.angle(d) - np.angle(a))
            # Normalize to [-pi,pi)
            lam = (lam + np.pi)%(2*np.pi) - np.pi
            return float(theta), float(phi), float(lam)

        # Otherwise, phases from elements and normalize
        phi = (np.angle(c)-np.angle(a)); phi=(phi+np.pi)%(2*np.pi)-np.pi
        lam = (np.angle(b)-np.angle(a)-np.pi); lam=(lam+np.pi)%(2*np.pi)-np.pi
        return float(theta),float(phi),float(lam)

    def _A_from_c(c1,c2,c3):
        X = np.array([[0,1],[1,0]], complex)
        Y = np.array([[0,-1j],[1j,0]], complex)
        Z = np.array([[1,0],[0,-1]], complex)
        XX = np.kron(X,X); YY = np.kron(Y,Y); ZZ = np.kron(Z,Z)
        H = c1*XX + c2*YY + c3*ZZ
        # use exp via eig (4x4) for robustness
        ew, EV = np.linalg.eig(1j*H)
        A = EV @ np.diag(np.exp(ew)) @ np.linalg.inv(EV)
        # project back to unitary (remove numeric drift)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A)
    # Factor K1, K2 → (2x2 ⊗ 2x2)
    def factor_local(K):
        # reshape to (2,2,2,2), SVD the (a,c ; b,d) unfolding
        M = K.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
        U,_,Vh = np.linalg.svd(M, full_matrices=False)
        A = U[:,0].reshape(2,2); B = Vh.conj().T[:,0].reshape(2,2)
        return N_Qubit_Decomposition_Guided_Tree._polar_unitary(A), N_Qubit_Decomposition_Guided_Tree._polar_unitary(B)
    def _magic_basis_plusYY():
        # Complex magic basis (matches A(c)=exp(-i/2*(c1 XX + c2 YY + c3 ZZ)) below)
        # Columns are (|Φ+>, i|Φ->, i|Ψ+>, |Ψ->) up to harmless phases
        return (1/np.sqrt(2))*np.array([
            [1, 0, 0,  1j],
            [0, 1j,1,  0 ],
            [0, 1j,-1, 0 ],
            [1j,0, 0, -1 ]
        ], dtype=complex)

    def _project_to_SO4(O):
        # nearest real orthogonal with det=+1
        O = np.real_if_close(O, tol=1e5)
        U, _, Vt = np.linalg.svd(O)
        O = U @ Vt
        if np.linalg.det(O) < 0:
            O[:,0] *= -1
        return O

    def _clean_col_phases(W):
        Wc = W.copy()
        for j in range(Wc.shape[1]):
            col = Wc[:, j]
            k = np.argmax(np.abs(col))
            if np.abs(col[k]) > 1e-14:
                Wc[:, j] *= np.exp(-1j * np.angle(col[k]))
        return Wc

    def closest_local_product(W4):
        A, B = N_Qubit_Decomposition_Guided_Tree.factor_local(W4)
        return N_Qubit_Decomposition_Guided_Tree._global_phase_fix(A), N_Qubit_Decomposition_Guided_Tree._global_phase_fix(B)
    def kak_u3s_around_cx(U, n, c, t, iters=3):
        U4 = N_Qubit_Decomposition_Guided_Tree.trace_out_qubits(U, n, (c, t))
        U4 = N_Qubit_Decomposition_Guided_Tree._global_phase_fix(U4)
        from qiskit.synthesis import TwoQubitWeylDecomposition
        twd = TwoQubitWeylDecomposition(U4)
        c1, c2, c3 = twd.a, twd.b, twd.c
        K1A, K1B, K2A, K2B = twd.K1l, twd.K1r, twd.K2l, twd.K2r
        A = N_Qubit_Decomposition_Guided_Tree._A_from_c(c1,c2,c3)
        U_rec = np.kron(K1A,K1B) @ A @ np.kron(K2A,K2B)
        z = np.trace(U_rec.conj().T @ U4)
        U_rec *= np.exp(1j * np.angle(z))
        print("Frob err:", np.linalg.norm(U_rec - U4), c1, c2, c3)
        thA_pre, phA_pre, laA_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K2A.conj().T)
        thB_pre, phB_pre, laB_pre = N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K2B.conj().T)
        thA_post,phA_post,laA_post= N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K1A.conj().T)  # left-apply ⇒ take dagger on outputs
        thB_post,phB_post,laB_post= N_Qubit_Decomposition_Guided_Tree.su2_to_u3_zyz(K1B.conj().T)
        return {
            "c": (c1,c2,c3),
            "pre":  { "A": (thA_pre/2, phA_pre, laA_pre),
                    "B": (thB_pre/2, phB_pre, laB_pre) },
            "post": { "A": (thA_post/2, phA_post, laA_post),
                    "B": (thB_post/2, phB_post, laB_post) }
        }
    def params_to_mat(self, params):
        allU = []
        for U, pspace in zip(self.Umtx, [None] if self.paramspace is None else self.paramspace):
            U = U.copy()
            scaled_params = np.sum(params.reshape(-1, 1+len(pspace)) * np.array((1.0,) + pspace), axis=1) if pspace is not None else params
            self.get_Circuit().apply_to(scaled_params if pspace is not None else params, U)
            allU.append(U)
        return allU
    def OSR_with_local_alignment(self, pairs, cuts, Fnorm, tol, rank, use_softmax, method="dual_annealing"):
        if len(pairs) != 0:
            self.set_Cost_Function_Variant( 10 )
            #self.Run_Decomposition(pairs, False)
            self.set_Gate_Structure(self.get_circuit_from_pairs(pairs, False))
            import scipy
            param_bound = np.array(([2*np.pi] + [1/x for x in self.paramscale])*self.get_Parameter_Num())
            def cost(x):
                allU = self.params_to_mat(x)
                S = [N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol)[1] for U in allU for cut in cuts]
                if use_softmax: return N_Qubit_Decomposition_Guided_Tree.cuts_softmax_rank_cost(S, rank)
                else: return N_Qubit_Decomposition_Guided_Tree.avg_loss(S, rank)
            def jacobian(x):
                allU = self.params_to_mat(x)
                grad = np.zeros(len(x), dtype=float)
                for Ubase, U, pspace in zip(self.Umtx, allU, [None] if self.paramspace is None else self.paramspace):
                    dL = N_Qubit_Decomposition_Guided_Tree.get_deriv_osr_entanglement(U, cuts, rank, use_softmax)
                    basevec = np.array((1.0,) if pspace is None else (1.0,) + pspace)
                    scaled_params = np.sum(x.reshape(-1, 1+len(pspace)) * basevec, axis=1) if pspace is not None else x                    
                    derivs = N_Qubit_Decomposition_Guided_Tree.param_derivs(self.get_Circuit(), Ubase, scaled_params)
                    newgrad = np.array([N_Qubit_Decomposition_Guided_Tree.real_trace_conj_dot(dL, deriv) for deriv in derivs])
                    if pspace is not None: newgrad = (np.array(newgrad)[:,np.newaxis] * basevec).reshape(-1)
                    grad += newgrad
                return grad / len(self.Umtx)
            if method == "differential_evolution":
                best = scipy.optimize.differential_evolution(cost, [ (0, x) for x in param_bound ], maxiter=100, polish=False)
                best = scipy.optimize.minimize(cost, best.x, method='BFGS', jac=jacobian, options={'maxiter': 200})
            elif method == "dual_annealing":
                best = None
                for seed in range(20):
                    res = scipy.optimize.dual_annealing(cost, [ (0, x) for x in param_bound ], maxiter=100)#, minimizer_kwargs={'jac': jacobian})
                    if best is None or res.fun < best.fun: best = res
            elif method == "basinhopping":
                best = scipy.optimize.basinhopping(cost, np.random.rand(len(param_bound))*param_bound, niter=50, stepsize=np.pi/2, minimizer_kwargs={'jac': jacobian})
            else:
                best = min([scipy.optimize.minimize(cost, np.random.rand(len(param_bound))*param_bound, method='BFGS', jac=jacobian, options={'maxiter': 200}) for _ in range(20)], key=lambda r: r.fun)
            #print(best)
            self.set_Cost_Function_Variant( 3 )
            allU = self.params_to_mat(best.x)
        else: allU = self.Umtx
        return [(N_Qubit_Decomposition_Guided_Tree.ceil_log2(rank), s) for U in allU for cut in cuts for rank, s in (N_Qubit_Decomposition_Guided_Tree.operator_schmidt_rank(U, self.qbit_num, cut, Fnorm, tol),)]
    def Run_Decomposition(self, pairs, finalizing=True):
        circ = self.get_circuit_from_pairs(pairs, finalizing)
        self.set_Gate_Structure(circ)
        self.set_Optimized_Parameters(np.random.rand(self.get_Parameter_Num())*(2*np.pi))
        super().Start_Decomposition()
        if finalizing:
            params = self.get_Optimized_Parameters()
            self.err = self.Optimization_Problem(params)
            return self.err < self.config.get('tolerance', 1e-8)
    def generate_insertions(curpath, topology, num_cnot):
        import itertools
        n = len(curpath)
        nslots = n + 1
        for places in itertools.combinations_with_replacement(range(nslots), num_cnot):
            for pairs in itertools.product(topology, repeat=num_cnot):
                out = []
                j = 0  # index into inserted pairs
                for slot in range(nslots):
                    while j < num_cnot and places[j] == slot:
                        out.append(pairs[j])
                        j += 1
                    if slot < n:
                        out.append(curpath[slot])
                yield tuple(out)
    def Start_Decomposition(self):
        import heapq, itertools
        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        #because we have U already conjugate transposed, must use prefix order
        B = self.config.get('beam', None)#8*len(self.topology))
        max_depth = self.config.get('tree_level_max', 14)
        tol = 1e-3
        Fnorm = np.sqrt(1<<self.qbit_num)
        best = []
        visited = set()
        all_ranks = list(range(min(2, self.qbit_num-1)))
        def get_osr_stats(path, rank, use_softmax):
            h = self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol, rank=rank, use_softmax=use_softmax, method="basin_hopping")
            min_cnots = max((x[0] for x in h), default=0)
            ranktot = sum(x[0] for x in h)
            kappa = sum(sum(y*y for y in x[1][1:]) for x in h)
            return min_cnots, ranktot+kappa, h
        def add_to_heap(path, parent_stats):
            if len(path) > max_depth: return False
            if path in visited: return False
            visited.add(path)
            if self.qbit_num > 1:
                min_cnots, rankkappa = min(get_osr_stats(path, rank, use_sm)[:2] for (rank, use_sm) in itertools.product(all_ranks, (False,))) #(False, True)
            else: min_cnots, rankkappa = 0, 0.0
            if parent_stats is not None and (min_cnots, rankkappa) >= parent_stats: return False
            heapq.heappush(best, (min_cnots, rankkappa, path))
            return True
        add_to_heap((), None)
        while best:
            #print(best[0])
            min_cnots, rankkappa, curpath = heapq.heappop(best)
            if min_cnots == 0:
                #print(path)
                for i in range(10):
                    if self.Run_Decomposition(curpath):
                        self.all_solutions.append((self.get_Circuit(), self.get_Optimized_Parameters()))
                        if stop_first_solution: return
                        break
                    #print("Looping", h)
            num_cnot = 1
            while True:
                any_added = False
                for newpath in N_Qubit_Decomposition_Guided_Tree.generate_insertions(curpath, self.topology, num_cnot):
                    if add_to_heap(newpath, (min_cnots, rankkappa)): any_added = True
                if any_added: break
                num_cnot += 1
                if len(curpath) + num_cnot > max_depth: break
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        #print("No decomposition found within the given CNOT limit.")
    """
    def Start_Decomposition(self):
        self.all_solutions = []
        self.err = 1.0
        stop_first_solution = self.config.get("stop_first_solution", True)
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(self.qbit_num))
        if self.topology is None:
            self.topology = [(i, j) for i in range(self.qbit_num) for j in range(i+1, self.qbit_num)]
        pair_affects = {
            pair: [i for i,A in enumerate(cuts) if (pair[0] in A) ^ (pair[1] in A)]
            for pair in self.topology
        }
        #because we have U already conjugate transposed, must use prefix order
        B = self.config.get('beam', None)#8*len(self.topology))
        max_depth = self.config.get('tree_level_max', 14)
        tol = 1e-3
        Fnorm = np.sqrt(1<<self.qbit_num)
        prior_level_info = None
        for depth in range(max_depth+1):
            remaining = max_depth - depth
            visited, seq_pairs_of, seq_dir_of, res = N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS_level(self.qbit_num, self.topology, prior_level_info, use_gl=False)
            nextprefixes = []
            for path in set(tuple(x[1]) for x in res):
                curh = None if len(path)==0 else prefixes[path[:-1]]
                check_cuts = pair_affects[tuple(sorted(path[-1]))] if not curh is None else range(len(cuts))
                #samples = [max(x[0] for x in self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol)) for _ in range(5)]
                #if len(set(samples)) != 1: print(samples)
                h = self.OSR_with_local_alignment(path, cuts, Fnorm, tol=tol, use_softmax=False, method="dual_annealing")
                min_cnots = max((x[0] for x in h), default=0)
                print(path, h, N_Qubit_Decomposition_Guided_Tree.avg_loss([x[1] for x in h]), remaining, min_cnots)
                if min_cnots == 0:
                    #print(path)
                    for i in range(10):
                        if self.Run_Decomposition(path):
                            self.all_solutions.append((self.get_Circuit(), self.get_Optimized_Parameters()))
                            if stop_first_solution: return
                            break
                        #print("Looping", h)
                if min_cnots > remaining: continue
                if not curh is None:
                    #print(path, [(h[i], curh[i]) for i in check_cuts])
                    #if any(h[i][0] > curh[i][0] for i in check_cuts): continue
                    if max((x[0] for x in curh), default=0) < min_cnots: continue
                nextprefixes.append((path, h))
            nextprefixes.sort(key=lambda t: (max((x[0] for x in t[1]), default=0), sum(x[0] for x in t[1]), N_Qubit_Decomposition_Guided_Tree.avg_loss([x[1] for x in t[1]])))
            prefixes = {x[0]: x[1] for x in nextprefixes[:B]}
            prior_level_info = (visited, seq_pairs_of, seq_dir_of, list(x[0] for x in reversed(res) if tuple(x[1]) in prefixes))
        self.set_Gate_Structure(Circuit(self.qbit_num))
        self.set_Optimized_Parameters(np.array([]))
        #print("No decomposition found within the given CNOT limit.")
    """
    def get_Decomposition_Error(self): return self.err
    def compositions(total, parts):
        """
        All nonnegative integer tuples of length `parts` summing to `total`.
        """
        if parts == 1:
            yield (total,)
            return
        for x in range(total + 1):
            for rest in N_Qubit_Decomposition_Guided_Tree.compositions(total - x, parts - 1):
                yield (x,) + rest
    def solve_best_min_cnots(num_qubits, cuts, rank_kappa, topology, use_surplus=True):
        m = len(topology)
        cut_to_edges = [[i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)] for cut in cuts]
        total = 0
        best_kappa = None
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(sum(edge_counts[j] for j in cut_to_edge)>=cut_bound[0] for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa)):
                    new_kappa = 0.0
                    for cut_to_edge, cut_bound in zip(cut_to_edges, rank_kappa):
                        coverage = sum(edge_counts[j] for j in cut_to_edge)
                        if use_surplus:
                            new_kappa += cut_bound[1] * (coverage - cut_bound[0])
                        else: new_kappa += cut_bound[1] * coverage
                    best_kappa = new_kappa if best_kappa is None else max(best_kappa, new_kappa)
            if best_kappa is not None: break
            total += 1
        return total, best_kappa
    def solve_min_cnots(num_qubits, cuts, cut_bounds, topology):
        m = len(topology)
        cut_to_edges = [[i for i, z in enumerate(topology) if (z[0] in cut) != (z[1] in cut)] for cut in cuts]
        total = 0
        while True:
            for edge_counts in N_Qubit_Decomposition_Guided_Tree.compositions(total, m):
                if all(sum(edge_counts[j] for j in cut_to_edge)>=cut_bound for cut_to_edge, cut_bound in zip(cut_to_edges, cut_bounds)):
                    return total
            total += 1
    def gen_all_min_cnots(num_qbits, topology=None): #OSR tells min CNOTs at most for 3 qubits 3, 4 qubits 6, 5 qubits 7
        import itertools
        cuts = list(N_Qubit_Decomposition_Guided_Tree.unique_cuts(num_qbits))
        min_cnot_bounds = [2*min(cut_size, num_qbits - cut_size) for cut_size in (len(cut) for cut in cuts)]
        if topology is None:
            topology = [(i, j) for i in range(num_qbits) for j in range(i+1, num_qbits)]
        for cnot_bounds in itertools.product(*(range(bound+1) for bound in min_cnot_bounds)):
            #if tuple(sorted(cnot_bounds)) != cnot_bounds: continue
            print(cnot_bounds, N_Qubit_Decomposition_Guided_Tree.solve_min_cnots(num_qbits, cuts, cnot_bounds, topology))
#N_Qubit_Decomposition_Guided_Tree.gen_all_min_cnots(3); assert False
#N_Qubit_Decomposition_Guided_Tree.build_sequence(); assert False
#print(len(list(N_Qubit_Decomposition_Guided_Tree.enumerate_unordered_cnot_BFS(3, [(0,1),(1,2),])))); assert False

import numpy as np
from squander import N_Qubit_Decomposition_custom, Circuit
import random

# Number of qubits
N = 3

# Generate a random qubit permutation (permutation of qubit indices)
qubit_permutation = list(range(N))
random.shuffle(qubit_permutation)

print(f"Target qubit permutation for {N} qubits: {qubit_permutation}")

# Create the full permutation matrix from qubit permutation
# For qubit permutation [q0, q1, q2, ...], basis state |i> maps to |permuted_i>
# where permuted_i is obtained by permuting the bits of i according to qubit_permutation
matrix_size = 2**N
target_matrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)

for i in range(matrix_size):
    # Convert i to binary representation
    bits = [(i >> k) & 1 for k in range(N)]

    # Apply qubit permutation
    permuted_bits = [bits[qubit_permutation[k]] for k in range(N)]

    # Convert back to integer
    j = sum(bit << k for k, bit in enumerate(permuted_bits))

    target_matrix[i, j] = 1.0

print("\nTarget permutation matrix:")
print(target_matrix)

# Build circuit with single Permutation_NU gate
print("Building circuit with single Permutation_NU gate...")
ansatz = Circuit(N)
ansatz.add_Permutation_NU()

# Create decomposition object
cDecompose = N_Qubit_Decomposition_custom(
    target_matrix.conj().T,
)
cDecompose.set_Gate_Structure(ansatz)

# Set optimizer
cDecompose.set_Optimizer("BFGS")

# Run decomposition
print("Starting decomposition...")
cDecompose.Start_Decomposition()

# Get final cost
final_params = cDecompose.get_Optimized_Parameters()
final_cost = cDecompose.Optimization_Problem(final_params)

print(f"\nFinal cost: {final_cost:.6e}")
print(f"Success: {final_cost < 1e-8}")
print(f"Learned parameter: {final_params}")

# Debug: Check what matrix we got
import math
print(f"\nNumber of possible permutations for {N} qubits: {math.factorial(N)}")
print(f"Centers at: {[i * 2 * np.pi / math.factorial(N) for i in range(math.factorial(N))]}")
print(f"Parameter {final_params[0]:.6f} selects center 0 (identity) if near 0.0 or {2*np.pi:.6f}")

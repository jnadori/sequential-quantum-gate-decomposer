import numpy as np
import squander
from squander import N_Qubit_Decomposition_custom, Circuit
from scipy.optimize import dual_annealing

np.set_printoptions(linewidth=200, precision=6, suppress=True)

def generate_ansatz_CNZ(levels, N):
    """Generate ansatz using regular CNZ gates (fixed, no parameters)"""
    ansatz = Circuit(N)
    for i in range(levels):
        for qubit in range(N):
            ansatz.add_U3(qubit)
        # Add one CNZ gate per layer, cycling through phase indices
        ansatz.add_CNZ()
    for qubit in range(N):
        ansatz.add_U3(qubit)
    return ansatz

def generate_ansatz_CNZ_NU(levels, N):
    """Generate ansatz using CNZ_NU gates (1 learnable parameter)"""
    ansatz = Circuit(N)
    for i in range(levels):
        for qubit in range(N):
            ansatz.add_U3(qubit)
        ansatz.add_CNZ_NU()
    for qubit in range(N):
        ansatz.add_U3(qubit)
    return ansatz

def F(N):
    """Generate N-qubit QFT matrix"""
    Fn = np.zeros([2**N, 2**N], dtype=np.complex128)
    for idx in range(2**N):
        for jdx in range(2**N):
            Fn[idx][jdx] = 1/np.sqrt(2**N) * (np.exp(2j*np.pi/(2**N))**(idx*jdx))
    return Fn

def test_decomposition(N, levels, gate_type, num_samples=10):
    """
    Test decomposition multiple times and return statistics
    Returns: (mean_cost, min_cost, max_cost, success_rate)
    """
    # Generate target unitary (QFT)
    Umtx = F(N)

    costs = []

    for sample in range(num_samples):
        # Create decomposition object
        cDecompose = N_Qubit_Decomposition_custom(Umtx.conj().T)

        # Set gate structure
        if gate_type == 'CNZ':
            ansatz = generate_ansatz_CNZ(levels, N)
        elif gate_type == 'CNZ_NU':
            ansatz = generate_ansatz_CNZ_NU(levels, N)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        cDecompose.set_Gate_Structure(ansatz)
        cDecompose.set_Verbose(0)
        cDecompose.set_Cost_Function_Variant(3)

        parameter_num = cDecompose.get_Parameter_Num()

        try:
            # Use BFGS with random initialization
            cDecompose.set_Optimizer("BFGS")
            parameters = np.random.rand(parameter_num) * np.pi * 2
            cDecompose.set_Optimized_Parameters(parameters)
            cDecompose.Start_Decomposition()

            final_params = cDecompose.get_Optimized_Parameters()
            final_cost = cDecompose.Optimization_Problem(final_params)

            costs.append(final_cost)

        except Exception as e:
            pass

    if len(costs) == 0:
        return None, None, None, 0.0

    mean_cost = np.mean(costs)
    min_cost = np.min(costs)
    max_cost = np.max(costs)
    success_rate = len(costs) / num_samples

    return mean_cost, min_cost, max_cost, success_rate

def find_minimum_layers(N, gate_type, target_cost=1e-8, max_layers=50, step=5, num_samples=10):
    """
    Find minimum layers needed to reach target cost
    """
    print(f"\n  Searching for {gate_type} with {num_samples} samples per layer count...", flush=True)

    for layers in range(step, max_layers + 1, step):
        print(f"    Testing {layers:2d} layers... ", end='', flush=True)

        mean_cost, min_cost, max_cost, success_rate = test_decomposition(N, layers, gate_type, num_samples=num_samples)

        if min_cost is not None:
            print(f"min={min_cost:.2e} mean={mean_cost:.2e} max={max_cost:.2e}", end='')
            if min_cost < target_cost:
                print(f"  ✓ SUCCESS!")
                return layers, min_cost, mean_cost, max_cost
            else:
                print()
        else:
            print("FAILED")

    print(f"    Could not reach target with {max_layers} layers")
    return None, None, None, None

def main():
    """Run comparison"""

    print("="*70)
    print("CNZ vs CNZ_NU: Minimum Layers for High Accuracy")
    print("Target: N-qubit Quantum Fourier Transform")
    print(f"Success criterion: min cost < 1e-8 (over 10 samples)")
    print("="*70)

    results = []
    target_cost = 1e-8
    num_samples = 100

    # Test 3 qubits
    print("\n" + "="*70)
    print("3-QUBIT QFT")
    print("="*70)

    cnz_result_3 = find_minimum_layers(3, 'CNZ', target_cost, max_layers=50, step=1, num_samples=num_samples)
    cnz_nu_result_3 = find_minimum_layers(3, 'CNZ_NU', target_cost, max_layers=50, step=1, num_samples=num_samples)

    results.append({
        'N': 3,
        'gate_type': 'CNZ',
        'layers': cnz_result_3[0],
        'min_cost': cnz_result_3[1],
        'mean_cost': cnz_result_3[2],
        'max_cost': cnz_result_3[3]
    })
    results.append({
        'N': 3,
        'gate_type': 'CNZ_NU',
        'layers': cnz_nu_result_3[0],
        'min_cost': cnz_nu_result_3[1],
        'mean_cost': cnz_nu_result_3[2],
        'max_cost': cnz_nu_result_3[3]
    })

    # Test 4 qubits
    print("\n" + "="*70)
    print("4-QUBIT QFT")
    print("="*70)

    cnz_result_4 = find_minimum_layers(4, 'CNZ', target_cost, max_layers=50, step=1, num_samples=int(num_samples/5))
    cnz_nu_result_4 = find_minimum_layers(4, 'CNZ_NU', target_cost, max_layers=50, step=1, num_samples=int(num_samples/5))

    results.append({
        'N': 4,
        'gate_type': 'CNZ',
        'layers': cnz_result_4[0],
        'min_cost': cnz_result_4[1],
        'mean_cost': cnz_result_4[2],
        'max_cost': cnz_result_4[3]
    })
    results.append({
        'N': 4,
        'gate_type': 'CNZ_NU',
        'layers': cnz_nu_result_4[0],
        'min_cost': cnz_nu_result_4[1],
        'mean_cost': cnz_nu_result_4[2],
        'max_cost': cnz_nu_result_4[3]
    })

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Minimum Layers to Reach min cost < 1e-8")
    print("="*70)
    print(f"{'Qubits':>7} {'Gate':>10} {'Layers':>8} {'Min Cost':>12} {'Mean Cost':>12} {'Max Cost':>12}")
    print("-"*70)

    for r in results:
        layers_str = str(r['layers']) if r['layers'] else "NOT FOUND"
        min_str = f"{r['min_cost']:.2e}" if r['min_cost'] else "N/A"
        mean_str = f"{r['mean_cost']:.2e}" if r['mean_cost'] else "N/A"
        max_str = f"{r['max_cost']:.2e}" if r['max_cost'] else "N/A"
        print(f"{r['N']:>7} {r['gate_type']:>10} {layers_str:>8} {min_str:>12} {mean_str:>12} {max_str:>12}")

    print("="*70)

if __name__ == "__main__":
    main()

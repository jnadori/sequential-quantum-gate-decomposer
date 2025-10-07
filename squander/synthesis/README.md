# Tree Search Quantum Circuit Synthesis (Python Implementation)

This module provides a **pure Python implementation** of tree search based quantum circuit synthesis for better debugging and custom optimizer integration.

## Overview

The tree search algorithm explores different quantum gate configurations by:
1. **Enumerating gate structures** using n-ary Gray codes
2. **Optimizing parameters** for each structure using scipy or custom optimizers
3. **Finding the minimal depth** circuit that achieves target fidelity

## Module Structure

```
squander/synthesis/
├── gray_code.py           # N-ary Gray code generator
├── circuit_builder.py     # Circuit construction from Gray codes
├── optimizer_wrapper.py   # Unified optimizer interface
├── tree_search.py         # Main tree search algorithm
└── README.md             # This file
```

## Components

### 1. Gray Code Generator (`gray_code.py`)

N-ary reflected Gray code iteration for enumerating gate configurations.

```python
from squander.synthesis.gray_code import NaryGrayCodeCounter

# Create counter for 2 positions with limits [3, 4]
counter = NaryGrayCodeCounter([3, 4])

# Iterate over all Gray codes
for gray_code in counter:
    print(gray_code)  # [0,0], [1,0], [1,1], ...
```

**Key Features:**
- Efficient Gray code generation
- Support for different limits per position
- Range iteration for parallelization

### 2. Circuit Builder (`circuit_builder.py`)

Constructs quantum circuits from Gray codes respecting topology constraints.

```python
from squander.synthesis.circuit_builder import (
    build_topology,
    construct_circuit_from_gray_code
)

# Build all-to-all topology
topology = build_topology(qbit_num=3)  # [(0,1), (1,0), (0,2), ...]

# Construct circuit from Gray code
circuit = construct_circuit_from_gray_code(
    gray_code=[0, 1, 2],
    qbit_num=3,
    topology=topology,
    use_u3=True
)
```

**Circuit Structure:**
- For each Gray code element:
  - U3 gates on target and control qubits
  - CNOT(target, control)
- Final layer of U3 gates on all qubits

### 3. Optimizer Wrapper (`optimizer_wrapper.py`)

Unified interface for different optimization methods.

```python
from squander.synthesis.optimizer_wrapper import optimize_circuit

# Optimize with scipy
result = optimize_circuit(
    Umtx=target_unitary,
    circuit=circuit,
    optimizer='scipy:L-BFGS-B',
    config={'max_iterations': 1000}
)

# Optimize with built-in SQUANDER
result = optimize_circuit(
    Umtx=target_unitary,
    circuit=circuit,
    optimizer='BFGS'
)

# Optimize with custom function
def my_optimizer(cost_fn, grad_fn, x0, **kwargs):
    # Custom optimization logic
    return x_opt, cost, info

result = optimize_circuit(
    Umtx=target_unitary,
    circuit=circuit,
    optimizer='custom',
    custom_optimizer=my_optimizer
)
```

**Supported Optimizers:**
- **scipy:** L-BFGS-B, BFGS, CG, Powell, Nelder-Mead, etc.
- **SQUANDER built-in:** BFGS, ADAM, AGENTS
- **Custom:** User-defined optimization functions

### 4. Tree Search (`tree_search.py`)

Main tree search decomposition algorithm.

```python
from squander.synthesis.tree_search import TreeSearchDecomposition

# Create decomposer
decomposer = TreeSearchDecomposition(
    Umtx=target_unitary.conj().T,
    topology=None,  # All-to-all connectivity
    config={
        'tree_level_max': 10,
        'optimization_tolerance': 1e-8,
        'optimizer': 'scipy:L-BFGS-B',
        'cost_function_variant': 3,
    },
    verbose=1
)

# Run decomposition
result = decomposer.start_decomposition()

# Get results
circuit = decomposer.get_circuit()
parameters = decomposer.get_optimized_parameters()
qiskit_circuit = decomposer.get_qiskit_circuit()
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tree_level_max` | Maximum circuit depth (CNOT layers) | 10 |
| `tree_level_min` | Minimum circuit depth to search | 0 |
| `optimization_tolerance` | Target cost function value | 1e-8 |
| `optimizer` | Optimizer specification | 'scipy:L-BFGS-B' |
| `max_iterations` | Max optimization iterations | 1000 |
| `cost_function_variant` | Cost function type (0-5) | 3 |
| `num_trials_per_level` | Random restarts per config | 1 |

**Cost Function Variants:**
- 0: FROBENIUS_NORM
- 1: FROBENIUS_NORM_CORRECTION1
- 2: FROBENIUS_NORM_CORRECTION2
- 3: HILBERT_SCHMIDT_TEST
- 4: HILBERT_SCHMIDT_TEST_CORRECTION1
- 5: HILBERT_SCHMIDT_TEST_CORRECTION2

## Circuit Visualization

The module provides Qiskit-based circuit visualization:

```python
from squander.synthesis.circuit_builder import visualize_circuit_structure

# Visualize as text (default)
viz = visualize_circuit_structure(gray_code, topology, qbit_num, output='text')
print(viz)

# Visualize with matplotlib
fig = visualize_circuit_structure(gray_code, topology, qbit_num, output='mpl')
fig.savefig('circuit.png')

# Export LaTeX source
latex = visualize_circuit_structure(gray_code, topology, qbit_num, output='latex')
```

When running tree search with `verbose >= 2`, the best circuit structure is automatically displayed:

```python
decomposer = TreeSearchDecomposition(Umtx.conj().T, config=config, verbose=2)
result = decomposer.start_decomposition()
# Outputs Qiskit circuit diagram at the end
```

## Usage Examples

See [examples/decomposition/example_tree_search_python.py](../../examples/decomposition/example_tree_search_python.py) for complete examples.

### Basic Example

```python
import numpy as np
from squander.synthesis.tree_search import TreeSearchDecomposition

# Generate 3-qubit QFT
def qft(N):
    dim = 2**N
    return np.array([[np.exp(2j*np.pi*i*j/dim)/np.sqrt(dim)
                      for j in range(dim)] for i in range(dim)])

Umtx = qft(3)

# Configure and run
decomposer = TreeSearchDecomposition(
    Umtx.conj().T,
    config={'tree_level_max': 5, 'optimization_tolerance': 1e-8},
    verbose=2
)
result = decomposer.start_decomposition()

print(f"Success: {result['success']}")
print(f"Cost: {result['cost']:.2e}")
print(f"Depth: {result['level']} CNOT layers")
```

### With Topology Constraints

```python
# Define linear topology (nearest-neighbor)
topology = [(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)]

decomposer = TreeSearchDecomposition(
    Umtx.conj().T,
    topology=topology,  # Limited connectivity
    config={'tree_level_max': 10},
    verbose=1
)
```

### With Custom Optimizer

```python
from scipy.optimize import differential_evolution

def custom_de_optimizer(cost_fn, grad_fn, x0, **kwargs):
    """Differential evolution optimizer."""
    bounds = [(0, 2*np.pi)] * len(x0)
    result = differential_evolution(cost_fn, bounds, seed=42)
    return result.x, result.fun, {'success': result.success}

result = optimize_circuit(
    Umtx,
    circuit,
    optimizer='custom',
    custom_optimizer=custom_de_optimizer
)
```

## Performance Comparison

| Feature | C++ Implementation | Python Implementation |
|---------|-------------------|----------------------|
| **Speed** | Fast (production) | Slower (debugging) |
| **Parallelization** | TBB | Sequential/multiprocessing |
| **Debugging** | Difficult | Easy |
| **Optimizer Flexibility** | Fixed | High (scipy + custom) |
| **Experimentation** | Requires rebuild | Immediate |

## Advantages of Python Version

1. **Debugging:**
   - Print statements, breakpoints, visualization
   - Inspect intermediate results
   - Step-by-step execution

2. **Custom Optimizers:**
   - Easy integration with scipy.optimize
   - Evolutionary algorithms (DEAP, PyGMO)
   - Meta-heuristics (simulated annealing, particle swarm)
   - ML-based optimizers (PyTorch, JAX)

3. **Rapid Prototyping:**
   - Test new search strategies
   - Experiment with cost functions
   - Modify circuit structures

4. **Integration:**
   - Combine with Qiskit, Cirq
   - Use with ML frameworks
   - Export to various formats

## Testing

Run the test suite:

```bash
conda activate qgd
python examples/decomposition/example_tree_search_python.py
```

## Reference

C++ implementation: [squander/src-cpp/decomposition/N_Qubit_Decomposition_Tree_Search.cpp](../src-cpp/decomposition/N_Qubit_Decomposition_Tree_Search.cpp)

## Future Improvements

- [ ] Parallel evaluation of gate configurations (multiprocessing)
- [ ] Caching of optimization results
- [ ] Adaptive search (skip unpromising branches)
- [ ] Integration with quantum machine learning
- [ ] Support for custom gate sets (beyond U3+CNOT)

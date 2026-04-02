"""Compile QASM circuits from circs/ using BQSKit QSearch or LEAP.

Uses:
- QSearch if qubit count <= 3
- LEAP otherwise

Keeps layout trivial by not applying SABRE layout/routing passes.
Uses SetModelPass to set a star-topology MachineModel with {cx, u3}.
Prints gate counts instead of saving compiled QASM files.
"""

from __future__ import annotations

import csv
import glob
import os
import time
from collections import Counter

from bqskit.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.crot import CROTGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.passes import (
    LEAPSynthesisPass,
    QSearchSynthesisPass,
    SetModelPass,
    UnfoldPass,
)
from bqskit.qis.graph import CouplingGraph

CIRCS_DIR = "circs_qasm2"
RESULTS_CSV = "compilation_results_bqskit.csv"
STAR_CENTER = 0


def choose_synthesis_method(num_qudits: int) -> str:
    return "qsearch"


def make_star_coupling_graph(num_qudits: int, center: int = 0) -> CouplingGraph:
    edges = [(center, q) for q in range(num_qudits) if q != center]
    return CouplingGraph(edges)


def make_star_machine_model(num_qudits: int, center: int = 0) -> MachineModel:
    coupling_graph = make_star_coupling_graph(num_qudits, center=center)
    gate_set = {U3Gate(), CROTGate()}
    return MachineModel(
        num_qudits=num_qudits,
        gate_set=gate_set,
        coupling_graph=coupling_graph,
    )


def load_qasm_circuit(filename: str) -> Circuit:
    with open(filename, "r") as f:
        qasm_text = f.read()
    return OPENQASM2Language().decode(qasm_text)


def count_two_qubit_gates(circuit: Circuit) -> int:
    total = 0
    for op in circuit:
        if len(op.location) == 2:
            total += 1
    return total


def count_total_gates(circuit: Circuit) -> int:
    total = 0
    for _ in circuit:
        total += 1
    return total


def get_gate_counts(circuit: Circuit) -> Counter[str]:
    counts: Counter[str] = Counter()
    for op in circuit:
        gate_name = op.gate.__class__.__name__
        counts[gate_name] += 1
    return counts


def format_gate_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{gate}:{counts[gate]}" for gate in sorted(counts))


def build_workflow(num_qudits: int, model: MachineModel):
    method = choose_synthesis_method(num_qudits)

    if method == "qsearch":
        synth_pass = QSearchSynthesisPass()
    else:
        synth_pass = LEAPSynthesisPass()

    return method, [
        SetModelPass(model),
        synth_pass,
        UnfoldPass(),
    ]


def main() -> None:
    qasm_files = sorted(glob.glob(os.path.join(CIRCS_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No QASM files found in {CIRCS_DIR}/")
        return

    write_header = not os.path.exists(RESULTS_CSV)
    if write_header:
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "circuit",
                    "qubits",
                    "synth_method",
                    "two_qbit_original",
                    "two_qbit_compiled",
                    "gate_num_compiled",
                    "gate_counts",
                    "compilation_time_s",
                ]
            )

    compiler = Compiler()

    for qasm_file in qasm_files:
        name = os.path.basename(qasm_file)
        print(f"\n{'=' * 60}")
        print(f"Circuit: {name}")
        print(f"{'=' * 60}")

        input_circuit = load_qasm_circuit(qasm_file)
        num_qudits = input_circuit.num_qudits
        two_qbit_original = count_two_qubit_gates(input_circuit)

        print(
            f"  Qubits: {num_qudits}, "
            f"Original 2-qubit gates: {two_qbit_original}"
        )

        model = make_star_machine_model(num_qudits, center=STAR_CENTER)
        synth_method, workflow = build_workflow(num_qudits, model)

        t0 = time.time()
        compiled_circuit = compiler.compile(input_circuit, workflow)
        elapsed = time.time() - t0

        two_qbit_compiled = count_two_qubit_gates(compiled_circuit)
        gate_num_compiled = count_total_gates(compiled_circuit)
        gate_counts = get_gate_counts(compiled_circuit)
        gate_counts_str = format_gate_counts(gate_counts)

        print(f"  Method: {synth_method}")
        print(f"  Compiled 2-qubit gates: {two_qbit_compiled}")
        print(f"  Total gate count: {gate_num_compiled}")
        print(f"  Gate counts: {gate_counts_str}")
        print(f"  Time: {elapsed:.1f}s")

        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    name,
                    num_qudits,
                    synth_method,
                    two_qbit_original,
                    two_qbit_compiled,
                    gate_num_compiled,
                    gate_counts_str,
                    f"{elapsed:.2f}",
                ]
            )

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
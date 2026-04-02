#!/usr/bin/env python3

from pathlib import Path

from qiskit import qasm2
from qiskit.qasm3 import loads

INPUT_DIR = Path("compiled_circs")
OUTPUT_DIR = Path("circs_qasm2")
EXTENSIONS = {".qasm", ".qasm3", ".txt"}


def convert_file(input_file: Path, output_file: Path) -> None:
    qasm3_text = input_file.read_text(encoding="utf-8")
    circuit = loads(qasm3_text)
    qasm2_text = qasm2.dumps(circuit)
    output_file.write_text(qasm2_text, encoding="utf-8")


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not files:
        print(f"No input files found in {INPUT_DIR}")
        return

    converted = 0
    failed = 0

    for input_file in files:
        output_file = OUTPUT_DIR / f"{input_file.stem}.qasm"

        try:
            convert_file(input_file, output_file)
            print(f"Converted: {input_file} -> {output_file}")
            converted += 1
        except Exception as exc:
            print(f"Failed: {input_file} ({exc})")
            failed += 1

    print()
    print(f"Done. Converted: {converted}, Failed: {failed}")


if __name__ == "__main__":
    main()
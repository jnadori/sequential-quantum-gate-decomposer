"""
Scan all QASM files in circs/ and shrink qreg/creg declarations
to match the maximum qubit index actually referenced in gates.
Overwrites files in-place.
"""

import re
import glob
import os

circs_dir = "circs"

def max_qubit_index(lines):
    """Return the highest qubit index referenced in gate instructions."""
    max_idx = -1
    # Matches q[<number>] anywhere on a line
    pattern = re.compile(r'\bq\[(\d+)\]')
    for line in lines:
        stripped = line.strip()
        # Skip declarations and comments
        if stripped.startswith('qreg') or stripped.startswith('creg') or stripped.startswith('//'):
            continue
        for m in pattern.finditer(line):
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx

def resize_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find declared size
    declared = None
    for line in lines:
        m = re.match(r'\s*qreg\s+q\[(\d+)\]', line)
        if m:
            declared = int(m.group(1))
            break

    if declared is None:
        print(f"  SKIP (no qreg): {path}")
        return

    max_used = max_qubit_index(lines)
    needed = max_used + 1  # indices are 0-based

    if needed >= declared:
        print(f"  OK ({declared} qubits used): {path}")
        return

    # Rewrite qreg and creg lines
    new_lines = []
    for line in lines:
        line = re.sub(r'(qreg\s+q\[)\d+(\])', lambda m: f"{m.group(1)}{needed}{m.group(2)}", line)
        line = re.sub(r'(creg\s+\w+\[)\d+(\])', lambda m: f"{m.group(1)}{needed}{m.group(2)}", line)
        new_lines.append(line)

    with open(path, 'w') as f:
        f.writelines(new_lines)

    print(f"  RESIZED {declared} -> {needed} qubits: {os.path.basename(path)}")

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(circs_dir, "*.qasm")))
    print(f"Processing {len(files)} QASM files in '{circs_dir}/'...\n")
    for path in files:
        resize_file(path)

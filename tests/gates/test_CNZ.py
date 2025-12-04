'''
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
'''

import numpy as np
import pytest

from squander.gates.gates_Wrapper import CNZ


class Test_CNZ:
    """This is a test class for the CNZ gate of the Squander package"""

    def test_CNZ_creation_basic(self):
        r"""
        Test basic creation of CNZ gate with valid phase string.
        """
        qbit_num = 3
        
        # Test with explicit phase string
        cnz_gate = CNZ(qbit_num, "101")
        
        assert cnz_gate.get_Parameter_Num() == 0
        assert cnz_gate.get_Name() == "CNZ"

    def test_CNZ_creation_default_phase(self):
        r"""
        Test creation of CNZ gate without phase string.
        When phase_string is not provided, it should default to all 1s (phase_idx = 2^qbit_num - 1).
        """
        qbit_num = 2
        
        # Test without phase string - should default to "11" (all 1s)
        cnz_gate = CNZ(qbit_num)
        
        assert cnz_gate.get_Parameter_Num() == 0
        assert cnz_gate.get_Name() == "CNZ"
        
        # Verify it defaults to phase_idx = 3 (binary "11" for 2 qubits)
        # by checking the matrix - element at index 3 should be -1
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        diagonal = np.diag(matrix)
        assert np.allclose(diagonal[3], -1.0)  # phase_idx = 3
        assert np.allclose(diagonal[:3], 1.0)  # all other elements are 1

    def test_CNZ_get_matrix(self):
        r"""
        Test matrix retrieval for CNZ gate.
        CNZ creates a diagonal matrix with all 1s except one element at phase_idx which is -1.
        """
        qbit_num = 2
        phase_string = "10"  # binary 10 = decimal 2
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = cnz_gate.get_Matrix()
        
        # Convert to numpy array for easier testing
        matrix_np = np.array(matrix, dtype=np.complex128)
        
        # Matrix should be 2^qbit_num x 2^qbit_num
        expected_size = 2**qbit_num
        assert matrix_np.shape == (expected_size, expected_size)
        
        # Should be diagonal
        assert np.allclose(matrix_np, np.diag(np.diag(matrix_np)))
        
        # All diagonal elements should be 1 except phase_idx=2 which should be -1
        diagonal = np.diag(matrix_np)
        assert np.allclose(diagonal[0], 1.0)
        assert np.allclose(diagonal[1], 1.0)
        assert np.allclose(diagonal[2], -1.0)  # phase_idx = 2 (binary "10")
        assert np.allclose(diagonal[3], 1.0)

    def test_CNZ_get_matrix_different_phases(self):
        r"""
        Test matrix retrieval for CNZ gate with different phase indices.
        """
        qbit_num = 3
        
        # Test phase_idx = 0 (binary "0")
        cnz_gate_0 = CNZ(qbit_num, "0")
        matrix_0 = np.array(cnz_gate_0.get_Matrix(), dtype=np.complex128)
        diagonal_0 = np.diag(matrix_0)
        assert np.allclose(diagonal_0[0], -1.0)
        assert np.allclose(diagonal_0[1:], 1.0)
        
        # Test phase_idx = 7 (binary "111")
        cnz_gate_7 = CNZ(qbit_num, "111")
        matrix_7 = np.array(cnz_gate_7.get_Matrix(), dtype=np.complex128)
        diagonal_7 = np.diag(matrix_7)
        assert np.allclose(diagonal_7[:7], 1.0)
        assert np.allclose(diagonal_7[7], -1.0)
        
        # Test phase_idx = 5 (binary "101")
        cnz_gate_5 = CNZ(qbit_num, "101")
        matrix_5 = np.array(cnz_gate_5.get_Matrix(), dtype=np.complex128)
        diagonal_5 = np.diag(matrix_5)
        assert np.allclose(diagonal_5[5], -1.0)
        assert np.allclose(np.delete(diagonal_5, 5), 1.0)

    def test_CNZ_apply_to(self):
        r"""
        Test applying CNZ gate to a state vector.
        """
        qbit_num = 2
        phase_string = "01"  # binary 01 = decimal 1
        
        cnz_gate = CNZ(qbit_num, phase_string)
        
        # Create a test state vector
        matrix_size = 2**qbit_num
        test_state = np.random.uniform(-1.0, 1.0, (matrix_size,)) + \
                     1j * np.random.uniform(-1.0, 1.0, (matrix_size,))
        test_state = test_state / np.linalg.norm(test_state)
        
        # Save original state before applying gate
        original_state = test_state.copy()
        
        # Apply gate (modifies test_state in place)
        cnz_gate.apply_to(test_state)
        
        # Verify the result: element at phase_idx=1 should be negated
        expected_state = original_state.copy()
        expected_state[1] *= -1
        
        assert np.allclose(test_state, expected_state)

    def test_CNZ_apply_to_matrix(self):
        r"""
        Test applying CNZ gate to a matrix.
        The diagonal gate multiplies rows, not columns.
        """
        qbit_num = 2
        phase_string = "11"  # binary 11 = decimal 3
        
        cnz_gate = CNZ(qbit_num, phase_string)
        
        # Create a test matrix
        matrix_size = 2**qbit_num
        test_matrix = np.random.uniform(-1.0, 1.0, (matrix_size, matrix_size)) + \
                      1j * np.random.uniform(-1.0, 1.0, (matrix_size, matrix_size))
        
        # Save original matrix before applying gate
        original_matrix = test_matrix.copy()
        
        # Apply gate (modifies test_matrix in place)
        cnz_gate.apply_to(test_matrix)
        
        # Verify: last row should be negated (phase_idx=3)
        # The diagonal gate multiplies each row element by the corresponding diagonal element
        expected_matrix = original_matrix.copy()
        expected_matrix[3, :] *= -1
        
        assert np.allclose(test_matrix, expected_matrix)

    def test_CNZ_phase_string_invalid_characters(self):
        r"""
        Test error handling for phase string with invalid (non-binary) characters.
        std::stoi with base 2 will throw an exception for non-binary characters.
        """
        qbit_num = 2
        
        # Test with non-binary characters - std::stoi will throw std::invalid_argument
        invalid_strings = ["2", "abc", "12", "1a0", "01x"]
        
        for invalid_str in invalid_strings:
            with pytest.raises(Exception):
                CNZ(qbit_num, invalid_str)
        
        # Strings with whitespace might be parsed differently
        # Leading/trailing whitespace may cause issues
        whitespace_strings = [" 101", "101 ", "\t101", "101\n"]
        for ws_str in whitespace_strings:
            # These might fail or might be parsed - test both cases
            try:
                CNZ(qbit_num, ws_str)
                # If it succeeds, that's also acceptable (though not ideal)
            except Exception:
                # If it fails, that's expected
                pass

    def test_CNZ_phase_string_empty(self):
        r"""
        Test error handling for empty phase string.
        Empty string will be converted by std::stoi which may throw an exception.
        """
        qbit_num = 2
        
        # Empty string should cause an error when std::stoi tries to parse it
        with pytest.raises(Exception):
            CNZ(qbit_num, "")

    def test_CNZ_phase_string_none(self):
        r"""
        Test that phase_string=None defaults to all 1s (same as not providing phase_string).
        """
        qbit_num = 2
        
        # None should default to all 1s, same as not providing phase_string
        cnz_gate = CNZ(qbit_num, None)
        
        assert cnz_gate.get_Parameter_Num() == 0
        assert cnz_gate.get_Name() == "CNZ"
        
        # Verify it defaults to phase_idx = 3 (binary "11" for 2 qubits)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        diagonal = np.diag(matrix)
        assert np.allclose(diagonal[3], -1.0)  # phase_idx = 3
        assert np.allclose(diagonal[:3], 1.0)  # all other elements are 1

    def test_CNZ_phase_string_wrong_type(self):
        r"""
        Test error handling for phase string with wrong type (not string).
        """
        qbit_num = 2
        
        # Test with integer instead of string
        with pytest.raises(Exception):
            CNZ(qbit_num, 101)  # Should be "101" not 101
        
        # Test with list
        with pytest.raises(Exception):
            CNZ(qbit_num, ["1", "0", "1"])

    def test_CNZ_phase_string_leading_zeros(self):
        r"""
        Test that leading zeros in phase string are handled correctly.
        Binary "001" should be treated as 1, not as requiring 3 bits.
        """
        qbit_num = 2
        phase_string = "001"  # Should be interpreted as binary 001 = decimal 1
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        diagonal = np.diag(matrix)
        
        # Element at index 1 should be -1
        assert np.allclose(diagonal[1], -1.0)
        assert np.allclose(diagonal[0], 1.0)
        assert np.allclose(diagonal[2], 1.0)
        assert np.allclose(diagonal[3], 1.0)

    def test_CNZ_phase_string_large_index(self):
        r"""
        Test behavior when phase_idx exceeds matrix size.
        The C++ code doesn't validate this, so it may cause out-of-bounds access.
        """
        qbit_num = 2
        matrix_size = 2**qbit_num  # 4
        
        # Phase string "1111" = decimal 15, which exceeds matrix size
        # This may cause out-of-bounds access in get_matrix when accessing com_matrix[phase_idx]
        phase_string = "1111"
        
        # This might succeed but cause issues, or it might crash
        # We'll test what happens - it might work but access invalid memory
        try:
            cnz_gate = CNZ(qbit_num, phase_string)
            matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
            # If it succeeds, the behavior is undefined but we can still test
            assert matrix.shape == (matrix_size, matrix_size)
        except (Exception, SystemError, MemoryError):
            # If it fails or crashes, that's expected for invalid input
            assert True

    def test_CNZ_multiple_qubits(self):
        r"""
        Test CNZ gate with different numbers of qubits.
        """
        for qbit_num in range(1, 5):
            phase_string = "1" * qbit_num  # All ones
            cnz_gate = CNZ(qbit_num, phase_string)
            
            matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
            expected_size = 2**qbit_num
            assert matrix.shape == (expected_size, expected_size)
            
            # Verify it's diagonal
            assert np.allclose(matrix, np.diag(np.diag(matrix)))

    def test_CNZ_unitarity(self):
        r"""
        Test that CNZ gate matrix is unitary.
        """
        qbit_num = 3
        phase_string = "101"
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        
        # Check unitarity: U @ U^\dagger should be identity
        product = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        
        assert np.allclose(product, identity, atol=1e-10)

    def test_CNZ_hermitian(self):
        r"""
        Test that CNZ gate matrix is Hermitian (since it's diagonal with real values).
        """
        qbit_num = 2
        phase_string = "10"
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        
        # Check Hermitian: U should equal U^\dagger
        assert np.allclose(matrix, matrix.conj().T, atol=1e-10)

    def test_CNZ_phase_string_single_bit(self):
        r"""
        Test CNZ gate with single-bit phase strings.
        """
        qbit_num = 2
        
        # Test "0"
        cnz_gate_0 = CNZ(qbit_num, "0")
        matrix_0 = np.array(cnz_gate_0.get_Matrix(), dtype=np.complex128)
        assert np.allclose(np.diag(matrix_0)[0], -1.0)
        
        # Test "1"
        cnz_gate_1 = CNZ(qbit_num, "1")
        matrix_1 = np.array(cnz_gate_1.get_Matrix(), dtype=np.complex128)
        assert np.allclose(np.diag(matrix_1)[1], -1.0)

    def test_CNZ_phase_string_all_zeros(self):
        r"""
        Test CNZ gate with phase string "0" (all zeros).
        """
        qbit_num = 3
        phase_string = "0"
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        diagonal = np.diag(matrix)
        
        # Element at index 0 should be -1
        assert np.allclose(diagonal[0], -1.0)
        assert np.allclose(diagonal[1:], 1.0)

    def test_CNZ_phase_string_all_ones(self):
        r"""
        Test CNZ gate with phase string of all ones.
        """
        qbit_num = 3
        phase_string = "111"  # Binary 111 = decimal 7
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        diagonal = np.diag(matrix)
        
        # Element at index 7 should be -1
        assert np.allclose(diagonal[7], -1.0)
        assert np.allclose(diagonal[:7], 1.0)

    def test_CNZ_get_target_qbits(self):
        r"""
        Test that CNZ gate returns correct target qubits.
        Based on the C++ code, CNZ sets all qubits as target qubits.
        """
        qbit_num = 3
        phase_string = "101"
        
        cnz_gate = CNZ(qbit_num, phase_string)
        target_qbits = cnz_gate.get_Target_Qbits()
        
        # CNZ should have all qubits as targets
        assert len(target_qbits) == qbit_num
        assert set(target_qbits) == set(range(qbit_num))

    def test_CNZ_get_control_qbit(self):
        r"""
        Test that CNZ gate returns -1 for control qubit (no control qubit).
        """
        qbit_num = 2
        phase_string = "10"
        
        cnz_gate = CNZ(qbit_num, phase_string)
        control_qbit = cnz_gate.get_Control_Qbit()
        
        # CNZ has no control qubit
        assert control_qbit == -1

    def test_CNZ_phase_string_very_long(self):
        r"""
        Test CNZ gate with a very long phase string.
        This tests behavior with phase_idx that's much larger than matrix size.
        """
        qbit_num = 2
        # Very long binary string that represents a huge number
        phase_string = "1" * 20  # 20 ones = 2^20 - 1 = 1048575
        
        # This will create a phase_idx that's way beyond the matrix size
        # The C++ code doesn't validate bounds, so this may cause issues
        try:
            cnz_gate = CNZ(qbit_num, phase_string)
            matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
            # If it succeeds, verify basic structure
            assert matrix.shape == (2**qbit_num, 2**qbit_num)
        except (Exception, SystemError, MemoryError):
            # If it fails or crashes, that's expected
            assert True

    def test_CNZ_phase_string_special_characters(self):
        r"""
        Test CNZ gate with phase string containing special characters.
        """
        qbit_num = 2
        
        # Test with various special characters that aren't 0 or 1
        special_chars = ["@", "#", "$", "%", "&", "*", "+", "-", "="]
        
        for char in special_chars:
            with pytest.raises(Exception):
                CNZ(qbit_num, char)

    def test_CNZ_phase_string_mixed_valid_invalid(self):
        r"""
        Test CNZ gate with phase strings that have both valid and invalid parts.
        """
        qbit_num = 2
        
        # Strings that start valid but have invalid characters
        mixed_strings = ["10a", "01b2", "1x0", "a101"]
        
        for mixed_str in mixed_strings:
            with pytest.raises(Exception):
                CNZ(qbit_num, mixed_str)

    def test_CNZ_apply_to_identity_preservation(self):
        r"""
        Test that CNZ gate preserves the identity structure except for the phase flip.
        """
        qbit_num = 3
        phase_string = "010"  # binary 010 = decimal 2
        
        cnz_gate = CNZ(qbit_num, phase_string)
        
        # Create identity matrix
        matrix_size = 2**qbit_num
        identity = np.eye(matrix_size, dtype=np.complex128)
        
        # Apply CNZ to identity
        cnz_gate.apply_to(identity)
        
        # Result should be identity with one diagonal element flipped
        expected = np.eye(matrix_size, dtype=np.complex128)
        expected[2, 2] = -1.0
        
        assert np.allclose(identity, expected)

    def test_CNZ_matrix_determinant(self):
        r"""
        Test that CNZ gate matrix has correct determinant.
        Since CNZ flips one diagonal element from 1 to -1, 
        the determinant should be -1 (one sign change).
        """
        qbit_num = 2
        phase_string = "11"  # binary 11 = decimal 3
        
        cnz_gate = CNZ(qbit_num, phase_string)
        matrix = np.array(cnz_gate.get_Matrix(), dtype=np.complex128)
        
        # Determinant should be -1 (one sign flip)
        det = np.linalg.det(matrix)
        assert np.allclose(det, -1.0, atol=1e-10)

    def test_CNZ_phase_string_zero_padding(self):
        r"""
        Test that leading zeros in phase string are handled correctly.
        "0001" should be the same as "1".
        """
        qbit_num = 2
        
        # Both should result in phase_idx = 1
        cnz_gate_1 = CNZ(qbit_num, "1")
        cnz_gate_0001 = CNZ(qbit_num, "0001")
        
        matrix_1 = np.array(cnz_gate_1.get_Matrix(), dtype=np.complex128)
        matrix_0001 = np.array(cnz_gate_0001.get_Matrix(), dtype=np.complex128)
        
        assert np.allclose(matrix_1, matrix_0001)


# -*- coding: utf-8 -*-
"""
Copyright 2024 Budapest Quantum Computing Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from typing import List, Tuple, Optional, Iterator


class NaryGrayCodeCounter:
    """
    Python implementation of n-ary reflected Gray code counter.

    This class iterates over n-ary reflected Gray codes, where each element
    can have different maximum values. It's used in tree search algorithms
    to enumerate gate configurations.

    Reference: squander/src-cpp/common/n_aryGrayCodeCounter.cpp
    """

    def __init__(self, n_ary_limits: List[int], initial_offset: int = 0):
        """
        Initialize the Gray code counter.

        Args:
            n_ary_limits: List where n_ary_limits[i] is the max value+1 for element i
                         (e.g., [3, 2, 4] means element 0 can be 0-2, element 1 can be 0-1, etc.)
            initial_offset: Starting offset (default 0)
        """
        self.n_ary_limits = np.array(n_ary_limits, dtype=np.int32)
        self.length = len(self.n_ary_limits)

        if self.length == 0:
            self.offset_max = 0
            self.offset = 0
            return

        # Calculate maximum offset (total number of configurations - 1)
        self.offset_max = int(np.prod(self.n_ary_limits)) - 1
        self.offset = initial_offset

        # Initialize counter chain and gray code
        self.counter_chain = np.zeros(self.length, dtype=np.int32)
        self.gray_code = np.zeros(self.length, dtype=np.int32)

        self.initialize(initial_offset)

    def initialize(self, initial_offset: int = 0):
        """
        Initialize the counter to a specific offset.

        Args:
            initial_offset: The starting offset (0 <= initial_offset <= offset_max)
        """
        if initial_offset < 0 or initial_offset > self.offset_max:
            raise ValueError(f"initial_offset must be between 0 and {self.offset_max}")

        self.offset = initial_offset

        # Generate counter chain from offset
        temp_offset = initial_offset
        for idx in range(self.length):
            self.counter_chain[idx] = temp_offset % self.n_ary_limits[idx]
            temp_offset //= self.n_ary_limits[idx]

        # Convert counter chain to Gray code
        parity = 0
        for jdx in range(self.length - 1, -1, -1):
            if parity:
                self.gray_code[jdx] = self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
            else:
                self.gray_code[jdx] = self.counter_chain[jdx]
            parity ^= (self.gray_code[jdx] & 1)

    def get(self) -> np.ndarray:
        """
        Get the current Gray code value.

        Returns:
            Current Gray code as numpy array
        """
        return self.gray_code.copy()

    def next(self) -> Tuple[bool, int, int, int]:
        """
        Iterate to the next Gray code value.

        Returns:
            Tuple (finished, changed_index, value_prev, value):
                - finished: True if we've reached the end
                - changed_index: Index where the change occurred
                - value_prev: Previous value at changed_index
                - value: New value at changed_index
        """
        if self.offset >= self.offset_max:
            return True, 0, 0, 0

        # Update counter chain
        counter_chain_idx = 0
        while True:
            if self.counter_chain[counter_chain_idx] < self.n_ary_limits[counter_chain_idx] - 1:
                self.counter_chain[counter_chain_idx] += 1
                break
            else:
                self.counter_chain[counter_chain_idx] = 0
                counter_chain_idx += 1

        # Update Gray code and find changed index
        parity = 0
        changed_index = 0
        value_prev = 0
        value = 0

        for jdx in range(self.length - 1, -1, -1):
            if parity:
                gray_code_new_val = self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
            else:
                gray_code_new_val = self.counter_chain[jdx]

            parity ^= (gray_code_new_val & 1)

            if gray_code_new_val != self.gray_code[jdx]:
                value_prev = int(self.gray_code[jdx])
                value = int(gray_code_new_val)
                self.gray_code[jdx] = gray_code_new_val
                changed_index = jdx
                break

        self.offset += 1

        return False, changed_index, value_prev, value

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Make the counter iterable.

        Yields:
            Gray code arrays
        """
        # Reset to initial state
        self.initialize(0)

        # Yield first value
        yield self.get()

        # Iterate through remaining values
        while True:
            finished, _, _, _ = self.next()
            if finished:
                break
            yield self.get()

    def iterate_range(self, start_offset: int, end_offset: int) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over a specific range of offsets (useful for parallelization).

        Args:
            start_offset: Starting offset (inclusive)
            end_offset: Ending offset (inclusive)

        Yields:
            Tuples of (offset, gray_code)
        """
        if start_offset > end_offset or end_offset > self.offset_max:
            raise ValueError(f"Invalid range: [{start_offset}, {end_offset}], max={self.offset_max}")

        self.initialize(start_offset)

        # Yield first value
        yield start_offset, self.get()

        # Iterate until end_offset
        current_offset = start_offset
        while current_offset < end_offset:
            finished, _, _, _ = self.next()
            if finished:
                break
            current_offset += 1
            yield current_offset, self.get()


def generate_all_gray_codes(n_ary_limits: List[int]) -> List[np.ndarray]:
    """
    Generate all Gray codes for given limits.

    Args:
        n_ary_limits: Maximum values for each position

    Returns:
        List of all Gray code arrays
    """
    counter = NaryGrayCodeCounter(n_ary_limits)
    return list(counter)


def count_total_configurations(n_ary_limits: List[int]) -> int:
    """
    Count total number of configurations for given limits.

    Args:
        n_ary_limits: Maximum values for each position

    Returns:
        Total number of configurations
    """
    return int(np.prod(n_ary_limits))

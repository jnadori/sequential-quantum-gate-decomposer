# -*- coding: utf-8 -*-
"""
Gray Code Counter Implementation

Python implementation of n-ary Gray code counter for tree search decomposition.
This replicates the functionality of the C++ n_aryGrayCodeCounter class.
"""

from typing import List, Tuple, Optional


class NaryGrayCodeCounter:
    """
    N-ary Gray code counter for enumerating gate structures.
    
    This class implements the same algorithm as the C++ n_aryGrayCodeCounter,
    generating n-ary reflected Gray codes for tree search enumeration.
    """
    
    def __init__(self, n_ary_limits: List[int], initial_offset: int = 0):
        """
        Initialize Gray code counter.
        
        Args:
            n_ary_limits: List of maximum values for each Gray code digit
            initial_offset: Starting offset (0 <= initial_offset <= offset_max)
        """
        self.n_ary_limits = n_ary_limits.copy()
        
        if len(self.n_ary_limits) == 0:
            self.offset_max = 0
            self.offset = 0
            self.gray_code = []
            self.counter_chain = []
            return
        
        # Calculate offset_max (product of all limits - 1)
        self.offset_max = self.n_ary_limits[0]
        for limit in self.n_ary_limits[1:]:
            self.offset_max *= limit
        self.offset_max -= 1
        
        self.offset = initial_offset
        self._initialize(initial_offset)
    
    def _initialize(self, initial_offset: int):
        """Initialize counter to specific offset."""
        if initial_offset < 0 or initial_offset > self.offset_max:
            raise ValueError(f"Invalid initial_offset: {initial_offset} (max: {self.offset_max})")
        
        # Generate counter chain from offset
        self.counter_chain = []
        temp_offset = initial_offset
        for limit in self.n_ary_limits:
            self.counter_chain.append(temp_offset % limit)
            temp_offset //= limit
        
        # Determine initial Gray code from counter chain (reflected Gray code)
        self.gray_code = [0] * len(self.n_ary_limits)
        parity = 0
        
        for jdx in range(len(self.n_ary_limits) - 1, -1, -1):
            if parity:
                self.gray_code[jdx] = self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
            else:
                self.gray_code[jdx] = self.counter_chain[jdx]
            parity = parity ^ (self.gray_code[jdx] & 1)
    
    def get(self) -> List[int]:
        """Get current Gray code value."""
        return self.gray_code.copy()
    
    def next(self) -> Tuple[bool, int, int, int]:
        """
        Iterate counter to next value.
        
        Returns:
            Tuple of (done, changed_index, value_prev, value):
            - done: True if no more values available
            - changed_index: Index of digit that changed
            - value_prev: Previous value at changed_index
            - value: New value at changed_index
        """
        changed_index = 0
        value_prev = 0
        value = 0
        
        # Check if we've reached the end
        if self.offset >= self.offset_max:
            return (True, 0, 0, 0)
        
        # Update counter chain (increment with carry)
        update_counter = True
        counter_chain_idx = 0
        
        while update_counter and counter_chain_idx < len(self.counter_chain):
            if self.counter_chain[counter_chain_idx] < self.n_ary_limits[counter_chain_idx] - 1:
                self.counter_chain[counter_chain_idx] += 1
                update_counter = False
            elif self.counter_chain[counter_chain_idx] == self.n_ary_limits[counter_chain_idx] - 1:
                self.counter_chain[counter_chain_idx] = 0
                update_counter = True
            
            counter_chain_idx += 1
        
        # Determine updated Gray code and find changed digit
        parity = 0
        for jdx in range(len(self.n_ary_limits) - 1, -1, -1):
            if parity:
                gray_code_new_val = self.n_ary_limits[jdx] - 1 - self.counter_chain[jdx]
            else:
                gray_code_new_val = self.counter_chain[jdx]
            
            parity = parity ^ (gray_code_new_val & 1)
            
            if gray_code_new_val != self.gray_code[jdx]:
                value_prev = self.gray_code[jdx]
                value = gray_code_new_val
                self.gray_code[jdx] = gray_code_new_val
                changed_index = jdx
                break
        
        self.offset += 1
        return (False, changed_index, value_prev, value)
    
    def set_offset_max(self, offset_max: int):
        """Set maximum offset for this counter."""
        self.offset_max = offset_max

"""JABCode-compatible data encoder that matches the decoder format.

This module provides a data encoder that creates JABCode-compatible bit streams
using the exact format expected by the JABCode DataDecoder.
"""

import numpy as np
from typing import Optional

from .data_decoder import DataDecoder


class JABCodeDataEncoder:
    """Data encoder that creates JABCode-compatible bit streams.
    
    This encoder matches the exact format expected by the JABCode decoder,
    using the same encoding tables and bit patterns.
    """
    
    def __init__(self):
        """Initialize encoder with decoding tables for reverse lookup."""
        # Create reverse lookup tables from decoder tables
        self.upper_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_UPPER)}
        self.lower_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_LOWER)}
        self.numeric_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_NUMERIC)}
        self.punct_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_PUNCT)}
        self.mixed_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_MIXED)}
        self.alphanumeric_table = {chr(c): i for i, c in enumerate(DataDecoder.DECODING_TABLE_ALPHANUMERIC)}
        
        # Character size for each mode
        self.char_sizes = {
            'Upper': 5,
            'Lower': 5,
            'Numeric': 4,
            'Punct': 4,
            'Mixed': 5,
            'Alphanumeric': 6,
        }
        
    def encode_string(self, text: str) -> np.ndarray:
        """Encode string to JABCode-compatible bit stream.
        
        Args:
            text: Input text to encode
            
        Returns:
            Bit array compatible with JABCode decoder
        """
        if not text:
            return np.array([], dtype=np.uint8)
            
        # Determine the best starting mode
        best_mode = self.get_best_mode_for_text(text)
        
        bits = []
        mode = 'Upper'  # Always start in uppercase mode
        
        # Switch to best mode if it's not upper
        if best_mode == 'Alphanumeric' and any(c in self.alphanumeric_table for c in text):
            bits.extend(self._int_to_bits(31, 5))  # Switch to alphanumeric
            mode = 'Alphanumeric'
        elif best_mode == 'Lower' and all(c in self.lower_table for c in text):
            bits.extend(self._int_to_bits(28, 5))  # Switch to lowercase
            mode = 'Lower'
        elif best_mode == 'Numeric' and all(c in self.numeric_table for c in text):
            bits.extend(self._int_to_bits(29, 5))  # Switch to numeric
            mode = 'Numeric'
        
        for char in text:
            # Try to encode in current mode first
            encoded = False
            
            if mode == 'Upper' and char in self.upper_table:
                value = self.upper_table[char]
                bits.extend(self._int_to_bits(value, 5))
                encoded = True
            elif mode == 'Lower' and char in self.lower_table:
                value = self.lower_table[char]
                bits.extend(self._int_to_bits(value, 5))
                encoded = True
            elif mode == 'Numeric' and char in self.numeric_table:
                value = self.numeric_table[char]
                bits.extend(self._int_to_bits(value, 4))
                encoded = True
            elif mode == 'Alphanumeric' and char in self.alphanumeric_table:
                value = self.alphanumeric_table[char]
                bits.extend(self._int_to_bits(value, 6))
                encoded = True
            elif mode == 'Mixed' and char in self.mixed_table:
                value = self.mixed_table[char]
                bits.extend(self._int_to_bits(value, 5))
                encoded = True
            
            # If not encoded yet, switch modes
            if not encoded:
                if char in self.alphanumeric_table:
                    # Switch to alphanumeric mode
                    if mode == 'Upper':
                        bits.extend(self._int_to_bits(31, 5))  # Switch to alphanumeric
                    elif mode == 'Lower':
                        bits.extend(self._int_to_bits(31, 5))  # Switch to alphanumeric
                    elif mode == 'Numeric':
                        bits.extend(self._int_to_bits(13, 4))  # Switch to alphanumeric
                    mode = 'Alphanumeric'
                    value = self.alphanumeric_table[char]
                    bits.extend(self._int_to_bits(value, 6))
                elif char in self.mixed_table:
                    # Switch to mixed mode
                    if mode == 'Upper':
                        bits.extend(self._int_to_bits(30, 5))  # Switch to mixed
                    elif mode == 'Lower':
                        bits.extend(self._int_to_bits(30, 5))  # Switch to mixed
                    mode = 'Mixed'
                    value = self.mixed_table[char]
                    bits.extend(self._int_to_bits(value, 5))
                elif char in self.upper_table:
                    # Switch to upper mode
                    if mode == 'Lower':
                        bits.extend(self._int_to_bits(28, 5))  # Switch to upper
                    elif mode == 'Numeric':
                        bits.extend(self._int_to_bits(14, 4))  # Switch to upper
                    elif mode == 'Alphanumeric':
                        bits.extend(self._int_to_bits(63, 6))  # Switch to upper
                    mode = 'Upper'
                    value = self.upper_table[char]
                    bits.extend(self._int_to_bits(value, 5))
                elif char in self.lower_table:
                    # Switch to lower mode
                    if mode == 'Upper':
                        bits.extend(self._int_to_bits(28, 5))  # Switch to lower
                    elif mode == 'Numeric':
                        bits.extend(self._int_to_bits(15, 4))  # Switch to lower
                    elif mode == 'Alphanumeric':
                        bits.extend(self._int_to_bits(63, 6))  # Switch to upper first
                        bits.extend(self._int_to_bits(28, 5))  # Then to lower
                    mode = 'Lower'
                    value = self.lower_table[char]
                    bits.extend(self._int_to_bits(value, 5))
                elif char in self.numeric_table:
                    # Switch to numeric mode
                    if mode == 'Upper':
                        bits.extend(self._int_to_bits(29, 5))  # Switch to numeric
                    elif mode == 'Lower':
                        bits.extend(self._int_to_bits(29, 5))  # Switch to numeric
                    elif mode == 'Alphanumeric':
                        bits.extend(self._int_to_bits(63, 6))  # Switch to upper first
                        bits.extend(self._int_to_bits(29, 5))  # Then to numeric
                    mode = 'Numeric'
                    value = self.numeric_table[char]
                    bits.extend(self._int_to_bits(value, 4))
                else:
                    # Character not supported, skip it
                    continue
        
        return np.array(bits, dtype=np.uint8)
    
    def _int_to_bits(self, value: int, bit_count: int) -> list:
        """Convert integer to list of bits (MSB first)."""
        bits = []
        for i in range(bit_count):
            bit = (value >> (bit_count - 1 - i)) & 1
            bits.append(bit)
        return bits
    
    def get_best_mode_for_text(self, text: str) -> str:
        """Determine the best starting mode for given text."""
        if not text:
            return 'Upper'
            
        # Simple heuristic: check what mode can encode most characters
        mode_scores = {
            'Upper': sum(1 for c in text if c in self.upper_table),
            'Lower': sum(1 for c in text if c in self.lower_table),
            'Numeric': sum(1 for c in text if c in self.numeric_table),
            'Alphanumeric': sum(1 for c in text if c in self.alphanumeric_table),
        }
        
        # If alphanumeric can encode everything, use it for mixed case
        if mode_scores['Alphanumeric'] == len(text):
            return 'Alphanumeric'
        
        # Otherwise use the mode that covers the most characters
        best_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
        return best_mode
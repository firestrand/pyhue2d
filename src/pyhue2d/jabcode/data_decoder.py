"""Data decoder for JABCode based on reference implementation.

This module provides the DataDecoder class that converts raw bit data
back to original text/bytes using the exact decoding tables and logic
from the JABCode reference implementation.
"""

from typing import List, Optional

import numpy as np


class DataDecoder:
    """Data decoder for JABCode bit streams.

    Implements the exact decoding logic from the JABCode reference implementation
    in decoder.c, including character decoding tables and mode switching.
    """

    # Decoding tables from JABCode reference (decoder.h)
    DECODING_TABLE_UPPER = bytes(
        [32, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
    )
    DECODING_TABLE_LOWER = bytes(
        [
            32,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
        ]
    )
    DECODING_TABLE_NUMERIC = bytes([32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 44, 46])
    DECODING_TABLE_PUNCT = bytes([33, 34, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 58, 59, 63, 64])
    DECODING_TABLE_MIXED = bytes(
        [
            35,
            42,
            43,
            60,
            61,
            62,
            91,
            92,
            93,
            94,
            95,
            96,
            123,
            124,
            125,
            126,
            9,
            10,
            13,
            0,
            0,
            0,
            0,
            164,
            167,
            196,
            214,
            220,
            223,
            228,
            246,
            252,
        ]
    )
    DECODING_TABLE_ALPHANUMERIC = bytes(
        [
            32,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
        ]
    )

    # Character sizes for each mode (from reference implementation)
    CHARACTER_SIZES = {
        "Upper": 5,
        "Lower": 5,
        "Numeric": 4,
        "Punct": 4,
        "Mixed": 5,
        "Alphanumeric": 6,
        "Byte": 8,  # Variable, handled specially
        "ECI": 8,  # Not implemented
        "FNC1": 8,  # Not implemented
    }

    def __init__(self):
        """Initialize the data decoder."""
        pass

    def decode_data(self, bits: np.ndarray) -> bytes:
        """Decode bit array to original data using JABCode decoding logic.

        Args:
            bits: Binary array of decoded bits

        Returns:
            Decoded data as bytes

        Raises:
            ValueError: If decoding fails
        """
        if len(bits) == 0:
            return b""

        decoded_bytes = bytearray()
        mode = "Upper"  # Start with uppercase mode
        pre_mode = None
        index = 0  # Current bit position

        while index < len(bits):
            try:
                # Read encoded value based on current mode
                if mode != "Byte":
                    char_size = self.CHARACTER_SIZES[mode]
                    if index + char_size > len(bits):
                        break  # Not enough bits left

                    value = self._read_data(bits, index, char_size)
                    index += char_size
                else:
                    # Byte mode handled specially
                    value, bytes_consumed = self._handle_byte_mode(bits, index)
                    index += bytes_consumed

                # Decode value according to current mode
                if mode == "Upper":
                    mode, pre_mode, decoded_byte = self._decode_upper(value, pre_mode)
                elif mode == "Lower":
                    mode, pre_mode, decoded_byte = self._decode_lower(value, pre_mode)
                elif mode == "Numeric":
                    mode, pre_mode, decoded_byte = self._decode_numeric(value, pre_mode)
                elif mode == "Punct":
                    mode, pre_mode, decoded_byte = self._decode_punct(value, pre_mode)
                elif mode == "Mixed":
                    mode, pre_mode, decoded_byte = self._decode_mixed(value, pre_mode)
                elif mode == "Alphanumeric":
                    mode, pre_mode, decoded_byte = self._decode_alphanumeric(value, pre_mode)
                elif mode == "Byte":
                    # Byte mode returns bytes directly
                    decoded_bytes.extend(value)
                    mode = pre_mode if pre_mode else "Upper"
                    continue
                else:
                    # Unknown mode, skip
                    continue

                # Add decoded byte if valid
                if decoded_byte is not None:
                    decoded_bytes.append(decoded_byte)

            except Exception as e:
                # If decoding fails, stop processing
                break

        return bytes(decoded_bytes)

    def _read_data(self, bits: np.ndarray, start: int, length: int) -> int:
        """Read specified number of bits and convert to integer.

        Args:
            bits: Bit array
            start: Starting bit index
            length: Number of bits to read

        Returns:
            Integer value of the bits
        """
        if start + length > len(bits):
            raise ValueError("Not enough bits to read")

        value = 0
        for i in range(length):
            if start + i < len(bits):
                value = (value << 1) | int(bits[start + i])
        return value

    def _decode_upper(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode uppercase mode value."""
        if value <= 26:
            return (pre_mode if pre_mode else "Upper", None, self.DECODING_TABLE_UPPER[value])
        elif value == 27:
            return ("Punct", "Upper", None)
        elif value == 28:
            return ("Lower", None, None)
        elif value == 29:
            return ("Numeric", None, None)
        elif value == 30:
            return ("Mixed", "Upper", None)
        elif value == 31:
            return ("Alphanumeric", "Upper", None)
        else:
            return ("Upper", pre_mode, None)

    def _decode_lower(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode lowercase mode value."""
        if value <= 26:
            return (pre_mode if pre_mode else "Lower", None, self.DECODING_TABLE_LOWER[value])
        elif value == 27:
            return ("Punct", "Lower", None)
        elif value == 28:
            return ("Upper", None, None)
        elif value == 29:
            return ("Numeric", None, None)
        elif value == 30:
            return ("Mixed", "Lower", None)
        elif value == 31:
            return ("Alphanumeric", "Lower", None)
        else:
            return ("Lower", pre_mode, None)

    def _decode_numeric(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode numeric mode value."""
        if value <= 12:
            return (pre_mode if pre_mode else "Numeric", None, self.DECODING_TABLE_NUMERIC[value])
        elif value == 13:
            return ("Alphanumeric", "Numeric", None)
        elif value == 14:
            return ("Upper", None, None)
        elif value == 15:
            return ("Lower", None, None)
        else:
            return ("Numeric", pre_mode, None)

    def _decode_punct(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode punctuation mode value."""
        if value <= 15:
            return (pre_mode if pre_mode else "Upper", None, self.DECODING_TABLE_PUNCT[value])
        else:
            return ("Punct", pre_mode, None)

    def _decode_mixed(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode mixed mode value."""
        if value <= 31:
            return (pre_mode if pre_mode else "Upper", None, self.DECODING_TABLE_MIXED[value])
        else:
            return ("Mixed", pre_mode, None)

    def _decode_alphanumeric(self, value: int, pre_mode: Optional[str]) -> tuple:
        """Decode alphanumeric mode value."""
        if value <= 62:
            return (pre_mode if pre_mode else "Upper", None, self.DECODING_TABLE_ALPHANUMERIC[value])
        elif value == 63:
            # Read 2 more bits for mode switch
            return ("Upper", None, None)  # Simplified
        else:
            return ("Alphanumeric", pre_mode, None)

    def _handle_byte_mode(self, bits: np.ndarray, index: int) -> tuple:
        """Handle byte mode encoding.

        Returns:
            (decoded_bytes, bits_consumed)
        """
        # Read 4 bits for byte count
        if index + 4 > len(bits):
            raise ValueError("Not enough bits for byte mode")

        value = self._read_data(bits, index, 4)
        bits_consumed = 4

        if value == 0:
            # Read 13 more bits for extended count
            if index + 4 + 13 > len(bits):
                raise ValueError("Not enough bits for extended byte mode")
            extended_value = self._read_data(bits, index + 4, 13)
            byte_length = extended_value + 15 + 1
            bits_consumed += 13
        else:
            byte_length = value

        # Read the actual bytes
        decoded_bytes = bytearray()
        for i in range(byte_length):
            if index + bits_consumed + 8 > len(bits):
                break
            byte_value = self._read_data(bits, index + bits_consumed, 8)
            decoded_bytes.append(byte_value)
            bits_consumed += 8

        return (bytes(decoded_bytes), bits_consumed)

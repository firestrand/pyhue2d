"""Data decoder for JABCode based on reference implementation.

This module provides the DataDecoder class that converts raw bit data
back to original text/bytes using the exact decoding tables and logic
from the JABCode reference implementation.
"""

from typing import Any, List, Optional, Tuple

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

    def _extract_net_data(self, bits: Any) -> List[int]:
        """Extract net message bits by stripping metadata flag and docked positions from tail."""
        bit_list = [int(b) for b in bits]
        offset = len(bit_list) - 1
        while offset >= 0 and bit_list[offset] == 0:
            offset -= 1
        # Skip flag bit 1
        offset -= 1
        # Skip 4 docked position bits
        offset -= 4
        net_length = max(0, offset + 1)
        return bit_list[:net_length]

    def decode_data_from_bits(self, bits: Any) -> bytes:
        """Extract net message bits and decode to original bytes.

        Args:
            bits: Binary array or list of pre-ECC data bits including tail flag.

        Returns:
            Decoded payload bytes.
        """
        net_bits = self._extract_net_data(bits)
        return self.decode_data(net_bits)

    def decode_data_from_hex(self, encoded_data_hex: str) -> bytes:
        """Decode pre-ECC hex string (where bits are at odd indices) to plaintext bytes.

        Args:
            encoded_data_hex: Hex-serialized pre-ECC bitstream.

        Returns:
            Decoded payload bytes.
        """
        bits = [int(c) for c in encoded_data_hex[1::2]]
        return self.decode_data_from_bits(bits)

    def decode_data(self, bits: Any) -> bytes:
        """Decode bit array to original data using JABCode decoding logic.

        Args:
            bits: Binary array or list of decoded bits

        Returns:
            Decoded data as bytes
        """
        bit_list = [int(b) for b in bits]
        if len(bit_list) == 0:
            return b""

        decoded_bytes = bytearray()
        mode = "Upper"
        pre_mode: Optional[str] = None
        index = 0

        def read_bits(idx: int, length: int) -> Tuple[Optional[int], int]:
            if idx + length > len(bit_list):
                return None, idx
            val = 0
            for i in range(length):
                val = (val << 1) | bit_list[idx + i]
            return val, idx + length

        while index < len(bit_list):
            flag = False
            if mode != "Byte":
                char_size = self.CHARACTER_SIZES[mode]
                value, new_index = read_bits(index, char_size)
                if value is None:
                    break
                index = new_index
            else:
                value = 0

            if mode == "Upper":
                if value <= 26:
                    decoded_bytes.append(self.DECODING_TABLE_UPPER[value])
                    if pre_mode is not None:
                        mode = pre_mode
                else:
                    if value == 27:
                        mode = "Punct"
                        pre_mode = "Upper"
                    elif value == 28:
                        mode = "Lower"
                        pre_mode = None
                    elif value == 29:
                        mode = "Numeric"
                        pre_mode = None
                    elif value == 30:
                        mode = "Alphanumeric"
                        pre_mode = None
                    elif value == 31:
                        val2, new_idx = read_bits(index, 2)
                        if val2 is None:
                            break
                        index = new_idx
                        if val2 == 0:
                            mode = "Byte"
                            pre_mode = "Upper"
                        elif val2 == 1:
                            mode = "Mixed"
                            pre_mode = "Upper"
                        elif val2 == 2:
                            mode = "ECI"
                            pre_mode = None
                        elif val2 == 3:
                            flag = True
            elif mode == "Lower":
                if value <= 26:
                    decoded_bytes.append(self.DECODING_TABLE_LOWER[value])
                    if pre_mode is not None:
                        mode = pre_mode
                else:
                    if value == 27:
                        mode = "Punct"
                        pre_mode = "Lower"
                    elif value == 28:
                        mode = "Upper"
                        pre_mode = "Lower"
                    elif value == 29:
                        mode = "Numeric"
                        pre_mode = None
                    elif value == 30:
                        mode = "Alphanumeric"
                        pre_mode = None
                    elif value == 31:
                        val2, new_idx = read_bits(index, 2)
                        if val2 is None:
                            break
                        index = new_idx
                        if val2 == 0:
                            mode = "Byte"
                            pre_mode = "Lower"
                        elif val2 == 1:
                            mode = "Mixed"
                            pre_mode = "Lower"
                        elif val2 == 2:
                            mode = "Upper"
                            pre_mode = None
                        elif val2 == 3:
                            mode = "FNC1"
                            pre_mode = None
            elif mode == "Numeric":
                if value <= 12:
                    decoded_bytes.append(self.DECODING_TABLE_NUMERIC[value])
                    if pre_mode is not None:
                        mode = pre_mode
                else:
                    if value == 13:
                        mode = "Punct"
                        pre_mode = "Numeric"
                    elif value == 14:
                        mode = "Upper"
                        pre_mode = None
                    elif value == 15:
                        val2, new_idx = read_bits(index, 2)
                        if val2 is None:
                            break
                        index = new_idx
                        if val2 == 0:
                            mode = "Byte"
                            pre_mode = "Numeric"
                        elif val2 == 1:
                            mode = "Mixed"
                            pre_mode = "Numeric"
                        elif val2 == 2:
                            mode = "Upper"
                            pre_mode = "Numeric"
                        elif val2 == 3:
                            mode = "Lower"
                            pre_mode = None
            elif mode == "Punct":
                if 0 <= value <= 15:
                    decoded_bytes.append(self.DECODING_TABLE_PUNCT[value])
                    mode = pre_mode if pre_mode else "Upper"
            elif mode == "Mixed":
                if 0 <= value <= 31:
                    if value == 19:
                        decoded_bytes.extend([10, 13])
                    elif value == 20:
                        decoded_bytes.extend([44, 32])  # ", "
                    elif value == 21:
                        decoded_bytes.extend([46, 32])  # ". "
                    elif value == 22:
                        decoded_bytes.extend([58, 32])  # ": "
                    else:
                        decoded_bytes.append(self.DECODING_TABLE_MIXED[value])
                    mode = pre_mode if pre_mode else "Upper"
            elif mode == "Alphanumeric":
                if value <= 62:
                    decoded_bytes.append(self.DECODING_TABLE_ALPHANUMERIC[value])
                    if pre_mode is not None:
                        mode = pre_mode
                elif value == 63:
                    val2, new_idx = read_bits(index, 2)
                    if val2 is None:
                        break
                    index = new_idx
                    if val2 == 0:
                        mode = "Byte"
                        pre_mode = "Alphanumeric"
                    elif val2 == 1:
                        mode = "Mixed"
                        pre_mode = "Alphanumeric"
                    elif val2 == 2:
                        mode = "Punct"
                        pre_mode = "Alphanumeric"
                    elif val2 == 3:
                        mode = "Upper"
                        pre_mode = None
            elif mode == "Byte":
                val4, new_idx = read_bits(index, 4)
                if val4 is None:
                    break
                index = new_idx
                if val4 == 0:
                    val13, new_idx = read_bits(index, 13)
                    if val13 is None:
                        break
                    index = new_idx
                    byte_len = val13 + 16
                else:
                    byte_len = val4

                for _ in range(byte_len):
                    b, new_idx = read_bits(index, 8)
                    if b is None:
                        break
                    index = new_idx
                    decoded_bytes.append(b)
                mode = pre_mode if pre_mode else "Upper"
            elif mode in ("ECI", "FNC1"):
                break

            if flag:
                break

        return bytes(decoded_bytes)

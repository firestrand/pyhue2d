"""Symbol matrix builder for JABCode symbols."""

from __future__ import annotations

import math
from typing import Union

from .jabcode_data_encoder import JABCodeDataEncoder
from .ldpc.codec import LDPCCodec
from .ldpc.parameters import LDPCParameters
from .ldpc.seed_config import RandomSeedConfig
from .module_data_extractor import ModuleDataExtractor

# Master Symbol Version 1 (21x21) Data Map: 1 = data, 0 = non-data
V1_DATA_MAP: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Standard Version 1 Master non-data module template (92 modules: 4 finders + 24 metadata modules)
V1_MASTER_TEMPLATE: list[list[int]] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 3, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 6, 6, 0, 0, 0],
    [0, 0, 3, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6, 0, 6, 0, 0],
    [0, 0, 0, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 6, 6, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 6, 6, 0],
    [0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 6, 0],
    [0, 3, 0, 3, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 6, 0, 6, 0, 6, 0],
    [0, 3, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 3, 3, 3, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 6, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


class SymbolMatrixBuilder:
    """Builds a complete JABCode symbol matrix with data and patterns."""

    def __init__(self) -> None:
        self.data_encoder = JABCodeDataEncoder()
        self.extractor = ModuleDataExtractor()

    def build_matrix(
        self,
        data: Union[str, bytes],
        colors: int = 8,
        ecc_level: Union[int, str] = 3,
        mask_pattern: int = 7,
    ) -> list[list[int]]:
        """Build 2D module color index matrix for the given data and parameters.

        Args:
            data: Plaintext string or raw bytes to encode.
            colors: Number of palette colors (e.g. 8).
            ecc_level: Error correction level integer (e.g. 3).
            mask_pattern: Mask pattern index (0-7, default 7).

        Returns:
            21x21 list of lists of color indices.
        """
        text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        pre_ecc_bits = self.data_encoder.encode_text(text, target_length=580)

        # LDPC encode
        params = LDPCParameters.for_ecc_level(ecc_level)
        codec = LDPCCodec(params, RandomSeedConfig())
        codeword_bits = codec.encode_codeword_bits(pre_ecc_bits)

        # Pack codeword bits into module color indices (3 bits per module for 8 colors)
        bits_per_mod = int(math.log2(colors)) if colors > 1 else 1
        color_indices: list[int] = []
        for i in range(0, len(codeword_bits), bits_per_mod):
            chunk = codeword_bits[i : i + bits_per_mod]
            val = 0
            for b in chunk:
                val = (val << 1) | b
            color_indices.append(val)

        # Padding module (module 348): bits 0, 1, 0 -> 2
        color_indices.append(2)

        # Populate matrix
        matrix = [[0] * 21 for _ in range(21)]
        idx = 0
        for x in range(21):
            for y in range(21):
                if V1_DATA_MAP[y][x] == 1:
                    raw_val = color_indices[idx]
                    idx += 1
                    mask_val = self.extractor._calculate_mask_value(x, y, mask_pattern, colors)
                    matrix[y][x] = raw_val ^ mask_val
                else:
                    matrix[y][x] = V1_MASTER_TEMPLATE[y][x]

        return matrix

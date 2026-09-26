"""Arbitrary-version eight-color JAB Code symbol encoding."""

import math
from typing import assert_never

from .ldpc.codec import interleave_bits
from .ldpc.generator import encode_ldpc_stream
from .ldpc.parameters import LDPCParameters
from .module_data_extractor import ModuleDataExtractor
from .symbol_layout import metadata_coordinates, pattern_modules, reserved_coordinates, slave_palette_coordinates


def integer_bits(value: int, width: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def payload_bits(data: str | bytes) -> list[int]:
    """Use uppercase or byte segments, preserving arbitrary binary and UTF-8 data."""
    match data:
        case str():
            payload = data.encode("utf-8")
        case bytes():
            payload = data
        case unreachable:
            assert_never(unreachable)
    if all(value == 32 or 65 <= value <= 90 for value in payload):
        return [bit for value in payload for bit in integer_bits(0 if value == 32 else value - 64, 5)]
    bits: list[int] = []
    for start in range(0, len(payload), 8207):
        chunk = payload[start : start + 8207]
        bits.extend(integer_bits(31, 5) + [0, 0])
        if len(chunk) <= 15:
            bits.extend(integer_bits(len(chunk), 4))
        else:
            bits.extend([0] * 4 + integer_bits(len(chunk) - 16, 13))
        for value in chunk:
            bits.extend(integer_bits(value, 8))
    return bits


def build_general_matrix(
    data: str | bytes,
    version: int,
    colors: int,
    ecc_level: int,
    mask_pattern: int,
    *,
    channel_bits: list[int] | None = None,
    is_master: bool = True,
) -> list[list[int]]:
    """Build a square master symbol with dynamic LDPC capacity and metadata."""
    if not 1 <= version <= 32:
        raise ValueError("Symbol version must be between 1 and 32")
    if colors not in (4, 8, 16, 32, 64):
        raise ValueError("Generalized encoding supports 4, 8, 16, 32, or 64 colors")
    if not 0 <= mask_pattern <= 7:
        raise ValueError("Mask pattern must be between 0 and 7")
    params = LDPCParameters.for_ecc_level(ecc_level)
    default_mode = colors == 8 and params.wc == 4 and params.wr == 9 and mask_pattern == 7
    dimension = 17 + 4 * version
    reserved = reserved_coordinates(version, version, colors, is_master, default_mode)
    bits_per_mod = int(math.log2(colors))
    capacity = ((dimension * dimension - len(reserved)) * bits_per_mod // params.wr) * (params.wr - params.wc)
    bits = list(channel_bits) if channel_bits is not None else payload_bits(data) + [0, 0, 0, 0, 1]
    if len(bits) > capacity:
        raise ValueError(f"Payload exceeds symbol capacity ({capacity} bits)")
    bits.extend([0] * (capacity - len(bits)))
    codeword = interleave_bits(encode_ldpc_stream(bits, params.wc, params.wr))
    matrix = [[0] * dimension for _ in range(dimension)]
    for (x, y), color in pattern_modules(version, version, is_master, colors).items():
        matrix[y][x] = color
    if is_master:
        palette_count = min(colors - 2, 62)
        total_meta = 4 * palette_count
        if not default_mode:
            total_meta += 4 + (38 + bits_per_mod - 1) // bits_per_mod
        coords = metadata_coordinates(dimension, dimension, total_meta)
        offset = 0
        if not default_mode:
            nc = bits_per_mod - 1
            metadata = encode_ldpc_stream(integer_bits(nc, 3), 2, 0)
            cyan_idx = {4: 3, 8: 3, 16: 3, 32: 7, 64: 15}.get(colors, 3)
            yellow_idx = {4: 2, 8: 6, 16: 14, 32: 30, 64: 60}.get(colors, 6)
            color_map = {0: 0, 3: cyan_idx, 6: yellow_idx}
            table = ((0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3))
            for start in (0, 3):
                value = 4 * metadata[start] + 2 * metadata[start + 1] + metadata[start + 2]
                c1, c2 = table[value]
                for c in (color_map[c1], color_map[c2]):
                    x, y = coords[offset]
                    matrix[y][x] = c
                    offset += 1
        for i in range(2, min(colors, 64)):
            if i == 2:
                palette = (5 % colors,) * 4
            elif i == 3:
                palette = (6 % colors, 3 % colors, 3 % colors, 6 % colors)
            elif i == 4:
                palette = (1,) * 4
            elif i == 5:
                palette = (2,) * 4
            elif i == 6:
                palette = (4,) * 4
            elif i == 7:
                palette = (7,) * 4
            else:
                palette = (i,) * 4
            for color in palette:
                x, y = coords[offset]
                matrix[y][x] = color
                offset += 1
        if not default_mode:
            metadata = (
                integer_bits(version - 1, 5) * 2
                + integer_bits(params.wc - 3, 3)
                + integer_bits(params.wr - 4, 3)
                + integer_bits(mask_pattern, 3)
            )
            encoded = encode_ldpc_stream(metadata, 2, 0)
            p2_idx = 0
            while p2_idx < len(encoded):
                color = 0
                for _ in range(bits_per_mod):
                    bit = encoded[p2_idx] if p2_idx < len(encoded) else 0
                    color = (color << 1) | bit
                    p2_idx += 1
                x, y = coords[offset]
                matrix[y][x] = color
                offset += 1
    else:
        palette_count = min(colors - 2, 62)
        coords = slave_palette_coordinates(dimension, dimension, palette_count)
        for index, (x, y) in enumerate(coords):
            entry = index // 4 + 2
            if entry == 2:
                c = 5 % colors
            elif entry == 3:
                c = 0 % colors
            elif entry == 4:
                c = 1 % colors
            elif entry == 5:
                c = 2 % colors
            elif entry == 6:
                c = 4 % colors
            elif entry == 7:
                c = 7 % colors
            else:
                c = entry % colors
            matrix[y][x] = c
    extractor = ModuleDataExtractor()
    position = 0
    for x in range(dimension):
        for y in range(dimension):
            if (x, y) in reserved:
                continue
            color = 0
            for _ in range(bits_per_mod):
                bit = codeword[position] if position < len(codeword) else (position - len(codeword)) % 2
                color = (color << 1) | bit
                position += 1
            matrix[y][x] = color ^ extractor._calculate_mask_value(x, y, mask_pattern, colors)
    return matrix

"""Arbitrary-version eight-color JAB Code symbol encoding."""

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
    if colors != 8:
        raise ValueError("Generalized encoding currently requires 8 colors")
    if not 0 <= mask_pattern <= 7:
        raise ValueError("Mask pattern must be between 0 and 7")
    params = LDPCParameters.for_ecc_level(ecc_level)
    default_mode = params.wc == 4 and params.wr == 9 and mask_pattern == 7
    dimension = 17 + 4 * version
    reserved = reserved_coordinates(version, version, colors, is_master, default_mode)
    capacity = ((dimension * dimension - len(reserved)) * 3 // params.wr) * (params.wr - params.wc)
    bits = list(channel_bits) if channel_bits is not None else payload_bits(data) + [0, 0, 0, 0, 1]
    if len(bits) > capacity:
        raise ValueError(f"Payload exceeds symbol capacity ({capacity} bits)")
    bits.extend([0] * (capacity - len(bits)))
    codeword = interleave_bits(encode_ldpc_stream(bits, params.wc, params.wr))
    matrix = [[0] * dimension for _ in range(dimension)]
    for (x, y), color in pattern_modules(version, version, is_master).items():
        matrix[y][x] = color
    if is_master:
        coords = metadata_coordinates(dimension, dimension, 41)
        offset = 0
        if not default_mode:
            metadata = encode_ldpc_stream(integer_bits(2, 3), 2, 0)
            pairs = ((0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3))
            for start in (0, 3):
                value = 4 * metadata[start] + 2 * metadata[start + 1] + metadata[start + 2]
                for color in pairs[value]:
                    x, y = coords[offset]
                    matrix[y][x] = color
                    offset += 1
        for entry, palette in enumerate(((5, 5, 5, 5), (6, 3, 3, 6), (1,) * 4, (2,) * 4, (4,) * 4, (7,) * 4)):
            for corner, color in enumerate(palette):
                x, y = coords[offset + entry * 4 + corner]
                matrix[y][x] = color
        offset += 24
        if not default_mode:
            metadata = (
                integer_bits(version - 1, 5) * 2
                + integer_bits(params.wc - 3, 3)
                + integer_bits(params.wr - 4, 3)
                + integer_bits(mask_pattern, 3)
            )
            encoded = encode_ldpc_stream(metadata, 2, 0) + [0]
            for start in range(0, 39, 3):
                x, y = coords[offset + start // 3]
                matrix[y][x] = 4 * encoded[start] + 2 * encoded[start + 1] + encoded[start + 2]
    else:
        coords = slave_palette_coordinates(dimension, dimension, 6)
        for index, (x, y) in enumerate(coords):
            matrix[y][x] = (5, 0, 1, 2, 4, 7)[index // 4]
    extractor = ModuleDataExtractor()
    position = 0
    for x in range(dimension):
        for y in range(dimension):
            if (x, y) in reserved:
                continue
            color = 0
            for _ in range(3):
                bit = codeword[position] if position < len(codeword) else (position - len(codeword)) % 2
                color = 2 * color + bit
                position += 1
            matrix[y][x] = color ^ extractor._calculate_mask_value(x, y, mask_pattern, colors)
    return matrix

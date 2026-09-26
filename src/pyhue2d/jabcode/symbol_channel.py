from __future__ import annotations

from dataclasses import dataclass, replace

from .exceptions import JABCodeError
from .ldpc.codec import deinterleave_bits
from .ldpc.generator import decode_ldpc_stream
from .module_data_extractor import ModuleDataExtractor
from .symbol_layout import metadata_coordinates, reserved_coordinates


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    version_x: int
    version_y: int
    color_count: int
    wc: int
    wr: int
    mask_pattern: int


@dataclass(frozen=True, slots=True)
class DecodedSymbolChannel:
    metadata: SymbolMetadata
    data_bits: tuple[int, ...]
    docks: tuple[tuple[int, SymbolMetadata], ...]
    corrected_errors: int


def _integer(bits: list[int]) -> int:
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    return result


def _palette_indices(matrix: list[list[int]], color_count: int) -> list[list[int]]:
    if color_count == 4 and any(value >= 4 for row in matrix for value in row):
        palette = {0: 0, 5: 1, 6: 2, 3: 3}
        try:
            return [[palette[value] for value in row] for row in matrix]
        except KeyError as error:
            raise JABCodeError("Invalid four-color module") from error
    return matrix


def _master_metadata(matrix: list[list[int]], error_correction: bool) -> tuple[SymbolMetadata, bool, int]:
    height, width = len(matrix), len(matrix[0])
    coords = metadata_coordinates(width, height, 4)
    colors = [matrix[y][x] for x, y in coords]
    if colors[0] not in (0, 3, 6):
        return SymbolMetadata((width - 17) // 4, (height - 17) // 4, 8, 4, 9, 7), True, 0
    pairs = {(0, 0): 0, (0, 3): 1, (0, 6): 2, (3, 0): 3, (3, 3): 4, (3, 6): 5, (6, 0): 6, (6, 3): 7}
    try:
        encoded = [pairs[(colors[index], colors[index + 1])] for index in (0, 2)]
    except KeyError as error:
        raise JABCodeError("Invalid master color metadata") from error
    part1 = [value >> shift & 1 for value in encoded for shift in (2, 1, 0)]
    decoded, corrected_part1 = decode_ldpc_stream(part1, 2, 0, error_correction=error_correction)
    color_count = 1 << (_integer(decoded) + 1)
    if color_count not in (4, 8):
        raise JABCodeError("Unsupported master color count")
    matrix = _palette_indices(matrix, color_count)
    bits_per_module = color_count.bit_length() - 1
    start = 4 + 4 * (color_count - 2)
    coords = metadata_coordinates(width, height, start + (38 + bits_per_module - 1) // bits_per_module)
    part2 = [matrix[y][x] >> shift & 1 for x, y in coords[start:] for shift in range(bits_per_module - 1, -1, -1)][:38]
    decoded, corrected_part2 = decode_ldpc_stream(part2, 2, 0, error_correction=error_correction)
    metadata = SymbolMetadata(
        _integer(decoded[:5]) + 1,
        _integer(decoded[5:10]) + 1,
        color_count,
        _integer(decoded[10:13]) + 3,
        _integer(decoded[13:16]) + 4,
        _integer(decoded[16:19]),
    )
    return metadata, False, corrected_part1 + corrected_part2


def _split_trailer(
    bits: list[int], metadata: SymbolMetadata, host_position: int | None
) -> tuple[tuple[int, ...], tuple[tuple[int, SymbolMetadata], ...]]:
    index = len(bits) - 1
    while index >= 0 and bits[index] == 0:
        index -= 1
    if index < 0:
        raise JABCodeError("Missing symbol trailer marker")
    index -= 1

    def read(count: int) -> int:
        nonlocal index
        if index - count + 1 < 0:
            raise JABCodeError("Truncated symbol docking metadata")
        value = _integer(list(reversed(bits[index - count + 1 : index + 1])))
        index -= count
        return value

    directions = [direction for direction in range(4) if direction != host_position]
    docked = [direction for direction in directions if read(1)]
    docks: list[tuple[int, SymbolMetadata]] = []
    for direction in docked:
        different_size, different_ecc = read(1), read(1)
        child = metadata
        if different_size:
            version = read(5) + 1
            child = replace(child, version_x=version) if direction >= 2 else replace(child, version_y=version)
        if different_ecc:
            child = replace(child, wc=read(3) + 3, wr=read(3) + 4)
        if child.wc >= child.wr:
            raise JABCodeError("Invalid slave error correction parameters")
        docks.append((direction, child))
    return tuple(bits[: index + 1]), tuple(docks)


def decode_symbol_channel(
    matrix: list[list[int]],
    metadata: SymbolMetadata | None = None,
    host_position: int | None = None,
    error_correction: bool = True,
) -> DecodedSymbolChannel:
    """Decode one symbol before concatenating payloads in docking traversal order."""
    height = len(matrix)
    width = len(matrix[0]) if height else 0
    if (
        not 21 <= height <= 145
        or not 21 <= width <= 145
        or (height - 17) % 4
        or (width - 17) % 4
        or any(len(row) != width for row in matrix)
    ):
        raise JABCodeError("Invalid symbol matrix dimensions")
    is_master = metadata is None
    if host_position is not None and (is_master or host_position not in range(4)):
        raise JABCodeError("Invalid slave host position")
    default_mode = False
    metadata_corrections = 0
    if metadata is None:
        try:
            metadata, default_mode, metadata_corrections = _master_metadata(matrix, error_correction)
        except ValueError as error:
            raise JABCodeError("Master metadata decoding failed") from error
    matrix = _palette_indices(matrix, metadata.color_count)
    if (
        metadata.version_x * 4 + 17 != width
        or metadata.version_y * 4 + 17 != height
        or not 3 <= metadata.wc < metadata.wr <= 11
        or metadata.color_count not in (4, 8)
        or not 0 <= metadata.mask_pattern <= 7
        or any(value < 0 or value >= metadata.color_count for row in matrix for value in row)
    ):
        raise JABCodeError("Symbol matrix does not match its metadata")
    reserved = reserved_coordinates(
        metadata.version_x, metadata.version_y, metadata.color_count, is_master, default_mode=default_mode
    )
    extractor = ModuleDataExtractor()
    bits_per_module = metadata.color_count.bit_length() - 1
    bits: list[int] = []
    for x in range(width):
        for y in range(height):
            if (x, y) not in reserved:
                value = matrix[y][x] ^ extractor._calculate_mask_value(
                    x, y, metadata.mask_pattern, metadata.color_count
                )
                bits.extend(value >> shift & 1 for shift in range(bits_per_module - 1, -1, -1))
    gross_length = len(bits) // metadata.wr * metadata.wr
    try:
        decoded, corrected = decode_ldpc_stream(
            deinterleave_bits(bits[:gross_length]), metadata.wc, metadata.wr, error_correction=error_correction
        )
    except ValueError as error:
        raise JABCodeError("Symbol LDPC decoding failed") from error
    payload, docks = _split_trailer(decoded, metadata, host_position)
    return DecodedSymbolChannel(metadata, payload, docks, corrected + metadata_corrections)

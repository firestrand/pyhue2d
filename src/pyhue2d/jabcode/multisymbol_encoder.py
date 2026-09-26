"""Compact docking trees and payload partitioning for multi-symbol encoding."""

from math import ceil, log2, sqrt

from .general_encoder import build_general_matrix, payload_bits
from .ldpc.parameters import LDPCParameters
from .symbol_layout import reserved_coordinates


def build_multisymbol_matrix(
    data: str | bytes,
    version: int,
    colors: int,
    ecc_level: int,
    mask_pattern: int,
    symbol_count: int,
    columns: int | None = None,
) -> list[list[int]]:
    """Encode an ordered stream over a breadth-first compact docking tree."""
    if not 1 <= symbol_count <= 61:
        raise ValueError("Symbol count must be between 1 and 61")
    if not 1 <= version <= 32:
        raise ValueError("Symbol version must be between 1 and 32")
    if columns is None:
        columns = ceil(sqrt(symbol_count))
    cells = {(index % columns, index // columns) for index in range(symbol_count)}
    order = [(0, 0)]
    seen = {(0, 0)}
    parents: list[int | None] = [None]
    docks: list[list[int]] = []
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for x, y in order:
        children: list[int] = []
        for direction, (dx, dy) in enumerate(directions):
            child = x + dx, y + dy
            if child in cells and child not in seen:
                seen.add(child)
                order.append(child)
                parents.append(direction ^ 1)
                children.append(direction)
        docks.append(children)
    dimension = 17 + 4 * version
    params = LDPCParameters.for_ecc_level(ecc_level)
    default_mode = colors == 8 and params.wc == 4 and params.wr == 9 and mask_pattern == 7
    bits_per_mod = int(log2(colors))
    capacities: list[int] = []
    footers: list[list[int]] = []
    for index, children in enumerate(docks):
        flags = [int(direction in children) for direction in range(4) if direction != parents[index]]
        footer = [0] * (2 * len(children)) + list(reversed(flags)) + [1]
        footers.append(footer)
        reserved = reserved_coordinates(version, version, colors, index == 0, default_mode)
        capacity = ((dimension * dimension - len(reserved)) * bits_per_mod // params.wr) * (params.wr - params.wc)
        capacities.append(capacity - len(footer))
    bits = payload_bits(data)
    if len(bits) > sum(capacities):
        raise ValueError("Payload exceeds combined symbol capacity")
    rows = ceil(symbol_count / columns)
    matrix = [[0] * (columns * dimension) for _ in range(rows * dimension)]
    start = 0
    remaining_capacity = sum(capacities)
    for index, (x, y) in enumerate(order):
        length = (
            len(bits) - start
            if index == symbol_count - 1
            else (len(bits) - start) * capacities[index] // remaining_capacity
        )
        channel = bits[start : start + length] + footers[index]
        tile = build_general_matrix(
            b"", version, colors, ecc_level, mask_pattern, channel_bits=channel, is_master=index == 0
        )
        for row, values in enumerate(tile):
            matrix[y * dimension + row][x * dimension : (x + 1) * dimension] = values
        start += length
        remaining_capacity -= capacities[index]
    return matrix

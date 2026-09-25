"""Test metadata responsiveness to module mutations.

Fact: JAB.METADATA.FOLLOWS_MODULES.v1
Evidence: EV-06
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_metadata_modules_responsiveness():
    """EV-06: Flipping a metadata module changes a reported parameter."""
    sidecar = load_sidecar("example1.png")
    original_matrix = sidecar.symbol_matrix

    coords_path = Path("tests/support/example1_metadata_modules.json")
    with open(coords_path, encoding="utf-8") as f:
        metadata_coords = json.load(f)

    # Decode original matrix
    orig_result = pyhue2d.decode(original_matrix)
    assert orig_result.version == sidecar.version
    assert orig_result.color_count == sidecar.color_number
    assert orig_result.mask_pattern in (sidecar.mask_pattern, 7)

    # Mutate one recorded metadata module: flip module 0 at (6, 1)
    target_coord = metadata_coords[0]
    tx, ty = target_coord["x"], target_coord["y"]

    mutated_matrix = copy.deepcopy(original_matrix)
    # Flip the color value at (tx, ty) to a different valid color index
    original_val = mutated_matrix[ty][tx]
    mutated_matrix[ty][tx] = (original_val + 1) % sidecar.color_number

    # Decoding the mutated matrix must not return identical parameters
    mut_result = pyhue2d.decode(mutated_matrix)
    parameters_changed = (
        mut_result.version != orig_result.version
        or mut_result.color_count != orig_result.color_count
        or mut_result.ecc_level != orig_result.ecc_level
        or mut_result.mask_pattern != orig_result.mask_pattern
    )
    assert parameters_changed, "Decoded parameters did not change when metadata module was flipped"

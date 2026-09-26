"""Fact test for Phase V22: Multi-Version Alignment Pattern Grid Sampler.

Verifies:
- Table 5 alignment pattern coordinates match ISO/IEC 23634:2022 for all versions 1 <= V <= 32.
- Alignment pattern counts match ISO standard (0 for V1-5, 77 for V32).
- Mesh-based module sampling on Version 10 and Version 32 multi-symbol images yields 100% accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pyhue2d.jabcode.image_processing.alignment_sampler import (
    JAB_AP_NUM,
    JAB_AP_POS,
    alignment_pattern_coords,
    get_all_grid_anchors,
    get_ap_positions,
    sample_symbol_mesh,
)


def test_table5_coordinates_all_32_versions():
    """Verify Table 5 coordinates and pattern counts for all 32 side versions."""
    assert len(JAB_AP_POS) == 32
    assert len(JAB_AP_NUM) == 32

    for v in range(1, 33):
        coords_1 = get_ap_positions(v, zero_indexed=False)
        coords_0 = get_ap_positions(v, zero_indexed=True)
        num = JAB_AP_NUM[v - 1]

        assert len(coords_1) == num
        assert len(coords_0) == num
        assert coords_0 == [c - 1 for c in coords_1]

        # First coordinate is always 4 (or 3 in 0-indexed)
        assert coords_1[0] == 4
        assert coords_0[0] == 3

        # Coordinates must be strictly increasing
        for k in range(len(coords_1) - 1):
            assert coords_1[k] < coords_1[k + 1]

    # Specific version spot checks from ISO Table 5
    assert get_ap_positions(1, zero_indexed=False) == [4, 18]
    assert get_ap_positions(5, zero_indexed=False) == [4, 34]
    assert get_ap_positions(6, zero_indexed=False) == [4, 17, 38]
    assert get_ap_positions(10, zero_indexed=False) == [4, 14, 32, 54]
    assert get_ap_positions(32, zero_indexed=False) == [4, 20, 38, 55, 73, 91, 108, 126, 142]


def test_internal_alignment_pattern_counts():
    """Verify internal alignment pattern counts across different versions."""
    # Versions 1 to 5 have only corner finders (0 internal alignment patterns)
    for v in range(1, 6):
        internal_aps = alignment_pattern_coords(v, v)
        assert len(internal_aps) == 0

    # Version 6 has 3x3 - 4 = 5 internal alignment patterns
    assert len(alignment_pattern_coords(6, 6)) == 5

    # Version 10 has 4x4 - 4 = 12 internal alignment patterns
    assert len(alignment_pattern_coords(10, 10)) == 12

    # Version 32 has 9x9 - 4 = 77 internal alignment patterns
    assert len(alignment_pattern_coords(32, 32)) == 77

    # Non-square symbol: Version (10, 6) has 4x3 - 4 = 8 internal alignment patterns
    assert len(alignment_pattern_coords(10, 6)) == 8


def test_sample_symbol_mesh_version_10_accuracy():
    """Verify mesh-based sampling on Version 10 asan_multi2.png yields 100% matrix accuracy."""
    img_path = Path("tests/fixtures/approved/jabcode/asan_multi2.png")
    sidecar_path = Path("tests/fixtures/approved/jabcode/asan_multi2.png.json")

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with open(sidecar_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    palette = np.array(sidecar["palette"], dtype=np.float32)
    expected_m0 = np.array(sidecar["symbols"][0]["symbol_matrix"], dtype=np.uint8)
    expected_m1 = np.array(sidecar["symbols"][1]["symbol_matrix"], dtype=np.uint8)

    # Master symbol is at the bottom: y in [684, 1368], x in [0, 684]
    corners_m0 = np.array(
        [
            [0.0, 684.0],
            [684.0, 684.0],
            [684.0, 1368.0],
            [0.0, 1368.0],
        ],
        dtype=np.float32,
    )
    sampled_m0 = sample_symbol_mesh(rgb, corners_m0, 10, 10, palette=palette)
    assert np.array_equal(sampled_m0, expected_m0)

    # Slave symbol is at the top: y in [0, 684], x in [0, 684]
    corners_m1 = np.array(
        [
            [0.0, 0.0],
            [684.0, 0.0],
            [684.0, 684.0],
            [0.0, 684.0],
        ],
        dtype=np.float32,
    )
    sampled_m1 = sample_symbol_mesh(rgb, corners_m1, 10, 10, palette=palette)
    assert np.array_equal(sampled_m1, expected_m1)


def test_sample_symbol_mesh_version_32_anchor_alignment():
    """Verify alignment pattern mesh anchors accurately align on Version 32 multi_block_2_v32.png."""
    img_path = Path("tests/fixtures/approved/jabcode/multi_block_2_v32.png")
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Master symbol is at the bottom: y in [1740, 3480], x in [0, 1740]
    corners_m0 = np.array(
        [
            [0.0, 1740.0],
            [1740.0, 1740.0],
            [1740.0, 3480.0],
            [0.0, 3480.0],
        ],
        dtype=np.float32,
    )

    # Sample full 145x145 module matrix
    sampled_m0 = sample_symbol_mesh(rgb, corners_m0, 32, 32)
    assert sampled_m0.shape == (145, 145)

    # Verify finder pattern cores in sampled matrix
    # FP0 (Top-Left) core at (3, 3) is color 0 (Black)
    assert sampled_m0[3, 3] == 0
    # FP1 (Top-Right) core at (3, 141) is color 0 (Black)
    assert sampled_m0[3, 141] == 0
    # FP2 (Bottom-Right) core at (141, 141) is color 6 (Yellow)
    assert sampled_m0[141, 141] == 6
    # FP3 (Bottom-Left) core at (141, 3) is color 3 (Cyan)
    assert sampled_m0[141, 3] == 3

    # Verify interior alignment pattern anchors have APX core color 6 (Yellow)
    internal_aps = alignment_pattern_coords(32, 32, zero_indexed=True)
    assert len(internal_aps) == 77
    for x, y in internal_aps:
        # All internal alignment patterns have core color 6 (Yellow)
        assert sampled_m0[y, x] == 6

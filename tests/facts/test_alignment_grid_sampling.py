"""Fact test for Phase V22: Multi-Version Alignment Pattern Grid Sampler.

Verifies:
- Table 5 alignment pattern coordinates match ISO/IEC 23634:2022 for all versions 1 <= V <= 32.
- Alignment pattern counts match ISO standard (0 for V1-5, 77 for V32).
- Mesh-based module sampling on Version 10 and Version 32 multi-symbol images yields 100% accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    # SHA-256 of the complete C encoder.h jab_ap_pos table as big-endian uint32.
    digest = hashlib.sha256(np.asarray(JAB_AP_POS, dtype=">u4").tobytes()).hexdigest()
    assert digest == "e82f86f77508a06f4393b139cdff55ef025c68d26e6b3eadf748ef5c376bd271"
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
    assert bgr is not None
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
    assert bgr is not None
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
    # The approved V32 sidecar omits its matrix; derive the exact raster oracle
    # independently from the known 12-pixel module pitch of this clean fixture.
    centers = rgb[1746:3480:12, 6:1740:12]
    expected = ((centers[:, :, 0] > 127) * 4 + (centers[:, :, 1] > 127) * 2 + (centers[:, :, 2] > 127)).astype(np.uint8)
    assert np.array_equal(sampled_m0, expected)

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


@pytest.mark.parametrize("version", [10, 32])
def test_mesh_sampling_preserves_every_module_under_perspective(version):
    """A projective capture retains every module, including outer edges."""
    side = 4 * version + 17
    matrix = np.random.default_rng(42).integers(0, 8, (side, side), dtype=np.uint8)
    palette = np.array([[r, g, b] for r in (0, 255) for g in (0, 255) for b in (0, 255)], dtype=np.uint8)
    image = np.repeat(np.repeat(palette[matrix], 8, axis=0), 8, axis=1)
    edge = side * 8
    source = np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], dtype=np.float32)
    corners = np.array([[60, 20], [edge - 70, 70], [edge + 40, edge], [10, edge - 30]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(image, transform, (edge + 100, edge + 100), flags=cv2.INTER_NEAREST)
    assert np.array_equal(sample_symbol_mesh(warped, corners, version, version), matrix)


@pytest.mark.parametrize("version", [10, 32])
@pytest.mark.parametrize("distortion", ["curved", "radial"])
def test_mesh_refines_alignment_patterns_on_curved_capture(version: int, distortion: str) -> None:
    name = "asan_multi2" if version == 10 else "multi_block_2_v32"
    bgr = cv2.imread(f"tests/fixtures/approved/jabcode/{name}.png")
    assert bgr is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    side = 4 * version + 17
    edge = side * 12
    image = rgb[edge : 2 * edge, :edge]
    expected_rgb = image[6::12, 6::12]
    expected = (
        (expected_rgb[..., 0] > 127) * 4 + (expected_rgb[..., 1] > 127) * 2 + (expected_rgb[..., 2] > 127)
    ).astype(np.uint8)
    yy, xx = np.indices((edge, edge), dtype=np.float32)
    displacement = 10 * np.sin(np.pi * xx / edge) * np.sin(np.pi * yy / edge)
    dx, dy = displacement, displacement * 0.6
    if distortion == "radial":
        rx, ry = 2 * xx / edge - 1, 2 * yy / edge - 1
        radial = 10 * (rx**2 + ry**2 - 2)
        dx, dy = rx * radial, ry * radial
    warped = cv2.remap(image, xx - dx, yy - dy, cv2.INTER_NEAREST)
    corners = np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], dtype=np.float32)
    assert np.array_equal(sample_symbol_mesh(warped, corners, version, version), expected)

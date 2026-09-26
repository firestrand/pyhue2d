"""Fact test for Phase V27: Photographed / perspective unwarping for multi-symbol docked topologies."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

import pyhue2d
from pyhue2d.jabcode.color_palette import ColorPalette
from pyhue2d.jabcode.multisymbol_encoder import build_multisymbol_matrix


def _warp_barcode_into_camera_frame(
    img_rgb: np.ndarray,
    homography_shifts: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    border_modules: int = 4,
    module_size: int = 12,
) -> Image.Image:
    """Simulate a printed barcode captured by a camera at an angle with paper border and illumination gradient."""
    h, w = img_rgb.shape[:2]
    border_px = border_modules * module_size
    bordered = np.full((h + 2 * border_px, w + 2 * border_px, 3), 255, dtype=np.uint8)
    bordered[border_px : border_px + h, border_px : border_px + w] = img_rgb

    bw, bh = bordered.shape[1], bordered.shape[0]
    src_corners = np.float32(
        [
            [border_px, border_px],
            [border_px + w, border_px],
            [border_px + w, border_px + h],
            [border_px, border_px + h],
        ]
    )

    (s0x, s0y), (s1x, s1y), (s2x, s2y), (s3x, s3y) = homography_shifts
    dst_corners = np.float32(
        [
            [80 + s0x, 70 + s0y],
            [80 + w + s1x, 70 + s1y],
            [80 + w + s2x, 70 + h + s2y],
            [80 + s3x, 70 + h + s3y],
        ]
    )

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    canvas_w = bw + 160
    canvas_h = bh + 140
    warped = cv2.warpPerspective(bordered, M, (canvas_w, canvas_h), borderValue=(248, 248, 248))

    # Add subtle illumination gradient
    y_grad = np.linspace(1.0, 0.95, canvas_h)[:, None, None]
    x_grad = np.linspace(0.96, 1.0, canvas_w)[None, :, None]
    gradient = y_grad * x_grad
    warped = np.clip(warped.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    return Image.fromarray(warped)


def test_camera_multisymbol_horizontal_docked_2():
    """Verify photographed 2-symbol horizontally docked barcode unwarps and decodes cleanly."""
    payload = b"Camera 2-symbol docked horizontal test."
    mat = build_multisymbol_matrix(payload, version=1, colors=8, ecc_level=3, mask_pattern=7, symbol_count=2)
    palette = np.array(ColorPalette(8).to_rgb_array(), dtype=np.uint8)
    img_rgb = np.repeat(np.repeat(palette[mat], 12, axis=0), 12, axis=1)

    camera_image = _warp_barcode_into_camera_frame(
        img_rgb,
        homography_shifts=((5, 8), (-15, -4), (12, 18), (-8, -6)),
    )

    result = pyhue2d.decode(camera_image)
    assert result is not None
    assert result.symbol_count == 2
    assert result.payload == payload


def test_camera_multisymbol_vertical_docked_2():
    """Verify photographed 2-symbol vertically docked barcode unwarps and decodes cleanly."""
    payload = b"Camera 2-symbol docked vertical test."
    mat = build_multisymbol_matrix(payload, version=1, colors=8, ecc_level=3, mask_pattern=7, symbol_count=2, columns=1)
    palette = np.array(ColorPalette(8).to_rgb_array(), dtype=np.uint8)
    img_rgb = np.repeat(np.repeat(palette[mat], 12, axis=0), 12, axis=1)

    camera_image = _warp_barcode_into_camera_frame(
        img_rgb,
        homography_shifts=((10, 6), (15, -8), (-6, 14), (-12, -10)),
    )

    result = pyhue2d.decode(camera_image)
    assert result is not None
    assert result.symbol_count == 2
    assert result.payload == payload


def test_camera_multisymbol_horizontal_docked_3():
    """Verify photographed 3-symbol horizontally docked barcode unwarps and decodes cleanly."""
    payload = b"Camera 3-symbol horizontal docking barcode capture with perspective distortion."
    mat = build_multisymbol_matrix(payload, version=1, colors=8, ecc_level=3, mask_pattern=7, symbol_count=3, columns=3)
    palette = np.array(ColorPalette(8).to_rgb_array(), dtype=np.uint8)
    img_rgb = np.repeat(np.repeat(palette[mat], 12, axis=0), 12, axis=1)

    camera_image = _warp_barcode_into_camera_frame(
        img_rgb,
        homography_shifts=((8, 5), (-12, -6), (10, 15), (-5, -8)),
    )

    result = pyhue2d.decode(camera_image)
    assert result is not None
    assert result.symbol_count == 3
    assert result.payload == payload

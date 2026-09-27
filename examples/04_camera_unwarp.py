"""Camera frame perspective unwarping example.

This example demonstrates:
- Encoding multi-symbol docked barcodes using the high-level pyhue2d.encode API.
- Synthetically applying camera capture distortions: paper margins, 4-corner perspective
  homography warping, and subtle illumination gradients.
- Automatic quadrilateral contour detection and aspect-ratio-aware unwarping by PyHue2D.
- Decoding docked topologies (horizontal and vertical) directly from distorted frames.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import pyhue2d  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def warp_into_camera_frame(
    img_rgb: np.ndarray,
    homography_shifts: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    border_modules: int = 4,
    module_size: int = 12,
) -> Image.Image:
    """Simulate a printed barcode captured at an angle with paper border and illumination gradient."""
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

    m_mat = cv2.getPerspectiveTransform(src_corners, dst_corners)
    canvas_w = bw + 160
    canvas_h = bh + 140
    warped = cv2.warpPerspective(bordered, m_mat, (canvas_w, canvas_h), borderValue=(248, 248, 248))

    # Add realistic illumination gradient across camera frame
    y_grad = np.linspace(1.0, 0.95, canvas_h)[:, None, None]
    x_grad = np.linspace(0.96, 1.0, canvas_w)[None, :, None]
    gradient = y_grad * x_grad
    warped_rgb = np.clip(warped.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    return Image.fromarray(warped_rgb)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== PyHue2D Camera Frame Perspective Unwarping ===\n")

    # -------------------------------------------------------------------------
    # 1. Horizontal Multi-Symbol Synthetic Camera Capture
    # -------------------------------------------------------------------------
    payload_h = b"Camera 3-symbol horizontal docking barcode capture with perspective distortion."
    out_h = OUTPUT_DIR / "04_camera_horizontal_docked_3.png"

    print("1. Simulating 3-symbol horizontal docked camera capture...")
    img_h = pyhue2d.encode(
        payload_h,
        version=1,
        colors=8,
        ecc_level=3,
        mask_pattern=7,
        symbol_count=3,
        columns=3,
    )
    photo_h = warp_into_camera_frame(
        np.array(img_h),
        homography_shifts=((8, 5), (-12, -6), (10, 15), (-5, -8)),
    )
    photo_h.save(out_h)
    print(f"   Saved: {out_h.name} ({photo_h.width}x{photo_h.height} px)")

    dec_h = pyhue2d.decode(photo_h)
    print(f"   Decoded payload: '{dec_h.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_h.symbol_count}, Corrected errors: {dec_h.corrected_error_count}\n")
    assert dec_h.payload == payload_h

    # -------------------------------------------------------------------------
    # 2. Vertical Multi-Symbol Synthetic Camera Capture
    # -------------------------------------------------------------------------
    payload_v = b"Camera 2-symbol docked vertical test."
    out_v = OUTPUT_DIR / "04_camera_vertical_docked_2.png"

    print("2. Simulating 2-symbol vertical docked camera capture...")
    img_v = pyhue2d.encode(
        payload_v,
        version=1,
        colors=8,
        ecc_level=3,
        mask_pattern=7,
        symbol_count=2,
        columns=1,
    )
    photo_v = warp_into_camera_frame(
        np.array(img_v),
        homography_shifts=((10, 6), (15, -8), (-6, 14), (-12, -10)),
    )
    photo_v.save(out_v)
    print(f"   Saved: {out_v.name} ({photo_v.width}x{photo_v.height} px)")

    dec_v = pyhue2d.decode(photo_v)
    print(f"   Decoded payload: '{dec_v.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_v.symbol_count}, Corrected errors: {dec_v.corrected_error_count}\n")
    assert dec_v.payload == payload_v

    print("Success: Camera frame perspective unwarping verified!")


if __name__ == "__main__":
    main()

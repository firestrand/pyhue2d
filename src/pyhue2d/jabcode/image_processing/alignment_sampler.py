"""Multi-version alignment pattern grid sampler conforming to ISO/IEC 23634:2022.

Implements ISO/IEC 23634 Table 5 alignment pattern coordinates and mesh-based
grid sampling for Version 1 through Version 32 symbols.
"""

from __future__ import annotations

import cv2
import numpy as np

# ISO/IEC 23634:2022 Table 5 - Positions of finder/alignment patterns (Side versions 1-32)
# Row index corresponds to version - 1. Coordinates are 1-indexed module positions.
JAB_AP_POS: tuple[tuple[int, ...], ...] = (
    (4, 18, 0, 0, 0, 0, 0, 0, 0),
    (4, 22, 0, 0, 0, 0, 0, 0, 0),
    (4, 26, 0, 0, 0, 0, 0, 0, 0),
    (4, 30, 0, 0, 0, 0, 0, 0, 0),
    (4, 34, 0, 0, 0, 0, 0, 0, 0),
    (4, 17, 38, 0, 0, 0, 0, 0, 0),
    (4, 20, 42, 0, 0, 0, 0, 0, 0),
    (4, 23, 46, 0, 0, 0, 0, 0, 0),
    (4, 26, 50, 0, 0, 0, 0, 0, 0),
    (4, 14, 32, 54, 0, 0, 0, 0, 0),
    (4, 17, 39, 58, 0, 0, 0, 0, 0),
    (4, 20, 46, 62, 0, 0, 0, 0, 0),
    (4, 23, 44, 66, 0, 0, 0, 0, 0),
    (4, 26, 37, 51, 70, 0, 0, 0, 0),
    (4, 14, 36, 58, 74, 0, 0, 0, 0),
    (4, 17, 39, 56, 78, 0, 0, 0, 0),
    (4, 20, 42, 63, 82, 0, 0, 0, 0),
    (4, 23, 38, 54, 70, 86, 0, 0, 0),
    (4, 26, 38, 56, 77, 90, 0, 0, 0),
    (4, 14, 33, 53, 72, 94, 0, 0, 0),
    (4, 17, 38, 59, 79, 98, 0, 0, 0),
    (4, 20, 36, 53, 70, 86, 102, 0, 0),
    (4, 23, 36, 55, 74, 93, 106, 0, 0),
    (4, 26, 36, 58, 79, 100, 110, 0, 0),
    (4, 14, 36, 58, 80, 92, 114, 0, 0),
    (4, 17, 34, 52, 70, 88, 99, 118, 0),
    (4, 20, 37, 54, 72, 89, 106, 122, 0),
    (4, 23, 38, 56, 74, 92, 113, 126, 0),
    (4, 26, 36, 58, 78, 98, 120, 130, 0),
    (4, 14, 32, 49, 67, 84, 102, 112, 134),
    (4, 17, 35, 53, 71, 89, 107, 119, 138),
    (4, 20, 38, 55, 73, 91, 108, 126, 142),
)

# Number of finder and alignment patterns along a side (Side versions 1-32)
JAB_AP_NUM: tuple[int, ...] = (
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
)


def get_ap_positions(version: int, zero_indexed: bool = True) -> list[int]:
    """Get the alignment and finder pattern center positions along one axis for a version.

    Args:
        version: Side version (1 to 32).
        zero_indexed: If True, returns 0-indexed module indices (subtracting 1).
            If False, returns 1-indexed module positions directly matching ISO Table 5.

    Returns:
        List of integer module positions along that axis.
    """
    if version < 1 or version > 32:
        raise ValueError(f"Version must be between 1 and 32, got {version}")
    idx = version - 1
    num = JAB_AP_NUM[idx]
    coords = list(JAB_AP_POS[idx][:num])
    if zero_indexed:
        return [c - 1 for c in coords]
    return coords


def get_all_grid_anchors(version_x: int, version_y: int, zero_indexed: bool = True) -> tuple[list[int], list[int]]:
    """Get pattern center coordinates along the X and Y axes for a symbol.

    Args:
        version_x: Horizontal side version (1 to 32).
        version_y: Vertical side version (1 to 32).
        zero_indexed: Whether coordinates are 0-indexed.

    Returns:
        Tuple of (x_positions, y_positions).
    """
    return (
        get_ap_positions(version_x, zero_indexed=zero_indexed),
        get_ap_positions(version_y, zero_indexed=zero_indexed),
    )


def alignment_pattern_coords(version_x: int, version_y: int, zero_indexed: bool = True) -> list[tuple[int, int]]:
    """Return coordinates of all internal alignment patterns for a symbol.

    Excludes the four outer corner finder/primary patterns:
    (TL, TR, BR, BL).

    Args:
        version_x: Horizontal side version (1 to 32).
        version_y: Vertical side version (1 to 32).
        zero_indexed: Whether coordinates are 0-indexed.

    Returns:
        List of (x, y) module coordinates for internal alignment patterns.
    """
    xs, ys = get_all_grid_anchors(version_x, version_y, zero_indexed=zero_indexed)
    nx, ny = len(xs), len(ys)
    coords: list[tuple[int, int]] = []

    for i in range(ny):
        for j in range(nx):
            # Skip the 4 corners (finder patterns in master, primary APs in slave)
            is_corner = (
                (i == 0 and j == 0)
                or (i == 0 and j == nx - 1)
                or (i == ny - 1 and j == nx - 1)
                or (i == ny - 1 and j == 0)
            )
            if not is_corner:
                coords.append((xs[j], ys[i]))

    return coords


def _refine_alignment_anchors(image: np.ndarray, module_anchors: np.ndarray, transform: np.ndarray) -> np.ndarray:
    anchors = cv2.perspectiveTransform(module_anchors.reshape(1, -1, 2), transform).reshape(module_anchors.shape)
    yellow = ((image[..., 0] > 160) & (image[..., 1] > 160) & (image[..., 2] < 100)).astype(np.uint8)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(yellow, connectivity=4)
    refined = anchors.copy()
    offsets = np.array([[0, -1], [-1, 0], [1, 0], [0, 1], [-1, -1], [1, 1], [-1, 1], [1, -1]], dtype=np.float32)
    matches = 0
    rows, cols = anchors.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if row in (0, rows - 1) and col in (0, cols - 1):
                continue
            center = module_anchors[row, col]
            projected = cv2.perspectiveTransform((center + offsets).reshape(1, -1, 2), transform)[0]
            vectors = projected - anchors[row, col]
            pitch = float(np.linalg.norm(vectors[1]))
            distance = np.linalg.norm(centroids - anchors[row, col], axis=1)
            candidates = np.flatnonzero(
                (distance < 2 * pitch)
                & (stats[:, cv2.CC_STAT_AREA] > 0.4 * pitch**2)
                & (stats[:, cv2.CC_STAT_AREA] < 1.8 * pitch**2)
            )
            for candidate in candidates[np.argsort(distance[candidates])]:
                points = np.rint(centroids[candidate] + vectors).astype(int)
                if (
                    np.any(points < 0)
                    or np.any(points[:, 0] >= image.shape[1])
                    or np.any(points[:, 1] >= image.shape[0])
                ):
                    continue
                colors = image[points[:, 1], points[:, 0]]
                cyan = (colors[:, 0] < 100) & (colors[:, 1] > 160) & (colors[:, 2] > 160)
                if np.all(cyan[:4]) and (np.all(cyan[4:6]) or np.all(cyan[6:])):
                    refined[row, col] = centroids[candidate]
                    matches += 1
                    break
    if matches < (rows * cols - 4) / 2:
        return anchors
    for row, col in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        center = anchors[row, col]
        neighbor = cv2.perspectiveTransform(
            (module_anchors[row, col] + np.array([[1, 0]], dtype=np.float32)).reshape(1, 1, 2), transform
        )[0, 0]
        pitch = float(np.linalg.norm(neighbor - center))
        x0, y0 = np.maximum(np.floor(center - 2 * pitch).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(center + 2 * pitch).astype(int), [image.shape[1], image.shape[0]])
        patch = image[y0:y1, x0:x1]
        cx, cy = np.rint(center - [x0, y0]).astype(int)
        if not (0 <= cy < patch.shape[0] and 0 <= cx < patch.shape[1]):
            continue
        mask = np.all((patch > 127) == (patch[cy, cx] > 127), axis=-1).astype(np.uint8)
        _, labels, component_stats, component_centers = cv2.connectedComponentsWithStats(mask, connectivity=4)
        label = labels[cy, cx]
        if 0.4 * pitch**2 < component_stats[label, cv2.CC_STAT_AREA] < 1.8 * pitch**2:
            refined[row, col] = component_centers[label] + [x0, y0]
    return refined


def sample_symbol_mesh_rgb(
    image: np.ndarray,
    corners: np.ndarray,
    version_x: int,
    version_y: int,
) -> np.ndarray:
    """Sample raw RGB module colors across an alignment pattern mesh.

    Refines projected anchors against APX patterns and samples local homographies.

    Args:
        image: Source RGB image array of shape (H, W, 3).
        corners: 4 corner coordinates in image space: [TL, TR, BR, BL],
            each as (x, y) float coordinates.
        version_x: Horizontal side version (1 to 32).
        version_y: Vertical side version (1 to 32).

    Returns:
        Sampled RGB array of shape (height, width, 3) where height = 4*Vy + 17
        and width = 4*Vx + 17.
    """
    width = 4 * version_x + 17
    height = 4 * version_y + 17

    corners_f32 = np.asarray(corners, dtype=np.float32)
    if corners_f32.shape != (4, 2):
        raise ValueError(f"Corners must have shape (4, 2), got {corners_f32.shape}")

    # For small versions (1 to 5), there are no internal alignment patterns
    if version_x <= 5 and version_y <= 5:
        dst = np.array(
            [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(corners_f32, dst)
        H_inv = np.linalg.inv(H)

        xs = np.arange(width) + 0.5
        ys = np.arange(height) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        pts = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x.ravel())], axis=0)
        img_pts = H_inv @ pts
        img_pts /= img_pts[2:3]

        map_x = img_pts[0].reshape((height, width)).astype(np.float32)
        map_y = img_pts[1].reshape((height, width)).astype(np.float32)
        sampled_bgr = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        return sampled_bgr

    # For versions >= 6, sample using the multi-anchor alignment mesh
    xs_ap = get_ap_positions(version_x, zero_indexed=True)
    ys_ap = get_ap_positions(version_y, zero_indexed=True)
    nx, ny = len(xs_ap), len(ys_ap)

    # Project module anchors through the same homography as the outer corners.
    # Bilinear corner interpolation does not preserve projective perspective.
    module_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(module_corners, corners_f32)
    anchor_x, anchor_y = np.meshgrid(np.asarray(xs_ap) + 0.5, np.asarray(ys_ap) + 0.5)
    module_anchors = np.stack([anchor_x, anchor_y], axis=-1).astype(np.float32)
    anchors = _refine_alignment_anchors(image, module_anchors, transform)

    sampled = np.zeros((height, width, 3), dtype=np.uint8)

    # Boundaries of sub-blocks: [0, ap1, ap2, ..., width]
    x_bounds = [0] + [xs_ap[j] for j in range(1, nx - 1)] + [width]
    y_bounds = [0] + [ys_ap[i] for i in range(1, ny - 1)] + [height]

    for i in range(ny - 1):
        y0, y1 = y_bounds[i], y_bounds[i + 1]
        for j in range(nx - 1):
            x0, x1 = x_bounds[j], x_bounds[j + 1]

            # Module centers for this local sub-block
            local_xs = np.arange(x0, x1) + 0.5
            local_ys = np.arange(y0, y1) + 0.5
            lx_grid, ly_grid = np.meshgrid(local_xs, local_ys)

            # Local 4 anchor quad in module space
            mod_quad = np.array(
                [
                    [float(xs_ap[j]) + 0.5, float(ys_ap[i]) + 0.5],
                    [float(xs_ap[j + 1]) + 0.5, float(ys_ap[i]) + 0.5],
                    [float(xs_ap[j + 1]) + 0.5, float(ys_ap[i + 1]) + 0.5],
                    [float(xs_ap[j]) + 0.5, float(ys_ap[i + 1]) + 0.5],
                ],
                dtype=np.float32,
            )

            # Corresponding 4 anchor quad in image space
            img_quad = np.array(
                [
                    anchors[i, j],
                    anchors[i, j + 1],
                    anchors[i + 1, j + 1],
                    anchors[i + 1, j],
                ],
                dtype=np.float32,
            )

            H_local = cv2.getPerspectiveTransform(mod_quad, img_quad)
            local_pts = np.stack([lx_grid.ravel(), ly_grid.ravel(), np.ones_like(lx_grid.ravel())], axis=0)
            mapped_pts = H_local @ local_pts
            mapped_pts /= mapped_pts[2:3]

            map_x = mapped_pts[0].reshape((y1 - y0, x1 - x0)).astype(np.float32)
            map_y = mapped_pts[1].reshape((y1 - y0, x1 - x0)).astype(np.float32)
            sub_sampled = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST)
            sampled[y0:y1, x0:x1] = sub_sampled

    return sampled


def sample_symbol_mesh(
    image: np.ndarray,
    corners: np.ndarray,
    version_x: int,
    version_y: int,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Sample and quantize a symbol module matrix across an alignment pattern mesh.

    Args:
        image: Source RGB image array of shape (H, W, 3).
        corners: 4 corner coordinates [TL, TR, BR, BL].
        version_x: Horizontal side version (1 to 32).
        version_y: Vertical side version (1 to 32).
        palette: Optional palette array of shape (C, 3). If omitted, standard
            8-color JAB Code RGB palette is used.

    Returns:
        Quantized module matrix of shape (height, width) with values in 0..C-1.
    """
    if palette is None:
        palette = np.array(
            [
                [0, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
                [0, 255, 255],
                [255, 0, 0],
                [255, 0, 255],
                [255, 255, 0],
                [255, 255, 255],
            ],
            dtype=np.float32,
        )
    else:
        palette = np.asarray(palette, dtype=np.float32)

    sampled_rgb = sample_symbol_mesh_rgb(image, corners, version_x, version_y).astype(np.float32)

    diff = sampled_rgb[:, :, None, :] - palette[None, None, :, :]
    dist = np.sum(diff**2, axis=-1)
    return np.argmin(dist, axis=-1).astype(np.uint8)

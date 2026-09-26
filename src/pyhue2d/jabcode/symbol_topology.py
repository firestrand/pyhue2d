"""Finder-driven raster geometry and channel-driven docking traversal."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .exceptions import JABCodeError
from .symbol_channel import DecodedSymbolChannel, decode_symbol_channel


@dataclass(frozen=True, slots=True)
class MasterGrid:
    """Oriented raster module grid and detected master rectangle."""

    grid: NDArray[np.uint8]
    x: int
    y: int
    width: int
    height: int

    @property
    def matrix(self) -> NDArray[np.uint8]:
        return self.grid[self.y : self.y + self.height, self.x : self.x + self.width]


def detect_master_grid(image: Image.Image) -> MasterGrid | None:
    """Locate a master from its four standard finder patterns."""
    pixels = np.asarray(image.convert("RGB"))
    x_axis = _module_axis(pixels[np.linspace(0, pixels.shape[0] - 1, 32, dtype=int)])
    y_axis = _module_axis(pixels[:, np.linspace(0, pixels.shape[1] - 1, 32, dtype=int)].transpose(1, 0, 2))
    if x_axis is None or y_axis is None:
        return None
    rgb = pixels[np.ix_(y_axis, x_axis)]
    grid = (
        (rgb[..., 0] > 127).astype(np.uint8) * 4
        + (rgb[..., 1] > 127).astype(np.uint8) * 2
        + (rgb[..., 2] > 127).astype(np.uint8)
    ).astype(np.uint8)
    for rotation in range(4):
        oriented = np.rot90(grid, rotation)
        geometry = _find_master(oriented)
        if geometry is not None:
            return geometry
    return None


def _module_axis(lines: NDArray[np.uint8]) -> NDArray[np.int64] | None:
    boundaries = [np.flatnonzero(np.any(line[1:] != line[:-1], axis=1)) + 1 for line in lines]
    lengths = [np.diff(points) for points in boundaries if len(points) > 1]
    if not lengths:
        return None
    pitch = int(np.argmax(np.bincount(np.concatenate(lengths))))
    phase = int(np.argmax(np.bincount(np.concatenate(boundaries) % pitch)))
    return np.arange(phase + pitch // 2, lines.shape[1], pitch, dtype=np.int64)


def _finder_centers(grid: NDArray[np.uint8], kind: int) -> set[tuple[int, int]]:
    if min(grid.shape) < 5:
        return set()
    matches = np.ones((grid.shape[0] - 4, grid.shape[1] - 4), dtype=bool)
    core, ring = ((0, 3), (0, 6), (6, 0), (3, 0))[kind]
    vertical = 1 if kind < 2 else -1
    for i in range(3):
        for j in range(i + 1):
            value = core if i % 2 == 0 else ring
            for sign in (-1, 1):
                dy, dx = 2 + sign * i * vertical, 2 + sign * j
                matches &= grid[dy : dy + matches.shape[0], dx : dx + matches.shape[1]] == value
    return {(int(x) + 2, int(y) + 2) for y, x in np.argwhere(matches)}


def _find_master(grid: NDArray[np.uint8]) -> MasterGrid | None:
    top_left, top_right, bottom_right, bottom_left = [_finder_centers(grid, kind) for kind in range(4)]
    for x, y in sorted(top_left):
        for right, top in sorted(top_right):
            width = right - x + 7
            if top != y or not 21 <= width <= 145 or (width - 17) % 4:
                continue
            for left, bottom in sorted(bottom_left):
                height = bottom - y + 7
                if left != x or not 21 <= height <= 145 or (height - 17) % 4:
                    continue
                if (right, bottom) in bottom_right and x >= 3 and y >= 3:
                    return MasterGrid(grid, x - 3, y - 3, width, height)
    return None


def decode_symbol_topology(
    image: Image.Image, error_correction: bool = True
) -> tuple[DecodedSymbolChannel, ...] | None:
    """Decode symbols breadth-first, following decoded channel docking flags."""
    master = detect_master_grid(image)
    if master is None:
        return None
    decoded = [decode_symbol_channel(master.matrix.tolist(), error_correction=error_correction)]
    rectangles = [(master.x, master.y, master.width, master.height)]
    for channel, (x, y, width, height) in zip(decoded, rectangles):
        for direction, metadata in channel.docks:
            child_width, child_height = metadata.version_x * 4 + 17, metadata.version_y * 4 + 17
            child_x, child_y = ((x, y - child_height), (x, y + height), (x - child_width, y), (x + width, y))[direction]
            rectangle = (child_x, child_y, child_width, child_height)
            if (
                child_x < 0
                or child_y < 0
                or child_x + child_width > master.grid.shape[1]
                or child_y + child_height > master.grid.shape[0]
            ):
                raise JABCodeError("Docked symbol extends outside the image")
            if any(
                child_x < rx + rw and rx < child_x + child_width and child_y < ry + rh and ry < child_y + child_height
                for rx, ry, rw, rh in rectangles
            ):
                raise JABCodeError("Docked symbol overlaps an already decoded symbol")
            matrix = master.grid[child_y : child_y + child_height, child_x : child_x + child_width]
            decoded.append(
                decode_symbol_channel(matrix.tolist(), metadata, direction ^ 1, error_correction=error_correction)
            )
            rectangles.append(rectangle)
    return tuple(decoded)

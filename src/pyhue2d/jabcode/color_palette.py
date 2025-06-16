"""Color palette management for JABCode implementation."""

import math
from typing import List, Optional, Tuple

import numpy as np

from .constants import SUPPORTED_COLOR_COUNTS, get_color_palette


class ColorPalette:
    """Manages color palettes for JABCode symbols."""

    def __init__(
        self,
        color_count: Optional[int] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
    ):
        """Initialize ColorPalette.

        Args:
            color_count: Number of colors (4, 8, 16, 32, 64, 128, 256)
            colors: Custom list of RGB tuples, overrides color_count
        """
        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("Colors must be a list of RGB tuples")
            self.colors = colors.copy()
            self.color_count = len(colors)
            self._validate_custom_colors(colors)
        else:
            if color_count is None:
                color_count = 8  # Default
            self.color_count = color_count
            self.colors = get_color_palette(color_count)
            self._validate_color_count(color_count)

    def _validate_color_count(self, color_count: int) -> None:
        """Validate color count is supported."""
        if color_count not in SUPPORTED_COLOR_COUNTS:
            raise ValueError(f"Color count must be one of {SUPPORTED_COLOR_COUNTS}")

    def _validate_custom_colors(self, colors: List[Tuple[int, int, int]]) -> None:
        """Validate custom colors list."""
        if not isinstance(colors, list):
            raise ValueError("Colors must be a list of RGB tuples")

        for color in colors:
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError("Each color must be an RGB tuple with 3 values")

            if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
                raise ValueError("RGB values must be between 0 and 255")

    def get_color(self, index: int) -> Tuple[int, int, int]:
        """Get color by index."""
        if not (0 <= index < self.color_count):
            raise IndexError(f"Color index out of range: {index}")
        return self.colors[index]

    def get_index(self, color: Tuple[int, int, int]) -> int:
        """Get index of color in palette."""
        try:
            return self.colors.index(color)
        except ValueError:
            raise ValueError("Color not found in palette")

    def find_closest_color(self, target_color: Tuple[int, int, int]) -> int:
        """Find index of closest color in palette."""
        min_distance = float("inf")
        closest_index = 0

        for i, palette_color in enumerate(self.colors):
            distance = self.get_color_distance(target_color, palette_color)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def get_color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two colors."""
        r_diff = color1[0] - color2[0]
        g_diff = color1[1] - color2[1]
        b_diff = color1[2] - color2[2]
        return math.sqrt(r_diff * r_diff + g_diff * g_diff + b_diff * b_diff)

    def is_high_contrast(self, min_distance_threshold: float = 100.0) -> bool:
        """Check if palette has good contrast (minimum distance between colors)."""
        if self.color_count < 2:
            return True

        min_distance = float("inf")
        for i in range(self.color_count):
            for j in range(i + 1, self.color_count):
                distance = self.get_color_distance(self.colors[i], self.colors[j])
                min_distance = min(min_distance, distance)

        return min_distance >= min_distance_threshold

    def to_rgb_array(self) -> np.ndarray:
        """Convert palette to numpy RGB array."""
        return np.array(self.colors, dtype=np.uint8)

    def copy(self) -> "ColorPalette":
        """Create a copy of the color palette."""
        return ColorPalette(colors=self.colors.copy())

    def __str__(self) -> str:
        """String representation of ColorPalette."""
        return f"ColorPalette({self.color_count} colors)"

    def __repr__(self) -> str:
        """Detailed representation of ColorPalette."""
        return f"ColorPalette(color_count={self.color_count}, colors={self.colors})"

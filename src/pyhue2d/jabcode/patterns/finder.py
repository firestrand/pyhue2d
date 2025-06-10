"""Finder pattern generation for JABCode symbols."""

import numpy as np
from ..constants import FinderPatternType


class FinderPatternGenerator:
    """Generates finder patterns for JABCode symbols.

    Finder patterns are used for symbol detection and orientation.
    JABCode uses 4 finder patterns: FP0 (top-left), FP1 (top-right),
    FP2 (bottom-left), FP3 (bottom-right).
    """

    def __init__(self):
        """Initialize finder pattern generator."""
        # Define the standard finder pattern templates
        # These are based on JABCode specification patterns
        self._pattern_templates = {
            FinderPatternType.FP0: self._create_fp0_template(),
            FinderPatternType.FP1: self._create_fp1_template(),
            FinderPatternType.FP2: self._create_fp2_template(),
            FinderPatternType.FP3: self._create_fp3_template(),
        }

        # Standard size for finder patterns
        self._standard_size = 7

    def generate_pattern(self, pattern_type: int, size: int = 7) -> np.ndarray:
        """Generate a finder pattern matrix.

        Args:
            pattern_type: Type of finder pattern (FP0, FP1, FP2, FP3)
            size: Size of the pattern matrix (default 7x7)

        Returns:
            numpy array representing the pattern

        Raises:
            ValueError: If pattern type is invalid
        """
        if not self.validate_pattern_type(pattern_type):
            raise ValueError(f"Invalid pattern type: {pattern_type}")

        if size <= 0 or size % 2 == 0:
            raise ValueError(f"Pattern size must be positive and odd: {size}")

        # Get the base template
        template = self._pattern_templates[pattern_type].copy()

        # If requested size matches template size, return as-is
        if size == template.shape[0]:
            return template

        # For different sizes, scale the pattern
        return self._scale_pattern(template, size)

    def get_pattern_size(self, pattern_type: int) -> int:
        """Get the standard size for a finder pattern type.

        Args:
            pattern_type: Type of finder pattern

        Returns:
            Standard size for the pattern

        Raises:
            ValueError: If pattern type is invalid
        """
        if not self.validate_pattern_type(pattern_type):
            raise ValueError(f"Invalid pattern type: {pattern_type}")

        return self._standard_size

    def validate_pattern_type(self, pattern_type: int) -> bool:
        """Validate if pattern type is supported.

        Args:
            pattern_type: Pattern type to validate

        Returns:
            True if pattern type is valid
        """
        return pattern_type in [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]

    def _create_fp0_template(self) -> np.ndarray:
        """Create FP0 (top-left) pattern template."""
        # FP0 pattern: distinctive L-shaped pattern for top-left corner
        pattern = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_fp1_template(self) -> np.ndarray:
        """Create FP1 (top-right) pattern template."""
        # FP1 pattern: mirrored version of FP0 for top-right corner
        pattern = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 1, 1],  # Different bottom row for identification
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_fp2_template(self) -> np.ndarray:
        """Create FP2 (bottom-left) pattern template."""
        # FP2 pattern: rotated version for bottom-left corner
        pattern = np.array(
            [
                [1, 1, 1, 0, 1, 1, 1],  # Different top row for identification
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_fp3_template(self) -> np.ndarray:
        """Create FP3 (bottom-right) pattern template."""
        # FP3 pattern: unique pattern for bottom-right corner
        pattern = np.array(
            [
                [1, 1, 1, 0, 1, 1, 1],  # Different top row for identification
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 1, 1],  # Different bottom row for identification
            ],
            dtype=np.uint8,
        )
        return pattern

    def _scale_pattern(self, pattern: np.ndarray, new_size: int) -> np.ndarray:
        """Scale a pattern to a new size.

        Args:
            pattern: Original pattern to scale
            new_size: New size (must be odd)

        Returns:
            Scaled pattern
        """
        original_size = pattern.shape[0]

        if new_size == original_size:
            return pattern.copy()

        # For simplicity, use nearest neighbor scaling
        # In a production implementation, this might use more sophisticated scaling
        scale_factor = new_size / original_size

        new_pattern = np.zeros((new_size, new_size), dtype=np.uint8)

        for i in range(new_size):
            for j in range(new_size):
                orig_i = int(i / scale_factor)
                orig_j = int(j / scale_factor)

                # Clamp to original bounds
                orig_i = min(orig_i, original_size - 1)
                orig_j = min(orig_j, original_size - 1)

                new_pattern[i, j] = pattern[orig_i, orig_j]

        return new_pattern

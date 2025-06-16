"""Alignment pattern generation for JABCode symbols."""

from typing import List, Tuple

import numpy as np

from ..constants import MAX_SYMBOL_VERSIONS, AlignmentPatternType


class AlignmentPatternGenerator:
    """Generates alignment patterns for JABCode symbols.

    Alignment patterns help with geometric distortion correction.
    JABCode uses 5 alignment patterns: AP0, AP1, AP2, AP3, and APX (extended).
    """

    def __init__(self):
        """Initialize alignment pattern generator."""
        # Define the standard alignment pattern templates
        self._pattern_templates = {
            AlignmentPatternType.AP0: self._create_ap0_template(),
            AlignmentPatternType.AP1: self._create_ap1_template(),
            AlignmentPatternType.AP2: self._create_ap2_template(),
            AlignmentPatternType.AP3: self._create_ap3_template(),
            AlignmentPatternType.APX: self._create_apx_template(),
        }

        # Standard size for alignment patterns
        self._standard_size = 5

        # Alignment pattern positions for different symbol versions
        # These are simplified positions based on JABCode specification
        self._position_tables = self._create_position_tables()

    def generate_pattern(self, pattern_type: int, size: int = 5) -> np.ndarray:
        """Generate an alignment pattern matrix.

        Args:
            pattern_type: Type of alignment pattern (AP0-AP3, APX)
            size: Size of the pattern matrix (default 5x5)

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

    def get_pattern_positions(self, symbol_version: int) -> List[Tuple[int, int]]:
        """Get positions where alignment patterns should be placed.

        Args:
            symbol_version: Version of the symbol

        Returns:
            List of (x, y) positions for alignment patterns

        Raises:
            ValueError: If symbol version is invalid
        """
        if symbol_version < 1 or symbol_version > MAX_SYMBOL_VERSIONS:
            raise ValueError(f"Invalid symbol version: {symbol_version}")

        return self._position_tables.get(symbol_version, [])

    def validate_pattern_type(self, pattern_type: int) -> bool:
        """Validate if alignment pattern type is supported.

        Args:
            pattern_type: Pattern type to validate

        Returns:
            True if pattern type is valid
        """
        return pattern_type in [
            AlignmentPatternType.AP0,
            AlignmentPatternType.AP1,
            AlignmentPatternType.AP2,
            AlignmentPatternType.AP3,
            AlignmentPatternType.APX,
        ]

    def _create_ap0_template(self) -> np.ndarray:
        """Create AP0 alignment pattern template."""
        # AP0: Standard cross pattern for alignment
        pattern = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_ap1_template(self) -> np.ndarray:
        """Create AP1 alignment pattern template."""
        # AP1: Alternative pattern with different center
        pattern = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_ap2_template(self) -> np.ndarray:
        """Create AP2 alignment pattern template."""
        # AP2: Diagonal emphasis pattern
        pattern = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_ap3_template(self) -> np.ndarray:
        """Create AP3 alignment pattern template."""
        # AP3: Corner emphasis pattern
        pattern = np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        return pattern

    def _create_apx_template(self) -> np.ndarray:
        """Create APX (extended) alignment pattern template."""
        # APX: Extended pattern for larger symbols
        pattern = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 1, 1, 0],
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

        # Use nearest neighbor scaling
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

    def _create_position_tables(self) -> dict:
        """Create position tables for alignment patterns by symbol version.

        Returns:
            Dictionary mapping symbol version to list of positions
        """
        positions: dict[int, list[tuple[int, int]]] = {}

        # Simplified position tables for JABCode symbols
        # In real implementation, these would be based on the full specification

        # Small symbols (versions 1-5) - fewer alignment patterns
        for version in range(1, 6):
            matrix_size = 21 + (version - 1) * 4  # Simplified size calculation
            center = matrix_size // 2

            if version == 1:
                # Version 1: no alignment patterns
                positions[version] = []
            elif version == 2:
                # Version 2: single center pattern
                positions[version] = [(center, center)]
            else:
                # Versions 3-5: center and corners
                edge_offset = 6
                positions[version] = [
                    (center, center),  # Center
                    (edge_offset, edge_offset),  # Top-left
                    (matrix_size - edge_offset - 1, edge_offset),  # Top-right
                    (edge_offset, matrix_size - edge_offset - 1),  # Bottom-left
                    (
                        matrix_size - edge_offset - 1,
                        matrix_size - edge_offset - 1,
                    ),  # Bottom-right
                ]

        # Medium symbols (versions 6-15) - more alignment patterns
        for version in range(6, 16):
            matrix_size = 21 + (version - 1) * 4
            center = matrix_size // 2
            edge_offset = 6
            mid_offset = matrix_size // 3

            positions[version] = [
                (center, center),  # Center
                (edge_offset, edge_offset),  # Corners
                (matrix_size - edge_offset - 1, edge_offset),
                (edge_offset, matrix_size - edge_offset - 1),
                (matrix_size - edge_offset - 1, matrix_size - edge_offset - 1),
                (mid_offset, mid_offset),  # Mid positions
                (matrix_size - mid_offset - 1, mid_offset),
                (mid_offset, matrix_size - mid_offset - 1),
                (matrix_size - mid_offset - 1, matrix_size - mid_offset - 1),
            ]

        # Large symbols (versions 16+) - dense alignment patterns
        for version in range(16, MAX_SYMBOL_VERSIONS + 1):
            matrix_size = 21 + (version - 1) * 4

            # Grid-based positioning for large symbols
            alignment_positions = []
            step = max(6, matrix_size // 8)  # Adaptive step size

            for x in range(step, matrix_size - step, step):
                for y in range(step, matrix_size - step, step):
                    alignment_positions.append((x, y))

            positions[version] = alignment_positions

        return positions

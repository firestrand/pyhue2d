"""Symbol version calculation for JABCode symbols."""

from typing import Tuple, Dict
from .constants import (
    MAX_SYMBOL_VERSIONS,
    SUPPORTED_COLOR_COUNTS,
    get_bits_per_module,
    validate_color_count,
    validate_ecc_level,
    validate_symbol_version,
)


class SymbolVersionCalculator:
    """Calculates appropriate symbol version for given data requirements.

    JABCode symbols have different versions (1-32) with varying matrix sizes
    and data capacities. This class helps determine the optimal version
    for given data and error correction requirements.
    """

    def __init__(self):
        """Initialize symbol version calculator."""
        # Create lookup tables for efficient calculation
        self._matrix_size_table = self._create_matrix_size_table()
        self._capacity_tables = self._create_capacity_tables()

        # ECC overhead factors (simplified)
        self._ecc_overhead = {
            "L": 0.15,  # 15% overhead for low ECC
            "M": 0.25,  # 25% overhead for medium ECC
            "Q": 0.35,  # 35% overhead for quartile ECC
            "H": 0.50,  # 50% overhead for high ECC
        }

    def calculate_version(
        self, data_size: int, color_count: int, ecc_level: str
    ) -> int:
        """Calculate minimum symbol version for given requirements.

        Args:
            data_size: Size of data to encode in bytes
            color_count: Number of colors in palette
            ecc_level: Error correction level ("L", "M", "Q", "H")

        Returns:
            Minimum symbol version that can accommodate the data

        Raises:
            ValueError: If inputs are invalid or data cannot fit in any version
        """
        # Validate inputs
        if data_size < 0:
            raise ValueError(f"Data size must be non-negative: {data_size}")

        if not validate_color_count(color_count):
            raise ValueError(f"Invalid color count: {color_count}")

        if not validate_ecc_level(ecc_level):
            raise ValueError(f"Invalid ECC level: {ecc_level}")

        # Special case: empty data
        if data_size == 0:
            return 1

        # Find minimum version that can accommodate the data
        for version in range(1, MAX_SYMBOL_VERSIONS + 1):
            capacity = self.get_data_capacity(version, color_count, ecc_level)
            if capacity >= data_size:
                return version

        # If no version can accommodate the data
        raise ValueError(
            f"Data size {data_size} bytes too large for maximum symbol version"
        )

    def get_matrix_size(self, version: int) -> Tuple[int, int]:
        """Get matrix dimensions for symbol version.

        Args:
            version: Symbol version (1-32)

        Returns:
            Tuple of (width, height) in modules

        Raises:
            ValueError: If version is invalid
        """
        if not validate_symbol_version(version):
            raise ValueError(f"Invalid symbol version: {version}")

        return self._matrix_size_table[version]

    def get_data_capacity(self, version: int, color_count: int, ecc_level: str) -> int:
        """Get data capacity for symbol configuration.

        Args:
            version: Symbol version
            color_count: Number of colors
            ecc_level: Error correction level

        Returns:
            Data capacity in bytes

        Raises:
            ValueError: If inputs are invalid
        """
        if not validate_symbol_version(version):
            raise ValueError(f"Invalid symbol version: {version}")

        if not validate_color_count(color_count):
            raise ValueError(f"Invalid color count: {color_count}")

        if not validate_ecc_level(ecc_level):
            raise ValueError(f"Invalid ECC level: {ecc_level}")

        # Get base capacity and apply ECC overhead
        base_capacity = self._capacity_tables[version][color_count]
        ecc_overhead = self._ecc_overhead[ecc_level]

        # Reduce capacity by ECC overhead
        actual_capacity = int(base_capacity * (1.0 - ecc_overhead))

        return max(1, actual_capacity)  # Ensure at least 1 byte capacity

    def validate_version(self, version: int) -> bool:
        """Validate if symbol version is supported.

        Args:
            version: Symbol version to validate

        Returns:
            True if version is valid
        """
        return validate_symbol_version(version)

    def _create_matrix_size_table(self) -> Dict[int, Tuple[int, int]]:
        """Create matrix size lookup table for all versions.

        Returns:
            Dictionary mapping version to (width, height) tuples
        """
        size_table = {}

        for version in range(1, MAX_SYMBOL_VERSIONS + 1):
            # JABCode matrix size formula: base size + version increment
            # Starting at 21x21 for version 1, increasing by 4 for each version
            size = 21 + (version - 1) * 4
            size_table[version] = (size, size)

        return size_table

    def _create_capacity_tables(self) -> Dict[int, Dict[int, int]]:
        """Create data capacity lookup tables for all configurations.

        Returns:
            Dictionary mapping version to color_count to capacity
        """
        capacity_tables: Dict[int, Dict[int, int]] = {}

        for version in range(1, MAX_SYMBOL_VERSIONS + 1):
            capacity_tables[version] = {}

            # Get matrix size for this version
            matrix_size = self._matrix_size_table[version][0]  # Square matrices
            total_modules = matrix_size * matrix_size

            # Reserve space for finder patterns, alignment patterns, etc.
            # Simplified calculation - in reality this would be more complex
            finder_pattern_modules = 4 * (7 * 7)  # 4 finder patterns
            alignment_pattern_overhead = min(50, version * 5)  # Simplified
            metadata_overhead = 20  # Simplified metadata overhead

            usable_modules = (
                total_modules
                - finder_pattern_modules
                - alignment_pattern_overhead
                - metadata_overhead
            )
            usable_modules = max(usable_modules, 10)  # Ensure minimum

            for color_count in SUPPORTED_COLOR_COUNTS:
                bits_per_module = get_bits_per_module(color_count)
                total_bits = usable_modules * bits_per_module
                total_bytes = total_bits // 8

                # Ensure reasonable minimum and maximum capacities
                capacity = max(5, min(total_bytes, 10000))
                capacity_tables[version][color_count] = capacity

        return capacity_tables

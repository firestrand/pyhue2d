"""Symbol version calculation for JABCode symbols."""

from typing import Dict, Tuple

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

        # Alignment pattern positions must be available before capacity calculations
        self._alignment_pattern_positions = self._create_alignment_pattern_positions_table()

        # ECC overhead factors (simplified) must be defined before capacity calculation
        self._ecc_overhead = {
            "L": 0.20,  # 20% overhead for low ECC
            "M": 0.35,  # 35% overhead for medium ECC
            "Q": 0.50,  # 50% overhead for quartile ECC
            "H": 0.80,  # 80% overhead for high ECC
        }

        # Capacity tables depend on all other lookups
        self._capacity_tables = self._create_capacity_tables()

    def calculate_version(self, data_size: int, color_count: int, ecc_level: str, is_master: bool = True) -> int:
        """Calculate minimum symbol version for given requirements.

        Args:
            data_size: Size of data to encode in bytes
            color_count: Number of colors in palette
            ecc_level: Error correction level ("L", "M", "Q", "H")
            is_master: Whether this is a master symbol (default True)

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
            capacity = self.get_data_capacity(version, color_count, ecc_level, is_master)
            if capacity >= data_size:
                return version

        # If no version can accommodate the data
        raise ValueError(f"Data size {data_size} bytes too large for maximum symbol version")

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

    def get_data_capacity(self, version: int, color_count: int, ecc_level: str, is_master: bool = True) -> int:
        """Get data capacity for symbol configuration.

        Args:
            version: Symbol version
            color_count: Number of colors
            ecc_level: Error correction level
            is_master: Whether this is a master symbol (default True)

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
        base_capacity = self._get_raw_capacity(version, color_count, is_master)
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

    def _get_raw_capacity(self, version: int, color_count: int, is_master: bool = True) -> int:
        """Calculate raw data capacity in bytes before ECC."""
        if not validate_symbol_version(version):
            raise ValueError(f"Invalid symbol version: {version}")

        matrix_size = self._matrix_size_table[version][0]
        total_modules = matrix_size * matrix_size

        # Overhead from finder patterns
        finder_pattern_modules = 4 * 17 if is_master else 4 * 7

        # Overhead from alignment patterns
        ap_positions = self._alignment_pattern_positions.get(version, [])
        alignment_pattern_modules = (len(ap_positions) - 4) * 7 if version > 1 else 0

        # Overhead from color palette modules
        # C-code: enc->color_number > 64 ? (64-2)*2 : (enc->color_number-2)*2;
        color_palette_number = 2  # from encoder.h COLOR_PALETTE_NUMBER
        if color_count > 64:
            palette_modules = (64 - 2) * color_palette_number
        elif color_count > 2:
            palette_modules = (color_count - 2) * color_palette_number
        else:
            palette_modules = 0

        # Overhead from metadata (simplified for now)
        metadata_modules = 20 if is_master else 5

        # Calculate usable modules
        usable_modules = total_modules - finder_pattern_modules - alignment_pattern_modules - palette_modules - metadata_modules
        usable_modules = max(0, usable_modules)

        # Calculate raw bit capacity
        bits_per_module = get_bits_per_module(color_count)
        total_bits = usable_modules * bits_per_module
        return total_bits // 8

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
        DEPRECATED: Use get_data_capacity for dynamic calculation.
        This table is kept for now to avoid breaking other parts of the code
        but should be phased out.
        """
        capacity_tables: Dict[int, Dict[int, int]] = {}

        for version in range(1, MAX_SYMBOL_VERSIONS + 1):
            capacity_tables[version] = {}
            for color_count in SUPPORTED_COLOR_COUNTS:
                # Use a simplified calculation for the table
                try:
                    capacity = self.get_data_capacity(version, color_count, "M")
                except ValueError:
                    capacity = 0
                capacity_tables[version][color_count] = capacity

        return capacity_tables

    def _create_alignment_pattern_positions_table(self) -> Dict[int, list[int]]:
        """Create alignment pattern position lookup table, from C reference."""
        # This table is derived from `jab_ap_num` in the C implementation.
        # It represents the number of alignment patterns along one edge for a given version.
        jab_ap_num = [
            0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7
        ]
        
        ap_positions: Dict[int, list[int]] = {}
        for version in range(1, MAX_SYMBOL_VERSIONS + 1):
            num_aps = jab_ap_num[version - 1]
            if num_aps == 0:
                ap_positions[version] = []
                continue
            
            size = 21 + (version - 1) * 4
            # Simplified logic from C to determine positions
            positions = [6]  # First AP is always at index 6
            if num_aps > 1:
                last_pos = size - 7
                spacing = (last_pos - 6) / (num_aps - 1)
                for i in range(1, num_aps - 1):
                    positions.append(round(6 + i * spacing))
                positions.append(last_pos)
            ap_positions[version] = positions
            
        return ap_positions

"""Tests for SymbolVersionCalculator class."""

import pytest

from pyhue2d.jabcode.constants import ECC_LEVELS, MAX_SYMBOL_VERSIONS, SUPPORTED_COLOR_COUNTS
from pyhue2d.jabcode.version_calculator import SymbolVersionCalculator


class TestSymbolVersionCalculator:
    """Test suite for SymbolVersionCalculator class."""

    def test_symbol_version_calculator_creation(self):
        """Test that SymbolVersionCalculator can be created."""
        calculator = SymbolVersionCalculator()
        assert calculator is not None

    def test_calculate_version_small_data(self):
        """Test calculating version for small data."""
        calculator = SymbolVersionCalculator()
        version = calculator.calculate_version(10, 8, "L")
        assert isinstance(version, int)
        assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_medium_data(self):
        """Test calculating version for medium data."""
        calculator = SymbolVersionCalculator()
        version = calculator.calculate_version(100, 8, "M")
        assert isinstance(version, int)
        assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_large_data(self):
        """Test calculating version for large data."""
        calculator = SymbolVersionCalculator()
        version = calculator.calculate_version(1000, 16, "H")
        assert isinstance(version, int)
        assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_different_color_counts(self):
        """Test calculating version with different color counts."""
        calculator = SymbolVersionCalculator()
        for color_count in SUPPORTED_COLOR_COUNTS:
            version = calculator.calculate_version(50, color_count, "M")
            assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_different_ecc_levels(self):
        """Test calculating version with different ECC levels."""
        calculator = SymbolVersionCalculator()
        for ecc_level in ECC_LEVELS:
            version = calculator.calculate_version(50, 8, ecc_level)
            assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_invalid_color_count(self):
        """Test that invalid color count raises error."""
        calculator = SymbolVersionCalculator()
        with pytest.raises(ValueError):
            calculator.calculate_version(50, 7, "M")  # Invalid color count

    def test_calculate_version_invalid_ecc_level(self):
        """Test that invalid ECC level raises error."""
        calculator = SymbolVersionCalculator()
        with pytest.raises(ValueError):
            calculator.calculate_version(50, 8, "X")  # Invalid ECC level

    def test_get_matrix_size_version_1(self):
        """Test getting matrix size for version 1."""
        calculator = SymbolVersionCalculator()
        size = calculator.get_matrix_size(1)
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert size == (21, 21)  # Version 1 should be 21x21

    def test_get_matrix_size_version_10(self):
        """Test getting matrix size for version 10."""
        calculator = SymbolVersionCalculator()
        size = calculator.get_matrix_size(10)
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert size == (57, 57)  # Version 10: 21 + (10-1)*4 = 57

    def test_get_matrix_size_version_32(self):
        """Test getting matrix size for version 32."""
        calculator = SymbolVersionCalculator()
        size = calculator.get_matrix_size(32)
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert size == (145, 145)  # Version 32: 21 + (32-1)*4 = 145

    def test_get_matrix_size_invalid_version(self):
        """Test that invalid version raises error."""
        calculator = SymbolVersionCalculator()
        with pytest.raises(ValueError):
            calculator.get_matrix_size(0)  # Invalid version

    def test_get_data_capacity_basic(self):
        """Test getting data capacity for basic configuration."""
        calculator = SymbolVersionCalculator()
        capacity = calculator.get_data_capacity(1, 8, "L")
        assert isinstance(capacity, int)
        assert capacity > 0

    def test_get_data_capacity_all_configurations(self):
        """Test getting data capacity for various configurations."""
        calculator = SymbolVersionCalculator()
        for version in [1, 5, 10]:
            for color_count in [4, 8, 16]:
                for ecc_level in ["L", "M", "H"]:
                    capacity = calculator.get_data_capacity(version, color_count, ecc_level)
                    assert isinstance(capacity, int)
                    assert capacity > 0

    def test_validate_version_valid(self):
        """Test validation of valid versions."""
        calculator = SymbolVersionCalculator()
        for version in [1, 16, 32]:
            assert calculator.validate_version(version) is True

    def test_validate_version_invalid(self):
        """Test validation of invalid versions."""
        calculator = SymbolVersionCalculator()
        invalid_versions = [0, -1, 33, 100]
        for version in invalid_versions:
            assert calculator.validate_version(version) is False


class TestSymbolVersionCalculatorImplementation:
    """Test suite for SymbolVersionCalculator implementation details.

    These tests will pass once the implementation is complete.
    """

    @pytest.fixture
    def calculator(self):
        """Create a symbol version calculator for testing."""
        try:
            return SymbolVersionCalculator()
        except NotImplementedError:
            pytest.skip("SymbolVersionCalculator not yet implemented")

    def test_calculator_creation(self, calculator):
        """Test that calculator can be created successfully."""
        assert calculator is not None

    def test_calculate_version_returns_valid_version(self, calculator):
        """Test that calculated version is within valid range."""
        version = calculator.calculate_version(50, 8, "M")
        assert isinstance(version, int)
        assert 1 <= version <= MAX_SYMBOL_VERSIONS

    def test_calculate_version_more_data_needs_higher_version(self, calculator):
        """Test that more data requires higher versions."""
        small_version = calculator.calculate_version(10, 8, "L")
        large_version = calculator.calculate_version(500, 8, "L")

        assert small_version <= large_version

    def test_calculate_version_more_colors_allows_smaller_version(self, calculator):
        """Test that more colors can use smaller versions for same data."""
        version_4_colors = calculator.calculate_version(100, 4, "M")
        version_16_colors = calculator.calculate_version(100, 16, "M")

        # More colors should allow same data in smaller or equal version
        assert version_16_colors <= version_4_colors

    def test_calculate_version_higher_ecc_needs_higher_version(self, calculator):
        """Test that higher ECC levels require larger versions."""
        version_l = calculator.calculate_version(100, 8, "L")
        version_h = calculator.calculate_version(100, 8, "H")

        # Higher ECC should need larger or equal version
        assert version_h >= version_l

    def test_calculate_version_invalid_inputs_raise_errors(self, calculator):
        """Test that invalid inputs raise appropriate errors."""
        # Invalid color count
        with pytest.raises(ValueError):
            calculator.calculate_version(50, 7, "M")

        # Invalid ECC level
        with pytest.raises(ValueError):
            calculator.calculate_version(50, 8, "X")

        # Negative data size
        with pytest.raises(ValueError):
            calculator.calculate_version(-1, 8, "M")

    def test_get_matrix_size_returns_tuple(self, calculator):
        """Test that matrix size returns (width, height) tuple."""
        size = calculator.get_matrix_size(1)
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert isinstance(size[0], int)
        assert isinstance(size[1], int)
        assert size[0] > 0
        assert size[1] > 0

    def test_get_matrix_size_increasing_versions(self, calculator):
        """Test that higher versions have larger matrix sizes."""
        size_1 = calculator.get_matrix_size(1)
        size_10 = calculator.get_matrix_size(10)
        size_20 = calculator.get_matrix_size(20)

        # Higher versions should have larger matrices
        assert size_1[0] <= size_10[0] <= size_20[0]
        assert size_1[1] <= size_10[1] <= size_20[1]

    def test_get_matrix_size_square_matrices(self, calculator):
        """Test that matrices are square."""
        for version in [1, 5, 10, 15, 20]:
            size = calculator.get_matrix_size(version)
            assert size[0] == size[1], f"Version {version} matrix should be square"

    def test_get_matrix_size_odd_dimensions(self, calculator):
        """Test that matrix dimensions are odd (typical for barcode symbols)."""
        for version in [1, 5, 10]:
            size = calculator.get_matrix_size(version)
            assert size[0] % 2 == 1, f"Version {version} width should be odd"
            assert size[1] % 2 == 1, f"Version {version} height should be odd"

    def test_get_matrix_size_invalid_version_raises_error(self, calculator):
        """Test that invalid versions raise errors."""
        with pytest.raises(ValueError):
            calculator.get_matrix_size(0)

        with pytest.raises(ValueError):
            calculator.get_matrix_size(33)

        with pytest.raises(ValueError):
            calculator.get_matrix_size(-1)

    def test_get_data_capacity_returns_positive_int(self, calculator):
        """Test that data capacity returns positive integer."""
        capacity = calculator.get_data_capacity(1, 8, "L")
        assert isinstance(capacity, int)
        assert capacity > 0

    def test_get_data_capacity_higher_version_more_capacity(self, calculator):
        """Test that higher versions have more capacity."""
        capacity_1 = calculator.get_data_capacity(1, 8, "M")
        capacity_10 = calculator.get_data_capacity(10, 8, "M")

        assert capacity_10 > capacity_1

    def test_get_data_capacity_more_colors_more_capacity(self, calculator):
        """Test that more colors provide more capacity."""
        capacity_4 = calculator.get_data_capacity(5, 4, "M")
        capacity_16 = calculator.get_data_capacity(5, 16, "M")

        assert capacity_16 > capacity_4

    def test_get_data_capacity_higher_ecc_less_capacity(self, calculator):
        """Test that higher ECC levels reduce capacity."""
        capacity_l = calculator.get_data_capacity(5, 8, "L")
        capacity_h = calculator.get_data_capacity(5, 8, "H")

        assert capacity_l > capacity_h

    def test_get_data_capacity_invalid_inputs_raise_errors(self, calculator):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            calculator.get_data_capacity(0, 8, "M")  # Invalid version

        with pytest.raises(ValueError):
            calculator.get_data_capacity(5, 7, "M")  # Invalid color count

        with pytest.raises(ValueError):
            calculator.get_data_capacity(5, 8, "X")  # Invalid ECC level

    def test_validate_version_valid_versions_return_true(self, calculator):
        """Test that valid versions return True."""
        for version in [1, 16, 32]:
            assert calculator.validate_version(version) is True

    def test_validate_version_invalid_versions_return_false(self, calculator):
        """Test that invalid versions return False."""
        invalid_versions = [0, -1, 33, 100]
        for version in invalid_versions:
            assert calculator.validate_version(version) is False

    def test_version_calculation_consistency(self, calculator):
        """Test that version calculation is consistent with capacity."""
        # If we calculate a version for some data, that version should
        # have enough capacity for the data
        data_size = 50
        color_count = 8
        ecc_level = "M"

        version = calculator.calculate_version(data_size, color_count, ecc_level)
        capacity = calculator.get_data_capacity(version, color_count, ecc_level)

        assert capacity >= data_size, "Calculated version should have sufficient capacity"

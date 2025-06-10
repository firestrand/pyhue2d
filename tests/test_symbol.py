"""Tests for Symbol class."""

import pytest
from pyhue2d.jabcode.core import Symbol


class TestSymbol:
    """Test cases for Symbol class."""

    def test_symbol_creation_with_valid_parameters(self):
        """Test Symbol can be created with valid parameters."""
        symbol = Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 21))

        assert symbol.version == 1
        assert symbol.color_count == 8
        assert symbol.ecc_level == "M"
        assert symbol.matrix_size == (21, 21)
        assert symbol.finder_patterns is None
        assert symbol.alignment_patterns is None

    def test_symbol_creation_with_patterns(self):
        """Test Symbol can be created with finder and alignment patterns."""
        finder_patterns = [(0, 0), (20, 0), (0, 20)]
        alignment_patterns = [(10, 10)]

        symbol = Symbol(
            version=2,
            color_count=16,
            ecc_level="H",
            matrix_size=(25, 25),
            finder_patterns=finder_patterns,
            alignment_patterns=alignment_patterns,
        )

        assert symbol.finder_patterns == finder_patterns
        assert symbol.alignment_patterns == alignment_patterns

    def test_symbol_version_validation(self):
        """Test Symbol validates version parameter."""
        with pytest.raises(ValueError, match="Version must be between 1 and 32"):
            Symbol(version=0, color_count=8, ecc_level="M", matrix_size=(21, 21))

        with pytest.raises(ValueError, match="Version must be between 1 and 32"):
            Symbol(version=33, color_count=8, ecc_level="M", matrix_size=(21, 21))

    def test_symbol_color_count_validation(self):
        """Test Symbol validates color_count parameter."""
        valid_colors = [4, 8, 16, 32, 64, 128, 256]

        # Test valid color counts
        for color_count in valid_colors:
            symbol = Symbol(
                version=1, color_count=color_count, ecc_level="M", matrix_size=(21, 21)
            )
            assert symbol.color_count == color_count

        # Test invalid color count
        with pytest.raises(ValueError, match="Color count must be one of"):
            Symbol(version=1, color_count=7, ecc_level="M", matrix_size=(21, 21))

    def test_symbol_ecc_level_validation(self):
        """Test Symbol validates ECC level parameter."""
        valid_levels = ["L", "M", "Q", "H"]

        # Test valid ECC levels
        for ecc_level in valid_levels:
            symbol = Symbol(
                version=1, color_count=8, ecc_level=ecc_level, matrix_size=(21, 21)
            )
            assert symbol.ecc_level == ecc_level

        # Test invalid ECC level
        with pytest.raises(ValueError, match="ECC level must be one of"):
            Symbol(version=1, color_count=8, ecc_level="X", matrix_size=(21, 21))

    def test_symbol_matrix_size_validation(self):
        """Test Symbol validates matrix_size parameter."""
        # Test valid matrix size
        symbol = Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 21))
        assert symbol.matrix_size == (21, 21)

        # Test invalid matrix sizes
        with pytest.raises(
            ValueError, match="Matrix size must be a tuple of two positive integers"
        ):
            Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(0, 21))

        with pytest.raises(
            ValueError, match="Matrix size must be a tuple of two positive integers"
        ):
            Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 0))

    def test_symbol_calculate_capacity(self):
        """Test Symbol can calculate its data capacity."""
        symbol = Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 21))

        capacity = symbol.calculate_capacity()
        assert isinstance(capacity, int)
        assert capacity > 0

    def test_symbol_is_master_slave(self):
        """Test Symbol can determine if it's master or slave in cascade."""
        # Master symbol (has finder patterns)
        master_symbol = Symbol(
            version=1,
            color_count=8,
            ecc_level="M",
            matrix_size=(21, 21),
            finder_patterns=[(0, 0), (20, 0), (0, 20)],
        )

        assert master_symbol.is_master() is True
        assert master_symbol.is_slave() is False

        # Slave symbol (no finder patterns)
        slave_symbol = Symbol(
            version=1,
            color_count=8,
            ecc_level="M",
            matrix_size=(21, 21),
            finder_patterns=None,
        )

        assert slave_symbol.is_master() is False
        assert slave_symbol.is_slave() is True

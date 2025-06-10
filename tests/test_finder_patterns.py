"""Tests for FinderPatternGenerator class."""

import pytest
import numpy as np
from pyhue2d.jabcode.patterns.finder import FinderPatternGenerator
from pyhue2d.jabcode.constants import FinderPatternType


class TestFinderPatternGenerator:
    """Test suite for FinderPatternGenerator class."""

    def test_finder_pattern_generator_creation(self):
        """Test that FinderPatternGenerator can be created."""
        generator = FinderPatternGenerator()
        assert generator is not None

    def test_generate_pattern_fp0(self):
        """Test generating FP0 (top-left) finder pattern."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP0)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (7, 7)

    def test_generate_pattern_fp1(self):
        """Test generating FP1 (top-right) finder pattern."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP1)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (7, 7)

    def test_generate_pattern_fp2(self):
        """Test generating FP2 (bottom-left) finder pattern."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP2)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (7, 7)

    def test_generate_pattern_fp3(self):
        """Test generating FP3 (bottom-right) finder pattern."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP3)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (7, 7)

    def test_generate_pattern_default_size(self):
        """Test that default pattern size is 7x7."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP0)
        assert pattern.shape == (7, 7)

    def test_generate_pattern_custom_size(self):
        """Test generating pattern with custom size."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP0, size=9)
        assert pattern.shape == (9, 9)

    def test_generate_pattern_invalid_type(self):
        """Test that invalid pattern type raises error."""
        generator = FinderPatternGenerator()
        with pytest.raises(ValueError):
            generator.generate_pattern(999)  # Invalid pattern type

    def test_get_pattern_size_fp0(self):
        """Test getting standard size for FP0 pattern."""
        generator = FinderPatternGenerator()
        size = generator.get_pattern_size(FinderPatternType.FP0)
        assert size == 7

    def test_get_pattern_size_all_types(self):
        """Test getting sizes for all finder pattern types."""
        generator = FinderPatternGenerator()
        pattern_types = [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]
        for pattern_type in pattern_types:
            size = generator.get_pattern_size(pattern_type)
            assert size == 7

    def test_validate_pattern_type_valid(self):
        """Test validation of valid pattern types."""
        generator = FinderPatternGenerator()
        valid_types = [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]
        for pattern_type in valid_types:
            assert generator.validate_pattern_type(pattern_type) is True

    def test_validate_pattern_type_invalid(self):
        """Test validation of invalid pattern types."""
        generator = FinderPatternGenerator()
        invalid_types = [-1, 4, 10, 999]
        for pattern_type in invalid_types:
            assert generator.validate_pattern_type(pattern_type) is False


class TestFinderPatternGeneratorImplementation:
    """Test suite for FinderPatternGenerator implementation details.

    These tests will pass once the implementation is complete.
    """

    @pytest.fixture
    def generator(self):
        """Create a finder pattern generator for testing."""
        # This will work once implemented
        try:
            return FinderPatternGenerator()
        except NotImplementedError:
            pytest.skip("FinderPatternGenerator not yet implemented")

    def test_fp0_pattern_structure(self, generator):
        """Test that FP0 pattern has correct structure."""
        pattern = generator.generate_pattern(FinderPatternType.FP0)

        # Should be numpy array
        assert isinstance(pattern, np.ndarray)
        # Should be square
        assert pattern.shape[0] == pattern.shape[1]
        # Should be 7x7 by default
        assert pattern.shape == (7, 7)
        # Should contain only 0s and 1s
        assert np.all(np.isin(pattern, [0, 1]))

    def test_fp1_pattern_structure(self, generator):
        """Test that FP1 pattern has correct structure."""
        pattern = generator.generate_pattern(FinderPatternType.FP1)

        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == pattern.shape[1]
        assert pattern.shape == (7, 7)
        assert np.all(np.isin(pattern, [0, 1]))

    def test_different_patterns_unique(self, generator):
        """Test that different pattern types generate unique patterns."""
        fp0 = generator.generate_pattern(FinderPatternType.FP0)
        fp1 = generator.generate_pattern(FinderPatternType.FP1)
        fp2 = generator.generate_pattern(FinderPatternType.FP2)
        fp3 = generator.generate_pattern(FinderPatternType.FP3)

        # Patterns should be different
        assert not np.array_equal(fp0, fp1)
        assert not np.array_equal(fp0, fp2)
        assert not np.array_equal(fp0, fp3)
        assert not np.array_equal(fp1, fp2)
        assert not np.array_equal(fp1, fp3)
        assert not np.array_equal(fp2, fp3)

    def test_custom_size_patterns(self, generator):
        """Test generating patterns with custom sizes."""
        sizes = [5, 9, 11]
        for size in sizes:
            pattern = generator.generate_pattern(FinderPatternType.FP0, size=size)
            assert pattern.shape == (size, size)

    def test_pattern_size_getter(self, generator):
        """Test getting standard pattern sizes."""
        for pattern_type in [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]:
            size = generator.get_pattern_size(pattern_type)
            assert isinstance(size, int)
            assert size > 0
            assert size % 2 == 1  # Should be odd for symmetric patterns

    def test_pattern_validation_valid(self, generator):
        """Test validation returns True for valid pattern types."""
        valid_types = [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]
        for pattern_type in valid_types:
            assert generator.validate_pattern_type(pattern_type) is True

    def test_pattern_validation_invalid(self, generator):
        """Test validation returns False for invalid pattern types."""
        invalid_types = [-1, 4, 10, 999]
        for pattern_type in invalid_types:
            assert generator.validate_pattern_type(pattern_type) is False

    def test_pattern_deterministic(self, generator):
        """Test that pattern generation is deterministic."""
        pattern1 = generator.generate_pattern(FinderPatternType.FP0)
        pattern2 = generator.generate_pattern(FinderPatternType.FP0)

        # Same input should produce same output
        assert np.array_equal(pattern1, pattern2)

    def test_pattern_border_structure(self, generator):
        """Test that patterns have proper border structure."""
        pattern = generator.generate_pattern(FinderPatternType.FP0)

        # Top and bottom rows should be all 1s (black border)
        assert np.all(pattern[0, :] == 1)
        assert np.all(pattern[-1, :] == 1)

        # Left and right columns should be all 1s (black border)
        assert np.all(pattern[:, 0] == 1)
        assert np.all(pattern[:, -1] == 1)

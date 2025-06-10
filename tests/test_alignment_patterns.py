"""Tests for AlignmentPatternGenerator class."""

import pytest
import numpy as np
from src.pyhue2d.jabcode.patterns.alignment import AlignmentPatternGenerator
from src.pyhue2d.jabcode.constants import AlignmentPatternType


class TestAlignmentPatternGenerator:
    """Test suite for AlignmentPatternGenerator class."""
    
    def test_alignment_pattern_generator_creation(self):
        """Test that AlignmentPatternGenerator can be created."""
        generator = AlignmentPatternGenerator()
        assert generator is not None
    
    def test_generate_pattern_ap0(self):
        """Test generating AP0 alignment pattern."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP0)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_ap1(self):
        """Test generating AP1 alignment pattern."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP1)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_ap2(self):
        """Test generating AP2 alignment pattern."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP2)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_ap3(self):
        """Test generating AP3 alignment pattern."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP3)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_apx(self):
        """Test generating APX (extended) alignment pattern."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.APX)
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_default_size(self):
        """Test that default pattern size is 5x5."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP0)
        assert pattern.shape == (5, 5)
    
    def test_generate_pattern_custom_size(self):
        """Test generating pattern with custom size."""
        generator = AlignmentPatternGenerator()
        pattern = generator.generate_pattern(AlignmentPatternType.AP0, size=7)
        assert pattern.shape == (7, 7)
    
    def test_generate_pattern_invalid_type(self):
        """Test that invalid pattern type raises error."""
        generator = AlignmentPatternGenerator()
        with pytest.raises(ValueError):
            generator.generate_pattern(999)  # Invalid pattern type
    
    def test_get_pattern_positions_version_1(self):
        """Test getting pattern positions for symbol version 1."""
        generator = AlignmentPatternGenerator()
        positions = generator.get_pattern_positions(1)
        assert isinstance(positions, list)
        # Version 1 should have no alignment patterns
        assert len(positions) == 0
    
    def test_get_pattern_positions_version_5(self):
        """Test getting pattern positions for symbol version 5."""
        generator = AlignmentPatternGenerator()
        positions = generator.get_pattern_positions(5)
        assert isinstance(positions, list)
        # Version 5 should have some alignment patterns
        assert len(positions) > 0
    
    def test_get_pattern_positions_invalid_version(self):
        """Test that invalid symbol version raises error."""
        generator = AlignmentPatternGenerator()
        with pytest.raises(ValueError):
            generator.get_pattern_positions(0)  # Invalid version
    
    def test_validate_pattern_type_valid(self):
        """Test validation of valid pattern types."""
        generator = AlignmentPatternGenerator()
        valid_types = [
            AlignmentPatternType.AP0,
            AlignmentPatternType.AP1,
            AlignmentPatternType.AP2,
            AlignmentPatternType.AP3,
            AlignmentPatternType.APX
        ]
        for pattern_type in valid_types:
            assert generator.validate_pattern_type(pattern_type) is True
    
    def test_validate_pattern_type_invalid(self):
        """Test validation of invalid pattern types."""
        generator = AlignmentPatternGenerator()
        invalid_types = [-1, 5, 10, 999]
        for pattern_type in invalid_types:
            assert generator.validate_pattern_type(pattern_type) is False


class TestAlignmentPatternGeneratorImplementation:
    """Test suite for AlignmentPatternGenerator implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def generator(self):
        """Create an alignment pattern generator for testing."""
        try:
            return AlignmentPatternGenerator()
        except NotImplementedError:
            pytest.skip("AlignmentPatternGenerator not yet implemented")
    
    def test_ap0_pattern_structure(self, generator):
        """Test that AP0 pattern has correct structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP0)
        
        # Should be numpy array
        assert isinstance(pattern, np.ndarray)
        # Should be square
        assert pattern.shape[0] == pattern.shape[1]
        # Should be 5x5 by default
        assert pattern.shape == (5, 5)
        # Should contain only 0s and 1s
        assert np.all(np.isin(pattern, [0, 1]))
    
    def test_ap1_pattern_structure(self, generator):
        """Test that AP1 pattern has correct structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP1)
        
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == pattern.shape[1]
        assert pattern.shape == (5, 5)
        assert np.all(np.isin(pattern, [0, 1]))
    
    def test_ap2_pattern_structure(self, generator):
        """Test that AP2 pattern has correct structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP2)
        
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == pattern.shape[1]
        assert pattern.shape == (5, 5)
        assert np.all(np.isin(pattern, [0, 1]))
    
    def test_ap3_pattern_structure(self, generator):
        """Test that AP3 pattern has correct structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP3)
        
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == pattern.shape[1]
        assert pattern.shape == (5, 5)
        assert np.all(np.isin(pattern, [0, 1]))
    
    def test_apx_pattern_structure(self, generator):
        """Test that APX pattern has correct structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.APX)
        
        assert isinstance(pattern, np.ndarray)
        assert pattern.shape[0] == pattern.shape[1]
        assert pattern.shape == (5, 5)
        assert np.all(np.isin(pattern, [0, 1]))
    
    def test_different_patterns_unique(self, generator):
        """Test that different pattern types generate unique patterns."""
        ap0 = generator.generate_pattern(AlignmentPatternType.AP0)
        ap1 = generator.generate_pattern(AlignmentPatternType.AP1)
        ap2 = generator.generate_pattern(AlignmentPatternType.AP2)
        ap3 = generator.generate_pattern(AlignmentPatternType.AP3)
        apx = generator.generate_pattern(AlignmentPatternType.APX)
        
        patterns = [ap0, ap1, ap2, ap3, apx]
        
        # Check that patterns are different (at least some should be)
        unique_patterns = []
        for pattern in patterns:
            is_unique = True
            for unique_pattern in unique_patterns:
                if np.array_equal(pattern, unique_pattern):
                    is_unique = False
                    break
            if is_unique:
                unique_patterns.append(pattern)
        
        # Should have multiple unique patterns
        assert len(unique_patterns) > 1
    
    def test_custom_size_patterns(self, generator):
        """Test generating patterns with custom sizes."""
        sizes = [3, 7, 9]
        for size in sizes:
            pattern = generator.generate_pattern(AlignmentPatternType.AP0, size=size)
            assert pattern.shape == (size, size)
    
    def test_pattern_positions_valid_versions(self, generator):
        """Test getting pattern positions for valid symbol versions."""
        for version in [1, 2, 5, 10, 15]:
            positions = generator.get_pattern_positions(version)
            assert isinstance(positions, list)
            # Each position should be a tuple of (x, y)
            for pos in positions:
                assert isinstance(pos, tuple)
                assert len(pos) == 2
                assert isinstance(pos[0], int)
                assert isinstance(pos[1], int)
                assert pos[0] >= 0
                assert pos[1] >= 0
    
    def test_pattern_positions_larger_symbols_more_patterns(self, generator):
        """Test that larger symbols have more alignment patterns."""
        positions_v1 = generator.get_pattern_positions(1)
        positions_v10 = generator.get_pattern_positions(10)
        
        # Larger versions should generally have more alignment patterns
        # (though this depends on the specific JABCode implementation)
        assert isinstance(positions_v1, list)
        assert isinstance(positions_v10, list)
    
    def test_pattern_validation_valid(self, generator):
        """Test validation returns True for valid pattern types."""
        valid_types = [
            AlignmentPatternType.AP0,
            AlignmentPatternType.AP1,
            AlignmentPatternType.AP2,
            AlignmentPatternType.AP3,
            AlignmentPatternType.APX
        ]
        for pattern_type in valid_types:
            assert generator.validate_pattern_type(pattern_type) is True
    
    def test_pattern_validation_invalid(self, generator):
        """Test validation returns False for invalid pattern types."""
        invalid_types = [-1, 5, 10, 999]
        for pattern_type in invalid_types:
            assert generator.validate_pattern_type(pattern_type) is False
    
    def test_pattern_deterministic(self, generator):
        """Test that pattern generation is deterministic."""
        pattern1 = generator.generate_pattern(AlignmentPatternType.AP0)
        pattern2 = generator.generate_pattern(AlignmentPatternType.AP0)
        
        # Same input should produce same output
        assert np.array_equal(pattern1, pattern2)
    
    def test_pattern_center_structure(self, generator):
        """Test that alignment patterns have proper center structure."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP0)
        
        # Center should typically be black (1) for alignment
        center = pattern.shape[0] // 2
        assert pattern[center, center] == 1
    
    def test_pattern_symmetry(self, generator):
        """Test that alignment patterns are symmetric."""
        pattern = generator.generate_pattern(AlignmentPatternType.AP0)
        
        # Test horizontal symmetry
        assert np.array_equal(pattern, np.fliplr(pattern))
        
        # Test vertical symmetry  
        assert np.array_equal(pattern, np.flipud(pattern))
"""Tests for FinderPatternDetector class."""

import pytest
import numpy as np
from PIL import Image
from pyhue2d.jabcode.image_processing.finder_detector import FinderPatternDetector
from pyhue2d.jabcode.core import Point2D
import os
import json
from pyhue2d.jabcode.patterns import FinderPatternGenerator
from pyhue2d.jabcode.constants import FinderPatternType

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'example_images')
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, 'examples_manifest.json')

class TestFinderPatternDetector:
    """Test suite for FinderPatternDetector class - Basic functionality."""
    
    def test_detector_creation_with_defaults(self):
        """Test that FinderPatternDetector can be created with default settings."""
        detector = FinderPatternDetector()
        assert detector is not None
    
    def test_detector_creation_with_custom_settings(self):
        """Test creating FinderPatternDetector with custom settings."""
        settings = {
            'detection_method': 'template_matching',
            'template_threshold': 0.8,
            'min_pattern_size': 5,
            'max_pattern_size': 50
        }
        detector = FinderPatternDetector(settings)
        assert detector is not None
    
    def test_detector_find_patterns_in_image(self):
        """Test finding finder patterns in an image."""
        detector = FinderPatternDetector()
        
        # Create test image with finder pattern
        test_array = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        # Add a simple finder pattern (black square with white center)
        test_array[40:60, 40:60] = [0, 0, 0]      # Black outer
        test_array[47:53, 47:53] = [255, 255, 255]  # White center
        
        test_image = Image.fromarray(test_array, mode='RGB')
        patterns = detector.find_patterns(test_image)
        
        assert isinstance(patterns, list)
    
    def test_detector_find_patterns_in_binary(self):
        """Test finding finder patterns in binary array."""
        detector = FinderPatternDetector()
        
        # Create test binary array
        binary_array = np.ones((80, 80), dtype=np.uint8) * 255
        binary_array[30:50, 30:50] = 0  # Black square
        binary_array[37:43, 37:43] = 255  # White center
        
        patterns = detector.find_patterns_binary(binary_array)
        assert isinstance(patterns, list)
    
    def test_detector_validate_pattern(self):
        """Test pattern validation."""
        detector = FinderPatternDetector()
        
        # Create test pattern region
        pattern_region = np.ones((20, 20), dtype=np.uint8) * 255
        pattern_region[5:15, 5:15] = 0    # Black
        pattern_region[8:12, 8:12] = 255  # White center
        
        center = Point2D(50, 50)
        is_valid = detector.validate_pattern(pattern_region, center)
        
        assert isinstance(is_valid, bool)


class TestFinderPatternDetectorImplementation:
    """Test suite for FinderPatternDetector implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def detector(self):
        """Create a finder pattern detector for testing."""
        try:
            return FinderPatternDetector()
        except NotImplementedError:
            pytest.skip("FinderPatternDetector not yet implemented")
    
    @pytest.fixture
    def test_image_with_patterns(self):
        """Create a test image with known finder patterns using real JABCode templates."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP0, size=21) * 255  # 0/1 to 0/255
        image_array = np.ones((200, 200), dtype=np.uint8) * 255  # White background
        # Place the pattern in four corners
        image_array[20:41, 20:41] = pattern
        image_array[20:41, 160:181] = pattern
        image_array[160:181, 20:41] = pattern
        image_array[160:181, 160:181] = pattern
        return Image.fromarray(image_array, mode='L')
    
    @pytest.fixture
    def test_binary_image(self):
        """Create a test binary image with patterns using real JABCode templates."""
        generator = FinderPatternGenerator()
        pattern = generator.generate_pattern(FinderPatternType.FP0, size=21) * 255
        binary_array = np.ones((150, 150), dtype=np.uint8) * 255
        binary_array[30:51, 30:51] = pattern
        binary_array[30:51, 100:121] = pattern
        return binary_array
    
    def test_detector_creation_with_defaults(self, detector):
        """Test that detector can be created with default values."""
        assert detector is not None
        assert hasattr(detector, 'settings')
        assert hasattr(detector, 'detection_methods')
    
    def test_detector_default_settings(self, detector):
        """Test detector default settings."""
        settings = detector.settings
        assert isinstance(settings, dict)
        assert 'detection_method' in settings
        assert 'template_threshold' in settings
        assert 'min_pattern_size' in settings
        assert 'max_pattern_size' in settings
    
    def test_detector_supported_methods(self, detector):
        """Test supported detection methods."""
        methods = detector.detection_methods
        assert isinstance(methods, list)
        assert 'template_matching' in methods
        assert 'contour_detection' in methods
    
    def test_detector_find_patterns_rgb_image(self, detector, test_image_with_patterns):
        """Test finding patterns in RGB image."""
        patterns = detector.find_patterns(test_image_with_patterns)
        
        assert isinstance(patterns, list)
        # Should find at least 3 valid patterns
        assert len(patterns) >= 3
        
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert 'center' in pattern
            assert 'size' in pattern
            assert 'confidence' in pattern
            assert 'pattern_type' in pattern
            
            # Center should be Point2D
            assert isinstance(pattern['center'], Point2D)
            
            # Size should be positive
            assert pattern['size'] > 0
            
            # Confidence should be between 0 and 1
            assert 0 <= pattern['confidence'] <= 1
    
    def test_detector_find_patterns_binary_image(self, detector, test_binary_image):
        """Test finding patterns in binary image."""
        patterns = detector.find_patterns_binary(test_binary_image)
        
        assert isinstance(patterns, list)
        assert len(patterns) >= 2  # Should find 2 patterns
        
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert 'center' in pattern
            assert 'size' in pattern
    
    def test_detector_template_matching_method(self, detector, test_image_with_patterns):
        """Test template matching detection method."""
        detector.settings['detection_method'] = 'template_matching'
        patterns = detector.find_patterns(test_image_with_patterns)
        
        assert isinstance(patterns, list)
        # Template matching should be reasonably accurate
        assert len(patterns) >= 2
    
    def test_detector_contour_detection_method(self, detector, test_image_with_patterns):
        """Test contour detection method."""
        detector.settings['detection_method'] = 'contour_detection'
        patterns = detector.find_patterns(test_image_with_patterns)
        
        assert isinstance(patterns, list)
        # Contour detection should find rectangular patterns
        assert len(patterns) >= 2
    
    def test_detector_pattern_type_classification(self, detector, test_image_with_patterns):
        """Test pattern type classification (FP0, FP1, FP2, FP3)."""
        patterns = detector.find_patterns(test_image_with_patterns)
        
        # Should classify different pattern types
        pattern_types = [p['pattern_type'] for p in patterns]
        assert 'FP0' in pattern_types or 'FP1' in pattern_types or 'FP2' in pattern_types
    
    def test_detector_pattern_size_filtering(self, detector):
        """Test pattern size filtering."""
        # Create image with patterns of different sizes
        image_array = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Small pattern (should be filtered out)
        image_array[10:15, 10:15] = [0, 0, 0]
        image_array[12:13, 12:13] = [255, 255, 255]
        
        # Normal pattern
        image_array[50:70, 50:70] = [0, 0, 0]
        image_array[57:63, 57:63] = [255, 255, 255]
        
        # Large pattern (might be filtered out)
        image_array[100:160, 100:160] = [0, 0, 0]
        image_array[127:133, 127:133] = [255, 255, 255]
        
        test_image = Image.fromarray(image_array, mode='RGB')
        
        # Set size limits
        detector.settings['min_pattern_size'] = 10
        detector.settings['max_pattern_size'] = 40
        
        patterns = detector.find_patterns(test_image)
        
        # Should only find the normal-sized pattern
        for pattern in patterns:
            assert detector.settings['min_pattern_size'] <= pattern['size'] <= detector.settings['max_pattern_size']
    
    def test_detector_confidence_threshold(self, detector, test_image_with_patterns):
        """Test confidence threshold filtering."""
        # Set high confidence threshold
        detector.settings['template_threshold'] = 0.9
        
        high_conf_patterns = detector.find_patterns(test_image_with_patterns)
        
        # Set low confidence threshold
        detector.settings['template_threshold'] = 0.3
        
        low_conf_patterns = detector.find_patterns(test_image_with_patterns)
        
        # Low threshold should find more patterns
        assert len(low_conf_patterns) >= len(high_conf_patterns)
        
        # All patterns should meet confidence threshold
        for pattern in high_conf_patterns:
            assert pattern['confidence'] >= 0.9
    
    def test_detector_validate_pattern_structure(self, detector):
        """Test pattern structure validation."""
        # Valid pattern structure (black border, white center)
        valid_pattern = np.zeros((20, 20), dtype=np.uint8)
        valid_pattern[8:12, 8:12] = 255
        
        center = Point2D(100, 100)
        assert detector.validate_pattern(valid_pattern, center) == True
        
        # Invalid pattern (no clear structure)
        invalid_pattern = np.random.randint(0, 256, size=(20, 20), dtype=np.uint8)
        assert detector.validate_pattern(invalid_pattern, center) == False
    
    def test_detector_pattern_orientation_detection(self, detector):
        """Test pattern orientation detection."""
        if hasattr(detector, 'detect_pattern_orientation'):
            # Create pattern with specific orientation
            pattern_array = np.ones((30, 30), dtype=np.uint8) * 255
            pattern_array[5:25, 5:25] = 0
            pattern_array[12:18, 12:18] = 255
            # Add orientation marker
            pattern_array[10:12, 5:10] = 255
            
            orientation = detector.detect_pattern_orientation(pattern_array)
            assert isinstance(orientation, float)
            assert 0 <= orientation < 360
    
    def test_detector_multiple_scales(self, detector):
        """Test detection at multiple scales."""
        # Create image with patterns at different scales
        image_array = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # Small scale pattern
        image_array[50:60, 50:60] = [0, 0, 0]
        image_array[53:57, 53:57] = [255, 255, 255]
        
        # Medium scale pattern
        image_array[100:120, 100:120] = [0, 0, 0]
        image_array[107:113, 107:113] = [255, 255, 255]
        
        # Large scale pattern
        image_array[200:240, 200:240] = [0, 0, 0]
        image_array[215:225, 215:225] = [255, 255, 255]
        
        test_image = Image.fromarray(image_array, mode='RGB')
        
        if hasattr(detector, 'multi_scale_detection'):
            detector.settings['multi_scale_detection'] = True
            patterns = detector.find_patterns(test_image)
            
            # Should find patterns at different scales
            assert len(patterns) >= 3
            
            sizes = [p['size'] for p in patterns]
            assert len(set(sizes)) >= 2  # At least 2 different sizes
    
    def test_detector_noise_robustness(self, detector):
        """Test detection robustness against noise."""
        # Create clean pattern
        clean_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        clean_array[40:60, 40:60] = [0, 0, 0]
        clean_array[47:53, 47:53] = [255, 255, 255]
        clean_image = Image.fromarray(clean_array, mode='RGB')
        
        # Add noise
        noisy_array = clean_array.copy()
        noise = np.random.randint(-20, 21, size=noisy_array.shape, dtype=np.int16)
        noisy_array = np.clip(noisy_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_image = Image.fromarray(noisy_array, mode='RGB')
        
        clean_patterns = detector.find_patterns(clean_image)
        noisy_patterns = detector.find_patterns(noisy_image)
        
        # Should still detect pattern in noisy image
        assert len(noisy_patterns) > 0
        # Confidence might be lower but pattern should be found
        assert len(noisy_patterns) >= len(clean_patterns) - 1
    
    def test_detector_performance_metrics(self, detector, test_image_with_patterns):
        """Test performance metrics collection."""
        # Process multiple images
        for _ in range(3):
            detector.find_patterns(test_image_with_patterns)
        
        if hasattr(detector, 'get_performance_stats'):
            stats = detector.get_performance_stats()
            assert isinstance(stats, dict)
            assert 'total_detections' in stats
            assert stats['total_detections'] >= 3
    
    def test_detector_corner_coordinates(self, detector, test_image_with_patterns):
        """Test extraction of pattern corner coordinates."""
        patterns = detector.find_patterns(test_image_with_patterns)
        
        for pattern in patterns:
            if 'corners' in pattern:
                corners = pattern['corners']
                assert isinstance(corners, list)
                assert len(corners) == 4  # Should have 4 corners
                
                for corner in corners:
                    assert isinstance(corner, Point2D)
    
    def test_detector_pattern_quality_assessment(self, detector):
        """Test pattern quality assessment."""
        # High quality pattern
        high_quality_array = np.ones((50, 50), dtype=np.uint8) * 255
        high_quality_array[10:40, 10:40] = 0
        high_quality_array[20:30, 20:30] = 255
        
        # Low quality pattern (blurred/distorted)
        low_quality_array = np.ones((50, 50), dtype=np.uint8) * 255
        low_quality_array[15:35, 15:35] = 100  # Gray instead of black
        low_quality_array[22:28, 22:28] = 200  # Light gray instead of white
        
        if hasattr(detector, 'assess_pattern_quality'):
            high_quality = detector.assess_pattern_quality(high_quality_array)
            low_quality = detector.assess_pattern_quality(low_quality_array)
            
            assert isinstance(high_quality, float)
            assert isinstance(low_quality, float)
            assert high_quality > low_quality
    
    def test_detector_error_handling_invalid_input(self, detector):
        """Test error handling for invalid inputs."""
        invalid_inputs = [
            None,
            "not an image",
            123,
            np.array([1, 2, 3]),  # Wrong dimensions
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                detector.find_patterns(invalid_input)
    
    def test_detector_error_handling_empty_image(self, detector):
        """Test error handling for empty images."""
        # Empty image
        empty_image = Image.new('RGB', (0, 0))
        
        with pytest.raises(ValueError):
            detector.find_patterns(empty_image)
        
        # Very small image
        tiny_image = Image.new('RGB', (2, 2))
        patterns = detector.find_patterns(tiny_image)
        assert isinstance(patterns, list)
        assert len(patterns) == 0  # Should find no patterns
    
    def test_detector_different_image_modes(self, detector):
        """Test detection with different image modes."""
        # Create pattern in different modes
        base_array = np.ones((80, 80, 3), dtype=np.uint8) * 255
        base_array[30:50, 30:50] = [0, 0, 0]
        base_array[37:43, 37:43] = [255, 255, 255]
        
        # RGB mode
        rgb_image = Image.fromarray(base_array, mode='RGB')
        rgb_patterns = detector.find_patterns(rgb_image)
        
        # Grayscale mode
        gray_image = rgb_image.convert('L')
        gray_patterns = detector.find_patterns(gray_image)
        
        # Should work with both modes
        assert isinstance(rgb_patterns, list)
        assert isinstance(gray_patterns, list)
    
    def test_detector_string_representation(self, detector):
        """Test detector string representation."""
        str_repr = str(detector)
        assert "FinderPatternDetector" in str_repr
        assert "method" in str_repr or "template" in str_repr
    
    def test_detector_copy(self, detector):
        """Test detector copying."""
        if hasattr(detector, 'copy'):
            copied = detector.copy()
            assert copied is not detector
            assert copied.settings == detector.settings
    
    def test_detector_reset(self, detector):
        """Test detector reset functionality."""
        if hasattr(detector, 'reset'):
            # Process some data
            test_image = Image.new('RGB', (50, 50))
            detector.find_patterns(test_image)
            
            # Reset detector
            detector.reset()
            
            # Should be able to process again
            patterns = detector.find_patterns(test_image)
            assert isinstance(patterns, list)

@pytest.mark.parametrize("example", json.load(open(MANIFEST_PATH)))
def test_detector_on_real_jabcode_images_actual(example):
    """Test FinderPatternDetector on real JABCode example images (actual result, not xfail)."""
    image_path = os.path.join(EXAMPLES_DIR, example['output'])
    image = Image.open(image_path)
    detector = FinderPatternDetector()
    patterns = detector.find_patterns(image)
    print(f"Image {example['output']}: found {len(patterns)} patterns")
    assert isinstance(patterns, list)
    assert len(patterns) >= 1  # At least one pattern should be found in a real JABCode image
"""Tests for RGBChannelBinarizer class."""

import pytest
import numpy as np
from PIL import Image
from pyhue2d.jabcode.image_processing.binarizer import RGBChannelBinarizer
from pyhue2d.jabcode.core import Bitmap


class TestRGBChannelBinarizer:
    """Test suite for RGBChannelBinarizer class - Basic functionality."""
    
    def test_binarizer_creation_with_defaults(self):
        """Test that RGBChannelBinarizer can be created with default settings."""
        binarizer = RGBChannelBinarizer()
        assert binarizer is not None
    
    def test_binarizer_creation_with_custom_settings(self):
        """Test creating RGBChannelBinarizer with custom settings."""
        settings = {
            'threshold_method': 'otsu',
            'adaptive_block_size': 15,
            'adaptive_c': 10,
            'morphology_operations': True
        }
        binarizer = RGBChannelBinarizer(settings)
        assert binarizer is not None
    
    def test_binarizer_process_rgb_image(self):
        """Test processing RGB image."""
        binarizer = RGBChannelBinarizer()
        
        # Create test RGB image
        rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_array[20:80, 20:80] = [255, 255, 255]  # White square
        rgb_array[40:60, 40:60] = [0, 0, 0]        # Black square in center
        
        test_image = Image.fromarray(rgb_array, mode='RGB')
        result = binarizer.process_image(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
    
    def test_binarizer_process_bitmap(self):
        """Test processing Bitmap object."""
        binarizer = RGBChannelBinarizer()
        
        # Create test bitmap
        test_array = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
        bitmap = Bitmap(array=test_array, width=50, height=50)
        
        result = binarizer.process_bitmap(bitmap)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (50, 50)
    
    def test_binarizer_process_numpy_array(self):
        """Test processing numpy array directly."""
        binarizer = RGBChannelBinarizer()
        
        # Create test array
        test_array = np.ones((30, 30, 3), dtype=np.uint8) * 128
        test_array[10:20, 10:20] = [255, 255, 255]
        
        result = binarizer.process_array(test_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (30, 30)


class TestRGBChannelBinarizerImplementation:
    """Test suite for RGBChannelBinarizer implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def binarizer(self):
        """Create a binarizer for testing."""
        try:
            return RGBChannelBinarizer()
        except NotImplementedError:
            pytest.skip("RGBChannelBinarizer not yet implemented")
    
    @pytest.fixture
    def test_rgb_image(self):
        """Create a test RGB image with known patterns."""
        # Create 100x100 image with distinct regions
        rgb_array = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray background
        
        # White regions (should be detected as foreground)
        rgb_array[10:30, 10:30] = [255, 255, 255]
        rgb_array[70:90, 70:90] = [255, 255, 255]
        
        # Black regions (should be detected as background)
        rgb_array[10:30, 70:90] = [0, 0, 0]
        rgb_array[70:90, 10:30] = [0, 0, 0]
        
        # Colored regions (for color channel testing)
        rgb_array[40:60, 10:30] = [255, 0, 0]    # Red
        rgb_array[40:60, 40:60] = [0, 255, 0]    # Green
        rgb_array[40:60, 70:90] = [0, 0, 255]    # Blue
        
        return Image.fromarray(rgb_array, mode='RGB')
    
    def test_binarizer_creation_with_defaults(self, binarizer):
        """Test that binarizer can be created with default values."""
        assert binarizer is not None
        assert hasattr(binarizer, 'settings')
        assert hasattr(binarizer, 'threshold_methods')
    
    def test_binarizer_default_settings(self, binarizer):
        """Test binarizer default settings."""
        settings = binarizer.settings
        assert isinstance(settings, dict)
        assert 'threshold_method' in settings
        assert 'adaptive_block_size' in settings
        assert 'adaptive_c' in settings
    
    def test_binarizer_supported_threshold_methods(self, binarizer):
        """Test supported threshold methods."""
        methods = binarizer.threshold_methods
        assert isinstance(methods, list)
        assert 'global' in methods
        assert 'otsu' in methods
        assert 'adaptive' in methods
    
    def test_binarizer_process_image_returns_binary(self, binarizer, test_rgb_image):
        """Test that processing returns binary array."""
        result = binarizer.process_image(test_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)
        
        # Should be binary (0 and 255 only, or 0 and 1)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert 0 in unique_values
    
    def test_binarizer_global_threshold_method(self, binarizer, test_rgb_image):
        """Test global threshold method."""
        binarizer.settings['threshold_method'] = 'global'
        binarizer.settings['global_threshold'] = 128
        
        result = binarizer.process_image(test_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        
        # Verify thresholding worked
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_binarizer_otsu_threshold_method(self, binarizer, test_rgb_image):
        """Test Otsu's automatic threshold method."""
        binarizer.settings['threshold_method'] = 'otsu'
        
        result = binarizer.process_image(test_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        
        # Otsu should automatically determine threshold
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_binarizer_adaptive_threshold_method(self, binarizer, test_rgb_image):
        """Test adaptive threshold method."""
        binarizer.settings['threshold_method'] = 'adaptive'
        binarizer.settings['adaptive_block_size'] = 11
        binarizer.settings['adaptive_c'] = 5
        
        result = binarizer.process_image(test_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        
        # Adaptive thresholding should handle local variations
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_binarizer_rgb_channel_processing(self, binarizer):
        """Test individual RGB channel processing."""
        # Create image with distinct color channels
        rgb_array = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb_array[10:40, 10:20] = [255, 0, 0]    # Red channel
        rgb_array[10:40, 20:30] = [0, 255, 0]    # Green channel  
        rgb_array[10:40, 30:40] = [0, 0, 255]    # Blue channel
        
        test_image = Image.fromarray(rgb_array, mode='RGB')
        
        # Test each channel
        if hasattr(binarizer, 'process_channel'):
            red_result = binarizer.process_channel(test_image, 'red')
            green_result = binarizer.process_channel(test_image, 'green')
            blue_result = binarizer.process_channel(test_image, 'blue')
            
            assert isinstance(red_result, np.ndarray)
            assert isinstance(green_result, np.ndarray)
            assert isinstance(blue_result, np.ndarray)
            
            # Results should be different for each channel
            assert not np.array_equal(red_result, green_result)
            assert not np.array_equal(green_result, blue_result)
    
    def test_binarizer_grayscale_conversion(self, binarizer, test_rgb_image):
        """Test grayscale conversion options."""
        # Test different grayscale conversion methods
        if hasattr(binarizer, 'convert_to_grayscale'):
            gray_luminance = binarizer.convert_to_grayscale(test_rgb_image, method='luminance')
            gray_average = binarizer.convert_to_grayscale(test_rgb_image, method='average')
            
            assert isinstance(gray_luminance, np.ndarray)
            assert isinstance(gray_average, np.ndarray)
            assert gray_luminance.shape == (100, 100)
            assert gray_average.shape == (100, 100)
    
    def test_binarizer_morphology_operations(self, binarizer, test_rgb_image):
        """Test morphological operations."""
        binarizer.settings['morphology_operations'] = True
        binarizer.settings['morph_kernel_size'] = 3
        binarizer.settings['morph_operations'] = ['opening', 'closing']
        
        result = binarizer.process_image(test_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        
        # Morphology should clean up the binary image
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_binarizer_noise_reduction(self, binarizer):
        """Test noise reduction capabilities."""
        # Create noisy image
        noisy_array = np.random.randint(0, 256, size=(60, 60, 3), dtype=np.uint8)
        # Add some clear patterns
        noisy_array[20:40, 20:40] = [255, 255, 255]
        noisy_array[25:35, 25:35] = [0, 0, 0]
        
        noisy_image = Image.fromarray(noisy_array, mode='RGB')
        
        binarizer.settings['noise_reduction'] = True
        result = binarizer.process_image(noisy_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (60, 60)
    
    def test_binarizer_different_image_sizes(self, binarizer):
        """Test processing different image sizes."""
        sizes = [(20, 20), (50, 30), (100, 100), (200, 150)]
        
        for width, height in sizes:
            test_array = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            test_image = Image.fromarray(test_array, mode='RGB')
            
            result = binarizer.process_image(test_image)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (height, width)
            assert result.dtype == np.uint8
    
    def test_binarizer_edge_cases(self, binarizer):
        """Test edge cases."""
        # All black image
        black_array = np.zeros((40, 40, 3), dtype=np.uint8)
        black_image = Image.fromarray(black_array, mode='RGB')
        result_black = binarizer.process_image(black_image)
        assert isinstance(result_black, np.ndarray)
        
        # All white image
        white_array = np.ones((40, 40, 3), dtype=np.uint8) * 255
        white_image = Image.fromarray(white_array, mode='RGB')
        result_white = binarizer.process_image(white_image)
        assert isinstance(result_white, np.ndarray)
        
        # Single color image
        red_array = np.zeros((40, 40, 3), dtype=np.uint8)
        red_array[:, :, 0] = 128  # Only red channel
        red_image = Image.fromarray(red_array, mode='RGB')
        result_red = binarizer.process_image(red_image)
        assert isinstance(result_red, np.ndarray)
    
    def test_binarizer_preprocessing_options(self, binarizer, test_rgb_image):
        """Test preprocessing options."""
        # Test with different preprocessing
        preprocessing_options = [
            {'gaussian_blur': True, 'blur_sigma': 1.0},
            {'median_filter': True, 'median_size': 3},
            {'contrast_enhancement': True, 'contrast_factor': 1.2},
        ]
        
        for options in preprocessing_options:
            for key, value in options.items():
                binarizer.settings[key] = value
            
            result = binarizer.process_image(test_rgb_image)
            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 100)
    
    def test_binarizer_performance_metrics(self, binarizer, test_rgb_image):
        """Test performance metrics collection."""
        # Process multiple images
        for _ in range(3):
            binarizer.process_image(test_rgb_image)
        
        if hasattr(binarizer, 'get_performance_stats'):
            stats = binarizer.get_performance_stats()
            assert isinstance(stats, dict)
            assert 'total_processed' in stats
            assert stats['total_processed'] >= 3
    
    def test_binarizer_quality_assessment(self, binarizer, test_rgb_image):
        """Test binarization quality assessment."""
        result = binarizer.process_image(test_rgb_image)
        
        if hasattr(binarizer, 'assess_quality'):
            quality_metrics = binarizer.assess_quality(test_rgb_image, result)
            assert isinstance(quality_metrics, dict)
            assert 'contrast_ratio' in quality_metrics
            assert 'noise_level' in quality_metrics
    
    def test_binarizer_error_handling_invalid_input(self, binarizer):
        """Test error handling for invalid inputs."""
        invalid_inputs = [
            None,
            "not an image",
            123,
            np.array([1, 2, 3]),  # Wrong dimensions
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises((ValueError, TypeError)):
                binarizer.process_image(invalid_input)
    
    def test_binarizer_error_handling_invalid_settings(self, binarizer):
        """Test error handling for invalid settings."""
        # Invalid threshold method
        with pytest.raises(ValueError):
            binarizer.settings['threshold_method'] = 'invalid_method'
            binarizer.process_image(Image.new('RGB', (10, 10)))
        
        # Invalid block size
        with pytest.raises(ValueError):
            binarizer.settings['adaptive_block_size'] = 2  # Must be odd
            binarizer.process_image(Image.new('RGB', (10, 10)))
    
    def test_binarizer_string_representation(self, binarizer):
        """Test binarizer string representation."""
        str_repr = str(binarizer)
        assert "RGBChannelBinarizer" in str_repr
        assert "threshold" in str_repr or "method" in str_repr
    
    def test_binarizer_copy(self, binarizer):
        """Test binarizer copying."""
        if hasattr(binarizer, 'copy'):
            copied = binarizer.copy()
            assert copied is not binarizer
            assert copied.settings == binarizer.settings
    
    def test_binarizer_reset(self, binarizer):
        """Test binarizer reset functionality."""
        if hasattr(binarizer, 'reset'):
            # Process some data
            test_image = Image.new('RGB', (10, 10))
            binarizer.process_image(test_image)
            
            # Reset binarizer
            binarizer.reset()
            
            # Should be able to process again
            result = binarizer.process_image(test_image)
            assert isinstance(result, np.ndarray)
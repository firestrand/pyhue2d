"""RGB Channel Binarizer for JABCode image processing.

This module provides the RGBChannelBinarizer class for converting color images
to binary representations using various thresholding methods, morphological
operations, and noise reduction techniques.
"""

import time
from typing import Union, Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage
from skimage import filters, morphology

from ..core import Bitmap


class RGBChannelBinarizer:
    """RGB Channel Binarizer for JABCode image processing.
    
    The RGBChannelBinarizer converts color images to binary representations
    using various thresholding methods, RGB channel processing, morphological
    operations, and noise reduction techniques.
    """
    
    DEFAULT_SETTINGS = {
        'threshold_method': 'otsu',
        'global_threshold': 128,
        'adaptive_block_size': 11,
        'adaptive_c': 5,
        'morphology_operations': False,
        'morph_kernel_size': 3,
        'morph_operations': ['opening', 'closing'],
        'noise_reduction': False,
        'gaussian_blur': False,
        'blur_sigma': 1.0,
        'median_filter': False,
        'median_size': 3,
        'contrast_enhancement': False,
        'contrast_factor': 1.2,
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize RGB Channel Binarizer.
        
        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Supported threshold methods
        self.threshold_methods = ['global', 'otsu', 'adaptive']
        
        # Validate settings
        self._validate_settings()
        
        # Performance statistics
        self._stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'processing_times': [],
        }
    
    def process_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Process PIL Image to binary array.
        
        Args:
            image: PIL Image object or numpy array
            
        Returns:
            Binary numpy array (0 and 255 values)
            
        Raises:
            ValueError: For invalid inputs
        """
        start_time = time.time()
        
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                array = np.array(image)
            elif isinstance(image, np.ndarray):
                array = image.copy()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Validate array
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape: {array.shape}")
            
            # Apply preprocessing
            if self.settings['contrast_enhancement']:
                array = self._enhance_contrast(array)
            
            if self.settings['gaussian_blur']:
                array = self._apply_gaussian_blur(array)
            
            if self.settings['median_filter']:
                array = self._apply_median_filter(array)
            
            # Convert to grayscale
            gray = self.convert_to_grayscale(array, method='luminance')
            
            # Apply thresholding
            binary = self._apply_threshold(gray)
            
            # Apply morphological operations
            if self.settings['morphology_operations']:
                binary = self._apply_morphology(binary)
            
            # Apply noise reduction
            if self.settings['noise_reduction']:
                binary = self._reduce_noise(binary)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return binary
            
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}") from e
    
    def process_bitmap(self, bitmap: Bitmap) -> np.ndarray:
        """Process Bitmap object to binary array.
        
        Args:
            bitmap: Bitmap object
            
        Returns:
            Binary numpy array
        """
        if not isinstance(bitmap, Bitmap):
            raise ValueError("Input must be a Bitmap object")
        
        return self.process_array(bitmap.array)
    
    def process_array(self, array: np.ndarray) -> np.ndarray:
        """Process numpy array to binary array.
        
        Args:
            array: Numpy array (RGB)
            
        Returns:
            Binary numpy array
        """
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        # Convert to PIL Image for processing
        if array.ndim == 3 and array.shape[2] == 3:
            pil_image = Image.fromarray(array.astype(np.uint8), mode='RGB')
        else:
            raise ValueError(f"Expected RGB array, got shape: {array.shape}")
        
        return self.process_image(pil_image)
    
    def process_channel(self, image: Union[Image.Image, np.ndarray], channel: str) -> np.ndarray:
        """Process individual RGB channel.
        
        Args:
            image: PIL Image object or numpy array
            channel: Channel name ('red', 'green', 'blue')
            
        Returns:
            Binary numpy array from specified channel
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            array = np.array(image)
        else:
            array = image.copy()
        
        # Extract channel
        channel_map = {'red': 0, 'green': 1, 'blue': 2}
        if channel not in channel_map:
            raise ValueError(f"Invalid channel: {channel}")
        
        channel_array = array[:, :, channel_map[channel]]
        
        # Apply thresholding to single channel
        binary = self._apply_threshold(channel_array)
        
        return binary
    
    def convert_to_grayscale(self, image: Union[Image.Image, np.ndarray], 
                           method: str = 'luminance') -> np.ndarray:
        """Convert RGB image to grayscale.
        
        Args:
            image: PIL Image or numpy array
            method: Conversion method ('luminance', 'average')
            
        Returns:
            Grayscale numpy array
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            array = np.array(image)
        else:
            array = image.copy()
        
        if method == 'luminance':
            # Standard luminance weights
            gray = np.dot(array, [0.2989, 0.5870, 0.1140])
        elif method == 'average':
            # Simple average
            gray = np.mean(array, axis=2)
        else:
            raise ValueError(f"Unsupported grayscale method: {method}")
        
        return gray.astype(np.uint8)
    
    def _apply_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply thresholding to grayscale image.
        
        Args:
            gray: Grayscale numpy array
            
        Returns:
            Binary numpy array
        """
        method = self.settings['threshold_method']
        
        if method == 'global':
            threshold = self.settings['global_threshold']
            binary = (gray > threshold).astype(np.uint8) * 255
            
        elif method == 'otsu':
            threshold = filters.threshold_otsu(gray)
            binary = (gray > threshold).astype(np.uint8) * 255
            
        elif method == 'adaptive':
            # Use local adaptive thresholding
            block_size = self.settings['adaptive_block_size']
            c = self.settings['adaptive_c']
            
            # Ensure block size is odd
            if block_size % 2 == 0:
                raise ValueError("Adaptive block size must be odd")
            
            threshold = filters.threshold_local(gray, block_size, offset=c)
            binary = (gray > threshold).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unsupported threshold method: {method}")
        
        return binary
    
    def _apply_morphology(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to binary image.
        
        Args:
            binary: Binary numpy array
            
        Returns:
            Processed binary array
        """
        kernel_size = self.settings['morph_kernel_size']
        operations = self.settings['morph_operations']
        
        # Create morphological kernel
        kernel = morphology.disk(kernel_size)
        
        # Convert to boolean for morphology operations
        binary_bool = binary > 0
        
        for operation in operations:
            if operation == 'opening':
                binary_bool = morphology.opening(binary_bool, kernel)
            elif operation == 'closing':
                binary_bool = morphology.closing(binary_bool, kernel)
            elif operation == 'erosion':
                binary_bool = morphology.erosion(binary_bool, kernel)
            elif operation == 'dilation':
                binary_bool = morphology.dilation(binary_bool, kernel)
        
        # Convert back to uint8
        return (binary_bool).astype(np.uint8) * 255
    
    def _reduce_noise(self, binary: np.ndarray) -> np.ndarray:
        """Apply noise reduction to binary image.
        
        Args:
            binary: Binary numpy array
            
        Returns:
            Noise-reduced binary array
        """
        # Remove small objects and fill holes
        binary_bool = binary > 0
        
        # Remove small objects
        binary_bool = morphology.remove_small_objects(binary_bool, min_size=10)
        
        # Fill small holes
        binary_bool = morphology.remove_small_holes(binary_bool, area_threshold=10)
        
        return (binary_bool).astype(np.uint8) * 255
    
    def _enhance_contrast(self, array: np.ndarray) -> np.ndarray:
        """Enhance contrast of RGB array.
        
        Args:
            array: RGB numpy array
            
        Returns:
            Contrast-enhanced array
        """
        factor = self.settings['contrast_factor']
        
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(array.astype(np.uint8), mode='RGB')
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        
        return np.array(enhanced)
    
    def _apply_gaussian_blur(self, array: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to RGB array.
        
        Args:
            array: RGB numpy array
            
        Returns:
            Blurred array
        """
        sigma = self.settings['blur_sigma']
        
        # Apply Gaussian filter to each channel
        blurred = np.zeros_like(array)
        for i in range(3):
            blurred[:, :, i] = ndimage.gaussian_filter(array[:, :, i], sigma=sigma)
        
        return blurred.astype(np.uint8)
    
    def _apply_median_filter(self, array: np.ndarray) -> np.ndarray:
        """Apply median filter to RGB array.
        
        Args:
            array: RGB numpy array
            
        Returns:
            Filtered array
        """
        size = self.settings['median_size']
        
        # Apply median filter to each channel
        filtered = np.zeros_like(array)
        for i in range(3):
            filtered[:, :, i] = ndimage.median_filter(array[:, :, i], size=size)
        
        return filtered.astype(np.uint8)
    
    def _validate_settings(self) -> None:
        """Validate binarizer settings."""
        if self.settings['threshold_method'] not in self.threshold_methods:
            raise ValueError(f"Invalid threshold method: {self.settings['threshold_method']}")
        
        if self.settings['adaptive_block_size'] % 2 == 0:
            raise ValueError("Adaptive block size must be odd")
        
        if not (0 <= self.settings['global_threshold'] <= 255):
            raise ValueError("Global threshold must be between 0 and 255")
    
    def _update_stats(self, processing_time: float) -> None:
        """Update processing statistics.
        
        Args:
            processing_time: Time taken for processing
        """
        self._stats['total_processed'] += 1
        self._stats['total_processing_time'] += processing_time
        self._stats['processing_times'].append(processing_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        if self._stats['total_processed'] == 0:
            return {
                'total_processed': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0,
            }
        
        return {
            'total_processed': self._stats['total_processed'],
            'total_processing_time': self._stats['total_processing_time'],
            'avg_processing_time': (
                self._stats['total_processing_time'] / self._stats['total_processed']
            ),
            'min_processing_time': min(self._stats['processing_times']),
            'max_processing_time': max(self._stats['processing_times']),
        }
    
    def assess_quality(self, original: Union[Image.Image, np.ndarray], 
                      binary: np.ndarray) -> Dict[str, float]:
        """Assess binarization quality metrics.
        
        Args:
            original: Original image
            binary: Binarized result
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert original to grayscale if needed
        if isinstance(original, Image.Image):
            if original.mode != 'RGB':
                original = original.convert('RGB')
            gray = np.array(original.convert('L'))
        else:
            gray = self.convert_to_grayscale(original)
        
        # Calculate contrast ratio
        foreground_mean = np.mean(gray[binary > 0])
        background_mean = np.mean(gray[binary == 0])
        contrast_ratio = abs(foreground_mean - background_mean) / 255.0
        
        # Calculate noise level (standard deviation of small regions)
        noise_level = np.std(binary) / 255.0
        
        return {
            'contrast_ratio': contrast_ratio,
            'noise_level': noise_level,
            'foreground_ratio': np.sum(binary > 0) / binary.size,
        }
    
    def reset(self) -> None:
        """Reset binarizer statistics and state."""
        self._stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'processing_times': [],
        }
    
    def copy(self) -> 'RGBChannelBinarizer':
        """Create a copy of this binarizer.
        
        Returns:
            New RGBChannelBinarizer instance with same settings
        """
        return RGBChannelBinarizer(self.settings.copy())
    
    def __str__(self) -> str:
        """String representation of binarizer."""
        return (
            f"RGBChannelBinarizer(method={self.settings['threshold_method']}, "
            f"processed={self._stats['total_processed']})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"RGBChannelBinarizer(settings={self.settings})"
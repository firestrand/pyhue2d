"""JABCode decoder class for high-level decoding interface.

This module provides the JABCodeDecoder class which serves as the main
entry point for decoding JABCode symbols from images.
"""

import time
from typing import Union, Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np

from .image_processing.finder_detector import FinderPatternDetector
from .image_processing.perspective_transformer import PerspectiveTransformer
from .image_processing.symbol_sampler import SymbolSampler
from .image_processing.binarizer import RGBChannelBinarizer
from .module_data_extractor import ModuleDataExtractor
from .core import Point2D


class JABCodeDecoder:
    """High-level JABCode decoder class.
    
    The JABCodeDecoder provides a simple, user-friendly interface for decoding
    JABCode symbols from images. It integrates the image processing pipeline
    and provides methods for extracting data from JABCode images.
    """
    
    DEFAULT_SETTINGS = {
        'detection_method': 'scanline',
        'perspective_correction': True,
        'error_correction': True,
        'validate_patterns': True,
        'multi_symbol_support': True,
        'noise_reduction': True,
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize JABCode decoder.
        
        Args:
            settings: Optional configuration settings (immutable after init)
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Initialize image processing components
        finder_settings = {
            'detection_method': self.settings['detection_method'],
            'noise_reduction': self.settings['noise_reduction'],
        }
        
        self.finder_detector = FinderPatternDetector(finder_settings)
        self.perspective_transformer = PerspectiveTransformer()
        self.symbol_sampler = SymbolSampler()
        self.binarizer = RGBChannelBinarizer()
        self.module_extractor = ModuleDataExtractor()
        
        # Statistics
        self._stats = {
            'total_decoded': 0,
            'total_detection_time': 0.0,
            'decoding_times': [],
            'patterns_detected': 0,
        }
    
    def decode(self, image_source: Union[str, Image.Image, np.ndarray]) -> bytes:
        """Decode JABCode symbol from image source.
        
        Args:
            image_source: Image to decode (file path, PIL Image, or numpy array)
            
        Returns:
            Decoded data as bytes
            
        Raises:
            ValueError: For invalid input or decoding errors
            NotImplementedError: Core decoding functionality not yet implemented
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess image
            image = self._load_image(image_source)
            
            # Step 2: Detect finder patterns
            pattern_dicts = self.finder_detector.find_patterns(image)
            
            if not pattern_dicts:
                raise ValueError("No JABCode patterns detected in image")
            
            # Extract Point2D centers from pattern dictionaries
            all_patterns = [pattern_dict['center'] for pattern_dict in pattern_dicts]
            
            # Select best 4 patterns for JABCode (if more than 4 detected)
            patterns = self._select_jabcode_patterns(all_patterns, pattern_dicts)
            
            # Step 3: Extract and validate symbol structure
            # Estimate symbol size based on finder patterns
            symbol_size = self._estimate_symbol_size(patterns, image)
            
            # Step 4: Apply perspective correction and sample symbol
            if self.settings['perspective_correction']:
                perspective_transform = self.perspective_transformer.get_jabcode_perspective_transform(
                    patterns, symbol_size
                )
                sampled_symbol = self.symbol_sampler.sample_symbol(
                    image, perspective_transform, symbol_size
                )
            else:
                # Direct sampling without perspective correction
                sampled_symbol = self._sample_symbol_direct(image, patterns, symbol_size)
            
            # Step 5: Extract module data from symbol
            symbol_metadata = self._extract_symbol_metadata(patterns, symbol_size)
            module_data = self.module_extractor.extract_module_data(
                sampled_symbol, symbol_metadata
            )
            
            # Step 6: Apply error correction
            if self.settings['error_correction']:
                corrected_data = self._apply_error_correction(module_data, symbol_metadata)
            else:
                corrected_data = module_data
            
            # Step 7: Reconstruct original data
            reconstructed_data = self._reconstruct_data(corrected_data, symbol_metadata)
            
            # Update statistics
            decoding_time = time.time() - start_time
            self._update_stats(decoding_time, len(patterns))
            
            # Return reconstructed data
            return reconstructed_data
            
        except Exception as e:
            if isinstance(e, NotImplementedError):
                raise
            raise ValueError(f"JABCode decoding failed: {str(e)}") from e
    
    def _load_image(self, image_source: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Load image from various source types.
        
        Args:
            image_source: Image source
            
        Returns:
            PIL Image object
        """
        if isinstance(image_source, str):
            # File path
            return Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            # Already a PIL Image
            return image_source
        elif isinstance(image_source, np.ndarray):
            # NumPy array
            if image_source.ndim == 2:
                # Grayscale
                return Image.fromarray(image_source, mode='L')
            elif image_source.ndim == 3:
                # RGB
                return Image.fromarray(image_source, mode='RGB')
            else:
                raise ValueError(f"Unsupported array dimensions: {image_source.ndim}")
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Return detection statistics as a dictionary."""
        stats = {
            'total_decoded': self._stats.get('total_decoded', 0),
            'total_detection_time': self._stats.get('total_detection_time', 0.0),
            'patterns_detected': self._stats.get('patterns_detected', 0),
        }
        
        if self._stats['total_decoded'] > 0:
            stats['avg_detection_time'] = (
                self._stats['total_detection_time'] / self._stats['total_decoded']
            )
        else:
            stats['avg_detection_time'] = 0.0
            
        return stats
    
    def _update_stats(self, decoding_time: float, patterns_found: int) -> None:
        """Update decoding statistics.
        
        Args:
            decoding_time: Time taken for decoding attempt
            patterns_found: Number of patterns detected
        """
        self._stats['total_decoded'] += 1
        self._stats['total_detection_time'] += decoding_time
        self._stats['decoding_times'].append(decoding_time)
        self._stats['patterns_detected'] += patterns_found
    
    def reset_stats(self) -> None:
        """Reset decoder statistics."""
        self._stats = {
            'total_decoded': 0,
            'total_detection_time': 0.0,
            'decoding_times': [],
            'patterns_detected': 0,
        }
    
    def _select_jabcode_patterns(self, all_patterns: List[Point2D], 
                               pattern_dicts: List[Dict[str, Any]]) -> List[Point2D]:
        """Select the best 4 patterns for JABCode decoding.
        
        JABCode requires exactly 4 finder patterns positioned at the corners.
        If more patterns are detected, select the 4 that best form a quadrilateral.
        
        Args:
            all_patterns: All detected pattern centers
            pattern_dicts: Complete pattern information
            
        Returns:
            List of 4 patterns ordered as [top-left, top-right, bottom-right, bottom-left]
            
        Raises:
            ValueError: If fewer than 4 patterns detected
        """
        if len(all_patterns) < 4:
            raise ValueError(f"JABCode requires at least 4 finder patterns, found {len(all_patterns)}")
        
        if len(all_patterns) == 4:
            # Exactly 4 patterns - order them correctly
            return self._order_patterns_for_jabcode(all_patterns)
        
        # More than 4 patterns - select best quadrilateral
        # For now, take the 4 patterns that form the largest quadrilateral
        # This is a simplified approach - could be enhanced with quality scores
        
        # Calculate center of all patterns
        center_x = sum(p.x for p in all_patterns) / len(all_patterns)
        center_y = sum(p.y for p in all_patterns) / len(all_patterns)
        
        # Find patterns closest to the 4 corners
        top_left = min(all_patterns, key=lambda p: (p.x - center_x)**2 + (p.y - center_y)**2 
                      if p.x < center_x and p.y < center_y else float('inf'))
        top_right = min(all_patterns, key=lambda p: (p.x - center_x)**2 + (p.y - center_y)**2 
                       if p.x >= center_x and p.y < center_y else float('inf'))
        bottom_right = min(all_patterns, key=lambda p: (p.x - center_x)**2 + (p.y - center_y)**2 
                          if p.x >= center_x and p.y >= center_y else float('inf'))
        bottom_left = min(all_patterns, key=lambda p: (p.x - center_x)**2 + (p.y - center_y)**2 
                         if p.x < center_x and p.y >= center_y else float('inf'))
        
        # Return ordered patterns
        return [top_left, top_right, bottom_right, bottom_left]
    
    def _order_patterns_for_jabcode(self, patterns: List[Point2D]) -> List[Point2D]:
        """Order 4 patterns for JABCode as [top-left, top-right, bottom-right, bottom-left].
        
        Args:
            patterns: List of 4 pattern centers
            
        Returns:
            Ordered pattern list
        """
        # Calculate center point
        center_x = sum(p.x for p in patterns) / 4
        center_y = sum(p.y for p in patterns) / 4
        
        # Classify patterns by quadrant
        quadrants = {'top_left': [], 'top_right': [], 'bottom_right': [], 'bottom_left': []}
        
        for pattern in patterns:
            if pattern.x < center_x and pattern.y < center_y:
                quadrants['top_left'].append(pattern)
            elif pattern.x >= center_x and pattern.y < center_y:
                quadrants['top_right'].append(pattern)
            elif pattern.x >= center_x and pattern.y >= center_y:
                quadrants['bottom_right'].append(pattern)
            else:
                quadrants['bottom_left'].append(pattern)
        
        # Select one pattern per quadrant (take first if multiple)
        ordered = []
        for quadrant in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if quadrants[quadrant]:
                ordered.append(quadrants[quadrant][0])
            else:
                # Fallback - use closest pattern
                closest = min(patterns, key=lambda p: abs(p.x - center_x) + abs(p.y - center_y))
                ordered.append(closest)
        
        return ordered[:4]  # Ensure exactly 4 patterns

    def _estimate_symbol_size(self, patterns: List[Point2D], image: Image.Image) -> Tuple[int, int]:
        """Estimate symbol size based on finder patterns.
        
        Args:
            patterns: List of detected finder patterns
            image: Source image
            
        Returns:
            Estimated (width, height) in modules
        """
        if len(patterns) < 4:
            # Default size for incomplete detection
            return (21, 21)
        
        # Calculate distances between patterns to estimate symbol size
        # This is simplified - real implementation would use pattern spacing
        # and module size estimation
        
        # Estimate based on distance between top patterns
        top_left = patterns[0]
        top_right = patterns[1] if len(patterns) > 1 else patterns[0]
        
        width_distance = abs(top_right.x - top_left.x)
        
        # Estimate module size (finder patterns are 7 modules apart + symbol width)
        estimated_module_size = max(1, width_distance / 21)  # Assume 21x21 minimum
        
        # Estimate symbol dimensions
        width = max(21, int(width_distance / estimated_module_size))
        height = width  # Assume square for now
        
        return (width, height)
    
    def _sample_symbol_direct(self, image: Image.Image, patterns: List[Point2D],
                            symbol_size: Tuple[int, int]) -> np.ndarray:
        """Sample symbol directly without perspective correction.
        
        Args:
            image: Source image
            patterns: Finder patterns
            symbol_size: Symbol dimensions
            
        Returns:
            Sampled symbol array
        """
        width, height = symbol_size
        
        # Create a simple sampling without perspective correction
        # This is a fallback method
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            # RGB image
            sampled = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Grayscale
            sampled = np.zeros((height, width), dtype=np.uint8)
        
        # Fill with background values for now
        sampled.fill(255)
        
        return sampled
    
    def _extract_symbol_metadata(self, patterns: List[Point2D],
                               symbol_size: Tuple[int, int]) -> Dict[str, Any]:
        """Extract symbol metadata from detected patterns.
        
        Args:
            patterns: Detected finder patterns
            symbol_size: Symbol dimensions
            
        Returns:
            Symbol metadata dictionary
        """
        width, height = symbol_size
        
        # This is simplified metadata extraction
        # Real implementation would read metadata from specific symbol positions
        metadata = {
            'width': width,
            'height': height,
            'color_count': 8,           # Default to 8 colors
            'ecc_level': 'M',          # Default to medium error correction
            'mask_pattern': 0,         # Default mask pattern
            'is_master': True,         # Assume master symbol
            'version': 1,              # Default version
            'finder_patterns': patterns,
        }
        
        return metadata
    
    def _apply_error_correction(self, module_data: List[int],
                              metadata: Dict[str, Any]) -> List[int]:
        """Apply LDPC error correction to module data.
        
        Args:
            module_data: Raw module data
            metadata: Symbol metadata
            
        Returns:
            Error-corrected data
        """
        # Placeholder for LDPC error correction
        # This will be implemented when LDPC integration is added
        
        # For now, return data as-is
        return module_data
    
    def _reconstruct_data(self, corrected_data: List[int],
                        metadata: Dict[str, Any]) -> bytes:
        """Reconstruct original data from corrected module data.
        
        Args:
            corrected_data: Error-corrected module data
            metadata: Symbol metadata
            
        Returns:
            Reconstructed original data
        """
        # Placeholder for data reconstruction
        # This would involve:
        # 1. Converting color indices to bits
        # 2. Deinterleaving data
        # 3. Mode-based decoding (alphanumeric, byte, etc.)
        # 4. Reassembling original message
        
        # For now, return placeholder data
        return b"JABCode decoding not yet fully implemented"
    
    def __str__(self) -> str:
        """String representation of decoder."""
        return (
            f"JABCodeDecoder(method={self.settings['detection_method']}, "
            f"perspective_correction={self.settings['perspective_correction']}, "
            f"decoded={self._stats['total_decoded']})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"JABCodeDecoder(settings={self.settings})"
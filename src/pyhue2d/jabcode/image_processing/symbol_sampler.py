"""Symbol Sampler for JABCode image processing.

This module provides the SymbolSampler class for extracting symbol data
from JABCode images with perspective correction and module sampling.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
from ..core import Point2D
from .perspective_transformer import PerspectiveTransform, PerspectiveTransformer


class SymbolSampler:
    """Symbol Sampler for JABCode image processing.
    
    Extracts module data from JABCode symbols using perspective correction
    and robust sampling algorithms.
    """
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize symbol sampler.
        
        Args:
            settings: Optional configuration settings
        """
        default_settings = {
            'neighborhood_size': 3,      # Size of sampling neighborhood
            'interpolation_method': 'bilinear',  # 'nearest', 'bilinear'
            'boundary_handling': 'background',   # 'background', 'replicate'
            'background_value': 255,     # Background value for out-of-bounds
        }
        
        self.settings = default_settings
        if settings:
            self.settings.update(settings)
        
        self.perspective_transformer = PerspectiveTransformer()
    
    def sample_symbol(self, image: Image.Image, 
                     perspective_transform: PerspectiveTransform,
                     symbol_size: Tuple[int, int]) -> np.ndarray:
        """Sample JABCode symbol with perspective correction.
        
        Args:
            image: Source image to sample from
            perspective_transform: Perspective transformation to apply
            symbol_size: (width, height) of symbol in modules
            
        Returns:
            2D numpy array of sampled symbol data
        """
        return self.perspective_transformer.sample_symbol_with_perspective(
            image, perspective_transform, symbol_size
        )
    
    def sample_modules_at_positions(self, image: Image.Image,
                                  positions: List[Point2D]) -> List[int]:
        """Sample module values at specified positions.
        
        Args:
            image: Source image
            positions: List of positions to sample
            
        Returns:
            List of sampled values
        """
        image_array = np.array(image)
        
        # Handle different image modes
        if len(image_array.shape) == 3:
            # RGB image - convert to grayscale
            grayscale = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            # Already grayscale
            grayscale = image_array
        
        img_height, img_width = grayscale.shape
        sampled_values = []
        
        for pos in positions:
            value = self._sample_at_position(grayscale, pos, img_width, img_height)
            sampled_values.append(value)
        
        return sampled_values
    
    def _sample_at_position(self, image_array: np.ndarray, position: Point2D,
                          img_width: int, img_height: int) -> int:
        """Sample image value at a specific position with neighborhood averaging.
        
        Args:
            image_array: Grayscale image array
            position: Position to sample
            img_width: Image width
            img_height: Image height
            
        Returns:
            Sampled value
        """
        x, y = position.x, position.y
        neighborhood_size = self.settings['neighborhood_size']
        half_size = neighborhood_size // 2
        
        total_value = 0
        sample_count = 0
        
        # Sample neighborhood around the position
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                sample_x = int(x + dx)
                sample_y = int(y + dy)
                
                # Check bounds
                if 0 <= sample_x < img_width and 0 <= sample_y < img_height:
                    total_value += image_array[sample_y, sample_x]
                    sample_count += 1
                elif self.settings['boundary_handling'] == 'background':
                    total_value += self.settings['background_value']
                    sample_count += 1
                elif self.settings['boundary_handling'] == 'replicate':
                    # Clamp to image boundaries
                    clamped_x = max(0, min(img_width - 1, sample_x))
                    clamped_y = max(0, min(img_height - 1, sample_y))
                    total_value += image_array[clamped_y, clamped_x]
                    sample_count += 1
        
        # Return average value
        if sample_count > 0:
            return total_value // sample_count
        else:
            return self.settings['background_value']
    
    def sample_with_interpolation(self, image: Image.Image,
                                positions: List[Point2D]) -> List[float]:
        """Sample image values with interpolation.
        
        Args:
            image: Source image
            positions: List of positions to sample (can be fractional)
            
        Returns:
            List of interpolated values
        """
        image_array = np.array(image)
        
        # Handle different image modes
        if len(image_array.shape) == 3:
            # RGB image - convert to grayscale
            grayscale = np.mean(image_array, axis=2)
        else:
            # Already grayscale
            grayscale = image_array.astype(float)
        
        img_height, img_width = grayscale.shape
        sampled_values = []
        
        for pos in positions:
            if self.settings['interpolation_method'] == 'nearest':
                x = int(round(pos.x))
                y = int(round(pos.y))
                if 0 <= x < img_width and 0 <= y < img_height:
                    value = grayscale[y, x]
                else:
                    value = self.settings['background_value']
            elif self.settings['interpolation_method'] == 'bilinear':
                value = self._bilinear_interpolation(grayscale, pos, img_width, img_height)
            else:
                raise ValueError(f"Unknown interpolation method: {self.settings['interpolation_method']}")
            
            sampled_values.append(value)
        
        return sampled_values
    
    def _bilinear_interpolation(self, image_array: np.ndarray, position: Point2D,
                              img_width: int, img_height: int) -> float:
        """Perform bilinear interpolation at a position.
        
        Args:
            image_array: Image array
            position: Position to interpolate
            img_width: Image width  
            img_height: Image height
            
        Returns:
            Interpolated value
        """
        x, y = position.x, position.y
        
        # Get the four surrounding integer coordinates
        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1
        
        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0
        
        # Get pixel values (with boundary handling)
        def get_pixel(px, py):
            if 0 <= px < img_width and 0 <= py < img_height:
                return image_array[py, px]
            else:
                return self.settings['background_value']
        
        p00 = get_pixel(x0, y0)
        p01 = get_pixel(x0, y1)
        p10 = get_pixel(x1, y0)
        p11 = get_pixel(x1, y1)
        
        # Bilinear interpolation
        value = (p00 * (1 - wx) * (1 - wy) +
                p10 * wx * (1 - wy) +
                p01 * (1 - wx) * wy +
                p11 * wx * wy)
        
        return value
    
    def extract_symbol_region(self, image: Image.Image,
                            finder_patterns: List[Point2D],
                            symbol_size: Tuple[int, int]) -> np.ndarray:
        """Extract symbol region with perspective correction.
        
        Args:
            image: Source image
            finder_patterns: List of 4 finder pattern positions
            symbol_size: (width, height) of symbol in modules
            
        Returns:
            Perspective-corrected symbol array
        """
        # Get perspective transformation
        transform = self.perspective_transformer.get_jabcode_perspective_transform(
            finder_patterns, symbol_size
        )
        
        # Sample the symbol
        return self.sample_symbol(image, transform, symbol_size)
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling operations.
        
        Returns:
            Dictionary with sampling statistics
        """
        return {
            'neighborhood_size': self.settings['neighborhood_size'],
            'interpolation_method': self.settings['interpolation_method'],
            'boundary_handling': self.settings['boundary_handling'],
            'background_value': self.settings['background_value'],
        }
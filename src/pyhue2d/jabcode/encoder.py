"""JABCode encoder class for high-level encoding interface.

This module provides the JABCodeEncoder class which serves as the main
entry point for encoding data into JABCode symbols.
"""

import time
from typing import Union, Dict, Any, Optional
from PIL import Image
import numpy as np

from .pipeline.encoding import EncodingPipeline
from .core import EncodedData, Bitmap
from .color_palette import ColorPalette


class JABCodeEncoder:
    """High-level JABCode encoder class.
    
    The JABCodeEncoder provides a simple, user-friendly interface for encoding
    data into JABCode symbols. It integrates the encoding pipeline and provides
    methods for generating both raw encoded data and visual representations.
    """
    
    DEFAULT_SETTINGS = {
        'color_count': 8,
        'ecc_level': 'M',
        'version': 'auto',
        'optimize': True,
        'master_symbol': True,
        'quiet_zone': 2,
        'module_size': 4,  # Pixels per module for image generation
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize JABCode encoder.
        
        Args:
            settings: Optional configuration settings (immutable after init)
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Initialize encoding pipeline
        pipeline_settings = {
            'color_count': self.settings['color_count'],
            'ecc_level': self.settings['ecc_level'],
            'version': self.settings['version'],
            'optimize': self.settings['optimize'],
            'master_symbol': self.settings['master_symbol'],
            'quiet_zone': self.settings['quiet_zone'],
        }
        
        self.pipeline = EncodingPipeline(pipeline_settings)
        
        # Statistics
        self._stats = {
            'total_encoded': 0,
            'total_encoding_time': 0.0,
            'encoding_times': [],
        }
    
    def encode(self, data: Union[str, bytes]) -> EncodedData:
        """Encode data into JABCode format.
        
        Args:
            data: Input data to encode (string or bytes)
            
        Returns:
            EncodedData containing the complete JABCode symbol data
            
        Raises:
            ValueError: For invalid input or encoding errors
        """
        # Validate color_count and ecc_level before proceeding
        valid_colors = [4, 8, 16, 32, 64, 128, 256]
        valid_levels = ["L", "M", "Q", "H"]
        if self.settings['color_count'] not in valid_colors:
            raise ValueError(f"Color count must be one of {valid_colors}")
        if self.settings['ecc_level'] not in valid_levels:
            raise ValueError(f"ECC level must be one of {valid_levels}")
        start_time = time.time()
        
        try:
            # Use the encoding pipeline to encode the data
            result = self.pipeline.encode(data)
            
            # Update statistics
            encoding_time = time.time() - start_time
            self._update_stats(encoding_time)
            
            return result
            
        except Exception as e:
            raise ValueError(f"JABCode encoding failed: {str(e)}") from e
    
    def encode_to_bitmap(self, data: Union[str, bytes]) -> Bitmap:
        """Encode data and generate bitmap representation.
        
        Args:
            data: Input data to encode
            
        Returns:
            Bitmap object containing the visual JABCode symbol
        """
        return self.pipeline.encode_to_bitmap(data)
    
    def encode_to_image(self, data: Union[str, bytes]) -> Image.Image:
        """Encode data and generate PIL Image representation.
        
        Args:
            data: Input data to encode
            
        Returns:
            PIL Image object containing the visual JABCode symbol
        """
        # Get bitmap from pipeline
        bitmap = self.encode_to_bitmap(data)
        
        # Scale up by module size for better visibility
        module_size = self.settings['module_size']
        scaled_width = bitmap.width * module_size
        scaled_height = bitmap.height * module_size
        
        # Convert to PIL Image
        if len(bitmap.array.shape) == 3:
            # RGB bitmap
            pil_image = Image.fromarray(bitmap.array, mode='RGB')
        else:
            # Grayscale bitmap - convert to RGB
            rgb_array = np.stack([bitmap.array] * 3, axis=-1)
            pil_image = Image.fromarray(rgb_array, mode='RGB')
        
        # Scale up using nearest neighbor to maintain sharp edges
        scaled_image = pil_image.resize(
            (scaled_width, scaled_height), 
            Image.NEAREST
        )
        
        return scaled_image
    
    def set_color_palette(self, palette: ColorPalette) -> None:
        """Set custom color palette."""
        self.settings['color_count'] = palette.color_count
        self.pipeline.set_color_palette(palette)
    
    def get_encoding_stats(self) -> dict:
        """Return encoding statistics as a dictionary."""
        stats = {
            'total_encoded': self._stats.get('total_encoded', 0),
            'encoder_total_time': self._stats.get('total_encoding_time', 0.0),
            'encoder_avg_time': (self._stats['total_encoding_time'] / self._stats['total_encoded']) if self._stats['total_encoded'] > 0 else 0.0,
        }
        # Add encoding_time if available from last encode
        if self._stats.get('encoding_times'):
            stats['encoding_time'] = self._stats['encoding_times'][-1]
        return stats
    
    def reset(self) -> None:
        """Reset encoder statistics and state."""
        self._stats = {
            'total_encoded': 0,
            'total_encoding_time': 0.0,
            'encoding_times': [],
        }
        self.pipeline.reset()
    
    def copy(self) -> 'JABCodeEncoder':
        """Create a copy of this encoder.
        
        Returns:
            New JABCodeEncoder instance with same settings
        """
        return JABCodeEncoder(self.settings.copy())
    
    def _update_stats(self, encoding_time: float) -> None:
        """Update encoding statistics.
        
        Args:
            encoding_time: Time taken for encoding in seconds
        """
        self._stats['total_encoded'] += 1
        self._stats['total_encoding_time'] += encoding_time
        self._stats['encoding_times'].append(encoding_time)
    
    def __str__(self) -> str:
        """String representation of encoder."""
        return (
            f"JABCodeEncoder(color_count={self.settings['color_count']}, "
            f"ecc_level={self.settings['ecc_level']}, "
            f"version={self.settings['version']})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"JABCodeEncoder(settings={self.settings})"
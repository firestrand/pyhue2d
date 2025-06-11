"""Bitmap renderer for JABCode symbols.

This module provides the BitmapRenderer class for converting JABCode symbol
matrices into visual bitmap representations with proper color mapping,
scaling, and formatting.
"""

import time
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from .core import Bitmap, Symbol
from .color_palette import ColorPalette


class BitmapRenderer:
    """Bitmap renderer for JABCode symbols.
    
    The BitmapRenderer handles the visual rendering of JABCode symbol matrices,
    providing control over module size, quiet zones, colors, and output format.
    It converts color index matrices into proper RGB bitmaps or PIL Images.
    """
    
    DEFAULT_SETTINGS = {
        'module_size': 4,           # Pixels per module
        'quiet_zone': 2,            # Modules of quiet zone
        'background_color': (255, 255, 255),  # White background
        'border_width': 0,          # Border width in pixels
        'anti_aliasing': False,     # Anti-aliasing (not implemented yet)
        'color_count': 8,           # Number of colors in palette
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize bitmap renderer.
        
        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Validate settings
        self._validate_settings()
        
        # Initialize color palette
        self.color_palette = ColorPalette(self.settings['color_count'])
        
        # Performance statistics
        self._stats = {
            'total_renders': 0,
            'total_render_time': 0.0,
            'render_times': [],
        }
    
    def render_matrix(self, color_matrix: np.ndarray) -> Bitmap:
        """Render a color index matrix to a Bitmap.
        
        Args:
            color_matrix: 2D numpy array of color indices
            
        Returns:
            Bitmap object containing the rendered symbol
            
        Raises:
            ValueError: For invalid matrix input
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_matrix(color_matrix)
            
            # Apply scaling (module size)
            scaled_matrix = self._scale_matrix(color_matrix)
            
            # Apply quiet zone
            with_quiet_zone = self._add_quiet_zone(scaled_matrix)
            
            # Apply border if specified
            with_border = self._add_border(with_quiet_zone)
            
            # Convert to RGB
            rgb_array = self._matrix_to_rgb(with_border)
            
            # Create bitmap
            bitmap = Bitmap(
                array=rgb_array,
                width=rgb_array.shape[1],
                height=rgb_array.shape[0]
            )
            
            # Update statistics
            render_time = time.time() - start_time
            self._update_stats(render_time)
            
            return bitmap
            
        except Exception as e:
            raise ValueError(f"Matrix rendering failed: {str(e)}") from e
    
    def render_to_image(self, color_matrix: np.ndarray) -> Image.Image:
        """Render a color index matrix to a PIL Image.
        
        Args:
            color_matrix: 2D numpy array of color indices
            
        Returns:
            PIL Image object containing the rendered symbol
        """
        bitmap = self.render_matrix(color_matrix)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(bitmap.array, mode='RGB')
        
        return pil_image
    
    def render_symbol(self, symbol: Symbol, symbol_matrix: np.ndarray) -> Bitmap:
        """Render a complete Symbol object with its matrix data.
        
        Args:
            symbol: Symbol object with metadata
            symbol_matrix: 2D numpy array of the symbol's color data
            
        Returns:
            Bitmap object containing the rendered symbol
        """
        # Validate that matrix matches symbol dimensions
        if symbol_matrix.shape != symbol.matrix_size:
            raise ValueError(
                f"Matrix shape {symbol_matrix.shape} doesn't match "
                f"symbol size {symbol.matrix_size}"
            )
        
        # Update color palette to match symbol
        if symbol.color_count != self.color_palette.color_count:
            self.color_palette = ColorPalette(symbol.color_count)
        
        return self.render_matrix(symbol_matrix)
    
    def set_color_palette(self, palette: ColorPalette) -> None:
        """Set custom color palette for rendering.
        
        Args:
            palette: Custom color palette to use
            
        Raises:
            ValueError: If palette doesn't match renderer settings
        """
        if palette.color_count != self.settings['color_count']:
            # Update settings to match palette
            self.settings['color_count'] = palette.color_count
        
        self.color_palette = palette
    
    def _validate_matrix(self, matrix: np.ndarray) -> None:
        """Validate input matrix.
        
        Args:
            matrix: Matrix to validate
            
        Raises:
            ValueError: For invalid matrices
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Matrix must be a numpy array")
        
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2-dimensional")
        
        if matrix.size == 0:
            raise ValueError("Matrix cannot be empty")
        
        if matrix.dtype not in [np.uint8, np.int32, np.int64]:
            matrix = matrix.astype(np.uint8)
        
        # Check color indices are within palette range
        max_color = np.max(matrix)
        if max_color >= self.color_palette.color_count:
            # Could clamp values or raise error - we'll clamp for robustness
            matrix = np.clip(matrix, 0, self.color_palette.color_count - 1)
    
    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Scale matrix by module size.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Scaled matrix
        """
        module_size = self.settings['module_size']
        
        if module_size == 1:
            return matrix
        
        # Use numpy repeat to scale
        scaled = np.repeat(matrix, module_size, axis=0)  # Scale height
        scaled = np.repeat(scaled, module_size, axis=1)  # Scale width
        
        return scaled
    
    def _add_quiet_zone(self, matrix: np.ndarray) -> np.ndarray:
        """Add quiet zone around the matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix with quiet zone
        """
        quiet_zone = self.settings['quiet_zone']
        
        if quiet_zone <= 0:
            return matrix
        
        module_size = self.settings['module_size']
        quiet_zone_pixels = quiet_zone * module_size
        
        # Create larger matrix with quiet zone
        new_height = matrix.shape[0] + 2 * quiet_zone_pixels
        new_width = matrix.shape[1] + 2 * quiet_zone_pixels
        
        # Fill with background color index (0)
        padded_matrix = np.zeros((new_height, new_width), dtype=matrix.dtype)
        
        # Place original matrix in center
        padded_matrix[
            quiet_zone_pixels:-quiet_zone_pixels,
            quiet_zone_pixels:-quiet_zone_pixels
        ] = matrix
        
        return padded_matrix
    
    def _add_border(self, matrix: np.ndarray) -> np.ndarray:
        """Add border around the matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Matrix with border
        """
        border_width = self.settings['border_width']
        
        if border_width <= 0:
            return matrix
        
        # Create larger matrix with border
        new_height = matrix.shape[0] + 2 * border_width
        new_width = matrix.shape[1] + 2 * border_width
        
        # Create border matrix (using color index 0 for border)
        bordered_matrix = np.zeros((new_height, new_width), dtype=matrix.dtype)
        
        # Place original matrix in center
        bordered_matrix[
            border_width:-border_width,
            border_width:-border_width
        ] = matrix
        
        return bordered_matrix
    
    def _matrix_to_rgb(self, matrix: np.ndarray) -> np.ndarray:
        """Convert color index matrix to RGB array.
        
        Args:
            matrix: Color index matrix
            
        Returns:
            RGB array (height, width, 3)
        """
        height, width = matrix.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get color palette as RGB array
        palette_rgb = self.color_palette.to_rgb_array()
        
        # Handle background color for quiet zone/border (index 0)
        bg_color = np.array(self.settings['background_color'], dtype=np.uint8)
        
        # Map each color index to RGB
        for y in range(height):
            for x in range(width):
                color_index = matrix[y, x]
                
                if color_index == 0:
                    # Use background color for index 0
                    rgb_array[y, x] = bg_color
                elif color_index < len(palette_rgb):
                    rgb_array[y, x] = palette_rgb[color_index]
                else:
                    # Fallback for out-of-range indices
                    rgb_array[y, x] = palette_rgb[0]
        
        return rgb_array
    
    def _validate_settings(self) -> None:
        """Validate renderer settings."""
        if self.settings['module_size'] < 1:
            raise ValueError("Module size must be at least 1")
        
        if self.settings['quiet_zone'] < 0:
            raise ValueError("Quiet zone cannot be negative")
        
        if self.settings['border_width'] < 0:
            raise ValueError("Border width cannot be negative")
        
        bg_color = self.settings['background_color']
        if (not isinstance(bg_color, (tuple, list)) or 
            len(bg_color) != 3 or 
            not all(0 <= c <= 255 for c in bg_color)):
            raise ValueError("Background color must be (R, G, B) tuple with values 0-255")
        
        if self.settings['color_count'] not in [4, 8, 16, 32, 64, 128, 256]:
            raise ValueError("Invalid color count")
    
    def _update_stats(self, render_time: float) -> None:
        """Update rendering statistics.
        
        Args:
            render_time: Time taken for rendering
        """
        self._stats['total_renders'] += 1
        self._stats['total_render_time'] += render_time
        self._stats['render_times'].append(render_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        if self._stats['total_renders'] == 0:
            return {
                'total_renders': 0,
                'total_render_time': 0.0,
                'avg_render_time': 0.0,
            }
        
        return {
            'total_renders': self._stats['total_renders'],
            'total_render_time': self._stats['total_render_time'],
            'avg_render_time': (
                self._stats['total_render_time'] / self._stats['total_renders']
            ),
            'min_render_time': min(self._stats['render_times']),
            'max_render_time': max(self._stats['render_times']),
        }
    
    def reset(self) -> None:
        """Reset renderer statistics and state."""
        self._stats = {
            'total_renders': 0,
            'total_render_time': 0.0,
            'render_times': [],
        }
    
    def copy(self) -> 'BitmapRenderer':
        """Create a copy of this renderer.
        
        Returns:
            New BitmapRenderer instance with same settings
        """
        return BitmapRenderer(self.settings.copy())
    
    def __str__(self) -> str:
        """String representation of renderer."""
        return (
            f"BitmapRenderer(module_size={self.settings['module_size']}, "
            f"quiet_zone={self.settings['quiet_zone']}, "
            f"color_count={self.settings['color_count']})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"BitmapRenderer(settings={self.settings})"
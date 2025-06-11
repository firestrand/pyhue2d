"""Module Data Extractor for JABCode decoding.

This module provides the ModuleDataExtractor class for extracting and processing
module data from sampled JABCode symbols, including color decoding, demasking,
and data organization based on the JABCode reference implementation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import math

from .core import Point2D
from .color_palette import ColorPalette
from .constants import DEFAULT_8_COLOR_PALETTE, DEFAULT_4_COLOR_PALETTE


class ModuleDataExtractor:
    """Module Data Extractor for JABCode decoding.
    
    Extracts and processes module data from sampled JABCode symbols,
    implementing color decoding, demasking, and data organization
    algorithms based on the JABCode reference implementation.
    """
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize module data extractor.
        
        Args:
            settings: Optional configuration settings
        """
        default_settings = {
            'color_decoding_method': 'hard_decision',  # 'hard_decision', 'soft_decision'
            'normalization_method': 'max_channel',     # 'max_channel', 'luminance'
            'black_threshold': 50,                     # Threshold for black module detection
            'interpolation_enabled': True,             # Enable color palette interpolation
            'demasking_enabled': True,                 # Enable demasking operations
        }
        
        self.settings = default_settings
        if settings:
            self.settings.update(settings)
        
        # Module processing statistics
        self._stats = {
            'modules_processed': 0,
            'color_decode_errors': 0,
            'mask_operations': 0,
            'interpolation_operations': 0,
        }
    
    def extract_module_data(self, sampled_symbol: np.ndarray,
                          symbol_metadata: Dict[str, Any]) -> List[int]:
        """Extract module data from sampled symbol.
        
        Args:
            sampled_symbol: 2D array of sampled RGB module values
            symbol_metadata: Metadata containing symbol parameters
            
        Returns:
            List of color indices for data modules
            
        Raises:
            ValueError: If symbol format is invalid
        """
        height, width = sampled_symbol.shape[:2]
        color_count = symbol_metadata.get('color_count', 8)
        mask_pattern = symbol_metadata.get('mask_pattern', 0)
        
        # Step 1: Read color palettes from symbol
        color_palettes = self._read_color_palettes(sampled_symbol, symbol_metadata)
        
        # Step 2: Create data map to identify data-bearing modules
        data_map = self._create_data_map(width, height, symbol_metadata)
        
        # Step 3: Decode module colors to indices
        color_indices = self._decode_module_colors(
            sampled_symbol, color_palettes, color_count
        )
        
        # Step 4: Filter to data modules only
        data_indices = self._filter_data_modules(color_indices, data_map)
        
        # Step 5: Apply demasking if enabled
        if self.settings['demasking_enabled']:
            data_indices = self._demask_data(data_indices, mask_pattern, 
                                           color_count, symbol_metadata)
        
        self._stats['modules_processed'] += len(data_indices)
        return data_indices
    
    def _read_color_palettes(self, sampled_symbol: np.ndarray,
                           metadata: Dict[str, Any]) -> List[np.ndarray]:
        """Read color palettes from symbol regions.
        
        Based on JABCode reference implementation's readColorPaletteInMaster/Slave.
        
        Args:
            sampled_symbol: Sampled symbol data
            metadata: Symbol metadata
            
        Returns:
            List of 4 color palettes (one for each quadrant)
        """
        color_count = metadata.get('color_count', 8)
        is_master = metadata.get('is_master', True)
        height, width = sampled_symbol.shape[:2]
        
        # Initialize 4 regional color palettes
        palettes = []
        
        if is_master:
            # Master symbol: read from finder pattern corners and metadata
            palettes = self._read_master_color_palettes(sampled_symbol, color_count)
        else:
            # Slave symbol: use predefined positions
            palettes = self._read_slave_color_palettes(sampled_symbol, color_count)
        
        # Apply interpolation for higher color counts
        if color_count > 64 and self.settings['interpolation_enabled']:
            palettes = self._interpolate_color_palettes(palettes, color_count)
        
        self._stats['interpolation_operations'] += 1
        return palettes
    
    def _read_master_color_palettes(self, sampled_symbol: np.ndarray,
                                  color_count: int) -> List[np.ndarray]:
        """Read color palettes from master symbol.
        
        Args:
            sampled_symbol: Sampled symbol data
            color_count: Number of colors
            
        Returns:
            List of 4 color palettes
        """
        height, width = sampled_symbol.shape[:2]
        
        # Sample color values from finder pattern corners and metadata modules
        # This is a simplified version - full implementation would read from
        # specific metadata module positions
        
        palettes = []
        
        # Generate base palettes for each quadrant
        for quadrant in range(4):
            # Get representative colors from quadrant
            if quadrant == 0:  # Top-left
                region = sampled_symbol[7:height//2, 7:width//2]
            elif quadrant == 1:  # Top-right
                region = sampled_symbol[7:height//2, width//2:-7]
            elif quadrant == 2:  # Bottom-right
                region = sampled_symbol[height//2:-7, width//2:-7]
            else:  # Bottom-left
                region = sampled_symbol[height//2:-7, 7:width//2]
            
            # Extract color palette from region
            palette = self._extract_palette_from_region(region, color_count)
            palettes.append(palette)
        
        return palettes
    
    def _read_slave_color_palettes(self, sampled_symbol: np.ndarray,
                                 color_count: int) -> List[np.ndarray]:
        """Read color palettes from slave symbol.
        
        Args:
            sampled_symbol: Sampled symbol data
            color_count: Number of colors
            
        Returns:
            List of 4 color palettes
        """
        # Slave symbols use predefined positions for color sampling
        # This is a simplified implementation
        height, width = sampled_symbol.shape[:2]
        
        # Use standard color palette for slave symbols
        color_palette = ColorPalette(color_count)
        base_palette = np.array(color_palette.colors)
        
        # Create 4 identical palettes for slave symbols
        palettes = [base_palette for _ in range(4)]
        
        return palettes
    
    def _extract_palette_from_region(self, region: np.ndarray,
                                   color_count: int) -> np.ndarray:
        """Extract color palette from a symbol region.
        
        Args:
            region: Region to sample colors from
            color_count: Number of colors in palette
            
        Returns:
            Color palette array
        """
        # This is a simplified implementation
        # Real implementation would sample specific positions
        
        color_palette = ColorPalette(color_count)
        return np.array(color_palette.colors)
    
    def _interpolate_color_palettes(self, palettes: List[np.ndarray],
                                  color_count: int) -> List[np.ndarray]:
        """Interpolate color palettes for higher color counts.
        
        Based on JABCode reference implementation's palette interpolation.
        
        Args:
            palettes: Base color palettes
            color_count: Target color count
            
        Returns:
            Interpolated color palettes
        """
        interpolated_palettes = []
        
        for palette in palettes:
            if color_count <= 64:
                # No interpolation needed
                interpolated_palettes.append(palette)
            else:
                # Interpolate between base colors
                interpolated = self._perform_color_interpolation(palette, color_count)
                interpolated_palettes.append(interpolated)
        
        return interpolated_palettes
    
    def _perform_color_interpolation(self, base_palette: np.ndarray,
                                   target_count: int) -> np.ndarray:
        """Perform color interpolation between palette colors.
        
        Args:
            base_palette: Base color palette
            target_count: Target number of colors
            
        Returns:
            Interpolated color palette
        """
        # Simplified interpolation - real implementation would follow
        # JABCode's specific interpolation algorithm
        base_count = len(base_palette)
        
        if target_count <= base_count:
            return base_palette[:target_count]
        
        # Linear interpolation between colors
        interpolated = np.zeros((target_count, 3), dtype=np.uint8)
        
        for i in range(target_count):
            # Map target index to base palette range
            base_idx = (i * (base_count - 1)) / (target_count - 1)
            lower_idx = int(base_idx)
            upper_idx = min(lower_idx + 1, base_count - 1)
            weight = base_idx - lower_idx
            
            # Interpolate RGB values
            if lower_idx == upper_idx:
                interpolated[i] = base_palette[lower_idx]
            else:
                interpolated[i] = (
                    (1 - weight) * base_palette[lower_idx] +
                    weight * base_palette[upper_idx]
                ).astype(np.uint8)
        
        return interpolated
    
    def _create_data_map(self, width: int, height: int,
                        metadata: Dict[str, Any]) -> np.ndarray:
        """Create data map identifying data-bearing modules.
        
        Args:
            width: Symbol width in modules
            height: Symbol height in modules  
            metadata: Symbol metadata
            
        Returns:
            Boolean array where True indicates data module
        """
        # Initialize all modules as data modules
        data_map = np.ones((height, width), dtype=bool)
        
        # Mark finder patterns as non-data (7x7 squares at corners)
        finder_size = 7
        
        # Top-left finder pattern
        data_map[:finder_size, :finder_size] = False
        
        # Top-right finder pattern
        data_map[:finder_size, -finder_size:] = False
        
        # Bottom-left finder pattern
        data_map[-finder_size:, :finder_size] = False
        
        # Bottom-right finder pattern
        data_map[-finder_size:, -finder_size:] = False
        
        # Mark alignment patterns as non-data
        # Simplified - real implementation would use alignment pattern positions
        # from version calculator
        
        return data_map
    
    def _decode_module_colors(self, sampled_symbol: np.ndarray,
                            color_palettes: List[np.ndarray],
                            color_count: int) -> np.ndarray:
        """Decode module colors to color indices.
        
        Based on JABCode reference implementation's decodeModuleHD.
        
        Args:
            sampled_symbol: Sampled symbol RGB values
            color_palettes: Regional color palettes
            color_count: Number of colors
            
        Returns:
            2D array of color indices
        """
        height, width = sampled_symbol.shape[:2]
        color_indices = np.zeros((height, width), dtype=np.int32)
        
        for y in range(height):
            for x in range(width):
                # Get module RGB value
                if len(sampled_symbol.shape) == 3:
                    rgb = sampled_symbol[y, x, :3]
                else:
                    # Grayscale - convert to RGB
                    gray_value = sampled_symbol[y, x]
                    rgb = np.array([gray_value, gray_value, gray_value])
                
                # Determine which quadrant palette to use
                palette_idx = self._get_palette_quadrant(x, y, width, height)
                palette = color_palettes[palette_idx]
                
                # Decode color using hard decision method
                if self.settings['color_decoding_method'] == 'hard_decision':
                    color_idx = self._decode_color_hard_decision(rgb, palette)
                else:
                    color_idx = self._decode_color_soft_decision(rgb, palette)
                
                color_indices[y, x] = color_idx
        
        return color_indices
    
    def _get_palette_quadrant(self, x: int, y: int, width: int, height: int) -> int:
        """Determine which quadrant palette to use for a module.
        
        Args:
            x, y: Module coordinates
            width, height: Symbol dimensions
            
        Returns:
            Quadrant index (0-3)
        """
        mid_x = width // 2
        mid_y = height // 2
        
        if x < mid_x and y < mid_y:
            return 0  # Top-left
        elif x >= mid_x and y < mid_y:
            return 1  # Top-right
        elif x >= mid_x and y >= mid_y:
            return 2  # Bottom-right
        else:
            return 3  # Bottom-left
    
    def _decode_color_hard_decision(self, rgb: np.ndarray,
                                  palette: np.ndarray) -> int:
        """Decode module color using hard decision method.
        
        Based on JABCode reference implementation's color normalization
        and nearest neighbor matching.
        
        Args:
            rgb: Module RGB values
            palette: Color palette
            
        Returns:
            Color index
        """
        # Check for black modules first
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if luminance < self.settings['black_threshold']:
            return 0  # Black color index
        
        # Normalize RGB by maximum channel value
        if self.settings['normalization_method'] == 'max_channel':
            rgb_max = max(rgb[0], rgb[1], rgb[2])
            if rgb_max > 0:
                normalized_rgb = rgb.astype(float) / rgb_max
            else:
                normalized_rgb = rgb.astype(float)
        else:
            # Luminance normalization
            if luminance > 0:
                normalized_rgb = rgb.astype(float) / luminance
            else:
                normalized_rgb = rgb.astype(float)
        
        # Find closest palette color using Euclidean distance
        min_distance = float('inf')
        best_idx = 0
        
        for i, palette_color in enumerate(palette):
            # Normalize palette color
            if self.settings['normalization_method'] == 'max_channel':
                pal_max = max(palette_color[0], palette_color[1], palette_color[2])
                if pal_max > 0:
                    normalized_pal = palette_color.astype(float) / pal_max
                else:
                    normalized_pal = palette_color.astype(float)
            else:
                pal_luminance = 0.299 * palette_color[0] + 0.587 * palette_color[1] + 0.114 * palette_color[2]
                if pal_luminance > 0:
                    normalized_pal = palette_color.astype(float) / pal_luminance
                else:
                    normalized_pal = palette_color.astype(float)
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((normalized_rgb - normalized_pal) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                best_idx = i
        
        return best_idx
    
    def _decode_color_soft_decision(self, rgb: np.ndarray,
                                  palette: np.ndarray) -> int:
        """Decode module color using soft decision method.
        
        Args:
            rgb: Module RGB values
            palette: Color palette
            
        Returns:
            Color index
        """
        # Simplified soft decision - could be enhanced with probability
        # distributions and confidence measures
        return self._decode_color_hard_decision(rgb, palette)
    
    def _filter_data_modules(self, color_indices: np.ndarray,
                           data_map: np.ndarray) -> List[int]:
        """Filter color indices to data modules only.
        
        Args:
            color_indices: 2D array of all color indices
            data_map: Boolean array indicating data modules
            
        Returns:
            List of color indices for data modules only
        """
        data_indices = []
        height, width = color_indices.shape
        
        # Scan column-wise, then row-wise (JABCode order)
        for x in range(width):
            for y in range(height):
                if data_map[y, x]:
                    data_indices.append(color_indices[y, x])
        
        return data_indices
    
    def _demask_data(self, data_indices: List[int], mask_pattern: int,
                    color_count: int, metadata: Dict[str, Any]) -> List[int]:
        """Apply demasking to data indices.
        
        Based on JABCode reference implementation's demaskSymbol function.
        
        Args:
            data_indices: List of masked color indices
            mask_pattern: Mask pattern index (0-7)
            color_count: Number of colors
            metadata: Symbol metadata
            
        Returns:
            List of demasked color indices
        """
        if not self.settings['demasking_enabled']:
            return data_indices
        
        demasked_indices = []
        
        # Apply demasking based on pattern
        for i, color_idx in enumerate(data_indices):
            # Calculate module position from index
            # This is simplified - real implementation would track x,y coordinates
            x = i % metadata.get('width', 21)
            y = i // metadata.get('width', 21)
            
            # Apply mask pattern
            mask_value = self._calculate_mask_value(x, y, mask_pattern, color_count)
            demasked_idx = color_idx ^ mask_value
            demasked_indices.append(demasked_idx)
        
        self._stats['mask_operations'] += len(data_indices)
        return demasked_indices
    
    def _calculate_mask_value(self, x: int, y: int, mask_pattern: int,
                            color_count: int) -> int:
        """Calculate mask value for a module position.
        
        Based on JABCode reference implementation's 8 mask patterns.
        
        Args:
            x, y: Module coordinates
            mask_pattern: Mask pattern index (0-7)
            color_count: Number of colors
            
        Returns:
            Mask value
        """
        if mask_pattern == 0:
            return (x + y) % color_count
        elif mask_pattern == 1:
            return x % color_count
        elif mask_pattern == 2:
            return y % color_count
        elif mask_pattern == 3:
            return (x + y) % 3 % color_count
        elif mask_pattern == 4:
            return (x // 2 + y // 3) % color_count
        elif mask_pattern == 5:
            return ((x * y) % 2 + (x * y) % 3) % color_count
        elif mask_pattern == 6:
            return ((x * x * y) % 7 + (2 * x * x + 2 * y) % 19) % color_count
        elif mask_pattern == 7:
            return ((x + y) % 7 + (x * y) % 13) % color_count
        else:
            return 0
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get module extraction statistics.
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            'modules_processed': self._stats['modules_processed'],
            'color_decode_errors': self._stats['color_decode_errors'],
            'mask_operations': self._stats['mask_operations'],
            'interpolation_operations': self._stats['interpolation_operations'],
        }
    
    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self._stats = {
            'modules_processed': 0,
            'color_decode_errors': 0,
            'mask_operations': 0,
            'interpolation_operations': 0,
        }
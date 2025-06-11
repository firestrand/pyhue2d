"""Multi-symbol cascade for JABCode implementation.

This module provides the MultiSymbolCascade class for handling large data
that needs to be split across multiple JABCode symbols in a master-slave
hierarchy according to ISO/IEC 23634:2022 specification.
"""

import time
import math
from typing import Union, Dict, Any, List, Optional
import numpy as np

from .encoder import JABCodeEncoder
from .core import EncodedData, Bitmap


class MultiSymbolCascade:
    """Multi-symbol cascade for large data encoding.
    
    The MultiSymbolCascade handles encoding of large data that exceeds the
    capacity of a single JABCode symbol by splitting it across multiple symbols
    in a master-slave hierarchy. The first symbol acts as the master and contains
    metadata about the entire cascade, while subsequent symbols are slaves
    containing data segments.
    
    According to JABCode specification, up to 61 symbols can be linked in a cascade.
    """
    
    DEFAULT_SETTINGS = {
        'max_symbols': 61,  # JABCode specification limit
        'color_count': 8,
        'ecc_level': 'M',
        'version': 'auto',
        'optimize': True,
        'chunk_size': 1024,
        'overlap_bytes': 16,  # Bytes of overlap between symbols for redundancy
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize multi-symbol cascade.
        
        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Validate settings
        self._validate_settings()
        
        self.max_symbols = self.settings['max_symbols']
        
        # Initialize encoder for individual symbols
        encoder_settings = {
            'color_count': self.settings['color_count'],
            'ecc_level': self.settings['ecc_level'],
            'version': self.settings['version'],
            'optimize': self.settings['optimize'],
        }
        self.encoder = JABCodeEncoder(encoder_settings)
        
        # Statistics
        self._stats = {
            'total_encodes': 0,
            'total_symbols_created': 0,
            'encoding_times': [],
            'symbol_counts': [],
        }
    
    def encode(self, data: Union[str, bytes]) -> List[EncodedData]:
        """Encode data into multiple JABCode symbols.
        
        Args:
            data: Input data to encode
            
        Returns:
            List of EncodedData objects, one for each symbol in the cascade
            
        Raises:
            ValueError: If data is too large for the maximum number of symbols
        """
        start_time = time.time()
        
        try:
            # Convert to bytes for consistent handling
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Estimate number of symbols needed
            estimated_symbols = self.estimate_symbol_count(data)
            if estimated_symbols > self.max_symbols:
                raise ValueError(
                    f"Data too large: estimated {estimated_symbols} symbols needed, "
                    f"maximum is {self.max_symbols}"
                )
            
            # Split data into segments
            data_segments = self._split_data(data_bytes)
            
            # Encode each segment
            encoded_symbols = []
            total_symbols = len(data_segments)
            
            for i, segment in enumerate(data_segments):
                # Create metadata for this symbol
                symbol_metadata = {
                    'symbol_index': i,
                    'total_symbols': total_symbols,
                    'is_master': (i == 0),
                    'data_segment': i,
                    'segment_start': sum(len(seg) for seg in data_segments[:i]),
                    'segment_end': sum(len(seg) for seg in data_segments[:i+1]),
                    'cascade_id': hash(data_bytes) % (2**31),  # Unique cascade identifier
                }
                
                # Add master-slave linking
                if i == 0:
                    # Master symbol contains slave references
                    symbol_metadata['slave_symbols'] = list(range(1, total_symbols))
                else:
                    # Slave symbols reference master
                    symbol_metadata['master_symbol_ref'] = 0
                
                # Encode this segment
                try:
                    encoded_segment = self.encoder.encode(segment)
                    
                    # Add cascade metadata to the encoded data
                    encoded_segment.metadata.update(symbol_metadata)
                    
                    encoded_symbols.append(encoded_segment)
                    
                except Exception as e:
                    raise ValueError(f"Failed to encode symbol {i}: {str(e)}") from e
            
            # Update statistics
            encoding_time = time.time() - start_time
            self._update_stats(encoding_time, len(encoded_symbols))
            
            return encoded_symbols
            
        except Exception as e:
            if "Data too large" in str(e):
                raise  # Re-raise size errors as-is
            raise ValueError(f"Multi-symbol encoding failed: {str(e)}") from e
    
    def encode_to_bitmaps(self, data: Union[str, bytes]) -> List[Bitmap]:
        """Encode data and generate bitmap representations.
        
        Args:
            data: Input data to encode
            
        Returns:
            List of Bitmap objects, one for each symbol in the cascade
        """
        encoded_symbols = self.encode(data)
        
        bitmaps = []
        for encoded_symbol in encoded_symbols:
            # Use the encoder to convert to bitmap
            # We need to temporarily store the encoded data and convert it
            bitmap = self.encoder.encode_to_bitmap(
                encoded_symbol.data[:100] if len(encoded_symbol.data) > 100 
                else encoded_symbol.data
            )
            bitmaps.append(bitmap)
        
        return bitmaps
    
    def estimate_symbol_count(self, data: Union[str, bytes]) -> int:
        """Estimate the number of symbols needed for given data.
        
        Args:
            data: Input data to estimate
            
        Returns:
            Estimated number of symbols needed
        """
        if isinstance(data, str):
            data_size = len(data.encode('utf-8'))
        else:
            data_size = len(data)
        
        if data_size == 0:
            return 1
        
        # Get approximate capacity per symbol
        symbol_capacity = self.calculate_symbol_capacity()
        
        # Estimate symbols needed with some overhead for metadata and redundancy
        estimated = math.ceil(data_size / symbol_capacity)
        
        # Add overhead for cascade metadata
        if estimated > 1:
            estimated = min(estimated + 1, self.max_symbols)
        
        return max(1, estimated)
    
    def calculate_symbol_capacity(self) -> int:
        """Calculate approximate data capacity per symbol.
        
        Returns:
            Approximate bytes per symbol
        """
        # This is a simplified estimation
        # In practice, capacity depends on version, color count, and ECC level
        base_capacity = 500  # More realistic estimate for JABCode symbols
        
        # Adjust based on color count (more colors = more bits per module)
        color_factor = math.log2(self.settings['color_count']) / 3  # Normalize to 8 colors
        
        # Adjust based on ECC level (higher ECC = less data capacity)
        ecc_factors = {'L': 1.2, 'M': 1.0, 'Q': 0.8, 'H': 0.6}
        ecc_factor = ecc_factors.get(self.settings['ecc_level'], 1.0)
        
        estimated_capacity = int(base_capacity * color_factor * ecc_factor)
        
        return max(200, estimated_capacity)  # Minimum reasonable capacity
    
    def _split_data(self, data: bytes) -> List[bytes]:
        """Split data into segments for multiple symbols.
        
        Args:
            data: Data to split
            
        Returns:
            List of data segments
        """
        if len(data) == 0:
            return [b""]
        
        symbol_capacity = self.calculate_symbol_capacity()
        overlap_bytes = self.settings.get('overlap_bytes', 0)
        
        # Calculate effective capacity per symbol (accounting for overlap)
        effective_capacity = symbol_capacity - overlap_bytes
        if effective_capacity <= 0:
            effective_capacity = symbol_capacity // 2
        
        segments = []
        offset = 0
        
        while offset < len(data):
            # Calculate segment end
            segment_end = min(offset + effective_capacity, len(data))
            
            # Add overlap from previous segment (except for first segment)
            segment_start = max(0, offset - overlap_bytes) if offset > 0 else 0
            
            segment = data[segment_start:segment_end]
            segments.append(segment)
            
            offset = segment_end
            
            # Safety check to prevent infinite loops
            if len(segments) >= self.max_symbols:
                if offset < len(data):
                    raise ValueError(
                        f"Data too large: requires more than {self.max_symbols} symbols"
                    )
                break
        
        return segments
    
    def _validate_settings(self) -> None:
        """Validate cascade settings."""
        if not (1 <= self.settings['max_symbols'] <= 61):
            raise ValueError("max_symbols must be between 1 and 61")
        
        valid_color_counts = [4, 8, 16, 32, 64, 128, 256]
        if self.settings['color_count'] not in valid_color_counts:
            raise ValueError(f"Invalid color count: {self.settings['color_count']}")
        
        valid_ecc_levels = ['L', 'M', 'Q', 'H']
        if self.settings['ecc_level'] not in valid_ecc_levels:
            raise ValueError(f"Invalid ECC level: {self.settings['ecc_level']}")
    
    def _update_stats(self, encoding_time: float, symbol_count: int) -> None:
        """Update encoding statistics.
        
        Args:
            encoding_time: Time taken for encoding
            symbol_count: Number of symbols created
        """
        self._stats['total_encodes'] += 1
        self._stats['total_symbols_created'] += symbol_count
        self._stats['encoding_times'].append(encoding_time)
        self._stats['symbol_counts'].append(symbol_count)
    
    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get encoding statistics.
        
        Returns:
            Dictionary of encoding statistics
        """
        if self._stats['total_encodes'] == 0:
            return {
                'total_encodes': 0,
                'total_symbols_created': 0,
                'avg_symbols_per_encode': 0.0,
                'avg_encoding_time': 0.0,
            }
        
        return {
            'total_encodes': self._stats['total_encodes'],
            'total_symbols_created': self._stats['total_symbols_created'],
            'avg_symbols_per_encode': (
                self._stats['total_symbols_created'] / self._stats['total_encodes']
            ),
            'avg_encoding_time': sum(self._stats['encoding_times']) / len(self._stats['encoding_times']),
            'min_symbols': min(self._stats['symbol_counts']) if self._stats['symbol_counts'] else 0,
            'max_symbols': max(self._stats['symbol_counts']) if self._stats['symbol_counts'] else 0,
        }
    
    def reset(self) -> None:
        """Reset cascade statistics and state."""
        self._stats = {
            'total_encodes': 0,
            'total_symbols_created': 0,
            'encoding_times': [],
            'symbol_counts': [],
        }
        self.encoder.reset()
    
    def copy(self) -> 'MultiSymbolCascade':
        """Create a copy of this cascade.
        
        Returns:
            New MultiSymbolCascade instance with same settings
        """
        return MultiSymbolCascade(self.settings.copy())
    
    def __str__(self) -> str:
        """String representation of cascade."""
        return (
            f"MultiSymbolCascade(max_symbols={self.max_symbols}, "
            f"color_count={self.settings['color_count']}, "
            f"ecc_level={self.settings['ecc_level']})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MultiSymbolCascade(settings={self.settings})"
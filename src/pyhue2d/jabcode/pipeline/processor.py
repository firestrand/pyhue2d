"""Data processor for JABCode encoding pipeline.

This module provides the DataProcessor class which handles the initial processing
of input data, including encoding mode selection, data chunking, and optimization.
"""

import time
from typing import Union, Dict, Any, List, Optional
from ..core import EncodedData
from ..encoding_modes import (
    EncodingModeDetector,
    UppercaseMode, LowercaseMode, NumericMode, 
    PunctuationMode, MixedMode, AlphanumericMode, ByteMode
)


class DataProcessor:
    """Processes input data for JABCode encoding.
    
    The DataProcessor handles the first stage of the JABCode encoding pipeline,
    responsible for:
    - Input validation and normalization
    - Optimal encoding mode detection
    - Data chunking for large inputs
    - Encoding optimization
    - Metadata generation
    """
    
    DEFAULT_SETTINGS = {
        'default_encoding_mode': 'auto',
        'optimize_encoding': True,
        'chunk_size': 1024,
        'max_data_size': 65536,  # 64KB max
        'enable_compression': False,
        'compression_threshold': 512,
    }
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize data processor.
        
        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)
        
        # Initialize encoding modes
        self.encoding_modes = {
            'Uppercase': UppercaseMode(),
            'Lowercase': LowercaseMode(),
            'Numeric': NumericMode(),
            'Punctuation': PunctuationMode(),
            'Mixed': MixedMode(),
            'Alphanumeric': AlphanumericMode(),
            'Byte': ByteMode(),
        }
        
        # Also create lowercase mappings for convenience
        lowercase_modes = {
            'uppercase': self.encoding_modes['Uppercase'],
            'lowercase': self.encoding_modes['Lowercase'],
            'numeric': self.encoding_modes['Numeric'],
            'punctuation': self.encoding_modes['Punctuation'],
            'mixed': self.encoding_modes['Mixed'],
            'alphanumeric': self.encoding_modes['Alphanumeric'],
            'byte': self.encoding_modes['Byte'],
        }
        self.encoding_modes.update(lowercase_modes)
        
        # Initialize detector
        self.detector = EncodingModeDetector()
        
        # Statistics
        self._stats = {
            'total_processed': 0,
            'total_bytes': 0,
            'mode_usage': {mode: 0 for mode in self.encoding_modes.keys()}
        }
    
    def process(self, data: Union[str, bytes], encoding_mode: Optional[str] = None) -> EncodedData:
        """Process input data into encoded format.
        
        Args:
            data: Input data to process (string or bytes)
            encoding_mode: Force specific encoding mode (optional)
            
        Returns:
            EncodedData object containing processed data
            
        Raises:
            ValueError: For invalid input or oversized data
        """
        start_time = time.time()
        
        # Validate input
        if not self.validate_input(data):
            raise ValueError(f"Invalid input data type: {type(data)}")
        
        # Handle empty data
        if not data:
            return EncodedData(b"", {'encoding_mode': 'byte', 'data_size': 0, 
                                   'chunk_count': 0, 'processing_time': 0.0,
                                   'original_size': 0})
        
        # Convert to string for processing
        if isinstance(data, bytes):
            try:
                data_str = data.decode('utf-8')
            except UnicodeDecodeError:
                # Handle binary data that's not UTF-8
                data_str = data.hex()
                encoding_mode = 'byte'  # Force byte mode for binary
        else:
            data_str = data
        
        # Check data size limits
        data_bytes = data_str.encode('utf-8')
        if len(data_bytes) > self.settings['max_data_size']:
            raise ValueError(f"Data size {len(data_bytes)} exceeds maximum {self.settings['max_data_size']}")
        
        # Determine encoding mode
        if encoding_mode is None:
            if self.settings['default_encoding_mode'] == 'auto':
                encoding_mode = self.get_optimal_encoding_mode(data_str)
            else:
                encoding_mode = self.settings['default_encoding_mode']
        
        # Validate encoding mode
        if encoding_mode not in self.encoding_modes:
            raise ValueError(f"Invalid encoding mode: {encoding_mode}")
        
        # Get the encoding mode instance
        mode_instance = self.encoding_modes[encoding_mode]
        
        # Chunk data if necessary
        chunks = self.chunk_data(data_str)
        
        # Encode data
        encoded_chunks = []
        for chunk in chunks:
            if mode_instance.can_encode(chunk):
                encoded_chunk = mode_instance.encode(chunk)
                encoded_chunks.append(encoded_chunk)
            else:
                # Fall back to byte mode if current mode can't encode
                byte_mode = self.encoding_modes['Byte']
                encoded_chunk = byte_mode.encode(chunk)
                encoded_chunks.append(encoded_chunk)
                encoding_mode = 'Byte'  # Update mode for metadata
        
        # Combine encoded chunks
        if encoded_chunks:
            combined_encoded = b''.join(encoded_chunks)
        else:
            combined_encoded = b''
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate metadata
        metadata = {
            'encoding_mode': encoding_mode,
            'data_size': len(combined_encoded),
            'original_size': len(data_bytes),
            'chunk_count': len(chunks),
            'processing_time': processing_time,
            'optimization_enabled': self.settings['optimize_encoding'],
            'chunks_info': [len(chunk) for chunk in chunks]
        }
        
        # Update statistics
        self._stats['total_processed'] += 1
        self._stats['total_bytes'] += len(data_bytes)
        self._stats['mode_usage'][encoding_mode] += 1
        
        return EncodedData(combined_encoded, metadata)
    
    def get_optimal_encoding_mode(self, data: str) -> str:
        """Determine the optimal encoding mode for given data.
        
        Args:
            data: Input data string
            
        Returns:
            Name of optimal encoding mode
        """
        if not self.settings['optimize_encoding']:
            return 'byte'  # Default to byte mode if optimization disabled
        
        # Use the detector to find the best mode
        best_mode = self.detector.detect_best_mode(data)
        return best_mode.name
    
    def chunk_data(self, data: str) -> List[str]:
        """Split data into chunks if necessary.
        
        Args:
            data: Input data string
            
        Returns:
            List of data chunks
        """
        chunk_size = self.settings['chunk_size']
        
        if len(data) <= chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def calculate_encoding_efficiency(self, data: str, mode_name: str) -> float:
        """Calculate encoding efficiency for given mode.
        
        Args:
            data: Input data
            mode_name: Encoding mode name
            
        Returns:
            Efficiency score (higher is better)
        """
        if mode_name not in self.encoding_modes:
            return 0.0
        
        mode = self.encoding_modes[mode_name]
        
        if not mode.can_encode(data):
            return 0.0
        
        # Simple efficiency calculation based on compression ratio
        original_size = len(data.encode('utf-8'))
        if original_size == 0:
            return 1.0
        
        try:
            encoded = mode.encode(data)
            compressed_size = len(encoded)
            efficiency = original_size / compressed_size if compressed_size > 0 else 0.0
        except:
            efficiency = 0.0
        
        return efficiency
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(data, (str, bytes))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            'total_processed': 0,
            'total_bytes': 0,
            'mode_usage': {mode: 0 for mode in self.encoding_modes.keys()}
        }
    
    def copy(self) -> 'DataProcessor':
        """Create a copy of this processor.
        
        Returns:
            New DataProcessor instance with same settings
        """
        return DataProcessor(self.settings.copy())
    
    def __str__(self) -> str:
        """String representation of processor."""
        mode = self.settings.get('default_encoding_mode', 'auto')
        optimize = self.settings.get('optimize_encoding', True)
        return f"DataProcessor(mode={mode}, optimize={optimize})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"DataProcessor(settings={self.settings})"
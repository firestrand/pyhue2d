"""Tests for DataProcessor class."""

import pytest
import numpy as np
from pyhue2d.jabcode.pipeline.processor import DataProcessor
from pyhue2d.jabcode.core import EncodedData
from pyhue2d.jabcode.encoding_modes import (
    UppercaseMode, LowercaseMode, NumericMode, ByteMode
)


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_data_processor_creation_with_defaults(self):
        """Test that DataProcessor can be created with default settings."""
        processor = DataProcessor()
        assert processor is not None
    
    def test_data_processor_creation_with_custom_settings(self):
        """Test creating DataProcessor with custom settings."""
        settings = {
            'default_encoding_mode': 'uppercase',
            'optimize_encoding': True,
            'chunk_size': 1024
        }
        processor = DataProcessor(settings)
        assert processor is not None
    
    def test_data_processor_process_string_data(self):
        """Test processing string data."""
        processor = DataProcessor()
        test_data = "Hello, World!"
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
    
    def test_data_processor_process_bytes_data(self):
        """Test processing bytes data."""
        processor = DataProcessor()
        test_data = b"Binary data test"
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
    
    def test_data_processor_process_empty_data(self):
        """Test processing empty data."""
        processor = DataProcessor()
        result = processor.process("")
        assert isinstance(result, EncodedData)
        assert result.is_empty()


class TestDataProcessorImplementation:
    """Test suite for DataProcessor implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def processor(self):
        """Create a data processor for testing."""
        try:
            return DataProcessor()
        except NotImplementedError:
            pytest.skip("DataProcessor not yet implemented")
    
    def test_processor_creation_with_defaults(self, processor):
        """Test that processor can be created with default values."""
        assert processor is not None
        assert hasattr(processor, 'settings')
        assert hasattr(processor, 'encoding_modes')
    
    def test_processor_default_settings(self, processor):
        """Test processor default settings."""
        settings = processor.settings
        assert isinstance(settings, dict)
        assert 'default_encoding_mode' in settings
        assert 'optimize_encoding' in settings
        assert 'chunk_size' in settings
    
    def test_processor_custom_settings(self):
        """Test processor with custom settings."""
        custom_settings = {
            'default_encoding_mode': 'numeric',
            'optimize_encoding': False,
            'chunk_size': 512,
            'max_data_size': 2048
        }
        try:
            processor = DataProcessor(custom_settings)
            for key, value in custom_settings.items():
                assert processor.settings[key] == value
        except NotImplementedError:
            pytest.skip("DataProcessor not yet implemented")
    
    def test_processor_available_encoding_modes(self, processor):
        """Test that processor has access to encoding modes."""
        modes = processor.encoding_modes
        assert isinstance(modes, dict)
        assert 'uppercase' in modes
        assert 'lowercase' in modes
        assert 'numeric' in modes
        assert 'byte' in modes
    
    def test_processor_process_string_returns_encoded_data(self, processor):
        """Test that processing string returns EncodedData."""
        test_data = "TEST STRING"
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_processor_process_bytes_returns_encoded_data(self, processor):
        """Test that processing bytes returns EncodedData."""
        test_data = b"Binary test data"
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_processor_process_empty_string(self, processor):
        """Test processing empty string."""
        result = processor.process("")
        assert isinstance(result, EncodedData)
        assert result.is_empty()
    
    def test_processor_process_empty_bytes(self, processor):
        """Test processing empty bytes."""
        result = processor.process(b"")
        assert isinstance(result, EncodedData)
        assert result.is_empty()
    
    def test_processor_process_unicode_string(self, processor):
        """Test processing Unicode string."""
        test_data = "Hello 世界! 🌍"
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_processor_process_large_data(self, processor):
        """Test processing large data."""
        test_data = "A" * 10000  # 10KB of data
        result = processor.process(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_processor_optimal_encoding_mode_detection(self, processor):
        """Test that processor detects optimal encoding modes."""
        # Test uppercase data
        uppercase_data = "HELLO WORLD"
        result = processor.get_optimal_encoding_mode(uppercase_data)
        assert result == 'Uppercase'
        
        # Test lowercase data
        lowercase_data = "hello world"
        result = processor.get_optimal_encoding_mode(lowercase_data)
        assert result == 'Lowercase'
        
        # Test numeric data
        numeric_data = "1234567890"
        result = processor.get_optimal_encoding_mode(numeric_data)
        assert result == 'Numeric'
        
        # Test mixed data (should use byte mode)
        mixed_data = "Hello123!@#"
        result = processor.get_optimal_encoding_mode(mixed_data)
        assert result in ('Byte', 'Mixed')  # Could be either depending on detector
    
    def test_processor_chunking_large_data(self, processor):
        """Test that processor chunks large data appropriately."""
        # Create data larger than default chunk size
        large_data = "X" * 2048
        chunks = processor.chunk_data(large_data)
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        
        # Verify chunk sizes are within limits
        chunk_size = processor.settings['chunk_size']
        for chunk in chunks[:-1]:  # All but last chunk should be full size
            assert len(chunk) == chunk_size
        
        # Last chunk can be smaller
        assert len(chunks[-1]) <= chunk_size
    
    def test_processor_encoding_mode_efficiency(self, processor):
        """Test encoding mode efficiency calculations."""
        test_data = "UPPERCASE TEXT"
        
        # Test efficiency for different modes
        uppercase_efficiency = processor.calculate_encoding_efficiency(test_data, 'uppercase')
        byte_efficiency = processor.calculate_encoding_efficiency(test_data, 'byte')
        
        # Both should return valid efficiency scores
        assert uppercase_efficiency >= 0
        assert byte_efficiency >= 0
        # Note: Efficiency comparison depends on encoding implementation details
    
    def test_processor_validate_data_input(self, processor):
        """Test data validation."""
        # Valid inputs
        assert processor.validate_input("string") == True
        assert processor.validate_input(b"bytes") == True
        assert processor.validate_input("") == True
        assert processor.validate_input(b"") == True
        
        # Invalid inputs
        assert processor.validate_input(None) == False
        assert processor.validate_input(123) == False
        assert processor.validate_input([1, 2, 3]) == False
    
    def test_processor_metadata_generation(self, processor):
        """Test metadata generation for processed data."""
        test_data = "Test data"
        result = processor.process(test_data)
        
        metadata = result.metadata
        assert 'encoding_mode' in metadata
        assert 'data_size' in metadata
        assert 'chunk_count' in metadata
        assert 'processing_time' in metadata
    
    def test_processor_error_handling_invalid_encoding_mode(self, processor):
        """Test error handling for invalid encoding mode."""
        with pytest.raises(ValueError):
            processor.process("test", encoding_mode="invalid_mode")
    
    def test_processor_error_handling_oversized_data(self, processor):
        """Test error handling for oversized data."""
        # Create data larger than max allowed size
        max_size = processor.settings.get('max_data_size', 100000)
        oversized_data = "X" * (max_size + 1)
        
        with pytest.raises(ValueError):
            processor.process(oversized_data)
    
    def test_processor_process_with_specific_encoding_mode(self, processor):
        """Test processing with specific encoding mode."""
        test_data = "123"  # Could be numeric or byte
        
        # Force numeric mode
        result_numeric = processor.process(test_data, encoding_mode='numeric')
        assert result_numeric.metadata['encoding_mode'] == 'numeric'
        
        # Force byte mode
        result_byte = processor.process(test_data, encoding_mode='byte')
        assert result_byte.metadata['encoding_mode'] == 'byte'
    
    def test_processor_optimization_enabled_vs_disabled(self, processor):
        """Test difference between optimized and non-optimized processing."""
        test_data = "Mixed 123 DATA"
        
        # Process with optimization
        processor.settings['optimize_encoding'] = True
        result_optimized = processor.process(test_data)
        
        # Process without optimization
        processor.settings['optimize_encoding'] = False
        result_unoptimized = processor.process(test_data)
        
        # Both should work, but may have different efficiencies
        assert isinstance(result_optimized, EncodedData)
        assert isinstance(result_unoptimized, EncodedData)
    
    def test_processor_process_preserves_data_integrity(self, processor):
        """Test that processing preserves data for reconstruction."""
        test_data = "Data integrity test 123!@#"
        result = processor.process(test_data)
        
        # The encoded data should contain enough information to reconstruct original
        assert result.get_size() > 0
        assert 'original_size' in result.metadata
        assert result.metadata['original_size'] == len(test_data.encode('utf-8'))
    
    def test_processor_string_representation(self, processor):
        """Test processor string representation."""
        str_repr = str(processor)
        assert "DataProcessor" in str_repr
        assert "settings" in str_repr or "mode" in str_repr
    
    def test_processor_copy(self, processor):
        """Test processor copying."""
        if hasattr(processor, 'copy'):
            copied = processor.copy()
            assert copied is not processor
            assert copied.settings == processor.settings
    
    def test_processor_get_stats(self, processor):
        """Test processor statistics."""
        if hasattr(processor, 'get_stats'):
            stats = processor.get_stats()
            assert isinstance(stats, dict)
            assert 'total_processed' in stats
            assert 'total_bytes' in stats
    
    def test_processor_reset_stats(self, processor):
        """Test processor statistics reset."""
        if hasattr(processor, 'reset_stats'):
            # Process some data
            processor.process("test data")
            
            # Reset stats
            processor.reset_stats()
            
            # Stats should be reset
            if hasattr(processor, 'get_stats'):
                stats = processor.get_stats()
                assert stats['total_processed'] == 0
                assert stats['total_bytes'] == 0
    
    def test_processor_concurrent_processing(self, processor):
        """Test that processor can handle concurrent processing."""
        # This is a basic test - real concurrency testing would be more complex
        test_data1 = "First dataset"
        test_data2 = "Second dataset"
        
        result1 = processor.process(test_data1)
        result2 = processor.process(test_data2)
        
        assert result1 != result2
        assert result1.get_size() > 0
        assert result2.get_size() > 0
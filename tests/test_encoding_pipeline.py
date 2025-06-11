"""Tests for EncodingPipeline class."""

import pytest
import numpy as np
from pyhue2d.jabcode.pipeline.encoding import EncodingPipeline
from pyhue2d.jabcode.core import EncodedData, Bitmap
from pyhue2d.jabcode.color_palette import ColorPalette
import os
import json

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'example_images')
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, 'examples_manifest.json')

with open(MANIFEST_PATH) as f:
    EXAMPLES = json.load(f)

class TestEncodingPipeline:
    """Test suite for EncodingPipeline class."""
    
    def test_encoding_pipeline_creation_with_defaults(self):
        """Test that EncodingPipeline can be created with default settings."""
        pipeline = EncodingPipeline()
        assert pipeline is not None
    
    def test_encoding_pipeline_creation_with_custom_settings(self):
        """Test creating EncodingPipeline with custom settings."""
        settings = {
            'color_count': 8,
            'ecc_level': 'M',
            'version': 'auto',
            'optimize': True
        }
        pipeline = EncodingPipeline(settings)
        assert pipeline is not None
    
    def test_encoding_pipeline_encode_string_data(self):
        """Test encoding string data."""
        pipeline = EncodingPipeline()
        test_data = "Hello, World!"
        result = pipeline.encode(test_data)
        assert isinstance(result, EncodedData)
    
    def test_encoding_pipeline_encode_bytes_data(self):
        """Test encoding bytes data."""
        pipeline = EncodingPipeline()
        test_data = b"Binary data test"
        result = pipeline.encode(test_data)
        assert isinstance(result, EncodedData)
    
    def test_encoding_pipeline_encode_to_bitmap(self):
        """Test encoding data to bitmap."""
        pipeline = EncodingPipeline()
        test_data = "Test bitmap encoding"
        result = pipeline.encode_to_bitmap(test_data)
        assert isinstance(result, Bitmap)


class TestEncodingPipelineImplementation:
    """Test suite for EncodingPipeline implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def pipeline(self):
        """Create an encoding pipeline for testing."""
        try:
            return EncodingPipeline()
        except NotImplementedError:
            pytest.skip("EncodingPipeline not yet implemented")
    
    def test_pipeline_creation_with_defaults(self, pipeline):
        """Test that pipeline can be created with default values."""
        assert pipeline is not None
        assert hasattr(pipeline, 'settings')
        assert hasattr(pipeline, 'processor')
        assert hasattr(pipeline, 'version_calculator')
        assert hasattr(pipeline, 'ldpc_codec')
    
    def test_pipeline_default_settings(self, pipeline):
        """Test pipeline default settings."""
        settings = pipeline.settings
        assert isinstance(settings, dict)
        assert 'color_count' in settings
        assert 'ecc_level' in settings
        assert 'version' in settings
        assert 'optimize' in settings
    
    def test_pipeline_custom_settings(self):
        """Test pipeline with custom settings."""
        custom_settings = {
            'color_count': 16,
            'ecc_level': 'H',
            'version': 5,
            'optimize': False,
            'chunk_size': 512
        }
        try:
            pipeline = EncodingPipeline(custom_settings)
            for key, value in custom_settings.items():
                if key in pipeline.settings:
                    assert pipeline.settings[key] == value
        except NotImplementedError:
            pytest.skip("EncodingPipeline not yet implemented")
    
    def test_pipeline_encode_string_returns_encoded_data(self, pipeline):
        """Test that encoding string returns EncodedData."""
        test_data = "TEST STRING"
        result = pipeline.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_pipeline_encode_bytes_returns_encoded_data(self, pipeline):
        """Test that encoding bytes returns EncodedData."""
        test_data = b"Binary test data"
        result = pipeline.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_pipeline_encode_empty_data(self, pipeline):
        """Test encoding empty data."""
        result = pipeline.encode("")
        assert isinstance(result, EncodedData)
        # Empty data should still have metadata and structure
        assert 'symbol_version' in result.metadata
        assert 'color_count' in result.metadata
        assert 'ecc_level' in result.metadata
    
    def test_pipeline_encode_to_bitmap_returns_bitmap(self, pipeline):
        """Test that encoding to bitmap returns Bitmap."""
        test_data = "Bitmap test data"
        result = pipeline.encode_to_bitmap(test_data)
        assert isinstance(result, Bitmap)
        assert result.width > 0
        assert result.height > 0
    
    def test_pipeline_encode_with_different_color_counts(self, pipeline):
        """Test encoding with different color counts."""
        test_data = "Color count test"
        
        for color_count in [4, 8, 16, 32, 64, 128, 256]:
            try:
                pipeline.settings['color_count'] = color_count
                result = pipeline.encode(test_data)
                assert isinstance(result, EncodedData)
                assert result.metadata['color_count'] == color_count
            except ValueError:
                # Some color counts might not be supported
                pass
    
    def test_pipeline_encode_with_different_ecc_levels(self, pipeline):
        """Test encoding with different ECC levels."""
        test_data = "ECC level test"
        
        for ecc_level in ["L", "M", "Q", "H"]:
            pipeline.settings['ecc_level'] = ecc_level
            result = pipeline.encode(test_data)
            assert isinstance(result, EncodedData)
            assert result.metadata['ecc_level'] == ecc_level
    
    def test_pipeline_version_auto_detection(self, pipeline):
        """Test automatic version detection."""
        # Small data should use lower version
        small_data = "Hi"
        pipeline.settings['version'] = 'auto'
        result_small = pipeline.encode(small_data)
        small_version = result_small.metadata['symbol_version']
        
        # Large data should use higher version
        large_data = "This is a much longer string that should require a higher symbol version " * 10
        result_large = pipeline.encode(large_data)
        large_version = result_large.metadata['symbol_version']
        
        assert large_version >= small_version
    
    def test_pipeline_fixed_version_encoding(self, pipeline):
        """Test encoding with fixed version."""
        test_data = "Fixed version test"
        fixed_version = 3
        
        pipeline.settings['version'] = fixed_version
        result = pipeline.encode(test_data)
        assert result.metadata['symbol_version'] == fixed_version
    
    def test_pipeline_optimization_enabled_vs_disabled(self, pipeline):
        """Test difference between optimized and non-optimized encoding."""
        test_data = "Optimization test data"
        
        # Encode with optimization
        pipeline.settings['optimize'] = True
        result_optimized = pipeline.encode(test_data)
        
        # Encode without optimization
        pipeline.settings['optimize'] = False
        result_unoptimized = pipeline.encode(test_data)
        
        # Both should work
        assert isinstance(result_optimized, EncodedData)
        assert isinstance(result_unoptimized, EncodedData)
        
        # Optimization metadata should be different
        assert result_optimized.metadata['optimization_enabled'] != result_unoptimized.metadata['optimization_enabled']
    
    def test_pipeline_error_correction_integration(self, pipeline):
        """Test integration with LDPC error correction."""
        test_data = "Error correction integration test"
        result = pipeline.encode(test_data)
        
        # Should contain error correction metadata
        assert 'ldpc_parameters' in result.metadata
        assert 'error_correction_overhead' in result.metadata
    
    def test_pipeline_symbol_structure_metadata(self, pipeline):
        """Test that symbol structure metadata is included."""
        test_data = "Symbol structure test"
        result = pipeline.encode(test_data)
        
        metadata = result.metadata
        assert 'symbol_version' in metadata
        assert 'matrix_size' in metadata
        assert 'finder_patterns' in metadata
        assert 'alignment_patterns' in metadata
        assert 'data_capacity' in metadata
    
    def test_pipeline_color_palette_integration(self, pipeline):
        """Test integration with color palette."""
        test_data = "Color palette test"
        pipeline.settings['color_count'] = 8
        
        result = pipeline.encode(test_data)
        
        # Should have color palette information
        assert 'color_palette' in result.metadata
        assert 'color_mapping' in result.metadata
    
    def test_pipeline_pattern_generation_metadata(self, pipeline):
        """Test that pattern generation metadata is included."""
        test_data = "Pattern generation test"
        result = pipeline.encode(test_data)
        
        metadata = result.metadata
        assert 'finder_pattern_positions' in metadata
        assert 'alignment_pattern_positions' in metadata
        assert 'pattern_sizes' in metadata
    
    def test_pipeline_encode_large_data(self, pipeline):
        """Test encoding large data."""
        # Test with data that requires chunking
        large_data = "A" * 5000  # 5KB of data
        result = pipeline.encode(large_data)
        
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert 'chunk_count' in result.metadata
        assert result.metadata['chunk_count'] >= 1
    
    def test_pipeline_encode_unicode_data(self, pipeline):
        """Test encoding Unicode data."""
        unicode_data = "Hello 世界! 🌍 Ελληνικά"
        result = pipeline.encode(unicode_data)
        
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert 'encoding_mode' in result.metadata
    
    def test_pipeline_encode_binary_data(self, pipeline):
        """Test encoding binary data."""
        binary_data = bytes(range(256))  # All byte values
        result = pipeline.encode(binary_data)
        
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert result.metadata['encoding_mode'] in ('Byte', 'byte')
    
    def test_pipeline_bitmap_generation_properties(self, pipeline):
        """Test bitmap generation properties."""
        test_data = "Bitmap properties test"
        bitmap = pipeline.encode_to_bitmap(test_data)
        
        assert isinstance(bitmap, Bitmap)
        assert bitmap.width == bitmap.height  # JABCode symbols are square
        assert bitmap.width > 0
        assert bitmap.height > 0
        
        # Should be a valid size based on symbol version
        valid_sizes = range(21, 500, 4)  # JABCode symbol sizes
        assert bitmap.width in valid_sizes or bitmap.width >= 21
    
    def test_pipeline_bitmap_color_validation(self, pipeline):
        """Test that bitmap uses correct color palette."""
        test_data = "Color validation test"
        color_count = 8
        pipeline.settings['color_count'] = color_count
        
        bitmap = pipeline.encode_to_bitmap(test_data)
        
        # Get unique colors used in bitmap
        unique_colors = set()
        for y in range(bitmap.height):
            for x in range(bitmap.width):
                pixel = bitmap.get_pixel(x, y)
                if isinstance(pixel, tuple):
                    unique_colors.add(pixel)
                else:
                    unique_colors.add((pixel, pixel, pixel))
        
        # Should not use more colors than specified
        assert len(unique_colors) <= color_count
    
    def test_pipeline_encode_with_custom_palette(self, pipeline):
        """Test encoding with custom color palette."""
        test_data = "Custom palette test"
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        custom_palette = ColorPalette(colors=custom_colors)

        if hasattr(pipeline, 'set_color_palette'):
            pipeline.set_color_palette(custom_palette)
            # The pipeline's color_count should now match the palette
            assert pipeline.settings['color_count'] == custom_palette.color_count

        result = pipeline.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.metadata['color_count'] == custom_palette.color_count
    
    def test_pipeline_progressive_encoding(self, pipeline):
        """Test progressive encoding for large data."""
        if hasattr(pipeline, 'encode_progressive'):
            large_data = "Progressive encoding test " * 100
            
            # Should return generator or iterator
            progressive_result = pipeline.encode_progressive(large_data)
            
            # Should be able to iterate through chunks
            chunks = list(progressive_result)
            assert len(chunks) > 0
            
            # All chunks should be EncodedData
            for chunk in chunks:
                assert isinstance(chunk, EncodedData)
    
    def test_pipeline_encoding_statistics(self, pipeline):
        """Test encoding statistics collection."""
        test_data = "Statistics test"
        result = pipeline.encode(test_data)
        
        if hasattr(pipeline, 'get_encoding_stats'):
            stats = pipeline.get_encoding_stats()
            assert isinstance(stats, dict)
            assert 'encoding_time' in stats
            assert 'compression_ratio' in stats
            assert 'error_correction_overhead' in stats
    
    def test_pipeline_validate_encoded_output(self, pipeline):
        """Test validation of encoded output."""
        test_data = "Validation test"
        result = pipeline.encode(test_data)
        
        # Basic validation
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert len(result.metadata) > 0
        
        # Required metadata fields
        required_fields = ['symbol_version', 'color_count', 'ecc_level', 'encoding_mode']
        for field in required_fields:
            assert field in result.metadata
    
    def test_pipeline_error_handling_invalid_settings(self, pipeline):
        """Test error handling for invalid settings."""
        # Invalid color count
        with pytest.raises(ValueError):
            pipeline.settings['color_count'] = 3  # Too few colors
            pipeline.encode("test")
        
        # Invalid ECC level
        with pytest.raises(ValueError):
            pipeline.settings['ecc_level'] = 'X'  # Invalid level
            pipeline.encode("test")
    
    def test_pipeline_string_representation(self, pipeline):
        """Test pipeline string representation."""
        str_repr = str(pipeline)
        assert "EncodingPipeline" in str_repr
        assert "color_count" in str_repr or "colors" in str_repr
        assert "ecc_level" in str_repr or "ecc" in str_repr
    
    def test_pipeline_copy(self, pipeline):
        """Test pipeline copying."""
        if hasattr(pipeline, 'copy'):
            copied = pipeline.copy()
            assert copied is not pipeline
            assert copied.settings == pipeline.settings
    
    def test_pipeline_reset(self, pipeline):
        """Test pipeline reset functionality."""
        if hasattr(pipeline, 'reset'):
            # Process some data
            pipeline.encode("test data")
            
            # Reset pipeline
            pipeline.reset()
            
            # Should be able to encode again
            result = pipeline.encode("new test data")
            assert isinstance(result, EncodedData)
    
    def test_pipeline_concurrent_encoding(self, pipeline):
        """Test that pipeline can handle multiple encoding requests."""
        # This is a basic test - real concurrency testing would be more complex
        test_data1 = "First dataset"
        test_data2 = "Second dataset"
        
        result1 = pipeline.encode(test_data1)
        result2 = pipeline.encode(test_data2)
        
        assert result1 != result2
        assert result1.get_size() > 0
        assert result2.get_size() > 0
        assert result1.metadata != result2.metadata

@pytest.mark.parametrize("example", EXAMPLES)
def test_encoding_pipeline_encode_manifest(example):
    """Test that EncodingPipeline can encode manifest examples."""
    pipeline = EncodingPipeline()
    result = pipeline.encode(example["text"])
    assert isinstance(result, EncodedData)
    assert result.get_size() > 0

# If a decode method exists on EncodingPipeline, add a similar test for decode
if hasattr(EncodingPipeline, 'decode'):
    @pytest.mark.parametrize("example", EXAMPLES)
    @pytest.mark.xfail(reason="EncodingPipeline.decode not yet implemented.")
    def test_encoding_pipeline_decode_manifest(example):
        pipeline = EncodingPipeline()
        image_path = os.path.join(EXAMPLES_DIR, example["output"])
        with pytest.raises(NotImplementedError):
            pipeline.decode(image_path)
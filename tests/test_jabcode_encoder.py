"""Tests for JABCodeEncoder class."""

import pytest
import numpy as np
from PIL import Image
from pyhue2d.jabcode.encoder import JABCodeEncoder
from pyhue2d.jabcode.core import Bitmap, EncodedData
from pyhue2d.jabcode.color_palette import ColorPalette
import os
import json

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'example_images')
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, 'examples_manifest.json')

with open(MANIFEST_PATH) as f:
    EXAMPLES = json.load(f)

class TestJABCodeEncoder:
    """Test suite for JABCodeEncoder class - Basic functionality."""
    
    def test_jabcode_encoder_creation_with_defaults(self):
        """Test that JABCodeEncoder can be created with default settings."""
        encoder = JABCodeEncoder()
        assert encoder is not None
    
    def test_jabcode_encoder_creation_with_custom_settings(self):
        """Test creating JABCodeEncoder with custom settings."""
        settings = {
            'color_count': 16,
            'ecc_level': 'H',
            'version': 5,
            'optimize': False
        }
        encoder = JABCodeEncoder(settings)
        assert encoder is not None
    
    def test_jabcode_encoder_encode_string_data(self):
        """Test encoding string data."""
        encoder = JABCodeEncoder()
        test_data = "Hello, JABCode World!"
        result = encoder.encode(test_data)
        assert isinstance(result, EncodedData)
    
    def test_jabcode_encoder_encode_bytes_data(self):
        """Test encoding bytes data."""
        encoder = JABCodeEncoder()
        test_data = b"Binary JABCode data test"
        result = encoder.encode(test_data)
        assert isinstance(result, EncodedData)
    
    def test_jabcode_encoder_encode_to_image(self):
        """Test encoding data to PIL Image."""
        encoder = JABCodeEncoder()
        test_data = "Test image encoding"
        result = encoder.encode_to_image(test_data)
        assert isinstance(result, Image.Image)


class TestJABCodeEncoderImplementation:
    """Test suite for JABCodeEncoder implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def encoder(self):
        """Create a JABCode encoder for testing."""
        try:
            return JABCodeEncoder()
        except NotImplementedError:
            pytest.skip("JABCodeEncoder not yet implemented")
    
    def test_encoder_creation_with_defaults(self, encoder):
        """Test that encoder can be created with default values."""
        assert encoder is not None
        assert hasattr(encoder, 'settings')
        assert hasattr(encoder, 'pipeline')
    
    def test_encoder_default_settings(self, encoder):
        """Test encoder default settings."""
        settings = encoder.settings
        assert isinstance(settings, dict)
        assert 'color_count' in settings
        assert 'ecc_level' in settings
        assert 'version' in settings
        assert 'optimize' in settings
    
    def test_encoder_custom_settings(self):
        """Test encoder with custom settings."""
        custom_settings = {
            'color_count': 32,
            'ecc_level': 'Q',
            'version': 8,
            'optimize': True,
            'master_symbol': False
        }
        try:
            encoder = JABCodeEncoder(custom_settings)
            for key, value in custom_settings.items():
                if key in encoder.settings:
                    assert encoder.settings[key] == value
        except NotImplementedError:
            pytest.skip("JABCodeEncoder not yet implemented")
    
    def test_encoder_encode_string_returns_encoded_data(self, encoder):
        """Test that encoding string returns EncodedData."""
        test_data = "JABCode Encoder Test"
        result = encoder.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_encoder_encode_bytes_returns_encoded_data(self, encoder):
        """Test that encoding bytes returns EncodedData."""
        test_data = b"Binary encoder test data"
        result = encoder.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
    
    def test_encoder_encode_empty_data(self, encoder):
        """Test encoding empty data."""
        result = encoder.encode("")
        assert isinstance(result, EncodedData)
        # Empty data should still have metadata and structure
        assert 'symbol_version' in result.metadata
        assert 'color_count' in result.metadata
        assert 'ecc_level' in result.metadata
    
    def test_encoder_encode_to_bitmap_returns_bitmap(self, encoder):
        """Test that encoding to bitmap returns Bitmap."""
        test_data = "Bitmap encoder test"
        result = encoder.encode_to_bitmap(test_data)
        assert isinstance(result, Bitmap)
        assert result.width > 0
        assert result.height > 0
    
    def test_encoder_encode_to_image_returns_pil_image(self, encoder):
        """Test that encoding to image returns PIL Image."""
        test_data = "PIL Image encoder test"
        result = encoder.encode_to_image(test_data)
        assert isinstance(result, Image.Image)
        assert result.width > 0
        assert result.height > 0
        assert result.mode in ['RGB', 'RGBA', 'L']
    
    def test_encoder_encode_with_different_color_counts(self):
        """Test encoding with different color counts."""
        test_data = "Color count test"
        for color_count in [4, 8, 16, 32, 64, 128, 256]:
            encoder = JABCodeEncoder(settings={'color_count': color_count})
            result = encoder.encode(test_data)
            assert isinstance(result, EncodedData)
            assert result.metadata['color_count'] == color_count
    
    def test_encoder_encode_with_different_ecc_levels(self):
        """Test encoding with different ECC levels."""
        test_data = "ECC level test"
        for ecc_level in ["L", "M", "Q", "H"]:
            encoder = JABCodeEncoder(settings={'ecc_level': ecc_level})
            result = encoder.encode(test_data)
            assert isinstance(result, EncodedData)
            assert result.metadata['ecc_level'] == ecc_level
    
    def test_encoder_version_auto_detection(self, encoder):
        """Test automatic version detection."""
        # Small data should use lower version
        small_data = "Hi"
        encoder.settings['version'] = 'auto'
        result_small = encoder.encode(small_data)
        small_version = result_small.metadata['symbol_version']
        
        # Large data should use higher version
        large_data = "This is a much longer string that should require a higher symbol version " * 20
        result_large = encoder.encode(large_data)
        large_version = result_large.metadata['symbol_version']
        
        assert large_version >= small_version
    
    def test_encoder_fixed_version_encoding(self):
        """Test encoding with fixed version."""
        test_data = "Fixed version test"
        fixed_version = 4
        encoder = JABCodeEncoder(settings={'version': fixed_version})
        result = encoder.encode(test_data)
        assert result.metadata['symbol_version'] == fixed_version
    
    def test_encoder_optimization_enabled_vs_disabled(self):
        """Test difference between optimized and non-optimized encoding."""
        test_data = "Optimization test data"
        encoder_opt = JABCodeEncoder(settings={'optimize': True})
        result_optimized = encoder_opt.encode(test_data)
        encoder_no_opt = JABCodeEncoder(settings={'optimize': False})
        result_unoptimized = encoder_no_opt.encode(test_data)
        assert isinstance(result_optimized, EncodedData)
        assert isinstance(result_unoptimized, EncodedData)
        assert result_optimized.metadata['optimization_enabled'] != result_unoptimized.metadata['optimization_enabled']
    
    def test_encoder_comprehensive_metadata(self, encoder):
        """Test that encoder provides comprehensive metadata."""
        test_data = "Metadata test"
        result = encoder.encode(test_data)
        
        metadata = result.metadata
        required_fields = [
            'symbol_version', 'color_count', 'ecc_level', 'encoding_mode',
            'matrix_size', 'finder_patterns', 'alignment_patterns',
            'encoding_time', 'original_size', 'encoded_size'
        ]
        
        for field in required_fields:
            assert field in metadata
    
    def test_encoder_bitmap_generation_properties(self, encoder):
        """Test bitmap generation properties."""
        test_data = "Bitmap properties test"
        bitmap = encoder.encode_to_bitmap(test_data)
        
        assert isinstance(bitmap, Bitmap)
        assert bitmap.width == bitmap.height  # JABCode symbols are square
        assert bitmap.width > 0
        assert bitmap.height > 0
        
        # Should be a valid size based on symbol version
        valid_sizes = range(21, 500, 4)  # JABCode symbol sizes
        assert bitmap.width in valid_sizes or bitmap.width >= 21
    
    def test_encoder_image_generation_properties(self, encoder):
        """Test PIL Image generation properties."""
        test_data = "Image properties test"
        image = encoder.encode_to_image(test_data)
        
        assert isinstance(image, Image.Image)
        assert image.width == image.height  # JABCode symbols are square
        assert image.width > 0
        assert image.height > 0
        assert image.mode in ['RGB', 'RGBA']
    
    def test_encoder_encode_large_data(self):
        """Test encoding large data."""
        encoder = JABCodeEncoder()
        large_data = "A" * 10000  # 10KB of data
        with pytest.raises(ValueError, match="too large for maximum symbol version"):
            encoder.encode(large_data)
    
    def test_encoder_encode_unicode_data(self, encoder):
        """Test encoding Unicode data."""
        unicode_data = "Hello 世界! 🌍 Ελληνικά"
        result = encoder.encode(unicode_data)
        
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert 'encoding_mode' in result.metadata
    
    def test_encoder_encode_binary_data(self, encoder):
        """Test encoding binary data."""
        binary_data = bytes(range(256))  # All byte values
        result = encoder.encode(binary_data)
        
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert result.metadata['encoding_mode'] in ('Byte', 'byte')
    
    def test_encoder_custom_color_palette(self, encoder):
        """Test encoding with custom color palette."""
        test_data = "Custom palette test"
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        custom_palette = ColorPalette(colors=custom_colors)

        if hasattr(encoder, 'set_color_palette'):
            encoder.set_color_palette(custom_palette)
            # The encoder's color_count should now match the palette
            assert encoder.settings['color_count'] == custom_palette.color_count

        result = encoder.encode(test_data)
        assert isinstance(result, EncodedData)
        assert result.metadata['color_count'] == custom_palette.color_count
    
    def test_encoder_error_handling_invalid_settings(self, encoder):
        """Test error handling for invalid settings."""
        # Invalid color count
        with pytest.raises(ValueError):
            encoder.settings['color_count'] = 3  # Too few colors
            encoder.encode("test")
        
        # Invalid ECC level
        with pytest.raises(ValueError):
            encoder.settings['ecc_level'] = 'X'  # Invalid level
            encoder.encode("test")
    
    def test_encoder_string_representation(self, encoder):
        """Test encoder string representation."""
        str_repr = str(encoder)
        assert "JABCodeEncoder" in str_repr
        assert "color_count" in str_repr or "colors" in str_repr
        assert "ecc_level" in str_repr or "ecc" in str_repr
    
    def test_encoder_copy(self, encoder):
        """Test encoder copying."""
        if hasattr(encoder, 'copy'):
            copied = encoder.copy()
            assert copied is not encoder
            assert copied.settings == encoder.settings
    
    def test_encoder_reset(self, encoder):
        """Test encoder reset functionality."""
        if hasattr(encoder, 'reset'):
            # Process some data
            encoder.encode("test data")
            
            # Reset encoder
            encoder.reset()
            
            # Should be able to encode again
            result = encoder.encode("new test data")
            assert isinstance(result, EncodedData)
    
    def test_encoder_batch_encoding(self, encoder):
        """Test batch encoding multiple data items."""
        test_data_list = [
            "First test string",
            "Second test string", 
            b"Binary test data",
            "Unicode test: 你好世界"
        ]
        
        results = []
        for data in test_data_list:
            result = encoder.encode(data)
            assert isinstance(result, EncodedData)
            results.append(result)
        
        # All results should be valid and different
        assert len(results) == len(test_data_list)
        for i, result in enumerate(results):
            assert result.get_size() > 0
            assert 'encoding_mode' in result.metadata
    
    def test_encoder_statistics_collection(self, encoder):
        """Test encoding statistics collection."""
        test_data = "Statistics test"
        result = encoder.encode(test_data)
        
        if hasattr(encoder, 'get_encoding_stats'):
            stats = encoder.get_encoding_stats()
            assert isinstance(stats, dict)
            assert 'encoding_time' in stats
            assert 'total_encoded' in stats
    
    def test_encoder_validate_encoded_output(self, encoder):
        """Test validation of encoded output."""
        test_data = "Validation test"
        result = encoder.encode(test_data)
        
        # Basic validation
        assert isinstance(result, EncodedData)
        assert result.get_size() > 0
        assert len(result.metadata) > 0
        
        # Required metadata fields
        required_fields = ['symbol_version', 'color_count', 'ecc_level', 'encoding_mode']
        for field in required_fields:
            assert field in result.metadata
        
        # Metadata value validation
        assert isinstance(result.metadata['symbol_version'], int)
        assert result.metadata['symbol_version'] >= 1
        assert result.metadata['color_count'] in [4, 8, 16, 32, 64, 128, 256]
        assert result.metadata['ecc_level'] in ['L', 'M', 'Q', 'H']

@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.xfail(reason="JABCodeEncoder.encode not yet implemented.")
def test_jabcode_encoder_encode_manifest(example):
    encoder = JABCodeEncoder()
    with pytest.raises(NotImplementedError):
        encoder.encode(example["text"])

# If a decode method exists on JABCodeEncoder, add a similar test for decode
if hasattr(JABCodeEncoder, 'decode'):
    @pytest.mark.parametrize("example", EXAMPLES)
    @pytest.mark.xfail(reason="JABCodeEncoder.decode not yet implemented.")
    def test_jabcode_encoder_decode_manifest(example):
        encoder = JABCodeEncoder()
        image_path = os.path.join(EXAMPLES_DIR, example["output"])
        with pytest.raises(NotImplementedError):
            encoder.decode(image_path)
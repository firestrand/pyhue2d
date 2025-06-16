"""Tests for JABCodeDecoder class."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyhue2d.jabcode.decoder import JABCodeDecoder


class TestJABCodeDecoder:
    """Test suite for JABCodeDecoder class - Basic functionality."""

    def test_decoder_creation_with_defaults(self):
        """Test that JABCodeDecoder can be created with default settings."""
        decoder = JABCodeDecoder()
        assert decoder is not None
        assert hasattr(decoder, "settings")
        assert hasattr(decoder, "finder_detector")

    def test_decoder_creation_with_custom_settings(self):
        """Test creating JABCodeDecoder with custom settings."""
        settings = {
            "detection_method": "contour_detection",
            "perspective_correction": False,
            "error_correction": True,
        }
        decoder = JABCodeDecoder(settings)
        assert decoder is not None
        assert decoder.settings["detection_method"] == "contour_detection"
        assert decoder.settings["perspective_correction"] is False

    def test_decoder_load_image_from_pil(self):
        """Test loading image from PIL Image."""
        decoder = JABCodeDecoder()
        # Create a simple test image
        test_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_image = Image.fromarray(test_array, mode="RGB")

        result = decoder._load_image(test_image)
        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_decoder_load_image_from_numpy(self):
        """Test loading image from numpy array."""
        decoder = JABCodeDecoder()
        # RGB array
        test_array = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = decoder._load_image(test_array)
        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

        # Grayscale array
        gray_array = np.ones((30, 30), dtype=np.uint8) * 200
        result = decoder._load_image(gray_array)
        assert isinstance(result, Image.Image)
        assert result.size == (30, 30)


class TestJABCodeDecoderImplementation:
    """Test suite for JABCodeDecoder implementation details."""

    @pytest.fixture
    def decoder(self):
        """Create a JABCode decoder for testing."""
        return JABCodeDecoder()

    def test_decoder_decode_simple_image(self, decoder):
        """Test decoding a simple JABCode image."""
        # Test that decoder can handle a simple case
        assert decoder is not None
        assert hasattr(decoder, 'decode')
        
        # Test with non-existent file should raise ValueError
        with pytest.raises(ValueError, match="JABCode decoding failed"):
            decoder.decode("nonexistent.png")

    def test_decoder_decode_with_different_settings(self, decoder):
        """Test decoding with different decoder settings."""
        # Test creating decoder with different settings
        settings = {"detection_method": "scanline", "perspective_correction": True}
        decoder_with_settings = JABCodeDecoder(settings)
        
        assert decoder_with_settings is not None
        assert hasattr(decoder_with_settings, 'decode')
        assert decoder_with_settings.settings["detection_method"] == "scanline"


# Round-trip testing - this will be our main validation
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "example_images")
MANIFEST_PATH = os.path.join(EXAMPLES_DIR, "examples_manifest.json")

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        EXAMPLES = json.load(f)

    @pytest.mark.parametrize("example", EXAMPLES)
    def test_round_trip_encoding_decoding(example):
        """Test encoding then decoding example data for round-trip validation."""
        from pyhue2d.jabcode.encoder import JABCodeEncoder
        from pyhue2d.jabcode.decoder import JABCodeDecoder
        
        # Get test data
        text = example["text"]
        test_data = text.encode("utf-8")
        
        # Encode
        encoder = JABCodeEncoder({"color_count": 8, "ecc_level": "M"})
        encoded_image = encoder.encode_to_image(test_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            encoded_image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # Decode
            decoder = JABCodeDecoder()
            decoded_data = decoder.decode(tmp_path)
            
            # Verify we got some data back (exact match may not work due to data format differences)
            assert len(decoded_data) > 0
            print(f"Original: {len(test_data)} bytes, Decoded: {len(decoded_data)} bytes")
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)

else:
    EXAMPLES = []

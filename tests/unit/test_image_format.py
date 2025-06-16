"""Unit tests for image format detection and conversion."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest
from PIL import Image

from pyhue2d.jabcode.exceptions import JABCodeFormatError, JABCodeImageError
from pyhue2d.jabcode.image_format import ImageConverter, ImageFormatDetector


class TestImageFormatDetector:
    """Test cases for ImageFormatDetector class."""

    def test_initialization(self):
        """Test ImageFormatDetector initialization."""
        detector = ImageFormatDetector()
        assert detector is not None
        assert hasattr(detector, "SUPPORTED_FORMATS")

    def test_supported_formats_structure(self):
        """Test structure of supported formats."""
        detector = ImageFormatDetector()
        formats = detector.SUPPORTED_FORMATS

        # Test that all expected formats are present
        expected_formats = ["PNG", "JPEG", "BMP", "TIFF", "WEBP"]
        for fmt in expected_formats:
            assert fmt in formats

        # Test structure of format info
        for fmt, info in formats.items():
            assert "extensions" in info
            assert "mime_types" in info
            assert "magic_bytes" in info
            assert "description" in info
            assert "supports_transparency" in info
            assert "lossless" in info

            assert isinstance(info["extensions"], list)
            assert isinstance(info["mime_types"], list)
            assert isinstance(info["magic_bytes"], list)
            assert isinstance(info["supports_transparency"], bool)
            assert isinstance(info["lossless"], bool)

    def test_detect_by_extension(self, tmp_path):
        """Test format detection by file extension."""
        detector = ImageFormatDetector()

        # Test various extensions
        test_cases = [
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),
            ("test.jpeg", "JPEG"),
            ("test.bmp", "BMP"),
            ("test.tiff", "TIFF"),
            ("test.tif", "TIFF"),
            ("test.webp", "WEBP"),
        ]

        for filename, expected_format in test_cases:
            file_path = Path(filename)
            detected = detector._detect_by_extension(file_path)
            assert detected == expected_format

    def test_detect_by_extension_case_insensitive(self):
        """Test case-insensitive extension detection."""
        detector = ImageFormatDetector()

        test_cases = [
            ("test.PNG", "PNG"),
            ("test.JPG", "JPEG"),
            ("test.BMP", "BMP"),
        ]

        for filename, expected_format in test_cases:
            file_path = Path(filename)
            detected = detector._detect_by_extension(file_path)
            assert detected == expected_format

    def test_detect_by_extension_unknown(self):
        """Test extension detection with unknown extension."""
        detector = ImageFormatDetector()

        file_path = Path("test.unknown")
        detected = detector._detect_by_extension(file_path)
        assert detected is None

    def test_detect_by_magic_bytes_png(self, tmp_path):
        """Test magic byte detection for PNG."""
        detector = ImageFormatDetector()

        png_magic = b"\x89PNG\r\n\x1a\n"
        test_file = tmp_path / "test.png"
        test_file.write_bytes(png_magic + b"rest of file")

        detected = detector._detect_by_magic_bytes(test_file)
        assert detected == "PNG"

    def test_detect_by_magic_bytes_jpeg(self, tmp_path):
        """Test magic byte detection for JPEG."""
        detector = ImageFormatDetector()

        jpeg_magic = b"\xff\xd8\xff"
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(jpeg_magic + b"rest of file")

        detected = detector._detect_by_magic_bytes(test_file)
        assert detected == "JPEG"

    def test_detect_by_magic_bytes_bmp(self, tmp_path):
        """Test magic byte detection for BMP."""
        detector = ImageFormatDetector()

        bmp_magic = b"BM"
        test_file = tmp_path / "test.bmp"
        test_file.write_bytes(bmp_magic + b"rest of file")

        detected = detector._detect_by_magic_bytes(test_file)
        assert detected == "BMP"

    def test_detect_by_magic_bytes_unknown(self, tmp_path):
        """Test magic byte detection with unknown format."""
        detector = ImageFormatDetector()

        test_file = tmp_path / "test.unknown"
        test_file.write_bytes(b"unknown format")

        detected = detector._detect_by_magic_bytes(test_file)
        assert detected is None

    def test_detect_by_magic_bytes_io_error(self):
        """Test magic byte detection with I/O error."""
        detector = ImageFormatDetector()

        nonexistent_file = Path("/nonexistent/file.png")
        detected = detector._detect_by_magic_bytes(nonexistent_file)
        assert detected is None

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_detect_by_pil_success(self, mock_open):
        """Test PIL-based format detection success."""
        detector = ImageFormatDetector()

        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.format = "PNG"
        mock_open.return_value.__enter__.return_value = mock_image

        test_file = Path("test.png")
        detected = detector._detect_by_pil(test_file)
        assert detected == "PNG"

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_detect_by_pil_unsupported_format(self, mock_open):
        """Test PIL-based detection with unsupported format."""
        detector = ImageFormatDetector()

        # Mock PIL Image with unsupported format
        mock_image = MagicMock()
        mock_image.format = "GIF"  # Not in our supported formats
        mock_open.return_value.__enter__.return_value = mock_image

        test_file = Path("test.gif")
        detected = detector._detect_by_pil(test_file)
        assert detected is None

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_detect_by_pil_exception(self, mock_open):
        """Test PIL-based detection with exception."""
        detector = ImageFormatDetector()

        # Mock PIL to raise exception
        mock_open.side_effect = Exception("Cannot open image")

        test_file = Path("test.png")
        detected = detector._detect_by_pil(test_file)
        assert detected is None

    def test_detect_format_nonexistent_file(self):
        """Test format detection with nonexistent file."""
        detector = ImageFormatDetector()

        with pytest.raises(JABCodeImageError) as exc_info:
            detector.detect_format("/nonexistent/file.png")

        assert "Image file does not exist" in str(exc_info.value)

    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_pil")
    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_magic_bytes")
    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_extension")
    def test_detect_format_priority(self, mock_ext, mock_magic, mock_pil, tmp_path):
        """Test format detection method priority."""
        detector = ImageFormatDetector()

        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")

        # Set up mocks to return different formats
        mock_ext.return_value = "BMP"
        mock_magic.return_value = "JPEG"
        mock_pil.return_value = "PNG"

        # PIL should have highest priority
        detected = detector.detect_format(test_file)
        assert detected == "PNG"

    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_pil")
    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_magic_bytes")
    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector._detect_by_extension")
    def test_detect_format_fallback(self, mock_ext, mock_magic, mock_pil, tmp_path):
        """Test format detection fallback chain."""
        detector = ImageFormatDetector()

        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")

        # PIL fails, magic bytes should be used
        mock_pil.return_value = None
        mock_magic.return_value = "PNG"
        mock_ext.return_value = "BMP"

        detected = detector.detect_format(test_file)
        assert detected == "PNG"

    def test_detect_format_unsupported(self, tmp_path):
        """Test detection of unsupported format."""
        detector = ImageFormatDetector()

        # Mock all detection methods to return unsupported format
        test_file = tmp_path / "test.gif"
        test_file.write_bytes(b"GIF89a")

        with patch.object(detector, "_detect_by_pil", return_value="GIF"):
            with pytest.raises(JABCodeFormatError) as exc_info:
                detector.detect_format(test_file)

            assert "Unsupported image format: GIF" in str(exc_info.value)

    def test_is_format_supported(self):
        """Test format support checking."""
        detector = ImageFormatDetector()

        # Test supported formats
        assert detector.is_format_supported("PNG")
        assert detector.is_format_supported("png")  # Case insensitive
        assert detector.is_format_supported("JPEG")

        # Test unsupported format
        assert not detector.is_format_supported("GIF")
        assert not detector.is_format_supported("UNKNOWN")

    def test_get_format_info(self):
        """Test getting format information."""
        detector = ImageFormatDetector()

        png_info = detector.get_format_info("PNG")
        assert png_info["lossless"] is True
        assert png_info["supports_transparency"] is True
        assert ".png" in png_info["extensions"]

        # Test case insensitive
        jpeg_info = detector.get_format_info("jpeg")
        assert jpeg_info["lossless"] is False
        assert ".jpg" in jpeg_info["extensions"]

    def test_get_format_info_unsupported(self):
        """Test getting info for unsupported format."""
        detector = ImageFormatDetector()

        with pytest.raises(JABCodeFormatError) as exc_info:
            detector.get_format_info("GIF")

        assert "Unsupported format: GIF" in str(exc_info.value)

    def test_get_supported_formats(self):
        """Test getting all supported formats."""
        detector = ImageFormatDetector()

        formats = detector.get_supported_formats()
        assert isinstance(formats, dict)
        assert "PNG" in formats
        assert "JPEG" in formats

        # Ensure it's a copy, not reference
        formats["TEST"] = {}
        assert "TEST" not in detector.SUPPORTED_FORMATS

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_validate_image_success(self, mock_open, tmp_path):
        """Test successful image validation."""
        detector = ImageFormatDetector()

        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.format = "PNG"
        mock_image.mode = "RGB"
        mock_image.size = (100, 100)
        mock_image.width = 100
        mock_image.height = 100
        mock_open.return_value.__enter__.return_value = mock_image

        # Mock file stats
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake png data")

        with patch("numpy.array") as mock_array:
            mock_array.return_value = np.zeros((100, 100, 3))

            result = detector.validate_image(test_file)

        assert result["format"] == "PNG"
        assert result["width"] == 100
        assert result["height"] == 100
        assert result["mode"] == "RGB"
        assert "jabcode_suitability" in result
        assert "recommendations" in result

    def test_analyze_jabcode_suitability_good_image(self):
        """Test JABCode suitability analysis for good image."""
        detector = ImageFormatDetector()

        # Mock good image
        mock_image = MagicMock()
        mock_image.size = (200, 200)
        mock_image.mode = "RGB"

        format_info = {"lossless": True}

        with patch("numpy.array") as mock_array:
            mock_array.return_value = np.zeros((200, 200, 3))

            suitability = detector._analyze_jabcode_suitability(mock_image, format_info)

        assert suitability["overall_score"] > 80
        assert len(suitability["strengths"]) > 0
        assert "Good image size" in str(suitability["strengths"])

    def test_analyze_jabcode_suitability_small_image(self):
        """Test JABCode suitability analysis for small image."""
        detector = ImageFormatDetector()

        # Mock small image
        mock_image = MagicMock()
        mock_image.size = (30, 30)
        mock_image.mode = "RGB"

        format_info = {"lossless": True}

        with patch("numpy.array") as mock_array:
            mock_array.return_value = np.zeros((30, 30, 3))

            suitability = detector._analyze_jabcode_suitability(mock_image, format_info)

        assert suitability["overall_score"] < 80
        assert len(suitability["issues"]) > 0
        assert "too small" in str(suitability["issues"]).lower()

    def test_analyze_jabcode_suitability_lossy_format(self):
        """Test JABCode suitability analysis for lossy format."""
        detector = ImageFormatDetector()

        # Mock image with lossy format
        mock_image = MagicMock()
        mock_image.size = (200, 200)
        mock_image.mode = "RGB"

        format_info = {"lossless": False}

        with patch("numpy.array") as mock_array:
            mock_array.return_value = np.zeros((200, 200, 3))

            suitability = detector._analyze_jabcode_suitability(mock_image, format_info)

        assert len(suitability["warnings"]) > 0
        assert "lossy" in str(suitability["warnings"]).lower()

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        detector = ImageFormatDetector()

        # Image info that needs recommendations
        image_info = {
            "format": "JPEG",
            "width": 50,
            "height": 50,
            "mode": "L",
            "lossless": False,
        }

        recommendations = detector._generate_recommendations(image_info)

        assert len(recommendations) > 0
        assert any("PNG" in rec for rec in recommendations)
        assert any("resolution" in rec.lower() for rec in recommendations)
        assert any("RGB" in rec for rec in recommendations)


class TestImageConverter:
    """Test cases for ImageConverter class."""

    def test_initialization(self):
        """Test ImageConverter initialization."""
        converter = ImageConverter()
        assert converter is not None
        assert hasattr(converter, "format_detector")
        assert isinstance(converter.format_detector, ImageFormatDetector)

    def test_initialization_with_detector(self):
        """Test ImageConverter initialization with custom detector."""
        custom_detector = ImageFormatDetector()
        converter = ImageConverter(custom_detector)
        assert converter.format_detector is custom_detector

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_prepare_for_format_jpeg(self, mock_open):
        """Test image preparation for JPEG format."""
        converter = ImageConverter()

        # Mock RGBA image
        mock_image = MagicMock()
        mock_image.mode = "RGBA"
        mock_image.size = (100, 100)
        mock_image.copy.return_value = mock_image
        mock_image.split.return_value = [
            None,
            None,
            None,
            MagicMock(),
        ]  # Mock alpha channel

        # Mock background creation
        mock_background = MagicMock()
        with patch("pyhue2d.jabcode.image_format.Image.new", return_value=mock_background):
            result = converter._prepare_for_format(mock_image, "JPEG")

        # Should create white background for JPEG
        mock_background.paste.assert_called_once()

    def test_prepare_for_format_png(self):
        """Test image preparation for PNG format."""
        converter = ImageConverter()

        # Mock image
        mock_image = MagicMock()
        mock_image.mode = "RGB"
        mock_image.copy.return_value = mock_image

        result = converter._prepare_for_format(mock_image, "PNG")

        # PNG supports all modes, should just copy
        mock_image.copy.assert_called_once()

    def test_get_save_parameters_jpeg(self):
        """Test save parameter generation for JPEG."""
        converter = ImageConverter()

        params = converter._get_save_parameters("JPEG", quality=90, optimize=True)

        assert params["format"] == "JPEG"
        assert params["quality"] == 90
        assert params["optimize"] is True
        assert "progressive" in params

    def test_get_save_parameters_png(self):
        """Test save parameter generation for PNG."""
        converter = ImageConverter()

        params = converter._get_save_parameters("PNG", quality=95, optimize=True)

        assert params["format"] == "PNG"
        assert params["optimize"] is True
        assert "compress_level" in params

    def test_get_save_parameters_webp(self):
        """Test save parameter generation for WebP."""
        converter = ImageConverter()

        params = converter._get_save_parameters("WEBP", quality=80, optimize=True, lossless=True)

        assert params["format"] == "WEBP"
        assert params["quality"] == 80
        assert params["lossless"] is True
        assert "method" in params

    @patch("pyhue2d.jabcode.image_format.Image.open")
    @patch("pyhue2d.jabcode.image_format.ImageFormatDetector.detect_format")
    def test_convert_image_success(self, mock_detect, mock_open, tmp_path):
        """Test successful image conversion."""
        converter = ImageConverter()

        # Setup mocks
        mock_detect.return_value = "PNG"
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.mode = "RGB"
        mock_open.return_value.__enter__.return_value = mock_image

        input_file = tmp_path / "input.png"
        output_file = tmp_path / "output.jpg"
        input_file.write_bytes(b"fake png data")

        with patch.object(converter, "_prepare_for_format", return_value=mock_image):
            result = converter.convert_image(input_path=input_file, output_path=output_file, target_format="JPEG")

        assert result["conversion_successful"] is True
        assert result["input_format"] == "PNG"
        assert result["target_format"] == "JPEG"
        assert result["input_size"] == (100, 100)
        mock_image.save.assert_called_once()

    def test_convert_image_unsupported_target(self, tmp_path):
        """Test conversion to unsupported target format."""
        converter = ImageConverter()

        input_file = tmp_path / "input.png"
        output_file = tmp_path / "output.gif"
        input_file.write_bytes(b"fake png data")

        with patch.object(converter.format_detector, "detect_format", return_value="PNG"):
            with pytest.raises(JABCodeFormatError) as exc_info:
                converter.convert_image(input_path=input_file, output_path=output_file, target_format="GIF")

        assert "Unsupported target format: GIF" in str(exc_info.value)

    @patch("pyhue2d.jabcode.image_format.Image.open")
    def test_convert_image_exception(self, mock_open, tmp_path):
        """Test conversion with exception."""
        converter = ImageConverter()

        input_file = tmp_path / "input.png"
        output_file = tmp_path / "output.jpg"
        input_file.write_bytes(b"fake png data")

        # Mock exception during image processing
        mock_open.side_effect = Exception("Cannot process image")

        with patch.object(converter.format_detector, "detect_format", return_value="PNG"):
            with pytest.raises(JABCodeImageError) as exc_info:
                converter.convert_image(input_path=input_file, output_path=output_file, target_format="JPEG")

        assert "Image conversion failed" in str(exc_info.value)

    def test_optimize_for_jabcode(self, tmp_path):
        """Test JABCode optimization."""
        converter = ImageConverter()

        input_file = tmp_path / "input.jpg"
        output_file = tmp_path / "output.png"
        input_file.write_bytes(b"fake jpg data")

        with patch.object(converter, "convert_image") as mock_convert:
            mock_convert.return_value = {"conversion_successful": True}

            result = converter.optimize_for_jabcode(input_file, output_file)

        # Should convert to PNG with optimization
        mock_convert.assert_called_once_with(
            input_path=input_file,
            output_path=output_file,
            target_format="PNG",
            optimize=True,
            compress_level=6,
        )


if __name__ == "__main__":
    pytest.main([__file__])

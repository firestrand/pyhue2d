"""Unit tests for CLI argument classes."""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from pyhue2d.jabcode.cli_args import CommonArgs, DecodeArgs, EncodeArgs, validate_args
from pyhue2d.jabcode.exceptions import JABCodeValidationError


class TestEncodeArgs:
    """Test cases for EncodeArgs class."""

    def test_valid_encode_args(self, tmp_path):
        """Test creation of valid encode arguments."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        args = EncodeArgs(input_source=input_file, output_path=output_file, palette=8, ecc_level="M")

        assert args.input_source == input_file
        assert args.output_path == output_file
        assert args.palette == 8
        assert args.ecc_level == "M"
        assert args.version == "auto"
        assert args.quiet_zone == 2
        assert args.mask_pattern == 7

    def test_input_source_validation_nonexistent_file(self):
        """Test validation fails for nonexistent input file."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source="/nonexistent/file.txt", output_path="/tmp/output.png")

        assert "Input file does not exist" in str(exc_info.value)
        assert exc_info.value.context["field"] == "input_source"

    def test_input_source_validation_directory(self, tmp_path):
        """Test validation fails when input is a directory."""
        directory = tmp_path / "test_dir"
        directory.mkdir()

        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source=directory, output_path=tmp_path / "output.png")

        assert "Input source is not a file" in str(exc_info.value)

    def test_output_path_validation_creates_directory(self, tmp_path):
        """Test output directory creation."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")

        output_dir = tmp_path / "new_dir"
        output_file = output_dir / "output.png"

        args = EncodeArgs(input_source=input_file, output_path=output_file)

        assert output_dir.exists()
        assert args.output_path == output_file

    def test_output_path_validation_existing_file_no_force(self, tmp_path):
        """Test validation fails for existing output file without force."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")

        output_file = tmp_path / "output.png"
        output_file.write_text("existing")

        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source=input_file, output_path=output_file, force_overwrite=False)

        assert "Output file exists" in str(exc_info.value)
        assert "use --force" in str(exc_info.value)

    def test_output_path_validation_existing_file_with_force(self, tmp_path):
        """Test validation passes for existing output file with force."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")

        output_file = tmp_path / "output.png"
        output_file.write_text("existing")

        args = EncodeArgs(input_source=input_file, output_path=output_file, force_overwrite=True)

        assert args.force_overwrite is True

    def test_palette_validation_valid_values(self, tmp_path):
        """Test palette validation with valid values."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        valid_palettes = [4, 8, 16, 32, 64, 128, 256]

        for palette in valid_palettes:
            args = EncodeArgs(input_source=input_file, output_path=output_file, palette=palette)
            assert args.palette == palette

    def test_palette_validation_invalid_value(self, tmp_path):
        """Test palette validation with invalid value."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source=input_file, output_path=output_file, palette=7)

        assert "Invalid palette size: 7" in str(exc_info.value)
        assert exc_info.value.context["field"] == "palette"
        assert exc_info.value.context["value"] == 7

    def test_ecc_level_validation_valid_values(self, tmp_path):
        """Test ECC level validation with valid values."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        valid_levels = ["L", "M", "Q", "H"]

        for level in valid_levels:
            args = EncodeArgs(input_source=input_file, output_path=output_file, ecc_level=level)
            assert args.ecc_level == level

    def test_ecc_level_validation_invalid_value(self, tmp_path):
        """Test ECC level validation with invalid value."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source=input_file, output_path=output_file, ecc_level="X")

        assert "Invalid ECC level: X" in str(exc_info.value)

    def test_version_validation_auto(self, tmp_path):
        """Test version validation with 'auto'."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        args = EncodeArgs(input_source=input_file, output_path=output_file, version="auto")

        assert args.version == "auto"

    def test_version_validation_integer(self, tmp_path):
        """Test version validation with valid integer."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        for version in [1, 16, 32]:
            args = EncodeArgs(input_source=input_file, output_path=output_file, version=version)
            assert args.version == version

    def test_version_validation_invalid_integer(self, tmp_path):
        """Test version validation with invalid integer."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        with pytest.raises(JABCodeValidationError) as exc_info:
            EncodeArgs(input_source=input_file, output_path=output_file, version=0)

        assert "Invalid version number: 0" in str(exc_info.value)

    def test_range_validations(self, tmp_path):
        """Test range validations for various parameters."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        # Test quiet_zone range
        with pytest.raises(JABCodeValidationError):
            EncodeArgs(input_source=input_file, output_path=output_file, quiet_zone=-1)

        # Test mask_pattern range
        with pytest.raises(JABCodeValidationError):
            EncodeArgs(input_source=input_file, output_path=output_file, mask_pattern=8)

        # Test module_size range
        with pytest.raises(JABCodeValidationError):
            EncodeArgs(input_source=input_file, output_path=output_file, module_size=0)

    def test_encoding_mode_validation(self, tmp_path):
        """Test encoding mode validation."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        valid_modes = [
            "Numeric",
            "Alphanumeric",
            "Uppercase",
            "Lowercase",
            "Mixed",
            "Punctuation",
            "Byte",
        ]

        for mode in valid_modes:
            args = EncodeArgs(input_source=input_file, output_path=output_file, encoding_mode=mode)
            assert args.encoding_mode == mode

        # Test invalid mode
        with pytest.raises(JABCodeValidationError):
            EncodeArgs(
                input_source=input_file,
                output_path=output_file,
                encoding_mode="Invalid",
            )

    def test_to_encoder_settings(self, tmp_path):
        """Test conversion to encoder settings."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        args = EncodeArgs(input_source=input_file, output_path=output_file, palette=16, ecc_level="H")

        settings = args.to_encoder_settings()

        assert settings["colors"] == 16
        assert settings["ecc_level"] == "H"


class TestDecodeArgs:
    """Test cases for DecodeArgs class."""

    def test_valid_decode_args(self, tmp_path):
        """Test creation of valid decode arguments."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"fake png data")  # Create fake image file

        args = DecodeArgs(input_path=input_file)

        assert args.input_path == input_file
        assert args.output_path is None
        assert args.detection_method == "scanline"
        assert args.perspective_correction is True
        assert args.error_correction is True

    def test_input_path_validation_nonexistent(self):
        """Test validation fails for nonexistent input file."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            DecodeArgs(input_path="/nonexistent/image.png")

        assert "Input image does not exist" in str(exc_info.value)

    def test_input_path_validation_invalid_extension(self, tmp_path):
        """Test validation fails for invalid image extension."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("not an image")

        with pytest.raises(JABCodeValidationError) as exc_info:
            DecodeArgs(input_path=input_file)

        assert "Unsupported image format" in str(exc_info.value)

    def test_input_path_validation_valid_extensions(self, tmp_path):
        """Test validation passes for valid image extensions."""
        valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

        for ext in valid_extensions:
            input_file = tmp_path / f"input{ext}"
            input_file.write_bytes(b"fake image data")

            args = DecodeArgs(input_path=input_file)
            assert args.input_path.suffix == ext

    def test_output_path_validation_none(self, tmp_path):
        """Test output path validation when None."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"fake png data")

        args = DecodeArgs(input_path=input_file, output_path=None)
        assert args.output_path is None

    def test_detection_method_validation(self, tmp_path):
        """Test detection method validation."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"fake png data")

        valid_methods = ["scanline", "contour", "hybrid"]

        for method in valid_methods:
            args = DecodeArgs(input_path=input_file, detection_method=method)
            assert args.detection_method == method

        # Test invalid method
        with pytest.raises(JABCodeValidationError):
            DecodeArgs(input_path=input_file, detection_method="invalid")

    def test_to_decoder_settings(self, tmp_path):
        """Test conversion to decoder settings."""
        input_file = tmp_path / "input.png"
        input_file.write_bytes(b"fake png data")

        args = DecodeArgs(
            input_path=input_file,
            detection_method="contour",
            perspective_correction=False,
            error_correction=False,
        )

        settings = args.to_decoder_settings()

        assert settings["detection_method"] == "contour"
        assert settings["perspective_correction"] is False
        assert settings["error_correction"] is False


class TestCommonArgs:
    """Test cases for CommonArgs class."""

    def test_valid_common_args(self):
        """Test creation of valid common arguments."""
        args = CommonArgs(verbose=True, quiet=False, log_level="DEBUG")

        assert args.verbose is True
        assert args.quiet is False
        assert args.log_level == "DEBUG"

    def test_log_level_validation_valid(self):
        """Test log level validation with valid values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            args = CommonArgs(log_level=level)
            assert args.log_level == level

    def test_log_level_validation_invalid(self):
        """Test log level validation with invalid value."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            CommonArgs(log_level="INVALID")

        assert "Invalid log level: INVALID" in str(exc_info.value)

    def test_verbose_quiet_conflict(self):
        """Test validation fails when both verbose and quiet are set."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            CommonArgs(verbose=True, quiet=True)

        assert "Cannot specify both --verbose and --quiet" in str(exc_info.value)

    def test_config_file_validation_nonexistent(self):
        """Test config file validation with nonexistent file."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            CommonArgs(config_file="/nonexistent/config.json")

        assert "Config file does not exist" in str(exc_info.value)

    def test_config_file_validation_valid(self, tmp_path):
        """Test config file validation with valid file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"test": true}')

        args = CommonArgs(config_file=config_file)
        assert args.config_file == config_file


class TestValidateArgsFunction:
    """Test cases for validate_args function."""

    def test_validate_args_success(self, tmp_path):
        """Test successful argument validation."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        args = validate_args(EncodeArgs, input_source=input_file, output_path=output_file)

        assert isinstance(args, EncodeArgs)
        assert args.input_source == input_file

    def test_validate_args_validation_error(self):
        """Test validation error handling."""
        with pytest.raises(JABCodeValidationError):
            validate_args(
                EncodeArgs,
                input_source="/nonexistent/file.txt",
                output_path="/tmp/output.png",
            )

    def test_validate_args_type_error(self):
        """Test type error handling."""
        with pytest.raises(JABCodeValidationError) as exc_info:
            validate_args(EncodeArgs, invalid_argument=True)

        assert "Invalid arguments" in str(exc_info.value)


class TestPathHandling:
    """Test cases for path handling and conversion."""

    def test_string_to_path_conversion(self, tmp_path):
        """Test automatic string to Path conversion."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")
        output_file = tmp_path / "output.png"

        # Pass strings instead of Path objects
        args = EncodeArgs(input_source=str(input_file), output_path=str(output_file))

        # Should be converted to Path objects
        assert isinstance(args.input_source, Path)
        assert isinstance(args.output_path, Path)
        assert args.input_source == input_file
        assert args.output_path == output_file

    def test_absolute_path_handling(self, tmp_path):
        """Test handling of absolute paths."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")

        args = EncodeArgs(
            input_source=input_file.absolute(),
            output_path=(tmp_path / "output.png").absolute(),
        )

        assert args.input_source.is_absolute()
        assert args.output_path.is_absolute()


if __name__ == "__main__":
    pytest.main([__file__])

"""Command-line argument classes with validation for JABCode operations.

This module provides structured argument classes for encoding and decoding
operations with comprehensive validation and type safety.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .exceptions import JABCodeValidationError, validation_error


@dataclass
class EncodeArgs:
    """Arguments for JABCode encoding operations.

    Provides validation and type safety for encoding parameters.
    """

    # Required arguments
    input_source: Union[str, Path]
    output_path: Union[str, Path]

    # Optional encoding parameters
    palette: int = 8
    ecc_level: str = "M"
    version: Union[int, str] = "auto"
    quiet_zone: int = 2
    mask_pattern: int = 7
    module_size: int = 1

    # Optional behavior settings
    optimize: bool = True
    force_overwrite: bool = False
    verbose: bool = False

    # Advanced options
    encoding_mode: Optional[str] = None
    chunk_size: int = 1024
    max_data_size: int = 32768

    def __post_init__(self):
        """Validate arguments after initialization."""
        self._validate_input_source()
        self._validate_output_path()
        self._validate_palette()
        self._validate_ecc_level()
        self._validate_version()
        self._validate_quiet_zone()
        self._validate_mask_pattern()
        self._validate_module_size()
        self._validate_encoding_mode()
        self._validate_sizes()

    def _validate_input_source(self):
        """Validate input source exists and is readable."""
        if isinstance(self.input_source, str):
            self.input_source = Path(self.input_source)

        if not self.input_source.exists():
            raise validation_error(
                f"Input file does not exist: {self.input_source}",
                field="input_source",
                value=str(self.input_source),
            )

        if not self.input_source.is_file():
            raise validation_error(
                f"Input source is not a file: {self.input_source}",
                field="input_source",
                value=str(self.input_source),
            )

        if not os.access(self.input_source, os.R_OK):
            raise validation_error(
                f"Input file is not readable: {self.input_source}",
                field="input_source",
                value=str(self.input_source),
            )

    def _validate_output_path(self):
        """Validate output path is writable."""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Check if output directory exists and is writable
        output_dir = self.output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise validation_error(
                    f"Cannot create output directory: {output_dir}",
                    field="output_path",
                    value=str(self.output_path),
                )

        if not os.access(output_dir, os.W_OK):
            raise validation_error(
                f"Output directory is not writable: {output_dir}",
                field="output_path",
                value=str(self.output_path),
            )

        # Check if output file exists and force_overwrite is not set
        if self.output_path.exists() and not self.force_overwrite:
            raise validation_error(
                f"Output file exists (use --force to overwrite): {self.output_path}",
                field="output_path",
                value=str(self.output_path),
            )

    def _validate_palette(self):
        """Validate color palette size."""
        valid_palettes = [4, 8, 16, 32, 64, 128, 256]
        if self.palette not in valid_palettes:
            raise validation_error(
                f"Invalid palette size: {self.palette}. Must be one of {valid_palettes}",
                field="palette",
                value=self.palette,
            )

    def _validate_ecc_level(self):
        """Validate error correction level."""
        valid_levels = ["L", "M", "Q", "H"]
        if self.ecc_level not in valid_levels:
            raise validation_error(
                f"Invalid ECC level: {self.ecc_level}. Must be one of {valid_levels}",
                field="ecc_level",
                value=self.ecc_level,
            )

    def _validate_version(self):
        """Validate symbol version."""
        if isinstance(self.version, str):
            if self.version != "auto":
                raise validation_error(
                    f"Invalid version string: {self.version}. Must be 'auto' or integer",
                    field="version",
                    value=self.version,
                )
        elif isinstance(self.version, int):
            if not (1 <= self.version <= 32):
                raise validation_error(
                    f"Invalid version number: {self.version}. Must be between 1 and 32",
                    field="version",
                    value=self.version,
                )
        else:
            raise validation_error(
                f"Invalid version type: {type(self.version)}. Must be 'auto' or integer",
                field="version",
                value=self.version,
            )

    def _validate_quiet_zone(self):
        """Validate quiet zone size."""
        if not (0 <= self.quiet_zone <= 10):
            raise validation_error(
                f"Invalid quiet zone: {self.quiet_zone}. Must be between 0 and 10",
                field="quiet_zone",
                value=self.quiet_zone,
            )

    def _validate_mask_pattern(self):
        """Validate mask pattern."""
        if not (0 <= self.mask_pattern <= 7):
            raise validation_error(
                f"Invalid mask pattern: {self.mask_pattern}. Must be between 0 and 7",
                field="mask_pattern",
                value=self.mask_pattern,
            )

    def _validate_module_size(self):
        """Validate module size."""
        if not (1 <= self.module_size <= 20):
            raise validation_error(
                f"Invalid module size: {self.module_size}. Must be between 1 and 20",
                field="module_size",
                value=self.module_size,
            )

    def _validate_encoding_mode(self):
        """Validate encoding mode if specified."""
        if self.encoding_mode is not None:
            valid_modes = [
                "Numeric",
                "Alphanumeric",
                "Uppercase",
                "Lowercase",
                "Mixed",
                "Punctuation",
                "Byte",
            ]
            if self.encoding_mode not in valid_modes:
                raise validation_error(
                    f"Invalid encoding mode: {self.encoding_mode}. Must be one of {valid_modes}",
                    field="encoding_mode",
                    value=self.encoding_mode,
                )

    def _validate_sizes(self):
        """Validate size parameters."""
        if not (512 <= self.chunk_size <= 8192):
            raise validation_error(
                f"Invalid chunk size: {self.chunk_size}. Must be between 512 and 8192",
                field="chunk_size",
                value=self.chunk_size,
            )

        if not (1024 <= self.max_data_size <= 1048576):  # 1MB max
            raise validation_error(
                f"Invalid max data size: {self.max_data_size}. Must be between 1024 and 1048576",
                field="max_data_size",
                value=self.max_data_size,
            )

    def to_encoder_settings(self) -> Dict[str, Any]:
        """Convert to encoder settings dictionary.

        Returns:
            Dictionary suitable for core.encode() function
        """
        return {
            "colors": self.palette,  # Core API expects 'colors' not 'color_count'
            "ecc_level": self.ecc_level,
        }


@dataclass
class DecodeArgs:
    """Arguments for JABCode decoding operations.

    Provides validation and type safety for decoding parameters.
    """

    # Required arguments
    input_path: Union[str, Path]

    # Optional output
    output_path: Optional[Union[str, Path]] = None

    # Decoding parameters
    detection_method: str = "scanline"
    perspective_correction: bool = True
    error_correction: bool = True
    validate_patterns: bool = True
    multi_symbol_support: bool = True
    noise_reduction: bool = True

    # Optional behavior settings
    force_overwrite: bool = False
    verbose: bool = False
    raw_output: bool = False

    def __post_init__(self):
        """Validate arguments after initialization."""
        self._validate_input_path()
        self._validate_output_path()
        self._validate_detection_method()

    def _validate_input_path(self):
        """Validate input image exists and is readable."""
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)

        if not self.input_path.exists():
            raise validation_error(
                f"Input image does not exist: {self.input_path}",
                field="input_path",
                value=str(self.input_path),
            )

        if not self.input_path.is_file():
            raise validation_error(
                f"Input path is not a file: {self.input_path}",
                field="input_path",
                value=str(self.input_path),
            )

        if not os.access(self.input_path, os.R_OK):
            raise validation_error(
                f"Input image is not readable: {self.input_path}",
                field="input_path",
                value=str(self.input_path),
            )

        # Validate image file extension
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        if self.input_path.suffix.lower() not in valid_extensions:
            raise validation_error(
                f"Unsupported image format: {self.input_path.suffix}. "
                f"Supported formats: {', '.join(valid_extensions)}",
                field="input_path",
                value=str(self.input_path),
            )

    def _validate_output_path(self):
        """Validate output path if specified."""
        if self.output_path is None:
            return

        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Check if output directory exists and is writable
        output_dir = self.output_path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise validation_error(
                    f"Cannot create output directory: {output_dir}",
                    field="output_path",
                    value=str(self.output_path),
                )

        if not os.access(output_dir, os.W_OK):
            raise validation_error(
                f"Output directory is not writable: {output_dir}",
                field="output_path",
                value=str(self.output_path),
            )

        # Check if output file exists and force_overwrite is not set
        if self.output_path.exists() and not self.force_overwrite:
            raise validation_error(
                f"Output file exists (use --force to overwrite): {self.output_path}",
                field="output_path",
                value=str(self.output_path),
            )

    def _validate_detection_method(self):
        """Validate pattern detection method."""
        valid_methods = ["scanline", "contour", "hybrid"]
        if self.detection_method not in valid_methods:
            raise validation_error(
                f"Invalid detection method: {self.detection_method}. " f"Must be one of {valid_methods}",
                field="detection_method",
                value=self.detection_method,
            )

    def to_decoder_settings(self) -> Dict[str, Any]:
        """Convert to decoder settings dictionary.

        Returns:
            Dictionary suitable for JABCodeDecoder initialization
        """
        return {
            "detection_method": self.detection_method,
            "perspective_correction": self.perspective_correction,
            "error_correction": self.error_correction,
            "validate_patterns": self.validate_patterns,
            "multi_symbol_support": self.multi_symbol_support,
            "noise_reduction": self.noise_reduction,
        }


@dataclass
class CommonArgs:
    """Common arguments shared across operations."""

    verbose: bool = False
    quiet: bool = False
    log_level: str = "INFO"
    config_file: Optional[Union[str, Path]] = None

    def __post_init__(self):
        """Validate common arguments."""
        self._validate_log_level()
        self._validate_config_file()
        self._validate_verbose_quiet()

    def _validate_log_level(self):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise validation_error(
                f"Invalid log level: {self.log_level}. Must be one of {valid_levels}",
                field="log_level",
                value=self.log_level,
            )

    def _validate_config_file(self):
        """Validate config file if specified."""
        if self.config_file is None:
            return

        if isinstance(self.config_file, str):
            self.config_file = Path(self.config_file)

        if not self.config_file.exists():
            raise validation_error(
                f"Config file does not exist: {self.config_file}",
                field="config_file",
                value=str(self.config_file),
            )

        if not os.access(self.config_file, os.R_OK):
            raise validation_error(
                f"Config file is not readable: {self.config_file}",
                field="config_file",
                value=str(self.config_file),
            )

    def _validate_verbose_quiet(self):
        """Validate verbose and quiet are not both set."""
        if self.verbose and self.quiet:
            raise validation_error(
                "Cannot specify both --verbose and --quiet",
                field="verbose_quiet",
                value={"verbose": self.verbose, "quiet": self.quiet},
            )


def validate_args(args_class, **kwargs) -> Union[EncodeArgs, DecodeArgs, CommonArgs]:
    """Validate and create argument instance.

    Args:
        args_class: Argument class to create
        **kwargs: Keyword arguments for the class

    Returns:
        Validated argument instance

    Raises:
        JABCodeValidationError: If validation fails
    """
    try:
        return args_class(**kwargs)
    except TypeError as e:
        raise validation_error(f"Invalid arguments: {e}") from e
    except Exception as e:
        if isinstance(e, JABCodeValidationError):
            raise
        raise validation_error(f"Argument validation failed: {e}") from e

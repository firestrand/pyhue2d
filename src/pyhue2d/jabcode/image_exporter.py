"""Image exporter for JABCode symbols.

This module provides the ImageExporter class for saving JABCode symbols
to various image formats (PNG, JPEG, BMP, TIFF) with customizable quality,
compression, and metadata settings.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, JpegImagePlugin, PngImagePlugin

from .core import Bitmap


class ImageExporter:
    """Image exporter for JABCode symbols.

    The ImageExporter handles saving JABCode symbols to various image formats
    with support for quality settings, compression levels, DPI, and metadata.
    """

    DEFAULT_SETTINGS = {
        "default_format": "PNG",
        "quality": 90,  # JPEG quality (1-100)
        "compression_level": 6,  # PNG compression (0-9)
        "dpi": (150, 150),  # DPI for print output
        "optimize": True,  # Optimize file size
        "progressive": False,  # Progressive JPEG
    }

    # Supported image formats
    SUPPORTED_FORMATS = ["PNG", "JPEG", "BMP", "TIFF", "WEBP"]

    # Format extensions mapping
    EXTENSION_MAP = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize image exporter.

        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        # Validate settings
        self._validate_settings()

        # Export statistics
        self._stats = {
            "total_exports": 0,
            "total_export_time": 0.0,
            "export_times": [],
            "format_counts": {fmt: 0 for fmt in self.SUPPORTED_FORMATS},
        }

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported image formats."""
        return self.SUPPORTED_FORMATS.copy()

    def export_bitmap(
        self,
        bitmap: Bitmap,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export bitmap to image file.

        Args:
            bitmap: Bitmap object to export
            filepath: Output file path
            format: Image format (auto-detected from extension if not specified)
            **kwargs: Additional format-specific options

        Raises:
            ValueError: For invalid inputs or unsupported formats
        """
        start_time = time.time()

        try:
            # Validate inputs
            self._validate_bitmap(bitmap)
            filepath = Path(filepath)

            # Determine format
            if format is None:
                format = self._detect_format_from_extension(filepath)

            format = format.upper()
            if format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {format}")

            # Create PIL Image from bitmap
            pil_image = Image.fromarray(bitmap.array, mode="RGB")

            # Apply format-specific settings
            save_kwargs = self._get_format_settings(format, **kwargs)

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            pil_image.save(str(filepath), format=format, **save_kwargs)

            # Update statistics
            export_time = time.time() - start_time
            self._update_stats(export_time, format)

        except Exception as e:
            raise ValueError(f"Failed to export bitmap: {str(e)}") from e

    def export_image(
        self,
        image: Image.Image,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export PIL Image to file.

        Args:
            image: PIL Image object to export
            filepath: Output file path
            format: Image format (auto-detected from extension if not specified)
            **kwargs: Additional format-specific options
        """
        start_time = time.time()

        try:
            filepath = Path(filepath)

            # Determine format
            if format is None:
                format = self._detect_format_from_extension(filepath)

            format = format.upper()
            if format not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {format}")

            # Apply format-specific settings
            save_kwargs = self._get_format_settings(format, **kwargs)

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save image
            image.save(str(filepath), format=format, **save_kwargs)

            # Update statistics
            export_time = time.time() - start_time
            self._update_stats(export_time, format)

        except Exception as e:
            raise ValueError(f"Failed to export image: {str(e)}") from e

    def export_matrix(
        self,
        matrix: np.ndarray,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export numpy matrix to image file.

        Args:
            matrix: Numpy array (2D grayscale or 3D RGB)
            filepath: Output file path
            format: Image format (auto-detected from extension if not specified)
            **kwargs: Additional format-specific options
        """
        # Convert matrix to PIL Image
        if matrix.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(matrix, mode="L")
        elif matrix.ndim == 3 and matrix.shape[2] == 3:
            # RGB
            pil_image = Image.fromarray(matrix, mode="RGB")
        elif matrix.ndim == 3 and matrix.shape[2] == 4:
            # RGBA
            pil_image = Image.fromarray(matrix, mode="RGBA")
        else:
            raise ValueError(f"Unsupported matrix shape: {matrix.shape}")

        self.export_image(pil_image, filepath, format, **kwargs)

    def export_batch(
        self,
        bitmaps: List[Bitmap],
        filepaths: List[Union[str, Path]],
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export multiple bitmaps to files.

        Args:
            bitmaps: List of bitmap objects to export
            filepaths: List of output file paths
            format: Image format for all files
            **kwargs: Additional format-specific options

        Raises:
            ValueError: If lists have different lengths
        """
        if len(bitmaps) != len(filepaths):
            raise ValueError("Number of bitmaps must match number of filepaths")

        for bitmap, filepath in zip(bitmaps, filepaths):
            self.export_bitmap(bitmap, filepath, format, **kwargs)

    def export_bitmap_with_metadata(
        self,
        bitmap: Bitmap,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Export bitmap with embedded metadata.

        Args:
            bitmap: Bitmap object to export
            filepath: Output file path
            metadata: Dictionary of metadata to embed
            format: Image format
            **kwargs: Additional format-specific options
        """
        filepath = Path(filepath)

        # Determine format
        if format is None:
            format = self._detect_format_from_extension(filepath)

        # Create PIL Image
        pil_image = Image.fromarray(bitmap.array, mode="RGB")

        # Add metadata if supported by format
        if metadata and format.upper() == "PNG":
            # PNG supports text metadata
            pnginfo = PngImagePlugin.PngInfo()
            for key, value in metadata.items():
                pnginfo.add_text(key, str(value))
            kwargs["pnginfo"] = pnginfo

        # For JPEG, metadata support is limited
        # For other formats, metadata might not be supported

        self.export_image(pil_image, filepath, format, **kwargs)

    def _validate_bitmap(self, bitmap: Bitmap) -> None:
        """Validate bitmap object.

        Args:
            bitmap: Bitmap to validate

        Raises:
            ValueError: For invalid bitmaps
        """
        if not isinstance(bitmap, Bitmap):
            raise ValueError("Input must be a Bitmap object")

        if bitmap.width <= 0 or bitmap.height <= 0:
            raise ValueError("Bitmap dimensions must be positive")

        if bitmap.array is None or bitmap.array.size == 0:
            raise ValueError("Bitmap array cannot be empty")

    def _detect_format_from_extension(self, filepath: Path) -> str:
        """Detect image format from file extension.

        Args:
            filepath: File path

        Returns:
            Image format string

        Raises:
            ValueError: For unsupported extensions
        """
        extension = filepath.suffix.lower()

        if extension in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[extension]
        else:
            # Default to PNG for unknown extensions
            return self.settings["default_format"]

    def _get_format_settings(self, format: str, **kwargs) -> Dict[str, Any]:
        """Get format-specific save settings.

        Args:
            format: Image format
            **kwargs: Override settings

        Returns:
            Dictionary of save parameters
        """
        settings = {}

        # Apply default settings
        if format == "PNG":
            settings.update(
                {
                    "optimize": kwargs.get("optimize", self.settings["optimize"]),
                    "compress_level": kwargs.get("compression_level", self.settings["compression_level"]),
                }
            )

        elif format == "JPEG":
            settings.update(
                {
                    "quality": kwargs.get("quality", self.settings["quality"]),
                    "optimize": kwargs.get("optimize", self.settings["optimize"]),
                    "progressive": kwargs.get("progressive", self.settings["progressive"]),
                }
            )

        elif format == "WEBP":
            settings.update(
                {
                    "quality": kwargs.get("quality", self.settings["quality"]),
                    "optimize": kwargs.get("optimize", self.settings["optimize"]),
                }
            )

        # Add DPI information for print formats
        if format in ["PNG", "JPEG", "TIFF"]:
            dpi = kwargs.get("dpi", self.settings["dpi"])
            if dpi:
                settings["dpi"] = dpi

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in [
                "optimize",
                "quality",
                "compression_level",
                "progressive",
                "dpi",
            ]:
                settings[key] = value

        return settings

    def _validate_settings(self) -> None:
        """Validate exporter settings."""
        if self.settings["default_format"] not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Invalid default format: {self.settings['default_format']}")

        if not (1 <= self.settings["quality"] <= 100):
            raise ValueError("Quality must be between 1 and 100")

        if not (0 <= self.settings["compression_level"] <= 9):
            raise ValueError("Compression level must be between 0 and 9")

        dpi = self.settings["dpi"]
        if dpi and (not isinstance(dpi, (tuple, list)) or len(dpi) != 2):
            raise ValueError("DPI must be a tuple of (x, y) values")

    def _update_stats(self, export_time: float, format: str) -> None:
        """Update export statistics.

        Args:
            export_time: Time taken for export
            format: Image format used
        """
        self._stats["total_exports"] += 1
        self._stats["total_export_time"] += export_time
        self._stats["export_times"].append(export_time)

        if format in self._stats["format_counts"]:
            self._stats["format_counts"][format] += 1

    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics.

        Returns:
            Dictionary of export statistics
        """
        if self._stats["total_exports"] == 0:
            return {
                "total_exports": 0,
                "total_export_time": 0.0,
                "avg_export_time": 0.0,
                "format_counts": {fmt: 0 for fmt in self.SUPPORTED_FORMATS},
            }

        return {
            "total_exports": self._stats["total_exports"],
            "total_export_time": self._stats["total_export_time"],
            "avg_export_time": (self._stats["total_export_time"] / self._stats["total_exports"]),
            "min_export_time": min(self._stats["export_times"]),
            "max_export_time": max(self._stats["export_times"]),
            "format_counts": self._stats["format_counts"].copy(),
        }

    def reset(self) -> None:
        """Reset export statistics and state."""
        self._stats = {
            "total_exports": 0,
            "total_export_time": 0.0,
            "export_times": [],
            "format_counts": {fmt: 0 for fmt in self.SUPPORTED_FORMATS},
        }

    def copy(self) -> "ImageExporter":
        """Create a copy of this exporter.

        Returns:
            New ImageExporter instance with same settings
        """
        return ImageExporter(self.settings.copy())

    def __str__(self) -> str:
        """String representation of exporter."""
        return (
            f"ImageExporter(default_format={self.settings['default_format']}, "
            f"quality={self.settings['quality']}, "
            f"dpi={self.settings['dpi']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ImageExporter(settings={self.settings})"

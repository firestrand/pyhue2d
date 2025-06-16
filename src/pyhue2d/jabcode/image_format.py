"""Image format detection and conversion utilities for JABCode operations.

This module provides utilities for detecting image formats, validating image
files, and converting between different image formats as needed for JABCode
encoding and decoding operations.
"""

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile

from .exceptions import JABCodeFormatError, JABCodeImageError, image_error


class ImageFormatDetector:
    """Detects and validates image formats for JABCode operations.

    Provides comprehensive image format detection using multiple methods
    including file extensions, MIME types, and magic number detection.
    """

    # Supported image formats for JABCode operations
    SUPPORTED_FORMATS = {
        "PNG": {
            "extensions": [".png"],
            "mime_types": ["image/png"],
            "magic_bytes": [b"\x89PNG\r\n\x1a\n"],
            "description": "Portable Network Graphics (recommended for JABCode)",
            "supports_transparency": True,
            "lossless": True,
        },
        "JPEG": {
            "extensions": [".jpg", ".jpeg"],
            "mime_types": ["image/jpeg"],
            "magic_bytes": [b"\xff\xd8\xff"],
            "description": "JPEG (lossy compression, not recommended for small codes)",
            "supports_transparency": False,
            "lossless": False,
        },
        "BMP": {
            "extensions": [".bmp"],
            "mime_types": ["image/bmp"],
            "magic_bytes": [b"BM"],
            "description": "Windows Bitmap (uncompressed)",
            "supports_transparency": False,
            "lossless": True,
        },
        "TIFF": {
            "extensions": [".tiff", ".tif"],
            "mime_types": ["image/tiff"],
            "magic_bytes": [b"II*\x00", b"MM\x00*"],
            "description": "Tagged Image File Format",
            "supports_transparency": True,
            "lossless": True,
        },
        "WEBP": {
            "extensions": [".webp"],
            "mime_types": ["image/webp"],
            "magic_bytes": [b"RIFF", b"WEBP"],
            "description": "WebP (modern format with good compression)",
            "supports_transparency": True,
            "lossless": True,  # Can be lossless or lossy
        },
    }

    def __init__(self):
        """Initialize image format detector."""
        # Enable loading of truncated images (helpful for debugging)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def detect_format(self, image_path: Union[str, Path]) -> str:
        """Detect image format using multiple methods.

        Args:
            image_path: Path to the image file

        Returns:
            Detected format name (e.g., 'PNG', 'JPEG')

        Raises:
            JABCodeFormatError: If format cannot be detected or is unsupported
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            raise image_error(f"Image file does not exist: {image_path}", image_path=str(image_path))

        # Try multiple detection methods
        format_from_extension = self._detect_by_extension(image_path)
        format_from_magic = self._detect_by_magic_bytes(image_path)
        format_from_pil = self._detect_by_pil(image_path)

        # Prioritize PIL detection as most reliable
        if format_from_pil:
            detected_format = format_from_pil
        elif format_from_magic:
            detected_format = format_from_magic
        elif format_from_extension:
            detected_format = format_from_extension
        else:
            raise JABCodeFormatError(
                f"Unable to detect image format for: {image_path}",
                error_code="FORMAT_DETECTION_FAILED",
                context={"image_path": str(image_path)},
            )

        # Validate format is supported
        if detected_format not in self.SUPPORTED_FORMATS:
            supported = list(self.SUPPORTED_FORMATS.keys())
            raise JABCodeFormatError(
                f"Unsupported image format: {detected_format}. " f"Supported formats: {', '.join(supported)}",
                error_code="UNSUPPORTED_FORMAT",
                context={
                    "detected_format": detected_format,
                    "supported_formats": supported,
                    "image_path": str(image_path),
                },
            )

        return detected_format

    def _detect_by_extension(self, image_path: Path) -> Optional[str]:
        """Detect format by file extension.

        Args:
            image_path: Path to the image file

        Returns:
            Format name if detected, None otherwise
        """
        extension = image_path.suffix.lower()

        for format_name, format_info in self.SUPPORTED_FORMATS.items():
            if extension in format_info["extensions"]:
                return format_name

        return None

    def _detect_by_magic_bytes(self, image_path: Path) -> Optional[str]:
        """Detect format by magic bytes at start of file.

        Args:
            image_path: Path to the image file

        Returns:
            Format name if detected, None otherwise
        """
        try:
            with open(image_path, "rb") as f:
                header = f.read(32)  # Read first 32 bytes

            for format_name, format_info in self.SUPPORTED_FORMATS.items():
                for magic in format_info["magic_bytes"]:
                    if header.startswith(magic):
                        return format_name
                    # Special case for WEBP (RIFF...WEBP)
                    if format_name == "WEBP" and magic == b"WEBP" and b"WEBP" in header[:12]:
                        return format_name

            return None

        except (OSError, IOError):
            return None

    def _detect_by_pil(self, image_path: Path) -> Optional[str]:
        """Detect format using PIL.

        Args:
            image_path: Path to the image file

        Returns:
            Format name if detected, None otherwise
        """
        try:
            with Image.open(image_path) as img:
                pil_format = img.format

                # Map PIL format names to our format names
                format_mapping = {
                    "PNG": "PNG",
                    "JPEG": "JPEG",
                    "BMP": "BMP",
                    "TIFF": "TIFF",
                    "WEBP": "WEBP",
                }

                return format_mapping.get(pil_format)

        except Exception:
            return None

    def validate_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate image file and return detailed information.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image validation results and metadata

        Raises:
            JABCodeImageError: If image validation fails
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)

        # Detect format
        format_name = self.detect_format(image_path)
        format_info = self.SUPPORTED_FORMATS[format_name]

        # Get file stats
        file_stats = image_path.stat()

        try:
            # Open and analyze image
            with Image.open(image_path) as img:
                # Basic image info
                image_info = {
                    "format": format_name,
                    "pil_format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                    "file_size": file_stats.st_size,
                    "path": str(image_path),
                }

                # Format-specific info
                image_info.update(format_info)

                # JABCode suitability analysis
                suitability = self._analyze_jabcode_suitability(img, format_info)
                image_info["jabcode_suitability"] = suitability

                # Recommendations
                recommendations = self._generate_recommendations(image_info)
                image_info["recommendations"] = recommendations

                return image_info

        except Exception as e:
            raise JABCodeImageError(
                f"Failed to validate image: {e}",
                error_code="IMAGE_VALIDATION_FAILED",
                context={
                    "image_path": str(image_path),
                    "format": format_name,
                    "error": str(e),
                },
            ) from e

    def _analyze_jabcode_suitability(self, img: Image.Image, format_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image suitability for JABCode operations.

        Args:
            img: PIL Image object
            format_info: Format information dictionary

        Returns:
            Suitability analysis results
        """
        # Calculate image statistics
        img_array = np.array(img)

        suitability = {
            "overall_score": 0,  # 0-100 score
            "issues": [],
            "warnings": [],
            "strengths": [],
        }

        score = 100

        # Size analysis
        width, height = img.size
        if width < 50 or height < 50:
            suitability["issues"].append("Image too small (minimum 50x50 recommended)")
            score -= 30
        elif width < 100 or height < 100:
            suitability["warnings"].append("Small image size may affect detection accuracy")
            score -= 10
        else:
            suitability["strengths"].append("Good image size for pattern detection")

        # Format analysis
        if format_info["lossless"]:
            suitability["strengths"].append("Lossless format preserves JABCode patterns")
        else:
            suitability["warnings"].append("Lossy format may degrade JABCode patterns")
            score -= 15

        # Color mode analysis
        if img.mode == "RGB":
            suitability["strengths"].append("RGB mode is ideal for color JABCode")
        elif img.mode == "RGBA":
            suitability["strengths"].append("RGBA mode supports transparency")
        elif img.mode in ("L", "P"):
            suitability["warnings"].append("Grayscale/palette mode may limit color detection")
            score -= 10
        else:
            suitability["issues"].append(f"Unusual color mode: {img.mode}")
            score -= 20

        # Aspect ratio analysis
        aspect_ratio = width / height
        if abs(aspect_ratio - 1.0) > 0.2:
            suitability["warnings"].append("Non-square aspect ratio may indicate perspective distortion")
            score -= 5

        suitability["overall_score"] = max(0, min(100, score))
        return suitability

    def _generate_recommendations(self, image_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving JABCode processing.

        Args:
            image_info: Image information dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Format recommendations
        if image_info["format"] != "PNG":
            recommendations.append("Convert to PNG format for best results")

        # Size recommendations
        if image_info["width"] < 200 or image_info["height"] < 200:
            recommendations.append("Use higher resolution images for better pattern detection")

        # Color mode recommendations
        if image_info["mode"] not in ("RGB", "RGBA"):
            recommendations.append("Convert to RGB color mode for optimal color detection")

        # Compression recommendations
        if not image_info["lossless"]:
            recommendations.append("Use lossless compression to preserve pattern integrity")

        return recommendations

    def get_supported_formats(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported formats.

        Returns:
            Dictionary of supported formats and their capabilities
        """
        return self.SUPPORTED_FORMATS.copy()

    def is_format_supported(self, format_name: str) -> bool:
        """Check if a format is supported.

        Args:
            format_name: Format name to check

        Returns:
            True if format is supported, False otherwise
        """
        return format_name.upper() in self.SUPPORTED_FORMATS

    def get_format_info(self, format_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific format.

        Args:
            format_name: Format name

        Returns:
            Format information dictionary

        Raises:
            JABCodeFormatError: If format is not supported
        """
        format_name = format_name.upper()
        if not self.is_format_supported(format_name):
            raise JABCodeFormatError(
                f"Unsupported format: {format_name}",
                error_code="UNSUPPORTED_FORMAT",
                context={"format": format_name},
            )

        return self.SUPPORTED_FORMATS[format_name].copy()


class ImageConverter:
    """Converts images between different formats for JABCode operations.

    Provides format conversion with optimization for JABCode encoding and decoding.
    """

    def __init__(self, format_detector: Optional[ImageFormatDetector] = None):
        """Initialize image converter.

        Args:
            format_detector: Optional format detector instance
        """
        self.format_detector = format_detector or ImageFormatDetector()

    def convert_image(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "PNG",
        quality: int = 95,
        optimize: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert image to target format.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            target_format: Target format ('PNG', 'JPEG', etc.)
            quality: JPEG quality (1-100)
            optimize: Enable optimization
            **kwargs: Additional PIL save parameters

        Returns:
            Conversion results dictionary

        Raises:
            JABCodeImageError: If conversion fails
        """
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Validate input
        input_format = self.format_detector.detect_format(input_path)

        # Validate target format
        target_format = target_format.upper()
        if not self.format_detector.is_format_supported(target_format):
            raise JABCodeFormatError(
                f"Unsupported target format: {target_format}",
                error_code="UNSUPPORTED_TARGET_FORMAT",
                context={"target_format": target_format},
            )

        try:
            with Image.open(input_path) as img:
                # Prepare image for target format
                converted_img = self._prepare_for_format(img, target_format)

                # Prepare save parameters
                save_params = self._get_save_parameters(target_format, quality, optimize, **kwargs)

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save converted image
                converted_img.save(output_path, **save_params)

                # Get result info
                result = {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_format": input_format,
                    "target_format": target_format,
                    "input_size": img.size,
                    "output_size": converted_img.size,
                    "input_mode": img.mode,
                    "output_mode": converted_img.mode,
                    "conversion_successful": True,
                }

                return result

        except Exception as e:
            raise JABCodeImageError(
                f"Image conversion failed: {e}",
                error_code="CONVERSION_FAILED",
                context={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_format": input_format,
                    "target_format": target_format,
                    "error": str(e),
                },
            ) from e

    def _prepare_for_format(self, img: Image.Image, target_format: str) -> Image.Image:
        """Prepare image for specific target format.

        Args:
            img: Source image
            target_format: Target format name

        Returns:
            Prepared image
        """
        # Make a copy to avoid modifying original
        prepared_img = img.copy()

        if target_format == "JPEG":
            # JPEG doesn't support transparency
            if prepared_img.mode in ("RGBA", "LA"):
                # Create white background
                background = Image.new("RGB", prepared_img.size, (255, 255, 255))
                if prepared_img.mode == "RGBA":
                    background.paste(prepared_img, mask=prepared_img.split()[-1])
                else:
                    background.paste(prepared_img)
                prepared_img = background
            elif prepared_img.mode != "RGB":
                prepared_img = prepared_img.convert("RGB")

        elif target_format == "PNG":
            # PNG supports all modes, but ensure we have the best mode
            if prepared_img.mode == "P":
                # Convert palette to RGB/RGBA
                if "transparency" in prepared_img.info:
                    prepared_img = prepared_img.convert("RGBA")
                else:
                    prepared_img = prepared_img.convert("RGB")

        elif target_format in ("BMP", "TIFF"):
            # These formats have good mode support
            pass

        elif target_format == "WEBP":
            # WebP supports most modes
            pass

        return prepared_img

    def _get_save_parameters(self, target_format: str, quality: int, optimize: bool, **kwargs) -> Dict[str, Any]:
        """Get save parameters for target format.

        Args:
            target_format: Target format name
            quality: Quality setting
            optimize: Optimization setting
            **kwargs: Additional parameters

        Returns:
            Save parameters dictionary
        """
        params = {"format": target_format}

        if target_format == "JPEG":
            params.update(
                {
                    "quality": quality,
                    "optimize": optimize,
                    "progressive": kwargs.get("progressive", False),
                }
            )
        elif target_format == "PNG":
            params.update(
                {
                    "optimize": optimize,
                    "compress_level": kwargs.get("compress_level", 6),
                }
            )
        elif target_format == "WEBP":
            params.update(
                {
                    "quality": quality,
                    "method": kwargs.get("method", 6),
                    "lossless": kwargs.get("lossless", False),
                }
            )
        elif target_format == "TIFF":
            params.update({"compression": kwargs.get("compression", "lzw")})

        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        return params

    def optimize_for_jabcode(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Dict[str, Any]:
        """Optimize image specifically for JABCode operations.

        Args:
            input_path: Path to input image
            output_path: Path for optimized output

        Returns:
            Optimization results dictionary
        """
        # Convert to PNG (best format for JABCode)
        return self.convert_image(
            input_path=input_path,
            output_path=output_path,
            target_format="PNG",
            optimize=True,
            compress_level=6,
        )

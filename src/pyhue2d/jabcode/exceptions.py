"""JABCode-specific exception hierarchy.

This module defines a comprehensive exception hierarchy for JABCode operations,
providing specific error types for different failure modes.
"""

from typing import Any, Dict, Optional


class JABCodeError(Exception):
    """Base exception for all JABCode-related errors.

    This is the root exception class that all other JABCode exceptions inherit from.
    It provides a consistent interface for error handling and includes optional
    error context information.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize JABCode error.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class JABCodeValidationError(JABCodeError):
    """Raised when input validation fails.

    This includes invalid parameters, unsupported values, or malformed input data.
    """

    pass


class JABCodeEncodingError(JABCodeError):
    """Raised when encoding operations fail.

    This covers errors during the encoding process, including data preparation,
    symbol generation, and bitmap creation.
    """

    pass


class JABCodeDecodingError(JABCodeError):
    """Raised when decoding operations fail.

    This covers errors during the decoding process, including pattern detection,
    perspective correction, and data reconstruction.
    """

    pass


class JABCodeImageError(JABCodeError):
    """Raised when image processing operations fail.

    This includes errors with image loading, format conversion, and image manipulation.
    """

    pass


class JABCodePatternError(JABCodeDecodingError):
    """Raised when finder pattern detection fails.

    This is a specific type of decoding error for pattern-related failures.
    """

    pass


class JABCodePerspectiveError(JABCodeDecodingError):
    """Raised when perspective correction fails.

    This is a specific type of decoding error for geometric transformation failures.
    """

    pass


class JABCodeDataError(JABCodeError):
    """Raised when data processing fails.

    This covers errors in data validation, encoding mode detection, and data formatting.
    """

    pass


class JABCodeLDPCError(JABCodeError):
    """Raised when LDPC error correction fails.

    This covers errors in both encoding and decoding LDPC operations.
    """

    pass


class JABCodeVersionError(JABCodeValidationError):
    """Raised when symbol version calculation or validation fails.

    This is a specific type of validation error for version-related failures.
    """

    pass


class JABCodeCapacityError(JABCodeValidationError):
    """Raised when data exceeds symbol capacity.

    This is a specific type of validation error for capacity-related failures.
    """

    pass


class JABCodeFormatError(JABCodeImageError):
    """Raised when image format is unsupported or invalid.

    This is a specific type of image error for format-related failures.
    """

    pass


# Convenience functions for creating specific errors
def validation_error(message: str, field: Optional[str] = None, value: Any = None) -> JABCodeValidationError:
    """Create a validation error with context.

    Args:
        message: Error message
        field: Optional field name that failed validation
        value: Optional value that failed validation

    Returns:
        JABCodeValidationError with context
    """
    context = {}
    if field is not None:
        context["field"] = field
    if value is not None:
        context["value"] = value

    return JABCodeValidationError(message, error_code="VALIDATION_FAILED", context=context)


def encoding_error(message: str, stage: Optional[str] = None) -> JABCodeEncodingError:
    """Create an encoding error with context.

    Args:
        message: Error message
        stage: Optional encoding stage where error occurred

    Returns:
        JABCodeEncodingError with context
    """
    context = {}
    if stage is not None:
        context["stage"] = stage

    return JABCodeEncodingError(message, error_code="ENCODING_FAILED", context=context)


def decoding_error(message: str, stage: Optional[str] = None) -> JABCodeDecodingError:
    """Create a decoding error with context.

    Args:
        message: Error message
        stage: Optional decoding stage where error occurred

    Returns:
        JABCodeDecodingError with context
    """
    context = {}
    if stage is not None:
        context["stage"] = stage

    return JABCodeDecodingError(message, error_code="DECODING_FAILED", context=context)


def pattern_error(message: str, pattern_count: Optional[int] = None) -> JABCodePatternError:
    """Create a pattern detection error with context.

    Args:
        message: Error message
        pattern_count: Optional number of patterns detected

    Returns:
        JABCodePatternError with context
    """
    context = {}
    if pattern_count is not None:
        context["pattern_count"] = pattern_count

    return JABCodePatternError(message, error_code="PATTERN_DETECTION_FAILED", context=context)


def image_error(
    message: str, image_path: Optional[str] = None, image_size: Optional[tuple] = None
) -> JABCodeImageError:
    """Create an image processing error with context.

    Args:
        message: Error message
        image_path: Optional path to the image that caused the error
        image_size: Optional size of the image that caused the error

    Returns:
        JABCodeImageError with context
    """
    context = {}
    if image_path is not None:
        context["image_path"] = image_path
    if image_size is not None:
        context["image_size"] = image_size

    return JABCodeImageError(message, error_code="IMAGE_PROCESSING_FAILED", context=context)


def capacity_error(data_size: int, max_capacity: int, version: Optional[int] = None) -> JABCodeCapacityError:
    """Create a capacity exceeded error with context.

    Args:
        data_size: Size of data that failed to fit
        max_capacity: Maximum capacity of the symbol
        version: Optional symbol version

    Returns:
        JABCodeCapacityError with context
    """
    message = f"Data size {data_size} bytes exceeds maximum capacity {max_capacity} bytes"
    context = {"data_size": data_size, "max_capacity": max_capacity}
    if version is not None:
        context["version"] = version
        message += f" for version {version}"

    return JABCodeCapacityError(message, error_code="CAPACITY_EXCEEDED", context=context)

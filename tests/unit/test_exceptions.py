"""Unit tests for JABCode exception hierarchy."""

import pytest

from pyhue2d.jabcode.exceptions import (
    JABCodeCapacityError,
    JABCodeDataError,
    JABCodeDecodingError,
    JABCodeEncodingError,
    JABCodeError,
    JABCodeFormatError,
    JABCodeImageError,
    JABCodeLDPCError,
    JABCodePatternError,
    JABCodePerspectiveError,
    JABCodeValidationError,
    JABCodeVersionError,
    capacity_error,
    decoding_error,
    encoding_error,
    image_error,
    pattern_error,
    validation_error,
)


class TestJABCodeError:
    """Test cases for base JABCodeError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = JABCodeError("Test message")
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.error_code is None
        assert error.context == {}

    def test_error_with_code(self):
        """Test error with error code."""
        error = JABCodeError("Test message", error_code="TEST_ERROR")
        assert str(error) == "[TEST_ERROR] Test message"
        assert error.error_code == "TEST_ERROR"

    def test_error_with_context(self):
        """Test error with context."""
        context = {"field": "test_field", "value": 42}
        error = JABCodeError("Test message", context=context)
        assert error.context == context

    def test_error_inheritance(self):
        """Test that JABCodeError inherits from Exception."""
        error = JABCodeError("Test message")
        assert isinstance(error, Exception)


class TestSpecificErrors:
    """Test cases for specific error types."""

    def test_validation_error(self):
        """Test JABCodeValidationError."""
        error = JABCodeValidationError("Invalid value")
        assert isinstance(error, JABCodeError)
        assert str(error) == "Invalid value"

    def test_encoding_error(self):
        """Test JABCodeEncodingError."""
        error = JABCodeEncodingError("Encoding failed")
        assert isinstance(error, JABCodeError)
        assert str(error) == "Encoding failed"

    def test_decoding_error(self):
        """Test JABCodeDecodingError."""
        error = JABCodeDecodingError("Decoding failed")
        assert isinstance(error, JABCodeError)
        assert str(error) == "Decoding failed"

    def test_image_error(self):
        """Test JABCodeImageError."""
        error = JABCodeImageError("Image processing failed")
        assert isinstance(error, JABCodeError)
        assert str(error) == "Image processing failed"

    def test_pattern_error_inheritance(self):
        """Test JABCodePatternError inherits from JABCodeDecodingError."""
        error = JABCodePatternError("Pattern not found")
        assert isinstance(error, JABCodeDecodingError)
        assert isinstance(error, JABCodeError)

    def test_perspective_error_inheritance(self):
        """Test JABCodePerspectiveError inherits from JABCodeDecodingError."""
        error = JABCodePerspectiveError("Perspective correction failed")
        assert isinstance(error, JABCodeDecodingError)
        assert isinstance(error, JABCodeError)

    def test_version_error_inheritance(self):
        """Test JABCodeVersionError inherits from JABCodeValidationError."""
        error = JABCodeVersionError("Invalid version")
        assert isinstance(error, JABCodeValidationError)
        assert isinstance(error, JABCodeError)

    def test_capacity_error_inheritance(self):
        """Test JABCodeCapacityError inherits from JABCodeValidationError."""
        error = JABCodeCapacityError("Capacity exceeded")
        assert isinstance(error, JABCodeValidationError)
        assert isinstance(error, JABCodeError)

    def test_format_error_inheritance(self):
        """Test JABCodeFormatError inherits from JABCodeImageError."""
        error = JABCodeFormatError("Unsupported format")
        assert isinstance(error, JABCodeImageError)
        assert isinstance(error, JABCodeError)


class TestConvenienceFunctions:
    """Test cases for convenience error creation functions."""

    def test_validation_error_function(self):
        """Test validation_error convenience function."""
        error = validation_error("Invalid input", field="test_field", value=123)

        assert isinstance(error, JABCodeValidationError)
        assert error.message == "Invalid input"
        assert error.error_code == "VALIDATION_FAILED"
        assert error.context["field"] == "test_field"
        assert error.context["value"] == 123

    def test_validation_error_minimal(self):
        """Test validation_error with minimal parameters."""
        error = validation_error("Invalid input")

        assert isinstance(error, JABCodeValidationError)
        assert error.message == "Invalid input"
        assert error.error_code == "VALIDATION_FAILED"
        assert error.context == {}

    def test_encoding_error_function(self):
        """Test encoding_error convenience function."""
        error = encoding_error("Encoding failed", stage="data_processing")

        assert isinstance(error, JABCodeEncodingError)
        assert error.message == "Encoding failed"
        assert error.error_code == "ENCODING_FAILED"
        assert error.context["stage"] == "data_processing"

    def test_decoding_error_function(self):
        """Test decoding_error convenience function."""
        error = decoding_error("Decoding failed", stage="pattern_detection")

        assert isinstance(error, JABCodeDecodingError)
        assert error.message == "Decoding failed"
        assert error.error_code == "DECODING_FAILED"
        assert error.context["stage"] == "pattern_detection"

    def test_pattern_error_function(self):
        """Test pattern_error convenience function."""
        error = pattern_error("No patterns found", pattern_count=0)

        assert isinstance(error, JABCodePatternError)
        assert error.message == "No patterns found"
        assert error.error_code == "PATTERN_DETECTION_FAILED"
        assert error.context["pattern_count"] == 0

    def test_image_error_function(self):
        """Test image_error convenience function."""
        error = image_error("Cannot open image", image_path="/test/path.png", image_size=(100, 100))

        assert isinstance(error, JABCodeImageError)
        assert error.message == "Cannot open image"
        assert error.error_code == "IMAGE_PROCESSING_FAILED"
        assert error.context["image_path"] == "/test/path.png"
        assert error.context["image_size"] == (100, 100)

    def test_capacity_error_function(self):
        """Test capacity_error convenience function."""
        error = capacity_error(1000, 800, version=5)

        assert isinstance(error, JABCodeCapacityError)
        assert "Data size 1000 bytes exceeds maximum capacity 800 bytes for version 5" in error.message
        assert error.error_code == "CAPACITY_EXCEEDED"
        assert error.context["data_size"] == 1000
        assert error.context["max_capacity"] == 800
        assert error.context["version"] == 5

    def test_capacity_error_without_version(self):
        """Test capacity_error without version parameter."""
        error = capacity_error(1000, 800)

        assert isinstance(error, JABCodeCapacityError)
        assert "Data size 1000 bytes exceeds maximum capacity 800 bytes" in error.message
        assert "version" not in error.context


class TestErrorChaining:
    """Test error chaining and exception handling patterns."""

    def test_error_chain_with_cause(self):
        """Test error chaining with original exception."""
        original = ValueError("Original error")

        try:
            raise original
        except ValueError as e:
            jabcode_error = JABCodeError("JABCode processing failed")
            jabcode_error.__cause__ = e

        assert jabcode_error.__cause__ is original

    def test_context_preservation(self):
        """Test that context is preserved during error handling."""
        context = {"operation": "encode", "data_size": 1024}
        error = JABCodeError("Operation failed", error_code="OP_FAILED", context=context)

        # Simulate error handling that preserves context
        assert error.context["operation"] == "encode"
        assert error.context["data_size"] == 1024

    def test_multiple_error_types(self):
        """Test handling of multiple error types."""
        errors = [
            JABCodeValidationError("Validation failed"),
            JABCodeEncodingError("Encoding failed"),
            JABCodeDecodingError("Decoding failed"),
            JABCodeImageError("Image failed"),
        ]

        for error in errors:
            assert isinstance(error, JABCodeError)

        # Test polymorphic handling
        validation_errors = [e for e in errors if isinstance(e, JABCodeValidationError)]
        assert len(validation_errors) == 1

        processing_errors = [e for e in errors if isinstance(e, (JABCodeEncodingError, JABCodeDecodingError))]
        assert len(processing_errors) == 2


class TestErrorMessages:
    """Test error message formatting and readability."""

    def test_error_message_clarity(self):
        """Test that error messages are clear and helpful."""
        error = validation_error(
            "Invalid palette size: 7. Must be one of [4, 8, 16, 32, 64, 128, 256]",
            field="palette",
            value=7,
        )

        assert "Invalid palette size" in error.message
        assert "Must be one of" in error.message
        assert error.context["value"] == 7

    def test_error_code_consistency(self):
        """Test that error codes follow consistent naming."""
        errors_with_codes = [
            validation_error("test"),
            encoding_error("test"),
            decoding_error("test"),
            pattern_error("test"),
            image_error("test"),
            capacity_error(100, 50),
        ]

        for error in errors_with_codes:
            assert error.error_code is not None
            assert "_" in error.error_code  # Snake case
            assert error.error_code.isupper()  # All uppercase
            assert error.error_code.endswith("_FAILED") or error.error_code.endswith("_EXCEEDED")


if __name__ == "__main__":
    pytest.main([__file__])

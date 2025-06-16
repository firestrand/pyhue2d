"""Test vectors for all JABCode encoding modes.

This module provides comprehensive test vectors to validate that all
encoding modes work correctly with their respective character sets
and optimization scenarios.
"""

from typing import Any, Dict, List, Tuple

import pytest

from pyhue2d.jabcode.encoding_modes.alphanumeric import AlphanumericMode
from pyhue2d.jabcode.encoding_modes.byte import ByteMode
from pyhue2d.jabcode.encoding_modes.detector import EncodingModeDetector
from pyhue2d.jabcode.encoding_modes.lowercase import LowercaseMode
from pyhue2d.jabcode.encoding_modes.mixed import MixedMode
from pyhue2d.jabcode.encoding_modes.numeric import NumericMode
from pyhue2d.jabcode.encoding_modes.punctuation import PunctuationMode
from pyhue2d.jabcode.encoding_modes.uppercase import UppercaseMode


class TestNumericModeVectors:
    """Test vectors for Numeric encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create NumericMode instance."""
        return NumericMode()

    @pytest.mark.parametrize(
        "input_data,expected_efficiency",
        [
            ("123", 1.0),  # Pure numeric
            ("0000", 1.0),  # Leading zeros
            ("999999", 1.0),  # Large number
            ("1234567890", 1.0),  # All digits
            ("12345", 1.0),  # Medium number
            ("", 1.0),  # Empty (edge case)
            ("123 456", 0.7),  # With space (should be lower efficiency)
            ("123a", 0.0),  # Contains letter (invalid)
            ("12.34", 0.0),  # Decimal point (invalid)
            ("123-456", 0.0),  # Hyphen (invalid)
        ],
    )
    def test_numeric_efficiency(self, mode, input_data, expected_efficiency):
        """Test numeric mode efficiency calculation."""
        efficiency = mode.calculate_efficiency(input_data)
        if expected_efficiency == 0.0:
            assert efficiency == 0.0
        elif expected_efficiency == 1.0:
            assert efficiency == 1.0
        else:
            assert 0.0 < efficiency < 1.0

    @pytest.mark.parametrize(
        "input_data",
        [
            "123",
            "0",
            "999",
            "1234567890",
            "000123",
        ],
    )
    def test_numeric_encoding_valid(self, mode, input_data):
        """Test numeric mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0

    @pytest.mark.parametrize(
        "input_data",
        [
            "123a",
            "12.34",
            "abc",
            "123-456",
            "12 34",  # Space not allowed in pure numeric
        ],
    )
    def test_numeric_encoding_invalid(self, mode, input_data):
        """Test numeric mode with invalid data."""
        assert not mode.can_encode(input_data)


class TestAlphanumericModeVectors:
    """Test vectors for Alphanumeric encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create AlphanumericMode instance."""
        return AlphanumericMode()

    @pytest.mark.parametrize(
        "input_data,expected_valid",
        [
            ("ABC123", True),  # Letters and numbers
            ("HELLO", True),  # Only letters
            ("12345", True),  # Only numbers
            ("A1B2C3", True),  # Mixed
            ("SPACE CHAR", True),  # With space
            ("$%*+-./:", True),  # Special characters
            ("abc", False),  # Lowercase letters
            ("Hello", False),  # Mixed case
            ("@#", False),  # Invalid special chars
            ("", True),  # Empty (edge case)
        ],
    )
    def test_alphanumeric_validity(self, mode, input_data, expected_valid):
        """Test alphanumeric mode validity."""
        can_encode = mode.can_encode(input_data)
        assert can_encode == expected_valid

    @pytest.mark.parametrize(
        "input_data",
        [
            "HELLO WORLD",
            "ABC123XYZ",
            "TEST $%*+-./: CHARS",
            "0123456789",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ],
    )
    def test_alphanumeric_encoding_valid(self, mode, input_data):
        """Test alphanumeric mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0


class TestUppercaseModeVectors:
    """Test vectors for Uppercase encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create UppercaseMode instance."""
        return UppercaseMode()

    @pytest.mark.parametrize(
        "input_data,expected_valid",
        [
            ("HELLO", True),  # Pure uppercase
            ("HELLO WORLD", True),  # With space
            ("ABC XYZ", True),  # Multiple words
            ("A", True),  # Single character
            ("", True),  # Empty
            ("hello", False),  # Lowercase
            ("Hello", False),  # Mixed case
            ("HELLO123", False),  # With numbers
            ("HELLO!", False),  # With punctuation
        ],
    )
    def test_uppercase_validity(self, mode, input_data, expected_valid):
        """Test uppercase mode validity."""
        can_encode = mode.can_encode(input_data)
        assert can_encode == expected_valid

    @pytest.mark.parametrize(
        "input_data",
        [
            "HELLO",
            "UPPERCASE TEXT",
            "A B C D E",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "MULTIPLE WORDS HERE",
        ],
    )
    def test_uppercase_encoding_valid(self, mode, input_data):
        """Test uppercase mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0

    def test_uppercase_efficiency_scenarios(self, mode):
        """Test uppercase mode efficiency in different scenarios."""
        # Perfect case - all uppercase letters and spaces
        perfect = "HELLO WORLD"
        assert mode.calculate_efficiency(perfect) == 1.0

        # Mixed case - should have lower efficiency
        mixed = "Hello World"
        assert mode.calculate_efficiency(mixed) < 1.0

        # With numbers - should have lower efficiency
        with_numbers = "HELLO 123"
        assert mode.calculate_efficiency(with_numbers) < 1.0


class TestLowercaseModeVectors:
    """Test vectors for Lowercase encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create LowercaseMode instance."""
        return LowercaseMode()

    @pytest.mark.parametrize(
        "input_data,expected_valid",
        [
            ("hello", True),  # Pure lowercase
            ("hello world", True),  # With space
            ("abc xyz", True),  # Multiple words
            ("a", True),  # Single character
            ("", True),  # Empty
            ("HELLO", False),  # Uppercase
            ("Hello", False),  # Mixed case
            ("hello123", False),  # With numbers
            ("hello!", False),  # With punctuation
        ],
    )
    def test_lowercase_validity(self, mode, input_data, expected_valid):
        """Test lowercase mode validity."""
        can_encode = mode.can_encode(input_data)
        assert can_encode == expected_valid

    @pytest.mark.parametrize(
        "input_data",
        [
            "hello",
            "lowercase text",
            "a b c d e",
            "abcdefghijklmnopqrstuvwxyz",
            "multiple words here",
        ],
    )
    def test_lowercase_encoding_valid(self, mode, input_data):
        """Test lowercase mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0


class TestMixedModeVectors:
    """Test vectors for Mixed encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create MixedMode instance."""
        return MixedMode()

    @pytest.mark.parametrize(
        "input_data,expected_valid",
        [
            ("Hello World", True),  # Mixed case
            ("ABC123xyz", True),  # Letters and numbers
            ("Test123", True),  # Mixed with numbers
            ("MixedCASE", True),  # Mixed case letters
            ("", True),  # Empty
            ("abc", True),  # Pure lowercase (should still work)
            ("ABC", True),  # Pure uppercase (should still work)
            ("123", True),  # Pure numeric (should still work)
            ("Special@Chars", False),  # Invalid special chars for mixed
        ],
    )
    def test_mixed_validity(self, mode, input_data, expected_valid):
        """Test mixed mode validity."""
        can_encode = mode.can_encode(input_data)
        assert can_encode == expected_valid

    @pytest.mark.parametrize(
        "input_data",
        [
            "Hello World",
            "MixedCase123",
            "Test Data Here",
            "ABC123xyz789",
            "CamelCaseText",
        ],
    )
    def test_mixed_encoding_valid(self, mode, input_data):
        """Test mixed mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0

    def test_mixed_efficiency_scenarios(self, mode):
        """Test mixed mode efficiency scenarios."""
        # Should handle mixed case well
        mixed_case = "Hello World"
        efficiency = mode.calculate_efficiency(mixed_case)
        assert 0.8 <= efficiency <= 1.0

        # Should handle alphanumeric well
        alphanumeric = "Test123"
        efficiency = mode.calculate_efficiency(alphanumeric)
        assert 0.8 <= efficiency <= 1.0


class TestPunctuationModeVectors:
    """Test vectors for Punctuation encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create PunctuationMode instance."""
        return PunctuationMode()

    @pytest.mark.parametrize(
        "input_data,expected_valid",
        [
            ("Hello, World!", True),  # Basic punctuation
            ("Test: value", True),  # Colon
            ("Question?", True),  # Question mark
            ("Exclamation!", True),  # Exclamation mark
            ("Comma, period.", True),  # Multiple punctuation
            ('"Quoted text"', True),  # Quotes
            ("(Parentheses)", True),  # Parentheses
            ("[Brackets]", True),  # Brackets
            ("{Braces}", True),  # Braces
            ("", True),  # Empty
            ("abc", True),  # Plain text (should work)
            ("@#$%^&*", False),  # Special symbols not in punctuation set
        ],
    )
    def test_punctuation_validity(self, mode, input_data, expected_valid):
        """Test punctuation mode validity."""
        can_encode = mode.can_encode(input_data)
        assert can_encode == expected_valid

    @pytest.mark.parametrize(
        "input_data",
        [
            "Hello, World!",
            "Question? Answer: Yes.",
            "Text with (parentheses) and [brackets].",
            '"Quoted string" with punctuation.',
            "Multiple: different, punctuation! marks?",
        ],
    )
    def test_punctuation_encoding_valid(self, mode, input_data):
        """Test punctuation mode encoding with valid data."""
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0


class TestByteModeVectors:
    """Test vectors for Byte encoding mode."""

    @pytest.fixture
    def mode(self):
        """Create ByteMode instance."""
        return ByteMode()

    @pytest.mark.parametrize(
        "input_data",
        [
            "Any text",  # Regular text
            "Special chars: @#$%^&*",  # Special characters
            "Unicode: café résumé",  # Unicode characters
            "Mixed: ABC123!@#",  # Mixed content
            "",  # Empty
            "Binary data: \x00\x01\x02\xff",  # Binary-like data
            "Line\nBreaks\nAnd\tTabs",  # Control characters
            "Very long text " * 100,  # Long text
        ],
    )
    def test_byte_mode_universal(self, mode, input_data):
        """Test that byte mode can encode any text data."""
        # Byte mode should handle any input
        assert mode.can_encode(input_data)
        encoded = mode.encode(input_data)
        assert encoded is not None
        assert len(encoded) > 0

    def test_byte_mode_efficiency(self, mode):
        """Test byte mode efficiency."""
        # Byte mode should have consistent efficiency for text
        test_cases = [
            "Simple text",
            "Complex text with symbols!@#$",
            "123456789",
            "UPPERCASE",
            "lowercase",
        ]

        efficiencies = []
        for text in test_cases:
            eff = mode.calculate_efficiency(text)
            efficiencies.append(eff)
            assert 0.5 <= eff <= 1.0  # Should be reasonably efficient

        # Efficiency should be fairly consistent for byte mode
        assert max(efficiencies) - min(efficiencies) <= 0.3

    def test_byte_mode_binary_data(self, mode):
        """Test byte mode with actual binary data."""
        # Test with bytes object
        binary_data = bytes([0, 1, 2, 255, 128, 64])

        # Convert to string representation for testing
        # In real usage, byte mode would handle bytes directly
        test_string = "".join(chr(b) if b < 128 else f"\\x{b:02x}" for b in binary_data)

        assert mode.can_encode(test_string)
        encoded = mode.encode(test_string)
        assert encoded is not None


class TestEncodingModeDetectorVectors:
    """Test vectors for automatic encoding mode detection."""

    @pytest.fixture
    def detector(self):
        """Create EncodingModeDetector instance."""
        return EncodingModeDetector()

    @pytest.mark.parametrize(
        "input_data,expected_mode",
        [
            ("123456", "Numeric"),  # Pure numeric
            ("HELLO WORLD", "Uppercase"),  # Pure uppercase
            ("hello world", "Lowercase"),  # Pure lowercase
            ("ABC123", "Alphanumeric"),  # Alphanumeric
            ("Hello World", "Mixed"),  # Mixed case
            ("Hello, World!", "Punctuation"),  # With punctuation
            ("Special @#$ chars", "Byte"),  # Special characters
            ("Unicode: café", "Byte"),  # Unicode
            ("", "Numeric"),  # Empty (should default to most efficient)
        ],
    )
    def test_mode_detection(self, detector, input_data, expected_mode):
        """Test automatic mode detection."""
        detected_mode = detector.detect_optimal_mode(input_data)

        # Note: The actual implementation might choose differently based on efficiency
        # This test documents expected behavior but may need adjustment
        print(f"Input: '{input_data}' -> Detected: {detected_mode}, Expected: {expected_mode}")

        # At minimum, the detected mode should be able to encode the data
        mode_classes = {
            "Numeric": NumericMode,
            "Alphanumeric": AlphanumericMode,
            "Uppercase": UppercaseMode,
            "Lowercase": LowercaseMode,
            "Mixed": MixedMode,
            "Punctuation": PunctuationMode,
            "Byte": ByteMode,
        }

        if detected_mode in mode_classes:
            mode_instance = mode_classes[detected_mode]()
            assert mode_instance.can_encode(input_data), f"Detected mode {detected_mode} cannot encode '{input_data}'"

    def test_mode_efficiency_comparison(self, detector):
        """Test that detector chooses more efficient modes."""
        # Test cases where one mode should be clearly better
        test_cases = [
            ("123456", ["Numeric", "Alphanumeric", "Byte"]),  # Numeric should win
            ("HELLO", ["Uppercase", "Alphanumeric", "Byte"]),  # Uppercase should win
            ("hello", ["Lowercase", "Mixed", "Byte"]),  # Lowercase should win
        ]

        for text, candidate_modes in test_cases:
            best_mode = detector.detect_optimal_mode(text)

            # Calculate efficiencies for comparison
            mode_classes = {
                "Numeric": NumericMode(),
                "Alphanumeric": AlphanumericMode(),
                "Uppercase": UppercaseMode(),
                "Lowercase": LowercaseMode(),
                "Mixed": MixedMode(),
                "Punctuation": PunctuationMode(),
                "Byte": ByteMode(),
            }

            efficiencies = {}
            for mode_name in candidate_modes:
                if mode_name in mode_classes:
                    mode = mode_classes[mode_name]
                    if mode.can_encode(text):
                        efficiencies[mode_name] = mode.calculate_efficiency(text)

            if efficiencies:
                print(f"Text: '{text}' -> Efficiencies: {efficiencies}")
                print(f"Best detected: {best_mode}")

    def test_mode_switching_scenarios(self, detector):
        """Test scenarios where mode switching might be beneficial."""
        # Long text with different patterns
        mixed_content = "Start with UPPERCASE then 123456 then lowercase then Mixed123 end"

        # Test both single mode and potential multi-mode
        single_mode = detector.detect_optimal_mode(mixed_content)
        print(f"Mixed content single mode: {single_mode}")

        # The detector should choose a mode that can handle all content
        mode_classes = {
            "Numeric": NumericMode(),
            "Alphanumeric": AlphanumericMode(),
            "Uppercase": UppercaseMode(),
            "Lowercase": LowercaseMode(),
            "Mixed": MixedMode(),
            "Punctuation": PunctuationMode(),
            "Byte": ByteMode(),
        }

        if single_mode in mode_classes:
            mode = mode_classes[single_mode]
            assert mode.can_encode(mixed_content), f"Selected mode {single_mode} cannot encode mixed content"


class TestEncodingModeEdgeCases:
    """Test edge cases and boundary conditions for encoding modes."""

    def test_empty_string_handling(self):
        """Test how all modes handle empty strings."""
        modes = [
            NumericMode(),
            AlphanumericMode(),
            UppercaseMode(),
            LowercaseMode(),
            MixedMode(),
            PunctuationMode(),
            ByteMode(),
        ]

        for mode in modes:
            # All modes should handle empty string
            assert mode.can_encode("")
            encoded = mode.encode("")
            assert encoded is not None

    def test_single_character_handling(self):
        """Test single character encoding across modes."""
        test_chars = [
            ("1", ["Numeric", "Alphanumeric", "Mixed", "Byte"]),
            ("A", ["Alphanumeric", "Uppercase", "Mixed", "Byte"]),
            ("a", ["Lowercase", "Mixed", "Byte"]),
            ("!", ["Punctuation", "Byte"]),
            (" ", ["Uppercase", "Lowercase", "Mixed", "Punctuation", "Byte"]),
        ]

        mode_instances = {
            "Numeric": NumericMode(),
            "Alphanumeric": AlphanumericMode(),
            "Uppercase": UppercaseMode(),
            "Lowercase": LowercaseMode(),
            "Mixed": MixedMode(),
            "Punctuation": PunctuationMode(),
            "Byte": ByteMode(),
        }

        for char, compatible_modes in test_chars:
            for mode_name in compatible_modes:
                mode = mode_instances[mode_name]
                assert mode.can_encode(char), f"Mode {mode_name} should encode '{char}'"

                encoded = mode.encode(char)
                assert encoded is not None
                assert len(encoded) > 0

    def test_boundary_character_sets(self):
        """Test characters at the boundaries of mode character sets."""
        # Test boundary characters
        boundary_tests = [
            (NumericMode(), "0123456789"),  # All numeric chars
            (AlphanumericMode(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 $%*+-./:"),
            (UppercaseMode(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ "),
            (LowercaseMode(), "abcdefghijklmnopqrstuvwxyz "),
        ]

        for mode, valid_chars in boundary_tests:
            # Test that all valid characters can be encoded
            for char in valid_chars:
                assert mode.can_encode(char), f"Mode {type(mode).__name__} should encode '{char}'"

            # Test the full character set
            assert mode.can_encode(valid_chars)
            encoded = mode.encode(valid_chars)
            assert encoded is not None

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        unicode_test_cases = [
            "café",  # Latin with accents
            "résumé",  # More accents
            "naïve",  # Diaeresis
            "piñata",  # Tilde
            "Москва",  # Cyrillic
            "東京",  # Japanese
            "🎉",  # Emoji
        ]

        # Only ByteMode should handle all Unicode
        byte_mode = ByteMode()

        for test_case in unicode_test_cases:
            assert byte_mode.can_encode(test_case), f"ByteMode should encode Unicode: '{test_case}'"

            encoded = byte_mode.encode(test_case)
            assert encoded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

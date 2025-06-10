"""Tests for encoding modes."""

import pytest
from pyhue2d.jabcode.encoding_modes import (
    EncodingModeBase,
    UppercaseMode,
    LowercaseMode,
    NumericMode,
    PunctuationMode,
    MixedMode,
    AlphanumericMode,
    ByteMode,
    EncodingModeDetector
)


class TestEncodingModeBase:
    """Test cases for EncodingModeBase."""
    
    def test_base_mode_is_abstract(self):
        """Test that base mode cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EncodingModeBase()


class TestUppercaseMode:
    """Test cases for UppercaseMode."""
    
    def test_uppercase_mode_creation(self):
        """Test UppercaseMode can be created."""
        mode = UppercaseMode()
        assert mode.mode_id == 0
        assert mode.name == "Uppercase"
    
    def test_uppercase_can_encode_uppercase_letters(self):
        """Test UppercaseMode can encode uppercase letters."""
        mode = UppercaseMode()
        assert mode.can_encode("A") is True
        assert mode.can_encode("Z") is True
        assert mode.can_encode("HELLO") is True
    
    def test_uppercase_cannot_encode_lowercase(self):
        """Test UppercaseMode cannot encode lowercase letters."""
        mode = UppercaseMode()
        assert mode.can_encode("a") is False
        assert mode.can_encode("hello") is False
    
    def test_uppercase_cannot_encode_numbers(self):
        """Test UppercaseMode cannot encode numbers."""
        mode = UppercaseMode()
        assert mode.can_encode("1") is False
        assert mode.can_encode("123") is False
    
    def test_uppercase_encode_string(self):
        """Test UppercaseMode can encode string to bits."""
        mode = UppercaseMode()
        bits = mode.encode("ABC")
        assert isinstance(bits, bytes)
        assert len(bits) > 0
    
    def test_uppercase_decode_bits(self):
        """Test UppercaseMode can decode bits back to string."""
        mode = UppercaseMode()
        original = "HELLO"
        bits = mode.encode(original)
        decoded = mode.decode(bits)
        assert decoded == original
    
    def test_uppercase_get_efficiency(self):
        """Test UppercaseMode returns encoding efficiency."""
        mode = UppercaseMode()
        efficiency = mode.get_efficiency("HELLO WORLD")
        assert 0.0 <= efficiency <= 1.0


class TestLowercaseMode:
    """Test cases for LowercaseMode."""
    
    def test_lowercase_mode_creation(self):
        """Test LowercaseMode can be created."""
        mode = LowercaseMode()
        assert mode.mode_id == 1
        assert mode.name == "Lowercase"
    
    def test_lowercase_can_encode_lowercase_letters(self):
        """Test LowercaseMode can encode lowercase letters."""
        mode = LowercaseMode()
        assert mode.can_encode("a") is True
        assert mode.can_encode("z") is True
        assert mode.can_encode("hello") is True
    
    def test_lowercase_cannot_encode_uppercase(self):
        """Test LowercaseMode cannot encode uppercase letters."""
        mode = LowercaseMode()
        assert mode.can_encode("A") is False
        assert mode.can_encode("HELLO") is False


class TestNumericMode:
    """Test cases for NumericMode."""
    
    def test_numeric_mode_creation(self):
        """Test NumericMode can be created."""
        mode = NumericMode()
        assert mode.mode_id == 2
        assert mode.name == "Numeric"
    
    def test_numeric_can_encode_digits(self):
        """Test NumericMode can encode digits."""
        mode = NumericMode()
        assert mode.can_encode("0") is True
        assert mode.can_encode("9") is True
        assert mode.can_encode("12345") is True
    
    def test_numeric_cannot_encode_letters(self):
        """Test NumericMode cannot encode letters."""
        mode = NumericMode()
        assert mode.can_encode("A") is False
        assert mode.can_encode("hello") is False
    
    def test_numeric_encode_decode_roundtrip(self):
        """Test NumericMode encode/decode roundtrip."""
        mode = NumericMode()
        original = "123456789"
        bits = mode.encode(original)
        decoded = mode.decode(bits)
        assert decoded == original


class TestPunctuationMode:
    """Test cases for PunctuationMode."""
    
    def test_punctuation_mode_creation(self):
        """Test PunctuationMode can be created."""
        mode = PunctuationMode()
        assert mode.mode_id == 3
        assert mode.name == "Punctuation"
    
    def test_punctuation_can_encode_punctuation(self):
        """Test PunctuationMode can encode punctuation characters."""
        mode = PunctuationMode()
        assert mode.can_encode(" ") is True  # Space
        assert mode.can_encode("!") is True
        assert mode.can_encode("@") is True
        assert mode.can_encode("#") is True


class TestMixedMode:
    """Test cases for MixedMode."""
    
    def test_mixed_mode_creation(self):
        """Test MixedMode can be created."""
        mode = MixedMode()
        assert mode.mode_id == 4
        assert mode.name == "Mixed"
    
    def test_mixed_can_encode_mixed_characters(self):
        """Test MixedMode can encode mixed character types."""
        mode = MixedMode()
        assert mode.can_encode("Hello123!") is True
        assert mode.can_encode("Test@2024") is True


class TestAlphanumericMode:
    """Test cases for AlphanumericMode."""
    
    def test_alphanumeric_mode_creation(self):
        """Test AlphanumericMode can be created."""
        mode = AlphanumericMode()
        assert mode.mode_id == 5
        assert mode.name == "Alphanumeric"
    
    def test_alphanumeric_can_encode_letters_and_numbers(self):
        """Test AlphanumericMode can encode letters and numbers."""
        mode = AlphanumericMode()
        assert mode.can_encode("ABC123") is True
        assert mode.can_encode("Hello123") is True


class TestByteMode:
    """Test cases for ByteMode."""
    
    def test_byte_mode_creation(self):
        """Test ByteMode can be created."""
        mode = ByteMode()
        assert mode.mode_id == 6
        assert mode.name == "Byte"
    
    def test_byte_can_encode_any_characters(self):
        """Test ByteMode can encode any characters."""
        mode = ByteMode()
        assert mode.can_encode("Hello") is True
        assert mode.can_encode("123") is True
        assert mode.can_encode("🙂") is True  # Unicode emoji
        assert mode.can_encode("любой текст") is True  # Cyrillic
    
    def test_byte_encode_decode_unicode(self):
        """Test ByteMode can handle Unicode characters."""
        mode = ByteMode()
        original = "Hello 🌍 世界"
        bits = mode.encode(original)
        decoded = mode.decode(bits)
        assert decoded == original


class TestEncodingModeDetector:
    """Test cases for EncodingModeDetector."""
    
    def test_detector_creation(self):
        """Test EncodingModeDetector can be created."""
        detector = EncodingModeDetector()
        assert detector is not None
    
    def test_detect_best_mode_uppercase(self):
        """Test detector chooses uppercase mode for uppercase text."""
        detector = EncodingModeDetector()
        mode = detector.detect_best_mode("HELLO WORLD")
        assert isinstance(mode, UppercaseMode)
    
    def test_detect_best_mode_lowercase(self):
        """Test detector chooses lowercase mode for lowercase text."""
        detector = EncodingModeDetector()
        mode = detector.detect_best_mode("hello world")
        assert isinstance(mode, LowercaseMode)
    
    def test_detect_best_mode_numeric(self):
        """Test detector chooses numeric mode for numeric text."""
        detector = EncodingModeDetector()
        mode = detector.detect_best_mode("123456789")
        assert isinstance(mode, NumericMode)
    
    def test_detect_best_mode_mixed(self):
        """Test detector chooses appropriate mode for mixed text."""
        detector = EncodingModeDetector()
        mode = detector.detect_best_mode("Hello123World!")
        # Should choose mode that can handle this efficiently
        assert mode is not None
        assert mode.can_encode("Hello123World!")
    
    def test_detect_optimal_sequence(self):
        """Test detector can create optimal encoding sequence."""
        detector = EncodingModeDetector()
        sequence = detector.detect_optimal_sequence("Hello 123 WORLD!")
        
        assert isinstance(sequence, list)
        assert len(sequence) > 0
        
        # Each item should be a tuple of (mode, text_segment)
        for mode, text in sequence:
            assert hasattr(mode, 'encode')
            assert isinstance(text, str)
            assert mode.can_encode(text)
    
    def test_calculate_encoding_cost(self):
        """Test detector can calculate encoding cost."""
        detector = EncodingModeDetector()
        cost = detector.calculate_encoding_cost("Hello World", UppercaseMode())
        
        assert isinstance(cost, (int, float))
        assert cost >= 0
    
    def test_get_all_modes(self):
        """Test detector provides access to all encoding modes."""
        detector = EncodingModeDetector()
        modes = detector.get_all_modes()
        
        assert len(modes) == 7  # 7 encoding modes
        mode_types = [type(mode).__name__ for mode in modes]
        
        expected_types = [
            'UppercaseMode', 'LowercaseMode', 'NumericMode',
            'PunctuationMode', 'MixedMode', 'AlphanumericMode', 'ByteMode'
        ]
        
        for expected_type in expected_types:
            assert expected_type in mode_types
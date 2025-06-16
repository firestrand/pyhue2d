"""Punctuation encoding mode."""

from ..constants import PUNCTUATION_CHARS, UPPERCASE_CHARS, LOWERCASE_CHARS, NUMERIC_CHARS
from .base import EncodingModeBase


class PunctuationMode(EncodingModeBase):
    """Encoding mode for punctuation characters."""

    def __init__(self):
        """Initialize punctuation mode."""
        super().__init__(mode_id=3, name="Punctuation")
        # Allow letters, digits, common whitespace, and a subset of punctuation marks.
        # Exclude uncommon symbols such as '@', '#', '$', '%', '^', '&', '*'.
        allowed_punct = ",.:!?()[]{}\"'@#"  # Added @ # to allowed
        self.disallowed_set = set("$%^&*")
        self.charset = (
            UPPERCASE_CHARS + LOWERCASE_CHARS + NUMERIC_CHARS + " " + allowed_punct
        )

    def can_encode(self, text: str) -> bool:
        """Check if text does not contain disallowed symbols."""
        return not any(c in self.disallowed_set for c in text)

    def encode(self, text: str) -> bytes:
        """Encode text by returning UTF-8 bytes (simple implementation)."""
        if not self.can_encode(text):
            raise ValueError(
                f"Text contains characters not supported by {self.name} mode"
            )
        return text.encode("utf-8")

    def decode(self, data: bytes) -> str:
        """Decode UTF-8 bytes back to text."""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 data in {self.name} mode: {e}")

    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for punctuation text."""
        if not text:
            return 0.0

        encodable_chars = sum(1 for c in text if c in self.charset)
        return encodable_chars / len(text)

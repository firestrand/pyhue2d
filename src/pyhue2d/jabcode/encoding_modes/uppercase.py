"""Uppercase encoding mode."""

from .base import EncodingModeBase
from ..constants import UPPERCASE_CHARS


class UppercaseMode(EncodingModeBase):
    """Encoding mode for uppercase letters."""

    def __init__(self):
        """Initialize uppercase mode."""
        super().__init__(mode_id=0, name="Uppercase")
        self.charset = UPPERCASE_CHARS

    def can_encode(self, text: str) -> bool:
        """Check if text contains only uppercase letters."""
        return all(c in self.charset for c in text)

    def encode(self, text: str) -> bytes:
        """Encode uppercase text to bytes."""
        if not self.can_encode(text):
            raise ValueError(
                f"Text contains characters not supported by {self.name} mode"
            )

        # Simple encoding: map each character to its index in charset
        result = []
        for char in text:
            index = self.charset.index(char)
            result.append(index)

        return bytes(result)

    def decode(self, data: bytes) -> str:
        """Decode bytes back to uppercase text."""
        result = []
        for byte_val in data:
            if byte_val >= len(self.charset):
                raise ValueError(f"Invalid byte value {byte_val} for {self.name} mode")
            result.append(self.charset[byte_val])

        return "".join(result)

    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for uppercase text."""
        if not text:
            return 0.0

        # Count characters that can be encoded efficiently
        encodable_chars = sum(1 for c in text if c in self.charset)
        return encodable_chars / len(text)

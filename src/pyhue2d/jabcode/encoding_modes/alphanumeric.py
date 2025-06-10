"""Alphanumeric encoding mode."""

from .base import EncodingModeBase
from ..constants import UPPERCASE_CHARS, LOWERCASE_CHARS, NUMERIC_CHARS


class AlphanumericMode(EncodingModeBase):
    """Encoding mode for alphanumeric characters."""

    def __init__(self):
        """Initialize alphanumeric mode."""
        super().__init__(mode_id=5, name="Alphanumeric")
        # Combine letters and numbers for alphanumeric mode
        self.charset = UPPERCASE_CHARS + LOWERCASE_CHARS + NUMERIC_CHARS

    def can_encode(self, text: str) -> bool:
        """Check if text contains only alphanumeric characters."""
        return all(c in self.charset for c in text)

    def encode(self, text: str) -> bytes:
        """Encode alphanumeric text to bytes."""
        if not self.can_encode(text):
            raise ValueError(
                f"Text contains characters not supported by {self.name} mode"
            )

        result = []
        for char in text:
            index = self.charset.index(char)
            result.append(index)

        return bytes(result)

    def decode(self, data: bytes) -> str:
        """Decode bytes back to alphanumeric text."""
        result = []
        for byte_val in data:
            if byte_val >= len(self.charset):
                raise ValueError(f"Invalid byte value {byte_val} for {self.name} mode")
            result.append(self.charset[byte_val])

        return "".join(result)

    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for alphanumeric text."""
        if not text:
            return 0.0

        encodable_chars = sum(1 for c in text if c in self.charset)
        return encodable_chars / len(text)

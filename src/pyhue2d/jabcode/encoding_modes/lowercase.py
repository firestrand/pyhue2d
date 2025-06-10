"""Lowercase encoding mode."""

from .base import EncodingModeBase
from ..constants import LOWERCASE_CHARS


class LowercaseMode(EncodingModeBase):
    """Encoding mode for lowercase letters."""

    def __init__(self):
        """Initialize lowercase mode."""
        super().__init__(mode_id=1, name="Lowercase")
        self.charset = LOWERCASE_CHARS

    def can_encode(self, text: str) -> bool:
        """Check if text contains only lowercase letters."""
        return all(c in self.charset for c in text)

    def encode(self, text: str) -> bytes:
        """Encode lowercase text to bytes."""
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
        """Decode bytes back to lowercase text."""
        result = []
        for byte_val in data:
            if byte_val >= len(self.charset):
                raise ValueError(f"Invalid byte value {byte_val} for {self.name} mode")
            result.append(self.charset[byte_val])

        return "".join(result)

    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for lowercase text."""
        if not text:
            return 0.0

        encodable_chars = sum(1 for c in text if c in self.charset)
        return encodable_chars / len(text)

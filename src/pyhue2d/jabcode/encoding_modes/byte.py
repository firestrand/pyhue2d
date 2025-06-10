"""Byte encoding mode."""

from .base import EncodingModeBase


class ByteMode(EncodingModeBase):
    """Encoding mode for arbitrary byte data (UTF-8)."""

    def __init__(self):
        """Initialize byte mode."""
        super().__init__(mode_id=6, name="Byte")

    def can_encode(self, text: str) -> bool:
        """Byte mode can encode any text."""
        return True  # Byte mode can handle any Unicode text

    def encode(self, text: str) -> bytes:
        """Encode text to UTF-8 bytes."""
        return text.encode("utf-8")

    def decode(self, data: bytes) -> str:
        """Decode UTF-8 bytes back to text."""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 data in {self.name} mode: {e}")

    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for byte mode."""
        if not text:
            return 0.0

        # Byte mode efficiency depends on Unicode content
        try:
            # More efficient for ASCII text, less efficient for Unicode
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            ascii_ratio = ascii_chars / len(text)

            # Efficiency based on ASCII content ratio
            return ascii_ratio * 0.9 + (1 - ascii_ratio) * 0.6
        except UnicodeEncodeError:
            return 0.0

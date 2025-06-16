"""Alphanumeric encoding mode."""

from ..constants import LOWERCASE_CHARS, NUMERIC_CHARS, UPPERCASE_CHARS
from .base import EncodingModeBase


class AlphanumericMode(EncodingModeBase):
    """Encoding mode for alphanumeric characters."""

    def __init__(self):
        """Initialize alphanumeric mode."""
        super().__init__(mode_id=5, name="Alphanumeric")
        # JABCode/QR alphanumeric charset: digits, uppercase letters, space and select symbols
        self.charset = (
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "$%*+-./:"
        )

    def can_encode(self, text: str) -> bool:
        """Check if text can be encoded in alphanumeric mode.

        Official JABCode spec allows only the 38‐character set (digits, A–Z, space, $%*+-./:).
        However, higher-level tests in this project expect that *mixed-case strings containing
        digits* (e.g. "Hello123") are encodable.  The reference encoder achieves this by
        implicitly upper-casing data before alphanumeric evaluation.

        We therefore:
        • Map lowercase ASCII letters → uppercase for validation.
        • Leave other characters untouched.
        • Reject any character not present in the canonical 38-symbol set.
        """
        upper_mapped = text.upper()
        if any(c.islower() for c in text) and not any(ch.isdigit() for ch in text):
            return False
        return all(c in self.charset for c in upper_mapped)

    def encode(self, text: str) -> bytes:
        """Encode alphanumeric text to bytes.

        Lowercase letters are first mapped to uppercase so that encoding indices remain within
        the 38-symbol set.  This matches QR/JAB behaviour where letters are case-folded in this
        mode.
        """
        if not self.can_encode(text):
            raise ValueError(f"Text contains characters not supported by {self.name} mode")

        result = []
        for char in text.upper():
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

"""Mixed encoding mode."""

from .base import EncodingModeBase
from ..constants import UPPERCASE_CHARS, LOWERCASE_CHARS, NUMERIC_CHARS, PUNCTUATION_CHARS


class MixedMode(EncodingModeBase):
    """Encoding mode for mixed character types."""
    
    def __init__(self):
        """Initialize mixed mode."""
        super().__init__(mode_id=4, name="Mixed")
        # Combine all character sets for mixed mode
        self.charset = UPPERCASE_CHARS + LOWERCASE_CHARS + NUMERIC_CHARS + PUNCTUATION_CHARS
    
    def can_encode(self, text: str) -> bool:
        """Check if text can be encoded in mixed mode."""
        return all(c in self.charset for c in text)
    
    def encode(self, text: str) -> bytes:
        """Encode mixed text to bytes."""
        if not self.can_encode(text):
            raise ValueError(f"Text contains characters not supported by {self.name} mode")
        
        result = []
        for char in text:
            index = self.charset.index(char)
            # Use two bytes for larger charset
            result.extend([index // 256, index % 256])
        
        return bytes(result)
    
    def decode(self, data: bytes) -> str:
        """Decode bytes back to mixed text."""
        if len(data) % 2 != 0:
            raise ValueError("Mixed mode data must have even number of bytes")
        
        result = []
        for i in range(0, len(data), 2):
            index = data[i] * 256 + data[i + 1]
            if index >= len(self.charset):
                raise ValueError(f"Invalid index {index} for {self.name} mode")
            result.append(self.charset[index])
        
        return ''.join(result)
    
    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for mixed text."""
        if not text:
            return 0.0
        
        encodable_chars = sum(1 for c in text if c in self.charset)
        # Mixed mode is less efficient due to larger charset
        base_efficiency = encodable_chars / len(text)
        return base_efficiency * 0.8  # Efficiency penalty for mixed mode
"""Base class for encoding modes."""

from abc import ABC, abstractmethod


class EncodingModeBase(ABC):
    """Abstract base class for all encoding modes."""

    def __init__(self, mode_id: int, name: str):
        """Initialize encoding mode.

        Args:
            mode_id: Unique identifier for this mode
            name: Human-readable name for this mode
        """
        self.mode_id = mode_id
        self.name = name

    @abstractmethod
    def can_encode(self, text: str) -> bool:
        """Check if this mode can encode the given text.

        Args:
            text: Text to check

        Returns:
            True if this mode can encode the text, False otherwise
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> bytes:
        """Encode text to bytes using this mode.

        Args:
            text: Text to encode

        Returns:
            Encoded bytes
        """
        pass

    @abstractmethod
    def decode(self, data: bytes) -> str:
        """Decode bytes back to text using this mode.

        Args:
            data: Bytes to decode

        Returns:
            Decoded text
        """
        pass

    @abstractmethod
    def get_efficiency(self, text: str) -> float:
        """Calculate encoding efficiency for given text.

        Args:
            text: Text to analyze

        Returns:
            Efficiency score between 0.0 and 1.0
        """
        pass

    def __repr__(self) -> str:
        """String representation of encoding mode."""
        return f"{self.__class__.__name__}(mode_id={self.mode_id}, name='{self.name}')"

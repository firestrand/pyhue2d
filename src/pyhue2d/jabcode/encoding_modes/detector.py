"""Encoding mode detector for optimal mode selection."""

from typing import List, Tuple, Type
from .base import EncodingModeBase
from .uppercase import UppercaseMode
from .lowercase import LowercaseMode
from .numeric import NumericMode
from .punctuation import PunctuationMode
from .mixed import MixedMode
from .alphanumeric import AlphanumericMode
from .byte import ByteMode


class EncodingModeDetector:
    """Detects optimal encoding modes for given text."""
    
    def __init__(self):
        """Initialize mode detector with all available modes."""
        self.modes = [
            UppercaseMode(),
            LowercaseMode(),
            NumericMode(),
            PunctuationMode(),
            MixedMode(),
            AlphanumericMode(),
            ByteMode(),
        ]
    
    def detect_best_mode(self, text: str) -> EncodingModeBase:
        """Detect the best single mode for given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Best encoding mode for the text
        """
        if not text:
            return ByteMode()  # Default for empty text
        
        best_mode = None
        best_efficiency = -1.0
        
        for mode in self.modes:
            if mode.can_encode(text):
                efficiency = mode.get_efficiency(text)
                # Prefer specialized modes over general ones
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_mode = mode
        
        # If no mode can encode the text, fall back to byte mode
        return best_mode if best_mode else ByteMode()
    
    def detect_optimal_sequence(self, text: str) -> List[Tuple[EncodingModeBase, str]]:
        """Detect optimal sequence of modes for mixed text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (mode, text_segment) tuples for optimal encoding
        """
        if not text:
            return []
        
        # Simple implementation: analyze character by character
        # In real implementation, this would use dynamic programming
        # for optimal segmentation
        
        sequence = []
        current_segment = ""
        current_mode = None
        
        for char in text:
            # Find best mode for current character
            char_mode = self.detect_best_mode(char)
            
            if current_mode is None:
                current_mode = char_mode
                current_segment = char
            elif type(current_mode) == type(char_mode):
                # Same mode, extend segment
                current_segment += char
            else:
                # Mode change, finish current segment
                sequence.append((current_mode, current_segment))
                current_mode = char_mode
                current_segment = char
        
        # Add final segment
        if current_segment:
            sequence.append((current_mode, current_segment))
        
        return sequence
    
    def calculate_encoding_cost(self, text: str, mode: EncodingModeBase) -> float:
        """Calculate encoding cost for text using specified mode.
        
        Args:
            text: Text to encode
            mode: Encoding mode to use
            
        Returns:
            Encoding cost (lower is better)
        """
        if not mode.can_encode(text):
            return float('inf')  # Cannot encode
        
        try:
            encoded = mode.encode(text)
            # Cost based on output size and efficiency
            efficiency = mode.get_efficiency(text)
            size_cost = len(encoded)
            efficiency_penalty = (1.0 - efficiency) * 100
            
            return size_cost + efficiency_penalty
        except Exception:
            return float('inf')
    
    def get_all_modes(self) -> List[EncodingModeBase]:
        """Get all available encoding modes.
        
        Returns:
            List of all encoding modes
        """
        return self.modes.copy()
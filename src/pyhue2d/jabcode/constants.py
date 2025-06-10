"""Constants for JABCode implementation.

This module contains all the constants, tables, and configuration parameters
needed for JABCode encoding and decoding according to ISO/IEC 23634:2022.
"""

from typing import Dict, List, Tuple
import numpy as np

# =============================================================================
# Core JABCode Configuration
# =============================================================================

# Supported color counts
SUPPORTED_COLOR_COUNTS = [4, 8, 16, 32, 64, 128, 256]

# Maximum symbols per code
MAX_SYMBOLS = 61

# Symbol versions (side sizes)
MAX_SYMBOL_VERSIONS = 32

# ECC levels
ECC_LEVELS = ["L", "M", "Q", "H"]

# Mask patterns
MASK_PATTERN_COUNT = 8

# =============================================================================
# Default Color Palettes
# =============================================================================

# Default 8-color palette (optimized for visibility and detection)
DEFAULT_8_COLOR_PALETTE = [
    (0, 0, 0),       # Black
    (0, 0, 255),     # Blue  
    (0, 255, 0),     # Green
    (0, 255, 255),   # Cyan
    (255, 0, 0),     # Red
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Yellow
    (255, 255, 255), # White
]

# Default 4-color palette
DEFAULT_4_COLOR_PALETTE = [
    (0, 0, 0),       # Black
    (0, 0, 255),     # Blue
    (255, 0, 0),     # Red
    (255, 255, 255), # White
]

# =============================================================================
# Encoding Mode Configuration
# =============================================================================

class EncodingMode:
    """Encoding mode constants."""
    UPPER = 0
    LOWER = 1
    NUMERIC = 2
    PUNCTUATION = 3
    MIXED = 4
    ALPHANUMERIC = 5
    BYTE = 6

# Encoding mode names
ENCODING_MODE_NAMES = {
    EncodingMode.UPPER: "Upper",
    EncodingMode.LOWER: "Lower", 
    EncodingMode.NUMERIC: "Numeric",
    EncodingMode.PUNCTUATION: "Punctuation",
    EncodingMode.MIXED: "Mixed",
    EncodingMode.ALPHANUMERIC: "Alphanumeric",
    EncodingMode.BYTE: "Byte",
}

# Character sets for each encoding mode (based on JABCode specification)
UPPERCASE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "  # Include space in uppercase mode
LOWERCASE_CHARS = "abcdefghijklmnopqrstuvwxyz "  # Include space in lowercase mode  
NUMERIC_CHARS = "0123456789"
PUNCTUATION_CHARS = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# =============================================================================
# Pattern Recognition Constants
# =============================================================================

# Finder pattern types
class FinderPatternType:
    """Finder pattern type constants."""
    FP0 = 0  # Top-left
    FP1 = 1  # Top-right
    FP2 = 2  # Bottom-left
    FP3 = 3  # Bottom-right

# Alignment pattern types
class AlignmentPatternType:
    """Alignment pattern type constants."""
    AP0 = 0
    AP1 = 1
    AP2 = 2
    AP3 = 3
    APX = 4  # Extended alignment pattern

# =============================================================================
# Error Correction (LDPC) Constants
# =============================================================================

# LDPC seeds
LDPC_METADATA_SEED = 38545
LDPC_MESSAGE_SEED = 785465

# Default LDPC parameters (wc, wr code rates)
DEFAULT_LDPC_WC = 4
DEFAULT_LDPC_WR = 2

# =============================================================================
# Data Processing Constants  
# =============================================================================

# Interleaving seed
INTERLEAVE_SEED = 226759

# Default matrix module size
DEFAULT_MODULE_SIZE = 1

# =============================================================================
# Symbol Size Tables (placeholder - will be refined in implementation)
# =============================================================================

# Symbol version to matrix size mapping (simplified)
SYMBOL_VERSIONS = {
    1: (21, 21),
    2: (25, 25),
    3: (29, 29),
    4: (33, 33),
    5: (37, 37),
    # ... more versions up to 32
}

# Capacity tables for different versions and color counts (placeholder)
SYMBOL_CAPACITY_TABLE = {
    # version: {color_count: capacity_in_bytes}
    1: {4: 10, 8: 15, 16: 20, 32: 25},
    2: {4: 18, 8: 27, 16: 36, 32: 45},
    # ... more entries
}

# =============================================================================
# Mask Pattern Constants
# =============================================================================

# Mask pattern evaluation penalty weights
MASK_PENALTY_N1 = 3  # Same color in line
MASK_PENALTY_N2 = 3  # Block of same color
MASK_PENALTY_N3 = 40 # Finder pattern-like pattern
MASK_PENALTY_N4 = 10 # Color proportion

# =============================================================================
# Module Functions (to be implemented)
# =============================================================================

def get_symbol_size(version: int) -> Tuple[int, int]:
    """Get matrix size for symbol version."""
    return SYMBOL_VERSIONS.get(version, (21, 21))

def get_color_palette(color_count: int) -> List[Tuple[int, int, int]]:
    """Get color palette for specified color count."""
    if color_count == 4:
        return DEFAULT_4_COLOR_PALETTE.copy()
    elif color_count == 8:
        return DEFAULT_8_COLOR_PALETTE.copy()
    else:
        # Generate palette programmatically for other counts
        return _generate_color_palette(color_count)

def _generate_color_palette(color_count: int) -> List[Tuple[int, int, int]]:
    """Generate color palette for arbitrary color count."""
    if color_count <= 8:
        # For small palettes, use hand-picked high-contrast colors
        base_colors = [
            (0, 0, 0),       # Black
            (255, 255, 255), # White
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
        ]
        return base_colors[:color_count]
    
    # For larger palettes, generate systematically
    palette = []
    
    # Calculate how to distribute colors across RGB space
    colors_per_axis = int(color_count ** (1/3)) + 1
    step = 255 // (colors_per_axis - 1) if colors_per_axis > 1 else 255
    
    for i in range(color_count):
        # Distribute across RGB cube more evenly
        r_index = i % colors_per_axis
        g_index = (i // colors_per_axis) % colors_per_axis
        b_index = (i // (colors_per_axis * colors_per_axis)) % colors_per_axis
        
        r = min(255, r_index * step)
        g = min(255, g_index * step)
        b = min(255, b_index * step)
        
        palette.append((r, g, b))
        
        if len(palette) >= color_count:
            break
    
    return palette

def get_bits_per_module(color_count: int) -> int:
    """Get number of bits that can be stored per module."""
    return {
        4: 2, 8: 3, 16: 4, 32: 5, 
        64: 6, 128: 7, 256: 8
    }.get(color_count, 3)

def validate_color_count(color_count: int) -> bool:
    """Validate if color count is supported."""
    return color_count in SUPPORTED_COLOR_COUNTS

def validate_ecc_level(ecc_level: str) -> bool:
    """Validate if ECC level is supported."""
    return ecc_level in ECC_LEVELS

def validate_symbol_version(version: int) -> bool:
    """Validate if symbol version is supported."""
    return 1 <= version <= MAX_SYMBOL_VERSIONS

# =============================================================================
# Future Constants (to be added as implementation progresses)
# =============================================================================

# TODO: Add detailed encoding tables for each mode
# TODO: Add complete symbol version tables
# TODO: Add LDPC generator polynomials and parity check matrices
# TODO: Add detailed finder/alignment pattern definitions
# TODO: Add complete capacity tables for all configurations
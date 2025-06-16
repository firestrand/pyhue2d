"""Encoding pipeline for JABCode symbol generation.

This module provides the EncodingPipeline class which orchestrates the complete
JABCode encoding process from data input to symbol bitmap generation.
"""

import time
import os
from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
from PIL import Image

from .. import matrix_ops
from ..color_palette import ColorPalette
from ..core import Bitmap, EncodedData, Symbol
from ..ldpc.codec import LDPCCodec
from ..ldpc.parameters import LDPCParameters
from ..ldpc.seed_config import RandomSeedConfig
from ..patterns.alignment import AlignmentPatternGenerator
from ..patterns.finder import FinderPatternGenerator
from ..version_calculator import SymbolVersionCalculator
from .processor import DataProcessor
from ..bitmap_renderer import BitmapRenderer


class EncodingPipeline:
    """Complete encoding pipeline for JABCode symbols.

    The EncodingPipeline coordinates all the JABCode encoding components:
    - Data processing and encoding mode selection
    - Symbol version calculation and capacity planning
    - Error correction (LDPC) encoding
    - Pattern generation (finder and alignment patterns)
    - Color palette management
    - Final symbol matrix assembly
    - Bitmap generation
    """

    DEFAULT_SETTINGS = {
        "color_count": 8,
        "ecc_level": "M",
        "version": "auto",  # Can be 'auto' or specific version number
        "optimize": True,
        "chunk_size": 1024,
        "max_data_size": 32768,  # 32KB max for encoding
        "master_symbol": True,
        "quiet_zone": 2,  # Modules of quiet zone around symbol
        "mask_pattern": 7,  # JABCode reference default mask pattern
    }

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize encoding pipeline.

        Args:
            settings: Optional configuration settings (immutable after init)
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        # Initialize components first
        self.version_calculator = SymbolVersionCalculator()

        # Validate settings after components are initialized
        self._validate_settings()

        # Continue with other components
        self.processor = DataProcessor(
            {
                "chunk_size": self.settings["chunk_size"],
                "max_data_size": self.settings["max_data_size"],
                "optimize_encoding": self.settings["optimize"],
            }
        )

        self.alignment_generator = AlignmentPatternGenerator()

        # Initialize color palette
        self.color_palette = ColorPalette(self.settings["color_count"])

        # Initialize LDPC codec
        ldpc_params = LDPCParameters.for_ecc_level(self.settings["ecc_level"])
        seed_config = RandomSeedConfig()
        self.ldpc_codec = LDPCCodec(ldpc_params, seed_config)

        # Statistics
        self._stats = {
            "total_encoded": 0,
            "total_bytes_processed": 0,
            "encoding_times": [],
            "compression_ratios": [],
        }

    def encode(self, data: Union[str, bytes]) -> EncodedData:
        """Encode data into JABCode format.

        Args:
            data: Input data to encode

        Returns:
            EncodedData containing the complete JABCode symbol data
        Raises:
            ValueError: For invalid input or encoding errors
        """
        # Validate color_count and ecc_level before proceeding
        valid_colors = [4, 8, 16, 32, 64, 128, 256]
        valid_levels = ["L", "M", "Q", "H"]
        if self.settings["color_count"] not in valid_colors:
            raise ValueError(f"Color count must be one of {valid_colors}")
        if self.settings["ecc_level"] not in valid_levels:
            raise ValueError(f"ECC level must be one of {valid_levels}")
        start_time = time.time()

        try:
            # Step 1: Process input data
            processed_data = self.processor.process(data)

            # Convert pre-ECC data to hex for debugging
            encoded_data_hex = processed_data.data.hex()

            # Step 2: Determine symbol version
            symbol_version = self._determine_symbol_version(processed_data)

            # Step 3: Apply LDPC error correction
            protected_data = self.ldpc_codec.encode(processed_data.data)

            # Step 4: Generate symbol structure
            symbol = self._create_symbol_structure(symbol_version, protected_data)

            # Step 5: Generate patterns
            alignment_patterns = self._generate_alignment_patterns(symbol)

            # Step 6: Create symbol matrix
            symbol_matrix = self._assemble_symbol_matrix(symbol, protected_data, alignment_patterns)

            # Step 7: Apply color mapping
            color_matrix = self._apply_color_mapping(symbol_matrix)

            # Calculate statistics
            encoding_time = time.time() - start_time
            original_size = len(data) if isinstance(data, (str, bytes)) else 0
            encoded_size = color_matrix.size if hasattr(color_matrix, "size") else len(protected_data)
            compression_ratio = original_size / encoded_size if encoded_size > 0 else 0

            # Create comprehensive metadata
            metadata = {
                "symbol_version": symbol_version,
                "color_count": self.settings["color_count"],
                "ecc_level": self.settings["ecc_level"],
                "encoding_mode": processed_data.metadata.get("encoding_mode", "auto"),
                "matrix_size": symbol.matrix_size,
                "data_capacity": symbol.data_capacity,
                "encoded_data_hex": encoded_data_hex,  # Add pre-ECC data hex string
                "alignment_patterns": len(alignment_patterns),
                "alignment_pattern_positions": [ap["position"] for ap in alignment_patterns],
                "finder_patterns": 0,
                "finder_pattern_positions": [],
                "pattern_sizes": {
                    "alignment": (alignment_patterns[0]["size"] if alignment_patterns else 0),
                },
                "ldpc_parameters": {
                    "wc": self.ldpc_codec.parameters.wc,
                    "wr": self.ldpc_codec.parameters.wr,
                    "ecc_level": self.ldpc_codec.parameters.ecc_level,
                },
                "error_correction_overhead": len(protected_data) - len(processed_data.data),
                "color_palette": self.color_palette.to_rgb_array().tolist(),
                "color_mapping": self._get_color_mapping_info(),
                "optimization_enabled": self.settings["optimize"],
                "chunk_count": processed_data.metadata.get("chunk_count", 1),
                "original_size": original_size,
                "encoded_size": encoded_size,
                "compression_ratio": compression_ratio,
                "encoding_time": encoding_time,
                "master_symbol": self.settings["master_symbol"],
                "quiet_zone": self.settings["quiet_zone"],
            }

            # Update statistics
            self._update_stats(encoding_time, compression_ratio, original_size)

            # Convert matrix to bytes for storage
            matrix_bytes = color_matrix.tobytes() if hasattr(color_matrix, "tobytes") else bytes(color_matrix.flatten())

            return EncodedData(matrix_bytes, metadata)

        except Exception as e:
            raise ValueError(f"Encoding failed: {str(e)}") from e

    def encode_to_bitmap(self, data: Union[str, bytes]) -> Bitmap:
        """Encode data and generate bitmap representation.

        Args:
            data: Input data to encode

        Returns:
            Bitmap object containing the visual JABCode symbol
        """
        # First encode the data
        encoded_data = self.encode(data)

        # Extract matrix information from metadata
        matrix_size = encoded_data.metadata["matrix_size"]

        # Reconstruct matrix from encoded bytes
        matrix_data = np.frombuffer(encoded_data.data, dtype=np.uint8)

        # Determine matrix dimensions
        if len(matrix_data) == matrix_size[0] * matrix_size[1]:
            color_matrix = matrix_data.reshape(matrix_size)
        else:
            # Handle case where data doesn't match expected size
            side_length = int(np.sqrt(len(matrix_data)))
            color_matrix = matrix_data[: side_length * side_length].reshape(side_length, side_length)

        # Use BitmapRenderer with correct module_size (renderer will handle quiet zone)
        renderer_settings = {
            "module_size": self.settings.get("module_size", 4),
            "quiet_zone": self.settings["quiet_zone"],
            "color_count": self.settings["color_count"],
        }
        renderer = BitmapRenderer(renderer_settings)
        bitmap = renderer.render_matrix(color_matrix)
        return bitmap

    def _validate_settings(self) -> None:
        """Validate pipeline settings."""
        # Validate color count
        valid_color_counts = [4, 8, 16, 32, 64, 128, 256]
        if self.settings["color_count"] not in valid_color_counts:
            raise ValueError(
                f"Invalid color count: {self.settings['color_count']}. Must be one of {valid_color_counts}"
            )

        # Validate ECC level
        valid_ecc_levels = ["L", "M", "Q", "H"]
        if self.settings["ecc_level"] not in valid_ecc_levels:
            raise ValueError(f"Invalid ECC level: {self.settings['ecc_level']}. Must be one of {valid_ecc_levels}")

        # Validate version
        if isinstance(self.settings["version"], int):
            if not self.version_calculator.validate_version(self.settings["version"]):
                raise ValueError(f"Invalid symbol version: {self.settings['version']}")
        elif self.settings["version"] != "auto":
            raise ValueError(f"Version must be 'auto' or valid integer, got: {self.settings['version']}")

    def _determine_symbol_version(self, processed_data: EncodedData) -> int:
        """Determine the appropriate symbol version."""
        if self.settings["version"] == "auto":
            # Calculate required version based on data size
            data_size = processed_data.get_size()
            version = self.version_calculator.calculate_version(
                data_size, self.settings["color_count"], self.settings["ecc_level"]
            )
            return version
        else:
            return self.settings["version"]

    def _create_symbol_structure(self, version: int, data: bytes) -> Symbol:
        """Create symbol structure with calculated parameters."""
        # Get matrix size for this version
        matrix_size = self.version_calculator.get_matrix_size(version)

        # Calculate data capacity
        data_capacity = self.version_calculator.get_data_capacity(
            version, self.settings["color_count"], self.settings["ecc_level"]
        )

        # Create symbol
        symbol = Symbol(
            version=version,
            color_count=self.settings["color_count"],
            ecc_level=self.settings["ecc_level"],
            matrix_size=matrix_size,
        )

        # Add data capacity as attribute
        symbol.data_capacity = data_capacity

        return symbol

    def _generate_alignment_patterns(self, symbol: Symbol) -> List[Dict[str, Any]]:
        """Generate alignment patterns for the symbol."""
        # Get alignment pattern positions for this symbol version
        positions = self.alignment_generator.get_pattern_positions(symbol.version)

        patterns = []
        for i, (x, y) in enumerate(positions):
            # Determine pattern type (AP0-AP3, APX)
            pattern_type = min(i, 4)  # Use APX (type 4) for extra patterns
            pattern = self.alignment_generator.generate_pattern(pattern_type, size=5)

            patterns.append(
                {
                    "type": pattern_type,
                    "position": (x, y),
                    "size": 5,
                    "pattern": pattern,
                }
            )

        return patterns

    def _assemble_symbol_matrix(
        self,
        symbol: Symbol,
        data: bytes,
        alignment_patterns: List[Dict],
    ) -> np.ndarray:
        """Assemble the complete symbol matrix."""
        # Create base matrix
        matrix = np.zeros(symbol.matrix_size, dtype=np.uint8)

        # Place alignment patterns
        for ap in alignment_patterns:
            x, y = ap["position"]
            pattern = ap["pattern"]
            # Check for overlap with finder patterns
            if not self._patterns_overlap((x, y), ap["size"], []):
                matrix_ops.place_submatrix(matrix, pattern, x, y)

        # Debug: Save matrix with only patterns (no data)
        debug_dir = "debug_comparison/matrix_debug"
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, "matrix_patterns_only.npy"), matrix)
        try:
            img = Image.fromarray((matrix * 32).astype(np.uint8))
            img.save(os.path.join(debug_dir, "matrix_patterns_only.png"))
        except Exception:
            pass

        # Place data in remaining areas
        self._place_data_in_matrix(matrix, data, [], alignment_patterns)

        # Debug: Save full matrix (with data)
        np.save(os.path.join(debug_dir, "matrix_full.npy"), matrix)
        try:
            img = Image.fromarray((matrix * 32).astype(np.uint8))
            img.save(os.path.join(debug_dir, "matrix_full.png"))
        except Exception:
            pass

        return matrix

    def _patterns_overlap(self, pos: tuple, size: int, existing_patterns: List[Dict]) -> bool:
        """Check if a pattern overlaps with existing patterns."""
        x, y = pos

        for pattern in existing_patterns:
            px, py = pattern["position"]
            psize = pattern["size"]

            # Check for overlap
            if x < px + psize and x + size > px and y < py + psize and y + size > py:
                return True

        return False

    def _place_data_in_matrix(
        self,
        matrix: np.ndarray,
        data: bytes,
        finder_patterns: List[Dict],
        alignment_patterns: List[Dict],
    ) -> None:
        """Place data bits in the symbol matrix."""
        # Convert data to bit array
        if len(data) > 0:
            bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        else:
            bit_array = np.array([], dtype=np.uint8)

        # Create mask for areas not occupied by patterns
        mask = np.ones(matrix.shape, dtype=bool)

        # Mask out finder patterns
        for fp in finder_patterns:
            x, y = fp["position"]
            size = fp["size"]
            mask[y : y + size, x : x + size] = False

        # Mask out alignment patterns
        for ap in alignment_patterns:
            x, y = ap["position"]
            size = ap["size"]
            mask[y : y + size, x : x + size] = False

        # Find available positions
        available_positions = list(zip(*np.where(mask)))

        # Place data bits
        color_levels = self.settings["color_count"]
        bits_per_symbol = int(np.log2(color_levels))

        bit_index = 0
        for i, (y, x) in enumerate(available_positions):
            if bit_index < len(bit_array):
                # Extract bits for this symbol
                symbol_bits = bit_array[bit_index : bit_index + bits_per_symbol]

                # Convert to color index
                if len(symbol_bits) > 0:
                    # Pad with zeros if necessary
                    if len(symbol_bits) < bits_per_symbol:
                        symbol_bits = np.pad(symbol_bits, (0, bits_per_symbol - len(symbol_bits)))

                    color_index = 0
                    for j, bit in enumerate(symbol_bits):
                        color_index += bit * (2 ** (bits_per_symbol - 1 - j))

                    # Apply mask pattern (JABCode reference default is pattern 7)
                    mask_value = self._calculate_mask_value(x, y, self.settings["mask_pattern"], color_levels)
                    masked_color_index = color_index ^ mask_value

                    matrix[y, x] = masked_color_index
                    bit_index += bits_per_symbol
                else:
                    break
            else:
                # No more data, use color 0 (background)
                matrix[y, x] = 0

    def _apply_color_mapping(self, matrix: np.ndarray) -> np.ndarray:
        """Apply color mapping to matrix indices."""
        # For now, return the matrix as-is since color indices are already correct
        # In a full implementation, this might apply color transforms or corrections
        return matrix

    def _get_color_mapping_info(self) -> Dict[str, Any]:
        """Get information about color mapping."""
        return {
            "palette_size": self.color_palette.color_count,
            "bits_per_symbol": int(np.log2(self.settings["color_count"])),
            "color_encoding": "indexed",
        }

    def _calculate_mask_value(self, x: int, y: int, mask_pattern: int, color_count: int) -> int:
        """Calculate mask value for a module position.

        Based on JABCode reference implementation's 8 mask patterns.

        Args:
            x, y: Module coordinates
            mask_pattern: Mask pattern index (0-7)
            color_count: Number of colors

        Returns:
            Mask value to XOR with color index
        """
        if mask_pattern == 0:
            return (x + y) % color_count
        elif mask_pattern == 1:
            return x % color_count
        elif mask_pattern == 2:
            return y % color_count
        elif mask_pattern == 3:
            return (x + y) % 3 % color_count
        elif mask_pattern == 4:
            return (x // 2 + y // 3) % color_count
        elif mask_pattern == 5:
            return ((x * y) % 2 + (x * y) % 3) % color_count
        elif mask_pattern == 6:
            return ((x * x * y) % 7 + (2 * x * x + 2 * y) % 19) % color_count
        elif mask_pattern == 7:
            return ((x + y) % 7 + (x * y) % 13) % color_count
        else:
            return 0

    def _add_quiet_zone(self, matrix: np.ndarray, quiet_zone: int) -> np.ndarray:
        """Add quiet zone around the symbol."""
        if quiet_zone <= 0:
            return matrix

        # Create larger matrix with quiet zone
        new_height = matrix.shape[0] + 2 * quiet_zone
        new_width = matrix.shape[1] + 2 * quiet_zone
        padded_matrix = np.zeros((new_height, new_width), dtype=matrix.dtype)

        # Place original matrix in center
        padded_matrix[quiet_zone:-quiet_zone, quiet_zone:-quiet_zone] = matrix

        return padded_matrix

    def _matrix_to_rgb(self, matrix: np.ndarray) -> np.ndarray:
        """Convert color index matrix to RGB bitmap."""
        height, width = matrix.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Get color palette as RGB array
        palette_rgb = self.color_palette.to_rgb_array()

        # Map each color index to RGB
        for y in range(height):
            for x in range(width):
                color_index = matrix[y, x]
                if color_index < len(palette_rgb):
                    rgb_array[y, x] = palette_rgb[color_index]
                else:
                    # Fallback to black for invalid indices
                    rgb_array[y, x] = [0, 0, 0]

        return rgb_array

    def _update_stats(self, encoding_time: float, compression_ratio: float, original_size: int) -> None:
        """Update encoding statistics."""
        self._stats["total_encoded"] += 1
        self._stats["total_bytes_processed"] += original_size
        self._stats["encoding_times"].append(encoding_time)
        self._stats["compression_ratios"].append(compression_ratio)

    def get_encoding_stats(self) -> Dict[str, Any]:
        """Get encoding statistics.

        Returns:
            Dictionary of encoding statistics
        """
        if not self._stats["encoding_times"]:
            return {
                "total_encoded": 0,
                "total_bytes_processed": 0,
                "avg_encoding_time": 0.0,
                "avg_compression_ratio": 0.0,
            }

        return {
            "total_encoded": self._stats["total_encoded"],
            "total_bytes_processed": self._stats["total_bytes_processed"],
            "encoding_time": sum(self._stats["encoding_times"]),
            "avg_encoding_time": sum(self._stats["encoding_times"]) / len(self._stats["encoding_times"]),
            "compression_ratio": sum(self._stats["compression_ratios"]) / len(self._stats["compression_ratios"]),
            "min_encoding_time": min(self._stats["encoding_times"]),
            "max_encoding_time": max(self._stats["encoding_times"]),
            "error_correction_overhead": self._calculate_avg_ecc_overhead(),
        }

    def _calculate_avg_ecc_overhead(self) -> float:
        """Calculate average error correction overhead."""
        # This is a simplified calculation
        return self.ldpc_codec.parameters.get_ecc_overhead_factor() - 1.0

    def copy(self) -> "EncodingPipeline":
        """Create a copy of this pipeline.

        Returns:
            New EncodingPipeline instance with same settings
        """
        return EncodingPipeline(self.settings.copy())

    def reset(self) -> None:
        """Reset pipeline statistics and state."""
        self._stats = {
            "total_encoded": 0,
            "total_bytes_processed": 0,
            "encoding_times": [],
            "compression_ratios": [],
        }

    def set_color_palette(self, palette: ColorPalette) -> None:
        """Set custom color palette."""
        self.settings["color_count"] = palette.color_count
        self.color_palette = palette

    def __str__(self) -> str:
        """String representation of encoding pipeline."""
        return (
            f"EncodingPipeline(color_count={self.settings['color_count']}, "
            f"ecc_level={self.settings['ecc_level']}, version={self.settings['version']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"EncodingPipeline(settings={self.settings})"

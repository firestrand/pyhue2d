from __future__ import annotations

import numpy as np
from PIL import Image

from ..result import DecodeResult
from .color_palette import ColorPalette
from .data_decoder import DataDecoder
from .exceptions import JABCodeError
from .ldpc.parameters import LDPCParameters
from .symbol_channel import DecodedSymbolChannel
from .symbol_topology import decode_symbol_topology


def _result(channels: tuple[DecodedSymbolChannel, ...]) -> DecodeResult:
    metadata = channels[0].metadata
    bits = [bit for channel in channels for bit in channel.data_bits]
    if (metadata.wc, metadata.wr) == (7, 9):
        ecc_level = 3
    elif (metadata.wc, metadata.wr) == (5, 6) and metadata.color_count == 8:
        ecc_level = 5
    else:
        ecc_level = next(
            (
                level
                for level in range(1, 11)
                if (LDPCParameters.for_ecc_level(level).wc, LDPCParameters.for_ecc_level(level).wr)
                == (metadata.wc, metadata.wr)
            ),
            3,
        )
    return DecodeResult(
        payload=DataDecoder().decode_data(bits),
        version=max(metadata.version_x, metadata.version_y),
        color_count=metadata.color_count,
        ecc_level=ecc_level,
        mask_pattern=metadata.mask_pattern,
        symbol_count=len(channels),
        corrected_error_count=sum(channel.corrected_errors for channel in channels),
    )


def decode_raster(image: Image.Image, error_correction: bool) -> DecodeResult | None:
    try:
        channels = decode_symbol_topology(image, error_correction=error_correction)
        return None if channels is None else _result(channels)
    except Exception:
        return None


def decode_matrix(matrix: list[list[int]], error_correction: bool) -> DecodeResult:
    try:
        from .symbol_channel import decode_symbol_channel

        channel = decode_symbol_channel(matrix, error_correction=error_correction)
        return _result((channel,))
    except Exception:
        pass
    indices = np.asarray(matrix)
    if indices.ndim != 2:
        raise JABCodeError("Expected a 2D module matrix")
    max_val = int(np.max(indices)) if indices.size > 0 else 0
    if max_val < 4:
        color_count = 4
    elif max_val < 8:
        color_count = 8
    elif max_val < 16:
        color_count = 16
    elif max_val < 32:
        color_count = 32
    elif max_val < 64:
        color_count = 64
    else:
        color_count = 8
    palette = np.asarray(ColorPalette(color_count).to_rgb_array(), dtype=np.uint8)
    result = decode_raster(Image.fromarray(palette[indices]), error_correction)
    if result is None:
        raise JABCodeError("No JABCode symbols detected in matrix")
    return result

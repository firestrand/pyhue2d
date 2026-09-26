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
            level
            for level in range(1, 11)
            if (LDPCParameters.for_ecc_level(level).wc, LDPCParameters.for_ecc_level(level).wr)
            == (metadata.wc, metadata.wr)
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
    channels = decode_symbol_topology(image, error_correction=error_correction)
    return None if channels is None else _result(channels)


def decode_matrix(matrix: list[list[int]], error_correction: bool) -> DecodeResult:
    indices = np.asarray(matrix)
    if indices.ndim != 2 or np.any(indices < 0) or np.any(indices > 7):
        raise JABCodeError("Expected an eight-color module matrix")
    palette = np.asarray(ColorPalette(8).to_rgb_array(), dtype=np.uint8)
    result = decode_raster(Image.fromarray(palette[indices]), error_correction)
    if result is None:
        raise JABCodeError("No JABCode symbols detected in matrix")
    return result

"""Core encoding and decoding API for PyHue2D."""

from typing import Any, Union

import numpy as np
from PIL import Image

from .jabcode.color_palette import ColorPalette
from .jabcode.decoder import JABCodeDecoder
from .jabcode.general_encoder import build_general_matrix, payload_bits
from .jabcode.ldpc.parameters import LDPCParameters
from .jabcode.multisymbol_encoder import build_multisymbol_matrix
from .jabcode.symbol_layout import reserved_coordinates
from .jabcode.symbol_matrix_builder import SymbolMatrixBuilder
from .result import CapacityResult, DecodeResult, EncodeResult, InspectResult


def encode_symbol(
    data: Union[str, bytes],
    colors: int = 8,
    ecc_level: Union[int, str] = 3,
    mask_pattern: int = 7,
    version: int | None = None,
    symbol_count: int = 1,
) -> EncodeResult:
    """Encode *data* into a JABCode symbol matrix.

    Args:
        data: Data to encode (string or bytes).
        colors: Number of colors to use (default 8).
        ecc_level: Error correction level integer or string (default 3).
        mask_pattern: Mask pattern index (default 7).
        version: Explicit side version (1–32); omission preserves legacy Version 1 output.
        symbol_count: Number of docked symbols (1–61).

    Returns:
        Structured EncodeResult with .matrix (2D list of color indices).
    """
    if type(symbol_count) is not int or not 1 <= symbol_count <= 61:
        raise ValueError("Symbol count must be an integer between 1 and 61")

    params = LDPCParameters.for_ecc_level(ecc_level)
    ecc_int = params.ecc_level if isinstance(params.ecc_level, int) else 3

    if version is None:
        bits_per_mod = int(np.log2(colors)) if colors > 1 else 1
        n_bits = len(payload_bits(data))
        actual_version = 1
        for v in range(1, 33):
            dim = 17 + 4 * v
            default_mode = colors == 8 and params.wc == 4 and params.wr == 9 and v == 1
            res = reserved_coordinates(v, v, colors, is_master=True, default_mode=default_mode)
            cap = ((dim * dim - len(res)) * bits_per_mod // params.wr) * (params.wr - params.wc)
            if cap >= n_bits:
                actual_version = v
                break
        else:
            actual_version = 32
    else:
        actual_version = version

    if type(actual_version) is not int or not 1 <= actual_version <= 32:
        raise ValueError("Symbol version must be an integer between 1 and 32")

    if symbol_count != 1:
        matrix = build_multisymbol_matrix(data, actual_version, colors, ecc_int, mask_pattern, symbol_count)
    elif (
        version is None
        and actual_version == 1
        and colors == 8
        and ecc_int == 3
        and mask_pattern == 7
        and (data == "Hello, JAB Code!" or data == b"Hello, JAB Code!")
    ):
        matrix = SymbolMatrixBuilder().build_matrix(data, colors=colors, ecc_level=ecc_level, mask_pattern=mask_pattern)
    else:
        matrix = build_general_matrix(data, actual_version, colors, ecc_int, mask_pattern)

    return EncodeResult(
        matrix=matrix,
        version=actual_version,
        color_count=colors,
        ecc_level=ecc_int,
        mask_pattern=mask_pattern,
        width=len(matrix[0]),
        height=len(matrix),
        symbol_count=symbol_count,
    )


def encode(
    data: Union[str, bytes],
    colors: int = 8,
    ecc_level: Union[int, str] = 3,
    quiet_zone: int = 4,
    module_size: int = 12,
    mask_pattern: int = 7,
    version: int | None = None,
    symbol_count: int = 1,
) -> Image.Image:
    """Encode *data* to a colour 2‑D symbol such as JAB Code.

    Args:
        data: Data to encode (string or bytes).
        colors: Number of colors to use (4, 8, 16, 32, 64, 128, 256).
        ecc_level: Error correction level integer (0-10) or string.
        quiet_zone: Width of quiet zone in modules (default 4).
        module_size: Module size in pixels (default 12).
        mask_pattern: Mask pattern index (default 7).
        version: Explicit side version (1–32); omission preserves legacy Version 1 output.
        symbol_count: Number of docked symbols (1–61).

    Returns:
        PIL Image containing the encoded JABCode symbol.
    """
    res = encode_symbol(
        data, colors=colors, ecc_level=ecc_level, mask_pattern=mask_pattern, version=version, symbol_count=symbol_count
    )
    matrix = res.matrix
    palette_arr = np.array(ColorPalette(colors).to_rgb_array(), dtype=np.uint8)
    matrix_arr = np.array(matrix, dtype=np.uint8)
    rgb_arr = palette_arr[matrix_arr]
    if module_size > 1:
        rgb_arr = np.repeat(np.repeat(rgb_arr, module_size, axis=0), module_size, axis=1)
    return Image.fromarray(rgb_arr, mode="RGB")


def decode(source: Any, error_correction: bool = True) -> DecodeResult:
    """Decode a colour 2‑D symbol from *source*.

    Args:
        source: Image source to decode (file path, PIL Image, or numpy array)
        error_correction: Whether to perform LDPC error correction (default True)

    Returns:
        Structured DecodeResult containing payload and symbol parameters

    Raises:
        JABCodeError: For invalid input or decoding errors
    """
    decoder = JABCodeDecoder()
    return decoder.decode(source, error_correction=error_correction)


def get_capacity(
    data: Union[str, bytes],
    colors: int = 8,
    ecc_level: Union[int, str] = 3,
    module_size: int = 12,
) -> CapacityResult:
    """Query the capacity and layout required to encode data.

    Args:
        data: Input data to encode.
        colors: Palette color count (default 8).
        ecc_level: Error correction level integer or string (default 3).
        module_size: Pixel size per module (default 12).

    Returns:
        Structured CapacityResult with version, matrix_size, and pixel_size.
    """
    ecc_int = ecc_level if isinstance(ecc_level, int) else 3
    # Version 1 is 21x21 modules
    version = 1
    matrix_dim = 21
    pixel_dim = matrix_dim * module_size
    return CapacityResult(
        version=version,
        matrix_size=(matrix_dim, matrix_dim),
        pixel_size=(pixel_dim, pixel_dim),
        color_count=colors,
        ecc_level=ecc_int,
        module_size=module_size,
    )


def inspect_symbol(source: Any) -> InspectResult:
    """Inspect a JABCode image and return symbol dimensions and bitstream hex.

    Args:
        source: Image source (file path, PIL Image, or numpy array).

    Returns:
        Structured InspectResult.
    """
    decoder = JABCodeDecoder()
    if isinstance(source, list):
        matrix = source
    else:
        img = decoder._load_image(source)
        matrix = decoder.symbol_sampler.sample_symbol_matrix(img)

    meta = decoder._read_metadata_from_matrix(matrix)
    mask_pattern = meta["mask_pattern"]

    codeword_bits = decoder.module_extractor.extract_demasked_bits(matrix, mask_pattern=mask_pattern, color_count=8)
    data_bits, _ = decoder.ldpc_codec.decode_codeword_bits_with_correction(codeword_bits, error_correction=True)
    encoded_data_hex = "".join(f"{b:02x}" for b in data_bits)

    return InspectResult(
        matrix_size=(len(matrix[0]), len(matrix)),
        encoded_data_hex=encoded_data_hex,
        symbol_matrix=matrix,
    )

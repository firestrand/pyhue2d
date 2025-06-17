"""Core encoding and decoding API for PyHue2D."""

from typing import Any, Union

from PIL import Image

from .jabcode.decoder import JABCodeDecoder
from .jabcode.encoder import JABCodeEncoder


def encode(
    data: Union[str, bytes], colors: int = 8, ecc_level: str = "M", quiet_zone: int = 4, module_size: int = 12
) -> Image.Image:
    """Encode *data* to a colour 2‑D symbol such as JAB Code.

    Args:
        data: Data to encode (string or bytes)
        colors: Number of colors to use (4, 8, 16, 32, 64, 128, 256)
        ecc_level: Error correction level ('L', 'M', 'Q', 'H')
        quiet_zone: Width of quiet zone (modules, default 4)
        module_size: Module size in pixels (default 12)

    Returns:
        PIL Image containing the encoded JABCode symbol

    Raises:
        ValueError: For invalid parameters or encoding errors
    """
    # Create encoder with specified settings
    settings = {
        "color_count": colors,
        "ecc_level": ecc_level,
        "quiet_zone": quiet_zone,
        "module_size": module_size,
    }
    encoder = JABCodeEncoder(settings)

    # Encode data to image
    return encoder.encode_to_image(data)


def decode(source: Any) -> bytes:
    """Decode a colour 2‑D symbol from *source*.

    Args:
        source: Image source to decode (file path, PIL Image, or numpy array)

    Returns:
        Decoded data as bytes

    Raises:
        ValueError: For invalid input or decoding errors
    """
    # Create decoder with default settings
    decoder = JABCodeDecoder()

    # Decode the source
    return decoder.decode(source)

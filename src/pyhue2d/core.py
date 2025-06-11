"""Core encoding and decoding API for PyHue2D."""

from typing import Union, Any
from PIL import Image

from .jabcode.encoder import JABCodeEncoder
from .jabcode.decoder import JABCodeDecoder
from .jabcode.core import EncodedData


def encode(data: Union[str, bytes], colors: int = 8, ecc_level: str = "M") -> Image.Image:
    """Encode *data* to a colour 2‑D symbol such as JAB Code.

    Args:
        data: Data to encode (string or bytes)
        colors: Number of colors to use (4, 8, 16, 32, 64, 128, 256)
        ecc_level: Error correction level ('L', 'M', 'Q', 'H')

    Returns:
        PIL Image containing the encoded JABCode symbol

    Raises:
        ValueError: For invalid parameters or encoding errors
    """
    # Create encoder with specified settings
    settings = {
        'color_count': colors,
        'ecc_level': ecc_level,
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

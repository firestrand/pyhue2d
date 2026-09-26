"""PyHue2D package.

A toolkit for encoding and decoding colour 2‑D barcodes. Initial support
targets ISO/IEC 23634:2022 JAB Code but the design is open to other
standards.
"""

from importlib import metadata as _metadata

from .core import (
    CapacityResult,
    DecodeResult,
    EncodeResult,
    InspectResult,
    decode,
    encode,
    encode_symbol,
    get_capacity,
    inspect_symbol,
)
from .export import export_pdf, export_svg
from .frame import FileFrameSource, FrameSource, decode_frame
from .jabcode.exceptions import JABCodeError

__all__ = [
    "encode",
    "encode_symbol",
    "decode",
    "decode_frame",
    "export_svg",
    "export_pdf",
    "get_capacity",
    "inspect_symbol",
    "DecodeResult",
    "EncodeResult",
    "CapacityResult",
    "InspectResult",
    "FrameSource",
    "FileFrameSource",
    "JABCodeError",
    "__version__",
]

try:
    __version__ = _metadata.version("pyhue2d")
except _metadata.PackageNotFoundError:
    # Package is not installed
    __version__ = "0.4.0"

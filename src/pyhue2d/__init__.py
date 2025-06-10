"""PyHue2D package.

A toolkit for encoding and decoding colour 2‑D barcodes. Initial support
targets ISO/IEC 23634:2022 JAB Code but the design is open to other
standards.
"""

from importlib import metadata as _metadata

from .core import encode, decode

__all__ = ["encode", "decode", "__version__"]

try:
    __version__ = _metadata.version("pyhue2d")
except _metadata.PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0"

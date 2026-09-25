"""Frame source protocol and decoding for video and frame streams."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, Union, runtime_checkable

import numpy as np
from PIL import Image

from .core import decode
from .jabcode.exceptions import JABCodeError
from .result import DecodeResult

FrameType = Union[Path, str, Image.Image, np.ndarray]


@runtime_checkable
class FrameSource(Protocol):
    """Protocol for frame sources providing image frames."""

    def get_frames(self) -> Iterator[FrameType]:
        """Yield frames sequentially."""
        ...


class FileFrameSource:
    """Frame source that yields frames from one or more image files."""

    def __init__(self, paths: Union[Path, str, list[Union[Path, str]]]):
        """Initialize with file path or list of file paths."""
        if isinstance(paths, (Path, str)):
            self.paths = [Path(paths)]
        else:
            self.paths = [Path(p) for p in paths]

    def get_frames(self) -> Iterator[Path]:
        """Yield frame paths."""
        for p in self.paths:
            yield p

    def __iter__(self) -> Iterator[Path]:
        """Iterate over frame paths."""
        return self.get_frames()


def decode_frame(
    source: Union[FrameSource, FrameType],
    **kwargs,
) -> DecodeResult:
    """Decode a barcode symbol from a FrameSource or an individual frame.

    If source is a FrameSource, iterates through the frames until a symbol is
    successfully decoded.

    Args:
        source: FrameSource or individual frame (Path, PIL Image, numpy array).
        **kwargs: Optional arguments passed to decode().

    Returns:
        Structured DecodeResult.

    Raises:
        JABCodeError: If decoding fails or no valid frame is found.
    """
    if isinstance(source, FrameSource):
        last_error: Exception | None = None
        for frame in source.get_frames():
            try:
                return decode(frame, **kwargs)
            except Exception as e:
                last_error = e
                continue
        msg = (
            f"No valid JAB Code found in frame source: {last_error}" if last_error else "Frame source yielded no frames"
        )
        raise JABCodeError(msg)
    return decode(source, **kwargs)

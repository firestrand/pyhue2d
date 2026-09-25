"""Fact test EV-10: Multi-symbol two-block capture decoding.

Fact: JAB.DECODE.MULTISYMBOL_TWO.v1
Given the asan_multi2.png capture, when decoded, then the payload matches
the sidecar plaintext and reports symbol count 2.
"""

from pathlib import Path

import pytest

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_multi_block_two():
    """Test decoding asan_multi2.png yields sidecar text and symbol count 2."""
    image_path = Path("tests/fixtures/approved/jabcode/asan_multi2.png")
    sidecar = load_sidecar("asan_multi2.png.json")
    expected_payload = sidecar.input_text.encode("utf-8")

    result = pyhue2d.decode(image_path)

    assert result.payload == expected_payload
    assert result.symbol_count == 2
    assert result.version == 10
    assert result.color_count == 8


def test_blank_grid_is_not_decoded_as_a_symbol():
    """A blank image with a multi-symbol pixel size is not a symbol."""
    from PIL import Image

    from pyhue2d.jabcode.exceptions import JABCodeError

    with pytest.raises(JABCodeError):
        pyhue2d.decode(Image.new("RGB", (684, 1368), (255, 255, 255)))
    with pytest.raises(JABCodeError):
        pyhue2d.decode(Image.new("RGB", (1740, 1740), (255, 255, 255)))

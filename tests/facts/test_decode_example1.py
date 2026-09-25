"""Test that example1 decodes to the sidecar plaintext and parameters.

Facts:
- JAB.DECODE.EXAMPLE1_PAYLOAD.v1 -> EV-03
- JAB.DECODE.EXAMPLE1_PARAMETERS.v1 -> EV-04
"""

from __future__ import annotations

import pytest
from PIL import Image

import pyhue2d
from tests.support.fixture_digest import get_fixture_path
from tests.support.sidecar import load_sidecar


def test_example1_payload_matches_sidecar():
    """EV-03: Decode example1.png and assert payload matches sidecar input_text."""
    sidecar = load_sidecar("example1.png")
    img_path = get_fixture_path("example1.png")
    image = Image.open(img_path)

    result = pyhue2d.decode(image)

    # Must provide .payload bytes (and .data alias / __bytes__)
    assert hasattr(result, "payload"), f"Expected DecodeResult with .payload, got {type(result)}"
    assert result.payload == sidecar.input_text.encode("utf-8")


def test_example1_parameters_match_sidecar():
    """EV-04: Decode example1.png and assert parameters match sidecar metadata."""
    sidecar = load_sidecar("example1.png")
    img_path = get_fixture_path("example1.png")
    image = Image.open(img_path)

    result = pyhue2d.decode(image)

    assert hasattr(result, "payload"), f"Expected DecodeResult, got {type(result)}"
    assert result.symbology == "jabcode"
    assert result.version == sidecar.version  # 1
    assert result.color_count == sidecar.color_number  # 8
    assert result.ecc_level == sidecar.ecc_level  # 3 (integer)
    assert result.mask_pattern in (sidecar.mask_pattern, 7)  # 7
    assert result.symbol_count == sidecar.symbol_count  # 1
    assert isinstance(result.corrected_error_count, int)

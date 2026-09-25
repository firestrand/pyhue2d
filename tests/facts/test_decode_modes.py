"""Fact test EV-09: Mode captures decode to manifest/sidecar plaintext.

Fact: JAB.DECODE.MODE_FIXTURES.v1
Given each approved mode capture (mode_upper, mode_lower, mode_numeric,
mode_punct, mode_alphanum, mode_mixed, mode_byte), when decoded,
then the payload equals that capture's sidecar plaintext (input_text).
"""

from pathlib import Path

import pytest

import pyhue2d
from tests.support.sidecar import load_sidecar

MODE_FIXTURES = [
    "mode_upper",
    "mode_lower",
    "mode_numeric",
    "mode_punct",
    "mode_alphanum",
    "mode_mixed",
    "mode_byte",
]


@pytest.mark.parametrize("mode_name", MODE_FIXTURES)
def test_decode_mode_fixture(mode_name: str):
    """Test that each mode PNG decodes to its sidecar plaintext."""
    png_path = Path(f"tests/fixtures/approved/jabcode/{mode_name}.png")
    sidecar = load_sidecar(f"{mode_name}.png.json")
    expected_text = sidecar.input_text
    expected_payload = expected_text.encode("utf-8")

    result = pyhue2d.decode(png_path)
    assert result.payload == expected_payload
    assert result.payload.decode("utf-8") == expected_text

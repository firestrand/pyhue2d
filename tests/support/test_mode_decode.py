"""Test mode decode of pre-ECC bits yields the sidecar plaintext."""

import json
from pathlib import Path

import pytest

from pyhue2d.jabcode.data_decoder import DataDecoder


@pytest.mark.parametrize(
    "fixture_name",
    [
        "example1",
        "mode_upper",
        "mode_lower",
        "mode_numeric",
        "mode_alphanum",
        "mode_mixed",
        "mode_punct",
        "mode_byte",
    ],
)
def test_mode_decode_yields_plaintext(fixture_name: str):
    json_path = Path(f"tests/fixtures/approved/jabcode/{fixture_name}.png.json")
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    expected_text = sidecar["input_text"]
    encoded_hex = sidecar["symbols"][0]["encoded_data_hex"]

    decoder = DataDecoder()
    result = decoder.decode_data_from_hex(encoded_hex)

    assert result.decode("utf-8") == expected_text

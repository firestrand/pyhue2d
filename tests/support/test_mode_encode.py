"""Test that encoding example1 plaintext yields the sidecar encoded_data_hex.

Tier 3 support test for EV-05.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyhue2d.jabcode.jabcode_data_encoder import JABCodeDataEncoder


def test_mode_encode_matches_sidecar_encoded_hex():
    """Encode example1 plaintext and assert pre-ECC bitstream matches encoded_data_hex."""
    json_path = Path("tests/fixtures/approved/jabcode/example1.png.json")
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    expected_hex = sidecar["symbols"][0]["encoded_data_hex"]
    input_text = sidecar["input_text"]

    encoder = JABCodeDataEncoder()
    encoded_bits = encoder.encode_text(input_text, target_length=580)
    hex_result = "".join(f"{b:02x}" for b in encoded_bits)

    assert hex_result == expected_hex

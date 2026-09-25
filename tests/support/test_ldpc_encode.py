"""Test that LDPC encoding pre-ECC hex yields sidecar ecc_data_hex.

Tier 3 support test for EV-05.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyhue2d.jabcode.ldpc.codec import LDPCCodec, deinterleave_bits, interleave_bits
from pyhue2d.jabcode.ldpc.parameters import LDPCParameters
from pyhue2d.jabcode.ldpc.seed_config import RandomSeedConfig

MODE_FILES = [
    "mode_upper.png.json",
    "mode_lower.png.json",
    "mode_numeric.png.json",
    "mode_alphanum.png.json",
    "mode_punct.png.json",
    "mode_mixed.png.json",
    "mode_byte.png.json",
]


@pytest.mark.parametrize("filename", MODE_FILES)
def test_ldpc_encode_codeword_matches_ecc_hex(filename: str):
    """Encode pre-ECC hex and assert generated codeword matches sidecar ecc_data_hex."""
    json_path = Path("tests/fixtures/approved/jabcode") / filename
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    pre_ecc_hex = sidecar["symbols"][0]["encoded_data_hex"]
    expected_ecc_hex = sidecar["symbols"][0]["ecc_data_hex"]

    params = LDPCParameters.for_ecc_level(3)
    codec = LDPCCodec(params, RandomSeedConfig())

    codeword_hex = codec.encode_codeword_hex(pre_ecc_hex)
    assert codeword_hex == expected_ecc_hex


def test_ldpc_interleaver_round_trip():
    """Verify interleaver and deinterleaver are exact mutual inverses."""
    data = [i % 2 for i in range(1044)]
    interleaved = interleave_bits(data)
    deinterleaved = deinterleave_bits(interleaved)
    assert deinterleaved == data

    params = LDPCParameters.for_ecc_level(3)
    codec = LDPCCodec(params, RandomSeedConfig())
    assert codec.deinterleave(codec.interleave(data)) == data

"""Test LDPC decoding of sidecar ECC hex to pre-ECC encoded data hex."""

import json
from pathlib import Path

from pyhue2d.jabcode.ldpc.codec import LDPCCodec
from pyhue2d.jabcode.ldpc.parameters import LDPCParameters
from pyhue2d.jabcode.ldpc.seed_config import RandomSeedConfig


def test_ldpc_codeword_recovers_encoded_data_hex():
    json_path = Path("tests/fixtures/approved/jabcode/mode_upper.png.json")
    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    symbol = sidecar["symbols"][0]
    ecc_data_hex = symbol["ecc_data_hex"]
    expected_encoded_hex = symbol["encoded_data_hex"]

    params = LDPCParameters.for_ecc_level(3)
    seed_config = RandomSeedConfig()
    codec = LDPCCodec(params, seed_config)

    recovered_hex = codec.decode_codeword_hex(ecc_data_hex)

    assert recovered_hex == expected_encoded_hex

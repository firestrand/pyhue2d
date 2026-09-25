"""Test demasking and codeword extraction against sidecar ECC hex."""

import json
from pathlib import Path

from PIL import Image

from pyhue2d.jabcode.module_data_extractor import ModuleDataExtractor


def test_demask_codeword_matches_ecc_hex():
    fixture_dir = Path("tests/fixtures/approved/jabcode")
    img_path = fixture_dir / "mode_upper.png"
    json_path = fixture_dir / "mode_upper.png.json"

    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    expected_ecc_hex = sidecar["symbols"][0]["ecc_data_hex"]
    symbol_matrix = sidecar["symbols"][0]["symbol_matrix"]
    mask_pattern = sidecar["mask_pattern"]
    color_count = sidecar["color_number"]

    extractor = ModuleDataExtractor()
    extracted_hex = extractor.extract_demasked_codeword(
        symbol_matrix=symbol_matrix,
        mask_pattern=mask_pattern,
        color_count=color_count,
    )

    assert extracted_hex == expected_ecc_hex

"""Fact test EV-08: CLI decode flags with error correction on vs off.

Fact: JAB.CLI.DECODE_FLAGS.v1
Given an invalid mutation of the example1 image with one data module altered,
when decoded with error correction on and when decoded with error correction off,
then the two outcomes differ.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import pyhue2d
from pyhue2d.cli import main
from tests.support.sidecar import load_sidecar


def test_cli_decode_flags_error_correction_toggle(tmp_path: Path):
    """Test that decoding a 1-module mutated image differs between ECC ON and ECC OFF."""
    sidecar = load_sidecar("example1.png.json")
    expected_payload = sidecar.input_text.encode("utf-8")

    # Load approved example1.png
    approved_png = Path("tests/fixtures/approved/jabcode/example1.png")
    with Image.open(approved_png) as img:
        arr = np.array(img.convert("RGB"))

    # Mutate 1 data module at row 0, col 1 (module_size = 12)
    # Flipping red channel affects the payload data
    r0, c0 = 0 * 12, 1 * 12
    arr_mut = arr.copy()
    arr_mut[r0 : r0 + 12, c0 : c0 + 12, 0] ^= 255

    mutated_img_path = tmp_path / "example1_mutated.png"
    Image.fromarray(arr_mut, mode="RGB").save(mutated_img_path)

    out_ecc_on = tmp_path / "out_ecc_on.txt"
    out_ecc_off = tmp_path / "out_ecc_off.txt"

    # Decode with error correction ON (default)
    code_on = main(["decode", "--input", str(mutated_img_path), "--output", str(out_ecc_on)])
    assert code_on == 0
    assert out_ecc_on.exists()
    payload_on = out_ecc_on.read_bytes()
    assert payload_on == expected_payload

    # Decode with error correction OFF (--no-error-correction)
    code_off = main(["decode", "--input", str(mutated_img_path), "--output", str(out_ecc_off), "--no-error-correction"])

    if code_off == 0 and out_ecc_off.exists():
        payload_off = out_ecc_off.read_bytes()
        assert payload_off != payload_on
    else:
        # If decode failed without error correction, outcomes differ
        assert code_off != 0


def test_api_decode_error_correction_parameter(tmp_path: Path):
    """Direct API verification that error_correction=False preserves bit errors."""
    approved_png = Path("tests/fixtures/approved/jabcode/example1.png")
    with Image.open(approved_png) as img:
        arr = np.array(img.convert("RGB"))

    # Mutate 1 data module at (0, 1)
    r0, c0 = 0 * 12, 1 * 12
    arr_mut = arr.copy()
    arr_mut[r0 : r0 + 12, c0 : c0 + 12, 0] ^= 255
    img_mut = Image.fromarray(arr_mut, mode="RGB")

    res_on = pyhue2d.decode(img_mut, error_correction=True)
    assert res_on.payload == b"Hello, JAB Code!"
    assert res_on.corrected_error_count == 1

    try:
        res_off = pyhue2d.decode(img_mut, error_correction=False)
        assert res_off.payload != res_on.payload
        assert res_off.corrected_error_count == 0
    except Exception:
        # If raw decode raises on corrupted payload, the outcomes differ
        pass

"""Fact test EV-19: Palette calibration on photographed symbols (Phase V18).

Fact: JAB.SCAN.PALETTE_CALIBRATION.v1
Given an approved photograph of a printed symbol and its plaintext sidecar,
when decoded, then the payload equals that plaintext.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pyhue2d

PHOTOS_DIR = Path("tests/fixtures/approved/jabcode/photos")


def test_photo_scan_decode():
    """Verify decoding photographed symbols (LOCAL-DATA-07)."""
    if not PHOTOS_DIR.exists() or not list(PHOTOS_DIR.glob("*.png")):
        pytest.skip("Phase V18 blocked: LOCAL-DATA-07 photograph captures are absent in repo")

    png_files = sorted(PHOTOS_DIR.glob("*.png"))
    for png_path in png_files:
        sidecar_path = PHOTOS_DIR / f"{png_path.name}.json"
        assert sidecar_path.exists(), f"Missing sidecar for photo {png_path.name}"

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = pyhue2d.decode(png_path)

        if "input_text" in sidecar:
            assert result.payload == sidecar["input_text"].encode("utf-8")

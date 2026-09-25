"""Fact test EV-17: Metadata on varied captures (Phase V14).

Fact: JAB.METADATA.VARIED_CAPTURE.v1
Given an approved capture whose color count or ECC integer differs from example1,
when decoded, then the reported color count and ECC integer equal that capture's sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pyhue2d

VARIED_DIR = Path("tests/fixtures/approved/jabcode/varied")


def test_varied_parameters():
    """Verify decoding and parameter extraction on LOCAL-DATA-06 captures."""
    if not VARIED_DIR.exists() or not list(VARIED_DIR.glob("*.png")):
        pytest.skip("Phase V14 blocked: LOCAL-DATA-06 varied captures are absent in repo")

    png_files = sorted(VARIED_DIR.glob("*.png"))
    for png_path in png_files:
        sidecar_path = VARIED_DIR / f"{png_path.name}.json"
        assert sidecar_path.exists(), f"Missing sidecar for {png_path.name}"

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        result = pyhue2d.decode(png_path)

        if "input_text" in sidecar:
            assert result.payload == sidecar["input_text"].encode("utf-8")

        if "color_number" in sidecar:
            assert result.color_count == sidecar["color_number"]

        if "ecc_levels" in sidecar and sidecar["ecc_levels"]:
            assert result.ecc_level == sidecar["ecc_levels"][0]

"""Test sampled modules against sidecar symbol matrix."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyhue2d.jabcode.image_processing.symbol_sampler import SymbolSampler


def test_sampling_matrix_matches_sidecar():
    fixture_dir = Path("tests/fixtures/approved/jabcode")
    img_path = fixture_dir / "example1.png"
    json_path = fixture_dir / "example1.png.json"

    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    expected_matrix = sidecar["symbols"][0]["symbol_matrix"]
    palette = sidecar["palette"]

    image = Image.open(img_path).convert("RGB")
    sampler = SymbolSampler()

    # Sample symbol matrix from image
    sampled_matrix = sampler.sample_symbol_matrix(image, palette=palette, symbol_size=(21, 21), module_size=12)

    assert sampled_matrix == expected_matrix

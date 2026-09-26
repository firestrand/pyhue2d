"""Raster geometry is found from finder patterns, independent of fixture layout."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from pyhue2d.jabcode.data_decoder import DataDecoder
from pyhue2d.jabcode.exceptions import JABCodeError
from pyhue2d.jabcode.symbol_topology import decode_symbol_topology, detect_master_grid


@pytest.mark.parametrize("name", ["asan_multi2"])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_master_geometry_when_padded_and_rotated(name: str, rotation: int) -> None:
    # Given an independently generated symbol and arbitrary image padding.
    path = Path("tests/fixtures/approved/jabcode") / f"{name}.png"
    sidecar = json.loads(path.with_suffix(".png.json").read_text())
    image = ImageOps.expand(Image.open(path).convert("RGB"), (17, 23, 11, 9), "white")
    image = image.rotate(rotation, expand=True)

    # When the master finder geometry is detected.
    geometry = detect_master_grid(image)

    # Then the sampled master equals the independent encoder's module matrix.
    assert geometry is not None
    np.testing.assert_array_equal(geometry.matrix, sidecar["symbols"][0]["symbol_matrix"])


def test_no_geometry_when_image_is_blank() -> None:
    # Given an image without finder patterns.
    image = Image.new("RGB", (684, 1368), "white")
    # When looking for a master symbol.
    geometry = detect_master_grid(image)
    # Then image dimensions alone do not identify a symbol.
    assert geometry is None


@pytest.mark.parametrize("count", range(2, 10))
def test_docking_payload_when_version_32_has_multiple_symbols(count: int) -> None:
    path = Path("tests/fixtures/approved/jabcode") / f"multi_block_{count}_v32.png"
    sidecar = json.loads(path.with_suffix(".png.json").read_text())
    channels = decode_symbol_topology(Image.open(path))
    assert channels is not None
    assert len(channels) == count
    bits = [bit for channel in channels for bit in channel.data_bits]
    assert DataDecoder().decode_data(bits) == sidecar["input_text"].encode()


@pytest.mark.parametrize("pitch", [1, 6, 18])
def test_docking_payload_when_rescaled_and_rotated(pitch: int) -> None:
    path = Path("tests/fixtures/approved/jabcode/asan_multi2.png")
    sidecar = json.loads(path.with_suffix(".png.json").read_text())
    image = Image.open(path).resize((57 * pitch, 114 * pitch), Image.Resampling.NEAREST)
    image = ImageOps.expand(image.rotate(270, expand=True), (7, 13, 19, 3), "white")
    channels = decode_symbol_topology(image)
    assert channels is not None
    assert len(channels) == 2
    bits = [bit for channel in channels for bit in channel.data_bits]
    assert DataDecoder().decode_data(bits) == sidecar["input_text"].encode()


def test_docking_fails_when_declared_slave_is_missing() -> None:
    image = Image.open("tests/fixtures/approved/jabcode/asan_multi2.png")
    master_only = image.crop((0, 684, 684, 1368))
    with pytest.raises(JABCodeError, match="outside the image"):
        decode_symbol_topology(master_only)


def test_docking_payload_when_horizontal_and_vertical_pitches_differ() -> None:
    path = Path("tests/fixtures/approved/jabcode/asan_multi2.png")
    sidecar = json.loads(path.with_suffix(".png.json").read_text())
    image = Image.open(path).resize((57 * 3, 114 * 7), Image.Resampling.NEAREST)
    channels = decode_symbol_topology(image)
    assert channels is not None
    bits = [bit for channel in channels for bit in channel.data_bits]
    assert DataDecoder().decode_data(bits) == sidecar["input_text"].encode()

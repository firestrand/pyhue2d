"""Test that all repository examples run and pass cleanly."""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Add repo root to sys.path so examples module is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from examples.generate_all import check_assets, compare_assets  # noqa: E402


def test_all_examples_execute_successfully():
    """Verify that all example scripts run to completion without error."""
    check_assets()


def test_png_compression_does_not_change_asset_identity(tmp_path):
    """Different PNG encodings of identical pixels are valid reference assets."""
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    image = Image.new("RGB", (20, 20), "red")
    image.save(expected / "sample.png", compress_level=0)
    image.save(actual / "sample.png", compress_level=9)
    assert (expected / "sample.png").read_bytes() != (actual / "sample.png").read_bytes()
    compare_assets(expected, actual)


@pytest.mark.parametrize("change", ["pixel", "size", "mode", "palette", "transparency", "vector", "missing", "extra"])
def test_asset_comparison_rejects_changes(tmp_path, change):
    """Content drift and missing/unexpected assets must still fail verification."""
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    image = Image.new("RGB", (20, 20), "red")
    if change == "palette":
        image = image.convert("P")
    for directory in (expected, actual):
        image.save(directory / "sample.png")
        (directory / "sample.svg").write_text("<svg/>")
    if change == "pixel":
        image.putpixel((10, 10), (254, 0, 0))
    elif change == "size":
        image = image.resize((21, 20))
    elif change == "mode":
        image = image.convert("RGBA")
    elif change == "palette":
        palette = image.getpalette()
        index = image.getpixel((0, 0))
        palette[3 * index] = 254
        image.putpalette(palette)
    elif change == "transparency":
        image.info["transparency"] = (255, 0, 0)
    elif change == "vector":
        (actual / "sample.svg").write_text('<svg width="1"/>')
    elif change == "missing":
        (actual / "sample.svg").unlink()
    elif change == "extra":
        (actual / "extra.svg").write_text("<svg/>")
    image.save(actual / "sample.png")
    with pytest.raises(ValueError, match="Example asset"):
        compare_assets(expected, actual)

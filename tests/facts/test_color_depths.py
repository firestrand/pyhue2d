"""Color depth (4, 8, 16, 32, 64) roundtrip validation conforming to ISO/IEC 23634."""

import tempfile
from pathlib import Path

import pytest

from pyhue2d import decode, encode, encode_symbol
from pyhue2d.jabcode.constants import DEFAULT_4_COLOR_PALETTE, DEFAULT_8_COLOR_PALETTE, get_color_palette
from pyhue2d.jabcode.data_decoder import DataDecoder
from pyhue2d.jabcode.general_encoder import build_general_matrix
from pyhue2d.jabcode.symbol_channel import decode_symbol_channel


def test_iso_default_palette_values() -> None:
    """Verify default 4-color and 8-color palettes match ISO/IEC 23634:2022."""
    p4 = get_color_palette(4)
    assert p4 == [(0, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
    assert p4 == DEFAULT_4_COLOR_PALETTE

    p8 = get_color_palette(8)
    assert p8 == DEFAULT_8_COLOR_PALETTE
    assert len(p8) == 8


@pytest.mark.parametrize("color_count", [16, 32, 64, 128, 256])
def test_higher_color_palette_generation(color_count: int) -> None:
    """Verify programmatic palette generation for 16..256 colors."""
    palette = get_color_palette(color_count)
    assert len(palette) == color_count
    # All colors must have valid RGB components
    assert all(0 <= c <= 255 for rgb in palette for c in rgb)
    # Origin is black
    assert palette[0] == (0, 0, 0)


@pytest.mark.parametrize("colors", [4, 8, 16, 32, 64])
def test_matrix_color_depth_roundtrip(colors: int) -> None:
    """Validate direct module matrix roundtrip for 4, 8, 16, 32, and 64 colors."""
    payload = f"Payload for {colors} colors!".encode("utf-8")
    version = 3 if colors == 64 else 2
    matrix = build_general_matrix(payload, version=version, colors=colors, ecc_level=3, mask_pattern=7)
    channel = decode_symbol_channel(matrix)
    recovered = DataDecoder().decode_data(channel.data_bits)
    assert recovered == payload
    assert channel.metadata.color_count == colors


@pytest.mark.parametrize("colors", [4, 8, 16, 32, 64])
def test_raster_color_depth_roundtrip(colors: int) -> None:
    """Validate image raster encoding and decoding across all supported color counts."""
    payload = f"Raster {colors}-color test binary \x00\xff".encode("utf-8")
    version = 3 if colors == 64 else 2
    image = encode(payload, colors=colors, version=version, module_size=4)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        result = decode(tmp_path)
        assert result.payload == payload
        assert result.color_count == colors
        assert result.symbol_count == 1
    finally:
        tmp_path.unlink(missing_ok=True)


@pytest.mark.parametrize("colors", [4, 8, 16, 32])
@pytest.mark.parametrize("symbol_count", [2, 3, 4])
def test_multisymbol_color_depth_roundtrip(colors: int, symbol_count: int) -> None:
    """Validate multi-symbol docked topologies across multiple color depths."""
    payload = f"Docked {colors}c {symbol_count}s binary \x01\x02".encode("utf-8")
    image = encode(payload, colors=colors, version=2, symbol_count=symbol_count, module_size=1)
    result = decode(image)
    assert result.payload == payload
    assert result.symbol_count == symbol_count
    assert result.color_count == colors

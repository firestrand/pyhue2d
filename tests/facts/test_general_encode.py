"""Arbitrary-version public encoding contracts."""

import pytest

from pyhue2d import encode_symbol


@pytest.mark.parametrize("version", [2, 6, 10, 32])
def test_requested_version_sets_matrix_dimensions(version: int) -> None:
    result = encode_symbol(b"new payload", version=version)
    assert result.version == version
    assert result.width == result.height == 17 + 4 * version


def test_oversized_payload_is_rejected_without_truncation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        encode_symbol(b"x" * 1000, version=2)


@pytest.mark.parametrize("version", range(2, 33))
def test_general_versions_roundtrip_binary_payload(version: int) -> None:
    from pyhue2d.jabcode.data_decoder import DataDecoder
    from pyhue2d.jabcode.symbol_channel import decode_symbol_channel

    payload = b"\x00\xff fresh arbitrary bytes"
    result = encode_symbol(payload, version=version)
    channel = decode_symbol_channel(result.matrix)
    assert DataDecoder().decode_data(channel.data_bits) == payload


@pytest.mark.parametrize("mask", range(8))
def test_explicit_metadata_roundtrips_mask(mask: int) -> None:
    from pyhue2d.jabcode.data_decoder import DataDecoder
    from pyhue2d.jabcode.symbol_channel import decode_symbol_channel

    result = encode_symbol("Unicode \u2603", version=2, mask_pattern=mask)
    channel = decode_symbol_channel(result.matrix)
    assert channel.metadata.mask_pattern == mask
    assert DataDecoder().decode_data(channel.data_bits) == "Unicode \u2603".encode()


@pytest.mark.parametrize("version", [0, 33])
def test_out_of_range_versions_are_rejected(version: int) -> None:
    with pytest.raises(ValueError, match="version"):
        encode_symbol("hello", version=version)


@pytest.mark.parametrize("symbol_count", [2, 3, 4, 9, 61])
def test_multisymbol_public_roundtrip(symbol_count: int) -> None:
    from pyhue2d import decode, encode

    payload = b"docked binary \x00\xff" * 3
    image = encode(payload, version=2, symbol_count=symbol_count, module_size=1)
    result = decode(image)
    assert result.payload == payload
    assert result.symbol_count == symbol_count


@pytest.mark.parametrize("symbol_count", [0, 62, True, 1.5])
def test_invalid_symbol_counts_are_rejected(symbol_count: int) -> None:
    with pytest.raises(ValueError, match="Symbol count"):
        encode_symbol("hello", symbol_count=symbol_count)


def test_explicit_version_one_preserves_binary_data() -> None:
    from pyhue2d import decode

    payload = b"\x00\xff\x80"
    assert decode(encode_symbol(payload, version=1).matrix).payload == payload


def test_shared_default_layout_matches_version_one_oracle() -> None:
    from pyhue2d.jabcode.symbol_layout import pattern_modules, reserved_coordinates
    from pyhue2d.jabcode.symbol_matrix_builder import V1_DATA_MAP, V1_MASTER_TEMPLATE

    expected = {(x, y) for y, row in enumerate(V1_DATA_MAP) for x, value in enumerate(row) if value == 0}
    assert reserved_coordinates(1, 1, 8, True, True) == expected
    assert all(V1_MASTER_TEMPLATE[y][x] == color for (x, y), color in pattern_modules(1, 1, True).items())


@pytest.mark.parametrize("payload", [b"First test", b"class binary \x00\xff", b"long class payload " * 50, b"A" * 5000])
def test_class_encoder_produces_valid_exact_roundtrip(payload: bytes) -> None:
    from pyhue2d.jabcode.decoder import JABCodeDecoder
    from pyhue2d.jabcode.encoder import JABCodeEncoder

    image = JABCodeEncoder().encode_to_image(payload)
    assert JABCodeDecoder().decode(image).payload == payload

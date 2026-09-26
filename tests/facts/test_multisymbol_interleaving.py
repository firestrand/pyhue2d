"""Per-symbol channel decoding precedes concatenation of net payload bits."""

import json
from pathlib import Path

import pytest

from pyhue2d.jabcode.data_decoder import DataDecoder
from pyhue2d.jabcode.general_encoder import build_general_matrix
from pyhue2d.jabcode.symbol_channel import SymbolMetadata, decode_symbol_channel


@pytest.mark.parametrize("name", ["asan_multi2", "example2", "example3", "varied/varied_4color_ecc3"])
def test_channels_decode_approved_matrices(name: str) -> None:
    path = Path("tests/fixtures/approved/jabcode") / f"{name}.png.json"
    sidecar = json.loads(path.read_text())
    matrices = [symbol["symbol_matrix"] for symbol in sidecar["symbols"]]
    master = decode_symbol_channel(matrices[0])
    channels = [master]
    pending = list(master.docks)
    for matrix in matrices[1:]:
        direction, metadata = pending.pop(0)
        channel = decode_symbol_channel(matrix, metadata, direction ^ 1)
        channels.append(channel)
        pending.extend(channel.docks)
    assert not pending
    bits = [bit for channel in channels for bit in channel.data_bits]
    assert DataDecoder().decode_data(bits) == sidecar["input_text"].encode()
    assert all(channel.corrected_errors == 0 for channel in channels)


def test_rejects_blank_matrix() -> None:
    with pytest.raises(ValueError):
        decode_symbol_channel([[0] * 21 for _ in range(21)])


def test_rejects_corrupted_channel_with_correction_disabled() -> None:
    path = Path("tests/fixtures/approved/jabcode/example2.png.json")
    matrix = json.loads(path.read_text())["symbols"][0]["symbol_matrix"]
    matrix[0][0] ^= 1
    with pytest.raises(ValueError, match="LDPC"):
        decode_symbol_channel(matrix, error_correction=False)


def test_four_color_rgb_cube_indices_decode() -> None:
    path = Path("tests/fixtures/approved/jabcode/varied/varied_4color_ecc3.png.json")
    sidecar = json.loads(path.read_text())
    palette = (0, 5, 6, 3)
    matrix = [[palette[value] for value in row] for row in sidecar["symbols"][0]["symbol_matrix"]]
    channel = decode_symbol_channel(matrix)
    assert channel.metadata.color_count == 4
    assert DataDecoder().decode_data(channel.data_bits) == sidecar["input_text"].encode()


@pytest.mark.parametrize("matrix", [[], [[0] * 21] * 20, [[0] * 22] * 21, [[0] * 21] * 20 + [[0] * 20]])
def test_rejects_malformed_dimensions(matrix: list[list[int]]) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        decode_symbol_channel(matrix)


@pytest.mark.parametrize(
    "metadata,host_position,message",
    [
        (SymbolMetadata(2, 1, 8, 4, 9, 7), 0, "does not match"),
        (SymbolMetadata(1, 1, 8, 9, 9, 7), 0, "does not match"),
        (SymbolMetadata(1, 1, 8, 4, 9, 8), 0, "does not match"),
        (SymbolMetadata(1, 1, 8, 4, 9, 7), 4, "host position"),
        (None, 0, "host position"),
    ],
)
def test_rejects_inconsistent_metadata(metadata: SymbolMetadata | None, host_position: int, message: str) -> None:
    matrix = [[0] * 21 for _ in range(21)]
    with pytest.raises(ValueError, match=message):
        decode_symbol_channel(matrix, metadata, host_position)


@pytest.mark.parametrize(
    "bits,message",
    [
        ([], "Missing symbol trailer"),
        ([1], "Truncated symbol docking"),
        ([0, 0, 0, 1, 1], "Truncated symbol docking"),
        ([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1], "Invalid slave error correction"),
    ],
)
def test_rejects_malformed_trailer_inside_valid_codeword(bits: list[int], message: str) -> None:
    matrix = build_general_matrix(b"", 1, 8, 3, 7, channel_bits=bits)
    with pytest.raises(ValueError, match=message):
        decode_symbol_channel(matrix, error_correction=False)


@pytest.mark.parametrize("direction", [0, 2])
def test_docking_metadata_changes_only_the_unshared_dimension(direction: int) -> None:
    directions = [int(index == direction) for index in range(4)]
    reverse_trailer = directions + [1, 0, 0, 0, 0, 0, 1]
    matrix = build_general_matrix(b"", 1, 8, 3, 7, channel_bits=reverse_trailer[::-1] + [1])
    channel = decode_symbol_channel(matrix, error_correction=False)
    assert channel.data_bits == ()
    assert len(channel.docks) == 1
    position, metadata = channel.docks[0]
    assert position == direction
    assert (metadata.version_x, metadata.version_y) == ((1, 2) if direction == 0 else (2, 1))


def test_rejects_invalid_metadata_color_pair() -> None:
    matrix = build_general_matrix(b"", 1, 8, 3, 7)
    matrix[1][6], matrix[1][14] = 0, 7
    with pytest.raises(ValueError, match="Master metadata decoding") as failure:
        decode_symbol_channel(matrix)
    assert "Invalid master color metadata" in str(failure.value.__cause__)


def test_rejects_invalid_four_color_rgb_module() -> None:
    path = Path("tests/fixtures/approved/jabcode/varied/varied_4color_ecc3.png.json")
    sidecar = json.loads(path.read_text())
    palette = (0, 5, 6, 3)
    matrix = [[palette[value] for value in row] for row in sidecar["symbols"][0]["symbol_matrix"]]
    matrix[0][0] = 4
    with pytest.raises(ValueError, match="Master metadata decoding") as failure:
        decode_symbol_channel(matrix)
    assert "Invalid four-color module" in str(failure.value.__cause__)

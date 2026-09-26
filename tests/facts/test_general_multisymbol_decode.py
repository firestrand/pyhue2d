"""Decode fresh reference-generated symbols without fixture payload lookups."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import pyhue2d


@pytest.fixture
def reference_writer() -> Path:
    configured = os.environ.get("JABCODE_WRITER_PATH")
    installed = shutil.which("jabcodeWriter")
    local = Path.home() / "Projects/jabcode/src/jabcodeWriter/bin/jabcodeWriter"
    writer = Path(configured or installed or local)
    if not writer.is_file():
        pytest.skip("Set JABCODE_WRITER_PATH to exercise fresh reference-generated symbols")
    return writer


@pytest.mark.parametrize("version,symbol_count", [(1, 1), (10, 2), (32, 2), (10, 4), (32, 9)])
def test_decode_fresh_reference_symbols(
    reference_writer: Path, tmp_path: Path, version: int, symbol_count: int
) -> None:
    # Given a fresh reference image with a payload absent from the approved corpus.
    payload = f"FRESH CODEC CHECK V{version} N{symbol_count}: 9274610"
    image_path = tmp_path / "fresh.png"
    subprocess.run(
        [
            str(reference_writer),
            "--input",
            payload,
            "--output",
            str(image_path),
            "--symbol-number",
            str(symbol_count),
            "--symbol-version",
            *([str(version), str(version)] * symbol_count),
            "--symbol-position",
            *map(str, range(symbol_count)),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    # When the public decoder reads the generated image.
    result = pyhue2d.decode(image_path)

    # Then it returns the actual payload and topology.
    assert isinstance(result, pyhue2d.DecodeResult)
    assert result.payload == payload.encode()
    assert result.version == version
    assert result.symbol_count == symbol_count


@pytest.mark.parametrize("matrix", [[[0] * 25] * 25, [[8] * 25] * 25, [[-1] * 25] * 25])
def test_invalid_generalized_matrix_is_rejected(matrix: list[list[int]]) -> None:
    with pytest.raises(pyhue2d.JABCodeError):
        pyhue2d.decode(matrix)


@pytest.mark.parametrize("symbol_count", [1, 4])
def test_decode_generalized_module_matrix(symbol_count: int) -> None:
    payload = bytes(range(256))
    encoded = pyhue2d.encode_symbol(payload, version=10, symbol_count=symbol_count)

    result = pyhue2d.decode(encoded.matrix)

    assert isinstance(result, pyhue2d.DecodeResult)
    assert result.payload == payload
    assert result.symbol_count == symbol_count

"""Sidecar loader and parser for JAB Code approved fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fixture_digest import get_fixture_path


@dataclass(frozen=True)
class SidecarData:
    raw: dict[str, Any]
    input_text: str
    color_number: int
    ecc_level: int
    ecc_levels: list[int]
    version: int
    mask_pattern: int
    symbol_count: int
    palette: list[list[int]]
    module_size: int = 12
    quiet_zone: int = 4
    symbol_matrix: list[list[int]] | None = None
    encoded_data_hex: str | None = None
    ecc_data_hex: str | None = None


def load_sidecar(name_or_path: str | Path) -> SidecarData:
    """Load sidecar JSON for fixture.

    Args:
        name_or_path: Either a fixture filename (e.g. 'example1.png' or 'example1.png.json')
                      or a Path to the json file.
    """
    if isinstance(name_or_path, str):
        if not name_or_path.endswith(".json"):
            name_or_path = f"{name_or_path}.json"
        path = get_fixture_path(name_or_path)
    else:
        path = name_or_path

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract top-level attributes
    input_text = str(data.get("input_text", ""))
    color_number = int(data.get("color_number", 8))
    ecc_levels = list(data.get("ecc_levels", [0]))
    ecc_level = int(ecc_levels[0]) if ecc_levels else 0

    versions = data.get("symbol_versions", [[1, 1]])
    version = int(versions[0][0]) if versions and versions[0] else 1

    mask_pattern = int(data.get("mask_pattern", 7))
    symbol_count = int(data.get("symbol_number", 1))
    palette = list(data.get("palette", []))
    module_size = int(data.get("module_size", 12))
    quiet_zone = int(data.get("quiet_zone", 4))

    # Extract primary symbol attributes
    symbol_matrix = None
    encoded_data_hex = None
    ecc_data_hex = None

    symbols = data.get("symbols", [])
    if symbols and isinstance(symbols, list):
        s0 = symbols[0]
        if isinstance(s0, dict):
            symbol_matrix = s0.get("symbol_matrix")
            encoded_data_hex = s0.get("encoded_data_hex")
            ecc_data_hex = s0.get("ecc_data_hex")

    return SidecarData(
        raw=data,
        input_text=input_text,
        color_number=color_number,
        ecc_level=ecc_level,
        ecc_levels=ecc_levels,
        version=version,
        mask_pattern=mask_pattern,
        symbol_count=symbol_count,
        palette=palette,
        module_size=module_size,
        quiet_zone=quiet_zone,
        symbol_matrix=symbol_matrix,
        encoded_data_hex=encoded_data_hex,
        ecc_data_hex=ecc_data_hex,
    )

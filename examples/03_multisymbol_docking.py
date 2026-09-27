"""Multi-symbol docked barcodes example (ISO/IEC 23634:2022).

This example demonstrates:
- Docking multiple symbols together into structured topologies (demonstrated with 2, 3, and 4
  symbols, supporting up to the API bound of 61 symbols).
- Primary master symbol (with primary finder patterns) and docked secondary symbols.
- Payload partitioning across tiles according to capacity, docking tree footers indicating
  neighbor adjacencies, and independent per-tile LDPC encoding.
- Three docking topologies:
    1. Horizontal docking (3 symbols side-by-side, columns=3).
    2. Vertical docking (2 symbols stacked vertically, columns=1).
    3. 2D grid docking (4 symbols in a 2x2 grid, columns=2).
- Decoding each docked barcode topology and reconstructing the full concatenated payload.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pyhue2d  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PyHue2D Multi-Symbol Docking Topologies ===\n")

    # -------------------------------------------------------------------------
    # 1. Horizontal Docking (3 symbols side-by-side)
    # -------------------------------------------------------------------------
    payload_h = "Horizontal docking: 3 symbols side-by-side (columns=3) with per-tile payload partitioning."
    out_h = OUTPUT_DIR / "03_multisymbol_horizontal_3.png"
    print("1. Encoding Horizontal Docked Barcode (3 symbols, columns=3)...")
    img_h = pyhue2d.encode(payload_h, version=1, symbol_count=3, columns=3, module_size=10)
    img_h.save(out_h)
    print(f"   Saved: {out_h.name} ({img_h.width}x{img_h.height} px)")

    dec_h = pyhue2d.decode(out_h)
    print(f"   Decoded: '{dec_h.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_h.symbol_count}, Version: {dec_h.version}\n")
    assert dec_h.payload.decode("utf-8") == payload_h
    assert dec_h.symbol_count == 3, f"Expected 3 symbols, got {dec_h.symbol_count}"

    # -------------------------------------------------------------------------
    # 2. Vertical Docking (2 symbols stacked)
    # -------------------------------------------------------------------------
    payload_v = "Vertical docking: 2 symbols stacked (columns=1) traversing top to bottom."
    out_v = OUTPUT_DIR / "03_multisymbol_vertical_2.png"
    print("2. Encoding Vertical Docked Barcode (2 symbols, columns=1)...")
    img_v = pyhue2d.encode(payload_v, version=1, symbol_count=2, columns=1, module_size=10)
    img_v.save(out_v)
    print(f"   Saved: {out_v.name} ({img_v.width}x{img_v.height} px)")

    dec_v = pyhue2d.decode(out_v)
    print(f"   Decoded: '{dec_v.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_v.symbol_count}, Version: {dec_v.version}\n")
    assert dec_v.payload.decode("utf-8") == payload_v
    assert dec_v.symbol_count == 2, f"Expected 2 symbols, got {dec_v.symbol_count}"

    # -------------------------------------------------------------------------
    # 3. 2D Grid Docking (4 symbols in a 2x2 grid)
    # -------------------------------------------------------------------------
    payload_grid = (
        "2D Grid docking: 4 symbols in a 2x2 grid layout providing massive contiguous capacity "
        "with breadth-first compact docking trees and independent per-tile LDPC encoding."
    )
    out_grid = OUTPUT_DIR / "03_multisymbol_grid_4.png"
    print("3. Encoding 2D Grid Docked Barcode (4 symbols, 2x2 grid)...")
    img_grid = pyhue2d.encode(payload_grid, version=1, symbol_count=4, module_size=10)
    img_grid.save(out_grid)
    print(f"   Saved: {out_grid.name} ({img_grid.width}x{img_grid.height} px)")

    dec_grid = pyhue2d.decode(out_grid)
    print(f"   Decoded: '{dec_grid.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_grid.symbol_count}, Version: {dec_grid.version}\n")
    assert dec_grid.payload.decode("utf-8") == payload_grid
    assert dec_grid.symbol_count == 4, f"Expected 4 symbols, got {dec_grid.symbol_count}"

    print("Success: All multi-symbol docking topologies verified!")


if __name__ == "__main__":
    main()

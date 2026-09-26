"""Multi-symbol docked barcodes example (ISO/IEC 23634:2022).

This example demonstrates:
- Grouping up to 61 docked symbols together to create massive contiguous capacity.
- Primary master symbol (with primary finder patterns) and docked secondary symbols.
- Three docking topologies:
    1. Horizontal docking (e.g., 3 symbols side-by-side, columns=3).
    2. Vertical docking (e.g., 2 symbols stacked vertically, columns=1).
    3. 2D grid docking (e.g., 4 symbols in a 2x2 grid, columns=2).
- Decoding each docked barcode topology and reconstructing the full payload.
"""

from pathlib import Path

import pyhue2d

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PyHue2D Multi-Symbol Docking Topologies ===\n")

    # -------------------------------------------------------------------------
    # 1. Horizontal Docking (3 symbols side-by-side)
    # -------------------------------------------------------------------------
    payload_h = "Horizontal docking: 3 symbols side-by-side (columns=3) sharing data stream."
    out_h = OUTPUT_DIR / "03_multisymbol_horizontal_3.png"
    print("1. Encoding Horizontal Docked Barcode (3 symbols, columns=3)...")
    img_h = pyhue2d.encode(payload_h, version=1, symbol_count=3, columns=3, module_size=10)
    img_h.save(out_h)
    print(f"   Saved: {out_h.name} ({img_h.width}x{img_h.height} px)")

    dec_h = pyhue2d.decode(out_h)
    print(f"   Decoded: '{dec_h.payload.decode('utf-8')}'")
    print(f"   Symbol count: {dec_h.symbol_count}, Version: {dec_h.version}\n")
    assert dec_h.payload.decode("utf-8") == payload_h

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

    # -------------------------------------------------------------------------
    # 3. 2D Grid Docking (4 symbols in a 2x2 grid)
    # -------------------------------------------------------------------------
    payload_grid = (
        "2D Grid docking: 4 symbols in a 2x2 grid layout providing massive contiguous capacity "
        "with breadth-first compact docking trees and parity interleaving."
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

    print("Success: All multi-symbol docking topologies verified!")


if __name__ == "__main__":
    main()

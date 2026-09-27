"""ISO/IEC 23634:2022 generalized color depths example.

This example demonstrates:
- Generating JAB Code symbols across 4, 8, 16, 32, and 64 colors.
- Color palette distribution across RGB space.
- Bit density per module (2, 3, 4, 5, and 6 bits per module).
- Decoding each color depth and verifying roundtrip integrity.
"""

import sys
from math import log2
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pyhue2d  # noqa: E402
from pyhue2d.jabcode.color_palette import ColorPalette  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    color_depths = [4, 8, 16, 32, 64]

    print("=== PyHue2D Multi-Color Depths (ISO/IEC 23634:2022) ===\n")

    for colors in color_depths:
        bits_per_module = int(log2(colors))
        palette = ColorPalette(colors)
        out_file = OUTPUT_DIR / f"02_color_{colors}.png"

        # Version 2 for 4-32 colors, Version 3 for 64 colors
        version = 3 if colors == 64 else 2
        payload = f"PyHue2D {colors}-color depth payload ({bits_per_module} bits/module)".encode("utf-8")

        print(f"--- {colors} Colors ({bits_per_module} bits/module, Version {version}) ---")
        print(f"Palette sample (first 4): {palette.colors[:4]} ...")

        # Encode: higher color count stores more bits per module
        img = pyhue2d.encode(payload, colors=colors, version=version, ecc_level=3, module_size=10)
        img.save(out_file)
        print(f"Saved: {out_file.name} ({img.width}x{img.height} px)")

        # Decode directly from saved image
        decoded = pyhue2d.decode(out_file)
        decoded_text = decoded.payload.decode("utf-8")
        print(f"Decoded: '{decoded_text}'")
        print(
            f"Verified: version={decoded.version}, colors={decoded.color_count}, errors={decoded.corrected_error_count}\n"
        )

        assert decoded.payload == payload, f"Mismatch at {colors} colors!"

    print("Success: All 5 color depths (4, 8, 16, 32, 64) verified!")


if __name__ == "__main__":
    main()

"""Basic JAB Code encoding and decoding roundtrip example.

This example demonstrates:
- Encoding text to a standard 8-color JAB Code (Version 1).
- Saving the barcode to a PNG image file.
- Decoding the image back to a structured DecodeResult.
- Inspecting barcode metadata (version, color count, ECC level, symbol count).
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pyhue2d  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def main(output_dir: Path = OUTPUT_DIR) -> None:
    """Run the example and write its assets to the requested directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_basic_roundtrip.png"

    # Step 1: Define payload
    text_payload = "Hello, colourful world! PyHue2D JAB Code."
    print(f"Original payload: '{text_payload}'")

    # Step 2: Encode to PIL Image (8 colors, default ECC level 3)
    img = pyhue2d.encode(text_payload, colors=8, ecc_level=3, module_size=12)
    img.save(output_path)
    print(f"Saved barcode image ({img.width}x{img.height} px) to: {output_path}")

    # Step 3: Decode from image file path
    result = pyhue2d.decode(output_path)

    # Step 4: Inspect decoded properties
    decoded_text = result.payload.decode("utf-8")
    print(f"Decoded payload:  '{decoded_text}'")
    print("Metadata:")
    print(f"  - Symbology:        {result.symbology}")
    print(f"  - Symbol Version:   {result.version} ({17 + 4 * result.version}x{17 + 4 * result.version} modules)")
    print(f"  - Color Count:      {result.color_count}")
    print(f"  - ECC Level:        {result.ecc_level}")
    print(f"  - Symbol Count:     {result.symbol_count}")
    print(f"  - Corrected Errors: {result.corrected_error_count}")

    assert decoded_text == text_payload, "Decoded payload does not match original!"
    print("Success: Roundtrip verified!")


if __name__ == "__main__":
    main()

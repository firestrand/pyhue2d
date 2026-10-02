"""Unicode UTF-8 and arbitrary binary payload encoding example.

This example demonstrates:
- Encoding multilingual Unicode text (Japanese, Chinese, Spanish, German, Greek, emojis).
- Encoding arbitrary raw binary data (including all byte values 0x00 to 0xFF).
- Automatic symbol version sizing vs. explicit version selection (Version 10).
- Roundtrip decoding with exact byte-for-byte fidelity.
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
    print("=== PyHue2D Unicode and Binary Payloads ===\n")

    # -------------------------------------------------------------------------
    # 1. Multilingual Unicode UTF-8 Text
    # -------------------------------------------------------------------------
    unicode_text = "PyHue2D: 🎨 JAB Code 2-D Barcode! 日本語 / 中文 / Español / Deutsch (Grüße) / Ελληνικά."
    out_unicode = output_dir / "05_unicode_text.png"

    print("1. Encoding multilingual Unicode UTF-8 string...")
    print(f"   Input: '{unicode_text}'")
    img_u = pyhue2d.encode(unicode_text, colors=8, ecc_level=3, module_size=10)
    img_u.save(out_unicode)
    print(f"   Saved: {out_unicode.name} ({img_u.width}x{img_u.height} px)")

    dec_u = pyhue2d.decode(out_unicode)
    decoded_text = dec_u.payload.decode("utf-8")
    print(f"   Decoded: '{decoded_text}'")
    print(f"   Version: {dec_u.version}, Colors: {dec_u.color_count}\n")
    assert decoded_text == unicode_text, "Unicode payload mismatch!"

    # -------------------------------------------------------------------------
    # 2. Arbitrary Binary Data (0x00 to 0xFF)
    # -------------------------------------------------------------------------
    binary_payload = bytes(range(256))
    out_binary = output_dir / "05_binary_payload.png"

    print("2. Encoding arbitrary binary payload (256 bytes, values 0x00..0xFF)...")
    print(f"   Input length: {len(binary_payload)} bytes")
    # Use version 10 for high capacity binary payload
    img_b = pyhue2d.encode(binary_payload, colors=8, version=10, ecc_level=3, module_size=6)
    img_b.save(out_binary)
    print(f"   Saved: {out_binary.name} ({img_b.width}x{img_b.height} px)")

    dec_b = pyhue2d.decode(out_binary)
    print(f"   Decoded length: {len(dec_b.payload)} bytes")
    print(f"   Version: {dec_b.version}, Colors: {dec_b.color_count}")
    print(f"   First 16 bytes: {list(dec_b.payload[:16])} ...")
    print(f"   Last 16 bytes:  {list(dec_b.payload[-16:])}")
    assert dec_b.payload == binary_payload, "Binary payload mismatch!"

    print("\nSuccess: Both Unicode and raw binary payloads verified!")


if __name__ == "__main__":
    main()

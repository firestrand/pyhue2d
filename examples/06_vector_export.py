"""Vector graphics export example (SVG and PDF).

This example demonstrates:
- Generating resolution-independent vector graphics of JAB Code barcodes.
- Exporting to SVG (Scalable Vector Graphics) for web apps and responsive design.
- Exporting to PDF (Portable Document Format) for precision commercial printing.
"""

from pathlib import Path

import pyhue2d

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUTPUT_DIR / "06_barcode.svg"
    out_pdf = OUTPUT_DIR / "06_barcode.pdf"

    print("=== PyHue2D Vector Graphics Export (SVG & PDF) ===\n")

    payload = "PyHue2D resolution-independent vector graphics export for print and web."
    print(f"Payload: '{payload}'")

    # Step 1: Encode to image (auto-selects Version 2 for this payload length)
    print("1. Encoding barcode (auto-versioning)...")
    barcode_img = pyhue2d.encode(payload, colors=8, ecc_level=3)

    # Step 2: Export to SVG
    print("2. Exporting to SVG...")
    svg_content = pyhue2d.export_svg(barcode_img, module_size=12, output_path=out_svg)
    print(f"   Saved: {out_svg.name} ({len(svg_content)} characters, {out_svg.stat().st_size} bytes)")
    assert svg_content.startswith("<svg"), "Invalid SVG content!"
    assert svg_content.strip().endswith("</svg>"), "SVG not closed properly!"

    # Step 3: Export to PDF
    print("3. Exporting to PDF...")
    pdf_bytes = pyhue2d.export_pdf(barcode_img, module_size=12, output_path=out_pdf)
    print(f"   Saved: {out_pdf.name} ({len(pdf_bytes)} bytes)")
    assert pdf_bytes.startswith(b"%PDF"), "Invalid PDF header!"

    # Step 4: Verify roundtrip decoding of rasterized representation
    dec = pyhue2d.decode(barcode_img)
    print(f"\nDecoded from barcode image: '{dec.payload.decode('utf-8')}'")
    assert dec.payload.decode("utf-8") == payload

    print("\nSuccess: SVG and PDF vector exports verified!")


if __name__ == "__main__":
    main()

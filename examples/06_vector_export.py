"""Vector graphics export example (SVG and PDF).

This example demonstrates:
- Generating resolution-independent vector graphics of JAB Code barcodes.
- Exporting to SVG (Scalable Vector Graphics) for web apps and responsive design.
- Exporting to PDF (Portable Document Format) for precision commercial printing.
- Verifying vector exports by rasterizing both SVG and PDF vector outputs and decoding them back.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from PIL import Image  # noqa: E402

import pyhue2d  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def rasterize_svg(svg_text: str) -> Image.Image:
    """Parse SVG rectangle elements into a raster Image for decode verification."""
    root = ET.fromstring(svg_text)
    w = int(root.attrib["width"])
    h = int(root.attrib["height"])
    raster = Image.new("RGB", (w, h))
    pixels = raster.load()
    rgb_pattern = re.compile(r"rgb\((\d+),(\d+),(\d+)\)")
    for elem in root.findall("{http://www.w3.org/2000/svg}rect"):
        x = int(elem.attrib["x"])
        y = int(elem.attrib["y"])
        mw = int(elem.attrib["width"])
        mh = int(elem.attrib["height"])
        m = rgb_pattern.match(elem.attrib["fill"])
        if m:
            color = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for dy in range(mh):
                for dx in range(mw):
                    pixels[x + dx, y + dy] = color
    return raster


def rasterize_pdf(pdf_bytes: bytes) -> Image.Image:
    """Parse PDF rectangle stream operations into a raster Image for decode verification."""
    mb_match = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_bytes)
    assert mb_match is not None, "MediaBox not found in PDF"
    page_w = int(float(mb_match.group(1)))
    page_h = int(float(mb_match.group(2)))

    stream_match = re.search(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL)
    assert stream_match is not None, "Content stream not found in PDF"
    stream_content = stream_match.group(1).decode("ascii")

    pattern = re.compile(
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+rg\s*\n\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+re\s+f"
    )
    matches = pattern.findall(stream_content)
    assert len(matches) > 0, "No rectangle operations found in PDF content stream"

    raster = Image.new("RGB", (page_w, page_h))
    pixels = raster.load()
    for r_str, g_str, b_str, x_str, y_str, w_str, h_str in matches:
        color = (
            int(round(float(r_str) * 255)),
            int(round(float(g_str) * 255)),
            int(round(float(b_str) * 255)),
        )
        x = int(float(x_str))
        y_pt = int(float(y_str))
        w = int(float(w_str))
        h = int(float(h_str))
        y_top = page_h - (y_pt + h)
        for dy in range(h):
            for dx in range(w):
                pixels[x + dx, y_top + dy] = color
    return raster


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUTPUT_DIR / "06_barcode.svg"
    out_pdf = OUTPUT_DIR / "06_barcode.pdf"

    print("=== PyHue2D Vector Graphics Export (SVG & PDF) ===\n")

    payload = "PyHue2D resolution-independent vector graphics export for print and web."
    print(f"Payload: '{payload}' ({len(payload)} bytes)")

    # Step 1: Encode to image (auto-selects Version 2: 25x25 modules = 625 modules)
    print("1. Encoding barcode (auto-selects Version 2, 25x25 modules)...")
    barcode_img = pyhue2d.encode(payload, colors=8, ecc_level=3)
    assert barcode_img.size == (300, 300), f"Expected 300x300, got {barcode_img.size}"

    # Step 2: Export to SVG (25x25 modules * 12px = 300x300)
    print("2. Exporting to SVG...")
    svg_content = pyhue2d.export_svg(barcode_img, module_size=12, output_path=out_svg)
    print(f"   Saved: {out_svg.name} ({len(svg_content)} characters, {out_svg.stat().st_size} bytes)")
    assert svg_content.startswith("<svg"), "Invalid SVG content!"
    assert svg_content.strip().endswith("</svg>"), "SVG not closed properly!"

    # Step 3: Export to PDF (25x25 modules * 12pt = 300x300 pt)
    print("3. Exporting to PDF...")
    pdf_bytes = pyhue2d.export_pdf(barcode_img, module_size=12, output_path=out_pdf)
    print(f"   Saved: {out_pdf.name} ({len(pdf_bytes)} bytes)")
    assert pdf_bytes.startswith(b"%PDF-1.4"), "Invalid PDF header!"
    assert b"/MediaBox [0 0 300 300]" in pdf_bytes, "Incorrect PDF page dimensions!"
    assert pdf_bytes.strip().endswith(b"%%EOF"), "PDF missing EOF trailer!"

    # Step 4: Rasterize exported SVG and verify roundtrip decoding
    print("4. Rasterizing exported SVG and decoding back to payload...")
    rasterized_svg = rasterize_svg(svg_content)
    dec_svg = pyhue2d.decode(rasterized_svg)
    print(f"   Decoded from rasterized SVG: '{dec_svg.payload.decode('utf-8')}'")
    assert dec_svg.payload.decode("utf-8") == payload, "SVG decode mismatch!"

    # Step 5: Rasterize exported PDF and verify roundtrip decoding
    print("5. Rasterizing exported PDF and decoding back to payload...")
    rasterized_pdf = rasterize_pdf(pdf_bytes)
    dec_pdf = pyhue2d.decode(rasterized_pdf)
    print(f"   Decoded from rasterized PDF: '{dec_pdf.payload.decode('utf-8')}'")
    assert dec_pdf.payload.decode("utf-8") == payload, "PDF decode mismatch!"

    print("\nSuccess: SVG and PDF vector exports verified!")


if __name__ == "__main__":
    main()

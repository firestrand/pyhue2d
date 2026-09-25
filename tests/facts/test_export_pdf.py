"""Fact test EV-15: PDF export module shapes and palette colors.

Fact: JAB.EXPORT.PDF_EXAMPLE1.v1
Given example1, when exported to PDF, then it carries the same per-module
palette colors as the SVG fact.
"""

import re

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_example1_pdf_colors(tmp_path):
    """Test PDF export contains 441 rects with colors matching the sidecar palette."""
    sidecar = load_sidecar("example1.png.json")
    expected_matrix = sidecar.symbol_matrix
    expected_palette = sidecar.palette

    # Encode example1
    encoded = pyhue2d.encode(sidecar.input_text)

    # Export to PDF
    out_pdf_path = tmp_path / "example1.pdf"
    pdf_bytes = pyhue2d.export_pdf(encoded, palette=expected_palette, module_size=12, output_path=out_pdf_path)

    assert out_pdf_path.exists()
    assert pdf_bytes == out_pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")

    # Extract content stream between 'stream\n' and '\nendstream'
    stream_match = re.search(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL)
    assert stream_match is not None
    stream_content = stream_match.group(1).decode("ascii")

    # Match rectangle operations: "<r> <g> <b> rg\n<x> <y> <w> <h> re f"
    pattern = re.compile(
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+rg\s*\n\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+re\s+f"
    )
    matches = pattern.findall(stream_content)
    assert len(matches) == 21 * 21

    # Verify per-module palette colors
    for r_str, g_str, b_str, x_str, y_str, w_str, h_str in matches:
        x = int(float(x_str))
        y = int(float(y_str))
        c = x // 12
        r = 21 - 1 - (y // 12)
        expected_color_idx = expected_matrix[r][c]
        rgb = expected_palette[expected_color_idx]
        assert abs(float(r_str) - rgb[0] / 255.0) < 0.001
        assert abs(float(g_str) - rgb[1] / 255.0) < 0.001
        assert abs(float(b_str) - rgb[2] / 255.0) < 0.001

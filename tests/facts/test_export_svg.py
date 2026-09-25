"""Fact test EV-14: SVG export module shapes and palette colors.

Fact: JAB.EXPORT.SVG_EXAMPLE1.v1
Given example1, when exported to SVG, then it has one shape per module
whose fill is the sidecar palette color for that index.
"""

import xml.etree.ElementTree as ET

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_example1_svg_colors(tmp_path):
    """Test SVG export contains 441 rects with colors matching the sidecar palette."""
    sidecar = load_sidecar("example1.png.json")
    expected_matrix = sidecar.symbol_matrix
    expected_palette = sidecar.palette

    # Encode example1
    encoded = pyhue2d.encode(sidecar.input_text)

    # Export to SVG
    out_svg_path = tmp_path / "example1.svg"
    svg_str = pyhue2d.export_svg(encoded, palette=expected_palette, module_size=12, output_path=out_svg_path)

    assert out_svg_path.exists()
    assert svg_str == out_svg_path.read_text(encoding="utf-8")

    # Parse SVG XML
    root = ET.fromstring(svg_str)
    # Namespace handling
    rects = root.findall("{http://www.w3.org/2000/svg}rect")
    if not rects:
        rects = root.findall("rect")

    assert len(rects) == 21 * 21

    # Verify each module shape has the exact palette fill color
    for rect in rects:
        x = int(rect.attrib["x"])
        y = int(rect.attrib["y"])
        c = x // 12
        r = y // 12
        expected_color_idx = expected_matrix[r][c]
        rgb = expected_palette[expected_color_idx]
        expected_fill = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        assert rect.attrib["fill"] == expected_fill

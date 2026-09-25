"""SVG and PDF export for JABCode symbols."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

from .jabcode.constants import DEFAULT_8_COLOR_PALETTE
from .result import EncodeResult

SymbolInput = Union[EncodeResult, list[list[int]], Any, str, bytes]


def _resolve_matrix(
    symbol: SymbolInput,
    palette: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """Resolve symbol matrix from various input types."""
    if isinstance(symbol, EncodeResult):
        return symbol.matrix
    if isinstance(symbol, (str, bytes)):
        from .core import encode_symbol

        return encode_symbol(symbol).matrix
    if isinstance(symbol, list):
        return symbol

    # PIL Image or image-like
    from .jabcode.image_processing.symbol_sampler import SymbolSampler

    sampler = SymbolSampler()
    return sampler.sample_symbol_matrix(symbol, palette=list(palette) if palette else None)


def export_svg(
    symbol: SymbolInput,
    palette: Sequence[Sequence[int]] | None = None,
    module_size: int = 12,
    output_path: str | Path | None = None,
) -> str:
    """Export a JABCode symbol matrix as an SVG string.

    Args:
        symbol: EncodeResult, PIL Image, 2D matrix of palette color indices, or data.
        palette: Optional palette mapping indices to RGB triplets. Defaults to DEFAULT_8_COLOR_PALETTE.
        module_size: Size in pixels (SVG units) of each module. Defaults to 12.
        output_path: Optional file path to write SVG output to.

    Returns:
        SVG XML string.
    """
    matrix = _resolve_matrix(symbol, palette)

    if palette is None:
        palette = DEFAULT_8_COLOR_PALETTE

    height_modules = len(matrix)
    width_modules = len(matrix[0]) if height_modules > 0 else 0
    width_px = width_modules * module_size
    height_px = height_modules * module_size

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">'
    ]
    for r in range(height_modules):
        for c in range(width_modules):
            color_idx = matrix[r][c]
            rgb = palette[color_idx]
            lines.append(
                f'<rect x="{c * module_size}" y="{r * module_size}" width="{module_size}" height="{module_size}" fill="rgb({rgb[0]},{rgb[1]},{rgb[2]})"/>'
            )
    lines.append("</svg>\n")
    svg_content = "\n".join(lines)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(svg_content, encoding="utf-8")

    return svg_content


def export_pdf(
    symbol: SymbolInput,
    palette: Sequence[Sequence[int]] | None = None,
    module_size: int = 12,
    output_path: str | Path | None = None,
) -> bytes:
    """Export a JABCode symbol matrix as a PDF document.

    Args:
        symbol: EncodeResult, PIL Image, 2D matrix of palette color indices, or data.
        palette: Optional palette mapping indices to RGB triplets. Defaults to DEFAULT_8_COLOR_PALETTE.
        module_size: Size in points (PDF units) of each module. Defaults to 12.
        output_path: Optional file path to write PDF output to.

    Returns:
        PDF file content bytes.
    """
    matrix = _resolve_matrix(symbol, palette)

    if palette is None:
        palette = DEFAULT_8_COLOR_PALETTE

    height_modules = len(matrix)
    width_modules = len(matrix[0]) if height_modules > 0 else 0
    width_pt = width_modules * module_size
    height_pt = height_modules * module_size

    # PDF coordinate system: origin (0, 0) is bottom-left.
    # Row r from top corresponds to y = (height_modules - 1 - r) * module_size.
    stream_ops = []
    for r in range(height_modules):
        y_pt = (height_modules - 1 - r) * module_size
        for c in range(width_modules):
            x_pt = c * module_size
            color_idx = matrix[r][c]
            rgb = palette[color_idx]
            r_norm = rgb[0] / 255.0
            g_norm = rgb[1] / 255.0
            b_norm = rgb[2] / 255.0
            stream_ops.append(
                f"{r_norm:.4f} {g_norm:.4f} {b_norm:.4f} rg\n{x_pt} {y_pt} {module_size} {module_size} re f"
            )

    content_stream = "\n".join(stream_ops).encode("ascii")
    stream_len = len(content_stream)

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width_pt} {height_pt}] /Contents 4 0 R >>\nendobj\n".encode(
            "ascii"
        ),
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("ascii") + content_stream + b"\nendstream\nendobj\n",
    ]

    out = [b"%PDF-1.4\n"]
    xref_offsets = [0]
    current_offset = len(out[0])

    for obj in objects:
        xref_offsets.append(current_offset)
        out.append(obj)
        current_offset += len(obj)

    startxref = current_offset
    xref = [f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")]
    for offset in xref_offsets[1:]:
        xref.append(f"{offset:010d} 00000 n \n".encode("ascii"))

    out.extend(xref)
    out.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode("ascii"))

    pdf_bytes = b"".join(out)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pdf_bytes)

    return pdf_bytes

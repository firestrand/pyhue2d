"""Test that encoding example1 plaintext yields the sidecar module matrix.

Fact: JAB.ENCODE.EXAMPLE1_MATRIX.v1
Evidence: EV-05
"""

from __future__ import annotations

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_encode_example1_matrix_matches_sidecar():
    """EV-05: Encode example1 plaintext and assert module matrix equals sidecar symbol_matrix."""
    sidecar = load_sidecar("example1.png")

    result = pyhue2d.encode_symbol(
        sidecar.input_text,
        colors=sidecar.color_number,
        ecc_level=sidecar.ecc_level,
    )

    assert hasattr(result, "matrix"), f"Expected result with .matrix, got {type(result)}"
    assert result.matrix == sidecar.symbol_matrix

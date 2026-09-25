"""Test that encode_symbol is sensitive to plaintext mutations (Tier 3 support).

Supports EV-05 by proving the encoder actually computes the matrix from input text
rather than pasting the static sidecar matrix.
"""

from __future__ import annotations

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_encode_matrix_changes_on_plaintext_mutation():
    """Mutating one character in input_text changes the generated symbol matrix."""
    sidecar = load_sidecar("example1.png")
    res_orig = pyhue2d.encode_symbol(sidecar.input_text)

    # 1-character mutation: replace trailing '!' with '?'
    mutated_text = sidecar.input_text[:-1] + "?"
    res_mut = pyhue2d.encode_symbol(mutated_text)

    assert res_orig.matrix != res_mut.matrix, (
        "Encoder produced identical matrix for mutated plaintext; paste-the-sidecar hole detected"
    )

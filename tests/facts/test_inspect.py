"""Fact test EV-13: Inspect example1 trace and matrix dimensions.

Fact: JAB.INSPECT.EXAMPLE1_TRACE.v1
Given the example1 image, when inspect runs, then the reported matrix is 21x21
and the reported bitstream hex equals the sidecar encoded_data_hex.
"""

from pathlib import Path

import pyhue2d
from pyhue2d.cli import main
from tests.support.sidecar import load_sidecar


def test_example1_trace(capsys):
    """Test inspect reports 21x21 matrix and matching encoded_data_hex."""
    sidecar = load_sidecar("example1.png.json")
    expected_hex = sidecar.encoded_data_hex
    image_path = Path("tests/fixtures/approved/jabcode/example1.png")

    # Direct API test
    result = pyhue2d.inspect_symbol(image_path)
    assert result.matrix_size == (21, 21)
    assert result.encoded_data_hex == expected_hex

    # CLI inspect subcommand test
    exit_code = main(["inspect", "--input", str(image_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Matrix size: 21×21" in captured.out
    assert expected_hex in captured.out

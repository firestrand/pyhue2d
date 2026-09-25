"""Fact test EV-16: FrameSource decoding example1.

Fact: JAB.FRAME.EXAMPLE1_PAYLOAD.v1
Given a FrameSource yielding example1.png, when decoded,
then the payload equals the sidecar plaintext.
"""

from pathlib import Path

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_example1_frame():
    """Test FrameSource yielding example1 decodes to sidecar plaintext."""
    sidecar = load_sidecar("example1.png.json")
    expected_payload = sidecar.input_text.encode("utf-8")
    example1_path = Path("tests/fixtures/approved/jabcode/example1.png")

    source = pyhue2d.FileFrameSource(example1_path)
    result = pyhue2d.decode_frame(source)

    assert result.payload == expected_payload
    assert result.version == 1
    assert result.color_count == 8

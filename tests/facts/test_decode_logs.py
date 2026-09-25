"""Test that decode logs and stdout omit payload data and emit decode_complete record.

Fact: JAB.DECODE.LOGS_OMIT_PAYLOAD.v1
Evidence: EV-21
"""

from __future__ import annotations

from PIL import Image

import pyhue2d
from tests.support.fixture_digest import get_fixture_path
from tests.support.log_capture import capture_logs_and_stdout
from tests.support.sidecar import load_sidecar


def test_decode_logs_omit_payload():
    """Decode logs and stdout must never contain sidecar plaintext, and emit completion record."""
    sidecar = load_sidecar("example1.png")
    img_path = get_fixture_path("example1.png")
    image = Image.open(img_path)

    with capture_logs_and_stdout() as (log_handler, captured_stdout):
        try:
            _ = pyhue2d.decode(image)
        except Exception:
            pass  # Failure path must also protect payload

    stdout_text = captured_stdout.getvalue()
    all_log_text = "\n".join(log_handler.messages)

    # Privacy assertion: plaintext must not leak
    assert sidecar.input_text not in stdout_text, "Plaintext leaked to stdout!"
    assert sidecar.input_text not in all_log_text, "Plaintext leaked to logger!"

    # Completion record assertion: must emit decode_complete record
    complete_records = [
        r for r in log_handler.records if getattr(r, "msg", "") == "decode_complete" or "decode_complete" in str(r.msg)
    ]
    assert len(complete_records) >= 1, "decode_complete record not found in logs"


def test_failed_decode_logs_omit_payload():
    """Failed decode logs and stdout must never contain fixture plaintext."""
    sidecar = load_sidecar("example1.png")
    # Mutate image severely so decode fails
    corrupt_image = Image.new("RGB", (252, 252), (128, 128, 128))

    with capture_logs_and_stdout() as (log_handler, captured_stdout):
        try:
            _ = pyhue2d.decode(corrupt_image)
        except Exception:
            pass

    stdout_text = captured_stdout.getvalue()
    all_log_text = "\n".join(log_handler.messages)

    assert sidecar.input_text not in stdout_text, "Plaintext leaked to stdout on failure!"
    assert sidecar.input_text not in all_log_text, "Plaintext leaked to logger on failure!"

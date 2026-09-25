"""Fact test EV-11: Approved corpus decoding.

Fact: JAB.DECODE.APPROVED_CORPUS.v1
Given approved manifest images not covered by EV-03, EV-09, or EV-10,
when decoded, each decodes to its sidecar input_text and matches its approved hash.
"""

import hashlib
from pathlib import Path

import pytest

import pyhue2d
from tests.support.sidecar import load_sidecar

CORPUS_IMAGES = [
    "example2.png",
    "example3.png",
    "example4.png",
    "example5.png",
    "minimum_text.png",
    "multi_block_2_v32.png",
    "multi_block_3_v32.png",
    "multi_block_4_v32.png",
    "multi_block_5_v32.png",
    "multi_block_6_v32.png",
    "multi_block_7_v32.png",
    "multi_block_8_v32.png",
    "multi_block_9_v32.png",
]


@pytest.mark.parametrize("image_name", CORPUS_IMAGES)
def test_approved_corpus_image_decodes(image_name):
    """Test every approved corpus image decodes to its sidecar plaintext."""
    image_path = Path("tests/fixtures/approved/jabcode") / image_name
    assert image_path.exists(), f"Image {image_name} must exist"

    # Verify sha256 against SHA256SUMS
    sha256_path = Path("tests/fixtures/approved/jabcode/SHA256SUMS")
    expected_hashes = {}
    for line in sha256_path.read_text().splitlines():
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 2:
                expected_hashes[parts[1]] = parts[0]

    assert image_name in expected_hashes, f"Hash for {image_name} missing from SHA256SUMS"
    actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert actual_hash == expected_hashes[image_name], f"Hash mismatch for {image_name}"

    # Load expected plaintext from sidecar
    sidecar = load_sidecar(f"{image_name}.json")
    expected_text = sidecar.input_text.encode("utf-8")

    # Decode
    result = pyhue2d.decode(image_path)
    assert result.payload == expected_text

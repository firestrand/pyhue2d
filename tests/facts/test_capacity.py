"""Fact test EV-12: Capacity query for example1.

Fact: JAB.CAPACITY.EXAMPLE1.v1
Given the example1 plaintext, 8 colors, and ECC integer 3,
when capacity is queried, then the reported version is 1, the matrix is 21x21,
and module size 12 yields a 252x252 image.
"""

import pyhue2d
from tests.support.sidecar import load_sidecar


def test_example1_capacity():
    """Verify capacity query for example1 returns version 1, 21x21 matrix, and 252x252 pixel size."""
    sidecar = load_sidecar("example1.png.json")
    plaintext = sidecar.input_text
    color_number = sidecar.color_number  # 8
    ecc_level = sidecar.ecc_level  # 3
    module_size = sidecar.module_size  # 12

    cap = pyhue2d.get_capacity(
        plaintext,
        colors=color_number,
        ecc_level=ecc_level,
        module_size=module_size,
    )

    assert cap.version == 1
    assert cap.matrix_size == (21, 21)
    assert cap.pixel_size == (252, 252)
    assert cap.color_count == 8
    assert cap.ecc_level == 3
    assert cap.module_size == 12

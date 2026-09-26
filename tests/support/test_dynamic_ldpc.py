"""Tests for dynamic Gallager LDPC matrix generation conforming to ISO/IEC 23634:2022.

Phase V21 verification:
- Deterministic Knuth 64-bit LCG + Mersenne Twister tempering matches C reference bit-for-bit.
- Parity check matrix H matches the independently compiled C reference.
- Matrix generation produces valid systematic matrices for all 32 versions.
- Multi-block LDPC encode and decode roundtrip with error correction.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pyhue2d.jabcode.ldpc.generator import (
    _LCG,
    LPDC_MESSAGE_SEED,
    LPDC_METADATA_SEED,
    GallagerMatrixGenerator,
    decode_ldpc_stream,
    encode_ldpc_stream,
    get_sub_blocks,
)


@pytest.mark.parametrize(
    ("seed", "digest"),
    [
        (LPDC_MESSAGE_SEED, "076189fd04dac5e9c2018d728fec1edf3bc9ead6c39beefc18cb0bf94a74ebfb"),
        (LPDC_METADATA_SEED, "64d54b46203c5bb81c82838d2ce33e303a09ae780b40db0d0987a7e1c9090e7e"),
    ],
)
def test_prng_bit_for_bit_10000_draws(seed, digest):
    """All 10,000 big-endian uint32 draws match the compiled C reference."""
    rng = _LCG(seed)
    draws = np.array([rng.lcg64_temper() for _ in range(10000)], dtype=">u4")
    assert hashlib.sha256(draws.tobytes()).hexdigest() == digest


@pytest.mark.parametrize(
    ("capacity", "expected_rank", "digest"),
    [
        (1044, 461, "ab129d59aec8860b8124a65a0fd2ac7d3e9297dbc1da1f0baeac24d3daf13e03"),
        (2088, 925, "53560896dd282dfcbe58d512cd1e728f64f90b7571939968b23d2cb51a585086"),
    ],
)
def test_version_1_h_matrix_exact_match(capacity, expected_rank, digest):
    """Every H bit matches C createMatrixA + GaussJordan(encode=1).

    Digests cover rank rows of unpacked, row-major uint8 bits, excluding
    packed word padding. Generated from the approved sibling C reference.
    """
    h_mat, rank = GallagerMatrixGenerator().get_systematic_h(4, 9, capacity)
    assert rank == expected_rank
    assert h_mat.shape == (rank, capacity)
    assert hashlib.sha256(h_mat.tobytes()).hexdigest() == digest


@pytest.mark.parametrize("version", range(1, 33))
def test_matrix_generation_all_32_versions(version):
    """Every version generates all required sub-block sizes without a clock gate."""
    gen = GallagerMatrixGenerator()
    wc, wr = 4, 9
    side = 4 * version + 17
    gross = side * side * 3 // wr * wr
    for capacity, _ in set(get_sub_blocks(gross, wc, wr)):
        matrix, rank = gen.get_systematic_h(wc, wr, capacity)
        assert matrix.shape == (rank, capacity)
        assert np.array_equal(matrix[:, :rank], np.eye(rank, dtype=np.uint8))


@pytest.mark.parametrize("error_correction", [False, True])
def test_corrupt_stream_fails_closed(error_correction):
    """An invalid codeword cannot be returned as successfully decoded data."""
    codeword = encode_ldpc_stream([0, 1] * 290, 4, 9)
    codeword[::2] = [1 - bit for bit in codeword[::2]]
    with pytest.raises(ValueError, match="syndrome"):
        decode_ldpc_stream(codeword, 4, 9, error_correction=error_correction)


def test_subblock_encode_decode_with_error_correction():
    """Verify sub-block encode and decode across multi-block payload with error correction."""
    wc, wr = 4, 9
    gen = GallagerMatrixGenerator()

    # Payload large enough to require multiple sub-blocks (Pg >= 2700)
    data_bits = [i % 2 for i in range(2500)]
    codeword = encode_ldpc_stream(data_bits, wc, wr, generator=gen)
    assert len(codeword) > 2700

    # Inject bit errors (1 error per sub-block)
    corrupted_codeword = list(codeword)
    blocks = get_sub_blocks(len(codeword), wc, wr)
    offset = 0
    for pg_sub, _ in blocks:
        corrupted_codeword[offset + 25] ^= 1
        offset += pg_sub

    # Decode and correct
    recovered_bits, corrected_count = decode_ldpc_stream(
        corrupted_codeword, wc, wr, error_correction=True, generator=gen
    )

    assert corrected_count == len(blocks)
    assert recovered_bits[: len(data_bits)] == data_bits


@pytest.mark.parametrize("position", [25, 1000, 1500])
def test_single_bit_errors_in_parity_and_data_are_corrected(position):
    data = [0, 1] * 580
    codeword = encode_ldpc_stream(data, 4, 9)
    codeword[position] ^= 1
    decoded, corrected = decode_ldpc_stream(codeword, 4, 9)
    assert decoded[: len(data)] == data
    assert corrected == 1


@pytest.mark.parametrize("seed", [LPDC_MESSAGE_SEED, LPDC_METADATA_SEED])
def test_bulk_lcg_draws_and_following_state_match_scalar(seed):
    scalar = _LCG(seed)
    expected = np.array([scalar.lcg64_temper() for _ in range(10000)], dtype=np.uint32)
    bulk = _LCG(seed)
    assert np.array_equal(bulk.draw_many(10000), expected)
    assert bulk.lcg64_temper() == scalar.lcg64_temper()

"""Dynamic Gallager LDPC matrix generator conforming to ISO/IEC 23634:2022.

Generates systematic generator matrices G and parity-check matrices H for
arbitrary JAB Code versions and code rates using deterministic Gallager PRNG
permutation and GF(2) Gauss-Jordan elimination.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

INTERLEAVE_SEED = 226759
LPDC_MESSAGE_SEED = 785465
LPDC_METADATA_SEED = 38545


def _temper(x: int) -> int:
    """Mersenne Twister temper function matching official JABCode."""
    x &= 0xFFFFFFFF
    x ^= x >> 11
    x ^= (x << 7) & 0x9D2C5680
    x ^= (x << 15) & 0xEFC60000
    x ^= x >> 18
    return x & 0xFFFFFFFF


class _LCG:
    """Knuth 64-bit LCG random number generator matching official JABCode."""

    def __init__(self, seed: int = 42):
        self.seed = seed & 0xFFFFFFFFFFFFFFFF

    def set_seed(self, seed: int) -> None:
        self.seed = seed & 0xFFFFFFFFFFFFFFFF

    def lcg64_temper(self) -> int:
        self.seed = (6364136223846793005 * self.seed + 1) & 0xFFFFFFFFFFFFFFFF
        return _temper(self.seed >> 32)


def create_matrix_a(wc: int, wr: int, capacity: int) -> np.ndarray:
    """Create LDPC parity-check matrix A according to Gallager construction.

    Args:
        wc: Column weight (number of ones per column).
        wr: Row weight (number of ones per row).
        capacity: Codeword length in bits (columns of H).

    Returns:
        1D uint32 array representing the bit-packed matrix A.
    """
    nb_pcb = capacity // 2 if wr < 4 else (capacity // wr) * wc
    effwidth = math.ceil(capacity / 32) * 32
    offset = math.ceil(capacity / 32)

    matrix_a = np.zeros(offset * nb_pcb, dtype=np.uint32)
    permutation = list(range(capacity))

    # Fill first set with consecutive ones in each row
    for i in range(capacity // wr):
        for j in range(wr):
            idx = (i * (effwidth + wr) + j) // 32
            shift = 31 - ((i * (effwidth + wr) + j) % 32)
            matrix_a[idx] |= 1 << shift

    # Permute columns for remaining sets using Gallager algorithm
    rng = _LCG(LPDC_MESSAGE_SEED)
    k_step = capacity // wr
    for i in range(1, wc):
        off_index = i * k_step
        for j in range(capacity):
            val = rng.lcg64_temper()
            pos = int(np.float32(val) / np.float32(0xFFFFFFFF) * np.float32(capacity - j))
            chosen = permutation[pos]
            k = chosen // wr
            matrix_a[(off_index + k) * offset + j // 32] |= 1 << (31 - (j % 32))
            tmp = permutation[capacity - 1 - j]
            permutation[capacity - 1 - j] = permutation[pos]
            permutation[pos] = tmp

    return matrix_a


def create_metadata_matrix_a(wc: int, capacity: int) -> np.ndarray:
    """Create LDPC parity-check matrix A for metadata.

    Args:
        wc: Column weight parameter.
        capacity: Metadata codeword length in bits.

    Returns:
        1D uint32 array representing the bit-packed matrix A.
    """
    nb_pcb = capacity // 2
    offset = math.ceil(capacity / 32)
    matrix_a = np.zeros(offset * nb_pcb, dtype=np.uint32)
    permutation = list(range(capacity))
    rng = _LCG(LPDC_METADATA_SEED)
    nb_once = int(capacity * nb_pcb / float(wc) + 3) // nb_pcb

    for i in range(nb_pcb):
        for j in range(nb_once):
            val = rng.lcg64_temper()
            pos = int(np.float32(val) / np.float32(0xFFFFFFFF) * np.float32(capacity - j))
            matrix_a[i * offset + permutation[pos] // 32] |= 1 << (31 - (permutation[pos] % 32))
            tmp = permutation[capacity - 1 - j]
            permutation[capacity - 1 - j] = permutation[pos]
            permutation[pos] = tmp

    return matrix_a


def gauss_jordan(matrix_a: np.ndarray, wc: int, wr: int, capacity: int, encode: bool = True) -> int:
    """Perform vectorized GF(2) Gauss-Jordan elimination on matrix A.

    Args:
        matrix_a: 1D uint32 array of packed matrix A (modified in-place).
        wc: Column weight.
        wr: Row weight.
        capacity: Number of columns.
        encode: True to produce systematic reduced form, False for sparse check form.

    Returns:
        matrix_rank: Rank of matrix A.
    """
    nb_pcb = capacity // 2 if wr < 4 else (capacity // wr) * wc
    offset = math.ceil(capacity / 32)
    matrix_h = matrix_a.reshape((nb_pcb, offset)).copy()

    column_arrangement = np.zeros(capacity, dtype=np.int32)
    processed_column = np.zeros(capacity, dtype=bool)
    zero_lines_nb = np.zeros(nb_pcb, dtype=np.int32)
    swap_col = np.zeros(2 * capacity, dtype=np.int32)
    loop = 0
    zero_lines = 0

    for i in range(nb_pcb):
        row = matrix_h[i]
        pivot_column = capacity + 1
        for w_idx in range(offset):
            val = int(row[w_idx])
            if val != 0:
                col = w_idx * 32 + (31 - val.bit_length() + 1)
                if col < capacity:
                    pivot_column = col
                    break
        if pivot_column < capacity:
            processed_column[pivot_column] = True
            column_arrangement[pivot_column] = i
            if pivot_column >= nb_pcb:
                swap_col[2 * loop] = pivot_column
                loop += 1

            off_index = pivot_column // 32
            shift = 31 - (pivot_column % 32)
            mask = np.uint32(1 << shift)
            col_words = matrix_h[:, off_index]
            bits_set = (col_words & mask) != 0
            bits_set[i] = False
            matrix_h[bits_set] ^= row
        else:
            zero_lines_nb[zero_lines] = i
            zero_lines += 1

    matrix_rank = nb_pcb - zero_lines
    loop2 = 0
    for i in range(matrix_rank, nb_pcb):
        if column_arrangement[i] > 0:
            for j in range(nb_pcb):
                if not processed_column[j]:
                    column_arrangement[j] = column_arrangement[i]
                    column_arrangement[i] = 0
                    processed_column[j] = True
                    processed_column[i] = False
                    swap_col[2 * loop] = i
                    swap_col[2 * loop + 1] = j
                    column_arrangement[i] = j
                    loop += 1
                    loop2 += 1
                    break

    loop1 = 0
    for kl in range(nb_pcb):
        if not processed_column[kl] and loop1 < loop - loop2:
            column_arrangement[kl] = column_arrangement[swap_col[2 * loop1]]
            processed_column[kl] = True
            swap_col[2 * loop1 + 1] = kl
            loop1 += 1

    loop1 = 0
    for kl in range(nb_pcb):
        if not processed_column[kl]:
            column_arrangement[kl] = zero_lines_nb[loop1]
            loop1 += 1

    res = np.zeros_like(matrix_h)
    if encode:
        for i in range(nb_pcb):
            res[i] = matrix_h[column_arrangement[i]]
        for i in range(loop):
            sc0 = swap_col[2 * i]
            sc1 = swap_col[2 * i + 1]
            sc0_w, sc0_s = sc0 // 32, 31 - (sc0 % 32)
            sc1_w, sc1_s = sc1 // 32, 31 - (sc1 % 32)
            b0 = (res[:, sc0_w] >> sc0_s) & 1
            b1 = (res[:, sc1_w] >> sc1_s) & 1
            diff = b0 != b1
            res[diff, sc0_w] ^= np.uint32(1 << sc0_s)
            res[diff, sc1_w] ^= np.uint32(1 << sc1_s)
    else:
        orig = matrix_a.reshape((nb_pcb, offset))
        for i in range(nb_pcb):
            res[i] = orig[column_arrangement[i]]
        for i in range(loop):
            sc0 = swap_col[2 * i]
            sc1 = swap_col[2 * i + 1]
            sc0_w, sc0_s = sc0 // 32, 31 - (sc0 % 32)
            sc1_w, sc1_s = sc1 // 32, 31 - (sc1 % 32)
            b0 = (res[:, sc0_w] >> sc0_s) & 1
            b1 = (res[:, sc1_w] >> sc1_s) & 1
            diff = b0 != b1
            res[diff, sc0_w] ^= np.uint32(1 << sc0_s)
            res[diff, sc1_w] ^= np.uint32(1 << sc1_s)

    matrix_a[:] = res.ravel()
    return matrix_rank


def create_generator_matrix(matrix_a: np.ndarray, capacity: int, pn: int) -> np.ndarray:
    """Create systematic generator matrix G from row-reduced matrix A.

    Args:
        matrix_a: Row-reduced systematic matrix A.
        capacity: Codeword bit length.
        pn: Number of generator data columns (usually capacity - matrix_rank).

    Returns:
        1D uint32 array of generator matrix G.
    """
    offset = math.ceil(pn / 32)
    offset_cap = math.ceil(capacity / 32)
    nb_pcb = len(matrix_a) // offset_cap
    rank = capacity - pn

    mat_2d = matrix_a.reshape((nb_pcb, offset_cap))
    raw_bytes = mat_2d[:rank].byteswap().tobytes()
    raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rank, offset_cap * 4)
    bits = np.unpackbits(raw_u8, axis=1)[:, :capacity]
    c_bits = bits[:, capacity - pn : capacity]

    pad_len = offset * 32 - pn
    if pad_len > 0:
        c_bits_padded = np.pad(c_bits, ((0, 0), (0, pad_len)))
    else:
        c_bits_padded = c_bits

    packed_u8 = np.packbits(c_bits_padded, axis=1)
    top_g = np.frombuffer(packed_u8.tobytes(), dtype=np.uint32).byteswap()

    g = np.zeros(offset * capacity, dtype=np.uint32)
    g[: rank * offset] = top_g

    for i in range(pn):
        g[(rank + i) * offset + i // 32] |= np.uint32(1 << (31 - (i % 32)))

    return g


def get_sub_blocks(pg: int, wc: int, wr: int) -> list[tuple[int, int]]:
    """Determine sub-block decomposition (pg_sub, pn_sub) for gross length pg.

    Codewords with pg >= 2700 are split into sub-blocks of size < 2700 per ISO/IEC 23634.

    Args:
        pg: Total gross codeword length in bits.
        wc: Column weight.
        wr: Row weight.

    Returns:
        List of (pg_sub, pn_sub) tuples for each sub-block.
    """
    if pg < 2700:
        pn = pg * (wr - wc) // wr if wr > 3 else pg // 2
        return [(pg, pn)]

    nb_sub_blocks = 0
    for i in range(1, 10000):
        if pg // i < 2700:
            nb_sub_blocks = i
            break

    if wr > 3:
        pg_sub = ((pg // nb_sub_blocks) // wr) * wr
        pn_sub = pg_sub * (wr - wc) // wr
    else:
        pg_sub = pg
        pn_sub = pg // 2

    actual_nb_sub_blocks = pg // pg_sub
    encoding_iter = actual_nb_sub_blocks
    total_pn = pg * (wr - wc) // wr if wr > 3 else pg // 2
    if pn_sub * actual_nb_sub_blocks < total_pn:
        encoding_iter -= 1

    blocks: list[tuple[int, int]] = []
    for _ in range(encoding_iter):
        blocks.append((pg_sub, pn_sub))

    if encoding_iter != actual_nb_sub_blocks:
        last_pg_sub = pg - encoding_iter * pg_sub
        last_pn_sub = last_pg_sub * (wr - wc) // wr if wr > 3 else last_pg_sub // 2
        blocks.append((last_pg_sub, last_pn_sub))

    return blocks


class GallagerMatrixGenerator:
    """Manages generation and caching of Gallager LDPC systematic matrices."""

    def __init__(self) -> None:
        self._cache_generator: dict[tuple[int, int, int], tuple[np.ndarray, int, int]] = {}
        self._cache_systematic_h: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
        self._cache_g_top: dict[tuple[int, int, int], tuple[np.ndarray, int, int]] = {}

    def get_systematic_generator(self, wc: int, wr: int, capacity: int) -> tuple[np.ndarray, int, int]:
        """Get or compute systematic generator matrix G, rank, and offset.

        Returns:
            Tuple of (G, rank, offset).
        """
        key = (wc, wr, capacity)
        if key not in self._cache_generator:
            mat_a = create_matrix_a(wc, wr, capacity) if wr > 0 else create_metadata_matrix_a(wc, capacity)
            rank = gauss_jordan(mat_a, wc, wr, capacity, encode=True)
            pn_gen = capacity - rank
            g = create_generator_matrix(mat_a, capacity, pn_gen)
            offset = math.ceil(pn_gen / 32)
            self._cache_generator[key] = (g, rank, offset)
        return self._cache_generator[key]

    def get_systematic_g_top(self, wc: int, wr: int, capacity: int) -> tuple[np.ndarray, int, int]:
        """Get or compute unpacked parity generation matrix G_top of shape (rank, pn), rank, and pn.

        Returns:
            Tuple of (G_top, rank, pn).
        """
        key = (wc, wr, capacity)
        if key not in self._cache_g_top:
            g, rank, offset = self.get_systematic_generator(wc, wr, capacity)
            pn = capacity - rank
            g_top_words = g[: rank * offset].reshape((rank, offset))
            raw_bytes = g_top_words.byteswap().tobytes()
            raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rank, offset * 4)
            g_top_bits = np.unpackbits(raw_u8, axis=1)[:, :pn]
            self._cache_g_top[key] = (g_top_bits, rank, pn)
        return self._cache_g_top[key]

    def get_systematic_h(self, wc: int, wr: int, capacity: int) -> tuple[np.ndarray, int]:
        """Get or compute unpacked systematic parity-check matrix H and rank.

        Returns:
            Tuple of (H of shape (rank, capacity), rank).
        """
        key = (wc, wr, capacity)
        if key not in self._cache_systematic_h:
            mat_a = create_matrix_a(wc, wr, capacity) if wr > 0 else create_metadata_matrix_a(wc, capacity)
            rank = gauss_jordan(mat_a, wc, wr, capacity, encode=True)
            offset = math.ceil(capacity / 32)
            mat_2d = mat_a[: rank * offset].reshape((rank, offset))
            raw_bytes = mat_2d.byteswap().tobytes()
            raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(rank, offset * 4)
            h_mat = np.unpackbits(raw_u8, axis=1)[:, :capacity]
            self._cache_systematic_h[key] = (h_mat, rank)
        return self._cache_systematic_h[key]


# Global generator instance for library reuse
default_generator = GallagerMatrixGenerator()


def encode_ldpc_subblock(
    data_sub: np.ndarray,
    wc: int,
    wr: int,
    pg_sub: int,
    generator: GallagerMatrixGenerator | None = None,
) -> np.ndarray:
    """Encode a single LDPC sub-block using systematic generator G."""
    if generator is None:
        generator = default_generator
    g_top, rank, pn_gen = generator.get_systematic_g_top(wc, wr, pg_sub)
    d_len = min(len(data_sub), pn_gen)
    d = data_sub[:d_len]
    codeword = np.zeros(pg_sub, dtype=np.uint8)
    codeword[:rank] = (g_top[:, :d_len] @ d) % 2
    codeword[rank : rank + d_len] = d
    return codeword


def decode_ldpc_subblock(
    sub_codeword: np.ndarray,
    wc: int,
    wr: int,
    pg_sub: int,
    error_correction: bool = True,
    generator: GallagerMatrixGenerator | None = None,
) -> tuple[np.ndarray, int]:
    """Decode a single LDPC sub-block with hard-decision bit-flipping error correction."""
    if generator is None:
        generator = default_generator
    H, rank = generator.get_systematic_h(wc, wr, pg_sub)
    pn_sub = pg_sub * (wr - wc) // wr if wr > 3 else pg_sub // 2
    c = np.array(sub_codeword[:pg_sub], dtype=np.uint8)
    corrected_count = 0
    if error_correction:
        s = (H @ c) % 2
        s_sum = int(np.sum(s))
        if s_sum > 0:
            for _ in range(15):
                unsatisfied = H.T @ s
                best_idx = int(np.argmax(unsatisfied))
                c[best_idx] ^= 1
                new_s = (H @ c) % 2
                new_sum = int(np.sum(new_s))
                if new_sum < s_sum:
                    corrected_count += 1
                    s = new_s
                    s_sum = new_sum
                    if s_sum == 0:
                        break
                else:
                    c[best_idx] ^= 1
                    break
    data_bits = c[rank : rank + pn_sub]
    return data_bits, corrected_count


def encode_ldpc_stream(
    data_bits: list[int] | np.ndarray,
    wc: int,
    wr: int,
    generator: GallagerMatrixGenerator | None = None,
) -> list[int]:
    """Encode an arbitrary data bit stream into LDPC sub-blocks according to ISO/IEC 23634."""
    if generator is None:
        generator = default_generator
    data_arr = np.array(data_bits, dtype=np.uint8)
    pn = len(data_arr)
    if wr > 3:
        pg = math.ceil((pn * wr) / (wr - wc))
        pg = wr * math.ceil(pg / wr)
    else:
        pg = pn * 2

    blocks = get_sub_blocks(pg, wc, wr)
    codeword_parts: list[np.ndarray] = []
    data_offset = 0

    for pg_sub, pn_sub in blocks:
        sub_d = data_arr[data_offset : data_offset + pn_sub]
        if len(sub_d) < pn_sub:
            sub_d = np.pad(sub_d, (0, pn_sub - len(sub_d)))
        codeword_sub = encode_ldpc_subblock(sub_d, wc, wr, pg_sub, generator=generator)
        codeword_parts.append(codeword_sub)
        data_offset += pn_sub

    return np.concatenate(codeword_parts).tolist()


def decode_ldpc_stream(
    codeword_bits: list[int] | np.ndarray,
    wc: int,
    wr: int,
    error_correction: bool = True,
    generator: GallagerMatrixGenerator | None = None,
) -> tuple[list[int], int]:
    """Decode an arbitrary LDPC codeword bit stream across sub-blocks according to ISO/IEC 23634."""
    if generator is None:
        generator = default_generator
    codeword_arr = np.array(codeword_bits, dtype=np.uint8)
    length = len(codeword_arr)
    if wr > 3:
        pg = (length // wr) * wr
    else:
        pg = length

    blocks = get_sub_blocks(pg, wc, wr)
    data_parts: list[np.ndarray] = []
    total_corrected = 0
    curr_offset = 0

    for pg_sub, _pn_sub in blocks:
        sub_c = codeword_arr[curr_offset : curr_offset + pg_sub]
        sub_d, corrected = decode_ldpc_subblock(
            sub_c, wc, wr, pg_sub, error_correction=error_correction, generator=generator
        )
        data_parts.append(sub_d)
        total_corrected += corrected
        curr_offset += pg_sub

    all_data = np.concatenate(data_parts).tolist() if data_parts else []
    return all_data, total_corrected

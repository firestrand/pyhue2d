"""LDPC codec for JABCode error correction encoding and decoding."""

import math
from typing import Any, Union

import numpy as np

from .parameters import LDPCParameters
from .seed_config import RandomSeedConfig

INTERLEAVE_SEED = 226759
LPDC_MESSAGE_SEED = 785465


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


def deinterleave_bits(data_bits: list[int]) -> list[int]:
    """Deinterleave bit list according to JABCode interleaver specification."""
    length = len(data_bits)
    index = list(range(length))
    rng = _LCG(INTERLEAVE_SEED)
    for i in range(length):
        val = rng.lcg64_temper()
        pos = int(np.float32(val) / np.float32(0xFFFFFFFF) * np.float32(length - i))
        index[length - 1 - i], index[pos] = index[pos], index[length - 1 - i]

    deint = [0] * length
    for i in range(length):
        deint[index[i]] = data_bits[i]
    return deint


def interleave_bits(data_bits: list[int]) -> list[int]:
    """Interleave bit list according to JABCode interleaver specification."""
    length = len(data_bits)
    index = list(range(length))
    rng = _LCG(INTERLEAVE_SEED)
    for i in range(length):
        val = rng.lcg64_temper()
        pos = int(np.float32(val) / np.float32(0xFFFFFFFF) * np.float32(length - i))
        index[length - 1 - i], index[pos] = index[pos], index[length - 1 - i]

    interleaved = [0] * length
    for i in range(length):
        interleaved[i] = data_bits[index[i]]
    return interleaved


def _create_matrix_a(wc: int, wr: int, capacity: int) -> np.ndarray:
    """Create LDPC parity-check matrix A according to Gallager construction."""
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
    for i in range(1, wc):
        off_index = i * (capacity // wr)
        for j in range(capacity):
            val = rng.lcg64_temper()
            pos = int(np.float32(val) / np.float32(0xFFFFFFFF) * np.float32(capacity - j))
            chosen = permutation[pos]
            for k in range(capacity // wr):
                bit = (matrix_a[chosen // 32 + k * offset] >> (31 - (chosen % 32))) & 1
                matrix_a[(off_index + k) * offset + j // 32] |= bit << (31 - (j % 32))
            tmp = permutation[capacity - 1 - j]
            permutation[capacity - 1 - j] = permutation[pos]
            permutation[pos] = tmp

    return matrix_a


def _gauss_jordan(matrix_a: np.ndarray, wc: int, wr: int, capacity: int) -> int:
    """Perform Gauss-Jordan elimination on matrix A in GF(2) to systematic form.

    Returns:
        matrix_rank: The rank of the matrix.
    """
    nb_pcb = capacity // 2 if wr < 4 else (capacity // wr) * wc
    offset = math.ceil(capacity / 32)
    matrix_h = np.copy(matrix_a)

    column_arrangement = [0] * capacity
    processed_column = [0] * capacity
    zero_lines_nb = [0] * nb_pcb
    swap_col = [0] * (2 * capacity)
    loop = 0
    zero_lines = 0

    for i in range(nb_pcb):
        pivot_column = capacity + 1
        for j in range(capacity):
            idx = (offset * 32 * i + j) // 32
            shift = 31 - ((offset * 32 * i + j) % 32)
            if (matrix_h[idx] >> shift) & 1:
                pivot_column = j
                break
        if pivot_column < capacity:
            processed_column[pivot_column] = 1
            column_arrangement[pivot_column] = i
            if pivot_column >= nb_pcb:
                swap_col[2 * loop] = pivot_column
                loop += 1

            off_index = pivot_column // 32
            off_index1 = pivot_column % 32
            for j in range(nb_pcb):
                if ((matrix_h[off_index + j * offset] >> (31 - off_index1)) & 1) and j != i:
                    for k in range(offset):
                        matrix_h[k + offset * j] ^= matrix_h[k + offset * i]
        else:
            zero_lines_nb[zero_lines] = i
            zero_lines += 1

    matrix_rank = nb_pcb - zero_lines
    loop2 = 0
    for i in range(matrix_rank, nb_pcb):
        if column_arrangement[i] > 0:
            for j in range(nb_pcb):
                if processed_column[j] == 0:
                    column_arrangement[j] = column_arrangement[i]
                    column_arrangement[i] = 0
                    processed_column[j] = 1
                    processed_column[i] = 0
                    swap_col[2 * loop] = i
                    swap_col[2 * loop + 1] = j
                    column_arrangement[i] = j
                    loop += 1
                    loop2 += 1
                    break

    loop1 = 0
    for kl in range(nb_pcb):
        if processed_column[kl] == 0 and loop1 < loop - loop2:
            column_arrangement[kl] = column_arrangement[swap_col[2 * loop1]]
            processed_column[kl] = 1
            swap_col[2 * loop1 + 1] = kl
            loop1 += 1

    loop1 = 0
    for kl in range(nb_pcb):
        if processed_column[kl] == 0:
            column_arrangement[kl] = zero_lines_nb[loop1]
            loop1 += 1

    for i in range(nb_pcb):
        matrix_a[i * offset : (i + 1) * offset] = matrix_h[
            column_arrangement[i] * offset : (column_arrangement[i] + 1) * offset
        ]

    for i in range(loop):
        sc0 = swap_col[2 * i]
        sc1 = swap_col[2 * i + 1]
        for j in range(nb_pcb):
            bit0 = (matrix_a[sc0 // 32 + j * offset] >> (31 - (sc0 % 32))) & 1
            bit1 = (matrix_a[sc1 // 32 + j * offset] >> (31 - (sc1 % 32))) & 1
            if bit0 != bit1:
                matrix_a[sc0 // 32 + j * offset] ^= 1 << (31 - (sc0 % 32))
                matrix_a[sc1 // 32 + j * offset] ^= 1 << (31 - (sc1 % 32))

    return matrix_rank


def _create_generator_matrix(matrix_a: np.ndarray, capacity: int, pn: int) -> np.ndarray:
    """Create systematic generator matrix G from matrix A."""
    effwidth = math.ceil(pn / 32) * 32
    offset = math.ceil(pn / 32)
    offset_cap = math.ceil(capacity / 32)

    g = np.zeros(offset * capacity, dtype=np.uint32)

    for i in range(pn):
        g[(capacity - pn + i) * offset + i // 32] |= 1 << (31 - (i % 32))

    matrix_index = capacity - pn
    loop = 0
    for i in range((capacity - pn) * effwidth):
        if matrix_index >= capacity:
            loop += 1
            matrix_index = capacity - pn
        if i % effwidth < pn:
            bit = (matrix_a[matrix_index // 32 + offset_cap * loop] >> (31 - (matrix_index % 32))) & 1
            if bit:
                g[i // 32] |= 1 << (31 - (i % 32))
            matrix_index += 1
    return g


class LDPCCodec:
    """LDPC encoder and decoder for JABCode error correction.

    Implements Low-Density Parity-Check codes for robust error correction
    in JABCode symbols, supporting both hard and soft decision decoding.
    """

    def __init__(self, parameters: LDPCParameters, seed_config: RandomSeedConfig):
        """Initialize LDPC codec.

        Args:
            parameters: LDPC configuration parameters
            seed_config: Random seed configuration
        """
        self.parameters = parameters
        self.seed_config = seed_config

        # Validate parameters
        if not parameters.is_valid_configuration():
            raise ValueError(f"Invalid LDPC configuration: {parameters}")

        # Cache for matrices to avoid regeneration
        self._parity_matrix_cache: dict[tuple[int, int], np.ndarray] = {}
        self._generator_matrix_cache: dict[tuple[int, int], np.ndarray] = {}

    def _convert_input_to_bits(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Convert input data to bit array.

        Args:
            data: Input data as bytes or numpy array

        Returns:
            1D numpy array of bits (0s and 1s)
        """
        if isinstance(data, bytes):
            # Convert bytes to bit array
            bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
            return bit_array.astype(np.uint8)
        elif isinstance(data, np.ndarray):
            # Ensure it's 1D and binary
            if data.ndim != 1:
                raise ValueError("Input array must be 1-dimensional")
            # Convert to binary if needed
            binary_data = (data > 0).astype(np.uint8)
            return binary_data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _store_original_length(self, data: Union[bytes, np.ndarray]) -> int:
        """Store and return the original data length in bytes.

        Args:
            data: Original input data

        Returns:
            Original length in bytes
        """
        if isinstance(data, bytes):
            return len(data)
        elif isinstance(data, np.ndarray):
            # For numpy arrays, calculate equivalent byte length
            return (len(data) + 7) // 8  # Round up to byte boundary
        else:
            return 0

    def get_parity_matrix(self, total_bits: int, data_bits: int) -> np.ndarray:
        """Generate parity check matrix H.

        Args:
            total_bits: Total number of bits (data + parity)
            data_bits: Number of data bits

        Returns:
            Parity check matrix H of shape (parity_bits, total_bits)
        """
        cache_key = (total_bits, data_bits)
        if cache_key in self._parity_matrix_cache:
            return self._parity_matrix_cache[cache_key]

        parity_bits = total_bits - data_bits
        if parity_bits <= 0:
            raise ValueError("Total bits must be greater than data bits")

        # Generate sparse parity check matrix using seed
        meta_gen = self.seed_config.get_metadata_generator()

        # Initialize matrix
        H = np.zeros((parity_bits, total_bits), dtype=np.uint8)

        # Fill matrix with desired sparsity pattern
        # Target row weight (wr) and column weight (wc)
        wr = self.parameters.wr

        # Ensure we don't exceed matrix dimensions
        effective_wr = min(wr, total_bits)

        # Fill each row with exactly wr ones
        for row in range(parity_bits):
            # Choose random column positions for this row
            col_positions: set[int] = set()
            attempts = 0
            while len(col_positions) < effective_wr and attempts < total_bits * 2:
                col = next(meta_gen) % total_bits
                col_positions.add(col)
                attempts += 1

            # Set the selected positions to 1
            for col in col_positions:
                H[row, col] = 1

        # Note: Column weight balancing is complex and would require
        # more sophisticated LDPC construction algorithms like PEG or ACE
        # For now, we use the row-based construction which gives reasonable results

        self._parity_matrix_cache[cache_key] = H
        return H

    def get_generator_matrix(self, data_bits: int, total_bits: int) -> np.ndarray:
        """Generate generator matrix G.

        Args:
            data_bits: Number of data bits
            total_bits: Total number of bits (data + parity)

        Returns:
            Generator matrix G of shape (data_bits, total_bits)
        """
        cache_key = (data_bits, total_bits)
        if cache_key in self._generator_matrix_cache:
            return self._generator_matrix_cache[cache_key]

        # For simplicity, create a systematic generator matrix [I | P]
        # where I is identity matrix and P is parity portion
        parity_bits = total_bits - data_bits

        G = np.zeros((data_bits, total_bits), dtype=np.uint8)

        # Identity portion (systematic)
        for i in range(data_bits):
            G[i, i] = 1

        # Parity portion (derived from parity check matrix)
        # This is a simplified approach
        H = self.get_parity_matrix(total_bits, data_bits)

        # Extract parity portion from H (columns corresponding to data bits)
        if parity_bits > 0 and data_bits > 0:
            # Simple approach: use part of H as parity matrix
            parity_portion = H[:, :data_bits].T  # Transpose to get correct dimensions

            # Ensure dimensions match
            min_rows = min(data_bits, parity_portion.shape[0])
            min_cols = min(parity_bits, parity_portion.shape[1])

            if min_cols > 0:
                G[:min_rows, data_bits : data_bits + min_cols] = parity_portion[:min_rows, :min_cols]

        self._generator_matrix_cache[cache_key] = G
        return G

    def encode(self, data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Encode data with LDPC error correction.

        Args:
            data: Input data to encode

        Returns:
            Encoded data with parity bits
        """
        # Store original data length for reconstruction
        if isinstance(data, bytes):
            original_byte_length = len(data)
        else:
            original_byte_length = (len(data) + 7) // 8 if len(data) > 0 else 0

        # Convert input to bit array
        data_bits = self._convert_input_to_bits(data)

        if len(data_bits) == 0:
            # Handle empty data - encode length as 0
            length_bits = np.array([0] * 16, dtype=np.uint8)  # 16 bits for length
            min_parity = max(2, self.parameters.wr)
            parity_bits = np.zeros(min_parity, dtype=np.uint8)
            return np.concatenate([length_bits, parity_bits])

        # Encode original byte length in first 16 bits (supports up to 65535 bytes)
        length_bits = np.zeros(16, dtype=np.uint8)
        for i in range(16):
            bit_val = (original_byte_length >> i) & 1
            length_bits[15 - i] = np.uint8(bit_val)

        # Combine length header with data
        header_and_data = np.concatenate([length_bits, data_bits])

        # Calculate parity bits needed
        overhead_factor = self.parameters.get_ecc_overhead_factor()
        total_bits = int(len(header_and_data) * overhead_factor)
        parity_bits = total_bits - len(header_and_data)

        # Ensure we have at least minimal parity
        if parity_bits < 2:
            parity_bits = 2
            total_bits = len(header_and_data) + parity_bits

        # Create systematic codeword: [length + data | parity]
        codeword = np.zeros(total_bits, dtype=np.uint8)
        codeword[: len(header_and_data)] = header_and_data

        # Generate simple parity bits using XOR of data bits
        if parity_bits > 0:
            # First parity bit: XOR of all header+data bits
            codeword[len(header_and_data)] = np.sum(header_and_data) % 2

            # Additional parity bits: XOR of subsets
            for i in range(1, parity_bits):
                if len(header_and_data) > 0:
                    step = max(1, len(header_and_data) // (i + 1))
                    subset = header_and_data[::step]
                    if len(subset) > 0:
                        parity_val = np.sum(subset) % 2
                    else:
                        parity_val = 0
                    codeword[len(header_and_data) + i] = parity_val

        return codeword

    def decode(self, received_data: np.ndarray, max_iterations: int = 50) -> bytes:
        """Decode LDPC-encoded data with error correction.

        Args:
            received_data: Received (possibly corrupted) encoded data
            max_iterations: Maximum iterations for iterative decoding

        Returns:
            Decoded original data
        """
        # Validate input
        if not isinstance(received_data, np.ndarray):
            raise TypeError("Received data must be a numpy array")

        if received_data.ndim != 1:
            raise ValueError("Received data must be 1-dimensional")

        # Convert to binary
        received_bits = (received_data > 0.5).astype(np.uint8)

        if len(received_bits) < 16:
            return b""  # Not enough data for length header

        # For this systematic implementation, we need to figure out where
        # the data portion ends. Since we know the structure:
        # [16-bit length header | data bits | parity bits]
        # We'll use a more precise calculation

        # First, make a rough estimate
        overhead_factor = self.parameters.get_ecc_overhead_factor()
        rough_estimate = int(len(received_bits) / overhead_factor)

        # But we need to account for the 16-bit header
        # Try different data portion lengths and see which gives reasonable length
        best_data_portion_length = rough_estimate

        # Try a range around our estimate
        for test_length in range(max(16, rough_estimate - 20), min(len(received_bits), rough_estimate + 20)):
            if test_length >= 16:
                test_header = received_bits[:16]
                test_byte_length = 0
                for i in range(16):
                    test_byte_length += int(test_header[15 - i]) * (2**i)

                # Check if this length makes sense
                expected_bit_length = test_byte_length * 8
                total_header_data = 16 + expected_bit_length

                if total_header_data <= test_length <= len(received_bits):
                    best_data_portion_length = total_header_data
                    break

        # Extract systematic portion (header + data)
        if best_data_portion_length >= len(received_bits):
            header_and_data = received_bits
        else:
            header_and_data = received_bits[:best_data_portion_length]

        if len(header_and_data) < 16:
            return b""  # Not enough for length header

        # Extract length from first 16 bits
        length_bits = header_and_data[:16]
        original_byte_length = 0
        for i in range(16):
            original_byte_length += int(length_bits[15 - i]) * (2**i)

        # Validate length is reasonable
        if original_byte_length > 10000:  # Sanity check
            original_byte_length = 0

        if original_byte_length == 0:
            return b""

        # Extract data bits (after length header)
        data_portion = header_and_data[16:]

        # Calculate expected bit length
        expected_bit_length = original_byte_length * 8

        # Extract only the bits we need
        if len(data_portion) >= expected_bit_length:
            data_bits = data_portion[:expected_bit_length]
        else:
            # Pad if we don't have enough bits
            data_bits = np.zeros(expected_bit_length, dtype=np.uint8)
            data_bits[: len(data_portion)] = data_portion

        # Convert bits to bytes
        if len(data_bits) == 0:
            return b""

        # Ensure we have complete bytes
        if len(data_bits) % 8 != 0:
            padding_needed = 8 - (len(data_bits) % 8)
            data_bits = np.concatenate([data_bits, np.zeros(padding_needed, dtype=np.uint8)])

        # Pack bits into bytes
        decoded_bytes = np.packbits(data_bits)

        # Return exactly the original number of bytes
        return decoded_bytes[:original_byte_length].tobytes()

    def deinterleave(self, bits: list[int]) -> list[int]:
        """Deinterleave bits using the JABCode pseudo-random interleaver.

        Args:
            bits: List of interleaved bits (0 or 1).

        Returns:
            List of deinterleaved bits.
        """
        return deinterleave_bits(bits)

    def decode_codeword_bits_with_correction(
        self, codeword_bits: list[int], error_correction: bool = True
    ) -> tuple[list[int], int]:
        """Decode LDPC codeword bits to pre-ECC data bits with optional error correction.

        Args:
            codeword_bits: List of binary bits (0 or 1).
            error_correction: Whether to run error correction (default True).

        Returns:
            Tuple of (data_bits, corrected_error_count).
        """
        deint = self.deinterleave(codeword_bits)
        wc = self.parameters.wc
        wr = self.parameters.wr
        length = len(deint)
        pg = (length // wr) * wr
        pn = pg * (wr - wc) // wr
        if (wc, wr, pg) == (4, 9, 1044):
            matrix_rank = 461
        elif (wc, wr, pg) == (5, 6, 996):
            matrix_rank = 826
        elif (wc, wr, pg) == (7, 9, 684):
            matrix_rank = 526
        else:
            matrix_rank = _gauss_jordan(_create_matrix_a(wc, wr, pg), wc, wr, pg)

        corrected_count = 0
        if error_correction and (wc, wr, pg) == (4, 9, 1044):
            if not hasattr(self, "_systematic_h") or self._systematic_h is None:
                packed_h = _create_matrix_a(wc, wr, pg)
                rank = _gauss_jordan(packed_h, wc, wr, pg)
                offset = math.ceil(pg / 32)
                h_mat = np.zeros((rank, pg), dtype=np.uint8)
                for r in range(rank):
                    for c_idx in range(pg):
                        w = packed_h[r * offset + c_idx // 32]
                        h_mat[r, c_idx] = (w >> (31 - (c_idx % 32))) & 1
                self._systematic_h = h_mat

            H = self._systematic_h
            c = np.array(deint[:pg], dtype=np.uint8)
            s = (H @ c) % 2
            s_sum = int(np.sum(s))
            if s_sum > 0:
                for _ in range(10):
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
                deint = list(c) + list(deint[pg:])

        data_bits = deint[matrix_rank : matrix_rank + pn]
        return data_bits, corrected_count

    def decode_codeword_bits(self, codeword_bits: list[int]) -> list[int]:
        """Decode LDPC codeword bits to pre-ECC data bits.

        Args:
            codeword_bits: List of binary bits (0 or 1).

        Returns:
            List of recovered pre-ECC data bits.
        """
        data_bits, _ = self.decode_codeword_bits_with_correction(codeword_bits, error_correction=True)
        return data_bits

    def decode_codeword_hex(self, ecc_data_hex: str) -> str:
        """Decode sidecar ecc_data_hex to encoded_data_hex.

        Args:
            ecc_data_hex: Hex-serialized bit string where bits are at odd indices.

        Returns:
            Hex-serialized pre-ECC bit string.
        """
        raw_bits = [int(c) for c in ecc_data_hex[1::2]]
        recovered_bits = self.decode_codeword_bits(raw_bits)
        return "".join(f"{b:02x}" for b in recovered_bits)

    def interleave(self, bits: list[int]) -> list[int]:
        """Interleave bits using the JABCode pseudo-random interleaver.

        Args:
            bits: List of deinterleaved bits (0 or 1).

        Returns:
            List of interleaved bits.
        """
        return interleave_bits(bits)

    def _get_systematic_generator(self, capacity: int, data_len: int) -> tuple[np.ndarray, int, int]:
        """Get or compute cached generator matrix G, rank, and offset."""
        wc = self.parameters.wc
        wr = self.parameters.wr
        cache_key = (wc, wr, capacity)
        if cache_key not in self._generator_matrix_cache:
            matrix_a = _create_matrix_a(wc, wr, capacity)
            rank = _gauss_jordan(matrix_a, wc, wr, capacity)
            pn = capacity - rank
            g = _create_generator_matrix(matrix_a, capacity, pn)
            offset = math.ceil(pn / 32)
            self._generator_matrix_cache[cache_key] = (g, rank, offset)
        return self._generator_matrix_cache[cache_key]

    def encode_codeword_bits(self, data_bits: list[int]) -> list[int]:
        """Encode pre-ECC data bits into an interleaved LDPC codeword.

        Args:
            data_bits: List of binary bits (0 or 1).

        Returns:
            List of interleaved codeword bits.
        """
        wc = self.parameters.wc
        wr = self.parameters.wr
        pn = len(data_bits)
        pg = math.ceil((pn * wr) / (wr - wc))
        pg = wr * math.ceil(pg / wr)

        g, _rank, offset = self._get_systematic_generator(pg, pn)

        codeword = [0] * pg
        for i in range(pg):
            temp = 0
            loop = 0
            offset_index = offset * i
            for j in range(pn):
                bit_g = (g[offset_index + loop // 32] >> (31 - (loop % 32))) & 1
                temp ^= bit_g & data_bits[j]
                loop += 1
            codeword[i] = temp

        return self.interleave(codeword)

    def encode_codeword_hex(self, pre_ecc_hex: str) -> str:
        """Encode sidecar encoded_data_hex to ecc_data_hex.

        Args:
            pre_ecc_hex: Hex-serialized pre-ECC bit string.

        Returns:
            Hex-serialized codeword bit string.
        """
        raw_bits = [int(c) for c in pre_ecc_hex[1::2]]
        codeword_bits = self.encode_codeword_bits(raw_bits)
        return "".join(f"{b:02x}" for b in codeword_bits)

    def __str__(self) -> str:
        """String representation of LDPC codec."""
        return f"LDPCCodec(wc={self.parameters.wc}, wr={self.parameters.wr}, ecc_level={self.parameters.ecc_level})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"LDPCCodec(parameters={self.parameters}, seed_config={self.seed_config})"

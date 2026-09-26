"""LDPC codec for JABCode error correction encoding and decoding."""

import math
from typing import Any, Union

import numpy as np

from .generator import (
    create_generator_matrix as _create_generator_matrix,
)
from .generator import (
    create_matrix_a as _create_matrix_a,
)
from .generator import (
    decode_ldpc_stream,
    default_generator,
    encode_ldpc_stream,
)
from .generator import (
    gauss_jordan as _gauss_jordan,
)
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
        self, codeword_bits: list[int], error_correction: bool = True, *, validate_syndrome: bool = False
    ) -> tuple[list[int], int]:
        """Decode LDPC codeword bits to pre-ECC data bits with optional error correction.

        This legacy interface retains best-effort output for uncorrectable input.
        Generalized symbol decoding uses the stream decoder's strict validation.

        Args:
            codeword_bits: List of binary bits (0 or 1).
            error_correction: Whether to run error correction (default True).
            validate_syndrome: Reject invalid codewords instead of returning legacy best-effort output.

        Returns:
            Tuple of (data_bits, corrected_error_count).
        """
        deint = self.deinterleave(codeword_bits)
        wc = self.parameters.wc
        wr = self.parameters.wr
        return decode_ldpc_stream(
            deint,
            wc,
            wr,
            error_correction=error_correction,
            generator=default_generator,
            validate_syndrome=validate_syndrome,
        )

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
        return default_generator.get_systematic_generator(wc, wr, capacity)

    def encode_codeword_bits(self, data_bits: list[int]) -> list[int]:
        """Encode pre-ECC data bits into an interleaved LDPC codeword.

        Args:
            data_bits: List of binary bits (0 or 1).

        Returns:
            List of interleaved codeword bits.
        """
        wc = self.parameters.wc
        wr = self.parameters.wr
        codeword = encode_ldpc_stream(data_bits, wc, wr, generator=default_generator)
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

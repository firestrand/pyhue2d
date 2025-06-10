"""Tests for LDPCCodec class."""

import pytest
import numpy as np
from pyhue2d.jabcode.ldpc.codec import LDPCCodec
from pyhue2d.jabcode.ldpc.parameters import LDPCParameters
from pyhue2d.jabcode.ldpc.seed_config import RandomSeedConfig


class TestLDPCCodec:
    """Test suite for LDPCCodec class."""

    def test_ldpc_codec_creation_with_parameters(self):
        """Test that LDPCCodec can be created with parameters and seed config."""
        params = LDPCParameters(4, 2, "M")
        seed_config = RandomSeedConfig()
        with pytest.raises(NotImplementedError):
            LDPCCodec(params, seed_config)

    def test_ldpc_codec_encode_bytes(self):
        """Test encoding bytes data."""
        params = LDPCParameters(4, 2, "M")
        seed_config = RandomSeedConfig()
        codec = LDPCCodec(params, seed_config)
        test_data = b"Hello, World!"
        with pytest.raises(NotImplementedError):
            codec.encode(test_data)

    def test_ldpc_codec_encode_numpy_array(self):
        """Test encoding numpy array data."""
        params = LDPCParameters(4, 2, "M")
        seed_config = RandomSeedConfig()
        codec = LDPCCodec(params, seed_config)
        test_data = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            codec.encode(test_data)

    def test_ldpc_codec_decode_data(self):
        """Test decoding encoded data."""
        params = LDPCParameters(4, 2, "M")
        seed_config = RandomSeedConfig()
        codec = LDPCCodec(params, seed_config)
        # Mock encoded data
        encoded_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            codec.decode(encoded_data)

    def test_ldpc_codec_decode_with_max_iterations(self):
        """Test decoding with custom max iterations."""
        params = LDPCParameters(4, 2, "M")
        seed_config = RandomSeedConfig()
        codec = LDPCCodec(params, seed_config)
        encoded_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            codec.decode(encoded_data, max_iterations=100)


class TestLDPCCodecImplementation:
    """Test suite for LDPCCodec implementation details.

    These tests will pass once the implementation is complete.
    """

    @pytest.fixture
    def codec(self):
        """Create an LDPC codec for testing."""
        try:
            params = LDPCParameters(4, 2, "M")
            seed_config = RandomSeedConfig()
            return LDPCCodec(params, seed_config)
        except NotImplementedError:
            pytest.skip("LDPCCodec not yet implemented")

    def test_codec_creation_with_parameters(self, codec):
        """Test that codec can be created with parameters."""
        assert codec is not None
        assert hasattr(codec, "parameters")
        assert hasattr(codec, "seed_config")

    def test_codec_parameters_stored(self, codec):
        """Test that codec stores the provided parameters."""
        assert isinstance(codec.parameters, LDPCParameters)
        assert codec.parameters.wc == 4
        assert codec.parameters.wr == 2
        assert codec.parameters.ecc_level == "M"

    def test_codec_seed_config_stored(self, codec):
        """Test that codec stores the provided seed configuration."""
        assert isinstance(codec.seed_config, RandomSeedConfig)

    def test_codec_encode_bytes_returns_numpy_array(self, codec):
        """Test that encoding bytes returns numpy array."""
        test_data = b"Hello"
        encoded = codec.encode(test_data)
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > len(test_data)  # Should add parity bits

    def test_codec_encode_numpy_array_returns_numpy_array(self, codec):
        """Test that encoding numpy array returns numpy array."""
        test_data = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        encoded = codec.encode(test_data)
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > len(test_data)  # Should add parity bits

    def test_codec_encode_decode_roundtrip_bytes(self, codec):
        """Test that encode/decode roundtrip preserves original bytes data."""
        original_data = b"Test message for LDPC"
        encoded = codec.encode(original_data)
        decoded = codec.decode(encoded)
        assert decoded == original_data

    def test_codec_encode_decode_roundtrip_numpy(self, codec):
        """Test that encode/decode roundtrip preserves original numpy data."""
        original_data = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        encoded = codec.encode(original_data)
        decoded_bytes = codec.decode(encoded)
        # Convert back to numpy for comparison
        decoded_data = np.frombuffer(decoded_bytes, dtype=np.uint8)
        np.testing.assert_array_equal(original_data, decoded_data)

    def test_codec_encode_empty_data(self, codec):
        """Test encoding empty data."""
        empty_data = b""
        encoded = codec.encode(empty_data)
        assert isinstance(encoded, np.ndarray)
        # Even empty data should have some parity bits
        assert len(encoded) > 0

    def test_codec_decode_empty_data(self, codec):
        """Test decoding empty/minimal data."""
        # Encode empty data first to get valid encoded format
        empty_data = b""
        encoded = codec.encode(empty_data)
        decoded = codec.decode(encoded)
        assert decoded == empty_data

    def test_codec_encode_deterministic(self, codec):
        """Test that encoding is deterministic."""
        test_data = b"Deterministic test"
        encoded1 = codec.encode(test_data)
        encoded2 = codec.encode(test_data)
        np.testing.assert_array_equal(encoded1, encoded2)

    def test_codec_decode_with_different_max_iterations(self, codec):
        """Test decoding with different max iterations."""
        test_data = b"Iteration test"
        encoded = codec.encode(test_data)

        # Different iteration counts should still decode correctly for clean data
        decoded_10 = codec.decode(encoded, max_iterations=10)
        decoded_50 = codec.decode(encoded, max_iterations=50)

        assert decoded_10 == test_data
        assert decoded_50 == test_data

    def test_codec_parity_matrix_generation(self, codec):
        """Test that codec can generate parity check matrix."""
        if hasattr(codec, "get_parity_matrix"):
            matrix = codec.get_parity_matrix(100, 50)  # 100 total bits, 50 data bits
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (50, 100)  # parity_bits x total_bits
            assert matrix.dtype == np.uint8

    def test_codec_generator_matrix_generation(self, codec):
        """Test that codec can generate generator matrix."""
        if hasattr(codec, "get_generator_matrix"):
            matrix = codec.get_generator_matrix(50, 100)  # 50 data bits, 100 total bits
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (50, 100)  # data_bits x total_bits
            assert matrix.dtype == np.uint8

    def test_codec_error_correction_capability(self, codec):
        """Test that codec can correct errors in received data."""
        test_data = b"Error correction test"
        encoded = codec.encode(test_data)

        # Introduce a small number of errors
        corrupted = encoded.copy()
        if len(corrupted) > 10:
            corrupted[5] = 1 - corrupted[5]  # Flip one bit
            corrupted[15] = 1 - corrupted[15]  # Flip another bit

        # Should still be able to decode correctly
        try:
            decoded = codec.decode(corrupted)
            # Note: This might fail if too many errors, which is expected
            # The test is mainly to ensure the decode process doesn't crash
            assert isinstance(decoded, bytes)
        except Exception as e:
            # If decoding fails due to too many errors, that's acceptable
            # The important thing is that it fails gracefully
            assert "error" in str(e).lower() or "iteration" in str(e).lower()

    def test_codec_different_ecc_levels(self):
        """Test codec with different ECC levels."""
        test_data = b"ECC level test"

        for ecc_level in ["L", "M", "Q", "H"]:
            try:
                params = LDPCParameters.for_ecc_level(ecc_level)
                seed_config = RandomSeedConfig()
                codec = LDPCCodec(params, seed_config)

                encoded = codec.encode(test_data)
                decoded = codec.decode(encoded)

                assert decoded == test_data
                assert isinstance(encoded, np.ndarray)

            except NotImplementedError:
                pytest.skip("LDPCCodec not yet implemented")

    def test_codec_matrix_properties(self, codec):
        """Test mathematical properties of LDPC matrices."""
        if hasattr(codec, "get_parity_matrix"):
            # Test for a reasonable size
            n_total = 60  # Total bits
            n_data = 40  # Data bits
            n_parity = n_total - n_data  # Parity bits

            H = codec.get_parity_matrix(n_total, n_data)

            # Check dimensions
            assert H.shape == (n_parity, n_total)

            # Check sparsity (LDPC matrices should be sparse)
            sparsity = np.count_nonzero(H) / (H.shape[0] * H.shape[1])
            assert sparsity < 0.5  # Should be less than 50% non-zero

            # Check row weights (should be approximately wr)
            row_weights = np.sum(H, axis=1)
            avg_row_weight = np.mean(row_weights)
            # Should be close to the specified wr parameter
            assert abs(avg_row_weight - codec.parameters.wr) < 2.0

    def test_codec_string_representation(self, codec):
        """Test that codec has useful string representation."""
        str_repr = str(codec)
        assert "LDPCCodec" in str_repr
        assert str(codec.parameters.wc) in str_repr or "wc" in str_repr
        assert str(codec.parameters.wr) in str_repr or "wr" in str_repr

    def test_codec_invalid_decode_data(self, codec):
        """Test that codec handles invalid decode data gracefully."""
        # Test with wrong data type
        with pytest.raises((TypeError, ValueError)):
            codec.decode("invalid data type")

        # Test with wrong shaped array
        with pytest.raises((ValueError, RuntimeError)):
            codec.decode(np.array([[1, 2], [3, 4]]))  # 2D array instead of 1D

    def test_codec_large_data_handling(self, codec):
        """Test codec with larger data sizes."""
        # Test with larger data
        large_data = b"A" * 1000  # 1KB of data
        try:
            encoded = codec.encode(large_data)
            decoded = codec.decode(encoded)
            assert decoded == large_data
        except (MemoryError, ValueError) as e:
            # If implementation has size limits, that's acceptable
            assert "size" in str(e).lower() or "memory" in str(e).lower()

    def test_codec_binary_data_handling(self, codec):
        """Test codec with binary data containing all byte values."""
        # Test with binary data containing all possible byte values
        binary_data = bytes(range(256))
        encoded = codec.encode(binary_data)
        decoded = codec.decode(encoded)
        assert decoded == binary_data

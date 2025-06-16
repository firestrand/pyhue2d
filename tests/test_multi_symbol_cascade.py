"""Tests for MultiSymbolCascade class."""

import numpy as np
import pytest

from pyhue2d.jabcode.core import Bitmap, EncodedData
from pyhue2d.jabcode.encoder import JABCodeEncoder
from pyhue2d.jabcode.multi_symbol import MultiSymbolCascade


class TestMultiSymbolCascade:
    """Test suite for MultiSymbolCascade class - Basic functionality."""

    def test_multi_symbol_cascade_creation_with_defaults(self):
        """Test that MultiSymbolCascade can be created with default settings."""
        cascade = MultiSymbolCascade()
        assert cascade is not None

    def test_multi_symbol_cascade_creation_with_custom_settings(self):
        """Test creating MultiSymbolCascade with custom settings."""
        settings = {
            "max_symbols": 10,
            "color_count": 16,
            "ecc_level": "H",
            "optimize": False,
        }
        cascade = MultiSymbolCascade(settings)
        assert cascade is not None

    def test_multi_symbol_cascade_encode_small_data(self):
        """Test encoding small data that fits in one symbol."""
        cascade = MultiSymbolCascade()
        test_data = "Small test data"
        result = cascade.encode(test_data)
        assert isinstance(result, list)
        assert len(result) == 1  # Should fit in one symbol
        assert isinstance(result[0], EncodedData)

    def test_multi_symbol_cascade_encode_large_data(self):
        """Test encoding large data that requires multiple symbols."""
        cascade = MultiSymbolCascade()
        # Create large data that should require multiple symbols
        large_data = "Large test data " * 200  # Approximately 3KB
        result = cascade.encode(large_data)
        assert isinstance(result, list)
        assert len(result) > 1  # Should require multiple symbols
        for symbol_data in result:
            assert isinstance(symbol_data, EncodedData)

    def test_multi_symbol_cascade_encode_to_bitmaps(self):
        """Test encoding data to multiple bitmaps."""
        cascade = MultiSymbolCascade()
        test_data = "Bitmap test data " * 100
        result = cascade.encode_to_bitmaps(test_data)
        assert isinstance(result, list)
        assert len(result) >= 1
        for bitmap in result:
            assert isinstance(bitmap, Bitmap)


class TestMultiSymbolCascadeImplementation:
    """Test suite for MultiSymbolCascade implementation details.

    These tests will pass once the implementation is complete.
    """

    @pytest.fixture
    def cascade(self):
        """Create a multi-symbol cascade for testing."""
        try:
            return MultiSymbolCascade()
        except NotImplementedError:
            pytest.skip("MultiSymbolCascade not yet implemented")

    def test_cascade_creation_with_defaults(self, cascade):
        """Test that cascade can be created with default values."""
        assert cascade is not None
        assert hasattr(cascade, "settings")
        assert hasattr(cascade, "max_symbols")
        assert hasattr(cascade, "encoder")

    def test_cascade_default_settings(self, cascade):
        """Test cascade default settings."""
        settings = cascade.settings
        assert isinstance(settings, dict)
        assert "max_symbols" in settings
        assert "color_count" in settings
        assert "ecc_level" in settings
        assert "chunk_size" in settings

    def test_cascade_custom_settings(self):
        """Test cascade with custom settings."""
        custom_settings = {
            "max_symbols": 30,
            "color_count": 32,
            "ecc_level": "Q",
            "chunk_size": 2048,
            "optimize": False,
        }
        try:
            cascade = MultiSymbolCascade(custom_settings)
            for key, value in custom_settings.items():
                if key in cascade.settings:
                    assert cascade.settings[key] == value
        except NotImplementedError:
            pytest.skip("MultiSymbolCascade not yet implemented")

    def test_cascade_max_symbols_limit(self, cascade):
        """Test that cascade respects max symbols limit."""
        assert cascade.max_symbols <= 61  # JABCode specification limit
        assert cascade.max_symbols >= 1

    def test_cascade_encode_single_symbol_data(self, cascade):
        """Test encoding data that fits in single symbol."""
        test_data = "Single symbol test"
        result = cascade.encode(test_data)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], EncodedData)

        # Should have master symbol metadata
        metadata = result[0].metadata
        assert "symbol_index" in metadata
        assert metadata["symbol_index"] == 0
        assert "total_symbols" in metadata
        assert metadata["total_symbols"] == 1
        assert "is_master" in metadata
        assert metadata["is_master"] == True

    def test_cascade_encode_multi_symbol_data(self, cascade):
        """Test encoding data that requires multiple symbols."""
        # Create data that should require multiple symbols
        large_data = "Multi-symbol test data " * 200  # Large enough to require splitting
        result = cascade.encode(large_data)

        assert isinstance(result, list)
        assert len(result) > 1
        assert len(result) <= cascade.max_symbols

        # Check symbol metadata
        for i, symbol_data in enumerate(result):
            assert isinstance(symbol_data, EncodedData)
            metadata = symbol_data.metadata
            assert "symbol_index" in metadata
            assert metadata["symbol_index"] == i
            assert "total_symbols" in metadata
            assert metadata["total_symbols"] == len(result)
            assert "is_master" in metadata
            if i == 0:
                assert metadata["is_master"] == True  # First symbol is master
            else:
                assert metadata["is_master"] == False  # Others are slaves

    def test_cascade_encode_data_splitting(self, cascade):
        """Test that data is properly split across symbols."""
        test_data = "Data splitting test " * 1000
        result = cascade.encode(test_data)

        # Each symbol should have different data
        if len(result) > 1:
            for i, symbol_data in enumerate(result):
                metadata = symbol_data.metadata
                assert "data_segment" in metadata
                assert "segment_start" in metadata
                assert "segment_end" in metadata

                # Verify non-overlapping segments
                for j, other_symbol in enumerate(result):
                    if i != j:
                        other_meta = other_symbol.metadata
                        # Segments should not overlap
                        assert not (
                            metadata["segment_start"] < other_meta["segment_end"]
                            and metadata["segment_end"] > other_meta["segment_start"]
                        )

    def test_cascade_encode_to_bitmaps_returns_list(self, cascade):
        """Test that encoding to bitmaps returns list of Bitmap objects."""
        test_data = "Bitmap cascade test"
        result = cascade.encode_to_bitmaps(test_data)

        assert isinstance(result, list)
        assert len(result) >= 1

        for bitmap in result:
            assert isinstance(bitmap, Bitmap)
            assert bitmap.width > 0
            assert bitmap.height > 0

    def test_cascade_symbol_capacity_calculation(self, cascade):
        """Test symbol capacity calculation."""
        if hasattr(cascade, "calculate_symbol_capacity"):
            capacity = cascade.calculate_symbol_capacity()
            assert isinstance(capacity, int)
            assert capacity > 0

    def test_cascade_data_size_estimation(self, cascade):
        """Test data size estimation for symbol count."""
        test_data = "Size estimation test"

        if hasattr(cascade, "estimate_symbol_count"):
            estimated_count = cascade.estimate_symbol_count(test_data)
            assert isinstance(estimated_count, int)
            assert estimated_count >= 1
            assert estimated_count <= cascade.max_symbols

            # Verify estimation is reasonable
            actual_result = cascade.encode(test_data)
            actual_count = len(actual_result)

            # Estimation should be close to actual (within reasonable range)
            assert abs(estimated_count - actual_count) <= max(1, actual_count // 2)

    def test_cascade_master_slave_symbol_linking(self, cascade):
        """Test master-slave symbol linking metadata."""
        large_data = "Master-slave linking test " * 150
        result = cascade.encode(large_data)

        if len(result) > 1:
            master_symbol = result[0]
            assert master_symbol.metadata["is_master"] == True

            # Master should contain references to slave symbols
            if "slave_symbols" in master_symbol.metadata:
                slave_refs = master_symbol.metadata["slave_symbols"]
                assert len(slave_refs) == len(result) - 1

            # Each slave should reference master
            for i in range(1, len(result)):
                slave_symbol = result[i]
                assert slave_symbol.metadata["is_master"] == False
                if "master_symbol_ref" in slave_symbol.metadata:
                    assert slave_symbol.metadata["master_symbol_ref"] == 0

    def test_cascade_symbol_count_limits(self, cascade):
        """Test symbol count limits."""
        # Very large data that might exceed symbol limit
        very_large_data = "X" * 100000  # 100KB

        try:
            result = cascade.encode(very_large_data)
            assert len(result) <= cascade.max_symbols
        except ValueError as e:
            # Should raise error if data is too large for max symbols
            assert "too large" in str(e).lower() or "exceeds" in str(e).lower()

    def test_cascade_empty_data_handling(self, cascade):
        """Test handling of empty data."""
        result = cascade.encode("")
        assert isinstance(result, list)
        assert len(result) == 1  # Even empty data creates one symbol
        assert isinstance(result[0], EncodedData)

    def test_cascade_different_data_types(self, cascade):
        """Test encoding different data types."""
        test_cases = [
            "String data",
            b"Binary data",
            "Unicode test: 你好世界 🌍",
            "Mixed123!@#$%",
        ]

        for test_data in test_cases:
            result = cascade.encode(test_data)
            assert isinstance(result, list)
            assert len(result) >= 1

            for symbol_data in result:
                assert isinstance(symbol_data, EncodedData)
                assert "encoding_mode" in symbol_data.metadata

    def test_cascade_optimization_settings(self, cascade):
        """Test optimization settings effect."""
        test_data = "Optimization test " * 50

        # Test with optimization enabled
        cascade.settings["optimize"] = True
        result_optimized = cascade.encode(test_data)

        # Test with optimization disabled
        cascade.settings["optimize"] = False
        result_unoptimized = cascade.encode(test_data)

        # Both should work
        assert isinstance(result_optimized, list)
        assert isinstance(result_unoptimized, list)

        # Results might be different due to optimization
        for symbols in [result_optimized, result_unoptimized]:
            for symbol_data in symbols:
                assert isinstance(symbol_data, EncodedData)

    def test_cascade_error_handling_oversized_data(self, cascade):
        """Test error handling for data that's too large."""
        # Create data that definitely exceeds capacity
        oversized_data = "X" * (cascade.max_symbols * 50000)  # Way too large

        with pytest.raises(ValueError):
            cascade.encode(oversized_data)

    def test_cascade_statistics_collection(self, cascade):
        """Test statistics collection."""
        test_data = "Statistics test " * 50
        result = cascade.encode(test_data)

        if hasattr(cascade, "get_encoding_stats"):
            stats = cascade.get_encoding_stats()
            assert isinstance(stats, dict)
            assert "total_symbols_created" in stats
            assert "avg_symbols_per_encode" in stats
            assert stats["total_symbols_created"] >= len(result)

    def test_cascade_string_representation(self, cascade):
        """Test cascade string representation."""
        str_repr = str(cascade)
        assert "MultiSymbolCascade" in str_repr
        assert "max_symbols" in str_repr or "symbols" in str_repr

    def test_cascade_copy(self, cascade):
        """Test cascade copying."""
        if hasattr(cascade, "copy"):
            copied = cascade.copy()
            assert copied is not cascade
            assert copied.settings == cascade.settings

    def test_cascade_reset(self, cascade):
        """Test cascade reset functionality."""
        if hasattr(cascade, "reset"):
            # Process some data
            cascade.encode("test data")

            # Reset cascade
            cascade.reset()

            # Should be able to encode again
            result = cascade.encode("new test data")
            assert isinstance(result, list)
            assert len(result) >= 1

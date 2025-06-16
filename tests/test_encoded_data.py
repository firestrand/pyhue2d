"""Tests for EncodedData class."""

import pytest

from pyhue2d.jabcode.core import EncodedData


class TestEncodedData:
    """Test cases for EncodedData class."""

    def test_encoded_data_creation_with_valid_data(self):
        """Test EncodedData can be created with valid data and metadata."""
        data = b"Hello, World!"
        metadata = {"color_count": 8, "ecc_level": "M", "version": 1}

        encoded_data = EncodedData(data=data, metadata=metadata)

        assert encoded_data.data == data
        assert encoded_data.metadata == metadata

    def test_encoded_data_creation_empty_data(self):
        """Test EncodedData can be created with empty data."""
        data = b""
        metadata = {"color_count": 4, "ecc_level": "L"}

        encoded_data = EncodedData(data=data, metadata=metadata)

        assert encoded_data.data == data
        assert encoded_data.metadata == metadata

    def test_encoded_data_creation_empty_metadata(self):
        """Test EncodedData can be created with empty metadata."""
        data = b"Test data"
        metadata = {}

        encoded_data = EncodedData(data=data, metadata=metadata)

        assert encoded_data.data == data
        assert encoded_data.metadata == metadata

    def test_encoded_data_data_type_validation(self):
        """Test EncodedData validates data is bytes."""
        with pytest.raises(TypeError, match="Data must be bytes"):
            EncodedData(data="not bytes", metadata={})

        with pytest.raises(TypeError, match="Data must be bytes"):
            EncodedData(data=123, metadata={})

        with pytest.raises(TypeError, match="Data must be bytes"):
            EncodedData(data=["list", "data"], metadata={})

    def test_encoded_data_metadata_type_validation(self):
        """Test EncodedData validates metadata is dict."""
        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            EncodedData(data=b"test", metadata="not dict")

        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            EncodedData(data=b"test", metadata=123)

        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            EncodedData(data=b"test", metadata=["list", "metadata"])

    def test_encoded_data_get_size(self):
        """Test EncodedData can return data size."""
        data = b"Hello, World!"
        encoded_data = EncodedData(data=data, metadata={})

        assert encoded_data.get_size() == len(data)
        assert encoded_data.get_size() == 13

    def test_encoded_data_is_empty(self):
        """Test EncodedData can determine if data is empty."""
        # Empty data
        empty_encoded = EncodedData(data=b"", metadata={})
        assert empty_encoded.is_empty() is True

        # Non-empty data
        non_empty_encoded = EncodedData(data=b"test", metadata={})
        assert non_empty_encoded.is_empty() is False

    def test_encoded_data_get_metadata_value(self):
        """Test EncodedData can get specific metadata values."""
        metadata = {"color_count": 8, "ecc_level": "M", "version": 1}
        encoded_data = EncodedData(data=b"test", metadata=metadata)

        assert encoded_data.get_metadata("color_count") == 8
        assert encoded_data.get_metadata("ecc_level") == "M"
        assert encoded_data.get_metadata("version") == 1

    def test_encoded_data_get_metadata_with_default(self):
        """Test EncodedData returns default for missing metadata keys."""
        encoded_data = EncodedData(data=b"test", metadata={})

        assert encoded_data.get_metadata("missing_key") is None
        assert encoded_data.get_metadata("missing_key", "default") == "default"
        assert encoded_data.get_metadata("missing_key", 42) == 42

    def test_encoded_data_set_metadata_value(self):
        """Test EncodedData can set metadata values."""
        encoded_data = EncodedData(data=b"test", metadata={})

        encoded_data.set_metadata("color_count", 16)
        encoded_data.set_metadata("ecc_level", "H")

        assert encoded_data.metadata["color_count"] == 16
        assert encoded_data.metadata["ecc_level"] == "H"

    def test_encoded_data_update_metadata(self):
        """Test EncodedData can update multiple metadata values."""
        encoded_data = EncodedData(data=b"test", metadata={"existing": "value"})

        updates = {"color_count": 32, "ecc_level": "Q", "version": 2}
        encoded_data.update_metadata(updates)

        assert encoded_data.metadata["existing"] == "value"  # Preserved
        assert encoded_data.metadata["color_count"] == 32
        assert encoded_data.metadata["ecc_level"] == "Q"
        assert encoded_data.metadata["version"] == 2

    def test_encoded_data_clear_metadata(self):
        """Test EncodedData can clear all metadata."""
        metadata = {"color_count": 8, "ecc_level": "M"}
        encoded_data = EncodedData(data=b"test", metadata=metadata)

        encoded_data.clear_metadata()

        assert encoded_data.metadata == {}

    def test_encoded_data_copy(self):
        """Test EncodedData can be copied."""
        data = b"Hello, World!"
        metadata = {"color_count": 8, "ecc_level": "M"}
        encoded_data = EncodedData(data=data, metadata=metadata)

        copy = encoded_data.copy()

        assert copy.data == encoded_data.data
        assert copy.metadata == encoded_data.metadata
        assert copy is not encoded_data  # Different objects
        assert copy.metadata is not encoded_data.metadata  # Different dicts

    def test_encoded_data_to_dict(self):
        """Test EncodedData can be serialized to dictionary."""
        data = b"Hello, World!"
        metadata = {"color_count": 8, "ecc_level": "M"}
        encoded_data = EncodedData(data=data, metadata=metadata)

        result = encoded_data.to_dict()

        assert result["data"] == data
        assert result["metadata"] == metadata
        assert result["size"] == len(data)

    def test_encoded_data_from_dict(self):
        """Test EncodedData can be created from dictionary."""
        data = b"Hello, World!"
        metadata = {"color_count": 8, "ecc_level": "M"}
        data_dict = {"data": data, "metadata": metadata}

        encoded_data = EncodedData.from_dict(data_dict)

        assert encoded_data.data == data
        assert encoded_data.metadata == metadata

    def test_encoded_data_string_representation(self):
        """Test EncodedData has useful string representation."""
        data = b"Hello"
        metadata = {"color_count": 8}
        encoded_data = EncodedData(data=data, metadata=metadata)

        str_repr = str(encoded_data)

        assert "EncodedData" in str_repr
        assert "5 bytes" in str_repr
        assert "color_count=8" in str_repr

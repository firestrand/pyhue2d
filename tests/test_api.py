import pytest
import pyhue2d


def test_encode_not_implemented():
    with pytest.raises(NotImplementedError):
        pyhue2d.encode(b"data")


def test_decode_not_implemented():
    with pytest.raises(NotImplementedError):
        pyhue2d.decode("image.png")

"""Tests for Bitmap class."""

import numpy as np
import pytest

from pyhue2d.jabcode.core import Bitmap


class TestBitmap:
    """Test cases for Bitmap class."""

    def test_bitmap_creation_with_valid_array(self):
        """Test Bitmap can be created with valid numpy array."""
        array = np.zeros((10, 10, 3), dtype=np.uint8)
        bitmap = Bitmap(array=array, width=10, height=10)

        assert np.array_equal(bitmap.array, array)
        assert bitmap.width == 10
        assert bitmap.height == 10

    def test_bitmap_creation_with_grayscale_array(self):
        """Test Bitmap can be created with grayscale array."""
        array = np.zeros((10, 10), dtype=np.uint8)
        bitmap = Bitmap(array=array, width=10, height=10)

        assert np.array_equal(bitmap.array, array)
        assert bitmap.width == 10
        assert bitmap.height == 10

    def test_bitmap_dimension_validation_mismatch(self):
        """Test Bitmap validates array dimensions match width/height."""
        array = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Array dimensions don't match width/height"):
            Bitmap(array=array, width=5, height=10)

        with pytest.raises(ValueError, match="Array dimensions don't match width/height"):
            Bitmap(array=array, width=10, height=5)

    def test_bitmap_array_type_validation(self):
        """Test Bitmap validates array is numpy array."""
        with pytest.raises(TypeError, match="Array must be a numpy array"):
            Bitmap(array=[[1, 2], [3, 4]], width=2, height=2)

    def test_bitmap_negative_dimensions(self):
        """Test Bitmap rejects negative dimensions."""
        array = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="Width and height must be positive"):
            Bitmap(array=array, width=-1, height=10)

        with pytest.raises(ValueError, match="Width and height must be positive"):
            Bitmap(array=array, width=10, height=-1)

    def test_bitmap_get_pixel_rgb(self):
        """Test Bitmap can get RGB pixel values."""
        array = np.zeros((3, 3, 3), dtype=np.uint8)
        array[1, 1] = [255, 128, 64]  # Set middle pixel to specific color

        bitmap = Bitmap(array=array, width=3, height=3)
        pixel = bitmap.get_pixel(1, 1)

        assert pixel == (255, 128, 64)

    def test_bitmap_get_pixel_grayscale(self):
        """Test Bitmap can get grayscale pixel values."""
        array = np.zeros((3, 3), dtype=np.uint8)
        array[1, 1] = 128  # Set middle pixel to gray

        bitmap = Bitmap(array=array, width=3, height=3)
        pixel = bitmap.get_pixel(1, 1)

        assert pixel == 128

    def test_bitmap_set_pixel_rgb(self):
        """Test Bitmap can set RGB pixel values."""
        array = np.zeros((3, 3, 3), dtype=np.uint8)
        bitmap = Bitmap(array=array, width=3, height=3)

        bitmap.set_pixel(1, 1, (255, 128, 64))

        assert np.array_equal(bitmap.array[1, 1], [255, 128, 64])

    def test_bitmap_set_pixel_grayscale(self):
        """Test Bitmap can set grayscale pixel values."""
        array = np.zeros((3, 3), dtype=np.uint8)
        bitmap = Bitmap(array=array, width=3, height=3)

        bitmap.set_pixel(1, 1, 128)

        assert bitmap.array[1, 1] == 128

    def test_bitmap_pixel_bounds_checking(self):
        """Test Bitmap validates pixel coordinates are in bounds."""
        array = np.zeros((3, 3, 3), dtype=np.uint8)
        bitmap = Bitmap(array=array, width=3, height=3)

        # Test out of bounds access
        with pytest.raises(IndexError, match="Pixel coordinates out of bounds"):
            bitmap.get_pixel(3, 1)

        with pytest.raises(IndexError, match="Pixel coordinates out of bounds"):
            bitmap.get_pixel(1, 3)

        with pytest.raises(IndexError, match="Pixel coordinates out of bounds"):
            bitmap.set_pixel(-1, 1, (255, 0, 0))

    def test_bitmap_is_grayscale(self):
        """Test Bitmap can determine if it's grayscale or color."""
        # Test grayscale
        gray_array = np.zeros((3, 3), dtype=np.uint8)
        gray_bitmap = Bitmap(array=gray_array, width=3, height=3)
        assert gray_bitmap.is_grayscale() is True

        # Test RGB
        rgb_array = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb_bitmap = Bitmap(array=rgb_array, width=3, height=3)
        assert rgb_bitmap.is_grayscale() is False

    def test_bitmap_to_pil_image(self):
        """Test Bitmap can convert to PIL Image."""
        array = np.full((3, 3, 3), 128, dtype=np.uint8)
        bitmap = Bitmap(array=array, width=3, height=3)

        pil_image = bitmap.to_pil_image()

        # PIL uses width, height format
        assert pil_image.size == (3, 3)
        assert pil_image.mode == "RGB"

    def test_bitmap_from_pil_image(self):
        """Test Bitmap can be created from PIL Image."""
        from PIL import Image

        # Create a PIL image
        pil_image = Image.new("RGB", (5, 5), color=(255, 128, 64))

        bitmap = Bitmap.from_pil_image(pil_image)

        assert bitmap.width == 5
        assert bitmap.height == 5
        assert bitmap.array.shape == (5, 5, 3)
        assert np.all(bitmap.array == [255, 128, 64])

    def test_bitmap_resize(self):
        """Test Bitmap can be resized."""
        array = np.ones((3, 3, 3), dtype=np.uint8) * 128
        bitmap = Bitmap(array=array, width=3, height=3)

        resized = bitmap.resize(6, 6)

        assert resized.width == 6
        assert resized.height == 6
        assert resized.array.shape == (6, 6, 3)

    def test_bitmap_copy(self):
        """Test Bitmap can be copied."""
        array = np.ones((3, 3, 3), dtype=np.uint8) * 128
        bitmap = Bitmap(array=array, width=3, height=3)

        copy = bitmap.copy()

        assert copy.width == bitmap.width
        assert copy.height == bitmap.height
        assert np.array_equal(copy.array, bitmap.array)
        assert copy is not bitmap  # Different objects
        assert copy.array is not bitmap.array  # Different arrays

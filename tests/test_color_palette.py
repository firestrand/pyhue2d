"""Tests for ColorPalette class."""

import pytest
from pyhue2d.jabcode.color_palette import ColorPalette


class TestColorPalette:
    """Test cases for ColorPalette class."""

    def test_colorpalette_creation_with_default_8_colors(self):
        """Test ColorPalette can be created with default 8-color palette."""
        palette = ColorPalette()

        assert palette.color_count == 8
        assert len(palette.colors) == 8
        assert palette.colors[0] == (0, 0, 0)  # Black
        assert palette.colors[7] == (255, 255, 255)  # White

    def test_colorpalette_creation_with_4_colors(self):
        """Test ColorPalette can be created with 4 colors."""
        palette = ColorPalette(color_count=4)

        assert palette.color_count == 4
        assert len(palette.colors) == 4
        assert (0, 0, 0) in palette.colors  # Black
        assert (255, 255, 255) in palette.colors  # White

    def test_colorpalette_creation_with_16_colors(self):
        """Test ColorPalette can be created with 16 colors."""
        palette = ColorPalette(color_count=16)

        assert palette.color_count == 16
        assert len(palette.colors) == 16
        assert len(set(palette.colors)) == 16  # All colors unique

    def test_colorpalette_creation_with_custom_colors(self):
        """Test ColorPalette can be created with custom color list."""
        custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        palette = ColorPalette(colors=custom_colors)

        assert palette.color_count == 3
        assert palette.colors == custom_colors

    def test_colorpalette_invalid_color_count(self):
        """Test ColorPalette rejects invalid color counts."""
        with pytest.raises(ValueError, match="Color count must be one of"):
            ColorPalette(color_count=7)

        with pytest.raises(ValueError, match="Color count must be one of"):
            ColorPalette(color_count=512)

    def test_colorpalette_invalid_custom_colors(self):
        """Test ColorPalette validates custom colors."""
        with pytest.raises(ValueError, match="Colors must be a list of RGB tuples"):
            ColorPalette(colors="invalid")

        with pytest.raises(ValueError, match="Each color must be an RGB tuple with 3 values"):
            ColorPalette(colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), 123])

        with pytest.raises(ValueError, match="RGB values must be between 0 and 255"):
            ColorPalette(colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (256, 0, 0)])

    def test_colorpalette_get_color_by_index(self):
        """Test ColorPalette can get color by index."""
        palette = ColorPalette(color_count=4)

        color = palette.get_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_colorpalette_get_color_invalid_index(self):
        """Test ColorPalette validates color index."""
        palette = ColorPalette(color_count=4)

        with pytest.raises(IndexError, match="Color index out of range"):
            palette.get_color(4)

        with pytest.raises(IndexError, match="Color index out of range"):
            palette.get_color(-1)

    def test_colorpalette_get_index_of_color(self):
        """Test ColorPalette can find index of color."""
        palette = ColorPalette(color_count=8)

        # Test known colors
        assert palette.get_index((0, 0, 0)) == 0  # Black
        assert palette.get_index((255, 255, 255)) == 7  # White

    def test_colorpalette_get_index_of_missing_color(self):
        """Test ColorPalette handles missing color lookup."""
        palette = ColorPalette(color_count=4)

        with pytest.raises(ValueError, match="Color not found in palette"):
            palette.get_index((128, 128, 128))  # Gray not in 4-color palette

    def test_colorpalette_find_closest_color(self):
        """Test ColorPalette can find closest color match."""
        palette = ColorPalette(color_count=8)

        # Should find black for dark gray
        closest_index = palette.find_closest_color((32, 32, 32))
        assert closest_index == 0  # Black

        # Should find white for light gray
        closest_index = palette.find_closest_color((224, 224, 224))
        assert closest_index == 7  # White

    def test_colorpalette_get_distance_between_colors(self):
        """Test ColorPalette can calculate color distance."""
        palette = ColorPalette()

        # Distance between black and white
        distance = palette.get_color_distance((0, 0, 0), (255, 255, 255))
        expected = (255**2 + 255**2 + 255**2) ** 0.5  # Euclidean distance
        assert abs(distance - expected) < 1e-10

        # Distance between same color
        distance = palette.get_color_distance((128, 128, 128), (128, 128, 128))
        assert distance == 0.0

    def test_colorpalette_is_high_contrast(self):
        """Test ColorPalette can determine if palette has good contrast."""
        # 8-color palette should have good contrast
        palette_8 = ColorPalette(color_count=8)
        assert palette_8.is_high_contrast() is True

        # Custom low-contrast palette
        similar_colors = [(100, 100, 100), (110, 110, 110), (120, 120, 120)]
        palette_low = ColorPalette(colors=similar_colors)
        assert palette_low.is_high_contrast() is False

    def test_colorpalette_to_rgb_array(self):
        """Test ColorPalette can convert to numpy RGB array."""
        import numpy as np

        palette = ColorPalette(color_count=4)
        rgb_array = palette.to_rgb_array()

        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.shape == (4, 3)
        assert rgb_array.dtype == np.uint8
        assert np.array_equal(rgb_array[0], [0, 0, 0])  # Black

    def test_colorpalette_copy(self):
        """Test ColorPalette can be copied."""
        original = ColorPalette(color_count=8)
        copy = original.copy()

        assert copy.color_count == original.color_count
        assert copy.colors == original.colors
        assert copy is not original
        assert copy.colors is not original.colors

    def test_colorpalette_string_representation(self):
        """Test ColorPalette has useful string representation."""
        palette = ColorPalette(color_count=4)
        str_repr = str(palette)

        assert "ColorPalette" in str_repr
        assert "4 colors" in str_repr

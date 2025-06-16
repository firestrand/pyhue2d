"""Tests for BitmapRenderer class."""

import numpy as np
import pytest
from PIL import Image

from pyhue2d.jabcode.bitmap_renderer import BitmapRenderer
from pyhue2d.jabcode.color_palette import ColorPalette
from pyhue2d.jabcode.core import Bitmap, Symbol


class TestBitmapRenderer:
    """Test suite for BitmapRenderer class - Basic functionality."""

    def test_bitmap_renderer_creation_with_defaults(self):
        """Test that BitmapRenderer can be created with default settings."""
        renderer = BitmapRenderer()
        assert renderer is not None

    def test_bitmap_renderer_creation_with_custom_settings(self):
        """Test creating BitmapRenderer with custom settings."""
        settings = {
            "module_size": 8,
            "quiet_zone": 4,
            "background_color": (255, 255, 255),
            "border_width": 2,
        }
        renderer = BitmapRenderer(settings)
        assert renderer is not None

    def test_bitmap_renderer_render_simple_matrix(self):
        """Test rendering a simple color matrix."""
        renderer = BitmapRenderer()
        # Create a simple 5x5 matrix with different colors
        color_matrix = np.array(
            [
                [0, 1, 2, 1, 0],
                [1, 2, 3, 2, 1],
                [2, 3, 4, 3, 2],
                [1, 2, 3, 2, 1],
                [0, 1, 2, 1, 0],
            ],
            dtype=np.uint8,
        )

        result = renderer.render_matrix(color_matrix)
        assert isinstance(result, Bitmap)

    def test_bitmap_renderer_render_to_pil_image(self):
        """Test rendering matrix to PIL Image."""
        renderer = BitmapRenderer()
        color_matrix = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.uint8)

        result = renderer.render_to_image(color_matrix)
        assert isinstance(result, Image.Image)

    def test_bitmap_renderer_render_symbol(self):
        """Test rendering a complete Symbol object."""
        renderer = BitmapRenderer()
        # Create a basic symbol for testing
        symbol = Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 21))

        # Create test matrix data
        test_matrix = np.random.randint(0, 8, size=(21, 21), dtype=np.uint8)

        result = renderer.render_symbol(symbol, test_matrix)
        assert isinstance(result, Bitmap)


class TestBitmapRendererImplementation:
    """Test suite for BitmapRenderer implementation details.

    These tests will pass once the implementation is complete.
    """

    @pytest.fixture
    def renderer(self):
        """Create a bitmap renderer for testing."""
        try:
            return BitmapRenderer()
        except NotImplementedError:
            pytest.skip("BitmapRenderer not yet implemented")

    def test_renderer_creation_with_defaults(self, renderer):
        """Test that renderer can be created with default values."""
        assert renderer is not None
        assert hasattr(renderer, "settings")
        assert hasattr(renderer, "color_palette")

    def test_renderer_default_settings(self, renderer):
        """Test renderer default settings."""
        settings = renderer.settings
        assert isinstance(settings, dict)
        assert "module_size" in settings
        assert "quiet_zone" in settings
        assert "background_color" in settings

    def test_renderer_custom_settings(self):
        """Test renderer with custom settings."""
        custom_settings = {
            "module_size": 10,
            "quiet_zone": 6,
            "background_color": (240, 240, 240),
            "border_width": 3,
            "anti_aliasing": True,
        }
        try:
            renderer = BitmapRenderer(custom_settings)
            for key, value in custom_settings.items():
                if key in renderer.settings:
                    assert renderer.settings[key] == value
        except NotImplementedError:
            pytest.skip("BitmapRenderer not yet implemented")

    def test_renderer_color_palette_integration(self, renderer):
        """Test color palette integration."""
        assert hasattr(renderer, "color_palette")
        assert isinstance(renderer.color_palette, ColorPalette)
        assert renderer.color_palette.color_count >= 4

    def test_renderer_render_matrix_returns_bitmap(self, renderer):
        """Test that rendering matrix returns Bitmap."""
        test_matrix = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]], dtype=np.uint8)

        result = renderer.render_matrix(test_matrix)
        assert isinstance(result, Bitmap)
        assert result.width > 0
        assert result.height > 0

    def test_renderer_render_to_image_returns_pil(self, renderer):
        """Test that rendering to image returns PIL Image."""
        test_matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        result = renderer.render_to_image(test_matrix)
        assert isinstance(result, Image.Image)
        assert result.width > 0
        assert result.height > 0
        assert result.mode in ["RGB", "RGBA"]

    def test_renderer_module_size_scaling(self, renderer):
        """Test module size scaling."""
        test_matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        # Test different module sizes
        for module_size in [1, 2, 4, 8, 16]:
            renderer.settings["module_size"] = module_size
            result = renderer.render_matrix(test_matrix)

            expected_width = test_matrix.shape[1] * module_size
            expected_height = test_matrix.shape[0] * module_size

            # Account for quiet zone
            quiet_zone = renderer.settings.get("quiet_zone", 0)
            expected_width += 2 * quiet_zone * module_size
            expected_height += 2 * quiet_zone * module_size

            assert result.width == expected_width
            assert result.height == expected_height

    def test_renderer_quiet_zone_application(self, renderer):
        """Test quiet zone application."""
        test_matrix = np.array([[1]], dtype=np.uint8)  # Single module

        # Test different quiet zone sizes
        for quiet_zone in [0, 1, 2, 4]:
            renderer.settings["quiet_zone"] = quiet_zone
            result = renderer.render_matrix(test_matrix)

            module_size = renderer.settings["module_size"]
            expected_size = (1 + 2 * quiet_zone) * module_size

            assert result.width == expected_size
            assert result.height == expected_size

    def test_renderer_color_mapping_accuracy(self, renderer):
        """Test color mapping accuracy."""
        # Create matrix with specific color indices
        test_matrix = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=np.uint8)

        result = renderer.render_matrix(test_matrix)

        # Check that colors are properly mapped
        # This is a basic check - more detailed color verification would require
        # checking specific pixel values
        assert result.array.shape[2] == 3  # RGB channels
        assert np.max(result.array) <= 255
        assert np.min(result.array) >= 0

    def test_renderer_background_color_setting(self, renderer):
        """Test background color setting."""
        test_matrix = np.array([[1]], dtype=np.uint8)

        # Test different background colors
        background_colors = [
            (255, 255, 255),  # White
            (0, 0, 0),  # Black
            (128, 128, 128),  # Gray
            (255, 0, 0),  # Red
        ]

        for bg_color in background_colors:
            renderer.settings["background_color"] = bg_color
            renderer.settings["quiet_zone"] = 2  # Ensure there's background area

            result = renderer.render_matrix(test_matrix)

            # Check that result is valid
            assert isinstance(result, Bitmap)
            assert result.width > 0
            assert result.height > 0

    def test_renderer_border_width_application(self, renderer):
        """Test border width application."""
        test_matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        # Test different border widths
        for border_width in [0, 1, 2, 4]:
            renderer.settings["border_width"] = border_width
            result = renderer.render_matrix(test_matrix)

            # Border should increase the overall size
            base_size = test_matrix.shape[0] * renderer.settings["module_size"]
            quiet_zone_size = 2 * renderer.settings.get("quiet_zone", 0) * renderer.settings["module_size"]
            expected_min_size = base_size + quiet_zone_size

            assert result.width >= expected_min_size
            assert result.height >= expected_min_size

    def test_renderer_symbol_rendering_integration(self, renderer):
        """Test complete symbol rendering."""
        symbol = Symbol(version=1, color_count=8, ecc_level="M", matrix_size=(21, 21))

        # Create realistic symbol matrix
        symbol_matrix = np.zeros((21, 21), dtype=np.uint8)

        # Add some pattern (simplified finder pattern)
        symbol_matrix[0:7, 0:7] = 1  # Top-left finder pattern
        symbol_matrix[0:7, 14:21] = 2  # Top-right finder pattern
        symbol_matrix[14:21, 0:7] = 3  # Bottom-left finder pattern

        # Add some data
        symbol_matrix[7:14, 7:14] = np.random.randint(0, 8, size=(7, 7))

        result = renderer.render_symbol(symbol, symbol_matrix)

        assert isinstance(result, Bitmap)
        assert result.width > 0
        assert result.height > 0

        # Symbol should be square
        assert result.width == result.height

    def test_renderer_custom_color_palette(self, renderer):
        """Test rendering with custom color palette."""
        test_matrix = np.array([[0, 1, 2, 3]], dtype=np.uint8)

        # Create custom color palette
        custom_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
        ]
        custom_palette = ColorPalette(colors=custom_colors)

        if hasattr(renderer, "set_color_palette"):
            renderer.set_color_palette(custom_palette)
            result = renderer.render_matrix(test_matrix)
            assert isinstance(result, Bitmap)

    def test_renderer_large_matrix_handling(self, renderer):
        """Test handling of large matrices."""
        # Create larger matrix
        large_matrix = np.random.randint(0, 8, size=(50, 50), dtype=np.uint8)

        # Use smaller module size to keep output manageable
        renderer.settings["module_size"] = 2
        renderer.settings["quiet_zone"] = 1

        result = renderer.render_matrix(large_matrix)

        assert isinstance(result, Bitmap)
        assert result.width > 0
        assert result.height > 0

        # Check dimensions are correct
        expected_size = 50 * 2 + 2 * 1 * 2  # matrix_size * module_size + quiet_zone
        assert result.width == expected_size
        assert result.height == expected_size

    def test_renderer_error_handling_invalid_matrix(self, renderer):
        """Test error handling for invalid matrices."""
        # Test with invalid matrix types
        invalid_matrices = [
            None,
            [],
            "not a matrix",
            np.array([]),  # Empty array
        ]

        for invalid_matrix in invalid_matrices:
            with pytest.raises((ValueError, TypeError)):
                renderer.render_matrix(invalid_matrix)

    def test_renderer_error_handling_invalid_color_indices(self, renderer):
        """Test error handling for invalid color indices."""
        # Matrix with color indices beyond palette range
        invalid_matrix = np.array([[0, 1, 255]], dtype=np.uint8)  # 255 likely beyond palette

        # Should handle gracefully (either clamp or raise error)
        try:
            result = renderer.render_matrix(invalid_matrix)
            assert isinstance(result, Bitmap)
        except ValueError:
            # Acceptable to raise error for out-of-range colors
            pass

    def test_renderer_performance_metrics(self, renderer):
        """Test performance metrics collection."""
        test_matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8)

        # Render multiple times
        for _ in range(5):
            result = renderer.render_matrix(test_matrix)
            assert isinstance(result, Bitmap)

        # Check if performance metrics are available
        if hasattr(renderer, "get_performance_stats"):
            stats = renderer.get_performance_stats()
            assert isinstance(stats, dict)
            assert "total_renders" in stats
            assert stats["total_renders"] >= 5

    def test_renderer_string_representation(self, renderer):
        """Test renderer string representation."""
        str_repr = str(renderer)
        assert "BitmapRenderer" in str_repr
        assert "module_size" in str_repr or "size" in str_repr

    def test_renderer_copy(self, renderer):
        """Test renderer copying."""
        if hasattr(renderer, "copy"):
            copied = renderer.copy()
            assert copied is not renderer
            assert copied.settings == renderer.settings

    def test_renderer_reset(self, renderer):
        """Test renderer reset functionality."""
        if hasattr(renderer, "reset"):
            # Render some data
            test_matrix = np.array([[0, 1]], dtype=np.uint8)
            renderer.render_matrix(test_matrix)

            # Reset renderer
            renderer.reset()

            # Should be able to render again
            result = renderer.render_matrix(test_matrix)
            assert isinstance(result, Bitmap)

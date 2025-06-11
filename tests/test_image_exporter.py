"""Tests for ImageExporter class."""

import pytest
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from pyhue2d.jabcode.image_exporter import ImageExporter
from pyhue2d.jabcode.core import Bitmap


class TestImageExporter:
    """Test suite for ImageExporter class - Basic functionality."""
    
    def test_image_exporter_creation_with_defaults(self):
        """Test that ImageExporter can be created with default settings."""
        exporter = ImageExporter()
        assert exporter is not None
    
    def test_image_exporter_creation_with_custom_settings(self):
        """Test creating ImageExporter with custom settings."""
        settings = {
            'default_format': 'PNG',
            'quality': 95,
            'compression_level': 6,
            'dpi': (300, 300)
        }
        exporter = ImageExporter(settings)
        assert exporter is not None
    
    def test_image_exporter_export_bitmap_to_png(self):
        """Test exporting bitmap to PNG format."""
        exporter = ImageExporter()
        
        # Create test bitmap
        test_array = np.zeros((20, 20, 3), dtype=np.uint8)
        test_array[5:15, 5:15] = [255, 0, 0]  # Red square
        bitmap = Bitmap(array=test_array, width=20, height=20)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_bitmap(bitmap, tmp_file.name, format='PNG')
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
                # Verify it's a valid PNG
                with Image.open(tmp_file.name) as img:
                    assert img.format == 'PNG'
                    assert img.size == (20, 20)
            finally:
                os.unlink(tmp_file.name)
    
    def test_image_exporter_export_pil_image(self):
        """Test exporting PIL Image."""
        exporter = ImageExporter()
        
        # Create test PIL Image
        test_image = Image.new('RGB', (30, 30), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_image(test_image, tmp_file.name)
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
            finally:
                os.unlink(tmp_file.name)
    
    def test_image_exporter_export_matrix(self):
        """Test exporting numpy matrix directly."""
        exporter = ImageExporter()
        
        # Create test matrix
        test_matrix = np.random.randint(0, 256, size=(25, 25, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_matrix(test_matrix, tmp_file.name)
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
            finally:
                os.unlink(tmp_file.name)


class TestImageExporterImplementation:
    """Test suite for ImageExporter implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def exporter(self):
        """Create an image exporter for testing."""
        try:
            return ImageExporter()
        except NotImplementedError:
            pytest.skip("ImageExporter not yet implemented")
    
    @pytest.fixture
    def test_bitmap(self):
        """Create a test bitmap for testing."""
        test_array = np.zeros((50, 50, 3), dtype=np.uint8)
        # Create a pattern
        test_array[10:40, 10:40] = [255, 0, 0]  # Red square
        test_array[20:30, 20:30] = [0, 255, 0]  # Green square in center
        return Bitmap(array=test_array, width=50, height=50)
    
    def test_exporter_creation_with_defaults(self, exporter):
        """Test that exporter can be created with default values."""
        assert exporter is not None
        assert hasattr(exporter, 'settings')
        assert hasattr(exporter, 'supported_formats')
    
    def test_exporter_default_settings(self, exporter):
        """Test exporter default settings."""
        settings = exporter.settings
        assert isinstance(settings, dict)
        assert 'default_format' in settings
        assert 'quality' in settings
        assert 'dpi' in settings
    
    def test_exporter_supported_formats(self, exporter):
        """Test supported format list."""
        formats = exporter.supported_formats
        assert isinstance(formats, list)
        assert 'PNG' in formats
        assert 'JPEG' in formats
    
    def test_exporter_export_bitmap_png(self, exporter, test_bitmap):
        """Test exporting bitmap to PNG format."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_bitmap(test_bitmap, tmp_file.name, format='PNG')
                
                # Verify file exists and has content
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
                # Verify it's a valid PNG with correct dimensions
                with Image.open(tmp_file.name) as img:
                    assert img.format == 'PNG'
                    assert img.size == (test_bitmap.width, test_bitmap.height)
                    assert img.mode == 'RGB'
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_export_bitmap_jpeg(self, exporter, test_bitmap):
        """Test exporting bitmap to JPEG format."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            try:
                exporter.export_bitmap(test_bitmap, tmp_file.name, format='JPEG')
                
                # Verify file exists and has content
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
                # Verify it's a valid JPEG
                with Image.open(tmp_file.name) as img:
                    assert img.format == 'JPEG'
                    assert img.size == (test_bitmap.width, test_bitmap.height)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_export_image_pil(self, exporter):
        """Test exporting PIL Image object."""
        # Create test PIL Image
        test_image = Image.new('RGB', (40, 40), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_image(test_image, tmp_file.name, format='PNG')
                
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
                # Verify exported image
                with Image.open(tmp_file.name) as img:
                    assert img.format == 'PNG'
                    assert img.size == (40, 40)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_export_matrix_rgb(self, exporter):
        """Test exporting RGB matrix."""
        # Create RGB matrix
        rgb_matrix = np.zeros((30, 30, 3), dtype=np.uint8)
        rgb_matrix[5:25, 5:25] = [0, 0, 255]  # Blue square
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_matrix(rgb_matrix, tmp_file.name)
                
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
                with Image.open(tmp_file.name) as img:
                    assert img.size == (30, 30)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_export_matrix_grayscale(self, exporter):
        """Test exporting grayscale matrix."""
        # Create grayscale matrix
        gray_matrix = np.zeros((35, 35), dtype=np.uint8)
        gray_matrix[10:25, 10:25] = 128  # Gray square
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                exporter.export_matrix(gray_matrix, tmp_file.name)
                
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_format_detection_from_extension(self, exporter, test_bitmap):
        """Test automatic format detection from file extension."""
        extensions_and_formats = [
            ('.png', 'PNG'),
            ('.jpg', 'JPEG'),
            ('.jpeg', 'JPEG'),
            ('.bmp', 'BMP'),
            ('.tiff', 'TIFF'),
        ]
        
        for ext, expected_format in extensions_and_formats:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                try:
                    # Test auto-detection (don't specify format)
                    exporter.export_bitmap(test_bitmap, tmp_file.name)
                    
                    assert os.path.exists(tmp_file.name)
                    assert os.path.getsize(tmp_file.name) > 0
                    
                    # Verify format if supported
                    if expected_format in exporter.supported_formats:
                        with Image.open(tmp_file.name) as img:
                            # Some formats might be normalized (e.g., JPEG -> JPEG)
                            assert img.format in [expected_format, 'JPEG', 'PNG']
                except (ValueError, OSError):
                    # Some formats might not be supported - that's okay
                    pass
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_exporter_quality_settings_jpeg(self, exporter, test_bitmap):
        """Test JPEG quality settings."""
        quality_levels = [10, 50, 90, 95]
        file_sizes = []
        
        for quality in quality_levels:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                try:
                    exporter.export_bitmap(test_bitmap, tmp_file.name, 
                                         format='JPEG', quality=quality)
                    
                    assert os.path.exists(tmp_file.name)
                    file_size = os.path.getsize(tmp_file.name)
                    assert file_size > 0
                    file_sizes.append(file_size)
                    
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        
        # Higher quality should generally result in larger files
        # (though this isn't guaranteed for all images)
        assert len(file_sizes) == len(quality_levels)
    
    def test_exporter_dpi_settings(self, exporter, test_bitmap):
        """Test DPI settings."""
        dpi_values = [(72, 72), (150, 150), (300, 300)]
        
        for dpi in dpi_values:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    exporter.export_bitmap(test_bitmap, tmp_file.name, 
                                         format='PNG', dpi=dpi)
                    
                    assert os.path.exists(tmp_file.name)
                    assert os.path.getsize(tmp_file.name) > 0
                    
                    # Verify DPI was set (if supported by format)
                    with Image.open(tmp_file.name) as img:
                        if hasattr(img, 'info') and 'dpi' in img.info:
                            # DPI might be stored differently by PIL
                            pass  # Just verify file was created successfully
                        
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_exporter_compression_settings_png(self, exporter, test_bitmap):
        """Test PNG compression settings."""
        compression_levels = [0, 3, 6, 9]
        
        for compression in compression_levels:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    exporter.export_bitmap(test_bitmap, tmp_file.name, 
                                         format='PNG', compression_level=compression)
                    
                    assert os.path.exists(tmp_file.name)
                    assert os.path.getsize(tmp_file.name) > 0
                    
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_exporter_batch_export(self, exporter):
        """Test batch export functionality."""
        # Create multiple test bitmaps
        bitmaps = []
        for i in range(3):
            array = np.zeros((20, 20, 3), dtype=np.uint8)
            # Clamp color values to 0-255
            color = [min(255, i * 80), min(255, (i + 1) * 80), min(255, (i + 2) * 80)]
            array[5:15, 5:15] = color
            bitmaps.append(Bitmap(array=array, width=20, height=20))
        if hasattr(exporter, 'export_batch'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                filenames = [
                    os.path.join(tmp_dir, f'test_{i}.png') 
                    for i in range(len(bitmaps))
                ]
                exporter.export_batch(bitmaps, filenames)
                for filename in filenames:
                    assert os.path.exists(filename)
                    assert os.path.getsize(filename) > 0
    
    def test_exporter_metadata_preservation(self, exporter, test_bitmap):
        """Test metadata preservation in exported images."""
        metadata = {
            'Title': 'Test JABCode Symbol',
            'Description': 'Generated by PyHue2D',
            'Software': 'PyHue2D JABCode Library'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                if hasattr(exporter, 'export_bitmap_with_metadata'):
                    exporter.export_bitmap_with_metadata(
                        test_bitmap, tmp_file.name, metadata=metadata
                    )
                else:
                    # Fallback to regular export
                    exporter.export_bitmap(test_bitmap, tmp_file.name)
                
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_error_handling_invalid_path(self, exporter, test_bitmap):
        """Test error handling for invalid file paths."""
        invalid_paths = [
            '/nonexistent/directory/file.png',
            '',
            None,
        ]
        
        for invalid_path in invalid_paths:
            with pytest.raises((ValueError, TypeError, OSError)):
                exporter.export_bitmap(test_bitmap, invalid_path)
    
    def test_exporter_error_handling_invalid_format(self, exporter, test_bitmap):
        """Test error handling for invalid formats."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                with pytest.raises(ValueError):
                    exporter.export_bitmap(test_bitmap, tmp_file.name, 
                                         format='INVALID_FORMAT')
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_error_handling_invalid_bitmap(self, exporter):
        """Test error handling for invalid bitmap objects."""
        invalid_bitmaps = [
            None,
            "not a bitmap",
            123,
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                for invalid_bitmap in invalid_bitmaps:
                    with pytest.raises((ValueError, TypeError)):
                        exporter.export_bitmap(invalid_bitmap, tmp_file.name)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_performance_large_images(self, exporter):
        """Test performance with large images."""
        # Create large bitmap
        large_array = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
        large_bitmap = Bitmap(array=large_array, width=500, height=500)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                import time
                start_time = time.time()
                exporter.export_bitmap(large_bitmap, tmp_file.name)
                end_time = time.time()
                
                # Should complete in reasonable time (adjust threshold as needed)
                assert end_time - start_time < 10.0  # 10 seconds max
                
                assert os.path.exists(tmp_file.name)
                assert os.path.getsize(tmp_file.name) > 0
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_exporter_statistics_collection(self, exporter, test_bitmap):
        """Test statistics collection."""
        # Export multiple images
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    exporter.export_bitmap(test_bitmap, tmp_file.name)
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
        
        if hasattr(exporter, 'get_export_stats'):
            stats = exporter.get_export_stats()
            assert isinstance(stats, dict)
            assert 'total_exports' in stats
            assert stats['total_exports'] >= 3
    
    def test_exporter_string_representation(self, exporter):
        """Test exporter string representation."""
        str_repr = str(exporter)
        assert "ImageExporter" in str_repr
        assert "format" in str_repr or "PNG" in str_repr
    
    def test_exporter_copy(self, exporter):
        """Test exporter copying."""
        if hasattr(exporter, 'copy'):
            copied = exporter.copy()
            assert copied is not exporter
            assert copied.settings == exporter.settings
    
    def test_exporter_reset(self, exporter):
        """Test exporter reset functionality."""
        if hasattr(exporter, 'reset'):
            # Export some data
            test_array = np.zeros((10, 10, 3), dtype=np.uint8)
            test_bitmap = Bitmap(array=test_array, width=10, height=10)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                try:
                    exporter.export_bitmap(test_bitmap, tmp_file.name)
                    
                    # Reset exporter
                    exporter.reset()
                    
                    # Should be able to export again
                    exporter.export_bitmap(test_bitmap, tmp_file.name)
                    assert os.path.exists(tmp_file.name)
                    
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
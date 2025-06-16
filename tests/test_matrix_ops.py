"""Tests for matrix operations module."""

import numpy as np
import pytest

from pyhue2d.jabcode.matrix_ops import (
    apply_mask_pattern,
    calculate_matrix_checksum,
    create_matrix,
    extract_submatrix,
    find_pattern_in_matrix,
    flip_matrix_horizontal,
    flip_matrix_vertical,
    place_submatrix,
    resize_matrix,
    rotate_matrix_90,
)


class TestMatrixOps:
    """Test cases for matrix operations."""

    def test_create_matrix_zeros(self):
        """Test creating zero matrix."""
        matrix = create_matrix(3, 4, fill_value=0)

        assert matrix.shape == (3, 4)
        assert matrix.dtype == np.uint8
        assert np.all(matrix == 0)

    def test_create_matrix_filled(self):
        """Test creating matrix with fill value."""
        matrix = create_matrix(2, 3, fill_value=5)

        assert matrix.shape == (2, 3)
        assert np.all(matrix == 5)

    def test_create_matrix_different_dtype(self):
        """Test creating matrix with different data type."""
        matrix = create_matrix(2, 2, fill_value=1.5, dtype=np.float32)

        assert matrix.dtype == np.float32
        assert np.all(matrix == 1.5)

    def test_rotate_matrix_90_clockwise(self):
        """Test 90-degree clockwise rotation."""
        original = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        rotated = rotate_matrix_90(original, clockwise=True)
        expected = np.array([[3, 1], [4, 2]], dtype=np.uint8)

        assert np.array_equal(rotated, expected)

    def test_rotate_matrix_90_counterclockwise(self):
        """Test 90-degree counterclockwise rotation."""
        original = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        rotated = rotate_matrix_90(original, clockwise=False)
        expected = np.array([[2, 4], [1, 3]], dtype=np.uint8)

        assert np.array_equal(rotated, expected)

    def test_flip_matrix_horizontal(self):
        """Test horizontal flip (left-right)."""
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        flipped = flip_matrix_horizontal(original)
        expected = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8)

        assert np.array_equal(flipped, expected)

    def test_flip_matrix_vertical(self):
        """Test vertical flip (up-down)."""
        original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        flipped = flip_matrix_vertical(original)
        expected = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8)

        assert np.array_equal(flipped, expected)

    def test_extract_submatrix_valid(self):
        """Test extracting submatrix with valid coordinates."""
        matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint8)
        submatrix = extract_submatrix(matrix, start_row=1, start_col=1, height=2, width=2)
        expected = np.array([[6, 7], [10, 11]], dtype=np.uint8)

        assert np.array_equal(submatrix, expected)

    def test_extract_submatrix_out_of_bounds(self):
        """Test extracting submatrix with out-of-bounds coordinates."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        with pytest.raises(ValueError, match="Submatrix extends beyond matrix bounds"):
            extract_submatrix(matrix, start_row=1, start_col=1, height=2, width=2)

    def test_place_submatrix_valid(self):
        """Test placing submatrix at valid position."""
        matrix = create_matrix(4, 4, fill_value=0)
        submatrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        result = place_submatrix(matrix, submatrix, start_row=1, start_col=1)

        assert result[1, 1] == 1
        assert result[1, 2] == 2
        assert result[2, 1] == 3
        assert result[2, 2] == 4
        assert result[0, 0] == 0  # Original value preserved

    def test_place_submatrix_out_of_bounds(self):
        """Test placing submatrix out of bounds."""
        matrix = create_matrix(2, 2, fill_value=0)
        submatrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        with pytest.raises(ValueError, match="Submatrix extends beyond matrix bounds"):
            place_submatrix(matrix, submatrix, start_row=1, start_col=1)

    def test_apply_mask_pattern_0(self):
        """Test applying mask pattern 0: (row + col) % 2 == 0."""
        matrix = create_matrix(3, 3, fill_value=1)
        masked = apply_mask_pattern(matrix, pattern=0)

        # Pattern should flip values where (row + col) % 2 == 0
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        assert np.array_equal(masked, expected)

    def test_apply_mask_pattern_1(self):
        """Test applying mask pattern 1: row % 2 == 0."""
        matrix = create_matrix(3, 3, fill_value=1)
        masked = apply_mask_pattern(matrix, pattern=1)

        # Pattern should flip values where row % 2 == 0
        expected = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        assert np.array_equal(masked, expected)

    def test_apply_mask_pattern_invalid(self):
        """Test applying invalid mask pattern."""
        matrix = create_matrix(2, 2, fill_value=1)

        with pytest.raises(ValueError, match="Invalid mask pattern"):
            apply_mask_pattern(matrix, pattern=8)

    def test_calculate_matrix_checksum(self):
        """Test calculating matrix checksum."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        checksum = calculate_matrix_checksum(matrix)

        assert isinstance(checksum, int)
        assert checksum > 0

        # Same matrix should give same checksum
        checksum2 = calculate_matrix_checksum(matrix)
        assert checksum == checksum2

        # Different matrix should give different checksum
        matrix2 = np.array([[1, 2], [3, 5]], dtype=np.uint8)
        checksum3 = calculate_matrix_checksum(matrix2)
        assert checksum != checksum3

    def test_resize_matrix_nearest(self):
        """Test resizing matrix with nearest neighbor."""
        matrix = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        resized = resize_matrix(matrix, new_height=4, new_width=4)

        assert resized.shape == (4, 4)
        assert resized.dtype == matrix.dtype
        # Check some expected values for nearest neighbor
        assert resized[0, 0] == 1
        assert resized[3, 3] == 4

    def test_resize_matrix_invalid_size(self):
        """Test resizing matrix with invalid dimensions."""
        matrix = create_matrix(2, 2, fill_value=1)

        with pytest.raises(ValueError, match="New dimensions must be positive"):
            resize_matrix(matrix, new_height=0, new_width=2)

    def test_find_pattern_in_matrix_found(self):
        """Test finding pattern in matrix when pattern exists."""
        matrix = np.array([[0, 0, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]], dtype=np.uint8)

        pattern = np.array([[1, 2]], dtype=np.uint8)
        positions = find_pattern_in_matrix(matrix, pattern)

        assert len(positions) == 1
        assert positions[0] == (1, 1)  # (row, col) where pattern starts

    def test_find_pattern_in_matrix_not_found(self):
        """Test finding pattern in matrix when pattern doesn't exist."""
        matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)

        pattern = np.array([[1, 2]], dtype=np.uint8)
        positions = find_pattern_in_matrix(matrix, pattern)

        assert len(positions) == 0

    def test_find_pattern_in_matrix_multiple(self):
        """Test finding multiple occurrences of pattern."""
        matrix = np.array([[1, 2, 0, 1, 2], [0, 0, 0, 0, 0], [1, 2, 0, 0, 0]], dtype=np.uint8)

        pattern = np.array([[1, 2]], dtype=np.uint8)
        positions = find_pattern_in_matrix(matrix, pattern)

        assert len(positions) == 3
        assert (0, 0) in positions
        assert (0, 3) in positions
        assert (2, 0) in positions

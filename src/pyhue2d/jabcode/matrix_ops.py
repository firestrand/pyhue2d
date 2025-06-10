"""Matrix operations for JABCode implementation.

This module provides basic matrix operations needed for JABCode encoding
and decoding, including rotation, flipping, pattern application, and more.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image


def create_matrix(
    height: int, width: int, fill_value: int = 0, dtype: type = np.uint8
) -> np.ndarray:
    """Create a matrix filled with specified value.

    Args:
        height: Matrix height
        width: Matrix width
        fill_value: Value to fill matrix with
        dtype: NumPy data type

    Returns:
        Filled matrix
    """
    return np.full((height, width), fill_value, dtype=dtype)


def rotate_matrix_90(matrix: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """Rotate matrix 90 degrees.

    Args:
        matrix: Input matrix
        clockwise: True for clockwise, False for counterclockwise

    Returns:
        Rotated matrix
    """
    if clockwise:
        return np.rot90(matrix, k=-1)  # k=-1 for clockwise
    else:
        return np.rot90(matrix, k=1)  # k=1 for counterclockwise


def flip_matrix_horizontal(matrix: np.ndarray) -> np.ndarray:
    """Flip matrix horizontally (left-right).

    Args:
        matrix: Input matrix

    Returns:
        Horizontally flipped matrix
    """
    return np.fliplr(matrix)


def flip_matrix_vertical(matrix: np.ndarray) -> np.ndarray:
    """Flip matrix vertically (up-down).

    Args:
        matrix: Input matrix

    Returns:
        Vertically flipped matrix
    """
    return np.flipud(matrix)


def extract_submatrix(
    matrix: np.ndarray, start_row: int, start_col: int, height: int, width: int
) -> np.ndarray:
    """Extract submatrix from matrix.

    Args:
        matrix: Source matrix
        start_row: Starting row index
        start_col: Starting column index
        height: Submatrix height
        width: Submatrix width

    Returns:
        Extracted submatrix

    Raises:
        ValueError: If submatrix extends beyond matrix bounds
    """
    end_row = start_row + height
    end_col = start_col + width

    if (
        start_row < 0
        or start_col < 0
        or end_row > matrix.shape[0]
        or end_col > matrix.shape[1]
    ):
        raise ValueError("Submatrix extends beyond matrix bounds")

    return matrix[start_row:end_row, start_col:end_col].copy()


def place_submatrix(
    matrix: np.ndarray, submatrix: np.ndarray, start_row: int, start_col: int
) -> np.ndarray:
    """Place submatrix into matrix at specified position.

    Args:
        matrix: Target matrix
        submatrix: Submatrix to place
        start_row: Starting row index
        start_col: Starting column index

    Returns:
        Matrix with submatrix placed

    Raises:
        ValueError: If submatrix extends beyond matrix bounds
    """
    result = matrix.copy()
    sub_height, sub_width = submatrix.shape[:2]
    end_row = start_row + sub_height
    end_col = start_col + sub_width

    if (
        start_row < 0
        or start_col < 0
        or end_row > matrix.shape[0]
        or end_col > matrix.shape[1]
    ):
        raise ValueError("Submatrix extends beyond matrix bounds")

    result[start_row:end_row, start_col:end_col] = submatrix
    return result


def apply_mask_pattern(matrix: np.ndarray, pattern: int) -> np.ndarray:
    """Apply mask pattern to matrix.

    Args:
        matrix: Input matrix
        pattern: Mask pattern number (0-7)

    Returns:
        Masked matrix

    Raises:
        ValueError: If pattern number is invalid
    """
    if not (0 <= pattern <= 7):
        raise ValueError(f"Invalid mask pattern: {pattern}")

    result = matrix.copy()
    height, width = matrix.shape[:2]

    for row in range(height):
        for col in range(width):
            mask_condition = False

            if pattern == 0:
                mask_condition = (row + col) % 2 == 0
            elif pattern == 1:
                mask_condition = row % 2 == 0
            elif pattern == 2:
                mask_condition = col % 3 == 0
            elif pattern == 3:
                mask_condition = (row + col) % 3 == 0
            elif pattern == 4:
                mask_condition = (row // 2 + col // 3) % 2 == 0
            elif pattern == 5:
                mask_condition = (row * col) % 2 + (row * col) % 3 == 0
            elif pattern == 6:
                mask_condition = ((row * col) % 2 + (row * col) % 3) % 2 == 0
            elif pattern == 7:
                mask_condition = ((row + col) % 2 + (row * col) % 3) % 2 == 0

            if mask_condition:
                # For binary data, flip the bit; for multi-level, XOR with pattern
                if matrix.dtype == np.bool_ or np.max(matrix) <= 1:
                    result[row, col] = 1 - result[row, col]
                else:
                    result[row, col] = result[row, col] ^ 1

    return result


def calculate_matrix_checksum(matrix: np.ndarray) -> int:
    """Calculate checksum of matrix for verification.

    Args:
        matrix: Input matrix

    Returns:
        Matrix checksum
    """
    # Simple checksum using sum with position weighting
    height, width = matrix.shape[:2]
    checksum = 0

    for row in range(height):
        for col in range(width):
            value = int(matrix[row, col])
            # Weight by position to make checksum more sensitive to changes
            weight = (row + 1) * (col + 1)
            checksum += value * weight

    return checksum % (2**31 - 1)  # Keep within reasonable range


def resize_matrix(matrix: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Resize matrix using nearest neighbor interpolation.

    Args:
        matrix: Input matrix
        new_height: New height
        new_width: New width

    Returns:
        Resized matrix

    Raises:
        ValueError: If new dimensions are invalid
    """
    if new_height <= 0 or new_width <= 0:
        raise ValueError("New dimensions must be positive")

    # Use PIL for resizing with nearest neighbor
    if len(matrix.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(matrix)
        resized_pil = pil_image.resize((new_width, new_height), Image.NEAREST)
        return np.array(resized_pil, dtype=matrix.dtype)
    else:
        # Multi-channel (shouldn't happen often in JABCode, but handle it)
        pil_image = Image.fromarray(matrix)
        resized_pil = pil_image.resize((new_width, new_height), Image.NEAREST)
        return np.array(resized_pil, dtype=matrix.dtype)


def find_pattern_in_matrix(
    matrix: np.ndarray, pattern: np.ndarray
) -> List[Tuple[int, int]]:
    """Find all occurrences of pattern in matrix.

    Args:
        matrix: Matrix to search in
        pattern: Pattern to find

    Returns:
        List of (row, col) positions where pattern starts
    """
    positions = []
    matrix_height, matrix_width = matrix.shape[:2]
    pattern_height, pattern_width = pattern.shape[:2]

    # Search through all possible positions
    for row in range(matrix_height - pattern_height + 1):
        for col in range(matrix_width - pattern_width + 1):
            # Extract submatrix and compare
            submatrix = matrix[row : row + pattern_height, col : col + pattern_width]
            if np.array_equal(submatrix, pattern):
                positions.append((row, col))

    return positions

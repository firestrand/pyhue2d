"""Core data structures for JABCode implementation."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import math
import numpy as np
from PIL import Image


@dataclass
class Point2D:
    """Represents a 2D coordinate point."""

    x: float
    y: float

    def __post_init__(self) -> None:
        """Convert coordinates to float after initialization."""
        # Ensure x and y are always floats
        self.x = float(self.x)
        self.y = float(self.y)

    def distance_to(self, other: "Point2D") -> float:
        """Calculate Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __add__(self, other: "Point2D") -> "Point2D":
        """Add two points (vector addition)."""
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """Subtract two points (vector subtraction)."""
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """Multiply point by scalar."""
        return Point2D(x=self.x * scalar, y=self.y * scalar)

    def __truediv__(self, scalar: float) -> "Point2D":
        """Divide point by scalar."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Point2D(x=self.x / scalar, y=self.y / scalar)

    def __eq__(self, other: object) -> bool:
        """Check equality with another point."""
        if not isinstance(other, Point2D):
            return False
        return self.x == other.x and self.y == other.y

    def equals(self, other: "Point2D", tolerance: float = 1e-9) -> bool:
        """Check equality with tolerance for floating point comparison."""
        return abs(self.x - other.x) <= tolerance and abs(self.y - other.y) <= tolerance

    def magnitude(self) -> float:
        """Calculate magnitude (distance from origin)."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Point2D":
        """Normalize to unit vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return Point2D(x=self.x / mag, y=self.y / mag)

    def dot(self, other: "Point2D") -> float:
        """Calculate dot product with another point."""
        return self.x * other.x + self.y * other.y

    def rotate(self, angle: float) -> "Point2D":
        """Rotate point around origin by angle (in radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_x = self.x * cos_a - self.y * sin_a
        new_y = self.x * sin_a + self.y * cos_a
        return Point2D(x=new_x, y=new_y)

    def rotate_around(self, angle: float, center: "Point2D") -> "Point2D":
        """Rotate point around arbitrary center by angle (in radians)."""
        # Translate to origin, rotate, translate back
        translated = self - center
        rotated = translated.rotate(angle)
        return rotated + center

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple of floats."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to tuple of integers (truncated)."""
        return (int(self.x), int(self.y))

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> "Point2D":
        """Create Point2D from tuple."""
        return cls(x=coords[0], y=coords[1])

    def __str__(self) -> str:
        """String representation of Point2D."""
        return f"Point2D(x={self.x}, y={self.y})"


@dataclass
class EncodedData:
    """Container for encoded data with metadata."""

    data: bytes
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate encoded data parameters after initialization."""
        self._validate_data()
        self._validate_metadata()

    def _validate_data(self) -> None:
        """Validate data is bytes type."""
        if not isinstance(self.data, bytes):
            raise TypeError("Data must be bytes")

    def _validate_metadata(self) -> None:
        """Validate metadata is dict type."""
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be a dictionary")

    def get_size(self) -> int:
        """Get the size of the data in bytes."""
        return len(self.data)

    def is_empty(self) -> bool:
        """Check if the data is empty."""
        return len(self.data) == 0

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key, with optional default."""
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self.metadata[key] = value

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Update multiple metadata values."""
        self.metadata.update(updates)

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata.clear()

    def copy(self) -> "EncodedData":
        """Create a deep copy of the EncodedData."""
        return EncodedData(
            data=self.data,  # bytes are immutable, so no need to copy
            metadata=self.metadata.copy(),  # shallow copy of metadata dict
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize EncodedData to dictionary."""
        return {"data": self.data, "metadata": self.metadata, "size": self.get_size()}

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> "EncodedData":
        """Create EncodedData from dictionary."""
        return cls(data=data_dict["data"], metadata=data_dict["metadata"])

    def __str__(self) -> str:
        """String representation of EncodedData."""
        metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
        return f"EncodedData({self.get_size()} bytes, {metadata_str})"


@dataclass
class Bitmap:
    """Represents image/matrix data using numpy arrays."""

    array: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate bitmap parameters after initialization."""
        self._validate_array()
        self._validate_dimensions()

    def _validate_array(self) -> None:
        """Validate array is a numpy array."""
        if not isinstance(self.array, np.ndarray):
            raise TypeError("Array must be a numpy array")

    def _validate_dimensions(self) -> None:
        """Validate dimensions and array shape match."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")

        # Check array dimensions match width/height
        if len(self.array.shape) == 2:  # Grayscale
            if self.array.shape != (self.height, self.width):
                raise ValueError("Array dimensions don't match width/height")
        elif len(self.array.shape) == 3:  # RGB
            if self.array.shape[:2] != (self.height, self.width):
                raise ValueError("Array dimensions don't match width/height")
        else:
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB)")

    def get_pixel(self, x: int, y: int) -> Union[int, Tuple[int, int, int]]:
        """Get pixel value at coordinates (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel coordinates out of bounds")

        pixel = self.array[y, x]
        if len(self.array.shape) == 2:  # Grayscale
            return int(pixel)
        elif len(self.array.shape) == 3 and self.array.shape[2] >= 3:
            return (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        else:
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB)")

    def set_pixel(
        self, x: int, y: int, value: Union[int, Tuple[int, int, int]]
    ) -> None:
        """Set pixel value at coordinates (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel coordinates out of bounds")

        self.array[y, x] = value

    def is_grayscale(self) -> bool:
        """Check if bitmap is grayscale (2D) or color (3D)."""
        return len(self.array.shape) == 2

    def to_pil_image(self) -> Image.Image:
        """Convert bitmap to PIL Image."""
        if self.is_grayscale():
            return Image.fromarray(self.array, mode="L")
        else:
            return Image.fromarray(self.array, mode="RGB")

    @classmethod
    def from_pil_image(cls, pil_image: Image.Image) -> "Bitmap":
        """Create bitmap from PIL Image."""
        array = np.array(pil_image)
        width, height = pil_image.size
        return cls(array=array, width=width, height=height)

    def resize(self, new_width: int, new_height: int) -> "Bitmap":
        """Resize bitmap to new dimensions."""
        pil_image = self.to_pil_image()
        resized_pil = pil_image.resize((new_width, new_height))
        return self.from_pil_image(resized_pil)

    def copy(self) -> "Bitmap":
        """Create a deep copy of the bitmap."""
        return Bitmap(array=self.array.copy(), width=self.width, height=self.height)


@dataclass
class Symbol:
    """Represents a JAB Code symbol with all its properties."""

    version: int
    color_count: int
    ecc_level: str
    matrix_size: Tuple[int, int]
    finder_patterns: Optional[list] = None
    alignment_patterns: Optional[list] = None

    def __post_init__(self) -> None:
        """Validate symbol parameters after initialization."""
        self._validate_version()
        self._validate_color_count()
        self._validate_ecc_level()
        self._validate_matrix_size()

    def _validate_version(self) -> None:
        """Validate version is within valid range."""
        if not (1 <= self.version <= 32):
            raise ValueError("Version must be between 1 and 32")

    def _validate_color_count(self) -> None:
        """Validate color count is a supported value."""
        valid_colors = [4, 8, 16, 32, 64, 128, 256]
        if self.color_count not in valid_colors:
            raise ValueError(f"Color count must be one of {valid_colors}")

    def _validate_ecc_level(self) -> None:
        """Validate ECC level is supported."""
        valid_levels = ["L", "M", "Q", "H"]
        if self.ecc_level not in valid_levels:
            raise ValueError(f"ECC level must be one of {valid_levels}")

    def _validate_matrix_size(self) -> None:
        """Validate matrix size is valid."""
        if (
            not isinstance(self.matrix_size, tuple)
            or len(self.matrix_size) != 2
            or not all(isinstance(x, int) and x > 0 for x in self.matrix_size)
        ):
            raise ValueError("Matrix size must be a tuple of two positive integers")

    def calculate_capacity(self) -> int:
        """Calculate the data capacity of this symbol."""
        # Simplified capacity calculation based on matrix size and color count
        total_modules = self.matrix_size[0] * self.matrix_size[1]
        # Account for finder patterns, alignment patterns, and error correction
        data_modules = total_modules * 0.7  # Rough estimate
        bits_per_module = {4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8}[
            self.color_count
        ]
        return int(data_modules * bits_per_module / 8)  # Convert to bytes

    def is_master(self) -> bool:
        """Check if this symbol is a master symbol (has finder patterns)."""
        return self.finder_patterns is not None

    def is_slave(self) -> bool:
        """Check if this symbol is a slave symbol (no finder patterns)."""
        return self.finder_patterns is None

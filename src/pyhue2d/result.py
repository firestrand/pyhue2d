"""Structured decode result type for PyHue2D."""

from dataclasses import dataclass


@dataclass
class EncodeResult:
    """Result of an encode symbol operation.

    Attributes:
        matrix: 2D list of color index integers (height, width).
        version: Symbol version number.
        color_count: Number of colors in the symbol palette.
        ecc_level: Error correction level as an integer.
        mask_pattern: Mask pattern index applied.
        width: Symbol width in modules.
        height: Symbol height in modules.
    """

    matrix: list[list[int]]
    version: int = 1
    color_count: int = 8
    ecc_level: int = 3
    mask_pattern: int = 7
    width: int = 21
    height: int = 21
    symbol_count: int = 1


@dataclass
class DecodeResult:
    """Result of a successful barcode decode operation.

    Attributes:
        payload: Decoded payload bytes.
        symbology: Symbology name, e.g. "jabcode".
        version: Symbol version number.
        color_count: Number of colors in the symbol palette.
        ecc_level: Error correction level as an integer.
        mask_pattern: Mask pattern index applied.
        symbol_count: Number of symbols in the barcode.
        corrected_error_count: Number of corrected errors during decoding.
    """

    payload: bytes
    symbology: str = "jabcode"
    version: int = 1
    color_count: int = 8
    ecc_level: int = 3
    mask_pattern: int = 7
    symbol_count: int = 1
    corrected_error_count: int = 0

    @property
    def data(self) -> bytes:
        """Compatibility alias for payload."""
        return self.payload

    def __bytes__(self) -> bytes:
        """Byte conversion returns payload bytes."""
        return self.payload

    def __len__(self) -> int:
        """Length of payload bytes."""
        return len(self.payload)

    def __getitem__(self, item: object) -> object:
        """Index access into payload bytes."""
        return self.payload[item]  # type: ignore[index]

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Decode payload bytes to string."""
        return self.payload.decode(encoding, errors=errors)


@dataclass(frozen=True)
class CapacityResult:
    """Result of a capacity query for a symbol layout.

    Attributes:
        version: Symbol version number (1-32).
        matrix_size: Module dimensions (width, height).
        pixel_size: Image dimensions in pixels (width, height) at the given module_size.
        color_count: Number of colors in the palette.
        ecc_level: Error correction level integer.
        module_size: Pixel size per module.
    """

    version: int
    matrix_size: tuple[int, int]
    pixel_size: tuple[int, int]
    color_count: int = 8
    ecc_level: int = 3
    module_size: int = 12


@dataclass(frozen=True)
class InspectResult:
    """Result of an inspection of a JABCode symbol.

    Attributes:
        matrix_size: Symbol dimensions in modules (width, height).
        encoded_data_hex: Pre-ECC bitstream encoded as hex string.
        symbol_matrix: 2D list of color index integers.
    """

    matrix_size: tuple[int, int]
    encoded_data_hex: str
    symbol_matrix: list[list[int]]

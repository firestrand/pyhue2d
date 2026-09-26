"""Version selection for standards-compliant class-encoder matrices."""

from ..general_encoder import build_general_matrix, payload_bits
from ..ldpc.parameters import LDPCParameters
from ..symbol_layout import reserved_coordinates


def encode_class_matrix(
    data: str | bytes, requested_version: int | str, ecc_level: str, mask_pattern: int
) -> tuple[list[list[int]], int]:
    """Choose capacity from actual byte-mode bits rather than compressed bytes."""
    levels = {"L": 3, "M": 3, "Q": 4, "H": 7}
    level = levels[ecc_level]
    params = LDPCParameters.for_ecc_level(level)
    required = len(payload_bits(data)) + 5
    version = 1
    if requested_version == "auto":
        for candidate in range(1, 33):
            dimension = 17 + 4 * candidate
            default = level == 3 and mask_pattern == 7
            reserved = reserved_coordinates(candidate, candidate, 8, True, default)
            capacity = ((dimension * dimension - len(reserved)) * 3 // params.wr) * (params.wr - params.wc)
            version = candidate
            if capacity >= required:
                break
    else:
        version = int(requested_version)
    return build_general_matrix(data, version, 8, level, mask_pattern), version

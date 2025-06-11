"""LDPC parameters configuration for JABCode error correction."""

from dataclasses import dataclass
from ..constants import ECC_LEVELS


@dataclass(frozen=True)
class LDPCParameters:
    """Configuration parameters for LDPC error correction.

    JABCode uses LDPC codes with specific wc (column weight) and wr (row weight)
    parameters for different error correction levels.
    """

    wc: int
    wr: int
    ecc_level: str = "M"

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate wc (column weight)
        if not isinstance(self.wc, int) or self.wc < 2:
            raise ValueError(f"Column weight (wc) must be an integer >= 2: {self.wc}")

        # Validate wr (row weight)
        if not isinstance(self.wr, int) or self.wr < 2:
            raise ValueError(f"Row weight (wr) must be an integer >= 2: {self.wr}")

        # Validate ECC level
        if self.ecc_level not in ECC_LEVELS:
            raise ValueError(
                f"Invalid ECC level: {self.ecc_level}. Must be one of {ECC_LEVELS}"
            )

    def get_code_rate(self) -> float:
        """Calculate the theoretical code rate.

        Returns:
            Code rate as a float between 0 and 1
        """
        # JABCode reference implementation code rate calculation
        # From the net capacity formula: (wr - wc) / wr
        return (self.wr - self.wc) / self.wr

    def get_parity_ratio(self) -> float:
        """Calculate the parity to data ratio.

        Returns:
            Ratio of parity bits to data bits
        """
        return self.wr / self.wc

    def get_redundancy_factor(self) -> float:
        """Calculate redundancy factor for this configuration.

        Returns:
            Redundancy factor (inverse of code rate)
        """
        code_rate = self.get_code_rate()
        if code_rate <= 0:
            raise ValueError("Invalid code rate for redundancy calculation")
        return 1.0 / code_rate

    def get_ecc_overhead_factor(self) -> float:
        """Get ECC overhead factor based on ECC level.

        Returns:
            Overhead factor for the ECC level
        """
        # ECC level specific overhead factors
        ecc_factors = {
            "L": 1.2,  # 20% overhead for low ECC
            "M": 1.35,  # 35% overhead for medium ECC
            "Q": 1.5,  # 50% overhead for quartile ECC
            "H": 1.8,  # 80% overhead for high ECC
        }
        return ecc_factors.get(self.ecc_level, 1.35)

    def is_valid_configuration(self) -> bool:
        """Check if this is a valid LDPC configuration.

        Returns:
            True if the configuration is theoretically valid
        """
        # Basic validity checks
        # In JABCode reference: wr (row weight) >= wc (column weight)
        # This ensures positive code rate: (wr - wc) / wr >= 0
        if self.wc > self.wr:
            return False  # Column weight should not exceed row weight

        if self.get_code_rate() < 0 or self.get_code_rate() >= 1:
            return False  # Code rate should be between 0 and 1

        return True

    def copy(self) -> "LDPCParameters":
        """Create a copy of these parameters.

        Returns:
            New LDPCParameters instance with same values
        """
        return LDPCParameters(self.wc, self.wr, self.ecc_level)

    def __str__(self) -> str:
        """String representation of LDPC parameters."""
        return f"LDPCParameters(wc={self.wc}, wr={self.wr}, ecc_level={self.ecc_level})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"LDPCParameters(wc={self.wc}, wr={self.wr}, ecc_level='{self.ecc_level}', "
            f"code_rate={self.get_code_rate():.3f})"
        )

    @classmethod
    def for_ecc_level(cls, ecc_level: str) -> "LDPCParameters":
        """Create standard parameters for given ECC level.

        Args:
            ecc_level: Error correction level ("L", "M", "Q", "H")

        Returns:
            LDPCParameters optimized for the ECC level
        """
        # JABCode reference implementation parameters
        # Based on ecclevel2wcwr[11][2] from encoder.h:
        # Level 0: {4,9}, Level 1: {3,8}, Level 2: {3,7}, Level 3: {4,9}
        # Level 4: {3,6}, Level 5: {4,7}, Level 6: {4,6}, Level 7: {3,4}
        # Level 8: {4,5}, Level 9: {5,6}, Level 10: {6,7}
        
        ecc_configs = {
            "L": (4, 9),   # Low error correction (level 0, code rate 0.55)
            "M": (4, 9),   # Medium error correction (level 3, code rate 0.55) - JABCode default
            "Q": (3, 6),   # Quartile error correction (level 4, code rate 0.50)
            "H": (3, 4),   # High error correction (level 7, code rate 0.25)
        }

        if ecc_level not in ecc_configs:
            raise ValueError(f"Invalid ECC level: {ecc_level}")

        wc, wr = ecc_configs[ecc_level]
        return cls(wc, wr, ecc_level)

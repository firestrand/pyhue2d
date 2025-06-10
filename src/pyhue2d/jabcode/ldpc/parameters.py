"""LDPC parameters configuration for JABCode error correction."""

from dataclasses import dataclass
from typing import Optional
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
            raise ValueError(f"Invalid ECC level: {self.ecc_level}. Must be one of {ECC_LEVELS}")
    
    def get_code_rate(self) -> float:
        """Calculate the theoretical code rate.
        
        Returns:
            Code rate as a float between 0 and 1
        """
        # Simplified code rate calculation
        # In practice, this would depend on the specific LDPC matrix construction
        return 1.0 - (self.wr / self.wc)
    
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
            "L": 1.2,   # 20% overhead for low ECC
            "M": 1.35,  # 35% overhead for medium ECC
            "Q": 1.5,   # 50% overhead for quartile ECC
            "H": 1.8,   # 80% overhead for high ECC
        }
        return ecc_factors.get(self.ecc_level, 1.35)
    
    def is_valid_configuration(self) -> bool:
        """Check if this is a valid LDPC configuration.
        
        Returns:
            True if the configuration is theoretically valid
        """
        # Basic validity checks
        if self.wc <= self.wr:
            return False  # Column weight should be greater than row weight
        
        if self.get_code_rate() <= 0 or self.get_code_rate() >= 1:
            return False  # Code rate should be between 0 and 1
        
        return True
    
    def copy(self) -> 'LDPCParameters':
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
        return (f"LDPCParameters(wc={self.wc}, wr={self.wr}, ecc_level='{self.ecc_level}', "
                f"code_rate={self.get_code_rate():.3f})")
    
    @classmethod
    def for_ecc_level(cls, ecc_level: str) -> 'LDPCParameters':
        """Create standard parameters for given ECC level.
        
        Args:
            ecc_level: Error correction level ("L", "M", "Q", "H")
            
        Returns:
            LDPCParameters optimized for the ECC level
        """
        # Standard configurations for each ECC level
        ecc_configs = {
            "L": (4, 2),   # Low error correction
            "M": (6, 3),   # Medium error correction  
            "Q": (8, 4),   # Quartile error correction
            "H": (10, 5),  # High error correction
        }
        
        if ecc_level not in ecc_configs:
            raise ValueError(f"Invalid ECC level: {ecc_level}")
        
        wc, wr = ecc_configs[ecc_level]
        return cls(wc, wr, ecc_level)
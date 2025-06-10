"""Random seed configuration for JABCode LDPC generation."""

import random
from typing import Dict, Any, Iterator
from ..constants import LDPC_METADATA_SEED, LDPC_MESSAGE_SEED


class RandomSeedConfig:
    """Configuration for random seeds used in LDPC matrix generation.

    JABCode uses specific seeds for metadata (38545) and message (785465)
    LDPC matrix generation to ensure deterministic behavior.
    """

    def __init__(
        self,
        metadata_seed: int = LDPC_METADATA_SEED,
        message_seed: int = LDPC_MESSAGE_SEED,
    ):
        """Initialize random seed configuration.

        Args:
            metadata_seed: Seed for metadata LDPC matrix generation
            message_seed: Seed for message LDPC matrix generation

        Raises:
            ValueError: If seeds are invalid
        """
        # Validate metadata seed
        if not isinstance(metadata_seed, int) or metadata_seed <= 0:
            raise ValueError(
                f"Metadata seed must be a positive integer: {metadata_seed}"
            )

        # Validate message seed
        if not isinstance(message_seed, int) or message_seed <= 0:
            raise ValueError(f"Message seed must be a positive integer: {message_seed}")

        self.metadata_seed = metadata_seed
        self.message_seed = message_seed

        # Initialize random number generators
        self._metadata_rng = random.Random(self.metadata_seed)
        self._message_rng = random.Random(self.message_seed)

    def get_metadata_generator(self) -> Iterator[int]:
        """Get a random number generator for metadata LDPC matrix.

        Returns:
            Iterator yielding deterministic random integers
        """
        # Reset to ensure deterministic behavior
        self._metadata_rng = random.Random(self.metadata_seed)
        while True:
            yield self._metadata_rng.randint(0, 2**32 - 1)

    def get_message_generator(self) -> Iterator[int]:
        """Get a random number generator for message LDPC matrix.

        Returns:
            Iterator yielding deterministic random integers
        """
        # Reset to ensure deterministic behavior
        self._message_rng = random.Random(self.message_seed)
        while True:
            yield self._message_rng.randint(0, 2**32 - 1)

    def reset_generators(self) -> None:
        """Reset both generators to their initial seeds."""
        self._metadata_rng = random.Random(self.metadata_seed)
        self._message_rng = random.Random(self.message_seed)

    def get_metadata_random(self, min_val: int = 0, max_val: int = 2**16 - 1) -> int:
        """Get a single random value for metadata operations.

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Random integer in specified range
        """
        return self._metadata_rng.randint(min_val, max_val)

    def get_message_random(self, min_val: int = 0, max_val: int = 2**16 - 1) -> int:
        """Get a single random value for message operations.

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Random integer in specified range
        """
        return self._message_rng.randint(min_val, max_val)

    def copy(self) -> "RandomSeedConfig":
        """Create a copy of this seed configuration.

        Returns:
            New RandomSeedConfig with same seed values
        """
        return RandomSeedConfig(self.metadata_seed, self.message_seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert seed configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {"metadata_seed": self.metadata_seed, "message_seed": self.message_seed}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RandomSeedConfig":
        """Create seed configuration from dictionary.

        Args:
            config_dict: Dictionary containing seed configuration

        Returns:
            New RandomSeedConfig instance

        Raises:
            KeyError: If required keys are missing
            ValueError: If seed values are invalid
        """
        return cls(
            metadata_seed=config_dict["metadata_seed"],
            message_seed=config_dict["message_seed"],
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another seed configuration."""
        if not isinstance(other, RandomSeedConfig):
            return NotImplemented
        return (
            self.metadata_seed == other.metadata_seed
            and self.message_seed == other.message_seed
        )

    def __hash__(self) -> int:
        """Get hash value for use in sets and dictionaries."""
        return hash((self.metadata_seed, self.message_seed))

    def __str__(self) -> str:
        """String representation of seed configuration."""
        return f"RandomSeedConfig(metadata_seed={self.metadata_seed}, message_seed={self.message_seed})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"RandomSeedConfig(metadata_seed={self.metadata_seed}, "
            f"message_seed={self.message_seed})"
        )

    @classmethod
    def create_default(cls) -> "RandomSeedConfig":
        """Create seed configuration with JABCode specification defaults.

        Returns:
            RandomSeedConfig with default JABCode seeds
        """
        return cls(LDPC_METADATA_SEED, LDPC_MESSAGE_SEED)

    def is_default_configuration(self) -> bool:
        """Check if this uses the default JABCode seeds.

        Returns:
            True if using default JABCode specification seeds
        """
        return (
            self.metadata_seed == LDPC_METADATA_SEED
            and self.message_seed == LDPC_MESSAGE_SEED
        )

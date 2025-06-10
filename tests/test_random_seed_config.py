"""Tests for RandomSeedConfig class."""

import pytest
from src.pyhue2d.jabcode.ldpc.seed_config import RandomSeedConfig
from src.pyhue2d.jabcode.constants import LDPC_METADATA_SEED, LDPC_MESSAGE_SEED


class TestRandomSeedConfig:
    """Test suite for RandomSeedConfig class."""
    
    def test_random_seed_config_creation_with_defaults(self):
        """Test that RandomSeedConfig can be created with default seeds."""
        config = RandomSeedConfig()
        assert config.metadata_seed == LDPC_METADATA_SEED
        assert config.message_seed == LDPC_MESSAGE_SEED
    
    def test_random_seed_config_creation_with_custom_seeds(self):
        """Test creating RandomSeedConfig with custom seed values."""
        config = RandomSeedConfig(metadata_seed=12345, message_seed=67890)
        assert config.metadata_seed == 12345
        assert config.message_seed == 67890
    
    def test_random_seed_config_creation_with_default_constants(self):
        """Test creating with JABCode specification default seeds."""
        config = RandomSeedConfig(LDPC_METADATA_SEED, LDPC_MESSAGE_SEED)
        assert config.metadata_seed == LDPC_METADATA_SEED
        assert config.message_seed == LDPC_MESSAGE_SEED
    
    def test_random_seed_config_invalid_metadata_seed(self):
        """Test that invalid metadata seed raises error."""
        with pytest.raises(ValueError):
            RandomSeedConfig(metadata_seed=-1)  # Invalid seed
    
    def test_random_seed_config_invalid_message_seed(self):
        """Test that invalid message seed raises error."""
        with pytest.raises(ValueError):
            RandomSeedConfig(message_seed=-1)  # Invalid seed


class TestRandomSeedConfigImplementation:
    """Test suite for RandomSeedConfig implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def seed_config(self):
        """Create a seed config for testing."""
        try:
            return RandomSeedConfig()
        except NotImplementedError:
            pytest.skip("RandomSeedConfig not yet implemented")
    
    def test_seed_config_creation_with_defaults(self, seed_config):
        """Test that seed config can be created with default values."""
        assert seed_config is not None
        assert hasattr(seed_config, 'metadata_seed')
        assert hasattr(seed_config, 'message_seed')
    
    def test_seed_config_default_metadata_seed(self, seed_config):
        """Test that default metadata seed matches JABCode specification."""
        assert seed_config.metadata_seed == LDPC_METADATA_SEED
        assert isinstance(seed_config.metadata_seed, int)
        assert seed_config.metadata_seed > 0
    
    def test_seed_config_default_message_seed(self, seed_config):
        """Test that default message seed matches JABCode specification."""
        assert seed_config.message_seed == LDPC_MESSAGE_SEED
        assert isinstance(seed_config.message_seed, int)
        assert seed_config.message_seed > 0
    
    def test_seed_config_custom_seeds(self):
        """Test creating seed config with custom seed values."""
        try:
            custom_config = RandomSeedConfig(metadata_seed=12345, message_seed=67890)
            assert custom_config.metadata_seed == 12345
            assert custom_config.message_seed == 67890
        except NotImplementedError:
            pytest.skip("RandomSeedConfig not yet implemented")
    
    def test_seed_config_seeds_are_different(self, seed_config):
        """Test that metadata and message seeds are different values."""
        assert seed_config.metadata_seed != seed_config.message_seed
    
    def test_seed_config_validation_positive_seeds(self):
        """Test that seeds must be positive integers."""
        # Valid positive seeds should work
        try:
            valid_config = RandomSeedConfig(metadata_seed=1000, message_seed=2000)
            assert valid_config.metadata_seed == 1000
            assert valid_config.message_seed == 2000
        except NotImplementedError:
            pytest.skip("RandomSeedConfig not yet implemented")
        
        # Invalid seeds should raise ValueError
        with pytest.raises(ValueError):
            RandomSeedConfig(metadata_seed=-1, message_seed=2000)
        
        with pytest.raises(ValueError):
            RandomSeedConfig(metadata_seed=1000, message_seed=-1)
        
        with pytest.raises(ValueError):
            RandomSeedConfig(metadata_seed=0, message_seed=2000)
    
    def test_seed_config_validation_integer_seeds(self):
        """Test that seeds must be integers."""
        with pytest.raises((ValueError, TypeError)):
            RandomSeedConfig(metadata_seed=12.5, message_seed=2000)
        
        with pytest.raises((ValueError, TypeError)):
            RandomSeedConfig(metadata_seed=1000, message_seed="invalid")
    
    def test_seed_config_get_metadata_generator(self, seed_config):
        """Test getting a random generator for metadata."""
        if hasattr(seed_config, 'get_metadata_generator'):
            generator = seed_config.get_metadata_generator()
            assert generator is not None
            # Should be reproducible
            gen1 = seed_config.get_metadata_generator()
            gen2 = seed_config.get_metadata_generator()
            # First few random numbers should be the same (deterministic)
            assert next(gen1) == next(gen2)
    
    def test_seed_config_get_message_generator(self, seed_config):
        """Test getting a random generator for message data."""
        if hasattr(seed_config, 'get_message_generator'):
            generator = seed_config.get_message_generator()
            assert generator is not None
            # Should be reproducible
            gen1 = seed_config.get_message_generator()
            gen2 = seed_config.get_message_generator()
            # First few random numbers should be the same (deterministic)
            assert next(gen1) == next(gen2)
    
    def test_seed_config_generators_are_different(self, seed_config):
        """Test that metadata and message generators produce different sequences."""
        if hasattr(seed_config, 'get_metadata_generator') and hasattr(seed_config, 'get_message_generator'):
            meta_gen = seed_config.get_metadata_generator()
            msg_gen = seed_config.get_message_generator()
            
            # Should produce different sequences due to different seeds
            meta_vals = [next(meta_gen) for _ in range(10)]
            msg_vals = [next(msg_gen) for _ in range(10)]
            
            assert meta_vals != msg_vals
    
    def test_seed_config_reset_generators(self, seed_config):
        """Test that generators can be reset to initial state."""
        if hasattr(seed_config, 'get_metadata_generator') and hasattr(seed_config, 'reset_generators'):
            gen1 = seed_config.get_metadata_generator()
            first_vals = [next(gen1) for _ in range(5)]
            
            seed_config.reset_generators()
            gen2 = seed_config.get_metadata_generator()
            second_vals = [next(gen2) for _ in range(5)]
            
            assert first_vals == second_vals
    
    def test_seed_config_string_representation(self, seed_config):
        """Test that seed config has a useful string representation."""
        str_repr = str(seed_config)
        assert "RandomSeedConfig" in str_repr
        assert str(seed_config.metadata_seed) in str_repr
        assert str(seed_config.message_seed) in str_repr
    
    def test_seed_config_equality(self):
        """Test that seed configs can be compared for equality."""
        try:
            config1 = RandomSeedConfig(metadata_seed=1000, message_seed=2000)
            config2 = RandomSeedConfig(metadata_seed=1000, message_seed=2000)
            config3 = RandomSeedConfig(metadata_seed=1111, message_seed=2222)
            
            assert config1 == config2
            assert config1 != config3
        except NotImplementedError:
            pytest.skip("RandomSeedConfig not yet implemented")
    
    def test_seed_config_hash(self):
        """Test that seed configs can be hashed."""
        try:
            config1 = RandomSeedConfig(metadata_seed=1000, message_seed=2000)
            config2 = RandomSeedConfig(metadata_seed=1000, message_seed=2000)
            
            # Equal objects should have same hash
            assert hash(config1) == hash(config2)
            
            # Should be able to use in set
            config_set = {config1, config2}
            assert len(config_set) == 1  # Should be deduplicated
        except NotImplementedError:
            pytest.skip("RandomSeedConfig not yet implemented")
    
    def test_seed_config_copy(self, seed_config):
        """Test that seed config can be copied."""
        if hasattr(seed_config, 'copy'):
            copied = seed_config.copy()
            assert copied == seed_config
            assert copied is not seed_config
    
    def test_seed_config_to_dict(self, seed_config):
        """Test that seed config can be converted to dictionary."""
        if hasattr(seed_config, 'to_dict'):
            config_dict = seed_config.to_dict()
            assert isinstance(config_dict, dict)
            assert 'metadata_seed' in config_dict
            assert 'message_seed' in config_dict
            assert config_dict['metadata_seed'] == seed_config.metadata_seed
            assert config_dict['message_seed'] == seed_config.message_seed
    
    def test_seed_config_from_dict(self):
        """Test that seed config can be created from dictionary."""
        if hasattr(RandomSeedConfig, 'from_dict'):
            config_dict = {
                'metadata_seed': 1000,
                'message_seed': 2000
            }
            config = RandomSeedConfig.from_dict(config_dict)
            assert config.metadata_seed == 1000
            assert config.message_seed == 2000
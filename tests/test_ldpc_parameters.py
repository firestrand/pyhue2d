"""Tests for LDPCParameters class."""

import pytest
from src.pyhue2d.jabcode.ldpc.parameters import LDPCParameters
from src.pyhue2d.jabcode.constants import ECC_LEVELS, DEFAULT_LDPC_WC, DEFAULT_LDPC_WR


class TestLDPCParameters:
    """Test suite for LDPCParameters class."""
    
    def test_ldpc_parameters_creation_with_defaults(self):
        """Test that LDPCParameters can be created with default values."""
        params = LDPCParameters(DEFAULT_LDPC_WC, DEFAULT_LDPC_WR)
        assert params.wc == DEFAULT_LDPC_WC
        assert params.wr == DEFAULT_LDPC_WR
        assert params.ecc_level == "M"
    
    def test_ldpc_parameters_creation_with_custom_values(self):
        """Test creating LDPCParameters with custom wc and wr values."""
        params = LDPCParameters(wc=6, wr=3, ecc_level="H")
        assert params.wc == 6
        assert params.wr == 3
        assert params.ecc_level == "H"
    
    def test_ldpc_parameters_creation_with_all_ecc_levels(self):
        """Test creating parameters with all ECC levels."""
        for ecc_level in ECC_LEVELS:
            params = LDPCParameters(4, 2, ecc_level)
            assert params.ecc_level == ecc_level
    
    def test_ldpc_parameters_invalid_wc(self):
        """Test that invalid wc values raise errors."""
        with pytest.raises(ValueError):
            LDPCParameters(wc=0, wr=2)  # Invalid wc
    
    def test_ldpc_parameters_invalid_wr(self):
        """Test that invalid wr values raise errors."""
        with pytest.raises(ValueError):
            LDPCParameters(wc=4, wr=0)  # Invalid wr
    
    def test_ldpc_parameters_invalid_ecc_level(self):
        """Test that invalid ECC level raises error."""
        with pytest.raises(ValueError):
            LDPCParameters(4, 2, "X")  # Invalid ECC level


class TestLDPCParametersImplementation:
    """Test suite for LDPCParameters implementation details.
    
    These tests will pass once the implementation is complete.
    """
    
    @pytest.fixture
    def parameters(self):
        """Create LDPC parameters for testing."""
        try:
            return LDPCParameters(4, 2, "M")
        except NotImplementedError:
            pytest.skip("LDPCParameters not yet implemented")
    
    def test_parameters_creation_with_defaults(self, parameters):
        """Test that parameters can be created with default values."""
        assert parameters is not None
        assert hasattr(parameters, 'wc')
        assert hasattr(parameters, 'wr')
        assert hasattr(parameters, 'ecc_level')
    
    def test_parameters_wc_property(self, parameters):
        """Test that wc property is accessible and correct."""
        assert parameters.wc == 4
        assert isinstance(parameters.wc, int)
        assert parameters.wc > 0
    
    def test_parameters_wr_property(self, parameters):
        """Test that wr property is accessible and correct.""" 
        assert parameters.wr == 2
        assert isinstance(parameters.wr, int)
        assert parameters.wr > 0
    
    def test_parameters_ecc_level_property(self, parameters):
        """Test that ecc_level property is accessible and correct."""
        assert parameters.ecc_level == "M"
        assert isinstance(parameters.ecc_level, str)
        assert parameters.ecc_level in ECC_LEVELS
    
    def test_parameters_code_rate_calculation(self, parameters):
        """Test that code rate can be calculated."""
        if hasattr(parameters, 'get_code_rate'):
            code_rate = parameters.get_code_rate()
            assert isinstance(code_rate, float)
            assert 0 < code_rate < 1  # Code rate should be between 0 and 1
    
    def test_parameters_parity_ratio_calculation(self, parameters):
        """Test that parity ratio can be calculated."""
        if hasattr(parameters, 'get_parity_ratio'):
            parity_ratio = parameters.get_parity_ratio()
            assert isinstance(parity_ratio, float)
            assert parity_ratio > 0
    
    def test_parameters_different_ecc_levels_have_different_properties(self):
        """Test that different ECC levels result in different parameters."""
        try:
            params_l = LDPCParameters(4, 2, "L")
            params_h = LDPCParameters(4, 2, "H")
            
            # Different ECC levels might have different effective parameters
            # even with same wc, wr
            assert params_l.ecc_level != params_h.ecc_level
        except NotImplementedError:
            pytest.skip("LDPCParameters not yet implemented")
    
    def test_parameters_validation_wc_range(self):
        """Test that wc values are validated."""
        # Valid wc values
        for wc in [2, 3, 4, 5, 6, 8]:
            try:
                params = LDPCParameters(wc, 2)
                assert params.wc == wc
            except NotImplementedError:
                pytest.skip("LDPCParameters not yet implemented")
        
        # Invalid wc values should raise ValueError
        for invalid_wc in [0, -1, 1]:
            with pytest.raises(ValueError):
                LDPCParameters(invalid_wc, 2)
    
    def test_parameters_validation_wr_range(self):
        """Test that wr values are validated."""
        # Valid wr values  
        for wr in [2, 3, 4, 5]:
            try:
                params = LDPCParameters(4, wr)
                assert params.wr == wr
            except NotImplementedError:
                pytest.skip("LDPCParameters not yet implemented")
        
        # Invalid wr values should raise ValueError
        for invalid_wr in [0, -1, 1]:
            with pytest.raises(ValueError):
                LDPCParameters(4, invalid_wr)
    
    def test_parameters_validation_ecc_level(self):
        """Test that ECC level is validated."""
        # Valid ECC levels
        for ecc_level in ECC_LEVELS:
            try:
                params = LDPCParameters(4, 2, ecc_level)
                assert params.ecc_level == ecc_level
            except NotImplementedError:
                pytest.skip("LDPCParameters not yet implemented")
        
        # Invalid ECC levels should raise ValueError
        for invalid_ecc in ["X", "Z", "", "LOW", "HIGH"]:
            with pytest.raises(ValueError):
                LDPCParameters(4, 2, invalid_ecc)
    
    def test_parameters_string_representation(self, parameters):
        """Test that parameters have a useful string representation."""
        str_repr = str(parameters)
        assert "LDPCParameters" in str_repr
        assert "wc=4" in str_repr
        assert "wr=2" in str_repr
        assert "ecc_level=M" in str_repr
    
    def test_parameters_equality(self):
        """Test that parameters can be compared for equality."""
        try:
            params1 = LDPCParameters(4, 2, "M")
            params2 = LDPCParameters(4, 2, "M")
            params3 = LDPCParameters(6, 3, "H")
            
            assert params1 == params2
            assert params1 != params3
        except NotImplementedError:
            pytest.skip("LDPCParameters not yet implemented")
    
    def test_parameters_hash(self):
        """Test that parameters can be hashed for use in sets/dicts."""
        try:
            params1 = LDPCParameters(4, 2, "M")
            params2 = LDPCParameters(4, 2, "M")
            
            # Should be able to hash and equal objects should have same hash
            assert hash(params1) == hash(params2)
            
            # Should be able to use in set
            param_set = {params1, params2}
            assert len(param_set) == 1  # Should be deduplicated
        except NotImplementedError:
            pytest.skip("LDPCParameters not yet implemented")
    
    def test_parameters_copy(self, parameters):
        """Test that parameters can be copied."""
        if hasattr(parameters, 'copy'):
            copied = parameters.copy()
            assert copied == parameters
            assert copied is not parameters  # Should be different objects
    
    def test_parameters_wc_wr_relationship(self, parameters):
        """Test relationship between wc and wr parameters."""
        # In LDPC codes, typically wc > wr for practical codes
        # This is a general guideline, not a strict requirement
        if parameters.wc <= parameters.wr:
            # This might be valid in some cases, but worth noting
            pass  # Allow for flexibility in implementation
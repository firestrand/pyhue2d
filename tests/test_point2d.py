"""Tests for Point2D class."""

import pytest
import math
from pyhue2d.jabcode.core import Point2D


class TestPoint2D:
    """Test cases for Point2D class."""
    
    def test_point2d_creation_with_integers(self):
        """Test Point2D can be created with integer coordinates."""
        point = Point2D(x=10, y=20)
        
        assert point.x == 10.0
        assert point.y == 20.0
    
    def test_point2d_creation_with_floats(self):
        """Test Point2D can be created with float coordinates."""
        point = Point2D(x=3.14, y=2.71)
        
        assert point.x == 3.14
        assert point.y == 2.71
    
    def test_point2d_creation_with_negative_coordinates(self):
        """Test Point2D can be created with negative coordinates."""
        point = Point2D(x=-5.5, y=-10.0)
        
        assert point.x == -5.5
        assert point.y == -10.0
    
    def test_point2d_creation_with_zero(self):
        """Test Point2D can be created with zero coordinates."""
        point = Point2D(x=0, y=0)
        
        assert point.x == 0.0
        assert point.y == 0.0
    
    def test_point2d_distance_to_point(self):
        """Test Point2D can calculate distance to another point."""
        point1 = Point2D(x=0, y=0)
        point2 = Point2D(x=3, y=4)
        
        distance = point1.distance_to(point2)
        
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_point2d_distance_to_same_point(self):
        """Test Point2D distance to itself is zero."""
        point = Point2D(x=5, y=10)
        
        distance = point.distance_to(point)
        
        assert distance == 0.0
    
    def test_point2d_distance_to_with_floats(self):
        """Test Point2D distance calculation with float coordinates."""
        point1 = Point2D(x=1.0, y=1.0)
        point2 = Point2D(x=4.0, y=5.0)
        
        distance = point1.distance_to(point2)
        expected = math.sqrt((4-1)**2 + (5-1)**2)  # sqrt(9 + 16) = 5
        
        assert abs(distance - expected) < 1e-10
    
    def test_point2d_add_points(self):
        """Test Point2D addition operation."""
        point1 = Point2D(x=2, y=3)
        point2 = Point2D(x=4, y=5)
        
        result = point1 + point2
        
        assert result.x == 6.0
        assert result.y == 8.0
        assert isinstance(result, Point2D)
    
    def test_point2d_subtract_points(self):
        """Test Point2D subtraction operation."""
        point1 = Point2D(x=10, y=7)
        point2 = Point2D(x=3, y=2)
        
        result = point1 - point2
        
        assert result.x == 7.0
        assert result.y == 5.0
        assert isinstance(result, Point2D)
    
    def test_point2d_multiply_by_scalar(self):
        """Test Point2D multiplication by scalar."""
        point = Point2D(x=3, y=4)
        
        result = point * 2.5
        
        assert result.x == 7.5
        assert result.y == 10.0
        assert isinstance(result, Point2D)
    
    def test_point2d_divide_by_scalar(self):
        """Test Point2D division by scalar."""
        point = Point2D(x=10, y=8)
        
        result = point / 2
        
        assert result.x == 5.0
        assert result.y == 4.0
        assert isinstance(result, Point2D)
    
    def test_point2d_divide_by_zero(self):
        """Test Point2D division by zero raises error."""
        point = Point2D(x=5, y=10)
        
        with pytest.raises(ZeroDivisionError):
            point / 0
    
    def test_point2d_equality(self):
        """Test Point2D equality comparison."""
        point1 = Point2D(x=3.0, y=4.0)
        point2 = Point2D(x=3.0, y=4.0)
        point3 = Point2D(x=3.1, y=4.0)
        
        assert point1 == point2
        assert point1 != point3
        assert point2 != point3
    
    def test_point2d_equality_with_tolerance(self):
        """Test Point2D equality with floating point tolerance."""
        point1 = Point2D(x=1.0, y=2.0)
        point2 = Point2D(x=1.0000001, y=2.0000001)
        
        assert point1.equals(point2, tolerance=1e-6) is True
        assert point1.equals(point2, tolerance=1e-8) is False
    
    def test_point2d_magnitude(self):
        """Test Point2D can calculate its magnitude (distance from origin)."""
        point = Point2D(x=3, y=4)
        
        magnitude = point.magnitude()
        
        assert magnitude == 5.0
    
    def test_point2d_magnitude_zero(self):
        """Test Point2D magnitude of zero point."""
        point = Point2D(x=0, y=0)
        
        magnitude = point.magnitude()
        
        assert magnitude == 0.0
    
    def test_point2d_normalize(self):
        """Test Point2D normalization to unit vector."""
        point = Point2D(x=3, y=4)
        
        normalized = point.normalize()
        
        assert abs(normalized.x - 0.6) < 1e-10  # 3/5
        assert abs(normalized.y - 0.8) < 1e-10  # 4/5
        assert abs(normalized.magnitude() - 1.0) < 1e-10
    
    def test_point2d_normalize_zero_vector(self):
        """Test Point2D normalization of zero vector raises error."""
        point = Point2D(x=0, y=0)
        
        with pytest.raises(ValueError, match="Cannot normalize zero vector"):
            point.normalize()
    
    def test_point2d_dot_product(self):
        """Test Point2D dot product calculation."""
        point1 = Point2D(x=2, y=3)
        point2 = Point2D(x=4, y=5)
        
        dot = point1.dot(point2)
        
        assert dot == 23.0  # 2*4 + 3*5 = 8 + 15 = 23
    
    def test_point2d_dot_product_perpendicular(self):
        """Test Point2D dot product of perpendicular vectors is zero."""
        point1 = Point2D(x=1, y=0)
        point2 = Point2D(x=0, y=1)
        
        dot = point1.dot(point2)
        
        assert dot == 0.0
    
    def test_point2d_rotate(self):
        """Test Point2D rotation around origin."""
        point = Point2D(x=1, y=0)
        
        # Rotate 90 degrees counterclockwise
        rotated = point.rotate(math.pi / 2)
        
        assert abs(rotated.x - 0.0) < 1e-10
        assert abs(rotated.y - 1.0) < 1e-10
    
    def test_point2d_rotate_around_point(self):
        """Test Point2D rotation around arbitrary point."""
        point = Point2D(x=2, y=1)
        center = Point2D(x=1, y=1)
        
        # Rotate 90 degrees around center
        rotated = point.rotate_around(math.pi / 2, center)
        
        assert abs(rotated.x - 1.0) < 1e-10
        assert abs(rotated.y - 2.0) < 1e-10
    
    def test_point2d_to_tuple(self):
        """Test Point2D conversion to tuple."""
        point = Point2D(x=3.5, y=7.2)
        
        result = point.to_tuple()
        
        assert result == (3.5, 7.2)
        assert isinstance(result, tuple)
    
    def test_point2d_to_int_tuple(self):
        """Test Point2D conversion to integer tuple."""
        point = Point2D(x=3.7, y=7.2)
        
        result = point.to_int_tuple()
        
        assert result == (3, 7)
        assert isinstance(result, tuple)
        assert all(isinstance(x, int) for x in result)
    
    def test_point2d_from_tuple(self):
        """Test Point2D creation from tuple."""
        point = Point2D.from_tuple((5.5, 10.0))
        
        assert point.x == 5.5
        assert point.y == 10.0
    
    def test_point2d_string_representation(self):
        """Test Point2D string representation."""
        point = Point2D(x=3.0, y=4.5)
        
        str_repr = str(point)
        
        assert "Point2D" in str_repr
        assert "3.0" in str_repr
        assert "4.5" in str_repr
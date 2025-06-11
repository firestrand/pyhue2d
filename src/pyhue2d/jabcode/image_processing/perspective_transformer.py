"""Perspective Transformer for JABCode image processing.

This module provides the PerspectiveTransformer class for correcting
perspective distortion in JABCode images based on the JABCode reference
implementation's perspective transformation algorithms.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from PIL import Image
from ..core import Point2D


class PerspectiveTransform:
    """Represents a 3x3 perspective transformation matrix.
    
    Based on the JABCode reference implementation's jab_perspective_transform structure.
    """
    
    def __init__(self, matrix: Optional[np.ndarray] = None):
        """Initialize perspective transformation.
        
        Args:
            matrix: 3x3 transformation matrix, defaults to identity
        """
        if matrix is None:
            self.matrix = np.eye(3, dtype=np.float64)
        else:
            if matrix.shape != (3, 3):
                raise ValueError("Transformation matrix must be 3x3")
            self.matrix = matrix.astype(np.float64)
    
    @property
    def a11(self) -> float:
        return self.matrix[0, 0]
    
    @property 
    def a12(self) -> float:
        return self.matrix[0, 1]
        
    @property
    def a13(self) -> float:
        return self.matrix[0, 2]
        
    @property
    def a21(self) -> float:
        return self.matrix[1, 0]
        
    @property
    def a22(self) -> float:
        return self.matrix[1, 1]
        
    @property
    def a23(self) -> float:
        return self.matrix[1, 2]
        
    @property
    def a31(self) -> float:
        return self.matrix[2, 0]
        
    @property
    def a32(self) -> float:
        return self.matrix[2, 1]
        
    @property
    def a33(self) -> float:
        return self.matrix[2, 2]


class PerspectiveTransformer:
    """Perspective Transformer for JABCode image processing.
    
    Implements perspective correction algorithms based on the JABCode
    reference implementation, providing quadrilateral-to-quadrilateral
    transformations for correcting perspective distortion.
    """
    
    def __init__(self):
        """Initialize perspective transformer."""
        pass
    
    def square_to_quad(self, quad_points: List[Point2D]) -> PerspectiveTransform:
        """Calculate transformation from unit square to quadrilateral.
        
        Based on the JABCode reference implementation's square2Quad function.
        Maps unit square (0,0)-(1,1) to arbitrary quadrilateral.
        
        Args:
            quad_points: List of 4 points defining target quadrilateral
                        in order: top-left, top-right, bottom-right, bottom-left
            
        Returns:
            PerspectiveTransform object
            
        Raises:
            ValueError: If quad_points doesn't contain exactly 4 points
        """
        if len(quad_points) != 4:
            raise ValueError("Quadrilateral must have exactly 4 points")
        
        x0, y0 = quad_points[0].x, quad_points[0].y  # top-left
        x1, y1 = quad_points[1].x, quad_points[1].y  # top-right  
        x2, y2 = quad_points[2].x, quad_points[2].y  # bottom-right
        x3, y3 = quad_points[3].x, quad_points[3].y  # bottom-left
        
        # Calculate differences
        dx1 = x1 - x2
        dx2 = x3 - x2
        dx3 = x0 - x1 + x2 - x3
        dy1 = y1 - y2
        dy2 = y3 - y2
        dy3 = y0 - y1 + y2 - y3
        
        # Check if transformation is affine (projective terms are zero)
        if abs(dx3) < 1e-10 and abs(dy3) < 1e-10:
            # Affine transformation
            matrix = np.array([
                [x1 - x0, x3 - x0, x0],
                [y1 - y0, y3 - y0, y0],
                [0.0, 0.0, 1.0]
            ])
        else:
            # Projective transformation
            denominator = dx1 * dy2 - dx2 * dy1
            if abs(denominator) < 1e-10:
                raise ValueError("Degenerate quadrilateral - cannot compute transformation")
            
            a13 = (dx3 * dy2 - dx2 * dy3) / denominator
            a23 = (dx1 * dy3 - dx3 * dy1) / denominator
            
            matrix = np.array([
                [x1 - x0 + a13 * x1, x3 - x0 + a23 * x3, x0],
                [y1 - y0 + a13 * y1, y3 - y0 + a23 * y3, y0],
                [a13, a23, 1.0]
            ])
        
        return PerspectiveTransform(matrix)
    
    def quad_to_square(self, quad_points: List[Point2D]) -> PerspectiveTransform:
        """Calculate transformation from quadrilateral to unit square.
        
        Based on the JABCode reference implementation's quad2Square function.
        Computes the adjugate matrix of the square2Quad transformation.
        
        Args:
            quad_points: List of 4 points defining source quadrilateral
            
        Returns:
            PerspectiveTransform object (inverse of square_to_quad)
        """
        # Get the forward transformation
        forward_transform = self.square_to_quad(quad_points)
        matrix = forward_transform.matrix
        
        # Compute adjugate matrix (transpose of cofactor matrix)
        # For 3x3 matrix, adjugate elements are:
        adjugate = np.zeros((3, 3))
        
        adjugate[0, 0] = matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]
        adjugate[0, 1] = matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]
        adjugate[0, 2] = matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]
        
        adjugate[1, 0] = matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]
        adjugate[1, 1] = matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]
        adjugate[1, 2] = matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]
        
        adjugate[2, 0] = matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]
        adjugate[2, 1] = matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]
        adjugate[2, 2] = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        
        return PerspectiveTransform(adjugate)
    
    def perspective_transform(self, src_quad: List[Point2D], dest_quad: List[Point2D]) -> PerspectiveTransform:
        """Calculate transformation from one quadrilateral to another.
        
        Based on the JABCode reference implementation's perspectiveTransform function.
        Composes quad2Square and square2Quad transformations.
        
        Args:
            src_quad: Source quadrilateral points
            dest_quad: Destination quadrilateral points
            
        Returns:
            PerspectiveTransform object
        """
        # Transform source quad to unit square
        q2s = self.quad_to_square(src_quad)
        
        # Transform unit square to destination quad
        s2q = self.square_to_quad(dest_quad)
        
        # Compose transformations: result = s2q * q2s
        result_matrix = np.dot(s2q.matrix, q2s.matrix)
        
        return PerspectiveTransform(result_matrix)
    
    def get_jabcode_perspective_transform(self, finder_patterns: List[Point2D], 
                                        symbol_size: Tuple[int, int]) -> PerspectiveTransform:
        """Get perspective transformation for JABCode symbol.
        
        Based on the JABCode reference implementation's getPerspectiveTransform function.
        Maps from canonical JABCode symbol coordinates to detected finder pattern positions.
        
        Args:
            finder_patterns: List of 4 finder pattern center points
                           in order: top-left, top-right, bottom-right, bottom-left
            symbol_size: (width, height) of symbol in modules
            
        Returns:
            PerspectiveTransform for correcting symbol perspective
        """
        if len(finder_patterns) != 4:
            raise ValueError("JABCode requires exactly 4 finder patterns")
        
        width, height = symbol_size
        
        # Source coordinates: canonical symbol with 3.5-pixel offset
        # This accounts for the finder pattern structure in JABCode
        src_quad = [
            Point2D(3.5, 3.5),                    # top-left
            Point2D(width - 3.5, 3.5),           # top-right
            Point2D(width - 3.5, height - 3.5),  # bottom-right
            Point2D(3.5, height - 3.5)           # bottom-left
        ]
        
        # Destination coordinates: detected finder pattern positions
        dest_quad = finder_patterns
        
        return self.perspective_transform(src_quad, dest_quad)
    
    def warp_points(self, transform: PerspectiveTransform, 
                   points: Union[List[Point2D], np.ndarray]) -> List[Point2D]:
        """Apply perspective transformation to array of points.
        
        Based on the JABCode reference implementation's warpPoints function.
        
        Args:
            transform: PerspectiveTransform to apply
            points: Points to transform
            
        Returns:
            List of transformed points
        """
        if isinstance(points, list):
            # Convert Point2D list to numpy array
            point_array = np.array([[p.x, p.y] for p in points])
        else:
            point_array = points
        
        if point_array.size == 0:
            return []
        
        # Ensure we have Nx2 array
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        
        # Add homogeneous coordinate (z=1)
        num_points = point_array.shape[0]
        homogeneous = np.ones((num_points, 3))
        homogeneous[:, :2] = point_array
        
        # Apply transformation
        transformed = np.dot(transform.matrix, homogeneous.T).T
        
        # Convert back to Cartesian coordinates
        result_points = []
        for i in range(num_points):
            if abs(transformed[i, 2]) < 1e-10:
                # Avoid division by zero
                result_points.append(Point2D(transformed[i, 0], transformed[i, 1]))
            else:
                x = transformed[i, 0] / transformed[i, 2]
                y = transformed[i, 1] / transformed[i, 2] 
                result_points.append(Point2D(x, y))
        
        return result_points
    
    def sample_symbol_with_perspective(self, image: Image.Image, 
                                     transform: PerspectiveTransform,
                                     symbol_size: Tuple[int, int]) -> np.ndarray:
        """Sample JABCode symbol with perspective correction applied.
        
        Based on the JABCode reference implementation's sampleSymbol function.
        Samples at module center positions with 3x3 neighborhood averaging.
        
        Args:
            image: Source image to sample from
            transform: Perspective transformation to apply
            symbol_size: (width, height) of symbol in modules
            
        Returns:
            2D numpy array of sampled symbol data
        """
        width, height = symbol_size
        image_array = np.array(image)
        
        # Handle different image modes
        if len(image_array.shape) == 3:
            # RGB image - convert to grayscale for sampling
            grayscale = np.mean(image_array, axis=2)
        else:
            # Already grayscale
            grayscale = image_array
        
        img_height, img_width = grayscale.shape
        sampled_symbol = np.zeros((height, width), dtype=np.uint8)
        
        # Sample at module center positions (offset by 0.5)
        for row in range(height):
            for col in range(width):
                # Module center position
                module_center = Point2D(col + 0.5, row + 0.5)
                
                # Transform to image coordinates
                transformed_points = self.warp_points(transform, [module_center])
                img_x, img_y = transformed_points[0].x, transformed_points[0].y
                
                # Sample with 3x3 neighborhood averaging for robustness
                total_value = 0
                sample_count = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        sample_x = int(img_x + dx)
                        sample_y = int(img_y + dy)
                        
                        # Check bounds
                        if (0 <= sample_x < img_width and 
                            0 <= sample_y < img_height):
                            total_value += grayscale[sample_y, sample_x]
                            sample_count += 1
                
                # Average the samples
                if sample_count > 0:
                    sampled_symbol[row, col] = total_value // sample_count
                else:
                    # Out of bounds - use background value
                    sampled_symbol[row, col] = 255
        
        return sampled_symbol
"""Finder Pattern Detector for JABCode image processing.

This module provides the FinderPatternDetector class for detecting JABCode
finder patterns in images using template matching, contour detection, and
pattern validation algorithms.
"""

import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import feature, measure, morphology

from ..constants import FinderPatternType
from ..core import Point2D
from ..patterns import FinderPatternGenerator


class FinderPatternDetector:
    """Finder Pattern Detector for JABCode image processing.

    The FinderPatternDetector identifies JABCode finder patterns (FP0, FP1, FP2, FP3)
    in images using various detection methods including template matching,
    contour detection, and pattern validation.
    """

    DEFAULT_SETTINGS = {
        "detection_method": "scanline",
        "template_threshold": 0.7,
        "min_pattern_size": 10,
        "max_pattern_size": 100,
        "contour_min_area": 100,
        "contour_max_area": 10000,
        "aspect_ratio_tolerance": 0.3,
        "multi_scale_detection": True,
        "scale_factors": [0.5, 0.75, 1.0, 1.25, 1.5],
        "noise_reduction": True,
        "quality_threshold": 0.5,
        # Optimization settings for large images
        "large_image_threshold": 1000000,  # pixels (1000x1000)
        "large_image_skip_factor": 2,  # Skip every N pixels for large images
        "max_patterns_per_symbol": 100,  # Limit patterns to prevent memory issues
    }

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize Finder Pattern Detector.

        Args:
            settings: Optional configuration settings
        """
        self.settings = self.DEFAULT_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        # Supported detection methods (aliases included for backward compatibility)
        self.detection_methods = [
            "template_matching",
            "contour_detection",  # Preferred explicit name
            "contour",  # Alias for contour_detection
            "hybrid",
            "scanline",
        ]

        # Validate settings
        self._validate_settings()

        # Create finder pattern templates
        self._create_templates()

        # Detection statistics
        self._stats = {
            "total_detections": 0,
            "total_detection_time": 0.0,
            "detection_times": [],
            "patterns_found": 0,
        }

    def find_patterns(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """Find finder patterns in an image.

        Args:
            image: PIL Image object or numpy array

        Returns:
            List of detected patterns with metadata

        Raises:
            ValueError: For invalid inputs
        """
        start_time = time.time()

        try:
            if image is None:
                raise ValueError("Input image cannot be None")
            if not (isinstance(image, Image.Image) or isinstance(image, np.ndarray)):
                raise TypeError(f"Unsupported image type: {type(image)}")

            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                if image.size == (0, 0):
                    raise ValueError("Image cannot be empty")
                # Handle RGBA by compositing onto white
                if image.mode == "RGBA":
                    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(bg, image).convert("RGB")
                elif image.mode not in ["RGB", "L"]:
                    image = image.convert("RGB")
                array = np.array(image)
            elif isinstance(image, np.ndarray):
                array = image.copy()
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Validate array
            if array.size == 0:
                raise ValueError("Array cannot be empty")

            # Check array dimensions
            if array.ndim not in [2, 3]:
                raise ValueError(f"Array must be 2D or 3D, got {array.ndim}D array with shape {array.shape}")

            if array.ndim == 3 and array.shape[2] not in [1, 3, 4]:
                raise ValueError(f"3D array must have 1, 3, or 4 channels, got {array.shape[2]} channels")

            if array.shape[0] < 10 or array.shape[1] < 10:
                return []  # Too small to contain patterns

            method = self.settings["detection_method"]
            if method == "template_matching":
                patterns = self._detect_by_template_matching(array)
            elif method == "scanline":
                patterns = self._detect_by_scanline(array)
            else:
                # For contour detection, convert to grayscale
                if array.ndim == 3:
                    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = array
                # Normalize alias "contour" to "contour_detection"
                if method in ("contour_detection", "contour"):
                    patterns = self._detect_by_contour_detection(gray)
                elif method == "hybrid":
                    patterns = self._detect_hybrid(gray)
                else:
                    raise ValueError(f"Unsupported detection method: {method}")

            # Filter by size and quality
            if method not in ["scanline"]:
                patterns = self._filter_patterns(patterns, array if method == "template_matching" else gray)

            # Update statistics
            detection_time = time.time() - start_time
            self._update_stats(detection_time, len(patterns))

            return patterns

        except Exception as e:
            raise ValueError(f"Failed to detect patterns: {str(e)}") from e

    def find_patterns_binary(self, binary_array: np.ndarray) -> List[Dict[str, Any]]:
        """Find finder patterns in binary array.

        Args:
            binary_array: Binary numpy array (0 and 255 values)

        Returns:
            List of detected patterns
        Raises:
            ValueError: For invalid inputs
        """
        if binary_array is None:
            raise ValueError("Input must be a numpy array")
        if not isinstance(binary_array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if binary_array.ndim != 2:
            raise ValueError("Binary array must be 2D")

        # Use contour detection for binary images
        return self._detect_by_contour_detection(binary_array)

    def validate_pattern(self, pattern_region: np.ndarray, center: Point2D) -> bool:
        """Validate if a region contains a valid finder pattern.

        Args:
            pattern_region: Image region to validate
            center: Center point of the pattern

        Returns:
            True if valid pattern, False otherwise
        """
        try:
            if pattern_region.size == 0:
                return False
            h, w = pattern_region.shape[:2]
            if h < 5 or w < 5:
                return False

            # JABCode patterns have complex internal structure
            # Check for reasonable contrast and variation in the pattern
            min_val = np.min(pattern_region)
            max_val = np.max(pattern_region)
            contrast = max_val - min_val

            # Must have significant contrast (both black and white regions)
            if contrast < 100:  # Not enough contrast
                return False

            # Check for structured variation (not just noise)
            # Count distinct regions by checking transitions
            binary = (pattern_region < 128).astype(np.uint8)

            # Count horizontal transitions in middle row
            mid_row = binary[h // 2, :]
            h_transitions = np.sum(np.diff(mid_row) != 0)

            # Count vertical transitions in middle column
            mid_col = binary[:, w // 2]
            v_transitions = np.sum(np.diff(mid_col) != 0)

            # Should have some transitions indicating structure, but not too many (noise)
            total_transitions = h_transitions + v_transitions
            if not (2 <= total_transitions <= 20):  # Broader range, but exclude extreme noise
                return False

            # Additional validation: check for pattern-like structure
            # JABCode finder patterns should have some regularity, not random noise

            # Check for excessive noise by looking at local variation
            # Detect purely random patterns vs structured patterns

            # Method 1: Check noise level with smoothing (if scipy available)
            try:
                from scipy import ndimage

                smoothed = ndimage.gaussian_filter(pattern_region.astype(float), sigma=1.0)
                noise_level = np.mean(np.abs(pattern_region.astype(float) - smoothed))

                # Too much noise indicates random pattern rather than structured finder pattern
                if (
                    noise_level > 60
                ):  # strict threshold
                    return False
            except ImportError:
                # Method 2: Alternative noise detection without scipy
                # Calculate local variation using standard deviation
                variation = np.std(pattern_region)
                if variation > 80:  # High variation typically indicates noise
                    return False

            # Method 3: Check for excessive transitions (noise characteristic)
            # Random noise typically has many more transitions than structured patterns
            # JABCode patterns can have up to ~12 transitions, so allow some margin
            if total_transitions > 16:  # Too many transitions likely indicates noise
                return False

            # Check for some basic structure rather than pure noise
            # Count the percentage of each binary value to ensure it's not uniform
            black_pixels = np.sum(binary == 0)
            white_pixels = np.sum(binary == 1)
            total_pixels = black_pixels + white_pixels

            if total_pixels > 0:
                black_ratio = black_pixels / total_pixels
                white_ratio = white_pixels / total_pixels

                # Pattern should have both black and white regions (not all one color)
                # Allow for a very broad range as JABCode patterns can have varying ratios
                # including patterns that are mostly one color with small amounts of the other
                if black_ratio < 0.02 or white_ratio < 0.02:  # Very lenient threshold for varied patterns
                    return False

            # Additional check: detect random noise vs structured patterns
            # Random noise has very different characteristics from structured patterns
            if contrast == 255:  # Full contrast range often indicates random noise
                # Check if pattern has too many single-pixel features (characteristic of noise)
                # Use erosion/dilation to detect isolated pixels
                try:
                    import cv2

                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(binary, kernel, iterations=1)
                    dilated = cv2.dilate(eroded, kernel, iterations=1)
                    isolated_pixels = np.sum(binary != dilated) / total_pixels

                    # Random noise typically has many isolated pixels
                    if isolated_pixels > 0.05:  # Very sensitive isolated pixel threshold
                        return False
                except ImportError:
                    # Fallback: check for excessive small regions
                    # Count 2x2 blocks that are uniform vs mixed
                    uniform_blocks = 0
                    total_blocks = 0
                    for i in range(0, h - 1, 2):
                        for j in range(0, w - 1, 2):
                            block = binary[i : i + 2, j : j + 2]
                            if block.size == 4:
                                total_blocks += 1
                                if np.all(block == block[0, 0]):  # All same value
                                    uniform_blocks += 1

                    if total_blocks > 0:
                        uniform_ratio = uniform_blocks / total_blocks
                        # Structured patterns should have more uniform blocks than pure noise
                        if uniform_ratio < 0.1:  # Too few uniform blocks suggests noise
                            return False

            return True

        except Exception:
            return False

    def detect_pattern_orientation(self, pattern_array: np.ndarray) -> float:
        """Detect pattern orientation in degrees.

        Args:
            pattern_array: Pattern region array

        Returns:
            Orientation angle in degrees (0-360)
        """
        try:
            # Use principal component analysis for orientation
            # Convert to binary
            binary = (pattern_array < 128).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit ellipse to get orientation
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                angle = ellipse[2]  # Angle of rotation
                return float(angle % 360)

            return 0.0

        except Exception:
            return 0.0

    def assess_pattern_quality(self, pattern_array: np.ndarray) -> float:
        """Assess the quality of a detected pattern.

        Args:
            pattern_array: Pattern region array

        Returns:
            Quality score between 0 and 1
        """
        try:
            if pattern_array.size == 0:
                return 0.0

            # Calculate various quality metrics

            # 1. Contrast
            contrast = np.std(pattern_array) / 255.0

            # 2. Edge strength
            edges = feature.canny(pattern_array, sigma=1.0)
            edge_strength = np.sum(edges) / edges.size

            # 3. Symmetry (approximate)
            h, w = pattern_array.shape
            if h > 4 and w > 4:
                top_half = pattern_array[: h // 2, :]
                bottom_half = np.flip(pattern_array[h // 2 :, :], axis=0)
                min_h = min(top_half.shape[0], bottom_half.shape[0])
                symmetry = 1.0 - np.mean(np.abs(top_half[:min_h] - bottom_half[:min_h])) / 255.0
            else:
                symmetry = 0.5

            # 4. Size appropriateness
            size_score = (
                1.0 if self.settings["min_pattern_size"] <= min(h, w) <= self.settings["max_pattern_size"] else 0.5
            )

            # Combine metrics
            quality = contrast * 0.3 + edge_strength * 0.3 + symmetry * 0.2 + size_score * 0.2

            return min(1.0, max(0.0, quality))

        except Exception:
            return 0.0

    def _create_templates(self) -> None:
        """Create finder pattern templates for matching using FinderPatternGenerator."""
        self.templates = {}
        generator = FinderPatternGenerator(color_count=8)
        # JABCode finder patterns are always 7x7
        pattern_types = [
            FinderPatternType.FP0,
            FinderPatternType.FP1,
            FinderPatternType.FP2,
            FinderPatternType.FP3,
        ]
        for pattern_type in pattern_types:
            template = generator.generate_pattern(pattern_type) * 255  # 0/1 to 0/255
            self.templates[f"FP{pattern_type}_7"] = template

    def _generate_pattern_template(self, pattern_type: str, size: int) -> np.ndarray:
        """(Deprecated) Use FinderPatternGenerator instead."""
        raise NotImplementedError("Use FinderPatternGenerator for pattern templates.")

    def _detect_by_template_matching(self, array: np.ndarray) -> List[Dict[str, Any]]:
        """DEPRECATED: Detect patterns using color template matching.
        This method is deprecated and will be removed in a future release.
        Use scanline/layer-based detection instead.
        """
        warnings.warn(
            "Template matching for finder pattern detection is deprecated. Use scanline/layer-based detection.",
            DeprecationWarning,
        )
        patterns = []
        threshold = self.settings["template_threshold"]
        # If grayscale, convert to 3D for uniformity
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        for template_name, template in self.templates.items():
            pattern_type, size_str = template_name.split("_")
            template_size = int(size_str)  # Will always be 7 for JABCode
            # Ensure template is 3D
            if template.ndim == 2:
                template = np.stack([template] * 3, axis=-1)
            # Match each channel and average
            results = []
            for c in range(3):
                result = cv2.matchTemplate(array[..., c], template[..., c], cv2.TM_CCOEFF_NORMED)
                results.append(result)
            avg_result = np.mean(results, axis=0)
            # Find matches above threshold
            locations = np.where(avg_result >= threshold)
            for y, x in zip(*locations):
                center_x = x + template_size // 2
                center_y = y + template_size // 2
                confidence = avg_result[y, x]
                pattern = {
                    "center": Point2D(center_x, center_y),
                    "size": template_size,
                    "confidence": float(confidence),
                    "pattern_type": pattern_type,
                    "detection_method": "template_matching",
                }
                # Add corners
                pattern["corners"] = [
                    Point2D(x, y),
                    Point2D(x + template_size, y),
                    Point2D(x + template_size, y + template_size),
                    Point2D(x, y + template_size),
                ]
                patterns.append(pattern)
        # Remove overlapping detections
        patterns = self._remove_overlapping_patterns(patterns)
        return patterns

    def _detect_by_scanline(self, array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect patterns using scanline/layer-based analysis (JABCode reference style)."""
        # Defensive input check
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if array.ndim not in (2, 3):
            raise ValueError(f"Unsupported array shape for scanline detection: {array.shape}")

        # Check if this is a large image and optimize accordingly
        height, width = array.shape[:2]
        total_pixels = height * width
        is_large_image = total_pixels > self.settings["large_image_threshold"]

        if is_large_image:
            # For large images, use a coarser scan to improve performance
            skip_factor = self.settings["large_image_skip_factor"]
            print(f"Large image detected ({width}x{height}), using skip factor {skip_factor}")
        else:
            skip_factor = 1
        # Step 1: Quantize to palette
        if array.ndim == 2:
            bw_threshold = 128
            quantized = (array >= bw_threshold).astype(int)
            palette = np.array([[0], [1]])
            tol_factor = 0.9
        elif array.ndim == 3 and array.shape[2] == 3:
            palette = np.array(
                [
                    [0, 0, 0],
                    [255, 255, 255],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 0],
                    [0, 255, 255],
                    [255, 0, 255],
                ],
                dtype=np.uint8,
            )
            flat = array.reshape(-1, 3)
            dists = np.linalg.norm(flat[:, None, :] - palette[None, :, :], axis=2)
            indices = np.argmin(dists, axis=1)
            quantized = indices.reshape(array.shape[:2])
            tol_factor = 0.5
        else:
            raise ValueError(f"Unsupported array shape for scanline detection: {array.shape}")
        patterns = []

        def classify_pattern_type(colors):
            # JABCode finder patterns have complex internal structure
            # Look for basic alternating patterns that could indicate finder patterns
            if set(colors) <= {0, 1}:
                # Look for patterns with alternating black/white regions
                if len(colors) >= 3:
                    # Count transitions
                    transitions = sum(1 for i in range(len(colors) - 1) if colors[i] != colors[i + 1])
                    if transitions >= 2:
                        # Basic heuristic: patterns with borders and internal structure
                        if colors[0] == colors[-1]:  # Same color at edges (typical for bordered patterns)
                            return "FP0"  # Default to FP0 for detection, classify later
                        else:
                            return "FP1"  # Alternative pattern
            return "unknown"

        def merge_short_runs(runs, min_len=3):
            merged = []
            for val, count in runs:
                if merged and count < min_len:
                    prev_val, prev_count = merged[-1]
                    merged[-1] = (prev_val, prev_count + count)
                else:
                    merged.append((val, count))
            return merged

        def check_pattern_cross(lengths):
            # JABCode reference-style validation: layer size proportion must be n-1-1-1-m where n>1, m>1
            if len(lengths) < 3:
                return False, 0

            # Look for 3-5 layer patterns like JABCode finder patterns
            if len(lengths) >= 3:
                # Calculate module size as average of inner layers
                if len(lengths) == 3:
                    # For 3-layer pattern: outer-inner-outer
                    layer_size = lengths[1]
                elif len(lengths) == 5:
                    # For 5-layer pattern: outer-inner-core-inner-outer (JABCode style)
                    inner_layers = lengths[1:4]  # Take inner 3 layers
                    layer_size = sum(inner_layers) / 3.0
                else:
                    # For 4-layer or other patterns
                    inner_layers = lengths[1:-1]
                    layer_size = sum(inner_layers) / len(inner_layers)

                # Validate layer size relationships with tolerance (based on JABCode reference)
                layer_tolerance = layer_size / 2.0

                # Basic size requirements
                if layer_size < 2:  # Minimum module size
                    return False, layer_size

                # Check layer proportions for JABCode-style patterns
                if len(lengths) == 5:
                    # 5-layer pattern validation (reference checkPatternCross)
                    valid = (
                        abs(layer_size - lengths[1]) < layer_tolerance
                        and abs(layer_size - lengths[2]) < layer_tolerance
                        and abs(layer_size - lengths[3]) < layer_tolerance
                        and lengths[0] > 0.5 * layer_tolerance
                        and lengths[4] > 0.5 * layer_tolerance
                        and abs(lengths[1] - lengths[3]) < layer_tolerance
                    )
                else:
                    # Simpler validation for 3-4 layer patterns
                    valid = all(l > 0 for l in lengths) and layer_size >= 2

                return valid, layer_size

            return False, 0

        def cross_check_orthogonal(quantized, axis, center, idx, size):
            # axis=0: check column at x=center, y=idx; axis=1: check row at y=center, x=idx
            if axis == 0:
                line = quantized[:, int(center)] if 0 <= int(center) < quantized.shape[1] else None
                pos = idx
            else:
                line = quantized[int(center), :] if 0 <= int(center) < quantized.shape[0] else None
                pos = idx
            if line is None:
                return False
            # Extract a window around pos
            window = line[max(0, pos - size) : min(len(line), pos + size + 1)]
            # Find runs in window
            runs = []
            prev = window[0]
            count = 1
            for val in window[1:]:
                if val == prev:
                    count += 1
                else:
                    runs.append((prev, count))
                    prev = val
                    count = 1
            runs.append((prev, count))
            runs = merge_short_runs(runs)
            if len(runs) < 5:
                return False
            for i in range(len(runs) - 4):
                lengths = [runs[j][1] for j in range(i, i + 5)]
                valid, _ = check_pattern_cross(lengths)
                if valid:
                    return True
            return False

        def scan_line(line, axis, idx):
            # Find runs of the same color
            runs = []
            prev = line[0]
            count = 1
            for val in line[1:]:
                if val == prev:
                    count += 1
                else:
                    runs.append((prev, count))
                    prev = val
                    count = 1
            runs.append((prev, count))
            runs = merge_short_runs(runs)

            # Look for pattern sequences of various lengths (3, 4, 5 runs) - prioritize 5 for JABCode
            for seq_len in [5, 4, 3]:  # Prefer 5-layer patterns (JABCode style)
                if len(runs) < seq_len:
                    continue

                for i in range(len(runs) - seq_len + 1):
                    colors = [runs[j][0] for j in range(i, i + seq_len)]
                    lengths = [runs[j][1] for j in range(i, i + seq_len)]
                    valid, module_size = check_pattern_cross(lengths)

                    if valid:
                        # Calculate center position correctly
                        start_pos = sum(runs[j][1] for j in range(i))
                        pattern_width = sum(lengths)
                        center_pos = start_pos + pattern_width // 2

                        # Improved confidence calculation based on JABCode reference style
                        # Consider layer size consistency and pattern quality
                        if len(lengths) >= 3:
                            # Calculate variance in layer sizes (lower is better)
                            if len(lengths) == 5:
                                inner_layers = lengths[1:4]
                            else:
                                inner_layers = lengths[1:-1]

                            if len(inner_layers) > 1:
                                layer_variance = np.var(inner_layers)
                                consistency_score = max(
                                    0.1,
                                    1.0 - (layer_variance / (module_size * module_size)),
                                )
                            else:
                                consistency_score = 0.8

                            # Pattern structure score (prefer alternating patterns)
                            transitions = sum(1 for k in range(len(colors) - 1) if colors[k] != colors[k + 1])
                            structure_score = min(1.0, transitions / 4.0)  # Expect 2-4 transitions

                            # Size reasonableness score
                            size_score = (
                                1.0
                                if self.settings["min_pattern_size"]
                                <= pattern_width
                                <= self.settings["max_pattern_size"]
                                else 0.3
                            )

                            # Combined confidence score
                            confidence = consistency_score * 0.5 + structure_score * 0.3 + size_score * 0.2
                        else:
                            confidence = 0.5

                        pattern_type = classify_pattern_type(colors)

                        # Create point based on scan axis
                        if axis == 0:  # Horizontal scan (y=idx, x=center_pos)
                            pt = Point2D(center_pos, idx)
                        else:  # Vertical scan (x=idx, y=center_pos)
                            pt = Point2D(idx, center_pos)

                        # Apply size and confidence filtering consistent with settings
                        pattern_size = max(pattern_width, int(module_size * 3))

                        # Only add patterns that meet basic quality criteria
                        if (
                            confidence >= 0.5  # Reasonable confidence threshold
                            and self.settings["min_pattern_size"] <= pattern_size <= self.settings["max_pattern_size"]
                        ):

                            pattern = {
                                "center": pt,
                                "lengths": lengths,
                                "colors": colors,
                                "size": pattern_size,
                                "axis": axis,
                                "confidence": min(1.0, confidence),  # Cap at 1.0
                                "pattern_type": pattern_type,
                                "detection_method": "scanline",
                                "module_size": module_size,
                            }
                            patterns.append(pattern)

        # Use skip factor for large images to improve performance
        for y in range(0, quantized.shape[0], skip_factor):
            scan_line(quantized[y, :], axis=0, idx=y)
        for x in range(0, quantized.shape[1], skip_factor):
            scan_line(quantized[:, x], axis=1, idx=x)

        # Limit patterns for large images to prevent memory issues
        if is_large_image and len(patterns) > self.settings["max_patterns_per_symbol"]:
            # Sort by confidence and keep the best patterns
            patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
            patterns = patterns[: self.settings["max_patterns_per_symbol"]]
            print(f"Limited to {len(patterns)} best patterns for large image")

        # Remove overlapping patterns and apply quality-based filtering
        patterns = self._remove_overlapping_patterns(patterns)

        # Additional filtering to improve pattern quality based on JABCode reference approach
        if patterns:
            # Calculate mean module size for filtering (JABCode reference style)
            module_sizes = [p.get("module_size", p["size"] / 3) for p in patterns]
            mean_module_size = np.mean(module_sizes)
            threshold = mean_module_size / 2.0  # Tolerance for module size variation

            # Filter patterns by module size consistency and confidence
            filtered_patterns = []
            for pattern in patterns:
                pattern_module_size = pattern.get("module_size", pattern["size"] / 3)
                size_diff = abs(pattern_module_size - mean_module_size)

                # Keep patterns with consistent module sizes and good confidence
                if size_diff <= threshold and pattern["confidence"] >= self.settings.get("quality_threshold", 0.5):
                    filtered_patterns.append(pattern)

            # If all patterns were filtered out, keep the best one (JABCode backup mechanism)
            if not filtered_patterns and patterns:
                best_pattern = max(patterns, key=lambda p: p["confidence"])
                filtered_patterns = [best_pattern]

            patterns = filtered_patterns

        return patterns

    def _detect_by_contour_detection(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect patterns using contour detection.

        Args:
            gray: Grayscale image array

        Returns:
            List of detected patterns
        """
        patterns = []

        # Threshold image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours - use RETR_LIST to find all contours including internal ones
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if not (self.settings["contour_min_area"] <= area <= self.settings["contour_max_area"]):
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio (should be approximately square)
            aspect_ratio = w / h if h > 0 else 0
            if not (
                1 - self.settings["aspect_ratio_tolerance"]
                <= aspect_ratio
                <= 1 + self.settings["aspect_ratio_tolerance"]
            ):
                continue

            # Extract pattern region
            pattern_region = gray[y : y + h, x : x + w]

            # Validate pattern
            center = Point2D(x + w // 2, y + h // 2)
            if not self.validate_pattern(pattern_region, center):
                continue

            # Assess quality
            quality = self.assess_pattern_quality(pattern_region)

            pattern = {
                "center": center,
                "size": max(w, h),
                "confidence": quality,
                "pattern_type": "Unknown",  # Would need classification
                "detection_method": "contour_detection",
                "area": area,
            }

            # Add corners
            pattern["corners"] = [
                Point2D(x, y),
                Point2D(x + w, y),
                Point2D(x + w, y + h),
                Point2D(x, y + h),
            ]

            patterns.append(pattern)

        return patterns

    def _detect_hybrid(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect patterns using hybrid approach.

        Args:
            gray: Grayscale image array

        Returns:
            List of detected patterns
        """
        # Use both methods and combine results
        template_patterns = self._detect_by_template_matching(gray)
        contour_patterns = self._detect_by_contour_detection(gray)

        # Combine and remove duplicates
        all_patterns = template_patterns + contour_patterns
        return self._remove_overlapping_patterns(all_patterns)

    def _filter_patterns(self, patterns: List[Dict[str, Any]], gray: np.ndarray) -> List[Dict[str, Any]]:
        """Filter patterns by size and quality criteria.

        Args:
            patterns: List of detected patterns
            gray: Original grayscale image

        Returns:
            Filtered patterns
        """
        filtered = []

        for pattern in patterns:
            size = pattern["size"]
            confidence = pattern["confidence"]

            # Size filter
            if not (self.settings["min_pattern_size"] <= size <= self.settings["max_pattern_size"]):
                continue

            # Quality filter
            if confidence < self.settings["quality_threshold"]:
                continue

            filtered.append(pattern)

        return filtered

    def _remove_overlapping_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping pattern detections.

        Args:
            patterns: List of patterns

        Returns:
            Non-overlapping patterns
        """
        if len(patterns) <= 1:
            return patterns

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p["confidence"], reverse=True)

        filtered = []
        for pattern in patterns:
            is_overlapping = False

            for existing in filtered:
                # Calculate distance between centers
                dx = pattern["center"].x - existing["center"].x
                dy = pattern["center"].y - existing["center"].y
                distance = math.sqrt(dx * dx + dy * dy)

                # Check if they overlap significantly
                min_distance = (pattern["size"] + existing["size"]) / 4
                if distance < min_distance:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered.append(pattern)

        return filtered

    def _reduce_noise(self, gray: np.ndarray) -> np.ndarray:
        """Apply noise reduction to grayscale image.

        Args:
            gray: Grayscale image array

        Returns:
            Denoised image
        """
        # Apply Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        return denoised

    def _validate_settings(self) -> None:
        """Validate detector settings."""
        if self.settings["detection_method"] not in self.detection_methods:
            raise ValueError(f"Invalid detection method: {self.settings['detection_method']}")

        if not (0 < self.settings["template_threshold"] <= 1):
            raise ValueError("Template threshold must be between 0 and 1")

        if self.settings["min_pattern_size"] >= self.settings["max_pattern_size"]:
            raise ValueError("Min pattern size must be less than max pattern size")

    def _update_stats(self, detection_time: float, patterns_found: int) -> None:
        """Update detection statistics.

        Args:
            detection_time: Time taken for detection
            patterns_found: Number of patterns found
        """
        self._stats["total_detections"] += 1
        self._stats["total_detection_time"] += detection_time
        self._stats["detection_times"].append(detection_time)
        self._stats["patterns_found"] += patterns_found

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        if self._stats["total_detections"] == 0:
            return {
                "total_detections": 0,
                "total_detection_time": 0.0,
                "avg_detection_time": 0.0,
                "patterns_found": 0,
            }

        return {
            "total_detections": self._stats["total_detections"],
            "total_detection_time": self._stats["total_detection_time"],
            "avg_detection_time": (self._stats["total_detection_time"] / self._stats["total_detections"]),
            "min_detection_time": min(self._stats["detection_times"]),
            "max_detection_time": max(self._stats["detection_times"]),
            "patterns_found": self._stats["patterns_found"],
            "avg_patterns_per_detection": (self._stats["patterns_found"] / self._stats["total_detections"]),
        }

    def reset(self) -> None:
        """Reset detector statistics and state."""
        self._stats = {
            "total_detections": 0,
            "total_detection_time": 0.0,
            "detection_times": [],
            "patterns_found": 0,
        }

    def copy(self) -> "FinderPatternDetector":
        """Create a copy of this detector.

        Returns:
            New FinderPatternDetector instance with same settings
        """
        return FinderPatternDetector(self.settings.copy())

    def __str__(self) -> str:
        """String representation of detector."""
        return (
            f"FinderPatternDetector(method={self.settings['detection_method']}, "
            f"threshold={self.settings['template_threshold']}, "
            f"detections={self._stats['total_detections']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"FinderPatternDetector(settings={self.settings})"

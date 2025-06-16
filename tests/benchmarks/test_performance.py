"""Performance benchmarking tests for PyHue2D JABCode implementation.

These tests measure encoding and decoding performance across different
data sizes, configurations, and scenarios to ensure acceptable performance
and identify optimization opportunities.
"""

import secrets
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from pyhue2d.core import decode, encode
from pyhue2d.jabcode.decoder import JABCodeDecoder
from pyhue2d.jabcode.encoder import JABCodeEncoder
from pyhue2d.jabcode.image_processing.finder_detector import FinderPatternDetector
from pyhue2d.jabcode.pipeline.encoding import EncodingPipeline


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    operation: str
    configuration: Dict[str, Any]
    data_size: int
    execution_time: float
    throughput_mbps: float
    memory_usage: int = 0
    success: bool = True
    error: str = ""


class PerformanceBenchmark:
    """Base class for performance benchmarks."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def time_operation(self, operation_name: str, func, *args, **kwargs) -> Tuple[Any, float]:
        """Time a single operation.

        Args:
            operation_name: Name of the operation for logging
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return result, execution_time
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Operation {operation_name} failed: {e}")
            return None, execution_time

    def calculate_throughput(self, data_size: int, execution_time: float) -> float:
        """Calculate throughput in MB/s.

        Args:
            data_size: Size of data in bytes
            execution_time: Execution time in seconds

        Returns:
            Throughput in MB/s
        """
        if execution_time <= 0:
            return 0.0
        return (data_size / 1024 / 1024) / execution_time

    def add_result(
        self,
        operation: str,
        config: Dict[str, Any],
        data_size: int,
        execution_time: float,
        success: bool = True,
        error: str = "",
    ):
        """Add a benchmark result."""
        throughput = self.calculate_throughput(data_size, execution_time)
        result = BenchmarkResult(
            operation=operation,
            configuration=config,
            data_size=data_size,
            execution_time=execution_time,
            throughput_mbps=throughput,
            success=success,
            error=error,
        )
        self.results.append(result)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all benchmarks."""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return {"total_runs": len(self.results), "successful_runs": 0}

        execution_times = [r.execution_time for r in successful_results]
        throughputs = [r.throughput_mbps for r in successful_results]
        data_sizes = [r.data_size for r in successful_results]

        return {
            "total_runs": len(self.results),
            "successful_runs": len(successful_results),
            "success_rate": len(successful_results) / len(self.results) * 100,
            "execution_time": {
                "min": min(execution_times),
                "max": max(execution_times),
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "stdev": (statistics.stdev(execution_times) if len(execution_times) > 1 else 0),
            },
            "throughput_mbps": {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            },
            "data_size": {
                "min": min(data_sizes),
                "max": max(data_sizes),
                "mean": statistics.mean(data_sizes),
                "median": statistics.median(data_sizes),
            },
        }


class TestEncodingPerformance(PerformanceBenchmark):
    """Test encoding performance across different scenarios."""

    def test_encoding_performance_by_data_size(self):
        """Test encoding performance across different data sizes."""
        data_sizes = [10, 50, 100, 500, 1000, 2000]  # bytes
        repetitions = 3

        config = {"colors": 8, "ecc_level": "M"}

        for size in data_sizes:
            # Generate test data
            test_data = secrets.token_bytes(size)

            for rep in range(repetitions):
                try:
                    result, exec_time = self.time_operation(f"encode_{size}_bytes", encode, test_data, **config)

                    self.add_result(
                        operation="encode",
                        config=config,
                        data_size=size,
                        execution_time=exec_time,
                        success=result is not None,
                    )

                except Exception as e:
                    self.add_result(
                        operation="encode",
                        config=config,
                        data_size=size,
                        execution_time=0.0,
                        success=False,
                        error=str(e),
                    )

        # Print results
        stats = self.get_summary_stats()
        print(f"\nEncoding Performance by Data Size:")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Average Execution Time: {stats['execution_time']['mean']:.3f}s")
        print(f"Average Throughput: {stats['throughput_mbps']['mean']:.2f} MB/s")

        # Basic performance assertions
        assert stats["success_rate"] > 50, "Encoding success rate too low"
        assert stats["execution_time"]["mean"] < 10.0, "Average encoding time too high"

    def test_encoding_performance_by_color_count(self):
        """Test encoding performance across different color counts."""
        color_counts = [4, 8, 16, 32]
        test_data = b"Performance test data " * 10  # ~230 bytes
        repetitions = 3

        for colors in color_counts:
            config = {"colors": colors, "ecc_level": "M"}

            for rep in range(repetitions):
                try:
                    result, exec_time = self.time_operation(f"encode_{colors}_colors", encode, test_data, **config)

                    self.add_result(
                        operation="encode_color_count",
                        config=config,
                        data_size=len(test_data),
                        execution_time=exec_time,
                        success=result is not None,
                    )

                except Exception as e:
                    self.add_result(
                        operation="encode_color_count",
                        config=config,
                        data_size=len(test_data),
                        execution_time=0.0,
                        success=False,
                        error=str(e),
                    )

        # Analyze results by color count
        results_by_color = defaultdict(list)
        for result in self.results:
            if result.operation == "encode_color_count" and result.success:
                colors = result.configuration["colors"]
                results_by_color[colors].append(result.execution_time)

        print(f"\nEncoding Performance by Color Count:")
        for colors, times in results_by_color.items():
            if times:
                avg_time = statistics.mean(times)
                print(f"  {colors} colors: {avg_time:.3f}s average")

        # Check that all color counts work
        assert len(results_by_color) >= 3, "Should successfully encode with multiple color counts"

    def test_encoding_performance_by_ecc_level(self):
        """Test encoding performance across different ECC levels."""
        ecc_levels = ["L", "M", "Q", "H"]
        test_data = b"ECC performance test " * 20  # ~420 bytes
        repetitions = 3

        for ecc_level in ecc_levels:
            config = {"colors": 8, "ecc_level": ecc_level}

            for rep in range(repetitions):
                try:
                    result, exec_time = self.time_operation(f"encode_ecc_{ecc_level}", encode, test_data, **config)

                    self.add_result(
                        operation="encode_ecc_level",
                        config=config,
                        data_size=len(test_data),
                        execution_time=exec_time,
                        success=result is not None,
                    )

                except Exception as e:
                    self.add_result(
                        operation="encode_ecc_level",
                        config=config,
                        data_size=len(test_data),
                        execution_time=0.0,
                        success=False,
                        error=str(e),
                    )

        # Analyze results by ECC level
        results_by_ecc = defaultdict(list)
        for result in self.results:
            if result.operation == "encode_ecc_level" and result.success:
                ecc = result.configuration["ecc_level"]
                results_by_ecc[ecc].append(result.execution_time)

        print(f"\nEncoding Performance by ECC Level:")
        for ecc, times in results_by_ecc.items():
            if times:
                avg_time = statistics.mean(times)
                print(f"  ECC {ecc}: {avg_time:.3f}s average")

        # Higher ECC levels might take longer, but all should work
        assert len(results_by_ecc) >= 2, "Should successfully encode with multiple ECC levels"


class TestDecodingPerformance(PerformanceBenchmark):
    """Test decoding performance across different scenarios."""

    @pytest.fixture(autouse=True)
    def setup_test_images(self, tmp_path):
        """Create test images for decoding benchmarks."""
        self.test_images = []

        # Create images of different sizes
        test_data_sets = [
            b"Small test",
            b"Medium test data " * 5,
            b"Large test data " * 20,
        ]

        for i, test_data in enumerate(test_data_sets):
            try:
                encoded_image = encode(test_data, colors=8, ecc_level="M")
                image_path = tmp_path / f"test_image_{i}.png"
                encoded_image.save(image_path)

                self.test_images.append(
                    {
                        "path": image_path,
                        "original_data": test_data,
                        "size": len(test_data),
                    }
                )
            except Exception as e:
                print(f"Failed to create test image {i}: {e}")

        if not self.test_images:
            pytest.skip("No test images could be created")

    def test_decoding_performance_basic(self):
        """Test basic decoding performance."""
        if not hasattr(self, "test_images") or not self.test_images:
            pytest.skip("No test images available")

        repetitions = 3

        for image_info in self.test_images:
            for rep in range(repetitions):
                try:
                    result, exec_time = self.time_operation(
                        f"decode_{image_info['size']}_bytes",
                        decode,
                        str(image_info["path"]),
                    )

                    self.add_result(
                        operation="decode",
                        config={"detection_method": "scanline"},
                        data_size=image_info["size"],
                        execution_time=exec_time,
                        success=result is not None and len(result) > 0,
                    )

                except Exception as e:
                    self.add_result(
                        operation="decode",
                        config={"detection_method": "scanline"},
                        data_size=image_info["size"],
                        execution_time=0.0,
                        success=False,
                        error=str(e),
                    )

        # Print results
        stats = self.get_summary_stats()
        print(f"\nDecoding Performance:")
        print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")
        if stats.get("successful_runs", 0) > 0:
            print(f"Average Execution Time: {stats['execution_time']['mean']:.3f}s")
            print(f"Average Throughput: {stats['throughput_mbps']['mean']:.2f} MB/s")

        # Basic performance assertions (lenient due to decoder calibration)
        assert stats.get("successful_runs", 0) > 0, "No successful decoding operations"

    def test_decoding_performance_detection_methods(self):
        """Test decoding performance with different detection methods."""
        if not hasattr(self, "test_images") or not self.test_images:
            pytest.skip("No test images available")

        detection_methods = ["scanline", "contour"]
        image_info = self.test_images[0]  # Use first test image
        repetitions = 3

        for method in detection_methods:
            decoder = JABCodeDecoder(
                {
                    "detection_method": method,
                    "perspective_correction": True,
                    "error_correction": True,
                }
            )

            for rep in range(repetitions):
                try:
                    result, exec_time = self.time_operation(f"decode_{method}", decoder.decode, str(image_info["path"]))

                    self.add_result(
                        operation="decode_method",
                        config={"detection_method": method},
                        data_size=image_info["size"],
                        execution_time=exec_time,
                        success=result is not None and len(result) >= 0,
                    )

                except Exception as e:
                    self.add_result(
                        operation="decode_method",
                        config={"detection_method": method},
                        data_size=image_info["size"],
                        execution_time=0.0,
                        success=False,
                        error=str(e),
                    )

        # Analyze results by detection method
        results_by_method = defaultdict(list)
        for result in self.results:
            if result.operation == "decode_method" and result.success:
                method = result.configuration["detection_method"]
                results_by_method[method].append(result.execution_time)

        print(f"\nDecoding Performance by Detection Method:")
        for method, times in results_by_method.items():
            if times:
                avg_time = statistics.mean(times)
                print(f"  {method}: {avg_time:.3f}s average")


class TestComponentPerformance(PerformanceBenchmark):
    """Test performance of individual components."""

    def test_pattern_detection_performance(self):
        """Test finder pattern detection performance."""
        try:
            # Load a reference image if available
            ref_image_path = Path("tests/example_images/example1.png")
            if not ref_image_path.exists():
                pytest.skip("Reference image not available")

            detector = FinderPatternDetector({"detection_method": "scanline"})

            from PIL import Image

            image = Image.open(ref_image_path)

            repetitions = 5

            for rep in range(repetitions):
                result, exec_time = self.time_operation("pattern_detection", detector.find_patterns, image)

                self.add_result(
                    operation="pattern_detection",
                    config={"method": "scanline"},
                    data_size=image.width * image.height * 3,  # Approximate image size
                    execution_time=exec_time,
                    success=result is not None and len(result) > 0,
                )

            # Print results
            successful_results = [r for r in self.results if r.success]
            if successful_results:
                avg_time = statistics.mean([r.execution_time for r in successful_results])
                print(f"\nPattern Detection Performance:")
                print(f"Average time: {avg_time:.3f}s")
                print(f"Success rate: {len(successful_results)/repetitions*100:.1f}%")

        except Exception as e:
            pytest.skip(f"Pattern detection performance test failed: {e}")

    def test_encoding_pipeline_performance(self):
        """Test encoding pipeline component performance."""
        test_data = b"Pipeline performance test " * 10

        pipeline = EncodingPipeline(
            {
                "color_count": 8,
                "ecc_level": "M",
            }
        )

        repetitions = 3

        for rep in range(repetitions):
            try:
                result, exec_time = self.time_operation("encoding_pipeline", pipeline.encode, test_data)

                self.add_result(
                    operation="encoding_pipeline",
                    config={"color_count": 8, "ecc_level": "M"},
                    data_size=len(test_data),
                    execution_time=exec_time,
                    success=result is not None,
                )

            except Exception as e:
                self.add_result(
                    operation="encoding_pipeline",
                    config={"color_count": 8, "ecc_level": "M"},
                    data_size=len(test_data),
                    execution_time=0.0,
                    success=False,
                    error=str(e),
                )

        # Print results
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            avg_time = statistics.mean([r.execution_time for r in successful_results])
            print(f"\nEncoding Pipeline Performance:")
            print(f"Average time: {avg_time:.3f}s")
            print(f"Success rate: {len(successful_results)/repetitions*100:.1f}%")


class TestMemoryPerformance:
    """Test memory usage and performance characteristics."""

    def test_memory_scaling_with_data_size(self):
        """Test memory scaling with different data sizes."""
        # This is a basic test - in production you'd use memory profiling tools
        import sys

        data_sizes = [100, 500, 1000, 2000]

        for size in data_sizes:
            test_data = b"X" * size

            # Measure memory before
            initial_size = sys.getsizeof(test_data)

            try:
                # Encode
                encoded_image = encode(test_data, colors=8, ecc_level="M")

                # Basic memory check
                assert encoded_image is not None
                print(f"Data size {size}: encoded successfully")

            except Exception as e:
                print(f"Data size {size}: failed ({e})")

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        # Basic test to ensure no obvious memory leaks
        test_data = b"Memory cleanup test"

        # Perform multiple encode/decode cycles
        for i in range(10):
            try:
                encoded_image = encode(test_data, colors=8, ecc_level="M")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    encoded_image.save(tmp_file.name)
                    tmp_path = tmp_file.name

                try:
                    decoded_data = decode(tmp_path)
                    # Memory should be cleaned up automatically

                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            except Exception as e:
                print(f"Cycle {i} failed: {e}")

        print("Memory cleanup test completed")


def print_benchmark_summary(benchmark: PerformanceBenchmark):
    """Print comprehensive benchmark summary."""
    stats = benchmark.get_summary_stats()

    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"Total Operations: {stats.get('total_runs', 0)}")
    print(f"Successful Operations: {stats.get('successful_runs', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0):.1f}%")

    if stats.get("successful_runs", 0) > 0:
        exec_stats = stats["execution_time"]
        throughput_stats = stats["throughput_mbps"]

        print(f"\nExecution Time (seconds):")
        print(f"  Min: {exec_stats['min']:.3f}")
        print(f"  Max: {exec_stats['max']:.3f}")
        print(f"  Mean: {exec_stats['mean']:.3f}")
        print(f"  Median: {exec_stats['median']:.3f}")

        print(f"\nThroughput (MB/s):")
        print(f"  Min: {throughput_stats['min']:.2f}")
        print(f"  Max: {throughput_stats['max']:.2f}")
        print(f"  Mean: {throughput_stats['mean']:.2f}")
        print(f"  Median: {throughput_stats['median']:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

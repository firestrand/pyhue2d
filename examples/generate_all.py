"""Master script to run all examples and generate all repository sample assets.

Usage:
    python examples/generate_all.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = Path(__file__).parent / "output"

EXAMPLE_MODULES = [
    ("examples.01_basic_roundtrip", "01_basic_roundtrip.py", "Basic roundtrip encoding and decoding"),
    ("examples.02_color_depths", "02_color_depths.py", "ISO 4, 8, 16, 32, 64 color depths"),
    (
        "examples.03_multisymbol_docking",
        "03_multisymbol_docking.py",
        "Multi-symbol horizontal, vertical, and grid docking",
    ),
    ("examples.04_camera_unwarp", "04_camera_unwarp.py", "Camera frame perspective unwarping"),
    ("examples.05_binary_and_unicode", "05_binary_and_unicode.py", "Unicode UTF-8 and arbitrary binary payloads"),
    ("examples.06_vector_export", "06_vector_export.py", "SVG and PDF vector graphic export"),
    ("examples.07_frame_stream_decoding", "07_frame_stream_decoding.py", "Frame-by-frame video stream decoding"),
]


def main(output_dir: Path = OUTPUT_DIR) -> None:
    """Run every example and write its assets to the requested directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_total = time.time()

    print("========================================================================")
    print(" PyHue2D - Running All Example Demonstrations & Asset Generation")
    print("========================================================================\n")

    for mod_name, file_name, desc in EXAMPLE_MODULES:
        print(f"\n>>> Running {file_name}: {desc}...")
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        mod.main(output_dir)
        dt = time.time() - t0
        print(f">>> Completed {file_name} in {dt:.2f}s.")

    elapsed = time.time() - start_total
    print("\n========================================================================")
    print(f" All examples completed successfully in {elapsed:.2f}s!")
    print("========================================================================")
    print(f"\nGenerated assets in {output_dir}:")
    for asset in sorted(output_dir.iterdir()):
        size_kb = asset.stat().st_size / 1024.0
        print(f"  - {asset.name:<32} ({size_kb:6.2f} KB)")


def compare_assets(expected_dir: Path, actual_dir: Path) -> None:
    """Require matching filenames, exact PNG pixels, and exact vector bytes."""
    expected_names = {path.name for path in expected_dir.iterdir()}
    actual_names = {path.name for path in actual_dir.iterdir()}
    if expected_names != actual_names:
        raise ValueError(
            f"Example asset set differs: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name in sorted(expected_names):
        expected = expected_dir / name
        actual = actual_dir / name
        if expected.suffix == ".png":
            with Image.open(expected) as left, Image.open(actual) as right:
                matches = (left.mode, left.size) == (right.mode, right.size)
                # RGBA expansion also detects palette and transparency changes.
                matches = matches and left.convert("RGBA").tobytes() == right.convert("RGBA").tobytes()
        else:
            matches = expected.read_bytes() == actual.read_bytes()
        if not matches:
            raise ValueError(f"Example asset differs: {name}")


def check_assets(expected_dir: Path = OUTPUT_DIR) -> None:
    """Regenerate and compare examples without changing the repository assets."""
    with TemporaryDirectory(prefix="pyhue2d-examples-") as directory:
        actual_dir = Path(directory)
        main(actual_dir)
        compare_assets(expected_dir, actual_dir)
    print("All example assets match (PNG pixels and vector bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify assets without overwriting them")
    if parser.parse_args().check:
        check_assets()
    else:
        main()

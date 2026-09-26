"""Master script to run all examples and generate all repository sample assets.

Usage:
    python examples/generate_all.py
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_total = time.time()

    print("========================================================================")
    print(" PyHue2D - Running All Example Demonstrations & Asset Generation")
    print("========================================================================\n")

    for mod_name, file_name, desc in EXAMPLE_MODULES:
        print(f"\n>>> Running {file_name}: {desc}...")
        t0 = time.time()
        mod = importlib.import_module(mod_name)
        mod.main()
        dt = time.time() - t0
        print(f">>> Completed {file_name} in {dt:.2f}s.")

    elapsed = time.time() - start_total
    print("\n========================================================================")
    print(f" All examples completed successfully in {elapsed:.2f}s!")
    print("========================================================================")
    print(f"\nGenerated assets in {OUTPUT_DIR}:")
    for asset in sorted(OUTPUT_DIR.iterdir()):
        size_kb = asset.stat().st_size / 1024.0
        print(f"  - {asset.name:<32} ({size_kb:6.2f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check integrity of approved JAB Code fixtures.

This script uses Pillow only and does not import pyhue2d.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

QUARANTINED_FILES = {
    "multi_block_2.png",
    "multi_block_3.png",
    "multi_block_4.png",
    "multi_block_5.png",
    "multi_block_6.png",
    "multi_block_7.png",
    "multi_block_8.png",
    "multi_block_9.png",
    "maximum_text.png",
    "multi_block_2.png.json",
    "multi_block_3.png.json",
    "multi_block_4.png.json",
    "multi_block_5.png.json",
    "multi_block_6.png.json",
    "multi_block_7.png.json",
    "multi_block_8.png.json",
    "multi_block_9.png.json",
    "maximum_text.png.json",
}

APPROVED_FIXTURE_NAMES = [
    "example1.png",
    "example2.png",
    "example3.png",
    "example4.png",
    "example5.png",
    "minimum_text.png",
    "test_block2.png",
    "mode_upper.png",
    "mode_lower.png",
    "mode_numeric.png",
    "mode_punct.png",
    "mode_alphanum.png",
    "mode_mixed.png",
    "mode_byte.png",
    "asan_multi2.png",
    "multi_block_2_v32.png",
    "multi_block_3_v32.png",
    "multi_block_4_v32.png",
    "multi_block_5_v32.png",
    "multi_block_6_v32.png",
    "multi_block_7_v32.png",
    "multi_block_8_v32.png",
    "multi_block_9_v32.png",
]


def check_fixtures(base_dir: Path | None = None) -> list[str]:
    errors: list[str] = []

    # Determine fixture directory
    if base_dir is None:
        approved_dir = Path("tests/fixtures/approved/jabcode")
        legacy_dir = Path("tests/example_images")
        if approved_dir.exists():
            fixture_dir = approved_dir
        elif legacy_dir.exists():
            fixture_dir = legacy_dir
        else:
            return ["Neither tests/fixtures/approved/jabcode nor tests/example_images found"]
    else:
        fixture_dir = base_dir

    sha_file = fixture_dir / "SHA256SUMS"
    sha_map: dict[str, str] = {}
    if sha_file.exists():
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                sha_map[Path(parts[1]).name] = parts[0]

    for png_name in APPROVED_FIXTURE_NAMES:
        png_path = fixture_dir / png_name
        sidecar_path = fixture_dir / f"{png_name}.json"

        if not png_path.exists():
            errors.append(f"{png_name}: PNG file missing at {png_path}")
            continue

        if not sidecar_path.exists():
            errors.append(f"{png_name}: Sidecar missing at {sidecar_path}")
            continue

        if sidecar_path.stat().st_size == 0:
            errors.append(f"{png_name}: Sidecar is empty 0-byte file (should be quarantined)")
            continue

        # Load sidecar JSON
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                content = f.read()
                if "BEGIN PRIVATE KEY" in content:
                    errors.append(f"{sidecar_path.name}: Contains PEM private key header!")
                sidecar_data = json.loads(content)
        except Exception as e:
            errors.append(f"{png_name}: Failed to parse sidecar JSON: {e}")
            continue

        # Verify image dimensions
        try:
            with Image.open(png_path) as img:
                actual_size = img.size
                if "final_image_size" in sidecar_data:
                    expected_size = tuple(sidecar_data["final_image_size"])
                    if actual_size != expected_size:
                        errors.append(
                            f"{png_name}: Size mismatch: actual {actual_size} != expected {expected_size} from sidecar"
                        )
        except Exception as e:
            errors.append(f"{png_name}: Failed to open image with Pillow: {e}")
            continue

        # Verify SHA256 if recorded
        if sha_map and png_name in sha_map:
            actual_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
            if actual_sha != sha_map[png_name]:
                errors.append(f"{png_name}: SHA256 mismatch: actual {actual_sha} != expected {sha_map[png_name]}")

    return errors


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check integrity of JAB Code fixtures.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory containing fixtures")
    parser.add_argument(
        "--set", type=str, default="approved", choices=["approved", "varied", "photos"], help="Fixture set to check"
    )
    args = parser.parse_args()

    target_dir = args.dir
    if target_dir is None and args.set != "approved":
        target_dir = Path("tests/fixtures/approved/jabcode") / args.set

    errors = check_fixtures(base_dir=target_dir)
    if errors:
        print("=== Fixture Integrity Errors ===", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("All approved JAB Code fixtures passed integrity checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

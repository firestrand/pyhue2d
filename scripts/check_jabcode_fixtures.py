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


def check_varied_fixtures(target_dir: Path) -> tuple[list[str], int]:
    """Check varied fixtures (LOCAL-DATA-06) for Phase V13."""
    if not target_dir.exists() or not list(target_dir.glob("*.png")):
        return [
            "Phase V14 is blocked: LOCAL-DATA-06 varied captures (color count != 8, ECC != 3) are not in the repo."
        ], 2

    png_files = sorted(target_dir.glob("*.png"))
    sha_file = target_dir / "SHA256SUMS"
    sha_map: dict[str, str] = {}
    if sha_file.exists():
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                sha_map[Path(parts[1]).name] = parts[0]

    errors = []
    has_non_8_color = False
    has_non_3_ecc = False

    for png_path in png_files:
        sidecar_path = target_dir / f"{png_path.name}.json"
        if not sidecar_path.exists():
            errors.append(f"{png_path.name}: Sidecar missing at {sidecar_path}")
            continue

        try:
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            color_num = sidecar_data.get("color_number")
            ecc_levels = sidecar_data.get("ecc_levels", [])
            if color_num is not None and color_num != 8:
                has_non_8_color = True
            if ecc_levels and ecc_levels != [3]:
                has_non_3_ecc = True
        except Exception as e:
            errors.append(f"{png_path.name}: Failed to parse sidecar: {e}")

        if sha_map and png_path.name in sha_map:
            actual_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
            if actual_sha != sha_map[png_path.name]:
                errors.append(
                    f"{png_path.name}: SHA256 mismatch: actual {actual_sha} != expected {sha_map[png_path.name]}"
                )

    if not has_non_8_color:
        errors.append("LOCAL-DATA-06 requires at least one capture with color count != 8")
    if not has_non_3_ecc:
        errors.append("LOCAL-DATA-06 requires at least one capture with ECC != 3")

    return errors, 1 if errors else 0


def check_photo_fixtures(target_dir: Path) -> tuple[list[str], int]:
    """Check photograph fixtures (LOCAL-DATA-07) for Phase V17."""
    if not target_dir.exists() or not list(target_dir.glob("*.png")):
        return ["Phase V18 is blocked: LOCAL-DATA-07 photographed symbol captures are not in the repo."], 2

    png_files = sorted(target_dir.glob("*.png"))
    sha_file = target_dir / "SHA256SUMS"
    sha_map: dict[str, str] = {}
    if sha_file.exists():
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                sha_map[Path(parts[1]).name] = parts[0]

    errors: list[str] = []
    for png_path in png_files:
        sidecar_path = target_dir / f"{png_path.name}.json"
        if not sidecar_path.exists():
            errors.append(f"{png_path.name}: Sidecar missing at {sidecar_path}")
            continue

        try:
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if "input_text" not in sidecar_data:
                errors.append(f"{png_path.name}: Sidecar missing input_text field")
        except Exception as e:
            errors.append(f"{png_path.name}: Failed to parse sidecar: {e}")

        if sha_map and png_path.name in sha_map:
            actual_sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
            if actual_sha != sha_map[png_path.name]:
                errors.append(
                    f"{png_path.name}: SHA256 mismatch: actual {actual_sha} != expected {sha_map[png_path.name]}"
                )

    return errors, 1 if errors else 0


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

    if args.set == "varied":
        errors, exit_code = check_varied_fixtures(target_dir or (Path("tests/fixtures/approved/jabcode") / "varied"))
        if errors:
            print("=== Varied Fixture Integrity Errors ===", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return exit_code
        print("All varied JAB Code fixtures passed integrity checks.")
        return 0

    if args.set == "photos":
        errors, exit_code = check_photo_fixtures(target_dir or (Path("tests/fixtures/approved/jabcode") / "photos"))
        if errors:
            print("=== Photo Fixture Integrity Errors ===", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return exit_code
        print("All photo JAB Code fixtures passed integrity checks.")
        return 0

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

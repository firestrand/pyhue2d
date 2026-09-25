#!/usr/bin/env python3
"""Check availability and functionality of the official jabcode binary (Phase V15 Data Gate).

Exit codes:
  0: Official binary is present and successfully verified against example1.png
  1: Binary failed execution or output validation
  2: Official binary is absent (Phase V16 is blocked on LOCAL-DATA-05)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pyhue2d.reference import SubprocessReferenceCodec, get_reference_codec  # noqa: E402


def check_import_boundary() -> list[str]:
    """Verify that domain modules do not import SubprocessReferenceCodec."""
    domain_dir = ROOT / "src" / "pyhue2d"
    violations = []
    for py_file in domain_dir.rglob("*.py"):
        if py_file.name == "reference.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        if "SubprocessReferenceCodec" in content:
            violations.append(f"{py_file}: imports or references SubprocessReferenceCodec")
        if "subprocess" in content and "cli.py" not in py_file.name:
            violations.append(f"{py_file}: imports subprocess outside reference adapter")
    return violations


def main() -> int:
    # Check import boundary first
    violations = check_import_boundary()
    if violations:
        print("=== Reference Import Boundary Violations ===", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    codec = get_reference_codec()
    if codec is None:
        print(
            "Phase V16 is blocked: official jabcode binary (LOCAL-DATA-05) is absent.\n"
            "Set JABCODE_READER_PATH or install jabcodeReader into PATH to unblock.",
            file=sys.stderr,
        )
        return 2

    example1_path = ROOT / "tests" / "fixtures" / "approved" / "jabcode" / "example1.png"
    if not example1_path.exists():
        print(f"Error: example1 fixture not found at {example1_path}", file=sys.stderr)
        return 1

    try:
        payload = codec.decode(example1_path)
        expected = b"Hello, JAB Code!"
        if payload != expected:
            print(f"Error: reference binary returned {payload!r}, expected {expected!r}", file=sys.stderr)
            return 1
        print("Official jabcode binary verified successfully against example1.png.")
        return 0
    except Exception as e:
        print(f"Error executing reference binary: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

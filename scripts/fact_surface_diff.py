#!/usr/bin/env python3
"""Check git diff for paths listed in tests/support/fact_surface.txt."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    fact_surface_file = Path("tests/support/fact_surface.txt")
    if not fact_surface_file.exists():
        print(f"Error: {fact_surface_file} not found", file=sys.stderr)
        return 1

    with open(fact_surface_file, encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not paths:
        return 0

    # Run git diff against HEAD for these paths
    cmd = ["git", "diff", "HEAD", "--"] + paths
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    diff_output = result.stdout.strip()
    if not diff_output:
        return 0

    print("=== Detected Fact Surface Changes ===")
    print(diff_output)

    # Check for review bypass marker or environment variable
    reviewed_env = os.environ.get("Evidence-Surface-Reviewed") or os.environ.get("EVIDENCE_SURFACE_REVIEWED")
    marker_file = Path("tests/support/.surface-reviewed")

    if reviewed_env == "yes" or marker_file.exists():
        print("Fact surface changes explicitly marked as reviewed.")
        return 0

    print(
        "\nError: Fact surface changed without 'Evidence-Surface-Reviewed: yes' "
        "or marker file tests/support/.surface-reviewed",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

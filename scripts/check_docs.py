"""Fail when the README names a command flag the CLI does not provide."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    errors: list[str] = []
    if "<username>" in pyproject or "<username>" in readme or "<username>" in contributing:
        errors.append("placeholder <username> is still in README, CONTRIBUTING, or pyproject URLs")
    if "utility_scripts" in readme:
        errors.append("README references utility_scripts, which is not in this tree")
    if "--camera" in readme:
        errors.append("README documents --camera, which the CLI does not provide")

    help_text = ""
    for args in (
        ["--help"],
        ["encode", "--help"],
        ["decode", "--help"],
        ["inspect", "--help"],
        ["validate", "--help"],
        ["convert", "--help"],
    ):
        help_proc = subprocess.run(
            [sys.executable, "-m", "pyhue2d", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        help_text += help_proc.stdout + help_proc.stderr
        if help_proc.returncode != 0:
            errors.append(f"pyhue2d {' '.join(args)} exited {help_proc.returncode}")
    for flag in sorted(set(re.findall(r"--[a-z][a-z0-9-]*", readme))):
        if flag not in help_text:
            errors.append(f"README mentions {flag}, which is absent from pyhue2d --help")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("docs check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

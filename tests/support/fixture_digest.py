"""Fixture digest and location helpers for JAB Code approved fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

APPROVED_DIR = Path("tests/fixtures/approved/jabcode")


def get_fixture_path(name: str) -> Path:
    """Return path to fixture file in approved directory."""
    path = APPROVED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture {name} not found in {APPROVED_DIR}")
    return path


def sha256_fixture(name_or_path: str | Path) -> str:
    """Compute sha256 of fixture file."""
    if isinstance(name_or_path, str):
        path = get_fixture_path(name_or_path)
    else:
        path = name_or_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sha256sums() -> dict[str, str]:
    """Load approved SHA256SUMS file into {filename: sha256} mapping."""
    sha_file = APPROVED_DIR / "SHA256SUMS"
    if not sha_file.exists():
        raise FileNotFoundError(f"SHA256SUMS not found in {APPROVED_DIR}")

    result: dict[str, str] = {}
    for line in sha_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            result[Path(parts[1]).name] = parts[0]
    return result

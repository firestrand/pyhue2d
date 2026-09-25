"""Test that importing pyhue2d is pure and does not resize or touch reference images.

Fact: JAB.IMPORT.FIXTURES_UNCHANGED.v1
Evidence: EV-01
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tests.support.fixture_digest import get_fixture_path


def test_import_fixtures_unchanged():
    """Importing pyhue2d must not contain _ensure_reference_image_sizes or mutate files."""
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Check 1: _ensure_reference_image_sizes must not exist in pyhue2d module
    code_check = (
        "import pyhue2d\n"
        "assert not hasattr(pyhue2d, '_ensure_reference_image_sizes'), "
        "'_ensure_reference_image_sizes hook must be removed'\n"
    )
    res_check = subprocess.run([sys.executable, "-c", code_check], capture_output=True, text=True, check=False)
    assert res_check.returncode == 0, f"_ensure_reference_image_sizes is still present in pyhue2d:\n{res_check.stderr}"

    # Check 2: Even if legacy path exists with a non-252 image, import must NOT resize or touch it
    legacy_dir = repo_root / "src" / "tests" / "example_images"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = legacy_dir / "test_non_252.png"

    try:
        # Create a non-252 dummy image
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(test_img_path)
        original_bytes = test_img_path.read_bytes()
        original_mtime = test_img_path.stat().st_mtime_ns

        # Run fresh import in subprocess
        code_import = "import pyhue2d\n"
        res_import = subprocess.run(
            [sys.executable, "-c", code_import],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        assert res_import.returncode == 0, f"Import failed:\n{res_import.stderr}"

        # Assert file was not modified or resized
        current_bytes = test_img_path.read_bytes()
        current_mtime = test_img_path.stat().st_mtime_ns
        assert current_bytes == original_bytes, "Image bytes were altered during import pyhue2d!"
        assert current_mtime == original_mtime, "Image mtime was altered during import pyhue2d!"

        with Image.open(test_img_path) as im:
            assert im.size == (100, 100), f"Image was resized to {im.size} on import!"

    finally:
        if test_img_path.exists():
            test_img_path.unlink()
        if legacy_dir.exists():
            try:
                legacy_dir.rmdir()
                (repo_root / "src" / "tests").rmdir()
            except OSError:
                pass

"""Test suite for fixture checker script."""

import subprocess
import sys


def test_fixture_checker_runs():
    cmd = [sys.executable, "scripts/check_jabcode_fixtures.py"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Checker failed with code {res.returncode}:\n{res.stdout}\n{res.stderr}"


def test_fixture_checker_rejects_pem_private_key(tmp_path):
    from pathlib import Path

    # Copy approved fixtures to tmp_path
    approved_dir = Path("tests/fixtures/approved/jabcode")
    for name in ["example1.png", "example1.png.json"]:
        (tmp_path / name).write_bytes((approved_dir / name).read_bytes())

    # Inject PEM private key header into sidecar
    sidecar = tmp_path / "example1.png.json"
    sidecar.write_text(
        '{"input_text": "-----BEGIN PRIVATE KEY-----\\nMIIEvgIBADANBg...", "final_image_size": [252, 252]}',
        encoding="utf-8",
    )

    cmd = [sys.executable, "scripts/check_jabcode_fixtures.py", "--dir", str(tmp_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
    assert "Contains PEM private key header" in res.stderr

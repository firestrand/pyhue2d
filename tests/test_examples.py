"""Test that all repository examples run and pass cleanly."""

import sys
from pathlib import Path

# Add repo root to sys.path so examples module is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from examples.generate_all import main as run_examples  # noqa: E402


def test_all_examples_execute_successfully():
    """Verify that all example scripts run to completion without error."""
    run_examples()

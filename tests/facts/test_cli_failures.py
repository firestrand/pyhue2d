"""Failure messages for a missing decode input and an unknown subcommand."""

from __future__ import annotations

import pytest

from pyhue2d.cli import main


def test_unknown_subcommand_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-command"])
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_missing_decode_input_fails(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["decode", "--input", "no-such-file.png"])
    assert code != 0
    captured = capsys.readouterr()
    assert "failed" in captured.err.lower()

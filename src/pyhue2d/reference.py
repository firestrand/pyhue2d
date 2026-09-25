"""Reference codec port and adapters for external JABCode binaries (Phase V15)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from .jabcode.exceptions import JABCodeError


@runtime_checkable
class ReferenceCodec(Protocol):
    """Port for interacting with an external reference JABCode implementation."""

    def decode(self, image_path: Path | str) -> bytes:
        """Decode JABCode image at image_path and return payload bytes."""
        ...


class SubprocessReferenceCodec:
    """Adapter that delegates decode operations to an external jabcode binary."""

    def __init__(self, binary_path: Path | str):
        self.binary_path = Path(binary_path)
        if not self.binary_path.exists() and not shutil.which(str(binary_path)):
            raise FileNotFoundError(f"Reference binary not found: {binary_path}")

    def decode(self, image_path: Path | str) -> bytes:
        """Decode image using the external reference CLI."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        proc = subprocess.run(
            [str(self.binary_path), str(path)],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")
            raise JABCodeError(f"Reference binary decode failed (exit {proc.returncode}): {err}")

        # Standard reference reader prints the decoded payload to stdout
        return proc.stdout.strip()


class NullReferenceCodec:
    """Capture-backed null adapter for contract testing when binary is absent."""

    def __init__(self, capture_dir: Path | str = "tests/fixtures/approved/jabcode"):
        self.capture_dir = Path(capture_dir)

    def decode(self, image_path: Path | str) -> bytes:
        """Look up expected plaintext from the approved fixture sidecar."""
        path = Path(image_path)
        sidecar_path = self.capture_dir / f"{path.name}.json"
        if sidecar_path.exists():
            import json

            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if "input_text" in data:
                return data["input_text"].encode("utf-8")
        raise JABCodeError(f"No captured reference oracle available for {path.name}")


def get_reference_codec(binary_path: Path | str | None = None) -> ReferenceCodec | None:
    """Obtain configured reference codec, or None if no official binary is configured."""
    candidates: list[Path | str | None] = [
        binary_path,
        os.environ.get("JABCODE_READER_PATH"),
        shutil.which("jabcodeReader"),
        shutil.which("jabcode"),
        Path.home() / "Projects" / "jabcode" / "src" / "jabcodeReader" / "bin" / "jabcodeReader",
    ]
    for cand in candidates:
        if cand is not None:
            cand_path = Path(cand)
            if cand_path.exists() and cand_path.is_file():
                try:
                    return SubprocessReferenceCodec(cand_path)
                except Exception:
                    continue
    return None

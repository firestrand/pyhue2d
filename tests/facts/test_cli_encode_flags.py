"""Fact test EV-07: CLI encode flags and default dimensions.

Fact: JAB.CLI.ENCODE_FLAGS.v1
Given the example1 plaintext, when it is encoded with the default CLI
and when it is encoded with an explicit non-default module size,
then the default image is 252x252 and the non-default image has a different pixel size.
Defaults equal sidecar module_size 12 and quiet_zone 4.
"""

from pathlib import Path

import pytest
from PIL import Image

from pyhue2d.cli import create_enhanced_parser, main
from pyhue2d.jabcode.cli_args import EncodeArgs
from tests.support.sidecar import load_sidecar


def test_cli_encode_flags_default_and_module_size(tmp_path: Path):
    """Test default CLI encode produces 252x252 and --module-size 1 produces 21x21."""
    sidecar = load_sidecar("example1.png.json")
    plaintext = sidecar.input_text
    expected_module_size = sidecar.module_size  # 12
    expected_quiet_zone = sidecar.quiet_zone  # 4

    # Verify parser defaults match sidecar
    parser = create_enhanced_parser()
    args_default = parser.parse_args(["encode", "--input", "dummy.txt", "--output", "dummy.png"])
    assert args_default.module_size == expected_module_size == 12
    assert args_default.quiet_zone == expected_quiet_zone == 4

    # Verify EncodeArgs defaults
    input_file = tmp_path / "example1_plaintext.txt"
    input_file.write_text(plaintext, encoding="utf-8")
    output_default = tmp_path / "example1_default.png"
    output_mod1 = tmp_path / "example1_mod1.png"

    encode_args_def = EncodeArgs(input_source=input_file, output_path=output_default)
    assert encode_args_def.module_size == 12
    assert encode_args_def.quiet_zone == 4

    # Encode with default CLI settings
    code_def = main(["encode", "--input", str(input_file), "--output", str(output_default)])
    assert code_def == 0
    assert output_default.exists()

    with Image.open(output_default) as img_def:
        assert img_def.size == (252, 252)

    # Encode with explicit non-default module size (--module-size 1)
    code_mod1 = main(["encode", "--input", str(input_file), "--output", str(output_mod1), "--module-size", "1"])
    assert code_mod1 == 0
    assert output_mod1.exists()

    with Image.open(output_mod1) as img_mod1:
        assert img_mod1.size == (21, 21)
        assert img_mod1.size != (252, 252)


def test_cli_encode_rejects_unsupported_flags(tmp_path: Path):
    """Verify that unsupported flags (out-of-range --version, --mask-pattern != 7, --encoding-mode) are rejected."""
    input_file = tmp_path / "in.txt"
    input_file.write_text("Hello", encoding="utf-8")
    out_file = tmp_path / "out.png"

    # Out-of-range version (valid range is 1-32)
    code_ver = main(["encode", "--input", str(input_file), "--output", str(out_file), "--version", "35"])
    assert code_ver != 0

    # Unsupported mask pattern
    code_mask = main(["encode", "--input", str(input_file), "--output", str(out_file), "--mask-pattern", "0"])
    assert code_mask != 0

    # Unsupported encoding mode
    code_mode = main(["encode", "--input", str(input_file), "--output", str(out_file), "--encoding-mode", "Byte"])
    assert code_mode != 0


def test_cli_encode_version_flag_supported(tmp_path: Path):
    """Verify that valid --version flag produces corresponding symbol dimensions."""
    input_file = tmp_path / "in.txt"
    input_file.write_text("Hello, version 4!", encoding="utf-8")
    out_file = tmp_path / "out_v4.png"

    code_ver = main(
        ["encode", "--input", str(input_file), "--output", str(out_file), "--palette", "16", "--version", "4"]
    )
    assert code_ver == 0
    assert out_file.exists()

    with Image.open(out_file) as img:
        # Version 4 side is 17 + 4*4 = 33 modules. Default module_size = 12 -> 396x396
        assert img.size == (396, 396)

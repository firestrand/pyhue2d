"""Enhanced command-line interface for PyHue2D with comprehensive validation."""

import argparse
import logging
import sys

from .core import decode, encode
from .jabcode.cli_args import DecodeArgs, EncodeArgs, validate_args
from .jabcode.exceptions import JABCodeError, JABCodeValidationError
from .jabcode.image_format import ImageConverter, ImageFormatDetector


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
        quiet: Enable quiet mode (errors only)
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def handle_jabcode_error(error: JABCodeError, operation: str) -> None:
    """Handle JABCode-specific errors with helpful messages.

    Args:
        error: JABCode error to handle
        operation: Operation that failed (encode/decode)
    """
    if isinstance(error, JABCodeValidationError):
        print(
            f"❌ {operation.capitalize()} validation failed: {error.message}",
            file=sys.stderr,
        )
        if error.context:
            for key, value in error.context.items():
                print(f"   {key}: {value}", file=sys.stderr)
    else:
        print(f"❌ {operation.capitalize()} failed: {error.message}", file=sys.stderr)
        if error.error_code:
            print(f"   Error code: {error.error_code}", file=sys.stderr)


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with comprehensive options.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="pyhue2d",
        description="Encode and decode colour 2-D barcodes (JABCode implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode text file to JABCode
  pyhue2d encode --input message.txt --output code.png --palette 8 --ecc-level M

  # Decode JABCode image
  pyhue2d decode --input code.png --output decoded.txt

  # Validate image format
  pyhue2d validate --input code.png

  # Convert image format
  pyhue2d convert --input code.jpg --output code.png --format PNG
""",
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Encode command
    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode data to JABCode image",
        description="Encode text or binary data into a JABCode color barcode image",
    )
    encode_parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="FILE",
        help="Input file containing data to encode",
    )
    encode_parser.add_argument(
        "--output",
        "-o",
        required=True,
        metavar="IMAGE",
        help="Output image file path (PNG recommended)",
    )
    encode_parser.add_argument(
        "--palette",
        "-p",
        type=int,
        default=8,
        choices=[4, 8, 16, 32, 64, 128, 256],
        help="Number of colors in palette (default: 8)",
    )
    encode_parser.add_argument(
        "--ecc-level",
        "-e",
        default="M",
        choices=["L", "M", "Q", "H"],
        help="Error correction level (default: M)",
    )
    encode_parser.add_argument("--version", type=int, metavar="N", help="Symbol version (1-32, default: auto)")
    encode_parser.add_argument(
        "--quiet-zone",
        type=int,
        default=2,
        metavar="N",
        help="Quiet zone size in modules (default: 2)",
    )
    encode_parser.add_argument(
        "--mask-pattern",
        type=int,
        default=7,
        choices=range(8),
        help="Mask pattern (0-7, default: 7)",
    )
    encode_parser.add_argument(
        "--module-size",
        type=int,
        default=1,
        metavar="N",
        help="Module size in pixels (default: 1)",
    )
    encode_parser.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")
    encode_parser.add_argument(
        "--encoding-mode",
        choices=[
            "Numeric",
            "Alphanumeric",
            "Uppercase",
            "Lowercase",
            "Mixed",
            "Punctuation",
            "Byte",
        ],
        help="Force specific encoding mode (default: auto-detect)",
    )

    # Decode command
    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode JABCode image to data",
        description="Decode JABCode color barcode image to original data",
    )
    decode_parser.add_argument("--input", "-i", required=True, metavar="IMAGE", help="Input JABCode image file")
    decode_parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output file for decoded data (default: print to stdout)",
    )
    decode_parser.add_argument(
        "--detection-method",
        default="scanline",
        choices=["scanline", "contour", "hybrid"],
        help="Pattern detection method (default: scanline)",
    )
    decode_parser.add_argument("--no-perspective", action="store_true", help="Disable perspective correction")
    decode_parser.add_argument("--no-error-correction", action="store_true", help="Disable error correction")
    decode_parser.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")
    decode_parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw bytes without text interpretation",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate image format and JABCode suitability",
        description="Analyze image format and suitability for JABCode operations",
    )
    validate_parser.add_argument("--input", "-i", required=True, metavar="IMAGE", help="Image file to validate")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert image format",
        description="Convert image between different formats",
    )
    convert_parser.add_argument("--input", "-i", required=True, metavar="IMAGE", help="Input image file")
    convert_parser.add_argument("--output", "-o", required=True, metavar="IMAGE", help="Output image file")
    convert_parser.add_argument(
        "--format",
        default="PNG",
        choices=["PNG", "JPEG", "BMP", "TIFF", "WEBP"],
        help="Target image format (default: PNG)",
    )
    convert_parser.add_argument(
        "--quality",
        type=int,
        default=95,
        metavar="N",
        help="JPEG quality (1-100, default: 95)",
    )
    convert_parser.add_argument("--optimize", action="store_true", default=True, help="Enable optimization")
    convert_parser.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")

    return parser


def command_encode(args: argparse.Namespace) -> int:
    """Handle encode command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Create and validate encode arguments
        encode_args = validate_args(
            EncodeArgs,
            input_source=args.input,
            output_path=args.output,
            palette=args.palette,
            ecc_level=args.ecc_level,
            version=args.version or "auto",
            quiet_zone=args.quiet_zone,
            mask_pattern=args.mask_pattern,
            module_size=args.module_size,
            force_overwrite=args.force,
            verbose=args.verbose,
            encoding_mode=args.encoding_mode,
        )

        # Read input data
        with open(encode_args.input_source, "rb") as fh:
            payload = fh.read()

        # Encode data
        image = encode(payload, **encode_args.to_encoder_settings())

        # Save image
        image.save(encode_args.output_path)

        if not args.quiet:
            print(f"✅ Successfully encoded '{encode_args.input_source}' to '{encode_args.output_path}'")
            print(f"   Image size: {image.size}")
            print(f"   Colors: {encode_args.palette}, ECC: {encode_args.ecc_level}")
            print(f"   Data size: {len(payload)} bytes")

        return 0

    except JABCodeError as e:
        handle_jabcode_error(e, "encode")
        return 1
    except Exception as e:
        print(f"❌ Encoding failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def command_decode(args: argparse.Namespace) -> int:
    """Handle decode command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Create and validate decode arguments
        decode_args = validate_args(
            DecodeArgs,
            input_path=args.input,
            output_path=args.output,
            detection_method=args.detection_method,
            perspective_correction=not args.no_perspective,
            error_correction=not args.no_error_correction,
            force_overwrite=args.force,
            verbose=args.verbose,
            raw_output=args.raw,
        )

        # Decode data
        decoded_data = decode(str(decode_args.input_path))

        if not args.quiet:
            print(f"✅ Successfully decoded '{decode_args.input_path}'")
            print(f"   Data length: {len(decoded_data)} bytes")

        # Handle output
        if decode_args.output_path:
            # Write to file
            with open(decode_args.output_path, "wb") as fh:
                fh.write(decoded_data)
            if not args.quiet:
                print(f"   Saved to: {decode_args.output_path}")
        else:
            # Print to stdout
            if decode_args.raw_output:
                # Raw binary output
                sys.stdout.buffer.write(decoded_data)
            else:
                # Try to decode as text
                try:
                    text = decoded_data.decode("utf-8")
                    if text.isprintable():
                        print(f"   Content: {text}")
                    else:
                        print("   Content: <binary data>")
                        sys.stdout.buffer.write(decoded_data)
                except UnicodeDecodeError:
                    print("   Content: <binary data>")
                    sys.stdout.buffer.write(decoded_data)

        return 0

    except JABCodeError as e:
        handle_jabcode_error(e, "decode")
        return 1
    except Exception as e:
        print(f"❌ Decoding failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def command_validate(args) -> int:
    """Handle validate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        detector = ImageFormatDetector()
        image_info = detector.validate_image(args.input)

        print(f"📊 Image Analysis: {args.input}")
        print(f"   Format: {image_info['format']} ({image_info['description']})")
        print(f"   Size: {image_info['width']}×{image_info['height']} pixels")
        print(f"   Color mode: {image_info['mode']}")
        print(f"   File size: {image_info['file_size']:,} bytes")
        print(f"   Lossless: {'Yes' if image_info['lossless'] else 'No'}")
        print(f"   Transparency: {'Yes' if image_info['has_transparency'] else 'No'}")

        suitability = image_info["jabcode_suitability"]
        print(f"\n🎯 JABCode Suitability: {suitability['overall_score']}/100")

        if suitability["strengths"]:
            print("   Strengths:")
            for strength in suitability["strengths"]:
                print(f"     ✅ {strength}")

        if suitability["warnings"]:
            print("   Warnings:")
            for warning in suitability["warnings"]:
                print(f"     ⚠️  {warning}")

        if suitability["issues"]:
            print("   Issues:")
            for issue in suitability["issues"]:
                print(f"     ❌ {issue}")

        if image_info["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in image_info["recommendations"]:
                print(f"     • {rec}")

        return 0

    except JABCodeError as e:
        handle_jabcode_error(e, "validate")
        return 1
    except Exception as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def command_convert(args) -> int:
    """Handle convert command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        converter = ImageConverter()
        result = converter.convert_image(
            input_path=args.input,
            output_path=args.output,
            target_format=args.format,
            quality=args.quality,
            optimize=args.optimize,
        )

        if not args.quiet:
            print(f"✅ Successfully converted '{result['input_path']}'")
            print(f"   {result['input_format']} → {result['target_format']}")
            print(f"   Size: {result['input_size']} → {result['output_size']}")
            print(f"   Mode: {result['input_mode']} → {result['output_mode']}")
            print(f"   Output: {result['output_path']}")

        return 0

    except JABCodeError as e:
        handle_jabcode_error(e, "convert")
        return 1
    except Exception as e:
        print(f"❌ Conversion failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main(argv=None) -> None:
    """Enhanced main function with comprehensive error handling.

    Args:
        argv: Command-line arguments (default: sys.argv)
    """
    parser = create_enhanced_parser()
    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    # Validate global arguments
    if getattr(args, "verbose", False) and getattr(args, "quiet", False):
        print("❌ Cannot specify both --verbose and --quiet", file=sys.stderr)
        sys.exit(1)

    # Route to command handlers
    command_map = {
        "encode": command_encode,
        "decode": command_decode,
        "validate": command_validate,
        "convert": command_convert,
    }

    if args.command in command_map:
        exit_code = command_map[args.command](args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()

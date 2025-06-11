"""Command-line interface for PyHue2D."""

import argparse

from .core import encode, decode


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyhue2d",
        description="Encode and decode colour 2-D barcodes",
    )
    sub = parser.add_subparsers(dest="command")

    enc_parser = sub.add_parser(
        "encode", help="Encode data to a colour barcode (e.g. JAB Code)"
    )
    enc_parser.add_argument(
        "--input", required=True, help="Input file"
    )
    enc_parser.add_argument(
        "--output", required=True, help="Output image path"
    )
    enc_parser.add_argument(
        "--palette", type=int, default=8, help="Number of colours"
    )
    enc_parser.add_argument(
        "--ecc-level", default="M", help="Error correction level"
    )

    dec_parser = sub.add_parser("decode", help="Decode a colour barcode image")
    dec_parser.add_argument("--input", required=True, help="Input image path")

    args = parser.parse_args(argv)

    if args.command == "encode":
        try:
            with open(args.input, "rb") as fh:
                payload = fh.read()
            image = encode(payload, colors=args.palette, ecc_level=args.ecc_level)
            image.save(args.output)
            print(f"✅ Successfully encoded '{args.input}' to '{args.output}'")
            print(f"   Image size: {image.size}")
            print(f"   Colors: {args.palette}, ECC: {args.ecc_level}")
        except FileNotFoundError:
            parser.error(f"Input file '{args.input}' not found")
        except Exception as exc:
            parser.error(f"Encoding failed: {exc}")
    elif args.command == "decode":
        try:
            decoded_data = decode(args.input)
            print(f"✅ Successfully decoded '{args.input}'")
            print(f"   Data length: {len(decoded_data)} bytes")
            
            # Try to decode as text if possible
            try:
                text = decoded_data.decode('utf-8')
                if text.isprintable():
                    print(f"   Content: {text}")
                else:
                    print(f"   Content: <binary data>")
            except UnicodeDecodeError:
                print(f"   Content: <binary data>")
                
        except FileNotFoundError:
            parser.error(f"Input file '{args.input}' not found")
        except Exception as exc:
            parser.error(f"Decoding failed: {exc}")
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()

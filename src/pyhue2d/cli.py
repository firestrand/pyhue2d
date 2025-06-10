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
            encode(payload, colors=args.palette, ecc_level=args.ecc_level)
        except NotImplementedError as exc:
            parser.error(str(exc))
    elif args.command == "decode":
        try:
            decode(args.input)
        except NotImplementedError as exc:
            parser.error(str(exc))
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()

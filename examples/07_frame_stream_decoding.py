"""Video stream and multi-frame barcode decoding example.

This example demonstrates:
- Decoding barcodes from a stream of video frames using `FileFrameSource` and `decode_frame`.
- Simulating a camera feed where initial frames contain non-barcode content or noise.
- Automatic frame iteration until a valid JAB Code symbol is detected and decoded.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import pyhue2d
from pyhue2d.frame import FileFrameSource, decode_frame

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_empty_path = OUTPUT_DIR / "07_stream_frame_1.png"
    frame_barcode_path = OUTPUT_DIR / "07_stream_frame_2.png"

    print("=== PyHue2D Multi-Frame Stream Decoding ===\n")

    payload = "Real-time camera video stream barcode detection."
    print(f"Target payload: '{payload}'")

    # Frame 1: Simulate a background scene / non-barcode video frame
    print("1. Creating Frame 1 (empty scene without barcode)...")
    noise_frame = Image.fromarray(np.random.randint(220, 256, (300, 300, 3), dtype=np.uint8))
    noise_frame.save(frame_empty_path)
    print(f"   Saved: {frame_empty_path.name}")

    # Frame 2: Video frame containing a valid barcode
    print("2. Creating Frame 2 (frame containing barcode)...")
    barcode_img = pyhue2d.encode(payload, colors=8, ecc_level=3)
    barcode_img.save(frame_barcode_path)
    print(f"   Saved: {frame_barcode_path.name}")

    # Decode across frame stream
    print("3. Scanning video stream with decode_frame()...")
    stream_source = FileFrameSource([frame_empty_path, frame_barcode_path])
    result = decode_frame(stream_source)

    decoded_text = result.payload.decode("utf-8")
    print(f"   Successfully decoded from stream: '{decoded_text}'")
    print(f"   Version: {result.version}, Colors: {result.color_count}")
    assert decoded_text == payload

    print("\nSuccess: Stream frame decoding verified!")


if __name__ == "__main__":
    main()

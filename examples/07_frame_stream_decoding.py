"""Video stream and multi-frame barcode decoding example.

This example demonstrates:
- Decoding barcodes from a stream of video frames using `FileFrameSource` and `decode_frame`.
- Simulating a camera feed where initial frames contain non-barcode background noise.
- Deterministic seeded frame generation to avoid git tree diffs.
- Verifying why non-barcode frames fail decode (nonzero LDPC syndrome error).
- Automatic frame iteration until a valid JAB Code symbol is detected and decoded.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import pyhue2d  # noqa: E402
from pyhue2d.frame import FileFrameSource, decode_frame  # noqa: E402
from pyhue2d.jabcode.exceptions import JABCodeError  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"


def main(output_dir: Path = OUTPUT_DIR) -> None:
    """Run the example and write its assets to the requested directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_empty_path = output_dir / "07_stream_frame_1.png"
    frame_barcode_path = output_dir / "07_stream_frame_2.png"

    print("=== PyHue2D Multi-Frame Stream Decoding ===\n")

    payload = "Real-time camera video stream barcode detection."
    print(f"Target payload: '{payload}'")

    # Frame 1: Deterministic seeded background scene / noise frame without barcode
    print("1. Creating Frame 1 (seeded background noise scene without barcode)...")
    rng = np.random.default_rng(42)
    noise_frame = Image.fromarray(rng.integers(220, 256, (300, 300, 3), dtype=np.uint8))
    noise_frame.save(frame_empty_path)
    print(f"   Saved: {frame_empty_path.name}")

    # Verify that decoding Frame 1 alone raises JABCodeError due to nonzero syndrome
    print("   Verifying Frame 1 rejection reason on standalone decode...")
    try:
        pyhue2d.decode(noise_frame)
        raise AssertionError("Frame 1 should not decode successfully!")
    except JABCodeError as e:
        print(f"   Correctly rejected Frame 1: {e}")

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

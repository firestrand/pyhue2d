# PyHue2D Examples Guide

This guide provides practical code walkthroughs for all core and advanced features of **PyHue2D**, including generalized multi-color palettes, multi-symbol docked topologies, camera perspective unwarping, binary/Unicode payloads, and vector exports.

Executable scripts generating sample outputs are available in the [`examples/`](../examples/) directory.

---

## 1. Basic Roundtrip

Encode a text payload to a standard 8-color JAB Code (Version 1) and decode it back:

```python
import pyhue2d

# 1. Encode text payload
payload = "Hello, colourful world! PyHue2D JAB Code."
img = pyhue2d.encode(payload, colors=8, ecc_level=3, module_size=12)
img.save("barcode.png")

# 2. Decode from file path or PIL Image
result = pyhue2d.decode("barcode.png")

print("Decoded text:", result.payload.decode("utf-8"))
print("Symbol version:", result.version)
print("Color count:", result.color_count)
print("Error correction level:", result.ecc_level)
print("Corrected errors:", result.corrected_error_count)
```

---

## 2. Multi-Color Palettes (ISO/IEC 23634:2022)

PyHue2D supports 4, 8, 16, 32, and 64 colors, dramatically scaling data capacity by increasing bits per module:

| Color Count | Bits / Module | Palette Distribution | Typical Use Case |
|---|---|---|---|
| **4** | 2 | Black, Magenta, Yellow, Cyan | High-contrast or monochrome-adjacent print |
| **8** | 3 | RGB + CMY + Black + White | Default standard JAB Code |
| **16** | 4 | 4 × 2 × 2 RGB color cube | Moderate density on digital displays |
| **32** | 5 | 4 × 4 × 2 RGB color cube | High-density digital barcodes |
| **64** | 6 | 4 × 4 × 4 RGB color cube | Maximum capacity per module |

### Code Example

```python
import pyhue2d

payload = b"Higher color counts allow packing more data into identical symbol dimensions."

# 16-color symbol
img_16 = pyhue2d.encode(payload, colors=16, version=2, ecc_level=3)
res_16 = pyhue2d.decode(img_16)
assert res_16.payload == payload
assert res_16.color_count == 16

# 64-color symbol (6 bits per module!)
img_64 = pyhue2d.encode(payload, colors=64, version=3, ecc_level=3)
res_64 = pyhue2d.decode(img_64)
assert res_64.payload == payload
assert res_64.color_count == 64
```

---

## 3. Multi-Symbol Docked Topologies

When data exceeds the capacity of a single symbol, multiple symbols can be docked together (supporting up to the API bound of 61 symbols; the examples demonstrate 2, 3, and 4 symbols). The primary symbol hosts primary finder patterns; docked secondary symbols attach via compact docking trees. The payload is split across tiles according to their available capacity, each tile appends docking tree footers indicating neighbor adjacencies, and each tile is LDPC-encoded independently. During decoding, the master symbol identifies docked neighbors, each tile is decoded and error-corrected independently, and the payload slices are reassembled in breadth-first traversal order.

PyHue2D supports arbitrary docking topologies via the `symbol_count` and `columns` parameters:

### Horizontal Docking (e.g. 3 symbols wide)

```python
import pyhue2d

payload = "Horizontal layout fits wide margins or banner surfaces."
img = pyhue2d.encode(payload, symbol_count=3, columns=3, module_size=10)
res = pyhue2d.decode(img)
assert res.symbol_count == 3
```

### Vertical Docking (e.g. 2 symbols tall)

```python
import pyhue2d

payload = "Vertical layout fits column gutters or receipt slips."
img = pyhue2d.encode(payload, symbol_count=2, columns=1, module_size=10)
res = pyhue2d.decode(img)
assert res.symbol_count == 2
```

### 2D Grid Docking (e.g. 4 symbols in 2 × 2)

```python
import pyhue2d

payload = "Massive payload distributed across a 2x2 multi-symbol grid..." * 10
img = pyhue2d.encode(payload, symbol_count=4, module_size=10)
res = pyhue2d.decode(img)
assert res.symbol_count == 4
```

---

## 4. Camera Frame Perspective Unwarping

Barcodes photographed by mobile cameras or scanned from paper surfaces frequently exhibit perspective skew, rotation, paper margins, and illumination gradients. PyHue2D automatically detects quadrilateral contours, unwarps perspective homographies, and decodes the symbols:

```python
from PIL import Image
import pyhue2d

# Open a photograph or scan containing a skewed barcode
photo = Image.open("examples/output/04_camera_horizontal_docked_3.png")

# decode() detects the symbol contour, corrects perspective, and decodes
result = pyhue2d.decode(photo)
print(f"Decoded from photo: {result.payload.decode('utf-8')}")
print(f"Detected symbols: {result.symbol_count}")
```

---

## 5. Multilingual Unicode & Arbitrary Binary Data

PyHue2D natively handles arbitrary binary bytes (such as compressed archives, encrypted payloads, or raw tokens) as well as multilingual UTF-8 text:

```python
import pyhue2d

# 1. Multilingual UTF-8
unicode_text = "PyHue2D: 🎨 JAB Code! 日本語 / 中文 / Español / Deutsch (Grüße) / Ελληνικά."
img_u = pyhue2d.encode(unicode_text, colors=8, ecc_level=3)
res_u = pyhue2d.decode(img_u)
assert res_u.payload.decode("utf-8") == unicode_text

# 2. Raw binary data (all 256 byte values 0x00..0xFF)
binary_payload = bytes(range(256))
img_b = pyhue2d.encode(binary_payload, colors=8, version=10, ecc_level=3)
res_b = pyhue2d.decode(img_b)
assert res_b.payload == binary_payload
```

---

## 6. Vector Graphics Export (SVG & PDF)

For web interfaces or high-resolution print production, export symbols directly to SVG or PDF without pixelation:

```python
import pyhue2d

payload = "Resolution-independent vector export."
barcode = pyhue2d.encode(payload, colors=8, ecc_level=3)

# Export to SVG
svg_xml = pyhue2d.export_svg(barcode, module_size=12, output_path="barcode.svg")

# Export to PDF
pdf_bytes = pyhue2d.export_pdf(barcode, module_size=12, output_path="barcode.pdf")
```

---

## 7. Video and Frame Stream Decoding

Process sequential frames from a camera or video stream with `FileFrameSource` and `decode_frame`:

```python
from pathlib import Path
from pyhue2d.frame import FileFrameSource, decode_frame

# Stream of image frame paths (e.g. from camera capture)
frames = [Path(f"frame_{i:04d}.png") for i in range(100)]
source = FileFrameSource(frames)

# decode_frame iterates through frames until a valid barcode is found
result = decode_frame(source)
print(f"Decoded: {result.payload}")
```

---

## 8. Command-Line Interface (CLI)

The functionality is also available via the command line:

```bash
# Encode text to PNG
pyhue2d encode --input message.txt --output message.png --palette 8

# Encode with specific version and color depth
pyhue2d encode --input binary.bin --output binary.png --palette 16 --version 4

# Decode image to stdout or file
pyhue2d decode --input message.png

# Inspect module matrix dimensions and raw bitstream hex
pyhue2d inspect --input message.png
```

---

## 9. Running Repository Examples

To generate all sample image assets and execute the full test suite of examples:

```bash
# Generate all example assets into examples/output/
python examples/generate_all.py
```

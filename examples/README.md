# PyHue2D Examples

This directory provides self-contained, executable examples showcasing the features and capabilities of **PyHue2D**.

---

## 🚀 Running the Examples

You can run all examples at once to generate sample barcode assets in `examples/output/`:

```bash
# Using Python
python examples/generate_all.py

# Or using just
just examples
```

You can also run any example individually:

```bash
python examples/01_basic_roundtrip.py
python examples/02_color_depths.py
python examples/03_multisymbol_docking.py
python examples/04_camera_unwarp.py
python examples/05_binary_and_unicode.py
python examples/06_vector_export.py
python examples/07_frame_stream_decoding.py
```

---

## 📂 Example Catalog

### 1. [01_basic_roundtrip.py](01_basic_roundtrip.py)
* **What it demonstrates:**
  * Encoding a text string to an 8-color JAB Code (Version 1).
  * Saving the barcode image to PNG.
  * Decoding the image to a structured `DecodeResult`.
  * Inspecting symbol version, color count, error correction level, and corrected error count.
* **Output:** `output/01_basic_roundtrip.png`

### 2. [02_color_depths.py](02_color_depths.py)
* **What it demonstrates:**
  * Multi-color palette support conforming to ISO/IEC 23634:2022:
    * **4 colors** (2 bits/module): Black, Magenta, Yellow, Cyan.
    * **8 colors** (3 bits/module): Primary RGB + CMY + White + Black.
    * **16 colors** (4 bits/module): 4×2×2 RGB color cube.
    * **32 colors** (5 bits/module): 4×4×2 RGB color cube.
    * **64 colors** (6 bits/module): 4×4×4 RGB color cube.
  * Demonstrating data density gains with higher color counts.
  * Full raster encoding and decoding roundtrips across all five color depths.
* **Outputs:**
  * `output/02_color_4.png`
  * `output/02_color_8.png`
  * `output/02_color_16.png`
  * `output/02_color_32.png`
  * `output/02_color_64.png`

### 3. [03_multisymbol_docking.py](03_multisymbol_docking.py)
* **What it demonstrates:**
  * Docking multiple symbols together into structured topologies (demonstrated with 2, 3, and 4 symbols, supporting up to the API bound of 61 symbols).
  * Payload partitioning across tiles according to capacity, docking tree footers indicating neighbor adjacencies, and independent per-tile LDPC encoding.
  * Three docking topologies:
    * **Horizontal docking** (3 symbols side-by-side, `columns=3`).
    * **Vertical docking** (2 symbols stacked, `columns=1`).
    * **2D grid docking** (4 symbols in a 2×2 grid).
  * Full multi-symbol decode and concatenated payload reassembly.
* **Outputs:**
  * `output/03_multisymbol_horizontal_3.png`
  * `output/03_multisymbol_vertical_2.png`
  * `output/03_multisymbol_grid_4.png`

### 4. [04_camera_unwarp.py](04_camera_unwarp.py)
* **What it demonstrates:**
  * Generating synthetic camera perspective homography warps with paper margins, 4-corner perspective distortion, and illumination gradients.
  * Aspect-ratio-aware quadrilateral contour detection and perspective unwarping.
  * Decoding docked multi-symbol barcodes directly from synthetic camera frames.
* **Outputs:**
  * `output/04_camera_horizontal_docked_3.png`
  * `output/04_camera_vertical_docked_2.png`

### 5. [05_binary_and_unicode.py](05_binary_and_unicode.py)
* **What it demonstrates:**
  * Multilingual Unicode UTF-8 strings (Japanese, Chinese, Spanish, German, Greek, emojis).
  * Arbitrary binary byte arrays (including all 256 byte values 0x00 to 0xFF).
  * Dynamic symbol version selection based on payload length.
  * Byte-for-byte fidelity verification upon decoding.
* **Outputs:**
  * `output/05_unicode_text.png`
  * `output/05_binary_payload.png`

### 6. [06_vector_export.py](06_vector_export.py)
* **What it demonstrates:**
  * Resolution-independent vector graphic exports:
    * **SVG** for responsive web interfaces and digital signage.
    * **PDF** for commercial printing and document embedding.
  * Parsing and rasterizing both SVG and PDF vector outputs and decoding them back to original payload.
* **Outputs:**
  * `output/06_barcode.svg`
  * `output/06_barcode.pdf`

### 7. [07_frame_stream_decoding.py](07_frame_stream_decoding.py)
* **What it demonstrates:**
  * Scanning a sequence of video frames using `FileFrameSource` and `decode_frame`.
  * Deterministic seeded noise frame generation ensuring clean git trees.
  * Verifying expected decode failure reason (nonzero syndrome error) on background scenes.
  * Skipping non-barcode frames and automatically decoding as soon as a barcode frame appears.
* **Outputs:**
  * `output/07_stream_frame_1.png`
  * `output/07_stream_frame_2.png`

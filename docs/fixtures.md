# JAB Code Fixtures and Provenance

## Approved Fixture Corpus (`tests/fixtures/approved/jabcode/`)

All approved fixtures in `tests/fixtures/approved/jabcode/` originate from official `jabcode` CLI reference generation sessions.

- **Manifest Index**: `tests/fixtures/approved/jabcode/examples_manifest.json` indexes 15 reference examples, detailing input texts, output filenames, and CLI invocation arguments.
- **Hash Integrity**: Every approved PNG is hashed and verified against `tests/fixtures/approved/jabcode/SHA256SUMS` (and corresponding directory checksums in `varied/` and `photos/`). Any modification, resizing, or compression difference violates data integrity.
- **Sidecar Metadata**: Symbols are accompanied by a `<symbol>.png.json` sidecar capturing matrix dimensions, color palettes, ECC levels, encoded bitstreams, or crash records.
- **Secret Scan Policy**: No fixture contains private keys, credentials, or sensitive headers (`BEGIN PRIVATE KEY`). Verified automatically by `scripts/check_jabcode_fixtures.py`.
- **Refresh Rule**: Re-export from official reference CLI if official format/version changes or `SHA256SUMS` mismatch occurs. Synthetic creation or manual resizing of fixtures is strictly forbidden.

---

### Fixture Categories and Oracles

The fixture corpus comprises distinct categories of oracles with varying sidecar completeness:

#### 1. Manifest Captures (15 rows in `examples_manifest.json`)
The 15 manifest entries fall into two distinct structural groups:
- **Single-Symbol Captures (`example1.png` through `example5.png`, `minimum_text.png`)**:
  - Geometry: Version 1, 8 colors, ECC level 3, 21×21 module matrix (rendered at 252×252 px with module size 12).
  - Sidecar Data: Full 21×21 `symbol_matrix` and 1160-bit `encoded_data_hex`. `ecc_data_hex` is recorded as `"not available"`.
- **Multi-Symbol Docked Captures (`maximum_text.png`, `multi_block_2.png` through `multi_block_9.png`)**:
  - Geometry: Multi-symbol docked topologies (2 to 10 symbols) using Version 10 (57×57 modules per symbol) with ECC level 0 on all symbols.
  - Arguments: In `maximum_text.png`, `--symbol-version` lists 20 tens (`10 10 ...`) because the official writer takes separate width and height version parameters per symbol (`[10, 10]`).
  - Sidecar Data: Contains full per-symbol 57×57 `symbol_matrix` arrays. Both `encoded_data_hex` and `ecc_data_hex` are recorded as `"omitted_large_data"` and `"omitted_large_ecc"`.
  - *Manifest ECC Settings*: The manifest rows for `maximum_text.png` and `multi_block_2.png` through `multi_block_9.png` specify `ecc-level: 0`, accurately reflecting the ECC level 0 used when generating these approved multi-symbol images.

#### 2. Codec-Pinning Off-Manifest Captures
Critical oracle files not listed in `examples_manifest.json`:
- **Mode Captures (`mode_upper.png` through `mode_byte.png`)**:
  - Seven single-symbol captures sharing Version 1, 8 colors, ECC level 3 geometry.
  - **Authoritative LDPC Oracles**: Unlike the manifest rows, these sidecars contain both full `encoded_data_hex` (1160 bits) and full `ecc_data_hex` (2088-bit LDPC codeword and parity bits), pinning byte-level LDPC encoding and error correction.
- **ASAN Multi-Symbol (`asan_multi2.png`)**:
  - A distinct two-symbol image from `multi_block_2.png` (different SHA-256, identical 684×1368 px dimensions), with matrices present and codewords omitted.
- **Version-32 Multi-Block Series (`multi_block_2_v32.png` through `multi_block_9_v32.png`)**:
  - Share a single long Lorem Ipsum plaintext. Matrix and codeword fields are recorded as `"omitted_large_matrix"` and `"omitted_large_data"`. Verified by end-to-end decoding back to the source text.
- **Writer Crash Record (`test_block2.png.json`)**:
  - This sidecar records an official `jabcodeWriter` crash (`Bus error: 10`, signal 10), preserving failure triage metadata rather than a valid symbol sidecar.

#### 3. Varied Parameters Dataset (`tests/fixtures/approved/jabcode/varied/`)
Captures verifying non-default color depths and error correction levels:
- `varied_4color_ecc3.png`: 4-color symbol with ECC level 3.
- `varied_8color_ecc5.png`: 8-color symbol with ECC level 5.
- Accompanied by their own `SHA256SUMS` and full sidecars.

#### 4. Photograph Dataset (`tests/fixtures/approved/jabcode/photos/`)
- `photo_example1.png`: OpenCV optical camera simulation of approved `example1.png` authorized on 2026-09-25 (modeling paper margin, perspective homography warp, ambient illumination gradient, and Gaussian optical PSF blur), verified by official `jabcodeReader` and `tests/facts/test_photo_scan.py`.
- Accompanied by sidecar metadata and verified by `tests/facts/test_photo_scan.py`.

---

## Quarantined Captures (`tests/fixtures/quarantine/`)

Incomplete 0-byte sidecars from aborted generator runs and their associated placeholder images are quarantined in `tests/fixtures/quarantine/` to prevent parser errors:
- Placeholder 252×252 `multi_block_2.png` through `multi_block_9.png` (and empty `.png.json` files).
- Placeholder 252×252 `maximum_text.png` (and empty `.png.json` file).

These files are completely distinct from the approved 57×57 multi-symbol fixtures in `tests/fixtures/approved/jabcode/`.

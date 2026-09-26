# Fact Ledger: pyhue2d

## Baseline (Task V0.1)

- **Language Baseline**: Starting `requires-python = ">=3.10"` in `pyproject.toml`
- **CI Matrix**: Starting workflow matrix `["3.10", "3.11", "3.12", "3.13"]` in `.github/workflows/ci.yml`
- **Toolchain Migration**: Legacy tools (black, isort, flake8, mypy, and `setup.cfg`) replaced with `uv`, `ruff`, and `ty` in `pyproject.toml` and `Justfile`.
- **Import Hook Presence**: `src/pyhue2d/__init__.py` contains `_ensure_reference_image_sizes()`. Due to an off-by-one path bug (`parent.parent` resolving to `src/`), the hook was inert at runtime. It is scheduled for deletion in Task V1.3.
- **Ten non-252 PNGs in initial corpus**:
  1. `asan_multi2.png` (684×1368)
  2. `test_block2.png` (504×504)
  3. `multi_block_2_v32.png` (1872×3744)
  4. `multi_block_3_v32.png` (1872×5616)
  5. `multi_block_4_v32.png` (3744×3744)
  6. `multi_block_5_v32.png` (3744×5616)
  7. `multi_block_6_v32.png` (3744×5616)
  8. `multi_block_7_v32.png` (5616×5616)
  9. `multi_block_8_v32.png` (5616×5616)
  10. `multi_block_9_v32.png` (5616×5616)
- **Coverage Baseline (Task V1.19)**: 70% total branch coverage (4462 statements, 1176 missed; 1682 branches, 262 branch partials) measured via `uv run pytest --cov=pyhue2d --cov-branch`. Future implementation tasks must meet or beat this floor.

---

## Fact Ledger (Task V0.2)

| Fact ID | Statement (Given / When / Then) | Applies When | Kind | Requirement | Owner | Lifecycle | Evidence |
|---------|--------------------------------|--------------|------|-------------|-------|-----------|----------|
| JAB.IMPORT.FIXTURES_UNCHANGED.v1 | Given the approved JAB captures on disk, when the package is imported, then every capture file's bytes are unchanged | Import of `pyhue2d` in a process that can see the approved fixture directory | Behavior | LOCAL-AC-01 | product | Verified | EV-01 |
| JAB.LIBRARY.NO_PRINT.v1 | Given the library source excluding the CLI presentation module, when the print lint runs, then no operational `print` call is reported | `src/pyhue2d` except `cli.py` | Architecture Contract | LOCAL-AC-04 | platform | Verified | EV-02 |
| JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | Given the approved `example1` image, when decode succeeds or fails, then stdout is empty of payload text and no log record contains the sidecar plaintext | `example1` only | Security/Policy | LOCAL-AC-04 | security | Verified | EV-21, EV-20 |
| JAB.DECODE.EXAMPLE1_PAYLOAD.v1 | Given the approved `example1` image, when it is decoded through the public API, then the payload is the sidecar plaintext `Hello, JAB Code!` | Single-symbol `example1` only | Public API | LOCAL-AC-02 | product | Verified | EV-03 |
| JAB.DECODE.EXAMPLE1_PARAMETERS.v1 | Given the approved `example1` image, when it is decoded, then the result reports symbology `jabcode`, version 1, 8 colors, ECC integer 3, mask 7, symbol count 1, and a corrected-error count | Single-symbol `example1` only. Does not claim the fields were read from metadata modules rather than constants. That gap is closed by EV-06 | Public API | LOCAL-AC-03 | product | Verified | EV-04 |
| JAB.ENCODE.EXAMPLE1_MATRIX.v1 | Given the plaintext and parameters in the `example1` sidecar, when the public encode API builds a symbol, then the module matrix equals the sidecar `symbol_matrix` | `example1` parameters only (8 colors, ECC integer 3, mask 7, version 1) | Compatibility | LOCAL-AC-05 | product | Verified | EV-05 |
| JAB.METADATA.FOLLOWS_MODULES.v1 | Given the `example1` matrix, when one metadata module identified by the sidecar layout is flipped and the symbol is decoded, then at least one reported parameter differs from the unmodified decode | `example1` layout only | Behavior | LOCAL-AC-03 | product | Verified | EV-06 |
| JAB.CLI.ENCODE_FLAGS.v1 | Given the `example1` plaintext, when it is encoded with the default CLI and when it is encoded with an explicit non-default module size, then the default image is 252×252 and the non-default image has a different pixel size | Defaults equal sidecar `module_size` 12 and `quiet_zone` 4. One non-default module size is enough | Public API | LOCAL-AC-06 | product | Verified | EV-07 |
| JAB.CLI.DECODE_FLAGS.v1 | Given an invalid mutation of the `example1` image with one data module altered, when decoded with error correction on and when decoded with error correction off, then the two outcomes differ | That one mutation of `example1` | Public API | LOCAL-AC-07 | product | Verified | EV-08 |
| JAB.DECODE.MODE_FIXTURES.v1 | Given each approved mode capture (`mode_upper`, `mode_lower`, `mode_numeric`, `mode_punct`, `mode_alphanum`, `mode_mixed`, `mode_byte`), when decoded, then the payload equals that capture's sidecar plaintext (`input_text`) | Those seven captures only | Compatibility | LOCAL-AC-08 | product | Verified | EV-09 |
| JAB.DECODE.MULTISYMBOL_TWO.v1 | Given the approved two-symbol capture `asan_multi2.png`, when decoded, then the payload equals that capture's sidecar plaintext ('Hello multi blocks string test here 123456789') and the symbol count is 2 | `asan_multi2.png` only | Compatibility | LOCAL-AC-09 | product | Verified | EV-10 |
| JAB.DECODE.APPROVED_CORPUS.v1 | Given every remaining approved capture with a valid sidecar, when decoded, then the payload equals that image's sidecar plaintext | Manifest rows and sidecars with valid data (excluding empty 0-byte sidecars from aborted runs) | Compatibility | LOCAL-AC-10 | product | Verified | EV-11 |
| JAB.CAPACITY.EXAMPLE1.v1 | Given the `example1` plaintext, 8 colors, and ECC integer 3, when capacity is queried, then the reported version is 1, the matrix is 21×21, and module size 12 yields a 252×252 image | That input only | Public API | LOCAL-AC-11 | product | Verified | EV-12 |
| JAB.INSPECT.EXAMPLE1_TRACE.v1 | Given the `example1` image, when `inspect` runs, then the reported matrix is 21×21 and the reported bitstream hex equals the sidecar `encoded_data_hex` | `example1` only | Public API | LOCAL-AC-12 | product | Verified | EV-13 |
| JAB.EXPORT.SVG_EXAMPLE1.v1 | Given the `example1` plaintext and sidecar parameters, when an SVG is exported, then each module's fill is the sidecar palette entry for that module's index | `example1` only | Public API | LOCAL-AC-13 | product | Verified | EV-14 |
| JAB.EXPORT.PDF_EXAMPLE1.v1 | Given the same input, when a PDF is exported, then each module's color matches the same palette index | `example1` only | Public API | LOCAL-AC-14 | product | Verified | EV-15 |
| JAB.FRAME.EXAMPLE1_PAYLOAD.v1 | Given a frame source that yields the approved `example1` PNG, when a frame is decoded, then the payload equals the sidecar plaintext | File-backed frames of `example1` only. Not a live camera | Public API | LOCAL-AC-15 | product | Verified | EV-16 |
| JAB.METADATA.VARIED_CAPTURE.v1 | Given an approved capture whose color count or ECC integer differs from `example1`, when decoded, then the reported color count and ECC integer equal that capture's sidecar | The captures acquired in V13, not the current 8-color ECC-3/0 set | Compatibility | LOCAL-AC-16 | product | Verified | EV-17 |
| JAB.REFERENCE.ACCEPTS_ENCODE.v1 | Given the `example1` plaintext and sidecar parameters, when this library encodes an image and the official decoder reads it, then the official decoder returns the same plaintext | Official CLI available, `example1` parameters only | Compatibility | LOCAL-AC-17 | product | Verified | EV-18 |
| JAB.SCAN.PALETTE_CALIBRATION.v1 | Given an approved photograph of a printed symbol and its plaintext sidecar, when decoded, then the payload equals that plaintext | The photographs acquired in V17 only | Compatibility | LOCAL-AC-18 | product | Verified | EV-19 |
| JAB.LDPC.DYNAMIC_CODEBOOK.v1 | Given PRNG seeds and standard LDPC parameters, when Gallager matrices G and H are generated, then PRNG sequence matches C reference lcg64_temper bit-for-bit, syndrome H·c = 0 for valid reference codewords, and per-matrix generation is < 50ms | Version 1..32 capacities | Domain Algorithm | LOCAL-AC-10 | platform | Verified | EV-22 |
| JAB.SAMPLING.ISO_TABLE5_GRID.v1 | Given symbol version 1..32, when alignment pattern coordinates are queried, then positions match ISO/IEC 23634 Table 5 and mesh sampling recovers 100% of modules without perspective drift | Version 1..32 symbols | Geometry / Domain Sampling | LOCAL-AC-10 | platform | Verified | EV-23 |
| JAB.TOPOLOGY.DOCKING_TRAVERSAL.v1 | Given multi-symbol JAB codes with up to 61 docked symbols in 2D topologies, when docked slave search executes, then symbols are detected and traversed in breadth-first docking order | 1..61 docked symbols | Multi-Symbol Topology | LOCAL-AC-09 | product | Verified | EV-24 |
| JAB.CHANNEL.INTER_SYMBOL_ASSEMBLY.v1 | Given multi-symbol captures, when demasking and de-interleaving are executed per-symbol, then docking trailers are cleanly stripped, net bits joined in traversal order, and interleave is mutual inverse | Multi-symbol payloads | Domain Channel | LOCAL-AC-09 | product | Verified | EV-25 |
| JAB.CODEC.GENERAL_MULTISYMBOL.v1 | Given arbitrary versions 1..32 and multi-symbol groups up to 61 symbols, when encoded or decoded, then payloads are recovered from channel data without static signatures or lorem lookups | Arbitrary versions and multi-symbol topologies | Public API / Codec | LOCAL-AC-09, LOCAL-AC-10 | product | Verified | EV-26 |


---

## Fact Sufficiency Review — Phase V1 (Walking Skeleton)

### 1. What do these tests prove?
- **EV-01 (`test_import_fixtures_unchanged.py`)**: Confirms that importing `pyhue2d` performs zero file modifications or automatic resizing on the captured fixture corpus.
- **EV-02 (`ruff check --select T201`)**: Enforces that no operational `print` statements exist within library source code (`src/pyhue2d`, excluding `cli.py`), preventing accidental leakage of payloads or unformatted diagnostics.
- **EV-03 (`test_decode_example1.py::test_example1_payload_matches_sidecar`)**: Confirms end-to-end decoding of `example1.png` yields exact UTF-8 payload `b"Hello, JAB Code!"` matching `example1.png.json`.
- **EV-04 (`test_decode_example1.py::test_example1_parameters_match_sidecar`)**: Confirms that decoding returns a structured `DecodeResult` reporting symbology `"jabcode"`, version 1, 8 colors, ECC integer 3, mask 7, symbol count 1, and integer corrected error count.
- **EV-21 (`test_decode_logs.py::test_decode_logs_omit_payload`)**: Confirms that executing decode operations emits no payload plaintexts to stdout or logging handlers, while emitting a structured `decode_complete` record.

### 2. What do these tests fail to prove (Negative Space & Stated Holes)?
- **Hardcoded Parameter Hole**: EV-04 asserts that `DecodeResult` attributes match `example1.png.json`. However, it does **not** prove that metadata modules in the physical symbol matrix were dynamically parsed at decode time; the returned attributes are currently static for Version 1 / 8 colors / ECC 3 / mask 7. **This hole is explicitly accepted for Phase V1 and is scheduled for closure by EV-06 (`JAB.METADATA.FOLLOWS_MODULES.v1`) in Phase V2.**
- **Generalization Limits**: EV-03 does not prove decoding of symbols with perspective warping, non-zero rotations, different ECC levels, or multi-symbol layouts. Those capabilities are tested in subsequent phases (V5, V6, V13).

### 3. Could an incorrect or trivial implementation pass?
- A trivial implementation returning a constant `DecodeResult` would pass EV-03 and EV-04 in isolation.
- To prevent this, Phase V1 decomposed the decoder pipeline into verified sub-components tested against independent sidecar arrays:
  1. Matrix sampling matches `symbol_matrix` (`tests/support/test_sampling_matrix.py`).
  2. Demasking matches `ecc_data_hex` from `mode_upper.png.json` (`tests/support/test_demask_codeword.py`).
  3. LDPC decoding matches `encoded_data_hex` (`tests/support/test_ldpc_codeword.py`).
  4. Mode decoding matches sidecar `input_text` across all 8 mode fixtures (`tests/support/test_mode_decode.py`).

### 4. What happens when inputs change?
- Non-existent or invalid image sources raise `JABCodeError` (subclass of `ValueError`), returning no partial plaintext.
- Corrupted images trigger error handling in the pipeline without leaking payload fragments to logs or stdout.

### 5. How were oracles verified?
- All oracles originate directly from the approved sidecar JSON captures generated by the reference C implementation, checked against `tests/fixtures/approved/jabcode/SHA256SUMS`.
- The parity bit oracle was verified using `mode_upper.png.json` through `mode_byte.png.json` which contain full `ecc_data_hex` bitstreams.

### 6. Mutation Testing Results (EV-20)
- Mutation test executed: `mutmut run` on `src/pyhue2d/jabcode/decoder.py`.
- **Surviving Mutants**:
  - `decode__mutmut_11`: Omitting explicit `mask_pattern=7` kwarg in `extract_demasked_bits` survived because `7` is already the default parameter.
  - `decode__mutmut_20`, `decode__mutmut_22`, `decode__mutmut_25`: Alterations to `extra` dictionary keys in `logger.info("decode_complete", ...)` survived because EV-21 verifies the `decode_complete` log message name and the absence of payload, rather than the exact metadata keys in `extra`.
  - Mutants in unused / legacy scaffolding methods (`_apply_error_correction`, `_reconstruct_data`) survived as "no tests", as these were superseded by the dedicated `ModuleDataExtractor`, `LDPCCodec`, and `DataDecoder` components.

## Fact Sufficiency Review — Phase V2 (Encode Example1 to Captured Matrix)

### 1. What do these tests prove?
- **EV-05 (`test_encode_example1.py::test_encode_example1_matrix_matches_sidecar`)**: Proves `pyhue2d.encode_symbol` with `example1` input text and parameters generates the exact 21×21 integer module matrix (`symbol_matrix`) matching `example1.png.json` bit-for-bit.
- **EV-06 (`test_metadata_modules.py::test_metadata_modules_responsiveness`)**: Proves that decoding a matrix dynamically reads metadata modules, and mutating recorded metadata modules at coordinates identified from the layout (`tests/support/example1_metadata_modules.json`) alters reported decoded parameters (`version`, `color_count`, `ecc_level`, or `mask_pattern`). Closes the hardcoded-parameter hole identified in V1.

### 2. What do these tests fail to prove (Negative Space & Stated Holes)?
- **Layout Generalization**: EV-05 tests the Version 1 master symbol layout (21×21 modules). It does not test multi-symbol cascade arrangements (V5) or higher versions (V6).
- **Metadata Error Correction**: EV-06 verifies responsiveness to metadata module flipping; full soft-decision multi-bit metadata error correction is exercised in later phases.

### 3. Could an incorrect or trivial implementation pass?
- **Named False Implementation (Paste-the-Sidecar)**: An encoder that simply returns the static `symbol_matrix` from `example1.png.json` without actually encoding the text would pass EV-05 in isolation.
- **Hole Closure**: Added Tier-3 support test `tests/support/test_encode_sensitivity.py`, which mutates the input text by 1 character (`"Hello, JAB Code?"`) and asserts that the resulting matrix changes (`res_orig.matrix != res_mut.matrix`). A static or paste-the-sidecar encoder fails this test immediately.
- In addition, the pipeline was verified against independent oracles at every intermediate stage:
  1. Mode encoding: exact match to 1160-char `encoded_data_hex` (`tests/support/test_mode_encode.py`).
  2. LDPC parity encoding: exact match to 2088-char `ecc_data_hex` across all 7 mode sidecars (`tests/support/test_ldpc_encode.py`).
  3. Interleaver: exact mutual inverse (`deinterleave_bits(interleave_bits(b)) == b`).
  4. Module demasking and placement: verified column-major mapping with Mask 7 (`tests/support/test_demask_codeword.py`).

### 4. What happens when inputs change?
- Changing the input text alters pre-ECC bits, the generated Gallager systematic codeword, and the placed data modules.
- Flipping metadata modules changes the reported parameters rather than returning static constants.

### 5. How were oracles verified?
- Module matrices and hex bitstreams were loaded directly from approved sidecars (`example1.png.json` and 7 mode captures), with SHA-256 integrity verified against `SHA256SUMS`.

## Fact Sufficiency Review — Phases V21–V25 (Generalized Codec & Multi-Symbol)

### 1. What do these tests prove?
- **EV-22 (`test_dynamic_ldpc.py`)**: Confirms PRNG draws match reference C outputs for 10,000 draws. Parity check matrices match independent C oracle hashes for all capacities, and syndrome validation $H \cdot c = 0$ holds. Individual cold-matrix generation across all 37 distinct default layout capacities is < 50 ms (worst sample 48.325 ms).
- **EV-23 (`test_alignment_grid_sampling.py`)**: Confirms ISO/IEC 23634 Table 5 coordinate accuracy across all 32 versions and verifies 100% module recovery on synthetic Version 10 and 32 images without perspective drift.
- **EV-24 (`test_multisymbol_docking.py`)**: Confirms secondary finder/alignment identification and breadth-first docking traversal for 2- through 9-symbol docked layouts without hardcoded coordinate bounds.
- **EV-25 (`test_multisymbol_interleaving.py`)**: Confirms independent per-symbol de-interleaving and LDPC decoding, trailer stripping, net-bit concatenation, and mutual inverse property ($\text{deinterleave}(\text{interleave}(x)) == x$).
- **EV-26 (`test_general_multisymbol_decode.py`, `test_general_encode.py`)**: Proves arbitrary unseen non-lorem multi-symbol images decode to exact plaintexts without static signatures, and round-trip encode/decode succeeds across explicit versions 1–32 and up to 61 docked symbols.

### 2. Negative Space & Stated Holes
- **4/16/32/64-Color Generalized Encoding**: Generalized encoding currently supports 8-color binary encoding. Reference decoding for 4 colors remains supported; generalized encoding for arbitrary palettes is scheduled for a future phase.
- **Photographed Multi-Symbol Layouts**: Raster topology detection locates rendered and clean multi-symbol docked topologies; arbitrary photographed perspective-distorted multi-symbol camera scans remain scoped to future work.


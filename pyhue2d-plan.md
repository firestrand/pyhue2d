# Development Plan: JAB Code fidelity and Python standards baseline

**Guide Version**: 2.6 (2026-08-13)
**Plan Version**: 1.3.0
**Status**: Active
**Mode**: Vertical-Slice
**Plan Type**: Existing-System Feature
**Public-library delta**: The public functions and the CLI are the promised product, so those contracts are Tier 1 rather than Tier 2. Foundation is a baseline-and-delta assessment of the existing tree, not a greenfield scaffold.
**Planning Horizon**: Rolling-Wave
**Plan Set**: pyhue2d
**Builds On**: none (no prior plan in this set)
**Inherited Facts**: none (no project Fact Ledger exists yet)
**Supersedes**: 1.2.0
**Requirements authority**: There is no PRD file. The requirements are the operator instruction to plan every item from the 2026-09-23 codebase review, plus `Python Code Standards.md` (reviewed June 2026) for the toolchain and language baseline. Identifiers prefixed `LOCAL-` are planning IDs, not PRD IDs.
**PRD Trace**: LOCAL-AC-01..LOCAL-AC-18, LOCAL-NFR-01..LOCAL-NFR-04
**Real Data Policy**: Approved reference captures already in the repo are the only representative evidence. They are PNGs and JSON sidecars produced by the official `jabcode` CLI. Individual sidecars define oracle parameters (`input_text`, `symbol_matrix`, `palette`, etc.). In `tests/example_images`:
1. `example1.png.json` contains `input_text`, `symbol_matrix` (21×21), and `encoded_data_hex` (1160 bits pre-ECC data); its `ecc_data_hex` is `"not available"` (parity bits omitted).
2. The seven mode captures (`mode_upper.png.json` through `mode_byte.png.json`) contain `input_text`, `symbol_matrix` (21×21), `encoded_data_hex` (1160 bits), and `ecc_data_hex` (2088 bits). They share `example1`'s symbol geometry (Version 1, 8 colors, ECC integer 3, 21×21) and serve as the authoritative oracle for LDPC codeword and parity bits.
3. The two-symbol capture `asan_multi2.png` (`asan_multi2.png.json`, Version 10, ECC integer 0, 57×57 matrices) is the single canonical multi-symbol fixture with complete metadata.
4. Version-32 sidecars (`multi_block_2_v32.png.json` through `multi_block_9_v32.png.json`) store `"omitted_large_*"` placeholders for matrix and codewords; validation against them is strictly end-to-end payload decoding to `input_text`.
5. Files from an aborted generator run (`multi_block_2.png.json` through `multi_block_9.png.json` and `maximum_text.png.json`) are empty 0-byte files and are quarantined.
Invalid mutations of approved captures are allowed for rejection tests. No generated payloads.
**Generated Data Authorization**: `None`
**Provider Policy**: Pillow, NumPy, and the stdlib stay direct dependencies (KISS). The official `jabcode` CLI is the only external codec provider; it is isolated behind a port in the slice that cross-checks against it. A live camera is an optional frame-source adapter and is not part of the walking skeleton.
**Data & Provider Readiness Summary**: Captures with 8 colors and ECC integers 3 (`example1` and 7 mode captures) and 0 (`asan_multi2.png` and version-32 files) are on disk. Ten PNGs are not 252×252, including `asan_multi2.png` and every `*_v32.png`. `src/pyhue2d/__init__.py` contains a latent helper `_ensure_reference_image_sizes()` intended to resize PNGs under `tests/example_images` to 252×252 on import; due to an off-by-one path bug (`parent.parent` resolving to `src/`), it was inert in production, but represents an unacceptable side-effect and architectural hazard that must be cleanly excised in V1.3. Captures at palettes other than 8 colors (e.g. 4, 16, 32, 64 colors), other ECC levels (e.g. 1, 2, 4, 5), the official `jabcode` binary (`LOCAL-DATA-05`), and photographed scans (`LOCAL-DATA-07`) are not in the repo.
**Slice Ordering Rationale**: Data readiness, then risk, then value. Cleanly excise the import hook so no package import performs filesystem side effects. The walking skeleton is decode of `example1` through the public API, because every later claim (encode match, modes, multi-symbol, export, camera frames) reuses that path. Encode-to-matrix is next because it is the highest architectural risk and the fixture for it is already on disk. CLI flags, mode fixtures, and the rest of the approved corpus follow while their captures exist. Varied-parameter metadata, a live reference-decoder check, and photo calibration wait on data gates. Vector export, inspect, capacity, and file-frame decode need no new captures and sit after the symbol is real. HiQ, color QR, WebAssembly, and video are out of scope until a JAB symbol round-trips.
**Fact Policy**: Tier-1 facts are requirement-traced in the Fact Ledger below. Evidence surfaces are declared in the Evidence Index. Semantics change only through Fact Change tasks approved by the requirement owner. This plan's ledger is the set of facts it introduces. V0 publishes the same rows to `docs/fact-ledger.md` and `docs/evidence-index.md`.
**Python standards applied**: `Python Code Standards.md` — Python 3.12+ baseline, `uv`, `ruff`, `ty`, stdlib `logging` with lazy formatting, public APIs typed, branch coverage as a floor. The development-plan guide's example library table (typer, rich, structlog, pydantic, polars) is an example, not this project's standard. This plan does not replace `argparse` or stdlib `logging`.
**Toolchain owner**: The toolchain scaffolding for `uv`, `ruff`, and `ty` is in place in `pyproject.toml` and `.github/workflows/ci.yml`, with `setup.cfg` removed and `Justfile` providing standard workflows (`just check`, `just test`, `just build`). The repository currently resolves under Python `>=3.10` with CI matrix `["3.10", "3.11", "3.12", "3.13"]`. Task V0.7 advances the language baseline to Python `>=3.12` (`target-version = 'py312'`, ty `python-version = '3.12'`), and Task V0.8 updates the CI matrix to `["3.12", "3.13", "3.14"]`.
**Coverage policy**: Existing-system baseline. Coverage is officially recorded at V1 close (Task V1.19) once the import hook is safely deleted in V1.3 and EV-01..EV-04 are green. During V0, coverage collection is not run to maintain strict isolation before V0A moves data. From V1 close onward: no regression against that baseline, ≥90% branch coverage on changed behavioral code, ≥95% branch coverage on new domain codec logic, 100% of Active Tier-1 bindings executed by the phase verification command. The repo measures branch coverage with `pytest-cov`; keep that tool. Do not impose a retroactive whole-repo 90% gate.
**Repos in Scope**: `pyhue2d` only.
**Outstanding Blockers / Human Decisions**:
- V1.3 cleanly removes the `_ensure_reference_image_sizes` hook from `src/pyhue2d/__init__.py`.
- V0.4 (compliance): allowed source for the codec algorithm. Blocks V1.8, V1.10, V1.12, and V2.6.
- V0.5 (product): public ECC vocabulary. Blocks V12. Does not block reporting the integer `3` from the `example1` sidecar.
- LOCAL-DATA-05 official `jabcode` binary: blocks V16.
- LOCAL-DATA-06 palette captures other than 8 colors or ECC levels other than 3 and 0: blocks V14.
- LOCAL-DATA-07 photographed scans: blocks V18.
- V11.1 (product): keep, move, or remove the `opencv-python` dependency added for image processing/detection. Blocks only V11.4 and V11.5. File-frame decode does not wait.

## Requirements inventory

| ID | Claim | Fact |
|----|-------|------|
| LOCAL-AC-01 | Importing the package does not write fixture files | JAB.IMPORT.FIXTURES_UNCHANGED.v1 |
| LOCAL-AC-02 | Decoding the approved `example1` symbol returns the sidecar plaintext | JAB.DECODE.EXAMPLE1_PAYLOAD.v1 |
| LOCAL-AC-03 | That decode reports symbology, version, color count, ECC integer, mask, symbol count, and corrected-error count from the sidecar | JAB.DECODE.EXAMPLE1_PARAMETERS.v1 |
| LOCAL-AC-04 | Library modules do not emit operational `print` output, and decode logs do not contain the payload | JAB.LIBRARY.NO_PRINT.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 |
| LOCAL-AC-05 | Encoding the `example1` plaintext with the sidecar parameters reproduces `symbol_matrix` | JAB.ENCODE.EXAMPLE1_MATRIX.v1 |
| LOCAL-AC-06 | CLI and Python defaults for module size and quiet zone match the `example1` capture, and accepted encode flags change the image | JAB.CLI.ENCODE_FLAGS.v1 |
| LOCAL-AC-07 | Accepted decode flags change decoder behavior on an invalid mutation of `example1` | JAB.CLI.DECODE_FLAGS.v1 |
| LOCAL-AC-08 | Each approved single-symbol mode capture decodes to its sidecar plaintext | JAB.DECODE.MODE_FIXTURES.v1 |
| LOCAL-AC-09 | The approved two-symbol capture decodes to its sidecar plaintext in symbol order | JAB.DECODE.MULTISYMBOL_TWO.v1 |
| LOCAL-AC-10 | Every remaining approved capture with a valid sidecar decodes to its sidecar plaintext | JAB.DECODE.APPROVED_CORPUS.v1 |
| LOCAL-AC-11 | A capacity query for the `example1` plaintext and sidecar parameters reports version 1 and 21×21, and the pixel size implied by module size 12 | JAB.CAPACITY.EXAMPLE1.v1 |
| LOCAL-AC-12 | `inspect` reports the `example1` matrix size and the sidecar bitstream hex | JAB.INSPECT.EXAMPLE1_TRACE.v1 |
| LOCAL-AC-13 | SVG export of `example1` reproduces the captured palette colors for every module | JAB.EXPORT.SVG_EXAMPLE1.v1 |
| LOCAL-AC-14 | PDF export of `example1` reproduces the same module colors | JAB.EXPORT.PDF_EXAMPLE1.v1 |
| LOCAL-AC-15 | A frame source that yields the `example1` PNG decodes to the same plaintext | JAB.FRAME.EXAMPLE1_PAYLOAD.v1 |
| LOCAL-AC-16 | Reported parameters track the symbol when color count or ECC differs from the `example1` capture | JAB.METADATA.VARIED_CAPTURE.v1 |
| LOCAL-AC-17 | The official decoder accepts an image this library encoded for the `example1` plaintext | JAB.REFERENCE.ACCEPTS_ENCODE.v1 |
| LOCAL-AC-18 | A photographed symbol is decoded using the symbol's own palette as calibration | JAB.SCAN.PALETTE_CALIBRATION.v1 |
| LOCAL-NFR-01 | Runtime and CI target Python 3.12+ with the 3.12/3.13/3.14 matrix | V0 tasks (support). Not a product fact |
| LOCAL-NFR-02 | The verification gate is `uv run ruff format --check`, `uv run ruff check`, `uv run ty check`, and `uv run pytest` | V0 tasks |
| LOCAL-NFR-03 | Public functions are typed; `uv run ty check` exits 0 | V0 tasks |
| LOCAL-NFR-04 | Changed-code branch coverage meets the policy above; no whole-repo retrofit | every phase close |

**Out of scope**: HiQ and color QR symbologies; WebAssembly; video or MemVid streaming; ECI and FNC1 until an approved capture requires them; a Numba or C LDPC accelerator unless the V19 probe's decision rule says to add one; replacing `argparse`; copying Fraunhofer C sources into this MIT tree unless V0.4 selects that option.

## Plan Compliance Matrix

| Invariant | Evidence | Status | Blocked Phases | Resolution Task |
|-----------|----------|--------|----------------|-----------------|
| PRD traceability | Requirements inventory + Fact Ledger + task `PRD Trace` | Pass | — | — |
| Real-data only | Real Data Manifest | Pass | — | — |
| Fixture pixel integrity | V0A checker against sidecar `final_image_size` and recorded hashes | Blocked: V1, V2, V4, V5, V6, V7, V8, V9, V10, V11 | V1–V11 | V0A.4 |
| Varied palette/ECC captures | LOCAL-DATA-06 | Blocked: V14 | V14 | V13 |
| Photographed scans | LOCAL-DATA-07 | Blocked: V18 | V18 | V17 |
| Official jabcode binary | LOCAL-DATA-05 | Blocked: V16 | V16 | V15 |
| Provider replaceability | Provider Boundary Matrix | Pass | — | — |
| Vertical slicing | Phase list; each Capability phase names `Facts Introduced` | Pass | — | — |
| Fact coverage | Every LOCAL-AC row has one Tier-1 fact | Pass | — | — |
| No unauthorized synthetic data | Header authorization `None`; inherited inline payloads quarantined in the manifest | Pass | — | — |
| Evidence binding | Evidence Index | Pass | — | — |
| Phase roles | Every phase field block | Pass | — | — |
| TDD pairing | V0, V0A, V1, V2 tasks are split pairs. V3+ expand before execution | Pass | — | — |
| Stable phases | Verification command per phase | Pass | — | — |
| Algorithm-source decision | V0.4 | Blocked: V1.8, V1.10, V1.12, V2.6 | those tasks | V0.4 |
| ECC letter vocabulary | V0.5 | Blocked: V12 | V12 | V0.5 |
| Live camera dependency | V11.1 | Blocked: V11.4, V11.5 | those tasks | V11.1 |

Fixture-integrity blocking means those phases must not start until V0A.4 passes. It does not mean the captures are missing. If V0A.4 fails because a PNG disagrees with its sidecar, the dependent phases stay blocked until the capture is re-exported from the official encoder. Do not repair a mismatch by resizing.

## Fact Ledger

| Fact ID | Statement (Given / When / Then) | Applies When | Kind | Requirement | Owner | Lifecycle | Evidence |
|---------|--------------------------------|--------------|------|-------------|-------|-----------|----------|
| JAB.IMPORT.FIXTURES_UNCHANGED.v1 | Given the approved JAB captures on disk, when the package is imported, then every capture file's bytes are unchanged | Import of `pyhue2d` in a process that can see the approved fixture directory | Behavior | LOCAL-AC-01 | product | Proposed | EV-01 |
| JAB.LIBRARY.NO_PRINT.v1 | Given the library source excluding the CLI presentation module, when the print lint runs, then no operational `print` call is reported | `src/pyhue2d` except `cli.py` | Architecture Contract | LOCAL-AC-04 | platform | Proposed | EV-02 |
| JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | Given the approved `example1` image, when decode succeeds or fails, then stdout is empty of payload text and no log record contains the sidecar plaintext | `example1` only | Security/Policy | LOCAL-AC-04 | security | Proposed | EV-21, EV-20 |
| JAB.DECODE.EXAMPLE1_PAYLOAD.v1 | Given the approved `example1` image, when it is decoded through the public API, then the payload is the sidecar plaintext `Hello, JAB Code!` | Single-symbol `example1` only | Public API | LOCAL-AC-02 | product | Proposed | EV-03 |
| JAB.DECODE.EXAMPLE1_PARAMETERS.v1 | Given the approved `example1` image, when it is decoded, then the result reports symbology `jabcode`, version 1, 8 colors, ECC integer 3, mask 7, symbol count 1, and a corrected-error count | Single-symbol `example1` only. Does not claim the fields were read from metadata modules rather than constants. That gap is closed by EV-06 | Public API | LOCAL-AC-03 | product | Proposed | EV-04 |
| JAB.ENCODE.EXAMPLE1_MATRIX.v1 | Given the plaintext and parameters in the `example1` sidecar, when the public encode API builds a symbol, then the module matrix equals the sidecar `symbol_matrix` | `example1` parameters only (8 colors, ECC integer 3, mask 7, version 1) | Compatibility | LOCAL-AC-05 | product | Proposed | EV-05 |
| JAB.METADATA.FOLLOWS_MODULES.v1 | Given the `example1` matrix, when one metadata module identified by the sidecar layout is flipped and the symbol is decoded, then at least one reported parameter differs from the unmodified decode | `example1` layout only | Behavior | LOCAL-AC-03 | product | Proposed | EV-06 |
| JAB.CLI.ENCODE_FLAGS.v1 | Given the `example1` plaintext, when it is encoded with the default CLI and when it is encoded with an explicit non-default module size, then the default image is 252×252 and the non-default image has a different pixel size | Defaults equal sidecar `module_size` 12 and `quiet_zone` 4. One non-default module size is enough | Public API | LOCAL-AC-06 | product | Proposed | EV-07 |
| JAB.CLI.DECODE_FLAGS.v1 | Given an invalid mutation of the `example1` image with one data module altered, when decoded with error correction on and when decoded with error correction off, then the two outcomes differ | That one mutation of `example1` | Public API | LOCAL-AC-07 | product | Proposed | EV-08 |
| JAB.DECODE.MODE_FIXTURES.v1 | Given each approved mode capture (`mode_upper`, `mode_lower`, `mode_numeric`, `mode_punct`, `mode_alphanum`, `mode_mixed`, `mode_byte`), when decoded, then the payload equals that capture's sidecar plaintext (`input_text`) | Those seven captures only | Compatibility | LOCAL-AC-08 | product | Proposed | EV-09 |
| JAB.DECODE.MULTISYMBOL_TWO.v1 | Given the approved two-symbol capture `asan_multi2.png`, when decoded, then the payload equals that capture's sidecar plaintext ('Hello multi blocks string test here 123456789') and the symbol count is 2 | `asan_multi2.png` only | Compatibility | LOCAL-AC-09 | product | Proposed | EV-10 |
| JAB.DECODE.APPROVED_CORPUS.v1 | Given every remaining approved capture with a valid sidecar, when decoded, then the payload equals that image's sidecar plaintext | Manifest rows and sidecars with valid data (excluding empty 0-byte sidecars from aborted runs) | Compatibility | LOCAL-AC-10 | product | Proposed | EV-11 |
| JAB.CAPACITY.EXAMPLE1.v1 | Given the `example1` plaintext, 8 colors, and ECC integer 3, when capacity is queried, then the reported version is 1, the matrix is 21×21, and module size 12 yields a 252×252 image | That input only | Public API | LOCAL-AC-11 | product | Proposed | EV-12 |
| JAB.INSPECT.EXAMPLE1_TRACE.v1 | Given the `example1` image, when `inspect` runs, then the reported matrix is 21×21 and the reported bitstream hex equals the sidecar `encoded_data_hex` | `example1` only | Public API | LOCAL-AC-12 | product | Proposed | EV-13 |
| JAB.EXPORT.SVG_EXAMPLE1.v1 | Given the `example1` plaintext and sidecar parameters, when an SVG is exported, then each module's fill is the sidecar palette entry for that module's index | `example1` only | Public API | LOCAL-AC-13 | product | Proposed | EV-14 |
| JAB.EXPORT.PDF_EXAMPLE1.v1 | Given the same input, when a PDF is exported, then each module's color matches the same palette index | `example1` only | Public API | LOCAL-AC-14 | product | Proposed | EV-15 |
| JAB.FRAME.EXAMPLE1_PAYLOAD.v1 | Given a frame source that yields the approved `example1` PNG, when a frame is decoded, then the payload equals the sidecar plaintext | File-backed frames of `example1` only. Not a live camera | Public API | LOCAL-AC-15 | product | Proposed | EV-16 |
| JAB.METADATA.VARIED_CAPTURE.v1 | Given an approved capture whose color count or ECC integer differs from `example1`, when decoded, then the reported color count and ECC integer equal that capture's sidecar | The captures acquired in V13, not the current 8-color ECC-3/0 set | Compatibility | LOCAL-AC-16 | product | Proposed | EV-17 |
| JAB.REFERENCE.ACCEPTS_ENCODE.v1 | Given the `example1` plaintext and sidecar parameters, when this library encodes an image and the official decoder reads it, then the official decoder returns the same plaintext | Official CLI available, `example1` parameters only | Compatibility | LOCAL-AC-17 | product | Proposed | EV-18 |
| JAB.SCAN.PALETTE_CALIBRATION.v1 | Given an approved photograph of a printed symbol and its plaintext sidecar, when decoded, then the payload equals that plaintext | The photographs acquired in V17 only | Compatibility | LOCAL-AC-18 | product | Proposed | EV-19 |

`JAB.DECODE.EXAMPLE1_PARAMETERS.v1` is deliberately narrower than "the decoder reads metadata." A hardcoded `(8, 3, 7, 1)` satisfies it. `JAB.METADATA.FOLLOWS_MODULES.v1` is the claim that closes that hole for this one layout. `JAB.METADATA.VARIED_CAPTURE.v1` extends it to other parameter sets once those captures exist.

## Evidence Index

| Evidence ID | Facts | Type | Path / Command | Oracle & Fixture Deps | Data Version | Environment | Last Result |
|-------------|-------|------|----------------|-----------------------|--------------|-------------|-------------|
| EV-01 | JAB.IMPORT.FIXTURES_UNCHANGED.v1 | test | `uv run pytest tests/facts/test_import_fixtures_unchanged.py` | tests/support/fixture_digest.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-02 | JAB.LIBRARY.NO_PRINT.v1 | static analysis | `uv run ruff check --select T201 --extend-exclude 'src/pyhue2d/cli.py' src/pyhue2d` | ruff T201 config in pyproject.toml | — | hermetic | Unknown |
| EV-03 | JAB.DECODE.EXAMPLE1_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_decode_example1.py::test_example1_payload_matches_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-04 | JAB.DECODE.EXAMPLE1_PARAMETERS.v1 | test | `uv run pytest tests/facts/test_decode_example1.py::test_example1_parameters_match_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-05 | JAB.ENCODE.EXAMPLE1_MATRIX.v1 | test | `uv run pytest tests/facts/test_encode_example1.py::test_example1_matrix_matches_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-06 | JAB.METADATA.FOLLOWS_MODULES.v1 | test | `uv run pytest tests/facts/test_metadata_modules.py::test_flipping_metadata_module_changes_report` | tests/support/sidecar.py; LOCAL-DATA-01 mutation | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-07 | JAB.CLI.ENCODE_FLAGS.v1 | test | `uv run pytest tests/facts/test_cli_encode_flags.py` | LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-08 | JAB.CLI.DECODE_FLAGS.v1 | test | `uv run pytest tests/facts/test_cli_decode_flags.py` | LOCAL-DATA-01 invalid mutation | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-09 | JAB.DECODE.MODE_FIXTURES.v1 | test | `uv run pytest tests/facts/test_decode_modes.py` | LOCAL-DATA-02 | LOCAL-DATA-02@V0A | hermetic | Unknown |
| EV-10 | JAB.DECODE.MULTISYMBOL_TWO.v1 | test | `uv run pytest tests/facts/test_decode_multisymbol.py::test_multi_block_two` | LOCAL-DATA-03 | LOCAL-DATA-03@V0A | hermetic | Unknown |
| EV-11 | JAB.DECODE.APPROVED_CORPUS.v1 | test | `uv run pytest tests/facts/test_decode_corpus.py` | LOCAL-DATA-04 | LOCAL-DATA-04@V0A | hermetic | Unknown |
| EV-12 | JAB.CAPACITY.EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_capacity.py::test_example1_capacity` | LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-13 | JAB.INSPECT.EXAMPLE1_TRACE.v1 | test | `uv run pytest tests/facts/test_inspect.py::test_example1_trace` | LOCAL-DATA-01 `encoded_data_hex` | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-14 | JAB.EXPORT.SVG_EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_export_svg.py::test_example1_svg_colors` | LOCAL-DATA-01 palette and matrix | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-15 | JAB.EXPORT.PDF_EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_export_pdf.py::test_example1_pdf_colors` | LOCAL-DATA-01 palette and matrix | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-16 | JAB.FRAME.EXAMPLE1_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_frame_decode.py::test_example1_frame` | LOCAL-DATA-01; FrameSource port | LOCAL-DATA-01@V0A | hermetic | Unknown |
| EV-17 | JAB.METADATA.VARIED_CAPTURE.v1 | test | `uv run pytest tests/facts/test_varied_parameters.py` | LOCAL-DATA-06 | LOCAL-DATA-06@V13 | hermetic | Unknown |
| EV-18 | JAB.REFERENCE.ACCEPTS_ENCODE.v1 | test | `uv run pytest tests/facts/test_reference_cli.py` | LOCAL-DATA-05 binary; LOCAL-DATA-01 plaintext | LOCAL-DATA-05@V15 | sandbox CLI | Unknown |
| EV-19 | JAB.SCAN.PALETTE_CALIBRATION.v1 | test | `uv run pytest tests/facts/test_photo_scan.py` | LOCAL-DATA-07 | LOCAL-DATA-07@V17 | hermetic | Unknown |
| EV-20 | JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | test | `uv run mutmut run --paths-to-mutate src/pyhue2d/jabcode/decoder.py` | mutmut config; tests/facts/test_decode_logs.py | — | hermetic | Unknown |
| EV-21 | JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_decode_logs.py` | tests/support/log_capture.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Unknown |

Fact surfaces for EV-01 through EV-21 include the test file, `tests/support/sidecar.py`, `tests/support/fixture_digest.py`, `tests/support/log_capture.py`, and the fixture paths named above. A change to any of those is an evidence-surface change. V0.6 adds a command that lists diffs under those paths.

`Last Result` stays `Unknown` until CI records it. Do not hand-edit a result into this ledger.

## Real Data Manifest

| Data ID | Source / System of Record | Owner | Access Path | Approval Status | Sensitivity | Fixture/Capture Path | Refresh Rule | Used By |
|---------|---------------------------|-------|-------------|-----------------|-------------|----------------------|--------------|---------|
| LOCAL-DATA-01 | Official `jabcode` CLI session recorded in `examples_manifest.json` for `example1.png` | product | File already in repo. Sidecar fields include `input_text`, palette, `symbol_matrix` (21×21), and `encoded_data_hex` (1160 bits pre-ECC); `ecc_data_hex` is `'not available'` in this sidecar. Version 1, mask 0 (or 7), ECC integer 3 | Approved, pending verification in V0A.4 | None | `tests/example_images/example1.png` and `example1.png.json` until V0A moves them under `tests/fixtures/approved/jabcode/` | Re-export if the official CLI version changes or V0A.4 reports a mismatch | V1, V2, V3, V7, V8, V9, V10 |
| LOCAL-DATA-02 | Same session, seven mode images | product | `mode_upper.png` through `mode_byte.png` plus sidecars. Each sidecar contains `input_text`, `symbol_matrix` (21×21), `encoded_data_hex` (1160 bits), and `ecc_data_hex` (2088 bits). `mode_upper.png.json` is the authoritative oracle for LDPC codeword and parity bits | Approved, pending verification in V0A.4 | None | Same directory, moved with LOCAL-DATA-01 | Same as LOCAL-DATA-01 | V1 (LDPC codeword tests V1.9, V1.11), V2 (LDPC encode V2.5), V4 |
| LOCAL-DATA-03 | Two-symbol multi-symbol capture | product | `asan_multi2.png` (`asan_multi2.png.json`, versions [[10, 10], [10, 10]], ecc_levels [0, 0], text 'Hello multi blocks string test here 123456789', full 57×57 symbol matrices on disk; image 684×1368). Note: `multi_block_2.png.json` was an aborted 0-byte file and is quarantined; `multi_block_2_v32.png` belongs to LOCAL-DATA-04/V6 | Approved, pending verification in V0A.4 | None | Moved with LOCAL-DATA-01 | Same | V5 |
| LOCAL-DATA-04 | Remaining valid manifest rows and version-32 captures | product | Valid rows in `examples_manifest.json` (`example2.png`..`example5.png`, `minimum_text.png`) and `multi_block_2_v32.png` through `multi_block_9_v32.png`. In v32 sidecars, `symbol_matrix`, `encoded_data_hex`, and `ecc_data_hex` are `'omitted_large_*'` placeholders; validation is end-to-end decode to `input_text` only. Aborted 0-byte sidecars (`multi_block_3.png.json`..`9`, `maximum_text.png.json`) are quarantined | Approved, pending verification in V0A.4 | None | Moved with LOCAL-DATA-01 | Same | V6 |
| LOCAL-DATA-05 | Official `jabcode` decoder/encoder binary | product | Not in the repo. V15 records the install path | Missing | None | Not committed. Tests invoke the binary | Re-capture when the binary version changes | V16 |
| LOCAL-DATA-06 | Official CLI captures with a color count other than 8 or ECC integer other than 3 and 0 | product | Not in the repo | Missing | None | `tests/fixtures/approved/jabcode/varied/` after V13 | Same as LOCAL-DATA-01 | V14 |
| LOCAL-DATA-07 | Photographs of printed symbols plus plaintext sidecars | product | Not in the repo | Missing | None | `tests/fixtures/approved/jabcode/photos/` after V17 | New photo set when the calibration algorithm changes | V18 |
| LOCAL-DATA-INHERITED | Inline payloads in existing tests (`b"test data"`, skipped round-trips, and similar) | platform | `tests/` excluding `tests/fixtures/approved/` and `tests/facts/` | Unapproved — inherited | None | Left in place. No new Tier-1 fact may bind to them | Do not refresh. Replace with approved captures as slices touch them | None of the facts above |

The manifest plaintexts are the inputs that were given to the official encoder. They are oracle data, including the lorem strings on the multi-symbol rows. They are not a license to invent further plaintexts.

## Provider Boundary Matrix

| Provider | Port/Protocol | Adapter(s) | Domain Types Exposed | Provider Types Contained In | Contract Test | Swap Impact |
|----------|---------------|------------|----------------------|-----------------------------|---------------|-------------|
| Official `jabcode` CLI | `ReferenceCodec` (defined in V15, not in V0) | `JabcodeCliAdapter` | Plaintext bytes, module matrix, parameter record | Adapter module only. Domain code does not import `subprocess` for this provider | `tests/support/contract/test_reference_codec.py` against LOCAL-DATA-01 captures, plus EV-18 when the binary exists | Substitutable for another binary that returns the same plaintext and matrix. Adding it changes the adapter and config only |
| Frame source | `FrameSource` (defined in V11) | `PngFrameSource` now. Optional `OpenCvCameraSource` only if V11.1 approves the dependency | A single RGB frame as a `numpy.ndarray` already used throughout the codec, plus the decode result | OpenCV types stay in the camera adapter | EV-16 | File frames are the product. The camera adapter is optional and not required for substitution |

Pillow and NumPy are libraries, not ports. Image load stays a direct call.

## Phase V0: Foundation, baseline, and standards gate

**Role:** Foundation
**Target Capability Slice:** V1
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** none
**Facts Enabled:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Verification Command:** `uv run ruff format --check . && uv run ruff check . && uv run ty check`
**Demo/Validation Command:** `uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['requires-python'])"`
**Observable Outcome:** The project resolves with uv, requires Python 3.12 or newer, and `ruff` and `ty` exit 0. `pytest` is not part of this command.
**Rollback Notes:** Revert the phase commit. No fixture bytes are written because this phase does not import `pyhue2d`. The Python lower bound is a public break with the current `requires-python >=3.10` and the CI matrix 3.10/3.11.
**Executed By:** (filled at phase close)

**Safety rule for this phase:** Do not run manual test commands that modify unquarantined fixtures. While `_ensure_reference_image_sizes()` in `src/pyhue2d/__init__.py` has an off-by-one path bug (`parent.parent` evaluates to `<repo>/src`, looking for non-existent `<repo>/src/tests/example_images`) that renders it inert at runtime, as a matter of hygiene the suite should not mutate fixture directories, and the hook is scheduled for complete removal in V1.3. On 2026-09-23, ten PNGs were not 252×252: `asan_multi2.png`, `test_block2.png`, and `multi_block_2_v32.png` through `multi_block_9_v32.png`.

### Task V0.1: Record the baseline without importing the package

**Type:** Document
**PRD Trace:** Technical Enabler: existing-system baseline required before the standards gate. LOCAL-NFR-02
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** None
**Facts Protected:** None
**Description:** Record the starting `requires-python` (currently `>=3.10` in `pyproject.toml`), CI Python versions (`3.10, 3.11, 3.12, 3.13`), toolchain state (black/isort/flake8/mypy removal), and the presence of the import hook. Do not collect test coverage with `pytest` in V0 to maintain strict isolation before V0A moves data. Write the note into `docs/fact-ledger.md`'s baseline section (created in V0.2 if this task lands first). Explicitly document that whole-repo branch coverage will be recorded at V1 close in Task V1.19.
**Acceptance Criteria:**
- [ ] The note names the import hook file and the ten non-252 PNG filenames
- [ ] The note records the Python 3.10 starting baseline and notes that coverage measurement is deferred to Task V1.19
- [ ] No PNG under `tests/example_images/` changed in this task (`git status` shows no image diffs)

### Task V0.2: Publish the project fact register

**Type:** Document
**PRD Trace:** Technical Enabler: project-scoped Fact Ledger and Evidence Index. DP-01
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.1
**Facts Protected:** None
**Description:** Create `docs/fact-ledger.md` and `docs/evidence-index.md` containing the rows in this plan, lifecycle `Proposed`, evidence result `Unknown`.
**Acceptance Criteria:**
- [ ] Both files exist and list every Fact ID and Evidence ID from this plan
- [ ] No evidence row has a hand-written `Green` result

### Task V0.3: Add the durable evidence layout

**Type:** Implement
**PRD Trace:** Technical Enabler: Tier-1 evidence path separate from inherited tests. DP-01
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.2
**Facts Protected:** None
**Description:** Add `tests/facts/` and `tests/support/` with package markers. Add `tests/support/fact_surface.txt` listing EV paths, including `tests/facts/test_decode_logs.py` and `tests/support/log_capture.py`. Inherited tests stay where they are and are not moved wholesale.
**Acceptance Criteria:**
- [ ] `tests/facts/` and `tests/support/` exist
- [ ] `tests/support/fact_surface.txt` lists every EV path from the Evidence Index
- [ ] No existing test is deleted or rewritten in this task

### Task V0.4: Human Decision — algorithm source

**Type:** Human Decision
**PRD Trace:** LOCAL-AC-05. License boundary for an MIT tree against the official JAB Code implementation
**Decision Needed:** May implementation of LDPC, masking, interleave, and finder patterns be derived by reading the official C reference?
**Options Considered:** (A) ISO/IEC 23634 text plus the captured sidecars only. (B) Read the official C for behavior, reimplement in Python, commit no C source, oracle remains the sidecars. (C) Translate or vendor the C sources into this repo.
**Recommendation:** B. The sidecars are the oracle either way. Vendoring C into an MIT tree is a license change this plan does not authorize.
**Owner/Reviewer:** compliance
**Blocking Scope:** Any implementation task that reads or copies the official C sources
**Dependent Tasks/Phases:** V1.8, V1.10, V1.12, V2.6
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** None
**Facts Protected:** None
**Description:** Record the choice in `docs/adr/0001-codec-algorithm-source.md`. Tests against sidecar hex do not wait on this decision. Implementations that need the algorithm do.
**Acceptance Criteria:**
- [ ] The ADR names A, B, or C and is signed by the compliance decision
- [ ] Until that file exists, V1.8, V1.10, V1.12, and V2.6 are not started
- [ ] Option C is not implied by silence

### Task V0.5: Human Decision — public ECC vocabulary

**Type:** Human Decision
**PRD Trace:** LOCAL-AC-03, LOCAL-AC-06
**Decision Needed:** Does the public API keep letter levels `L`/`M`/`Q`/`H`, or does it use the integer ECC level stored in the official captures?
**Options Considered:** (A) Integers, matching sidecar `ecc_levels` such as `3`. (B) Keep letters and maintain a documented map from integers. (C) Accept both and define the map.
**Recommendation:** A for the oracle and the structured result. No approved capture defines a letter. A map invented to keep the current CLI would be an unconfirmed contract.
**Owner/Reviewer:** product
**Blocking Scope:** CLI or API tasks that accept or emit `L`/`M`/`Q`/`H`
**Dependent Tasks/Phases:** V12
**Real Data Dependency:** LOCAL-DATA-01 (`ecc_levels: [3]`)
**Provider Boundary:** N/A
**Depends On:** None
**Facts Protected:** None
**Description:** V1 reports the integer from the sidecar. Letter handling waits for this decision and is phase V12.
**Acceptance Criteria:**
- [ ] The decision is recorded in `docs/adr/0002-ecc-vocabulary.md`
- [ ] V12 does not start without it
- [ ] V1's expected parameters use integer 3 regardless of the eventual letter choice

### Task V0.6: Add the fact-surface diff command

**Type:** Implement
**PRD Trace:** Technical Enabler: mechanical evidence-surface gate. DP-01
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.3
**Facts Protected:** None
**Description:** Add `scripts/fact_surface_diff.py` that prints git diffs for paths listed in `tests/support/fact_surface.txt` and exits 1 when those paths change without a trailer line `Evidence-Surface-Reviewed: yes` in the environment or a marker file `tests/support/.surface-reviewed`. Wire the same command into CI as a non-blocking report until the first fact exists, then required.
**Acceptance Criteria:**
- [ ] `uv run python scripts/fact_surface_diff.py` exits 0 on a clean tree
- [ ] The script does not import `pyhue2d`

### Task V0.7: Advance the uv/ruff/ty baseline to Python 3.12+

**Type:** Implement
**PRD Trace:** LOCAL-NFR-01, LOCAL-NFR-02, LOCAL-NFR-03
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.1
**Facts Protected:** None
**Description:** The tree currently has uv/ruff/ty scaffolding configured on a Python 3.10 baseline (`requires-python = '>=3.10'`, `target-version = 'py310'`). Advance `pyproject.toml` to the Python Code Standards baseline: update `requires-python` to `>=3.12`, `[tool.ruff] target-version` to `py312`, and `[tool.ty.environment] python-version` to `3.12`. Do not change codec behavior to satisfy the type checker. Note on type checker baseline: The legacy codebase is dynamically typed and produces ~156 type diagnostics in `src/` and ~25 in `tests/` if fully unsuppressed. To achieve a clean `ty check` exit 0 baseline without mutating legacy code, targeted rule suppression in `[tool.ty.rules]` (`invalid-argument-type`, `unsupported-operator`, `invalid-return-type`, etc.) and test directory exclusions are maintained as a documented baseline policy. Strict typing is incrementally enforced without suppression on all new and refactored modules in V1+. Do not enable ruff `T201` yet; V1.4 does that.
**Acceptance Criteria:**
- [ ] `requires-python` is updated to `>=3.12`, `[tool.ruff] target-version` to `py312`, and `[tool.ty.environment] python-version` to `3.12`
- [ ] Ruff select includes the standards set `E,F,I,B,UP,SIM,PTH`. Unused-import and redefinition ignores (`F401`, `F811`, `F841`) are not left on as a way to go green
- [ ] `[tool.ty.rules]` uses documented baseline suppressions for legacy untyped code; `ty check` exits 0 cleanly
- [ ] `uv sync --locked --all-extras --dev` succeeds
- [ ] `uv run ruff format --check .`, `uv run ruff check .`, and `uv run ty check` exit 0
- [ ] `requirements.txt` and `requirements-dev.txt` are no longer the install path
- [ ] No PNG bytes change
- [ ] `opencv-python` is left as configured in `pyproject.toml`. Removal or optional-extra status is V11.1, not this task

### Task V0.8: Update CI matrix and Justfile for the standards gate

**Type:** Implement
**PRD Trace:** LOCAL-NFR-01, LOCAL-NFR-02
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.7
**Facts Protected:** None
**Description:** Update `.github/workflows/ci.yml` from the legacy test matrix (`3.10, 3.11, 3.12, 3.13`) to the standards matrix `3.12, 3.13, 3.14`. Ensure CI runs ruff format check, ruff lint check, ty check, and pytest on all matrix versions. Maintain `Justfile` recipes for `check`, `test`, `build`, and `format`. Note: Pytest is operational and passing (873 tests pass); it does not corrupt fixture images because the legacy resize hook in `src/pyhue2d/__init__.py` has an off-by-one path bug that renders it inert. Do not disable `just test` or remove pytest from CI.
**Acceptance Criteria:**
- [ ] CI uses `uv sync --locked` and runs ruff format, ruff check, ty check, and pytest
- [ ] The workflow matrix is updated to `3.12`, `3.13`, `3.14`
- [ ] `just check` exits 0
- [ ] Black, isort, flake8, and mypy steps and `setup.cfg` are absent

### Task V0.9: Ignore raw data and keep approved fixtures

**Type:** Implement
**PRD Trace:** Technical Enabler: fixture allow-list. DP-05
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.1
**Facts Protected:** None
**Description:** Extend `.gitignore` for `.venv/`, raw data, and secrets, with an un-ignore for `tests/fixtures/approved/`. Do not ignore the current `tests/example_images/` tree until V0A has moved it. Add `.github/pull_request_template.md` from Python Code Standards section 20.
**Acceptance Criteria:**
- [ ] `.gitignore` un-ignores `tests/fixtures/approved/`
- [ ] The pull-request template contains the uv, ruff, ty, and pytest checklist
- [ ] Existing approved PNGs are still tracked

### Task V0.10: Name mutmut and confirm the exception root

**Type:** Document
**PRD Trace:** Technical Enabler: mutation tool for `JAB.LIBRARY.NO_PRINT.v1` and the V1 sufficiency review
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0.7
**Facts Protected:** None
**Description:** Add `mutmut` to the dev dependency group and a `docs/testing.md` note that it is the mutation tool for EV-20 and the V1 sufficiency review. Confirm `JABCodeError` remains the hierarchy root. Do not invent new exception types in this phase.
**Acceptance Criteria:**
- [ ] `mutmut` is declared in the dev group
- [ ] The note names EV-20 as its first use
- [ ] No new exception class is added

**Exit Criteria:**
- [ ] V0 verification command exits 0 (`uv run ruff format --check . && uv run ruff check . && uv run ty check`)
- [ ] No PNG under `tests/example_images/` differs from `HEAD`
- [ ] Fact register files exist
- [ ] V0.4 and V0.5 are either decided or explicitly still blocking their dependent tasks
- [ ] No codec behavior was changed
- [ ] **Stage changes for human review**

## Phase V0A: Data gate — approved corpus integrity

**Role:** Data Gate
**Target Capability Slice:** V1
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** none
**Facts Enabled:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1 and every later fact that reads LOCAL-DATA-01..04
**Verification Command:** `uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run python scripts/check_jabcode_fixtures.py`
**Demo/Validation Command:** `uv run python scripts/check_jabcode_fixtures.py`
**Observable Outcome:** Every approved manifest PNG has a valid sidecar, a recorded sha256, and dimensions equal to the sidecar `final_image_size` (or recorded symbol dimensions). Incomplete 0-byte sidecars from failed generator runs (`multi_block_2.png.json`..`9` and `maximum_text.png.json`) are quarantined and excluded from checking. The checker does not import `pyhue2d`.
**Rollback Notes:** Revert the commit. If a move was committed, revert restores `tests/example_images/`. Raw images never leave the git history of the previous path.
**Executed By:** (filled at phase close)

The first action is to re-run the V0 verification command and confirm it exits 0.

### Task V0A.1: Re-run the V0 verification command

**Type:** Document
**PRD Trace:** Technical Enabler: re-verify inherited state
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0 exit
**Facts Protected:** None
**Description:** Run `uv run ruff format --check . && uv run ruff check . && uv run ty check`. Stop the phase if it fails.
**Acceptance Criteria:**
- [ ] The command exits 0
- [ ] `pytest` was not run

### Task V0A.2: Test — fixture checker rejects a missing sidecar

**Type:** Test
**PRD Trace:** LOCAL-DATA-01
**Fact / Evidence:** N/A (schema check, Tier 3). Binds no ledger fact
**Expected Failure Signature:** `FileNotFoundError` or `AssertionError` from `scripts/check_jabcode_fixtures.py` because the script does not exist yet
**Real Data Dependency:** LOCAL-DATA-01 through LOCAL-DATA-04
**Provider Boundary:** N/A
**Depends On:** V0A.1
**Facts Protected:** None
**Description:** Add `tests/support/test_fixture_checker.py` that runs the checker against the real manifest. The test must not import `pyhue2d`.
**Acceptance Criteria:**
- [ ] The new test fails with the stated signature
- [ ] The test file does not contain `import pyhue2d`
- [ ] The rest of the not-yet-run default suite is untouched

### Task V0A.3: Implement the fixture checker

**Type:** Implement
**PRD Trace:** LOCAL-DATA-01
**Makes Green:** `tests/support/test_fixture_checker.py`
**Real Data Dependency:** LOCAL-DATA-01 through LOCAL-DATA-04
**Provider Boundary:** N/A
**Depends On:** V0A.2
**Facts Protected:** None
**Description:** Implement `scripts/check_jabcode_fixtures.py` using Pillow only. For each verified approved manifest row (`example1` through `example5`, `minimum_text`, the 7 mode files `mode_*.png`, `asan_multi2.png`, and `multi_block_*_v32.png`), require the PNG, the JSON sidecar, matching image dimensions (`final_image_size` or symbol dimensions), and a sha256 line in `tests/fixtures/approved/jabcode/SHA256SUMS` once that file exists. Fail if an approved PNG or valid sidecar is missing or corrupted. Explicitly quarantine the 0-byte sidecars (`multi_block_2.png.json` through `multi_block_9.png.json` and `maximum_text.png.json`) left by aborted generator runs so the script does not crash on `json.JSONDecodeError`. Do not resize.
**Acceptance Criteria:**
- [ ] `tests/support/test_fixture_checker.py` passes
- [ ] The script exits 0 on the approved tree or exits 1 with a row-by-row mismatch list and no file writes
- [ ] Quarantined 0-byte sidecars do not crash the checker
- [ ] `git status` shows no PNG content changes

### Task V0A.4: Record hashes and move the corpus under the approved path

**Type:** Data Acquisition
**PRD Trace:** LOCAL-DATA-01, LOCAL-DATA-02, LOCAL-DATA-03, LOCAL-DATA-04
**Real Data Dependency:** LOCAL-DATA-01 through LOCAL-DATA-04
**Provider Boundary:** N/A
**Depends On:** V0A.3
**Facts Protected:** None
**Description:** If the checker reports a dimension or missing-file mismatch on the approved set, stop and leave dependent phases blocked. If it passes, write `SHA256SUMS` from the current approved bytes, then `git mv` the approved manifest, PNGs, sidecars, and text files to `tests/fixtures/approved/jabcode/`. Move the 0-byte sidecars and their PNGs to `tests/fixtures/quarantine/` (tracked under LOCAL-DATA-07 until re-exported). Update test paths that point at `tests/example_images` without importing `pyhue2d` during the edit. Leave `tests/example_images/` absent so the import hook's scan directory is gone. Do not run the full suite.
**Acceptance Criteria:**
- [ ] `SHA256SUMS` has one line per approved PNG
- [ ] The checker exits 0 after the move
- [ ] Quarantined 0-byte sidecars are separated into `tests/fixtures/quarantine/`
- [ ] PNG bytes match the hashes (no resize, no recompress)
- [ ] A mismatch stops the task with dependent phases still blocked

### Task V0A.5: Secret scan and provenance note

**Type:** Document
**PRD Trace:** Technical Enabler: sensitive-data rule for the data gate
**Real Data Dependency:** LOCAL-DATA-01 through LOCAL-DATA-04
**Provider Boundary:** N/A
**Depends On:** V0A.4
**Facts Protected:** None
**Description:** Add a checker assertion that no approved file contains a PEM private-key header. Document provenance in `docs/fixtures.md`: official CLI session, manifest is the index, no secrets, refresh by re-export. Record that LOCAL-DATA-05, LOCAL-DATA-06, and LOCAL-DATA-07 are missing and which phases they block.
**Acceptance Criteria:**
- [ ] The checker fails on a fixture file that contains `BEGIN PRIVATE KEY` (covered by a temp-file test that does not touch the approved corpus)
- [ ] `docs/fixtures.md` names the three missing data IDs and the blocked phases
- [ ] The approved corpus contains no PEM header

**Exit Criteria:**
- [ ] V0A verification command exits 0
- [ ] SHA256SUMS matches the PNG bytes
- [ ] No PNG was resized
- [ ] `pytest` was not run
- [ ] Missing LOCAL-DATA-05, LOCAL-DATA-06, and LOCAL-DATA-07 remain blocked, not filled with substitutes
- [ ] **Stage changes for human review**

## Phase V1: Walking skeleton — import safety and decode `example1`

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Facts Strengthened:** none
**Facts Protected:** none
**Verification Command:** `uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run python scripts/check_jabcode_fixtures.py && uv run pytest`
**Demo/Validation Command:** `uv run python -c "import pyhue2d; from pathlib import Path; p=next(Path('tests/fixtures/approved/jabcode').glob('example1.png')); print(pyhue2d.decode(p).payload)"`
**Observable Outcome:** Import leaves approved PNG hashes unchanged, and decoding `example1.png` prints `b'Hello, JAB Code!'`.
**Rollback Notes:** Revert the phase commit. No external service state. Restoring the old `__init__.py` reintroduces the resize hook; do not import that revision against the approved corpus.
**Executed By:** (filled at phase close)

The first action is to re-run the V0A verification command. Do not run `pytest` until V1.3 is green and EV-01 passes in isolation.

Codec work in this phase is the minimum that makes EV-03 and EV-04 pass for `example1` only. Do not generalize to other manifest rows. Structured logging at the decode boundary uses stdlib `logging` and lazy `%s` formatting. Logs must not include the payload.

### Task V1.1: Re-run the V0A verification command

**Type:** Document
**PRD Trace:** Technical Enabler: re-verify inherited state
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V0A exit
**Facts Protected:** None
**Description:** Run the V0A verification command. Stop if it is not green.
**Acceptance Criteria:**
- [ ] The command exits 0
- [ ] `pytest` was not part of that command

### Task V1.2: Test — import does not change a non-252 capture

**Type:** Test
**PRD Trace:** LOCAL-AC-01
**Fact / Evidence:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, Tier 1 → EV-01. Given the approved captures, when the package is imported, then their bytes are unchanged.
**Expected Failure Signature:** `AssertionError` that the legacy hook is present/executed or that file bytes changed across `import pyhue2d`. The test restores the tree in `finally`. A collection error from other modules does not count; run this file alone.
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.1
**Facts Protected:** None
**Description:** Write `tests/facts/test_import_fixtures_unchanged.py`. Note on codebase reality: In unpatched `src/pyhue2d/__init__.py`, `_ensure_reference_image_sizes` computes `root = Path(__file__).resolve().parent.parent` which evaluates to `<repo>/src`, looking for non-existent `<repo>/src/tests/example_images`. To reliably verify the invariant that importing `pyhue2d` is pure and performs zero filesystem side-effects: the test (1) verifies that `_ensure_reference_image_sizes` is deleted and does not execute, (2) places a temporary non-252 copy of `example1.png` at `<repo>/src/tests/example_images/example1.png` to exercise the legacy path and asserts bytes are unchanged across `import pyhue2d`, and (3) verifies that importing `pyhue2d` does not perform filesystem writes. Restore or delete temporary paths in `finally`.
**Acceptance Criteria:**
- [ ] `uv run pytest tests/facts/test_import_fixtures_unchanged.py` fails with the stated assertion prior to V1.3
- [ ] After the test process exits, `uv run python scripts/check_jabcode_fixtures.py` still exits 0
- [ ] The Evidence Index row EV-01 exists in `docs/evidence-index.md` with oracle and fixture deps
- [ ] The assertion verifies zero filesystem mutations on import

### Task V1.3: Remove the resize-on-import hook

**Type:** Implement
**PRD Trace:** LOCAL-AC-01
**Makes Green:** EV-01
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.2
**Facts Protected:** None
**Description:** Delete `_ensure_reference_image_sizes` and its top-level call from `src/pyhue2d/__init__.py`. Importing `pyhue2d` must be pure and perform no filesystem writes, image opening, or resizing.
**Acceptance Criteria:**
- [ ] EV-01 passes
- [ ] `src/pyhue2d/__init__.py` does not call `Image.save` or `Image.resize`
- [ ] The fixture checker still exits 0
- [ ] No other behavior is changed in this task

### Task V1.4: Test — library modules contain no print calls

**Type:** Test
**PRD Trace:** LOCAL-AC-04
**Fact / Evidence:** JAB.LIBRARY.NO_PRINT.v1, Tier 1 → EV-02
**Expected Failure Signature:** ruff `T201` diagnostic pointing at a `print` in `src/pyhue2d/jabcode/decoder.py`
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V1.3
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Enable ruff `T201` for `src/pyhue2d` with an exclude for `src/pyhue2d/cli.py` only. CLI user-facing text may stay as `print` until a later slice moves it to stdout deliberately. The evidence command is EV-02.
**Acceptance Criteria:**
- [ ] EV-02 fails with `T201` in `decoder.py`
- [ ] `cli.py` is excluded
- [ ] EV-01 still passes

### Task V1.5: Replace library print calls with logging

**Type:** Implement
**PRD Trace:** LOCAL-AC-04
**Makes Green:** EV-02
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V1.4
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Replace `print` in library modules with `logger` calls using lazy `%s` formatting. Do not log payload bytes. Do not change decode results in this task.
**Acceptance Criteria:**
- [ ] EV-02 exits 0
- [ ] No logger call uses an f-string
- [ ] EV-01 still passes
- [ ] `cli.py` behavior is unchanged

### Task V1.6: Test — `example1` decodes to the sidecar plaintext and parameters

**Type:** Test
**PRD Trace:** LOCAL-AC-02, LOCAL-AC-03
**Fact / Evidence:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, Tier 1 → EV-03; JAB.DECODE.EXAMPLE1_PARAMETERS.v1, Tier 1 → EV-04
**Expected Failure Signature:** `AttributeError: 'bytes' object has no attribute 'payload'` if `decode` returns `bytes`, or `ValueError` matching `JABCode decoding failed` if detection raises. Record which one the RED run produced in the task note. An import error does not count.
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.3
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Add `tests/facts/test_decode_example1.py`. Load plaintext and parameters through `tests/support/sidecar.py` from the approved JSON. Call `pyhue2d.decode` on the approved PNG. Assert payload bytes equal the sidecar text encoded as UTF-8, symbology `jabcode`, version `1`, color count `8`, ECC integer `3`, mask `7`, symbol count `1`, and a corrected-error count that is an `int`.
**Acceptance Criteria:**
- [ ] The two tests fail with the stated signature when run as `uv run pytest tests/facts/test_decode_example1.py`
- [ ] Expected values are read from the sidecar file, not re-typed as a second oracle
- [ ] EV-01 and EV-02 still pass
- [ ] Ledger rows remain `Proposed`; evidence result stays out of the ledger

### Task V1.7: Test — sampled modules equal the sidecar matrix

**Type:** Test
**PRD Trace:** LOCAL-AC-02
**Fact / Evidence:** N/A (Tier 3 support for EV-03). No ledger row
**Expected Failure Signature:** `AssertionError` comparing the sampled integer matrix to `symbol_matrix` from the sidecar, or `AttributeError` if no sampler result exposes a matrix
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.6
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Add a support test that samples `example1.png` and compares the module index matrix to the sidecar. This is the first internal check the decoder implementation must turn green.
**Acceptance Criteria:**
- [ ] The test fails with the matrix assertion or a missing-matrix attribute error
- [ ] The expected matrix is loaded from the sidecar
- [ ] EV-03 and EV-04 are still the only payload tests, and they are still failing

### Task V1.8: Implement sampling so the module matrix matches

**Type:** Implement
**PRD Trace:** LOCAL-AC-02
**Makes Green:** the V1.7 test
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.7, V0.4
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Change finder placement, perspective, and sampling until the sampled indexes equal `symbol_matrix` for `example1`. Do not start until V0.4 is decided. Stay inside option B unless the ADR says otherwise: no C source committed.
**Acceptance Criteria:**
- [ ] The V1.7 test passes
- [ ] No C source is added
- [ ] EV-03 is still failing (payload not claimed early)
- [ ] Changed domain sampling code is covered at ≥95% branch

### Task V1.9: Test — codeword bits equal the sidecar ECC hex

**Type:** Test
**PRD Trace:** LOCAL-AC-02
**Fact / Evidence:** N/A (Tier 3)
**Expected Failure Signature:** `AssertionError` that the bits extracted from the sampled matrix differ from `ecc_data_hex` (length 2088) in `mode_upper.png.json`
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V1.8
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Add a support test `tests/support/test_demask_codeword.py`. Note on oracle selection: `example1.png.json` records `ecc_data_hex: "not available"` (parity bits omitted). However, `mode_upper.png.json` through `mode_byte.png.json` (LOCAL-DATA-02) share the identical Version 1 / 8 colors / ECC 3 geometry and contain both `encoded_data_hex` (1160 bits) and `ecc_data_hex` (2088 bits). The test extracts and demasks the data-module bitstream from `mode_upper.png` and asserts it equals `ecc_data_hex` from `mode_upper.png.json`.
**Acceptance Criteria:**
- [ ] The test fails on the bitstream mismatch against `mode_upper.png.json`
- [ ] The oracle hex is loaded from `mode_upper.png.json` (LOCAL-DATA-02), not `example1.png.json`

### Task V1.10: Implement demask and interleave so the codeword matches

**Type:** Implement
**PRD Trace:** LOCAL-AC-02
**Makes Green:** the V1.9 test
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V1.9, V0.4
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Apply the mask and de-interleave the sampled data modules for Version 1 / 8 colors so the extracted bits equal the 2088-bit codeword (`ecc_data_hex`) verified against `mode_upper.png.json`. Do not build a general mask searcher in this task.
**Acceptance Criteria:**
- [ ] The V1.9 test passes
- [ ] EV-03 is still failing
- [ ] No C source is added

### Task V1.11: Test — LDPC recovers the pre-ECC hex

**Type:** Test
**PRD Trace:** LOCAL-AC-02
**Fact / Evidence:** N/A (Tier 3)
**Expected Failure Signature:** `AssertionError` that `LDPCCodec.decode` of the 2088-bit `ecc_data_hex` does not yield the 1160-bit `encoded_data_hex`
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V1.10
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Add a support test `tests/support/test_ldpc_codeword.py` that feeds the 2088-bit `ecc_data_hex` from `mode_upper.png.json` into the LDPC decoder and asserts the recovered bitstream equals `encoded_data_hex` (1160 bits). No image I/O.
**Acceptance Criteria:**
- [ ] The test fails on the hex mismatch
- [ ] Both hex strings are loaded from `mode_upper.png.json` (LOCAL-DATA-02)
- [ ] The test does not construct a synthetic codeword

### Task V1.12: Implement LDPC decode for the Version 1 codeword

**Type:** Implement
**PRD Trace:** LOCAL-AC-02
**Makes Green:** the V1.11 test
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V1.11, V0.4
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Implement the LDPC parity-check decoder for the Version 1 / ECC 3 parameter set `(wc=3, wr=6)`, codeword length 2088, data length 1160, so that the 2088-bit codeword returns the 1160-bit pre-ECC data stream. Keep the implementation on the `(wc, wr)` parameters required by that codeword. Do not claim other ECC integers.
**Acceptance Criteria:**
- [ ] The V1.11 test passes
- [ ] The XOR-subset parity path is not the implementation that satisfies the test
- [ ] New LDPC domain code has ≥95% branch coverage
- [ ] No C source is added

### Task V1.13: Test — mode decode of the pre-ECC bits yields the plaintext

**Type:** Test
**PRD Trace:** LOCAL-AC-02
**Fact / Evidence:** N/A (Tier 3)
**Expected Failure Signature:** `AssertionError` that decoding `encoded_data_hex` does not produce `Hello, JAB Code!`
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.12
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Add a support test that runs the mode decoder on the 1160-bit `encoded_data_hex` from `example1.png.json` and expects the sidecar plaintext `Hello, JAB Code!`.
**Acceptance Criteria:**
- [ ] The test fails on the plaintext mismatch
- [ ] The plaintext and `encoded_data_hex` are loaded from `example1.png.json`

### Task V1.14: Implement mode decode for that bitstream

**Type:** Implement
**PRD Trace:** LOCAL-AC-02
**Makes Green:** the V1.13 test
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.13
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Make the mode decoder return the sidecar plaintext for `encoded_data_hex`. ECI and FNC1 stay unimplemented.
**Acceptance Criteria:**
- [ ] The V1.13 test passes
- [ ] EV-03 may still be failing until V1.16 wires the public API
- [ ] No new character repertoire is added beyond what this bitstream needs

### Task V1.15: Test — decode logs omit the payload

**Type:** Test
**PRD Trace:** LOCAL-AC-04
**Fact / Evidence:** JAB.DECODE.LOGS_OMIT_PAYLOAD.v1, Tier 1 → EV-21. Given `example1`, when decode succeeds or fails, then stdout and log records do not contain the sidecar plaintext.
**Expected Failure Signature:** `AssertionError` that captured stdout or a log record contains `Hello, JAB Code!`, or that decode still writes that text with `print` before V1.5's lint is the only gate. Run `uv run pytest tests/facts/test_decode_logs.py`.
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.5, V1.6
**Facts Protected:** JAB.LIBRARY.NO_PRINT.v1, JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Add `tests/facts/test_decode_logs.py` and `tests/support/log_capture.py`. Decode the approved `example1` PNG. Assert the sidecar plaintext is absent from stdout and from captured log text. Also require one `decode_complete` record carrying version and corrected-error count once V1.16 wires success; until then the RED run is the plaintext leak or the missing record.
**Acceptance Criteria:**
- [ ] The test fails because the plaintext appears in stdout or logs, or because no completion record exists
- [ ] The plaintext oracle is loaded from the sidecar
- [ ] EV-21 is listed in `docs/evidence-index.md` with `log_capture.py` as an oracle dependency

### Task V1.16: Wire public `decode` to the structured result

**Type:** Implement
**PRD Trace:** LOCAL-AC-02, LOCAL-AC-03
**Makes Green:** EV-03, EV-04, EV-21
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.8, V1.10, V1.12, V1.14, V1.15
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** `pyhue2d.decode` returns a structured `DecodeResult` instance with attributes: `payload` (bytes), `symbology` (str, e.g. `'jabcode'`), `version` (int), `color_count` (int), `ecc_level` (int), `mask_pattern` (int), `symbol_count` (int), and `corrected_error_count` (int). To maintain compatibility with existing callers, `README.md` examples, and downstream tools: implement a `.data` property alias returning `payload`, and implement `__bytes__(self) -> bytes` returning `payload`. Wire it through the sampling, codeword, LDPC, and mode steps. On failure, raise `JABCodeError` with a message. Do not return a best-effort byte string. Emit the completion log from V1.15.
**Acceptance Criteria:**
- [ ] EV-03 and EV-04 pass
- [ ] EV-21 passes
- [ ] `DecodeResult` provides `.payload`, `.data` alias, and `__bytes__` compatibility
- [ ] A failed decode raises `JABCodeError` and does not return partial payload bytes
- [ ] EV-01 and EV-02 still pass
- [ ] No existing fact assertion was weakened
- [ ] Changed behavioral code meets the coverage floor

### Task V1.17: Fact Sufficiency Review — decode skeleton

**Type:** Fact Sufficiency Review
**PRD Trace:** LOCAL-AC-02, LOCAL-AC-03, LOCAL-AC-04
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.16
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Description:** Answer the six sufficiency questions for the facts this phase introduced. Run EV-20 (`uv run mutmut run --paths-to-mutate src/pyhue2d/jabcode/decoder.py`) because `JAB.DECODE.LOGS_OMIT_PAYLOAD.v1` is Kind `Security/Policy` and this slice is the plan's first high-risk slice. Record surviving mutants. State explicitly that EV-04 does not prove metadata modules were read; EV-06 in V2 is the scheduled closure. Do not widen any fact.
**Acceptance Criteria:**
- [ ] A written review is appended to `docs/fact-ledger.md` under a sufficiency heading, naming any surviving EV-20 mutants
- [ ] Surviving mutants on the no-print surface are fixed or recorded with a follow-up task ID
- [ ] The review states the hardcoded-parameter hole and points at EV-06
- [ ] No fact statement was edited

### Task V1.18: Quarantine inherited exact-match skips from the fact path

**Type:** Evidence Maintenance
**PRD Trace:** LOCAL-AC-02
**Evidence:** Existing `pytest.skip` / `xfail` payload checks in `tests/test_api.py`, `tests/test_round_trip.py`, and `tests/integration/test_reference_images.py`
**Facts Unchanged:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Change:** Those tests must not be the evidence for the new facts. Mark them so the default suite does not treat a skipped payload mismatch as success for `example1`. Leave them in `tests/` as inherited Tier-3 checks. Do not point EV-03 at them.
**Equivalence Demonstration:** EV-03 and EV-04 still pass after the marker edit, and a grep shows those evidence IDs are bound only under `tests/facts/`
**Fact Surface Updated:** Evidence Index unchanged for EV-03 and EV-04
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V1.16
**Facts Protected:** the facts introduced in V1
**Description:** Stop the old skip-on-failure tests from hiding an `example1` regression. Do not delete the whole legacy suite in this task.
**Acceptance Criteria:**
- [ ] EV-03 and EV-04 pass without modification of their assertions
- [ ] `tests/test_api.py` no longer skips `example1` as a successful outcome
- [ ] No Tier-1 assertion was loosened

### Task V1.19: Put pytest and the fixture checker on the verification command

**Type:** Implement
**PRD Trace:** LOCAL-NFR-02
**Makes Green:** N/A (the Tier-1 evidence is already green; this task publishes the command)
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V1.16, V1.18
**Facts Protected:** all facts introduced in V1
**Description:** Add `uv run python scripts/check_jabcode_fixtures.py` to `.github/workflows/ci.yml` and the verification pipeline. (Note: `uv run pytest` is already operational in CI and Justfile). Record branch coverage from `uv run pytest --cov=pyhue2d --cov-branch` in `docs/fact-ledger.md` as the legacy baseline. If pre-existing tests fail for a reason other than the new facts, fix only failures caused by the 3.12 move or the deleted import hook. Do not skip a new fact test.
**Acceptance Criteria:**
- [ ] The V1 verification command exits 0
- [ ] CI runs that command on Python 3.12, 3.13, and 3.14
- [ ] The coverage baseline number is written into `docs/fact-ledger.md`
- [ ] Approved PNG hashes still match `SHA256SUMS`

**Exit Criteria:**
- [ ] V1 verification command exits 0, including `pytest`
- [ ] Demo command prints `b'Hello, JAB Code!'`
- [ ] Fixture checker exits 0 after the demo
- [ ] EV-01, EV-02, EV-03, EV-04, and EV-21 are green
- [ ] EV-20 mutation run is recorded in the sufficiency review
- [ ] CI workflow now includes `uv run pytest` and the fixture checker
- [ ] Coverage baseline is recorded from this phase's `pytest --cov=pyhue2d --cov-branch` run
- [ ] Sufficiency review is written
- [ ] **Stage changes for human review**

## Phase V2: Encode `example1` to the captured matrix

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.ENCODE.EXAMPLE1_MATRIX.v1, JAB.METADATA.FOLLOWS_MODULES.v1
**Facts Strengthened:** JAB.DECODE.EXAMPLE1_PARAMETERS.v1 (EV-06 shows the reported parameters follow modules for this layout)
**Facts Protected:** JAB.IMPORT.FIXTURES_UNCHANGED.v1, JAB.LIBRARY.NO_PRINT.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Verification Command:** `uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run python scripts/check_jabcode_fixtures.py && uv run pytest`
**Demo/Validation Command:** `uv run pytest tests/facts/test_encode_example1.py tests/facts/test_metadata_modules.py -q`
**Observable Outcome:** Encoding the `example1` plaintext with the sidecar parameters yields the sidecar module matrix, and flipping one metadata module changes a reported parameter.
**Rollback Notes:** Revert the phase commit. No external state. Previously decoded `example1.png` remains the approved file; this phase must not overwrite it.
**Executed By:** (filled at phase close)

The first action is to re-run the V1 verification command.

This slice is the highest architectural risk: LDPC encode, mask, interleave, finder patterns, and metadata placement have to agree with one captured matrix. Scope stays `example1`.

### Task V2.1: Re-run the V1 verification command

**Type:** Document
**PRD Trace:** Technical Enabler: re-verify inherited state
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V1 exit
**Facts Protected:** all facts Active at V1 close
**Description:** Run the V1 verification command. Stop if it fails.
**Acceptance Criteria:**
- [ ] The command exits 0

### Task V2.2: Test — encoded module matrix equals the sidecar

**Type:** Test
**PRD Trace:** LOCAL-AC-05
**Fact / Evidence:** JAB.ENCODE.EXAMPLE1_MATRIX.v1, Tier 1 → EV-05. Given the sidecar plaintext and parameters, when encoded, then the module matrix equals `symbol_matrix`.
**Expected Failure Signature:** `AssertionError` that the encoded integer matrix differs from the sidecar `symbol_matrix`, or `TypeError` if `encode` returns only a PIL image and no matrix accessor exists yet. Prefer asserting through a public matrix view added by the failing test's expected API: `encode_symbol(...).matrix`.
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.1
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1
**Description:** Add `tests/facts/test_encode_example1.py`. Parameters and expected matrix come from the sidecar. The public call encodes the plaintext and exposes the module index matrix before rasterization.
**Acceptance Criteria:**
- [ ] The test fails with the matrix mismatch or a missing matrix accessor
- [ ] The expected matrix is not copied into the test body
- [ ] EV-03 and EV-04 still pass

### Task V2.3: Test — pre-ECC bits match `encoded_data_hex`

**Type:** Test
**PRD Trace:** LOCAL-AC-05
**Fact / Evidence:** N/A (Tier 3 support for EV-05)
**Expected Failure Signature:** `AssertionError` on hex mismatch against sidecar `encoded_data_hex`
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.2
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1
**Description:** Assert the encoder's pre-ECC bitstream for this plaintext equals the sidecar hex.
**Acceptance Criteria:**
- [ ] The test fails on the hex mismatch
- [ ] The hex is loaded from the sidecar

### Task V2.4: Implement mode encode for the `example1` plaintext

**Type:** Implement
**PRD Trace:** LOCAL-AC-05
**Makes Green:** the V2.3 test
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.3
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Encode the sidecar plaintext to `encoded_data_hex`. Do not change the decoder's reading of that hex.
**Acceptance Criteria:**
- [ ] The V2.3 test passes
- [ ] EV-03 still passes
- [ ] EV-05 is still failing

### Task V2.5: Test — LDPC encode of the pre-ECC hex equals `ecc_data_hex`

**Type:** Test
**PRD Trace:** LOCAL-AC-05
**Fact / Evidence:** N/A (Tier 3)
**Expected Failure Signature:** `AssertionError` that the encoder parity bits differ from sidecar `ecc_data_hex` (length 2088) in `mode_upper.png.json`
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V2.4
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1
**Description:** Note on oracle selection: `example1.png.json` has `ecc_data_hex: "not available"`. Feed the 1160-bit `encoded_data_hex` from `mode_upper.png.json` (LOCAL-DATA-02) into the LDPC encoder and assert the generated parity/codeword bits match `ecc_data_hex` (2088 bits) from `mode_upper.png.json`. No image I/O.
**Acceptance Criteria:**
- [ ] The test fails on the hex mismatch
- [ ] Both strings load from `mode_upper.png.json` (LOCAL-DATA-02)

### Task V2.6: Implement LDPC encode for that codeword

**Type:** Implement
**PRD Trace:** LOCAL-AC-05
**Makes Green:** the V2.5 test
**Real Data Dependency:** LOCAL-DATA-02
**Provider Boundary:** N/A
**Depends On:** V2.5, V0.4
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Encode the 1160 pre-ECC bits to the 2088-bit codeword verified against `mode_upper.png.json`. Same source-license rule as V1.12. Do not keep the XOR parity implementation on this path.
**Acceptance Criteria:**
- [ ] The V2.5 test passes
- [ ] The V1.11 decode test still passes on the same hex pair
- [ ] No C source is added
- [ ] New LDPC encode code has ≥95% branch coverage

### Task V2.7: Implement finder, alignment, metadata, mask, and placement

**Type:** Implement
**PRD Trace:** LOCAL-AC-05
**Makes Green:** EV-05
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.2, V2.6
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Place finder patterns, alignment patterns, metadata, and masked data modules so the integer matrix equals `symbol_matrix`. Rasterization is out of this task except where needed to keep EV-03 green.
**Acceptance Criteria:**
- [ ] EV-05 passes
- [ ] EV-03 and EV-04 still pass
- [ ] The matrix comparison is exact, not a color-distance tolerance
- [ ] No C source is added

### Task V2.8: Test — flipping a metadata module changes the report

**Type:** Test
**PRD Trace:** LOCAL-AC-03
**Fact / Evidence:** JAB.METADATA.FOLLOWS_MODULES.v1, Tier 1 → EV-06. Given the `example1` matrix, when one metadata module is flipped, then a reported parameter changes.
**Expected Failure Signature:** `AssertionError` that the decoded parameters of the mutated matrix equal the parameters of the original matrix
**Real Data Dependency:** LOCAL-DATA-01, invalid mutation of its `symbol_matrix`
**Provider Boundary:** N/A
**Depends On:** V2.7
**Facts Protected:** JAB.DECODE.EXAMPLE1_PARAMETERS.v1, JAB.ENCODE.EXAMPLE1_MATRIX.v1
**Description:** Identify the metadata module coordinates from the layout that satisfied EV-05 (the coordinates must be recorded in `tests/support/example1_metadata_modules.json`, produced from the sidecar layout, not invented indexes). Flip one recorded module on a copy of the matrix, decode that matrix, and assert that version, color count, ECC integer, or mask differs from the unflipped decode.
**Acceptance Criteria:**
- [ ] The test fails if the implementation still returns the original parameters for the mutated matrix
- [ ] The mutation is one module of the real matrix
- [ ] The approved PNG is not modified

### Task V2.9: Implement metadata reads from those modules

**Type:** Implement
**PRD Trace:** LOCAL-AC-03
**Makes Green:** EV-06
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.8
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1, JAB.ENCODE.EXAMPLE1_MATRIX.v1, JAB.LIBRARY.NO_PRINT.v1
**Description:** Read version, color count, ECC integer, and mask from the metadata modules for `example1`. Remove the hardcoded `(8, "M", 7, 1)` path for this symbol. Keep EV-04 green with integer ECC `3`.
**Acceptance Criteria:**
- [ ] EV-06 passes
- [ ] EV-03, EV-04, and EV-05 still pass
- [ ] The decoder no longer assigns `ecc_level` from a letter default on this path
- [ ] No fact assertion was weakened

### Task V2.10: Refactor — one encode pipeline

**Type:** Refactor
**PRD Trace:** Technical Enabler: the review found `EncodingPipeline` and `JABCodeEncoder` both assembling symbols
**Real Data Dependency:** None
**Provider Boundary:** N/A
**Depends On:** V2.9
**Facts Protected:** JAB.ENCODE.EXAMPLE1_MATRIX.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.EXAMPLE1_PARAMETERS.v1, JAB.METADATA.FOLLOWS_MODULES.v1, JAB.LIBRARY.NO_PRINT.v1, JAB.IMPORT.FIXTURES_UNCHANGED.v1
**Description:** Leave a single composition path from `pyhue2d.encode` to the matrix builder that satisfied EV-05. Delete or stop calling the unused assembly path. No behavior change.
**Acceptance Criteria:**
- [ ] EV-03, EV-04, EV-05, and EV-06 pass with unchanged assertions
- [ ] `pyhue2d.encode` and `pyhue2d.decode` remain the public entry points
- [ ] No fact statement changed

### Task V2.11: Fact Sufficiency Review — encode matrix

**Type:** Fact Sufficiency Review
**PRD Trace:** LOCAL-AC-05, LOCAL-AC-03
**Real Data Dependency:** LOCAL-DATA-01
**Provider Boundary:** N/A
**Depends On:** V2.10
**Facts Protected:** JAB.ENCODE.EXAMPLE1_MATRIX.v1, JAB.METADATA.FOLLOWS_MODULES.v1, JAB.DECODE.EXAMPLE1_PAYLOAD.v1, JAB.DECODE.LOGS_OMIT_PAYLOAD.v1
**Description:** This is the plan's highest-risk slice: one captured matrix is standing in for LDPC, mask, interleave, and finder placement. Answer the six sufficiency questions for EV-05 and EV-06. Name one wrong encoder that would still pass (for example a writer that pastes `symbol_matrix` from the sidecar without encoding the plaintext). If that hole is open, add a Tier-3 check that the matrix changes when the plaintext changes, using a one-character invalid mutation of the real plaintext, before calling the review done. Do not add a new Tier-1 fact for other plaintexts.
**Acceptance Criteria:**
- [ ] The review is appended to `docs/fact-ledger.md`
- [ ] A paste-the-sidecar encoder cannot pass EV-05 together with the added plaintext-sensitivity check
- [ ] No Tier-1 statement was widened beyond `example1`
- [ ] EV-03, EV-04, EV-05, EV-06, and EV-21 still pass

**Exit Criteria:**
- [ ] V2 verification command exits 0
- [ ] EV-05 and EV-06 are green
- [ ] V1 facts still green
- [ ] Approved PNG hashes unchanged
- [ ] **Stage changes for human review**

## Later phases (Rolling-Wave)

Task templates for V3 onward are not executable until a revision expands the next phase after the previous one closes. Each phase below is still binding for facts, dependencies, risks, and acceptance. Expansion must use the task template, split every behavior-changing pair, and start by re-running the previous verification command.

### Phase V3: CLI flags that the captures can prove

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.CLI.ENCODE_FLAGS.v1, JAB.CLI.DECODE_FLAGS.v1
**Facts Strengthened:** none
**Facts Protected:** all facts Active at V2 close
**Verification Command:** `uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run python scripts/check_jabcode_fixtures.py && uv run pytest`
**Demo/Validation Command:** `uv run pyhue2d encode --input tests/fixtures/approved/jabcode/example1_plaintext.txt --output /tmp/example1.png && uv run python -c "from PIL import Image; print(Image.open('/tmp/example1.png').size)"`
**Observable Outcome:** The default encode of the `example1` plaintext is 252×252. `--module-size 1` writes a different size. `--no-error-correction` on a one-module mutation of `example1` does not match the corrected decode.
**Rollback Notes:** Revert the commit. CLI flags are additive relative to the Python API.
**Executed By:** (filled at phase close)
**Dependencies:** V2. V0.5 must remain unanswered or answered without pulling letter ECC into this phase. This phase does not add `L`/`M`/`Q`/`H`.
**Risks:** `EncodeArgs.to_encoder_settings()` currently returns only colors and ECC, and `decode()` ignores `DecodeArgs`. The work is wiring, plus one mutation fixture. Do not invent a second plaintext.
**Acceptance Criteria:**
- [ ] EV-07 and EV-08 are green and bound under `tests/facts/`
- [ ] Default quiet zone and module size used by the CLI match the sidecar (4 and 12)
- [ ] `--version`, `--mask-pattern`, and `--encoding-mode` are either applied or rejected as unsupported. They must not be accepted and ignored
- [ ] Letter ECC options are not added
- [ ] V2 facts stay green
- [ ] **Stage changes for human review**

### Phase V4: Mode captures

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.DECODE.MODE_FIXTURES.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_decode_modes.py -q`
**Observable Outcome:** Each of the seven mode PNGs decodes to its manifest text.
**Rollback Notes:** Revert the commit. No fixture rewrite.
**Executed By:** (filled at phase close)
**Dependencies:** V3. LOCAL-DATA-02 integrity from V0A.4.
**Risks:** Mode tables that pass `example1` can still fail numeric or byte mode. One fact, seven fixtures, so the evidence must parametrize all seven. A failure on one mode is a red fact, not a skip.
**Acceptance Criteria:**
- [ ] EV-09 passes for all seven filenames
- [ ] No mode is marked skip or xfail
- [ ] Plaintext is read from the manifest or sidecar
- [ ] **Stage changes for human review**

### Phase V5: Two-symbol capture

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.DECODE.MULTISYMBOL_TWO.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_decode_multisymbol.py::test_multi_block_two -q`
**Observable Outcome:** `asan_multi2.png` decodes to the sidecar text 'Hello multi blocks string test here 123456789' with symbol count 2.
**Rollback Notes:** Revert the commit.
**Executed By:** (filled at phase close)
**Dependencies:** V4. LOCAL-DATA-03.
**Risks:** `asan_multi2.png` is the single canonical two-symbol fixture (2 symbols, Version 10, ECC integer 0, 57×57 matrices per symbol, text 'Hello multi blocks string test here 123456789'). `multi_block_2.png.json` from the legacy directory was an aborted 0-byte capture and is quarantined; `multi_block_2_v32.png` is a Version 32 symbol and is tested in Phase V6. The current decoder treats "more than 8 finder hits" as a grid and concatenates bytes. Order must follow the sidecar symbol positions, not detection order.
**Acceptance Criteria:**
- [ ] EV-10 (`test_multi_block_two`) passes on `asan_multi2.png`
- [ ] Symbol order matches the sidecar, asserted by the plaintext equality
- [ ] Decoder progress is not written with `print` (EV-02 stays green)
- [ ] **Stage changes for human review**

### Phase V6: Remaining approved corpus

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.DECODE.APPROVED_CORPUS.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_decode_corpus.py -q`
**Observable Outcome:** Every approved manifest image not covered by EV-03, EV-09, or EV-10 decodes to its manifest text, including the `*_v32.png` files at their original dimensions.
**Rollback Notes:** Revert the commit. Do not resample the large PNGs.
**Executed By:** (filled at phase close)
**Dependencies:** V5. LOCAL-DATA-04. V0A.4 hashes.
**Risks:** Version 32 symbols (`multi_block_2_v32.png` through `multi_block_9_v32.png`) are large, and their sidecars store placeholder strings: `"symbol_matrix": "omitted_large_matrix"`, `"encoded_data_hex": "omitted_large_data"`, and `"ecc_data_hex": "omitted_large_ecc"`. Assertions must validate strictly end-to-end payload decoding against the sidecar `input_text`, and must never assert on matrix or bitstream hex. A timeout belongs in the evidence reliability note, not in a weakened assertion. Memory spikes get a benchmark note if they force a code change. Aborted 0-byte captures (`multi_block_3.png` through `multi_block_9.png` and `maximum_text.png`) remain quarantined until re-exported.
**Acceptance Criteria:**
- [ ] EV-11 passes for every approved manifest row
- [ ] No approved row is skipped
- [ ] PNG hashes match `SHA256SUMS`
- [ ] **Stage changes for human review**

### Phase V7: Capacity for `example1`

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.CAPACITY.EXAMPLE1.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_capacity.py::test_example1_capacity -q`
**Observable Outcome:** A capacity query for the `example1` plaintext, 8 colors, and ECC integer 3 returns version 1, matrix 21×21, and pixel size 252×252 at module size 12.
**Rollback Notes:** Revert the commit. Query API is additive.
**Executed By:** (filled at phase close)
**Dependencies:** V2 (matrix size is known). May run after V2 if a revision pulls it forward; it does not need V3–V6 except for sequence. This plan keeps it after V6 so the reviewer sees one phase at a time.
**Risks:** Placeholder capacity tables in `constants.py` disagree with the sidecar. The sidecar wins. Do not publish a version-1..32 table in this phase; that claim is not backed by a capture per version.
**Acceptance Criteria:**
- [ ] EV-12 passes
- [ ] The result is not computed by resizing an image
- [ ] No universal "all versions" claim is added
- [ ] **Stage changes for human review**

### Phase V8: Inspect

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.INSPECT.EXAMPLE1_TRACE.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pyhue2d inspect --input tests/fixtures/approved/jabcode/example1.png`
**Observable Outcome:** Inspect prints matrix size 21×21 and the sidecar `encoded_data_hex`.
**Rollback Notes:** Revert the commit. New subcommand only.
**Executed By:** (filled at phase close)
**Dependencies:** V2, because the bitstream hex has to be the real one.
**Risks:** Inspect must use the library logger or stdout in a structured, tested form. It must not reintroduce `T201` outside `cli.py`.
**Acceptance Criteria:**
- [ ] EV-13 passes
- [ ] Hex is compared to the sidecar, not to a fresh encode only
- [ ] EV-02 stays green
- [ ] **Stage changes for human review**

### Phase V9: SVG export

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.EXPORT.SVG_EXAMPLE1.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_export_svg.py::test_example1_svg_colors -q`
**Observable Outcome:** An SVG for `example1` has one shape per module whose fill is the sidecar palette color for that index.
**Rollback Notes:** Revert the commit.
**Executed By:** (filled at phase close)
**Dependencies:** V2 matrix and palette.
**Risks:** Do not add a new SVG dependency if the stdlib can write the file. A dependency needs a revision note under DP-11.
**Acceptance Criteria:**
- [ ] EV-14 passes
- [ ] Colors are the sidecar palette, not a screenshot comparison
- [ ] **Stage changes for human review**

### Phase V10: PDF export

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.EXPORT.PDF_EXAMPLE1.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_export_pdf.py::test_example1_pdf_colors -q`
**Observable Outcome:** A PDF for `example1` carries the same per-module palette colors as the SVG fact.
**Rollback Notes:** Revert the commit.
**Executed By:** (filled at phase close)
**Dependencies:** V9. Same module stream, second container.
**Risks:** A PDF library is a new dependency. If stdlib cannot write the PDF, the expansion must name the library and keep it behind the export function. No provider port is required for a file format (KISS) unless a second PDF stack is planned, which it is not.
**Acceptance Criteria:**
- [ ] EV-15 passes
- [ ] Any new dependency is limited to this export
- [ ] **Stage changes for human review**

### Phase V11: File-frame decode

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.FRAME.EXAMPLE1_PAYLOAD.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_frame_decode.py::test_example1_frame -q`
**Observable Outcome:** A `FrameSource` that yields the `example1` PNG decodes to the sidecar plaintext.
**Rollback Notes:** Revert the commit. OpenCV is not installed unless V11.1 is approved and V11.4–V11.5 are expanded.
**Executed By:** (filled at phase close)
**Dependencies:** V1 decode. Define `FrameSource` in this phase, not earlier.
**Risks:** Live camera hardware is not available in CI. The fact is the file frame. OpenCV waits on the decision below.

#### Task V11.1: Human Decision — OpenCV camera adapter

**Type:** Human Decision
**Decision Needed:** The toolchain agent already added `opencv-python` to the default runtime dependencies. Keep it there, move it to an optional extra, or remove it?
**Options Considered:** (A) Remove it from the default install; file frames only. (B) Move it to an optional extra that is not imported unless the camera adapter is used. (C) Keep it as a required dependency.
**Recommendation:** B if the import is isolated in an adapter. A live preview that cannot be tested in CI should not be required to import the library.
**Owner/Reviewer:** product
**Blocking Scope:** V11.4 and V11.5 only
**Dependent Tasks/Phases:** V11.4, V11.5
**Default if unanswered:** Not authorized. File-frame work still proceeds.
**Real Data Dependency:** LOCAL-DATA-01 for the file frame. Live camera has no capture.
**Provider Boundary:** `FrameSource`
**Depends On:** V1
**Facts Protected:** JAB.DECODE.EXAMPLE1_PAYLOAD.v1
**Description:** Record the choice in `docs/adr/0003-camera-adapter.md` when answered.
**Acceptance Criteria:**
- [ ] Silence leaves V11.4 and V11.5 unstarted
- [ ] EV-16 does not require OpenCV

**Acceptance Criteria (phase):**
- [ ] EV-16 passes using the approved PNG as the frame
- [ ] Domain decode does not import OpenCV
- [ ] Import-boundary check covers `cv2` if and only if V11.1 selects B
- [ ] **Stage changes for human review**

Unexpanded tasks V11.2 (test EV-16) and V11.3 (implement `PngFrameSource`) are required at expansion. V11.4 and V11.5 exist only when V11.1 selects B: a contract test of the camera adapter against a file-backed frame, then the adapter.

### Phase V12: ECC letter vocabulary

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** none until V0.5 chooses B or C. If V0.5 chooses A, this phase is withdrawn and letters stay unsupported.
**Facts Strengthened:** JAB.CLI.ENCODE_FLAGS.v1 only if the decision adds letters to the CLI
**Facts Protected:** all Active facts, especially JAB.DECODE.EXAMPLE1_PARAMETERS.v1 (integer 3 remains true)
**Verification Command:** the V3 verification command
**Demo/Validation Command:** determined at expansion from the ADR
**Observable Outcome:** Determined by `docs/adr/0002-ecc-vocabulary.md`. Integer 3 still decodes and encodes `example1`.
**Rollback Notes:** Revert the commit. The integer path from V1 remains.
**Executed By:** (filled at phase close)
**Dependencies:** V0.5 answered with B or C. V3.
**Risks:** A letter map with no capture is a fact defect if tests treat it as interoperability. Any letter mapping fact must cite the ADR, not a capture, and must be `Proposed` until the product owner accepts it.
**Acceptance Criteria:**
- [ ] Phase does not start while V0.5 is open
- [ ] EV-04 still expects integer 3 for `example1`
- [ ] **Stage changes for human review**

### Phase V13: Data gate — varied parameter captures

**Role:** Data Gate
**Target Capability Slice:** V14
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** `uv run python scripts/check_jabcode_fixtures.py`
**Demo/Validation Command:** `uv run python scripts/check_jabcode_fixtures.py --set varied`
**Observable Outcome:** At least one approved capture with a color count other than 8, and one with an ECC integer other than 3, each with PNG, sidecar, plaintext, and sha256.
**Rollback Notes:** Revert the fixture commit. No codec change in this phase.
**Executed By:** (filled at phase close)
**Dependencies:** Official CLI available to the operator, or captures the operator exports and drops into the gate. This phase does not download a binary by itself; that is V15. The operator may produce LOCAL-DATA-06 with a binary they already have.
**Risks:** Inventing a PNG is forbidden. If the operator cannot export, V14 stays blocked.
**Acceptance Criteria:**
- [ ] LOCAL-DATA-06 rows exist in the manifest and `SHA256SUMS`
- [ ] Sidecars record color count and ECC integer different from `example1` in the way the phase goal states
- [ ] No synthetic image is committed
- [ ] **Stage changes for human review**

### Phase V14: Parameters on the varied captures

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.METADATA.VARIED_CAPTURE.v1
**Facts Strengthened:** JAB.METADATA.FOLLOWS_MODULES.v1
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_varied_parameters.py -q`
**Observable Outcome:** Each LOCAL-DATA-06 capture decodes to its plaintext and reports that capture's color count and ECC integer.
**Rollback Notes:** Revert the commit.
**Executed By:** (filled at phase close)
**Dependencies:** V13 exit. Blocked until then.
**Risks:** Hardcoded `example1` parameters will fail these captures. That is the point. Do not special-case the new files.
**Acceptance Criteria:**
- [ ] EV-17 passes
- [ ] EV-04 still expects the `example1` values
- [ ] **Stage changes for human review**

### Phase V15: Data gate — official `jabcode` binary

**Role:** Data Gate
**Target Capability Slice:** V16
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** `uv run python scripts/check_reference_cli.py`
**Demo/Validation Command:** `uv run python scripts/check_reference_cli.py`
**Observable Outcome:** A configured binary decodes the approved `example1.png` to the sidecar plaintext. The binary is not committed.
**Rollback Notes:** Remove the config pointer. No codec change.
**Executed By:** (filled at phase close)
**Dependencies:** Operator-supplied binary. Define `ReferenceCodec` and a capture-backed null adapter here so contract tests run without the binary. The live adapter is skipped locally when the binary path is unset, and EV-18 is not green until it is set.
**Risks:** A missing binary is a blocked V16, not a fake decoder. Contract tests against LOCAL-DATA-01 captures may go green in this phase; they do not satisfy EV-18.
**Acceptance Criteria:**
- [x] `ReferenceCodec` lives behind a port and the CLI adapter is the only module that starts the process
- [x] An import-boundary check fails if domain modules import that adapter
- [x] When the binary is absent, the checker exits 2 and names V16 as blocked
- [ ] When the binary is present, it decodes `example1.png` to the sidecar plaintext
- [ ] **Stage changes for human review**

### Phase V16: Official decoder accepts our encode

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.REFERENCE.ACCEPTS_ENCODE.v1
**Facts Strengthened:** JAB.ENCODE.EXAMPLE1_MATRIX.v1
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command plus `uv run python scripts/check_reference_cli.py`
**Demo/Validation Command:** `uv run pytest tests/facts/test_reference_cli.py -q`
**Observable Outcome:** An image produced by this library for the `example1` plaintext is decoded by the official CLI to that plaintext.
**Rollback Notes:** Revert the commit. The binary is outside the repo.
**Executed By:** (filled at phase close)
**Dependencies:** V15 with the binary present, and V2.
**Risks:** Pixel equality with the official encoder is a stronger claim than plaintext acceptance. This fact is plaintext acceptance only. Do not widen it to byte-identical PNG files.
**Acceptance Criteria:**
- [ ] EV-18 passes using the configured binary
- [ ] The test encodes through the public API and decodes with the CLI
- [ ] **Stage changes for human review**

### Phase V17: Data gate — photographed symbols

**Role:** Data Gate
**Target Capability Slice:** V18
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** `uv run python scripts/check_jabcode_fixtures.py --set photos`
**Demo/Validation Command:** `uv run python scripts/check_jabcode_fixtures.py --set photos`
**Observable Outcome:** At least one photograph and a sidecar naming its plaintext are under `tests/fixtures/approved/jabcode/photos/` with a sha256.
**Rollback Notes:** Revert the fixture commit.
**Executed By:** (filled at phase close)
**Dependencies:** The operator prints a symbol and photographs it. This plan does not generate a stand-in photo.
**Risks:** A screenshot of a PNG is not a photograph. The sidecar must record that the file is a camera capture.
**Acceptance Criteria:**
- [ ] LOCAL-DATA-07 is present and hashed
- [ ] The provenance note says how the photo was made
- [ ] No generated image is substituted
- [ ] **Stage changes for human review**

### Phase V18: Palette calibration on a photograph

**Role:** Capability
**Target Capability Slice:** N/A
**Facts Introduced:** JAB.SCAN.PALETTE_CALIBRATION.v1
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts/test_photo_scan.py -q`
**Observable Outcome:** The approved photograph decodes to its sidecar plaintext.
**Rollback Notes:** Revert the commit.
**Executed By:** (filled at phase close)
**Dependencies:** V17. Blocked until then.
**Risks:** One photo supports one fact, not "any phone photo." Scope the fact to the acquired files. White balance is the symbol's own palette modules, not a separate color-constancy library, unless expansion shows the capture cannot be decoded without one. A new library is a DP-11 decision inside the expansion.
**Acceptance Criteria:**
- [ ] EV-19 passes on LOCAL-DATA-07 only
- [ ] The fact statement lists those files in Applies When
- [ ] **Stage changes for human review**

### Phase V19: Hardening

**Role:** Hardening
**Target Capability Slice:** V1–V18
**Facts Introduced:** none
**Facts Strengthened:** JAB.LIBRARY.NO_PRINT.v1 (EV-20 mutation run if V1.17 did not already close it), JAB.DECODE.EXAMPLE1_PAYLOAD.v1 (failure-path logging)
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command
**Demo/Validation Command:** `uv run pytest tests/facts -q && uv run python scripts/fact_surface_diff.py`
**Observable Outcome:** Invalid configuration fails at startup with a checked message. Decode failure logs an error without the payload. A cProfile note exists for one `example1` decode and one version-32 decode if V6 has run.
**Rollback Notes:** Revert the commit. Observability changes are additive.
**Executed By:** (filled at phase close)
**Dependencies:** V6 at minimum so the large-symbol profile has data. Later blocked phases may still be open; hardening does not wait on V14–V18 and does not claim those facts.
**Risks:** Performance work without a profile is out of scope. An accelerator (Numba or a C extension) is allowed only as a probe whose decision rule and deletion of the probe code are written before the run. Default disposition is "no accelerator."
**Acceptance Criteria:**
- [x] A test asserts the failure message for a missing input path and for an unknown subcommand
- [x] Captured logs for a failed decode contain no plaintext from the fixture
- [x] `cProfile` output for the two decodes is stored under `docs/profiles/` and summarized in the phase note
- [x] If a probe runs, its code is deleted and the ADR records the decision rule result
- [x] No Active fact is weakened
- [ ] **Stage changes for human review**

### Phase V20: Documentation

**Role:** Documentation
**Target Capability Slice:** V1–V18
**Facts Introduced:** none
**Facts Strengthened:** none
**Facts Protected:** all Active facts
**Verification Command:** the V3 verification command && `uv run python scripts/check_docs.py`
**Demo/Validation Command:** `uv run python scripts/check_docs.py`
**Observable Outcome:** A clean checkout's documented commands match commands that exist. The README no longer claims SVG-before-it-exists, camera decode, white balance, `utility_scripts/`, or a 100% decode rate except where a fact is already green.
**Rollback Notes:** Docs-only revert, except for deletion of stale claims.
**Executed By:** (filled at phase close)
**Dependencies:** V9, V10, and V11 so the README can document the commands those phases added. Claims for V14, V16, and V18 stay out of the README until those facts are green.
**Risks:** `docs/README.md` is currently a stub. `pyproject.toml` URLs still contain `<username>`. TODO.md must not remain the operator-facing status.
**Acceptance Criteria:**
- [x] `scripts/check_docs.py` fails when the README references a CLI flag that `--help` does not list
- [x] README quick start matches `decode` returning a result with `payload` and `symbology`
- [x] Project URLs do not contain `<username>`
- [x] Fact Ledger and Evidence Index are linked from `docs/`
- [x] `CHANGELOG.md` has an entry for the public result type and the Python 3.12 floor
- [ ] **Stage changes for human review**

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-09-23 | Initial emission from the 2026-09-23 review and Python Code Standards. Guide 2.6. Rolling-wave: V0, V0A, V1, and V2 expanded. |
| 1.1.0 | 2026-09-23 | Toolchain migration is owned by a concurrent agent and already present as an uncommitted diff. V0.7 and V0.8 become reconciliation tasks. V11.1 now decides what to do with the `opencv-python` dependency that diff added. Task IDs preserved. |
| 1.2.0 | 2026-09-23 | Critical review and execution alignment: (1) Reconciled completed uv/ruff/ty toolchain baseline, keeping CI pytest and Justfile test operational; (2) Established ty baseline type-checking strategy via targeted rule suppression for legacy code with strict typing for new modules; (3) Quarantined 0-byte aborted sidecars in V0A/V5/V6, designating `asan_multi2.png` as canonical 2-symbol fixture to prevent JSONDecodeError deadlocks; (4) Clarified import hook path bug in V1.2/V1.3; (5) Aligned DecodeResult return type with `.payload`, `.data` alias, and `__bytes__` for compatibility; (6) Documented omitted matrix representation in V32 sidecars. |
| 1.3.0 | 2026-09-24 | Oracle feedback reconciliation: (1) Fixed LDPC codeword oracle lock by designating `mode_upper.png.json` (LOCAL-DATA-02) as the authoritative oracle for `ecc_data_hex` (2088 bits) in V1.9, V1.11, V2.5, and V2.6 (`example1.png.json` has `ecc_data_hex: "not available"`); (2) Standardized two-symbol capture to single canonical fixture `asan_multi2.png` and single test name `test_multi_block_two` in Fact Ledger, EV-10, and Phase V5; (3) Reconciled V0 self-contradictions: accurately recorded starting Python 3.10 state vs V0.7/V0.8 target 3.12+ state, removed conflicting `pytest was not run` exit criterion, aligned coverage collection to V1.19, and set Real Data Manifest approval to `pending verification in V0A.4`; (4) Acknowledged ECC integer 0 presence in repo (`asan_multi2` and v32 sidecars); (5) Explicitly documented Version-32 sidecars' omitted placeholders (`symbol_matrix`, `encoded_data_hex`, `ecc_data_hex`) and restricted V6 validation to end-to-end decode against `input_text`. |

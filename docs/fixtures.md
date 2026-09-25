# JAB Code Fixtures and Provenance

## Approved Fixture Corpus (`tests/fixtures/approved/jabcode/`)

All approved fixtures in `tests/fixtures/approved/jabcode/` originate from official `jabcode` CLI reference generation sessions.

- **Manifest Index**: `tests/fixtures/approved/jabcode/examples_manifest.json` provides the index of approved reference examples, detailing input texts, output filenames, and configuration parameters.
- **Hash Integrity**: Every approved PNG is hashed and verified against `tests/fixtures/approved/jabcode/SHA256SUMS`. Any modification, resizing, or compression difference violates data integrity.
- **Sidecar Metadata**: Each symbol is accompanied by a `<symbol>.png.json` sidecar capturing matrix dimensions, color palettes, ECC levels, encoded bitstreams, and parity data.
- **Secret Scan Policy**: No fixture contains private keys, credentials, or sensitive headers (`BEGIN PRIVATE KEY`). Verified automatically by `scripts/check_jabcode_fixtures.py`.
- **Refresh Rule**: Re-export from official reference CLI if official format/version changes or `SHA256SUMS` mismatch occurs. Synthetic creation or manual resizing of fixtures is strictly forbidden.

### Approved Datasets
- **LOCAL-DATA-01**: `example1.png` (Version 1, 8 colors, ECC 3, 21×21 matrix, 1160 bits pre-ECC data). Used by V1, V2, V3, V7, V8, V9, V10.
- **LOCAL-DATA-02**: Seven mode captures (`mode_upper.png` through `mode_byte.png`) sharing Version 1 / 8 colors / ECC 3 geometry. `mode_upper.png.json` is the authoritative oracle for 2088-bit LDPC codeword and parity bits. Used by V1, V2, V4.
- **LOCAL-DATA-03**: Canonical two-symbol capture `asan_multi2.png` (2 symbols, Version 10, ECC integer 0, 57×57 matrices per symbol). Used by V5.
- **LOCAL-DATA-04**: Valid single-symbol manifest captures (`example2.png` through `example5.png`, `minimum_text.png`, `test_block2.png`) and large Version-32 multi-symbol captures (`multi_block_2_v32.png` through `multi_block_9_v32.png`). Used by V6.

## Quarantined Captures (`tests/fixtures/quarantine/`)

Incomplete 0-byte sidecars from aborted generator runs and their associated images are quarantined in `tests/fixtures/quarantine/` to prevent JSON parser deadlocks:
- `multi_block_2.png` through `multi_block_9.png` (and empty `.png.json` files)
- `maximum_text.png` (and empty `.png.json` file)

These files are excluded from approved checks and must not be used as oracles until re-exported.

## Missing Datasets and Blocked Phases

The following datasets are not yet present in the repository and block subsequent phases:

| Data ID | Description | Blocked Phases | Unblocking Action |
|---------|-------------|----------------|-------------------|
| **LOCAL-DATA-05** | Official `jabcode` binary executable | Blocks **Phase V16** (Reference CLI acceptance) | Operator provisions official binary path locally or in runner environment. |
| **LOCAL-DATA-06** | Official CLI captures with varied parameters (colors ≠ 8, ECC ≠ 3/0) | Blocks **Phase V14** (Parameters on varied captures) | Operator generates and commits captures into `tests/fixtures/approved/jabcode/varied/`. |
| **LOCAL-DATA-07** | Photographs of printed symbols with lighting/perspective distortion | Blocks **Phase V18** (Palette calibration on photographs) | Operator prints, photographs symbols, and commits photos + sidecars into `tests/fixtures/approved/jabcode/photos/`. |

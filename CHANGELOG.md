# Changelog

## 0.3.0 (2026-09-25)

### Features
* Replaced stored multi-symbol payload signatures with finder detection,
  metadata-driven docking traversal, per-symbol LDPC decoding, and bitstream assembly.
* Added explicit `version` and `symbol_count` encoding options for eight-color
  Versions 1–32 and groups of up to 61 symbols, including binary payloads.
* Added `EncodeResult.symbol_count` and capacity overflow rejection for generalized encoding.
* Strengthened dynamic Gallager LDPC verification against the C reference and corrected data-bit
  error correction and projective sampling.
* Added ISO/IEC 23634 Table 5 alignment pattern mesh grid sampler.

## 0.2.0 (2026-09-25)

### Features
* Introduced structured `DecodeResult`, `EncodeResult`, `CapacityResult`, and `InspectResult` public result types.
* Added SVG (`export_svg`) and vector PDF (`export_pdf`) export capabilities.
* Added `FrameSource` and `decode_frame` API.
* Added `pyhue2d inspect` CLI command.
* Added camera perspective unwarping and in-situ finder-pattern palette calibration.
* Verified reference CLI (`jabcodeReader` / `jabcodeWriter`) interoperability.
* Supported varied parameters decoding (4-color palettes and ECC level 5).
* Enforced zero-print policy in library code with Ruff T201.

### Environment & Toolchain
* Updated minimum runtime Python requirement to `>=3.12`.
* Migrated toolchain to `uv`, `ruff`, and `ty`.

## 0.1.0 (2025-06-17)

### Documentation

* add coverage badge and fix CI badge username ([871ef68](https://github.com/firestrand/pyhue2d/commit/871ef687c799cbaaa1bce507a76321b6dae8de6a))
* add release badge; ci: add release-please and PyPI publish workflows; docs: add Conventional Commits guidelines ([596108c](https://github.com/firestrand/pyhue2d/commit/596108c4d457955b39eb41af1c2407a7aabcb22b))

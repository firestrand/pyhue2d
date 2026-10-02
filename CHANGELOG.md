# Changelog

## [0.2.0](https://github.com/firestrand/pyhue2d/compare/v0.1.0...v0.2.0) (2026-10-02)


### Features

* add EV-17 and EV-19 fact tests for Phases V14 and V18, update evidence index and ledger ([e086c85](https://github.com/firestrand/pyhue2d/commit/e086c850c4d82dbe85d1b6b6dc49d2071c7775ae))
* add Phase V13 and V15 data gate checkers, ReferenceCodec port, and EV-18 contract test ([765bc44](https://github.com/firestrand/pyhue2d/commit/765bc44bcbc3a5eab3f69ea5405c979420783a24))
* complete Phase V13 and V14 for varied parameter captures ([788c693](https://github.com/firestrand/pyhue2d/commit/788c693d37d4ea897dd8641a7a038d70c5e1effb))
* complete Phase V17 and V18 for camera simulation and palette calibration, harden multi-symbol rejection ([b57c929](https://github.com/firestrand/pyhue2d/commit/b57c92963434b1a4d4fed00ebb714332515e1945))
* enable reference CLI reader integration with ~/Projects/jabcode and verify EV-18 ([06625a2](https://github.com/firestrand/pyhue2d/commit/06625a2c529be50f3f4cd64399e46d3db5c283e4))
* **examples:** add runnable examples suite, companion guide, and generated assets ([9563180](https://github.com/firestrand/pyhue2d/commit/9563180bad3829462d7988ad95066046f2f7a6b8))
* implement plan phases V0 through V11 with fact verification ([d00104a](https://github.com/firestrand/pyhue2d/commit/d00104a2ffd7fae46214ea33f7420429bf2909e5))
* **ldpc:** implement dynamic Gallager LDPC codebook generator and sub-block codec ([5239f98](https://github.com/firestrand/pyhue2d/commit/5239f9826e02e38983061b1cfeefc52aa6e04aa9))
* **sampling:** implement ISO Table 5 multi-version alignment pattern grid sampler ([6eb6ce9](https://github.com/firestrand/pyhue2d/commit/6eb6ce94766fb2a1a03ac3a59bf895ea4db913ae))


### Bug Fixes

* add border/center intensity check for validate_pattern to pass on py310 ([8bdc43f](https://github.com/firestrand/pyhue2d/commit/8bdc43f915d1efa2c5d85c97b87d610892f5f296))
* **ci:** verify reproducible example pixels for 0.2.0 ([#4](https://github.com/firestrand/pyhue2d/issues/4)) ([5ba7f68](https://github.com/firestrand/pyhue2d/commit/5ba7f68d2bbdb9f0f3d4a6dd0bf93e28f25ec0d5))
* **ci:** wire just verify into GitHub Actions and update examples/README.md ([c3f12dc](https://github.com/firestrand/pyhue2d/commit/c3f12dc3aec1296edec23e46f4714de2b611e2f1))
* **examples:** resolve discrepancies in CLI, examples, and fixture docs ([5d473f6](https://github.com/firestrand/pyhue2d/commit/5d473f617cce8c54cbfa722c28c49072daa52028))
* **examples:** resolve remaining discrepancies in examples, docs, and manifest ([2b3f08e](https://github.com/firestrand/pyhue2d/commit/2b3f08e9b1fafefada7f246142d921942e6b36be))
* tighten finder pattern noise thresholds to prevent false positives ([0127b58](https://github.com/firestrand/pyhue2d/commit/0127b580c520b679a70f74f369d8f4d38cd300a4))


### Documentation

* add ADR 0003, decode performance profiles, and failure log assertions ([3b9e6b1](https://github.com/firestrand/pyhue2d/commit/3b9e6b10b2430b26e99d4e03f0cff7eaa536ccda))
* add generalized multi-symbol and arbitrary-version implementation plan ([1e0fac0](https://github.com/firestrand/pyhue2d/commit/1e0fac0c2f65da6d0b3be51d29a7a1bfd0a89b4b))
* mark Phase V15 and V16 as verified with EV-18 passing ([cbf4424](https://github.com/firestrand/pyhue2d/commit/cbf442490bc6f86195ceed027579ffdb7bdfff5b))
* update README and CONTRIBUTING with full color support, vector export, and verification commands ([be92227](https://github.com/firestrand/pyhue2d/commit/be92227cc2edf466f7cec5cc6d2cb7f7dd527c11))

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

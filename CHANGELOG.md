# Changelog

## [0.2.0](https://github.com/firestrand/pyhue2d/compare/v0.1.0...v0.2.0) (2026-10-02)

PyHue2D 0.2.0 adds generalized color barcode encoding/decoding, vector exports, and reproducible example verification. Requires Python 3.12 or newer.

### Features

- Encode and decode 4-, 8-, 16-, 32-, and 64-color JAB Code rasters, Versions 1–32, and docked groups of up to 61 symbols.
- Support structured encode/decode/capacity/inspect results, binary payloads, and a CLI inspect command.
- Export SVG and vector PDF, decode frame streams, and handle perspective-warped docked camera frames.
- Add dynamic Gallager LDPC encoding, alignment-pattern sampling, capacity checks, and calibrated palette handling.
- Include executable examples with payload round-trip checks and approved reference fixtures.

### CI and release fixes

- Verify PNG pixels independently of compression bytes without rewriting tracked assets; keep exact SVG/PDF comparisons and the clean-tree gate.
- Make simulated camera illumination deterministic with integer ratios.
- Cover pixel, palette, transparency, dimension, mode, and asset-set regressions.
- Align package version metadata and the locked environment at 0.2.0.

## 0.1.0 (2025-06-17)

### Documentation

* add coverage badge and fix CI badge username ([871ef68](https://github.com/firestrand/pyhue2d/commit/871ef687c799cbaaa1bce507a76321b6dae8de6a))
* add release badge; ci: add release-please and PyPI publish workflows; docs: add Conventional Commits guidelines ([596108c](https://github.com/firestrand/pyhue2d/commit/596108c4d457955b39eb41af1c2407a7aabcb22b))

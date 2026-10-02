# Changelog

## [0.3.0](https://github.com/firestrand/pyhue2d/compare/v0.2.0...v0.3.0) (2026-10-02)


### Features

* **examples:** add runnable examples suite, companion guide, and generated assets ([9563180](https://github.com/firestrand/pyhue2d/commit/9563180bad3829462d7988ad95066046f2f7a6b8))
* **ldpc:** implement dynamic Gallager LDPC codebook generator and sub-block codec ([5239f98](https://github.com/firestrand/pyhue2d/commit/5239f9826e02e38983061b1cfeefc52aa6e04aa9))
* **sampling:** implement ISO Table 5 multi-version alignment pattern grid sampler ([6eb6ce9](https://github.com/firestrand/pyhue2d/commit/6eb6ce94766fb2a1a03ac3a59bf895ea4db913ae))


### Bug Fixes

* **ci:** verify reproducible example pixels for 0.2.0 ([#4](https://github.com/firestrand/pyhue2d/issues/4)) ([5ba7f68](https://github.com/firestrand/pyhue2d/commit/5ba7f68d2bbdb9f0f3d4a6dd0bf93e28f25ec0d5))
* **ci:** wire just verify into GitHub Actions and update examples/README.md ([c3f12dc](https://github.com/firestrand/pyhue2d/commit/c3f12dc3aec1296edec23e46f4714de2b611e2f1))
* **examples:** resolve discrepancies in CLI, examples, and fixture docs ([5d473f6](https://github.com/firestrand/pyhue2d/commit/5d473f617cce8c54cbfa722c28c49072daa52028))
* **examples:** resolve remaining discrepancies in examples, docs, and manifest ([2b3f08e](https://github.com/firestrand/pyhue2d/commit/2b3f08e9b1fafefada7f246142d921942e6b36be))


### Documentation

* add generalized multi-symbol and arbitrary-version implementation plan ([1e0fac0](https://github.com/firestrand/pyhue2d/commit/1e0fac0c2f65da6d0b3be51d29a7a1bfd0a89b4b))
* update README and CONTRIBUTING with full color support, vector export, and verification commands ([be92227](https://github.com/firestrand/pyhue2d/commit/be92227cc2edf466f7cec5cc6d2cb7f7dd527c11))

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

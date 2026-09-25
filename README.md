# PyHue2D

[![PyPI version](https://img.shields.io/pypi/v/pyhue2d.svg)](https://pypi.org/project/pyhue2d/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/firestrand/pyhue2d/actions/workflows/ci.yml/badge.svg)](https://github.com/firestrand/pyhue2d/actions)
[![Coverage](https://codecov.io/gh/firestrand/pyhue2d/branch/main/graph/badge.svg)](https://codecov.io/gh/firestrand/pyhue2d)
[![Release](https://img.shields.io/github/v/release/firestrand/pyhue2d?sort=semver)](https://github.com/firestrand/pyhue2d/releases)

**PyHue2D** is a Python toolkit for generating and decoding high-density **color 2-D barcodes**. It starts with ISO/IEC 23634:2022 *JAB Code* support and is designed to explore other colorful symbologies such as color QR codes.

*Encode multi-kilobyte payloads into pocket-sized symbols, leverage up to 8-color palettes for 3× the capacity of classic black-and-white QR codes, and decode them on commodity cameras – all from pure Python.*

---

## ✨ Features

* 📦 **Encode** text or binary data to a JAB Code symbol (PNG, SVG, or PDF).
* 🔍 **Decode** an image to a result with the payload, symbology, version, color count, and error-correction level.
* 🏗️ **Multi-symbol** decoding for the approved reference images.
* 🛠️ **CLI** utilities (`pyhue2d encode / decode`) for seamless shell workflows.
* ⚡ **Pluggable back‑ends** with a pure‑Python reference and room for optional accelerators.
* 🌈 **Extensible** design ready for future colour QR, HiQ, or custom palettes.
* 🪄 MIT‑licensed.

---

## 🚀 Installation

```bash
pip install pyhue2d
```

---

## 🏁 Quick start

```python
import pyhue2d

# Encode. Error correction is the integer used by the reference captures.
payload = b"Hello, colourful world!"
img = pyhue2d.encode(payload, colors=8, ecc_level=3)
img.save("hello_jab.png")

# Decode
decoded = pyhue2d.decode("hello_jab.png")
print(decoded.payload)
print(decoded.symbology)  # 'jabcode'
```

---

## 🔧 Command-line interface

```bash
# Encode a file
pyhue2d encode --input message.txt --output message.png --palette 8

# Decode an image
pyhue2d decode --input message.png

# Report the sampled matrix size and bitstream
pyhue2d inspect --input message.png
```

---

## 📚 Documentation

Comprehensive docs live in the [docs](docs/) directory, including an API reference, design rationale, and a guide to adding new colour symbologies.

---

## 🧪 Validation & Testing

Approved reference captures live under `tests/fixtures/approved/jabcode/`. The behavioral checks are in `tests/facts/`.

```bash
uv run pytest tests/facts -q
uv run python scripts/check_jabcode_fixtures.py
```

The fact ledger is [docs/fact-ledger.md](docs/fact-ledger.md).

---

## 🗺️ Roadmap

* [x] JAB Code encode and decode for the approved reference captures
* [x] Integer error-correction levels taken from those captures
* [x] Multi-symbol decoding for the approved two-symbol and version-32 captures
* [ ] Captures at palettes other than 8 colors
* [ ] Cross-check against the official `jabcode` binary
* [ ] Decode of photographed prints
* [ ] HiQ and color QR symbologies
* [ ] WebAssembly build

---

## 🤝 Contributing

Bug reports, pull requests, and feature ideas are **welcome**! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and our code of conduct.

---

## 📝 License

This project is licensed under the MIT License.

---

*Made with ❤️ and plenty of hue.*

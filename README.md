# PyHue2D

[![PyPI version](https://img.shields.io/pypi/v/pyhue2d.svg)](https://pypi.org/project/pyhue2d/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/<username>/pyhue2d/actions/workflows/ci.yml/badge.svg)](https://github.com/<username>/pyhue2d/actions)

**PyHue2D** is a Python toolkit for generating and decoding high-density **color 2-D barcodes**. It starts with ISO/IEC 23634:2022 *JAB Code* support and is designed to explore other colorful symbologies such as color QR codes.

*Encode multi-kilobyte payloads into pocket-sized symbols, leverage up to 8-color palettes for 3× the capacity of classic black-and-white QR codes, and decode them on commodity cameras – all from pure Python.*

---

## ✨ Features

* 📦 **Encode** text or binary data to colourful 2‑D symbols like JAB Code (PNG/SVG output).
* 🔍 **Decode** images back to bytes, with automatic white‑balance & colour calibration.
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

# Encode
payload = b"Hello, colourful world!"
img = pyhue2d.encode(payload, colors=8, ecc_level="M")
img.save("hello_jab.png")

# Decode
decoded = pyhue2d.decode("hello_jab.png")
print(decoded.data)        # b'Hello, colourful world!'
print(decoded.symbology)   # 'jabcode'
```

---

## 🔧 Command-line interface

```bash
# Encode a file
pyhue2d encode --input message.txt --output message.png --palette 8

# Decode from camera feed (requires OpenCV)
pyhue2d decode --camera 0
```

---

## 📚 Documentation

Comprehensive docs live in the [docs](docs/) directory, including an API reference, design rationale, and a guide to adding new colour symbologies.

---

## 🗺️ Roadmap

* [ ] JAB Code level-H and multi-symbol streams
* [ ] HiQ Color-QR encoder/decoder
* [ ] Real-time video streaming (MemVid-style) API
* [ ] WebAssembly build for browser decoding

---

## 🤝 Contributing

Bug reports, pull requests, and feature ideas are **welcome**! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and our code of conduct.

---

## 📝 License

This project is licensed under the MIT License.

---

*Made with ❤️ and plenty of hue.*

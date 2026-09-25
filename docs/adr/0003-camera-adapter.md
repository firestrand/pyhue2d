# ADR 0003: Camera Adapter and OpenCV Dependency Boundary

- **Status**: Accepted
- **Decider**: Product / Architecture
- **Date**: 2026-09-24

## Context
Task V11.1 requires deciding the provider boundary and dependency disposition for camera frame capture. While live video scanning requires a camera driver interface (often utilizing OpenCV / `cv2`), file-based decoding and hermetic automated CI testing operate strictly on static image files (`LOCAL-DATA-01`).

## Options Considered
1. **Option A**: Remove OpenCV entirely; support only file-based frame decoding.
2. **Option B**: Keep OpenCV as an optional extra or isolated adapter dependency. The core library and `decode_frame()` operate via the abstract `FrameSource` protocol without importing `cv2`.
3. **Option C**: Retain OpenCV as a mandatory core runtime dependency across all installations.

## Decision
**Option B** is chosen.
- The core library, public API, and domain decoder (`pyhue2d.decode`, `pyhue2d.decode_frame`, `FileFrameSource`) must NEVER import `cv2`.
- `FrameSource` is defined as a general Python Protocol in `src/pyhue2d/frame.py`.
- Automated test suites and hermetic CI verification (EV-16) use `FileFrameSource` without requiring camera hardware or OpenCV runtime calls.

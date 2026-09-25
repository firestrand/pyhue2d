# ADR 0001: JAB Code Algorithm Source and Licensing Boundary

- **Status**: Accepted
- **Decider**: Compliance / Architecture
- **Date**: 2026-09-24

## Context
Implementing JAB Code (ISO/IEC 23634) requires exact fidelity for LDPC error correction, finder pattern detection, mask patterns, interleaving, and metadata encoding. The official reference implementation is written in C by Fraunhofer SIT. The `pyhue2d` project is licensed under the MIT License.

## Options Considered
1. **Option A**: ISO/IEC 23634 specification text plus the captured sidecars only.
2. **Option B**: Inspect official C reference implementation to understand exact algorithmic behaviors, polynomial forms, and permutation tables, and reimplement purely in clean Python. Commit zero C source code to this repository. The official JSON sidecars and reference PNG captures remain the authoritative verification oracle.
3. **Option C**: Translate or vendor official C source files into this repository.

## Decision
**Option B** is chosen.
- No C source code is vendored or committed to the repository, preserving the clean MIT license boundary.
- Python implementations of LDPC, masking, interleaving, and sampling must faithfully match the ISO standard and reference CLI behavior.
- Every algorithm stage is validated against the official sidecar captures.

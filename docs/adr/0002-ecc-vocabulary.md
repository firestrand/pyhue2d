# ADR 0002: Error Correction Coding (ECC) Vocabulary

- **Status**: Accepted
- **Decider**: Product / Architecture
- **Date**: 2026-09-24

## Context
ISO/IEC 23634 and the official `jabcode` reference captures specify error correction levels as integers (e.g. `ecc_levels: [3]`, `ecc_levels: [0, 0]`). Some barcode systems (like QR code) use letter levels (`L`, `M`, `Q`, `H`).

## Options Considered
1. **Option A**: Integer-based ECC representation (0..10), directly matching the official captures and ISO/IEC 23634 specification.
2. **Option B**: Exclusively letter-based ECC levels (`L`, `M`, `Q`, `H`), mapping internally to arbitrary integers.
3. **Option C**: Accept both letters and integers across all interfaces.

## Decision
**Option A** is chosen for the core codec, `DecodeResult`, and Tier-1 facts.
- The decoder structured result and oracle verification use exact integer levels (`ecc_level: int`, e.g. 3 or 0) as provided by reference sidecars.
- Any convenience mappings from legacy letter levels (`L`, `M`, `Q`, `H`) in the CLI or high-level wrapper are deferred to Phase V12.

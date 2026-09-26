# Generalized codec execution evidence

## Recovered starting point

The original `pyhue2d-plan.md` V0–V20 milestone was committed as `2c6093b`.
The follow-on plan was committed as `1e0fac0`; V21's generator as `5239f98`,
and V22's sampler as `6eb6ce9`. At resumption the only untracked file was
`a.out`; it was left untouched. No earlier local coding-agent transcript was
found. Initial `just verify`: 923 passed, 18 skipped, 7 xfailed.

## Implemented and observed

- Dynamic LDPC validates all 10,000 PRNG draws for both seeds and complete
  parity-check matrix hashes against independently compiled C oracles.
  Residual syndromes reject invalid codewords; int32 unsatisfied-check counts
  fix overflow when correcting data-bit errors.
- The alignment sampler verifies all 32 table rows and refines actual alignment
  motifs. V10/V32 projective, curved, and radial image tests recover every module.
- Finder-based raster detection and breadth-first docking recover every approved
  two- through nine-symbol capture. Tests cover padding, quarter turns, unequal
  pixel pitches, missing slaves, and malformed channel metadata.
- The public decoder contains no payload signature table. Fresh reference-writer
  images, including binary payloads and unequal symbol dimensions, decode from
  their channel data. Deinterleaving and LDPC happen per symbol, not on the
  combined codeword. Net bits are concatenated before mode decoding.
- `encode` and `encode_symbol` accept `version` and `symbol_count`. Explicit
  versions 1–32 support eight-color binary encoding and 1–61 docked symbols.
  The existing default Version-1 matrix oracle is preserved. The class encoder
  now uses valid channel matrices and preserves black modules in its renderer.
- The official C reader independently recovered Python-generated Version-10,
  explicit-metadata, four-symbol, 61-symbol, and class-encoder PNG payloads.
- Manual public API roundtrips covered `(version, count)` values `(1, 4)`,
  `(10, 4)`, `(32, 2)`, and `(2, 61)`. CLI decode, help, and missing-input
  behavior were exercised successfully.

## Final verification

`just verify` passes, and `uv build` produces both wheel and source distribution.
Fixture integrity, documentation commands, and the fact-surface script pass.
The final verification run returned **1082 passed, 18 skipped, 7 xfailed**.
The full coverage run measured **77% total branch-aware coverage**, above the
recorded 70% baseline. Malformed-channel tests pass independently (24 total)
and achieve 100% statement coverage for `symbol_channel.py`.

An intermediate run exposed the legacy metadata-responsiveness contract: it
corrupts a metadata/palette module in a Version-1 matrix and requires successful
decoding with changed parameters. That test was not weakened or removed.
The legacy low-level codec and Version-1 matrix path retain best-effort output;
generalized channels and image decoding require valid LDPC syndromes. The
existing contract and strict generalized-channel tests both pass. No fact
change or owner approval is needed for this compatibility-preserving resolution.

V21's measured individual-matrix `<50 ms` criterion is met. Bulk LCG generation
uses exact uint64 prefix products and sums, with all 10,000 draws plus following
state verified against the scalar generator for both seeds. Matrix construction
improved 67–69%. Alternating baseline/optimized cold-H measurements covered all
37 distinct capacities in the 32 default layouts: **259 optimized samples**,
worst median **44.606 ms**, worst individual sample **48.325 ms**. Median H
generation improved 11.31%. These are per-matrix measurements on this machine,
not a portable latency guarantee or a combined per-version codebook budget.
Uint64 and batched-elimination probes did not meet their 20% improvement rule
and were removed. The final implementation retains vectorized construction and
bulk PRNG generation, with the original elimination algorithm.

## Scope boundaries

Generalized encoding supports eight colors; approved four-color decoding remains
supported. Raster topology detection does not yet locate arbitrary photographed
multi-symbol layouts. The refined mesh sampler operates with supplied corners;
the existing approved single-symbol photo path remains available. Alignment
refinement assumes standard eight-color RGB and bounded displacement.

Requested ECC labels cannot always be reconstructed uniquely from optimized
reference weights; the decoder retains the project's existing compatibility
mapping for those weights. The original default encoder's seven expected
failures and the 18 quarantined/missing-fixture skips are unchanged.

No commits or pushes were made. Verified changes are staged for inspection;
the original untracked `a.out` remains untouched.

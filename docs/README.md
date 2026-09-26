# Documentation

- [Fact ledger](fact-ledger.md) — the behavioral claims and their evidence.
- [Evidence index](evidence-index.md) — the commands that check those claims.
- [Fixtures](fixtures.md) — where the approved JAB Code captures come from.
- [Generalized codec plan](plans/2026-09-25-generalized-multisymbol-plan.md) — follow-on implementation and verification status.
- [ADRs](adr/0001-codec-algorithm-source.md) — algorithm source, and [ECC vocabulary](adr/0002-ecc-vocabulary.md).

Public entry points are `pyhue2d.encode`, `pyhue2d.decode`, `pyhue2d.inspect_symbol`, `pyhue2d.export_svg`, and `pyhue2d.export_pdf`. Error correction levels in those calls are integers, matching the reference captures.

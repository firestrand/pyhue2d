# Testing Strategy & Exception Hierarchy: pyhue2d

## Testing Strategy

- **Test Runner**: `pytest` executed with `uv run pytest`.
- **Coverage**: Measured with `pytest-cov` via `uv run pytest --cov=pyhue2d --cov-branch`.
- **Mutation Testing**: `mutmut` is configured as the mutation testing tool for high-risk policy validation (specifically EV-20: verifying `JAB.LIBRARY.NO_PRINT.v1` and `JAB.DECODE.LOGS_OMIT_PAYLOAD.v1`). First execution occurs in the Phase V1 sufficiency review (Task V1.17).

## Exception Hierarchy

All library exceptions inherit from `JABCodeError`:

```
JABCodeError (src/pyhue2d/jabcode/exceptions.py)
├── JABCodeDecodeError
│   ├── FinderPatternError
│   ├── AlignmentPatternError
│   ├── MetadataDecodeError
│   └── LDPCDecodeError
├── JABCodeEncodeError
└── JABCodeParameterError
```

`JABCodeError` remains the root of all package-specific exceptions.

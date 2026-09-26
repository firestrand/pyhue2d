# Evidence Index: pyhue2d

| Evidence ID | Facts | Type | Path / Command | Oracle & Fixture Deps | Data Version | Environment | Last Result |
|-------------|-------|------|----------------|-----------------------|--------------|-------------|-------------|
| EV-01 | JAB.IMPORT.FIXTURES_UNCHANGED.v1 | test | `uv run pytest tests/facts/test_import_fixtures_unchanged.py` | tests/support/fixture_digest.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-02 | JAB.LIBRARY.NO_PRINT.v1 | static analysis | `uv run ruff check --select T201 --extend-exclude 'src/pyhue2d/cli.py' src/pyhue2d` | ruff T201 config in pyproject.toml | — | hermetic | Passed |
| EV-03 | JAB.DECODE.EXAMPLE1_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_decode_example1.py::test_example1_payload_matches_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-04 | JAB.DECODE.EXAMPLE1_PARAMETERS.v1 | test | `uv run pytest tests/facts/test_decode_example1.py::test_example1_parameters_match_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-05 | JAB.ENCODE.EXAMPLE1_MATRIX.v1 | test | `uv run pytest tests/facts/test_encode_example1.py::test_example1_matrix_matches_sidecar` | tests/support/sidecar.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-06 | JAB.METADATA.FOLLOWS_MODULES.v1 | test | `uv run pytest tests/facts/test_metadata_modules.py::test_flipping_metadata_module_changes_report` | tests/support/sidecar.py; LOCAL-DATA-01 mutation | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-07 | JAB.CLI.ENCODE_FLAGS.v1 | test | `uv run pytest tests/facts/test_cli_encode_flags.py` | LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-08 | JAB.CLI.DECODE_FLAGS.v1 | test | `uv run pytest tests/facts/test_cli_decode_flags.py` | LOCAL-DATA-01 invalid mutation | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-09 | JAB.DECODE.MODE_FIXTURES.v1 | test | `uv run pytest tests/facts/test_decode_modes.py` | LOCAL-DATA-02 | LOCAL-DATA-02@V0A | hermetic | Passed |
| EV-10 | JAB.DECODE.MULTISYMBOL_TWO.v1 | test | `uv run pytest tests/facts/test_decode_multisymbol.py::test_multi_block_two` | LOCAL-DATA-03 | LOCAL-DATA-03@V0A | hermetic | Passed |
| EV-11 | JAB.DECODE.APPROVED_CORPUS.v1 | test | `uv run pytest tests/facts/test_decode_corpus.py` | LOCAL-DATA-04 | LOCAL-DATA-04@V0A | hermetic | Passed |
| EV-12 | JAB.CAPACITY.EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_capacity.py::test_example1_capacity` | LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-13 | JAB.INSPECT.EXAMPLE1_TRACE.v1 | test | `uv run pytest tests/facts/test_inspect.py::test_example1_trace` | LOCAL-DATA-01 `encoded_data_hex` | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-14 | JAB.EXPORT.SVG_EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_export_svg.py::test_example1_svg_colors` | LOCAL-DATA-01 palette and matrix | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-15 | JAB.EXPORT.PDF_EXAMPLE1.v1 | test | `uv run pytest tests/facts/test_export_pdf.py::test_example1_pdf_colors` | LOCAL-DATA-01 palette and matrix | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-16 | JAB.FRAME.EXAMPLE1_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_frame_decode.py::test_example1_frame` | LOCAL-DATA-01; FrameSource port | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-17 | JAB.METADATA.VARIED_CAPTURE.v1 | test | `uv run pytest tests/facts/test_varied_parameters.py` | LOCAL-DATA-06 | LOCAL-DATA-06@V13 | hermetic | Passed |
| EV-18 | JAB.REFERENCE.ACCEPTS_ENCODE.v1 | test | `uv run pytest tests/facts/test_reference_cli.py` | LOCAL-DATA-05 binary; LOCAL-DATA-01 plaintext | LOCAL-DATA-05@V15 | sandbox CLI | Passed |
| EV-19 | JAB.SCAN.PALETTE_CALIBRATION.v1 | test | `uv run pytest tests/facts/test_photo_scan.py` | LOCAL-DATA-07 | LOCAL-DATA-07@V17 | hermetic | Passed |
| EV-20 | JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | test | `uv run mutmut run` | mutmut config; tests/facts/test_decode_logs.py | — | hermetic | Passed |
| EV-21 | JAB.DECODE.LOGS_OMIT_PAYLOAD.v1 | test | `uv run pytest tests/facts/test_decode_logs.py` | tests/support/log_capture.py; LOCAL-DATA-01 | LOCAL-DATA-01@V0A | hermetic | Passed |
| EV-22 | JAB.LDPC.DYNAMIC_CODEBOOK.v1 | test | `uv run pytest tests/support/test_dynamic_ldpc.py` | ISO/IEC 23634 §7.5, Annex A; C reference PRNG oracle | LOCAL-DATA-01@V21 | hermetic | Passed |
| EV-23 | JAB.SAMPLING.ISO_TABLE5_GRID.v1 | test | `uv run pytest tests/facts/test_alignment_grid_sampling.py` | ISO/IEC 23634 Table 5 coordinates | LOCAL-DATA-01@V22 | hermetic | Passed |
| EV-24 | JAB.TOPOLOGY.DOCKING_TRAVERSAL.v1 | test | `uv run pytest tests/facts/test_multisymbol_docking.py` | ISO/IEC 23634 §6.4; asan_multi2 and multi_block_v32 captures | LOCAL-DATA-03@V23 | hermetic | Passed |
| EV-25 | JAB.CHANNEL.INTER_SYMBOL_ASSEMBLY.v1 | test | `uv run pytest tests/facts/test_multisymbol_interleaving.py` | ISO/IEC 23634 §7.4, §7.6 | LOCAL-DATA-03@V24 | hermetic | Passed |
| EV-26 | JAB.CODEC.GENERAL_MULTISYMBOL.v1 | test | `uv run pytest tests/facts/test_general_multisymbol_decode.py tests/facts/test_general_encode.py` | ISO/IEC 23634 complete spec; arbitrary multi-symbol roundtrips | LOCAL-DATA-03@V25 | hermetic | Passed |


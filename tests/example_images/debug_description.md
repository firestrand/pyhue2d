# JABCode Debug JSON Output Description

This document explains the structure and meaning of the debug JSON files generated by the C reference JABCode encoder. Each debug JSON file is produced alongside its corresponding output image and contains all relevant encoding parameters, symbol data, and metadata needed for deep debugging and cross-implementation comparison.

---

## Top-Level Structure
The JSON file is an object with the following main sections:

- **input_text**: The original input string or data encoded in the symbol. This is parsed from the `--input` CLI argument.
- **color_number**: Number of colors used in the symbol (e.g., 8).
- **module_size**: Size (in pixels) of each module (cell) in the symbol.
- **symbol_number**: Number of symbols (for multi-symbol codes).
- **master_symbol_width / master_symbol_height**: Dimensions of the master symbol (if set).
- **ecc_levels**: Array of error correction levels used for each symbol.
- **symbol_versions**: Array of [x, y] pairs indicating the version (side size) of each symbol.
- **color_space**: Color space used, parsed from the `--color-space` argument (e.g., 0 for RGB).
- **mask_pattern**: The final mask pattern applied to the symbol.
- **quiet_zone**: Width (in modules) of the quiet zone (margin) around the symbol (hardcoded to a typical value).
- **palette**: Array of RGB triplets, listing the color palette used for encoding (in the order of color indices).
- **symbols**: Array of objects, one per symbol, each containing detailed symbol data (see below).
- **final_image_size**: [width, height] of the rendered output image in pixels.
- **notes**: Any additional debug notes or information about missing fields.

---

## Per-Symbol Section (`symbols` array)
Each entry in the `symbols` array contains:

- **matrix_size**: [width, height] of the symbol matrix (in modules).
- **symbol_matrix**: 2D array (rows of columns) of color indices, representing the raw symbol matrix before rendering. Each integer refers to an index in the `palette` array.
- **encoded_data_hex**: The per-symbol encoded data bytes (before error correction), as a hexadecimal string. This represents the data assigned to this specific symbol.
- **ecc_data_hex**: The error-corrected and interleaved data bytes for this symbol, as a hexadecimal string. This is the final data that gets placed into the matrix.
- **finder_patterns**: Placeholder for finder pattern positions and color indices (not available in current build).
- **alignment_patterns**: Placeholder for alignment pattern positions and color indices (not available in current build).

---

## Field Usage and Comparison
- **All fields are intended to allow step-by-step comparison between the C reference and other implementations (e.g., Python).**
- **symbol_matrix** and **palette** allow for direct visual and algorithmic comparison of the encoded structure.
- **encoded_data_hex** and **ecc_data_hex** enable byte-level comparison of the data assignment, encoding, and error correction process for each symbol.
- **finder_patterns** and **alignment_patterns** (when available) help verify correct placement and coloring of key patterns.
- **All input parameters** (color number, ECC, module size, etc.) are logged for reproducibility.

---

## Notes
- Some fields (e.g., finder/alignment patterns) are marked as "not available" as they are not explicitly stored in a simple format in the C encoder. These can be added with further instrumentation if needed.
- The debug JSON is designed to be both human-readable and machine-parseable for automated test harnesses.

---

For any questions or to extend the debug output, see the encoder instrumentation or contact the maintainers. 
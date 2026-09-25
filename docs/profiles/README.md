# Profiling Notes (Phase V19 Hardening)

This directory contains `cProfile` traces for single-symbol (`example1.png`) and large multi-symbol (`multi_block_2_v32.png`) decode execution.

## 1. `example1.png` Decode Trace (`example1_decode.prof.txt`)
- **Total Duration**: ~0.36 seconds
- **Function Breakdown**:
  - `decode_codeword_bits_with_correction`: ~0.347s cumulative
    - `_gauss_jordan`: 0.158s (systematic generator / check matrix reduction over GF(2))
    - `_create_matrix_a`: 0.095s (LDPC base permutation matrix generation)
  - `sample_symbol_matrix`: ~0.004s (pixel extraction and Euclidean color palette matching)
  - `decode_data_from_bits`: < 0.001s (character decoding and state transitions)
- **Observations**: Matrix algebra (Gauss-Jordan elimination) dominates single-symbol decode time. Since the systematic matrix $H_{sys}$ depends only on $(wc=3, wr=6, v=1)$, caching $H_{sys}$ would reduce decode times to < 0.01 seconds.

## 2. Version 32 Multi-Block Decode Trace (`v32_decode.prof.txt`)
- **Total Duration**: ~0.046 seconds
- **Function Breakdown**:
  - NumPy array loading and variance computation: ~0.031s
  - Image file decoding (`PIL.PngImagePlugin`): ~0.012s
- **Observations**: Image loading and memory array access account for 100% of runtime; no performance bottlenecks detected.

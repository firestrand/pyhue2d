# Decode profiling

These `cProfile` reports were regenerated after replacing payload signatures with
real channel decoding. They measure one cold decode in a process; profiling
overhead and concurrent work affect timings.

- [Single symbol](example1_decode.prof.txt): `example1.png`, about 0.019 seconds,
  including finder detection, dynamic LDPC, and mode decoding.
- [Two Version-32 symbols](v32_decode.prof.txt): `multi_block_2_v32.png`, about
  1.156 seconds. The actual channel path now dominates, including LDPC decoding
  and deinterleaving; the linked report includes the per-function breakdown.

The old Version-32 report measured image loading and a stored payload lookup;
its timing was not evidence of codec performance. Current reports validate that
symbol content is decoded. Generator caches reuse matrices across matching
subblocks. No optional accelerator was added.

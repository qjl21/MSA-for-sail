# MSA CHANGELOG

## [0.1.1+v0.1.0] - 2026-11

Alpha release of MSA (MiniMax Sparse Attention), the `fmha_ppu` package for
PPU (HGGC). Provides dense FlashAttention and sparse top-k attention through
a single JIT-compiled `csrc` stack.

### Features

- Dense attention: prefill and decode, contiguous and paged KV cache, in fp8
  and bf16.
- Sparse attention: proxy pass with per-block max score, `sparse_topk_select`
  indexer producing top-k KV block indexes, and sparse prefill/decode over
  `kv_block_indexes`.
- Public API: `fmha_ppu_plan`, `fmha_ppu`, `sparse_topk_select`, plus
  `allocate_graph_workspace`; also registered as `minfer.ops.*` TVM FFI globals.
- Kernels JIT-compiled at first import from `csrc` Jinja templates with
  `hgcc` + `ninja`; ACTLIZE submodule pinned to v1.0.0 (`664f5c19`).
- Benchmark suite covering dense/paged/sparse prefill and decode in fp8 and
  bf16 (`benchmarks/bench_sparse_attention_ops.py`).

### Compatibility

- Python ≥ 3.10; Linux x86_64; PPU SDK with `hgcc` on `PATH` (or `PPU_SDK`).
- FP8 sparse requires PPU0015 (890 / M890); not supported on PPU0010 (810E).

### Packaging

- Package version is composed at build time from the release tag and the PPU SDK:
  `0.1.1+v0.1.0.ppu<sdk>` on this tag, and
  `0.1.1.dev<distance>+v0.1.0.ppu<sdk>.<sha>` after it.
- Building a wheel requires the `0.1.1+v0.1.0` git tag and `hgcc` on `PATH`
  (or `PPU_SDK` set).

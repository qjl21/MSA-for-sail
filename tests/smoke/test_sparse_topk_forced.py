# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Test forced begin/end block selection in sparse_topk_select.

Verifies that force_begin_blocks and force_end_blocks guarantee those
block indices appear in the output regardless of their scores.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch
from msa_for_sail import sparse_topk_select


def test_forced_blocks(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=8,
    topk=16, num_valid_pages=200,
    force_begin=3, force_end=2,
    seed=42, dtype=torch.float32,
):
    """Core test: forced blocks must appear in output even with worst scores."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    # Generate random scores, then deliberately set forced blocks to WORST scores
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=dtype)
    # Fill padding with -inf
    max_score[:, num_valid_pages:, :] = float('-inf')
    # Give forced blocks the WORST possible valid scores to stress-test
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    # Run with forced selection
    result = sparse_topk_select(
        max_score, topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin,
        force_end_blocks=force_end,
    )
    assert result.shape == (total_qo_len, num_qo_heads, topk)
    assert result.dtype == torch.int32

    # Verify: all forced begin indices [0, force_begin) must appear in every row
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, (
                    f"[FAIL] force_begin idx={idx} missing from row (t={t}, h={h}): {sorted(row)}"
                )
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, (
                    f"[FAIL] force_end idx={idx} missing from row (t={t}, h={h}): {sorted(row)}"
                )
    print(f"  [PASS] force_begin={force_begin}, force_end={force_end}, "
          f"max_k={max_k_tiles}, nvp={num_valid_pages}")


def test_forced_zero_is_noop(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=8,
    topk=16, num_valid_pages=200, seed=42, dtype=torch.float32,
):
    """force_begin=0, force_end=0 must produce identical results to no-force."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=dtype)
    max_score[:, num_valid_pages:, :] = float('-inf')

    result_noop = sparse_topk_select(max_score, topk, num_valid_pages=num_valid_pages)
    result_zero = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=0, force_end_blocks=0,
    )
    assert torch.equal(result_noop, result_zero), (
        f"[FAIL] force_begin=0, force_end=0 differs from default"
    )
    print("  [PASS] force_begin=0, force_end=0 == no-force (bitwise identical)")


def test_forced_ascending_order(
    num_qo_heads=4, max_k_tiles=512, total_qo_len=4,
    topk=16, num_valid_pages=400,
    force_begin=4, force_end=3, seed=123, dtype=torch.float32,
):
    """Output must still be in ascending order with forced blocks."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=dtype)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = result_cpu[t, h].tolist()
            valid = [x for x in row if x >= 0]
            assert valid == sorted(valid), (
                f"[FAIL] row not ascending at (t={t}, h={h}): {row}"
            )
    print(f"  [PASS] ascending order preserved with force_begin={force_begin}, force_end={force_end}")


def test_forced_with_xor_fast_path(
    num_qo_heads=4, max_k_tiles=256, total_qo_len=32,
    topk=16, num_valid_pages=224,
    force_begin=2, force_end=3, seed=77, dtype=torch.float32,
):
    """XorF4 transpose fast path requires qo%32==0 and K%32==0."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    assert total_qo_len % 32 == 0 and max_k_tiles % 32 == 0
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=dtype)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, f"[FAIL] XorF4 path: begin idx={idx} missing"
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, f"[FAIL] XorF4 path: end idx={idx} missing"
    print(f"  [PASS] XorF4 fast path: force_begin={force_begin}, force_end={force_end}, "
          f"qo={total_qo_len}, K={max_k_tiles}")


def test_forced_large_k(
    num_qo_heads=4, max_k_tiles=4096, total_qo_len=4,
    topk=16, num_valid_pages=4000,
    force_begin=2, force_end=4, seed=99, dtype=torch.float32,
):
    """Large K (> 4096 tiles) to stress histogram multi-pass path."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    max_score = torch.randn(num_qo_heads, max_k_tiles, total_qo_len,
                            device=dev, dtype=dtype)
    max_score[:, num_valid_pages:, :] = float('-inf')
    max_score[:, :force_begin, :] = -1e10
    max_score[:, num_valid_pages - force_end:num_valid_pages, :] = -1e10

    result = sparse_topk_select(
        max_score, topk, num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin, force_end_blocks=force_end,
    )
    result_cpu = result.cpu()
    for t in range(total_qo_len):
        for h in range(num_qo_heads):
            row = set(result_cpu[t, h].tolist())
            row.discard(-1)
            for idx in range(force_begin):
                assert idx in row, f"[FAIL] large_k: begin idx={idx} missing"
            for idx in range(num_valid_pages - force_end, num_valid_pages):
                assert idx in row, f"[FAIL] large_k: end idx={idx} missing"
    print(f"  [PASS] large K={max_k_tiles}: force_begin={force_begin}, force_end={force_end}")


def test_dist_matrix(topk=16, seed=42, dtype=torch.float32):
    """Same shape, different data distributions: exercise every Phase-4
    tie-break branch of IndexerTopKRegHistBF16Kernel and check exactness.

    Shape is fixed at the online decode CUDA-graph case (H=4, K=8192, T=8,
    topk=16, nvp=8192); only the value distribution changes:

      case                distribution                        Phase-4 branch
      ------------------- ----------------------------------- -------------------------
      randn               dense random scores                 tiny threshold bin (healthy baseline)
      tie32               7x 2.0 + 25x -1.0 + -inf sea        num_ties=25  -> single-ballot
      tie64               7x 2.0 + 50x -1.0 + -inf sea        num_ties=50  -> dual-ballot
      uniform-sea-random  7x 2.0 at random pos + -inf sea     num_ties=8185 all-equal -> uniform fast
                                                                path; non-contiguous tie idx strictly
                                                                checks ascending-idx selection
      uniform-sea-prefix  randn at [0,7) + -inf sea           online static-nvp CUDA-graph shape
                                                                (K == buffer capacity, kv_lens ~ 840):
                                                                finite < topk, remaining slots legally
                                                                take -inf positions, never -1
      zero-sea            7x 2.0 + 8185x 0.0                  uniform fast path on a non -inf score
      nonuniform-sea      7x 2.0 + 1.0/1.0078125 mixed sea    num_ties=8185 mixed scores in ONE fp16
                                                                bin -> argmax-rounds fallback
                                                                (quantized-score path, e.g. fp8)

    1.0078125 = 1 + 2^-7: distinct from 1.0 in bf16/fp32, but the fp16
    top-12-bit bin key is identical (0x3C00>>4 == 0x3C08>>4), so the whole
    sea shares one coarse bin with mixed scores.

    Correctness is checked per dtype:
      bf16 (RegHist): exact per-row match against its deterministic contract
        -- top-16 by (score desc, idx asc), emitted ascending by idx --
        computed with a stable torch.sort (ties keep ascending idx).
      fp32 (IndexerTopK): tie selection is NOT idx-ascending, so only the
        selected score multiset is contractual (tie-immune check).
    """
    H, K, T = 4, 8192, 8
    nvp = K
    dev = torch.device("cuda")

    def build(kind):
        g = torch.Generator().manual_seed(seed)
        if kind == "randn":
            return torch.randn(H, K, T, generator=g)
        # Random per-row positions for the 7 above-threshold scores.
        order = torch.rand(H, K, T, generator=g).argsort(dim=1)
        if kind in ("tie32", "tie64"):
            n_tie = 25 if kind == "tie32" else 50
            m = torch.full((H, K, T), float("-inf"))
            m.scatter_(1, order[:, :7, :], 2.0)
            m.scatter_(1, order[:, 7:7 + n_tie, :], -1.0)
            return m
        if kind == "uniform-sea-random":
            m = torch.full((H, K, T), float("-inf"))
            m.scatter_(1, order[:, :7, :], 2.0)
            return m
        if kind == "uniform-sea-prefix":
            m = torch.full((H, K, T), float("-inf"))
            m[:, :7, :] = torch.randn(H, 7, T, generator=g)
            return m
        if kind == "zero-sea":
            m = torch.zeros(H, K, T)
            m.scatter_(1, order[:, :7, :], 2.0)
            return m
        if kind == "nonuniform-sea":
            m = torch.ones(H, K, T)
            m[:, 1::2, :] = 1.0078125  # same fp16 bin as 1.0, different score
            m.scatter_(1, order[:, :7, :], 2.0)
            return m
        raise ValueError(kind)

    CASES = [
        ("randn",              "healthy dense path"),
        ("tie32",              "num_ties=25  -> single-ballot"),
        ("tie64",              "num_ties=50  -> dual-ballot"),
        ("uniform-sea-random", "num_ties=8185 all -inf  -> uniform fast path"),
        ("uniform-sea-prefix", "static-nvp: finite at [0,7) + -inf sea -> uniform fast path"),
        ("zero-sea",           "num_ties=8185 all 0.0   -> uniform fast path"),
        ("nonuniform-sea",  "num_ties=8185 mixed 1bin -> argmax-rounds fallback"),
    ]

    def check(result, m_f32, tag):
        assert result.shape == (T, H, topk) and result.dtype == torch.int32
        r = result.cpu()
        assert bool(((r >= 0) & (r < nvp)).all()), \
            f"[FAIL] {tag}: idx out of [0,{nvp}) or unexpected -1"
        assert bool((r[..., 1:] >= r[..., :-1]).all()), \
            f"[FAIL] {tag}: not ascending"
        if dtype == torch.bfloat16:
            # bf16 RegHist deterministic contract: top-16 by (score desc,
            # idx asc), emitted ascending by idx; stable sort keeps ties in
            # ascending idx order.
            _, order = m_f32.sort(dim=1, descending=True, stable=True)
            ref = order[:, :topk, :].sort(dim=1).values.permute(2, 0, 1)
            assert torch.equal(r.long(), ref), (
                f"[FAIL] {tag}: output != (score desc, idx asc) top-{topk}\n"
                f"  row(t=0,h=0) got {r[0, 0].tolist()}\n"
                f"  row(t=0,h=0) ref {ref[0, 0].tolist()}")
        else:
            # fp32 IndexerTopK: tie selection is not idx-ascending; verify
            # the selected score multiset instead (tie-immune).
            ref = m_f32.topk(topk, dim=1).values                       # (H, topk, T)
            sel = m_f32.gather(1, r.permute(1, 2, 0).long())           # (H, topk, T)
            assert torch.equal(sel.sort(dim=1).values, ref.sort(dim=1).values), \
                f"[FAIL] {tag}: selected scores do not match top-{topk}"

    for kind, tag in CASES:
        max_score = build(kind).to(dev).to(dtype)
        check(sparse_topk_select(max_score, topk, num_valid_pages=nvp),
              max_score.float().cpu(), kind)
        print(f"  [PASS] {kind:<20} H={H} K={K} T={T} nvp={nvp} | {tag}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                        help="Score dtype for the forced-block functional cases "
                             "(bf16 exercises the transpose_bf16 + RegHist path)")
    args = parser.parse_args()

    dev = torch.device("cuda")

    DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"=== Testing forced block selection (dtype={args.dtype}) ===")
    test_forced_zero_is_noop(dtype=DTYPE)
    test_forced_blocks(dtype=DTYPE)
    test_forced_blocks(force_begin=1, force_end=0, seed=10, dtype=DTYPE)
    test_forced_blocks(force_begin=0, force_end=5, seed=20, dtype=DTYPE)
    test_forced_blocks(force_begin=8, force_end=8, seed=30, dtype=DTYPE)
    test_forced_ascending_order(dtype=DTYPE)
    test_forced_with_xor_fast_path(dtype=DTYPE)
    test_forced_large_k(dtype=DTYPE)
    test_dist_matrix(dtype=DTYPE)
    print(f"\nAll forced-block tests PASSED (dtype={args.dtype})!")

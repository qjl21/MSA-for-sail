# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: MIT

"""Trimmed from test_proxy_kv_smoke.py:

    KV cache A (proxy, MQA-compressed)
        ↓
    fmha_sail dense (num_kv_heads=1) → max_score
"""
import argparse
import math
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import fp8_test_dtypes
from msa_for_sail import fmha_sail, fmha_sail_plan, sparse_topk_select


def quantize_qkv(x, storage_dtype):
    """Return (kernel_input, bf16 reference) for the given Q/K/V storage dtype."""
    if storage_dtype == "fp8":
        q = x.to(torch.float8_e4m3fn)
        return q, q.float().to(torch.bfloat16)
    return x, x

def verify_topk_selection(max_score, kv_block_indexes, topk, atol=1e-4):
    """Kernel-selected score multiset vs torch.topk on the same max_score."""
    s_THK = max_score.permute(2, 0, 1).contiguous()  # (T, H, K)
    ref_topk = torch.topk(s_THK, k=topk, dim=-1)
    ref_scores, _ = torch.sort(ref_topk.values, dim=-1)

    valid_mask = kv_block_indexes >= 0
    safe_idx = torch.where(valid_mask, kv_block_indexes, torch.zeros_like(kv_block_indexes))
    kernel_scores = torch.gather(s_THK, -1, safe_idx.long())
    kernel_scores = torch.where(valid_mask, kernel_scores,
                                torch.full_like(kernel_scores, float("-inf")))
    kernel_scores, _ = torch.sort(kernel_scores, dim=-1)

    both_finite = kernel_scores.isfinite() & ref_scores.isfinite()
    diff = (kernel_scores - ref_scores).abs()
    diff_finite = torch.where(both_finite, diff, torch.zeros_like(diff))
    max_diff = diff_finite.max().item()
    n_inf_mismatch = (kernel_scores.isfinite() != ref_scores.isfinite()).sum().item()
    return (max_diff < atol) and (n_inf_mismatch == 0), max_diff, n_inf_mismatch

def ref_tile_max_score(q, k, qo_segment_lens, kv_segment_lens, num_kv_heads, max_k_tiles, causal=False, qo_offset_tensor=None):
    """Reference per-tile max QK score. Output layout: [head, k_tile, token]."""
    total_qo_len, num_qo_heads, head_dim = q.shape
    batch_size = qo_segment_lens.shape[0]
    h_r = num_qo_heads // num_kv_heads
    kv_tile_size = 128

    result = torch.full(
        (num_qo_heads, max_k_tiles, total_qo_len),
        -float("inf"), dtype=torch.float32, device=q.device,
    )

    qo_off = 0
    kv_offset = 0
    for b in range(batch_size):
        qo_len = int(qo_segment_lens[b].item())
        kv_len = int(kv_segment_lens[b].item())
        n_tiles = (kv_len + kv_tile_size - 1) // kv_tile_size

        q_seq = q[qo_off : qo_off + qo_len].float()
        k_seq = k[kv_offset : kv_offset + kv_len].float()

        causal_off = int(qo_offset_tensor[b].item()) if qo_offset_tensor is not None else (kv_len - qo_len)

        for t in range(n_tiles):
            k_start = t * kv_tile_size
            k_end = min(k_start + kv_tile_size, kv_len)
            k_chunk = k_seq[k_start:k_end]
            k_expanded = k_chunk.repeat_interleave(h_r, dim=1)
            scores = torch.einsum("qhd,khd->qhk", q_seq, k_expanded)
            if causal:
                q_pos = torch.arange(qo_len, device=q.device).unsqueeze(1) + causal_off
                k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(~(q_pos >= k_pos).unsqueeze(1), -float("inf"))
            tile_max = scores.max(dim=-1).values  # [qo_len, num_qo_heads]
            result[:, t, qo_off : qo_off + qo_len] = tile_max.permute(1, 0)

        qo_off += qo_len
        kv_offset += kv_len

    return result

def check_maxscore(max_score, ref):
    """Compare max_score against precomputed reference. Returns True if passed."""
    print(f"max_score shape {tuple(max_score.shape)}"
          f" ref shape {tuple(ref.shape)}")
    print(f"max_score\n{max_score[0, :5]}")
    print(f"ref\n{ref[0, :5]}")
    valid = ref != -float("inf")
    if not valid.any():
        return True
    # max_score may be bf16 (opt-in) or fp32; compare in fp32.
    kv = max_score[valid].float()
    rv = ref[valid].float()
    diff = (kv - rv).abs()
    passed = torch.allclose(kv, rv, atol=1e-1, rtol=1e-1)
    torch.cuda.synchronize()
    status = "PASS" if passed else "FAIL"
    print(f"[{status} maxscore] max_diff={diff.max().item():.4f}")
    return passed

def run(check=False, seed=0,
        total_qo_len=8, num_kv_heads_real=4, num_kv_splits=-1,
        total_kv_len=256, page_size=128, head_dim=128, topk=16,
        storage_dtype="bf16", max_score_dtype="float32"):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    ms_dtype = {"float32": torch.float32, "bf16": torch.bfloat16}[max_score_dtype]

    num_qo_heads_dense = num_kv_heads_real
    num_kv_heads_dense = 1

    kv_len = total_kv_len
    pages_per_seq = (kv_len + page_size - 1) // page_size

    k_pages_dense, k_pages_dense_ref = quantize_qkv(torch.randn(
        pages_per_seq, num_kv_heads_dense, page_size, head_dim, device=dev, dtype=dtype), storage_dtype)
    v_pages_dense, _ = quantize_qkv(torch.randn(
        pages_per_seq, num_kv_heads_dense, page_size, head_dim, device=dev, dtype=dtype), storage_dtype)
    q_dense, q_dense_ref = quantize_qkv(torch.randn(
        total_qo_len, num_qo_heads_dense, head_dim, device=dev, dtype=dtype), storage_dtype)

    # ── Stage 1: Dense pass on KV cache A → max_score ──
    qo_segment_lens_d = torch.tensor([total_qo_len], dtype=torch.int32)
    kv_segment_lens_d = torch.tensor([kv_len], dtype=torch.int32)
    kv_indices_d = torch.arange(pages_per_seq, device=dev, dtype=torch.int32)
    plan_dense = fmha_sail_plan(
        qo_segment_lens_d, kv_segment_lens_d, num_qo_heads_dense,
        num_kv_splits=num_kv_splits,
        causal=True, page_size=page_size,
        output_maxscore=True,
    )
    _, max_score = fmha_sail(
        q_dense, k_pages_dense, v_pages_dense, plan_dense,
        sm_scale=1.0 / math.sqrt(head_dim), kv_indices=kv_indices_d,
        output_o=False, output_maxscore=True,
        max_score_dtype=ms_dtype,
    )
    assert max_score is not None
    assert max_score.dim() == 3 and max_score.shape[0] == num_qo_heads_dense \
        and max_score.shape[2] == total_qo_len, f"max_score shape {tuple(max_score.shape)}"
    # max_score dtype follows max_score_dtype (float32 default; bf16 selects
    # the OnlyScoreBF16 variant) for both bf16 and fp8 Q/K.
    assert max_score.dtype == ms_dtype, \
        f"max_score dtype {max_score.dtype} != {ms_dtype}"
    assert not max_score.isnan().any()

    # ── Stage 2: sparse_topk_select → kv_block_indexes ──
    # num_valid_pages caps OOB indices (max_k_tiles is round-up-aligned > actual pages).
    kv_block_indexes = sparse_topk_select(
        max_score, topk, num_valid_pages=pages_per_seq)
    assert kv_block_indexes.shape == (total_qo_len, num_qo_heads_dense, topk)
    assert kv_block_indexes.dtype == torch.int32

    print(f"[OK] max_score{tuple(max_score.shape)} (max_k_tiles={max_score.shape[1]}) "
          f"→ kv_block_indexes{tuple(kv_block_indexes.shape)} ")

    # ── Optional: full correctness check ──
    if check:
        topk_ok, topk_diff, n_inf_mis = verify_topk_selection(
            max_score, kv_block_indexes, topk)
        print(f"  topk select : max_diff={topk_diff:.6f}, n_inf_mis={n_inf_mis} "
              f"→ {'OK' if topk_ok else 'FAIL'}")
        assert topk_ok
        # Reconstruct contiguous K from the paged cache (following the SAME page table) for reference.
        k_dense_contig = torch.zeros((kv_len + page_size - 1) // page_size * page_size, num_kv_heads_dense, head_dim, dtype=torch.bfloat16, device=dev)
        for pi in range(pages_per_seq):
            phys_idx = kv_indices_d[pi].item()
            start = pi * page_size
            end = start + page_size
            k_dense_contig[start:end] = k_pages_dense_ref[phys_idx].transpose(0, 1)
        k_dense_contig = k_dense_contig[:kv_len]  # crop padding
        ms_ref = ref_tile_max_score(q_dense_ref, k_dense_contig, qo_segment_lens_d, kv_segment_lens_d, num_kv_heads_dense, max_score.shape[1], causal=True)
        ms_ok = check_maxscore(max_score, ms_ref)

        # Cross-dtype check: rerun with the other max_score dtype; the two
        # outputs must agree within bf16 rounding (and on -inf positions).
        other_dtype = (torch.bfloat16 if ms_dtype == torch.float32
                       else torch.float32)
        _, max_score_other = fmha_sail(
            q_dense, k_pages_dense, v_pages_dense, plan_dense,
            sm_scale=1.0 / math.sqrt(head_dim), kv_indices=kv_indices_d,
            output_o=False, output_maxscore=True,
            max_score_dtype=other_dtype,
        )
        assert max_score_other.dtype == other_dtype, \
            f"max_score dtype {max_score_other.dtype} != {other_dtype}"
        a, b = max_score.float(), max_score_other.float()
        assert torch.equal(a.isfinite(), b.isfinite()), "-inf pattern differs across dtypes"
        finite = a.isfinite()
        if finite.any():
            rel = ((a - b).abs()[finite] / a.abs()[finite].clamp_min(1.0)).max().item()
        else:
            rel = 0.0
        cross_ok = rel < 0.01
        print(f"  cross-dtype ({ms_dtype} vs {other_dtype}): max_rel_diff={rel:.4f} "
              f"→ {'OK' if cross_ok else 'FAIL'}")
        assert cross_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal sparse attention E2E "
                    "(max_score → sparse_topk_select → sparse fmha output).",
    )
    parser.add_argument("--check", action="store_true",
                        help="Also run topk-multiset + PyTorch sparse-ref cosine checks.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-qo-len", type=int, default=8192,
                        help="T = number of query tokens. 1 = decode, large = prefill.")
    parser.add_argument("--num-kv-heads-real", type=int, default=4,
                        help="KV head count of the REAL GQA cache.")
    parser.add_argument("--num-kv-splits", type=int, default=-1,
                        help="Number of KV splits. -1 → auto (one of [1, 2, 4, 8]).")
    parser.add_argument("--total-kv-len", type=int, default=8192,
                        help="KV prefix length. 256 → max_k_tiles=128 (Filtered single-CTA "
                             "path); 524000 → max_k_tiles=4096 (Multi-CTA Lookback path).")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", dest="storage_dtype", default="bf16",
                        choices=["bf16", "fp8"] if torch.float8_e4m3fn in
                        fp8_test_dtypes(torch.float8_e4m3fn) else ["bf16"],
                        help="Q/K/V storage dtype: fp8=float8_e4m3fn "
                             "(fp8 OnlyScore path). fp8 is PPU0015-only.")
    parser.add_argument("--max-score-dtype", default="float32",
                        choices=["float32", "bf16"],
                        help="max_score output dtype: bf16 selects the "
                             "OnlyScoreBF16 variant. --check additionally "
                             "cross-compares both dtypes.")
    # NOTE: minfer.sparse_topk_select currently asserts topk == 16,
    # so we hard-fix it here rather than expose a misleading CLI flag.
    args = parser.parse_args()
    run(**vars(args))

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: MIT

"""Smoke-level max_score correctness tests (trimmed from test_correctness.py).

Only validates the max_score output against a PyTorch tile-level reference
(`ref_tile_max_score`).  Output (o) correctness is NOT checked here.

Test categories mirror test_correctness.py:
  - Prefill (TILE_Q=256)
  - Short Q / Decode (TILE_Q=128)
  - Corner Cases
"""

import sys
import torch
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import fp8_test_dtypes
from fmha_ppu import fmha_ppu as _fmha_ppu
from fmha_ppu import fmha_ppu_plan

fmha_ppu = None

page_sizes = [128]

def ref_tile_max_score(q, k, qo_segment_lens, kv_segment_lens,
                       num_kv_heads, max_k_tiles, causal=False,
                       qo_offset_tensor=None):
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

        causal_off = (int(qo_offset_tensor[b].item())
                      if qo_offset_tensor is not None
                      else (kv_len - qo_len))

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


def check_maxscore(name, max_score, ref, log=True):
    """Compare max_score against precomputed reference. Returns True if passed.

    float32 max_score: allclose with atol/rtol=1e-1 (fp32 accumulation-order
    noise vs the einsum reference).
    bfloat16 max_score: tightened to 1 bf16 ulp of the once-rounded reference.
    The kernel rounds the fp32 tile max exactly once at the store, so any diff
    beyond 1 ulp (2^-7 relative, floored at 1e-2) is a real error; the 1 ulp
    slack absorbs MMA-vs-einsum fp32 noise crossing a rounding boundary.
    """
    valid = ref != -float("inf")
    if not valid.any():
        return True
    # max_score may be bf16 (OnlyScoreBF16 variant); compare in fp32.
    kv = max_score[valid].float()
    rv = ref[valid].float()
    if max_score.dtype == torch.bfloat16:
        rv = rv.to(torch.bfloat16).float()
        ulp = torch.exp2(torch.floor(torch.log2(rv.abs().clamp_min(1e-30))) - 7)
        diff = (kv - rv).abs()
        passed = bool((diff <= ulp.clamp_min(1e-2)).all())
    else:
        diff = (kv - rv).abs()
        passed = torch.allclose(kv, rv, atol=1e-1, rtol=1e-1)
    torch.cuda.synchronize()
    status = "PASS" if passed else "FAIL"
    if log:
        print(f"    [{status} maxscore] max_diff={diff.max().item():.4f}")
    if not passed:
        failed_cases.append(f"{name} maxscore: max_diff={diff.max().item():.4f}")
    return passed


failed_cases = []


# ---------------------------------------------------------------------------
# Single-case runner (max_score only)
# ---------------------------------------------------------------------------

def _run_case_maxscore(b, q_len, k_len, h_q, h_k, d, dtype,
                       num_kv_splits=-1, causal=True,
                       max_score_dtype=torch.float32):
    """Run max_score correctness for one ragged (dense) case."""
    device = "cuda:0"
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
    ms_name = {torch.float32: "ms_fp32", torch.bfloat16: "ms_bf16"}[max_score_dtype]
    name = (f"b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} d={d} "
            f"{dtype_name} {ms_name} splitKV={num_kv_splits} "
            f"{'causal' if causal else 'noncausal'}")

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = (torch.full((b,), qo_offset_val, device=device, dtype=torch.int32)
                        if causal else None)

    if dtype == torch.float8_e4m3fn:
        q_fi = q_bf16.reshape(b * q_len, h_q, d).to(dtype)
        k_fi = k_bf16.reshape(b * k_len, h_k, d).to(dtype)
        v_fi = v_bf16.reshape(b * k_len, h_k, d).to(dtype)
    else:
        q_fi = q_bf16.reshape(b * q_len, h_q, d)
        k_fi = k_bf16.reshape(b * k_len, h_k, d)
        v_fi = v_bf16.reshape(b * k_len, h_k, d)

    # Skip if max_score tensor would exceed int32 element count
    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    if maxscore_elems >= (1 << 31):
        print(f"  [SKIP] {name}: maxscore_elems={maxscore_elems} >= 2^31")
        return True

    passed = True
    try:
        qo_lens = torch.full((b,), q_len, device=device, dtype=torch.int32)
        kv_lens = torch.full((b,), k_len, device=device, dtype=torch.int32)
        plan_info = fmha_ppu_plan(
            qo_lens, kv_lens, h_q,
            causal=causal, qo_offset=qo_offset_tensor,
            num_kv_splits=num_kv_splits,
            output_maxscore=True,
            num_kv_heads=h_k,
            device=device
        )
        torch.cuda.synchronize()
        o, max_score = fmha_ppu(
            q_fi, k_fi, v_fi,
            plan_info=plan_info,
            output_maxscore=True,
            output_o=False,
            max_score_dtype=max_score_dtype,
        )
        assert max_score.dtype == max_score_dtype, \
            f"{name}: max_score dtype {max_score.dtype} != {max_score_dtype}"
        max_k_tiles = max_score.shape[1]

        ms_ref = None
        if causal:
            qo_lens = torch.full((b,), q_len, device=device, dtype=torch.int32)
            kv_lens = torch.full((b,), k_len, device=device, dtype=torch.int32)
            q_ref_flat = (q_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else q_fi).reshape(b * q_len, h_q, d)
            k_ref_flat = (k_fi.to(torch.bfloat16) if dtype == torch.float8_e4m3fn else k_fi).reshape(b * k_len, h_k, d)
            ms_ref = ref_tile_max_score(
                q_ref_flat, k_ref_flat, qo_lens, kv_lens, h_k,
                max_k_tiles, causal=True,
            )

        if causal and max_score is not None and ms_ref is not None:
            passed &= check_maxscore(name + " ms_only", max_score, ms_ref)

    except Exception as e:
        print(f"    [FAIL maxscore] {name}: {type(e).__name__}: {str(e)[:120]}")
        failed_cases.append(f"{name} maxscore: {type(e).__name__}")
        passed = False

    return passed


def _run_case_paged_maxscore(b, q_len, k_len, h_q, h_k, d, dtype, page_size=128,
                             causal=True, max_score_dtype=torch.float32):
    """Paged KV cache max_score correctness (shuffled page indices)."""
    device = "cuda:0"
    dtype_name = {torch.float8_e4m3fn: "fp8", torch.bfloat16: "bf16"}[dtype]
    ms_name = {torch.float32: "ms_fp32", torch.bfloat16: "ms_bf16"}[max_score_dtype]
    name = (f"paged-ms b={b} q={q_len} k={k_len} h_q={h_q} h_k={h_k} d={d} "
            f"ps={page_size} {dtype_name} {ms_name}")

    maxscore_elems = h_q * ((k_len + 127) // 128) * b * q_len
    if maxscore_elems >= (1 << 31):
        print(f"  [SKIP] {name}: maxscore_elems={maxscore_elems} >= 2^31")
        return True

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_offset_val = k_len - q_len if causal else 0
    qo_offset_tensor = torch.full((b,), qo_offset_val, dtype=torch.int32) if causal else None

    # Build paged KV cache with shuffled page indices
    pages_per_seq = (k_len + page_size - 1) // page_size
    padded_len = pages_per_seq * page_size
    total_pages = b * pages_per_seq

    k_bf16_padded = k_bf16
    v_bf16_padded = v_bf16
    if padded_len > k_len:
        pad = torch.zeros(b, padded_len - k_len, h_k, d, dtype=torch.bfloat16, device=device)
        k_bf16_padded = torch.cat([k_bf16, pad], dim=1)
        v_bf16_padded = torch.cat([v_bf16, pad], dim=1)
    k_pages_nhd = k_bf16_padded.reshape(b, pages_per_seq, page_size, h_k, d).reshape(total_pages, page_size, h_k, d)
    v_pages_nhd = v_bf16_padded.reshape(b, pages_per_seq, page_size, h_k, d).reshape(total_pages, page_size, h_k, d)
    k_pages = k_pages_nhd.transpose(1, 2).contiguous()   # (total_pages, h_k, page_size, d)
    v_pages = v_pages_nhd.transpose(1, 2).contiguous()

    perm = torch.randperm(total_pages, device=device, dtype=torch.int64)
    k_paged_bf16 = torch.empty_like(k_pages)
    v_paged_bf16 = torch.empty_like(v_pages)
    k_paged_bf16[perm] = k_pages
    v_paged_bf16[perm] = v_pages
    kv_indices = perm.to(torch.int32)

    if dtype == torch.float8_e4m3fn:
        q_fi = q_bf16.reshape(b * q_len, h_q, d).to(dtype)
        k_paged_fi = k_paged_bf16.to(dtype)
        v_paged_fi = v_paged_bf16.to(dtype)
        q_ref_bf16 = q_fi.reshape(b, q_len, h_q, d).to(torch.bfloat16)
        k_paged_for_ref = k_paged_fi.to(torch.bfloat16)
        v_paged_for_ref = v_paged_fi.to(torch.bfloat16)
    else:
        q_fi = q_bf16.reshape(b * q_len, h_q, d)
        k_paged_fi = k_paged_bf16
        v_paged_fi = v_paged_bf16
        q_ref_bf16 = q_bf16
        k_paged_for_ref = k_paged_bf16
        v_paged_for_ref = v_paged_bf16

    # Reconstruct contiguous K/V from paged cache for reference
    k_ref_contig = torch.zeros(b, padded_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_ref_contig = torch.zeros(b, padded_len, h_k, d, dtype=torch.bfloat16, device=device)
    for bi in range(b):
        for pi in range(pages_per_seq):
            logical_idx = bi * pages_per_seq + pi
            phys_idx = kv_indices[logical_idx].item()
            start = pi * page_size
            end = start + page_size
            k_ref_contig[bi, start:end] = k_paged_for_ref[phys_idx].transpose(0, 1)
            v_ref_contig[bi, start:end] = v_paged_for_ref[phys_idx].transpose(0, 1)
    k_ref_contig = k_ref_contig[:, :k_len]
    v_ref_contig = v_ref_contig[:, :k_len]

    q_ref_flat = q_ref_bf16.reshape(b * q_len, h_q, d)
    k_ref_flat = k_ref_contig.reshape(b * k_len, h_k, d)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)

    passed = True
    try:
        torch.cuda.synchronize()
        plan_info_ms = fmha_ppu_plan(qo_lens, kv_lens, h_q,
            causal=causal, qo_offset=qo_offset_tensor,
            page_size=page_size,
            output_maxscore=True,
            num_kv_heads=h_k,
        )
        torch.cuda.synchronize()
        max_k_tiles = max(plan_info_ms[3].get("max_k_tiles", -1),
                          plan_info_ms[4].get("max_k_tiles", -1) if plan_info_ms[4] else -1)

        ms_ref = ref_tile_max_score(q_ref_flat.float(), k_ref_flat.float(),
                                    qo_lens, kv_lens, h_k, max_k_tiles,
                                    causal=True) if causal else None

        # Mode 2: output_o=False, output_maxscore=True
        torch.cuda.synchronize()
        _, max_score = fmha_ppu(
            q_fi, k_paged_fi, v_paged_fi,
            plan_info=plan_info_ms,
            kv_indices=kv_indices,
            output_o=False, output_maxscore=True,
            max_score_dtype=max_score_dtype,
        )
        torch.cuda.synchronize()
        assert max_score.dtype == max_score_dtype, \
            f"{name}: max_score dtype {max_score.dtype} != {max_score_dtype}"
        if ms_ref is not None and max_score is not None:
            passed &= check_maxscore(name + " ms_only", max_score, ms_ref)
    except Exception as e:
        print(f"    [FAIL maxscore] {name}: {type(e).__name__}: {str(e)[:120]}")
        failed_cases.append(f"{name} maxscore: {type(e).__name__}")
        passed = False

    return passed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_pass = True

    # OnlyScore does not require fp8 on PPU0010 (810E); it is not validated there.
    dtype_list = fp8_test_dtypes(torch.bfloat16, torch.float8_e4m3fn)
    split_list = [-1, 2, 4]

    # --- Prefill tests (TILE_Q=256 path, q_len > 128) ---
    print("=== Prefill (TILE_Q=256) — max_score ===")
    for dtype in dtype_list:
        for split in split_list:
            all_pass &= _run_case_maxscore(1, 8192, 200*1024, 48, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(1, 131072, 131072, 48, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(1, 65536, 65536, 48, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(1, 8192, 8192, 48, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(1, 256, 256, 48, 8, 128, dtype, split, causal=True)
    print()

    # --- Short Q / Decode tests (TILE_Q=128 path, q_len <= 128) ---
    print("=== Short Q / Decode (TILE_Q=128) — max_score ===")
    for dtype in dtype_list:
        for split in split_list:
            all_pass &= _run_case_maxscore(1, 65, 512, 8, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(1, 64, 512, 8, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(32, 6, 512, 8, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(32, 6, 8192, 8, 8, 128, dtype, split, causal=True)
            all_pass &= _run_case_maxscore(4, 128, 2048, 8, 8, 128, dtype, split, causal=True)
    print()

    # --- Corner cases ---
    print("=== Corner Cases — max_score ===")
    for dtype in dtype_list:
        # TILE_Q boundary: 128 → TILE_Q=128, 129 → TILE_Q=256
        all_pass &= _run_case_maxscore(1, 127, 1024, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 128, 1024, 24, 4, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 129, 1024, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 255, 1024, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 257, 1024, 48, 8, 128, dtype, causal=True)

        # Pure decode: q_len=1
        all_pass &= _run_case_maxscore(1, 1, 512, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(32, 1, 8192, 48, 8, 128, dtype, causal=True)

        # Minimal sizes
        all_pass &= _run_case_maxscore(1, 1, 1, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 2, 2, 48, 8, 128, dtype, causal=True)

        # Odd sizes
        all_pass &= _run_case_maxscore(1, 7, 1024, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 33, 1024, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(3, 17, 513, 48, 8, 128, dtype, causal=True)

        # Single head (h_q=1, h_k=1)
        all_pass &= _run_case_maxscore(1, 256, 256, 1, 1, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(4, 128, 2048, 1, 1, 128, dtype, causal=True)

        # Extreme GQA ratio (h_q=64, h_k=1)
        all_pass &= _run_case_maxscore(1, 256, 256, 64, 1, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(1, 8192, 8192, 64, 8, 128, dtype, causal=True)

        # Large batch + small sequences
        all_pass &= _run_case_maxscore(64, 1, 128, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(128, 1, 64, 8, 8, 128, dtype, causal=True)

        # Long KV + short Q (append scenario)
        all_pass &= _run_case_maxscore(1, 1, 131072, 48, 8, 128, dtype, causal=True)
        all_pass &= _run_case_maxscore(4, 6, 65536, 48, 8, 128, dtype, causal=True)
        print()

    # --- Paged KV tests (small + boundary cases) ---
    print("=== Paged KV Cache — max_score ===")
    for dtype in dtype_list:
        for ps in page_sizes:
            all_pass &= _run_case_paged_maxscore(1, 256, 512, 8, 8, 128, dtype, page_size=ps)
            all_pass &= _run_case_paged_maxscore(4, 128, 2048, 48, 8, 128, dtype, page_size=ps)
            all_pass &= _run_case_maxscore(4, 128, 2048, 48, 8, 128, dtype)
            all_pass &= _run_case_paged_maxscore(1, 1, 512, 48, 8, 128, dtype, page_size=ps)
            all_pass &= _run_case_paged_maxscore(32, 1, 8192, 8, 8, 128, dtype, page_size=ps)
            all_pass &= _run_case_maxscore(32, 1, 8192, 8, 8, 128, dtype)
    print()

    # --- Paged Prefill (q == k, causal) ---
    print("=== Paged Prefill — max_score ===")
    for dtype in dtype_list:
        for page_size in page_sizes:
            for b, q, k, hq, hk in [(1, 65536, 65536, 64, 4),
                                    (1, 8192, 8192, 64, 4),
                                    (1, 256, 256, 64, 4)]:
                try:
                    all_pass &= _run_case_paged_maxscore(b, q, k, hq, hk, 128, dtype, page_size=page_size)
                except Exception as e:
                    case_name = f"paged b={b} q={q} k={k} h_q={hq} h_k={hk} ps={page_size} {dtype}"
                    print(f"  [EXCEPTION] {case_name}: {e}")
                    failed_cases.append(f"{case_name}: EXCEPTION {e}")
                    all_pass = False

    # --- Paged Decode (small q, large k) ---
    print("\n=== Paged Decode — max_score ===")
    paged_decode_cases = [(32, 24, 8192, 8, 8),
                          (32, 1, 8192, 48, 8),
                          (64, 24, 8192, 8, 8)]
    for dtype in dtype_list:
        for page_size in page_sizes:
            for b, q, k, hq, hk in paged_decode_cases:
                try:
                    all_pass &= _run_case_paged_maxscore(b, q, k, hq, hk, 128, dtype, page_size=page_size)
                except Exception as e:
                    case_name = f"paged b={b} q={q} k={k} h_q={hq} h_k={hk} ps={page_size} {dtype}"
                    print(f"  [EXCEPTION] {case_name}: {type(e).__name__}: {str(e)[:200]}")
                    failed_cases.append(f"{case_name}: EXCEPTION {type(e).__name__}")
                    all_pass = False
    print()

    # --- bf16 max_score output (OnlyScoreBF16 variant) ---
    print("=== bf16 max_score (OnlyScoreBF16) ===")
    bf16_ms = torch.bfloat16
    for dtype in dtype_list:
        # Prefill (TILE_Q=256) + tile boundary
        all_pass &= _run_case_maxscore(1, 8192, 8192, 48, 8, 128, dtype, causal=True,
                                       max_score_dtype=bf16_ms)
        all_pass &= _run_case_maxscore(1, 129, 1024, 48, 8, 128, dtype, causal=True,
                                       max_score_dtype=bf16_ms)
        # Short Q / decode (pack-GQA direct max_score path)
        all_pass &= _run_case_maxscore(32, 6, 512, 8, 8, 128, dtype, causal=True,
                                       max_score_dtype=bf16_ms)
        all_pass &= _run_case_maxscore(1, 1, 8192, 48, 8, 128, dtype, causal=True,
                                       max_score_dtype=bf16_ms)
        # Paged KV
        all_pass &= _run_case_paged_maxscore(4, 128, 2048, 48, 8, 128, dtype,
                                             page_size=128, max_score_dtype=bf16_ms)
    print()

    if not all_pass:
        print(f"\n=== FAILED CASES ({len(failed_cases)}) ===")
        for i, fc in enumerate(failed_cases):
            print(f"  {fc}")
            if i == 7 and len(failed_cases) > 8:
                print(f"  ..and {len(failed_cases) - 8} more")
                break
    else:
        print("ALL TESTS PASSED")

    import os
    os._exit(0 if all_pass else 1)


if __name__ == "__main__":
    if "--check" in sys.argv:
        fmha_ppu = partial(_fmha_ppu, check_input_valid=True)
    else:
        fmha_ppu = _fmha_ppu

    main()

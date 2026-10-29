#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: MIT
"""Reproduce / benchmark a single MSA FMHA case from a --format string.

The format string is the single-line log printed by the MSA FMHA kernel when
``PPU_LIB_SHOW_PARAMS=1`` is set, e.g.::

    [MSA] --format=dtype:bf16,batch_size:33,seqlen_q:[1,...],seqlen_k:[4096,...],\
qo_offset:[1141,...],num_qo_heads:4,num_kv_heads:4,head_dim_qk:128,head_dim_vo:128,\
num_kv_splits:1,is_paged:1,page_size:128,causal:1,sm_scale:0.088388.

Only the user-facing inputs of ``fmha_ppu`` / ``fmha_ppu_plan`` are carried
in the string; internally computed quantities (variant key, sparse_mode,
qo_tile_size, single_wg, pack_factor, max_qo_len, max_k_tiles) are re-derived by
the plan. Random Q/K/V data with the parsed shapes is used (the original tensor
contents are not recoverable from the log), which is sufficient to reproduce the
kernel path / timing for that configuration.

Usage::

    python run_msa.py --format "<the printed line>"
    python run_msa.py --caselist cases.txt [--case_idx N]
    python run_msa.py --format "..." --bench          # time + report MFU/MBU
    python run_msa.py --format "..." --ref_device cpu # allocate inputs on CPU

The kernel path (sparse_mode: Off/Full/OnlyScore/Sparse, i.e. whether max_score
and/or kv_block_indexes are used) is fully driven by the ``sparse_mode`` field in
the format string.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from fmha_ppu import fmha_ppu, fmha_ppu_plan  # noqa: E402


def _convert(value: str):
    """Convert a format-string value to list / int / float / str."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [int(x) for x in json.loads(value)]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_format(line: str) -> dict:
    """Parse '[MSA] --format=k:v,k:v.' into a config dict."""
    s = re.sub(r"^.*?--format=", "", line).strip().rstrip(".")
    # key:value pairs; value is either a [...] list or a comma-free scalar.
    pattern = r"(\w+):(\[.*?\]|[^,]*?)(?=,\w+:|$)"
    pairs = re.findall(pattern, s)
    return {k: _convert(v) for k, v in pairs}


# Hardware peak specs per platform: per-dtype compute peak (TFLOPs) and HBM
# bandwidth peak (GB/s). Selected automatically from the CUDA device name.
HW_PEAKS = {
    "890":  {"tflops": {"fp8": 830.6, "bf16": 415.3}, "hbm_gbs": 2664.0},
    "810e": {"tflops": {"fp8": 278.5, "bf16": 139.2}, "hbm_gbs": 2764.8},
    "b200": {"tflops": {"fp8": 4500.0, "bf16": 2250.0}, "hbm_gbs": 8000.0},
}


def detect_hw(device_name: str):
    """Detect the hardware platform from the CUDA device name.

    Mirrors run_flash_mla.py: recognizes PPU ('ppu'/'zw') and NVIDIA ('nvidia')
    devices and warns on unrecognized names, then maps the model number to a
    HW_PEAKS key.
    """
    n = device_name.lower()
    # Known device names: 890 -> 'ZW-M890P', 810E -> 'PPU-ZW810E'.
    if "890" in n:  # ZW-M890P
        return "890"
    if "810e" in n:  # PPU-ZW810E
        return "810e"
    if "b200" in n:  # NVIDIA B200
        return "b200"
    return None


def compute_bytes(cfg: dict, elem_bytes: int) -> int:
    """Total HBM traffic (bytes) for one attention pass: Q + K + V + O."""
    h_q = int(cfg["num_qo_heads"]); h_k = int(cfg["num_kv_heads"])
    d_qk = int(cfg["head_dim_qk"]); d_vo = int(cfg["head_dim_vo"])
    seqlen_q = list(cfg["seqlen_q"]); seqlen_k = list(cfg["seqlen_k"])
    total_qo = sum(seqlen_q)
    sparse_mode = str(cfg.get("sparse_mode", "Off"))
    kv_block_num = int(cfg.get("kv_block_num", 0))
    page_size = int(cfg.get("page_size", 128))

    q_bytes = total_qo * h_q * d_qk * elem_bytes
    o_bytes = total_qo * h_q * d_vo * 2  # O is always bf16
    if sparse_mode == "Sparse" and kv_block_num > 0:
        # Each (token, kv_head) gathers kv_block_num blocks of page_size tokens.
        kv_tokens = total_qo * kv_block_num * page_size
        kv_bytes = kv_tokens * h_k * (d_qk + d_vo) * elem_bytes
    else:
        total_kv = sum(seqlen_k)
        kv_bytes = total_kv * h_k * (d_qk + d_vo) * elem_bytes
    return q_bytes + kv_bytes + o_bytes


def compute_min_dram_bytes(cfg: dict, elem_bytes: int) -> int:
    """Minimum DRAM traffic accounting for intra-batch page reuse and GQA.

    In sparse decode, all tokens within the same batch share the same set of
    KV pages (kv_block_indexes is per-batch, broadcast to all tokens).  GQA
    qo_heads share kv_heads.  So unique KV data = batch * h_k * topk * page.
    """
    h_q = int(cfg["num_qo_heads"]); h_k = int(cfg["num_kv_heads"])
    d_qk = int(cfg["head_dim_qk"]); d_vo = int(cfg["head_dim_vo"])
    seqlen_q = list(cfg["seqlen_q"])
    batch_size = len(seqlen_q)
    total_qo = sum(seqlen_q)
    sparse_mode = str(cfg.get("sparse_mode", "Off"))
    kv_block_num = int(cfg.get("kv_block_num", 0))
    page_size = int(cfg.get("page_size", 128))

    q_bytes = total_qo * h_q * d_qk * elem_bytes
    o_bytes = total_qo * h_q * d_vo * 2
    if sparse_mode == "Sparse" and kv_block_num > 0:
        # Unique pages: each (batch, kv_head) has kv_block_num distinct pages.
        unique_pages = batch_size * h_k * kv_block_num
        kv_bytes = unique_pages * page_size * (d_qk + d_vo) * elem_bytes
    else:
        total_kv = sum(cfg["seqlen_k"])
        kv_bytes = total_kv * h_k * (d_qk + d_vo) * elem_bytes
    return q_bytes + kv_bytes + o_bytes


def compute_tflops(cfg: dict, mean_ms: float) -> float:
    """Achieved TFLOP/s; sparse uses accessed-kv (kv_block_num*page_size) FLOPs."""
    from fmha_ppu.bench_utils import attention_tflops
    h_q = int(cfg["num_qo_heads"])
    d_qk = int(cfg["head_dim_qk"]); d_vo = int(cfg["head_dim_vo"])
    seqlen_q = list(cfg["seqlen_q"]); seqlen_k = list(cfg["seqlen_k"])
    sparse_mode = str(cfg.get("sparse_mode", "Off"))
    kv_block_num = int(cfg.get("kv_block_num", 0))
    page_size = int(cfg.get("page_size", 128))
    causal = bool(int(cfg.get("causal", 1)))
    if sparse_mode == "Sparse" and kv_block_num > 0:
        eff_kv = kv_block_num * page_size
        total_qo = sum(seqlen_q)
        total_flops = total_qo * h_q * eff_kv * (d_qk + d_vo) * 2
        return total_flops / (mean_ms * 1e-3) / 1e12
    qo_lens = torch.tensor(seqlen_q, dtype=torch.int32)
    kv_lens = torch.tensor(seqlen_k, dtype=torch.int32)
    return attention_tflops(qo_lens, kv_lens, d_qk, d_vo, h_q, causal, mean_ms)


def build_case(cfg: dict, device: str = "cuda:0", ref_device: str = None):
    """Synthesize random inputs matching the parsed config and build a plan.

    Inputs (q/k/v/indices) are allocated on ``ref_device``; the plan workspaces
    live on ``device`` (the kernel device). When ``ref_device`` is "cpu", only
    the kernel call moves tensors to GPU (mirrors run_flash_mla.py ref_device).
    """
    if ref_device is None:
        ref_device = device
    dtype = torch.bfloat16 if cfg.get("dtype", "bf16") == "bf16" else torch.float8_e4m3fn
    init_dtype = torch.half if dtype.itemsize == 1 else dtype

    h_q = int(cfg["num_qo_heads"])
    h_k = int(cfg["num_kv_heads"])
    d_qk = int(cfg["head_dim_qk"])
    d_vo = int(cfg["head_dim_vo"])
    b = int(cfg["batch_size"])
    seqlen_q = list(cfg["seqlen_q"])
    seqlen_k = list(cfg["seqlen_k"])
    assert len(seqlen_q) == b and len(seqlen_k) == b, "seqlen list length != batch_size"

    is_paged = bool(int(cfg.get("is_paged", 0)))
    page_size = int(cfg.get("page_size", 128))
    causal = bool(int(cfg.get("causal", 1)))
    num_kv_splits = int(cfg.get("num_kv_splits", 1))
    sm_scale = float(cfg.get("sm_scale", 1.0 / math.sqrt(d_qk)))
    sparse_mode = str(cfg.get("sparse_mode", "Off"))
    kv_block_num = int(cfg.get("kv_block_num", 0))
    is_sparse = (sparse_mode == "Sparse") and kv_block_num > 0
    is_full = (sparse_mode == "Full")

    qo_lens = torch.tensor(seqlen_q, dtype=torch.int32)
    kv_lens = torch.tensor(seqlen_k, dtype=torch.int32)
    total_qo = int(sum(seqlen_q))

    if cfg.get("qo_offset"):
        qo_offset = torch.tensor(cfg["qo_offset"], dtype=torch.int32)
    else:
        qo_offset = (kv_lens - qo_lens).to(torch.int32) if causal else None

    torch.manual_seed(0)
    q = torch.randn(total_qo, h_q, d_qk, dtype=init_dtype, device=ref_device).to(dtype)

    kv_indices = None
    kv_block_indexes = None
    if is_paged:
        pages_per_seq = [max(1, (sk + page_size - 1) // page_size) for sk in seqlen_k]
        total_pages = int(sum(pages_per_seq))
        k = torch.randn(total_pages, h_k, page_size, d_vo, dtype=init_dtype, device=ref_device).to(dtype)
        v = torch.randn(total_pages, h_k, page_size, d_vo, dtype=init_dtype, device=ref_device).to(dtype)
        # Identity page table is enough to reproduce the kernel path/timing.
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=ref_device)
        if is_sparse:
            # kv_block_indexes: [total_qo, h_k, kv_block_num], per-batch block
            # indices sampled from [0, pages_per_seq[b]), -1 padded (mirrors
            # tests/regression/test_sparse_attn.py).
            import random
            random.seed(0)
            kv_block_indexes = torch.full(
                (total_qo, h_k, kv_block_num), -1, dtype=torch.int32, device=ref_device)
            q_pos = 0
            for bi in range(b):
                n = min(kv_block_num, pages_per_seq[bi])
                for h in range(h_k):
                    blocks = sorted(random.sample(range(pages_per_seq[bi]), n))
                    kv_block_indexes[q_pos:q_pos + seqlen_q[bi], h, :n] = \
                        torch.tensor(blocks, dtype=torch.int32, device=ref_device)
                q_pos += seqlen_q[bi]
    else:
        total_kv = int(sum(seqlen_k))
        k = torch.randn(total_kv, h_k, d_vo, dtype=init_dtype, device=ref_device).to(dtype)
        v = torch.randn(total_kv, h_k, d_vo, dtype=init_dtype, device=ref_device).to(dtype)

    # Plan workspaces live on the kernel device (cuda); segment_lens stay on CPU.
    plan_info = fmha_ppu_plan(
        qo_lens, kv_lens, h_q,
        num_kv_heads=h_k,
        qo_offset=qo_offset,
        num_kv_splits=num_kv_splits,
        page_size=page_size if is_paged else -1,
        kv_block_num=kv_block_num if is_sparse else -1,
        causal=causal,
        device=device,
    )
    return {
        "q": q, "k": k, "v": v, "plan_info": plan_info,
        "kv_indices": kv_indices, "kv_block_indexes": kv_block_indexes,
        "sm_scale": sm_scale, "qo_lens": qo_lens, "kv_lens": kv_lens,
        "h_q": h_q, "d_qk": d_qk, "d_vo": d_vo, "causal": causal,
        "is_sparse": is_sparse, "is_full": is_full, "sparse_mode": sparse_mode,
    }


def run_case(cfg: dict, bench: bool = False,
             device: str = "cuda:0", ref_device: str = None,
             hw: dict = None, hw_name: str = "?"):
    c = build_case(cfg, device, ref_device)
    output_maxscore = c["is_full"]

    def _to_dev(t):
        return t.to(device) if t is not None else None

    def fn():
        return fmha_ppu(
            _to_dev(c["q"]), _to_dev(c["k"]), _to_dev(c["v"]),
            plan_info=c["plan_info"],
            kv_indices=_to_dev(c["kv_indices"]),
            kv_block_indexes=_to_dev(c["kv_block_indexes"]),
            sm_scale=c["sm_scale"],
            output_o=True, output_maxscore=output_maxscore,
            check_input_valid=False,  # skip D2H sync in benchmark loop
        )

    o, ms = fn()
    torch.cuda.synchronize()
    ms_info = f", max_score.shape={tuple(ms.shape)}" if ms is not None else ""
    print(f"[run_msa] RAN OK [sparse_mode={c['sparse_mode']}]: "
          f"o.shape={tuple(o.shape)} dtype={o.dtype}{ms_info}", flush=True)

    if bench:
        import numpy as np
        from fmha_ppu.bench_utils import bench_gpu_time
        times = bench_gpu_time(fn)
        med = float(np.median(times))
        tflops = compute_tflops(cfg, med)
        elem_bytes = 1 if cfg.get("dtype", "bf16") == "fp8" else 2
        total_bytes = compute_bytes(cfg, elem_bytes)
        gbs = total_bytes / (med * 1e-3) / 1e9
        dtype_key = "fp8" if cfg.get("dtype", "bf16") == "fp8" else "bf16"
        peak_tflops = hw["tflops"].get(dtype_key, 0.0)
        peak_hbm = hw["hbm_gbs"]
        mfu = (tflops / peak_tflops) if peak_tflops > 0 else 0.0
        mbu = (gbs / peak_hbm) if peak_hbm > 0 else 0.0
        min_dram_bytes = compute_min_dram_bytes(cfg, elem_bytes)
        dram_gbs = min_dram_bytes / (med * 1e-3) / 1e9
        dram_pct = (dram_gbs / peak_hbm) if peak_hbm > 0 else 0.0
        print(f"[run_msa] median={med:.4f} ms  TFLOP/s={tflops:.2f}  MFU={mfu * 100:.2f}%  "
              f"GB/s={gbs:.2f}  MBU={mbu * 100:.2f}%  DRAM={dram_pct * 100:.2f}%  (hw={hw_name})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Reproduce/benchmark a single MSA case from a --format string")
    ap.add_argument("--format", type=str, default=None,
                    help="the printed '[MSA] --format=...' line")
    ap.add_argument("--caselist", type=str, default=None,
                    help="file with one --format line per case")
    ap.add_argument("--case_idx", type=int, default=None,
                    help="1-based index into --caselist to run a single case")
    ap.add_argument("--bench", action="store_true", help="time the case with bench_gpu_time")
    ap.add_argument("--ref_device", type=str, default=None,
                    help="where to allocate input tensors (e.g. cpu); only the "
                         "kernel call moves them to cuda. Default: cuda")
    args = ap.parse_args()

    try:
        dev_name = torch.cuda.get_device_name()
    except Exception:
        dev_name = ""
    hw_name = detect_hw(dev_name)
    if hw_name is None:
        print(f"[run_msa] WARN: no peak spec matched for '{dev_name}', "
              f"defaulting hw to '890'", flush=True)
        hw_name = "890"
    hw = HW_PEAKS[hw_name]

    cases = []
    if args.format:
        cases = [args.format]
    elif args.caselist:
        lines = [ln.strip() for ln in open(args.caselist)
                 if ln.strip() and "--format=" in ln]
        if not lines:
            print("no cases found in caselist", file=sys.stderr)
            sys.exit(1)
        cases = [lines[args.case_idx - 1]] if args.case_idx else lines
    else:
        ap.error("must give --format or --caselist")

    for line in cases:
        cfg = parse_format(line)
        print(f"[run_msa] case cfg: {cfg}", flush=True)
        run_case(cfg, bench=args.bench,
                 device="cuda:0", ref_device=args.ref_device,
                 hw=hw, hw_name=hw_name)


if __name__ == "__main__":
    main()

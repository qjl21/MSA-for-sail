#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Pre-compile all FMHA PPU kernel variants with a single ninja build.

Usage:
    python3 scripts/warmup_fmha_ppu.py          # Compile all with max parallelism
    python3 scripts/warmup_fmha_ppu.py -j 64    # Limit to 64 parallel compilations
    python3 scripts/warmup_fmha_ppu.py --clear   # Clear cache first, then compile all
"""

import argparse
import glob
import itertools
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from fmha_ppu.jit import _FMHA_PPU_DISPATCH, _FMHA_PPU_IMPOSSIBLE, _get_hgcc_flags

def enumerate_all_variants():

    dims = _FMHA_PPU_DISPATCH
    all_values = [values for _, values in dims]

    variants = []
    for combo in itertools.product(*all_values):
        params = {}
        for _, template_params in combo:
            params.update(template_params)
        if _FMHA_PPU_IMPOSSIBLE(params):
            continue
        indices = [values.index(val) for (_, values), val in zip(dims, combo)]
        variant_name = "_".join(str(i) for i in indices)
        variant = dict(params)
        variant["func_name"] = "fmha_ppu_" + variant_name
        variant["variant_name"] = variant_name
        variants.append(variant)

    return variants


def select_variants_for_preset(variants, preset: str):
    """Filter FMHA CUTLASS variants for image warmup presets.

    `all` preserves the historical full matrix.  `m3-infer` keeps the paged
    M3 inference paths used by dense FP8KV, sparse scoring, and sparse decode.
    `aime` is a narrower experiment preset for the current M3 sparse AIME shape.
    """
    if preset == "all":
        return variants

    def common_m3(v):
        if v["page_size"] != 128:
            return False
        if v["sparse_mode"] not in {"Off", "OnlyScore", "Sparse"}:
            return False
        if v["sparse_mode"] == "Sparse" and v["tile_q"] != "_128":
            return False
        return True

    if preset == "m3-infer":
        return [v for v in variants if common_m3(v)]

    if preset == "aime":
        return [
            v for v in variants
            if common_m3(v) and v["pack_factor"] in {1, 4, 16}
        ]

    raise ValueError(f"unknown preset: {preset}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compile all FMHA PPU kernel variants")
    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Parallel jobs for ninja (0 = auto)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear JIT cache before compiling")
    parser.add_argument("--preset", choices=["all", "m3-infer", "aime"], default="all",
                        help="FMHA CUTLASS variant preset (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List variants without compiling")
    args = parser.parse_args()

    _cleanup_compiler_temps()

    if args.all:
        args.clear = True
    from fmha_ppu.jit import (
        CACHE_BASE, _FMHA_VARLEN_DIR, _get_compiler, _get_link_libs,
    )
    import jinja2

    all_variants = enumerate_all_variants()
    variants = select_variants_for_preset(all_variants, args.preset)
    print(
        f"Preset: {args.preset} | FMHA variants: {len(variants)}/{len(all_variants)} "
        "+ plan + sparse_topk + reduction"
    )

    if args.dry_run:
        for v in variants:
            print(f"  {v['variant_name']}: dtype={v['dtype_in']} tile={v['tile_q']}x{v['tile_kv']} "
                  f"wg={v['single_wg']} sparse={v['sparse_mode']} page={v['page_size']} "
                  f"split={v['is_split_kv']} pack={v['pack_factor']}")
        return

    if args.clear:
        if CACHE_BASE.exists():
            shutil.rmtree(CACHE_BASE)
            print(f"Cleared cache: {CACHE_BASE}")

    # Filter already-cached variants
    to_compile = []
    cached = 0
    for v in variants:
        so_path = CACHE_BASE / v["variant_name"] / f"{v['variant_name']}.so"
        if so_path.exists():
            cached += 1
        else:
            to_compile.append(v)

    plan_so = CACHE_BASE / "plan" / "fmha_sm80_plan.so"
    need_plan = not plan_so.exists()
    sparse_topk_dir = CACHE_BASE / "sparse_topk"
    sparse_topk_so = sparse_topk_dir / "sparse_topk_select.so"
    need_sparse_topk = not sparse_topk_so.exists()
    for src in [
        _FMHA_VARLEN_DIR / "sparse_topk_select.cu",
        _FMHA_VARLEN_DIR / "include" / "sparse_topk_select.cuh",
        _FMHA_VARLEN_DIR / "tvm_ffi_utils.h",
    ]:
        dst = sparse_topk_dir / src.name
        if not dst.exists() or dst.read_text() != src.read_text():
            need_sparse_topk = True
    reduction_so = CACHE_BASE / "reduction" / "fmha_sm80_reduction.so"
    need_reduction = not reduction_so.exists()

    total = len(to_compile) + (1 if need_plan else 0) + (1 if need_sparse_topk else 0) + (1 if need_reduction else 0)
    print(f"Already cached: {cached}, to compile: {total}")
    if total == 0:
        print("All variants already compiled.")
        if include_sparse_aot:
            warmup_sparse_attn(clear=args.clear)
        return

    # ── Generate all .cu files ──
    build_dir = CACHE_BASE / "_warmup_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    with open(_FMHA_VARLEN_DIR / "fmha_sm80_inst.jinja") as f:
        inst_template = jinja2.Template(f.read())
    with open(_FMHA_VARLEN_DIR / "fmha_ppu_variant_run.cu.jinja") as f:
        run_template = jinja2.Template(f.read())

    # Copy shared headers
    for name in ["fmha_ppu_params.h", "msa_profiling_interface.hpp",
                 "tvm_ffi_utils.h", "gmem_bounds_check.h"]:
        shutil.copy2(_FMHA_VARLEN_DIR / name, build_dir / name)

    hgcc = _get_compiler()

    ninja_lines = [
        f"ninja_required_version = 1.5\n",
        f"hgcc = {hgcc}\n",
        f"hgcc_flags = {_get_hgcc_flags(build_dir)}\n",
        f"link_libs = {_get_link_libs()}\n",
        "rule hgcc_compile\n",
        "  command = $hgcc $hgcc_flags -c $in -o $out\n",
        "  description = Compiling $in\n\n",
        "rule hgcc_link\n",
        "  command = $hgcc -shared $in -o $out $link_libs\n",
        "  description = Linking $out\n\n",
        "rule hgcc_compile_other\n",
        f"  command = $hgcc {_get_hgcc_flags(build_dir, False)} -c $in -o $out\n",
        "  description = Compiling other kernel\n\n"
    ]

    # Generate variant .cu files and ninja rules → all link into one .so
    all_variant_objs = []
    for v in to_compile:
        vname = v["variant_name"]

        inst_cu = build_dir / f"inst_{vname}.cu"
        run_cu = build_dir / f"run_{vname}.cu"
        inst_cu.write_text(inst_template.render(**v))
        run_cu.write_text(run_template.render(**v))

        inst_o = build_dir / f"inst_{vname}.o"
        run_o = build_dir / f"run_{vname}.o"
        all_variant_objs.extend([inst_o, run_o])

        ninja_lines.append(f"build {inst_o}: hgcc_compile {inst_cu}\n")
        ninja_lines.append(f"build {run_o}: hgcc_compile {run_cu}\n")

    # Link all variants into one .so
    all_so_dir = CACHE_BASE / "_all_variants"
    all_so_dir.mkdir(parents=True, exist_ok=True)
    all_so_path = all_so_dir / "all_variants.so"
    if all_variant_objs:
        objs_str = " ".join(str(o) for o in all_variant_objs)
        ninja_lines.append(f"build {all_so_path}: hgcc_link {objs_str}\n\n")

    # Plan kernel
    if need_plan:
        plan_dir = CACHE_BASE / "plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_cu_src = _FMHA_VARLEN_DIR / "fmha_sm80_plan.cu"
        plan_cu = build_dir / "fmha_sm80_plan.cu"
        shutil.copy2(plan_cu_src, plan_cu)

        plan_o = build_dir / "fmha_sm80_plan.o"
        plan_so = plan_dir / "fmha_sm80_plan.so"

        ninja_lines.append(f"build {plan_o}: hgcc_compile_other {plan_cu}\n")
        ninja_lines.append(f"build {plan_so}: hgcc_link {plan_o}\n\n")

    # Sparse topk kernel
    if need_sparse_topk:
        topk_dir = CACHE_BASE / "sparse_topk"
        topk_dir.mkdir(parents=True, exist_ok=True)
        topk_cu_src = _FMHA_VARLEN_DIR / "sparse_topk_select.cu"
        if topk_cu_src.exists():
            topk_cu = build_dir / "sparse_topk_select.cu"
            shutil.copy2(topk_cu_src, topk_cu)
            shutil.copy2(topk_cu_src, topk_dir / "sparse_topk_select.cu")
            shutil.copy2(_FMHA_VARLEN_DIR / "include" / "sparse_topk_select.cuh",
                         topk_dir / "sparse_topk_select.cuh")
            shutil.copy2(_FMHA_VARLEN_DIR / "tvm_ffi_utils.h",
                         topk_dir / "tvm_ffi_utils.h")
            topk_o = build_dir / "sparse_topk_select.o"
            topk_so = topk_dir / "sparse_topk_select.so"
            ninja_lines.append(f"build {topk_o}: hgcc_compile_other {topk_cu}\n")
            ninja_lines.append(f"build {topk_so}: hgcc_link {topk_o}\n\n")

    # Split-KV reduction kernel
    if need_reduction:
        red_dir = CACHE_BASE / "reduction"
        red_dir.mkdir(parents=True, exist_ok=True)
        red_cu_src = _FMHA_VARLEN_DIR / "fmha_sm80_reduction.cu"
        if red_cu_src.exists():
            red_cu = build_dir / "fmha_sm80_reduction.cu"
            shutil.copy2(red_cu_src, red_cu)
            red_o = build_dir / "fmha_sm80_reduction.o"
            red_so = red_dir / "fmha_sm80_reduction.so"
            ninja_lines.append(f"build {red_o}: hgcc_compile_other {red_cu}\n")
            ninja_lines.append(f"build {red_so}: hgcc_link {red_o}\n\n")

    (build_dir / "build.ninja").write_text("".join(ninja_lines))

    # ── Run single ninja build ──
    jobs_flag = f"-j{args.jobs}" if args.jobs > 0 else ""
    cmd = f"ninja {jobs_flag}".strip()
    print(f"Running: {cmd} (in {build_dir})", flush=True)
    print(f"Compiling {total} kernels...", flush=True)

    start = time.time()
    result = subprocess.run(
        cmd.split(),
        cwd=str(build_dir),
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\nDone in {elapsed:.1f}s: {total} compiled, {cached} cached")
    else:
        print(f"\nBuild failed after {elapsed:.1f}s (exit code {result.returncode})")
        sys.exit(1)

    _cleanup_compiler_temps()


def _cleanup_compiler_temps():
    temps = glob.glob("/tmp/tmpxft_*")
    if temps:
        total = sum(os.path.getsize(f) for f in temps if os.path.isfile(f))
        for f in temps:
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
            except OSError:
                pass
        print(f"Cleaned {len(temps)} compiler temp files ({total / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()

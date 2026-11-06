// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "cutlass/cutlass.h"
#include "fmha_common.hpp"

namespace cutlass::fmha::kernel {

// SM80 reduction kernel for split-KV FMHA.
// Merges partial bf16 O using float32 LSE (log-sum-exp) from multiple KV splits.
// Identical algorithm to Sm100FmhaReductionKernel — the merge math is
// architecture-independent (no MMA unit or TMA usage).
//
// Grid:  (num_qo_heads, total_qo_len, 1)
// Block: 128 threads; each thread handles one d element.
//
// Structure mirrors Triton's _merge_topk_attn_out_kernel (measured 4us vs our
// 11-13us on [D][1,2000] h64/kv4):
//  - Weights are computed cooperatively per warp with butterfly shuffles: lane
//    c owns split c + 32*j, so exp2f runs 32 times per warp instead of once per
//    thread. No __syncthreads, no shared memory, no per-thread weight arrays.
//    Pass 2 gathers weights with warp-uniform __shfl_sync broadcasts.
//  - 128 threads (4 warps) is deliberate: with 64 (row,head) blocks the kernel
//    is latency-bound and warp count is what hides memory latency. A 64-thread
//    2-elements-per-thread variant measured SLOWER (13us) despite issuing fewer
//    total instructions (73K vs this version's ~90K) — halving the block cut
//    the resident warps from 256 to 128 and lengthened the per-warp critical
//    path. 1 d/thread with scalar bf16 loads keeps every warp saturated.
//  - Pass 2 reads weights from a warp-private shared table (v5a): the
//    butterfly-computed weights are staged to smem once (__syncwarp only), and
//    every lane reads the same address per split — no runtime-indexed
//    __shfl_sync gather (v.shuffle.idx.i -> 0) and no cross-warp barrier.
// Also measured slower on PPU (kept for the record): shared-memory two-pass
// weights with four barriers; warp-0 butterfly with a single barrier (warp 0
// became the serial critical path); 64-thread blocks with paired 32-bit loads
// (v3 — instruction count beat Triton but warp count halved).
template <class ElementPartial,  // bf16 (workspace O type)
          class ElementOut,      // output dtype (fp8 or bf16)
          int kMaxSplits = 128>
struct Sm80FmhaReductionKernel {
  static const int MaxThreadsPerBlock = 128;

  struct Params {
    const ElementPartial* ptr_O_partial;  // [total_qo_len * num_splits, num_heads, head_dim]
    ElementOut* ptr_O;                    // [total_qo_len, num_heads, head_dim]
    const float* ptr_lse;                // [num_splits, total_qo_len, num_heads]
    const int* num_kv_splits_per_row;     // [total_qo_len]: actual split count per row
    float scale_softmax_log2;
    float inv_scale_o;
    int num_kv_splits;
    int total_qo_len;
    int num_qo_heads;
    int head_dim_vo;
    int stride_o_n;       // output: token stride
    int stride_o_h;       // output: head stride
    int stride_partial_n; // partial: token stride
    int stride_partial_h; // partial: head stride
    // Fused unpack-GQA: when ptr_O_direct != nullptr, write to unpacked layout
    // [nnz_qo, num_qo_heads_orig, head_dim_vo] instead of packed ptr_O.
    ElementOut* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int num_kv_heads = 0;
    int pack_factor = 1;
  };

  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.num_qo_heads, params.total_qo_len, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* /* smem */) {
    int head_idx = blockIdx.x;
    int abs_row = blockIdx.y;
    int d = threadIdx.x;
    int lane = d & 31;
    int warp = d >> 5;
    int warp_d = 32 * warp;

    // Block-uniform guard.
    if (abs_row >= params.total_qo_len) return;

    int num_splits = warp_uniform(params.num_kv_splits_per_row[abs_row]);
    // Use 64-bit offset: total_qo_len * num_splits * num_heads * head_dim can exceed INT32_MAX
    int64_t partial_off = (int64_t)abs_row * params.stride_partial_n +
                          (int64_t)head_idx * params.stride_partial_h + d;

    // Decide output base + offset. Fused unpack remaps packed (abs_row, head_idx)
    // to unpacked (actual_tok, unpacked_head); plain path uses packed strides.
    ElementOut* o_ptr_used;
    int64_t dst_off;
    if (params.ptr_O_direct != nullptr) {
      int pf            = params.pack_factor;
      int rem_hr        = params.num_qo_heads / params.num_kv_heads;  // packed h_r
      int kv_head       = head_idx / rem_hr;
      int rhr_idx       = head_idx % rem_hr;
      int h_r_orig      = rem_hr * pf;
      int pf_idx        = abs_row % pf;
      int actual_tok    = abs_row / pf;
      int unpacked_head = kv_head * h_r_orig + rhr_idx * pf + pf_idx;
      dst_off = (int64_t)actual_tok * params.num_qo_heads_orig * params.head_dim_vo
              + (int64_t)unpacked_head * params.head_dim_vo
              + d;
      o_ptr_used = params.ptr_O_direct;
    } else {
      dst_off = (int64_t)abs_row * params.stride_o_n
              + (int64_t)head_idx * params.stride_o_h
              + d;
      o_ptr_used = params.ptr_O;
    }

    // Fast path: single split — just scale and copy, no merge needed
    if (num_splits == 1) {
      if (d >= params.head_dim_vo) return;  // no shfl on this path -> safe
      int64_t stats_base = warp_uniform((int64_t)abs_row * params.num_qo_heads + head_idx);
      float lse = __ldg(&params.ptr_lse[stats_base]);
      if (lse == -INFINITY) {
        o_ptr_used[dst_off] = ElementOut(0.f);
      } else {
        float o_s = static_cast<float>(params.ptr_O_partial[partial_off]);
        o_ptr_used[dst_off] = ElementOut(o_s * params.inv_scale_o);
      }
      return;
    }

    // Whole warps beyond head_dim_vo have no work; exiting here keeps every
    // remaining warp fully populated for the __shfl_sync sections below
    // (a partially-populated warp would break the full-mask shuffles).
    if (warp_d >= params.head_dim_vo) return;

    // ---- Merge weights: butterfly shuffle per warp (no smem, no barriers) --
    int64_t stats_base   = warp_uniform((int64_t)abs_row * params.num_qo_heads + head_idx);
    int64_t stats_stride = warp_uniform((int64_t)params.total_qo_len * params.num_qo_heads);

    constexpr int kGroups = (kMaxSplits + 31) / 32;  // splits per lane: ceil(128/32)
    float w_reg[kGroups];
    float inv_sum;
    {
      float lse_reg[kGroups];
      #pragma unroll
      for (int j = 0; j < kGroups; ++j) {
        int s = lane + 32 * j;
        lse_reg[j] = (s < num_splits)
            ? __ldg(&params.ptr_lse[stats_base + (int64_t)s * stats_stride])
            : -INFINITY;
      }
      float mx = lse_reg[0];
      #pragma unroll
      for (int j = 1; j < kGroups; ++j) mx = fmaxf(mx, lse_reg[j]);
      #pragma unroll
      for (int off = 16; off >= 1; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
      float sw = 0.f;
      #pragma unroll
      for (int j = 0; j < kGroups; ++j) {
        w_reg[j] = (lse_reg[j] == -INFINITY) ? 0.f
             : exp2f(params.scale_softmax_log2 * (lse_reg[j] - mx));
        sw += w_reg[j];
      }
      #pragma unroll
      for (int off = 16; off >= 1; off >>= 1)
        sw += __shfl_xor_sync(0xffffffffu, sw, off);
      inv_sum = (sw > 0.f) ? 1.f / sw : 0.f;
    }

    // v5a: stage the lane-distributed weights into a warp-private shared table
    // so pass 2 reads them as plain same-address broadcast loads. This removes
    // the per-split __shfl_sync gather of v4, whose runtime lane index compiled
    // to v.shuffle.idx.i sitting on the fma dependency chain (measured 5376 on
    // [D][1,2000]; Triton's merge has none — it broadcasts via tsm.st/ld).
    __shared__ float s_w[MaxThreadsPerBlock / 32][kMaxSplits];
    #pragma unroll
    for (int j = 0; j < kGroups; ++j)
      s_w[warp][lane + 32 * j] = w_reg[j];
    __syncwarp();

    // ---- Per-d weighted sum --------------------------------------------
    // Indexed addressing with a partial unroll keeps several O_partial loads
    // in flight; the loop runs to the warp-uniform num_splits bound so no
    // address guards are needed. plan's num_kv_splits_per_row can exceed the
    // splits the mainloop actually wrote (adaptive split assignment + phase3
    // max-update across heads); those slots keep lse == -INFINITY so w_s == 0,
    // but their O_partial bytes are stale workspace content, and fmaf(0,
    // stale, acc) is only benign when stale is finite (-INF lse bytes from a
    // previous case alias to bf16 -Inf). So the VALUE is sanitized after an
    // unconditional load — matching the pre-v5a guard
    //   o_s = (lse_s != -INFINITY) ? o_s : 0.f;
    // Never predicate the load itself: that serialized the unroll-4 loads on
    // PPU and cost 3x kernel time (5us -> 15us).
    int64_t partial_stride = (int64_t)params.total_qo_len * params.stride_partial_n;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < kGroups; ++j) {
      if (32 * j < num_splits) {                     // warp-uniform
        int n_here = min(32, num_splits - 32 * j);   // warp-uniform
        #pragma unroll 4
        for (int c = 0; c < n_here; ++c) {
          // Broadcast weight for split 32*j + c via the warp-private table:
          // every lane reads the same address (shared-memory broadcast, no
          // bank conflict) instead of a runtime-indexed shuffle gather.
          float w_s = s_w[warp][32 * j + c];
          // Unconditional load keeps all unroll-4 loads in flight (MLP);
          // neutralize stale bytes AFTER the load, like the pre-v5a guard.
          float o_s = static_cast<float>(
              params.ptr_O_partial[partial_off + (int64_t)(32 * j + c) * partial_stride]);
          o_s = (w_s != 0.f) ? o_s : 0.f;
          acc = fmaf(w_s, o_s, acc);
        }
      }
    }
    if (d >= params.head_dim_vo) return;  // partial-warp tail: no more shfl below
    float scale_out = (inv_sum == 0.f) ? 0.f : inv_sum * params.inv_scale_o;
    o_ptr_used[dst_off] = ElementOut(acc * scale_out);
  }
};

}  // namespace cutlass::fmha::kernel

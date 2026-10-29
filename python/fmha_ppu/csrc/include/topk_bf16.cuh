// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

// =============================================================================
// topk_bf16.cuh
//
// BF16 sparse top-k selection for the MSA indexer: a register-path
// algorithm operating on the transposed buffer, plus a warp-0
// ascending-sort output tail.
//
// Contracts (enforced by the host dispatcher):
//   * 16 < max_k_tiles < kMaxTies
//   * TOPK == 16
//   * |logit| <= 65504 (bin extraction goes through fp16 and saturates)
// =============================================================================

#pragma once

#include <cfloat>
#include <cstdint>
#include <hggc_bf16.h>
#include <hggc_fp16.h>

namespace flashinfer {
namespace sparse_topk {
namespace topk_bf16 {
// ============================================================
// Register-path top-K kernel (topk_bf16 namespace).
// Single-pass design over the transposed buffer:
//   - rows < kMaxTies; host-enforced by the top-level dispatcher
//   - the host guarantees max_k_tiles > topk
//   - TOPK is a template parameter (dispatch fixes it to 16)
//   - kMaxTies covers the worst-case threshold bin (K <= 12287);
//     the union is tie_buffer-dominated (96KB vs 16KB histogram)
// ============================================================
constexpr uint32_t kBlockSize = 1024;
constexpr uint32_t kNumWarps = kBlockSize / 32;

constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistBins = 1u << kHistBits;  // 4096
constexpr uint32_t kMaxTies = 12288;          // >= worst-case threshold-bin occupancy (K <= 12287)
constexpr uint32_t kVecsPerThread = 4;
constexpr uint32_t kElemsPerVec = 4;  // elements per Vec4 (one float4 per thread per v-iter)

// --- Minimal shims: only what this kernel references. ---

template <typename T, uint32_t N>
struct alignas(sizeof(T) * N) AlignedVector {
  T data[N];
  __device__ __forceinline__ T& operator[](uint32_t i) { return data[i]; }
  __device__ __forceinline__ const T& operator[](uint32_t i) const { return data[i]; }
  __device__ __forceinline__ void fill(T value) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) data[i] = value;
  }
};

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
  }
  return val;
}

template <uint32_t kBits>
__device__ __forceinline__ uint32_t extract_coarse_bin(float x) {
  // Order-preserving fp32->fp16->uint16 mapping, then top-kBits as bin idx.
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

__device__ __forceinline__ uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

// ============================================================
// Lightweight POD helpers.
// ============================================================
struct alignas(16) MatchBinV3 {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) TieV3 {
  uint32_t idx;
  float score;
};

// (score desc, idx asc) total order over ties; NaN fails both score
// comparisons and falls through to the deterministic idx tiebreaker.
__device__ __forceinline__ bool tie_greater(const TieV3& a, const TieV3& b) {
  if (a.score > b.score) return true;
  if (a.score < b.score) return false;
  return a.idx < b.idx;
}

// Register-path shared memory layout (~96.6 KB, single-pass;
// no staging buffer).  Requires cudaFuncSetAttribute opt-in at
// the launch point (> 48KB default dynamic smem limit).
//   counter_gt / counter_eq : 256B
//   match + warp_sum        : 16B + 128B
//   union { histogram[4096]   -> 16KB
//           histogram_vec     -> same storage as histogram
//           tie_buffer[kMaxTies] -> 96KB }   // union size driven by tie_buffer
// s_topk_indices (TOPK * int32) lives at the tail of dynamic smem.
using Vec4 = AlignedVector<float, kElemsPerVec>;

struct RegisterTopKSmem {
  using HistVec = AlignedVector<uint32_t, kHistBins / kBlockSize>;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBinV3 match;
  uint32_t warp_sum[kNumWarps];
  TieV3 warp_best[kNumWarps];         // Phase-4 fallback: per-warp argmax scratch
  uint32_t warp_best_slot[kNumWarps];
  union {
    uint32_t histogram[kHistBins];
    HistVec histogram_vec[kBlockSize];
    TieV3 tie_buffer[kMaxTies];
  };
};

// ============================================================
// Vectorized bf16 score loader: reads kVecsPerThread*4 = 16 bf16
// scores per thread into a Vec4[] register array.  Uses 8B (uint2)
// loads + batched __bfloat1622float2 conversion when the row is
// 8-byte aligned; falls back to per-element scalar load otherwise.
//
// Alignment contract: scores + base is 8-byte aligned when the row
// start (scores) is 8B aligned AND base is a multiple of 4 (always
// true since base = (tx + v*kBlockSize)*4).  Transpose buffers from
// cudaMalloc are >=256B aligned, so rows with max_k_tiles % 4 == 0
// take the fast path; odd max_k_tiles (e.g. edge_unaligned K=65)
// fall back to scalar.
// ============================================================
__device__ __forceinline__ void load_scores_vec(
    const __ppu_bfloat16* __restrict__ scores,
    uint32_t tx, uint32_t length,
    Vec4 local[kVecsPerThread]) {
  const bool row_aligned8 = (reinterpret_cast<uintptr_t>(scores) & 7u) == 0;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    const uint32_t base = (tx + v * kBlockSize) * kElemsPerVec;
    if (base >= length) break;
    if (row_aligned8 && base + 3 < length) {
      // Fast path: one 8B load yields 4 bf16, then 2x bf16x2->float2.
      const uint2 raw = *reinterpret_cast<const uint2*>(scores + base);
      const __ppu_bfloat162 lo = *reinterpret_cast<const __ppu_bfloat162*>(&raw.x);
      const __ppu_bfloat162 hi = *reinterpret_cast<const __ppu_bfloat162*>(&raw.y);
      const float2 f0 = __bfloat1622float2(lo);
      const float2 f1 = __bfloat1622float2(hi);
      local[v][0] = f0.x; local[v][1] = f0.y;
      local[v][2] = f1.x; local[v][3] = f1.y;
    } else {
      // Fallback: per-element scalar load (misaligned row or tail).
#pragma unroll
      for (uint32_t e = 0; e < kElemsPerVec; ++e) {
        const uint32_t idx = base + e;
        local[v][e] = (idx < length) ? static_cast<float>(scores[idx]) : 0.0f;
      }
    }
  }
}


// Phase-1/3 shared element walk over (v, e, idx, valid, bin); the skip is
// warp-uniform: callback may use warp collectives, never early-out per lane.
template <typename F>
__device__ __forceinline__ void for_each_lane_element(
    const Vec4 (&local)[kVecsPerThread], uint32_t length, uint32_t tx, uint32_t warp_id, F&& f) {
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    if ((warp_id * 32 + v * kBlockSize) * kElemsPerVec >= length) continue;
#pragma unroll
    for (uint32_t e = 0; e < kElemsPerVec; ++e) {
      const uint32_t idx = (tx + v * kBlockSize) * kElemsPerVec + e;
      const bool valid = idx < length;
      const uint32_t bin =
          valid ? extract_coarse_bin<kHistBits>(local[v][e]) : 0xFFFFFFFFu;
      f(v, e, idx, valid, bin);
    }
  }
}


// Phase-4 uniform fast path: if every tie shares one score (NaN fails ==
// and safely declines), ties are already idx-ordered — fill the remaining
// topk slots directly and skip ranking.  Returns true when taken.
__device__ __forceinline__ bool try_uniform_take(
    const Vec4 (&local)[kVecsPerThread], uint32_t length, uint32_t thr_bin,
    uint32_t num_ties, uint32_t num_above, uint32_t topk_remain,
    int32_t* topk_indices, RegisterTopKSmem* smem) {
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;
  const auto tie_buffer = smem->tie_buffer;
  const float first_score = tie_buffer[0].score;
  bool uniform = true;
  for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
    // NaN fails == and safely falls back to the mixed-score fallback.
    if (tie_buffer[i].score != first_score) {
      uniform = false;
      break;
    }
  }
  if (!__syncthreads_and(uniform)) return false;
  uint32_t seq_base = 0;
#pragma unroll
  for (uint32_t v = 0; v < kVecsPerThread; ++v) {
    uint32_t hit_mask = 0;
#pragma unroll
    for (uint32_t e = 0; e < kElemsPerVec; ++e) {
      const uint32_t idx = (tx + v * kBlockSize) * kElemsPerVec + e;
      if (idx < length && extract_coarse_bin<kHistBits>(local[v][e]) == thr_bin) {
        hit_mask |= 1u << e;
      }
    }
    const uint32_t cnt = __popc(hit_mask);
    const auto warp_inc = warp_inclusive_sum(lane_id, cnt);
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }
    __syncthreads();
    const auto tmp = smem->warp_sum[lane_id];
    const uint32_t warp_prefix = warp_reduce_sum(lane_id < warp_id ? tmp : 0u);
    const uint32_t block_total = warp_reduce_sum(tmp);
    uint32_t seq = seq_base + warp_prefix + warp_inc - cnt;
#pragma unroll
    for (uint32_t e = 0; e < kElemsPerVec; ++e) {
      if (hit_mask & (1u << e)) {
        if (seq < topk_remain) {
          topk_indices[num_above + seq] = (tx + v * kBlockSize) * kElemsPerVec + e;
        }
        ++seq;
      }
    }
    seq_base += block_total;
    __syncthreads();
  }
  return true;
}


// Mixed-score fallback: topk_remain rounds of block-wide argmax in
// (score desc, idx asc) order; winner slots are consumed via the sentinel
// (-inf, ~0u), which ranks below every real tie.  O(topk_remain * num_ties).
__device__ __forceinline__ void argmax_select_fallback(
    uint32_t num_ties, uint32_t num_above, uint32_t topk_remain,
    int32_t* topk_indices, RegisterTopKSmem* smem) {
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;
  const auto tie_buffer = smem->tie_buffer;
  const auto invalid =
      TieV3{.idx = 0xFFFFFFFFu, .score = __int_as_float(0xff800000u)};  // -inf
  for (uint32_t sel = 0; sel < topk_remain; ++sel) {
    auto best = invalid;
    uint32_t best_slot = 0xFFFFFFFFu;
    for (uint32_t i = tx; i < num_ties; i += kBlockSize) {
      const auto cand = tie_buffer[i];
      if (tie_greater(cand, best)) {
        best = cand;
        best_slot = i;
      }
    }
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      TieV3 other;
      other.score = __shfl_xor_sync(0xFFFFFFFFu, best.score, offset);
      other.idx = __shfl_xor_sync(0xFFFFFFFFu, best.idx, offset);
      const uint32_t other_slot = __shfl_xor_sync(0xFFFFFFFFu, best_slot, offset);
      if (tie_greater(other, best)) {
        best = other;
        best_slot = other_slot;
      }
    }
    if (lane_id == 0) {
      smem->warp_best[warp_id] = best;
      smem->warp_best_slot[warp_id] = best_slot;
    }
    __syncthreads();
    if (warp_id == 0) {
      auto wbest = smem->warp_best[lane_id];
      uint32_t wslot = smem->warp_best_slot[lane_id];
#pragma unroll
      for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        TieV3 other;
        other.score = __shfl_xor_sync(0xFFFFFFFFu, wbest.score, offset);
        other.idx = __shfl_xor_sync(0xFFFFFFFFu, wbest.idx, offset);
        const uint32_t other_slot = __shfl_xor_sync(0xFFFFFFFFu, wslot, offset);
        if (tie_greater(other, wbest)) {
          wbest = other;
          wslot = other_slot;
        }
      }
      if (lane_id == 0) {
        topk_indices[num_above + sel] = wbest.idx;
        tie_buffer[wslot] = invalid;  // consume the winner
      }
    }
    __syncthreads();
  }
}


// ============================================================
// register_topk_bf16 (single-pass; MSA rows are < kMaxTies).
//
// Loads the entire row (<= 16K bf16 scores) into per-thread
// registers, then runs the coarse 12-bit histogram + tie-breaking
// pipeline.
//
// `_smem` is reinterpret_cast<RegisterTopKSmem*>; the caller is
// responsible for sizing the dynamic smem accordingly.
// Requires length > TOPK (the host guarantees max_k_tiles > topk),
// so every one of the TOPK output slots is written exactly once.
// ============================================================
template <uint32_t TOPK>
__device__ __forceinline__ void register_topk_bf16 (
  const __ppu_bfloat16* __restrict__ scores, int32_t* __restrict__ indices, const uint32_t length, void* _smem) {
  auto* smem = static_cast<RegisterTopKSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  // Initialize shared memory histogram
  {
    typename RegisterTopKSmem::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    if (tx == 0) {
      smem->counter_gt = smem->counter_eq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers (bf16 -> float cast per element)
  Vec4 local[kVecsPerThread];
  load_scores_vec(scores, tx, length, local);

  // Warp-aggregated histogram: a whole warp landing in one bin (uniform
  // -inf sea) collapses 32 same-address atomics into one leader +32.
  // No per-lane early-out: shfl/__all_sync need a fully converged warp.
  for_each_lane_element(local, length, tx, warp_id,
                        [&](uint32_t /*v*/, uint32_t /*e*/, uint32_t idx, bool valid, uint32_t bin) {
    const uint32_t bin0 = __shfl_sync(0xFFFFFFFFu, bin, 0);
    if (__all_sync(0xFFFFFFFFu, valid && bin == bin0)) {
      if (lane_id == 0) atomicAdd(&smem->histogram[bin0], 32u);
    } else if (valid) {
      atomicAdd(&smem->histogram[bin], 1u);
    }
  });
  __syncthreads();

// Phase 2: Exclusive prefix scan -> find threshold bin
{
    constexpr uint32_t kItems = kHistBins / kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = warp_reduce_sum(lane_id < warp_id ? tmp : 0u);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = length - prefix_sum;
      if (above < TOPK && above + orig[i] >= TOPK) {
        smem->match = {
            .bin = tx * kItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
}

  const auto [thr_bin, num_above, num_equal] = smem->match;

  // Phase 3: Scatter
  const bool need_tiebreak = num_equal + num_above > TOPK;
  const auto topk_indices = indices;
  const auto tie_buffer = smem->tie_buffer;

  // Warp-aggregated counters: one atomic per warp per class (gt/eq), same
  // uniform-sea motivation as Phase-1; slot order is unobservable here.
  for_each_lane_element(local, length, tx, warp_id,
                        [&](uint32_t v, uint32_t e, uint32_t idx, bool valid, uint32_t bin) {
    const bool is_gt = valid && bin > thr_bin;
    const bool is_eq = valid && bin == thr_bin;
    const uint32_t gt_mask = __ballot_sync(0xFFFFFFFFu, is_gt);
    const uint32_t eq_mask = __ballot_sync(0xFFFFFFFFu, is_eq);
    uint32_t gt_base = 0, eq_base = 0;
    if (lane_id == 0) {
      if (gt_mask != 0u) gt_base = atomicAdd(&smem->counter_gt, __popc(gt_mask));
      if (eq_mask != 0u) eq_base = atomicAdd(&smem->counter_eq, __popc(eq_mask));
    }
    gt_base = __shfl_sync(0xFFFFFFFFu, gt_base, 0);
    eq_base = __shfl_sync(0xFFFFFFFFu, eq_base, 0);
    if (is_gt) {
      topk_indices[gt_base + __popc(gt_mask & ((1u << lane_id) - 1u))] = idx;
    } else if (is_eq) {
      const auto pos = eq_base + __popc(eq_mask & ((1u << lane_id) - 1u));
      if (need_tiebreak) {
        if (pos < kMaxTies) {
          tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
        }
      } else {
        if (const auto which = pos + num_above; which < TOPK) {
          topk_indices[which] = idx;
        }
      }
    }
  });

  if (!need_tiebreak) return;

  // Phase 4: Tie-breaking within the threshold bin
  __syncthreads();

  const uint32_t num_ties = min(num_equal, kMaxTies);
  const uint32_t topk_remain = TOPK - num_above;

  if (num_ties <= 32u) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target_tie = tie_buffer[warp_id];
    const bool pred = tie_greater(tie, target_tie);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      topk_indices[num_above + rank] = target_tie.idx;
    }
  } else if (num_ties <= 64u) {
    const auto lane_id_1 = lane_id + 32;
    const auto warp_id_1 = warp_id + 32;
    // The sentinel must rank strictly below every real tie, including -inf
    // scores: score=-inf + idx=0xFFFFFFFF loses both comparison keys, so
    // tie_greater(x, invalid) holds for every real tie x.  (-FLT_MAX would
    // still beat -inf ties and corrupt the rank.)
    const auto invalid = TieV3{.idx = 0xFFFFFFFF, .score = __int_as_float(0xff800000u)};  // -inf
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;

    // Rank a target tie by counting how many of the 64 comparison sources
    // (tie_0 across 32 lanes + tie_1 across 32 lanes) are strictly greater.
    auto rank_target = [&](uint32_t target_idx) {
      const auto target = tie_buffer[target_idx];
      const bool pred_0 = tie_greater(tie_0, target);
      const bool pred_1 = tie_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    };

    // Pass 1: targets [0, 32). Invariant: warp_id < 32 <= num_ties in this branch.
    rank_target(warp_id);
    // Pass 2: targets [32, num_ties). Guarded for num_ties < 64.
    if (warp_id_1 < num_ties) rank_target(warp_id_1);
  } else {
    [[unlikely]];
    if (try_uniform_take(local, length, thr_bin, num_ties, num_above, topk_remain,
                         topk_indices, smem)) {
      return;
    }
    argmax_select_fallback(num_ties, num_above, topk_remain, topk_indices, smem);
  }

}

// =============================================================================
// Warp-only ascending sort (v2.1 fix replacing the heavy cub::BlockRadixSort)
// =============================================================================
//
// v2.0 used cub::BlockRadixSort<uint32_t, 512, 1> over all 512 threads to
// sort the topk indices ascending — but only ≤64 of the 512 keys are real
// (rest are sentinel ~0u), so 92%+ of the sort work was wasted, and nsys
// measured this as ~13 us per kernel call (largest single component of the
// IndexerTopK kernel duration).
//
// v2.1 replaces it with a warp-only bitonic sort (single warp, 32 lanes ×
// {1, 2} keys depending on MAX_TOPK), modelled on v1.6.5's WarpBitonicSortAsc16.
// Expected cost: < 1 us, savings ~12 us per kernel call.

// 32-element ascending sort, 32 lanes × 1 key each (for MAX_TOPK ≤ 32).
// Lanes with no real data should pass key = ~0u sentinel, which sorts to the end.
__device__ __forceinline__ void WarpBitonicSortAsc32(uint32_t& key, uint32_t lane) {
  constexpr uint32_t MASK = 0xFFFFFFFFu;
#pragma unroll
  for (int k = 2; k <= 32; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      uint32_t partner = __shfl_xor_sync(MASK, key, j);
      const bool asc_pair = ((lane & k) == 0);
      const bool I_am_lower = ((lane & j) == 0);
      if (I_am_lower) {
        key = asc_pair ? min(key, partner) : max(key, partner);
      } else {
        key = asc_pair ? max(key, partner) : min(key, partner);
      }
    }
  }
}

// 64-element ascending sort, 32 lanes × 2 keys each (for MAX_TOPK == 64).
// BLOCKED layout: lane L holds positions 2L, 2L+1.
// Lanes/slots without real data pass ~0u sentinel.
__device__ __forceinline__ void WarpBitonicSortAsc64(uint32_t* keys, uint32_t lane) {
  constexpr uint32_t MASK = 0xFFFFFFFFu;
  constexpr int KPT = 2;
  constexpr int N = 64;
#pragma unroll
  for (int k = 2; k <= N; k *= 2) {
#pragma unroll
    for (int j = k / 2; j > 0; j /= 2) {
      // For BLOCKED KPT=2 layout: position i = 2L + s
      // (i & k) for k power of 2 ≥ 2: == 0 iff (L & (k/2)) == 0
      // For k == N: (i & N) == 0 always since i < N → asc_pair = true (final merge)
      const bool asc_pair = (k < N) ? ((lane & (k / 2)) == 0) : true;

      if (j >= KPT) {
        // Cross-lane shuffle (mask in lane-space = j / KPT)
        const int mask = j / KPT;
        const bool I_am_lower = ((lane & mask) == 0);
#pragma unroll
        for (int s = 0; s < KPT; ++s) {
          uint32_t my = keys[s];
          uint32_t partner = __shfl_xor_sync(MASK, my, mask);
          if (I_am_lower) {
            keys[s] = asc_pair ? min(my, partner) : max(my, partner);
          } else {
            keys[s] = asc_pair ? max(my, partner) : min(my, partner);
          }
        }
      } else {
        // j == 1, within-thread: compare slot 0 vs slot 1 of same lane
        if (asc_pair) {
          if (keys[0] > keys[1]) {
            uint32_t t = keys[0]; keys[0] = keys[1]; keys[1] = t;
          }
        } else {
          if (keys[0] < keys[1]) {
            uint32_t t = keys[0]; keys[0] = keys[1]; keys[1] = t;
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// sort_write_output ? shared warp-0 tail: ascending-by-index bitonic sort
// of the selected indices with the OOB (>= num_valid_pages) clamp folded in,
// then the gmem write (-1 for clamped slots).
// -----------------------------------------------------------------------------
template <uint32_t TOPK>
__device__ __forceinline__ void sort_write_output(
    const int32_t* __restrict__ smemOutput, int32_t* __restrict__ row_out, uint32_t num_valid_pages) {
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  if (warp_id != 0) return;  // only warp 0 sorts and writes the output

  if constexpr (TOPK <= 32) {
    uint32_t key = ~0u;
    if (lane < static_cast<uint32_t>(TOPK)) {
      const int32_t idx = smemOutput[lane];
      const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < num_valid_pages);
      key = valid ? static_cast<uint32_t>(idx) : ~0u;
    }
    WarpBitonicSortAsc32(key, lane);
    if (lane < static_cast<uint32_t>(TOPK)) {
      row_out[lane] = (key == ~0u) ? -1 : static_cast<int32_t>(key);
    }
  } else {
    // TOPK == 64: 32 lanes x 2 keys, BLOCKED layout (lane L holds 2L, 2L+1).
    uint32_t keys[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < TOPK) {
        const int32_t idx = smemOutput[pos];
        const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < num_valid_pages);
        keys[s] = valid ? static_cast<uint32_t>(idx) : ~0u;
      } else {
        keys[s] = ~0u;
      }
    }
    WarpBitonicSortAsc64(keys, lane);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < TOPK) {
        const uint32_t k = keys[s];
        row_out[pos] = (k == ~0u) ? -1 : static_cast<int32_t>(k);
      }
    }
  }
}

// ============================================================
// Block-per-row register select on the bf16 transposed buffer:
// the register algorithm (1-pass) + the shared warp-0
// ascending-sort tail, so the output format matches the streaming
// path exactly (ascending k-tile index, -1 for out-of-range).
// ============================================================
template <uint32_t TOPK>
__global__ void __launch_bounds__(kBlockSize, 1) IndexerTopKRegHistBF16Kernel(
    const __ppu_bfloat16* __restrict__ in, int32_t* __restrict__ out, uint32_t total_qo_len,
    uint32_t num_qo_heads, uint32_t max_k_tiles, uint32_t topk, uint32_t num_valid_pages) {

  extern __shared__ char smem_raw[];
  auto* s_topk_indices =
      reinterpret_cast<int32_t*>(smem_raw + sizeof(RegisterTopKSmem));

  // ---- bid -> row decode: bid = qo_head_idx * total_qo_len + t ----
  const uint32_t bid = blockIdx.x;
  const uint32_t qo_head_idx = bid / total_qo_len;
  const uint32_t t = bid % total_qo_len;
  int32_t* row_out = out + (static_cast<size_t>(t) * num_qo_heads + qo_head_idx) * topk;
  const __ppu_bfloat16* logits = in + static_cast<size_t>(bid) * max_k_tiles;

  register_topk_bf16<TOPK>(logits, s_topk_indices, max_k_tiles, smem_raw);
  __syncthreads();
  sort_write_output<TOPK>(s_topk_indices, row_out, num_valid_pages);
}

}  // namespace topk_bf16
}  // namespace sparse_topk
}  // namespace flashinfer

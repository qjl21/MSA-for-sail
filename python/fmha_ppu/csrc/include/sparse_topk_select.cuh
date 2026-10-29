/*
 * Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 *
 * Copyright (c) 2024-2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * ------------------------------------------------------------------------
 * Portions of this file (indexerTopK histogram-step + insertion-sort
 * algorithm) are derived from NVIDIA TensorRT-LLM:
 *   Copyright (c) 2019-2026 NVIDIA CORPORATION
 *   Copyright (c) 2021 NAVER Corp. (CLOVA)
 * licensed under Apache License 2.0.  Original source:
 *   tensorrt_llm/cpp/tensorrt_llm/kernels/indexerTopK.cu
 *
 * Function provenance map (NVIDIA TensorRT-LLM, indexerTopK.cu line ranges):
 *   extractBinIdx                indexerTopK.cu:43-71
 *   isPartialMatch               indexerTopK.cu:73-83
 *   vectorized_process           indexerTopK.cu:99-162
 *   processHistogramStep         indexerTopK.cu:164-362  (multipleBlocksPerRow /
 *                                                        mergeBlocks branches dropped;
 *                                                        stride1!=1 branch dropped)
 *   topKPerRowJob                indexerTopK.cu:365-595  (insertion-sort branch only;
 *                                                        radix-sort branch dropped;
 *                                                        multipleBlocksPerRow /
 *                                                        mergeBlocks dropped;
 *                                                        ascending-by-index sort fused)
 *
 * v2.0_indexer_topk_qo_outermost fork additions (FlashInfer-original):
 *   SparseTopKTransposeKernel        — input layout adapter (reused from v1.6.5)
 *   SparseTopKIdentityFillKernel     — trivial-row fast path  (reused from v1.6.5)
 *   IndexerTopKWithSortKernel        — wraps topKPerRowJob and fuses an
 *                                      ascending-by-index cub::BlockRadixSort over
 *                                      smemOutput before the gmem write, so the
 *                                      output contract matches the v1.6.5 fork
 *                                      (qo_outermost layout, asc-by-index)
 *   SparseTopKWarpRankKernel / SparseTopKWarpBallotKernel
 *                                    — warp-per-row top-K (shuffle rank-count
 *                                      for K<=64 fp32; ballot binary search
 *                                      with KPT-scaled coverage up to K<=512)
 *   SparseTopKStreamMinReplaceKernel — fused streaming top-K (no transpose)
 *   SparseTopKSelect                 — top-level dispatcher
 * ------------------------------------------------------------------------
 */
#ifndef FLASHINFER_SPARSE_TOPK_SELECT_CUH_
#define FLASHINFER_SPARSE_TOPK_SELECT_CUH_

#include <hggc.h>
#include <hggc_fp16.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <type_traits>
#include "topk_bf16.cuh"

namespace flashinfer {
namespace sparse_topk {

// =============================================================================
// v2.0 design overview
// =============================================================================
//
// Replaces the v1.6.5_2cta_per_row_qo_outermost pipeline (transpose +
// Filtered/Multi-CTA + Merge + SortByIndex = 3-4 kernels) with the
// indexerTopK algorithm (transpose + indexerTopK-with-fused-sort = 2 kernels).
//
// Pipeline:
//
//   [in (Hq, K, qo) row-contig fp32]
//          |
//          v  SparseTopKTransposeKernel  (per-head 32x32 SMEM tile transpose)
//          |
//   [transpose_buf (Hq, qo, K) row-contig fp32 — workspace]
//          |
//          v  IndexerTopKWithSortKernel<MAX_TOPK>(grid=num_rows, block=256)
//          |    Stage 0: 10-bit fp16 hist + cub::BlockScan + threshold + classify
//          |    Stage 1/2/3: 10+10+10 bit fp32 (only if Stage 0 didn't finish)
//          |    Final pass: insertion sort over kNumFinalItems=2048 candidates
//          |    v2.1: warp-0-only WarpBitonicSortAsc{32,64} on smemOutput (asc by idx)
//          |    write gmem with qo_outermost offset
//          |
//   [out (qo, num_qo_heads, topk) int32, ascending-by-index]
//
//
// Fork extensions on top of v2.0:
//   - 16 < K <= 96: transpose + warp-per-row kernels (WarpRank / WarpBallot)
//     instead of IndexerTopKWithSortKernel.
//   - K >= 256 with many rows (num_rows >= 256 * num_sms): no transpose —
//     SparseTopKStreamMinReplaceKernel reads the original (H, K, T) layout.
//
// Workspace = transpose_buf only (no descriptors / merge_in / unsorted_tmp /
// counter slabs needed; v1.6.5's workspace formula is preserved as a
// conservative upper bound so callers that allocated v1.6.5-sized workspaces
// still work without any change).
constexpr uint32_t numThreadsPerBlock = 256u;
constexpr uint32_t numThreadsPerWarp = 32u;

// =============================================================================
// indexerTopK helpers — vendored from TRT-LLM indexerTopK.cu
// =============================================================================

template <int step>
__device__ __forceinline__ uint32_t extractBinIdx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
    return bits >> 6;          // 10-bit fp16 (was >> 5 for 11-bit)
  } else {
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

    if constexpr (step == 1) {
      return bits >> 22;                // high 10 bits (was >> 21)
    } else if constexpr (step == 2) {
      return (bits >> 12) & 0x3ff;      // mid 10 bits (was (>> 10) & 0x7ff)
    } else if constexpr (step == 3) {
      return (bits >> 2) & 0x3ff;       // next-low 10 bits, drops bits 0-1
    }
  }
}

template <int shift>
__device__ __forceinline__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

// Map Func over the input data, using vectorized float4 loads when possible.
// stride1==1 only (we always feed transpose_buf, which is row-contig).
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len,
                                   Func f) {
  constexpr int WARP_SIZE = 32;
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(__ldg(in + i), i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);

    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                       ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) /
                          sizeof(T))
                       : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    idxT const len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = __ldg(in_cast + i);
      idxT const real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    if (thread_rank < skip_cnt) {
      f(__ldg(in + thread_rank), thread_rank);
    }
    idxT const remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(__ldg(in + remain_i), remain_i);
    }
  }
}

// Map a float to an order-preserving uint32 (higher float -> higher uint).
// Used by the warp-level and streaming top-K kernels below.
__device__ __forceinline__ uint32_t float_to_ord(float f) {
  uint32_t bits = __float_as_uint(f);
  uint32_t mask = -static_cast<int32_t>(bits >> 31) | 0x80000000u;
  return bits ^ mask;
}

// bf16 overload: 16-bit order-preserving key, zero-extended into uint32.
//   negative -> ~bits & 0xFFFF  (in [0x0000, 0x7FFF]; ord(-inf) = 0x007F)
//   positive -> bits | 0x8000   (in [0x8000, 0xFFFF]; NaN still above +inf)
// Every bf16 key stays below kForcedOrd, so the forced-page sentinel keeps
// outranking all genuine scores, and the 0u padding sentinel stays below
// every genuine key (ord(-inf) == 0x007F > 0) — the warp/streaming kernels
// below work unchanged on bf16.
__device__ __forceinline__ uint32_t float_to_ord(__ppu_bfloat16 v) {
  const uint32_t bits = static_cast<uint32_t>(__bfloat16_as_ushort(v));
  return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

// Forced pages use the same sentinel rank as the transpose path's FLT_MAX
// (transpose writes is_forced ? FLT_MAX : value): ranks above every finite
// score, ties with a genuine FLT_MAX score, and sits below +inf / positive
// NaN — identical to extractBinIdx's bin ordering (FLT_MAX bin2 < +inf bin1
// < posNaN bin0). Keeps streaming output consistent with transpose+warp and
// transpose+histogram paths for NaN/inf inputs.
constexpr uint32_t kForcedOrd = 0xFF7FFFFFu;  // == float_to_ord(FLT_MAX)

// SMEM histogram bank padding: +1 bin per 32 bins breaks the power-of-two bank
// aliasing on the atomicAdd-heavy histogram path (1024 bins -> 1056 slots).
#define HISTO_PADDED_IDX(binIdx) ((binIdx) + ((binIdx) >> 5))

// =============================================================================
// processHistogramStep — vendored from indexerTopK.cu (single-block-per-row only)
// =============================================================================
//
// Differences vs upstream:
//   - Drops `multipleBlocksPerRow` and `mergeBlocks` template branches: v2.0
//     always processes one row per CTA, no inter-CTA merge.
//   - Drops `stride1` runtime branch: v2.0 always feeds transpose_buf, so input
//     is always row-contiguous (stride1 == 1) and we always go through
//     vectorized_process.
//   - `indices` arg removed (was only used in mergeBlocks branch).
//   - step == 0 distributes via an SMEM tile prefetch (smemTileLogits, one
//     kNumThreadsPerBlock-wide tile at a time) instead of vectorized_process;
//     the scan / threshold / classify stages are identical for all steps.
template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(float const* logits, int rowEnd, uint32_t& logitPattern,
                                     int& thresholdBinIdx, SmemOutputType& smemOutput,
                                     int* smemThresholdBinIdx, int* smemFinalDstIdx,
                                     int* smemFinalBinSize, int* smemFoundTopKValues,
                                     SmemFinalType& smemFinal, float* smemTileLogits,
                                     int rowStart, int topK) {
  // Step 0 is the fp16 fast path; if it could not resolve top-K (threshold bin
  // exceeded kNumFinalItems) we restart the fp32 radix from scratch in steps 1-3.
  // Discard any candidates step 0 wrote into smemOutput so step 1 doesn't
  // double-count valid entries that fall under both fp16 and fp32 thresholds
  // (ported from upstream indexerTopK.cu; without this reset, elements already
  // emitted at step 0 are re-emitted at step 1 and the output gets duplicated
  // indices when the step-0 threshold bin overflows).
  if constexpr (step == 1) {
    if (threadIdx.x == 0) {
      smemFoundTopKValues[0] = 0;
      smemFinalDstIdx[0] = 0;
    }
  }
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[HISTO_PADDED_IDX(idx)] = 0;
  }
  __syncthreads();

  // Update pattern.
  constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 22 : 12;
  if constexpr (step == 2) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x3ff) << patternShift;
  } else if constexpr (step == 3) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x3ff) << patternShift;
  }

  // Distribute the elements to the histogram bins.
  if constexpr (step == 0) {
    // SMEM tile prefetch: cooperative load of one kNumThreadsPerBlock-wide
    // tile into SMEM, then bin from SMEM (replaces vectorized_process).
    const int totalElems = rowEnd - rowStart;
    for (int tileStart = 0; tileStart < totalElems; tileStart += kNumThreadsPerBlock) {
      const int globalIdx = rowStart + tileStart + threadIdx.x;
      smemTileLogits[threadIdx.x] =
          (tileStart + threadIdx.x < totalElems) ? __ldg(logits + globalIdx) : 0.0f;
      __syncthreads();

      const int tileSize = (totalElems - tileStart < kNumThreadsPerBlock)
                               ? totalElems - tileStart
                               : kNumThreadsPerBlock;
      for (int i = threadIdx.x; i < tileSize; i += kNumThreadsPerBlock) {
        const uint32_t binIdx = extractBinIdx<step>(smemTileLogits[i]);
        atomicAdd(&smemFinal.histo.data[HISTO_PADDED_IDX(binIdx)], 1);
      }
      __syncthreads();
    }
  } else {
    auto distributeToBins = [&](float logit, int /* idx */ = 0) {
      if (isPartialMatch<patternShift>(logit, logitPattern)) {
        uint32_t binIdx = extractBinIdx<step>(logit);
        atomicAdd(&smemFinal.histo.data[HISTO_PADDED_IDX(binIdx)], 1);
      }
    };

    // Distribute the elements to the histogram bins.
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart,
                       distributeToBins);
    __syncthreads();
  }

  // Reads the value of the starting position in smemOutput accounting.
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount = smemFinal.histo.data[HISTO_PADDED_IDX(idx)];
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[HISTO_PADDED_IDX(idx)] = prefixSum;
    __syncthreads();

    bool foundThreshold = false;
    if (prefixSum < topK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                              ? totalSum
                              : smemFinal.histo.data[HISTO_PADDED_IDX(idx + 1)];
      if (nextPrefixSum >= topK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    if (__syncthreads_or(foundThreshold)) {
      break;
    }
    lastValue = totalSum;
  }
  if constexpr (step == 0) {
    __syncthreads();
  }

  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection.
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
        smemOutput[dstIdx] = idx;
      }
      if constexpr (step < 3) {
        // Only fill final items for sorting if the threshold bin fits.
        if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.ord_logits[dstIdx] = float_to_ord(logit);
          smemFinal.items.indices[dstIdx] = idx;
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // Elements in the threshold bin share bits [31:2] at step 3.
          int dstIdx = atomicAdd(&smemFinal.histo.data[HISTO_PADDED_IDX(binIdx)], 1);
          if (dstIdx < topK) {
            smemOutput[dstIdx] = idx;
          }
        }
      }
    }
  };

  vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart,
                     processBins);
  __syncthreads();

  // Continue to next step if the threshold bin overflows the staging buffer.
  return smemFinalBinSize[0] > kNumFinalItems;
}

// =============================================================================
// Transpose kernel — reused verbatim from v1.6.5_2cta_per_row_qo_outermost
// =============================================================================
//
// Per-head 32x32 SMEM tile transpose.  Source layout has qo as the innermost
// dim (stride 1, K-stride = qo); dest swaps so K becomes innermost (stride 1,
// qo-stride = K).  Heads dim stays outermost in both.
//
// Block: (32, 8) = 256 threads.  Each thread does 4 elements per pass.
// Tile : 32x32 floats with [32][33] padding to remove SMEM bank conflicts.
// Grid : (ceil(qo / 32), ceil(K / 32), num_qo_heads)
//
// Fallback used when XorF4 preconditions (qo % TILE == 0 && K % TILE == 0 &&
// qo % 4 == 0 && K % 4 == 0) are not met.  All decode/spec workloads with
// qo, K divisible by 32 take the faster TransposeXorF4 path.

// forced-max sentinel written by both transpose kernels into the
// forced-selection region.  Defined before its first use (two-phase lookup).
template<typename DataType>
__device__ __forceinline__ DataType forced_max_score() {
  if constexpr (std::is_same_v<DataType, float>) {
    return FLT_MAX;
  } else {
    return __ushort_as_bfloat16(static_cast<uint16_t>(0x7F80u));  // +inf for bf16
  }
}

constexpr int kTransposeTile = 32;
constexpr int kTransposeBlockRows = 8;
template<typename DataType>
__global__ void __launch_bounds__(kTransposeTile * kTransposeBlockRows)
    SparseTopKTransposeKernel(const DataType* __restrict__ in, DataType* __restrict__ out, uint32_t K,
                              uint32_t qo, uint32_t force_begin, uint32_t force_end_start,
                              uint32_t num_valid_pages) {
  __shared__ DataType tile[kTransposeTile][kTransposeTile + 1];

  const uint32_t q_base = blockIdx.x * kTransposeTile;
  const uint32_t k_base = blockIdx.y * kTransposeTile;
  const uint32_t head = blockIdx.z;
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;

  const DataType* in_head = in + static_cast<size_t>(head) * K * qo;
  DataType* out_head = out + static_cast<size_t>(head) * qo * K;

  const uint32_t q_load = q_base + tx;
#pragma unroll
  for (int dk = 0; dk < kTransposeTile; dk += kTransposeBlockRows) {
    const uint32_t k_load = k_base + ty + dk;
    if (q_load < qo && k_load < K) {
      tile[ty + dk][tx] = in_head[static_cast<size_t>(k_load) * qo + q_load];
    }
  }
  __syncthreads();

  const uint32_t k_store = k_base + tx;
  const bool is_forced = (k_store < force_begin) ||
                         (k_store >= force_end_start && k_store < num_valid_pages);
#pragma unroll
  for (int dq = 0; dq < kTransposeTile; dq += kTransposeBlockRows) {
    const uint32_t q_store = q_base + ty + dq;
    if (q_store < qo && k_store < K) {
      out_head[static_cast<size_t>(q_store) * K + k_store] =
      is_forced ? forced_max_score<DataType>() : tile[tx][ty + dq];
    }
  }
}

// =============================================================================
// SparseTopKTransposeXorKernel — FAST PATH transpose
// =============================================================================
//
// Vendored from paged_attention_doc/sparse_topk_experiments/transpose_bench/
// benchmark_swizzle.py — XOR-swizzle + float4 GMEM I/O, no SMEM padding.
//
// Theory:
//   Standard +1-padding k32: scalar float GMEM loads/stores.
//   This kernel: float4 GMEM loads (4 Q values per thread), scalar SMEM
//   writes (XOR-swizzled), scalar SMEM reads (transposed, conflict-free by
//   XOR), float4 GMEM stores (4 K values per thread).
//
//   The XOR swizzle: value[k][q] stored at S[k * TILE + (q ^ k)].
//     WRITE bank (q^k): for fixed k, varying q → all different banks → no conflict.
//     READ bank ((q^k) for varying k, fixed q): all different banks → no conflict.
//
// Preconditions (caller must ensure):
//   - TILE divides qo and K
//   - qo and K divisible by 4 (for float4 alignment)
//   - input/output base pointers 16-B aligned (cudaMalloc guarantees this for
//     the workspace pointer; transpose_buf offset is at the start so aligned)
//
// Grid: (qo / TILE, K / TILE, num_qo_heads), Block: TILE * TILE / 4 threads.
template <int TILE, typename DataType>
__global__ void __launch_bounds__(TILE * TILE / 4) SparseTopKTransposeXorKernel(
    const DataType* __restrict__ in, DataType* __restrict__ out, uint32_t K, uint32_t qo,
    uint32_t force_begin, uint32_t force_end_start, uint32_t num_valid_pages) {
  constexpr int N_THR = TILE / 4;  // threads along N (qo) direction
  using PackType = std::conditional_t<
          std::is_same_v<DataType, float>,
          float4, uint2>;

  __shared__ DataType S[TILE * TILE];

  const uint32_t qb = blockIdx.x * TILE;
  const uint32_t kb = blockIdx.y * TILE;
  const uint32_t h  = blockIdx.z;
  const uint32_t tx = threadIdx.x;

  const int tm = tx / N_THR;        // K-tile row index (0..TILE-1)
  const int tn = tx % N_THR;        // Q-tile col group (0..N_THR-1, covers 4 Q each)

  const DataType* in_head  = in  + static_cast<size_t>(h) * K * qo;
  DataType*       out_head = out + static_cast<size_t>(h) * qo * K;

  // ---- LOAD: float4 GMEM → XOR-swizzled SMEM ---------------------------
  {
    const PackType* in_pack = reinterpret_cast<const PackType*>(
        in_head + static_cast<size_t>(kb + tm) * qo + qb + tn * 4);
    const PackType v = *in_pack;
    DataType e0, e1, e2, e3;
    if constexpr (std::is_same_v<DataType, __ppu_bfloat16>) {
      e0 = __ushort_as_bfloat16(static_cast<uint16_t>(v.x & 0xFFFFu));
      e1 = __ushort_as_bfloat16(static_cast<uint16_t>(v.x >> 16));
      e2 = __ushort_as_bfloat16(static_cast<uint16_t>(v.y & 0xFFFFu));
      e3 = __ushort_as_bfloat16(static_cast<uint16_t>(v.y >> 16));
    } else {
      e0 = v.x; e1 = v.y; e2 = v.z; e3 = v.w;
    }
    const int base = tm * TILE;
    S[base + ((tn * 4 + 0) ^ tm)] = e0;
    S[base + ((tn * 4 + 1) ^ tm)] = e1;
    S[base + ((tn * 4 + 2) ^ tm)] = e2;
    S[base + ((tn * 4 + 3) ^ tm)] = e3;
  }
  __syncthreads();

  // ---- STORE: scalar swizzled SMEM reads → float4 GMEM -----------------
  {
    const int q_out = qb + tm;
    const int k0    = kb + tn * 4;

    DataType v0 = S[(tn * 4 + 0) * TILE + (tm ^ (tn * 4 + 0))];
    DataType v1 = S[(tn * 4 + 1) * TILE + (tm ^ (tn * 4 + 1))];
    DataType v2 = S[(tn * 4 + 2) * TILE + (tm ^ (tn * 4 + 2))];
    DataType v3 = S[(tn * 4 + 3) * TILE + (tm ^ (tn * 4 + 3))];

    auto is_forced = [&](uint32_t k) -> bool {
      return (k < force_begin) || (k >= force_end_start && k < num_valid_pages);
    };
    const uint32_t uk0 = static_cast<uint32_t>(k0);
    const DataType max_score = forced_max_score<DataType>();
    if (is_forced(uk0))     v0 = max_score;
    if (is_forced(uk0 + 1)) v1 = max_score;
    if (is_forced(uk0 + 2)) v2 = max_score;
    if (is_forced(uk0 + 3)) v3 = max_score;

    PackType out_v;
    if constexpr (std::is_same_v<DataType, __ppu_bfloat16>){
      out_v.x = static_cast<uint32_t>(__bfloat16_as_ushort(v0)) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(v1)) << 16);
      out_v.y = static_cast<uint32_t>(__bfloat16_as_ushort(v2)) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(v3)) << 16);
    }else{
      out_v = make_float4(v0, v1, v2, v3);
    }
    PackType* out_pack = reinterpret_cast<PackType*>(
        out_head + static_cast<size_t>(q_out) * K + k0);
    *out_pack = out_v;
  }
}


// =============================================================================
// Identity-fill kernel — reused verbatim from v1.6.5_2cta_per_row_qo_outermost
// =============================================================================
// Used when max_k_tiles <= topk (trivial row): output = [0, 1, ..., K-1, -1, ..., -1].
// Skips both transpose + indexerTopK to save VRAM and a kernel launch.
//
// v2.5_oob_clamp_in_kernel: also handles num_valid_pages clamp — indices in
// [num_valid_pages, max_k_tiles) become -1 (already sorted to tail by
// construction since the trivial fill is ascending). Pass num_valid_pages =
// max_k_tiles (or any value >= max_k_tiles) to disable clamping.
__global__ void SparseTopKIdentityFillKernel(int32_t* __restrict__ out, uint32_t total_qo_len,
                                             uint32_t num_qo_heads, uint32_t max_k_tiles,
                                             uint32_t topk, uint32_t num_valid_pages) {
  // grid = (num_rows = total_qo_len * num_qo_heads,), block = (min(topk, 256),)
  // bid encodes (qo_head_idx, t) with t innermost: bid = qo_head_idx * total_qo_len + t
  // out physical layout: (total_qo_len, num_qo_heads, topk) row-major contiguous
  //   element (t, qo_head_idx, k) at offset = (t * num_qo_heads + qo_head_idx) * topk + k
  const uint32_t bid = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t qo_head_idx = bid / total_qo_len;
  const uint32_t t = bid % total_qo_len;
  int32_t* row = out + (static_cast<size_t>(t) * num_qo_heads + qo_head_idx) * topk;
  // valid_count = min(max_k_tiles, num_valid_pages); positions [valid_count, topk) → -1
  const uint32_t valid_count = (num_valid_pages < max_k_tiles) ? num_valid_pages : max_k_tiles;
  for (uint32_t i = tx; i < topk; i += blockDim.x) {
    row[i] = (i < valid_count) ? static_cast<int32_t>(i) : static_cast<int32_t>(-1);
  }
}

// =============================================================================
// Warp-rank path: warp-shuffle rank-count for max_k_tiles <= 32*KeysPerThread
// KeysPerThread: keys per thread, 1 or 2
// Each lane holds KeysPerThread elements at positions numThreadsPerWarp*i+lane; fully coalesced.
// TOPK<=numThreadsPerWarp
// =============================================================================
template <uint32_t TOPK, int KeysPerThread, int kWarpCount = 8, typename DataType = float>
__global__ void __launch_bounds__(kWarpCount * numThreadsPerWarp, 4)
SparseTopKWarpRankKernel(
    const DataType* __restrict__ in,
    int32_t* __restrict__ out,
    uint32_t total_qo_len,
    uint32_t num_qo_heads,
    uint32_t max_k_tiles,
    uint32_t num_valid_pages) {
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane    = threadIdx.x & 31;
  const uint32_t global_row = blockIdx.x * kWarpCount + warp_id;
  const uint32_t num_rows   = total_qo_len * num_qo_heads;
  if (global_row >= num_rows) return;

  const uint32_t head = global_row / total_qo_len;
  const uint32_t t    = global_row % total_qo_len;
  const DataType* row_base = in + static_cast<size_t>(global_row) * max_k_tiles;

  // FULL_MASK, __shfl_sync and WarpBitonicSortAsc32 below are hard-tied to a
  // 32-lane hardware warp;
  static_assert(TOPK <= numThreadsPerWarp, "SparseTopKWarpRankKernel: TOPK must be <= 32.");
  static_assert(KeysPerThread == 1 || KeysPerThread == 2,
              "Only KeysPerThread 1 (max_k_tiles<=32) and 2 (max_k_tiles<=64) are supported");

  // ---- Load: 1pt or 2pt depending on KeysPerThread ----
  uint32_t ord_logits[KeysPerThread];
  #pragma unroll
  for (int i = 0; i < KeysPerThread; ++i) {
    const uint32_t pos = numThreadsPerWarp * i + lane;
    ord_logits[i] = (pos < max_k_tiles) ? float_to_ord(__ldg(row_base + pos)) : 0u;
  }
  // ---- Warp shuffle rank-count ----
  int ranks[KeysPerThread] = {0};
  constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;
  #pragma unroll
  for (uint32_t j = 0; j < numThreadsPerWarp; ++j) {
    uint32_t other0 = __shfl_sync(FULL_MASK, ord_logits[0], j);
    uint32_t other1 = 0u;
    if constexpr (KeysPerThread == 2) {
      other1 = __shfl_sync(FULL_MASK, ord_logits[1], j);
    }

    // --- ranks[0]: element at position `lane` ---
    if (j != static_cast<int>(lane)) {
      if (other0 > ord_logits[0] || (other0 == ord_logits[0] && static_cast<uint32_t>(j) < lane))
        ranks[0]++;
    }
    if constexpr (KeysPerThread == 2) {
      if (other1 > ord_logits[0]) ranks[0]++;
    }

    // --- ranks[1]: element at position `numThreadsPerWarp + lane` ---
    if constexpr (KeysPerThread == 2) {
      if (other0 >= ord_logits[1]) ranks[1]++;
      // vs other1 at pos numThreadsPerWarp+j: skip self
      if (j != static_cast<int>(lane)) {
        if (other1 > ord_logits[1] || (other1 == ord_logits[1] && static_cast<uint32_t>(j) < lane))
          ranks[1]++;
      }
    }
  }

  // ---- Scatter top-TOPK indices into shared memory ----
  __shared__ int32_t smem_topk[kWarpCount][numThreadsPerWarp];
  smem_topk[warp_id][lane] = -1;
  __syncwarp(FULL_MASK);
  #pragma unroll
  for (int i = 0; i < KeysPerThread; ++i) {
    const uint32_t pos = numThreadsPerWarp * i + lane;
    if (ranks[i] < static_cast<int>(TOPK) && pos < max_k_tiles && pos < num_valid_pages)
      smem_topk[warp_id][ranks[i]] = static_cast<int32_t>(pos);
  }
  __syncwarp(FULL_MASK);

  // ---- Gather and sort top-K indices ----
  uint32_t key = ~0u;
  if (lane < TOPK) {
    const int32_t idx = smem_topk[warp_id][lane];
    key = (idx >= 0) ? static_cast<uint32_t>(idx) : ~0u;
  }
  topk_bf16::WarpBitonicSortAsc32(key, lane);

  // ---- Write output ----
  if (lane < TOPK) {
    int32_t* row_out = out + (static_cast<size_t>(t) * num_qo_heads + head) * TOPK;
    row_out[lane] = (key == ~0u) ? static_cast<int32_t>(-1) : static_cast<int32_t>(key);
  }
}


// =============================================================================
// LargeK path: warp-ballot binary-search selection for 64 < max_k_tiles <= 96
// Each lane holds 3 elements (positions: lane, 32+lane, 64+lane).
// Reads from TRANSPOSED buffer — K dimension contiguous, fully coalesced.
// =============================================================================
template <uint32_t TOPK, int KeysPerThread = 3, int kWarpCount = 8, typename DataType = float>
__global__ void __launch_bounds__(kWarpCount * numThreadsPerWarp, 4)
SparseTopKWarpBallotKernel(
    const DataType* __restrict__ in,
    int32_t* __restrict__ out,
    uint32_t total_qo_len,
    uint32_t num_qo_heads,
    uint32_t max_k_tiles,
    uint32_t num_valid_pages) {

  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane    = threadIdx.x & 31;
  const uint32_t global_row = blockIdx.x * kWarpCount + warp_id;
  const uint32_t num_rows   = total_qo_len * num_qo_heads;
  static_assert(TOPK <= numThreadsPerWarp, "SparseTopKWarpBallotKernel: TOPK must be <= 32.");
  static_assert(KeysPerThread >= 1 && KeysPerThread <= 16,
                "SparseTopKWarpBallotKernel: KeysPerThread in [1,16] (covers max_k_tiles <= 32*KPT).");
  if (global_row >= num_rows) return;

  const uint32_t head = global_row / total_qo_len;
  const uint32_t t    = global_row % total_qo_len;

  // ---- Coalesced load: KeysPerThread segments of 32 elements ----
  // 16-bit dtypes (bf16/fp16): the float_to_ord overload natively yields a
  // 16-bit key zero-extended into uint32 (value range [0, 0xFFFF]), so the
  // binary search over [0, kOrdHi] converges in 16 iterations instead of 32.
  constexpr uint32_t kOrdBits = (sizeof(DataType) == 2) ? 16u : 32u;
  constexpr uint32_t kOrdHi   = (kOrdBits == 16) ? 0xFFFFu : 0xFFFFFFFFu;
  const DataType* row_base = in + static_cast<size_t>(global_row) * max_k_tiles;
  uint32_t ord_logits[KeysPerThread];
  // Convert to comparison-preserving uint32: higher float -> higher uint
  #pragma unroll
  for (int i = 0; i < KeysPerThread; ++i) {
    const uint32_t pos = numThreadsPerWarp * i + lane;
    ord_logits[i] = (pos < max_k_tiles) ? float_to_ord(__ldg(row_base + pos)) : 0u;
  }

  // ---- Top-K selection via binary search + ballot counting ----
  // Algorithm: O(32 ballot iterations) instead of O(32·KPT²) rank-count.
  //   1. Convert floats to ordered uint32 (preserves comparison order)
  //   2. Binary search the full uint32 range for the threshold ord_logits of the
  //      TOPK-th value (count(>thr) < TOPK <= count(>=thr))
  //   3. Select elements above threshold + tie-break at threshold by position
  //   4. Scatter to smem, sort, write output
  constexpr uint32_t FULL_MASK = 0xFFFFFFFFu;

  // Binary search on uint32 space: find threshold such that
  //   count(ord_logits > threshold) < TOPK  AND  count(ord_logits >= threshold) >= TOPK
  // 32 iterations exactly cover the full uint32 domain, so start from the
  // full range: the answer is range-independent and extra iterations after
  // convergence are stable no-ops (cnt(>thr) < TOPK always takes the hi branch).
  uint32_t lo = 0u, hi = kOrdHi;
  #pragma unroll
  for (uint32_t iter = 0; iter < kOrdBits; ++iter) {
    uint32_t mid = lo + ((hi - lo) >> 1);
    int cnt = 0;
    #pragma unroll
    for (int i = 0; i < KeysPerThread; ++i) {
      cnt += __popc(__ballot_sync(FULL_MASK, ord_logits[i] > mid));
    }
    if (cnt >= static_cast<int>(TOPK)) lo = mid + 1; else hi = mid;
  }
  // lo == hi == threshold_ord: the ordered representation of the TOPK-th value
  const uint32_t thr = lo;

  // Count elements strictly above threshold (guaranteed in top-K)
  int n_above = 0;
  #pragma unroll
  for (int i = 0; i < KeysPerThread; ++i) {
    n_above += __popc(__ballot_sync(FULL_MASK, ord_logits[i] > thr));
  }
  int need_left = static_cast<int>(TOPK) - n_above;  // how many ties to select

  // Elements at threshold (ties): select by segment priority (seg0 < seg1 < ... ,
  // lane order within segment), then rank each selected element in the selected set.
  const uint32_t below = (1u << lane) - 1u;  // bitmask of lanes before me
  bool sel[KeysPerThread];
  int ranks[KeysPerThread];
  int prior_sel = 0;        // selected count in previous segments (warp-uniform)

  #pragma unroll
  for (int s = 0; s < KeysPerThread; ++s) {
    const uint32_t at = __ballot_sync(FULL_MASK, ord_logits[s] == thr);
    // Above-threshold always selected; at-threshold selected if my lane-order
    // position among ties fits in the remaining slots.
    sel[s] = (ord_logits[s] > thr) ||
             (ord_logits[s] == thr && __popc(at & below) < need_left);
    // Rank = selected in previous segments + selected before me in this segment.
    const uint32_t sel_mask = __ballot_sync(FULL_MASK, sel[s]);
    ranks[s] = prior_sel + __popc(sel_mask & below);
    // Advance per-segment state.
    need_left -= __popc(at);
    prior_sel += __popc(sel_mask);
  }

  // ---- Scatter top-TOPK indices into shared memory ----
  __shared__ int32_t smem_topk[kWarpCount][numThreadsPerWarp];
  smem_topk[warp_id][lane] = -1;
  __syncwarp(FULL_MASK);

  // Scatter each of the 3 elements if selected into top-TOPK
  #pragma unroll
  for (int m = 0; m < KeysPerThread; ++m) {
    const uint32_t pos = static_cast<uint32_t>(m) * numThreadsPerWarp + lane;
    if (sel[m] && pos < max_k_tiles && pos < num_valid_pages) {
      smem_topk[warp_id][ranks[m]] = static_cast<int32_t>(pos);
    }
  }
  __syncwarp(FULL_MASK);

  // ---- Gather and sort top-K indices ----
  uint32_t key = ~0u;
  if (lane < TOPK) {
    const int32_t idx = smem_topk[warp_id][lane];
    key = (idx >= 0) ? static_cast<uint32_t>(idx) : ~0u;
  }
  topk_bf16::WarpBitonicSortAsc32(key, lane);

  // ---- Write output ----
  if (lane < TOPK) {
    int32_t* row_out = out + (static_cast<size_t>(t) * num_qo_heads + head) * TOPK;
    row_out[lane] = (key == ~0u) ? static_cast<int32_t>(-1) : static_cast<int32_t>(key);
  }
}

// =============================================================================
// SparseTopKStreamMinReplaceKernel — fused streaming top-K over the original
// (H, K, T) layout (no transpose). Each thread handles one query, iterating
// over K with a register-resident top-TOPK (min-tracked) array.
//   - k_step-unrolled main loop (16 for TOPK <= 16, else 8) for load prefetch
//   - force_begin/force_end pages are pinned via kForcedOrd: [0, force_begin)
//     is forced at init; the force_end zone is forced in the checked tail and
//     re-forced by a fixup pass for the part the unrolled loop scanned raw
//   - padding pages [num_valid_pages, max_k_tiles) compete raw and are
//     clamped to -1 after the scan
//   - __launch_bounds__(256)
// =============================================================================
template <uint32_t TOPK, typename DataType = float>
__global__ void __launch_bounds__(256)
SparseTopKStreamMinReplaceKernel(
    const DataType* __restrict__ in,   // original layout: (H, K, T), T-contiguous
    int32_t* __restrict__ out,      // (T, H, topk)
    uint32_t total_qo_len,
    uint32_t num_qo_heads,
    uint32_t max_k_tiles,
    uint32_t num_valid_pages,
    uint32_t force_begin,
    uint32_t force_end) {

  const uint32_t tid = threadIdx.x;
  const uint32_t global_query = blockIdx.x * blockDim.x + tid;
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  if (global_query >= num_rows) return;

  const uint32_t head = global_query / total_qo_len;
  const uint32_t t    = global_query % total_qo_len;
  const size_t T = static_cast<size_t>(total_qo_len);  
  static_assert(TOPK <= 32, "SparseTopKStreamMinReplaceKernel: TOPK must be <= 32.");

  // Base: in[head, k, t] = in + head*K*T + k*T + t
  const DataType* base = in + static_cast<size_t>(head) * max_k_tiles * T + t;
  const uint32_t force_end_start =
      (force_end <= num_valid_pages) ? num_valid_pages - force_end : 0;
  auto is_forced = [&](uint32_t k) -> bool {
    return (k < force_begin) || (k >= force_end_start && k < num_valid_pages);
  };

  // ---- Initialize top-K registers: direct-fill the first TOPK elements ----
  // Precondition: max_k_tiles >= TOPK (guaranteed by dispatch).
  // top_ord holds float_to_ord values (ordered-uint space), top_idx the k-positions.
  uint32_t top_ord[TOPK];
  uint32_t top_idx[TOPK];
  #pragma unroll
  for (uint32_t i = 0; i < TOPK; ++i) {
    top_ord[i] = is_forced(i) ? kForcedOrd : float_to_ord(__ldg(base + i * T));
    top_idx[i] = i;
  }

  // Recompute the minimum (value + slot) of the register-resident top set.
  uint32_t min_ord, min_slot;
  auto find_min = [&]() {
    min_ord = top_ord[0]; min_slot = 0;
    #pragma unroll
    for (uint32_t j = 1; j < TOPK; ++j) {
      if (top_ord[j] < min_ord) { min_ord = top_ord[j]; min_slot = j; }
    }
  };

  // Insert (ord_val, idx) if it beats the current minimum, then recompute it.
  // Keeps exactly the TOPK largest ord_logits values seen so far.
  auto insert_min = [&](uint32_t ord_val, uint32_t idx) {
    if (ord_val > min_ord) {
      top_ord[min_slot] = ord_val;
      top_idx[min_slot] = idx;
      find_min();
    }
  };

  find_min();

  // ---- Streaming main loop with k_step unroll (start from TOPK) ----
  constexpr uint32_t k_step = (TOPK <= 16) ? 16 : 8;
  const uint32_t k_left = (num_valid_pages > force_end + TOPK) ?  num_valid_pages - force_end - TOPK : 0;
  const uint32_t K16 = TOPK + (k_left & ~(k_step - 1u));
  uint32_t k = TOPK;

  for (; k < K16; k += k_step) {
    // Three phases: batch-load keeps k_step loads in flight (MLP),
    // then batch-convert, then batch-insert (serial on min-slot anyway).
    DataType f[k_step];
    uint32_t o[k_step];
    #pragma unroll
    for (uint32_t u = 0; u < k_step; ++u)
      f[u] = __ldg(base + static_cast<size_t>(k + u) * T);
    #pragma unroll
    for (uint32_t u = 0; u < k_step; ++u)
      o[u] = float_to_ord(f[u]);
    #pragma unroll
    for (uint32_t u = 0; u < k_step; ++u)
      insert_min(o[u], k + u);
  }

  // Checked region [K16, num_valid_pages): the force_end zone lives entirely
  // here (raw region stops at force_end_start) and is also re-forced below.
  for (; k < num_valid_pages; ++k) {
    const uint32_t o = is_forced(k) ? kForcedOrd : float_to_ord(__ldg(base + k * T));
    insert_min(o, k);
  }

  //Entries in [force_end_start, num_valid_pages), upgrade in place if already in the set else insert at kForcedOrd.
  // Zero-trip when force_end == 0.
  auto find_slot = [&](uint32_t idx) -> int {
    for (uint32_t s = 0; s < TOPK; ++s)
      if (top_idx[s] == idx) return static_cast<int>(s);
    return -1;
  };
  for (uint32_t j = force_end_start; j < num_valid_pages; ++j){
    const int s = find_slot(j);
    if (s >= 0) {
     top_ord[s] = kForcedOrd;
     if (static_cast<uint32_t>(s) == min_slot) find_min();
    }else{
      insert_min(kForcedOrd, j);
    }
  }

  // ---- Apply num_valid_pages clamping ----
  #pragma unroll
  for (uint32_t i = 0; i < TOPK; ++i) {
    if (top_idx[i] >= num_valid_pages) {
      top_idx[i] = ~0u;
    }
  }

  // ---- Sort output indices ascending (bubble sort on TOPK elements) ----
  #pragma unroll
  for (int pass = 0; pass < static_cast<int>(TOPK) - 1; ++pass) {
    #pragma unroll
    for (int j = 0; j < static_cast<int>(TOPK) - 1 - pass; ++j) {
      if (top_idx[j] > top_idx[j + 1]) {
        uint32_t ti = top_idx[j]; top_idx[j] = top_idx[j+1]; top_idx[j+1] = ti;
      }
    }
  }

  // ---- Write output ----
  int32_t* row_out = out + (static_cast<size_t>(t) * num_qo_heads + head) * TOPK;
  #pragma unroll
  for (uint32_t i = 0; i < TOPK; ++i) {
    row_out[i] = (top_idx[i] == ~0u) ? static_cast<int32_t>(-1) : static_cast<int32_t>(top_idx[i]);
  }
}

// =============================================================================
// IndexerTopKWithSortKernel — fused indexerTopK + asc-by-index sort + qo_outermost write
// =============================================================================
//
// Per-CTA work (1 CTA per (qo_head, qo_pos) row):
//   1. Stage 0: 10-bit fp16 histogram (1024 bins) → cub::BlockScan threshold
//      → classify each element: bin < threshold ⇒ direct emit to smemOutput;
//        bin == threshold ⇒ stage to smemFinal.items if it fits (≤ 2048).
//   2. Stage 1/2/3 (only if needed): 10+10+10 bit fp32 histogram refinement
//      on the threshold-bin elements (covers bits [31:2]; low 2 mantissa
//      bits are not examined — such near-ties are selected arbitrarily).
//   3. If we never fell through to step 3, the staging buffer has a
//      well-defined subset of "ties at the boundary" — insertion sort over
//      smemFinal.items emits the remaining (topk - foundSoFar) slots into
//      smemOutput.
//   4. Fused asc-by-index sort (v2.1): warp-0-only WarpBitonicSortAsc{32,64}
//      over smemOutput[0..topk-1] → ascending by index.
//   5. Write to gmem with qo_outermost offset:
//        out[t][qo_head_idx][k] at (t*num_qo_heads + qo_head_idx) * topk + k
//
// Template params:
//   MAX_TOPK            — round-up template bin (16/32/64), runtime topk ≤ MAX_TOPK
//   kNumThreadsPerBlock — fixed at 256 (matches trtllm canonical config)
//   kNumBins            — fixed at 1024 (10-bit fp16 / fp32 hist)
//   kNumFinalItems      — fixed at 2048 (insertion sort staging capacity)
//
// Dynamic SMEM = topk * sizeof(int32_t) — host MUST cudaFuncSetAttribute the
// max value (kSparseTopkMaxK * sizeof(int32_t) = 256 B).
constexpr uint32_t kSparseTopkMaxK = 64;
constexpr int kIndexerNumThreadsPerBlock = 256;
constexpr int kIndexerNumBins = 1024;  // 10-bit hist (was 2048 / 11-bit)
constexpr int kIndexerNumBinsPadded = kIndexerNumBins + (kIndexerNumBins >> 5);
constexpr int kIndexerNumFinalItems = 2048;

template <uint32_t MAX_TOPK>
__global__ void __launch_bounds__(kIndexerNumThreadsPerBlock, /*minBlocks=*/2) IndexerTopKWithSortKernel(
    const float* __restrict__ in,        // (num_qo_heads, qo, K) row-contig fp32 (transpose_buf)
    int32_t* __restrict__ out,           // (qo, num_qo_heads, topk) row-major contig int32
    uint32_t total_qo_len, uint32_t num_qo_heads, uint32_t max_k_tiles,
    uint32_t topk,
    // v2.5_oob_clamp_in_kernel: indices >= num_valid_pages get rewritten to -1
    // and sorted to the tail.  Pass num_valid_pages = max_k_tiles to disable
    // (the comparison `idx >= num_valid_pages` will then never trigger because
    // every valid idx is in [0, max_k_tiles)).
    uint32_t num_valid_pages) {
  static_assert(MAX_TOPK <= kSparseTopkMaxK, "MAX_TOPK exceeds supported max");

  constexpr int kNumThreadsPerBlock = kIndexerNumThreadsPerBlock;
  constexpr int kNumBins = kIndexerNumBins;
  constexpr int kNumFinalItems = kIndexerNumFinalItems;
  constexpr int kNumBinsPadded = kIndexerNumBinsPadded;

  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  struct FinalItems {
    int indices[kNumFinalItems];
    uint32_t ord_logits[kNumFinalItems];//map:higher float score -> higher ordinal
  };
  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBinsPadded];
  };

  // SMEM union — v2.1 dropped the cub::BlockRadixSort sortByIndex member
  // since the asc-by-index sort is now warp-only (uses no SMEM).
  __shared__ union {
    FinalItems items;
    Histogram histo;
  } smemFinal;

  // Dynamic SMEM holds the top-K accumulator (one int32 per slot, sized by topk).
  extern __shared__ int32_t smemOutput[];

  __shared__ int smemThresholdBinIdx[1];
  __shared__ int smemFinalDstIdx[1];
  __shared__ int smemFinalBinSize[1];
  __shared__ int smemFoundTopKValues[1];

  // Stage 0 SMEM tile buffer: kNumThreadsPerBlock floats, only live during
  // the step-0 SMEM tile prefetch inside processHistogramStep.
  __shared__ float smemTileLogits[kNumThreadsPerBlock];

  // ---- bid → row decode ---------------------------------------------------
  // bid encodes (qo_head, token): bid = qo_head_idx * total_qo_len + t
  const uint32_t bid = blockIdx.x;
  const uint32_t qo_head_idx = bid / total_qo_len;
  const uint32_t t = bid % total_qo_len;
  // qo_outermost output offset:
  //   out (t, qo_head_idx, k) at offset (t*num_qo_heads + qo_head_idx) * topk + k
  const size_t row_offset_out =
      (static_cast<size_t>(t) * num_qo_heads + qo_head_idx) * topk;

  // Per-row input pointer (transposed layout: row stride = max_k_tiles).
  const float* logits = in + static_cast<size_t>(bid) * max_k_tiles;
  const int rowStart = 0;
  const int rowEnd = static_cast<int>(num_valid_pages);
  const int rowLen = rowEnd - rowStart;
  const int topK = static_cast<int>(topk);

  // ---- Trivial path: rowLen <= topK (defensive — host dispatcher already
  //      catches max_k_tiles <= topk via SparseTopKIdentityFillKernel).
  if (rowLen <= topK) {
    int32_t* row_out = out + row_offset_out;
    // OOB clamp: indices >= num_valid_pages → -1
    const int valid_count =
        (static_cast<int>(num_valid_pages) < rowLen) ? static_cast<int>(num_valid_pages) : rowLen;
    for (int rowIt = threadIdx.x; rowIt < valid_count; rowIt += kNumThreadsPerBlock) {
      row_out[rowIt] = rowIt;
    }
    for (int rowIt = valid_count + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
      row_out[rowIt] = -1;
    }
    return;
  }

  // ---- Init scalar SMEM counters -----------------------------------------
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // ---- Stage 0: fp16 10-bit hist (SMEM tiled prefetch) --------------------
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
          logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
          smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, smemTileLogits, rowStart, topK);

  if (continueToNextStep) {
    // Stage 1: fp32 high 10 bits.
    continueToNextStep =
        processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
            logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, /*smemTileLogits=*/nullptr, rowStart, topK);
  }
  if (continueToNextStep) {
    // Stage 2: fp32 mid 10 bits.
    continueToNextStep =
        processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
            logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
            smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, /*smemTileLogits=*/nullptr, rowStart, topK);
  }
  if (continueToNextStep) {
    // Stage 3: fp32 bits [11:2].  Remaining ties share bits [31:2] (low 2
    // mantissa bits not examined), so we just fill the topk window directly.
    processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems>(
        logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput, smemThresholdBinIdx,
        smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, /*smemTileLogits=*/nullptr, rowStart, topK);
  }

  if (!continueToNextStep) {
    // Insertion sort path: the threshold bin fit within kNumFinalItems, so we
    // sort the staged candidates by score (ties by staging arrival order) and emit the
    // remaining (topK - foundSoFar) entries to smemOutput.
    const int baseIdx = smemFoundTopKValues[0];
    const int finalCount = smemFinalDstIdx[0];
    for (int i = threadIdx.x; i < finalCount; i += kNumThreadsPerBlock) {
      int outIndex = 0;
      const uint32_t curOrd = smemFinal.items.ord_logits[i];
      for (int j = 0; j < finalCount; j++) {
        const uint32_t otherOrd = smemFinal.items.ord_logits[j];
        if (otherOrd > curOrd || (otherOrd == curOrd && i < j)) {
          outIndex++;
        }
      }
      if (outIndex + baseIdx < topK) {
        smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
      }
    }
    __syncthreads();
  }

  // ---- v2.1 warp-only fused asc-by-index sort -----------------------------
  // After smemOutput[0..topk-1] is filled (step-3 direct fill, or insertion
  // sort on the staged ties), only warp 0 (32 lanes) participates in the
  // sort.  The other 7 warps return early at the guard below.
  //
  // Why warp-only vs cub::BlockRadixSort<512, 1>:
  //   - cub::BlockRadixSort<uint32_t, 512, 1>.Sort() processes ALL 512 thread
  //     positions even though only ≤64 are real (rest are sentinel ~0u).
  //     Per nsys, that variant cost ~13 us per kernel call.
  //   - WarpBitonicSortAsc{32,64} only sorts the 32/64 real slots.  Estimated
  //     cost: < 1 us per kernel.  Savings: ~12 us per call = ~50% of the
  //     v2.0 IndexerTopKWithSortKernel runtime on n1024_K8192.
  const uint32_t warp_id = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31;
  if (warp_id != 0) return;  // only warp 0 sorts and writes the output

  int32_t* row_out = out + row_offset_out;

  if constexpr (MAX_TOPK <= 32) {
    // 32 lanes × 1 key.  Lanes >= topk pad with sentinel ~0u (sorts to end).
    // v2.5_oob_clamp_in_kernel: also fold OOB clamp here — indices >=
    // num_valid_pages get rewritten to ~0u, so they sort to the tail and the
    // existing `(key == ~0u) ? -1` write contract converts them to -1.
    uint32_t key = ~0u;
    if (lane < topK) {
      const int32_t idx = smemOutput[lane];
      const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < num_valid_pages);
      key = valid ? static_cast<uint32_t>(idx) : ~0u;
    }
    topk_bf16::WarpBitonicSortAsc32(key, lane);
    if (lane < topK) {
      row_out[lane] = (key == ~0u) ? -1 : static_cast<int32_t>(key);
    }
  } else {
    // MAX_TOPK == 64: 32 lanes × 2 keys, BLOCKED layout (lane L holds 2L, 2L+1).
    uint32_t keys[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < topK) {
        const int32_t idx = smemOutput[pos];
        const bool valid = (idx >= 0) && (static_cast<uint32_t>(idx) < num_valid_pages);
        keys[s] = valid ? static_cast<uint32_t>(idx) : ~0u;
      } else {
        keys[s] = ~0u;
      }
    }
    topk_bf16::WarpBitonicSortAsc64(keys, lane);
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int pos = static_cast<int>(lane) * 2 + s;
      if (pos < topK) {
        const uint32_t k = keys[s];
        row_out[pos] = (k == ~0u) ? -1 : static_cast<int32_t>(k);
      }
    }
  }
}

// transpose launch: (H,K,T) -> (H,T,K).
// Xor fast path when qo & K are multiples of 32/4; else padded-tile fallback.
// transpose_k: leading K tiles to actually transpose (32-aligned, <= max_k_tiles);
// buffer strides still use max_k_tiles. Pass max_k_tiles for a full transpose.
template <typename DataType>
inline hggcError_t LaunchTranspose(const DataType* in_strided, DataType* transposed,
                                   uint32_t total_qo_len, uint32_t num_qo_heads,
                                   uint32_t max_k_tiles, uint32_t num_valid_pages,
                                   uint32_t force_begin, uint32_t force_end_start,
                                   uint32_t transpose_k, hggcStream_t stream) {
  constexpr int kXorTile = 32;
  const bool xor_ok = (total_qo_len % kXorTile == 0u) && (max_k_tiles % kXorTile == 0u) &&
                      (total_qo_len % 4u == 0u) && (max_k_tiles % 4u == 0u);
  if (xor_ok) {
    dim3 grid(total_qo_len / kXorTile, transpose_k / kXorTile, num_qo_heads);
    dim3 block(kXorTile * kXorTile / 4);  // = 256 threads
    void* tr_args[] = {(void*)&in_strided, (void*)&transposed, (void*)&max_k_tiles,
                       (void*)&total_qo_len, (void*)&force_begin, (void*)&force_end_start,
                       (void*)&num_valid_pages};
    return hggcLaunchKernel((const void*)SparseTopKTransposeXorKernel<kXorTile, DataType>,
                            grid, block, tr_args, 0, stream);
  }
  dim3 grid((total_qo_len + kTransposeTile - 1) / kTransposeTile,
            (transpose_k + kTransposeTile - 1) / kTransposeTile, num_qo_heads);
  dim3 block(kTransposeTile, kTransposeBlockRows);
  void* tr_args[] = {(void*)&in_strided, (void*)&transposed, (void*)&max_k_tiles,
                     (void*)&total_qo_len, (void*)&force_begin, (void*)&force_end_start,
                     (void*)&num_valid_pages};
  return hggcLaunchKernel((const void*)SparseTopKTransposeKernel<DataType>, grid, block,
                          tr_args, 0, stream);
}

// =============================================================================
// Host-side dispatcher
// =============================================================================

// Launch one of the warp-level top-K kernels (WarpRank / WarpBallot):
// 1 warp per row, 8 rows per 256-thread block.
template <typename KernelT, typename DataType>
hggcError_t LaunchWarpRowTopK(KernelT kernel, const DataType* transposed, int32_t* out,
                              uint32_t total_qo_len, uint32_t num_qo_heads,
                              uint32_t max_k_tiles, uint32_t num_valid_pages,
                              uint32_t num_rows, hggcStream_t stream) {
  constexpr int kRowsPerBlock = 8;
  dim3 grid((num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
  dim3 block(numThreadsPerBlock);
  void* args[] = {(void*)&transposed, (void*)&out, (void*)&total_qo_len,
                  (void*)&num_qo_heads, (void*)&max_k_tiles, (void*)&num_valid_pages};
  return hggcLaunchKernel((const void*)kernel, grid, block, args, 0, stream);
}

hggcError_t LaunchTransposeAndIndexerTopK(const float* in_strided, float* transposed, int32_t* out,
                                          uint32_t total_qo_len, uint32_t num_qo_heads,
                                          uint32_t max_k_tiles, uint32_t topk,
                                          uint32_t num_valid_pages,
                                          uint32_t force_begin, uint32_t force_end_start,
                                          hggcStream_t stream) {
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  if (num_rows == 0) return hggcSuccess;

  const uint32_t scan_cover = (num_valid_pages + 31u) & ~31u;
  const uint32_t transpose_k = (scan_cover < max_k_tiles) ? scan_cover : max_k_tiles;

  // ---- (1) Transpose: (H,K,T) -> (H,T,K) ----
  hggcError_t tr_err = LaunchTranspose(in_strided, transposed, total_qo_len, num_qo_heads,
                                      max_k_tiles, num_valid_pages, force_begin,
                                      force_end_start, transpose_k, stream);
  if (tr_err != hggcSuccess) return tr_err;

  // ---- (2) IndexerTopK selection ----
  // based on register usage and occupancy
  if (num_valid_pages <= 32) {
    // SmallK path: 1 element per lane, 32 shuffle rounds
    return LaunchWarpRowTopK(SparseTopKWarpRankKernel<16, 1>, transposed, out, total_qo_len,
                             num_qo_heads, max_k_tiles, num_valid_pages, num_rows, stream);
  } else if (num_valid_pages <= 64) {
    // MediumK path: 2 elements per lane, 32 shuffle rounds
    return LaunchWarpRowTopK(SparseTopKWarpRankKernel<16, 2>, transposed, out, total_qo_len,
                             num_qo_heads, max_k_tiles, num_valid_pages, num_rows, stream);
  } else if (num_valid_pages <= 96) {
    // LargeK path: 3 elements per lane, 32 ballot iterations
    return LaunchWarpRowTopK(SparseTopKWarpBallotKernel<16>, transposed, out, total_qo_len,
                              num_qo_heads, max_k_tiles, num_valid_pages, num_rows, stream);
  } else if (num_valid_pages <= 128) {
    // XLargeK path: 4 elements per lane, 32 ballot iterations
    return LaunchWarpRowTopK(SparseTopKWarpBallotKernel<16, 4>, transposed, out, total_qo_len,
                             num_qo_heads, max_k_tiles, num_valid_pages, num_rows, stream);
  } else {
    // based on smem usage
    auto kernel = IndexerTopKWithSortKernel<16>;
    const size_t dyn_smem_bytes = static_cast<size_t>(topk) * sizeof(int32_t);
    hggcError_t err = hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(dyn_smem_bytes));
    if (err != hggcSuccess) return err;
    void* args[] = {(void*)&transposed, (void*)&out, (void*)&total_qo_len,
                    (void*)&num_qo_heads, (void*)&max_k_tiles, (void*)&topk,
                    (void*)&num_valid_pages};
    dim3 grid(num_rows);
    dim3 block(kIndexerNumThreadsPerBlock);
    return hggcLaunchKernel((const void*)kernel, grid, block, args, dyn_smem_bytes, stream);
  }
}

// =============================================================================
// Workspace size — transpose_buf only: (num_qo_heads, max_k_tiles, total_qo_len) element.
// =============================================================================
inline size_t SparseTopKWorkspaceSize(uint32_t total_qo_len, uint32_t num_qo_heads,
                                      uint32_t max_k_tiles) {
  return static_cast<size_t>(num_qo_heads) * max_k_tiles * total_qo_len;
}

// =============================================================================
// bf16-input dispatcher (max_score is __nv_bfloat16).
//
// Tiering (all selection kernels are DataType-generic over the 16-bit
// float_to_ord key, so every tier enjoys the 2x bandwidth cut):
//   * K >= 256 && num_rows >= 256*num_sms
//                   -> SparseTopKStreamMinReplaceKernel<16, bf16> (no transpose;
//                      1x gmem traffic beats transpose+ballot's ~3x at scale)
//   * 16 < K <= 512 -> transpose_bf16 + SparseTopKWarpBallotKernel<16,...,bf16,KPT>
//                      (rows below the stream cutoff), KPT = ceil(K/32) in
//                      {1,2,3,4,8,16}; 16-round binary search on the 16-bit ord
//   * otherwise     -> transpose_bf16 + topk_bf16::IndexerTopKRegHistBF16Kernel<16>
//                      (register path; the fp32 side runs
//                      IndexerTopKWithSortKernel here)
// (K <= 16 goes to the identity-fill path and K >= 12288 is rejected by the
// top-level dispatcher, so K always fits the register capacity of 16384.)
// =============================================================================
inline hggcError_t SparseTopKSelectBF16(const __ppu_bfloat16* in, int32_t* out,
                                        int32_t* workspace, uint32_t total_qo_len,
                                        uint32_t num_qo_heads, uint32_t max_k_tiles,
                                        uint32_t num_valid_pages, uint32_t force_begin,
                                        uint32_t force_end, hggcStream_t stream) {
  constexpr uint32_t topk = 16;
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  const uint32_t force_end_start =
      (force_end <= num_valid_pages) ? num_valid_pages - force_end : 0;
  __ppu_bfloat16* transpose_buf = reinterpret_cast<__ppu_bfloat16*>(workspace);

  // ---- Fused streaming path (no transpose) --------------------------------
  // bf16: K >= 256 with rows >= 256*SMs stays on the streaming path (1x gmem
  // traffic beats transpose+ballot's ~3x); smaller rows go transpose+ballot.
  int num_sms = 1;
  hggcDeviceGetAttribute(&num_sms, hggcDevAttrMultiProcessorCount, 0);
  const uint32_t kNumThreads = numThreadsPerBlock * static_cast<uint32_t>(num_sms);
  if (max_k_tiles >= numThreadsPerBlock && num_rows >= kNumThreads) {
    dim3 grid((num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock);
    dim3 block(numThreadsPerBlock);
    void* args[] = {(void*)&in, (void*)&out, (void*)&total_qo_len,
                    (void*)&num_qo_heads, (void*)&max_k_tiles, (void*)&num_valid_pages,
                    (void*)&force_begin, (void*)&force_end};
    return hggcLaunchKernel((const void*)SparseTopKStreamMinReplaceKernel<topk, __ppu_bfloat16>,
                            grid, block, args, 0, stream);
  }

  // ---- Transpose (H,K,T) -> (H,T,K) bf16 into the workspace buffer ----
  hggcError_t tr_err = LaunchTranspose(in, transpose_buf, total_qo_len, num_qo_heads,
                                      max_k_tiles, num_valid_pages, force_begin,
                                      force_end_start, max_k_tiles, stream);
  if (tr_err != hggcSuccess) return tr_err;

  // ---- Small-K warp paths (16-round ballot on the 16-bit ord) -----------
  bool launched = false;
  hggcError_t err = hggcSuccess;
  auto launch_topk_warp_kernel = [&](auto kpt) {
    constexpr int KPT = decltype(kpt)::value;
    if (!launched && max_k_tiles <= KPT * 32) {
      launched = true;
      err = LaunchWarpRowTopK(SparseTopKWarpBallotKernel<16, KPT, 8, __ppu_bfloat16>, transpose_buf,
                              out, total_qo_len, num_qo_heads, max_k_tiles, num_valid_pages,
                              num_rows, stream);
    }
  };
  launch_topk_warp_kernel(std::integral_constant<int, 1>{});
  launch_topk_warp_kernel(std::integral_constant<int, 2>{});
  launch_topk_warp_kernel(std::integral_constant<int, 3>{});
  launch_topk_warp_kernel(std::integral_constant<int, 4>{});
  launch_topk_warp_kernel(std::integral_constant<int, 8>{});
  launch_topk_warp_kernel(std::integral_constant<int, 16>{});
  if (launched) return err;

  // ---- Large-K: register kernel (1024-thread block, row in registers,
  //      one GMEM pass) ------------------------------------------------------
  constexpr uint32_t kRegSmemBytes = sizeof(topk_bf16::RegisterTopKSmem) + topk * sizeof(int32_t);
  auto kernel = topk_bf16::IndexerTopKRegHistBF16Kernel<16>;
  hggcError_t attr_err = hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kRegSmemBytes));
  if (attr_err != hggcSuccess) return attr_err;
  dim3 grid(num_rows);
  dim3 block(topk_bf16::kBlockSize);
  void* args[] = {(void*)&transpose_buf, (void*)&out,
                  (void*)&total_qo_len,  (void*)&num_qo_heads,
                  (void*)&max_k_tiles,   (void*)&topk,
                  (void*)&num_valid_pages};
  return hggcLaunchKernel((const void*)kernel, grid, block, args, kRegSmemBytes, stream);
}

// =============================================================================
// Top-level dispatcher
// =============================================================================
//   in        : (num_qo_heads, max_k_tiles, total_qo_len) contiguous DataType
//   out       : (total_qo_len, num_qo_heads, topk=16) contiguous int32 asc by k_tile index
//   workspace : DataType, at least SparseTopKWorkspaceSize(...) elements
//   num_valid_pages : indices >= num_valid_pages are emitted as -1 (sorted to
//                     tail).  Pass max_k_tiles (or any value >= max_k_tiles)
//                     to disable clamping.
//
//   * num_valid_pages <= 16        → SparseTopKIdentityFillKernel (trivial)
//   * num_rows >= 256*num_sms      → SparseTopKStreamMinReplaceKernel<16>
//                                    (no transpose; checked before the nvp
//                                    tiers below, at any nvp / max_k_tiles)
//   * 16 < num_valid_pages <= 32   → Transpose + SparseTopKWarpRankKernel<16, 1>
//   * 32 < num_valid_pages <= 64   → Transpose + SparseTopKWarpRankKernel<16, 2>
//   * 64 < num_valid_pages <= 96   → Transpose + SparseTopKWarpBallotKernel<16>
//   * 96 < num_valid_pages <= 128  → Transpose + SparseTopKWarpBallotKernel<16, 4>
//   * num_valid_pages > 128 (rows below the stream cutoff)
//                                → Transpose + IndexerTopKWithSortKernel<16>
//   * max_k_tiles >= 12288         → cudaErrorNotSupported
template <typename DataType>
inline hggcError_t SparseTopKSelect(const DataType* in, int32_t* out, int32_t* workspace,
                                    uint32_t total_qo_len, uint32_t num_qo_heads,
                                    uint32_t max_k_tiles, uint32_t num_valid_pages,
                                    uint32_t force_begin, uint32_t force_end,
                                    hggcStream_t stream) {
  constexpr uint32_t topk = 16;
  const uint32_t num_rows = total_qo_len * num_qo_heads;
  static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, __ppu_bfloat16>,
                "SparseTopKSelect: DataType must be float or __ppu_bfloat16");

  if (num_rows == 0) return hggcSuccess;
  // ---- Trivial path: num_valid_pages <= topk → identity fill -----------------
  if (num_valid_pages <= topk) {
    dim3 grid(num_rows);
    dim3 block(topk);
    void* args[] = {(void*)&out, (void*)&total_qo_len, (void*)&num_qo_heads,
                    (void*)&max_k_tiles, (void*)&topk, (void*)&num_valid_pages};
    return hggcLaunchKernel((const void*)SparseTopKIdentityFillKernel, grid, block, args, 0,
                            stream);
  }

  // ---- IndexerTopK insertion-sort path ------------------------------------
  if (max_k_tiles >= 12288) return hggcErrorNotSupported;

  if constexpr (std::is_same_v<DataType, __ppu_bfloat16>){
      return SparseTopKSelectBF16(in, out, workspace,
                                  total_qo_len, num_qo_heads, max_k_tiles, num_valid_pages,
                                  force_begin, force_end, stream);
  } else {
    int num_sms = 1;
    hggcDeviceGetAttribute(&num_sms, hggcDevAttrMultiProcessorCount, 0);
    const uint32_t kNumThreads = numThreadsPerBlock * static_cast<uint32_t>(num_sms);
    if (num_rows >= kNumThreads) {
      dim3 grid((num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock);
      dim3 block(numThreadsPerBlock);
      void* args[] = {(void*)&in, (void*)&out, (void*)&total_qo_len,
                      (void*)&num_qo_heads, (void*)&max_k_tiles, (void*)&num_valid_pages,
                      (void*)&force_begin, (void*)&force_end};
      return hggcLaunchKernel((const void*)SparseTopKStreamMinReplaceKernel<topk>,
                              grid, block, args, 0, stream);
    }
    const uint32_t force_end_start =
        (force_end <= num_valid_pages) ? num_valid_pages - force_end : 0;

    float* transpose_buf = reinterpret_cast<float*>(workspace);
    return LaunchTransposeAndIndexerTopK(in, transpose_buf, out, total_qo_len, num_qo_heads,
                                        max_k_tiles, topk, num_valid_pages,
                                        force_begin, force_end_start, stream);
}

}
}  // namespace sparse_topk
}  // namespace flashinfer

#endif  // FLASHINFER_SPARSE_TOPK_SELECT_CUH_

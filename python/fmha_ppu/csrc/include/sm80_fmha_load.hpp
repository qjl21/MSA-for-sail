// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>

#include "gmem_bounds_check.h"
#include "cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_ppu.h"
#include "cutlass/cutlass.h"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "ppu_include.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// Portable size accessors: when FMHA_GMEM_BOUNDS_CHECK is off the _size
// fields don't exist on Params, so pass 0 (the macros ignore the sz arg).
#ifdef FMHA_GMEM_BOUNDS_CHECK
#define _KV_INDPTR_SZ(p)  (p).kv_page_indptr_size
#define _KV_INDICES_SZ(p) (p).kv_indices_size
#else
#define _KV_INDPTR_SZ(p)  0
#define _KV_INDICES_SZ(p) 0
#endif

template <typename T>
struct MaskPackFactor { static constexpr int value = 1; };
template <int P>
struct MaskPackFactor<PackedCausalMask<P>> { static constexpr int value = P; };

// SM80 load collective: uses cp.async to move Q/K/V tiles from gmem to smem.
// Replaces SM100 TMA loads with equivalent cp.async-based copies.
// Dense-paged page table, hoisted out of the tile loop.  init()
// pre-reads the batch-constant kv_page_indptr pair once per CTA (the
// compiler cannot hoist them above the per-tile gmem stores -- max_score
// and/or output O -- due to potential aliasing); translate() maps kv_base
// to its physical page, memoizing the kv_indices load across a tile's
// stages (same page when TileN <= PageSize).  Dense-paged modes only (the
// negation of load_K's sparse branch); translate() returns -1 for sparse
// mode or TileN > PageSize, where load_K ignores the pre-resolved page.
template <int PageSize, int TileN>
struct DensePagedPageTable {
  bool hoist = false;
  int page_start = 0, num_pages = 0;
  int phys = -1, key = -1;

  template <class LoadParams>
  CUTLASS_DEVICE void init(LoadParams const& load, int batch_idx) {
    if constexpr (PageSize > 0) {
      if (load.kv_block_indexes == nullptr || load.kv_block_num == 0) {
        if (load.kv_page_indptr != nullptr) {
          page_start = KV_INDPTR_LOAD(load.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(load));
          num_pages = KV_INDPTR_LOAD(load.kv_page_indptr, batch_idx + 1, _KV_INDPTR_SZ(load)) - page_start;
          hoist = true;
        } else if constexpr (TileN <= PageSize) {
          // No indptr: page_start = 0 and no clamp (matches load_K's
          // fallback); TileN <= PageSize keeps each tile in one page, so
          // the memoized read stays inside the batch's page table.
          hoist = true;
        }
      }
    }
  }

  template <class LoadParams>
  CUTLASS_DEVICE int translate(LoadParams const& load, int kv_base) {
    if constexpr (PageSize > 0) {
      if (!hoist) return -1;
      int page = kv_base / PageSize;
      if (load.kv_page_indptr != nullptr) {
        page = min(page, max(0, num_pages - 1));
      }
      if (load.kv_indices == nullptr) return page;
      if (key != page) {
        phys = KV_INDICES_LOAD(load.kv_indices, page_start + page, _KV_INDICES_SZ(load));
        key = page;
      }
      return phys;
    } else {
      return -1;
    }
  }
};

//
// The interface mirrors Sm100FmhaLoadTmaWarpspecialized so the mainloop and
// kernel can use it as a drop-in replacement.
template <class Element, class TileShapeQK_, class TileShapePV_,
          class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV,
          class TensorStorage, class Mask, class TileShape,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off,
          int StageCountKV_ = 2>
struct Sm80FmhaLoadCpAsync {
  using TileShapeQK = TileShapeQK_;
  using TileShapePV = TileShapePV_;
  static constexpr int StageCountKV = StageCountKV_;

  static constexpr bool kNeedV = !is_only_score(kSparseAttnMode);
  static constexpr int KVPageSize = KVPageSize_;

  // SM80 does not use TMA — use raw pointers and cp.async.
  // We keep the same Layout types as SM100 for interface compatibility.
  using ShapeQ = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using LayoutQ = cute::Layout<ShapeQ, StrideQ>;

  // Paged: (page_size, D, H_kv, total_page_num) — 4-mode flat
  // Non-paged: (N, D, (H_R, H_G)) — 3-mode hierarchical
  using ShapeK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, _1, int32_t, int32_t>,
      cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>>;
  using ShapeV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<_1, int32_t, int32_t, int32_t>,
      cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>>;
  using LayoutK = cute::Layout<ShapeK, StrideK>;
  using LayoutV = cute::Layout<ShapeV, StrideV>;

  static constexpr int kPackFactor = MaskPackFactor<Mask>::value;

  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    int pack_factor = 1;
    int q_stride_n_original = 0;
    int q_stride_h_original = 0;
    int h_r_original = 0;
  };

  struct Params {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    const Element* ptr_Q_orig = nullptr;
    int q_stride_n_orig = 0;
    int q_stride_h_orig = 0;
    int h_r_orig = 0;
    int pack_factor = 1;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& /*problem_shape*/,
                                        Arguments const& args, void* /*workspace*/) {
    return Params{
      args.ptr_Q, args.layout_Q, args.ptr_K, args.layout_K, args.ptr_V, args.layout_V,
      args.kv_indices, args.kv_page_indptr, args.kv_block_indexes, args.kv_block_num,
#ifdef FMHA_GMEM_BOUNDS_CHECK
      args.kv_page_indptr_size, args.kv_indices_size, args.kv_block_indexes_numel,
#endif
      args.ptr_Q, args.q_stride_n_original, args.q_stride_h_original, args.h_r_original,
      args.pack_factor
    };
  }

  // -----------------------------------------------------------------------
  // cp.async tile loaders
  // -----------------------------------------------------------------------

  // Load a Q tile from gmem to smem using cp.async.
  // When kPackFactor > 1, the Q tensor is logically packed (fewer heads, more tokens),
  // so we must read from the original (unpacked) Q buffer and scatter into the packed
  // smem tile.  Mirrors SM100 load_Q_cp_async packed path.
  CUTLASS_DEVICE
  static void load_Q(int q_tile_index, int qo_head_idx, int qo_segment_offset,
                     int qo_len, Params const& params,
                     TensorStorage& storage) {
    constexpr int tile_m = get<0>(TileShapeQK{});
    constexpr int tile_k = get<2>(TileShapeQK{});
    constexpr int kVecElems = 16 / sizeof(Element);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto sQ_grouped = group_modes<0, rank(SmemLayoutQ{}) - 1>(sQ);

    int tid = threadIdx.x;

    int packed_len = qo_segment_offset + qo_len;
    int valid_m = min(tile_m, packed_len - (qo_segment_offset + q_tile_index * tile_m));
    if (valid_m < 0) valid_m = 0;

    constexpr int k_iters = tile_k / kVecElems;
    int total_ops = valid_m * k_iters;

    if constexpr (kPackFactor > 1) {
      // Packed Q: read from original (unpacked) Q buffer, scatter into packed smem.
      int h_r_orig = params.h_r_orig;
      int h_r_packed = h_r_orig / kPackFactor;
      int kv_head = qo_head_idx / h_r_packed;
      int rem_h = qo_head_idx % h_r_packed;
      int tile_base = q_tile_index * tile_m;

      for (int idx = tid; idx < total_ops; idx += blockDim.x) {
        int m = idx / k_iters;
        int k = (idx % k_iters) * kVecElems;

        int packed_pos = qo_segment_offset + tile_base + m;
        int orig_pos = packed_pos / kPackFactor;
        int pack_idx = packed_pos % kPackFactor;
        int orig_head = kv_head * h_r_orig + rem_h * kPackFactor + pack_idx;

        const Element* gmem_src = params.ptr_Q_orig
            + orig_pos * params.q_stride_n_orig
            + orig_head * params.q_stride_h_orig
            + k;

        Element* smem_dst = reinterpret_cast<Element*>(&sQ_grouped(m + tile_m * k));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    } else {
      // Non-packed Q: direct copy from packed Q buffer.
      for (int idx = tid; idx < total_ops; idx += blockDim.x) {
        int m = idx / k_iters;
        int k = (idx % k_iters) * kVecElems;

        int q_pos = qo_segment_offset + q_tile_index * tile_m + m;
        const Element* gmem_src = params.ptr_Q
          + q_pos * get<0>(params.layout_Q.stride())
          + qo_head_idx * get<2, 0>(params.layout_Q.stride())
          + k;

        Element* smem_dst = reinterpret_cast<Element*>(&sQ(m, k));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    }
  }

  // Load a K tile from gmem (or paged KV) to smem using cp.async.
  // Returns the actual KV tile size (may be < tile_kv for the last tile).
  // PreResolvedPage / pg_table: hoisted dense-paged page table (mainloop's
  // kHoistPageTable). true: one translate() per call replaces the per-call
  // indptr reads and the per-row kv_indices reloads; false: inline lookup.
  template <bool PreResolvedPage = false, class PageTable>
  CUTLASS_DEVICE
  static void load_K(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params,
                     TensorStorage& storage, int stage, PageTable& pg_table) {
    constexpr int tile_n = get<1>(TileShapeQK{});
    constexpr int tile_k = get<2>(TileShapeQK{});
    constexpr int kVecElems = 16 / sizeof(Element);

    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    int tid = threadIdx.x;
    // For paged KV, each batch's pages are accessed via kv_page_indptr,
    // so the logical KV position within the batch starts at 0 (ignore
    // the cumulative kv_segment_offset).
    if constexpr (KVPageSize > 0) {
      kv_segment_offset = 0;
    }
    int total_seq = kv_segment_offset + kv_len;
    int kv_base = kv_segment_offset + kv_tile_index * tile_n + stage * (tile_n / StageCountKV);
    int valid_n = min(tile_n / StageCountKV, total_seq - kv_base);
    if (valid_n < 0) valid_n = 0;

    constexpr int k_iters = tile_k / kVecElems;
    int total_ops = valid_n * k_iters;

    if constexpr (KVPageSize > 0) {
      if (params.kv_block_indexes != nullptr && params.kv_block_num > 0) {
        // Sparse mode: kv_tile_index indexes into kv_block_indexes.
        // Assumes tile_n <= KVPageSize (one tile fits within one page).
        int num_kv_heads_sp = size<2>(params.layout_K);  // H_kv from 4D layout
        int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                            + kv_head_idx * params.kv_block_num;
        if (kv_tile_index >= params.kv_block_num) return;
        int blk = __ldg(&params.kv_block_indexes[kv_block_offset + kv_tile_index]);
        if (blk < 0) return;  // padding entry
        int page_idx = blk;

        for (int idx = tid; idx < total_ops; idx += blockDim.x) {
          int n = idx / k_iters;
          int k = (idx % k_iters) * kVecElems;

          const Element* gmem_src = params.ptr_K
            + static_cast<int64_t>(page_idx) * get<3>(params.layout_K.stride())
            + kv_head_idx * get<2>(params.layout_K.stride())
            + stage * (tile_n / StageCountKV) * get<0>(params.layout_K.stride())
            + n * get<0>(params.layout_K.stride())
            + k;

          Element* smem_dst = reinterpret_cast<Element*>(&sK(n, k, stage));
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
              smem_dst, gmem_src);
        }
      } else {
        // Paged dense: per-element page lookup (handles tile_n > KVPageSize).
        // PreResolvedPage=true implies kTileN <= KVPageSize, so the whole
        // stage subtile lives in one page: query the hoisted table once and
        // skip the per-call indptr reads and per-row kv_indices reloads.
        // pre_phys < 0 (init() declined, e.g. sparse) falls back inline.
        int pre_phys = -1;
        if constexpr (PreResolvedPage) {
          pre_phys = pg_table.translate(params, kv_base);
        }
        int kv_page_start = 0;
        int num_pages_batch = 0;
        if (pre_phys < 0) {
          kv_page_start = (params.kv_page_indptr != nullptr)
              ? KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx,
                               _KV_INDPTR_SZ(params))
              : 0;
          if (params.kv_page_indptr != nullptr) {
            int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1,
                                              _KV_INDPTR_SZ(params));
            num_pages_batch = kv_page_end - kv_page_start;
          }
        }

        for (int idx = tid; idx < total_ops; idx += blockDim.x) {
          int n = idx / k_iters;
          int k = (idx % k_iters) * kVecElems;

          int kv_pos = kv_base + n;
          int n_in_page = kv_pos % KVPageSize;
          int phys_page = pre_phys;
          if (pre_phys < 0) {
            int logical_page = kv_pos / KVPageSize;
            int page_for_lookup = logical_page;
            if (num_pages_batch > 0)
              page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
            phys_page = (params.kv_indices != nullptr)
                ? KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup,
                                  _KV_INDICES_SZ(params))
                : page_for_lookup;
          }

          const Element* gmem_src = params.ptr_K
            + static_cast<int64_t>(phys_page) * get<3>(params.layout_K.stride())
            + kv_head_idx * get<2>(params.layout_K.stride())
            + n_in_page * get<0>(params.layout_K.stride())
            + k;

          Element* smem_dst = reinterpret_cast<Element*>(&sK(n, k, stage));
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
              smem_dst, gmem_src);
        }
      }
    } else {
      // Dense (non-paged) KV
      for (int idx = tid; idx < total_ops; idx += blockDim.x) {
        int n = idx / k_iters;
        int k = (idx % k_iters) * kVecElems;

        int kv_pos = kv_base + n;
        const Element* gmem_src = params.ptr_K
          + kv_pos * get<0>(params.layout_K.stride())
          + kv_head_idx * get<2, 1>(params.layout_K.stride())
          + k;

        Element* smem_dst = reinterpret_cast<Element*>(&sK(n, k, stage));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    }
  }

  // Load a V tile from gmem (or paged KV) to smem using cp.async.
  CUTLASS_DEVICE
  static void load_V(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params,
                     TensorStorage& storage, int stage) {
    constexpr int tile_n = get<2>(TileShapePV{});  // KV dim (N)
    constexpr int tile_d = get<1>(TileShapePV{});  // head_dim (D)
    constexpr int kVecElems = 16 / sizeof(Element);

    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    int tid = threadIdx.x;
    // For paged KV, each batch's pages are accessed via kv_page_indptr,
    // so the logical KV position within the batch starts at 0.
    if constexpr (KVPageSize > 0) {
      kv_segment_offset = 0;
    }
    int total_seq = kv_segment_offset + kv_len;
    int kv_base = kv_segment_offset + kv_tile_index * tile_n + stage * (tile_n / StageCountKV);
    int valid_n = min(tile_n / StageCountKV, total_seq - kv_base);
    if (valid_n < 0) valid_n = 0;

    constexpr int k_iters = tile_d / kVecElems;
    int total_ops = valid_n * k_iters;

    if constexpr (KVPageSize > 0) {
      if (params.kv_block_indexes != nullptr && params.kv_block_num > 0) {
        // Sparse mode: kv_tile_index indexes into kv_block_indexes.
        // Assumes tile_n <= KVPageSize (one tile fits within one page).
        int num_kv_heads_sp = size<2>(params.layout_V);  // H_kv from 4D layout
        int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                            + kv_head_idx * params.kv_block_num;
        if (kv_tile_index >= params.kv_block_num) return;
        int blk = __ldg(&params.kv_block_indexes[kv_block_offset + kv_tile_index]);
        if (blk < 0) return;
        int page_idx = blk;

        for (int idx = tid; idx < total_ops; idx += blockDim.x) {
          int n = idx / k_iters;
          int d = (idx % k_iters) * kVecElems;

          const Element* gmem_src = params.ptr_V
            + static_cast<int64_t>(page_idx) * get<3>(params.layout_V.stride())
            + kv_head_idx * get<2>(params.layout_V.stride())
            + stage * (tile_n / StageCountKV) * get<1>(params.layout_V.stride())
            + n * get<1>(params.layout_V.stride())
            + d;

          Element* smem_dst = reinterpret_cast<Element*>(&sV(d, n, stage));
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
              smem_dst, gmem_src);
        }
      } else {
        // Paged dense: per-element page lookup (handles tile_n > KVPageSize)
        int kv_page_start = (params.kv_page_indptr != nullptr)
            ? KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx,
                             _KV_INDPTR_SZ(params))
            : 0;
        int num_pages_batch = 0;
        if (params.kv_page_indptr != nullptr) {
          int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1,
                                            _KV_INDPTR_SZ(params));
          num_pages_batch = kv_page_end - kv_page_start;
        }

        for (int idx = tid; idx < total_ops; idx += blockDim.x) {
          int n = idx / k_iters;
          int d = (idx % k_iters) * kVecElems;

          int kv_pos = kv_base + n;
          int logical_page = kv_pos / KVPageSize;
          int n_in_page = kv_pos % KVPageSize;
          int page_for_lookup = logical_page;
          if (num_pages_batch > 0)
            page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
          int phys_page = (params.kv_indices != nullptr)
              ? KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup,
                                _KV_INDICES_SZ(params))
              : page_for_lookup;

          const Element* gmem_src = params.ptr_V
            + static_cast<int64_t>(phys_page) * get<3>(params.layout_V.stride())
            + kv_head_idx * get<2>(params.layout_V.stride())
            + n_in_page * get<1>(params.layout_V.stride())
            + d;

          Element* smem_dst = reinterpret_cast<Element*>(&sV(d, n, stage));
          cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
              smem_dst, gmem_src);
        }
      }
    } else {
      for (int idx = tid; idx < total_ops; idx += blockDim.x) {
        int n = idx / k_iters;
        int d = (idx % k_iters) * kVecElems;

        int kv_pos = kv_base + n;
        const Element* gmem_src = params.ptr_V
          + d * get<0>(params.layout_V.stride())
          + kv_pos * get<1>(params.layout_V.stride())
          + kv_head_idx * get<2, 1>(params.layout_V.stride());

        Element* smem_dst = reinterpret_cast<Element*>(&sV(d, n, stage));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    }
  }

  // Commit all pending cp.async loads and wait for completion.
  CUTLASS_DEVICE
  static void commit_and_wait() {
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
  }

  // Commit cp.async without waiting (for pipelining).
  CUTLASS_DEVICE
  static void commit() {
    cutlass::arch::cp_async_fence();
  }

  // Wait for all pending cp.async to complete.
  CUTLASS_DEVICE
  static void wait_all() {
    cutlass::arch::cp_async_wait<0>();
  }

  template <int N>
  CUTE_HOST_DEVICE
  static void wait() {
    cutlass::arch::cp_async_wait<N>();
  }
};

// =====================================================================
// PPU AIU-based load collective (drop-in replacement for Sm80FmhaLoadCpAsync)
//
// Uses PPU0015_AIU_LOAD / PPU0010_AIU_LOAD (gmem->smem bulk copy) paired with
// PPU0015_TSM_LD_SWZL / PPU0010_TSM_LD_SWZL (smem->vreg, set up in the mainloop).
// Falls back to cp.async for packed Q (kPackFactor > 1).
// =====================================================================
#if defined(USE_PPU) && USE_AIU
template <class Element, class TileShapeQK_, class TileShapePV_,
          class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV,
          class TensorStorage, class Mask, class TileShape,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off,
          int StageCountKV_ = 2>
struct Sm80FmhaLoadAiu
    : public Sm80FmhaLoadCpAsync<Element, TileShapeQK_, TileShapePV_,
          SmemLayoutQ, SmemLayoutK, SmemLayoutV,
          TensorStorage, Mask, TileShape,
          KVPageSize_, kSparseAttnMode, StageCountKV_> {
  using Base = Sm80FmhaLoadCpAsync<Element, TileShapeQK_, TileShapePV_,
      SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      TensorStorage, Mask, TileShape,
      KVPageSize_, kSparseAttnMode, StageCountKV_>;

  using typename Base::TileShapeQK;
  using typename Base::TileShapePV;
  using typename Base::Params;
  using typename Base::Arguments;
  using typename Base::LayoutQ;
  using typename Base::LayoutK;
  using typename Base::LayoutV;

  static constexpr int KVPageSize = Base::KVPageSize;
  static constexpr int kPackFactor = Base::kPackFactor;
  static constexpr int StageCountKV = Base::StageCountKV;

  // Tile dimensions
  static constexpr int kTileM = get<0>(TileShapeQK{});
  static constexpr int kTileN = get<1>(TileShapeQK{});
  static constexpr int kTileK = get<2>(TileShapeQK{});
  static constexpr int kTileD = get<1>(TileShapePV{});
  static constexpr int kStageSplitN = kTileN / StageCountKV;

  // kBlockKGmem mirrors Sm80FmhaFwdMainloop's definition
  static constexpr int kBytePerRow = kTileK * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

  // AIU gmem tiled copy types (KV loads are stage-split)
  static constexpr int bits_per_aiu_Q = kTileM * kBlockKGmem * sizeof(Element) * 8;
  static constexpr int bits_per_aiu_KV = kStageSplitN * kBlockKGmem * sizeof(Element) * 8;

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using GmemCopyStructQ  = PPU0015_AIU_LOAD<C<bits_per_aiu_Q>,  Element, false, kTileM, kBlockKGmem>;
  using GmemCopyStructKV = PPU0015_AIU_LOAD<C<bits_per_aiu_KV>, Element, false, kStageSplitN, kBlockKGmem>;
#else
  using GmemCopyStructQ  = PPU0010_AIU_LOAD<C<bits_per_aiu_Q>,  Element, false>;
  using GmemCopyStructKV = PPU0010_AIU_LOAD<C<bits_per_aiu_KV>, Element, false>;
#endif

  using GmemTiledCopyQ = decltype(
      make_tiled_copy(Copy_Atom<GmemCopyStructQ, Element>{},
                      Layout<Shape<_1,_1>, Stride<_1,_1>>{},
                      Layout<Shape<Int<kTileM>, Int<kBlockKGmem>>>{}));
  using GmemTiledCopyKV = decltype(
      make_tiled_copy(Copy_Atom<GmemCopyStructKV, Element>{},
                      Layout<Shape<_1,_1>, Stride<_1,_1>>{},
                      Layout<Shape<Int<kStageSplitN>, Int<kBlockKGmem>>>{}));

  // -------------------------------------------------------------------
  // load_Q: AIU for dense (kPackFactor==1), fallback for packed
  // -------------------------------------------------------------------
  CUTLASS_DEVICE
  static void load_Q(int q_tile_index, int qo_head_idx, int qo_segment_offset,
                     int qo_len, Params const& params,
                     TensorStorage& storage) {
    if constexpr (kPackFactor > 1) {
      // Packed Q: scatter from unpacked layout -> full fallback
      Base::load_Q(q_tile_index, qo_head_idx, qo_segment_offset, qo_len,
                   params, storage);
      return;
    }
    // Dense Q: AIU bulk copy via mix tensor
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});

    int packed_len = qo_segment_offset + qo_len;
    int valid_m = min(kTileM, packed_len - (qo_segment_offset + q_tile_index * kTileM));
    if (valid_m <= 0) return;

    int stride_n = get<0>(params.layout_Q.stride());
    const Element* q_base = params.ptr_Q
        + (qo_segment_offset + q_tile_index * kTileM) * stride_n
        + qo_head_idx * get<2, 0>(params.layout_Q.stride());

    Tensor mQ = make_tensor(make_gmem_ptr(q_base),
        make_shape(Int<kTileM>{}, Int<kTileK>{}),
        make_stride(stride_n, _1{}));
    Tensor gQ = make_mix_tensor_like(mQ);

    GmemTiledCopyQ gmem_tiled_copy;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
    gmem_tiled_copy.desc_.init(nullptr, valid_m, kTileK, stride_n);
#else
    int aiu_offset_q = get<1>(params.layout_Q.shape()) == kTileK ? 0 : (stride_n - get<1>(params.layout_Q.shape()));
    gmem_tiled_copy.desc_ = AiuDesc{nullptr, valid_m, stride_n, kTileM, kBlockKGmem, aiu_offset_q};
#endif
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
    if (warp_idx == 0) cute::copy(gmem_tiled_copy, tQgQ, tQsQ);
  }

  // -------------------------------------------------------------------
  // load_K: AIU for dense/paged/sparse (tile_n <= page_size), fallback
  // PreResolvedPage=true: query the mainloop-hoisted pg_table (dense-paged,
  // kTileN <= KVPageSize; all dtypes); false: run the page-table query inline.
  // -------------------------------------------------------------------
  template <bool PreResolvedPage = false, class PageTable>
  CUTLASS_DEVICE
  static void load_K(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params,
                     TensorStorage& storage, int stage, PageTable& pg_table) {
    if constexpr (KVPageSize > 0) {
      kv_segment_offset = 0;
    }
    int total_seq = kv_segment_offset + kv_len;
    int kv_base = kv_segment_offset + kv_tile_index * kTileN + stage * kStageSplitN;
    int valid_n = min(kStageSplitN, total_seq - kv_base);
    if (valid_n <= 0) return;

    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});

    if constexpr (KVPageSize > 0) {
      if constexpr (kTileN > KVPageSize) {
        // tile spans multiple pages -> full fallback
        Base::load_K(kv_tile_index, kv_head_idx, batch_idx, kv_segment_offset,
                     kv_len, params, storage, stage, pg_table);
        return;
      }
      // kTileN <= KVPageSize: one tile fits in one page
      const Element* page_ptr = nullptr;
      if (params.kv_block_indexes != nullptr && params.kv_block_num > 0) {
        // Sparse mode
        int num_kv_heads_sp = size<2>(params.layout_K);
        int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                            + kv_head_idx * params.kv_block_num;
        if (kv_tile_index >= params.kv_block_num) return;
        int blk = __ldg(&params.kv_block_indexes[kv_block_offset + kv_tile_index]);
        if (blk < 0) return;
        int stride_page = get<3>(params.layout_K.stride());
        page_ptr = params.ptr_K + blk * stride_page
                   + kv_head_idx * get<2>(params.layout_K.stride());
      } else {
        // Dense-paged mode
		    // PreResolvedPage=true (dense-paged, kTileN<=KVPageSize) queries the hoisted page table.
        int phys_page = -1;
        if constexpr (PreResolvedPage) {
          phys_page = pg_table.translate(params, kv_base);
        } else {
          int kv_page_start = (params.kv_page_indptr != nullptr)
              ? KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(params))
              : 0;
          int page_for_lookup = kv_base / KVPageSize;
          int num_pages_batch = 0;
          if (params.kv_page_indptr != nullptr) {
            int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1,
                                              _KV_INDPTR_SZ(params));
            num_pages_batch = kv_page_end - kv_page_start;
            page_for_lookup = min(page_for_lookup, max(0, num_pages_batch - 1));
          }
          phys_page = (params.kv_indices != nullptr)
              ? KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup,
                                _KV_INDICES_SZ(params))
              : page_for_lookup;
        }
        int stride_page = get<3>(params.layout_K.stride());
        page_ptr = params.ptr_K + phys_page * stride_page
                   + kv_head_idx * get<2>(params.layout_K.stride());
      }
      // AIU copy from page to smem via mix tensor
      int stride_n = get<0>(params.layout_K.stride());
      page_ptr += stage * kStageSplitN * stride_n;

      Tensor mK = make_tensor(make_gmem_ptr(page_ptr),
          make_shape(Int<kStageSplitN>{}, Int<kTileK>{}),
          make_stride(stride_n, _1{}));
      Tensor gK = make_mix_tensor_like(mK);

      GmemTiledCopyKV gmem_tiled_copy;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      gmem_tiled_copy.desc_.init(nullptr, valid_n, kTileK, stride_n);
#else
      int aiu_offset_k = get<1>(params.layout_K.shape()) == kTileK ? 0 : (stride_n - get<1>(params.layout_K.shape()));
      gmem_tiled_copy.desc_ = AiuDesc{nullptr, valid_n, stride_n, kStageSplitN, kBlockKGmem, aiu_offset_k};
#endif
      auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
      Tensor tKgK = gmem_thr_copy.partition_S(gK);
      Tensor tKsK = gmem_thr_copy.partition_D(sK)(_, _, _, stage);
      int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
      if (warp_idx == 0) cute::copy(gmem_tiled_copy, tKgK, tKsK);
    } else {
      // Dense (non-paged) KV: AIU bulk copy
      int stride_n = get<0>(params.layout_K.stride());
      const Element* k_base = params.ptr_K
          + kv_base * stride_n
          + kv_head_idx * get<2, 1>(params.layout_K.stride());

      Tensor mK = make_tensor(make_gmem_ptr(k_base),
          make_shape(Int<kStageSplitN>{}, Int<kTileK>{}),
          make_stride(stride_n, _1{}));
      Tensor gK = make_mix_tensor_like(mK);

      GmemTiledCopyKV gmem_tiled_copy;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      gmem_tiled_copy.desc_.init(nullptr, valid_n, kTileK, stride_n);
#else
      int aiu_offset_k = get<1>(params.layout_K.shape()) == kTileK ? 0 : (stride_n - get<1>(params.layout_K.shape()));
      gmem_tiled_copy.desc_ = AiuDesc{nullptr, valid_n, stride_n, kStageSplitN, kBlockKGmem, aiu_offset_k};
#endif
      auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
      Tensor tKgK = gmem_thr_copy.partition_S(gK);
      Tensor tKsK = gmem_thr_copy.partition_D(sK)(_, _, _, stage);
      int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
      if (warp_idx == 0) cute::copy(gmem_tiled_copy, tKgK, tKsK);
    }
  }

  // -------------------------------------------------------------------
  // load_V: AIU for dense/paged/sparse (tile_n <= page_size), fallback
  // -------------------------------------------------------------------
  CUTLASS_DEVICE
  static void load_V(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params,
                     TensorStorage& storage, int stage) {
    if constexpr (KVPageSize > 0) {
      kv_segment_offset = 0;
    }
    int total_seq = kv_segment_offset + kv_len;
    int kv_base = kv_segment_offset + kv_tile_index * kTileN + stage * kStageSplitN;
    int valid_n = min(kStageSplitN, total_seq - kv_base);
    if (valid_n <= 0) return;

    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    if constexpr (KVPageSize > 0) {
      if constexpr (kTileN > KVPageSize) {
        Base::load_V(kv_tile_index, kv_head_idx, batch_idx, kv_segment_offset,
                     kv_len, params, storage, stage);
        return;
      }
      const Element* page_ptr = nullptr;
      if (params.kv_block_indexes != nullptr && params.kv_block_num > 0) {
        int num_kv_heads_sp = size<2>(params.layout_V);
        int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                            + kv_head_idx * params.kv_block_num;
        if (kv_tile_index >= params.kv_block_num) return;
        int blk = __ldg(&params.kv_block_indexes[kv_block_offset + kv_tile_index]);
        if (blk < 0) return;
        int stride_page = get<3>(params.layout_V.stride());
        page_ptr = params.ptr_V + blk * stride_page
                   + kv_head_idx * get<2>(params.layout_V.stride());
      } else {
        int kv_page_start = (params.kv_page_indptr != nullptr)
            ? KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(params))
            : 0;
        int page_for_lookup = kv_base / KVPageSize;
        int num_pages_batch = 0;
        if (params.kv_page_indptr != nullptr) {
          int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1,
                                            _KV_INDPTR_SZ(params));
          num_pages_batch = kv_page_end - kv_page_start;
          page_for_lookup = min(page_for_lookup, max(0, num_pages_batch - 1));
        }
        int phys_page = (params.kv_indices != nullptr)
            ? KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup,
                              _KV_INDICES_SZ(params))
            : page_for_lookup;
        int stride_page = get<3>(params.layout_V.stride());
        page_ptr = params.ptr_V + phys_page * stride_page
                   + kv_head_idx * get<2>(params.layout_V.stride());
      }
      // AIU copy from page to smem via mix tensor
      // V layout: (D, N, H_kv, pages) with stride (_1, stride_n, stride_h, stride_page)
      int stride_n = get<1>(params.layout_V.stride());
      page_ptr += stage * kStageSplitN * stride_n;

      Tensor mV = make_tensor(make_gmem_ptr(page_ptr),
          make_shape(Int<kStageSplitN>{}, Int<kTileD>{}),
          make_stride(stride_n, _1{}));
      Tensor gV = make_mix_tensor_like(mV);

      GmemTiledCopyKV gmem_tiled_copy;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      gmem_tiled_copy.desc_.init(nullptr, valid_n, kTileD, stride_n);
#else
      int aiu_offset_v = get<0>(params.layout_V.shape()) == kTileD ? 0 : (stride_n - get<0>(params.layout_V.shape()));
      gmem_tiled_copy.desc_ = AiuDesc{nullptr, valid_n, stride_n, kStageSplitN, kBlockKGmem, aiu_offset_v};
#endif
      auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
      Tensor tVgV = gmem_thr_copy.partition_S(gV);
      Tensor tVsV = gmem_thr_copy.partition_D(sV)(_, _, _, stage);
      int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
      if (warp_idx == 0) cute::copy(gmem_tiled_copy, tVgV, tVsV);
    } else {
      // Dense (non-paged) V: AIU bulk copy
      // V layout: (D, N, (H_R, H_G)) with stride (_1, stride_n, (_0, stride_h))
      int stride_n = get<1>(params.layout_V.stride());
      const Element* v_base = params.ptr_V
          + kv_base * stride_n
          + kv_head_idx * get<2, 1>(params.layout_V.stride());

      Tensor mV = make_tensor(make_gmem_ptr(v_base),
          make_shape(Int<kStageSplitN>{}, Int<kTileD>{}),
          make_stride(stride_n, _1{}));
      Tensor gV = make_mix_tensor_like(mV);

      GmemTiledCopyKV gmem_tiled_copy;
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      gmem_tiled_copy.desc_.init(nullptr, valid_n, kTileD, stride_n);
#else
      int aiu_offset_v = get<0>(params.layout_V.shape()) == kTileD ? 0 : (stride_n - get<0>(params.layout_V.shape()));
      gmem_tiled_copy.desc_ = AiuDesc{nullptr, valid_n, stride_n, kStageSplitN, kBlockKGmem, aiu_offset_v};
#endif
      auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
      Tensor tVgV = gmem_thr_copy.partition_S(gV);
      Tensor tVsV = gmem_thr_copy.partition_D(sV)(_, _, _, stage);
      int warp_idx = __ppu_read_firstlane(threadIdx.x / 32);
      if (warp_idx == 0) cute::copy(gmem_tiled_copy, tVgV, tVsV);
    }
  }

  // commit_and_wait, commit, wait_all are inherited from Base (same cp.async pipeline)
};
#endif  // USE_PPU && USE_AIU

template <class Element, class TileShapeQK_, class TileShapePV_,
          class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV,
          class TensorStorage, class Mask, class TileShape,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off>
struct Sm80SparseFmhaLoadAiu {
  using TileShapeQK = TileShapeQK_;
  using TileShapePV = TileShapePV_;

  static constexpr bool kNeedV = !is_only_score(kSparseAttnMode);
  static constexpr int KVPageSize = KVPageSize_;

  // Same Layout types as Sm80FmhaLoadCpAsync for interface compatibility.
  using ShapeQ = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using LayoutQ = cute::Layout<ShapeQ, StrideQ>;

  using ShapeK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, _1, int32_t, int32_t>,
      cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>>;
  using ShapeV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<_1, int32_t, int32_t, int32_t>,
      cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>>;
  using LayoutK = cute::Layout<ShapeK, StrideK>;
  using LayoutV = cute::Layout<ShapeV, StrideV>;

  static constexpr int kPackFactor = MaskPackFactor<Mask>::value;

  // Field-for-field identical to Sm80FmhaLoadCpAsync::Arguments/Params so the
  // aggregate-initialization call sites in fmha_cutlass_sm80.cuh keep working
  // unchanged for the dedicated Sparse mainloop.
  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    int pack_factor = 1;
    int q_stride_n_original = 0;
    int q_stride_h_original = 0;
    int h_r_original = 0;
  };

  struct Params {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    const Element* ptr_Q_orig = nullptr;
    int q_stride_n_orig = 0;
    int q_stride_h_orig = 0;
    int h_r_orig = 0;
    int pack_factor = 1;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& /*problem_shape*/,
                                        Arguments const& args, void* /*workspace*/) {
    return Params{
      args.ptr_Q, args.layout_Q, args.ptr_K, args.layout_K, args.ptr_V, args.layout_V,
      args.kv_indices, args.kv_page_indptr, args.kv_block_indexes, args.kv_block_num,
#ifdef FMHA_GMEM_BOUNDS_CHECK
      args.kv_page_indptr_size, args.kv_indices_size, args.kv_block_indexes_numel,
#endif
      args.ptr_Q, args.q_stride_n_original, args.q_stride_h_original, args.h_r_original,
      args.pack_factor
    };
  }

  // -----------------------------------------------------------------------
  // AIU tile loaders. Tile dims mirror the mainloop's kTileM/kTileN/kTileK/
  // kTileD/kBlockKGmem so the AIU_LOAD atom's transfer granularity matches
  // what the mainloop's TSM_LD_SWZL smem-copy-atom expects to read back.
  // -----------------------------------------------------------------------
  static constexpr int kTileM = CUTE_STATIC_V(cute::get<0>(TileShapeQK{}));
  static constexpr int kTileN = CUTE_STATIC_V(cute::get<1>(TileShapeQK{}));
  static constexpr int kTileK = CUTE_STATIC_V(cute::get<2>(TileShapeQK{}));
  static constexpr int kTileD = CUTE_STATIC_V(cute::get<1>(TileShapePV{}));
  static constexpr int kBytePerRow = kTileK * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

  static constexpr int kBitsPerAiuQ = kTileM * kBlockKGmem * sizeof(Element) * 8;
  static constexpr int kBitsPerAiuK = kTileN * kBlockKGmem * sizeof(Element) * 8;
  static constexpr int kBitsPerAiuV = kTileN * kBlockKGmem * sizeof(Element) * 8;

  static constexpr bool IsFP8 = std::is_same_v<Element, cutlass::float_e4m3_t>;
  static_assert(sizeof(Element) != 1 || IsFP8, "Sparse FP8 supports only E4M3FN");
  static_assert(!IsFP8 || (KVPageSize == 128 && kTileK == 128 && kTileD == 128 &&
                          (kTileN == 64 || kTileN == 128) && kBlockKGmem == 128),
                "Sparse FP8 AIU requires page/head/cube size 128 and full N of 64 or 128");
#if defined(__HGGC_ARCH__)
  static_assert(!IsFP8 || __HGGC_ARCH__ == 150, "Sparse FP8 supports only PPU0015");
#endif
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  static constexpr bool UsePpu15Aiu = true;
#else
  static constexpr bool UsePpu15Aiu = IsFP8;
#endif
  using GmemCopyStructQ = std::conditional_t<UsePpu15Aiu,
      PPU0015_AIU_LOAD<C<kBitsPerAiuQ>, Element, false, kTileM, kBlockKGmem>,
      PPU0010_AIU_LOAD<C<kBitsPerAiuQ>, Element, false>>;
  using GmemCopyStructK = std::conditional_t<UsePpu15Aiu,
      PPU0015_AIU_LOAD<C<kBitsPerAiuK>, Element, false, kTileN, kBlockKGmem>,
      PPU0010_AIU_LOAD<C<kBitsPerAiuK>, Element, false>>;
  using GmemCopyStructV = std::conditional_t<UsePpu15Aiu,
      PPU0015_AIU_LOAD<C<kBitsPerAiuV>, Element, false, kTileN, kBlockKGmem>,
      PPU0010_AIU_LOAD<C<kBitsPerAiuV>, Element, false>>;

  using GmemTiledCopyQ = decltype(make_tiled_copy(Copy_Atom<GmemCopyStructQ, Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{}, Layout<Shape<Int<kTileM>, Int<kBlockKGmem>>>{}));
  using GmemTiledCopyK = decltype(make_tiled_copy(Copy_Atom<GmemCopyStructK, Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{}, Layout<Shape<Int<kTileN>, Int<kBlockKGmem>>>{}));
  using GmemTiledCopyV = decltype(make_tiled_copy(Copy_Atom<GmemCopyStructV, Element>{},
      Layout<Shape<_1, _1>, Stride<_1, _1>>{}, Layout<Shape<Int<kTileN>, Int<kBlockKGmem>>>{}));

  struct KVLoadState {
    const Element* const ptr_K_head;
    const Element* const ptr_V_head;
    const int k_row_stride;
    const int v_row_stride;
    const int k_page_stride;
    const int v_page_stride;
    GmemTiledCopyK tiled_copy_K;
    GmemTiledCopyV tiled_copy_V;

    CUTLASS_DEVICE
    KVLoadState(Params const& params, int kv_head_idx)
        : ptr_K_head(params.ptr_K + static_cast<int64_t>(kv_head_idx) * get<2>(params.layout_K.stride())),
          ptr_V_head(params.ptr_V + static_cast<int64_t>(kv_head_idx) * get<2>(params.layout_V.stride())),
          k_row_stride(get<0>(params.layout_K.stride())),
          v_row_stride(get<1>(params.layout_V.stride())),
          k_page_stride(get<3>(params.layout_K.stride())),
          v_page_stride(get<3>(params.layout_V.stride())) {
      static_assert(KVPageSize > 0, "Sparse KVLoadState requires paged KV");
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      tiled_copy_K.desc_.init(nullptr, kTileN, kTileK, k_row_stride);
      tiled_copy_V.desc_.init(nullptr, kTileN, kTileD, v_row_stride);
#else
      tiled_copy_K.desc_ = AiuDesc{nullptr, kTileN, k_row_stride, kTileN, kBlockKGmem, 0};
      tiled_copy_V.desc_ = AiuDesc{nullptr, kTileN, v_row_stride, kTileN, kBlockKGmem, 0};
#endif
    }
  };

  // Load a Q tile from gmem to smem.  All pack_factor values use AIU_LOAD bulk
  // copy so that TSM_LD_SWZL reads a matching swizzled layout.  Packed-Q
  // (kPackFactor > 1) in per-token sparse is contiguous in the original Q
  // buffer (pf consecutive heads of one token), so it also uses AIU bulk copy.
  CUTLASS_DEVICE
  static void load_Q(int q_tile_index, int qo_head_idx, int qo_segment_offset,
                     int qo_len, Params const& params,
                     TensorStorage& storage) {
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});

    if constexpr (kPackFactor > 1) {
      int warp_idx = threadIdx.x / 32;
      int h_r_orig = params.h_r_orig;
      int h_r_packed = h_r_orig / kPackFactor;
      int kv_head = qo_head_idx / h_r_packed;
      int rem_h = qo_head_idx % h_r_packed;
      int base = kv_head * h_r_orig + rem_h * kPackFactor;
      int orig_pos = qo_segment_offset / kPackFactor;  // original token index

      int valid_m = min(kTileM, qo_len - q_tile_index * kTileM);
      if (valid_m <= 0) return;

      const Element* q_gmem = params.ptr_Q_orig
          + orig_pos * params.q_stride_n_orig
          + base * params.q_stride_h_orig;
      Tensor gQ = make_tensor(make_gmem_ptr(q_gmem),
          make_shape(Int<kTileM>{}, Int<kTileK>{}),
          make_stride(params.q_stride_h_orig, _1{}));

      GmemTiledCopyQ gmem_tiled_copy_Q;
      auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(0);
      if constexpr (UsePpu15Aiu) {
        gmem_tiled_copy_Q.desc_.init(nullptr, valid_m, kTileK, params.q_stride_h_orig);
      } else {
        gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, valid_m, params.q_stride_h_orig, kTileM, kBlockKGmem, 0};
      }
      Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
      Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
      if (warp_idx == 0) cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);
      return;
    }

    int warp_idx = threadIdx.x / 32;

    // Align with dense Sm80FmhaLoadAiu::load_Q: use valid_m and actual
    // gmem stride instead of fixed kTileM / kTileK.
    int packed_len = qo_segment_offset + qo_len;
    int valid_m = min(kTileM, packed_len - (qo_segment_offset + q_tile_index * kTileM));
    if (valid_m <= 0) return;

    int stride_n = get<0>(params.layout_Q.stride());
    const Element* q_gmem = params.ptr_Q
        + (qo_segment_offset + q_tile_index * kTileM) * stride_n
        + qo_head_idx * get<2, 0>(params.layout_Q.stride());
    Tensor gQ = make_tensor(make_gmem_ptr(q_gmem),
        make_shape(Int<kTileM>{}, Int<kTileK>{}),
        make_stride(stride_n, _1{}));

    GmemTiledCopyQ gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(0);
    if constexpr (UsePpu15Aiu) {
      gmem_tiled_copy_Q.desc_.init(nullptr, valid_m, kTileK, stride_n);
    } else {
      int aiu_offset_q = get<1>(params.layout_Q.shape()) == kTileK ? 0 : (stride_n - get<1>(params.layout_Q.shape()));
      gmem_tiled_copy_Q.desc_ = AiuDesc{nullptr, valid_m, stride_n, kTileM, kBlockKGmem, aiu_offset_q};
    }
    Tensor tQgQ = gmem_thr_copy_Q.partition_S(make_mix_tensor_like(gQ));
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
    if (warp_idx == 0) cute::copy(gmem_tiled_copy_Q, tQgQ, tQsQ);
  }

  CUTLASS_DEVICE
  static void load_K(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params, KVLoadState const& kv_load,
                     TensorStorage& storage, int stage) {
    constexpr int kTilesPerPage = (KVPageSize > 0 && kTileN < KVPageSize)
        ? KVPageSize / kTileN : 1;
    int page_tile_idx = kv_tile_index / kTilesPerPage;

    if (page_tile_idx >= params.kv_block_num) return;
    int num_kv_heads_sp = size<2>(params.layout_K);
    int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                        + kv_head_idx * params.kv_block_num;
    int blk = __ldg(params.kv_block_indexes + (kv_block_offset + page_tile_idx));
    if (blk < 0) return;  // padding entry
    int page_idx = blk;
    if constexpr (KVPageSize > 0) {
      if (params.kv_page_indptr != nullptr && params.kv_indices != nullptr) {
        int page_begin = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(params));
        page_idx = KV_INDICES_LOAD(params.kv_indices, page_begin + blk, _KV_INDICES_SZ(params));
      }
    }
    load_K_page(kv_tile_index, page_idx, kv_load, storage, stage);
  }

  CUTLASS_DEVICE
  static void load_V(int kv_tile_index, int kv_head_idx, int batch_idx,
                     int kv_segment_offset, int kv_len,
                     Params const& params, KVLoadState const& kv_load,
                     TensorStorage& storage, int stage) {
    constexpr int kTilesPerPage = (KVPageSize > 0 && kTileN < KVPageSize)
        ? KVPageSize / kTileN : 1;
    int page_tile_idx = kv_tile_index / kTilesPerPage;

    if (page_tile_idx >= params.kv_block_num) return;
    int num_kv_heads_sp = size<2>(params.layout_V);
    int kv_block_offset = batch_idx * num_kv_heads_sp * params.kv_block_num
                        + kv_head_idx * params.kv_block_num;
    int blk = __ldg(params.kv_block_indexes + (kv_block_offset + page_tile_idx));
    if (blk < 0) return;  // padding entry
    int page_idx = blk;
    if constexpr (KVPageSize > 0) {
      if (params.kv_page_indptr != nullptr && params.kv_indices != nullptr) {
        int page_begin = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(params));
        page_idx = KV_INDICES_LOAD(params.kv_indices, page_begin + blk, _KV_INDICES_SZ(params));
      }
    }
    load_V_page(kv_tile_index, page_idx, kv_load, storage, stage);
  }

  CUTLASS_DEVICE
  static void load_K_page(int kv_tile_index, int page_idx,
                          KVLoadState const& kv_load,
                          TensorStorage& storage, int stage) {
    if (page_idx < 0) return;
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sK_stage = sK(_, _, stage);
    int warp_idx = threadIdx.x / 32;

    constexpr int kTilesPerPage = (KVPageSize > 0 && kTileN < KVPageSize)
        ? KVPageSize / kTileN : 1;
    int offset_in_page = (kv_tile_index % kTilesPerPage) * kTileN;

    const Element* k_gmem = kv_load.ptr_K_head
        + static_cast<int64_t>(page_idx) * kv_load.k_page_stride
        + offset_in_page * kv_load.k_row_stride;
    Tensor gK = make_tensor(make_gmem_ptr(k_gmem),
        Shape<Int<kTileN>, Int<kTileK>>{}, make_stride(kv_load.k_row_stride, _1{}));

    auto const& gmem_tiled_copy_K = kv_load.tiled_copy_K;
    auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(0);
    Tensor tKgK = gmem_thr_copy_K.partition_S(make_mix_tensor_like(gK));
    Tensor tKsK = gmem_thr_copy_K.partition_D(sK_stage);
    if (warp_idx == 0) cute::copy(gmem_tiled_copy_K, tKgK, tKsK);
  }

  // V variant of load_K_page (same pre-resolved-page contract).
  CUTLASS_DEVICE
  static void load_V_page(int kv_tile_index, int page_idx,
                          KVLoadState const& kv_load,
                          TensorStorage& storage, int stage) {
    if (page_idx < 0) return;
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    Tensor sV_stage = sV(_, _, stage);
    int warp_idx = threadIdx.x / 32;

    constexpr int kTilesPerPage = (KVPageSize > 0 && kTileN < KVPageSize)
        ? KVPageSize / kTileN : 1;
    int offset_in_page = (kv_tile_index % kTilesPerPage) * kTileN;

    const Element* v_gmem = kv_load.ptr_V_head
        + static_cast<int64_t>(page_idx) * kv_load.v_page_stride
        + offset_in_page * kv_load.v_row_stride;
    Tensor gV = make_tensor(make_gmem_ptr(v_gmem),
        Shape<Int<kTileN>, Int<kTileD>>{}, make_stride(kv_load.v_row_stride, _1{}));

    auto const& gmem_tiled_copy_V = kv_load.tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(0);
    Tensor tVgV = gmem_thr_copy_V.partition_S(make_mix_tensor_like(gV));
    Tensor tVsV = gmem_thr_copy_V.partition_D(sV_stage);
    if (warp_idx == 0) cute::copy(gmem_tiled_copy_V, tVgV, tVsV);
  }

  // AIU_LOAD is issued through the same async-copy queue as cp.async on PPU
  // (see fmha_sm80_sparse.cu's established pattern), so commit/wait reuse
  // the identical cp.async fence/wait primitives as Sm80FmhaLoadCpAsync.
  CUTLASS_DEVICE
  static void commit_and_wait() {
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
  }

  CUTLASS_DEVICE
  static void commit() {
    cutlass::arch::cp_async_fence();
  }

  template <int N>
  CUTLASS_DEVICE
  static void wait_group() {
    cutlass::arch::cp_async_wait<N>();
  }

  CUTLASS_DEVICE
  static void wait_all() {
    cutlass::arch::cp_async_wait<0>();
  }
};


}  // namespace cutlass::fmha::collective

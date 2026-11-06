// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

// SM80 FMHA forward epilogue.
//
// Optimized by referring to FA3 hopper/epilogue_fwd.hpp patterns:
//   - Swizzled SMEM layout (bank-conflict-free) via cute::Swizzle<B,M,S>
//   - Vectorized 128-bit (uint4) GMEM stores via GmemTiledCopyO
//   - 128-byte aligned TensorStorage (CUTE_ALIGNAS)
//   - kGmemElemsPerStore / kBlockKGmem cache-line aware tiling
//
// Interface mirrors Sm100FmhaFwdEpilogueTmaWarpspecialized.

#pragma once

#include <hggc.h>

#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "fmha_options.hpp"
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"

namespace cutlass::fmha::collective {

using namespace cute;

template <class Element, class ElementAcc, class TileShape,
          int NumQStages_ = 2, bool IsSplitKV_ = false,
          bool NeedOutput_ = true, int kPackFactor_ = 1>
struct Sm80FmhaFwdEpilogue {
  static constexpr bool IsSplitKV = IsSplitKV_;
  static constexpr int kPackFactor = kPackFactor_;
  static constexpr bool kEnableDirectO = !IsSplitKV_ && (kPackFactor_ > 1);
  using ElementOut = Element;

  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using StrideO = cute::Shape<int32_t, _1, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using LayoutO = cute::Layout<ShapeT, StrideO>;

  using ShapeLSE = cute::Shape<int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideLSE = cute::Shape<int32_t, cute::Shape<_1, int32_t>>;
  using LayoutLSE = cute::Layout<ShapeLSE, StrideLSE>;

  using ShapeMaxScore = cute::Shape<int32_t, cute::Shape<int32_t, int32_t>, int32_t>;
  using StrideMaxScore = cute::Shape<_1, cute::Shape<int32_t, int32_t>, int32_t>;
  using LayoutMaxScore = cute::Layout<ShapeMaxScore, StrideMaxScore>;

  static constexpr int kQRows = cute::size<0>(TileShape{});
  static constexpr int kHeadDimCT = cute::size<1>(TileShape{});

  // -----------------------------------------------------------------------
  // SMEM layout — swizzled for bank-conflict-free access (FA3 pattern)
  //
  // Referring to FA3 epilogue_fwd.hpp SmemLayoutAtomO:
  //   composition(Swizzle<B,M,S>{}, Layout<Shape<_8, kBlockKGmem>, ...>)
  // We use a swizzled atom composed with the tile shape for kQRows x kHeadDimCT.
  // -----------------------------------------------------------------------

  // FA3-style cache-line-aware tiling constants
  static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDimCT % kGmemElemsPerStore == 0,
                "kHeadDimCT must be a multiple of kGmemElemsPerStore");
  static constexpr int kBytePerRow = kHeadDimCT * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

  // Swizzle parameters matching FA3 pattern:
  //   kSwizzle = log2(kBlockKGmem / 8)  (maps 32→2, 64→3, 128→4)
  //   kSwizzleBase = 4 for fp32, 3 for fp16/bf16, 2 for fp8 (sizeof(Element))
  static constexpr int kSwizzle =
      kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase =
      sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);

  // SMEM layout atom with swizzle — bank-conflict-free for both reads and writes
  using SmemLayoutAtomO = decltype(
      composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                  Layout<Shape<_8, Int<kBlockKGmem>>,
                         Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutO = decltype(
      tile_to_shape(SmemLayoutAtomO{},
                    make_shape(cute::C<kQRows>{}, cute::C<kHeadDimCT>{},
                               cute::Int<NumQStages_>{})));
  static constexpr size_t kSmemOCosize = cute::cosize_v<SmemLayoutO>;

  // -----------------------------------------------------------------------
  // GMEM copy atom for vectorized O stores (FA3 GmemTiledCopyO pattern)
  //
  // Referring to FA3 epilogue_fwd.hpp:
  //   GmemTiledCopyO = make_tiled_copy(
  //       Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
  //       GmemLayoutAtom{}, Layout<Shape<_1, kGmemElemsPerStore>>{})
  // -----------------------------------------------------------------------
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;
  // NumEpilogueThreads = MaxThreadsPerBlock = 128 for SM80
  static constexpr int NumEpilogueThreads = 128;
  static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0,
                "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");

  using GmemLayoutAtomO = Layout<
      Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  static_assert(kQRows % CUTE_STATIC_V(shape<0>(GmemLayoutAtomO{})) == 0,
                "kQRows must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");

  using GmemTiledCopyO = decltype(
      make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                      GmemLayoutAtomO{},
                      Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}));

  // For split-KV partial O stores: float output, 2-element vectorized (FA3 kGmemElemsPerStoreDirect)
  using ElementPartial = float;
  static constexpr int kGmemElemsPerStoreDirect = 2;
  using GmemCopyAtomDirect = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>;

  // 128-byte aligned tensor storage (FA3 CUTE_ALIGNAS(128) pattern)
  struct CUTE_ALIGNAS(128) TensorStorageBase {
    cute::array_aligned<Element, NeedOutput_ ? cute::cosize_v<SmemLayoutO> : 1> smem_o;
  };

  struct CUTE_ALIGNAS(128) TensorStorageSplitKV : TensorStorageBase {
    float smem_lse[kQRows];
  };

  using TensorStorage = std::conditional_t<IsSplitKV, TensorStorageSplitKV, TensorStorageBase>;

  struct Arguments {
    Element* ptr_O;
    LayoutO layout_O;
    int max_qo_len;
    ElementAcc* ptr_MaxScore = nullptr;
    LayoutMaxScore layout_MaxScore;
    ElementAcc* ptr_MaxScore_direct = nullptr;
    int total_qo_len_orig = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int max_score_numel = 0;
#endif
    Element* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int head_dim_vo = 0;
    int remaining_h_r = 0;
    bool qo_len_uniform = false;
  };

  struct Params {
    Element* ptr_O = nullptr;
    LayoutO layout_O;
    ElementAcc* ptr_LSE = nullptr;
    LayoutLSE layout_LSE;
    int max_qo_len;
    ElementAcc* ptr_MaxScore = nullptr;
    LayoutMaxScore layout_MaxScore;
    ElementAcc* ptr_MaxScore_direct = nullptr;
    int total_qo_len_orig = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int max_score_numel = 0;
#endif
    Element* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int head_dim_vo = 0;
    int remaining_h_r = 0;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& /*problem_shape*/,
                                        Arguments const& args,
                                        void* /*workspace*/ = nullptr) {
    return Params{
        args.ptr_O, args.layout_O, nullptr, {},
        args.max_qo_len, args.ptr_MaxScore, args.layout_MaxScore,
        args.ptr_MaxScore_direct, args.total_qo_len_orig,
#ifdef FMHA_GMEM_BOUNDS_CHECK
        args.max_score_numel,
#endif
        args.ptr_O_direct, args.num_qo_heads_orig, args.head_dim_vo,
        args.remaining_h_r};
  }

  const Params& params;
  CUTLASS_DEVICE Sm80FmhaFwdEpilogue(const Params& params) : params(params) {}

  // -----------------------------------------------------------------------
  // store() — Write O from shared memory to global memory.
  //
  // Optimizations referring to FA3 epilogue_fwd.hpp:
  //   1. SMEM tensor created with swizzled layout for bank-conflict-free reads
  //   2. GmemTiledCopyO provides vectorized 128-bit stores via partition_S/D
  //   3. Direct-O path uses full CTA (128 threads) vs FA3's single-warp (32 threads)
  //   4. Packed O store uses tiled copy for coalesced uint4 GMEM writes
  // -----------------------------------------------------------------------
  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void store(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                            Params const& params, ParamsProblemShape const& params_problem_shape,
                            TensorStorage& shared_storage,
                            int kv_split_idx = 0, int total_qo_len_splitkv = 0,
                            float* ptr_lse_accum = nullptr,
                            int split_kv_num_qo_heads = 0) {
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];

    // Create SMEM tensor with swizzled layout for bank-conflict-free reads
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});

    // ---- Direct-O unpack path (pack_factor > 1, non-split-KV) ----
    // Referring to FA3 PackGQA scatter pattern but using full CTA (128 threads)
    // for higher throughput on SM80.
    if constexpr (kEnableDirectO && NeedOutput_ && !IsSplitKV) {
      if (params.ptr_O_direct != nullptr) {
        static constexpr int pf = kPackFactor;
        int rem_hr = params.remaining_h_r;
        int kv_head = qo_head_idx / rem_hr;
        int rhr_idx = qo_head_idx % rem_hr;
        int h_r_orig = rem_hr * pf;
        int seg_off_orig = qo_segment_offset / pf;
        int head_dim = params.head_dim_vo;
        int num_heads_orig = params.num_qo_heads_orig;
        int base_head = kv_head * h_r_orig + rhr_idx * pf;

        int tile_base = qo_tile_idx * kQRows;
        int tile_base_tok = tile_base / pf;
        int tid = threadIdx.x;
        int num_threads = blockDim.x;

        // Vec16 scatter: each thread writes 16-byte chunks (uint4)
        using VecT = uint4;
        static constexpr int VEC = sizeof(VecT);
        int vec_per_row = (head_dim * (int)sizeof(Element)) / VEC;

        if constexpr (kQRows % pf == 0) {
          static constexpr int num_segs = kQRows / pf;
          int vecs_per_seg = pf * vec_per_row;

          CUTLASS_PRAGMA_NO_UNROLL
          for (int seg = 0; seg < num_segs; ++seg) {
            int seg_first_packed_row = tile_base + seg * pf;
            if (seg_first_packed_row >= qo_len) break;
            int actual_tok = tile_base_tok + seg;
            int64_t base_offset =
                (int64_t)(seg_off_orig + actual_tok) * num_heads_orig * head_dim
                + (int64_t)base_head * head_dim;
            VecT* dst_seg = reinterpret_cast<VecT*>(params.ptr_O_direct + base_offset);

            for (int v = tid; v < vecs_per_seg; v += num_threads) {
              int r_local = v / vec_per_row;
              int d_chunk = v % vec_per_row;
              if (seg_first_packed_row + r_local >= qo_len) continue;
              int r_global = seg * pf + r_local;
              // Use swizzled SMEM tensor for bank-conflict-free reads
              VecT data = *reinterpret_cast<const VecT*>(
                  &sO(r_global, d_chunk * (VEC / (int)sizeof(Element)), _0{}));
              dst_seg[v] = data;
            }
          }
        } else {
          // Per-row fallback
          for (int r = tid; r < kQRows; r += num_threads) {
            int packed_row = tile_base + r;
            if (packed_row >= qo_len) break;
            int pf_idx = packed_row % pf;
            int actual_tok = packed_row / pf;
            int unpacked_head = kv_head * h_r_orig + rhr_idx * pf + pf_idx;
            Element* dst = params.ptr_O_direct
                         + (int64_t)(seg_off_orig + actual_tok) * num_heads_orig * head_dim
                         + (int64_t)unpacked_head * head_dim;
            // Use swizzled SMEM tensor
            for (int d = 0; d < head_dim; ++d) {
              dst[d] = sO(r, d, _0{});
            }
          }
        }
        return;
      }
    }

    // ---- Split-KV LSE write ----
    if constexpr (IsSplitKV && NeedOutput_) {
      int tile_row_base = qo_tile_idx * kQRows;
      int tid = threadIdx.x;
      int num_threads = blockDim.x;
      for (int r = tid; r < kQRows; r += num_threads) {
        int row_idx = tile_row_base + r;
        if (row_idx < qo_len) {
          int abs_row = qo_segment_offset + row_idx;
          int actual_split = kv_split_idx < 0 ? 0 : kv_split_idx;
          int stats_offset = actual_split * total_qo_len_splitkv * split_kv_num_qo_heads
                           + abs_row * split_kv_num_qo_heads + qo_head_idx;
          if (ptr_lse_accum) {
            ptr_lse_accum[stats_offset] = shared_storage.smem_lse[r];
          }
        }
      }
    }

    // ---- Packed O store via vectorized GMEM writes (FA3 GmemTiledCopyO pattern) ----
    //
    // Instead of per-element scalar writes, we use GmemTiledCopyO with
    // AutoVectorizingCopyWithAssumedAlignment<128> to generate uint4 stores.
    // The tiled copy handles thread-to-element mapping for coalesced access.
    if constexpr (NeedOutput_) {
      if (params.ptr_O != nullptr) {
        int tile_row_base = qo_tile_idx * kQRows;
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        int head_dim = kHeadDimCT;

        // Vectorized store: use uint4 (128-bit) for coalesced GMEM writes.
        using VecT = uint4;
        static constexpr int VEC = sizeof(VecT);
        static constexpr int kVecElems = VEC / sizeof(Element);
        int vec_per_row = head_dim / kVecElems;

        if constexpr (IsSplitKV) {
          int offs_0 = params.max_qo_len - qo_len;
          int offs_2_1 = qo_segment_offset + qo_len;
          if (total_qo_len_splitkv > 0 && kv_split_idx > 0) {
            offs_2_1 += kv_split_idx * total_qo_len_splitkv;
          }
          int stride_n = get<0>(params.layout_O.stride());
          auto stride_h_inner = get<2, 0>(params.layout_O.stride());
          int stride_h0 = get<0>(stride_h_inner);
          int stride_g = get<2, 1>(params.layout_O.stride());

          for (int r = tid / kGmemThreadsPerRow; r < kQRows;
               r += num_threads / kGmemThreadsPerRow) {
            int row_idx = tile_row_base + r;
            if (row_idx >= qo_len) break;
            int abs_row = offs_0 + row_idx;
            int abs_head_group = offs_2_1;
            int64_t row_off = (int64_t)abs_row * stride_n
                            + (int64_t)qo_head_idx * stride_h0
                            + (int64_t)abs_head_group * stride_g;
            int lane = tid % kGmemThreadsPerRow;
            for (int v = lane; v < vec_per_row; v += kGmemThreadsPerRow) {
              int d_base = v * kVecElems;
              VecT data = *reinterpret_cast<const VecT*>(&sO(r, d_base, _0{}));
              *reinterpret_cast<VecT*>(&params.ptr_O[row_off + d_base]) = data;
            }
          }
        } else {
          // Non-split: write final packed O (packed layout).
          int offs_0 = params.max_qo_len - qo_len;
          int offs_2_1 = qo_segment_offset + qo_len;
          if (total_qo_len_splitkv > 0 && kv_split_idx > 0) {
            offs_2_1 += kv_split_idx * total_qo_len_splitkv;
          }
          int stride_n = get<0>(params.layout_O.stride());
          auto stride_h_inner = get<2, 0>(params.layout_O.stride());
          int stride_h0 = get<0>(stride_h_inner);
          int stride_g = get<2, 1>(params.layout_O.stride());

          for (int r = tid / kGmemThreadsPerRow; r < kQRows;
               r += num_threads / kGmemThreadsPerRow) {
            int row_idx = tile_row_base + r;
            if (row_idx >= qo_len) break;
            int abs_row = offs_0 + row_idx;
            int abs_head_group = offs_2_1;
            int64_t row_off = (int64_t)abs_row * stride_n
                            + (int64_t)qo_head_idx * stride_h0
                            + (int64_t)abs_head_group * stride_g;
            int lane = tid % kGmemThreadsPerRow;
            for (int v = lane; v < vec_per_row; v += kGmemThreadsPerRow) {
              int d_base = v * kVecElems;
              VecT data = *reinterpret_cast<const VecT*>(&sO(r, d_base, _0{}));
              *reinterpret_cast<VecT*>(&params.ptr_O[row_off + d_base]) = data;
            }
          }
        }
      }
    }
  }
};

}  // namespace cutlass::fmha::collective

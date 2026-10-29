// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <cmath>

#include "gmem_bounds_check.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_ppu.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "sm80_fmha_load.hpp"
#include "ppu_include.hpp"

#if !USE_AIU
#define make_mix_tensor_like(x)  x
#endif

namespace cutlass::fmha::collective {

using namespace cute;

// Interface mirrors Sm100FmhaFwdMainloopTmaWarpspecialized.
template <class Element_, class ElementQK_, class ElementPV_, class TileShapeQK_,
          class TileShapePV_, class StrideQ_, class StrideK_, class StrideV_, class Mask_,
          class ThreadShape = Shape<_1, _1, _1>,
          bool IsSplitKV_ = false,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off>
struct Sm80FmhaFwdMainloop {
  static constexpr bool IsSplitKV = IsSplitKV_;
  static constexpr int KVPageSize = KVPageSize_;

  static constexpr bool kNeedMaxScore = (is_only_score(kSparseAttnMode) ||
                                         kSparseAttnMode == SparseAttnMode::Full);
  static constexpr bool kNeedOutput = !is_only_score(kSparseAttnMode);
  static constexpr bool kNeedSparse = kSparseAttnMode == SparseAttnMode::Sparse;
  static_assert(kNeedSparse == false, "Sm80FmhaFwdMainloop do not supports SparseAttnMode::Sparse");

  struct SplitKVParams {
    float scale_output_splitkv = 1.0f;
    float scale_output_nosplit = 1.0f;
    float* ptr_lse_accum = nullptr;
    int* kv_tile_begin_indices = nullptr;
    int* kv_tile_end_indices = nullptr;
    int* kv_split_indices = nullptr;
    int total_qo_len = 0;
    int num_qo_heads = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int split_kv_buf_size = 0;
#endif
  };

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  // FP8 QK path (OnlyScore): Q/K stay e4m3 all the way into the MMA unit (fp8 x fp8 -> fp32);
  static_assert(!std::is_same_v<Element, cutlass::float_e4m3_t> || (std::is_same_v<Element, cutlass::float_e4m3_t> && is_only_score(kSparseAttnMode)),
                "FP8 QK is only supported for OnlyScore");
  using TileShape = decltype(select<0, 1>(TileShapeQK_{}));
  using TileShapeQK = TileShapeQK_;
  using TileShapePV = TileShapePV_;
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  static constexpr int kTileM = CUTE_STATIC_V(shape<0>(TileShapeQK{}));
  static constexpr int kTileN = CUTE_STATIC_V(shape<1>(TileShapeQK{}));
  static constexpr int kTileK = CUTE_STATIC_V(shape<2>(TileShapeQK{}));
  static constexpr int kTileD = CUTE_STATIC_V(shape<1>(TileShapePV{}));

  static constexpr int kNWarps = (kTileM < 64 ? kTileM : 64) / 16;
  static constexpr int StageCountKV = 2;  // number of K/V stages per tile
  static_assert(kTileN % StageCountKV == 0, "kTileN must be divisible by StageCountKV");
  static constexpr int kStageSplitN = kTileN / StageCountKV;
  // Structural gate for the dense-paged page-table hoist. init() and load_K's
  // PreResolvedPage argument MUST use the same constant: translate() on an
  // uninitialized table returns -1, which load_K would use as a physical
  // page. Only the AIU dense-paged load_K branch consults the table, and that
  // branch is entered exactly when kTileN <= KVPageSize; sparse mode branches
  // earlier (kv_block_indexes), and init() re-checks it at runtime.
  static constexpr bool kHoistPageTable = (KVPageSize > 0) && (kTileN <= KVPageSize);
  static constexpr bool Q_in_regs = true;

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<Element, cutlass::half_t>,
      MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
      std::conditional_t<
          std::is_same_v<Element, cutlass::float_e4m3_t>,
          MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
          MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>>>;
#else
  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<Element, cutlass::half_t>,
      MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
      std::conditional_t<
          std::is_same_v<Element, cutlass::float_e4m3_t>,
          MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
          MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>>>;
#endif
  static constexpr int kMmaAtomK =
      std::is_same_v<Element, cutlass::float_e4m3_t> ? 32 : 16;
  using TiledMma = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, Int<kMmaAtomK>>>;

  // ====================== SMEM layouts ======================
  // All Q/K/V use the same swizzled layout atom (like FA3)
  static constexpr int kBytePerRow = kTileK * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  // A 128B row of 8-bit elements holds 8 16B units: the XOR width and the
  // row-index shift are both 3 (Swizzle<3,4,3>), matching the AIU/TSM 128B
  // hardware swizzle.
  static constexpr int kSwizzleB = sizeof(Element) == 1 ? 3 : kSwizzle;
  static constexpr int kSwizzleShift = sizeof(Element) == 1 ? 3 : kSwizzleBase;

  using SmemLayoutAtomQKV = decltype(
      composition(Swizzle<kSwizzleB, kSwizzleBase, kSwizzleShift>{},
                  Layout<Shape<_8, Int<kBlockKGmem>>,
                         Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShapeQK{})));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(Int<kStageSplitN>{}, Int<kTileK>{}, Int<StageCountKV>{})));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(Int<kStageSplitN>{}, Int<kTileD>{}, Int<StageCountKV>{})));
  using SmemLayoutVt = decltype(
      composition(SmemLayoutV{},
                  make_ordered_layout(
                      make_shape(Int<kTileD>{}, Int<kStageSplitN>{}, Int<StageCountKV>{}),
                      Step<_2, _1, _3>{})));

  static constexpr int kPackFactor = MaskPackFactor<Mask>::value;

#if defined(USE_PPU) && USE_AIU
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using SmemCopyOpQ  = PPU0015_TSM_LD_SWZL<Element, kTileM, kBlockKGmem, false, false, 1>;
  using SmemCopyOpK  = PPU0015_TSM_LD_SWZL<Element, kStageSplitN, kBlockKGmem, true,  false, kTileK / kBlockKGmem>;
  using SmemCopyOpVt = PPU0015_TSM_LD_SWZL<Element, kStageSplitN, kBlockKGmem, true,  true,  kTileD / kBlockKGmem>;
#else
  using SmemCopyOpQ  = PPU0010_TSM_LD_SWZL<Element, kTileM, kBlockKGmem, false, false>;
  using SmemCopyOpK  = PPU0010_TSM_LD_SWZL<Element, kStageSplitN, kBlockKGmem, false, false, kTileK / kBlockKGmem>;
  using SmemCopyOpVt = PPU0010_TSM_LD_SWZL<Element, kStageSplitN, kBlockKGmem, false, true,  kTileD / kBlockKGmem>;
#endif
  using SmemCopyAtomQ  = std::conditional_t<kPackFactor == 1, Copy_Atom<SmemCopyOpQ,  Element>, Copy_Atom<PPU_U32x4_LDSM_N, Element>>;
  using SmemCopyAtomK  = Copy_Atom<SmemCopyOpK,  Element>;
  using SmemCopyAtomVt = Copy_Atom<SmemCopyOpVt, Element>;
#else
  using SmemCopyAtomQ  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomK  = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomVt = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
#endif

  static constexpr int kMMAValM = 16 * kNWarps;  // value tile M
  static constexpr int kMMA_M = kTileM / kMMAValM;
  static constexpr int kNRows = 2 * kMMA_M;

  // ====================== Shared memory ======================
  // K and V use separate buffers; sizes from swizzled layouts.
  struct CUTE_ALIGNAS(128) TensorStorageFull {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
  };

  struct CUTE_ALIGNAS(128) TensorStorageOnlyScore {
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_v;   // we don't need smem_v for OnlyScore mode. it's a placeholder for consistency with the code below.
    };
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
  };

  using TensorStorage = std::conditional_t<kNeedOutput, TensorStorageFull, TensorStorageOnlyScore>;

  // ====================== Load collective ======================
#if defined(USE_PPU) && USE_AIU
  using Load = Sm80FmhaLoadAiu<
      Element, TileShapeQK, TileShapePV, SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      TensorStorage, Mask, TileShape, KVPageSize, kSparseAttnMode, StageCountKV>;
#else
  using Load = Sm80FmhaLoadCpAsync<
      Element, TileShapeQK, TileShapePV, SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      TensorStorage, Mask, TileShape, KVPageSize, kSparseAttnMode, StageCountKV>;
#endif

  using LayoutQ = typename Load::LayoutQ;
  using LayoutK = typename Load::LayoutK;
  using LayoutV = typename Load::LayoutV;

  // ====================== Arguments / Params ======================
  struct Arguments {
    typename Load::Arguments load;

    float scale_softmax;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    float scale_softmax = args.scale_softmax;
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{Load::to_underlying_arguments(problem_shape, args.load, workspace),
                  args.scale_q * args.scale_k * scale_softmax,
                  args.scale_q * args.scale_k * log2_e * scale_softmax,
                  args.scale_v * args.inv_scale_o};
  }

  static bool can_implement(auto const&, Arguments const&) { return true; }

  // ====================== Trip count helpers ======================
  template <class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE static int get_full_trip_count(BlkCoord const& blk_coord,
                                                 ProblemShape const& problem_shape,
                                                 int kv_block_num = 0) {
    return Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
  }

  template <class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE static int get_effective_trip_count(BlkCoord const& blk_coord,
                                                      ProblemShape const& problem_shape,
                                                      int kv_tile_begin, int kv_tile_end,
                                                      int kv_block_num = 0) {
    int full = get_full_trip_count(blk_coord, problem_shape, kv_block_num);
    int eff_end = full < kv_tile_end ? full : kv_tile_end;
    int count = eff_end - kv_tile_begin;
    return count > 0 ? count : 0;
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape, class TensorStorageType, class CollectiveEpilogue, class EpiTensorStorage>
  CUTLASS_DEVICE
  void mma(BlkCoord const& blk_coord, Params const& params,
           ParamsProblemShape const& params_problem_shape, ProblemShape const& problem_shape,
           TensorStorageType& storage, CollectiveEpilogue& epilogue,
           int kv_tile_begin = 0, int kv_tile_end = INT_MAX,
           int kv_split_idx = 0, SplitKVParams const* split_kv = nullptr,
           EpiTensorStorage* epi_storage = nullptr) {
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];

    int h_r = get<3, 0, 0>(problem_shape);
    int kv_head_idx = qo_head_idx / h_r;
    float scale = params.scale_softmax_log2;
    int thread_idx = threadIdx.x;
#if defined(USE_PPU) && USE_AIU
    const int tid_thread_slice = __ppu_read_firstlane(thread_idx / 32) * 32;
#else
    const int tid_thread_slice = thread_idx;
#endif

    int valid_m = min(kTileM, qo_len - qo_tile_idx * kTileM);
    if (valid_m <= 0) return;

    if constexpr (kNeedOutput) {
      __syncthreads();   // If persistent, need to sync to make sure all threads have finished with epilogue shared memory
    }

    // ---- Clear V SMEM when stage count > 1: partial V stage tiles may leave uninitialized, cause 0 * NaN = NaN in PV MMA ----
    if constexpr (kNeedOutput && StageCountKV > 1) {
      Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
      clear(sV);
    }
    // ---- Load Q tile to smem ----
    Load::load_Q(qo_tile_idx, qo_head_idx, qo_segment_offset, qo_len,
                 params.load, storage);
    Load::commit_and_wait();
    __syncthreads();

    // ---- Setup TiledMMA and partitions ----
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutVt{});

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShapePV{}));

    // Smem tiled copies
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma);
    auto smem_thr_copy_Q = kPackFactor == 1 ? smem_tiled_copy_Q.get_thread_slice(tid_thread_slice) : smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomK{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomVt{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid_thread_slice);

#if defined(USE_PPU) && USE_AIU
    Tensor tSsQ = [&]() {
      if constexpr (kPackFactor == 1) {
        return smem_thr_copy_Q.partition_S(make_mix_tensor_like(sQ));
      } else {
        return smem_thr_copy_Q.partition_S(sQ);
      }
    }();
    Tensor tSsK = smem_thr_copy_K.partition_S(make_mix_tensor_like(sK));
    Tensor tOsVt = smem_thr_copy_V.partition_S(make_mix_tensor_like(sVt));
#else
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
#endif

    // Copy Q from smem to registers
    if constexpr (Q_in_regs) {
      Tensor tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);
      cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_view);
    }

    // Coordinate tensor for masking (full tile)
    Tensor cS = cute::make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
    Tensor tScS = thr_mma.partition_C(cS);
    // Coordinate tensor for stage-tile masking
    Tensor cS_stage = cute::make_identity_tensor(make_shape(Int<kTileM>{}, Int<kStageSplitN>{}));

    // ---- Initialize softmax and O accumulator ----
    Softmax<kNRows> softmax(scale);
    if constexpr (kNeedOutput) {
      clear(tOrO);
    }

    // ---- Trip count ----
    int full_trip_count = get_full_trip_count(blk_coord, problem_shape,
                                         params.load.kv_block_num);
    int eff_end = full_trip_count < kv_tile_end ? full_trip_count : kv_tile_end;

    int total_trip_count, skip_count, real_masked_count, unmasked_count;

    if constexpr (IsSplitKV) {
      total_trip_count = max(0, eff_end - kv_tile_begin);
      int sub_q_off = 0, sub_q_sz = kTileM;
      int full_pw = Mask{}.get_trip_count(blk_coord, TileShape{},
                                           problem_shape, sub_q_off, sub_q_sz);
      int masked_pw = Mask{}.get_masked_trip_count(blk_coord, TileShape{},
                                                    problem_shape, sub_q_off, sub_q_sz);
      int unmask_bound = full_trip_count - masked_pw;
      skip_count = max(0, min(eff_end, full_trip_count) - max(kv_tile_begin, full_pw));
      real_masked_count = max(0, min(eff_end, full_pw) - max(kv_tile_begin, unmask_bound));
      unmasked_count = max(0, min(eff_end, unmask_bound) - kv_tile_begin);
    } else {
      total_trip_count = full_trip_count;
      int sub_q_off = 0, sub_q_sz = kTileM;
      int per_warp = Mask{}.get_trip_count(blk_coord, TileShape{},
                                            problem_shape, sub_q_off, sub_q_sz);
      int masked = Mask{}.get_masked_trip_count(blk_coord, TileShape{},
                                                 problem_shape, sub_q_off, sub_q_sz);
      skip_count = total_trip_count - per_warp;
      real_masked_count = max(0, masked - skip_count);
      unmasked_count = total_trip_count - masked;
    }

    // max_score output for (OnlyScore, OnlyScoreBF16)/Full. OnlyScoreBF16 stores bf16, other modes store float32.
    using ElementMaxScore = std::conditional_t<
        kSparseAttnMode == SparseAttnMode::OnlyScoreBF16, __ppu_bfloat16, float>;
    ElementMaxScore* ms_base = nullptr;
    int ms_stride_k = 0;
    int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
    int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);
    Tensor tScS_rowcol = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
    Tensor tScS_row = tScS_rowcol(_, _0{});

    // Per-row base pointer for direct (unpacked) max_score write.
    // Mirrors SM100's compute_ms_base pattern.
    auto compute_ms_base = [&](int abs_row) -> ElementMaxScore* {
      int rem_hr = get<3, 0, 0>(params_problem_shape);
      int h_r_orig = rem_hr * kPackFactor;
      int kv_head = qo_head_idx / rem_hr;
      int rem_hr_idx = qo_head_idx % rem_hr;
      int max_k = cute::size<2>(epilogue.params.layout_MaxScore);
      int tql_orig = epilogue.params.total_qo_len_orig;
      int seg_off_orig = segment_offset / kPackFactor;
      int actual_tok = abs_row / kPackFactor;
      int pf_idx = abs_row % kPackFactor;
      int unpacked_head = kv_head * h_r_orig + rem_hr_idx * kPackFactor + pf_idx;
      return reinterpret_cast<ElementMaxScore*>(epilogue.params.ptr_MaxScore_direct)
             + long(unpacked_head) * (long(max_k) * tql_orig)
             + seg_off_orig + actual_tok;
    };

    auto packed_maxscore_path = [&]() {
      Tensor gMaxScore = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementMaxScore*>(epilogue.params.ptr_MaxScore)),
        epilogue.params.layout_MaxScore);
      ms_base = &gMaxScore(segment_offset, qo_head_idx, 0);
      ms_stride_k = int(&gMaxScore(0, 0, 1) - &gMaxScore(0, 0, 0));
    };

    if constexpr (kNeedMaxScore) {
      if constexpr (kPackFactor > 1) {
        if (epilogue.params.ptr_MaxScore_direct != nullptr) {
          ms_stride_k = epilogue.params.total_qo_len_orig;
        } else {
          packed_maxscore_path();
        }
      } else {
        packed_maxscore_path();
      }
    }

    // Hoisted dense-paged page table (see kHoistPageTable): reads the
    // batch-constant indptr pair once per work item and memoizes kv_indices
    // across a tile's stages, for all dtypes on the AIU dense-paged path.
    DensePagedPageTable<KVPageSize, kTileN> pg_table;
    if constexpr (kHoistPageTable) {
      pg_table.init(params.load, batch_idx);
    }

    // Load all K stages for a tile
    auto load_K_next = [&](int kv_tile, int stage) {
      int stage_next = stage;
      int kv_tile_next = kv_tile - 1;
      if (kv_tile_next >= kv_tile_begin) {
        Load::load_K<kHoistPageTable>(kv_tile_next, kv_head_idx, batch_idx, kv_segment_offset, kv_len, params.load, storage, stage_next, pg_table);
      }
      Load::commit();
    };

    // ---- fwd_step: core per-KV-tile computation ----
    bool is_first_iter = true;
    Tensor row_max = make_tensor<float>(Shape<Int<kNRows>>{});

    auto fwd_step = [&](const int kv_tile, const int stage, const bool apply_mask, auto const check_inf) {
      Tensor tSrS = partition_fragment_C(tiled_mma, make_shape(Int<kTileM>{}, Int<kStageSplitN>{}));
      clear(tSrS);
      Tensor tSrK_cur = thr_mma.partition_fragment_B(sK(_, _, stage));
      if constexpr (kNeedOutput) {
        Load::template wait<2 * StageCountKV - 2>();
      } else {
        Load::template wait<StageCountKV - 1>();
      }
      __syncthreads();
      auto load_V_next = [&]() {
        if constexpr (kNeedOutput) {
          int stage_next = stage + StageCountKV - 1;
          int kv_tile_next = kv_tile - stage_next / StageCountKV;
          if (kv_tile_next >= kv_tile_begin) {
            stage_next = stage_next % StageCountKV;
            Load::load_V(kv_tile_next, kv_head_idx, batch_idx, kv_segment_offset, kv_len, params.load, storage, stage_next);
          }
          Load::commit();
        }
      };
      gemm_sm80<Q_in_regs>(
          tSrS, tSrQ, tSrK_cur, tSsQ, tSsK(_, _, _, stage),
          tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K, load_V_next);

      // Apply mask using coordinate tensor (shifted to global coordinates)
      if (apply_mask) {
        Tensor cS_stage_global = domain_offset(
            make_coord(qo_tile_idx * kTileM, kv_tile * kTileN + stage * kStageSplitN), cS_stage);
        Tensor tScS_stage_global = thr_mma.partition_C(cS_stage_global);
        Mask{}.apply_mask(tSrS, tScS_stage_global, problem_shape);
      }

      if constexpr (kNeedMaxScore) {
        Tensor scores_rc = make_tensor(tSrS.data(), convert_layout_acc_rowcol(tSrS.layout()));
        // Tree reduce per stage + quad allreduce deferred to the last stage
        // (all dtypes/modes; max is exact so results are order-independent).
        reduce_max_staged<StageCountKV>(scores_rc, row_max, stage);
        if (stage == StageCountKV - 1) {
          // Last stage: write the full-tile max
          bool use_direct = (kPackFactor > 1 && epilogue.params.ptr_MaxScore_direct != nullptr);
          if (use_direct || ms_base != nullptr) {
            uint qo_len = uint(get<0>(problem_shape));
            #pragma unroll
            for (int mi = 0; mi < kNRows; ++mi) {
              int abs_row = qo_tile_idx * kTileM + get<0>(tScS_row(mi));
              if (threadIdx.x % 4 == 0 && uint(abs_row) < qo_len) {
                if (use_direct) {
                  __stcg(compute_ms_base(abs_row) + long(kv_tile) * ms_stride_k,
                         ElementMaxScore(row_max(mi)));
                } else {
                  __stcg(ms_base + long(kv_tile) * ms_stride_k + abs_row,
                         ElementMaxScore(row_max(mi)));
                }
              }
            }
          }
        }
      }
      if constexpr (kNeedOutput) {
        Load::template wait<2 * StageCountKV - 2>();
      }
      __syncthreads();
      load_K_next(kv_tile, stage);

      if constexpr (kNeedOutput) {
        static constexpr bool Check_inf = decltype(check_inf)::value;
        Tensor scores_scale = [&]() -> auto {
          if (is_first_iter) {
            return softmax.template max_get_scale<true, Check_inf>(tSrS);
          } else {
            return softmax.template max_get_scale<false, Check_inf>(tSrS);
          }
        }();
        if (is_first_iter) {
          softmax.template online_softmax<true, Check_inf>(tSrS);
        } else {
          softmax.template online_softmax<false, Check_inf>(tSrS);
        }
        // Convert S (float) -> P (Element) for PV GEMM
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        Tensor tOrP_acc = make_tensor(tSrS.data(),
            convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
        convert_type_out(tOrP_acc, tOrP);
#else
        Tensor tOrP_acc = convert_acc<Element>(tSrS);
        Tensor tOrP = make_tensor(tOrP_acc.data(), make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
#endif
        // Rescale O by exp2(old_max - new_max)
        if (!is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
        // O += P_stage @ V_stage
        Tensor tOrV_cur = thr_mma.partition_fragment_B(sVt(_, _, stage));
        gemm_rs_sm80(tOrO, tOrP, tOrV_cur, tOsVt(_, _, _, stage),
                      tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
      }
      is_first_iter = false;
    };

    // ---- Three-phase mask loop (reverse iteration) ----
    int kv_tile = eff_end - 1;

    #pragma unroll
    for (int s = 0; s < StageCountKV - 1; ++s) {
      Load::load_K<kHoistPageTable>(kv_tile, kv_head_idx, batch_idx, kv_segment_offset, kv_len, params.load, storage, /*stage=*/s, pg_table);
      Load::commit();
      if constexpr (kNeedOutput) {
        Load::load_V(kv_tile, kv_head_idx, batch_idx, kv_segment_offset, kv_len, params.load, storage, /*stage=*/s);
        Load::commit();
      }
    }
    Load::load_K<kHoistPageTable>(kv_tile, kv_head_idx, batch_idx, kv_segment_offset, kv_len, params.load, storage, StageCountKV - 1, pg_table);
    Load::commit();
    // Phase 1: Skip (above causal boundary or sparse padding)
    for (int iter = 0; iter < skip_count && kv_tile >= kv_tile_begin; ++iter, --kv_tile) {
      #pragma unroll
      for (int s = 0; s < StageCountKV; ++s) {
        fwd_step(kv_tile, s, /*apply_mask=*/true, /*check_inf=*/cute::true_type{});
      }
    }

    // Phase 2: Masked iterations (near causal diagonal)
    for (int iter = 0; iter < real_masked_count && kv_tile >= kv_tile_begin; ++iter, --kv_tile) {
      #pragma unroll
      for (int s = 0; s < StageCountKV; ++s) {
        fwd_step(kv_tile, s, /*apply_mask=*/true, /*check_inf=*/cute::true_type{});
      }
    }

    // Phase 3: Unmasked iterations (no mask needed)
    for (int iter = 0; iter < unmasked_count && kv_tile >= kv_tile_begin; ++iter, --kv_tile) {
      #pragma unroll
      for (int s = 0; s < StageCountKV; ++s) {
        fwd_step(kv_tile, s, /*apply_mask=*/false, /*check_inf=*/cute::false_type{});
      }
    }

    // ---- Finalize: normalize O and compute LSE ----
    if constexpr (kNeedOutput) {
      // Handle case where no iterations ran
      if (is_first_iter) {
        Tensor tSrS_dummy = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeQK{}));
        fill(tSrS_dummy, -INFINITY);
        softmax.template max_get_scale<true, true>(tSrS_dummy);
        softmax.template online_softmax<true, true>(tSrS_dummy);
      }
      Tensor scores_scale = softmax.finalize();
      softmax.rescale_o(tOrO, scores_scale);

      // Write O from registers to epilogue smem
      if (epi_storage != nullptr) {
        __syncthreads();   // mainloop and epilogue use the same shared memory space.
        // Convert O from float to Element type, then write to smem_o
        auto& smem_o_ref = *epi_storage;
        Tensor sO_write = make_tensor(make_smem_ptr(smem_o_ref.smem_o.data()),
                                       typename CollectiveEpilogue::SmemLayoutO{});
        // tOrO is float; smem_o is Element. Convert via make_tensor_like + convert_type_out.
        Tensor tOrO_elem = make_tensor_like<Element>(tOrO);
        convert_type_out(tOrO, tOrO_elem);
        // Write to smem using partition_C (matching thread→element mapping)
        Tensor tOsO = thr_mma.partition_C(sO_write(_, _, _0{}));
        cute::copy(tOrO_elem, tOsO);

        // Write LSE for split-KV
        if constexpr (IsSplitKV) {
          Tensor cO_idx = cute::make_identity_tensor(select<0, 1>(TileShapeQK{}));
          Tensor tOcO = thr_mma.partition_C(cO_idx);
          Tensor tOcO_rc = make_tensor(tOcO.data(), convert_layout_acc_rowcol(tOcO.layout()));
          for (int mi = 0; mi < kNRows; ++mi) {
            int row = int(get<0>(tOcO_rc(mi, 0)));
            if (threadIdx.x % 4 == 0) {
              smem_o_ref.smem_lse[row] = softmax.row_sum(mi);
            }
          }
        }
        __syncthreads();   // make sure all writes to smem_o_ref are complete
      }
    }
  }

  // ====================== correction_empty ======================
  template <class BlkCoord, class ProblemShape, class ParamsProblemShape,
            class TensorStorageEpi, class CollectiveEpilogue>
  CUTLASS_DEVICE void correction_empty(
      BlkCoord const&, Params const&, ProblemShape const&, ParamsProblemShape const&,
      TensorStorageEpi& shared_storage_epi,
      CollectiveEpilogue&) {
    if constexpr (kNeedOutput) {
      Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()),
                              typename CollectiveEpilogue::SmemLayoutO{});
      clear(sO);
      if constexpr (IsSplitKV) {
        if (threadIdx.x < int(get<0>(TileShape{}))) {
          shared_storage_epi.smem_lse[threadIdx.x] = -INFINITY;
        }
      }
    }
    __syncthreads();
  }
};
}  // namespace cutlass::fmha::collective

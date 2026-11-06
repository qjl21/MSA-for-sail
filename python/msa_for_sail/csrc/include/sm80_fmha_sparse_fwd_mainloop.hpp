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
#include "acc_vreg_fraga.h"
#include "sm80_fmha_load.hpp"
#include "ppu_include.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

// Sparse-only SM80/PPU forward mainloop.
//
// This struct is intentionally split from Sm80FmhaFwdMainloop so the ordinary
// Off/Full/OnlyScore path does not carry PPU Sparse-specific AIU/TSM layout,
// sparse page mapping, sparse mask coordinate, padding skip, and future sparse
// optimizations.  The initial implementation is behavior-equivalent to the
// validated Sparse branch that used to live inside Sm80FmhaFwdMainloop.
template <class Element_, class ElementQK_, class ElementPV_, class TileShapeQK_,
          class TileShapePV_, class StrideQ_, class StrideK_, class StrideV_, class Mask_,
          class ThreadShape = Shape<_1, _1, _1>,
          bool IsSplitKV_ = false,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Sparse>
struct Sm80FmhaSparseFwdMainloop {
  static constexpr bool IsSplitKV = IsSplitKV_;
  static constexpr int KVPageSize = KVPageSize_;

  static_assert(kSparseAttnMode == SparseAttnMode::Sparse,
                "Sm80FmhaSparseFwdMainloop only supports SparseAttnMode::Sparse");

  static constexpr bool kNeedMaxScore = false;
  static constexpr bool kNeedOutput = true;
  static constexpr bool kNeedSparse = true;
  static constexpr int kPackFactor = MaskPackFactor<Mask_>::value;

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
  static constexpr int StageCountK = 1;   // K single-buffered: next K issued after QK (flies during PV)
  static constexpr int StageCountV = 1;   // V single-buffered (issued at iteration start)

  static constexpr bool IsFP8 = std::is_same_v<Element, cutlass::float_e4m3_t>;
  static_assert(sizeof(Element) != 1 || IsFP8, "Sparse FP8 supports only E4M3FN");
#if defined(__HGGC_ARCH__)
  static_assert(!IsFP8 || __HGGC_ARCH__ == 150, "Sparse FP8 supports only PPU0015");
#endif

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using MMA_Atom_16bit = std::conditional_t<
      std::is_same_v<Element, cutlass::half_t>,
      MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
      MMA_Atom<PPU0015_16x16x16_F32BF16BF16F32_TN>>;
#else
  using MMA_Atom_16bit = std::conditional_t<
      std::is_same_v<Element, cutlass::half_t>,
      MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
      MMA_Atom<PPU0010_16x16x16_F32BF16BF16F32_TN>>;
#endif
  using MMA_Atom_Arch = std::conditional_t<IsFP8,
      MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>, MMA_Atom_16bit>;
  using TiledMma = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, Int<IsFP8 ? 32 : 16>>>;
  static_assert(std::is_same_v<typename TiledMma::ValTypeC, float>);
  using TiledMmaOPerm = std::conditional_t<IsFP8,
      TiledMMA<MMA_Atom<cute::MSASparseFP8EpiPerm>,
               Layout<Shape<Int<kNWarps>, _1, _1>>,
               Tile<Int<16 * kNWarps>, _16, _32>>, TiledMma>;

  static constexpr int kBytePerRow = kTileK * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

  static constexpr int kSwizzle =
      kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase =
      sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomPlain =
      Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>;
  using SmemLayoutAtomSwizzled = decltype(
      composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                  Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutAtomQ = SmemLayoutAtomPlain;
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShapeQK{})));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomPlain{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<StageCountK>{})));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomPlain{},
      make_shape(Int<kTileN>{}, Int<kTileD>{}, Int<StageCountV>{})));
  using SmemLayoutVt = decltype(
      composition(SmemLayoutV{},
                  make_ordered_layout(
                      make_shape(Int<kTileD>{}, Int<kTileN>{}, Int<StageCountV>{}),
                      Step<_2, _1, _3>{})));

#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
  using SmemCopyOpQ = PPU0015_TSM_LD_SWZL<Element, kTileM, kBlockKGmem, false, false, 1>;
  using SmemCopyAtom = Copy_Atom<PPU0015_TSM_LD_SWZL<Element, kTileN, kBlockKGmem, true, false, kTileK / kBlockKGmem>, Element>;
#else
  using SmemCopyOpQ = std::conditional_t<IsFP8,
      PPU0015_TSM_LD_SWZL<Element, kTileM, kBlockKGmem, false, false, 1>,
      PPU0010_TSM_LD_SWZL<Element, kTileM, kBlockKGmem, false, false>>;
  using SmemCopyAtom = Copy_Atom<std::conditional_t<IsFP8,
      PPU0015_TSM_LD_SWZL<Element, kTileN, kBlockKGmem, true, false, kTileK / kBlockKGmem>,
      PPU0010_TSM_LD_SWZL<Element, kTileN, kBlockKGmem, false, false, kTileK / kBlockKGmem>>, Element>;
#endif
  using SmemCopyAtomQ = Copy_Atom<SmemCopyOpQ, Element>;

  CUTE_HOST_DEVICE static constexpr auto make_smem_copy_atom_transposed() {
    if constexpr (IsFP8) {
      return nullptr;
    } else {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
      return Copy_Atom<PPU0015_TSM_LD_SWZL<Element, kTileN, kBlockKGmem, true, true, kTileD / kBlockKGmem>, Element>{};
#else
      return Copy_Atom<PPU0010_TSM_LD_SWZL<Element, kTileN, kBlockKGmem, false, true, kTileD / kBlockKGmem>, Element>{};
#endif
    }
  }
  using SmemCopyAtomTransposed = decltype(make_smem_copy_atom_transposed());

  static constexpr int kMMAValM = 16 * kNWarps;
  static constexpr int kMMA_M = kTileM / kMMAValM;
  static constexpr int kNRows = 2 * kMMA_M;

  struct CUTE_ALIGNAS(128) TensorStorageSparse {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    // smem_q aliases smem_v: Q is preloaded into registers before the KV loop;
    // each loop-head __syncthreads (which precedes the iteration's V issue)
    // guards the Q->V hand-off. Saves one Q tile of smem per block.
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
  };
  static_assert(cute::cosize_v<SmemLayoutQ> <= cute::cosize_v<SmemLayoutV>,
                "smem_q aliases smem_v: Q tile must fit in the V buffer");

  using TensorStorage = TensorStorageSparse;

  using Load = Sm80SparseFmhaLoadAiu<
      Element, TileShapeQK, TileShapePV, SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      TensorStorage, Mask, TileShape, KVPageSize, kSparseAttnMode>;

  using LayoutQ = typename Load::LayoutQ;
  using LayoutK = typename Load::LayoutK;
  using LayoutV = typename Load::LayoutV;

  struct Arguments {
    typename Load::Arguments load;
    float scale_softmax;
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;
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

  template <class TensorT>
  CUTLASS_DEVICE
  static auto smem_read_src(TensorT const& t) {
    return make_mix_tensor_like(t);
  }

  template <class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE static int get_full_trip_count(BlkCoord const& blk_coord,
                                                 ProblemShape const& problem_shape,
                                                 int kv_block_num = 0) {
    return (kv_block_num * KVPageSize + kTileN - 1) / kTileN;
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

    Load::load_Q(qo_tile_idx, qo_head_idx, qo_segment_offset, qo_len,
                 params.load, storage);

    int valid_m = min(kTileM, qo_len - qo_tile_idx * kTileM);
    if (valid_m <= 0) return;

    typename Load::KVLoadState const kv_load(params.load, kv_head_idx);

    // ==== Page-table lookup pipelining (tuning spec §2) ====
    // Hoisted loop invariants; the topk blk entries for the first tiles are
    // issued BEFORE the Q wait so hop 1 hides behind it. See the spec for
    // the five exposure points this eliminates.
    int full_trip_count = get_full_trip_count(blk_coord, problem_shape,
                                         params.load.kv_block_num);
    int eff_end = full_trip_count < kv_tile_end ? full_trip_count : kv_tile_end;

    int total_trip_count, skip_count, real_masked_count;
    int num_kv_heads_s = get<3, 0, 1>(problem_shape);
    int kbi_off_s = batch_idx * num_kv_heads_s * params.load.kv_block_num
              + kv_head_idx * params.load.kv_block_num;
    total_trip_count = IsSplitKV ? max(0, eff_end - kv_tile_begin) : full_trip_count;
    // SM100-style: no -1 tail scan; padding tiles (topk entry < 0) load a
    // harmless dummy page (entry 0) and are masked out via
    // sparse_k_base = INT_MAX/2 -> scores -inf. On top of that, cap the trip
    // count by kv_len: tiles fully beyond kv_len are all-masked by causal
    // anyway, and computing them wastes up to 8x on short-kv large-batch
    // (mixload) shapes (e.g. kv_len=225 -> 4 real tiles vs 32 from the
    // kv_block_num formula). kv_len is already a register; no gmem read.
    int valid_tiles = min(full_trip_count, (kv_len + kTileN - 1) / kTileN);
    if constexpr (IsSplitKV) {
      skip_count = max(0, eff_end - max(valid_tiles, kv_tile_begin));
      real_masked_count = max(0, min(valid_tiles, eff_end) - kv_tile_begin);
    } else {
      skip_count = max(0, total_trip_count - valid_tiles);
      real_masked_count = valid_tiles;
    }

    int kv_tile = eff_end - 1;
    kv_tile -= skip_count;

    if constexpr (IsFP8) {
      if (real_masked_count <= 0 || kv_tile < kv_tile_begin) {
        Load::commit_and_wait();
        __syncthreads();
        if (epi_storage != nullptr) {
          using ElementOut = typename CollectiveEpilogue::ElementOut;
          static_assert(std::is_same_v<ElementOut, cutlass::bfloat16_t>);
          Tensor sO_zero = make_tensor(make_smem_ptr(epi_storage->smem_o.data()),
                                      typename CollectiveEpilogue::SmemLayoutO{});
          for (int idx = threadIdx.x; idx < kTileM * kTileD; idx += blockDim.x) {
            sO_zero(idx / kTileD, idx % kTileD, _0{}) = ElementOut(0.f);
          }
          if constexpr (IsSplitKV) {
            for (int row = threadIdx.x; row < kTileM; row += blockDim.x) {
              epi_storage->smem_lse[row] = -INFINITY;
            }
          }
        }
        __syncthreads();
        return;
      }
    }

    constexpr int kTilesPerPage = (KVPageSize > 0 && kTileN < KVPageSize)
        ? KVPageSize / kTileN : 1;
    const bool has_indmap = (params.load.kv_page_indptr != nullptr)
                         && (params.load.kv_indices != nullptr);
    // Work-item constant: base of this batch's slice in kv_indices.
    int page_begin = has_indmap
        ? KV_INDPTR_LOAD(params.load.kv_page_indptr, batch_idx, _KV_INDPTR_SZ(params.load))
        : 0;
    // Pipeline registers: page resolution of the current V tile (cur_*) and
    // one tile ahead (pf_*); fwd_step produces blk_n/pf_blk_n/pf_page_n for
    // two tiles ahead and the loop rotates them.
    int cur_blk = -1, cur_page_idx = -1;
    int pf_blk = -1, pf_page_idx = -1;
    auto resolve_page_idx = [&](int blk) -> int {
      // SM100-style dummy: a padding entry (blk < 0) falls back to logical
      // page 0, whose physical page is loaded harmlessly and masked out.
      int blk_eff = (blk < 0) ? 0 : blk;
      if (has_indmap)
        return KV_INDICES_LOAD(params.load.kv_indices, page_begin + blk_eff, _KV_INDICES_SZ(params.load));
      return blk_eff;
    };
    if (real_masked_count > 0 && kv_tile >= kv_tile_begin) {
      int pt_cur = kv_tile / kTilesPerPage;
      cur_blk = (pt_cur < params.load.kv_block_num)
          ? __ldg(params.load.kv_block_indexes + (kbi_off_s + pt_cur)) : -1;
      if (kv_tile - 1 >= kv_tile_begin) {
        int pt_pf = (kv_tile - 1) / kTilesPerPage;
        pf_blk = (pt_pf < params.load.kv_block_num)
            ? __ldg(params.load.kv_block_indexes + (kbi_off_s + pt_pf)) : -1;
      }
    }

    Load::commit_and_wait();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutVt{});

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tOrO = partition_fragment_C(tiled_mma, select<0, 1>(TileShapePV{}));

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma);
    // Q now uses AIU + TSM_LD_SWZL for all pack_factor values (same as K/V), so
    // it always needs the warp-aligned thread slice.

    const int tid_thread_slice = __ppu_read_firstlane(threadIdx.x / 32) * 32;
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid_thread_slice);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid_thread_slice);
    auto smem_tiled_copy_V = [&]() {
      if constexpr (IsFP8) { return nullptr; }
      else { return make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma); }
    }();
    auto smem_thr_copy_V = [&]() {
      if constexpr (IsFP8) { return nullptr; }
      else { return smem_tiled_copy_V.get_thread_slice(tid_thread_slice); }
    }();

    Tensor tSsQ = smem_thr_copy_Q.partition_S(smem_read_src(sQ));
    // Pre-partition K/V outside the loop (eliminate per-iteration divergent
    // address computation from make_mix_tensor_like + partition_S).
    Tensor tSsK = smem_thr_copy_K.partition_S(smem_read_src(sK));
    auto tOsVt = [&]() {
      if constexpr (IsFP8) { return nullptr; }
      else { return smem_thr_copy_V.partition_S(smem_read_src(sVt)); }
    }();

    // Pre-load Q from smem to registers (align with dense/FA3 Q_in_regs=true).
    // Avoids re-reading Q from smem in every KV iteration, which can cause
    // non-deterministic TSM port conflicts with concurrent K TSM_LD_SWZL reads.
    Tensor tSrQ_view = smem_thr_copy_Q.retile_D(tSrQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_view);

    Tensor cS = cute::make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
    Tensor tScS = thr_mma.partition_C(cS);

    Softmax<kNRows, IsFP8 ? 8 : 0> softmax(scale);
    clear(tOrO);

    bool is_first_iter = true;

    cur_page_idx = resolve_page_idx(cur_blk);
    pf_page_idx = resolve_page_idx(pf_blk);

    int blk_n = -1, pf_page_n = -1;

    auto fwd_step = [&](int kv_tile, int next_tile, bool has_next, bool apply_mask, auto check_inf) {
      int k_buf = kv_tile % StageCountK;

      // K_T is already in smem. Issue V_T via the pre-resolved page index
      // (no gmem index lookup on the issue path).
      Load::load_V_page(kv_tile, cur_page_idx, kv_load, storage, 0);
      Load::commit();  // group: V_T

      // QK GEMM (V + K_next loading in background)
      Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeQK{}));
      clear(tSrS);

      Tensor tSrK_cur = thr_mma.partition_fragment_B(sK(_, _, k_buf));
      auto load_V_hook = []() {};
      gemm_sm80<true>(
          tSrS, tSrQ, tSrK_cur, tSsQ, tSsK(_, _, _, k_buf),
          tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
          smem_thr_copy_Q, smem_thr_copy_K, load_V_hook);

      blk_n = -1; pf_page_n = -1;
      if (has_next && (next_tile - 1 >= kv_tile_begin)) {
        int next2_tile = next_tile - 1;
        int pt_n = next2_tile / kTilesPerPage;
        blk_n = (pt_n < params.load.kv_block_num)
            ? __ldg(params.load.kv_block_indexes + (kbi_off_s + pt_n)) : -1;
      }

      if (apply_mask) {
        int sparse_linear_pos = kv_tile * kTileN;
        int sparse_page_idx = sparse_linear_pos / KVPageSize;
        int offset_in_page = sparse_linear_pos % KVPageSize;

        int sparse_block = (sparse_page_idx < params.load.kv_block_num)
            ? cur_blk : -1;
        int sparse_k_base = sparse_block >= 0
            ? sparse_block * KVPageSize + offset_in_page
            : INT_MAX / 2;
        Tensor cS_global = domain_offset(
            make_coord(qo_tile_idx * kTileM, sparse_k_base), cS);
        Tensor tScS_global = thr_mma.partition_C(cS_global);
        Mask{}.apply_mask(tSrS, tScS_global, problem_shape);
      }

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
      if constexpr (IsFP8) { permute_Cregs_fp8(tSrS); }
#if defined(USE_PPU) && (!defined(__HGGC_ARCH__) || (__HGGC_ARCH__ != 150))
      constexpr bool UseAregsView = IsFP8;
#else
      constexpr bool UseAregsView = true;
#endif
      Tensor tOrP_acc = [&]() {
        if constexpr (IsFP8) {
          return make_tensor(tSrS.data(), convert_layout_acc_AregsFP8<TiledMma>(tSrS.layout()));
        } else if constexpr (UseAregsView) {
          return make_tensor(tSrS.data(), convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
        } else {
          return convert_acc<Element>(tSrS);
        }
      }();
      Tensor tOrP = [&]() {
        if constexpr (UseAregsView) {
          return make_tensor_like<Element>(tOrP_acc);
        } else {
          return make_tensor(tOrP_acc.data(),
              make_layout(get<0>(tSrQ.layout()), get<1>(tSrS.layout()), get<2>(tSrS.layout())));
        }
      }();
      if constexpr (UseAregsView) { convert_type_out(tOrP_acc, tOrP); }
      // All warps finished reading K_T (QK GEMM done); the single K buffer
      // can now be overwritten with K_next, which flies during PV. K_next's
      // page was prefetched (pf_page_idx), so the issue path is lookup-free.
      __syncthreads();
      if (has_next) {
        Load::load_K_page(next_tile, pf_page_idx, kv_load, storage, next_tile % StageCountK);
        Load::commit();  // group: K_next
      }
      // Hop 2 (kv_indices) for the tile two steps ahead; its latency hides
      // behind wait + PV + the next iteration's K wait/QK.
      pf_page_n = resolve_page_idx(blk_n);

      // Wait for V_T; K_next (if issued) stays pending into the next iteration.
      if (has_next) {
        Load::template wait_group<1>();
      } else {
        Load::wait_all();
      }
      __syncthreads();

      if (!is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
      Tensor tOrV_cur = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
      if constexpr (IsFP8) {
        gemm_rs_sm80_pv_fp8_vdirect<kTileN, kTileD, kBlockKGmem * sizeof(Element), true>(
            tOrO, tOrP, tOrV_cur, storage.smem_v.data(), tiled_mma);
      } else {
        gemm_rs_sm80(tOrO, tOrP, tOrV_cur, tOsVt(_, _, _, 0),
                     tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
      }

      is_first_iter = false;
    };

    // Prologue: prefetch K for the first tile into the single K buffer
    // (V_0 is issued inside iter 0, after the loop-head sync that also
    // guards the smem_q alias).
    if (real_masked_count > 0 && kv_tile >= kv_tile_begin) {
      Load::load_K_page(kv_tile, cur_page_idx, kv_load, storage, kv_tile % StageCountK);
      Load::commit();
    }

    for (int iter = 0; iter < real_masked_count && kv_tile >= kv_tile_begin; ++iter, --kv_tile) {
      // Wait for K_T (prefetched by the previous iteration / prologue);
      // the sync also frees the stages' previous contents for reuse.
      Load::wait_all();
      __syncthreads();
      int next_tile = kv_tile - 1;
      bool has_next = (iter + 1 < real_masked_count) && (next_tile >= kv_tile_begin);
      fwd_step(kv_tile, next_tile, has_next, true, cute::true_type{});
      // Rotate the page pipeline: pf_* becomes current, the freshly
      // prefetched pair becomes one-ahead.
      cur_blk = pf_blk; cur_page_idx = pf_page_idx;
      pf_blk = blk_n; pf_page_idx = pf_page_n;
    }

    if (is_first_iter) {
      Tensor tSrS_dummy = partition_fragment_C(tiled_mma, select<0, 1>(TileShapeQK{}));
      fill(tSrS_dummy, -INFINITY);
      softmax.template max_get_scale<true, true>(tSrS_dummy);
      softmax.template online_softmax<true, true>(tSrS_dummy);
    }
    Tensor scores_scale = [&]() {
      if constexpr (IsFP8) {
        if constexpr (IsSplitKV) {
          return softmax.finalize(split_kv->scale_output_splitkv);
        } else {
          return softmax.finalize(params.scale_output);
        }
      } else {
        return softmax.finalize();
      }
    }();
    softmax.rescale_o(tOrO, scores_scale);
    if constexpr (IsFP8) { permute_output_fp8(tOrO); }

    if (epi_storage != nullptr) {
      __syncthreads();
      auto& smem_o_ref = *epi_storage;
      Tensor sO_write = make_tensor(make_smem_ptr(smem_o_ref.smem_o.data()),
                                     typename CollectiveEpilogue::SmemLayoutO{});
      using ElementOut = typename CollectiveEpilogue::ElementOut;
      static_assert(!IsFP8 || std::is_same_v<ElementOut, cutlass::bfloat16_t>);
      Tensor tOrO_elem = make_tensor_like<ElementOut>(tOrO);
      convert_type_out(tOrO, tOrO_elem);
      TiledMmaOPerm tiled_mma_o;
      auto thr_mma_o = tiled_mma_o.get_slice(threadIdx.x);
      Tensor tOsO = thr_mma_o.partition_C(sO_write(_, _, _0{}));
      cute::copy(tOrO_elem, tOsO);

      if constexpr (IsSplitKV) {
        Tensor cO_idx = cute::make_identity_tensor(select<0, 1>(TileShapeQK{}));
        Tensor tOcO = thr_mma.partition_C(cO_idx);
        Tensor tOcO_rc = make_tensor(tOcO.data(), convert_layout_acc_rowcol(tOcO.layout()));
        for (int mi = 0; mi < kNRows; ++mi) {
          int row = int(get<0>(tOcO_rc(mi, 0)));
          if (row < int(get<0>(TileShape{}))) {
            smem_o_ref.smem_lse[row] = softmax.row_sum(mi);
          }
        }
      }
      __syncthreads();
    }
  }

};
}  // namespace cutlass::fmha::collective

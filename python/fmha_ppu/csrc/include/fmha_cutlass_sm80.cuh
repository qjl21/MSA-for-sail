/*
 * Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 *
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// SM80 top-level FMHA dispatch.
//
// Drop-in replacement for fmha_cutlass_sm100.cuh that targets SM80 (Ampere).
// Keeps the exact same FwdRunner template parameters, run_fmha_fwd signature,
// launch_fmha_reduction signature, and PackGQAUnpackParams struct so the
// Jinja-generated variant bindings can switch between SM100 and SM80 by
// changing only the #include.
// NOTE: fmha_cutlass_sm100.cuh has since been removed from this tree, so the
// parity above is historical. run_fmha_fwd / FwdRunner::run now carry two extra
// parameters (k_stride_token, v_stride_token) for NHD paged KV support, inserted
// right after v_stride_h and before batch_size.
//
// Key architectural differences from SM100:
//   TMA           → cp.async (16-byte vectors)
//   TMEM          → register-file accumulation
//   Warp-specialized pipeline → single-warp-group loop
//   SM100 MMA     → HMMA m16n8k16
//   ClusterShape  → single CTA (1,1,1)
//   16 warps      → 4 warps (128 threads)

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "allocator.h"
#include "fmha_fusion.hpp"
#include "gpu_trace.h"
#include "sm80_fmha_fwd_mainloop.hpp"
#include "sm80_fmha_sparse_fwd_mainloop.hpp"
#include "sm80_fmha_fwd_epilogue.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "fmha.hpp"
#include "fmha_tile_scheduler.hpp"
#include "sm80_fmha_fwd_kernel.hpp"
#include "sm80_fmha_reduction.hpp"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

struct PackGQAUnpackParams {
  float* max_score_direct = nullptr;
  int total_qo_len_orig = 0;
  void* o_direct = nullptr;
  int num_qo_heads_orig = 0;
  bool tma_direct_o_enabled = false;
  int packed_work_info_size = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
  GmemBounds gmem_bounds = {};
#endif
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask,
          class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false,
          bool SingleSoftmaxWarpGroup = false,
          int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off>
struct FwdRunner {
  using Element = DTypeIn;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = DTypeOut;

  // Q K D ((H_R, H_KV), B)
  using ProblemShapeVarlen =
      cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>,
                  PerBatchOffset>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::tuple<int, _1, int, int>,
      cute::tuple<int, _1, cute::tuple<_0, int>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::tuple<_1, int, int, int>,
      cute::tuple<_1, int, cute::tuple<_0, int>>>;
  using StrideO =
      cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
  using StrideLSE = cute::tuple<int, cute::tuple<_1, int>>;

  static constexpr int NumQStages = cute::size<0>(ThreadShape{});
  static constexpr bool bNeedOutput = !cutlass::fmha::collective::is_only_score(kSparseAttnMode);
  static constexpr int kPackFactor = cutlass::fmha::collective::pack_factor_of<ActiveMask>::value;

  // SM80 types — same template parameter order as SM100 but using Sm80 collectives.
  // Sparse uses a dedicated mainloop so AIU/TSM/sparse-page/mask logic does not
  // pollute the ordinary Off/Full/OnlyScore mainloop.
  using DenseMainloop = cutlass::fmha::collective::Sm80FmhaFwdMainloop<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeQK, TileShapePV, StrideQ,
      StrideK, StrideV, ActiveMask, ThreadShape, IsSplitKV, KVPageSize, kSparseAttnMode>;
  using SparseMainloop = cutlass::fmha::collective::Sm80FmhaSparseFwdMainloop<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeQK, TileShapePV, StrideQ,
      StrideK, StrideV, ActiveMask, ThreadShape, IsSplitKV, KVPageSize, kSparseAttnMode>;
  using Mainloop = std::conditional_t<
      kSparseAttnMode == cutlass::fmha::collective::SparseAttnMode::Sparse,
      SparseMainloop,
      DenseMainloop>;
  using Epilogue = cutlass::fmha::collective::Sm80FmhaFwdEpilogue<
      ElementOut, ElementAccumulatorPV, typename Mainloop::TileShapePV, NumQStages, IsSplitKV, bNeedOutput, kPackFactor>;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm80FmhaFwdKernel<
          ProblemShapeVarlen, Mainloop, Epilogue,
          cutlass::fmha::kernel::HostPrecomputedTileScheduler>>;
  using LayoutQ = typename Mainloop::LayoutQ;
  using LayoutK = typename Mainloop::LayoutK;
  using LayoutV = typename Mainloop::LayoutV;
  using LayoutO = typename Epilogue::LayoutO;
  using LayoutLSE = typename Epilogue::LayoutLSE;

  // show_log: print kernel tile config, occupancy, and runtime params
  static void show_log_if_enabled(const Operation& op,
                                  const cutlass::KernelHardwareInfo& hw_info,
                                  int batch_size, int total_qo_len, int max_qo_len,
                                  int head_dim_qk, int head_dim_vo,
                                  int num_kv_splits, int pack_factor,
                                  int kv_block_num, int num_ctas) {
    const char* p = std::getenv("show_log");
    if (!p || !std::isdigit(static_cast<unsigned char>(*p)) || std::atoi(p) <= 0) return;

    constexpr int kBlockM = cute::size<0>(TileShapeQK{});
    constexpr int kBlockN = cute::size<1>(TileShapeQK{});
    constexpr int kBlockPV_M = cute::size<0>(TileShapePV{});
    constexpr int kBlockPV_N = cute::size<1>(TileShapePV{});
    constexpr int kThreads = Operation::kThreadCount;
    constexpr int kSmemSize = Operation::Kernel::SharedStorageSize;

    dim3 grid = Operation::get_grid_shape(op.params());
    dim3 block = Operation::Kernel::get_block_shape();

    int ctas_per_sm = Operation::maximum_active_blocks();
    hggcFuncAttributes attr;
    hggcFuncGetAttributes(&attr,
        reinterpret_cast<const void*>(cutlass::device_kernel<typename Operation::Kernel>));

    int sm_count = hw_info.cu_count;
    float occupancy = (ctas_per_sm > 0 && sm_count > 0)
        ? static_cast<float>(num_ctas) / static_cast<float>(sm_count * ctas_per_sm)
        : 0.0f;

    printf("[MSA show_log]:\n");
    printf("  TileQK=(%d,%d), TilePV=(%d,%d), NumQStages=%d, threads=%d\n",
           kBlockM, kBlockN, kBlockPV_M, kBlockPV_N,
           static_cast<int>(NumQStages), kThreads);
    printf("  grid=(%u,%u,%u), block=(%u,%u,%u)\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("  smem_size=%d, CTAs_per_SM=%d, numRegs=%d, stack=%d\n",
           kSmemSize, ctas_per_sm, static_cast<int>(attr.numRegs),
           static_cast<int>(attr.localSizeBytes));
    printf("  sm_count=%d, num_ctas=%d, occupancy=%.3f\n",
           sm_count, num_ctas, occupancy);
    printf("  batch_size=%d, total_qo_len=%d, max_qo_len=%d, head_dim_qk=%d, head_dim_vo=%d\n",
           batch_size, total_qo_len, max_qo_len, head_dim_qk, head_dim_vo);
    printf("  num_kv_splits=%d, pack_factor=%d, kv_block_num=%d\n",
           num_kv_splits, pack_factor, kv_block_num);
    fflush(stdout);
  }

  static hggcError_t run(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_lens, IdType* kv_segment_lens,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         uint64_t* packed_work_range, uint64_t* packed_work_info,
                         DTypeOut* o, int mask_mode_code,
                         float sm_scale, float q_scale, float k_scale, float v_scale,
                         float o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h,
                         int k_stride_token, int v_stride_token, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, int* qo_offsets,
                         hggcStream_t stream,
                         int num_kv_splits = 1,
                         int* kv_tile_begin_indices = nullptr,
                         int* kv_tile_end_indices = nullptr,
                         int* kv_split_indices = nullptr,
                         float* ptr_lse_accum = nullptr,
                         int* kv_indices = nullptr,
                         int* kv_page_indptr = nullptr,
                         int total_page_num = 0,
                         float* maybe_max_score = nullptr,
                         int max_k_tiles = 0,
                         int* kv_block_indexes = nullptr,
                         int kv_block_num = 0,
                         int pack_factor = 1,
                         int q_stride_n_original = 0,
                         int q_stride_h_original = 0,
                         int h_r_original = 0,
                         PackGQAUnpackParams pack_gqa = {},
                         int num_ctas = 0) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = (num_ctas > 0)
        ? num_ctas
        : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    [[maybe_unused]] auto sched_args = typename cutlass::fmha::kernel::HostPrecomputedTileScheduler::Arguments{
        packed_work_range, packed_work_info};
    sched_args.packed_work_info_size = pack_gqa.packed_work_info_size;

    StrideQ stride_Q;
    StrideK stride_K;
    StrideV stride_V;
    StrideO stride_O;
    StrideLSE stride_LSE;

    int h_r = num_qo_heads / num_kv_heads;
    assert(num_qo_heads % num_kv_heads == 0);
    ProblemShapeVarlen problem_shape = cute::make_tuple(
        VariableLength{qo_segment_lens, qo_segment_offsets},
        VariableLength{kv_segment_lens, kv_segment_offsets}, head_dim_qk,
        cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size),
        PerBatchOffset{qo_offsets});

    stride_Q = make_stride(q_stride_n, _1{}, make_stride(q_stride_h, h_r * q_stride_h));
    stride_O = make_stride(
        num_qo_heads * head_dim_vo, _1{},
        make_stride(make_stride(head_dim_vo, h_r * head_dim_vo), num_qo_heads * head_dim_vo));
    stride_LSE = make_stride(num_qo_heads, make_stride(_1{}, h_r));

    if constexpr (KVPageSize > 0) {
      int k_stride_page = k_stride_n;
      int k_stride_head = k_stride_h;
      int v_stride_page = v_stride_n;
      int v_stride_head = v_stride_h;
      // In-page token stride. The 0 sentinel means "not supplied", which keeps the
      // historical head_dim value (HND layout, tokens of one head contiguous).
      // NHD layout supplies Hkv * head_dim. Logical mode order stays
      // (n_in_page, D, head, page) so size<2>(layout_K) keeps meaning Hkv.
      // See docs/msa_sparse_nhd_kv_layout_spec.md decisions D3/D4/D5.
      int k_stride_token_eff = (k_stride_token > 0) ? k_stride_token : head_dim_qk;
      int v_stride_token_eff = (v_stride_token > 0) ? v_stride_token : head_dim_vo;
      stride_K = make_stride(k_stride_token_eff, _1{}, k_stride_head, k_stride_page);
      stride_V = make_stride(_1{}, v_stride_token_eff, v_stride_head, v_stride_page);
      auto shape_K = make_shape(KVPageSize, head_dim_qk, num_kv_heads, total_page_num);
      auto shape_V = make_shape(head_dim_vo, KVPageSize, num_kv_heads, total_page_num);

      LayoutK layout_K = make_layout(shape_K, stride_K);
      LayoutV layout_V = make_layout(shape_V, stride_V);
      auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_O = make_shape(max_qo_len, head_dim_vo,
                                make_shape(make_shape(h_r, num_kv_heads),
                                           max_qo_len + total_qo_len * num_kv_splits));
      LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
      LayoutO layout_O = make_layout(shape_O, stride_O);

      auto shape_MaxScore = make_shape(total_qo_len, make_shape(h_r, num_kv_heads), max_k_tiles);
      auto stride_MaxScore = make_stride(
          _1{},
          make_stride(total_qo_len * max_k_tiles, h_r * total_qo_len * max_k_tiles),
          total_qo_len);
      typename Epilogue::LayoutMaxScore layout_MaxScore = make_layout(shape_MaxScore, stride_MaxScore);

      typename Epilogue::Arguments epi_args;
      epi_args.ptr_O = o ? (o - max_qo_len * get<0>(stride_O)) : nullptr;
      epi_args.layout_O = layout_O;
      epi_args.max_qo_len = max_qo_len;
      epi_args.ptr_MaxScore = maybe_max_score;
      epi_args.layout_MaxScore = layout_MaxScore;
      epi_args.ptr_MaxScore_direct = pack_gqa.max_score_direct;
      epi_args.total_qo_len_orig = pack_gqa.total_qo_len_orig;
#ifdef FMHA_GMEM_BOUNDS_CHECK
      epi_args.max_score_numel = pack_gqa.gmem_bounds.max_score_numel;
#endif
      if constexpr (IsSplitKV) {
        epi_args.ptr_O_direct = nullptr;
      } else {
        epi_args.ptr_O_direct = static_cast<ElementOut*>(pack_gqa.o_direct);
        if (epi_args.ptr_O_direct) {
          epi_args.ptr_O = nullptr;
        }
      }
      epi_args.num_qo_heads_orig = pack_gqa.num_qo_heads_orig;
      epi_args.head_dim_vo = head_dim_vo;
      epi_args.remaining_h_r = h_r;
      epi_args.qo_len_uniform = pack_gqa.tma_direct_o_enabled;

      typename Operation::Arguments arguments;
      if constexpr (IsSplitKV) {
        arguments = typename Operation::Arguments{
          {problem_shape,
           {{q, layout_Q, k, layout_K, v, layout_V, kv_indices, kv_page_indptr, kv_block_indexes, kv_block_num,
#ifdef FMHA_GMEM_BOUNDS_CHECK
             pack_gqa.gmem_bounds.kv_page_indptr_size, pack_gqa.gmem_bounds.kv_indices_size, pack_gqa.gmem_bounds.kv_block_indexes_numel,
#endif
             pack_factor, q_stride_n_original, q_stride_h_original, h_r_original},
            sm_scale, q_scale, k_scale, v_scale, o_scale},
           epi_args,
           sched_args,
           hw_info},
          {static_cast<float>(v_scale),
           static_cast<float>(v_scale * o_scale),
           ptr_lse_accum,
           kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
           total_qo_len, num_qo_heads
#ifdef FMHA_GMEM_BOUNDS_CHECK
           , pack_gqa.gmem_bounds.split_kv_size
#endif
           }};
      } else {
        arguments = typename Operation::Arguments{
          problem_shape,
          {{q, layout_Q, k, layout_K, v, layout_V, kv_indices, kv_page_indptr, kv_block_indexes, kv_block_num,
#ifdef FMHA_GMEM_BOUNDS_CHECK
            pack_gqa.gmem_bounds.kv_page_indptr_size, pack_gqa.gmem_bounds.kv_indices_size, pack_gqa.gmem_bounds.kv_block_indexes_numel,
#endif
            pack_factor, q_stride_n_original, q_stride_h_original, h_r_original},
           sm_scale, q_scale, k_scale, v_scale, o_scale},
          epi_args,
          sched_args,
          hw_info};
      }

      Operation op;
      size_t workspace_size = Operation::get_workspace_size(arguments);
      AlignedAllocator allocator(workspace_buffer, workspace_size);
      uint8_t* workspace_ptr =
          allocator.aligned_alloc<uint8_t>(workspace_size, 16, "fmha_cutlass_sm80_workspace");

      cutlass::Status status = op.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "This kernel is not supported. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        return err != hggcSuccess ? err : hggcErrorNotSupported;
      }
      status = op.initialize(arguments, workspace_ptr);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        return err != hggcSuccess ? err : hggcErrorLaunchFailure;
      }
      show_log_if_enabled(op, hw_info, batch_size, total_qo_len, max_qo_len,
                          head_dim_qk, head_dim_vo, num_kv_splits, pack_factor,
                          kv_block_num, num_ctas);
      GPUTraceParam _gt_param;
      gpu_trace::setup_from_env(_gt_param, stream);
      status = op.run(stream);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        gpu_trace::teardown(_gt_param, stream);
        return err != hggcSuccess ? err : hggcErrorLaunchFailure;
      }
      gpu_trace::teardown(_gt_param, stream);
      return hggcSuccess;
    } else {
      stride_K = make_stride(k_stride_n, _1{}, make_stride(_0{}, k_stride_h));
      stride_V = make_stride(_1{}, v_stride_n, make_stride(_0{}, v_stride_h));

      auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_O = make_shape(max_qo_len, head_dim_vo,
                                make_shape(make_shape(h_r, num_kv_heads),
                                           max_qo_len + total_qo_len * num_kv_splits));
      auto shape_K = make_shape(total_kv_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_V = make_shape(head_dim_vo, total_kv_len, make_shape(h_r, num_kv_heads));

      LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
      LayoutK layout_K = make_layout(shape_K, stride_K);
      LayoutV layout_V = make_layout(shape_V, stride_V);
      LayoutO layout_O = make_layout(shape_O, stride_O);

      auto shape_MaxScore = make_shape(total_qo_len, make_shape(h_r, num_kv_heads), max_k_tiles);
      auto stride_MaxScore = make_stride(
          _1{},
          make_stride(total_qo_len * max_k_tiles, h_r * total_qo_len * max_k_tiles),
          total_qo_len);
      typename Epilogue::LayoutMaxScore layout_MaxScore = make_layout(shape_MaxScore, stride_MaxScore);

      typename Epilogue::Arguments epi_args;
      epi_args.ptr_O = o ? (o - max_qo_len * get<0>(stride_O)) : nullptr;
      epi_args.layout_O = layout_O;
      epi_args.max_qo_len = max_qo_len;
      epi_args.ptr_MaxScore = maybe_max_score;
      epi_args.layout_MaxScore = layout_MaxScore;
      epi_args.ptr_MaxScore_direct = pack_gqa.max_score_direct;
      epi_args.total_qo_len_orig = pack_gqa.total_qo_len_orig;
#ifdef FMHA_GMEM_BOUNDS_CHECK
      epi_args.max_score_numel = pack_gqa.gmem_bounds.max_score_numel;
#endif
      if constexpr (IsSplitKV) {
        epi_args.ptr_O_direct = nullptr;
      } else {
        epi_args.ptr_O_direct = static_cast<ElementOut*>(pack_gqa.o_direct);
        if (epi_args.ptr_O_direct) {
          epi_args.ptr_O = nullptr;
        }
      }
      epi_args.num_qo_heads_orig = pack_gqa.num_qo_heads_orig;
      epi_args.head_dim_vo = head_dim_vo;
      epi_args.remaining_h_r = h_r;
      epi_args.qo_len_uniform = pack_gqa.tma_direct_o_enabled;

      typename Operation::Arguments arguments;
      if constexpr (IsSplitKV) {
        arguments = typename Operation::Arguments{
          {problem_shape,
           {{q, layout_Q, k, layout_K, v, layout_V, nullptr, nullptr, nullptr, 0,
             pack_factor, q_stride_n_original, q_stride_h_original, h_r_original},
            sm_scale, q_scale, k_scale, v_scale, o_scale},
           epi_args,
           sched_args,
           hw_info},
          {static_cast<float>(v_scale),
           static_cast<float>(v_scale * o_scale),
           ptr_lse_accum,
           kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
           total_qo_len, num_qo_heads
#ifdef FMHA_GMEM_BOUNDS_CHECK
           , pack_gqa.gmem_bounds.split_kv_size
#endif
           }};
      } else {
        arguments = typename Operation::Arguments{
          problem_shape,
          {{q, layout_Q, k, layout_K, v, layout_V, nullptr, nullptr, nullptr, 0,
            pack_factor, q_stride_n_original, q_stride_h_original, h_r_original},
           sm_scale, q_scale, k_scale, v_scale, o_scale},
          epi_args,
          sched_args,
          hw_info};
      }

      Operation op;
      size_t workspace_size = Operation::get_workspace_size(arguments);
      AlignedAllocator allocator(workspace_buffer, workspace_size);
      uint8_t* workspace_ptr =
          allocator.aligned_alloc<uint8_t>(workspace_size, 16, "fmha_cutlass_sm80_workspace");

      cutlass::Status status = op.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "This kernel is not supported. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        return err != hggcSuccess ? err : hggcErrorNotSupported;
      }
      status = op.initialize(arguments, workspace_ptr);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        return err != hggcSuccess ? err : hggcErrorLaunchFailure;
      }
      show_log_if_enabled(op, hw_info, batch_size, total_qo_len, max_qo_len,
                          head_dim_qk, head_dim_vo, num_kv_splits, pack_factor,
                          kv_block_num, num_ctas);
      GPUTraceParam _gt_param;
      gpu_trace::setup_from_env(_gt_param, stream);
      status = op.run(stream);
      if (status != cutlass::Status::kSuccess) {
        hggcError_t err = hggcGetLastError();
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << hggcGetErrorString(err) << std::endl;
        gpu_trace::teardown(_gt_param, stream);
        return err != hggcSuccess ? err : hggcErrorLaunchFailure;
      }
      gpu_trace::teardown(_gt_param, stream);
      return hggcSuccess;
    }
  }
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask,
          class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false,
          bool SingleSoftmaxWarpGroup = false,
          int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off>
hggcError_t run_fmha_fwd(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_lens, IdType* kv_segment_lens,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         uint64_t* packed_work_range, uint64_t* packed_work_info,
                         DTypeOut* o, int mask_mode_code,
                         double sm_scale, double q_scale, double k_scale, double v_scale,
                         double o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h,
                         int k_stride_token, int v_stride_token, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, int* qo_offsets,
                         hggcStream_t stream,
                         int num_kv_splits = 1,
                         int* kv_tile_begin_indices = nullptr,
                         int* kv_tile_end_indices = nullptr,
                         int* kv_split_indices = nullptr,
                         float* ptr_lse_accum = nullptr,
                         int* kv_indices = nullptr,
                         int* kv_page_indptr = nullptr,
                         int total_page_num = 0,
                         float* maybe_max_score = nullptr,
                         int max_k_tiles = 0,
                         int* kv_block_indexes = nullptr,
                         int kv_block_num = 0,
                         int pack_factor = 1,
                         int q_stride_n_original = 0,
                         int q_stride_h_original = 0,
                         int h_r_original = 0,
                         PackGQAUnpackParams pack_gqa = {},
                         int num_ctas = 0) {
  return FwdRunner<DTypeIn, DTypeOut, IdType, TileShapeQK, TileShapePV, ActiveMask,
                   ThreadShape, IsSplitKV, SingleSoftmaxWarpGroup, KVPageSize, kSparseAttnMode>::run(
      workspace_buffer, q, k, v, qo_segment_lens, kv_segment_lens,
      qo_segment_offsets, kv_segment_offsets, packed_work_range,
      packed_work_info, o, mask_mode_code, sm_scale,
      q_scale, k_scale, v_scale, o_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h,
      k_stride_token, v_stride_token, batch_size,
      total_qo_len, total_kv_len, max_qo_len, qo_offsets, stream,
      num_kv_splits, kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
      ptr_lse_accum,
      kv_indices, kv_page_indptr, total_page_num,
      maybe_max_score, max_k_tiles,
      kv_block_indexes, kv_block_num,
      pack_factor, q_stride_n_original, q_stride_h_original, h_r_original,
      pack_gqa, num_ctas);
}

};  // namespace flashinfer

namespace flashinfer {

// Launch the split-KV reduction kernel
template <typename ElementPartial, typename ElementOut>
__global__ void fmha_reduction_kernel_sm80(
    typename Sm80FmhaReductionKernel<ElementPartial, ElementOut>::Params params) {
  Sm80FmhaReductionKernel<ElementPartial, ElementOut> kernel;
  kernel(params, nullptr);
}

template <typename ElementPartial, typename ElementOut>
hggcError_t launch_fmha_reduction(
    const ElementPartial* ptr_O_partial, ElementOut* ptr_O,
    const float* ptr_lse,
    const int* num_kv_splits_per_row,
    float scale_softmax_log2, float inv_scale_o,
    int num_kv_splits, int total_qo_len, int num_qo_heads, int head_dim_vo,
    int stride_o_n, int stride_o_h, int stride_partial_n, int stride_partial_h,
    ElementOut* ptr_O_direct, int num_qo_heads_orig,
    int num_kv_heads, int pack_factor,
    hggcStream_t stream) {
  if (total_qo_len <= 0) return hggcSuccess;
  using ReductionKernel = Sm80FmhaReductionKernel<ElementPartial, ElementOut>;
  typename ReductionKernel::Params params{
      ptr_O_partial, ptr_O, ptr_lse,
      num_kv_splits_per_row,
      scale_softmax_log2, inv_scale_o,
      num_kv_splits, total_qo_len, num_qo_heads, head_dim_vo,
      stride_o_n, stride_o_h, stride_partial_n, stride_partial_h,
      ptr_O_direct, num_qo_heads_orig, num_kv_heads, pack_factor};
  dim3 grid = ReductionKernel::get_grid_shape(params);
  dim3 block = ReductionKernel::get_block_shape();
  fmha_reduction_kernel_sm80<ElementPartial, ElementOut><<<grid, block, 0, stream>>>(params);
  return hggcGetLastError();
}

}  // namespace flashinfer

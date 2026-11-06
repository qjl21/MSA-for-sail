// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <hggc.h>

#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "gmem_bounds_check.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "fmha_options.hpp"
#include "fmha_tile_scheduler.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;

// Interface mirrors Sm100FmhaFwdKernelTmaWarpspecialized.
template <class ProblemShapeIn, class CollectiveMainloop, class CollectiveEpilogue,
          class TileScheduler>
struct Sm80FmhaFwdKernel {
  using TileShape = typename CollectiveMainloop::TileShape;
  using ProblemShape = ProblemShapeIn;

  using TiledMma = typename CollectiveMainloop::TiledMma;
  static constexpr uint32_t NumThreads = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using ArchTag = cutlass::arch::PPU0015;

  using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
  using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;


  struct CUTE_ALIGNAS(128) SharedStorage {
    union {
      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr bool IsSplitKV = CollectiveMainloop::IsSplitKV;
  using SplitKVParams = typename CollectiveMainloop::SplitKVParams;

  struct ArgumentsBase {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments tile_scheduler;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct ArgumentsSplitKV : ArgumentsBase {
    SplitKVParams split_kv{};
  };

  using Arguments = std::conditional_t<IsSplitKV, ArgumentsSplitKV, ArgumentsBase>;

  struct ParamsBase {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  struct ParamsSplitKV : ParamsBase {
    SplitKVParams split_kv;
  };

  using Params = std::conditional_t<IsSplitKV, ParamsSplitKV, ParamsBase>;

  static size_t get_workspace_size(Arguments const& /*args*/) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, hggcStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    if constexpr (IsSplitKV) {
      return Params{
          {args.problem_shape,
           CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
           CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
           TileScheduler::to_underlying_arguments(args.tile_scheduler, args.hw_info)},
           args.split_kv};
    } else {
      return Params{
          args.problem_shape,
          CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
          CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
          TileScheduler::to_underlying_arguments(args.tile_scheduler, args.hw_info)};
    }
  }

  CUTLASS_DEVICE auto apply_batch(const Params& params, ProblemShape const& problem_shape,
                                  int batch_idx) {
    return apply_variable_length(params.problem_shape, batch_idx);
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    TileScheduler tile_scheduler{params.tile_scheduler};
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue{params.epilogue};

    CUTLASS_PRAGMA_NO_UNROLL
    #pragma clang loop licm(disable)
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto blk_coord = tile_scheduler.get_block_coord();

      auto logical_problem_shape =
          apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

      // Skip if Q tile is out of range
      if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
        continue;
      }

      // Handle empty KV sequences: write zeros + -inf LSE
      if (get<1>(logical_problem_shape) == 0) {
        if constexpr (!CollectiveMainloop::kNeedSparse) {
          mainloop.correction_empty(blk_coord, params.mainloop, logical_problem_shape,
                                    params.problem_shape, shared_storage.epilogue, epilogue);
          epilogue.store(blk_coord, logical_problem_shape, params.epilogue,
                         params.problem_shape, shared_storage.epilogue);
          __syncthreads();
        }
        continue;
      }

      if constexpr (IsSplitKV) {
        int wp = tile_scheduler.get_work_ptr();
        int kv_tile_begin = SPLIT_KV_BEGIN(params.split_kv, wp);
        int kv_tile_end = SPLIT_KV_END(params.split_kv, wp);
        int kv_split_idx = SPLIT_KV_SPLIT(params.split_kv, wp);

        if (CollectiveMainloop::get_effective_trip_count(
                blk_coord, logical_problem_shape,
                kv_tile_begin, kv_tile_end, params.mainloop.load.kv_block_num) <= 0) {
          continue;
        }

        mainloop.mma(
            blk_coord, params.mainloop, params.problem_shape, logical_problem_shape,
            shared_storage.mainloop, epilogue,
            kv_tile_begin, kv_tile_end, kv_split_idx, &params.split_kv,
            &shared_storage.epilogue);

        // Store output
        epilogue.store(blk_coord, logical_problem_shape, params.epilogue,
                       params.problem_shape, shared_storage.epilogue,
                       kv_split_idx, params.split_kv.total_qo_len,
                       params.split_kv.ptr_lse_accum, params.split_kv.num_qo_heads);
      } else {
        // Non-split path
        mainloop.mma(
            blk_coord, params.mainloop, params.problem_shape, logical_problem_shape,
            shared_storage.mainloop, epilogue,
            0, INT_MAX, 0, nullptr,
            &shared_storage.epilogue);

        // Store output
        epilogue.store(blk_coord, logical_problem_shape, params.epilogue,
                       params.problem_shape, shared_storage.epilogue);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel

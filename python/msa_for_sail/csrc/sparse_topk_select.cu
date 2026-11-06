// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "sparse_topk_select.cuh"
#include "tvm_ffi_utils.h"
#include "msa_profiling_interface.hpp"

using namespace flashinfer;

// v2.5_oob_clamp_in_kernel:
//   Adds num_valid_pages parameter (between topk and stream_ptr).  When
//   num_valid_pages < max_k_tiles, indices >= num_valid_pages are emitted as
//   -1 and sorted to the tail by the kernel — replacing the prod wrapper's
//   torch.where + sort + torch.where post-processing (~84-101 us measured).
//
//   To disable clamping, pass num_valid_pages = max_k_tiles (or any value
//   >= max_k_tiles).
void sparse_topk_select(TensorView max_score, TensorView output_indices,
                        TensorView workspace_buffer, int64_t topk,
                        int64_t num_valid_pages,
                        int64_t force_begin_blocks, int64_t force_end_blocks,
                        int64_t stream_ptr) {
  CHECK_INPUT(max_score);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(3, max_score);
  CHECK_DIM(3, output_indices);
  CHECK_DIM(1, workspace_buffer);

  const int64_t ms_dtype = encode_dlpack_dtype(max_score.dtype());
  TVM_FFI_ICHECK(ms_dtype == float32_code || ms_dtype == bfloat16_code)
      << "max_score must be float32 or bfloat16";
  TVM_FFI_ICHECK(encode_dlpack_dtype(output_indices.dtype()) == int32_code)
      << "output_indices must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(workspace_buffer.dtype()) == ms_dtype)
      << "workspace_buffer must be same with max_score";

  const int64_t num_qo_heads = max_score.size(0);
  const int64_t max_k_tiles = max_score.size(1);
  const int64_t total_qo_len = max_score.size(2);

  TVM_FFI_ICHECK(output_indices.size(0) == total_qo_len);
  TVM_FFI_ICHECK(output_indices.size(1) == num_qo_heads);
  TVM_FFI_ICHECK(output_indices.size(2) == topk);
  TVM_FFI_ICHECK(topk == 16) << "this kernel only supports topk == 16, got " << topk;
  TVM_FFI_ICHECK(num_valid_pages > 0)
      << "num_valid_pages must be > 0, got " << num_valid_pages;

  const size_t needed_workspace = sparse_topk::SparseTopKWorkspaceSize(
      static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(max_k_tiles));
  TVM_FFI_ICHECK(static_cast<size_t>(workspace_buffer.size(0)) >= needed_workspace)
      << "workspace_buffer too small: need " << needed_workspace << " elements";

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);

  // HGTX range + [MSA] param line around the kernel; no-op unless
  // PPU_LIB_PERF_INSTRUMENT=1 / PPU_LIB_SHOW_PARAMS=1.
  msa::fmha::FmhaProfParam msa_prof_params;
  if (msa::fmha::ProfilingInterface::Instance().get_op_info()) {
    msa_prof_params.set_topk_params(
        ms_dtype == bfloat16_code ? "bf16" : "fp32",
        num_qo_heads, max_k_tiles, total_qo_len, topk, num_valid_pages,
        force_begin_blocks, force_end_blocks);
  }
  msa::fmha::ProfilingInterface::Instance().instrument(true, msa_prof_params, "MSA");

  auto launch_topk = [&](auto* dtype_ptr) {
    return sparse_topk::SparseTopKSelect(
        dtype_ptr,
        static_cast<int32_t*>(output_indices.data_ptr()),
        static_cast<int32_t*>(workspace_buffer.data_ptr()),
        static_cast<uint32_t>(total_qo_len), static_cast<uint32_t>(num_qo_heads),
        static_cast<uint32_t>(max_k_tiles), static_cast<uint32_t>(num_valid_pages),
        static_cast<uint32_t>(force_begin_blocks), static_cast<uint32_t>(force_end_blocks),
        stream);
  };
  hggcError_t status = (ms_dtype == bfloat16_code)
      ? launch_topk(static_cast<const ppu_bfloat16*>(max_score.data_ptr()))
      : launch_topk(static_cast<const float*>(max_score.data_ptr()));

  TVM_FFI_ICHECK(status == hggcSuccess)
      << "sparse_topk_select failed: " << hggcGetErrorString(status);
  msa::fmha::ProfilingInterface::Instance().instrument(false, msa_prof_params, "MSA");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_topk_select, sparse_topk_select);

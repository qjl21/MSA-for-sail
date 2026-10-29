// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#include "prepare_plan_metadata.cuh"
#include "msa_profiling_interface.hpp"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace minfer {
namespace ops {

void prepare_plan_metadata_forward(
    TensorView qo_lens,
    TensorView kv_lens,
    Optional<TensorView> maybe_qo_offset,
    int64_t batch_size,
    int64_t total_q,
    int64_t page_size,
    int64_t kv_block_num,
    int64_t pack_factor,
    int64_t max_kv_len,
    bool causal,
    TensorView ws_qo_segment_lens,
    TensorView ws_kv_segment_lens,
    TensorView ws_qo_offset,
    TensorView ws_plan_kv_lens,
    TensorView ws_kv_page_indptr,
    TensorView ws_qo_segment_offsets,
    TensorView ws_kv_segment_offsets,
    int64_t stream_ptr) {

  const hggcStream_t stream = reinterpret_cast<hggcStream_t>(stream_ptr);
  const int* qo_offset_ptr = maybe_qo_offset.has_value()
      ? static_cast<const int*>(maybe_qo_offset.value().data_ptr())
      : nullptr;

  auto& profiler = msa::fmha::ProfilingInterface::Instance();
  msa::fmha::FmhaProfParam msa_prof_params;
  if (profiler.get_op_info()) {
    msa_prof_params.set_prepare_plan_metadata_params(
        batch_size, total_q, page_size, kv_block_num, pack_factor,
        max_kv_len, causal, maybe_qo_offset.has_value());
  }
  profiler.instrument(true, msa_prof_params, "MSA");

  auto status = flashinfer::launch_prepare_plan_metadata(
      static_cast<const int*>(qo_lens.data_ptr()),
      static_cast<const int*>(kv_lens.data_ptr()),
      qo_offset_ptr,
      static_cast<int>(batch_size),
      static_cast<int>(total_q),
      static_cast<int>(page_size),
      static_cast<int>(kv_block_num),
      static_cast<int>(pack_factor),
      static_cast<int>(max_kv_len),
      causal ? 1 : 0,
      static_cast<int*>(ws_qo_segment_lens.data_ptr()),
      static_cast<int*>(ws_kv_segment_lens.data_ptr()),
      static_cast<int*>(ws_qo_offset.data_ptr()),
      static_cast<int*>(ws_plan_kv_lens.data_ptr()),
      static_cast<int*>(ws_kv_page_indptr.data_ptr()),
      static_cast<int*>(ws_qo_segment_offsets.data_ptr()),
      static_cast<int*>(ws_kv_segment_offsets.data_ptr()),
      stream);
  profiler.instrument(false, msa_prof_params, "MSA");

  if (status != hggcSuccess) {
    TVM_FFI_THROW(RuntimeError) << "prepare_plan_metadata failed: " << hggcGetErrorString(status);
  }
}

}  // namespace ops
}  // namespace minfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_plan_metadata, minfer::ops::prepare_plan_metadata_forward);

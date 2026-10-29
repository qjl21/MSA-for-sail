// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT
//
// Parameter profiling interface for MSA FMHA kernels.
//
// Mirrors ppu/FlashMLA/csrc/fmha_profiling_interface.hpp:
//   - PPU_LIB_SHOW_PARAMS=1      -> print a single-line "[MSA] --format=..." string
//   - PPU_LIB_PERF_INSTRUMENT=1  -> push/pop an HGTX range around the kernel
//
// The HGTX dependency is optional: it is only enabled when <hgtx3/hgToolsExt.h>
// is available at compile time, so the header stays buildable in environments
// without the HGTX headers.

#pragma once

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <hggc_runtime.h>

#if defined(__has_include)
#if __has_include(<hgtx3/hgToolsExt.h>)
#include <hgtx3/hgToolsExt.h>
#define MSA_PROFILING_HAS_HGTX 1
#endif
#endif

namespace msa {
namespace fmha {

// RAII guard that temporarily relaxes the current thread's CUDA stream-capture
// mode so that D2H copies issued while a graph is being captured do not error
// out. Mirrors the verified GraphCaptureModeSuspender in FlashMLA flash_api.cpp.
struct GraphCaptureModeSuspender {
  hggcStreamCaptureMode original_mode;

  explicit GraphCaptureModeSuspender(
      hggcStreamCaptureMode relaxed_mode = hggcStreamCaptureModeRelaxed) {
    original_mode = relaxed_mode;
    // Exchange current thread's mode with relaxed_mode, save previous mode.
    hggcThreadExchangeStreamCaptureMode(&original_mode);
  }

  ~GraphCaptureModeSuspender() {
    // Restore the saved original mode back to the current thread.
    hggcThreadExchangeStreamCaptureMode(&original_mode);
  }

  // Disable copying to prevent accidental double-restoration.
  GraphCaptureModeSuspender(const GraphCaptureModeSuspender&) = delete;
  GraphCaptureModeSuspender& operator=(const GraphCaptureModeSuspender&) = delete;
};

// Collects key/value parameters and formats them as a single-line string that
// can later be parsed back by a benchmark / reproduction script.
class FmhaProfParam {
public:
  FmhaProfParam() {}

  template <typename T>
  void add_param(const std::string& key, const T& val) {
    args_[key] = val_to_string(val);
    insertionOrder.push_back(key);
  }

  // Read a device array back to host and format it as "[v0,v1,...]".
  // Returns an empty string when the pointer is null or the size is <= 0.
  template <typename T>
  std::string device_array2str(const void* addr, int sz, hggcStream_t stream,
                               int div = 1) {
    if (addr == nullptr || sz <= 0) {
      return "";
    }
    std::vector<T> tmp(sz);
    hggcMemcpyAsync(tmp.data(), addr, sizeof(T) * sz, hggcMemcpyDeviceToHost, stream);
    hggcStreamSynchronize(stream);

    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < sz; ++i) {
      oss << (div > 1 ? tmp[i] / div : tmp[i]);
      if (i < sz - 1) oss << ",";
    }
    oss << "]";
    return oss.str();
  }

  // Populate all user-facing fmha_ppu / fmha_ppu_plan input parameters.
  // seqlen_q / seqlen_k / qo_offset are read back from device (D2H). When a
  // CUDA graph is being captured, the capture mode is relaxed and a non-blocking
  // side stream is used for the D2H copies (mirrors FlashMLA's verified
  // handling) so the reads do not break the capture.
  void set_fmha_params(const std::string& dtype,
                       const std::string& sparse_mode,
                       int batch_size,
                       const int* qo_segment_lens_ptr,
                       const int* kv_segment_lens_ptr,
                       const int* qo_offsets_ptr,
                       int num_qo_heads, int num_kv_heads,
                       int head_dim_qk, int head_dim_vo,
                       int num_kv_splits, int is_paged, int page_size,
                       int kv_block_num, int causal, float sm_scale,
                       hggcStream_t stream, int pack_factor = 1) {
    // Field ORDER and naming follow the OperatorsForge MSA case-name
    // convention ("[MSA]:[D][1,2000]msa_sparse:sparse_type:decode,topk:16,
    // page_size:128,num_heads:64,num_heads_kv:4,head_dim:128,head_dim_v:128,...")
    // so the printed line can be aligned/matched against opf list files
    // directly. dtype is emitted right after the op tag for a stable prefix.
    add_param("op", "fmha");
    add_param("dtype", dtype);
    add_param("sparse_mode", sparse_mode);
    add_param("topk", kv_block_num);
    add_param("page_size", page_size);
    add_param("batch_size", batch_size);

    hggcStreamCaptureStatus capture_status;
    hggcStreamIsCapturing(stream, &capture_status);
    if (capture_status != hggcStreamCaptureStatusNone) {
      GraphCaptureModeSuspender protector(hggcStreamCaptureModeRelaxed);
      hggcStream_t side_stream;
      if (hggcStreamCreateWithFlags(&side_stream, hggcStreamNonBlocking) == hggcSuccess) {
        // seqlen_q is printed in RAW token count (qo_segment_lens holds the
        // packed length = raw * pack_factor; divide back so the value matches
        // the opf case-name field "seqlen_q"). pack_factor == 1 is a no-op.
        add_param("seqlen_q", device_array2str<int>(qo_segment_lens_ptr, batch_size, side_stream, pack_factor));
        add_param("seqlen_k", device_array2str<int>(kv_segment_lens_ptr, batch_size, side_stream));
        if (qo_offsets_ptr) {
          add_param("qo_offset", device_array2str<int>(qo_offsets_ptr, batch_size, side_stream));
        }
        hggcStreamDestroy(side_stream);
      }
    } else {
      add_param("seqlen_q", device_array2str<int>(qo_segment_lens_ptr, batch_size, stream, pack_factor));
      add_param("seqlen_k", device_array2str<int>(kv_segment_lens_ptr, batch_size, stream));
      if (qo_offsets_ptr) {
        add_param("qo_offset", device_array2str<int>(qo_offsets_ptr, batch_size, stream));
      }
    }

    add_param("num_heads", num_qo_heads);
    add_param("num_heads_kv", num_kv_heads);
    add_param("head_dim", head_dim_qk);
    add_param("head_dim_v", head_dim_vo);
    add_param("num_kv_splits", num_kv_splits);
    add_param("is_paged", is_paged);
    // kv_block_num == topk (number of selected pages per token); kept for
    // backward readability of the log line.
    add_param("kv_block_num", kv_block_num);
    add_param("causal", causal);
    add_param("sm_scale", sm_scale);
  }

  void set_reduction_params(const std::string& dtype,
                            int64_t total_qo_len, int64_t num_qo_heads,
                            int64_t num_qo_heads_orig, int64_t num_kv_heads,
                            int64_t head_dim_vo, int64_t num_kv_splits,
                            int64_t pack_factor,
                            int64_t stride_o_n, int64_t stride_o_h,
                            int64_t stride_partial_n, int64_t stride_partial_h,
                            double scale_softmax_log2, double inv_scale_o) {
    add_param("op", "reduction");
    add_param("dtype", dtype);
    add_param("total_qo_len", total_qo_len);
    add_param("num_qo_heads", num_qo_heads);
    add_param("num_qo_heads_orig", num_qo_heads_orig);
    add_param("num_kv_heads", num_kv_heads);
    add_param("head_dim_vo", head_dim_vo);
    add_param("num_kv_splits", num_kv_splits);
    add_param("pack_factor", pack_factor);
    add_param("stride_o_n", stride_o_n);
    add_param("stride_o_h", stride_o_h);
    add_param("stride_partial_n", stride_partial_n);
    add_param("stride_partial_h", stride_partial_h);
    add_param("scale_softmax_log2", scale_softmax_log2);
    add_param("inv_scale_o", inv_scale_o);
  }

  void set_prepare_plan_metadata_params(
      int64_t batch_size, int64_t total_qo_len, int64_t page_size,
      int64_t kv_block_num, int64_t pack_factor, int64_t max_kv_len,
      bool causal, bool has_qo_offset) {
    add_param("op", "prepare_plan_metadata");
    add_param("batch_size", batch_size);
    add_param("total_qo_len", total_qo_len);
    add_param("page_size", page_size);
    add_param("kv_block_num", kv_block_num);
    add_param("pack_factor", pack_factor);
    add_param("max_kv_len", max_kv_len);
    add_param("causal", causal);
    add_param("has_qo_offset", has_qo_offset);
  }

  void set_plan_params(
      int64_t batch_size, int64_t qo_tile_size, int64_t kv_tile_size,
      int64_t num_heads, int64_t num_buckets, bool causal,
      int64_t num_kv_splits, int64_t adaptive_chunk_size,
      int64_t lse_total_size, int64_t pack_factor, bool has_qo_offset,
      bool is_sparse = false) {
    // op tag carries the plan routing mode: "plan_sparse" (adaptive path,
    // num_kv_splits < 0) vs "plan" (non-sparse direct-greedy path). The
    // routing mirrors plan_kernel's sign-based dispatch so the log line can
    // be filtered/grepped by mode without parsing num_kv_splits.
    add_param("op", is_sparse ? "plan_sparse" : "plan");
    add_param("batch_size", batch_size);
    add_param("qo_tile_size", qo_tile_size);
    add_param("kv_tile_size", kv_tile_size);
    add_param("num_heads", num_heads);
    add_param("num_buckets", num_buckets);
    add_param("causal", causal);
    add_param("num_kv_splits", num_kv_splits);
    add_param("adaptive_chunk_size", adaptive_chunk_size);
    add_param("lse_total_size", lse_total_size);
    add_param("pack_factor", pack_factor);
    add_param("has_qo_offset", has_qo_offset);
  }

  // Populate user-facing sparse_topk_select input parameters.
  void set_topk_params(const std::string& dtype,
                       int64_t num_qo_heads, int64_t max_k_tiles,
                       int64_t total_qo_len, int64_t topk,
                       int64_t num_valid_pages,
                       int64_t force_begin_blocks, int64_t force_end_blocks) {
    add_param("op", "sparse_topk_select");
    add_param("dtype", dtype);
    add_param("num_qo_heads", num_qo_heads);
    add_param("max_k_tiles", max_k_tiles);
    add_param("total_qo_len", total_qo_len);
    add_param("topk", topk);
    add_param("num_valid_pages", num_valid_pages);
    add_param("force_begin_blocks", force_begin_blocks);
    add_param("force_end_blocks", force_end_blocks);
  }

  // Format as "[<tag>] --format=key:val,key:val."
  std::string format(const std::string& tag) {
    std::stringstream ss;
    ss << "[" << tag << "] --format=";
    for (size_t i = 0; i < insertionOrder.size(); ++i) {
      const std::string& key = insertionOrder[i];
      ss << key << ":" << args_[key];
      if (i + 1 < insertionOrder.size()) ss << ",";
    }
    ss << ".";
    return ss.str();
  }

private:
  std::string val_to_string(int val) { return std::to_string(val); }
  std::string val_to_string(int64_t val) { return std::to_string(val); }
  std::string val_to_string(bool val) { return std::to_string(int(val)); }
  std::string val_to_string(float val) {
    std::stringstream s;
    s << std::fixed << std::setprecision(6) << val;
    return s.str();
  }
  std::string val_to_string(double val) {
    std::stringstream s;
    s << std::fixed << std::setprecision(6) << val;
    return s.str();
  }
  std::string val_to_string(const std::string& val) { return val; }
  std::string val_to_string(const char* val) { return std::string(val); }

  std::unordered_map<std::string, std::string> args_;
  std::vector<std::string> insertionOrder;
};

// Singleton switch board. Reads the env vars once at construction; both off
// means get_op_info() is false and call sites skip all extraction work.
class ProfilingInterface {
public:
  ProfilingInterface(ProfilingInterface const&) = delete;
  void operator=(ProfilingInterface const&) = delete;

  static ProfilingInterface& Instance() {
    static ProfilingInterface instance;
    return instance;
  }

  bool get_op_info() { return show_params_ || use_hgtx_; }
  bool show_params() { return show_params_; }

  void instrument(bool start, FmhaProfParam& params, const std::string& tag) {
    if (!get_op_info()) {
      return;
    }
    if (start) {
      std::string op_name = params.format(tag);
      if (show_params_) {
        std::cout << op_name << std::endl;
      }
#ifdef MSA_PROFILING_HAS_HGTX
      if (use_hgtx_) {
        hgtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = HGTX_VERSION;
        eventAttrib.messageType = HGTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = op_name.c_str();
        hgtxDomainRangePushEx(domain_, &eventAttrib);
      }
#endif
    } else {
#ifdef MSA_PROFILING_HAS_HGTX
      if (use_hgtx_) {
        hgtxDomainRangePop(domain_);
      }
#endif
    }
  }

private:
  ProfilingInterface() {
    use_hgtx_ = false;
    show_params_ = false;
#ifdef MSA_PROFILING_HAS_HGTX
    domain_ = hgtxDomainCreateA("msa");
#endif

    char* pEnv_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
    if (pEnv_perf && std::isdigit(*pEnv_perf)) {
      int value = std::stoi(std::string(pEnv_perf));
      if (value == 1) {
        use_hgtx_ = true;
      } else if (value != 0) {
        printf("Invalid value for PPU_LIB_PERF_INSTRUMENT : %d\n", value);
      }
    }

    char* pEnv_params = std::getenv("PPU_LIB_SHOW_PARAMS");
    if (pEnv_params && std::isdigit(*pEnv_params)) {
      int value = std::stoi(std::string(pEnv_params));
      if (value == 1) {
        show_params_ = true;
      } else if (value != 0) {
        printf("Invalid value for PPU_LIB_SHOW_PARAMS : %d\n", value);
      }
    }
  }

  ~ProfilingInterface() {
#ifdef MSA_PROFILING_HAS_HGTX
    hgtxDomainDestroy(domain_);
#endif
  }

  bool use_hgtx_;
  bool show_params_;
#ifdef MSA_PROFILING_HAS_HGTX
  hgtxDomainHandle_t domain_;
#endif
};

} // namespace fmha
} // namespace msa

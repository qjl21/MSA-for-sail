/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
 * SPDX-License-Identifier: MIT
 *
 * prepare_plan_metadata kernel: computes all plan metadata on device
 * in a single kernel launch, replacing 20+ Python torch ops.
 *
 * Design (inspired by FlashMLA get_mla_metadata_kernel):
 * - 1 CTA, BLOCK_SIZE threads
 * - Stride loop for batch-level and token-level operations
 * - Thread 0 serial prefix sum (no cub::BlockScan, no size limit)
 * - Supports arbitrary batch_size and total_q
 */
#pragma once
#include "utils.cuh"

namespace flashinfer {

constexpr int PREPARE_BLOCK_SIZE = 256;

/*
 * prepare_plan_metadata kernel
 *
 * Grid: 1 block
 * Block: PREPARE_BLOCK_SIZE threads
 *
 * Shared memory is sized dynamically as 2*(batch_size+1) ints at launch, so
 * there is no hard batch_size limit (bounded only by device dynamic-smem cap).
 *
 * Phase 0: Load batch data + serial prefix sum (thread 0)
 * Phase 1: Fill per-token outputs (stride loop, all threads)
 * Phase 2: Segment offsets prefix sum (thread 0 serial)
 */
__global__ void prepare_plan_metadata_kernel(
    const int* __restrict__ qo_lens,       // [batch_size]
    const int* __restrict__ kv_lens,       // [batch_size]
    const int* __restrict__ qo_offset_in,  // [batch_size] or nullptr
    int batch_size,
    int total_q,
    int page_size,        // >0 means paged KV
    int kv_block_num,     // >0 means sparse mode
    int pack_factor,
    int max_kv_len,       // used when causal=false
    int causal,
    // outputs
    int* __restrict__ ws_qo_segment_lens,
    int* __restrict__ ws_kv_segment_lens,
    int* __restrict__ ws_qo_offset,
    int* __restrict__ ws_plan_kv_lens,
    int* __restrict__ ws_kv_page_indptr,
    int* __restrict__ ws_qo_segment_offsets,
    int* __restrict__ ws_kv_segment_offsets)
{
    // Shared memory layout:
    // [0..batch_size]: qo_cumsum (exclusive prefix sum of qo_lens)
    // [batch_size+1..2*batch_size+1]: page_cumsum (exclusive prefix sum of ceil_div(kv_lens, page_size))
    extern __shared__ int smem[];
    int* s_qo_cumsum = smem;                    // [batch_size + 1]
    int* s_page_cumsum = smem + batch_size + 1; // [batch_size + 1]

    int tid = threadIdx.x;
    bool is_sparse = (kv_block_num > 0 && page_size > 0);

    // === Phase 0: Prefix sums (thread 0 serial) ===
    if (tid == 0) {
        s_qo_cumsum[0] = 0;
        s_page_cumsum[0] = 0;
        for (int i = 0; i < batch_size; i++) {
            s_qo_cumsum[i + 1] = s_qo_cumsum[i] + qo_lens[i];
            if (page_size > 0) {
                s_page_cumsum[i + 1] = s_page_cumsum[i] + (kv_lens[i] + page_size - 1) / page_size;
            }
        }
    }
    __syncthreads();

    // === Phase 1: Fill per-token outputs (stride loop) ===
    int actual_total_q = s_qo_cumsum[batch_size];  // real token count
    for (int idx = tid; idx < total_q; idx += PREPARE_BLOCK_SIZE) {
        if (idx >= actual_total_q) {
            // Padding token (max_total_q > actual): zero-fill outputs
            ws_qo_segment_lens[idx] = 0;
            ws_kv_segment_lens[idx] = 0;
            ws_qo_offset[idx] = 0;
            ws_plan_kv_lens[idx] = 0;
            if (page_size > 0) ws_kv_page_indptr[idx] = s_page_cumsum[batch_size];
            continue;
        }
        int b, within;
        if (is_sparse) {
            // Binary search: find batch b such that qo_cumsum[b] <= idx < qo_cumsum[b+1]
            int lo = 0, hi = batch_size;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (s_qo_cumsum[mid + 1] <= idx)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            b = lo;
            within = idx - s_qo_cumsum[b];
        } else {
            b = idx;
            within = 0;
        }

        int qo = qo_lens[b];
        int kv = kv_lens[b];

        // qo_segment_lens
        ws_qo_segment_lens[idx] = is_sparse ? (pack_factor > 1 ? pack_factor : 1)
                                            : qo * pack_factor;
        // kv_segment_lens
        ws_kv_segment_lens[idx] = kv;

        // plan_kv_lens
        ws_plan_kv_lens[idx] = is_sparse ? (kv_block_num * page_size) : kv;

        // qo_offset
        if (causal) {
            int base;
            if (qo_offset_in != nullptr) {
                base = qo_offset_in[b];
            } else {
                base = kv - qo;
            }
            ws_qo_offset[idx] = base + within;
        } else {
            ws_qo_offset[idx] = max_kv_len;
        }

        // kv_page_indptr
        if (page_size > 0) {
            ws_kv_page_indptr[idx] = s_page_cumsum[b];
        }
    }
    // Last element of kv_page_indptr
    if (page_size > 0 && tid == 0) {
        ws_kv_page_indptr[total_q] = s_page_cumsum[batch_size];
    }
    __syncthreads();

    // === Phase 2: Segment offsets prefix sum (thread 0 serial) ===
    if (tid == 0) {
        ws_qo_segment_offsets[0] = 0;
        for (int i = 0; i < total_q; i++) {
            ws_qo_segment_offsets[i + 1] = ws_qo_segment_offsets[i] + ws_qo_segment_lens[i];
        }
        ws_kv_segment_offsets[0] = 0;
        for (int i = 0; i < total_q; i++) {
            ws_kv_segment_offsets[i + 1] = ws_kv_segment_offsets[i] + ws_kv_segment_lens[i];
        }
    }
}

// Host wrapper
inline hggcError_t launch_prepare_plan_metadata(
    const int* qo_lens, const int* kv_lens, const int* qo_offset_in,
    int batch_size, int total_q, int page_size, int kv_block_num,
    int pack_factor, int max_kv_len, int causal,
    int* ws_qo_segment_lens, int* ws_kv_segment_lens,
    int* ws_qo_offset, int* ws_plan_kv_lens,
    int* ws_kv_page_indptr,
    int* ws_qo_segment_offsets, int* ws_kv_segment_offsets,
    hggcStream_t stream)
{
    // Shared memory: 2 * (batch_size + 1) ints
    int smem_size = 2 * (batch_size + 1) * sizeof(int);
    hggcFuncSetAttribute(prepare_plan_metadata_kernel,
        hggcFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    prepare_plan_metadata_kernel<<<1, PREPARE_BLOCK_SIZE, smem_size, stream>>>(
        qo_lens, kv_lens, qo_offset_in,
        batch_size, total_q, page_size, kv_block_num,
        pack_factor, max_kv_len, causal,
        ws_qo_segment_lens, ws_kv_segment_lens,
        ws_qo_offset, ws_plan_kv_lens,
        ws_kv_page_indptr,
        ws_qo_segment_offsets, ws_kv_segment_offsets);
    return hggcGetLastError();
}

}  // namespace flashinfer

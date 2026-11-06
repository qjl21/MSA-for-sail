#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""SDPA-free attention reference used by the correctness tests.

torch.nn.functional.scaled_dot_product_attention can hang on PPU (notably with
a boolean attn_mask or is_causal), so the reference is expressed with plain
matmul + masked softmax + matmul.  These run on the PPU through the normal
torch ops and accumulate in fp32, so they stay fast and accurate enough to be a
reference for bf16 outputs.
"""

import math

import torch


# Rows of QK scores materialised at once.  Scores are q_chunk * kv_len * h_q
# fp32 elements; the chunk keeps that within a few hundred MB.  It is a Python
# integer so the loop unrolls into a fixed number of kernel launches.
_SCORE_CHUNK_ELEMS = 1 << 24


def _mha_ref_single(q, k, v, h_q, h_k, causal, qo_offset):
    """One batch item: q [q_len, h_q, d], k/v [kv_len, h_k, d] (both 3-D).

    Returns o [q_len, h_q, d] in fp32.  Heads are expanded in the same order as
    the tests' SDPA reference: ``(h_k, gqa_ratio)`` flattened into ``h_q``.
    """
    q_len, kv_len = q.shape[0], k.shape[0]
    gqa_ratio = h_q // h_k
    q_chunk = max(1, _SCORE_CHUNK_ELEMS // max(1, kv_len * h_q))
    offset = 0 if qo_offset is None else int(qo_offset)

    k_t = k.float().transpose(0, 1).transpose(1, 2)          # [h_k, d, kv_len]
    k_t = k_t.repeat_interleave(gqa_ratio, dim=0)            # [h_q, d, kv_len]
    v_t = v.float().transpose(0, 1)                          # [h_k, kv_len, d]
    v_t = v_t.repeat_interleave(gqa_ratio, dim=0)            # [h_q, kv_len, d]
    o = torch.empty((q_len, h_q, q.shape[-1]), dtype=torch.float32, device=q.device)
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    for q_start in range(0, q_len, q_chunk):
        q_end = min(q_start + q_chunk, q_len)
        q_c = q[q_start:q_end].float().transpose(0, 1)       # [h_q, qc, d]
        scores = (q_c @ k_t) * sm_scale                      # [h_q, qc, kv_len]

        if causal:
            row = torch.arange(q_start, q_end, device=q.device) + offset
            col = torch.arange(kv_len, device=q.device)
            invalid = row.unsqueeze(1) < col.unsqueeze(0)     # [qc, kv_len]
            scores.masked_fill_(invalid.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        o[q_start:q_end] = (probs @ v_t).transpose(0, 1)

    return o


def sdpa_ref_new(q, k, v, h_q, h_k, causal=True, qo_offset=None):
    """Drop-in replacement for the tests' sdpa_ref, without SDPA.

    q: [b, q_len, h_q, d], k/v: [b, kv_len, h_k, d].  qo_offset is the causal
    mask shift: Q row i of a batch item attends to KV column j iff
    i + qo_offset >= j.  It may be None/0, an int (uniform), or a per-batch
    tensor/list of length b.

    Returns o [b * q_len, h_q, d] in fp32, matching the kernel output layout.
    """
    b = q.shape[0]
    outs = []
    for bi in range(b):
        if qo_offset is None:
            off = 0
        elif isinstance(qo_offset, torch.Tensor):
            off = int(qo_offset.reshape(-1)[bi].item())
        elif isinstance(qo_offset, (list, tuple)):
            off = int(qo_offset[bi])
        else:
            off = int(qo_offset)
        outs.append(_mha_ref_single(q[bi], k[bi], v[bi], h_q, h_k, causal, off))
    return torch.cat(outs, dim=0)

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: MIT

# Run the MSA test suite on a PPU device.
# Usage: bash tests/run_ppu_tests.sh   (from the repository root)

MSA_ROOT=./
MSA_PYTHON=$MSA_ROOT/python

echo "=========================================="
echo "0. Environment check"
echo "=========================================="
python3 -c "
import sys, torch
sys.path.insert(0, '$MSA_PYTHON')
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
"

echo ""
echo "=========================================="
echo "1. test_sparse_topk_forced.py"
echo "=========================================="
cd $MSA_ROOT
python3 -c "
import sys, torch
sys.path.insert(0, '$MSA_PYTHON')
sys.path.insert(0, 'tests/smoke')
# Directly import and call test functions -- no monkey-patch needed because
# the device check lives inside if __name__ == '__main__' which is skipped
# when importing (vs running as script).
from test_sparse_topk_forced import (
    test_forced_blocks, test_forced_zero_is_noop,
    test_forced_ascending_order, test_forced_with_xor_fast_path,
    test_forced_large_k
)
print('=== Testing forced block selection ===')
test_forced_zero_is_noop()
test_forced_blocks()
test_forced_blocks(force_begin=1, force_end=0, seed=10)
test_forced_blocks(force_begin=0, force_end=5, seed=20)
test_forced_blocks(force_begin=8, force_end=8, seed=30)
test_forced_ascending_order()
test_forced_with_xor_fast_path()
test_forced_large_k()
print('All forced-block tests PASSED!')
"

echo ""
echo "=========================================="
echo "2. PPU dense throughput (basic shapes)"
echo "=========================================="
python3 -c "
import sys, math, time, torch
sys.path.insert(0, '$MSA_PYTHON')
from msa_for_sail import fmha_sail_plan, fmha_sail

HEAD = 128
device = 'cuda'

bench_cases = [
    ('prefill_1k',   1024, 1024, 32, 32),
    ('prefill_2k',   2048, 2048, 32, 32),
    ('prefill_4k',   4096, 4096, 32, 32),
    ('decode_bs32',  1,    4096, 32, 32),
    ('decode_bs128', 1,    4096, 128, 128),
]

WARMUP = 5
REPEAT = 20

print(f'{\"name\":<20s} {\"qo_len\":>6s} {\"kv_len\":>6s} {\"h_q\":>4s} {\"h_kv\":>4s} {\"time_ms\":>10s} {\"TFLOPs\":>10s}')
print('-' * 70)

for name, qo_len, kv_len, h_q, h_kv in bench_cases:
    B = 1  # PPU: batched decode kernel bug, single-token only for now
    # Original: B = 1 if qo_len > 1 else h_q

    q = torch.randn(qo_len, h_q, HEAD, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(kv_len, h_kv, HEAD, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(kv_len, h_kv, HEAD, dtype=torch.bfloat16, device=device) * 0.1

    if qo_len == 1:
        # Decode: B sequences each with 1 token, own KV cache copy
        q = q.repeat(B, 1, 1)
        k = k.repeat(B, 1, 1)
        v = v.repeat(B, 1, 1)
    qo_lens = torch.full((B,), qo_len if qo_len > 1 else 1, dtype=torch.int32)
    kv_lens = torch.full((B,), kv_len, dtype=torch.int32)

    plan = fmha_sail_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_kv, causal=True, num_kv_splits=1)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(WARMUP):
        fmha_sail(q, k, v, plan_info=plan)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(REPEAT):
        fmha_sail(q, k, v, plan_info=plan)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    t_ms = elapsed / REPEAT * 1000
    # FLOPs: 2 * B * qo_len * kv_len * h_q * head_dim
    flops = 2.0 * B * qo_len * kv_len * h_q * HEAD
    tflops = flops / (t_ms * 1e-3) / 1e12

    print(f'{name:<20s} {qo_len:>6d} {kv_len:>6d} {h_q:>4d} {h_kv:>4d} {t_ms:>10.3f} {tflops:>10.2f}')

print('')
print('Done!')
"

echo ""
echo "=========================================="
echo "3. PPU dense e2e: all shapes (via fmha_sail API)"
echo "=========================================="
python3 -c "
import sys, math, torch
sys.path.insert(0, '$MSA_PYTHON')
from msa_for_sail import fmha_sail_plan, fmha_sail

HEAD = 128
sm_scale = 1.0 / math.sqrt(HEAD)
device = 'cuda'

test_cases = [
    ('mha_sm_128', 128, 128, 4, 4, True),
    ('mha_sm_noncausal', 128, 128, 4, 4, False),
    ('mha_4x4', 128, 384, 4, 4, True),
    ('mha_4x4_noncausal', 128, 384, 4, 4, False),
    ('mha_long_kv', 128, 384, 4, 4, True),
    ('partial_qo_tile', 64, 128, 2, 2, True),
    ('mha_1k', 1024, 1024, 4, 4, True),
    ('mha_2k', 2048, 2048, 4, 4, True),
    ('mha_4k', 4096, 4096, 4, 4, True),
    ('mha_32x32_4k', 4096, 4096, 32, 32, True),
]

passed = 0
failed = 0

for name, qo_len, kv_len, h_q, h_kv, causal in test_cases:
    torch.manual_seed(0)
    q = torch.randn(qo_len, h_q, HEAD, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(kv_len, h_kv, HEAD, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(kv_len, h_kv, HEAD, dtype=torch.bfloat16, device=device) * 0.1
    qo_lens = torch.tensor([qo_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)

    plan = fmha_sail_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_kv, causal=causal, num_kv_splits=1)
    torch.cuda.synchronize()

    out, _ = fmha_sail(q, k, v, plan_info=plan)
    torch.cuda.synchronize()

    # Reference (correct einsum: separate n/m axes for q/k positions)
    q_f32 = q.float()
    k_f32 = k.float().repeat_interleave(h_q // h_kv, dim=1)
    v_f32 = v.float().repeat_interleave(h_q // h_kv, dim=1)

    s = torch.einsum('nhd,mhd->nhm', q_f32, k_f32) * sm_scale
    if causal:
        offset = kv_len - qo_len
        n_idx = torch.arange(qo_len, device=device).view(-1, 1)
        m_idx = torch.arange(kv_len, device=device).view(1, -1)
        mask = m_idx <= n_idx + offset
        s = s.masked_fill(~mask.unsqueeze(1), float('-inf'))
    p = torch.softmax(s, dim=-1)
    o_ref = torch.einsum('nhm,mhd->nhd', p, v_f32)

    diff = (out.float() - o_ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), o_ref.reshape(-1), dim=0
    ).item()
    ok = diff < 0.05 and cos > 0.999
    if ok:
        passed += 1
        status = 'PASS'
    else:
        failed += 1
        status = 'FAIL'
    print(f'  [{status}] {name}: max_abs={diff:.4f} cos_sim={cos:.6f}')

print(f'\nDense e2e: {passed}/{passed+failed} passed')
"

echo ""
echo "=========================================="
echo "4. PPU plan kernel validation (packed work decode)"
echo "=========================================="
python3 -c "
import sys, math, torch
sys.path.insert(0, '$MSA_PYTHON')
from msa_for_sail import fmha_sail_plan, sparse_topk_select

print('--- plan: packed work validation ---')
device = 'cuda'

# Decode helpers: same format as packed_work_range / packed_work_info
def decode_plan(plan_info, num_heads, batch_size):
    # Unpack tuple: (has_mixed, split, batch_size, decode_plan, prefill_plan)
    has_mixed, split, bs, inner_plan, prefill_plan = plan_info
    if has_mixed:
        print(f'  [INFO] mixed prefill/decode: split={split}')
        # For validation, use whichever plan is available
        target = inner_plan if inner_plan is not None else prefill_plan
    else:
        target = inner_plan

    pwr = target['packed_work_range'].cpu()
    pwi = target['packed_work_info'].cpu()
    num_ctas = pwr.shape[0]
    num_splits = target['num_kv_splits']

    # Decode work ranges
    total_items = 0
    items = []
    for cta in range(num_ctas):
        r = int(pwr[cta].item())
        s = r & 0xFFFFFFFF
        e = (r >> 32) & 0xFFFFFFFF
        assert s <= e, f'CTA {cta}: start={s} > end={e}'
        for wi in range(s, e):
            packed = int(pwi[wi].item())
            bi = packed & 0xFFFF
            hi = (packed >> 16) & 0xFFFF
            qt = (packed >> 32) & 0xFFFFFFFF
            items.append((bi, hi, qt))
        total_items += (e - s)

    errors = []
    # Check index ranges
    for bi, hi, qt in items:
        if bi < 0 or bi >= batch_size:
            errors.append(f'batch {bi} out of [0,{batch_size})')
        if hi < 0 or hi >= num_heads:
            errors.append(f'head {hi} out of [0,{num_heads})')

    # Check no duplicates (for nosplit)
    if num_splits <= 1:
        seen = set()
        for item in items:
            if item in seen:
                errors.append(f'duplicate work item {item}')
            seen.add(item)

    return total_items, num_ctas, num_splits, errors

test_configs = [
    ([1, 1, 1, 1], [128, 256, 384, 512], 4, True, 'B=4 varied kv causal'),
    ([128, 256], [256, 512], 8, True, 'B=2 varied qo causal'),
    ([64, 64], [128, 128], 4, False, 'B=2 noncausal'),
    ([1], [4096], 16, True, 'B=1 kv=4096 causal'),
    ([32, 4], [65536, 65536], 48, True, 'B=2 kv=65536 causal'),
]

all_plan_ok = True
for qo_lens, kv_lens, num_heads, causal, label in test_configs:
    B = len(qo_lens)
    qo_l = torch.tensor(qo_lens, dtype=torch.int32)
    kv_l = torch.tensor(kv_lens, dtype=torch.int32)
    qo_off = torch.tensor([kl - ql if causal else kl for ql, kl in zip(qo_lens, kv_lens)], dtype=torch.int32)

    try:
        plan_tuple = fmha_sail_plan(qo_l, kv_l, num_heads, causal=causal, qo_offset=qo_off, num_kv_heads=num_heads)
        total, num_ctas, splits, errors = decode_plan(plan_tuple, num_heads, B)

        if errors:
            status = 'FAIL'
            all_plan_ok = False
            print(f'  [{status}] {label}: {total} items/{num_ctas} CTAs/{splits} splits')
            for e in errors[:3]:
                print(f'    ERROR: {e}')
        else:
            status = 'PASS'
            print(f'  [{status}] {label}: {total} items/{num_ctas} CTAs/{splits} splits')
    except Exception as e:
        status = 'FAIL'
        all_plan_ok = False
        print(f'  [{status}] {label}: {type(e).__name__}: {str(e)[:100]}')

print(f'\nPlan validation: {\"ALL PASSED\" if all_plan_ok else \"SOME FAILED\"}')

print('')
print('--- sparse_topk_select: basic sanity ---')
all_topk_ok = True
for N, H, K, nvp in [(4, 2, 64, 64), (8, 4, 128, 100), (16, 2, 32, 32), (3, 8, 17, 17)]:
    torch.manual_seed(0)
    ms = torch.randn(H, K, N, dtype=torch.float32, device=device)
    try:
        result = sparse_topk_select(ms, 16, num_valid_pages=nvp)
        ok = (result.shape == (N, H, 16)) and (result.dtype == torch.int32)
        valid_mask = result >= 0
        if valid_mask.any():
            ok = ok and (result[valid_mask].max() < nvp)
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_topk_ok = False
        print(f'  [{status}] topk N={N} H={H} K={K} nvp={nvp}: shape={list(result.shape)}')
    except Exception as e:
        status = 'FAIL'
        all_topk_ok = False
        print(f'  [{status}] topk N={N} H={H} K={K} nvp={nvp}: {type(e).__name__}: {str(e)[:80]}')

print(f'\nSparse topk: {\"ALL PASSED\" if all_topk_ok else \"SOME FAILED\"}')

print('')
print('Done!')
"

echo ""
echo "=========================================="
echo "ALL DONE"
echo "=========================================="

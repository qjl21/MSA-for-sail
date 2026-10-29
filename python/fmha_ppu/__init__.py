# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FMHA varlen attention for PPU (HGGC).

Provides fmha_ppu_plan() and fmha_ppu() interfaces
with per-variant lazy JIT compilation.
"""

from .api import fmha_ppu, fmha_ppu_plan, sparse_topk_select, allocate_graph_workspace

__all__ = [
    "fmha_ppu_plan",
    "fmha_ppu",
    "sparse_topk_select",
    "allocate_graph_workspace",
]

try:
    import ctypes
    import tvm_ffi
    import torch

    _FP8_DTYPE_MAP = {
        'float8_e4m3fn': torch.float8_e4m3fn,
        'float8_e5m2': torch.float8_e5m2,
    }

    def _dlpack_capsule_set_dtype_int8(capsule):
        """Patch DLPack capsule dtype to int8 for zero-copy fp8→torch conversion.

        apache-tvm-ffi<=0.1.10 maps fp8 to DLPack type_code=10 (kDLBool)
        instead of 12/13, causing torch.from_dlpack to fail.
        Since fp8 and int8 are both 8-bit, we reinterpret the dtype field
        as int8 (kDLInt=0, bits=8, lanes=1) so torch can accept it.
        Remove this when apache-tvm-ffi ships correct fp8 DLPack codes.
        """
        get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        get_ptr.restype = ctypes.c_void_p
        get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
        dl_ptr = get_ptr(capsule, b'dltensor')
        # DLTensor layout: data(8) + device(8) + ndim(4) + dtype{code(1),bits(1),lanes(2)}
        dtype_addr = dl_ptr + 20
        ctypes.c_uint8.from_address(dtype_addr).value = 0      # kDLInt
        ctypes.c_uint8.from_address(dtype_addr + 1).value = 8   # 8 bits
        ctypes.c_uint16.from_address(dtype_addr + 2).value = 1  # 1 lane

    def _tvm_to_torch(x):
        if type(x).__module__ == 'tvm_ffi.core' and type(x).__name__ == 'Tensor':
            fp8_dtype = _FP8_DTYPE_MAP.get(str(x.dtype))
            if fp8_dtype is not None:
                capsule = x._to_dlpack()
                _dlpack_capsule_set_dtype_int8(capsule)
                return torch.from_dlpack(capsule).view(fp8_dtype)
            return torch.from_dlpack(x)
        if type(x).__module__ == 'tvm_ffi.container' and type(x).__name__ == 'Map':
            return {str(k): _tvm_to_torch(x[k]) for k in x}
        return x

    def _fmha_ppu_plan_ffi(*args):
        args = [_tvm_to_torch(a) for a in args]
        return fmha_ppu_plan(*args)

    def _fmha_ppu_ffi(*args):
        args = [_tvm_to_torch(a) for a in args]
        out, max_score = fmha_ppu(*args)
        return out, max_score

    def _sparse_topk_select_ffi(*args):
        args = [_tvm_to_torch(a) for a in args]
        return sparse_topk_select(*args)

    tvm_ffi.register_global_func("minfer.ops.fmha_ppu_plan", _fmha_ppu_plan_ffi)
    tvm_ffi.register_global_func("minfer.ops.fmha_ppu", _fmha_ppu_ffi)
    tvm_ffi.register_global_func("minfer.ops.sparse_topk_select", _sparse_topk_select_ffi)
except ImportError:
    pass

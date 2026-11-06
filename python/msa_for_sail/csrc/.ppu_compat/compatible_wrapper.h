// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef COMPATIBLE_WRAPPER_H
#define COMPATIBLE_WRAPPER_H

/*
 * compatible_wrapper.h — sailify compat layer for the msa_for_sail PPU build
 *
 * Force-included by jit.py into every hgcc translation unit.  Trimmed to
 * exactly what the converted sources consume:
 *   * <stdio.h>/<hggc.h> — pow_2.hpp and fmha_fusion.hpp call printf
 *     without including <cstdio>, and every TU uses hggcError_t /
 *     hggcStream_t from <hggc.h>.
 *   * COMPATIBLE_VERSION / __COMPATIBLECC_VER_* — CUDA/nvcc version gates
 *     in vec_dtypes.cuh, cutlass_utils.cuh and tvm_ffi_utils.h.
 *   * COMPATIBLE_ARCH — __CUDA_ARCH__ mirror, mapped from __HGGC_ARCH__.
 * The rest of what sailify generates (PPU SDK library headers, version
 * macros, GetVersion/GetProperty stubs, SDK fixups) served only the
 * unsupported-API stub layer and was removed with it.
 */

#include <stdio.h>
#include <hggc.h>

/* CUDA 13.0.0 / nvcc 13.0.0 equivalents (sailify version config). */
#define COMPATIBLE_VERSION 13000

#define __COMPATIBLECC_VER_MAJOR__  13
#define __COMPATIBLECC_VER_MINOR__  0
#define __COMPATIBLECC_VER_BUILD__  0

/* COMPATIBLE_ARCH mirrors __CUDA_ARCH__: device-only.
 * Do NOT use __HGGCCC__ here — it's defined for all -x cu code. */
#if defined(__HGGC_ARCH__)
  #ifndef COMPATIBLE_ARCH
    #if __HGGC_ARCH__ == 100
      #define COMPATIBLE_ARCH 800
    #elif __HGGC_ARCH__ == 150
      #define COMPATIBLE_ARCH 890
    #else
      #define COMPATIBLE_ARCH __HGGC_ARCH__
    #endif
  #endif
#endif

#endif /* COMPATIBLE_WRAPPER_H */

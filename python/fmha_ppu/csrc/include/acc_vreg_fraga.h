// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

CUTLASS_DEVICE int32_t get_lane_id_ppu() {
  return threadIdx.x % 32;
}

template <typename T, typename S, int N>
struct NumericArrayConverterPPU {
  using ElementCompute = T;
  CUTLASS_DEVICE bool is_source_needed() { return false; }
};

static constexpr int kConverterBaseUnit32 = 32;
static constexpr int kConverterBaseUnit8 = 8;

template <typename T>
struct NumericArrayConverterPPU<T, float, kConverterBaseUnit32> {
  using result_type = cutlass::Array<T, kConverterBaseUnit32>;
  using source_type = cutlass::Array<float, kConverterBaseUnit32>;
  using ElementCompute = T;

  CUTLASS_DEVICE result_type convert(source_type const& src_const) {
    result_type result;
    source_type src = src_const;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i++) {
      result[i] = static_cast<T>(src[i]);
    }

    result_type final_result;
    bool high_phase = get_lane_id_ppu() >> 4;
    uint32_t inner_idx = get_lane_id_ppu() & 0xf;
    uint32_t row = inner_idx >> 2;
    uint32_t col = inner_idx & 0x3;
    bool value1 = (!high_phase) && (row == col);
    bool value2 = (high_phase) && (row == col);
    uint32_t vreg0 = 0;
    uint32_t vreg1 = 0;
    uint32_t vreg2 = 0;
    uint32_t vreg3 = 0;
    vreg0 = value1 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c00 : 0x3f80) : vreg0;
    vreg0 = value2 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c000000 : 0x3f800000) : vreg0;
    vreg3 = vreg0;
    uint32_t weight[4] = {vreg0, vreg1, vreg2, vreg3};
    uint32_t zero_accum[4] = {0};

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 8) {
      float* d = reinterpret_cast<float*>(final_result.data() + i);
      float* a = reinterpret_cast<float*>(result.data() + i);
      float* b = reinterpret_cast<float*>(&weight[0]);
      float* c = reinterpret_cast<float*>(&zero_accum[0]);

      if (std::is_same<cutlass::half_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
        asm volatile(
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        asm volatile(
            "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
      } else if (std::is_same<cutlass::bfloat16_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
        asm volatile(
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        asm volatile(
            "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
      } else {
        assert(false && "NumericArrayConverterPPU only supports half_t and bfloat16_t outputs");
      }
    }
    return final_result;
  }

  CUTLASS_DEVICE bool is_source_needed() { return false; }
  CUTLASS_DEVICE result_type operator()(source_type const& s) { return convert(s); }
};

template <typename T>
struct NumericArrayConverterPPU<T, float, kConverterBaseUnit8> {
  using result_type = cutlass::Array<T, kConverterBaseUnit8>;
  using source_type = cutlass::Array<float, kConverterBaseUnit8>;
  using ElementCompute = T;

  CUTLASS_DEVICE result_type convert(source_type const& src_const) {
    result_type result;
    source_type src = src_const;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i++) {
      result[i] = static_cast<T>(src[i]);
    }

    result_type final_result;
    bool high_phase = get_lane_id_ppu() >> 4;
    uint32_t inner_idx = get_lane_id_ppu() & 0xf;
    uint32_t row = inner_idx >> 2;
    uint32_t col = inner_idx & 0x3;
    bool value1 = (!high_phase) && (row == col);
    bool value2 = (high_phase) && (row == col);
    uint32_t vreg0 = 0;
    uint32_t vreg1 = 0;
    uint32_t vreg2 = 0;
    uint32_t vreg3 = 0;
    vreg0 = value1 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c00 : 0x3f80) : vreg0;
    vreg0 = value2 ? (std::is_same<cutlass::half_t, T>::value ? 0x3c000000 : 0x3f800000) : vreg0;
    vreg3 = vreg0;
    uint32_t weight[4] = {vreg0, vreg1, vreg2, vreg3};
    uint32_t zero_accum[4] = {0};

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 8) {
      float* d = reinterpret_cast<float*>(final_result.data() + i);
      float* a = reinterpret_cast<float*>(result.data() + i);
      float* b = reinterpret_cast<float*>(&weight[0]);
      float* c = reinterpret_cast<float*>(&zero_accum[0]);

      if (std::is_same<cutlass::half_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
        asm volatile(
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        asm volatile(
            "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
      } else if (std::is_same<cutlass::bfloat16_t, T>::value) {
#if (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 100)
        asm volatile(
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#elif (defined __HGGC_ARCH__) && (__HGGC_ARCH__ == 150)
        asm volatile(
            "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11},"
            "{%12,%13,%14,%15};\n"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
              "f"(b[0]), "f"(b[1]), "f"(b[2]), "f"(b[3]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
      } else {
        assert(false && "NumericArrayConverterPPU only supports half_t and bfloat16_t outputs");
      }
    }
    return final_result;
  }

  CUTLASS_DEVICE bool is_source_needed() { return false; }
  CUTLASS_DEVICE result_type operator()(source_type const& s) { return convert(s); }
};

template <typename T, int N>
struct NumericArrayConverterPPU<T, float, N> {
  using result_type = cutlass::Array<T, N>;
  using source_type = cutlass::Array<float, N>;
  using ElementCompute = T;

  CUTLASS_DEVICE static result_type convert(source_type const& source) {
    static_assert(N % kConverterBaseUnit8 == 0,
                  "NumericArrayConverterPPU requires fragment size to be a multiple of 8");
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    result_type result;

    if constexpr (N % kConverterBaseUnit32 == 0) {
      constexpr int kVecWidth = kConverterBaseUnit32;
      NumericArrayConverterPPU<scalar_result_type, scalar_source_type, kVecWidth> convert_vector;
      using vec_result = cutlass::Array<scalar_result_type, kVecWidth>;
      using vec_source = cutlass::Array<scalar_source_type, kVecWidth>;
      vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
      vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / kVecWidth; ++i) {
        result_ptr[i] = convert_vector.convert(source_ptr[i]);
      }
    } else {
      constexpr int kVecWidth = kConverterBaseUnit8;
      NumericArrayConverterPPU<scalar_result_type, scalar_source_type, kVecWidth> convert_vector;
      using vec_result = cutlass::Array<scalar_result_type, kVecWidth>;
      using vec_source = cutlass::Array<scalar_source_type, kVecWidth>;
      vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
      vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / kVecWidth; ++i) {
        result_ptr[i] = convert_vector.convert(source_ptr[i]);
      }
    }

    return result;
  }

  CUTLASS_DEVICE bool is_source_needed() { return false; }
  CUTLASS_DEVICE result_type operator()(source_type const& s) { return convert(s); }
};

template <typename ToType, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_acc(Tensor<Engine, Layout> const& tensor) {
  using FromType = typename Engine::value_type;
  constexpr int kNumel = decltype(size(tensor))::value;
  NumericArrayConverterPPU<ToType, FromType, kNumel> convert_op;
  auto frag = convert_op(*reinterpret_cast<cutlass::Array<FromType, kNumel> const*>(tensor.data()));
  return make_tensor(make_rmem_ptr<ToType>(&frag), tensor.layout());
}

}  // namespace cutlass::fmha::collective

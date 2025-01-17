// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vclamp.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCLAMP__NEON_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__NEON_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__NEON_X4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__neon_x4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VCLAMP__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__NEON_X8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__NEON_X8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__neon_x8, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VCLAMP__RVV_X1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vclamp_ukernel__rvv_x1v, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__RVV_X1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x1v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__rvv_x1v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X1V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__rvv_x1v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__RVV_X1V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__rvv_x1v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VCLAMP__RVV_X2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__RVV_X2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 2 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X2V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__RVV_X2V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__rvv_x2v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VCLAMP__RVV_X4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__RVV_X4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X4V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__RVV_X4V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__rvv_x4v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VCLAMP__RVV_X8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__RVV_X8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__RVV_X8V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__RVV_X8V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__rvv_x8v, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__SSE_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VCLAMP__SSE_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X4, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VCLAMP__SSE_X4, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__sse_x4, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__SSE_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VCLAMP__SSE_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VCLAMP__SSE_X8, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_VCLAMP__SSE_X8, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__sse_x8, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__AVX_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VCLAMP__AVX_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_VCLAMP__AVX_X8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__avx_x8, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__AVX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VCLAMP__AVX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VCLAMP__AVX_X16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_VCLAMP__AVX_X16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__avx_x16, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__AVX512F_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__AVX512F_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X16, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__AVX512F_X16, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__avx512f_x16, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VCLAMP__AVX512F_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__AVX512F_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__AVX512F_X32, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__AVX512F_X32, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__avx512f_x32, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X4, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_ARM_X8, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASMSIMD_X86_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X4, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASMSIMD_X86_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASMSIMD_X86_X8, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASM_X1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vclamp_ukernel__wasm_x1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__WASM_X1, batch_gt_1) {
    for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasm_x1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X1, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasm_x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASM_X1, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasm_x1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASM_X2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__WASM_X2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X2, batch_gt_2) {
    for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X2, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASM_X2, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasm_x2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VCLAMP__WASM_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VCLAMP__WASM_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VCLAMP__WASM_X4, qmin) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_VCLAMP__WASM_X4, qmax) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f32_vclamp_ukernel__wasm_x4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VCLAMP__SCALAR_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vclamp_ukernel__scalar_x1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VCLAMP__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vclamp_ukernel__scalar_x1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X1, qmin) {
  for (uint8_t qmin = 1; qmin < 255; qmin++) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(qmin)
        .Test(xnn_f32_vclamp_ukernel__scalar_x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VCLAMP__SCALAR_X1, qmax) {
  for (uint8_t qmax = 1; qmax < 255; qmax++) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(qmax)
        .Test(xnn_f32_vclamp_ukernel__scalar_x1, xnn_init_f32_minmax_scalar_params);
    }
  }
}


TEST(F32_VCLAMP__SCALAR_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VCLAMP__SCALAR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X2, qmin) {
  for (uint8_t qmin = 1; qmin < 255; qmin++) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(qmin)
        .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VCLAMP__SCALAR_X2, qmax) {
  for (uint8_t qmax = 1; qmax < 255; qmax++) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(qmax)
        .Test(xnn_f32_vclamp_ukernel__scalar_x2, xnn_init_f32_minmax_scalar_params);
    }
  }
}


TEST(F32_VCLAMP__SCALAR_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VCLAMP__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VCLAMP__SCALAR_X4, qmin) {
  for (uint8_t qmin = 1; qmin < 255; qmin++) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(qmin)
        .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_VCLAMP__SCALAR_X4, qmax) {
  for (uint8_t qmax = 1; qmax < 255; qmax++) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(qmax)
        .Test(xnn_f32_vclamp_ukernel__scalar_x4, xnn_init_f32_minmax_scalar_params);
    }
  }
}

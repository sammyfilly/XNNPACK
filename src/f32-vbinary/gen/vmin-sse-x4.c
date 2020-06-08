// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vmin_ukernel__sse_x4(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);


  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const __m128 va0123 = _mm_loadu_ps(a);
    a += 4;

    const __m128 vb0123 = _mm_loadu_ps(b);
    b += 4;

    __m128 vy0123 = _mm_min_ps(va0123, vb0123);



    _mm_storeu_ps(y, vy0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const __m128 va0123 = _mm_loadu_ps(a);
    const __m128 vb0123 = _mm_loadu_ps(b);

    __m128 vy0123 = _mm_min_ps(va0123, vb0123);
    if (n & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) y, vy0123);
      vy0123 = _mm_movehl_ps(vy0123, vy0123);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      _mm_store_ss(y, vy0123);
    }
  }
}

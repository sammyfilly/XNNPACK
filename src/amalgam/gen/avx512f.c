// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/packw.h>
#include <xnnpack/prefetch.h>
#include <xnnpack/prelu.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>


void xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_loadu_ps(i4);
      i4 += 16;

      const __m512 vk4x0123456789ABCDEF = _mm512_load_ps(w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_loadu_ps(i5);
      i5 += 16;

      const __m512 vk5x0123456789ABCDEF = _mm512_load_ps(w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_loadu_ps(i6);
      i6 += 16;

      const __m512 vk6x0123456789ABCDEF = _mm512_load_ps(w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_loadu_ps(i7);
      i7 += 16;

      const __m512 vk7x0123456789ABCDEF = _mm512_load_ps(w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_loadu_ps(i8);
      i8 += 16;

      const __m512 vk8x0123456789ABCDEF = _mm512_load_ps(w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi9x0123456789ABCDEF = _mm512_loadu_ps(i9);
      i9 += 16;

      const __m512 vk9x0123456789ABCDEF = _mm512_load_ps(w + 160);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi10x0123456789ABCDEF = _mm512_loadu_ps(i10);
      i10 += 16;

      const __m512 vk10x0123456789ABCDEF = _mm512_load_ps(w + 176);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi11x0123456789ABCDEF = _mm512_loadu_ps(i11);
      i11 += 16;

      const __m512 vk11x0123456789ABCDEF = _mm512_load_ps(w + 192);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi12x0123456789ABCDEF = _mm512_loadu_ps(i12);
      i12 += 16;

      const __m512 vk12x0123456789ABCDEF = _mm512_load_ps(w + 208);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi13x0123456789ABCDEF = _mm512_loadu_ps(i13);
      i13 += 16;

      const __m512 vk13x0123456789ABCDEF = _mm512_load_ps(w + 224);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi14x0123456789ABCDEF = _mm512_loadu_ps(i14);
      i14 += 16;

      const __m512 vk14x0123456789ABCDEF = _mm512_load_ps(w + 240);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi15x0123456789ABCDEF = _mm512_loadu_ps(i15);
      i15 += 16;

      const __m512 vk15x0123456789ABCDEF = _mm512_load_ps(w + 256);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi16x0123456789ABCDEF = _mm512_loadu_ps(i16);
      i16 += 16;

      const __m512 vk16x0123456789ABCDEF = _mm512_load_ps(w + 272);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi17x0123456789ABCDEF = _mm512_loadu_ps(i17);
      i17 += 16;

      const __m512 vk17x0123456789ABCDEF = _mm512_load_ps(w + 288);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi18x0123456789ABCDEF = _mm512_loadu_ps(i18);
      i18 += 16;

      const __m512 vk18x0123456789ABCDEF = _mm512_load_ps(w + 304);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi19x0123456789ABCDEF = _mm512_loadu_ps(i19);
      i19 += 16;

      const __m512 vk19x0123456789ABCDEF = _mm512_load_ps(w + 320);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi20x0123456789ABCDEF = _mm512_loadu_ps(i20);
      i20 += 16;

      const __m512 vk20x0123456789ABCDEF = _mm512_load_ps(w + 336);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi21x0123456789ABCDEF = _mm512_loadu_ps(i21);
      i21 += 16;

      const __m512 vk21x0123456789ABCDEF = _mm512_load_ps(w + 352);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi22x0123456789ABCDEF = _mm512_loadu_ps(i22);
      i22 += 16;

      const __m512 vk22x0123456789ABCDEF = _mm512_load_ps(w + 368);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi23x0123456789ABCDEF = _mm512_loadu_ps(i23);
      i23 += 16;

      const __m512 vk23x0123456789ABCDEF = _mm512_load_ps(w + 384);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi24x0123456789ABCDEF = _mm512_loadu_ps(i24);
      i24 += 16;

      const __m512 vk24x0123456789ABCDEF = _mm512_load_ps(w + 400);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 416;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i4);
      const __m512 vk4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i5);
      const __m512 vk5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i6);
      const __m512 vk6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i7);
      const __m512 vk7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i8);
      const __m512 vk8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi9x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i9);
      const __m512 vk9x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 160);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi10x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i10);
      const __m512 vk10x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 176);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi11x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i11);
      const __m512 vk11x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 192);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi12x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i12);
      const __m512 vk12x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 208);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi13x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i13);
      const __m512 vk13x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 224);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi14x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i14);
      const __m512 vk14x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 240);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi15x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i15);
      const __m512 vk15x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 256);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi16x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i16);
      const __m512 vk16x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 272);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi17x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i17);
      const __m512 vk17x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 288);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi18x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i18);
      const __m512 vk18x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 304);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi19x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i19);
      const __m512 vk19x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 320);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi20x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i20);
      const __m512 vk20x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 336);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi21x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i21);
      const __m512 vk21x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 352);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi22x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i22);
      const __m512 vk22x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 368);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi23x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i23);
      const __m512 vk23x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 384);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi24x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i24);
      const __m512 vk24x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 400);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 64;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 80;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(w);
        __m512 vacc1p0 = _mm512_load_ps(w + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w + 32);
        const __m512 vk0x1 = _mm512_load_ps(w + 48);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 64);
        const __m512 vk1x1 = _mm512_load_ps(w + 80);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        const __m512 vk2x0 = _mm512_load_ps(w + 96);
        const __m512 vk2x1 = _mm512_load_ps(w + 112);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        const __m512 vk3x0 = _mm512_load_ps(w + 128);
        const __m512 vk3x1 = _mm512_load_ps(w + 144);
        vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);
        vacc1p1 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        const __m512 vk4x0 = _mm512_load_ps(w + 160);
        const __m512 vk4x1 = _mm512_load_ps(w + 176);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 192;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(w);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w + 16);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 48);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 64);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 80);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 96;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        const __m512 vk0x0 = _mm512_load_ps(w);
        const __m512 vk0x1 = _mm512_load_ps(w + 16);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        const __m512 vk1x0 = _mm512_load_ps(w + 32);
        const __m512 vk1x1 = _mm512_load_ps(w + 48);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        const __m512 vk2x0 = _mm512_load_ps(w + 64);
        const __m512 vk2x1 = _mm512_load_ps(w + 80);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        const __m512 vk3x0 = _mm512_load_ps(w + 96);
        const __m512 vk3x1 = _mm512_load_ps(w + 112);
        vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);
        vacc1p1 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        const __m512 vk4x0 = _mm512_load_ps(w + 128);
        const __m512 vk4x1 = _mm512_load_ps(w + 144);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 160;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        _mm512_store_ps(b, vacc0p0);
        _mm512_store_ps(b + 16, vacc1p0);
        b += 32;
      }

      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
        b += 16;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));
        __m512 vaccp0 = _mm512_load_ps(b);


        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);

        const __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);

        const __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);

        const __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);

        const __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);

        const __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        _mm512_store_ps(b, vaccp0);
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m512 vacc0p0 = _mm512_load_ps(b);
        __m512 vacc1p0 = _mm512_load_ps(b + 16);
        b += 32;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        const __m512 vi0x1 = _mm512_loadu_ps(i0 + 16);
        i0 += 32;

        __m512 vk0x0 = _mm512_load_ps(w);
        __m512 vk0x1 = _mm512_load_ps(w + 16);

        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi0x1, vk0x1, vacc1p0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        const __m512 vi1x1 = _mm512_loadu_ps(i1 + 16);
        i1 += 32;

        __m512 vk1x0 = _mm512_load_ps(w + 32);
        __m512 vk1x1 = _mm512_load_ps(w + 48);

        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
        __m512 vacc1p1 = _mm512_mul_ps(vi1x1, vk1x1);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        const __m512 vi2x1 = _mm512_loadu_ps(i2 + 16);
        i2 += 32;

        __m512 vk2x0 = _mm512_load_ps(w + 64);
        __m512 vk2x1 = _mm512_load_ps(w + 80);

        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi2x1, vk2x1, vacc1p0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        const __m512 vi3x1 = _mm512_loadu_ps(i3 + 16);
        i3 += 32;

        __m512 vk3x0 = _mm512_load_ps(w + 96);
        __m512 vk3x1 = _mm512_load_ps(w + 112);

        vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);
        vacc1p1 = _mm512_fmadd_ps(vi3x1, vk3x1, vacc1p1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        const __m512 vi4x1 = _mm512_loadu_ps(i4 + 16);
        i4 += 32;

        __m512 vk4x0 = _mm512_load_ps(w + 128);
        __m512 vk4x1 = _mm512_load_ps(w + 144);

        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
        vacc1p0 = _mm512_fmadd_ps(vi4x1, vk4x1, vacc1p0);

        w += 160;

        // Add up all accumulators to vacc0p0
        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
        vacc1p0 = _mm512_add_ps(vacc1p0, vacc1p1);

        __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
        __m512 vacc1 = _mm512_max_ps(vmin, vacc1p0);

        vacc0 = _mm512_min_ps(vmax, vacc0);
        vacc1 = _mm512_min_ps(vmax, vacc1);

        _mm512_storeu_ps(output, vacc0);
        _mm512_storeu_ps(output + 16, vacc1);
        output += 32;
      }


      for (; c >= 16; c -= 16) {
        __m512 vaccp0 = _mm512_load_ps(b);
        b += 16;


        const __m512 vi0x0 = _mm512_loadu_ps(i0);
        i0 += 16;

        __m512 vk0x0 = _mm512_load_ps(w);

        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_loadu_ps(i1);
        i1 += 16;

        __m512 vk1x0 = _mm512_load_ps(w + 16);

        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_loadu_ps(i2);
        i2 += 16;

        __m512 vk2x0 = _mm512_load_ps(w + 32);

        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_loadu_ps(i3);
        i3 += 16;

        __m512 vk3x0 = _mm512_load_ps(w + 48);

        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_loadu_ps(i4);
        i4 += 16;

        __m512 vk4x0 = _mm512_load_ps(w + 64);

        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        w += 80;


        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vmin, vaccp0);

        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_storeu_ps(output, vacc);
        output += 16;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 15);
        __m512 vaccp0 = _mm512_load_ps(b);
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

        const __m512 vi0x0 = _mm512_maskz_loadu_ps(vmask, i0);
        __m512 vk0x0 = _mm512_load_ps(w);
        vaccp0 = _mm512_fmadd_ps(vi0x0, vk0x0, vaccp0);

        const __m512 vi1x0 = _mm512_maskz_loadu_ps(vmask, i1);
        __m512 vk1x0 = _mm512_load_ps(w + 16);
        __m512 vaccp1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = _mm512_maskz_loadu_ps(vmask, i2);
        __m512 vk2x0 = _mm512_load_ps(w + 32);
        vaccp0 = _mm512_fmadd_ps(vi2x0, vk2x0, vaccp0);

        const __m512 vi3x0 = _mm512_maskz_loadu_ps(vmask, i3);
        __m512 vk3x0 = _mm512_load_ps(w + 48);
        vaccp1 = _mm512_fmadd_ps(vi3x0, vk3x0, vaccp1);

        const __m512 vi4x0 = _mm512_maskz_loadu_ps(vmask, i4);
        __m512 vk4x0 = _mm512_load_ps(w + 64);
        vaccp0 = _mm512_fmadd_ps(vi4x0, vk4x0, vaccp0);

        // Add up all accumulators to vaccp0
        vaccp0 = _mm512_add_ps(vaccp0, vaccp1);

        __m512 vacc = _mm512_max_ps(vmin, vaccp0);
        vacc = _mm512_min_ps(vmax, vacc);

        _mm512_mask_storeu_ps(output, vmask, vacc);
        output += c;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m512 vacc0123456789ABCDEFp0 = _mm512_load_ps(w);


      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;

      const __m512 vk0x0123456789ABCDEF = _mm512_load_ps(w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __m512 vk1x0123456789ABCDEF = _mm512_load_ps(w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_loadu_ps(i2);
      i2 += 16;

      const __m512 vk2x0123456789ABCDEF = _mm512_load_ps(w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_loadu_ps(i3);
      i3 += 16;

      const __m512 vk3x0123456789ABCDEF = _mm512_load_ps(w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_loadu_ps(i4);
      i4 += 16;

      const __m512 vk4x0123456789ABCDEF = _mm512_load_ps(w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_loadu_ps(i5);
      i5 += 16;

      const __m512 vk5x0123456789ABCDEF = _mm512_load_ps(w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_loadu_ps(i6);
      i6 += 16;

      const __m512 vk6x0123456789ABCDEF = _mm512_load_ps(w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_loadu_ps(i7);
      i7 += 16;

      const __m512 vk7x0123456789ABCDEF = _mm512_load_ps(w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_loadu_ps(i8);
      i8 += 16;

      const __m512 vk8x0123456789ABCDEF = _mm512_load_ps(w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      w += 160;


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_storeu_ps(output, vacc0123456789ABCDEF);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 16);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << c) - UINT32_C(1)));

      __m512 vacc0123456789ABCDEFp0 = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i0);
      const __m512 vk0x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 16);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i1);
      const __m512 vk1x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 32);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i2);
      const __m512 vk2x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 48);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i3);
      const __m512 vk3x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 64);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i4);
      const __m512 vk4x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 80);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i5);
      const __m512 vk5x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 96);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i6);
      const __m512 vk6x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 112);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i7);
      const __m512 vk7x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 128);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF, vacc0123456789ABCDEFp0);

      const __m512 vi8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, i8);
      const __m512 vk8x0123456789ABCDEF = _mm512_maskz_loadu_ps(vmask, w + 144);
      vacc0123456789ABCDEFp0 = _mm512_fmadd_ps(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF, vacc0123456789ABCDEFp0);


      __m512 vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEFp0);
      vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

      _mm512_mask_storeu_ps(output, vmask, vacc0123456789ABCDEF);
      output += c;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);

      a0 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
      const __m512 va3 = _mm512_set1_ps(*a3);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
      const __m512 va4 = _mm512_set1_ps(*a4);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
      const __m512 va5 = _mm512_set1_ps(*a5);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
      const __m512 va6 = _mm512_set1_ps(*a6);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a6 = (const float*) ((uintptr_t) a6 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c6, vmask, vacc6x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
        w += 16;

        const __m512 va0 = _mm512_set1_ps(*a0);
        vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);

        a0 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
        w += 16;

        const __m512 va0 = _mm512_set1_ps(*a0);
        vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
        const __m512 va1 = _mm512_set1_ps(*a1);
        vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
        const __m512 va2 = _mm512_set1_ps(*a2);
        vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
        const __m512 va3 = _mm512_set1_ps(*a3);
        vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
        const __m512 va4 = _mm512_set1_ps(*a4);
        vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
        const __m512 va5 = _mm512_set1_ps(*a5);
        vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
        const __m512 va6 = _mm512_set1_ps(*a6);
        vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);

        a0 += 1;
        a1 += 1;
        a2 += 1;
        a3 += 1;
        a4 += 1;
        a5 += 1;
        a6 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c6, vmask, vacc6x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_prelu_ukernel__avx512f_2x16(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const __m512 vzero = _mm512_setzero_ps();
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const __m512 vw0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;
      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __mmask16 vsign0x0123456789ABCDEF = _mm512_cmp_ps_mask(vi0x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc0x0123456789ABCDEF = _mm512_mask_mul_ps(vi0x0123456789ABCDEF, vsign0x0123456789ABCDEF, vi0x0123456789ABCDEF, vw0123456789ABCDEF);
      const __mmask16 vsign1x0123456789ABCDEF = _mm512_cmp_ps_mask(vi1x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc1x0123456789ABCDEF = _mm512_mask_mul_ps(vi1x0123456789ABCDEF, vsign1x0123456789ABCDEF, vi1x0123456789ABCDEF, vw0123456789ABCDEF);

      _mm512_storeu_ps(o0, vacc0x0123456789ABCDEF);
      o0 += 16;
      _mm512_storeu_ps(o1, vacc1x0123456789ABCDEF);
      o1 += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1 * sizeof(float));
      assert(c <= 15 * sizeof(float));
      // Prepare mask for valid 32-bit elements (depends on c).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (c >> XNN_LOG2_SIZEOF_FLOAT)) - UINT32_C(1)));

      const __m512 vw = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0 = _mm512_maskz_loadu_ps(vmask, i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m512 vi1 = _mm512_maskz_loadu_ps(vmask, i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __mmask16 vsign0 = _mm512_cmp_ps_mask(vi0, vzero, _CMP_LT_OQ);
      const __m512 vacc0 = _mm512_mask_mul_ps(vi0, vsign0, vi0, vw);
      const __mmask16 vsign1 = _mm512_cmp_ps_mask(vi1, vzero, _CMP_LT_OQ);
      const __m512 vacc1 = _mm512_mask_mul_ps(vi1, vsign1, vi1, vw);

      _mm512_mask_storeu_ps(o0, vmask, vacc0);
      o0 = (float*) ((uintptr_t) o0 + c);
      _mm512_mask_storeu_ps(o1, vmask, vacc1);
      o1 = (float*) ((uintptr_t) o1 + c);
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_vadd_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_add_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_add_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_add_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vaddc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_add_ps(vacc0, vb);
    vacc1 = _mm512_add_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_add_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_add_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vdiv_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_div_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vdivc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vacc0, vb);
    vacc1 = _mm512_div_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_max_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_max_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_max_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;


    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_max_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmaxc_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_max_ps(vacc0, vb);
    vacc1 = _mm512_max_ps(vacc1, vb);



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_max_ps(vacc, vb);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_max_ps(vmask, vacc, vb);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmin_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_min_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_min_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_min_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;


    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_min_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vminc_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_min_ps(vacc0, vb);
    vacc1 = _mm512_min_ps(vacc1, vb);



    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_min_ps(vacc, vb);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_min_ps(vmask, vacc, vb);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmul_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_mul_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_mul_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_mul_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vmulc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_mul_ps(vacc0, vb);
    vacc1 = _mm512_mul_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_mul_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrdivc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_div_ps(vb, vacc0);
    vacc1 = _mm512_div_ps(vb, vacc1);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_div_ps(vb, vacc);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_div_ps(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrsubc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vb, vacc0);
    vacc1 = _mm512_sub_ps(vb, vacc1);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vb, vacc);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsqrdiff_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_sub_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;

    vacc0 = _mm512_mul_ps(vacc0, vacc0);
    vacc1 = _mm512_mul_ps(vacc1, vacc1);


    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_mul_ps(vacc, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsqrdiffc_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, vb);
    vacc1 = _mm512_sub_ps(vacc1, vb);

    vacc0 = _mm512_mul_ps(vacc0, vacc0);
    vacc1 = _mm512_mul_ps(vacc1, vacc1);


    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, vb);
    vacc = _mm512_mul_ps(vacc, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_mul_ps(vmask, vacc, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsub_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, _mm512_loadu_ps(input_b));
    vacc1 = _mm512_sub_ps(vacc1, _mm512_loadu_ps(input_b + 16));
    input_b += 32;


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, _mm512_loadu_ps(input_b));
    input_b += 16;

    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, _mm512_maskz_loadu_ps(vmask, input_b));
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vsubc_minmax_ukernel__avx512f_x32(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  const __m512 vb = _mm512_set1_ps(*input_b);

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vacc0 = _mm512_loadu_ps(input_a);
    __m512 vacc1 = _mm512_loadu_ps(input_a + 16);
    input_a += 32;

    vacc0 = _mm512_sub_ps(vacc0, vb);
    vacc1 = _mm512_sub_ps(vacc1, vb);


    vacc0 = _mm512_max_ps(voutput_min, vacc0);
    vacc1 = _mm512_max_ps(voutput_min, vacc1);

    vacc0 = _mm512_min_ps(voutput_max, vacc0);
    vacc1 = _mm512_min_ps(voutput_max, vacc1);

    _mm512_storeu_ps(output, vacc0);
    _mm512_storeu_ps(output + 16, vacc1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc = _mm512_loadu_ps(input_a);
    input_a += 16;

    vacc = _mm512_sub_ps(vacc, vb);
    vacc = _mm512_max_ps(voutput_min, vacc);
    vacc = _mm512_min_ps(voutput_max, vacc);

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input_a);
    vacc = _mm512_maskz_sub_ps(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ps(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ps(vmask, voutput_max, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vclamp_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    vacc0123456789ABCDEF = _mm512_max_ps(vmin, vacc0123456789ABCDEF);

    vacc0123456789ABCDEF = _mm512_min_ps(vmax, vacc0123456789ABCDEF);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input);
    vacc = _mm512_max_ps(vmin, vacc);
    vacc = _mm512_min_ps(vmax, vacc);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vprescale = _mm512_set1_ps(params->avx512_rr1_lut16_p3.prescale);
  const __m512 valpha = _mm512_set1_ps(params->avx512_rr1_lut16_p3.alpha);
  const __m512 vbeta = _mm512_set1_ps(params->avx512_rr1_lut16_p3.beta);
  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_rr1_lut16_p3.sat_cutoff);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr1_lut16_p3.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr1_lut16_p3.log2e);
  const __m512 vminus_ln2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.minus_ln2);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr1_lut16_p3.c2);
  const __m512i vtable = _mm512_load_si512(params->avx512_rr1_lut16_p3.table);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    __m512 vx0 = _mm512_loadu_ps(input);
    __m512 vx1 = _mm512_loadu_ps(input + 16);
    __m512 vx2 = _mm512_loadu_ps(input + 32);
    __m512 vx3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vz0 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx0, vprescale));
    const __m512 vz1 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx1, vprescale));
    const __m512 vz2 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx2, vprescale));
    const __m512 vz3 = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx3, vprescale));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m512i ven0 = _mm512_slli_epi32(_mm512_castps_si512(vn0), 19);
    const __m512i vl0 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn0), vtable);
    const __m512i ven1 = _mm512_slli_epi32(_mm512_castps_si512(vn1), 19);
    const __m512i vl1 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn1), vtable);
    const __m512i ven2 = _mm512_slli_epi32(_mm512_castps_si512(vn2), 19);
    const __m512i vl2 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn2), vtable);
    const __m512i ven3 = _mm512_slli_epi32(_mm512_castps_si512(vn3), 19);
    const __m512i vl3 = _mm512_permutexvar_epi32(_mm512_castps_si512(vn3), vtable);

    __m512 vs0 = _mm512_castsi512_ps(_mm512_add_epi32(vl0, ven0));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    __m512 vs1 = _mm512_castsi512_ps(_mm512_add_epi32(vl1, ven1));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    __m512 vs2 = _mm512_castsi512_ps(_mm512_add_epi32(vl2, ven2));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    __m512 vs3 = _mm512_castsi512_ps(_mm512_add_epi32(vl3, ven3));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2, vz3);

    __m512 vp0 = _mm512_fmadd_ps(vc3, vt0, vc2);
    __m512 vp1 = _mm512_fmadd_ps(vc3, vt1, vc2);
    __m512 vp2 = _mm512_fmadd_ps(vc3, vt2, vc2);
    __m512 vp3 = _mm512_fmadd_ps(vc3, vt3, vc2);

    vp0 = _mm512_mul_ps(vp0, vt0);
    vt0 = _mm512_mul_ps(vt0, vs0);
    vp1 = _mm512_mul_ps(vp1, vt1);
    vt1 = _mm512_mul_ps(vt1, vs1);
    vp2 = _mm512_mul_ps(vp2, vt2);
    vt2 = _mm512_mul_ps(vt2, vs2);
    vp3 = _mm512_mul_ps(vp3, vt3);
    vt3 = _mm512_mul_ps(vt3, vs3);

    vs0 = _mm512_fmsub_ps(vs0, valpha, valpha);
    vs1 = _mm512_fmsub_ps(vs1, valpha, valpha);
    vs2 = _mm512_fmsub_ps(vs2, valpha, valpha);
    vs3 = _mm512_fmsub_ps(vs3, valpha, valpha);

    vp0 = _mm512_fmadd_ps(vp0, vt0, vt0);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vt1);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vt2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vt3);

    const __m512 vzero = _mm512_setzero_ps();
    __m512 vy0 = _mm512_fmadd_ps(vp0, valpha, vs0);
    const __mmask16 vsign0 = _mm512_cmp_ps_mask(vx0, vzero, _CMP_NLT_US);
    __m512 vy1 = _mm512_fmadd_ps(vp1, valpha, vs1);
    const __mmask16 vsign1 = _mm512_cmp_ps_mask(vx1, vzero, _CMP_NLT_US);
    __m512 vy2 = _mm512_fmadd_ps(vp2, valpha, vs2);
    const __mmask16 vsign2 = _mm512_cmp_ps_mask(vx2, vzero, _CMP_NLT_US);
    __m512 vy3 = _mm512_fmadd_ps(vp3, valpha, vs3);
    const __mmask16 vsign3 = _mm512_cmp_ps_mask(vx3, vzero, _CMP_NLT_US);

    vy0 = _mm512_mask_mul_ps(vy0, vsign0, vx0, vbeta);
    vy1 = _mm512_mask_mul_ps(vy1, vsign1, vx1, vbeta);
    vy2 = _mm512_mask_mul_ps(vy2, vsign2, vx2, vbeta);
    vy3 = _mm512_mask_mul_ps(vy3, vsign3, vx3, vbeta);

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vz = _mm512_max_ps(vsat_cutoff, _mm512_mul_ps(vx, vprescale));
    const __mmask16 vsign = _mm512_cmp_ps_mask(vx, _mm512_setzero_ps(), _CMP_NLT_US);

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512i ven = _mm512_slli_epi32(_mm512_castps_si512(vn), 19);
    const __m512i vl = _mm512_permutexvar_epi32(_mm512_castps_si512(vn), vtable);
    __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ven));
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2, vz);

    __m512 vp = _mm512_fmadd_ps(vc3, vt, vc2);
    vp = _mm512_mul_ps(vp, vt);

    vt = _mm512_mul_ps(vt, vs);
    vs = _mm512_fmsub_ps(vs, valpha, valpha);
    vp = _mm512_fmadd_ps(vp, vt, vt);
    __m512 vy = _mm512_fmadd_ps(vp, valpha, vs);

    vy = _mm512_mask_mul_ps(vy, vsign, vx, vbeta);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vhswish_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsixth = _mm512_set1_ps(params->avx512.sixth);
  const __m512 vhalf = _mm512_set1_ps(params->avx512.half);
  const __m512 vone = _mm512_set1_ps(params->avx512.one);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    __m512 vacc = _mm512_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm512_max_ps(vacc, vzero);
    vacc = _mm512_min_ps(vacc, vone);
    vacc = _mm512_mul_ps(vacc, vx);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vlrelu_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vslope = _mm512_set1_ps(params->scalar.slope);
  const __m512 vzero = _mm512_setzero_ps();

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vacc0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __mmask16 vsign0123456789ABCDEF = _mm512_cmp_ps_mask(vacc0123456789ABCDEF, vzero, _CMP_LT_OQ);

    vacc0123456789ABCDEF = _mm512_mask_mul_ps(vacc0123456789ABCDEF, vsign0123456789ABCDEF, vacc0123456789ABCDEF, vslope);

    _mm512_storeu_ps(output, vacc0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vacc = _mm512_maskz_loadu_ps(vmask, input);
    const __mmask16 vsign = _mm512_cmp_ps_mask(vacc, vzero, _CMP_LT_OQ);
    vacc = _mm512_mask_mul_ps(vacc, vsign, vacc, vslope);
    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_vrndd_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEG_INF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEG_INF);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndne_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_NEAREST_INT);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_NEAREST_INT);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndu_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_POS_INF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_POS_INF);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vrndz_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_roundscale_ps(vx0123456789ABCDEF, _MM_FROUND_TO_ZERO);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_maskz_roundscale_ps(vmask, vx, _MM_FROUND_TO_ZERO);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_rr2_lut32_p2.sign_mask);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_rr2_lut32_p2.magic_bias);
  const __m512 vlog2e = _mm512_set1_ps(params->avx512_rr2_lut32_p2.log2e);
  const __m512 vtable_lo = _mm512_load_ps(params->avx512_rr2_lut32_p2.table_lo);
  const __m512 vtable_hi = _mm512_load_ps(params->avx512_rr2_lut32_p2.table_hi);
  const __m512 vminus_ln2_hi = _mm512_set1_ps(params->avx512_rr2_lut32_p2.minus_ln2_hi);
  const __m512 vminus_ln2_lo = _mm512_set1_ps(params->avx512_rr2_lut32_p2.minus_ln2_lo);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_rr2_lut32_p2.c2);
  const __m512 vc1 = _mm512_set1_ps(params->avx512_rr2_lut32_p2.c1);
  const __m512 vone = _mm512_set1_ps(params->avx512_rr2_lut32_p2.one);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vz0 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx0), vsign_mask));
    const __m512 vz1 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx1), vsign_mask));
    const __m512 vz2 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx2), vsign_mask));
    const __m512 vz3 = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx3), vsign_mask));

    __m512 vn0 = _mm512_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vlog2e, vmagic_bias);

    const __m512 vl0 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn0), vtable_hi);
    const __m512 vl1 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn1), vtable_hi);
    const __m512 vl2 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn2), vtable_hi);
    const __m512 vl3 = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn3), vtable_hi);

    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    __m512 vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_hi, vz0);
    __m512 vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_hi, vz1);
    __m512 vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_hi, vz2);
    __m512 vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_hi, vz3);

    vt0 = _mm512_fmadd_ps(vn0, vminus_ln2_lo, vt0);
    vt1 = _mm512_fmadd_ps(vn1, vminus_ln2_lo, vt1);
    vt2 = _mm512_fmadd_ps(vn2, vminus_ln2_lo, vt2);
    vt3 = _mm512_fmadd_ps(vn3, vminus_ln2_lo, vt3);

    __m512 vp0 = _mm512_fmadd_ps(vt0, vc2, vc1);
    __m512 vp1 = _mm512_fmadd_ps(vt1, vc2, vc1);
    __m512 vp2 = _mm512_fmadd_ps(vt2, vc2, vc1);
    __m512 vp3 = _mm512_fmadd_ps(vt3, vc2, vc1);

    vt0 = _mm512_mul_ps(vt0, vl0);
    vt1 = _mm512_mul_ps(vt1, vl1);
    vt2 = _mm512_mul_ps(vt2, vl2);
    vt3 = _mm512_mul_ps(vt3, vl3);

    vp0 = _mm512_fmadd_ps(vt0, vp0, vl0);
    vp1 = _mm512_fmadd_ps(vt1, vp1, vl1);
    vp2 = _mm512_fmadd_ps(vt2, vp2, vl2);
    vp3 = _mm512_fmadd_ps(vt3, vp3, vl3);

    const __m512 ve0 = _mm512_scalef_ps(vp0, vn0);
    const __m512 ve1 = _mm512_scalef_ps(vp1, vn1);
    const __m512 ve2 = _mm512_scalef_ps(vp2, vn2);
    const __m512 ve3 = _mm512_scalef_ps(vp3, vn3);

    const __m512 vd0 = _mm512_add_ps(ve0, vone);
    const __m512 vd1 = _mm512_add_ps(ve1, vone);
    const __m512 vd2 = _mm512_add_ps(ve2, vone);
    const __m512 vd3 = _mm512_add_ps(ve3, vone);

    __m512 vf0 = _mm512_div_ps(ve0, vd0);
    __m512 vf1 = _mm512_div_ps(ve1, vd1);
    __m512 vf2 = _mm512_div_ps(ve2, vd2);
    __m512 vf3 = _mm512_div_ps(ve3, vd3);

    vf0 = _mm512_mask_sub_ps(vf0, _mm512_testn_epi32_mask(_mm512_castps_si512(vx0), vsign_mask), vone, vf0);
    vf1 = _mm512_mask_sub_ps(vf1, _mm512_testn_epi32_mask(_mm512_castps_si512(vx1), vsign_mask), vone, vf1);
    vf2 = _mm512_mask_sub_ps(vf2, _mm512_testn_epi32_mask(_mm512_castps_si512(vx2), vsign_mask), vone, vf2);
    vf3 = _mm512_mask_sub_ps(vf3, _mm512_testn_epi32_mask(_mm512_castps_si512(vx3), vsign_mask), vone, vf3);

    _mm512_storeu_ps(output, vf0);
    _mm512_storeu_ps(output + 16, vf1);
    _mm512_storeu_ps(output + 32, vf2);
    _mm512_storeu_ps(output + 48, vf3);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vz);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_storeu_ps(output, vf);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vz = _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(vx), vsign_mask));

    __m512 vn = _mm512_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m512 vl = _mm512_permutex2var_ps(vtable_lo, _mm512_castps_si512(vn), vtable_hi);
    vn = _mm512_sub_ps(vn, vmagic_bias);

    __m512 vt = _mm512_fmadd_ps(vn, vminus_ln2_hi, vz);
    vt = _mm512_fmadd_ps(vn, vminus_ln2_lo, vt);

    __m512 vp = _mm512_fmadd_ps(vt, vc2, vc1);
    vt = _mm512_mul_ps(vt, vl);
    vp = _mm512_fmadd_ps(vt, vp, vl);

    const __m512 ve = _mm512_scalef_ps(vp, vn);
    const __m512 vd = _mm512_add_ps(ve, vone);

    __m512 vf = _mm512_div_ps(ve, vd);

    vf = _mm512_mask_sub_ps(vf, _mm512_testn_epi32_mask(_mm512_castps_si512(vx), vsign_mask), vone, vf);

    _mm512_mask_storeu_ps(output, vmask, vf);
  }
}

void xnn_f32_vabs_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vnonsign_mask = _mm512_set1_epi32((int) params->avx512.nonsign_mask);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(input);
    input += 16;

    const __m512i vy0123456789ABCDEF = _mm512_and_epi32(vx0123456789ABCDEF, vnonsign_mask);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, input);
    const __m512i vy = _mm512_and_epi32(vx, vnonsign_mask);
    _mm512_mask_storeu_epi32(output, vmask, vy);
  }
}

void xnn_f32_vneg_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_neg_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512.sign_mask);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512i vx0123456789ABCDEF = _mm512_loadu_si512(input);
    input += 16;

    const __m512i vy0123456789ABCDEF = _mm512_xor_epi32(vx0123456789ABCDEF, vsign_mask);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512i vx = _mm512_maskz_loadu_epi32(vmask, input);
    const __m512i vy = _mm512_xor_epi32(vx, vsign_mask);
    _mm512_mask_storeu_epi32(output, vmask, vy);
  }
}

void xnn_f32_vsqr_ukernel__avx512f_x16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx0123456789ABCDEF = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vy0123456789ABCDEF = _mm512_mul_ps(vx0123456789ABCDEF, vx0123456789ABCDEF);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));
    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);
    const __m512 vy = _mm512_mul_ps(vx, vx);
    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m512 vb0 = _mm512_loadu_ps(b);
        _mm512_store_ps(packed_w, vb0);
        b += 16;
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
      }
      packed_w += 16;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
      const float* w8 = w7 + kc;
      const float* w9 = w8 + kc;
      const float* w10 = w9 + kc;
      const float* w11 = w10 + kc;
      const float* w12 = w11 + kc;
      const float* w13 = w12 + kc;
      const float* w14 = w13 + kc;
      const float* w15 = w14 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        __m512 v3x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w15), 3);
        w15 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
         // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v1x0123);
        _mm512_store_ps(packed_w + 32, v2x0123);
        _mm512_store_ps(packed_w + 48, v3x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;
          __m128 v15 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w15));
          w15 += 2;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;
          __m128 v15 = _mm_load_ss(w15);
          w15 += 1;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v15);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m512 vzero = _mm512_setzero_ps();
        _mm512_store_ps(packed_w, vzero);
        packed_w += 16;
      }

      // NR remainder has less than 16 rows so last row is not loaded
      // For SR=4 the
      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      // KC main loop multiple of 4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 4x4
        // a b c d
        // e f g h
        // i j k l
        // m n o p
        // Load first 4 rows of N into low part of each register
        __m512 v0x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w0));
        w0 += 4;
        __m512 v1x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w1));
        w1 += 4;
        __m512 v2x0123 = _mm512_castps128_ps512(_mm_loadu_ps(w2));
        w2 += 4;
        // castps leaves upper 128 bits undefined, so zero them.
        __m512 v3x0123 = _mm512_zextps128_ps512(_mm_loadu_ps(w3));
        w3 += 4;
        // Load next 4 rows of N into the high part of each register
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w4), 1);
        w4 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w5), 1);
        w5 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w6), 1);
        w6 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w7), 1);
        w7 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w8), 2);
        w8 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w9), 2);
        w9 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w10), 2);
        w10 += 4;
        v3x0123 = _mm512_insertf32x4(v3x0123, _mm_loadu_ps(w11), 2);
        w11 += 4;
        v0x0123 = _mm512_insertf32x4(v0x0123, _mm_loadu_ps(w12), 3);
        w12 += 4;
        v1x0123 = _mm512_insertf32x4(v1x0123, _mm_loadu_ps(w13), 3);
        w13 += 4;
        v2x0123 = _mm512_insertf32x4(v2x0123, _mm_loadu_ps(w14), 3);
        w14 += 4;

        // Transpose 2x2
        const __m512 vtmp0x0123 = _mm512_unpacklo_ps(v0x0123, v1x0123);  // a e b f   from row 0, 1
        const __m512 vtmp1x0123 = _mm512_unpacklo_ps(v2x0123, v3x0123);  // i m j n   from row 2, 3
        const __m512 vtmp2x0123 = _mm512_unpackhi_ps(v0x0123, v1x0123);  // c g d h   from row 0, 1
        const __m512 vtmp3x0123 = _mm512_unpackhi_ps(v2x0123, v3x0123);  // k o l p   from row 2, 3
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        // Transpose 4x4
        v0x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // a e i m   from row 0, 1
        v1x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp0x0123), _mm512_castps_pd(vtmp1x0123)));  // b f j n   from row 0, 1
        v2x0123 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // c g k o   from row 2, 3
        v3x0123 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(vtmp2x0123), _mm512_castps_pd(vtmp3x0123)));  // d h l p   from row 2, 3

        _mm512_store_ps(packed_w, v0x0123);
        _mm512_store_ps(packed_w + 16, v1x0123);
        _mm512_store_ps(packed_w + 32, v2x0123);
        _mm512_store_ps(packed_w + 48, v3x0123);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        if (k & 2) {
          // Read blocks of 4x2
          // a b
          // c d
          // e f
          // g h
          __m128 v0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
          w0 += 2;
          __m128 v1 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
          w1 += 2;
          __m128 v2 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
          w2 += 2;
          __m128 v3 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
          w3 += 2;
          __m128 v4 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
          w4 += 2;
          __m128 v5 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
          w5 += 2;
          __m128 v6 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
          w6 += 2;
          __m128 v7 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
          w7 += 2;
          __m128 v8 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
          w8 += 2;
          __m128 v9 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
          w9 += 2;
          __m128 v10 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
          w10 += 2;
          __m128 v11 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
          w11 += 2;
          __m128 v12 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
          w12 += 2;
          __m128 v13 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
          w13 += 2;
          __m128 v14 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
          w14 += 2;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a c b d   from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // e g f h   from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a c e g   from row 0, 1
          v1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a c b d   from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // e g f h   from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a c e g   from row 0, 1
          v5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a c b d   from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // e g f h   from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a c e g   from row 0, 1
          v9 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // b d f h   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a c b d   from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // e g f h   from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a c e g   from row 0, 1
          v13 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // b d f h   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          _mm_store_ps(packed_w + 16, v1);
          _mm_store_ps(packed_w + 20, v5);
          _mm_store_ps(packed_w + 24, v9);
          _mm_store_ps(packed_w + 28, v13);
          packed_w += 32;
        }
        if (k & 1) {
          // Read blocks of 4x1
          // a
          // b
          // c
          // d
          __m128 v0 = _mm_load_ss(w0);
          w0 += 1;
          __m128 v1 = _mm_load_ss(w1);
          w1 += 1;
          __m128 v2 = _mm_load_ss(w2);
          w2 += 1;
          __m128 v3 = _mm_load_ss(w3);
          w3 += 1;
          __m128 v4 = _mm_load_ss(w4);
          w4 += 1;
          __m128 v5 = _mm_load_ss(w5);
          w5 += 1;
          __m128 v6 = _mm_load_ss(w6);
          w6 += 1;
          __m128 v7 = _mm_load_ss(w7);
          w7 += 1;
          __m128 v8 = _mm_load_ss(w8);
          w8 += 1;
          __m128 v9 = _mm_load_ss(w9);
          w9 += 1;
          __m128 v10 = _mm_load_ss(w10);
          w10 += 1;
          __m128 v11 = _mm_load_ss(w11);
          w11 += 1;
          __m128 v12 = _mm_load_ss(w12);
          w12 += 1;
          __m128 v13 = _mm_load_ss(w13);
          w13 += 1;
          __m128 v14 = _mm_load_ss(w14);
          w14 += 1;

          // Transpose 2x2
          const __m128 vtmp0 = _mm_unpacklo_ps(v0, v1);  // a b  from row 0, 1
          const __m128 vtmp1 = _mm_unpacklo_ps(v2, v3);  // c d  from row 2, 3
          // Transpose 4x4
          v0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp0), _mm_castps_pd(vtmp1)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp4 = _mm_unpacklo_ps(v4, v5);  // a b  from row 0, 1
          const __m128 vtmp5 = _mm_unpacklo_ps(v6, v7);  // c d  from row 2, 3
          // Transpose 4x4
          v4 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp4), _mm_castps_pd(vtmp5)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp8 = _mm_unpacklo_ps(v8, v9);  // a b  from row 0, 1
          const __m128 vtmp9 = _mm_unpacklo_ps(v10, v11);  // c d  from row 2, 3
          // Transpose 4x4
          v8 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp8), _mm_castps_pd(vtmp9)));  // a b c d   from row 0, 1
          // Transpose 2x2
          const __m128 vtmp12 = _mm_unpacklo_ps(v12, v13);  // a b  from row 0, 1
          const __m128 vtmp13 = _mm_unpacklo_ps(v14, v14);  // c d  from row 2, 3
          // Transpose 4x4
          v12 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(vtmp12), _mm_castps_pd(vtmp13)));  // a b c d   from row 0, 1

          _mm_store_ps(packed_w, v0);
          _mm_store_ps(packed_w + 4, v4);
          _mm_store_ps(packed_w + 8, v8);
          _mm_store_ps(packed_w + 12, v12);
          packed_w += 16;
        }
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}

void xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_x4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const uint32_t* bias,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 32
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        ((uint32_t*) out)[0] = b[0];
        ((uint32_t*) out)[1] = b[1];
        ((uint32_t*) out)[2] = b[2];
        ((uint32_t*) out)[3] = b[3];
        ((uint32_t*) out)[4] = b[4];
        ((uint32_t*) out)[5] = b[5];
        ((uint32_t*) out)[6] = b[6];
        ((uint32_t*) out)[7] = b[7];
        ((uint32_t*) out)[8] = b[8];
        ((uint32_t*) out)[9] = b[9];
        ((uint32_t*) out)[10] = b[10];
        ((uint32_t*) out)[11] = b[11];
        ((uint32_t*) out)[12] = b[12];
        ((uint32_t*) out)[13] = b[13];
        ((uint32_t*) out)[14] = b[14];
        ((uint32_t*) out)[15] = b[15];
        ((uint32_t*) out)[16] = b[16];
        ((uint32_t*) out)[17] = b[17];
        ((uint32_t*) out)[18] = b[18];
        ((uint32_t*) out)[19] = b[19];
        ((uint32_t*) out)[20] = b[20];
        ((uint32_t*) out)[21] = b[21];
        ((uint32_t*) out)[22] = b[22];
        ((uint32_t*) out)[23] = b[23];
        ((uint32_t*) out)[24] = b[24];
        ((uint32_t*) out)[25] = b[25];
        ((uint32_t*) out)[26] = b[26];
        ((uint32_t*) out)[27] = b[27];
        ((uint32_t*) out)[28] = b[28];
        ((uint32_t*) out)[29] = b[29];
        ((uint32_t*) out)[30] = b[30];
        ((uint32_t*) out)[31] = b[31];
        b += 32;
      } else {
        ((uint32_t*) out)[0] = 0;
        ((uint32_t*) out)[1] = 0;
        ((uint32_t*) out)[2] = 0;
        ((uint32_t*) out)[3] = 0;
        ((uint32_t*) out)[4] = 0;
        ((uint32_t*) out)[5] = 0;
        ((uint32_t*) out)[6] = 0;
        ((uint32_t*) out)[7] = 0;
        ((uint32_t*) out)[8] = 0;
        ((uint32_t*) out)[9] = 0;
        ((uint32_t*) out)[10] = 0;
        ((uint32_t*) out)[11] = 0;
        ((uint32_t*) out)[12] = 0;
        ((uint32_t*) out)[13] = 0;
        ((uint32_t*) out)[14] = 0;
        ((uint32_t*) out)[15] = 0;
        ((uint32_t*) out)[16] = 0;
        ((uint32_t*) out)[17] = 0;
        ((uint32_t*) out)[18] = 0;
        ((uint32_t*) out)[19] = 0;
        ((uint32_t*) out)[20] = 0;
        ((uint32_t*) out)[21] = 0;
        ((uint32_t*) out)[22] = 0;
        ((uint32_t*) out)[23] = 0;
        ((uint32_t*) out)[24] = 0;
        ((uint32_t*) out)[25] = 0;
        ((uint32_t*) out)[26] = 0;
        ((uint32_t*) out)[27] = 0;
        ((uint32_t*) out)[28] = 0;
        ((uint32_t*) out)[29] = 0;
        ((uint32_t*) out)[30] = 0;
        ((uint32_t*) out)[31] = 0;
      }
      out += 32 * sizeof(uint32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;
      const int8_t* w8 = w7 + kc;
      const int8_t* w9 = w8 + kc;
      const int8_t* w10 = w9 + kc;
      const int8_t* w11 = w10 + kc;
      const int8_t* w12 = w11 + kc;
      const int8_t* w13 = w12 + kc;
      const int8_t* w14 = w13 + kc;
      const int8_t* w15 = w14 + kc;
      const int8_t* w16 = w15 + kc;
      const int8_t* w17 = w16 + kc;
      const int8_t* w18 = w17 + kc;
      const int8_t* w19 = w18 + kc;
      const int8_t* w20 = w19 + kc;
      const int8_t* w21 = w20 + kc;
      const int8_t* w22 = w21 + kc;
      const int8_t* w23 = w22 + kc;
      const int8_t* w24 = w23 + kc;
      const int8_t* w25 = w24 + kc;
      const int8_t* w26 = w25 + kc;
      const int8_t* w27 = w26 + kc;
      const int8_t* w28 = w27 + kc;
      const int8_t* w29 = w28 + kc;
      const int8_t* w30 = w29 + kc;
      const int8_t* w31 = w30 + kc;

      // KC main loop multiple of 32x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        const int8_t v01 = w0[1];
        const int8_t v02 = w0[2];
        const int8_t v03 = w0[3];
        w0 += 4;
        const int8_t v10 = w1[0];
        const int8_t v11 = w1[1];
        const int8_t v12 = w1[2];
        const int8_t v13 = w1[3];
        w1 += 4;
        const int8_t v20 = w2[0];
        const int8_t v21 = w2[1];
        const int8_t v22 = w2[2];
        const int8_t v23 = w2[3];
        w2 += 4;
        const int8_t v30 = w3[0];
        const int8_t v31 = w3[1];
        const int8_t v32 = w3[2];
        const int8_t v33 = w3[3];
        w3 += 4;
        const int8_t v40 = w4[0];
        const int8_t v41 = w4[1];
        const int8_t v42 = w4[2];
        const int8_t v43 = w4[3];
        w4 += 4;
        const int8_t v50 = w5[0];
        const int8_t v51 = w5[1];
        const int8_t v52 = w5[2];
        const int8_t v53 = w5[3];
        w5 += 4;
        const int8_t v60 = w6[0];
        const int8_t v61 = w6[1];
        const int8_t v62 = w6[2];
        const int8_t v63 = w6[3];
        w6 += 4;
        const int8_t v70 = w7[0];
        const int8_t v71 = w7[1];
        const int8_t v72 = w7[2];
        const int8_t v73 = w7[3];
        w7 += 4;
        const int8_t v80 = w8[0];
        const int8_t v81 = w8[1];
        const int8_t v82 = w8[2];
        const int8_t v83 = w8[3];
        w8 += 4;
        const int8_t v90 = w9[0];
        const int8_t v91 = w9[1];
        const int8_t v92 = w9[2];
        const int8_t v93 = w9[3];
        w9 += 4;
        const int8_t v100 = w10[0];
        const int8_t v101 = w10[1];
        const int8_t v102 = w10[2];
        const int8_t v103 = w10[3];
        w10 += 4;
        const int8_t v110 = w11[0];
        const int8_t v111 = w11[1];
        const int8_t v112 = w11[2];
        const int8_t v113 = w11[3];
        w11 += 4;
        const int8_t v120 = w12[0];
        const int8_t v121 = w12[1];
        const int8_t v122 = w12[2];
        const int8_t v123 = w12[3];
        w12 += 4;
        const int8_t v130 = w13[0];
        const int8_t v131 = w13[1];
        const int8_t v132 = w13[2];
        const int8_t v133 = w13[3];
        w13 += 4;
        const int8_t v140 = w14[0];
        const int8_t v141 = w14[1];
        const int8_t v142 = w14[2];
        const int8_t v143 = w14[3];
        w14 += 4;
        const int8_t v150 = w15[0];
        const int8_t v151 = w15[1];
        const int8_t v152 = w15[2];
        const int8_t v153 = w15[3];
        w15 += 4;
        const int8_t v160 = w16[0];
        const int8_t v161 = w16[1];
        const int8_t v162 = w16[2];
        const int8_t v163 = w16[3];
        w16 += 4;
        const int8_t v170 = w17[0];
        const int8_t v171 = w17[1];
        const int8_t v172 = w17[2];
        const int8_t v173 = w17[3];
        w17 += 4;
        const int8_t v180 = w18[0];
        const int8_t v181 = w18[1];
        const int8_t v182 = w18[2];
        const int8_t v183 = w18[3];
        w18 += 4;
        const int8_t v190 = w19[0];
        const int8_t v191 = w19[1];
        const int8_t v192 = w19[2];
        const int8_t v193 = w19[3];
        w19 += 4;
        const int8_t v200 = w20[0];
        const int8_t v201 = w20[1];
        const int8_t v202 = w20[2];
        const int8_t v203 = w20[3];
        w20 += 4;
        const int8_t v210 = w21[0];
        const int8_t v211 = w21[1];
        const int8_t v212 = w21[2];
        const int8_t v213 = w21[3];
        w21 += 4;
        const int8_t v220 = w22[0];
        const int8_t v221 = w22[1];
        const int8_t v222 = w22[2];
        const int8_t v223 = w22[3];
        w22 += 4;
        const int8_t v230 = w23[0];
        const int8_t v231 = w23[1];
        const int8_t v232 = w23[2];
        const int8_t v233 = w23[3];
        w23 += 4;
        const int8_t v240 = w24[0];
        const int8_t v241 = w24[1];
        const int8_t v242 = w24[2];
        const int8_t v243 = w24[3];
        w24 += 4;
        const int8_t v250 = w25[0];
        const int8_t v251 = w25[1];
        const int8_t v252 = w25[2];
        const int8_t v253 = w25[3];
        w25 += 4;
        const int8_t v260 = w26[0];
        const int8_t v261 = w26[1];
        const int8_t v262 = w26[2];
        const int8_t v263 = w26[3];
        w26 += 4;
        const int8_t v270 = w27[0];
        const int8_t v271 = w27[1];
        const int8_t v272 = w27[2];
        const int8_t v273 = w27[3];
        w27 += 4;
        const int8_t v280 = w28[0];
        const int8_t v281 = w28[1];
        const int8_t v282 = w28[2];
        const int8_t v283 = w28[3];
        w28 += 4;
        const int8_t v290 = w29[0];
        const int8_t v291 = w29[1];
        const int8_t v292 = w29[2];
        const int8_t v293 = w29[3];
        w29 += 4;
        const int8_t v300 = w30[0];
        const int8_t v301 = w30[1];
        const int8_t v302 = w30[2];
        const int8_t v303 = w30[3];
        w30 += 4;
        const int8_t v310 = w31[0];
        const int8_t v311 = w31[1];
        const int8_t v312 = w31[2];
        const int8_t v313 = w31[3];
        w31 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[7] = v70;
        out[8] = v80;
        out[9] = v90;
        out[10] = v100;
        out[11] = v110;
        out[12] = v120;
        out[13] = v130;
        out[14] = v140;
        out[15] = v150;
        out[16] = v160;
        out[17] = v170;
        out[18] = v180;
        out[19] = v190;
        out[20] = v200;
        out[21] = v210;
        out[22] = v220;
        out[23] = v230;
        out[24] = v240;
        out[25] = v250;
        out[26] = v260;
        out[27] = v270;
        out[28] = v280;
        out[29] = v290;
        out[30] = v300;
        out[31] = v310;
        out[32] = v01;
        out[33] = v11;
        out[34] = v21;
        out[35] = v31;
        out[36] = v41;
        out[37] = v51;
        out[38] = v61;
        out[39] = v71;
        out[40] = v81;
        out[41] = v91;
        out[42] = v101;
        out[43] = v111;
        out[44] = v121;
        out[45] = v131;
        out[46] = v141;
        out[47] = v151;
        out[48] = v161;
        out[49] = v171;
        out[50] = v181;
        out[51] = v191;
        out[52] = v201;
        out[53] = v211;
        out[54] = v221;
        out[55] = v231;
        out[56] = v241;
        out[57] = v251;
        out[58] = v261;
        out[59] = v271;
        out[60] = v281;
        out[61] = v291;
        out[62] = v301;
        out[63] = v311;
        out[64] = v02;
        out[65] = v12;
        out[66] = v22;
        out[67] = v32;
        out[68] = v42;
        out[69] = v52;
        out[70] = v62;
        out[71] = v72;
        out[72] = v82;
        out[73] = v92;
        out[74] = v102;
        out[75] = v112;
        out[76] = v122;
        out[77] = v132;
        out[78] = v142;
        out[79] = v152;
        out[80] = v162;
        out[81] = v172;
        out[82] = v182;
        out[83] = v192;
        out[84] = v202;
        out[85] = v212;
        out[86] = v222;
        out[87] = v232;
        out[88] = v242;
        out[89] = v252;
        out[90] = v262;
        out[91] = v272;
        out[92] = v282;
        out[93] = v292;
        out[94] = v302;
        out[95] = v312;
        out[96] = v03;
        out[97] = v13;
        out[98] = v23;
        out[99] = v33;
        out[100] = v43;
        out[101] = v53;
        out[102] = v63;
        out[103] = v73;
        out[104] = v83;
        out[105] = v93;
        out[106] = v103;
        out[107] = v113;
        out[108] = v123;
        out[109] = v133;
        out[110] = v143;
        out[111] = v153;
        out[112] = v163;
        out[113] = v173;
        out[114] = v183;
        out[115] = v193;
        out[116] = v203;
        out[117] = v213;
        out[118] = v223;
        out[119] = v233;
        out[120] = v243;
        out[121] = v253;
        out[122] = v263;
        out[123] = v273;
        out[124] = v283;
        out[125] = v293;
        out[126] = v303;
        out[127] = v313;
        out += 128;
      }

      // KC remainder
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        out[0] = v0;
        const int8_t v1 = *w1++;
        out[1] = v1;
        const int8_t v2 = *w2++;
        out[2] = v2;
        const int8_t v3 = *w3++;
        out[3] = v3;
        const int8_t v4 = *w4++;
        out[4] = v4;
        const int8_t v5 = *w5++;
        out[5] = v5;
        const int8_t v6 = *w6++;
        out[6] = v6;
        const int8_t v7 = *w7++;
        out[7] = v7;
        const int8_t v8 = *w8++;
        out[8] = v8;
        const int8_t v9 = *w9++;
        out[9] = v9;
        const int8_t v10 = *w10++;
        out[10] = v10;
        const int8_t v11 = *w11++;
        out[11] = v11;
        const int8_t v12 = *w12++;
        out[12] = v12;
        const int8_t v13 = *w13++;
        out[13] = v13;
        const int8_t v14 = *w14++;
        out[14] = v14;
        const int8_t v15 = *w15++;
        out[15] = v15;
        const int8_t v16 = *w16++;
        out[16] = v16;
        const int8_t v17 = *w17++;
        out[17] = v17;
        const int8_t v18 = *w18++;
        out[18] = v18;
        const int8_t v19 = *w19++;
        out[19] = v19;
        const int8_t v20 = *w20++;
        out[20] = v20;
        const int8_t v21 = *w21++;
        out[21] = v21;
        const int8_t v22 = *w22++;
        out[22] = v22;
        const int8_t v23 = *w23++;
        out[23] = v23;
        const int8_t v24 = *w24++;
        out[24] = v24;
        const int8_t v25 = *w25++;
        out[25] = v25;
        const int8_t v26 = *w26++;
        out[26] = v26;
        const int8_t v27 = *w27++;
        out[27] = v27;
        const int8_t v28 = *w28++;
        out[28] = v28;
        const int8_t v29 = *w29++;
        out[29] = v29;
        const int8_t v30 = *w30++;
        out[30] = v30;
        const int8_t v31 = *w31++;
        out[31] = v31;
        out += 32;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t);
        } while (--nb != 0);
      }
      out += (32 - n) * sizeof(uint32_t);

      // NR remainder has less than 32 rows so last row is not loaded
      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int8_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int8_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int8_t* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int8_t* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int8_t* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int8_t* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int8_t* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int8_t* w15 = w14 + kc;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const int8_t* w16 = w15 + kc;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int8_t* w17 = w16 + kc;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int8_t* w18 = w17 + kc;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int8_t* w19 = w18 + kc;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int8_t* w20 = w19 + kc;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int8_t* w21 = w20 + kc;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int8_t* w22 = w21 + kc;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int8_t* w23 = w22 + kc;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int8_t* w24 = w23 + kc;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int8_t* w25 = w24 + kc;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int8_t* w26 = w25 + kc;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int8_t* w27 = w26 + kc;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int8_t* w28 = w27 + kc;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int8_t* w29 = w28 + kc;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int8_t* w30 = w29 + kc;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 32x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v00 = w0[0];
        const int8_t v01 = w0[1];
        const int8_t v02 = w0[2];
        const int8_t v03 = w0[3];
        w0 += 4;
        const int8_t v10 = w1[0];
        const int8_t v11 = w1[1];
        const int8_t v12 = w1[2];
        const int8_t v13 = w1[3];
        w1 += 4;
        const int8_t v20 = w2[0];
        const int8_t v21 = w2[1];
        const int8_t v22 = w2[2];
        const int8_t v23 = w2[3];
        w2 += 4;
        const int8_t v30 = w3[0];
        const int8_t v31 = w3[1];
        const int8_t v32 = w3[2];
        const int8_t v33 = w3[3];
        w3 += 4;
        const int8_t v40 = w4[0];
        const int8_t v41 = w4[1];
        const int8_t v42 = w4[2];
        const int8_t v43 = w4[3];
        w4 += 4;
        const int8_t v50 = w5[0];
        const int8_t v51 = w5[1];
        const int8_t v52 = w5[2];
        const int8_t v53 = w5[3];
        w5 += 4;
        const int8_t v60 = w6[0];
        const int8_t v61 = w6[1];
        const int8_t v62 = w6[2];
        const int8_t v63 = w6[3];
        w6 += 4;
        const int8_t v70 = w7[0];
        const int8_t v71 = w7[1];
        const int8_t v72 = w7[2];
        const int8_t v73 = w7[3];
        w7 += 4;
        const int8_t v80 = w8[0];
        const int8_t v81 = w8[1];
        const int8_t v82 = w8[2];
        const int8_t v83 = w8[3];
        w8 += 4;
        const int8_t v90 = w9[0];
        const int8_t v91 = w9[1];
        const int8_t v92 = w9[2];
        const int8_t v93 = w9[3];
        w9 += 4;
        const int8_t v100 = w10[0];
        const int8_t v101 = w10[1];
        const int8_t v102 = w10[2];
        const int8_t v103 = w10[3];
        w10 += 4;
        const int8_t v110 = w11[0];
        const int8_t v111 = w11[1];
        const int8_t v112 = w11[2];
        const int8_t v113 = w11[3];
        w11 += 4;
        const int8_t v120 = w12[0];
        const int8_t v121 = w12[1];
        const int8_t v122 = w12[2];
        const int8_t v123 = w12[3];
        w12 += 4;
        const int8_t v130 = w13[0];
        const int8_t v131 = w13[1];
        const int8_t v132 = w13[2];
        const int8_t v133 = w13[3];
        w13 += 4;
        const int8_t v140 = w14[0];
        const int8_t v141 = w14[1];
        const int8_t v142 = w14[2];
        const int8_t v143 = w14[3];
        w14 += 4;
        const int8_t v150 = w15[0];
        const int8_t v151 = w15[1];
        const int8_t v152 = w15[2];
        const int8_t v153 = w15[3];
        w15 += 4;
        const int8_t v160 = w16[0];
        const int8_t v161 = w16[1];
        const int8_t v162 = w16[2];
        const int8_t v163 = w16[3];
        w16 += 4;
        const int8_t v170 = w17[0];
        const int8_t v171 = w17[1];
        const int8_t v172 = w17[2];
        const int8_t v173 = w17[3];
        w17 += 4;
        const int8_t v180 = w18[0];
        const int8_t v181 = w18[1];
        const int8_t v182 = w18[2];
        const int8_t v183 = w18[3];
        w18 += 4;
        const int8_t v190 = w19[0];
        const int8_t v191 = w19[1];
        const int8_t v192 = w19[2];
        const int8_t v193 = w19[3];
        w19 += 4;
        const int8_t v200 = w20[0];
        const int8_t v201 = w20[1];
        const int8_t v202 = w20[2];
        const int8_t v203 = w20[3];
        w20 += 4;
        const int8_t v210 = w21[0];
        const int8_t v211 = w21[1];
        const int8_t v212 = w21[2];
        const int8_t v213 = w21[3];
        w21 += 4;
        const int8_t v220 = w22[0];
        const int8_t v221 = w22[1];
        const int8_t v222 = w22[2];
        const int8_t v223 = w22[3];
        w22 += 4;
        const int8_t v230 = w23[0];
        const int8_t v231 = w23[1];
        const int8_t v232 = w23[2];
        const int8_t v233 = w23[3];
        w23 += 4;
        const int8_t v240 = w24[0];
        const int8_t v241 = w24[1];
        const int8_t v242 = w24[2];
        const int8_t v243 = w24[3];
        w24 += 4;
        const int8_t v250 = w25[0];
        const int8_t v251 = w25[1];
        const int8_t v252 = w25[2];
        const int8_t v253 = w25[3];
        w25 += 4;
        const int8_t v260 = w26[0];
        const int8_t v261 = w26[1];
        const int8_t v262 = w26[2];
        const int8_t v263 = w26[3];
        w26 += 4;
        const int8_t v270 = w27[0];
        const int8_t v271 = w27[1];
        const int8_t v272 = w27[2];
        const int8_t v273 = w27[3];
        w27 += 4;
        const int8_t v280 = w28[0];
        const int8_t v281 = w28[1];
        const int8_t v282 = w28[2];
        const int8_t v283 = w28[3];
        w28 += 4;
        const int8_t v290 = w29[0];
        const int8_t v291 = w29[1];
        const int8_t v292 = w29[2];
        const int8_t v293 = w29[3];
        w29 += 4;
        const int8_t v300 = w30[0];
        const int8_t v301 = w30[1];
        const int8_t v302 = w30[2];
        const int8_t v303 = w30[3];
        w30 += 4;
        out[0] = v00;
        out[1] = v10;
        out[2] = v20;
        out[3] = v30;
        out[4] = v40;
        out[5] = v50;
        out[6] = v60;
        out[7] = v70;
        out[8] = v80;
        out[9] = v90;
        out[10] = v100;
        out[11] = v110;
        out[12] = v120;
        out[13] = v130;
        out[14] = v140;
        out[15] = v150;
        out[16] = v160;
        out[17] = v170;
        out[18] = v180;
        out[19] = v190;
        out[20] = v200;
        out[21] = v210;
        out[22] = v220;
        out[23] = v230;
        out[24] = v240;
        out[25] = v250;
        out[26] = v260;
        out[27] = v270;
        out[28] = v280;
        out[29] = v290;
        out[30] = v300;
        out[32] = v01;
        out[33] = v11;
        out[34] = v21;
        out[35] = v31;
        out[36] = v41;
        out[37] = v51;
        out[38] = v61;
        out[39] = v71;
        out[40] = v81;
        out[41] = v91;
        out[42] = v101;
        out[43] = v111;
        out[44] = v121;
        out[45] = v131;
        out[46] = v141;
        out[47] = v151;
        out[48] = v161;
        out[49] = v171;
        out[50] = v181;
        out[51] = v191;
        out[52] = v201;
        out[53] = v211;
        out[54] = v221;
        out[55] = v231;
        out[56] = v241;
        out[57] = v251;
        out[58] = v261;
        out[59] = v271;
        out[60] = v281;
        out[61] = v291;
        out[62] = v301;
        out[64] = v02;
        out[65] = v12;
        out[66] = v22;
        out[67] = v32;
        out[68] = v42;
        out[69] = v52;
        out[70] = v62;
        out[71] = v72;
        out[72] = v82;
        out[73] = v92;
        out[74] = v102;
        out[75] = v112;
        out[76] = v122;
        out[77] = v132;
        out[78] = v142;
        out[79] = v152;
        out[80] = v162;
        out[81] = v172;
        out[82] = v182;
        out[83] = v192;
        out[84] = v202;
        out[85] = v212;
        out[86] = v222;
        out[87] = v232;
        out[88] = v242;
        out[89] = v252;
        out[90] = v262;
        out[91] = v272;
        out[92] = v282;
        out[93] = v292;
        out[94] = v302;
        out[96] = v03;
        out[97] = v13;
        out[98] = v23;
        out[99] = v33;
        out[100] = v43;
        out[101] = v53;
        out[102] = v63;
        out[103] = v73;
        out[104] = v83;
        out[105] = v93;
        out[106] = v103;
        out[107] = v113;
        out[108] = v123;
        out[109] = v133;
        out[110] = v143;
        out[111] = v153;
        out[112] = v163;
        out[113] = v173;
        out[114] = v183;
        out[115] = v193;
        out[116] = v203;
        out[117] = v213;
        out[118] = v223;
        out[119] = v233;
        out[120] = v243;
        out[121] = v253;
        out[122] = v263;
        out[123] = v273;
        out[124] = v283;
        out[125] = v293;
        out[126] = v303;
        out += 128;
      }

      // KC remainder of 1..3
      for (; k != 0; --k) {
        const int8_t v0 = *w0++;
        out[0] = v0;
        const int8_t v1 = *w1++;
        out[1] = v1;
        const int8_t v2 = *w2++;
        out[2] = v2;
        const int8_t v3 = *w3++;
        out[3] = v3;
        const int8_t v4 = *w4++;
        out[4] = v4;
        const int8_t v5 = *w5++;
        out[5] = v5;
        const int8_t v6 = *w6++;
        out[6] = v6;
        const int8_t v7 = *w7++;
        out[7] = v7;
        const int8_t v8 = *w8++;
        out[8] = v8;
        const int8_t v9 = *w9++;
        out[9] = v9;
        const int8_t v10 = *w10++;
        out[10] = v10;
        const int8_t v11 = *w11++;
        out[11] = v11;
        const int8_t v12 = *w12++;
        out[12] = v12;
        const int8_t v13 = *w13++;
        out[13] = v13;
        const int8_t v14 = *w14++;
        out[14] = v14;
        const int8_t v15 = *w15++;
        out[15] = v15;
        const int8_t v16 = *w16++;
        out[16] = v16;
        const int8_t v17 = *w17++;
        out[17] = v17;
        const int8_t v18 = *w18++;
        out[18] = v18;
        const int8_t v19 = *w19++;
        out[19] = v19;
        const int8_t v20 = *w20++;
        out[20] = v20;
        const int8_t v21 = *w21++;
        out[21] = v21;
        const int8_t v22 = *w22++;
        out[22] = v22;
        const int8_t v23 = *w23++;
        out[23] = v23;
        const int8_t v24 = *w24++;
        out[24] = v24;
        const int8_t v25 = *w25++;
        out[25] = v25;
        const int8_t v26 = *w26++;
        out[26] = v26;
        const int8_t v27 = *w27++;
        out[27] = v27;
        const int8_t v28 = *w28++;
        out[28] = v28;
        const int8_t v29 = *w29++;
        out[29] = v29;
        const int8_t v30 = *w30++;
        out[30] = v30;
        out += 32;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}

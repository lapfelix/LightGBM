/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief Metal histogram-construction kernels for MetalTreeLearner.
 *
 * This file is included in a C++11 source file as a string literal (same
 * trick as the OpenCL kernels in src/treelearner/ocl/*.cl), for runtime
 * compilation via -[MTLDevice newLibraryWithSource:options:error:].
 * The host must compile starting from the __METAL_KERNELS_BEGIN__ marker
 * below: __METAL_VERSION__ is not predefined for source-string compilation,
 * so compiling from the top would swallow the kernels into the raw string.
 */
#ifndef __METAL_VERSION__
// If we are including this file in C++,
// the entire source file following (except the last #endif) will become
// a raw string literal. The extra ")" is just for matching parentheses
// to make the editor happy. The extra ")" and extra endif will be skipped.
// The host locates the marker below with strstr, so no manual skip count.
R""()
#endif
// __METAL_KERNELS_BEGIN__
#include <metal_stdlib>
using namespace metal;

#ifndef _METAL_HISTOGRAMS_KERNEL_
#define _METAL_HISTOGRAMS_KERNEL_

// Compile-time specialization (via MTLCompileOptions.preprocessorMacros):
//   NB            - bins per dense feature group (64 or 256)
//   USE_INDICES   - 1: rows come from an index list; 0: identity rows
//   CONST_HESSIAN - 1: single hessian value for all rows; 0: per-row hessians
#ifndef NB
#define NB 64
#endif
#ifndef USE_INDICES
#define USE_INDICES 1
#endif
#ifndef CONST_HESSIAN
#define CONST_HESSIAN 0
#endif

// Dense histogram construction: one thread per leaf row, looping over all
// dense feature groups, accumulating via device float atomics into a
// [num_groups x NB] gradient/hessian histogram pair.
//
// Bins are row-major bytes: bins[row * row_stride + group]. Feature groups
// disabled by column sampling are skipped via the mask buffer.
kernel void metal_histogram(
    device const uchar* bins       [[buffer(0)]],
    device const float* grad       [[buffer(1)]],
    device const float* hess       [[buffer(2)]],
    device const int* indices      [[buffer(3)]],
    device const char* masks       [[buffer(4)]],
    device atomic_float* out_g     [[buffer(5)]],
    device atomic_float* out_h     [[buffer(6)]],
    constant uint& row_stride      [[buffer(7)]],
    constant uint& num_groups      [[buffer(8)]],
    constant uint& num_rows        [[buffer(9)]],
    constant float& const_hess_val [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= num_rows) {
    return;
  }
#if USE_INDICES
  uint r = (uint)indices[tid];
#else
  uint r = tid;
#endif
  float g = grad[tid];
#if CONST_HESSIAN
  float h = const_hess_val;
#else
  float h = hess[tid];
#endif
  size_t base = (size_t)r * row_stride;
  for (uint grp = 0; grp < num_groups; ++grp) {
    if (!masks[grp]) {
      continue;
    }
    uint b = bins[base + grp];
    atomic_fetch_add_explicit(out_g + (size_t)grp * NB + b, g, memory_order_relaxed);
    atomic_fetch_add_explicit(out_h + (size_t)grp * NB + b, h, memory_order_relaxed);
  }
}

// NOTE: no guard-closing #endif here. As in ocl/histogram64.cl, the last
// #endif below dual-purposes: it closes _METAL_HISTOGRAMS_KERNEL_ when
// compiled as Metal, and __METAL_VERSION__ when included as a C++ string.
// The following line ends the string literal, adds an extra #endif at the end
// )"" "\n#endif"
#endif

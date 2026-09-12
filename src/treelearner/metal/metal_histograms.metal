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

// ---- Stable leaf partition (GPU DataPartition::Split) ----
//
// DataPartition::Split runs each leaf block through a serial predicate and
// concatenates per-block survivors, which is exactly a global stable
// partition of the leaf's index range. These three kernels reproduce that
// result bit-for-bit:
//
//   metal_partition_count    per-block count of left-going rows
//   metal_partition_scan     exclusive prefix over block counts (+ total)
//   metal_partition_scatter  stable scatter into the output index array
//
// The predicate mirrors DenseBin::SplitInner / SplitCategoricalInner exactly
// (including uint8 wraparound and the single-bin branch), reading the same
// packed row-major bytes as metal_histogram. Multival and sparse groups are
// never dispatched here; the host falls back to the CPU path for those.
//
// Dispatch contract: threadgroup size is exactly PART_TG_SIZE threads, each
// thread owns PART_ROWS_PER_THREAD consecutive rows, so one threadgroup owns
// PART_BLOCK_ROWS rows. The host sizes block_counts/block_offsets as
// ceil(num_rows / PART_BLOCK_ROWS).
#define PART_TG_SIZE 256
#define PART_ROWS_PER_THREAD 8
#define PART_BLOCK_ROWS (PART_TG_SIZE * PART_ROWS_PER_THREAD)

// Flags for MetalPartitionParams.flags.
#define PART_MISS_IS_ZERO  (1u << 0)
#define PART_MISS_IS_NA    (1u << 1)
#define PART_MFB_IS_ZERO   (1u << 2)
#define PART_MFB_IS_NA     (1u << 3)
#define PART_USE_MIN_BIN   (1u << 4)
#define PART_CATEGORICAL   (1u << 5)
#define PART_DEFAULT_LEFT  (1u << 6)

// Must match MetalHistogramContext::PartitionParams in metal_context.h.
struct MetalPartitionParams {
  uint group_col;      // column d in the packed bins
  uint row_stride;     // packed row width (num dense groups)
  uint min_bin;
  uint max_bin;
  uint default_bin;
  uint most_freq_bin;
  uint threshold;      // numeric threshold bin (unused for categorical)
  uint flags;
  uint num_cat_words;  // bitset length for categorical splits
};

inline bool part_find_in_bitset(device const uint* bits, uint n, uint pos) {
  uint i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  return ((bits[i1] >> (pos % 32)) & 1u) != 0;
}

// Returns true when the row goes to the left child. Byte-for-byte port of
// DenseBin<uint8_t>::SplitInner / SplitCategoricalInner.
inline bool part_goes_left(uchar bin, constant MetalPartitionParams& p,
                           device const uint* cat_bits) {
  const uint flags = p.flags;
  if ((flags & PART_CATEGORICAL) == 0) {
    const bool miss_is_zero = (flags & PART_MISS_IS_ZERO) != 0;
    const bool miss_is_na = (flags & PART_MISS_IS_NA) != 0;
    const bool mfb_is_zero = (flags & PART_MFB_IS_ZERO) != 0;
    const bool mfb_is_na = (flags & PART_MFB_IS_NA) != 0;
    const bool use_min_bin = (flags & PART_USE_MIN_BIN) != 0;
    const bool default_left = (flags & PART_DEFAULT_LEFT) != 0;
    uchar th = (uchar)(p.threshold + p.min_bin);
    uchar t_zero_bin = (uchar)(p.min_bin + p.default_bin);
    if (p.most_freq_bin == 0) {
      th--;
      t_zero_bin--;
    }
    const uchar minb = (uchar)p.min_bin;
    const uchar maxb = (uchar)p.max_bin;
    const bool default_goes_left = (p.most_freq_bin <= p.threshold);
    bool missing_goes_left = false;
    if ((miss_is_zero || miss_is_na) && default_left) {
      missing_goes_left = true;
    }
    if (p.min_bin < p.max_bin) {
      if ((miss_is_zero && !mfb_is_zero && bin == t_zero_bin) ||
          (miss_is_na && !mfb_is_na && bin == maxb)) {
        return missing_goes_left;
      } else if ((use_min_bin && (bin < minb || bin > maxb)) ||
                 (!use_min_bin && bin == 0)) {
        if ((miss_is_na && mfb_is_na) || (miss_is_zero && mfb_is_zero)) {
          return missing_goes_left;
        }
        return default_goes_left;
      } else {
        return !(bin > th);
      }
    } else {
      const bool max_goes_left = (maxb <= th);
      if (miss_is_zero && !mfb_is_zero && bin == t_zero_bin) {
        return missing_goes_left;
      } else if (bin != maxb) {
        if ((miss_is_na && mfb_is_na) || (miss_is_zero && mfb_is_zero)) {
          return missing_goes_left;
        }
        return default_goes_left;
      } else {
        if (miss_is_na && !mfb_is_na) {
          return missing_goes_left;
        }
        return max_goes_left;
      }
    }
  } else {
    const bool use_min_bin = (flags & PART_USE_MIN_BIN) != 0;
    bool default_goes_left = false;
    if (p.most_freq_bin > 0 &&
        part_find_in_bitset(cat_bits, p.num_cat_words, p.most_freq_bin)) {
      default_goes_left = true;
    }
    const uint offset = (p.most_freq_bin == 0) ? 1 : 0;
    const uint b = (uint)bin;
    if ((use_min_bin && (b < p.min_bin || b > p.max_bin)) ||
        (!use_min_bin && b == 0)) {
      return default_goes_left;
    }
    return part_find_in_bitset(cat_bits, p.num_cat_words,
                               b - p.min_bin + offset);
  }
}

kernel void metal_partition_count(
    device const uchar* bins      [[buffer(0)]],
    device const int* idx_in      [[buffer(1)]],
    device uint* block_counts     [[buffer(2)]],
    device const uint* cat_bits   [[buffer(3)]],
    constant MetalPartitionParams& p [[buffer(4)]],
    constant uint& num_rows       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tls [[thread_position_in_threadgroup]]) {
  const uint block_start = tgid * (uint)PART_BLOCK_ROWS;
  uint local_left = 0;
  for (uint k = 0; k < (uint)PART_ROWS_PER_THREAD; ++k) {
    const uint pos = block_start + tls * (uint)PART_ROWS_PER_THREAD + k;
    if (pos >= num_rows) {
      break;
    }
    const uint r = (uint)idx_in[pos];
    const uchar b = bins[(size_t)r * p.row_stride + p.group_col];
    if (part_goes_left(b, p, cat_bits)) {
      ++local_left;
    }
  }
  threadgroup uint red[PART_TG_SIZE];
  red[tls] = local_left;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tls == 0) {
    uint sum = 0;
    for (uint i = 0; i < (uint)PART_TG_SIZE; ++i) {
      sum += red[i];
    }
    block_counts[tgid] = sum;
  }
}

kernel void metal_partition_scan(
    device const uint* block_counts [[buffer(0)]],
    device uint* block_offsets      [[buffer(1)]],
    device uint* result             [[buffer(2)]],
    constant uint& num_blocks       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }
  uint sum = 0;
  for (uint b = 0; b < num_blocks; ++b) {
    block_offsets[b] = sum;
    sum += block_counts[b];
  }
  result[0] = sum;
}

kernel void metal_partition_scatter(
    device const uchar* bins      [[buffer(0)]],
    device const int* idx_in      [[buffer(1)]],
    device int* idx_out           [[buffer(2)]],
    device const uint* block_offsets [[buffer(3)]],
    device const uint* block_counts  [[buffer(4)]],
    device const uint* result        [[buffer(5)]],
    device const uint* cat_bits      [[buffer(6)]],
    constant MetalPartitionParams& p [[buffer(7)]],
    constant uint& num_rows          [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tls [[thread_position_in_threadgroup]]) {
  const uint block_start = tgid * (uint)PART_BLOCK_ROWS;
  bool side[PART_ROWS_PER_THREAD];
  uint valid = 0;
  uint local_left = 0;
  for (uint k = 0; k < (uint)PART_ROWS_PER_THREAD; ++k) {
    const uint pos = block_start + tls * (uint)PART_ROWS_PER_THREAD + k;
    if (pos >= num_rows) {
      side[k] = false;
      continue;
    }
    ++valid;
    const uint r = (uint)idx_in[pos];
    const uchar b = bins[(size_t)r * p.row_stride + p.group_col];
    const bool left = part_goes_left(b, p, cat_bits);
    side[k] = left;
    if (left) {
      ++local_left;
    }
  }
  threadgroup uint tg_left[PART_TG_SIZE];
  threadgroup uint tg_valid[PART_TG_SIZE];
  tg_left[tls] = local_left;
  tg_valid[tls] = valid;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tls == 0) {
    uint sum_left = 0;
    uint sum_valid = 0;
    for (uint i = 0; i < (uint)PART_TG_SIZE; ++i) {
      const uint vl = tg_left[i];
      const uint vv = tg_valid[i];
      tg_left[i] = sum_left;
      tg_valid[i] = sum_valid;
      sum_left += vl;
      sum_valid += vv;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total_left = result[0];
  const uint lte_base = block_offsets[tgid];
  // gt rows of this block start after all left rows of all blocks, ordered
  // by block: total_left + (rows before this block that went right).
  const uint gt_base = total_left + (block_start - lte_base);
  uint lj = tg_left[tls];
  uint gj = tg_valid[tls] - tg_left[tls];
  for (uint k = 0; k < (uint)PART_ROWS_PER_THREAD; ++k) {
    const uint pos = block_start + tls * (uint)PART_ROWS_PER_THREAD + k;
    if (pos >= num_rows) {
      break;
    }
    const int v = idx_in[pos];
    if (side[k]) {
      idx_out[lte_base + lj] = v;
      ++lj;
    } else {
      idx_out[gt_base + gj] = v;
      ++gj;
    }
  }
  (void)block_counts;
}

// NOTE: no guard-closing #endif here. As in ocl/histogram64.cl, the last
// #endif below dual-purposes: it closes _METAL_HISTOGRAMS_KERNEL_ when
// compiled as Metal, and __METAL_VERSION__ when included as a C++ string.
// The following line ends the string literal, adds an extra #endif at the end
// )"" "\n#endif"
#endif

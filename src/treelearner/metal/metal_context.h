/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_

#include <cstddef>
#include <cstdint>

namespace LightGBM {

//! \brief C++-only interface to the Metal histogram backend.
//!
//! Implemented in metal_context.mm (Objective-C++). All Metal objects are
//! shared-memory (zero-copy on Apple Silicon unified memory): CPU writes via
//! BufferContents() are visible to subsequently committed GPU work, and
//! GPU writes are visible to the CPU after Wait() returns. No Objective-C
//! types appear here, so this header is safe to include from pure C++.
class MetalHistogramContext {
 public:
  MetalHistogramContext();
  ~MetalHistogramContext();

  MetalHistogramContext(const MetalHistogramContext&) = delete;
  MetalHistogramContext& operator=(const MetalHistogramContext&) = delete;

  //! \brief Name of the selected Metal device (owned by this object).
  const char* device_name() const;

  //! \brief Allocate a CPU/GPU shared buffer; FreeBuffer() releases it.
  void* AllocShared(size_t bytes);
  void FreeBuffer(void* buffer);
  //! \brief CPU-visible pointer to a shared buffer's contents.
  void* BufferContents(void* buffer);

  //! \brief Kernel specialization selected at pipeline-creation time.
  struct LaunchParams {
    int num_bins = 64;            // bins per dense feature group (64/256)
    bool use_indices = true;      // false => identity rows (root w/o bagging)
    bool const_hessian = false;   // true => single hessian value for all rows
    float const_hessian_value = 0.0f;
  };

  //! \brief Dispatch histogram construction; returns without waiting.
  //! The caller must have zeroed out_g/out_h (atomics accumulate).
  void LaunchHistogramAsync(const void* bins, const void* grads, const void* hess,
                            const void* indices, const void* masks,
                            void* out_g, void* out_h,
                            int row_stride, int num_groups, int num_rows,
                            const LaunchParams& params);
  //! \brief Block until the outstanding dispatch (if any) completes.
  void Wait();
  //! \brief Log cumulative dispatch statistics (count, rows, GPU-wait time).
  void LogStats() const;

  //! \brief Flags for PartitionParams::flags (mirror PART_* in the .metal file).
  enum PartitionFlags : uint32_t {
    kPartMissIsZero = 1u << 0,
    kPartMissIsNa = 1u << 1,
    kPartMfbIsZero = 1u << 2,
    kPartMfbIsNa = 1u << 3,
    kPartUseMinBin = 1u << 4,
    kPartCategorical = 1u << 5,
    kPartDefaultLeft = 1u << 6,
  };

  //! \brief Parameters for a stable leaf partition dispatch.
  //! Must match MetalPartitionParams in metal_histograms.metal field for field.
  struct PartitionParams {
    uint32_t group_col = 0;
    uint32_t row_stride = 0;
    uint32_t min_bin = 0;
    uint32_t max_bin = 0;
    uint32_t default_bin = 0;
    uint32_t most_freq_bin = 0;
    uint32_t threshold = 0;
    uint32_t flags = 0;
    uint32_t num_cat_words = 0;
  };

  //! \brief Leaf rows owned by one threadgroup in the partition kernels
  //! (PART_TG_SIZE x PART_ROWS_PER_THREAD in the .metal file).
  static const int kPartitionBlockRows = 256 * 8;

  //! \brief Run count+scan+scatter partition and block until it completes.
  //! idx_in holds num_rows row ids; idx_out receives the stable partition
  //! (left rows first); result[0] receives the left count. block_counts and
  //! block_offsets hold num_blocks uint32 entries. cat_bits holds the
  //! categorical bitset (num_cat_words entries; ignored unless the
  //! kPartCategorical flag is set). No histogram dispatch may be in flight.
  void LaunchPartitionSync(const void* bins, const void* idx_in, void* idx_out,
                           void* block_counts, void* block_offsets, void* result,
                           const void* cat_bits, int num_rows, int num_blocks,
                           const PartitionParams& params);

 private:
  void* impl_;  // MetalContextImpl* (see metal_context.mm)
  char device_name_[256];
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_

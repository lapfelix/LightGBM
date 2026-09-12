/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_

#include <cstddef>

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

 private:
  void* impl_;  // MetalContextImpl* (see metal_context.mm)
  char device_name_[256];
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_CONTEXT_H_

/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_
#define LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_

#include <LightGBM/dataset.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/random.h>

#include <memory>
#include <vector>

#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

#ifdef USE_METAL

#include "metal/metal_context.h"

namespace LightGBM {

/*!
 * \brief Metal-accelerated tree learner (Apple Silicon).
 *
 * Mirrors GPUTreeLearner structurally: dense (single) feature groups and
 * multival sub-features are histogrammed on the GPU via Metal device float
 * atomics, while oversized (>256 bins) multival sub-features stay on the CPU
 * wrapper path. Split search, partitioning, tree construction, objectives,
 * and the saved model format are unchanged from SerialTreeLearner, so
 * Metal-trained models load and predict anywhere.
 */
class MetalTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* config);
  ~MetalTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) override;
  void ResetIsConstantHessian(bool is_constant_hessian) override {
    // Pipelines are specialized lazily at dispatch time; nothing to rebuild.
    SerialTreeLearner::ResetIsConstantHessian(is_constant_hessian);
  }

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    SerialTreeLearner::SetBaggingData(subset, used_indices, num_data);
    if (subset == nullptr && used_indices != nullptr) {
      // determine if we are using bagging before we construct the data partition
      // thus we can start data movement to shared buffers earlier
      if (num_data != num_data_) {
        use_bagging_ = true;
        return;
      }
    }
    use_bagging_ = false;
  }

 protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
  void PartitionLeaf(int leaf, int feature, const uint32_t* threshold,
                     int num_threshold, bool default_left, int right_leaf) override;

 private:
  /*!
   * \brief Select bin size, create the Metal context, pack bins to shared buffers.
   */
  void InitMetal();
  //! \brief (Re)allocate shared buffers and repack dense bins row-major.
  void AllocateMetalMemory();
  void FreeMetalMemory();
  /*!
   * \brief Gather ordered gradients/Hessians/indices into shared buffers and
   *        dispatch histogram construction asynchronously (mirrors
   *        GPUTreeLearner::ConstructGPUHistogramsAsync).
   * \return true if a Metal dispatch was launched, false if Metal is not used
   */
  bool ConstructMetalHistogramsAsync(
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* data_indices, data_size_t num_data,
    const score_t* gradients, const score_t* hessians);
  //! \brief Wait for the outstanding dispatch and merge float histograms to double.
  void WaitAndGetHistograms(hist_t* histograms);
  //! \brief True when a leaf's histogram workload justifies a GPU dispatch.
  bool ShouldUseMetal(data_size_t num_rows, int num_used_gpu_columns) const;
  //! \brief True when a leaf's row count justifies a GPU partition dispatch.
  bool ShouldUseMetalPartition(data_size_t num_rows) const;
  /*!
   * \brief Partition one leaf on the GPU (stable partition, bit-identical to
   *        DataPartition::Split) and install the result.
   * \return true if the GPU path ran, false to fall back to the CPU path
   *         (multival/sparse group, oversized categorical bitset, tiny leaf).
   */
  bool TryPartitionMetal(int leaf, int feature, const uint32_t* threshold,
                         int num_threshold, bool default_left, int right_leaf);

  //! \brief True if bagging is used
  bool use_bagging_ = false;
  //! \brief Metal device/queue/pipeline management (see metal/metal_context.h)
  std::unique_ptr<MetalHistogramContext> ctx_;
  //! \brief total number of feature-groups
  int num_feature_groups_ = 0;
  //! \brief total number of dense feature-groups, which will be processed on GPU
  int num_dense_feature_groups_ = 0;
  //! \brief Max number of bins of training data, used to determine device_bin_size_
  int max_num_bin_ = 0;
  //! \brief Bins per dense group on device (64 or 256)
  int device_bin_size_ = 64;
  //! \brief Indices of all dense feature-groups
  std::vector<int> dense_feature_group_map_;
  //! \brief Reverse map: feature group -> dense ordinal, or -1 for
  //! multival/sparse groups (partitioning always uses the CPU path for those)
  std::vector<int> group_to_dense_col_;
  //! \brief Inner feature index of each GPU-packed multival sub-feature
  //! column. Multival sub-features with more than 256 bins stay on the CPU
  //! wrapper path; all others are histogrammed on the GPU exactly like dense
  //! sub-features (one packed byte column each).
  std::vector<int> mv_column_feature_;
  //! \brief Reverse map: inner feature -> multival column ordinal, or -1 for
  //! dense-group features and multival features left on the CPU path.
  std::vector<int> feature_to_mv_col_;
  //! \brief One packed copy segment: GPU group slots
  //! [src_bin_start, src_bin_start + count) land at leaf-pool bins
  //! [dst_hist_bin, dst_hist_bin + count).
  struct MetalCopySegment {
    int src_bin_start;
    int dst_hist_bin;
    int count;
  };
  //! \brief Packed copy plan per dense group (outer index = dense ordinal).
  //! The leaf pool stores each feature packed: the most-frequent bin is
  //! omitted (its mass folds in via leaf totals), so a dense group's packed
  //! width is smaller than its stored width whenever a sub-feature has its
  //! most-frequent bin at 0. In row-wise mode the pool additionally starts
  //! at bin 1 with multi-val groups consuming packed space that group
  //! boundaries don't account for. Copying whole groups at
  //! Dataset::GroupBinBoundary is therefore wrong twice over: it lands at
  //! the wrong base and never skips the packed-out bins. Each segment maps
  //! one sub-feature's stored range onto its packed range instead.
  std::vector<std::vector<MetalCopySegment>> dense_group_copy_plan_;
  //! \brief Packed copy segment per multival column (exactly one segment
  //! each: natural bins map onto the feature's packed range the same way as
  //! single-value sub-features).
  std::vector<MetalCopySegment> mv_column_copy_plan_;
  //! \brief Indices of all sparse feature-groups (informational only)
  std::vector<int> sparse_feature_group_map_;
  //! \brief Per-dense-group enable mask (1 = build histogram, 0 = skip)
  std::vector<char> feature_masks_;
  //! \brief Hessian value captured for constant-hessian dispatches
  score_t const_hessian_value_ = 0;
  //! \brief True while a Metal dispatch is in flight
  bool metal_pending_ = false;
  //! \brief Total packed byte columns (dense groups + multival sub-features).
  //! This is the packed row stride, the histogram kernel's group count, and
  //! the mask/output array width.
  int num_packed_columns_ = 0;
  //! \brief Shared buffers (MTLBuffer*, owned here, freed via the context)
  void* buf_bins_ = nullptr;     // num_data x num_dense uint8, row-major
  void* buf_grads_ = nullptr;    // num_data score_t (ordered per leaf)
  void* buf_hess_ = nullptr;     // num_data score_t (ordered per leaf)
  void* buf_indices_ = nullptr;  // num_data int32 leaf row ids
  void* buf_masks_ = nullptr;    // num_dense enable bytes
  void* buf_out_g_ = nullptr;    // num_dense x device_bin_size float
  void* buf_out_h_ = nullptr;    // num_dense x device_bin_size float
  // Stable-partition scratch (scatter output reuses buf_indices_).
  void* buf_part_in_ = nullptr;      // num_data int32 leaf row ids (input)
  void* buf_part_counts_ = nullptr;  // max_blocks uint32 per-block left counts
  void* buf_part_offsets_ = nullptr;  // max_blocks uint32 exclusive prefix
  void* buf_part_result_ = nullptr;  // uint32 total left count
  void* buf_cat_bits_ = nullptr;     // 256 uint32 categorical bitset staging
};

}  // namespace LightGBM

#else

// When Metal support is not compiled in, quit with an error message

namespace LightGBM {

class MetalTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("Metal Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DUSE_METAL=1 (macOS, Apple Silicon only)");
  }
};

}  // namespace LightGBM

#endif  // USE_METAL

#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_TREE_LEARNER_H_

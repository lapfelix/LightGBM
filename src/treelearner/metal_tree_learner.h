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
 * Mirrors GPUTreeLearner structurally: dense (single) feature groups are
 * histogrammed on the GPU via Metal device float atomics, while sparse
 * (multi-value) groups stay on the CPU. Split search, partitioning, tree
 * construction, objectives, and the saved model format are unchanged from
 * SerialTreeLearner, so Metal-trained models load and predict anywhere.
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
  bool ShouldUseMetal(data_size_t num_rows, int num_used_dense_groups) const;

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
  //! \brief Indices of all sparse feature-groups (informational only)
  std::vector<int> sparse_feature_group_map_;
  //! \brief Per-dense-group enable mask (1 = build histogram, 0 = skip)
  std::vector<char> feature_masks_;
  //! \brief Hessian value captured for constant-hessian dispatches
  score_t const_hessian_value_ = 0;
  //! \brief True while a Metal dispatch is in flight
  bool metal_pending_ = false;
  //! \brief Shared buffers (MTLBuffer*, owned here, freed via the context)
  void* buf_bins_ = nullptr;     // num_data x num_dense uint8, row-major
  void* buf_grads_ = nullptr;    // num_data score_t (ordered per leaf)
  void* buf_hess_ = nullptr;     // num_data score_t (ordered per leaf)
  void* buf_indices_ = nullptr;  // num_data int32 leaf row ids
  void* buf_masks_ = nullptr;    // num_dense enable bytes
  void* buf_out_g_ = nullptr;    // num_dense x device_bin_size float
  void* buf_out_h_ = nullptr;    // num_dense x device_bin_size float
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

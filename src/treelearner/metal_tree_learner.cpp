/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_METAL

#include "metal_tree_learner.h"

#include <LightGBM/bin.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../io/dense_bin.hpp"

namespace LightGBM {

// The Metal kernels read gradients/Hessians as float and indices as int32.
static_assert(sizeof(score_t) == sizeof(float),
              "Metal tree learner requires single-precision score_t");
static_assert(sizeof(data_size_t) == 4,
              "Metal tree learner requires 32-bit data_size_t");

MetalTreeLearner::MetalTreeLearner(const Config* config)
  : SerialTreeLearner(config) {
  use_bagging_ = false;
}

MetalTreeLearner::~MetalTreeLearner() {
  if (ctx_ != nullptr) {
    ctx_->LogStats();
  }
  FreeMetalMemory();
  ctx_.reset();
}

void MetalTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_feature_groups_ = train_data_->num_feature_groups();
  InitMetal();
}

void MetalTreeLearner::InitMetal() {
  max_num_bin_ = 0;
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (train_data_->IsMultiGroup(i)) {
      continue;
    }
    max_num_bin_ = std::max(max_num_bin_, train_data_->FeatureGroupNumBin(i));
  }
  if (max_num_bin_ <= 64) {
    device_bin_size_ = 64;
  } else if (max_num_bin_ <= 256) {
    device_bin_size_ = 256;
  } else {
    Log::Fatal("bin size %d cannot run on Metal", max_num_bin_);
  }
  // Same guidance as the OpenCL backend: padded bin counts waste device bins.
  int max_num_bin_no_categorical = 0;
  int cur_feature_group = 0;
  bool categorical_feature_found = false;
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    const int feature_group = train_data_->Feature2Group(inner_feature_index);
    const BinMapper* feature_bin_mapper = train_data_->FeatureBinMapper(inner_feature_index);
    if (feature_bin_mapper->bin_type() == BinType::CategoricalBin) {
      categorical_feature_found = true;
    }
    if (feature_group != cur_feature_group || inner_feature_index == num_features_ - 1) {
      if (!categorical_feature_found) {
        max_num_bin_no_categorical = std::max(max_num_bin_no_categorical,
                                              train_data_->FeatureGroupNumBin(cur_feature_group));
      }
      categorical_feature_found = false;
      cur_feature_group = feature_group;
    }
  }
  if (max_num_bin_no_categorical == 65) {
    Log::Warning("Setting max_bin to 63 is suggested for best performance");
  }
  ctx_ = std::unique_ptr<MetalHistogramContext>(new MetalHistogramContext());
  AllocateMetalMemory();
}

void MetalTreeLearner::FreeMetalMemory() {
  if (ctx_ == nullptr) {
    return;
  }
  ctx_->FreeBuffer(buf_bins_); buf_bins_ = nullptr;
  ctx_->FreeBuffer(buf_grads_); buf_grads_ = nullptr;
  ctx_->FreeBuffer(buf_hess_); buf_hess_ = nullptr;
  ctx_->FreeBuffer(buf_indices_); buf_indices_ = nullptr;
  ctx_->FreeBuffer(buf_masks_); buf_masks_ = nullptr;
  ctx_->FreeBuffer(buf_out_g_); buf_out_g_ = nullptr;
  ctx_->FreeBuffer(buf_out_h_); buf_out_h_ = nullptr;
}

void MetalTreeLearner::AllocateMetalMemory() {
  FreeMetalMemory();
  num_dense_feature_groups_ = 0;
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      ++num_dense_feature_groups_;
    }
  }
  dense_feature_group_map_.clear();
  sparse_feature_group_map_.clear();
  if (!num_dense_feature_groups_) {
    Log::Warning("Metal acceleration is disabled because no non-trivial dense features can be found");
    return;
  }
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      dense_feature_group_map_.push_back(i);
    } else {
      sparse_feature_group_map_.push_back(i);
    }
  }
  const int D = num_dense_feature_groups_;
  buf_bins_ = ctx_->AllocShared((size_t)num_data_ * D * sizeof(uint8_t));
  buf_grads_ = ctx_->AllocShared((size_t)num_data_ * sizeof(score_t));
  buf_hess_ = ctx_->AllocShared((size_t)num_data_ * sizeof(score_t));
  buf_indices_ = ctx_->AllocShared((size_t)num_data_ * sizeof(data_size_t));
  buf_masks_ = ctx_->AllocShared((size_t)D * sizeof(char));
  buf_out_g_ = ctx_->AllocShared((size_t)D * device_bin_size_ * sizeof(float));
  buf_out_h_ = ctx_->AllocShared((size_t)D * device_bin_size_ * sizeof(float));
  feature_masks_.assign(D, 0);

  // Pack dense bins row-major: byte (row, d) = raw stored bin of that group.
  // RawGet returns exactly the stored value the CPU histogram path accumulates
  // (see DenseBin::ConstructHistogramInner), so GPU/CPU layouts match.
  auto pack_start = std::chrono::steady_clock::now();
  std::vector<BinIterator*> iters(D);
  bool all_8bit = true, all_4bit = true;
  for (int d = 0; d < D; ++d) {
    BinIterator* it = train_data_->FeatureGroupIterator(dense_feature_group_map_[d]);
    const bool is_8bit = dynamic_cast<DenseBinIterator<uint8_t, false>*>(it) != nullptr;
    const bool is_4bit = dynamic_cast<DenseBinIterator<uint8_t, true>*>(it) != nullptr;
    if (!is_8bit && !is_4bit) {
      Log::Fatal("Metal tree learner only supports DenseBin and Dense4bitsBin for dense feature groups");
    }
    all_8bit = all_8bit && is_8bit;
    all_4bit = all_4bit && is_4bit;
    iters[d] = it;
  }
  uint8_t* bins = static_cast<uint8_t*>(ctx_->BufferContents(buf_bins_));
  if (all_8bit || all_4bit) {
    // Monomorphic loop: copying iterators by value lets the compiler resolve
    // RawGet statically (direct array access) instead of virtual dispatch.
    #define METAL_PACK_ROWS(ITER_T) \
      { \
        std::vector<ITER_T> concrete; \
        concrete.reserve(D); \
        for (int d = 0; d < D; ++d) { \
          concrete.push_back(*static_cast<ITER_T*>(iters[d])); \
        } \
        ITER_T* cits = concrete.data(); \
        _Pragma("omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)") \
        for (data_size_t r = 0; r < num_data_; ++r) { \
          uint8_t* dst = bins + (size_t)r * D; \
          for (int d = 0; d < D; ++d) { \
            dst[d] = static_cast<uint8_t>(cits[d].RawGet(r)); \
          } \
        } \
      }
    typedef DenseBinIterator<uint8_t, false> Dense8It;
    typedef DenseBinIterator<uint8_t, true> Dense4It;
    if (all_8bit) {
      METAL_PACK_ROWS(Dense8It)
    } else {
      METAL_PACK_ROWS(Dense4It)
    }
    #undef METAL_PACK_ROWS
  } else {
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t r = 0; r < num_data_; ++r) {
      uint8_t* dst = bins + (size_t)r * D;
      for (int d = 0; d < D; ++d) {
        dst[d] = static_cast<uint8_t>(iters[d]->RawGet(r));
      }
    }
  }
  for (auto* it : iters) {
    delete it;
  }
  std::chrono::duration<double> pack_seconds = std::chrono::steady_clock::now() - pack_start;
  Log::Info("%d dense feature groups (%.2f MB) packed for Metal in shared memory in %.3f s. %d sparse feature groups",
            D, (size_t)num_data_ * D / (1024.0 * 1024.0), pack_seconds.count(),
            (int)sparse_feature_group_map_.size());
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  num_feature_groups_ = train_data_->num_feature_groups();
  AllocateMetalMemory();
}

bool MetalTreeLearner::ShouldUseMetal(data_size_t num_rows, int num_used_dense_groups) const {
  if (num_dense_feature_groups_ == 0 || num_used_dense_groups <= 0) {
    return false;
  }
  // Synchronous GPU dispatches cost ~0.1-0.2 ms of scheduling latency on top
  // of kernel time; below this workload the CPU finishes first.
  const int64_t workload = (int64_t)num_rows * num_used_dense_groups;
  return workload >= (int64_t)config_->metal_min_hist_workload;
}

void MetalTreeLearner::BeforeTrain() {
  // Stage full gradients/Hessians in shared buffers (zero-copy on unified memory).
  if (!use_bagging_ && ShouldUseMetal(num_data_, num_dense_feature_groups_)) {
    if (!share_state_->is_constant_hessian) {
      std::memcpy(ctx_->BufferContents(buf_hess_), hessians_, (size_t)num_data_ * sizeof(score_t));
    } else if (num_data_ > 0) {
      const_hessian_value_ = hessians_[0];
    }
    std::memcpy(ctx_->BufferContents(buf_grads_), gradients_, (size_t)num_data_ * sizeof(score_t));
  }

  SerialTreeLearner::BeforeTrain();

  // use bagging: gather the used subset now, instead of at ConstructHistogram()
  if (data_partition_->leaf_count(0) != num_data_ &&
      ShouldUseMetal(data_partition_->leaf_count(0), num_dense_feature_groups_)) {
    const data_size_t* indices = data_partition_->indices();
    data_size_t cnt = data_partition_->leaf_count(0);
    std::memcpy(ctx_->BufferContents(buf_indices_), indices, (size_t)cnt * sizeof(data_size_t));
    score_t* ordered_hess = static_cast<score_t*>(ctx_->BufferContents(buf_hess_));
    score_t* ordered_grad = static_cast<score_t*>(ctx_->BufferContents(buf_grads_));
    if (!share_state_->is_constant_hessian) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < cnt; ++i) {
        ordered_hess[i] = hessians_[indices[i]];
      }
    } else if (cnt > 0) {
      const_hessian_value_ = hessians_[indices[0]];
    }
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < cnt; ++i) {
      ordered_grad[i] = gradients_[indices[i]];
    }
  }
}

bool MetalTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  int smaller_leaf;
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // only have root
  if (right_leaf < 0) {
    smaller_leaf = -1;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    smaller_leaf = left_leaf;
  } else {
    smaller_leaf = right_leaf;
  }

  // Stage indices, gradients and Hessians for the smaller leaf as early as possible.
  // Decided with the upper-bound group count; ConstructMetalHistogramsAsync
  // re-decides with the exact used count before gathering for larger leaves.
  if (smaller_leaf >= 0) {
    const data_size_t* indices = data_partition_->indices();
    data_size_t begin = data_partition_->leaf_begin(smaller_leaf);
    data_size_t end = begin + data_partition_->leaf_count(smaller_leaf);
    if (!ShouldUseMetal(end - begin, num_dense_feature_groups_)) {
      return SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);
    }
    std::memcpy(ctx_->BufferContents(buf_indices_), indices + begin,
                (size_t)(end - begin) * sizeof(data_size_t));
    score_t* ordered_hess = static_cast<score_t*>(ctx_->BufferContents(buf_hess_));
    score_t* ordered_grad = static_cast<score_t*>(ctx_->BufferContents(buf_grads_));
    if (!share_state_->is_constant_hessian) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        ordered_hess[i - begin] = hessians_[indices[i]];
      }
    } else if (end > begin) {
      const_hessian_value_ = hessians_[indices[begin]];
    }
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      ordered_grad[i - begin] = gradients_[indices[i]];
    }
  }
  return SerialTreeLearner::BeforeFindBestSplit(tree, left_leaf, right_leaf);
}

bool MetalTreeLearner::ConstructMetalHistogramsAsync(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data,
  const score_t* gradients, const score_t* hessians) {
  if (num_data <= 0) {
    return false;
  }
  if (!num_dense_feature_groups_) {
    return false;
  }
  // Identity rows when the leaf holds every row (root without bagging).
  const bool use_indices = (num_data != num_data_);

  // converted indices in is_feature_used to feature-group indices
  std::vector<int8_t> is_feature_group_used(num_feature_groups_, 0);
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1024) if (num_features_ >= 2048)
  for (int i = 0; i < num_features_; ++i) {
    if (is_feature_used[i]) {
      is_feature_group_used[train_data_->Feature2Group(i)] = 1;
    }
  }
  int used_dense_feature_groups = 0;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1024) reduction(+:used_dense_feature_groups) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      feature_masks_[i] = 1;
      ++used_dense_feature_groups;
    } else {
      feature_masks_[i] = 0;
    }
  }
  // if no feature group is used, just return and do not use Metal
  if (used_dense_feature_groups == 0) {
    return false;
  }
  // Small leaves stay on the CPU: decide before any gathering or dispatch.
  if (!ShouldUseMetal(num_data, used_dense_feature_groups)) {
    return false;
  }

  if (data_indices != nullptr && use_indices) {
    std::memcpy(ctx_->BufferContents(buf_indices_), data_indices,
                (size_t)num_data * sizeof(data_size_t));
  }
  score_t* ordered_grad = static_cast<score_t*>(ctx_->BufferContents(buf_grads_));
  score_t* ordered_hess = static_cast<score_t*>(ctx_->BufferContents(buf_hess_));
  if (gradients != nullptr) {
    if (use_indices) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_grad[i] = gradients[data_indices[i]];
      }
    } else {
      std::memcpy(ordered_grad, gradients, (size_t)num_data * sizeof(score_t));
    }
  }
  if (hessians != nullptr) {
    if (share_state_->is_constant_hessian) {
      const_hessian_value_ = use_indices ? hessians[data_indices[0]] : hessians[0];
    } else if (use_indices) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_hess[i] = hessians[data_indices[i]];
      }
    } else {
      std::memcpy(ordered_hess, hessians, (size_t)num_data * sizeof(score_t));
    }
  }
  std::memcpy(ctx_->BufferContents(buf_masks_), feature_masks_.data(),
              (size_t)num_dense_feature_groups_ * sizeof(char));
  // Atomics accumulate: outputs must start at zero for every dispatch.
  const size_t out_bytes = (size_t)num_dense_feature_groups_ * device_bin_size_ * sizeof(float);
  std::memset(ctx_->BufferContents(buf_out_g_), 0, out_bytes);
  std::memset(ctx_->BufferContents(buf_out_h_), 0, out_bytes);

  MetalHistogramContext::LaunchParams params;
  params.num_bins = device_bin_size_;
  params.use_indices = use_indices;
  params.const_hessian = share_state_->is_constant_hessian;
  params.const_hessian_value = const_hessian_value_;
  ctx_->LaunchHistogramAsync(buf_bins_, buf_grads_, buf_hess_, buf_indices_, buf_masks_,
                             buf_out_g_, buf_out_h_,
                             num_dense_feature_groups_, num_dense_feature_groups_,
                             num_data, params);
  metal_pending_ = true;
  return true;
}

void MetalTreeLearner::WaitAndGetHistograms(hist_t* histograms) {
  Common::FunctionTimer fun_timer("MetalTreeLearner::WaitAndGetHistograms", global_timer);
  ctx_->Wait();
  metal_pending_ = false;
  const float* hist_g = static_cast<const float*>(ctx_->BufferContents(buf_out_g_));
  const float* hist_h = static_cast<const float*>(ctx_->BufferContents(buf_out_h_));
  const int NB = device_bin_size_;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (!feature_masks_[i]) {
      continue;
    }
    const int dense_group_index = dense_feature_group_map_[i];
    hist_t* out = histograms + train_data_->GroupBinBoundary(dense_group_index) * 2;
    const int bin_size = train_data_->FeatureGroupNumBin(dense_group_index);
    const float* hg = hist_g + (size_t)i * NB;
    const float* hh = hist_h + (size_t)i * NB;
    for (int j = 0; j < bin_size; ++j) {
      GET_GRAD(out, j) = hg[j];
      GET_HESS(out, j) = hh[j];
    }
  }
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  Common::FunctionTimer fun_timer("MetalTreeLearner::ConstructHistograms", global_timer);
  const data_size_t smaller_cnt = smaller_leaf_splits_->num_data_in_leaf();
  const bool larger_needed = (larger_leaf_histogram_array_ != nullptr && !use_subtract);
  const data_size_t larger_cnt = larger_needed ? larger_leaf_splits_->num_data_in_leaf() : 0;
  // Provisional per-leaf decision with the upper-bound group count (the exact
  // used count is only known after the mask build, which re-decides).
  const bool smaller_maybe_gpu = ShouldUseMetal(smaller_cnt, num_dense_feature_groups_);
  const bool larger_maybe_gpu = larger_needed && ShouldUseMetal(larger_cnt, num_dense_feature_groups_);
  if (!smaller_maybe_gpu && !larger_maybe_gpu) {
    // Small leaves (or no dense groups): identical to the serial CPU path.
    SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
    return;
  }
  std::vector<int8_t> is_sparse_feature_used(num_features_, 0);
  int num_sparse_used = 0;
  int num_total_used = 0;
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    if (!is_feature_used[feature_index]) continue;
    ++num_total_used;
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(feature_index))) {
      is_sparse_feature_used[feature_index] = 1;
      ++num_sparse_used;
    }
  }
  // construct smaller leaf (indices/gradients prepared in BeforeFindBestSplit/BeforeTrain)
  hist_t* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  bool is_metal_used = false;
  if (smaller_maybe_gpu) {
    is_metal_used = ConstructMetalHistogramsAsync(is_feature_used,
      nullptr, smaller_cnt, nullptr, nullptr);
  }
  // CPU part: sparse features while Metal runs, or the full serial-equivalent
  // mask when Metal declined this leaf. Skipped entirely when the CPU has no
  // features to cover (an empty Dataset call still pays multi-val setup).
  const bool need_cpu_smaller = is_metal_used ? (num_sparse_used > 0) : (num_total_used > 0);
  if (need_cpu_smaller) {
    train_data_->ConstructHistograms<false, 0>(
      is_metal_used ? is_sparse_feature_used : is_feature_used,
      smaller_leaf_splits_->data_indices(), smaller_cnt,
      gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      share_state_.get(),
      ptr_smaller_leaf_hist_data);
  }
  // wait for Metal to finish, only if Metal is actually used
  if (is_metal_used) {
    WaitAndGetHistograms(ptr_smaller_leaf_hist_data);
  }

  if (larger_needed) {
    // construct larger leaf
    hist_t* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    is_metal_used = false;
    if (larger_maybe_gpu) {
      is_metal_used = ConstructMetalHistogramsAsync(is_feature_used,
        larger_leaf_splits_->data_indices(), larger_cnt,
        gradients_, hessians_);
    }
    // then construct sparse features on CPU (skipped when nothing to cover)
    const bool need_cpu_larger = is_metal_used ? (num_sparse_used > 0) : (num_total_used > 0);
    if (need_cpu_larger) {
      train_data_->ConstructHistograms<false, 0>(
        is_metal_used ? is_sparse_feature_used : is_feature_used,
        larger_leaf_splits_->data_indices(), larger_cnt,
        gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(),
        ptr_larger_leaf_hist_data);
    }
    // wait for Metal to finish, only if Metal is actually used
    if (is_metal_used) {
      WaitAndGetHistograms(ptr_larger_leaf_hist_data);
    }
  }
}

}  // namespace LightGBM

#endif  // USE_METAL

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

// Largest categorical threshold bitset staged for a GPU partition; larger
// bitsets (very high-cardinality categoricals) fall back to the CPU path.
static const int kMaxCatBitsetWords = 256;

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
  // Multival sub-features packed as GPU columns need device bins for their
  // natural bins too; oversized ones (>256 bins) stay on the CPU path and
  // must not raise the device bin size.
  for (int f = 0; f < num_features_; ++f) {
    if (!train_data_->IsMultiGroup(train_data_->Feature2Group(f))) {
      continue;
    }
    const int sub_num_bin = train_data_->FeatureNumBin(f);
    if (sub_num_bin <= 256) {
      max_num_bin_ = std::max(max_num_bin_, sub_num_bin);
    }
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
  ctx_->FreeBuffer(buf_part_in_); buf_part_in_ = nullptr;
  ctx_->FreeBuffer(buf_part_counts_); buf_part_counts_ = nullptr;
  ctx_->FreeBuffer(buf_part_offsets_); buf_part_offsets_ = nullptr;
  ctx_->FreeBuffer(buf_part_result_); buf_part_result_ = nullptr;
  ctx_->FreeBuffer(buf_cat_bits_); buf_cat_bits_ = nullptr;
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
  group_to_dense_col_.assign(num_feature_groups_, -1);
  // Multival sub-features become GPU columns holding natural bins. The packed
  // reader contract is universal across share-state layouts (col/row-wise,
  // sparse/dense wrapper): reader slot k holds natural bin k + (mfb == 0),
  // see FeatureMetainfo::offset. All CPU layouts produce bit-identical
  // models, so filling packed slots per that contract matches every CPU
  // path. Only oversized sub-features (>256 bins) stay on the CPU wrapper.
  mv_column_feature_.clear();
  mv_column_copy_plan_.clear();
  feature_to_mv_col_.assign(num_features_, -1);
  for (int f = 0; f < num_features_; ++f) {
    if (!train_data_->IsMultiGroup(train_data_->Feature2Group(f))) {
      continue;
    }
    if (train_data_->FeatureNumBin(f) > 256) {
      continue;
    }
    feature_to_mv_col_[f] = static_cast<int>(mv_column_feature_.size());
    mv_column_feature_.push_back(f);
  }
  num_packed_columns_ = num_dense_feature_groups_ + static_cast<int>(mv_column_feature_.size());
  if (num_packed_columns_ == 0) {
    Log::Warning("Metal acceleration is disabled because no non-trivial dense features can be found");
    return;
  }
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      group_to_dense_col_[i] = static_cast<int>(dense_feature_group_map_.size());
      dense_feature_group_map_.push_back(i);
    } else {
      sparse_feature_group_map_.push_back(i);
    }
  }
  // Packed copy plan: map each dense group's stored slots onto the leaf
  // pool's packed ranges. Sources come from the group's own cumulative
  // stored offsets (FeatureGroupBinOffsets); destinations come from the
  // share-state packed feature offsets, which is also what the readers use.
  // Stored slot 0 (rows at a most-frequent bin) is skipped on both sides:
  // readers fold that mass via leaf totals, so the GPU copy must too.
  // NOTE: Dataset::SubFeatureBinOffset is a 1/0 skip flag, not a cumulative
  // stored offset; do not use it here (it happens to work only for
  // single-feature groups).
  dense_group_copy_plan_.assign(num_dense_feature_groups_, {});
  {
    const std::vector<uint32_t>& packed_offsets =
        share_state_->feature_hist_offsets();
    const int total_bins = share_state_->num_hist_total_bin();
    for (int d = 0; d < num_dense_feature_groups_; ++d) {
      const int group = dense_feature_group_map_[d];
      std::vector<int> subs;
      for (int f = 0; f < num_features_; ++f) {
        if (train_data_->Feature2Group(f) == group) {
          subs.push_back(f);
        }
      }
      // Stored (sub-index) order, not feature-index order.
      std::sort(subs.begin(), subs.end(), [&](int a, int b) {
        return train_data_->Feature2SubFeature(a) <
               train_data_->Feature2SubFeature(b);
      });
      const std::vector<uint32_t>& stored_offsets =
          train_data_->FeatureGroupBinOffsets(group);
      CHECK_EQ(static_cast<int>(stored_offsets.size()), static_cast<int>(subs.size()) + 1);
      for (size_t s = 0; s < subs.size(); ++s) {
        const int f = subs[s];
        CHECK_EQ(train_data_->Feature2SubFeature(f), static_cast<int>(s));
        MetalCopySegment seg;
        seg.src_bin_start = static_cast<int>(stored_offsets[s]);
        seg.dst_hist_bin = static_cast<int>(packed_offsets[f]);
        seg.count = static_cast<int>(stored_offsets[s + 1] - stored_offsets[s]);
        // The segment must cover exactly what readers read for this
        // feature (num_bin slots minus the omitted most-frequent bin).
        const int reader_width = train_data_->FeatureNumBin(f) -
            (train_data_->FeatureBinMapper(f)->GetMostFreqBin() == 0 ? 1 : 0);
        CHECK_EQ(seg.count, reader_width);
        CHECK_GT(seg.count, 0);
        CHECK_GE(seg.dst_hist_bin, 0);
        CHECK_LE(seg.dst_hist_bin + seg.count, total_bins);
        dense_group_copy_plan_[d].push_back(seg);
      }
    }
  }
  // One copy segment per multival column: packed columns hold natural bins,
  // so stored slot s maps to packed slot dst like single-value sub-features
  // (stored slot 0 skipped exactly when the most-frequent bin is 0).
  {
    const std::vector<uint32_t>& packed_offsets =
        share_state_->feature_hist_offsets();
    const int total_bins = share_state_->num_hist_total_bin();
    for (size_t c = 0; c < mv_column_feature_.size(); ++c) {
      const int f = mv_column_feature_[c];
      const int num_bin = train_data_->FeatureNumBin(f);
      const bool mfb_is_zero =
          train_data_->FeatureBinMapper(f)->GetMostFreqBin() == 0;
      MetalCopySegment seg;
      seg.src_bin_start = mfb_is_zero ? 1 : 0;
      seg.dst_hist_bin = static_cast<int>(packed_offsets[f]);
      seg.count = num_bin - (mfb_is_zero ? 1 : 0);
      CHECK_GT(seg.count, 0);
      CHECK_GE(seg.dst_hist_bin, 0);
      CHECK_LE(seg.dst_hist_bin + seg.count, total_bins);
      mv_column_copy_plan_.push_back(seg);
    }
  }
  const int D = num_packed_columns_;
  const int Dd = num_dense_feature_groups_;
  buf_bins_ = ctx_->AllocShared((size_t)num_data_ * D * sizeof(uint8_t));
  buf_grads_ = ctx_->AllocShared((size_t)num_data_ * sizeof(score_t));
  buf_hess_ = ctx_->AllocShared((size_t)num_data_ * sizeof(score_t));
  buf_indices_ = ctx_->AllocShared((size_t)num_data_ * sizeof(data_size_t));
  buf_masks_ = ctx_->AllocShared((size_t)D * sizeof(char));
  buf_out_g_ = ctx_->AllocShared((size_t)D * device_bin_size_ * sizeof(float));
  buf_out_h_ = ctx_->AllocShared((size_t)D * device_bin_size_ * sizeof(float));
  // Stable-partition scratch. Scatter output reuses buf_indices_ (which is
  // restaged from the CPU index array by every histogram dispatch anyway).
  const int part_max_blocks =
      (num_data_ + MetalHistogramContext::kPartitionBlockRows - 1) / MetalHistogramContext::kPartitionBlockRows;
  buf_part_in_ = ctx_->AllocShared((size_t)num_data_ * sizeof(data_size_t));
  buf_part_counts_ = ctx_->AllocShared((size_t)part_max_blocks * sizeof(uint32_t));
  buf_part_offsets_ = ctx_->AllocShared((size_t)part_max_blocks * sizeof(uint32_t));
  buf_part_result_ = ctx_->AllocShared(4 * sizeof(uint32_t));
  buf_cat_bits_ = ctx_->AllocShared((size_t)kMaxCatBitsetWords * sizeof(uint32_t));
  feature_masks_.assign(D, 0);

  // Pack dense bins row-major: byte (row, d) = raw stored bin of that group.
  // RawGet returns exactly the stored value the CPU histogram path accumulates
  // (see DenseBin::ConstructHistogramInner), so GPU/CPU layouts match.
  auto pack_start = std::chrono::steady_clock::now();
  std::vector<BinIterator*> iters(Dd);
  bool all_8bit = true, all_4bit = true;
  for (int d = 0; d < Dd; ++d) {
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
        concrete.reserve(Dd); \
        for (int d = 0; d < Dd; ++d) { \
          concrete.push_back(*static_cast<ITER_T*>(iters[d])); \
        } \
        ITER_T* cits = concrete.data(); \
        _Pragma("omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)") \
        for (data_size_t r = 0; r < num_data_; ++r) { \
          uint8_t* dst = bins + (size_t)r * D; \
          for (int d = 0; d < Dd; ++d) { \
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
      for (int d = 0; d < Dd; ++d) {
        dst[d] = static_cast<uint8_t>(iters[d]->RawGet(r));
      }
    }
  }
  for (auto* it : iters) {
    delete it;
  }
  // Pack multival columns after the dense groups. Values are natural bins via
  // per-subfeature iterators (the same source PushDataToMultiValBin uses):
  // PushData stores most-frequent rows as absent and shifts the rest so that
  // Get() returns the natural bin in all cases. Sparse iterators carry a row
  // cursor, so each thread owns its iterators and visits row blocks in
  // ascending order.
  const int num_mv = static_cast<int>(mv_column_feature_.size());
  if (num_mv > 0) {
    const int num_threads = OMP_NUM_THREADS();
    std::vector<std::vector<BinIterator*>> mv_iters(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
      for (int c = 0; c < num_mv; ++c) {
        mv_iters[tid].push_back(train_data_->FeatureIterator(mv_column_feature_[c]));
      }
    }
    std::vector<int> mv_num_bin(num_mv, 0);
    for (int c = 0; c < num_mv; ++c) {
      mv_num_bin[c] = train_data_->FeatureNumBin(mv_column_feature_[c]);
    }
    Threading::For<data_size_t>(
        0, num_data_, 1024, [&](int tid, data_size_t start, data_size_t end) {
          for (int c = 0; c < num_mv; ++c) {
            mv_iters[tid][c]->Reset(start);
          }
          for (data_size_t r = start; r < end; ++r) {
            uint8_t* dst = bins + (size_t)r * D + Dd;
            for (int c = 0; c < num_mv; ++c) {
              const uint32_t v = mv_iters[tid][c]->Get(r);
              CHECK_LT(v, static_cast<uint32_t>(mv_num_bin[c]));
              dst[c] = static_cast<uint8_t>(v);
            }
          }
        });
    for (int tid = 0; tid < num_threads; ++tid) {
      for (auto* it : mv_iters[tid]) {
        delete it;
      }
    }
  }
  std::chrono::duration<double> pack_seconds = std::chrono::steady_clock::now() - pack_start;
  int mv_left_on_cpu = 0;
  for (int f = 0; f < num_features_; ++f) {
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(f)) &&
        feature_to_mv_col_[f] < 0) {
      ++mv_left_on_cpu;
    }
  }
  Log::Info("%d dense feature groups + %d multival columns (%.2f MB) packed for Metal in shared memory in %.3f s. %d sparse feature groups (%d multival features left on CPU)",
            Dd, num_mv, (size_t)num_data_ * D / (1024.0 * 1024.0), pack_seconds.count(),
            (int)sparse_feature_group_map_.size(), mv_left_on_cpu);
}

void MetalTreeLearner::ResetTrainingDataInner(const Dataset* train_data, bool is_constant_hessian, bool reset_multi_val_bin) {
  SerialTreeLearner::ResetTrainingDataInner(train_data, is_constant_hessian, reset_multi_val_bin);
  num_feature_groups_ = train_data_->num_feature_groups();
  AllocateMetalMemory();
}

bool MetalTreeLearner::ShouldUseMetal(data_size_t num_rows, int num_used_gpu_columns) const {
  if (num_packed_columns_ == 0 || num_used_gpu_columns <= 0) {
    return false;
  }
  // Synchronous GPU dispatches cost ~0.1-0.2 ms of scheduling latency on top
  // of kernel time; below this workload the CPU finishes first.
  const int64_t workload = (int64_t)num_rows * num_used_gpu_columns;
  return workload >= (int64_t)config_->metal_min_hist_workload;
}

bool MetalTreeLearner::ShouldUseMetalPartition(data_size_t num_rows) const {
  if (num_dense_feature_groups_ == 0 || buf_part_in_ == nullptr) {
    return false;
  }
  return num_rows >= config_->metal_min_partition_rows;
}

void MetalTreeLearner::PartitionLeaf(int leaf, int feature, const uint32_t* threshold,
                                     int num_threshold, bool default_left, int right_leaf) {
  Common::FunctionTimer fun_timer("MetalTreeLearner::PartitionLeaf", global_timer);
  if (TryPartitionMetal(leaf, feature, threshold, num_threshold, default_left, right_leaf)) {
    return;
  }
  data_partition_->Split(leaf, train_data_, feature, threshold, num_threshold,
                         default_left, right_leaf);
}

bool MetalTreeLearner::TryPartitionMetal(int leaf, int feature, const uint32_t* threshold,
                                         int num_threshold, bool default_left, int right_leaf) {
  const data_size_t cnt = data_partition_->leaf_count(leaf);
  if (cnt <= 0 || !ShouldUseMetalPartition(cnt)) {
    return false;
  }
  // Multival and sparse groups have no packed column; the CPU path owns them.
  const int group = train_data_->Feature2Group(feature);
  if (group < 0 || group >= static_cast<int>(group_to_dense_col_.size())) {
    return false;
  }
  const int d = group_to_dense_col_[group];
  if (d < 0) {
    return false;
  }
  const int sub = train_data_->Feature2SubFeature(feature);
  const std::vector<uint32_t>& stored_offsets = train_data_->FeatureGroupBinOffsets(group);
  if (sub < 0 || sub + 1 >= static_cast<int>(stored_offsets.size())) {
    return false;
  }
  // Mirror FeatureGroup::Split's dispatch: single-feature groups use the
  // min_bin=1 form, multi-feature groups the (min_bin, max_bin) form.
  const bool single_feature = (stored_offsets.size() == 2);
  MetalHistogramContext::PartitionParams params;
  params.group_col = static_cast<uint32_t>(d);
  params.row_stride = static_cast<uint32_t>(num_packed_columns_);
  if (single_feature) {
    params.min_bin = 1;
    params.max_bin = stored_offsets[1] - 1;
  } else {
    params.min_bin = stored_offsets[sub];
    params.max_bin = stored_offsets[sub + 1] - 1;
    params.flags |= MetalHistogramContext::kPartUseMinBin;
  }
  const BinMapper* mapper = train_data_->FeatureBinMapper(feature);
  params.default_bin = mapper->GetDefaultBin();
  params.most_freq_bin = mapper->GetMostFreqBin();
  if (default_left) {
    params.flags |= MetalHistogramContext::kPartDefaultLeft;
  }
  if (mapper->bin_type() != BinType::NumericalBin) {
    if (num_threshold <= 0 || num_threshold > kMaxCatBitsetWords) {
      return false;
    }
    params.flags |= MetalHistogramContext::kPartCategorical;
    params.num_cat_words = static_cast<uint32_t>(num_threshold);
    std::memcpy(ctx_->BufferContents(buf_cat_bits_), threshold,
                static_cast<size_t>(num_threshold) * sizeof(uint32_t));
  } else {
    if (num_threshold != 1) {
      return false;
    }
    params.threshold = threshold[0];
    // Mirror DenseBin::Split's missing-type dispatch exactly.
    const MissingType missing_type = mapper->missing_type();
    if (missing_type == MissingType::Zero) {
      params.flags |= MetalHistogramContext::kPartMissIsZero;
      if (params.default_bin == params.most_freq_bin) {
        params.flags |= MetalHistogramContext::kPartMfbIsZero;
      }
    } else if (missing_type == MissingType::NaN) {
      params.flags |= MetalHistogramContext::kPartMissIsNa;
      if (params.max_bin == params.most_freq_bin + params.min_bin &&
          params.most_freq_bin > 0) {
        params.flags |= MetalHistogramContext::kPartMfbIsNa;
      }
    }
  }
  if (metal_pending_) {
    ctx_->Wait();
    metal_pending_ = false;
  }
  const data_size_t begin = data_partition_->leaf_begin(leaf);
  std::memcpy(ctx_->BufferContents(buf_part_in_), data_partition_->indices() + begin,
              static_cast<size_t>(cnt) * sizeof(data_size_t));
  const int num_blocks =
      (cnt + MetalHistogramContext::kPartitionBlockRows - 1) / MetalHistogramContext::kPartitionBlockRows;
  ctx_->LaunchPartitionSync(buf_bins_, buf_part_in_, buf_indices_,
                            buf_part_counts_, buf_part_offsets_, buf_part_result_,
                            buf_cat_bits_, cnt, num_blocks, params);
  const uint32_t left_cnt =
      static_cast<const uint32_t*>(ctx_->BufferContents(buf_part_result_))[0];
  CHECK_LE(left_cnt, static_cast<uint32_t>(cnt));
  data_partition_->ApplyPartition(
      leaf, static_cast<const data_size_t*>(ctx_->BufferContents(buf_indices_)),
      static_cast<data_size_t>(left_cnt), right_leaf);
  return true;
}

void MetalTreeLearner::BeforeTrain() {
  // Stage full gradients/Hessians in shared buffers (zero-copy on unified memory).
  if (!use_bagging_ && ShouldUseMetal(num_data_, num_packed_columns_)) {
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
      ShouldUseMetal(data_partition_->leaf_count(0), num_packed_columns_)) {
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
    if (!ShouldUseMetal(end - begin, num_packed_columns_)) {
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
  if (num_packed_columns_ == 0) {
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
  int used_gpu_columns = 0;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 1024) reduction(+:used_gpu_columns) if (num_dense_feature_groups_ >= 2048)
  for (int i = 0; i < num_dense_feature_groups_; ++i) {
    if (is_feature_group_used[dense_feature_group_map_[i]]) {
      feature_masks_[i] = 1;
      ++used_gpu_columns;
    } else {
      feature_masks_[i] = 0;
    }
  }
  // Multival columns are masked per feature, with the same used definition
  // the CPU sparse path uses (bytree && is_feature_used), so the two paths
  // partition the multival features exactly: a feature is histogrammed
  // either on the GPU or by the CPU wrapper, never both.
  for (size_t c = 0; c < mv_column_feature_.size(); ++c) {
    const int f = mv_column_feature_[c];
    if (col_sampler_.is_feature_used_bytree()[f] && is_feature_used[f]) {
      feature_masks_[num_dense_feature_groups_ + c] = 1;
      ++used_gpu_columns;
    } else {
      feature_masks_[num_dense_feature_groups_ + c] = 0;
    }
  }
  // if no feature group is used, just return and do not use Metal
  if (used_gpu_columns == 0) {
    return false;
  }
  // Small leaves stay on the CPU: decide before any gathering or dispatch.
  if (!ShouldUseMetal(num_data, used_gpu_columns)) {
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
              (size_t)num_packed_columns_ * sizeof(char));
  // Atomics accumulate: outputs must start at zero for every dispatch.
  const size_t out_bytes = (size_t)num_packed_columns_ * device_bin_size_ * sizeof(float);
  std::memset(ctx_->BufferContents(buf_out_g_), 0, out_bytes);
  std::memset(ctx_->BufferContents(buf_out_h_), 0, out_bytes);

  MetalHistogramContext::LaunchParams params;
  params.num_bins = device_bin_size_;
  params.use_indices = use_indices;
  params.const_hessian = share_state_->is_constant_hessian;
  params.const_hessian_value = const_hessian_value_;
  ctx_->LaunchHistogramAsync(buf_bins_, buf_grads_, buf_hess_, buf_indices_, buf_masks_,
                             buf_out_g_, buf_out_h_,
                             num_packed_columns_, num_packed_columns_,
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
  const int num_dense = num_dense_feature_groups_;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int i = 0; i < num_packed_columns_; ++i) {
    if (!feature_masks_[i]) {
      continue;
    }
    const float* hg = hist_g + (size_t)i * NB;
    const float* hh = hist_h + (size_t)i * NB;
    if (i < num_dense) {
      for (const auto& seg : dense_group_copy_plan_[i]) {
        hist_t* out = histograms + seg.dst_hist_bin * 2;
        for (int j = 0; j < seg.count; ++j) {
          GET_GRAD(out, j) = hg[seg.src_bin_start + j];
          GET_HESS(out, j) = hh[seg.src_bin_start + j];
        }
      }
    } else {
      const MetalCopySegment& seg = mv_column_copy_plan_[i - num_dense];
      hist_t* out = histograms + seg.dst_hist_bin * 2;
      for (int j = 0; j < seg.count; ++j) {
        GET_GRAD(out, j) = hg[seg.src_bin_start + j];
        GET_HESS(out, j) = hh[seg.src_bin_start + j];
      }
    }
  }
}

void MetalTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  Common::FunctionTimer fun_timer("MetalTreeLearner::ConstructHistograms", global_timer);
  const data_size_t smaller_cnt = smaller_leaf_splits_->num_data_in_leaf();
  const bool larger_needed = (larger_leaf_histogram_array_ != nullptr && !use_subtract);
  const data_size_t larger_cnt = larger_needed ? larger_leaf_splits_->num_data_in_leaf() : 0;
  // Provisional per-leaf decision with the upper-bound column count (the exact
  // used count is only known after the mask build, which re-decides).
  const bool smaller_maybe_gpu = ShouldUseMetal(smaller_cnt, num_packed_columns_);
  const bool larger_maybe_gpu = larger_needed && ShouldUseMetal(larger_cnt, num_packed_columns_);
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
    // Multival features with a GPU column are histogrammed on the GPU (see
    // the mask build); only the remainder goes to the CPU wrapper.
    if (train_data_->IsMultiGroup(train_data_->Feature2Group(feature_index)) &&
        feature_to_mv_col_[feature_index] < 0) {
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

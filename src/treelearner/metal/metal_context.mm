/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <LightGBM/utils/log.h>

#include <time.h>

#include <cstring>
#include <map>
#include <string>
#include <tuple>

#include "metal_context.h"

// Embedded Metal kernel source (dual-mode header trick, see the .metal file).
static const char* kMetalKernelRawSource = {
#include "metal_histograms.metal"
};

static const char* kMetalSourceMarker = "// __METAL_KERNELS_BEGIN__";

namespace LightGBM {

// Pipeline cache key: (num_bins, use_indices, const_hessian).
typedef std::tuple<int, bool, bool> PipelineKey;

struct MetalContextImpl {
  // All id-typed members are __bridge_retained and CFRelease'd (ARC forbids
  // object pointers in structs, hence the void* storage).
  void* device = nil;
  void* queue = nil;
  void* pending = nil;
  std::map<PipelineKey, void*> pipelines;
  uint64_t num_launches = 0;
  uint64_t num_rows = 0;
  double wait_seconds = 0.0;
  double exec_seconds = 0.0;
};

static void ReleaseImpl(MetalContextImpl* impl) {
  if (impl->pending != nil) {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)(impl->pending);
    [cb waitUntilCompleted];
    CFRelease(impl->pending);
    impl->pending = nil;
  }
  for (auto& kv : impl->pipelines) {
    CFRelease(kv.second);
  }
  impl->pipelines.clear();
  if (impl->queue != nil) { CFRelease(impl->queue); }
  if (impl->device != nil) { CFRelease(impl->device); }
  delete impl;
}

MetalHistogramContext::MetalHistogramContext() : impl_(nullptr) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      Log::Fatal("Metal backend requested, but no Metal device is available on this system");
    }
    if (std::strstr(kMetalKernelRawSource, kMetalSourceMarker) == nullptr) {
      Log::Fatal("Metal backend: embedded kernel source marker not found");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      Log::Fatal("Metal backend: failed to create a command queue");
    }
    MetalContextImpl* impl = new MetalContextImpl();
    impl->device = (__bridge_retained void*)device;
    impl->queue = (__bridge_retained void*)queue;
    impl_ = impl;
    std::memset(device_name_, 0, sizeof(device_name_));
    std::strncpy(device_name_, [[device name] UTF8String], sizeof(device_name_) - 1);
    Log::Info("Metal backend: using device %s", device_name_);
  }
}

MetalHistogramContext::~MetalHistogramContext() {
  if (impl_ != nullptr) {
    @autoreleasepool {
      ReleaseImpl(static_cast<MetalContextImpl*>(impl_));
    }
    impl_ = nullptr;
  }
}

const char* MetalHistogramContext::device_name() const {
  return device_name_;
}

void* MetalHistogramContext::AllocShared(size_t bytes) {
  @autoreleasepool {
    MetalContextImpl* impl = static_cast<MetalContextImpl*>(impl_);
    id<MTLDevice> device = (__bridge id<MTLDevice>)(impl->device);
    if (bytes == 0) {
      bytes = 1;
    }
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      Log::Fatal("Metal backend: failed to allocate %zu shared bytes", bytes);
    }
    return (__bridge_retained void*)buffer;
  }
}

void MetalHistogramContext::FreeBuffer(void* buffer) {
  if (buffer != nullptr) {
    CFRelease(buffer);
  }
}

void* MetalHistogramContext::BufferContents(void* buffer) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(buffer);
  return [buf contents];
}

static id GetPipeline(MetalContextImpl* impl, const MetalHistogramContext::LaunchParams& params) {
  PipelineKey key(params.num_bins, params.use_indices, params.const_hessian);
  auto it = impl->pipelines.find(key);
  if (it != impl->pipelines.end()) {
    return (__bridge id)(it->second);
  }
  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)(impl->device);
    // Specialize via preprocessor macros in a per-key library. At most
    // 8 combinations exist (NB x use_indices x const_hessian); each is
    // compiled once and cached.
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_1;
    options.preprocessorMacros = @{
      @"NB": @(params.num_bins),
      @"USE_INDICES": @(params.use_indices ? 1 : 0),
      @"CONST_HESSIAN": @(params.const_hessian ? 1 : 0),
    };
    const char* src = std::strstr(kMetalKernelRawSource, kMetalSourceMarker);
    NSString* source = [NSString stringWithUTF8String:src];
    NSError* error = nil;
    id<MTLLibrary> specialized =
        [device newLibraryWithSource:source options:options error:&error];
    if (specialized == nil) {
      Log::Fatal("Metal backend: kernel specialization failed (NB=%d, indices=%d, const_h=%d): %s",
                 params.num_bins, (int)params.use_indices, (int)params.const_hessian,
                 error != nil ? [[error description] UTF8String] : "unknown error");
    }
    id<MTLFunction> function = [specialized newFunctionWithName:@"metal_histogram"];
    if (function == nil) {
      Log::Fatal("Metal backend: function 'metal_histogram' not found in specialized library");
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      Log::Fatal("Metal backend: pipeline creation failed: %s",
                 error != nil ? [[error description] UTF8String] : "unknown error");
    }
    impl->pipelines[key] = (__bridge_retained void*)pipeline;
    return pipeline;
  }
}

void MetalHistogramContext::LaunchHistogramAsync(
    const void* bins, const void* grads, const void* hess,
    const void* indices, const void* masks,
    void* out_g, void* out_h,
    int row_stride, int num_groups, int num_rows,
    const LaunchParams& params) {
  @autoreleasepool {
    MetalContextImpl* impl = static_cast<MetalContextImpl*>(impl_);
    if (impl->pending != nil) {
      // The learner always waits before relaunching; fail loudly otherwise.
      Log::Fatal("Metal backend: LaunchHistogramAsync called while a dispatch is still in flight");
    }
    id<MTLComputePipelineState> pipeline = GetPipeline(impl, params);
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(impl->queue);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:(__bridge id<MTLBuffer>)bins offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)grads offset:0 atIndex:1];
    if (!params.const_hessian) {
      [enc setBuffer:(__bridge id<MTLBuffer>)hess offset:0 atIndex:2];
    }
    if (params.use_indices) {
      [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:3];
    }
    [enc setBuffer:(__bridge id<MTLBuffer>)masks offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_g offset:0 atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)out_h offset:0 atIndex:6];
    uint32_t stride = (uint32_t)row_stride;
    uint32_t groups = (uint32_t)num_groups;
    uint32_t rows = (uint32_t)num_rows;
    [enc setBytes:&stride length:sizeof(stride) atIndex:7];
    [enc setBytes:&groups length:sizeof(groups) atIndex:8];
    [enc setBytes:&rows length:sizeof(rows) atIndex:9];
    [enc setBytes:&params.const_hessian_value length:sizeof(float) atIndex:10];
    const uint32_t tg_size = 256;
    uint32_t grid = ((uint32_t)num_rows + tg_size - 1) / tg_size * tg_size;
    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
    [cb commit];
    impl->pending = (__bridge_retained void*)cb;
    impl->num_launches += 1;
    impl->num_rows += (uint64_t)num_rows;
  }
}

void MetalHistogramContext::Wait() {
  MetalContextImpl* impl = static_cast<MetalContextImpl*>(impl_);
  if (impl->pending == nil) {
    return;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)(impl->pending);
    impl->pending = nil;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    [cb waitUntilCompleted];
    clock_gettime(CLOCK_MONOTONIC, &t1);
    impl->wait_seconds += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    impl->exec_seconds += [cb GPUEndTime] - [cb GPUStartTime];
    if ([cb status] == MTLCommandBufferStatusError) {
      NSError* error = [cb error];
      Log::Fatal("Metal backend: command buffer failed: %s",
                 error != nil ? [[error description] UTF8String] : "unknown error");
    }
  }
}

void MetalHistogramContext::LogStats() const {
  const MetalContextImpl* impl = static_cast<const MetalContextImpl*>(impl_);
  if (impl->num_launches == 0) {
    return;
  }
  Log::Info("Metal backend: %llu dispatches, %llu leaf rows, %.3f s in GPU wait (%.1f us/dispatch), %.3f s GPU exec",
            impl->num_launches, impl->num_rows, impl->wait_seconds,
            impl->wait_seconds * 1e6 / impl->num_launches, impl->exec_seconds);
}

}  // namespace LightGBM

#endif  // USE_METAL

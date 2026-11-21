#include "runtime/metal_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace m1triton::runtime {

namespace {

constexpr uintptr_t kApplePageSize = 16384;

bool is_page_aligned(uintptr_t ptr, uint64_t nbytes) {
  return ptr != 0 && (ptr % kApplePageSize) == 0 && nbytes > 0 && (nbytes % kApplePageSize) == 0;
}

}  // namespace

struct CompiledEntry {
  std::string fn_name;
  id<MTLComputePipelineState> pso = nil;
};

struct MetalRuntime::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLBinaryArchive> binary_archive = nil;
  std::string archive_path;
  id<MTLCommandBuffer> last_cmd_buf = nil;
  std::atomic<uint64_t> dispatches{0};
  mutable std::mutex mu;
  std::string last_kernel;
  std::unordered_map<std::string, uint64_t> key_to_id;
  std::vector<CompiledEntry> kernels;

  void init_binary_archive() {
    if (device == nil) return;
    const char* env_dir = std::getenv("M1TRITON_CACHE_DIR");
    std::filesystem::path cache_dir =
        (env_dir && env_dir[0] != '\0') ? std::filesystem::path(env_dir) : std::filesystem::path(".m1triton_cache");
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    archive_path = (cache_dir / "pipelines.metallib").string();

    MTLBinaryArchiveDescriptor* desc = [[MTLBinaryArchiveDescriptor alloc] init];
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path.c_str()]];
    if (std::filesystem::exists(archive_path)) {
      desc.url = url;
    }
    NSError* err = nil;
    binary_archive = [device newBinaryArchiveWithDescriptor:desc error:&err];
    if (binary_archive == nil && desc.url != nil) {
      desc.url = nil;
      binary_archive = [device newBinaryArchiveWithDescriptor:desc error:&err];
    }
  }
};

MetalRuntime::MetalRuntime() : impl_(new Impl()) {
  @autoreleasepool {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (impl_->device != nil) {
      impl_->queue = [impl_->device newCommandQueue];
      impl_->init_binary_archive();
    }
    impl_->kernels.push_back(CompiledEntry{});
  }
}

MetalRuntime::~MetalRuntime() {
  delete impl_;
}

MetalRuntime& MetalRuntime::instance() {
  static MetalRuntime rt;
  return rt;
}

DeviceInfo MetalRuntime::device_info() const {
  DeviceInfo info;
  @autoreleasepool {
    if (impl_->device == nil) {
      return info;
    }
    info.available = true;
    info.name = [[impl_->device name] UTF8String];
    info.max_threads_per_threadgroup = [impl_->device maxThreadsPerThreadgroup].width;
    info.max_threadgroup_memory_bytes = [impl_->device maxThreadgroupMemoryLength];
    info.recommended_max_working_set_bytes = [impl_->device recommendedMaxWorkingSetSize];
    info.has_unified_memory = [impl_->device hasUnifiedMemory];
  }
  return info;
}

uint64_t MetalRuntime::dispatch_count() const {
  return impl_->dispatches.load(std::memory_order_acquire);
}

void MetalRuntime::reset_dispatch_count() {
  impl_->dispatches.store(0, std::memory_order_release);
}

std::string MetalRuntime::last_kernel_name() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->last_kernel;
}

void MetalRuntime::synchronize() {
  @autoreleasepool {
    id<MTLCommandBuffer> pending = nil;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      pending = impl_->last_cmd_buf;
      impl_->last_cmd_buf = nil;
    }
    if (pending != nil) {
      [pending waitUntilCompleted];
    }
  }
}

uint64_t MetalRuntime::compile_kernel(
    const std::string& cache_key,
    const std::string& msl_source,
    const std::string& fn_name,
    bool fast_math,
    std::string* error_out) {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->key_to_id.find(cache_key);
    if (it != impl_->key_to_id.end()) {
      return it->second;
    }
  }

  @autoreleasepool {
    if (impl_->device == nil) {
      if (error_out) *error_out = "No Metal device available";
      return 0;
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;

    id<MTLLibrary> lib = [impl_->device newLibraryWithSource:[NSString stringWithUTF8String:msl_source.c_str()]
                                                     options:opts
                                                       error:&err];
    if (lib == nil) {
      if (error_out) {
        *error_out = err ? [[err localizedDescription] UTF8String] : "Unknown Metal compile error";
      }
      return 0;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fn_name.c_str()]];
    if (fn == nil) {
      if (error_out) *error_out = "Metal function not found: " + fn_name;
      return 0;
    }

    MTLComputePipelineDescriptor* pipe_desc = [[MTLComputePipelineDescriptor alloc] init];
    pipe_desc.computeFunction = fn;
    pipe_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    if (impl_->binary_archive != nil) {
      pipe_desc.binaryArchives = @[ impl_->binary_archive ];
    }

    id<MTLComputePipelineState> pso = [impl_->device newComputePipelineStateWithDescriptor:pipe_desc
                                                                                   options:MTLPipelineOptionNone
                                                                                reflection:nil
                                                                                     error:&err];
    if (pso == nil) {
      pipe_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = NO;
      pso = [impl_->device newComputePipelineStateWithDescriptor:pipe_desc
                                                         options:MTLPipelineOptionNone
                                                      reflection:nil
                                                           error:&err];
    }
    if (pso == nil) {
      if (error_out) {
        *error_out = err ? [[err localizedDescription] UTF8String] : "Pipeline creation failed";
      }
      return 0;
    }

    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->key_to_id.find(cache_key);
    if (it != impl_->key_to_id.end()) {
      return it->second;
    }
    if (impl_->binary_archive != nil && !impl_->archive_path.empty()) {
      NSError* arch_err = nil;
      [impl_->binary_archive addComputePipelineFunctionsWithDescriptor:pipe_desc error:&arch_err];
      NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:impl_->archive_path.c_str()]];
      [impl_->binary_archive serializeToURL:url error:&arch_err];
    }
    uint64_t new_id = static_cast<uint64_t>(impl_->kernels.size());
    impl_->kernels.push_back(CompiledEntry{fn_name, pso});
    impl_->key_to_id.emplace(cache_key, new_id);
    return new_id;
  }
}

double MetalRuntime::launch_kernel(
    uint64_t kernel_id,
    const std::vector<BufferArg>& buffers,
    const std::vector<ScalarArg>& scalars,
    const LaunchConfig& cfg,
    std::string* error_out) {
  CompiledEntry entry;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (kernel_id == 0 || kernel_id >= impl_->kernels.size()) {
      if (error_out) *error_out = "Invalid compiled Metal kernel id";
      return -1.0;
    }
    entry = impl_->kernels[kernel_id];
  }

  @autoreleasepool {
    if (impl_->device == nil || impl_->queue == nil) {
      if (error_out) *error_out = "Metal runtime uninitialized";
      return -1.0;
    }

    struct CpuCopyBack {
      void* host_ptr = nullptr;
      id<MTLBuffer> mtl_buf = nil;
      uint64_t nbytes = 0;
    };
    std::vector<CpuCopyBack> copy_backs;
    std::vector<id<MTLBuffer>> temp_buffers;
    bool needs_sync = cfg.wait;

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:entry.pso];

    for (const auto& b : buffers) {
      if (b.is_metal_buffer) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)b.handle;
        [enc setBuffer:buf offset:static_cast<NSUInteger>(b.byte_offset) atIndex:b.index];
      } else {
        uint8_t* raw_ptr = reinterpret_cast<uint8_t*>(b.handle) + b.byte_offset;
        uint64_t alloc_bytes = b.nbytes > 0 ? b.nbytes : 4;
        if (is_page_aligned(reinterpret_cast<uintptr_t>(raw_ptr), alloc_bytes)) {
          id<MTLBuffer> buf = [impl_->device newBufferWithBytesNoCopy:raw_ptr
                                                               length:static_cast<NSUInteger>(alloc_bytes)
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
          temp_buffers.push_back(buf);
          [enc setBuffer:buf offset:0 atIndex:b.index];
          needs_sync = true;
        } else {
          id<MTLBuffer> buf = [impl_->device newBufferWithBytes:raw_ptr
                                                         length:static_cast<NSUInteger>(alloc_bytes)
                                                        options:MTLResourceStorageModeShared];
          temp_buffers.push_back(buf);
          [enc setBuffer:buf offset:0 atIndex:b.index];
          if (b.writable) {
            copy_backs.push_back(CpuCopyBack{raw_ptr, buf, b.nbytes});
            needs_sync = true;
          }
        }
      }
    }

    for (const auto& s : scalars) {
      if (!s.bytes.empty()) {
        [enc setBytes:s.bytes.data() length:s.bytes.size() atIndex:s.index];
      }
    }

    if (cfg.threadgroup_mem_bytes > 0) {
      [enc setThreadgroupMemoryLength:cfg.threadgroup_mem_bytes atIndex:0];
    }

    MTLSize grid = MTLSizeMake(cfg.grid_x, cfg.grid_y, cfg.grid_z);
    MTLSize tg = MTLSizeMake(cfg.tg_x, cfg.tg_y, cfg.tg_z);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];

    impl_->dispatches.fetch_add(1, std::memory_order_acq_rel);
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      impl_->last_kernel = entry.fn_name;
      impl_->last_cmd_buf = needs_sync ? nil : cb;
    }

    if (needs_sync) {
      [cb waitUntilCompleted];
      if ([cb status] == MTLCommandBufferStatusError) {
        if (error_out) {
          NSError* err = [cb error];
          *error_out = err ? [[err localizedDescription] UTF8String] : "Metal command buffer error";
        }
        return -1.0;
      }
      for (const auto& cbk : copy_backs) {
        std::memcpy(cbk.host_ptr, [cbk.mtl_buf contents], static_cast<size_t>(cbk.nbytes));
      }
      return ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    }
    return 0.0;
  }
}

bool MetalRuntime::compile_msl_smoke(
    const std::string& msl_source, const std::string& fn_name, std::string* error_out) {
  std::string key = "smoke::" + fn_name + "::" + std::to_string(std::hash<std::string>{}(msl_source));
  return compile_kernel(key, msl_source, fn_name, false, error_out) != 0;
}

}  // namespace m1triton::runtime

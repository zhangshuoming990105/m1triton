#include "runtime/metal_runtime.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace m1triton::runtime {

namespace {

constexpr uintptr_t kApplePageSize = 16384;
constexpr size_t kScalarBufferAlign = 256;
constexpr NSUInteger kDefaultIcbMaxBindCount = 16;
constexpr NSUInteger kMetalMaxBufferBindings = 31;

bool is_page_aligned(uintptr_t ptr, uint64_t nbytes) {
  return ptr != 0 && (ptr % kApplePageSize) == 0 && nbytes > 0 && (nbytes % kApplePageSize) == 0;
}

bool buffers_overlap(const BufferArg& a, const BufferArg& b) {
  if (a.handle != b.handle) {
    return false;
  }
  if (a.nbytes == 0 || b.nbytes == 0) {
    return true;
  }
  const uint64_t a_end = a.byte_offset + a.nbytes;
  const uint64_t b_end = b.byte_offset + b.nbytes;
  return (a.byte_offset < b_end) && (b.byte_offset < a_end);
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

bool probe_compute_icb(id<MTLDevice> dev) {
  if (dev == nil) {
    return false;
  }
  if (@available(macOS 11.0, *)) {
    MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = kDefaultIcbMaxBindCount;
    id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:desc
                                                                   maxCommandCount:1
                                                                           options:MTLResourceStorageModeShared];
    return icb != nil;
  }
  return false;
}

double command_buffer_gpu_ms(id<MTLCommandBuffer> cb) {
  const double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
  return ms > 0.0 ? ms : 0.0;
}

struct CpuCopyBack {
  void* host_ptr = nullptr;
  id<MTLBuffer> mtl_buf = nil;
  uint64_t nbytes = 0;
};

struct CpuBufKey {
  uintptr_t ptr = 0;
  uint64_t nbytes = 0;

  bool operator==(const CpuBufKey& other) const { return ptr == other.ptr && nbytes == other.nbytes; }
};

struct CpuBufKeyHash {
  size_t operator()(const CpuBufKey& key) const {
    return std::hash<uintptr_t>{}(key.ptr) ^ (std::hash<uint64_t>{}(key.nbytes) << 1);
  }
};

}  // namespace

struct CompiledEntry {
  std::string fn_name;
  id<MTLComputePipelineState> pso = nil;
  bool supports_icb = false;
};

struct BatchState {
  bool active = false;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  std::vector<id<MTLBuffer>> temp_buffers;
  std::unordered_map<CpuBufKey, id<MTLBuffer>, CpuBufKeyHash> cpu_buf_cache;
  std::unordered_set<uintptr_t> copy_back_ptrs;
  std::vector<CpuCopyBack> copy_backs;
  bool needs_sync = false;
  uint64_t dispatch_count = 0;
  uint32_t last_tg_mem_bytes = 0;
};

struct CapturedDispatch {
  uint64_t kernel_id = 0;
  std::string fn_name;
  id<MTLComputePipelineState> pso = nil;
  std::vector<BufferArg> buffers;
  std::vector<id<MTLBuffer>> retained_buffers;
  std::vector<ScalarArg> scalars;
  LaunchConfig cfg;
};

struct GraphCaptureState {
  bool active = false;
  std::vector<CapturedDispatch> dispatches;
};

struct MetalGraphEntry {
  id<MTLIndirectCommandBuffer> icb = nil;
  id<MTLBuffer> scalar_buf = nil;
  std::vector<id<MTLBuffer>> retained_buffers;
  std::vector<__unsafe_unretained id<MTLResource>> resource_ptrs;
  std::vector<id<MTLComputePipelineState>> retained_psos;
  std::vector<NSRange> exec_ranges;
  size_t command_count = 0;
  std::string last_kernel;
};

struct MetalRuntime::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLBinaryArchive> binary_archive = nil;
  std::string archive_path;
  id<MTLCommandBuffer> last_cmd_buf = nil;
  bool supports_metal3 = false;
  bool supports_icb = false;
  std::atomic<uint64_t> dispatches{0};
  mutable std::mutex mu;
  std::string last_kernel;
  std::unordered_map<std::string, uint64_t> key_to_id;
  std::vector<CompiledEntry> kernels;
  BatchState batch;
  GraphCaptureState graph_capture;
  uint64_t next_graph_id = 1;
  std::unordered_map<uint64_t, std::shared_ptr<MetalGraphEntry>> graphs;

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
      if (@available(macOS 13.0, *)) {
        impl_->supports_metal3 = [impl_->device supportsFamily:MTLGPUFamilyMetal3];
      }
      impl_->supports_icb = probe_compute_icb(impl_->device);
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
    info.supports_metal3 = impl_->supports_metal3;
    info.supports_indirect_command_buffers = impl_->supports_icb;
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
    if (impl_->supports_metal3) {
      if (@available(macOS 13.0, *)) {
        opts.languageVersion = MTLLanguageVersion3_0;
      }
    }

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
    if (@available(macOS 11.0, *)) {
      pipe_desc.supportIndirectCommandBuffers = impl_->supports_icb ? YES : NO;
    }
    if (impl_->binary_archive != nil) {
      pipe_desc.binaryArchives = @[ impl_->binary_archive ];
    }

    auto try_create_pso = [&](BOOL tg_mult, BOOL icb) -> id<MTLComputePipelineState> {
      pipe_desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = tg_mult;
      if (@available(macOS 11.0, *)) {
        pipe_desc.supportIndirectCommandBuffers = icb;
      }
      return [impl_->device newComputePipelineStateWithDescriptor:pipe_desc
                                                          options:MTLPipelineOptionNone
                                                       reflection:nil
                                                            error:&err];
    };

    id<MTLComputePipelineState> pso = nil;
    if (impl_->supports_icb) {
      pso = try_create_pso(YES, YES);
      if (pso == nil) {
        pso = try_create_pso(NO, YES);
      }
    }
    if (pso == nil) {
      pso = try_create_pso(YES, NO);
    }
    if (pso == nil) {
      pso = try_create_pso(NO, NO);
    }
    if (pso == nil) {
      if (error_out) {
        *error_out = err ? [[err localizedDescription] UTF8String] : "Pipeline creation failed";
      }
      return 0;
    }

    bool pso_supports_icb = false;
    if (@available(macOS 11.0, *)) {
      pso_supports_icb = ([pso supportIndirectCommandBuffers] == YES);
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
    impl_->kernels.push_back(CompiledEntry{fn_name, pso, pso_supports_icb});
    impl_->key_to_id.emplace(cache_key, new_id);
    return new_id;
  }
}

uint64_t MetalRuntime::kernel_max_threads_per_threadgroup(uint64_t kernel_id) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (kernel_id == 0 || kernel_id >= impl_->kernels.size()) {
    return 0;
  }
  id<MTLComputePipelineState> pso = impl_->kernels[kernel_id].pso;
  if (pso == nil) {
    return 0;
  }
  return static_cast<uint64_t>([pso maxTotalThreadsPerThreadgroup]);
}

bool MetalRuntime::kernel_supports_icb(uint64_t kernel_id) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (kernel_id == 0 || kernel_id >= impl_->kernels.size()) {
    return false;
  }
  return impl_->kernels[kernel_id].supports_icb;
}

bool MetalRuntime::begin_batch(std::string* error_out) {
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->device == nil || impl_->queue == nil) {
      if (error_out) *error_out = "Metal runtime uninitialized";
      return false;
    }
    if (impl_->batch.active) {
      if (error_out) *error_out = "A Metal command batch is already active";
      return false;
    }
    if (impl_->graph_capture.active) {
      if (error_out) *error_out = "Cannot begin command batch while Metal graph capture is active";
      return false;
    }

    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    if (cb == nil) {
      if (error_out) *error_out = "Failed to create Metal command buffer for batch";
      return false;
    }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (enc == nil) {
      if (error_out) *error_out = "Failed to create Metal compute command encoder for batch";
      return false;
    }

    impl_->batch = BatchState{};
    impl_->batch.active = true;
    impl_->batch.cb = cb;
    impl_->batch.enc = enc;
    return true;
  }
}

double MetalRuntime::end_batch(bool wait, std::string* error_out) {
  @autoreleasepool {
    BatchState finished;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      if (!impl_->batch.active) {
        if (error_out) *error_out = "No active Metal command batch";
        return -1.0;
      }
      finished = std::move(impl_->batch);
      impl_->batch = BatchState{};
    }

    if (finished.enc != nil) {
      [finished.enc endEncoding];
    }
    if (finished.dispatch_count == 0) {
      return 0.0;
    }

    [finished.cb commit];
    const bool do_sync = wait || finished.needs_sync;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      impl_->last_cmd_buf = do_sync ? nil : finished.cb;
    }

    if (do_sync) {
      [finished.cb waitUntilCompleted];
      if ([finished.cb status] == MTLCommandBufferStatusError) {
        if (error_out) {
          NSError* err = [finished.cb error];
          *error_out = err ? [[err localizedDescription] UTF8String] : "Metal command batch error";
        }
        return -1.0;
      }
      for (const auto& cbk : finished.copy_backs) {
        std::memcpy(cbk.host_ptr, [cbk.mtl_buf contents], static_cast<size_t>(cbk.nbytes));
      }
      return command_buffer_gpu_ms(finished.cb);
    }
    return 0.0;
  }
}

void MetalRuntime::abort_batch() {
  @autoreleasepool {
    BatchState aborted;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      if (!impl_->batch.active) {
        return;
      }
      aborted = std::move(impl_->batch);
      impl_->batch = BatchState{};
    }
    if (aborted.enc != nil) {
      [aborted.enc endEncoding];
    }
  }
}

bool MetalRuntime::in_batch() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->batch.active;
}

bool MetalRuntime::begin_graph_capture(std::string* error_out) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->device == nil || impl_->queue == nil) {
    if (error_out) *error_out = "Metal runtime uninitialized";
    return false;
  }
  if (!impl_->supports_icb) {
    if (error_out) *error_out = "Metal device does not support Compute Indirect Command Buffers";
    return false;
  }
  if (impl_->graph_capture.active) {
    if (error_out) *error_out = "Metal graph capture is already active";
    return false;
  }
  if (impl_->batch.active) {
    if (error_out) *error_out = "Cannot begin Metal graph capture while a command batch is active";
    return false;
  }
  impl_->graph_capture = GraphCaptureState{};
  impl_->graph_capture.active = true;
  return true;
}

uint64_t MetalRuntime::end_graph_capture(std::string* error_out) {
  @autoreleasepool {
    std::vector<CapturedDispatch> dispatches;
    {
      std::lock_guard<std::mutex> lock(impl_->mu);
      if (!impl_->graph_capture.active) {
        if (error_out) *error_out = "No active Metal graph capture";
        return 0;
      }
      dispatches = std::move(impl_->graph_capture.dispatches);
      impl_->graph_capture = GraphCaptureState{};
    }

    if (dispatches.empty()) {
      if (error_out) *error_out = "Cannot finalize empty MetalGraph: no kernels were captured";
      return 0;
    }

    if (@available(macOS 11.0, *)) {
      NSUInteger max_bind = kDefaultIcbMaxBindCount;
      std::vector<uint8_t> scalar_blob;
      std::vector<std::vector<size_t>> scalar_offsets(dispatches.size());

      for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        for (const auto& b : d.buffers) {
          if (b.index + 1 > max_bind) {
            max_bind = static_cast<NSUInteger>(b.index + 1);
          }
        }
        scalar_offsets[i].resize(d.scalars.size(), 0);
        for (size_t si = 0; si < d.scalars.size(); ++si) {
          const auto& s = d.scalars[si];
          if (s.bytes.empty()) {
            continue;
          }
          if (s.index + 1 > max_bind) {
            max_bind = static_cast<NSUInteger>(s.index + 1);
          }
          const size_t offset = align_up(scalar_blob.size(), kScalarBufferAlign);
          const size_t padded_end = offset + align_up(s.bytes.size(), kScalarBufferAlign);
          scalar_blob.resize(padded_end, 0);
          std::memcpy(scalar_blob.data() + offset, s.bytes.data(), s.bytes.size());
          scalar_offsets[i][si] = offset;
        }
      }

      if (max_bind > kMetalMaxBufferBindings) {
        if (error_out) {
          *error_out = "MetalGraph buffer binding index exceeds Metal limit (" +
                       std::to_string(static_cast<uint64_t>(kMetalMaxBufferBindings)) + ")";
        }
        return 0;
      }

      id<MTLBuffer> scalar_buf = nil;
      if (!scalar_blob.empty()) {
        scalar_buf = [impl_->device newBufferWithBytes:scalar_blob.data()
                                                length:static_cast<NSUInteger>(scalar_blob.size())
                                               options:MTLResourceStorageModeShared];
        if (scalar_buf == nil) {
          if (error_out) *error_out = "Failed to allocate scalar constant buffer for MetalGraph";
          return 0;
        }
      }

      MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
      desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
      desc.inheritBuffers = NO;
      desc.inheritPipelineState = NO;
      desc.maxKernelBufferBindCount = max_bind;

      id<MTLIndirectCommandBuffer> icb =
          [impl_->device newIndirectCommandBufferWithDescriptor:desc
                                                maxCommandCount:static_cast<NSUInteger>(dispatches.size())
                                                        options:MTLResourceStorageModeShared];
      if (icb == nil) {
        if (error_out) *error_out = "Failed to allocate MTLIndirectCommandBuffer";
        return 0;
      }

      auto graph = std::make_shared<MetalGraphEntry>();
      graph->icb = icb;
      graph->scalar_buf = scalar_buf;
      graph->command_count = dispatches.size();
      graph->last_kernel = dispatches.back().fn_name;

      std::unordered_set<uintptr_t> seen_buffers;
      std::unordered_set<uint64_t> seen_kernels;

      for (size_t i = 0; i < dispatches.size(); ++i) {
        const auto& d = dispatches[i];
        if (seen_kernels.insert(d.kernel_id).second) {
          graph->retained_psos.push_back(d.pso);
        }

        id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:static_cast<NSUInteger>(i)];
        [cmd reset];
        [cmd setComputePipelineState:d.pso];

        for (size_t bi = 0; bi < d.buffers.size(); ++bi) {
          const auto& b = d.buffers[bi];
          id<MTLBuffer> buf = d.retained_buffers[bi];
          if (seen_buffers.insert(b.handle).second) {
            graph->retained_buffers.push_back(buf);
            graph->resource_ptrs.push_back(buf);
          }
          [cmd setKernelBuffer:buf
                        offset:static_cast<NSUInteger>(b.byte_offset)
                       atIndex:static_cast<NSUInteger>(b.index)];
        }

        for (size_t si = 0; si < d.scalars.size(); ++si) {
          const auto& s = d.scalars[si];
          if (!s.bytes.empty()) {
            [cmd setKernelBuffer:scalar_buf
                          offset:static_cast<NSUInteger>(scalar_offsets[i][si])
                         atIndex:static_cast<NSUInteger>(s.index)];
          }
        }

        if (d.cfg.threadgroup_mem_bytes > 0) {
          [cmd setThreadgroupMemoryLength:static_cast<NSUInteger>(d.cfg.threadgroup_mem_bytes) atIndex:0];
        }

        MTLSize grid = MTLSizeMake(d.cfg.grid_x, d.cfg.grid_y, d.cfg.grid_z);
        MTLSize tg = MTLSizeMake(d.cfg.tg_x, d.cfg.tg_y, d.cfg.tg_z);
        [cmd concurrentDispatchThreadgroups:grid threadsPerThreadgroup:tg];
      }

      size_t range_start = 0;
      std::vector<BufferArg> range_buffers;
      for (size_t i = 0; i < dispatches.size(); ++i) {
        bool has_hazard = false;
        for (const auto& nb : dispatches[i].buffers) {
          for (const auto& pb : range_buffers) {
            if (buffers_overlap(pb, nb) && (pb.writable || nb.writable)) {
              has_hazard = true;
              break;
            }
          }
          if (has_hazard) {
            break;
          }
        }
        if (has_hazard && i > range_start) {
          id<MTLIndirectComputeCommand> barrier_cmd =
              [icb indirectComputeCommandAtIndex:static_cast<NSUInteger>(i)];
          [barrier_cmd setBarrier];
          range_start = i;
          range_buffers.clear();
        }
        for (const auto& nb : dispatches[i].buffers) {
          range_buffers.push_back(nb);
        }
      }
      graph->exec_ranges.push_back(NSMakeRange(0, static_cast<NSUInteger>(dispatches.size())));

      id<MTLCommandBuffer> opt_cb = [impl_->queue commandBuffer];
      if (opt_cb != nil) {
        id<MTLBlitCommandEncoder> blit = [opt_cb blitCommandEncoder];
        if (blit != nil) {
          [blit optimizeIndirectCommandBuffer:icb
                                    withRange:NSMakeRange(0, static_cast<NSUInteger>(dispatches.size()))];
          [blit endEncoding];
          [opt_cb commit];
          [opt_cb waitUntilCompleted];
        }
      }

      std::lock_guard<std::mutex> lock(impl_->mu);
      const uint64_t gid = impl_->next_graph_id++;
      impl_->graphs.emplace(gid, std::move(graph));
      return gid;
    }

    if (error_out) *error_out = "Compute Indirect Command Buffers require macOS 11.0+";
    return 0;
  }
}

void MetalRuntime::abort_graph_capture() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->graph_capture = GraphCaptureState{};
}

bool MetalRuntime::in_graph_capture() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->graph_capture.active;
}

double MetalRuntime::replay_graph(uint64_t graph_id, bool wait, std::string* error_out) {
  std::shared_ptr<MetalGraphEntry> graph;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->device == nil || impl_->queue == nil) {
      if (error_out) *error_out = "Metal runtime uninitialized";
      return -1.0;
    }
    if (impl_->graph_capture.active) {
      if (error_out) *error_out = "Cannot replay MetalGraph while graph capture is active";
      return -1.0;
    }
    auto it = impl_->graphs.find(graph_id);
    if (it == impl_->graphs.end()) {
      if (error_out) *error_out = "Invalid or destroyed MetalGraph id";
      return -1.0;
    }
    graph = it->second;
  }

  @autoreleasepool {
    if (@available(macOS 11.0, *)) {
      id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
      if (cb == nil) {
        if (error_out) *error_out = "Failed to create Metal command buffer for graph replay";
        return -1.0;
      }
      id<MTLComputeCommandEncoder> enc =
          [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      if (enc == nil) {
        if (error_out) *error_out = "Failed to create concurrent compute encoder for graph replay";
        return -1.0;
      }

      if (!graph->resource_ptrs.empty()) {
        [enc useResources:graph->resource_ptrs.data()
                    count:static_cast<NSUInteger>(graph->resource_ptrs.size())
                    usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
      }
      if (graph->scalar_buf != nil) {
        [enc useResource:graph->scalar_buf usage:MTLResourceUsageRead];
      }

      for (size_t ri = 0; ri < graph->exec_ranges.size(); ++ri) {
        if (ri > 0) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc executeCommandsInBuffer:graph->icb withRange:graph->exec_ranges[ri]];
      }

      [enc endEncoding];
      [cb commit];

      impl_->dispatches.fetch_add(static_cast<uint64_t>(graph->command_count), std::memory_order_acq_rel);
      {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->last_kernel = graph->last_kernel;
        impl_->last_cmd_buf = wait ? nil : cb;
      }

      if (wait) {
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) {
          if (error_out) {
            NSError* err = [cb error];
            *error_out = err ? [[err localizedDescription] UTF8String] : "MetalGraph replay command buffer error";
          }
          return -1.0;
        }
        return command_buffer_gpu_ms(cb);
      }
      return 0.0;
    }

    if (error_out) *error_out = "Compute Indirect Command Buffers require macOS 11.0+";
    return -1.0;
  }
}

void MetalRuntime::destroy_graph(uint64_t graph_id) {
  std::shared_ptr<MetalGraphEntry> doomed;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->graphs.find(graph_id);
    if (it != impl_->graphs.end()) {
      doomed = std::move(it->second);
      impl_->graphs.erase(it);
    }
  }
}

size_t MetalRuntime::graph_command_count(uint64_t graph_id) const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it = impl_->graphs.find(graph_id);
  if (it == impl_->graphs.end()) {
    return 0;
  }
  return it->second->command_count;
}

double MetalRuntime::launch_kernel(
    uint64_t kernel_id,
    const std::vector<BufferArg>& buffers,
    const std::vector<ScalarArg>& scalars,
    const LaunchConfig& cfg,
    std::string* error_out) {
  CompiledEntry entry;
  bool capturing_graph = false;
  bool batching = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (kernel_id == 0 || kernel_id >= impl_->kernels.size()) {
      if (error_out) *error_out = "Invalid compiled Metal kernel id";
      return -1.0;
    }
    entry = impl_->kernels[kernel_id];
    capturing_graph = impl_->graph_capture.active;
    batching = impl_->batch.active;
  }

  @autoreleasepool {
    if (impl_->device == nil || impl_->queue == nil) {
      if (error_out) *error_out = "Metal runtime uninitialized";
      return -1.0;
    }

    const uint64_t tg_threads = static_cast<uint64_t>(cfg.tg_x) * cfg.tg_y * cfg.tg_z;
    const NSUInteger max_tg_threads = [entry.pso maxTotalThreadsPerThreadgroup];
    if (tg_threads == 0 || tg_threads > static_cast<uint64_t>(max_tg_threads)) {
      if (error_out) {
        *error_out = "Threadgroup size (" + std::to_string(tg_threads) +
                     ") exceeds PSO maxTotalThreadsPerThreadgroup (" +
                     std::to_string(static_cast<uint64_t>(max_tg_threads)) + ") for kernel " +
                     entry.fn_name;
      }
      return -1.0;
    }

    if (capturing_graph) {
      if (cfg.wait) {
        if (error_out) {
          *error_out = "launch_kernel(wait=True) is not supported during MetalGraph capture";
        }
        return -1.0;
      }
      if (!entry.supports_icb) {
        if (error_out) {
          *error_out = "Kernel '" + entry.fn_name + "' pipeline does not support Metal Indirect Command Buffers";
        }
        return -1.0;
      }

      CapturedDispatch rec;
      rec.kernel_id = kernel_id;
      rec.fn_name = entry.fn_name;
      rec.pso = entry.pso;
      rec.buffers = buffers;
      rec.retained_buffers.reserve(buffers.size());
      for (const auto& b : buffers) {
        if (!b.is_metal_buffer || b.handle == 0) {
          if (error_out) {
            *error_out = "MetalGraph capture requires Metal GPU buffers (MPS tensors); non-Metal buffer at index " +
                         std::to_string(b.index) + " is not supported";
          }
          return -1.0;
        }
        if (b.index >= kMetalMaxBufferBindings) {
          if (error_out) {
            *error_out = "MetalGraph buffer index " + std::to_string(b.index) + " exceeds Metal limit";
          }
          return -1.0;
        }
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)b.handle;
        rec.retained_buffers.push_back(buf);
      }
      for (const auto& s : scalars) {
        if (!s.bytes.empty() && s.index >= kMetalMaxBufferBindings) {
          if (error_out) {
            *error_out = "MetalGraph scalar index " + std::to_string(s.index) + " exceeds Metal limit";
          }
          return -1.0;
        }
      }
      rec.scalars = scalars;
      rec.cfg = cfg;

      std::lock_guard<std::mutex> lock(impl_->mu);
      if (!impl_->graph_capture.active) {
        if (error_out) *error_out = "MetalGraph capture ended concurrently";
        return -1.0;
      }
      impl_->graph_capture.dispatches.push_back(std::move(rec));
      return 0.0;
    }

    if (batching) {
      if (cfg.wait) {
        if (error_out) {
          *error_out =
              "launch_kernel(wait=True) is not permitted inside an active command_batch; use end_batch(wait=True)";
        }
        return -1.0;
      }

      std::lock_guard<std::mutex> lock(impl_->mu);
      if (!impl_->batch.active || impl_->batch.enc == nil) {
        if (error_out) *error_out = "Command batch ended concurrently";
        return -1.0;
      }

      id<MTLComputeCommandEncoder> enc = impl_->batch.enc;
      [enc setComputePipelineState:entry.pso];

      for (const auto& b : buffers) {
        if (b.is_metal_buffer) {
          id<MTLBuffer> buf = (__bridge id<MTLBuffer>)(void*)b.handle;
          [enc setBuffer:buf offset:static_cast<NSUInteger>(b.byte_offset) atIndex:b.index];
        } else {
          uint8_t* raw_ptr = reinterpret_cast<uint8_t*>(b.handle) + b.byte_offset;
          const uint64_t alloc_bytes = b.nbytes > 0 ? b.nbytes : 4;
          const CpuBufKey cache_key{reinterpret_cast<uintptr_t>(raw_ptr), alloc_bytes};
          id<MTLBuffer> buf = nil;
          auto it = impl_->batch.cpu_buf_cache.find(cache_key);
          if (it != impl_->batch.cpu_buf_cache.end()) {
            buf = it->second;
            if (b.writable && !is_page_aligned(reinterpret_cast<uintptr_t>(raw_ptr), alloc_bytes)) {
              if (impl_->batch.copy_back_ptrs.insert(reinterpret_cast<uintptr_t>(raw_ptr)).second) {
                impl_->batch.copy_backs.push_back(CpuCopyBack{raw_ptr, buf, b.nbytes});
              }
              impl_->batch.needs_sync = true;
            }
          } else if (is_page_aligned(reinterpret_cast<uintptr_t>(raw_ptr), alloc_bytes)) {
            buf = [impl_->device newBufferWithBytesNoCopy:raw_ptr
                                                   length:static_cast<NSUInteger>(alloc_bytes)
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
            impl_->batch.temp_buffers.push_back(buf);
            impl_->batch.cpu_buf_cache.emplace(cache_key, buf);
            impl_->batch.needs_sync = true;
          } else {
            buf = [impl_->device newBufferWithBytes:raw_ptr
                                             length:static_cast<NSUInteger>(alloc_bytes)
                                            options:MTLResourceStorageModeShared];
            impl_->batch.temp_buffers.push_back(buf);
            impl_->batch.cpu_buf_cache.emplace(cache_key, buf);
            if (b.writable) {
              if (impl_->batch.copy_back_ptrs.insert(reinterpret_cast<uintptr_t>(raw_ptr)).second) {
                impl_->batch.copy_backs.push_back(CpuCopyBack{raw_ptr, buf, b.nbytes});
              }
              impl_->batch.needs_sync = true;
            }
          }
          [enc setBuffer:buf offset:0 atIndex:b.index];
        }
      }

      for (const auto& s : scalars) {
        if (!s.bytes.empty()) {
          [enc setBytes:s.bytes.data() length:s.bytes.size() atIndex:s.index];
        }
      }

      if (cfg.threadgroup_mem_bytes > 0) {
        [enc setThreadgroupMemoryLength:cfg.threadgroup_mem_bytes atIndex:0];
        impl_->batch.last_tg_mem_bytes = cfg.threadgroup_mem_bytes;
      } else if (impl_->batch.last_tg_mem_bytes > 0) {
        [enc setThreadgroupMemoryLength:0 atIndex:0];
        impl_->batch.last_tg_mem_bytes = 0;
      }

      MTLSize grid = MTLSizeMake(cfg.grid_x, cfg.grid_y, cfg.grid_z);
      MTLSize tg = MTLSizeMake(cfg.tg_x, cfg.tg_y, cfg.tg_z);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

      impl_->batch.dispatch_count += 1;
      impl_->dispatches.fetch_add(1, std::memory_order_acq_rel);
      impl_->last_kernel = entry.fn_name;
      return 0.0;
    }

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
      return command_buffer_gpu_ms(cb);
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

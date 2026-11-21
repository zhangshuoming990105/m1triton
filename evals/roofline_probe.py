#!/usr/bin/env python3
"""Independent M1 Max hardware roofline probe (STREAM bandwidth + SGMM FP32/FP16 TFLOPS).

Compiles and runs minimal native Metal compute shaders via Objective-C++
MTLDevice::newLibraryWithSource (no xcrun metal CLI required) to measure real
empirical memory bandwidth and simdgroup_matrix_multiply_accumulate ceilings.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

OBJCXX_PROBE_SOURCE = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static const char* kProbeMSL = R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void stream_triad_f4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* c [[buffer(2)]],
    constant uint& n4 [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n4) {
        c[gid] = a[gid] + 2.5f * b[gid];
    }
}

kernel void stream_copy_f4(
    device const float4* src [[buffer(0)]],
    device float4* dst [[buffer(1)]],
    constant uint& n4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < n4) {
        dst[gid] = src[gid];
    }
}

// Compute-bound SGMM FP32 register throughput microbenchmark:
// Each SIMD-group runs 4x4 grid of 8x8 simdgroup_float8x8 accumulators (32x32 tile)
// in registers for `iters` outer steps -> 4*4 * (2*8*8*8) = 16384 FLOPs per inner iter per SIMD-group.
kernel void sgmm_peak_fp32(
    device float* out [[buffer(0)]],
    constant uint& iters [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem_a[64];
    threadgroup float smem_b[64];
    if (tid < 64) {
        smem_a[tid] = 0.01f + 0.0001f * float((tid + tg_id) & 15);
        smem_b[tid] = 0.01f + 0.0001f * float((tid * 3 + tg_id) & 15);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 a0, a1, a2, a3;
    simdgroup_float8x8 b0, b1, b2, b3;
    simdgroup_load(a0, smem_a, 8);
    simdgroup_load(a1, smem_a, 8);
    simdgroup_load(a2, smem_a, 8);
    simdgroup_load(a3, smem_a, 8);
    simdgroup_load(b0, smem_b, 8);
    simdgroup_load(b1, smem_b, 8);
    simdgroup_load(b2, smem_b, 8);
    simdgroup_load(b3, smem_b, 8);

    simdgroup_float8x8 c00(0.0f), c01(0.0f), c02(0.0f), c03(0.0f);
    simdgroup_float8x8 c10(0.0f), c11(0.0f), c12(0.0f), c13(0.0f);
    simdgroup_float8x8 c20(0.0f), c21(0.0f), c22(0.0f), c23(0.0f);
    simdgroup_float8x8 c30(0.0f), c31(0.0f), c32(0.0f), c33(0.0f);

    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c02, a0, b2, c02);
        simdgroup_multiply_accumulate(c03, a0, b3, c03);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
        simdgroup_multiply_accumulate(c12, a1, b2, c12);
        simdgroup_multiply_accumulate(c13, a1, b3, c13);
        simdgroup_multiply_accumulate(c20, a2, b0, c20);
        simdgroup_multiply_accumulate(c21, a2, b1, c21);
        simdgroup_multiply_accumulate(c22, a2, b2, c22);
        simdgroup_multiply_accumulate(c23, a2, b3, c23);
        simdgroup_multiply_accumulate(c30, a3, b0, c30);
        simdgroup_multiply_accumulate(c31, a3, b1, c31);
        simdgroup_multiply_accumulate(c32, a3, b2, c32);
        simdgroup_multiply_accumulate(c33, a3, b3, c33);
    }

    threadgroup float scratch[4 * 64];
    float sum_val = 0.0f;
    simdgroup_float8x8 tiles[16] = {
        c00, c01, c02, c03, c10, c11, c12, c13,
        c20, c21, c22, c23, c30, c31, c32, c33
    };
    for (uint t = 0; t < 16; ++t) {
        simdgroup_store(tiles[t], scratch + simd_id * 64, 8);
        sum_val += scratch[simd_id * 64 + (t & 63)];
    }
    threadgroup float tg_red[4];
    if ((tid & 31) == 0) {
        tg_red[simd_id] = sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        out[tg_id] = tg_red[0] + tg_red[1] + tg_red[2] + tg_red[3];
    }
}

kernel void sgmm_peak_fp16(
    device float* out [[buffer(0)]],
    constant uint& iters [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    threadgroup half smem_a[64];
    threadgroup half smem_b[64];
    if (tid < 64) {
        smem_a[tid] = half(0.01f + 0.0001f * float((tid + tg_id) & 15));
        smem_b[tid] = half(0.01f + 0.0001f * float((tid * 3 + tg_id) & 15));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 a0, a1, a2, a3;
    simdgroup_half8x8 b0, b1, b2, b3;
    simdgroup_load(a0, smem_a, 8);
    simdgroup_load(a1, smem_a, 8);
    simdgroup_load(a2, smem_a, 8);
    simdgroup_load(a3, smem_a, 8);
    simdgroup_load(b0, smem_b, 8);
    simdgroup_load(b1, smem_b, 8);
    simdgroup_load(b2, smem_b, 8);
    simdgroup_load(b3, smem_b, 8);

    simdgroup_half8x8 c00(0.0h), c01(0.0h), c02(0.0h), c03(0.0h);
    simdgroup_half8x8 c10(0.0h), c11(0.0h), c12(0.0h), c13(0.0h);
    simdgroup_half8x8 c20(0.0h), c21(0.0h), c22(0.0h), c23(0.0h);
    simdgroup_half8x8 c30(0.0h), c31(0.0h), c32(0.0h), c33(0.0h);

    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c02, a0, b2, c02);
        simdgroup_multiply_accumulate(c03, a0, b3, c03);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
        simdgroup_multiply_accumulate(c12, a1, b2, c12);
        simdgroup_multiply_accumulate(c13, a1, b3, c13);
        simdgroup_multiply_accumulate(c20, a2, b0, c20);
        simdgroup_multiply_accumulate(c21, a2, b1, c21);
        simdgroup_multiply_accumulate(c22, a2, b2, c22);
        simdgroup_multiply_accumulate(c23, a2, b3, c23);
        simdgroup_multiply_accumulate(c30, a3, b0, c30);
        simdgroup_multiply_accumulate(c31, a3, b1, c31);
        simdgroup_multiply_accumulate(c32, a3, b2, c32);
        simdgroup_multiply_accumulate(c33, a3, b3, c33);
    }

    threadgroup half scratch[4 * 64];
    float sum_val = 0.0f;
    simdgroup_half8x8 tiles[16] = {
        c00, c01, c02, c03, c10, c11, c12, c13,
        c20, c21, c22, c23, c30, c31, c32, c33
    };
    for (uint t = 0; t < 16; ++t) {
        simdgroup_store(tiles[t], scratch + simd_id * 64, 8);
        sum_val += float(scratch[simd_id * 64 + (t & 63)]);
    }
    threadgroup float tg_red[4];
    if ((tid & 31) == 0) {
        tg_red[simd_id] = sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        out[tg_id] = tg_red[0] + tg_red[1] + tg_red[2] + tg_red[3];
    }
}
)MSL";

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double iqr(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n < 4) return 0.0;
    return v[(3 * n) / 4] - v[n / 4];
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "ERROR: No Metal device available\n";
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kProbeMSL]
                                                  options:opts
                                                    error:&err];
        if (!lib) {
            std::cerr << "ERROR: Metal compile failed: " << [[err localizedDescription] UTF8String] << "\n";
            return 1;
        }

        auto makePipeline = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            NSError* perr = nil;
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&perr];
            if (!pso) {
                std::cerr << "ERROR: Pipeline failed for " << name << ": "
                          << [[perr localizedDescription] UTF8String] << "\n";
                std::exit(1);
            }
            return pso;
        };

        id<MTLComputePipelineState> psoCopy = makePipeline("stream_copy_f4");
        id<MTLComputePipelineState> psoTriad = makePipeline("stream_triad_f4");
        id<MTLComputePipelineState> psoFp32 = makePipeline("sgmm_peak_fp32");
        id<MTLComputePipelineState> psoFp16 = makePipeline("sgmm_peak_fp16");

        // 64M float32 = 256 MB per buffer (exceeds M1 Max 48 MB SLC so measures real DRAM bandwidth)
        const uint32_t nFloats = 64u * 1024u * 1024u;
        const uint32_t n4 = nFloats / 4u;
        const size_t nBytes = size_t(nFloats) * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:nBytes options:MTLResourceStorageModeShared];
        float* ptrA = static_cast<float*>([bufA contents]);
        float* ptrB = static_cast<float*>([bufB contents]);
        for (uint32_t i = 0; i < nFloats; i += 4096) {
            ptrA[i] = 1.0f;
            ptrB[i] = 2.0f;
        }

        const int warmup = 5;
        const int trials = 21;

        auto benchCopy = [&]() {
            std::vector<double> sec;
            for (int r = 0; r < warmup + trials; ++r) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:psoCopy];
                [enc setBuffer:bufA offset:0 atIndex:0];
                [enc setBuffer:bufC offset:0 atIndex:1];
                [enc setBytes:&n4 length:sizeof(n4) atIndex:2];
                MTLSize grid = MTLSizeMake(n4, 1, 1);
                MTLSize tg = MTLSizeMake(256, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double dt = [cb GPUEndTime] - [cb GPUStartTime];
                if (r >= warmup) sec.push_back(dt);
            }
            return sec;
        };

        auto benchTriad = [&]() {
            std::vector<double> sec;
            for (int r = 0; r < warmup + trials; ++r) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:psoTriad];
                [enc setBuffer:bufA offset:0 atIndex:0];
                [enc setBuffer:bufB offset:0 atIndex:1];
                [enc setBuffer:bufC offset:0 atIndex:2];
                [enc setBytes:&n4 length:sizeof(n4) atIndex:3];
                MTLSize grid = MTLSizeMake(n4, 1, 1);
                MTLSize tg = MTLSizeMake(256, 1, 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double dt = [cb GPUEndTime] - [cb GPUStartTime];
                if (r >= warmup) sec.push_back(dt);
            }
            return sec;
        };

        // Verify triad work happened
        auto copyTimes = benchCopy();
        auto triadTimes = benchTriad();
        float* ptrC = static_cast<float*>([bufC contents]);
        bool triadOk = std::fabs(ptrC[0] - 6.0f) < 1e-4f && std::fabs(ptrC[4096] - 6.0f) < 1e-4f;

        // SGMM compute probes
        const uint32_t numTGs = 2048;
        const uint32_t simdsPerTG = 4;
        const uint32_t threadsPerTG = simdsPerTG * 32;
        const uint32_t iters32 = 2048;
        const uint32_t iters16 = 4096;
        id<MTLBuffer> bufOut = [device newBufferWithLength:numTGs * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        auto benchSgmm = [&](id<MTLComputePipelineState> pso, uint32_t iters) {
            std::vector<double> sec;
            for (int r = 0; r < warmup + trials; ++r) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bufOut offset:0 atIndex:0];
                [enc setBytes:&iters length:sizeof(iters) atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(numTGs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadsPerTG, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double dt = [cb GPUEndTime] - [cb GPUStartTime];
                if (r >= warmup) sec.push_back(dt);
            }
            return sec;
        };

        auto fp32Times = benchSgmm(psoFp32, iters32);
        float fp32Sample = static_cast<float*>([bufOut contents])[0];
        auto fp16Times = benchSgmm(psoFp16, iters16);
        float fp16Sample = static_cast<float*>([bufOut contents])[0];

        double copyMed = median(copyTimes);
        double triadMed = median(triadTimes);
        double fp32Med = median(fp32Times);
        double fp16Med = median(fp16Times);

        double copyBytes = 2.0 * double(nBytes);
        double triadBytes = 3.0 * double(nBytes);
        double copyGBs = (copyBytes / copyMed) / 1e9;
        double triadGBs = (triadBytes / triadMed) / 1e9;

        // 16 SGMM ops per iter * 2*8*8*8 (=1024) FLOPs per SGMM * (numTGs * simdsPerTG)
        double fp32Flops = double(numTGs) * double(simdsPerTG) * double(iters32) * 16.0 * 1024.0;
        double fp16Flops = double(numTGs) * double(simdsPerTG) * double(iters16) * 16.0 * 1024.0;
        double fp32Tflops = (fp32Flops / fp32Med) / 1e12;
        double fp16Tflops = (fp16Flops / fp16Med) / 1e12;

        std::printf(
            "{\n"
            "  \"device_name\": \"%s\",\n"
            "  \"max_threads_per_threadgroup\": %lu,\n"
            "  \"max_threadgroup_memory_bytes\": %lu,\n"
            "  \"recommended_max_working_set_gb\": %.2f,\n"
            "  \"has_unified_memory\": %s,\n"
            "  \"trials\": %d,\n"
            "  \"stream_copy\": {\"bytes\": %.0f, \"median_ms\": %.4f, \"iqr_ms\": %.4f, \"bandwidth_gbs\": %.2f},\n"
            "  \"stream_triad\": {\"bytes\": %.0f, \"median_ms\": %.4f, \"iqr_ms\": %.4f, \"bandwidth_gbs\": %.2f, \"verified\": %s},\n"
            "  \"sgmm_fp32\": {\"flops\": %.0f, \"median_ms\": %.4f, \"iqr_ms\": %.4f, \"tflops\": %.3f, \"sample_out\": %.3f},\n"
            "  \"sgmm_fp16\": {\"flops\": %.0f, \"median_ms\": %.4f, \"iqr_ms\": %.4f, \"tflops\": %.3f, \"sample_out\": %.3f}\n"
            "}\n",
            [[device name] UTF8String],
            (unsigned long)[device maxThreadsPerThreadgroup].width,
            (unsigned long)[device maxThreadgroupMemoryLength],
            double([device recommendedMaxWorkingSetSize]) / (1024.0 * 1024.0 * 1024.0),
            [device hasUnifiedMemory] ? "true" : "false",
            trials,
            copyBytes, copyMed * 1e3, iqr(copyTimes) * 1e3, copyGBs,
            triadBytes, triadMed * 1e3, iqr(triadTimes) * 1e3, triadGBs, triadOk ? "true" : "false",
            fp32Flops, fp32Med * 1e3, iqr(fp32Times) * 1e3, fp32Tflops, fp32Sample,
            fp16Flops, fp16Med * 1e3, iqr(fp16Times) * 1e3, fp16Tflops, fp16Sample
        );
    }
    return 0;
}
'''


def run_probe() -> dict:
    cxx = "/usr/bin/clang++" if Path("/usr/bin/clang++").exists() else "clang++"
    with tempfile.TemporaryDirectory(prefix="m1triton_roofline_") as tmp:
        src = Path(tmp) / "probe.mm"
        exe = Path(tmp) / "probe"
        src.write_text(OBJCXX_PROBE_SOURCE, encoding="utf-8")
        subprocess.run(
            [
                cxx,
                "-std=c++20",
                "-O3",
                "-fobjc-arc",
                str(src),
                "-o",
                str(exe),
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
            ],
            check=True,
        )
        out = subprocess.check_output([str(exe)], text=True)
    result = json.loads(out)
    result["host_platform"] = platform.platform()
    result["host_machine"] = platform.machine()
    result["loadavg_1m"] = round(os.getloadavg()[0], 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure M1 Max Metal roofline ceilings.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = run_probe()
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

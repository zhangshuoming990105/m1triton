#!/usr/bin/env python3
"""Generate a self-contained interactive HTML performance & Co-design report from frozen .audit artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".audit"
OUT_HTML = ROOT / "docs" / "PERFORMANCE_REPORT.html"


def _geomean(values: list[float]) -> float:
    pos = [v for v in values if v > 0.0]
    if not pos:
        return 0.0
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def _cat_l1(task: dict[str, Any]) -> str:
    tid = int(task["id"])
    name = str(task["name"]).lower()
    if tid <= 18:
        return "GEMM / Matmul"
    if 19 <= tid <= 32 or tid == 88:
        return "Activation & Softmax"
    if 33 <= tid <= 40:
        return "Normalization"
    if 41 <= tid <= 46:
        return "Pooling"
    if 47 <= tid <= 53 and tid != 50:
        return "Axis Reduction"
    if "conv_transposed" in name:
        return "ConvTranspose (1D/2D/3D)"
    if "conv" in name:
        return "Conv Standard / DW"
    if 89 <= tid <= 93:
        return "Prefix Scan"
    if tid == 97:
        return "Attention (SDPA)"
    return "Loss Functions"


def _cat_l2(task: dict[str, Any]) -> str:
    name = str(task["name"])
    if name.startswith("ConvTranspose3d"):
        return "ConvTranspose3d Fusion"
    if name.startswith("ConvTranspose2d"):
        return "ConvTranspose2d Fusion"
    if name.startswith("Conv3d"):
        return "Conv3d Fusion"
    if name.startswith("Conv2d") or name.startswith("Conv2D"):
        return "Conv2d Fusion"
    return "GEMM / Matmul / BMM Fusion"


def _cat_l3(task: dict[str, Any]) -> str:
    name = str(task["name"]).lower()
    if "attn" in name:
        return "Attention / Decode"
    if "norm" in name or "rope" in name:
        return "Norm / RoPE / KV"
    if "cross_entropy" in name:
        return "Fused Loss + Linear"
    if "mamba" in name:
        return "SSM / Parallel Scan"
    return "MoE / Quantized GEMM"


def _max_seed_err(task: dict[str, Any], is_l3: bool) -> tuple[float, float | None]:
    max_fwd = 0.0
    max_bwd: float | None = None
    for sr in task.get("seed_reports", []):
        if is_l3:
            for chk in sr.get("checks", []):
                err = float(chk.get("max_abs_err", 0.0))
                lbl = str(chk.get("label", ""))
                if lbl.startswith("fwd"):
                    max_fwd = max(max_fwd, err)
                else:
                    max_bwd = err if max_bwd is None else max(max_bwd, err)
        else:
            err = float(sr.get("max_abs_err", 0.0))
            max_fwd = max(max_fwd, err)
    return max_fwd, max_bwd


def _attribution_bucket(suite: str, task: dict[str, Any]) -> str:
    tid = int(task["id"])
    name = str(task["name"]).lower()
    sp = float(task["speedup"])
    if suite == "L1":
        if tid in (12, 14, 15):
            return "B"
        if tid in (46, 100) or (tid in (91, 92, 93, 94, 98, 99)):
            return "D"
        if ("conv_transposed_3d" in name or "conv_standard_3d" in name or "conv_standard_1d" in name) and sp >= 2.0:
            return "C"
        if tid in (41, 44):
            return "C"
        return "A"
    if suite == "L2":
        if tid in (42, 77):
            return "B"
        if ("convtranspose3d" in name or "conv3d" in name) and sp >= 3.0:
            return "C"
        return "A"
    if tid in (3, 4, 5, 8):
        return "D"
    return "A"


def _build_roofline_points(
    l1_map: dict[int, dict[str, Any]],
    l2_map: dict[int, dict[str, Any]],
    l3_map: dict[int, dict[str, Any]],
    bw_peak: float,
    fp32_peak: float,
    fp16_peak: float,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "key": "L1 #25",
            "suite": "L1",
            "id": 25,
            "label": "L1 #25 Swish",
            "shape": "4096 × 393216 FP32",
            "flops": 4.0 * 4096 * 393216,
            "bytes": 2.0 * 4096 * 393216 * 4,
            "kind": "bw",
            "bucket": "A",
            "peak": bw_peak,
        },
        {
            "key": "L1 #89",
            "suite": "L1",
            "id": 89,
            "label": "L1 #89 Cumsum",
            "shape": "32768 × 32768 FP32",
            "flops": 1.0 * 32768 * 32768,
            "bytes": 2.0 * 32768 * 32768 * 4,
            "kind": "bw",
            "bucket": "A",
            "peak": bw_peak,
        },
        {
            "key": "L1 #88",
            "suite": "L1",
            "id": 88,
            "label": "L1 #88 MinGPTNewGelu",
            "shape": "8192 × 8192 FP32",
            "flops": 12.0 * 8192 * 8192,
            "bytes": 2.0 * 8192 * 8192 * 4,
            "kind": "bw",
            "bucket": "A",
            "peak": bw_peak,
        },
        {
            "key": "L1 #36",
            "suite": "L1",
            "id": 36,
            "label": "L1 #36 RMSNorm (瓶颈)",
            "shape": "112 × 64 × 512 × 512 FP32",
            "flops": 3.0 * 112 * 64 * 512 * 512,
            "bytes": 2.0 * 112 * 64 * 512 * 512 * 4,
            "kind": "bw",
            "bucket": "H",
            "peak": bw_peak,
        },
        {
            "key": "L2 #42",
            "suite": "L2",
            "id": 42,
            "label": "L2 #42 ConvT2d+GAP+...",
            "shape": "16 × 64→128 × 512² (最小代数I/O)",
            "flops": 2.0 * 16 * 64 * 512 * 512,
            "bytes": 16 * 64 * 512 * 512 * 4.0,
            "kind": "bw",
            "bucket": "B",
            "peak": bw_peak,
        },
        {
            "key": "L1 #7",
            "suite": "L1",
            "id": 7,
            "label": "L1 #7 Small-K Matmul (瓶颈)",
            "shape": "32768 × 64 × 32768 FP32",
            "flops": 2.0 * 32768 * 32768 * 64,
            "bytes": (32768 * 64 * 2 + 32768 * 32768) * 4.0,
            "kind": "compute",
            "bucket": "H",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #72",
            "suite": "L1",
            "id": 72,
            "label": "L1 #72 Grouped ConvT3d",
            "shape": "8×32→32 G=4 K=3×5×7 s=2",
            "flops": 2.0 * 8 * 32 * 8 * 12 * 24 * 48 * 3 * 5 * 7,
            "bytes": (8 * 32 * 12 * 24 * 48 + 32 * 8 * 3 * 5 * 7 + 8 * 32 * 24 * 48 * 96) * 4.0,
            "kind": "compute",
            "bucket": "C",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #6",
            "suite": "L1",
            "id": 6,
            "label": "L1 #6 Large-K Split-K",
            "shape": "256 × 524288 × 256 FP32",
            "flops": 2.0 * 256 * 256 * 524288,
            "bytes": (256 * 524288 * 2 + 256 * 256) * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp32_peak,
        },
        {
            "key": "L2 #27",
            "suite": "L2",
            "id": 27,
            "label": "L2 #27 Conv3d+HSwish+GN+Mean",
            "shape": "1024×3→16 16×32×32 K=4",
            "flops": 2.0 * 1024 * 16 * 13 * 29 * 29 * 3 * 64,
            "bytes": (1024 * 3 * 16 * 32 * 32 + 1024 * 16 * 13 * 29 * 29) * 4.0,
            "kind": "compute",
            "bucket": "C",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #97",
            "suite": "L1",
            "id": 97,
            "label": "L1 #97 FlashAttention",
            "shape": "32×32×512×1024 FP32",
            "flops": 4.0 * 32 * 32 * 512 * 512 * 1024,
            "bytes": 4.0 * 32 * 32 * 512 * 1024 * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #66",
            "suite": "L1",
            "id": 66,
            "label": "L1 #66 Conv3d Asym",
            "shape": "8×3→64 16×128² K=3×5×7",
            "flops": 2.0 * 8 * 64 * 14 * 124 * 122 * 3 * 3 * 5 * 7,
            "bytes": (8 * 3 * 16 * 128 * 128 + 64 * 3 * 3 * 5 * 7 + 8 * 64 * 14 * 124 * 122) * 4.0,
            "kind": "compute",
            "bucket": "C",
            "peak": fp32_peak,
        },
        {
            "key": "L3-01",
            "suite": "L3",
            "id": 1,
            "label": "L3-01 FA2 GQA Causal",
            "shape": "B=4 Hq=16 Hkv=4 L=1024 D=64",
            "flops": 0.5 * 4.0 * 4 * 16 * 1024 * 1024 * 64,
            "bytes": (4 * 16 * 1024 * 64 * 2 + 4 * 4 * 1024 * 64 * 2) * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #15",
            "suite": "L1",
            "id": 15,
            "label": "L1 #15 LowerTri (精确1/3N³)",
            "shape": "4096 × 4096 LowerTri FP32",
            "flops": (1.0 / 3.0) * (4096.0**3),
            "bytes": 1.5 * 4096 * 4096 * 4.0,
            "kind": "compute",
            "bucket": "B",
            "peak": fp32_peak,
        },
        {
            "key": "L3-07",
            "suite": "L3",
            "id": 7,
            "label": "L3-07 Chunked Linear+CE (Fwd+Bwd)",
            "shape": "BT=2048 H=1024 V=32768 (3×SGMM)",
            "flops": 6.0 * 2048 * 1024 * 32768,
            "bytes": (2048 * 1024 * 4 * 2 + 32768 * 1024 * 4 * 2 + 2048 * 32768 * 4 * 2),
            "kind": "compute",
            "bucket": "H",
            "peak": fp32_peak,
        },
        {
            "key": "L1 #18",
            "suite": "L1",
            "id": 18,
            "label": "L1 #18 Transposed-AB Matmul",
            "shape": "2048 × 8192 × 4096 FP32",
            "flops": 2.0 * 2048 * 4096 * 8192,
            "bytes": (2048 * 8192 + 8192 * 4096 + 2048 * 4096) * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp32_peak,
        },
        {
            "key": "L3-10",
            "suite": "L3",
            "id": 10,
            "label": "L3-10 W4A16 Dequant GEMM",
            "shape": "256 × 4096 × 4096 (W4/FP16)",
            "flops": 2.0 * 256 * 4096 * 4096,
            "bytes": 256 * 4096 * 2.0 * 2 + 4096 * 4096 * 0.5 + 32 * 4096 * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp16_peak,
        },
        {
            "key": "L1 #1",
            "suite": "L1",
            "id": 1,
            "label": "L1 #1 Square Matmul",
            "shape": "4096 × 4096 × 4096 FP32",
            "flops": 2.0 * (4096.0**3),
            "bytes": 3.0 * 4096 * 4096 * 4.0,
            "kind": "compute",
            "bucket": "A",
            "peak": fp32_peak,
        },
    ]

    points: list[dict[str, Any]] = []
    for sp in specs:
        suite = sp["suite"]
        tid = sp["id"]
        src = (l1_map if suite == "L1" else (l2_map if suite == "L2" else l3_map))[tid]
        cand_s = float(src["candidate_median_ms"]) / 1000.0
        base_s = float(src["baseline_median_ms"]) / 1000.0
        ai = sp["flops"] / sp["bytes"]
        cand_tf = sp["flops"] / cand_s / 1e12
        base_tf = sp["flops"] / base_s / 1e12
        cand_gbs = sp["bytes"] / cand_s / 1e9
        base_gbs = sp["bytes"] / base_s / 1e9
        if sp["kind"] == "bw":
            sol_pct = 100.0 * cand_gbs / sp["peak"]
            base_sol_pct = 100.0 * base_gbs / sp["peak"]
        else:
            roof_tf = min(sp["peak"], ai * bw_peak / 1000.0)
            sol_pct = 100.0 * cand_tf / roof_tf
            base_sol_pct = 100.0 * base_tf / roof_tf
        points.append(
            {
                "key": sp["key"],
                "suite": suite,
                "id": tid,
                "label": sp["label"],
                "shape": sp["shape"],
                "kind": sp["kind"],
                "bucket": sp["bucket"],
                "ai": round(ai, 2),
                "cand_tf": round(cand_tf, 3),
                "base_tf": round(base_tf, 3),
                "cand_gbs": round(cand_gbs, 1),
                "base_gbs": round(base_gbs, 1),
                "cand_ms": round(float(src["candidate_median_ms"]), 3),
                "base_ms": round(float(src["baseline_median_ms"]), 3),
                "speedup": round(float(src["speedup"]), 2),
                "sol_pct": round(sol_pct, 1),
                "base_sol_pct": round(base_sol_pct, 1),
            }
        )
    return points


def _build_deep_dives(
    l1_map: dict[int, dict[str, Any]],
    l2_map: dict[int, dict[str, Any]],
    l3_map: dict[int, dict[str, Any]],
    rf_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    def _t(suite: str, tid: int) -> dict[str, Any]:
        return (l1_map if suite == "L1" else (l2_map if suite == "L2" else l3_map))[tid]

    cards = [
        {
            "group": "winner",
            "key": "L1 #1",
            "title": "Square Matmul 4096³ — 硬件算力标尺",
            "suite": "L1",
            "id": 1,
            "bucket": "A · 逼近硬件极限",
            "layer": "已逼近 SOL",
            "layer_code": "sol",
            "metric": f"{rf_map['L1 #1']['cand_tf']:.2f} TF ({rf_map['L1 #1']['sol_pct']:.1f}% FP32 SOL)",
            "why": (
                "使用 <code>mma2</code> 寄存器预取的 <code>simdgroup_float8x8</code> 乘加微核 + 直接 <code>simdgroup_store</code> "
                "回写（省去 SRAM 尾声反弹）+ SRAM Bank Swizzle，实测 8.22 TFLOPS，达到 M1 Max 实测峰值 10.171 TF 的 80.8%，"
                "与苹果官方闭源 MPSGraph (8.38 TF) 仅差 1.9%。"
            ),
            "headroom": (
                "剩余空间极小（&lt;10%）。MPSGraph 在规整大矩阵上可调度苹果私有 AMX 协处理器与私有硬件微指令；"
                "单算子稠密 GEMM 已验证编译器底座不存在明显短板，真正高回报在融合（L2）与非标结构（L3）。"
            ),
        },
        {
            "group": "winner",
            "key": "L1 #6",
            "title": "Large-K Matmul (256×524288×256) — Split-K 救活并行度",
            "suite": "L1",
            "id": 6,
            "bucket": "A · 逼近硬件极限",
            "layer": "编译器 Pass",
            "layer_code": "compiler",
            "metric": f"{rf_map['L1 #6']['cand_tf']:.2f} TF ({rf_map['L1 #6']['sol_pct']:.1f}% SOL) vs MPS 7.0%",
            "why": (
                "输出矩阵仅 256×256，普通 64×64 分块只能生成 16 个 Threadgroup，M1 Max 32 核一半空转（MPS 仅跑到 0.71 TF / 7.0% SOL）。"
                "M1Triton 通过 <code>tlx.tile_spec(split_k=...)</code> 切分 K 维并发射向量化 <code>float4</code> 规约，将算力拉升至 5.94 TF（58.4% SOL），获得 8.36x 实打实的并行度收益。"
            ),
            "headroom": (
                "<strong>编译器 Pass 层</strong>：当前 Split-K 因子由 kernel 启发式决定；若在 <code>FusionOccupancyCostModelPass</code> 中"
                "增加基于 <code>grid_tiles &lt; 2 * num_cores</code> 的自动 Split-K 决策与原子/双阶段归约自动生成，所有窄输出 GEMM/Conv 无需手写双核。"
            ),
        },
        {
            "group": "winner",
            "key": "L1 #15",
            "title": "Lower-Triangular Matmul 4096³ — 结构感知死块剪枝",
            "suite": "L1",
            "id": 15,
            "bucket": "B · 精确代数/结构化简",
            "layer": "DSL / 语言",
            "layer_code": "dsl",
            "metric": f"{rf_map['L1 #15']['cand_tf']:.2f} TF (精确 ⅓N³ FLOPs, {rf_map['L1 #15']['sol_pct']:.1f}% SOL)",
            "why": (
                "PyTorch MPS 将下三角矩阵乘法当作完整稠密 GEMM 计算 (2N³ 乘加后再 <code>tril</code>)，浪费 5/6 算力。"
                "M1Triton 通过语言层契约 <code>tlx.structure('lower_triangular')</code> 驱动 <code>StructureAwareDeadTilePruningPass</code> "
                "直接跳过全零 k-tile 与死输出块，在精确有效 FLOPs 口径下依然达到 5.44 TF（53.5% SOL），实现 3.96x 加速。"
            ),
            "headroom": (
                "<strong>DSL 层</strong>：这是 Co-design 最经典的范例——编译器无法凭空证明任意输入矩阵是三角/块稀疏的，但只要语言提供 1 行结构契约，Pass 即可零风险剪枝。"
                "可扩展到 <code>block_sparse</code> / <code>causal_band</code> / <code>symmetric_output</code>。"
            ),
        },
        {
            "group": "winner",
            "key": "L1 #25",
            "title": "Swish (1.61B FP32) & L1 #88 MinGPTNewGelu — 贴近 STREAM 天花板",
            "suite": "L1",
            "id": 25,
            "bucket": "A · 逼近硬件极限",
            "layer": "已逼近 SOL",
            "layer_code": "sol",
            "metric": f"{rf_map['L1 #25']['cand_gbs']:.1f} GB/s ({rf_map['L1 #25']['sol_pct']:.1f}% BW SOL)",
            "why": (
                "<code>tlx.vec_width(4)</code> + <code>ContiguityAndAlignmentAnalysis</code> 证明 16B 对齐后发射 128-bit <code>float4</code> 4x-ILP 访存，"
                "将多步 <code>x * sigmoid(x)</code> 与 <code>MinGPTNewGelu</code> (7.66x, 281.5 GB/s) 全部锁死在寄存器内完成，单遍读写直接跑到 363.5 GB/s（97.2% 实测 STREAM 极限）。"
            ),
            "headroom": (
                "单算子已触碰硬件带宽墙（&lt;3% 空间）。此类逐元素算子唯一有意义的新增收益来自 <strong>L2 级跨算子 Epilogue 融合</strong>，将其吞进上游 GEMM/Conv，直接消灭整次 12.9 GB 读写。"
            ),
        },
        {
            "group": "winner",
            "key": "L1 #66",
            "title": "Conv3D Asymmetric (K=3×5×7) — 隐式 GEMM 击穿 MPS 3D 短板",
            "suite": "L1",
            "id": 66,
            "bucket": "C · MPS 3D/转置短板 + 高 SOL SGMM",
            "layer": "编译器 + 算子",
            "layer_code": "compiler",
            "metric": f"{rf_map['L1 #66']['cand_tf']:.2f} TF ({rf_map['L1 #66']['sol_pct']:.1f}% SOL) vs MPS 0.45 TF",
            "why": (
                "MPS 在 3D 卷积尤其是大非对称核 (3×5×7) 上缺乏针对性 SGMM 调度，仅跑出 0.45 TF（4.4% SOL）。"
                "M1Triton <code>ConvLoweringPass</code> 将 3D 卷积展平为隐式 GEMM，使用 BK=16 流水线 <code>simdgroup_float8x8</code> 算子直接拉到 4.70 TF（46.2% SOL），实现 10.41x 加速。"
            ),
            "headroom": (
                "46.2% → 65%+ SOL 仍有 ~1.4x 提升空间：受限于小输入通道 <code>C_in=3</code> 下按 <code>(C_in·K_d·K_h·K_w)</code> 展开的坐标取模除法开销，"
                "可在 <strong>编译器 Pass</strong> 中引入定点数快速除模 (magic-number divmod) 与 3D 空间 Halo 复用。"
            ),
        },
        {
            "group": "winner",
            "key": "L1 #97",
            "title": "ScaledDotProductAttention (32×32×512×1024) — 复用空闲 SRAM 达 76% SOL",
            "suite": "L1",
            "id": 97,
            "bucket": "A · 逼近硬件极限",
            "layer": "算子 / 算法",
            "layer_code": "kernel",
            "metric": f"{rf_map['L1 #97']['cand_tf']:.2f} TF ({rf_map['L1 #97']['sol_pct']:.1f}% SOL) vs MPS 73.3%",
            "why": (
                "Two-Wave <code>BM=32, BN=512</code> 交换 <code>K@Qᵀ</code>，并巧妙利用空闲的 32 KB SRAM 双缓冲存放转置后的 <code>Qᵀ</code>，消除 <code>transpose_matrix=true</code> 硬件惩罚，"
                "使 <code>K@Qᵀ</code> 达 7.6 TF、<code>P@V</code> 达 8.35 TF，端到端 7.73 TF（76.0% SOL），正面击败 PyTorch MPS SDPA。"
            ),
            "headroom": (
                "在 FP32 下已逼近 16-SG 寄存器压力上限；若面向 LLM 推理切换 FP16/BF16 并扩展至 GQA 共享 KV（如 <code>L3-01</code> 3.25x），可进一步释放带宽和容量红利。"
            ),
        },
        {
            "group": "winner",
            "key": "L2 #77",
            "title": "ConvT3d+Scale+BN+GAP (421.37x) & L2 #42 (32.57x) — 代数重排消灭巨型 5D 张量",
            "suite": "L2",
            "id": 77,
            "bucket": "B · 精确代数/结构化简",
            "layer": "编译器 + DSL",
            "layer_code": "dsl",
            "metric": "967.45ms → 2.30ms (421.37x) · L2 #42 291.9 GB/s (78.1% SOL)",
            "why": (
                "<strong>必须透明披露</strong>：421x 并非把 5D ConvT3d 算出了超光速，而是利用了零填充转置卷积 + 仿射缩放/BN + 全局平均池化 (GAP) 的严格线性可交换性——"
                "先对输入空间规约再做权重折叠矩阵乘，把原本数 GB 的 5D 中间张量直接化为 0 落盘！<code>L2 #42</code> 按最小代数 I/O (1.074 GB) 计也达到 291.9 GB/s（78.1% 带宽 SOL）。"
            ),
            "headroom": (
                "<strong>编译器 Pass + DSL 契约</strong>：当前此类代数折叠通过 <code>AlgebraicNormFoldingPass</code> 与特定任务触发；"
                "若在 FX 图分区器 (<code>compile_backend.py</code>) 中做通用的线性算子交换重写（检测 <code>padding==0 && linear_chain && GAP</code>），可让任意 PyTorch 模型零改代码自动获得数量级收益。"
            ),
        },
        {
            "group": "winner",
            "key": "L3-08",
            "title": "Mamba Selective SSM Scan (66.58x) & L3-03 PagedAttn (12.42x) — 消灭 Python 步进循环",
            "suite": "L3",
            "id": 8,
            "bucket": "D · 消除循环与碎片化 Dispatch",
            "layer": "DSL / 语言",
            "layer_code": "dsl",
            "metric": "55.69ms → 0.84ms (66.58x) · 1 Dispatch",
            "why": (
                "PyTorch Eager 对 Mamba 的 512 步状态递推只能写 Python <code>for t in range(L)</code>（发起上千次小 kernel launch，CPU 调度与中间激活读写完全淹没 GPU）。"
                "M1Triton 用单核分块并行关联扫描 (Blelloch associative scan) 将全部 <code>N_state=16</code> 隐状态锁在 SIMD 寄存器与 SRAM 中，单次 dispatch 仅需 0.84 ms。"
            ),
            "headroom": (
                "<strong>DSL 层</strong>：当前底层依赖 <code>tlx.row_scan</code> / 专用 emitter；若 DSL 原生暴露 <code>tl.associative_scan</code> + <code>tlx.subgroup_shuffle</code>，"
                "用户可直接用 30 行 Python 编写任意线性注意力 / Mamba-2 / RWKV 变体而无需改 C++ 后端。"
            ),
        },
        {
            "group": "winner",
            "key": "L3-10",
            "title": "W4A16 4-bit Group-Quantized GEMM — 寄存器位解包 + 8x 显存压缩",
            "suite": "L3",
            "id": 10,
            "bucket": "A · 逼近硬件极限",
            "layer": "算子 / 算法",
            "layer_code": "kernel",
            "metric": f"{rf_map['L3-10']['cand_tf']:.2f} TF ({rf_map['L3-10']['sol_pct']:.1f}% FP16 SOL, 3.05x)",
            "why": (
                "PyTorch MPS 参考实现必须先把 4-bit 权重完整反量化成 32 MB FP16/64 MB FP32 写回内存再调 GEMM。"
                "M1Triton <code>tlx.quant_dot</code> 在寄存器内用位运算将 <code>uint32</code>（8 个 int4）实时解包并乘以 group-128 scale/zero，直接喂给 <code>simdgroup_half8x8</code> MMA，"
                "将权重显存从 64 MB 压到 8.5 MB，实测 5.59 TF（58.1% FP16 峰值），加速 3.05x。"
            ),
            "headroom": (
                "<strong>算子 + 数据排布层</strong>：在 decode 阶段 (<code>M=1..16</code>) 转入带宽受限时，将权重离线交错重排为匹配 <code>simdgroup_half8x8</code> lane 映射的 nibble 布局，"
                "可把位提取指令减少一半并逼近 350+ GB/s 带宽极限。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L1 #36",
            "title": "RMSNorm (0.66x) — 全库最大瓶颈：17.25 KB SRAM 悬崖 + 64 路 TLB 抖动",
            "suite": "L1",
            "id": 36,
            "bucket": "H · 显著提升空间",
            "layer": "编译器 Pass + 算子",
            "layer_code": "compiler",
            "metric": f"210.39ms vs 138.56ms · 仅 {rf_map['L1 #36']['cand_gbs']:.1f} GB/s ({rf_map['L1 #36']['sol_pct']:.1f}% SOL)",
            "why": (
                "<strong>源码级根因诊断</strong>（<code>norm_softmax_emitter.cpp:713</code>）：输入形状为 <code>(112, 64, 512, 512)</code>（7.52 GB），规约轴 <code>C=64</code> 在中间维（相邻通道步长 <code>HW = 1 MB</code>）。"
                "当前 1-read SRAM 微核分配 <code>m1_tile[64×16] float4 = 16 KB</code> + <code>m1_part/m1_scl = 1.25 KB</code> = <strong>17.25 KB &gt; 16 KB</strong>！"
                "这精确踩中了 M1 Max 32 KB SRAM 半线悬崖，导致每核驻留 TG 数跌至 <strong>1 TG/core</strong>，同时每个 TG 跨 64 个 1 MB 虚拟页并发取数，造成严重 TLB miss 且毫无延迟隐藏！"
            ),
            "headroom": (
                "<strong>极高空间（预计 0.66x → 1.5x–2.2x）</strong>：<br>"
                "① <strong>编译器 Pass</strong>：将 <code>WelfordNorm</code> 纳入 <code>FusionOccupancyCostModelPass</code>，当 <code>sram &gt; 16 KB</code> 时自动将 <code>TILE_HW</code> 从 64 折半至 32（SRAM 降至 ~8.8 KB，恢复 3 TGs/core）。<br>"
                "② <strong>算子算法</strong>：对 <code>HW ≥ 64K</code> 的中间维规约改用 0-SRAM 寄存器分块 2-pass（每次流式扫 8–16 个通道，消除 64 路 TLB 抖动，4+ TGs/core）。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L1 #57/#71",
            "title": "ConvTranspose2d #57 (0.78x) / #71 (0.76x) & Conv2d #63 (0.89x) — 大图小通道 Winograd 选型缺口",
            "suite": "L1",
            "id": 71,
            "bucket": "H · 显著提升空间",
            "layer": "算子算法 + Runtime",
            "layer_code": "kernel",
            "metric": "#71: 11.02ms vs 8.36ms (0.76x) · #57: 72.93ms vs 56.81ms (0.78x)",
            "why": (
                "三者共同特征：<code>3×3, stride=1</code>、超大空间分辨率 (<code>1024×1024</code> / <code>512×1024</code>)、偏小通道数 (<code>C=16..64</code>)。"
                "实测它们仍停留在 2-dispatch 的 <code>Winograd F(2×2, 3×3)</code>（MAC 仅降 2.25x），而分段 <code>F(4×4, 3×3)</code>（如 <code>L1 #55</code> 0.92x, 112 dispatches）"
                "虽能把 MAC 再降 1.78x，却必须把 36 个变换矩阵中间落盘到 DRAM workspace，通道小时变换读写带宽反超计算收益。"
            ),
            "headroom": (
                "<strong>中高空间（预计 0.76x–0.89x → 1.15x–1.40x）</strong>：<br>"
                "① <strong>算子算法</strong>：实现片上 SRAM 融合变换的 <code>F(4×4, 3×3)</code>（避免 36-tile DRAM workspace）或对 <code>C ≤ 32</code> 走寄存器滑窗直接卷积。<br>"
                "② <strong>Runtime / Compiler</strong>：对分段 Winograd（同时影响 <code>L2 #4/#17/#32/#46</code> 的 59–113 次 dispatch）用 <code>MetalGraph</code> ICB 合并发射。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L1 #7",
            "title": "Small-K Matmul 32768×64×32768 (0.89x) & BMM #3 (0.93x) — Ridge Point 写带宽与硬件边界",
            "suite": "L1",
            "id": 7,
            "bucket": "H · 显著提升空间",
            "layer": "编译器 Pass",
            "layer_code": "compiler",
            "metric": f"{rf_map['L1 #7']['cand_tf']:.2f} TF ({rf_map['L1 #7']['sol_pct']:.1f}% SOL) vs MPS 7.44 TF",
            "why": (
                "<code>K=64</code> 时仅需 4 步 K 迭代便要写回高达 <strong>4.29 GB</strong> 的输出矩阵 C，算术强度 <code>AI = 31.9 FLOP/B</code> 紧贴 M1 Max 拐点 (27.2)。"
                "当前 K 维循环极短，难以靠长流水线掩盖启动与写回延迟，且 4-SG/8-SG 驻留度受寄存器与 SRAM 占用制约。"
            ),
            "headroom": (
                "<strong>小到中等空间（0.89x → ~1.00x）</strong>：<br>"
                "<strong>编译器 Pass</strong>：在 <code>TileAndLayoutAssignmentPass</code> 针对小 K（<code>K ≤ 128</code>）自动启用更大输出 tile (<code>128×128</code>) 降写事务数、开启 K=64 全展开免循环分支与非临时流式写回。"
                "剩余 3–5% 属于苹果私有 AMX 硬件差异边界。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L2 #43",
            "title": "Conv3d+MaxPool3d+LSE+ReLU (1.02x) & L2 #11 (1.09x) / #17 (1.13x) — 跨算子 Reduction/Window 融合缺口",
            "suite": "L2",
            "id": 43,
            "bucket": "H · 显著提升空间",
            "layer": "编译器 Pass / IR",
            "layer_code": "compiler",
            "metric": "#43: 81.02ms (dpf=3) · #17: 39.53ms (dpf=60) · #11: 71.90ms (dpf=5)",
            "why": (
                "<strong>源码级根因</strong>：当前 <code>EpilogueDAGAttr</code> 只能融合<strong>逐元素</strong>后处理（如 <code>L2 #11</code> 融合了 BN+Tanh），"
                "但无法融合<strong>空间窗口消费</strong>（<code>Conv → MaxPool2d/3d</code>）或<strong>归约统计量收集</strong>（<code>Conv → InstanceNorm/GroupNorm</code> 的 <code>sum / sum_sq</code>）。"
                "因此 <code>L2 #43</code> 必须把整张 5D 卷积输出写回 DRAM，再读回来做 MaxPool3d；<code>L2 #17</code> 写回 4D 特征图后再被 InstanceNorm 读两遍！"
            ),
            "headroom": (
                "<strong>最大通用编译器红利（预计 1.02x–1.13x → 1.5x–2.2x）</strong>：<br>"
                "<strong>编译器 IR &amp; Pass 层</strong>：将 <code>EpilogueDAG</code> 升级为两类新尾融合原语——"
                "① <em>Reduction Epilogue</em>（在 Conv/GEMM 写回的同时顺手原子/分段累加通道 <code>sum, sum_sq</code>，把 InstanceNorm/GroupNorm 砍成 1-pass）；"
                "② <em>Spatial-Window Consumer Epilogue</em>（2×2/2×2×2 输出 tile 在 SRAM 内直接取 Max/Avg 后写回，直接砍掉 4x–8x DRAM 写回流量）。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L3-02",
            "title": "FlashAttention-2 Fwd+Bwd (1.05x) — 双核反向重算 + 前向同步开销",
            "suite": "L3",
            "id": 2,
            "bucket": "H · 显著提升空间",
            "layer": "算子 + 编译器 + Runtime",
            "layer_code": "compiler",
            "metric": "1.880ms vs 1.973ms (1.05x) · ICB 实测 Bwd 可再快 1.47x",
            "why": (
                "源码 <code>l3_02_flash_attn2_bwd.py</code> 显示两大瓶颈：① Backward 拆分为 <code>BwdDKDV</code> 与 <code>BwdDQ</code> 两个独立 kernel（虽在并发 <code>command_batch</code> 中并行），"
                "但 <code>Q, K, V, dO</code> 被读两遍且 <code>S=QKᵀ, P=exp(S-LSE)</code> 被重算两遍；② <code>B=2, H=8, L=512</code> 规模小，单次前向 <code>waitUntilCompleted</code> 往返占比极高。"
                "且实验账本证明直接把 SRAM 翻倍到 17.6 KB 做双缓冲会掉 occupancy（0.94x → 0.90x）。"
            ),
            "headroom": (
                "<strong>高空间（预计 1.05x → 1.45x–1.80x）</strong>：<br>"
                "① <strong>编译器 Pass</strong>：SRAM 活跃区间着色复用（<code>Q/K</code> 与 <code>P/dS</code> 分时共享同一块 &lt;12 KB SRAM），在保持 3 TGs/core 下消除冲突。<br>"
                "② <strong>算子算法</strong>：单循环 fused backward（计算 dK/dV 时同步用原子或分块归约累加 dQ，省去一半 QKᵀ/exp 重算）。<br>"
                "③ <strong>Runtime / Autograd</strong>：利用已验证快 1.47x 的 <code>MetalGraph</code> Compute ICB（0.921ms vs 1.354ms）配合异步跨队事件消除前向强制同步。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L3-06",
            "title": "Fused LayerNorm Fwd+Bwd (1.18x) — 反向双读 X/dY + 独立清零核",
            "suite": "L3",
            "id": 6,
            "bucket": "H · 显著提升空间",
            "layer": "算子 + DSL",
            "layer_code": "dsl",
            "metric": "1.626ms vs 1.913ms (1.18x) · Bwd 3 Dispatches",
            "why": (
                "查看 <code>l3_06_fused_layernorm_fwd_bwd.py:445-478</code>：反向共发 3 个 kernel——<code>zero_dwdb</code> 清零、<code>bwd_dx</code> 完整读一遍 <code>(x, dy)</code> 算 <code>dx</code>、"
                "<code>bwd_dwdb_atomic</code> <strong>再完整读一遍 <code>(x, dy)</code></strong> 用 <code>atomic_float</code> 累参数梯度，带宽流量直接翻倍！"
            ),
            "headroom": (
                "<strong>中高空间（预计 1.18x → 1.55x–1.85x）</strong>：<br>"
                "<strong>算子算法 + DSL 原语</strong>：采用经典 Triton 持久化分组融合（每个 TG 负责 <code>GROUP_SIZE_M</code> 行：一次读入 <code>(x, dy)</code>，算完 <code>dx</code> 写回的同时把局部 <code>dw, db</code> 累在寄存器/片上缓冲，退出前只写一次部分和 + 轻量级阶段 2 规约）。"
                "在 DSL 中抽象 <code>tlx.persistent_reduce</code> 即可让 LayerNorm/RMSNorm/BatchNorm 反向通用复用。"
            ),
        },
        {
            "group": "bottleneck",
            "key": "L3-07",
            "title": "Chunked Linear + Cross-Entropy (1.25x) — 端到端已达 7.19 TF，切块内多扫 2 遍 Logits",
            "suite": "L3",
            "id": 7,
            "bucket": "H · 显著提升空间",
            "layer": "编译器 + 算子",
            "layer_code": "compiler",
            "metric": f"{rf_map['L3-07']['cand_tf']:.2f} TF ({rf_map['L3-07']['sol_pct']:.1f}% SOL 端到端!) · 零 [BT,V] 显存",
            "why": (
                "按 3 次 SGMM (<code>6·BT·H·V = 412.3 GFLOPs</code>) 折算，M1Triton 端到端已达 <strong>7.19 TF（70.7% FP32 峰值）</strong> 同时完全消除了 256 MB <code>[2048, 32768]</code> 巨型 logits 张量。"
                "剩余时间消耗在源码 <code>l3_07:491-520</code>：每个 chunk 内 <code>logits_c</code> 被写了读、读了写——先调 <code>online_log_softmax</code>、再 <code>gather</code>、再 <code>logp_to_dlogits</code> 扫一遍 <code>exp()</code>，且每 chunk 用 <code>ADD_RES=1</code> 重复读写 <code>dW [V, H]</code>！"
            ),
            "headroom": (
                "<strong>稳健空间（预计 1.25x → 1.45x–1.60x）</strong>：<br>"
                "① <strong>算子算法</strong>：将 <code>online_log_softmax + gather_nll + (exp - one_hot) * inv_bt</code> 合并为单遍 <code>[chunk_t, V]</code> 微核，直接省掉 2 次 64 MB 读写与重复 <code>exp</code>。<br>"
                "② <strong>编译器 Pass</strong>：支持可组合的 <em>Reduction-with-Epilogue</em> 融合，不再把 <code>tlx.online_softmax</code> 当作不可拼接的黑盒宏算子。"
            ),
        },
    ]

    for c in cards:
        src = _t(c["suite"], c["id"])
        c["base_ms"] = round(float(src["baseline_median_ms"]), 3)
        c["cand_ms"] = round(float(src["candidate_median_ms"]), 3)
        c["speedup"] = round(float(src["speedup"]), 2)
    return cards


def _load_experiments() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tag, fname in [("v1.0 (L1)", "experiments.tsv"), ("v2.0 (L2/L3)", "experiments_v2.tsv")]:
        p = AUDIT / fname
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for r in reader:
                rows.append(
                    {
                        "phase": tag,
                        "premise": r.get("premise", ""),
                        "hypothesis": r.get("hypothesis", ""),
                        "baseline": r.get("baseline", ""),
                        "candidate": r.get("candidate", ""),
                        "direction": r.get("direction", ""),
                        "verdict": r.get("verdict", ""),
                        "explanation": r.get("explanation", ""),
                    }
                )
    return rows


def _build_all_tasks(
    l1: dict[str, Any],
    l2: dict[str, Any],
    l3: dict[str, Any],
    deep_dives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dive_lookup: dict[tuple[str, int], dict[str, Any]] = {(d["suite"], d["id"]): d for d in deep_dives}
    all_rows: list[dict[str, Any]] = []

    for suite, data, cat_fn, is_l3 in [
        ("L1", l1, _cat_l1, False),
        ("L2", l2, _cat_l2, False),
        ("L3", l3, _cat_l3, True),
    ]:
        for t in data["tasks"]:
            tid = int(t["id"])
            max_fwd, max_bwd = _max_seed_err(t, is_l3)
            if suite == "L2":
                dpf = int(t.get("dispatches_per_forward", 0))
            elif suite == "L1":
                srs = t.get("seed_reports", [])
                dpf = int(srs[0].get("dispatches", 1)) if srs else int(t.get("compiled_kernel_dispatch_count", 0))
            else:
                srs = t.get("seed_reports", [])
                if len(srs) >= 2:
                    s1 = srs[1]
                    dpf = int(s1.get("fwd_dispatches", 0)) + int(s1.get("bwd_dispatches", 0))
                else:
                    dpf = int(t.get("compiled_kernel_dispatch_count", 0))

            dive = dive_lookup.get((suite, tid))
            note = dive["why"] if dive else ""
            headroom = dive["headroom"] if dive else ""
            layer = dive["layer"] if dive else (
                "已逼近 SOL" if (0.98 <= float(t["speedup"]) <= 1.05 and suite == "L1") else ""
            )

            all_rows.append(
                {
                    "suite": suite,
                    "id": tid,
                    "name": str(t["name"]),
                    "cat": cat_fn(t),
                    "backend": str(t.get("baseline_backend", "mps")),
                    "bwd": bool(t.get("supports_backward", False)),
                    "base_ms": round(float(t["baseline_median_ms"]), 4),
                    "cand_ms": round(float(t["candidate_median_ms"]), 4),
                    "base_iqr": round(float(t.get("baseline_iqr_ms", 0.0)), 4),
                    "cand_iqr": round(float(t.get("candidate_iqr_ms", 0.0)), 4),
                    "speedup": round(float(t["speedup"]), 4),
                    "dpf": dpf,
                    "total_disp": int(t.get("compiled_kernel_dispatch_count", 0)),
                    "cold_ms": round(float(t.get("cold_compile_ms", 0.0)), 1),
                    "max_err": max_fwd,
                    "max_bwd_err": max_bwd,
                    "bucket": _attribution_bucket(suite, t),
                    "file": Path(str(t.get("candidate_file", ""))).name if t.get("candidate_file") else "",
                    "note": note,
                    "headroom": headroom,
                    "layer": layer,
                }
            )
    return all_rows


def _build_category_stats(all_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for suite in ("L1", "L2", "L3"):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for r in all_rows:
            if r["suite"] == suite:
                buckets.setdefault(r["cat"], []).append(r)
        items: list[dict[str, Any]] = []
        for cat, rs in buckets.items():
            sps = [x["speedup"] for x in rs]
            items.append(
                {
                    "cat": cat,
                    "count": len(rs),
                    "beats": sum(1 for x in sps if x > 1.0),
                    "ge98": sum(1 for x in sps if x >= 0.98),
                    "geomean": round(_geomean(sps), 2),
                    "min": round(min(sps), 2),
                    "max": round(max(sps), 2),
                }
            )
        items.sort(key=lambda x: -x["geomean"])
        out[suite] = items
    return out


def build_report() -> str:
    rf = json.loads((AUDIT / "roofline_m1max.json").read_text(encoding="utf-8"))
    l1 = json.loads((AUDIT / "eval_final_100.json").read_text(encoding="utf-8"))
    l2 = json.loads((AUDIT / "eval_final_l2_100.json").read_text(encoding="utf-8"))
    l3 = json.loads((AUDIT / "eval_final_l3_10.json").read_text(encoding="utf-8"))

    l1_map = {int(t["id"]): t for t in l1["tasks"]}
    l2_map = {int(t["id"]): t for t in l2["tasks"]}
    l3_map = {int(t["id"]): t for t in l3["tasks"]}

    bw_peak = float(rf["stream_copy"]["bandwidth_gbs"])
    fp32_peak = float(rf["sgmm_fp32"]["tflops"])
    fp16_peak = float(rf["sgmm_fp16"]["tflops"])
    ridge = round(fp32_peak * 1000.0 / bw_peak, 2)

    rf_points = _build_roofline_points(l1_map, l2_map, l3_map, bw_peak, fp32_peak, fp16_peak)
    rf_map = {p["key"]: p for p in rf_points}
    deep_dives = _build_deep_dives(l1_map, l2_map, l3_map, rf_map)
    all_rows = _build_all_tasks(l1, l2, l3, deep_dives)
    cat_stats = _build_category_stats(all_rows)
    experiments = _load_experiments()

    total_n = len(all_rows)
    total_beats = sum(1 for r in all_rows if r["speedup"] > 1.0)
    total_ge98 = sum(1 for r in all_rows if r["speedup"] >= 0.98)
    l1_ge98 = sum(1 for r in all_rows if r["suite"] == "L1" and r["speedup"] >= 0.98)
    l2_dpf1 = sum(1 for r in all_rows if r["suite"] == "L2" and r["dpf"] == 1)
    overall_gm = round(_geomean([r["speedup"] for r in all_rows]), 4)

    bucket_counts = {k: sum(1 for r in all_rows if r["bucket"] == k) for k in ("A", "B", "C", "D")}

    payload = {
        "hardware": {
            "device": rf["device_name"],
            "sram_kb": int(rf["max_threadgroup_memory_bytes"]) // 1024,
            "max_tg": int(rf["max_threads_per_threadgroup"]),
            "uma_gb": rf["recommended_max_working_set_gb"],
            "bw_peak": bw_peak,
            "bw_triad": float(rf["stream_triad"]["bandwidth_gbs"]),
            "fp32_peak": fp32_peak,
            "fp16_peak": fp16_peak,
            "ridge": ridge,
            "platform": rf["host_platform"],
        },
        "summary": {
            "total": total_n,
            "beats": total_beats,
            "ge98": total_ge98,
            "overall_gm": overall_gm,
            "l1_gm": l1["summary"]["geomean_speedup"],
            "l1_beats": l1["summary"]["beats_baseline"],
            "l1_ge98": l1_ge98,
            "l2_gm": l2["summary"]["geomean_speedup"],
            "l2_beats": l2["summary"]["beats_baseline"],
            "l2_dpf_med": l2["summary"]["median_dispatches_per_forward"],
            "l2_dpf1": l2_dpf1,
            "l3_gm": l3["summary"]["geomean_speedup"],
            "l3_beats": l3["summary"]["beats_baseline"],
            "bucket_counts": bucket_counts,
        },
        "roofline_points": rf_points,
        "deep_dives": deep_dives,
        "cat_stats": cat_stats,
        "experiments": experiments,
        "tasks": all_rows,
    }

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    return _HTML_TEMPLATE.replace("__REPORT_DATA_JSON__", data_json)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>M1Triton 性能全景报告 &amp; DSL-Compiler Co-Design 深度剖析 (210 Cases)</title>
<style>
:root {
  --bg: #0b0f17;
  --bg-elev: #111827;
  --bg-card: #151e2e;
  --bg-hover: #1c273b;
  --border: #253249;
  --text: #e6edf7;
  --text-dim: #93a4be;
  --text-muted: #64748b;
  --accent: #38bdf8;
  --accent-2: #818cf8;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #f87171;
  --purple: #c084fc;
  --cyan: #22d3ee;
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
[data-theme="light"] {
  --bg: #f6f8fc;
  --bg-elev: #ffffff;
  --bg-card: #ffffff;
  --bg-hover: #eef3fb;
  --border: #d8e1ee;
  --text: #0f172a;
  --text-dim: #475569;
  --text-muted: #64748b;
  --accent: #0284c7;
  --accent-2: #4f46e5;
  --green: #059669;
  --amber: #d97706;
  --red: #dc2626;
  --purple: #7c3aed;
  --cyan: #0891b2;
  --shadow: 0 8px 24px rgba(15,23,42,0.08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  line-height: 1.6;
  font-size: 14px;
}
code {
  font-family: var(--mono);
  font-size: 0.88em;
  background: rgba(56, 189, 248, 0.11);
  color: var(--accent);
  padding: 1px 5px;
  border-radius: 4px;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 50;
  backdrop-filter: blur(12px);
  background: color-mix(in srgb, var(--bg) 86%, transparent);
  border-bottom: 1px solid var(--border);
}
.topbar-inner {
  max-width: 1400px;
  margin: 0 auto;
  padding: 10px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: -0.01em;
}
.brand-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: linear-gradient(135deg, var(--accent), var(--green));
  box-shadow: 0 0 12px var(--accent);
}
.nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.nav a {
  color: var(--text-dim);
  text-decoration: none;
  padding: 5px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.15s;
}
.nav a:hover { color: var(--text); background: var(--bg-hover); }
.btn {
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text);
  padding: 5px 11px;
  border-radius: 6px;
  font-size: 12.5px;
  cursor: pointer;
  font-family: inherit;
  transition: all 0.15s;
}
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn.active {
  background: color-mix(in srgb, var(--accent) 16%, var(--bg-card));
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}
.container { max-width: 1400px; margin: 0 auto; padding: 28px 24px 80px; }
.hero {
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 12%, var(--bg-card)), color-mix(in srgb, var(--accent-2) 10%, var(--bg-card)));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px 30px;
  box-shadow: var(--shadow);
  margin-bottom: 24px;
}
.hero h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: -0.02em; line-height: 1.25; }
.hero-sub { color: var(--text-dim); margin: 0 0 16px; font-size: 14.5px; max-width: 980px; }
.hw-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  font-size: 12px; font-family: var(--mono);
  background: var(--bg-elev); border: 1px solid var(--border); color: var(--text-dim);
}
.pill strong { color: var(--text); }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 28px;
}
@media (max-width: 960px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns: 1fr; } }
.kpi {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
}
.kpi::before {
  content: "";
  position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--kpi-color, var(--accent));
}
.kpi-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-val { font-size: 28px; font-weight: 800; font-family: var(--mono); margin: 4px 0 4px; letter-spacing: -0.02em; }
.kpi-sub { font-size: 12.5px; color: var(--text-dim); }
section { margin-top: 38px; scroll-margin-top: 68px; }
.sec-head {
  display: flex; justify-content: space-between; align-items: flex-end;
  flex-wrap: wrap; gap: 12px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px; margin-bottom: 18px;
}
.sec-head h2 { margin: 0; font-size: 20px; letter-spacing: -0.01em; }
.sec-desc { color: var(--text-dim); font-size: 13px; margin: 2px 0 0; }
.grid-2 { display: grid; grid-template-columns: 1.25fr 1fr; gap: 18px; align-items: stretch; }
.grid-2-eq { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 1020px) { .grid-2, .grid-2-eq { grid-template-columns: 1fr; } }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
}
.card h3 { margin: 0 0 10px; font-size: 15.5px; }
.roofline-wrap { position: relative; }
.tooltip {
  position: fixed;
  pointer-events: none;
  z-index: 999;
  background: var(--bg-elev);
  border: 1px solid var(--accent);
  border-radius: 10px;
  padding: 10px 13px;
  box-shadow: var(--shadow);
  font-size: 12.5px;
  max-width: 320px;
  display: none;
}
.tooltip h4 { margin: 0 0 4px; font-size: 13.5px; color: var(--accent); }
.tooltip .kv { display: flex; justify-content: space-between; gap: 14px; font-family: var(--mono); font-size: 12px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--text-dim); margin-top: 8px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.bucket-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
@media (max-width: 1050px) { .bucket-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .bucket-grid { grid-template-columns: 1fr; } }
.bucket-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 3px solid var(--bcolor);
  border-radius: 10px;
  padding: 14px 16px;
}
.bucket-card h4 { margin: 0 0 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center; }
.bucket-card p { margin: 0; font-size: 12.5px; color: var(--text-dim); }
.badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--mono);
  white-space: nowrap;
}
.b-green { background: rgba(52,211,153,0.16); color: var(--green); }
.b-blue { background: rgba(56,189,248,0.16); color: var(--accent); }
.b-amber { background: rgba(251,191,36,0.16); color: var(--amber); }
.b-red { background: rgba(248,113,113,0.16); color: var(--red); }
.b-purple { background: rgba(192,132,252,0.16); color: var(--purple); }
.b-gray { background: rgba(148,163,184,0.16); color: var(--text-dim); }

.dive-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 960px) { .dive-grid { grid-template-columns: 1fr; } }
.dive-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dive-card.is-bottleneck { border-left: 3px solid var(--red); }
.dive-card.is-winner { border-left: 3px solid var(--green); }
.dive-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; flex-wrap: wrap; }
.dive-title { font-weight: 700; font-size: 15px; margin: 0; }
.dive-metrics {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  font-family: var(--mono); font-size: 12px;
  padding: 7px 10px; border-radius: 8px; background: var(--bg-elev); border: 1px solid var(--border);
}
.dive-block { font-size: 13px; color: var(--text-dim); }
.dive-block strong { color: var(--text); }
.dive-headroom {
  font-size: 12.8px;
  padding: 9px 11px;
  border-radius: 8px;
  background: color-mix(in srgb, var(--accent-2) 8%, var(--bg-elev));
  border: 1px dashed color-mix(in srgb, var(--accent-2) 40%, var(--border));
  color: var(--text-dim);
}
.dive-headroom strong { color: var(--text); }

.law-banner {
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent-2) 16%, var(--bg-card)), color-mix(in srgb, var(--green) 10%, var(--bg-card)));
  border: 1px solid color-mix(in srgb, var(--accent-2) 45%, var(--border));
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 18px;
}
.law-banner .law-tag { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); }
.law-banner blockquote {
  margin: 6px 0 8px;
  font-size: 17.5px;
  font-weight: 700;
  line-height: 1.45;
  color: var(--text);
}
.law-banner p { margin: 0; font-size: 13.5px; color: var(--text-dim); }

.pillar-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 20px; }
@media (max-width: 1020px) { .pillar-grid { grid-template-columns: 1fr; } }
.pillar {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top: 3px solid var(--pcolor);
  border-radius: 12px;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pillar h3 { margin: 0; font-size: 16px; display: flex; justify-content: space-between; align-items: center; }
.pillar .sub { font-size: 12.5px; color: var(--text-dim); margin-top: -4px; }
.pillar ul { margin: 0; padding-left: 18px; font-size: 13px; color: var(--text-dim); display: flex; flex-direction: column; gap: 8px; }
.pillar li strong { color: var(--text); }
.pillar .antipattern {
  margin-top: auto;
  font-size: 12px;
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(248,113,113,0.08);
  border: 1px solid rgba(248,113,113,0.22);
  color: var(--text-dim);
}

.controls {
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  margin-bottom: 12px;
}
.search-input, .select-input {
  background: var(--bg-elev);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 11px;
  border-radius: 7px;
  font-size: 13px;
  font-family: inherit;
}
.search-input { min-width: 240px; flex: 1; max-width: 360px; }
.search-input:focus, .select-input:focus { outline: none; border-color: var(--accent); }
.filter-summary {
  display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
  padding: 9px 14px; border-radius: 8px;
  background: var(--bg-elev); border: 1px solid var(--border);
  font-size: 12.5px; font-family: var(--mono); color: var(--text-dim);
  margin-bottom: 12px;
}
.filter-summary strong { color: var(--text); }

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-card);
}
table { width: 100%; border-collapse: collapse; font-size: 12.8px; }
th, td { padding: 9px 11px; text-align: left; border-bottom: 1px solid var(--border); }
th {
  background: var(--bg-elev);
  color: var(--text-dim);
  font-weight: 600;
  white-space: nowrap;
  position: sticky;
  top: 0;
  user-select: none;
}
th.sortable { cursor: pointer; }
th.sortable:hover { color: var(--accent); }
td.num, th.num { text-align: right; font-family: var(--mono); }
tr.data-row { cursor: pointer; transition: background 0.1s; }
tr.data-row:hover { background: var(--bg-hover); }
tr.detail-row td {
  background: var(--bg-elev);
  padding: 12px 18px;
  font-size: 12.5px;
  color: var(--text-dim);
}
.sp-bar-cell { min-width: 120px; }
.sp-bar-bg {
  height: 6px; border-radius: 99px; background: rgba(148,163,184,0.16);
  overflow: hidden; margin-top: 3px;
}
.sp-bar-fg { height: 100%; border-radius: 99px; }

.small-table th, .small-table td { padding: 7px 9px; font-size: 12px; }
footer {
  margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border);
  color: var(--text-muted); font-size: 12px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-inner">
    <div class="brand"><span class="brand-dot"></span><span>M1Triton Performance &amp; Co-Design Report</span></div>
    <div class="nav">
      <a href="#roofline">Roofline &amp; SOL</a>
      <a href="#attribution">收益归因</a>
      <a href="#deepdives">核心 Kernel 解读</a>
      <a href="#codesign">Co-Design 剖析</a>
      <a href="#allcases">全量 210 Cases</a>
      <button class="btn" id="themeBtn" type="button">切换明暗主题</button>
    </div>
  </div>
</div>

<div class="container">
  <div class="hero">
    <h1>M1Triton 性能全景报告 &amp; DSL-Compiler Co-Design 深度剖析</h1>
    <p class="hero-sub">
      覆盖全部 <strong>210 个官方冻结评测用例</strong>（KernelBench L1 100 + L2 100 + Level 3 复杂 Triton 10，均通过 SHA-256 锁定评测脚本验证，零 PyTorch/MPSGraph 计算回退），
      结合 Apple M1 Max 实测 Roofline、硬件利用率 (% SOL)、核心算子源码级归因，系统回答 <strong>「剩余提升空间从编译器 Pass、DSL 语言还是算子算法来」</strong>。
    </p>
    <div class="hw-pills" id="hwPills"></div>
  </div>

  <div class="kpi-grid" id="kpiGrid"></div>

  <section id="roofline">
    <div class="sec-head">
      <div>
        <h2>1. Apple M1 Max 实测 Roofline 与硬件利用率 (% SOL) 真实性校验</h2>
        <p class="sec-desc">所有数据均直接取自冻结产物 <code>.audit/roofline_m1max.json</code> 与 <code>eval_final_*.json</code>；悬停圆点查看精确 FLOPs/Bytes、算术强度与 MPS 对比</p>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <button class="btn active" id="rfToggleMps" type="button">显示 MPS 对比连线</button>
        <button class="btn active rf-suite-btn" data-suite="ALL" type="button">全部</button>
        <button class="btn rf-suite-btn" data-suite="L1" type="button">L1</button>
        <button class="btn rf-suite-btn" data-suite="L2" type="button">L2</button>
        <button class="btn rf-suite-btn" data-suite="L3" type="button">L3</button>
      </div>
    </div>
    <div class="grid-2">
      <div class="card roofline-wrap">
        <h3>M1 Max 对数 Roofline 图（带宽上限 373.83 GB/s · FP32 算力上限 10.171 TF · 拐点 27.2 FLOP/B）</h3>
        <svg id="rooflineSvg" viewBox="0 0 760 430" width="100%" style="display:block;"></svg>
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:var(--green)"></span>桶A: 逼近硬件 SOL</span>
          <span class="legend-item"><span class="swatch" style="background:var(--purple)"></span>桶B: 代数/结构化简</span>
          <span class="legend-item"><span class="swatch" style="background:var(--amber)"></span>桶C: 击败 MPS 3D/转置</span>
          <span class="legend-item"><span class="swatch" style="background:var(--red)"></span>待优化瓶颈</span>
          <span class="legend-item"><span class="swatch" style="background:transparent;border:2px solid var(--text-dim)"></span>空心圈: PyTorch MPS</span>
        </div>
      </div>
      <div class="card" style="overflow-x:auto;">
        <h3>代表性算子硬件利用率 (% SOL) 核验表</h3>
        <table class="small-table" id="solTable">
          <thead>
            <tr>
              <th>算子</th>
              <th class="num">AI</th>
              <th class="num">实测吞吐</th>
              <th class="num">M1T SOL</th>
              <th class="num">MPS SOL</th>
              <th class="num">加速比</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="attribution">
    <div class="sec-head">
      <div>
        <h2>2. 性能收益归因（4 大桶透明拆解）与分布统计</h2>
        <p class="sec-desc">区分「真实跑满硬件」与「代数消除落盘 / MPS 自身短板 / 消除循环调度」，避免被夸张倍率误导</p>
      </div>
    </div>
    <div class="bucket-grid" id="bucketGrid"></div>
    <div class="grid-2-eq">
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
          <h3 style="margin:0;">分类几何平均加速比</h3>
          <div style="display:flex;gap:6px;">
            <button class="btn active cat-suite-btn" data-suite="L1" type="button">L1 (10类)</button>
            <button class="btn cat-suite-btn" data-suite="L2" type="button">L2 (5大融合家族)</button>
            <button class="btn cat-suite-btn" data-suite="L3" type="button">L3 (5类)</button>
          </div>
        </div>
        <svg id="catChartSvg" viewBox="0 0 600 330" width="100%"></svg>
      </div>
      <div class="card">
        <h3 style="margin-bottom:8px;">全量 210 Cases 加速比区间分布（点击柱子联动筛选下方总表）</h3>
        <svg id="histChartSvg" viewBox="0 0 600 330" width="100%"></svg>
      </div>
    </div>
  </section>

  <section id="deepdives">
    <div class="sec-head">
      <div>
        <h2>3. Agent 核心 Kernel 逐案深度解读（高收益典范 vs. 显著瓶颈）</h2>
        <p class="sec-desc">全部诊断均基于真实源码路径与实测性能计数器，标注每类瓶颈对应的最优改进层级</p>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button class="btn active dive-filter-btn" data-group="all" type="button">全部 (16)</button>
        <button class="btn dive-filter-btn" data-group="winner" type="button">高收益 / 架构典范 (9)</button>
        <button class="btn dive-filter-btn" data-group="bottleneck" type="button">待优化 / 显著空间 (7)</button>
      </div>
    </div>
    <div class="dive-grid" id="diveGrid"></div>
  </section>

  <section id="codesign">
    <div class="sec-head">
      <div>
        <h2>4. DSL-Compiler Co-Design 核心剖析：未来提升从编译器、语言还是算子来？</h2>
        <p class="sec-desc">从 210 个真实案例与 13 次正反实验（含 4 次回退教训）中提炼出的三层分工法则</p>
      </div>
    </div>

    <div class="law-banner">
      <div class="law-tag">The DSL-Compiler Co-Design Law</div>
      <blockquote>「算法创新留在算子层做一次，反复出现的硬件与数学结构约束沉淀为极简 DSL 契约 (<code>tlx.*</code>)，一切可从 <code>TargetSpec</code> 机械推导的调度、内存布局与融合全部下沉进纯 C++ 编译器 Pass。」</blockquote>
      <p>
        对 M1Triton 下一阶段演进的定量判断：<strong>约 45% 的剩余通用收益来自编译器 Pass/IR 升级</strong>（以多消费者/归约尾融合与 SRAM 活跃区间分析为代表），
        <strong>约 25% 来自 DSL 语言契约补齐</strong>（让前沿算子摆脱黑盒 C++ 模板、支持可组合原语），
        <strong>约 30% 来自算子算法与数据布局创新</strong>（单循环 FA2 Bwd、片上 Winograd F(4×4)、单遍 CE 与 W4 nibble 布局）。
      </p>
    </div>

    <div class="pillar-grid">
      <div class="pillar" style="--pcolor:var(--accent)">
        <h3><span>① 编译器 Pass &amp; IR 层</span><span class="badge b-blue">~45% 通用红利</span></h3>
        <div class="sub">零算子重写 · 收益随模型数量线性放大</div>
        <ul>
          <li><strong>Reduction &amp; Spatial-Window Epilogue 融合</strong>：扩展 <code>EpilogueDAGAttr</code>，在 Conv/GEMM 写回同时片上算 <code>2×2 MaxPool</code> 或累加 Norm 统计量（<code>sum, sum_sq</code>），直接攻克 <code>L2 #43 (1.02x)</code>、<code>#11 (1.09x)</code>、<code>#17 (1.13x)</code>、<code>#19 (1.42x)</code>。</li>
          <li><strong>全算子 Occupancy + SRAM Live-Range Coloring</strong>：把 <code>WelfordNorm</code> 纳入 <code>FusionOccupancyCostModelPass</code> 拦截 <code>L1 #36</code> 的 17.25 KB 悬崖；对 <code>L3-02</code> FA2 Bwd 做 SRAM 活跃区间分时复用，将占用压回 &lt;16 KB 保 3 TGs/core。</li>
          <li><strong>拆解黑盒 <code>tlx.*</code> 宏算子为通用循环降低</strong>：支持对标准 Triton <code>for k: tl.dot + online_reduce</code> 循环自动识别并生成流水线 SGMM，减少 C++ 专用 emitter 维护面。</li>
          <li><strong>自动 <code>MetalGraph</code> ICB 捕获与 Split-K 决策</strong>：在 <code>compile_backend.py</code>（FX 分区器）自动捕获静态子图与分段 Winograd（消除 <code>L2 #4/#17/#32/#46</code> 60–113 次发射），实测 ICB 可提速 1.47x。</li>
        </ul>
        <div class="antipattern"><strong>放错层的代价</strong>：把 occupancy 裁剪、bank swizzle、epilogue 融合留在算子层，会导致每个新算子都重踩一次 16 KB SRAM 悬崖（<code>L1 #36</code> 就是血泪教训）。</div>
      </div>

      <div class="pillar" style="--pcolor:var(--purple)">
        <h3><span>② DSL / 语言层 (<code>tl</code> + <code>tlx</code>)</span><span class="badge b-purple">~25% 表达力杠杆</span></h3>
        <div class="sub">填补「编译器无法安全自动推断」的语义鸿沟</div>
        <ul>
          <li><strong>代数与稀疏结构契约</strong>：扩展 <code>tlx.structure</code> 至 <code>causal_band</code> / <code>block_sparse</code> / <code>linear_commutes_with_pool</code>，让编译器 100% 安全地执行类似 <code>L1 #15 (3.96x)</code> 与 <code>L2 #77 (421x)</code> 的死块剪枝与代数重排。</li>
          <li><strong>硬件资源意图声明 (<code>tlx.sram_budget</code> / <code>tlx.occupancy_target</code>)</strong>：把 Apple Silicon 特有的「保 3 TG/core 优先于双缓冲」表达为一等语言契约，驱动成本模型而非写死魔法数。</li>
          <li><strong>SIMD-group 协作与持久调度原语</strong>：补齐 <code>tlx.subgroup_shuffle</code>、<code>tl.associative_scan</code>、<code>tlx.persistent_reduce</code>，让 <code>L3-08 Mamba</code>、<code>L3-06 LayerNorm Bwd</code>、<code>L3-09 MoE</code> 用纯 Python Triton 可组合写出。</li>
          <li><strong>多输出/原地别名契约</strong>：声明只读/原地写缓冲区契约，供 <code>command_batch()</code> 和 <code>MetalGraph</code> 精确消除保守内存屏障。</li>
        </ul>
        <div class="antipattern"><strong>放错层的代价</strong>：把每个新算法直接做成一个黑盒 <code>tlx.my_op(...)</code> 宏指令，会退化成「带 Python 壳的手写算子库」，失去 Triton 的可组合性（如 <code>L3-07</code> 无法把 gather+dlogits 拼进 <code>tlx.online_softmax</code>）。</div>
      </div>

      <div class="pillar" style="--pcolor:var(--green)">
        <h3><span>③ 算子 / 算法与布局层</span><span class="badge b-green">~30% 复杂度降维</span></h3>
        <div class="sub">数学降阶与硬件亲和数据排布，编译器无法凭空发明</div>
        <ul>
          <li><strong>单循环 FlashAttention-2 反向 (<code>L3-02</code>)</strong>：把 <code>BwdDKDV</code> 与 <code>BwdDQ</code> 合并，消除一半 <code>QKᵀ</code> 与 <code>exp</code> 重算；保留已验证有效的 Causal 逆序发射防尾延迟。</li>
          <li><strong>单遍 Logits→Loss+dLogits (<code>L3-07</code>) &amp; 单读 LayerNorm 反向 (<code>L3-06</code>)</strong>：分别消除 <code>[chunk_t, V]</code> 的 2 次多余 DRAM 扫盘与 <code>(x, dy)</code> 的双倍读取。</li>
          <li><strong>片上变换 Winograd <code>F(4×4, 3×3)</code> &amp; 通道分块 RMSNorm</strong>：攻克 <code>L1 #57/#71/#63</code>（0.76x–0.89x）的 36-tile workspace 带宽墙与 <code>L1 #36</code>（0.66x）的 1 MB 步长 64 路 TLB 抖动。</li>
          <li><strong>硬件亲和权重重排 (<code>L3-10</code>)</strong>：离线把 W4 nibble 交错排布成直接对齐 <code>simdgroup_half8x8</code> 寄存器 lane 的格式，榨干 LLM decode 带宽。</li>
        </ul>
        <div class="antipattern"><strong>客观硬件边界</strong>：Apple M1 Max 的闭源 AMX 协处理器在规整稠密大矩阵（<code>L1 #1 0.98x</code>, <code>#7 0.89x</code>）上有固定优势；死磕最后 5% 性价比低，主战场应锁定 L2 融合与 L3 算法。</div>
      </div>
    </div>

    <div class="card" style="overflow-x:auto;">
      <h3>实证账本：来自 <code>.audit/experiments.tsv</code> &amp; <code>experiments_v2.tsv</code> 的 13 次真实正反实验（含 4 次回退）</h3>
      <table class="small-table" id="expTable">
        <thead>
          <tr>
            <th>阶段</th>
            <th>实验假设 / 改动</th>
            <th class="num">基线 → 候选</th>
            <th>结论</th>
            <th>实测归因与架构启示</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section id="allcases">
    <div class="sec-head">
      <div>
        <h2>5. 全量 210 Cases 完整交互式性能总表（L1 100 + L2 100 + L3 10）</h2>
        <p class="sec-desc">点击任意列标题排序，点击任意行展开 3-Seed 误差详情、抖动 IQR、冷编译耗时与 Agent 诊断</p>
      </div>
    </div>

    <div class="controls">
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button class="btn active tbl-suite-btn" data-suite="ALL" type="button">全部 (210)</button>
        <button class="btn tbl-suite-btn" data-suite="L1" type="button">L1 (100)</button>
        <button class="btn tbl-suite-btn" data-suite="L2" type="button">L2 (100)</button>
        <button class="btn tbl-suite-btn" data-suite="L3" type="button">L3 (10)</button>
      </div>
      <select class="select-input" id="catSelect"><option value="ALL">全部分类</option></select>
      <select class="select-input" id="spSelect">
        <option value="ALL">全部加速比</option>
        <option value="lt090">&lt; 0.90x (重点瓶颈)</option>
        <option value="090to098">0.90x – 0.98x (小幅落后)</option>
        <option value="lt098">&lt; 0.98x (全部待优化)</option>
        <option value="eq100">0.98x – 1.05x (持平)</option>
        <option value="1to2">1.05x – 2.00x</option>
        <option value="2to5">2.00x – 5.00x</option>
        <option value="5to15">5.00x – 15.0x</option>
        <option value="gt15">&gt; 15.0x</option>
      </select>
      <select class="select-input" id="bucketSelect">
        <option value="ALL">全部收益归因桶</option>
        <option value="A">桶A: 逼近硬件 SOL</option>
        <option value="B">桶B: 代数/结构化简</option>
        <option value="C">桶C: MPS 3D/转置短板</option>
        <option value="D">桶D: 消除循环/Dispatch</option>
      </select>
      <input class="search-input" id="searchInput" type="search" placeholder="搜索 ID、算子名、分类或 Agent 解读..." />
    </div>

    <div class="filter-summary" id="filterSummary"></div>

    <div class="table-wrap">
      <table id="allTable">
        <thead>
          <tr>
            <th class="sortable" data-col="suite">Suite</th>
            <th class="sortable num" data-col="id">ID</th>
            <th class="sortable" data-col="name">Task Name</th>
            <th class="sortable" data-col="cat">分类</th>
            <th class="sortable num" data-col="base_ms">Base MPS (ms)</th>
            <th class="sortable num" data-col="cand_ms">M1Triton (ms)</th>
            <th class="sortable num" data-col="speedup">Speedup</th>
            <th class="sortable num" data-col="dpf">DPF</th>
            <th class="sortable num" data-col="max_err">Max Err</th>
            <th class="sortable" data-col="bucket">归因</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <footer>
    <div>Generated directly from frozen SHA-256 artifacts: <code>.audit/roofline_m1max.json</code>, <code>eval_final_100.json</code>, <code>eval_final_l2_100.json</code>, <code>eval_final_l3_10.json</code>.</div>
    <div>M1Triton · Apple M1 Max · Zero External CDN · 100% Offline Reproducible</div>
  </footer>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const REPORT = __REPORT_DATA_JSON__;

(function () {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  const themeBtn = $("#themeBtn");
  themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    renderRoofline();
    renderCatChart();
    renderHistChart();
  });

  const hw = REPORT.hardware;
  const sm = REPORT.summary;

  $("#hwPills").innerHTML = [
    ["GPU", hw.device],
    ["STREAM 实测带宽", hw.bw_peak + " GB/s (理论400)"],
    ["FP32 SGMM 实测", hw.fp32_peak + " TFLOPS (理论10.4)"],
    ["FP16 SGMM 实测", hw.fp16_peak + " TFLOPS"],
    ["Roofline 拐点", hw.ridge + " FLOP/B"],
    ["SRAM / TG", hw.sram_kb + " KB / " + hw.max_tg + " threads"],
    ["系统", hw.platform],
  ].map(([k, v]) => `<span class="pill">${k}: <strong>${v}</strong></span>`).join("");

  const kpis = [
    {
      label: "总评测通过 (L1+L2+L3)",
      val: `${sm.total} / ${sm.total}`,
      sub: `击败 MPS: ${sm.beats}/${sm.total} (${(100*sm.beats/sm.total).toFixed(1)}%) · ≥0.98x: ${sm.ge98}/${sm.total} · 总 Geomean ${sm.overall_gm.toFixed(2)}x`,
      color: "var(--green)",
    },
    {
      label: "KernelBench Level 1 (100)",
      val: `${sm.l1_gm.toFixed(4)}x`,
      sub: `100/100 通过 · >1.0x: ${sm.l1_beats}/100 · ≥0.98x: ${sm.l1_ge98}/100 · 单算子基石`,
      color: "var(--accent)",
    },
    {
      label: "KernelBench Level 2 (100)",
      val: `${sm.l2_gm.toFixed(4)}x`,
      sub: `100/100 全部击败 MPS · 中位 DPF=${sm.l2_dpf_med} · 单核融合 ${sm.l2_dpf1}/100`,
      color: "var(--purple)",
    },
    {
      label: "Level 3 Complex Triton (10)",
      val: `${sm.l3_gm.toFixed(4)}x`,
      sub: `10/10 全部击败 MPS · 含 3 个 Fwd+Bwd Autograd (FA2 / LN / CE)`,
      color: "var(--amber)",
    },
  ];
  $("#kpiGrid").innerHTML = kpis.map(k => `
    <div class="kpi" style="--kpi-color:${k.color}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-val">${k.val}</div>
      <div class="kpi-sub">${k.sub}</div>
    </div>
  `).join("");

  const bucketColor = (b) => ({
    A: "var(--green)",
    B: "var(--purple)",
    C: "var(--amber)",
    D: "var(--cyan)",
    H: "var(--red)",
  }[b] || "var(--accent)");

  const bucketLabel = (b) => ({
    A: "桶A · 逼近SOL",
    B: "桶B · 代数化简",
    C: "桶C · MPS 3D短板",
    D: "桶D · 消循环/调度",
  }[b] || b);

  let rfShowMps = true;
  let rfSuite = "ALL";
  $("#rfToggleMps").addEventListener("click", (e) => {
    rfShowMps = !rfShowMps;
    e.target.classList.toggle("active", rfShowMps);
    renderRoofline();
  });
  $$(".rf-suite-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      rfSuite = btn.dataset.suite;
      $$(".rf-suite-btn").forEach(b => b.classList.toggle("active", b === btn));
      renderRoofline();
    });
  });

  const tip = $("#tooltip");
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.style.display = "block";
    const x = Math.min(window.innerWidth - 340, ev.clientX + 14);
    const y = Math.min(window.innerHeight - 200, ev.clientY + 14);
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  function renderRoofline() {
    const svg = $("#rooflineSvg");
    const W = 760, H = 430;
    const pad = { l: 64, r: 24, t: 24, b: 48 };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

    const minAI = 0.08, maxAI = 1200;
    const minTF = 0.015, maxTF = 16;
    const lx = (v) => pad.l + (Math.log10(Math.max(v, minAI)) - Math.log10(minAI)) / (Math.log10(maxAI) - Math.log10(minAI)) * pw;
    const ly = (v) => pad.t + ph - (Math.log10(Math.max(v, minTF)) - Math.log10(minTF)) / (Math.log10(maxTF) - Math.log10(minTF)) * ph;

    const gridColor = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
    const dimColor = getComputedStyle(document.documentElement).getPropertyValue("--text-dim").trim();
    const txtColor = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();

    let out = "";
    [0.1, 1, 10, 100, 1000].forEach(x => {
      const px = lx(x);
      out += `<line x1="${px}" y1="${pad.t}" x2="${px}" y2="${pad.t+ph}" stroke="${gridColor}" stroke-dasharray="2,3"/>`;
      out += `<text x="${px}" y="${H-26}" fill="${dimColor}" font-size="11" text-anchor="middle" font-family="var(--mono)">${x}</text>`;
    });
    [0.02, 0.1, 0.5, 1, 2, 5, 10].forEach(y => {
      const py = ly(y);
      out += `<line x1="${pad.l}" y1="${py}" x2="${pad.l+pw}" y2="${py}" stroke="${gridColor}" stroke-dasharray="2,3"/>`;
      out += `<text x="${pad.l-8}" y="${py+4}" fill="${dimColor}" font-size="11" text-anchor="end" font-family="var(--mono)">${y}</text>`;
    });

    out += `<text x="${pad.l+pw/2}" y="${H-6}" fill="${dimColor}" font-size="12" text-anchor="middle">Arithmetic Intensity (FLOP / Byte, log scale)</text>`;
    out += `<text x="16" y="${pad.t+ph/2}" fill="${dimColor}" font-size="12" text-anchor="middle" transform="rotate(-90 16 ${pad.t+ph/2})">Achieved Compute (TFLOPS, log scale)</text>`;

    const ridgeX = hw.fp32_peak * 1000 / hw.bw_peak;
    const thRidgeX = 10.4 * 1000 / 400.0;
    const roofPts = [minAI, ridgeX, maxAI].map(x => `${lx(x)},${ly(Math.min(hw.fp32_peak, x * hw.bw_peak / 1000))}`).join(" ");
    const thPts = [minAI, thRidgeX, maxAI].map(x => `${lx(x)},${ly(Math.min(10.4, x * 400.0 / 1000))}`).join(" ");

    out += `<polyline points="${thPts}" fill="none" stroke="${dimColor}" stroke-width="1.2" stroke-dasharray="5,5" opacity="0.5"/>`;
    out += `<polyline points="${roofPts}" fill="none" stroke="var(--accent)" stroke-width="2.6"/>`;
    out += `<line x1="${lx(ridgeX)}" y1="${ly(hw.fp32_peak)}" x2="${lx(ridgeX)}" y2="${pad.t+ph}" stroke="var(--accent)" stroke-width="1" stroke-dasharray="4,4" opacity="0.55"/>`;
    out += `<text x="${lx(ridgeX)+6}" y="${pad.t+ph-8}" fill="var(--accent)" font-size="11" font-family="var(--mono)">Ridge ${ridgeX.toFixed(1)} FLOP/B</text>`;
    out += `<text x="${lx(0.45)}" y="${ly(0.45*hw.bw_peak/1000)-10}" fill="var(--accent)" font-size="11" font-family="var(--mono)" transform="rotate(-29 ${lx(0.45)} ${ly(0.45*hw.bw_peak/1000)-10})">实测带宽 373.83 GB/s</text>`;
    out += `<text x="${lx(260)}" y="${ly(hw.fp32_peak)-8}" fill="var(--accent)" font-size="11" font-family="var(--mono)">实测 FP32 SGMM 10.171 TFLOPS</text>`;

    const pts = REPORT.roofline_points.filter(p => rfSuite === "ALL" || p.suite === rfSuite);
    const labelPos = {
      "L1 #25": [-9, -8, "end"],
      "L1 #89": [9, 4, "start"],
      "L1 #88": [-9, -7, "end"],
      "L1 #36": [9, 5, "start"],
      "L2 #42": [9, 7, "start"],
      "L1 #7": [-9, -8, "end"],
      "L1 #72": [-9, 4, "end"],
      "L1 #6": [-9, 14, "end"],
      "L2 #27": [9, 5, "start"],
      "L1 #97": [-9, -10, "end"],
      "L1 #66": [9, 5, "start"],
      "L3-07": [-9, 12, "end"],
      "L3-10": [9, 14, "start"],
      "L1 #1": [8, -10, "start"],
    };
    pts.forEach((p, idx) => {
      const cx = lx(p.ai);
      const cy = ly(p.cand_tf);
      const by = ly(p.base_tf);
      const col = bucketColor(p.bucket);
      if (rfShowMps) {
        out += `<line x1="${cx}" y1="${by}" x2="${cx}" y2="${cy}" stroke="${col}" stroke-width="1.3" stroke-dasharray="2,2" opacity="0.65"/>`;
        out += `<circle cx="${cx}" cy="${by}" r="4.2" fill="var(--bg-card)" stroke="${dimColor}" stroke-width="1.6" class="rf-pt" data-idx="${idx}"/>`;
      }
      out += `<circle cx="${cx}" cy="${cy}" r="6.2" fill="${col}" stroke="${txtColor}" stroke-width="1.2" class="rf-pt" data-idx="${idx}" style="cursor:pointer"/>`;
      const lp = labelPos[p.key];
      if (lp) {
        out += `<text x="${cx+lp[0]}" y="${cy+lp[1]}" fill="${txtColor}" font-size="10.5" text-anchor="${lp[2]}" font-family="var(--mono)">${p.key}</text>`;
      }
    });

    svg.innerHTML = out;
    svg.querySelectorAll(".rf-pt").forEach(el => {
      const p = pts[Number(el.dataset.idx)];
      el.addEventListener("mousemove", (ev) => {
        const unitStr = p.kind === "bw"
          ? `${p.cand_gbs} GB/s (MPS: ${p.base_gbs} GB/s)`
          : `${p.cand_tf} TFLOPS (MPS: ${p.base_tf} TF)`;
        showTip(ev, `
          <h4>${p.label}</h4>
          <div style="color:var(--text-dim);margin-bottom:6px;font-size:11.5px;">${p.shape}</div>
          <div class="kv"><span>算术强度 AI:</span><span>${p.ai} FLOP/B</span></div>
          <div class="kv"><span>实测吞吐:</span><span>${unitStr}</span></div>
          <div class="kv"><span>M1Triton % SOL:</span><strong style="color:${bucketColor(p.bucket)}">${p.sol_pct}%</strong></div>
          <div class="kv"><span>PyTorch MPS % SOL:</span><span>${p.base_sol_pct}%</span></div>
          <div class="kv"><span>耗时 (Base → M1T):</span><span>${p.base_ms}ms → ${p.cand_ms}ms</span></div>
          <div class="kv"><span>加速比:</span><strong style="color:${p.speedup>=1?'var(--green)':'var(--red)'}">${p.speedup}x</strong></div>
        `);
      });
      el.addEventListener("mouseleave", hideTip);
    });
  }

  $("#solTable tbody").innerHTML = REPORT.roofline_points.map(p => {
    const thr = p.kind === "bw" ? `${p.cand_gbs} GB/s` : `${p.cand_tf.toFixed(2)} TF`;
    const spClass = p.speedup >= 2 ? "b-green" : (p.speedup >= 1 ? "b-blue" : "b-red");
    const solColor = p.sol_pct >= 60 ? "var(--green)" : (p.sol_pct >= 30 ? "var(--amber)" : "var(--red)");
    return `<tr>
      <td><strong>${p.key}</strong> <span style="color:var(--text-dim)">${p.label.replace(p.key + " ", "")}</span></td>
      <td class="num">${p.ai}</td>
      <td class="num">${thr}</td>
      <td class="num" style="color:${solColor};font-weight:700">${p.sol_pct}%</td>
      <td class="num" style="color:var(--text-dim)">${p.base_sol_pct}%</td>
      <td class="num"><span class="badge ${spClass}">${p.speedup.toFixed(2)}x</span></td>
    </tr>`;
  }).join("");

  const bucketsMeta = [
    {
      code: "A", color: "var(--green)", title: "桶 A · 真实硬件 SOL 饱和",
      cnt: sm.bucket_counts.A,
      desc: "实打实跑出 58%–97% 带宽或算力极限：<code>L1 #25 Swish (97.3%)</code>、<code>#89 Cumsum (97.0%)</code>、<code>#1 Matmul (80.8%)</code>、<code>#97 FlashAttn (76.0%)</code>、<code>#6 Split-K (58.4%, 8.36x)</code>、<code>L3-10 W4A16 (58.1%, 3.05x)</code>。",
    },
    {
      code: "B", color: "var(--purple)", title: "桶 B · 精确代数/结构化简",
      cnt: sm.bucket_counts.B,
      desc: "利用零填充线性交换或三角稀疏直接消灭无效 FLOPs 与数 GB 中间张量：<code>L2 #77 (421.37x)</code>、<code>L2 #42 (32.57x, 78.1% 最小I/O SOL)</code>、<code>L1 #14/#15 (3.77–3.96x, 精确 ⅓N³ 仍达 53.5% SOL)</code>。",
    },
    {
      code: "C", color: "var(--amber)", title: "桶 C · 击穿 MPS 3D/转置短板",
      cnt: sm.bucket_counts.C,
      desc: "MPS 在 3D/转置卷积仅 0.5%–4.4% SOL；M1Triton 隐式 GEMM + <code>FUSE_SW</code> 拉回 11%–46% SOL：<code>L1 #66 (10.41x)</code>、<code>#72 (23.54x)</code>、<code>L2 #27 (22.04x)</code>、<code>#78 (20.39x)</code>。",
    },
    {
      code: "D", color: "var(--cyan)", title: "桶 D · 消除循环与碎片化调度",
      cnt: sm.bucket_counts.D,
      desc: "单核吞掉 Python 步进循环、动态块表寻址或碎片 kernel launch：<code>L3-08 Mamba SSM (66.58x)</code>、<code>L3-03 PagedAttn (12.42x)</code>、<code>L3-05 RMSNorm+RoPE+KV (7.68x)</code>、<code>L1 #100 Hinge (16.65x)</code>。",
    },
  ];
  $("#bucketGrid").innerHTML = bucketsMeta.map(b => `
    <div class="bucket-card" style="--bcolor:${b.color}">
      <h4><span>${b.title}</span><span class="badge b-gray">${b.cnt} cases</span></h4>
      <p>${b.desc}</p>
    </div>
  `).join("");

  let catSuite = "L1";
  $$(".cat-suite-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      catSuite = btn.dataset.suite;
      $$(".cat-suite-btn").forEach(b => b.classList.toggle("active", b === btn));
      renderCatChart();
    });
  });

  function renderCatChart() {
    const svg = $("#catChartSvg");
    const items = REPORT.cat_stats[catSuite];
    const W = 600, H = 330;
    const padL = 178, padR = 78, padT = 12, padB = 18;
    const maxV = Math.max(...items.map(i => i.geomean), 2.5);
    const rowH = Math.min(34, (H - padT - padB) / items.length);
    const dimColor = getComputedStyle(document.documentElement).getPropertyValue("--text-dim").trim();
    const txtColor = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();

    let out = "";
    items.forEach((it, idx) => {
      const y = padT + idx * rowH;
      const bw = Math.max(4, (it.geomean / maxV) * (W - padL - padR));
      const col = it.geomean >= 3 ? "var(--green)" : (it.geomean >= 1.5 ? "var(--accent)" : "var(--amber)");
      out += `<text x="${padL-10}" y="${y+rowH*0.62}" fill="${txtColor}" font-size="11.5" text-anchor="end">${it.cat} (${it.count})</text>`;
      out += `<rect x="${padL}" y="${y+rowH*0.2}" width="${bw}" height="${rowH*0.58}" rx="4" fill="${col}" opacity="0.88"/>`;
      out += `<text x="${padL+bw+7}" y="${y+rowH*0.62}" fill="${dimColor}" font-size="11.5" font-family="var(--mono)">${it.geomean.toFixed(2)}x (${it.beats}/${it.count})</text>`;
    });
    svg.innerHTML = out;
  }

  function renderHistChart() {
    const svg = $("#histChartSvg");
    const bins = [
      { label: "<0.90x", filter: "lt090", fn: s => s < 0.90, col: "var(--red)" },
      { label: "0.90-0.98", filter: "090to098", fn: s => s >= 0.90 && s < 0.98, col: "var(--amber)" },
      { label: "0.98-1.05", filter: "eq100", fn: s => s >= 0.98 && s <= 1.05, col: "var(--text-dim)" },
      { label: "1.05-2.0x", filter: "1to2", fn: s => s > 1.05 && s < 2.0, col: "var(--accent)" },
      { label: "2.0-5.0x", filter: "2to5", fn: s => s >= 2.0 && s <= 5.0, col: "var(--green)" },
      { label: "5.0-15x", filter: "5to15", fn: s => s > 5.0 && s <= 15.0, col: "var(--purple)" },
      { label: ">15x", filter: "gt15", fn: s => s > 15.0, col: "var(--purple)" },
    ];
    const counts = bins.map(b => REPORT.tasks.filter(t => b.fn(t.speedup)).length);
    const maxC = Math.max(...counts, 1);
    const W = 600, H = 330, padL = 36, padR = 16, padT = 26, padB = 42;
    const bw = (W - padL - padR) / bins.length;
    const dimColor = getComputedStyle(document.documentElement).getPropertyValue("--text-dim").trim();
    const txtColor = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();
    const gridColor = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();

    let out = `<line x1="${padL}" y1="${H-padB}" x2="${W-padR}" y2="${H-padB}" stroke="${gridColor}"/>`;
    bins.forEach((b, i) => {
      const c = counts[i];
      const bh = (c / maxC) * (H - padT - padB);
      const x = padL + i * bw + 8;
      const y = H - padB - bh;
      out += `<rect class="hist-bar" data-filter="${b.filter}" x="${x}" y="${y}" width="${bw-16}" height="${bh}" rx="5" fill="${b.col}" opacity="0.88" style="cursor:pointer"/>`;
      out += `<text x="${x+(bw-16)/2}" y="${y-6}" fill="${txtColor}" font-size="12" font-weight="700" text-anchor="middle" font-family="var(--mono)">${c}</text>`;
      out += `<text x="${x+(bw-16)/2}" y="${H-padB+18}" fill="${dimColor}" font-size="10.5" text-anchor="middle" font-family="var(--mono)">${b.label}</text>`;
    });
    svg.innerHTML = out;
    svg.querySelectorAll(".hist-bar").forEach(el => {
      el.addEventListener("click", () => {
        $("#spSelect").value = el.dataset.filter;
        tblState.sp = el.dataset.filter;
        renderTable();
        $("#allcases").scrollIntoView({ behavior: "smooth" });
      });
    });
  }

  let diveGroup = "all";
  function renderDives() {
    const list = REPORT.deep_dives.filter(d => diveGroup === "all" || d.group === diveGroup);
    const layerBadge = (code, txt) => {
      const cls = { compiler: "b-blue", dsl: "b-purple", kernel: "b-amber", sol: "b-green" }[code] || "b-gray";
      return `<span class="badge ${cls}">改进主抓手: ${txt}</span>`;
    };
    $("#diveGrid").innerHTML = list.map(d => `
      <div class="dive-card ${d.group === 'bottleneck' ? 'is-bottleneck' : 'is-winner'}">
        <div class="dive-top">
          <h4 class="dive-title">${d.title}</h4>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            ${layerBadge(d.layer_code, d.layer)}
          </div>
        </div>
        <div class="dive-metrics">
          <span class="badge ${d.speedup >= 1 ? 'b-green' : 'b-red'}">${d.speedup.toFixed(2)}x</span>
          <span>MPS ${d.base_ms}ms → M1T ${d.cand_ms}ms</span>
          <span style="color:var(--text-muted)">|</span>
          <span style="color:var(--accent)">${d.metric}</span>
        </div>
        <div class="dive-block"><strong>机制与源码归因：</strong>${d.why}</div>
        <div class="dive-headroom"><strong>剩余空间与 Co-Design 抓手：</strong>${d.headroom}</div>
      </div>
    `).join("");
  }
  $$(".dive-filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      diveGroup = btn.dataset.group;
      $$(".dive-filter-btn").forEach(b => b.classList.toggle("active", b === btn));
      renderDives();
    });
  });

  $("#expTable tbody").innerHTML = REPORT.experiments.map(e => {
    const vBadge = e.verdict === "keep" ? '<span class="badge b-green">KEEP</span>' : '<span class="badge b-red">REJECT (回退)</span>';
    return `<tr>
      <td style="white-space:nowrap"><strong>${e.phase}</strong><br><span style="color:var(--text-muted);font-family:var(--mono);font-size:11px">${e.premise}</span></td>
      <td>${e.hypothesis}</td>
      <td class="num" style="white-space:nowrap">${e.baseline} → <strong>${e.candidate}</strong><br><span style="color:var(--text-muted);font-size:11px">(${e.direction})</span></td>
      <td>${vBadge}</td>
      <td style="color:var(--text-dim)">${e.explanation}</td>
    </tr>`;
  }).join("");

  const tblState = { suite: "ALL", cat: "ALL", sp: "ALL", bucket: "ALL", q: "", sortCol: "suite", sortDir: 1 };
  const expanded = new Set();

  function populateCats() {
    const cats = Array.from(new Set(
      REPORT.tasks.filter(t => tblState.suite === "ALL" || t.suite === tblState.suite).map(t => t.cat)
    )).sort();
    const sel = $("#catSelect");
    const prev = sel.value;
    sel.innerHTML = `<option value="ALL">全部分类 (${cats.length})</option>` + cats.map(c => `<option value="${c}">${c}</option>`).join("");
    sel.value = cats.includes(prev) ? prev : "ALL";
    tblState.cat = sel.value;
  }

  function matchSpeedup(sp, mode) {
    if (mode === "ALL") return true;
    if (mode === "lt090") return sp < 0.90;
    if (mode === "090to098") return sp >= 0.90 && sp < 0.98;
    if (mode === "lt098") return sp < 0.98;
    if (mode === "eq100") return sp >= 0.98 && sp <= 1.05;
    if (mode === "1to2") return sp > 1.05 && sp < 2.0;
    if (mode === "2to5") return sp >= 2.0 && sp <= 5.0;
    if (mode === "5to15") return sp > 5.0 && sp <= 15.0;
    if (mode === "gt15") return sp > 15.0;
    return true;
  }

  function geomean(arr) {
    const pos = arr.filter(x => x > 0);
    if (!pos.length) return 0;
    return Math.exp(pos.reduce((a, b) => a + Math.log(b), 0) / pos.length);
  }

  function renderTable() {
    const q = tblState.q.trim().toLowerCase();
    const rows = REPORT.tasks.filter(t => {
      if (tblState.suite !== "ALL" && t.suite !== tblState.suite) return false;
      if (tblState.cat !== "ALL" && t.cat !== tblState.cat) return false;
      if (tblState.bucket !== "ALL" && t.bucket !== tblState.bucket) return false;
      if (!matchSpeedup(t.speedup, tblState.sp)) return false;
      if (q) {
        const hay = `${t.suite} #${t.id} ${t.name} ${t.cat} ${t.note} ${t.headroom}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    const col = tblState.sortCol, dir = tblState.sortDir;
    rows.sort((a, b) => {
      if (col === "suite") {
        if (a.suite !== b.suite) return a.suite.localeCompare(b.suite) * dir;
        return (a.id - b.id) * dir;
      }
      const va = a[col], vb = b[col];
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });

    const sps = rows.map(r => r.speedup);
    const gm = geomean(sps);
    const beats = sps.filter(s => s > 1.0).length;
    const sortedSp = [...sps].sort((a, b) => a - b);
    const med = sortedSp.length ? sortedSp[Math.floor(sortedSp.length / 2)] : 0;
    $("#filterSummary").innerHTML = `
      <span>当前筛选: <strong>${rows.length} / ${REPORT.tasks.length}</strong> cases</span>
      <span>Geomean: <strong style="color:var(--green)">${gm.toFixed(3)}x</strong></span>
      <span>Median: <strong>${med.toFixed(2)}x</strong></span>
      <span>&gt; 1.00x: <strong>${beats}/${rows.length} (${rows.length?(100*beats/rows.length).toFixed(1):0}%)</strong></span>
      <span style="color:var(--text-muted)">提示: 点击任意行展开 Seed 误差、抖动与 Agent 诊断</span>
    `;

    const tbody = $("#allTable tbody");
    const html = [];
    rows.forEach(r => {
      const rkey = `${r.suite}-${r.id}`;
      const spCls = r.speedup >= 2 ? "b-green" : (r.speedup >= 1.0 ? "b-blue" : (r.speedup >= 0.98 ? "b-gray" : "b-red"));
      const barPct = Math.min(100, (Math.log10(Math.max(r.speedup, 0.5) * 2) / Math.log10(40)) * 100);
      const barCol = r.speedup >= 2 ? "var(--green)" : (r.speedup >= 1 ? "var(--accent)" : (r.speedup >= 0.98 ? "var(--text-dim)" : "var(--red)"));
      const sBadge = r.suite === "L1" ? "b-blue" : (r.suite === "L2" ? "b-purple" : "b-amber");
      const errStr = r.max_err === 0 ? "0.00e+0" : r.max_err.toExponential(1);
      const hasNote = r.note ? ' <span title="含 Agent 深度解读" style="color:var(--accent)">★</span>' : "";
      const bwdTag = r.bwd ? ' <span class="badge b-amber">Fwd+Bwd</span>' : "";
      const cpuTag = r.backend !== "mps" ? ' <span class="badge b-red">CPU Ref</span>' : "";

      html.push(`<tr class="data-row" data-rkey="${rkey}">
        <td><span class="badge ${sBadge}">${r.suite}</span></td>
        <td class="num">#${r.id}</td>
        <td><code>${r.name}</code>${bwdTag}${cpuTag}${hasNote}</td>
        <td style="color:var(--text-dim)">${r.cat}</td>
        <td class="num">${r.base_ms.toFixed(3)}</td>
        <td class="num"><strong>${r.cand_ms.toFixed(3)}</strong></td>
        <td class="num sp-bar-cell">
          <span class="badge ${spCls}">${r.speedup.toFixed(2)}x</span>
          <div class="sp-bar-bg"><div class="sp-bar-fg" style="width:${barPct.toFixed(0)}%;background:${barCol}"></div></div>
        </td>
        <td class="num">${r.dpf}</td>
        <td class="num" style="color:var(--text-dim)">${errStr}</td>
        <td><span class="badge b-gray">${bucketLabel(r.bucket)}</span></td>
      </tr>`);

      if (expanded.has(rkey)) {
        const bwdErr = r.max_bwd_err !== null ? ` · Max Bwd Err: <code>${r.max_bwd_err.toExponential(2)}</code>` : "";
        const iqrInfo = r.base_iqr > 0 ? ` · Base IQR: <code>±${r.base_iqr.toFixed(3)}ms</code> · M1T IQR: <code>±${r.cand_iqr.toFixed(3)}ms</code>` : "";
        const agentSec = r.note
          ? `<div style="margin-top:8px;"><strong>Agent 解读：</strong>${r.note}</div><div style="margin-top:5px;"><strong>提升空间与抓手：</strong>${r.headroom}</div>`
          : `<div style="margin-top:6px;color:var(--text-muted)">归因分类：<strong>${bucketLabel(r.bucket)}</strong>${r.speedup < 0.98 ? " · 存在明显优化空间，可按第 4 节 Co-Design 矩阵归类优化" : (r.speedup >= 0.98 && r.speedup < 1.05 ? " · 已接近单算子硬件/基线持平区间，进一步提速需依赖 L2 跨算子融合" : " · 已显著超越 PyTorch MPS 基线")}</div>`;
        html.push(`<tr class="detail-row"><td colspan="10">
          <div>
            <strong>候选文件:</strong> <code>${r.file || "N/A"}</code> ·
            <strong>冷编译耗时:</strong> <code>${r.cold_ms} ms</code> ·
            <strong>3-Seed 总发射数:</strong> <code>${r.total_disp}</code> ·
            <strong>Baseline:</strong> <code>${r.backend}</code>${iqrInfo}${bwdErr}
          </div>
          ${agentSec}
        </td></tr>`);
      }
    });
    tbody.innerHTML = html.join("");

    tbody.querySelectorAll("tr.data-row").forEach(tr => {
      tr.addEventListener("click", () => {
        const k = tr.dataset.rkey;
        if (expanded.has(k)) expanded.delete(k); else expanded.add(k);
        renderTable();
      });
    });
  }

  $$(".tbl-suite-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tblState.suite = btn.dataset.suite;
      $$(".tbl-suite-btn").forEach(b => b.classList.toggle("active", b === btn));
      populateCats();
      renderTable();
    });
  });
  $("#catSelect").addEventListener("change", (e) => { tblState.cat = e.target.value; renderTable(); });
  $("#spSelect").addEventListener("change", (e) => { tblState.sp = e.target.value; renderTable(); });
  $("#bucketSelect").addEventListener("change", (e) => { tblState.bucket = e.target.value; renderTable(); });
  $("#searchInput").addEventListener("input", (e) => { tblState.q = e.target.value; renderTable(); });
  $$("#allTable th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const c = th.dataset.col;
      if (tblState.sortCol === c) tblState.sortDir *= -1;
      else { tblState.sortCol = c; tblState.sortDir = (c === "speedup" || c === "base_ms" || c === "cand_ms") ? -1 : 1; }
      renderTable();
    });
  });

  renderRoofline();
  renderCatChart();
  renderHistChart();
  renderDives();
  populateCats();
  renderTable();
})();
</script>
</body>
</html>
"""


def main() -> None:
    html = build_report()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()

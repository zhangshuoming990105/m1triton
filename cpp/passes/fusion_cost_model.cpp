#include "passes/fusion_cost_model.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <variant>
#include <vector>

#include "passes/threadgroup_staging.hpp"

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

constexpr uint32_t kMaxThreadsPerCore = 2048u;
constexpr uint32_t kRegsPerCore = 65536u;
constexpr uint32_t kBaseAddressingRegs = 16u;
constexpr uint32_t kSgmmScratchRegs = 8u;
constexpr uint32_t kTailOverheadRegs = 6u;
constexpr uint32_t kMaxFragsPerSimdgroup = 16u;

uint32_t round_down_pow2(uint32_t v) {
  if (v == 0u) return 1u;
  uint32_t p = 1u;
  while (p <= (v >> 1u)) {
    p <<= 1u;
  }
  return p;
}

bool is_transcendental_epilogue_op(EpilogueOpKind op) {
  switch (op) {
    case EpilogueOpKind::GELU:
    case EpilogueOpKind::GELUTanh:
    case EpilogueOpKind::Mish:
    case EpilogueOpKind::Tanh:
    case EpilogueOpKind::Sigmoid:
    case EpilogueOpKind::Swish:
    case EpilogueOpKind::Exp:
    case EpilogueOpKind::Log:
    case EpilogueOpKind::Softplus:
    case EpilogueOpKind::ELU:
    case EpilogueOpKind::SELU:
      return true;
    default:
      return false;
  }
}

double epilogue_step_flops(EpilogueOpKind op) {
  if (op == EpilogueOpKind::SaveReg) return 0.0;
  if (is_transcendental_epilogue_op(op)) return 8.0;
  switch (op) {
    case EpilogueOpKind::HardSwish:
    case EpilogueOpKind::HardSigmoid:
    case EpilogueOpKind::Softsign:
    case EpilogueOpKind::LeakyReLU:
    case EpilogueOpKind::Clamp:
    case EpilogueOpKind::HardTanh:
      return 3.0;
    default:
      return 1.0;
  }
}

struct EpilogueSummary {
  uint32_t num_steps = 0;
  uint32_t distinct_save_regs = 0;
  bool has_transcendental = false;
  bool has_fused_tail = false;
  double epilogue_flops = 0.0;

  bool is_deep_epilogue() const {
    return num_steps >= 4u || (has_transcendental && distinct_save_regs > 0u);
  }

  uint32_t extra_regs() const {
    uint32_t regs = 0u;
    if (num_steps > 0u) {
      regs += 2u * distinct_save_regs;
      if (has_transcendental) regs += 4u;
      regs += num_steps / 4u;
    }
    if (has_fused_tail) {
      regs += kTailOverheadRegs;
    }
    return regs;
  }
};

EpilogueSummary summarize_epilogue_hints(const std::optional<EpilogueDAGAttr>& dag,
                                         const std::optional<FusedTailAttr>& tail) {
  EpilogueSummary out;
  if (dag.has_value()) {
    out.num_steps = static_cast<uint32_t>(dag->steps.size());
    std::unordered_set<int32_t> save_slots;
    for (const auto& step : dag->steps) {
      if (step.op == EpilogueOpKind::SaveReg) {
        save_slots.insert(step.arg_idx);
      }
      if (is_transcendental_epilogue_op(step.op)) {
        out.has_transcendental = true;
      }
      out.epilogue_flops += epilogue_step_flops(step.op);
    }
    out.distinct_save_regs = static_cast<uint32_t>(save_slots.size());
  }
  if (tail.has_value() && tail->kind != TailKind::None) {
    out.has_fused_tail = true;
  }
  return out;
}

EpilogueSummary collect_kernel_epilogue(const Kernel& kernel) {
  const std::optional<EpilogueDAGAttr>* dag = kernel.hints.epilogue_dag.has_value() ? &kernel.hints.epilogue_dag : nullptr;
  const std::optional<FusedTailAttr>* tail =
      (kernel.hints.fused_tail.has_value() && kernel.hints.fused_tail->kind != TailKind::None)
          ? &kernel.hints.fused_tail
          : nullptr;

  for (const auto& op_ptr : kernel.ops) {
    if (!op_ptr || op_ptr->erased) continue;
    if (dag == nullptr && op_ptr->hints.epilogue_dag.has_value()) {
      dag = &op_ptr->hints.epilogue_dag;
    }
    if (tail == nullptr && op_ptr->hints.fused_tail.has_value() && op_ptr->hints.fused_tail->kind != TailKind::None) {
      tail = &op_ptr->hints.fused_tail;
    }
  }

  return summarize_epilogue_hints(dag != nullptr ? *dag : std::optional<EpilogueDAGAttr>{},
                                  tail != nullptr ? *tail : std::optional<FusedTailAttr>{});
}

ScalarDType infer_ptr_operand_dtype(const Kernel& kernel, const Op& op, size_t idx, ScalarDType fallback) {
  if (idx >= op.operands.size()) return fallback;
  const ValueId vid = op.operands[idx];
  if (vid == kInvalidId || vid >= kernel.values.size()) return fallback;
  const Type& ty = kernel.value(vid).type;
  if (ty.is_ptr()) {
    return ty.as_ptr().pointee;
  }
  if (ty.is_tile() && !ty.is_ptr_elem()) {
    return ty.scalar_elem();
  }
  return fallback;
}

ScalarDType infer_kernel_float_dtype(const Kernel& kernel) {
  for (const auto& p : kernel.params) {
    if (p.type.is_ptr()) {
      const ScalarDType pt = p.type.as_ptr().pointee;
      if (pt == ScalarDType::F16 || pt == ScalarDType::F32) return pt;
    } else if (p.type.is_tile() && p.type.is_float_elem()) {
      return p.type.scalar_elem();
    }
  }
  return ScalarDType::F32;
}

bool find_sg_grid(uint32_t bm, uint32_t bn, uint32_t num_sg, uint32_t sgmm_m, uint32_t sgmm_n,
                  uint32_t& out_sg_m, uint32_t& out_sg_n) {
  if (num_sg == 0u || bm == 0u || bn == 0u || sgmm_m == 0u || sgmm_n == 0u) return false;

  bool found = false;
  uint32_t best_diff = UINT32_MAX;
  uint32_t best_m = 1u;
  uint32_t best_n = num_sg;

  for (uint32_t sg_m = 1u; sg_m <= num_sg; ++sg_m) {
    if ((num_sg % sg_m) != 0u) continue;
    const uint32_t sg_n = num_sg / sg_m;
    const uint64_t unit_m = static_cast<uint64_t>(sg_m) * sgmm_m;
    const uint64_t unit_n = static_cast<uint64_t>(sg_n) * sgmm_n;
    if ((bm % unit_m) != 0u || (bn % unit_n) != 0u) continue;

    const uint32_t fm = static_cast<uint32_t>(bm / unit_m);
    const uint32_t fn = static_cast<uint32_t>(bn / unit_n);
    const uint32_t diff = (fm >= fn) ? (fm - fn) : (fn - fm);
    const bool prefer_aspect = (bm >= bn) ? (sg_m >= sg_n) : (sg_n >= sg_m);
    const bool best_aspect = (bm >= bn) ? (best_m >= best_n) : (best_n >= best_m);

    if (!found || diff < best_diff || (diff == best_diff && prefer_aspect && !best_aspect)) {
      found = true;
      best_diff = diff;
      best_m = sg_m;
      best_n = sg_n;
    }
  }

  if (!found) return false;
  out_sg_m = best_m;
  out_sg_n = best_n;
  return true;
}

void fallback_sg_grid(uint32_t bm, uint32_t bn, uint32_t num_sg, uint32_t& out_sg_m, uint32_t& out_sg_n) {
  if (num_sg <= 1u) {
    out_sg_m = 1u;
    out_sg_n = 1u;
    return;
  }
  uint32_t best_m = 1u;
  uint32_t best_n = num_sg;
  uint32_t best_gap = num_sg;
  for (uint32_t sg_m = 1u; sg_m <= num_sg; ++sg_m) {
    if ((num_sg % sg_m) != 0u) continue;
    const uint32_t sg_n = num_sg / sg_m;
    const uint32_t gap = (sg_m >= sg_n) ? (sg_m - sg_n) : (sg_n - sg_m);
    const bool prefer = (bm >= bn) ? (sg_m >= sg_n) : (sg_n >= sg_m);
    const bool best_pref = (bm >= bn) ? (best_m >= best_n) : (best_n >= best_m);
    if (gap < best_gap || (gap == best_gap && prefer && !best_pref)) {
      best_gap = gap;
      best_m = sg_m;
      best_n = sg_n;
    }
  }
  out_sg_m = best_m;
  out_sg_n = best_n;
}

struct ComputeTileInfo {
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t num_simdgroups = 4;
  bool double_buffer = false;
  ScalarDType dt_a = ScalarDType::F32;
  ScalarDType dt_b = ScalarDType::F32;
};

struct KernelScanInfo {
  uint64_t existing_alloc_bytes = 0;
  std::vector<ComputeTileInfo> compute_ops;
  bool has_flash_attn = false;
  uint64_t flash_attn_sram_bytes = 0;
  uint32_t flash_attn_tg_threads = 0;
  uint32_t flash_attn_bm = 16;
  uint32_t flash_attn_bn = 64;
  uint32_t flash_attn_bd = 32;
  ScalarDType flash_attn_dt = ScalarDType::F32;
  bool has_scan_op = false;
  bool has_depthwise_conv = false;
  uint32_t fallback_tg_threads = 0;
  uint32_t pointwise_op_count = 0;
};

void scan_block_for_cost(const Kernel& kernel, BlockId blk_id, const TargetSpec& target, KernelScanInfo& info) {
  const uint32_t simd_width = std::max<uint32_t>(1u, target.simd_width);
  const Block& blk = kernel.block(blk_id);

  for (ValueId arg_id : blk.args) {
    const Type& ty = kernel.value(arg_id).type;
    if (ty.is_tile() && ty.as_tile().layout.kind != LayoutKind::Unassigned) {
      const auto& l = ty.as_tile().layout;
      const uint32_t t = l.tg_threads[0] * l.tg_threads[1] * l.tg_threads[2];
      info.fallback_tg_threads = std::max(info.fallback_tg_threads, t);
    }
  }

  for (OpId op_id : blk.ops) {
    const Op& node = kernel.op(op_id);
    if (node.erased) continue;

    for (ValueId res_id : node.results) {
      const Type& ty = kernel.value(res_id).type;
      if (ty.is_tile() && ty.as_tile().layout.kind != LayoutKind::Unassigned) {
        const auto& l = ty.as_tile().layout;
        const uint32_t t = l.tg_threads[0] * l.tg_threads[1] * l.tg_threads[2];
        info.fallback_tg_threads = std::max(info.fallback_tg_threads, t);
      }
    }

    if (node.kind == OpKind::ThreadgroupAlloc && std::holds_alternative<ThreadgroupAllocAttr>(node.attr)) {
      const auto& alloc = std::get<ThreadgroupAllocAttr>(node.attr);
      if (!alloc.shape.empty() && alloc.stages > 0u) {
        uint64_t elems = 1u;
        for (size_t i = 0; i + 1 < alloc.shape.size(); ++i) {
          elems *= static_cast<uint64_t>(alloc.shape[i]);
        }
        elems *= static_cast<uint64_t>(alloc.shape.back()) + alloc.pad_elems;
        info.existing_alloc_bytes += elems * dtype_bytes(alloc.elem) * alloc.stages;
      }
    } else if (node.kind == OpKind::Dot && node.operands.size() >= 2 && !node.results.empty() &&
               std::holds_alternative<DotAttr>(node.attr)) {
      const auto& da = std::get<DotAttr>(node.attr);
      const Type& lty = kernel.value(node.operands[0]).type;
      const Type& rty = kernel.value(node.operands[1]).type;
      const Type& oty = kernel.value(node.results[0]).type;
      if (lty.is_tile() && rty.is_tile() && oty.is_tile() && oty.as_tile().shape.size() >= 2) {
        const auto& ls = lty.as_tile().shape;
        const auto& rs = rty.as_tile().shape;
        const auto& os = oty.as_tile().shape;
        const size_t off = (os.size() == 3) ? 1 : 0;
        const uint32_t shape_bm = static_cast<uint32_t>(da.transpose_a ? ls[off + 1] : ls[off]);
        const uint32_t shape_bk = static_cast<uint32_t>(da.transpose_a ? ls[off] : ls[off + 1]);
        const uint32_t shape_bn = static_cast<uint32_t>(da.transpose_b ? rs[off] : rs[off + 1]);

        ComputeTileInfo ci;
        ci.dt_a = lty.is_ptr_elem() ? ScalarDType::F32 : lty.scalar_elem();
        ci.dt_b = rty.is_ptr_elem() ? ScalarDType::F32 : rty.scalar_elem();

        const std::optional<TileSpecAttr>& ts =
            node.hints.tile_spec.has_value() ? node.hints.tile_spec : kernel.hints.tile_spec;
        if (ts.has_value()) {
          ci.bm = ts->bm > 0u ? ts->bm : shape_bm;
          ci.bn = ts->bn > 0u ? ts->bn : shape_bn;
          ci.bk = ts->bk > 0u ? ts->bk : shape_bk;
          ci.num_simdgroups =
              ts->num_simdgroups > 0u ? ts->num_simdgroups : ((ci.bm >= 128u && ci.bn >= 64u) ? 8u : 4u);
          ci.double_buffer = ts->double_buffer;
        } else {
          ci.bm = shape_bm;
          ci.bn = shape_bn;
          ci.bk = shape_bk;
          const LayoutAttr& out_l = oty.as_tile().layout;
          if (out_l.kind == LayoutKind::SimdgroupFragment && out_l.simd_grid[0] > 0u && out_l.simd_grid[1] > 0u) {
            ci.num_simdgroups = out_l.simd_grid[0] * out_l.simd_grid[1];
          } else {
            ci.num_simdgroups = (ci.bm >= 128u && ci.bn >= 64u) ? 8u : 4u;
          }
          const LayoutAttr& lhs_l = lty.as_tile().layout;
          ci.double_buffer = (lhs_l.kind == LayoutKind::ThreadgroupStaged && lhs_l.stages >= 2u);
        }
        info.compute_ops.push_back(ci);
      }
    } else if (node.kind == OpKind::ImplicitGemmConv && std::holds_alternative<ConvAttr>(node.attr)) {
      const auto& ca = std::get<ConvAttr>(node.attr);
      if (ca.is_depthwise) {
        info.has_depthwise_conv = true;
        info.fallback_tg_threads =
            std::max(info.fallback_tg_threads, std::max<uint32_t>(1u, ca.num_simdgroups) * simd_width);
      } else {
        ComputeTileInfo ci;
        const ScalarDType dt = infer_ptr_operand_dtype(kernel, node, 0, ScalarDType::F32);
        ci.dt_a = dt;
        ci.dt_b = dt;
        const std::optional<TileSpecAttr>& ts =
            node.hints.tile_spec.has_value() ? node.hints.tile_spec : kernel.hints.tile_spec;
        ci.bm = (ts.has_value() && ts->bm > 0u) ? ts->bm : ca.bm;
        ci.bn = (ts.has_value() && ts->bn > 0u) ? ts->bn : ca.bn;
        ci.bk = (ts.has_value() && ts->bk > 0u) ? ts->bk : ca.bk;
        ci.num_simdgroups = (ts.has_value() && ts->num_simdgroups > 0u) ? ts->num_simdgroups : ca.num_simdgroups;
        ci.double_buffer = ts.has_value() && ts->double_buffer;
        info.compute_ops.push_back(ci);
      }
    } else if (node.kind == OpKind::GroupedGemm && std::holds_alternative<GroupedGemmAttr>(node.attr)) {
      const auto& gg = std::get<GroupedGemmAttr>(node.attr);
      ComputeTileInfo ci;
      const ScalarDType dt = infer_ptr_operand_dtype(kernel, node, 0, ScalarDType::F32);
      ci.dt_a = dt;
      ci.dt_b = dt;
      const std::optional<TileSpecAttr>& ts =
          node.hints.tile_spec.has_value() ? node.hints.tile_spec : kernel.hints.tile_spec;
      ci.bm = (ts.has_value() && ts->bm > 0u) ? ts->bm : gg.bm;
      ci.bn = (ts.has_value() && ts->bn > 0u) ? ts->bn : gg.bn;
      ci.bk = (ts.has_value() && ts->bk > 0u) ? ts->bk : gg.bk;
      ci.num_simdgroups = (ts.has_value() && ts->num_simdgroups > 0u) ? ts->num_simdgroups : gg.num_simdgroups;
      ci.double_buffer = ts.has_value() && ts->double_buffer;
      info.compute_ops.push_back(ci);
    } else if (node.kind == OpKind::QuantDot && std::holds_alternative<QuantDotAttr>(node.attr)) {
      const auto& qd = std::get<QuantDotAttr>(node.attr);
      ComputeTileInfo ci;
      const ScalarDType dt = infer_ptr_operand_dtype(kernel, node, 0, ScalarDType::F16);
      ci.dt_a = dt;
      ci.dt_b = dt;
      const std::optional<TileSpecAttr>& ts =
          node.hints.tile_spec.has_value() ? node.hints.tile_spec : kernel.hints.tile_spec;
      ci.bm = (ts.has_value() && ts->bm > 0u) ? ts->bm : qd.bm;
      ci.bn = (ts.has_value() && ts->bn > 0u) ? ts->bn : qd.bn;
      ci.bk = (ts.has_value() && ts->bk > 0u) ? ts->bk : qd.bk;
      ci.num_simdgroups = (ts.has_value() && ts->num_simdgroups > 0u) ? ts->num_simdgroups : qd.num_simdgroups;
      ci.double_buffer = ts.has_value() && ts->double_buffer;
      info.compute_ops.push_back(ci);
    } else if (node.kind == OpKind::FlashAttentionTile && std::holds_alternative<FlashAttnAttr>(node.attr)) {
      const auto& fa = std::get<FlashAttnAttr>(node.attr);
      info.has_flash_attn = true;
      info.flash_attn_bm = fa.bm;
      info.flash_attn_bn = fa.bn;
      info.flash_attn_bd = fa.bd;
      info.flash_attn_dt = infer_ptr_operand_dtype(kernel, node, 0, ScalarDType::F32);
      info.flash_attn_tg_threads = std::max<uint32_t>(1u, fa.num_simdgroups) * simd_width;
      const uint64_t ldp = static_cast<uint64_t>(fa.bn) + 4u;
      info.flash_attn_sram_bytes += (static_cast<uint64_t>(fa.bm) * ldp + 3u * fa.bm) * 4u;
    } else if (node.kind == OpKind::Scan) {
      info.has_scan_op = true;
    } else if (node.kind == OpKind::WelfordReduce && std::holds_alternative<WelfordNormAttr>(node.attr)) {
      info.fallback_tg_threads =
          std::max(info.fallback_tg_threads, std::get<WelfordNormAttr>(node.attr).tg_threads);
    } else if (node.kind == OpKind::OnlineSoftmax && std::holds_alternative<OnlineSoftmaxAttr>(node.attr)) {
      info.fallback_tg_threads =
          std::max(info.fallback_tg_threads, std::get<OnlineSoftmaxAttr>(node.attr).tg_threads);
    } else if (node.kind == OpKind::FusedLoss && std::holds_alternative<LossAttr>(node.attr)) {
      info.fallback_tg_threads = std::max(info.fallback_tg_threads, std::get<LossAttr>(node.attr).tg_threads);
    } else if (node.kind == OpKind::Unary || node.kind == OpKind::Binary || node.kind == OpKind::Select) {
      ++info.pointwise_op_count;
    }

    for (BlockId reg_id : node.regions) {
      scan_block_for_cost(kernel, reg_id, target, info);
    }
  }
}

bool clamp_tile_spec(TileSpecAttr& ts, ScalarDType dt_a, ScalarDType dt_b, uint64_t existing_alloc_bytes,
                     const EpilogueSummary& epilogue, const TargetSpec& target, bool sgmm_guards,
                     FusionCostReport* flags) {
  const TileSpecAttr before = ts;
  const uint32_t simd_width = std::max<uint32_t>(1u, target.simd_width);
  const uint32_t sgmm_m = std::max<uint32_t>(1u, target.sgmm_m);
  const uint32_t sgmm_n = std::max<uint32_t>(1u, target.sgmm_n);
  const uint32_t max_tg = std::max<uint32_t>(simd_width, target.max_threads_per_threadgroup);
  const uint32_t max_sg_pow2 = round_down_pow2(std::max<uint32_t>(1u, max_tg / simd_width));
  const uint64_t max_sram = target.max_threadgroup_memory_bytes;

  // PSO threadgroup limit clamp applies to every kernel kind.
  if (static_cast<uint64_t>(ts.num_simdgroups) * simd_width > max_tg) {
    ts.num_simdgroups = max_sg_pow2;
    if (flags != nullptr) flags->clamped_simdgroups = true;
  }

  if (!sgmm_guards || ts.bm == 0u || ts.bn == 0u || ts.bk == 0u) {
    return !(ts == before);
  }

  const uint32_t pad_a = ThreadgroupStagingAndSwizzlePass::recommended_pad_elems(dt_a);
  const uint32_t pad_b = ThreadgroupStagingAndSwizzlePass::recommended_pad_elems(dt_b);

  auto total_sram_for = [&](uint32_t bm, uint32_t bn, uint32_t bk, uint32_t stages) -> uint64_t {
    const uint64_t staging =
        ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(bm, bn, bk, dt_a, dt_b, pad_a, pad_b, stages);
    const uint64_t tail = epilogue.has_fused_tail ? static_cast<uint64_t>(bm) * 4u : 0u;
    return existing_alloc_bytes + staging + tail;
  };

  // Double-buffer SRAM overflow guard.
  if (ts.double_buffer && total_sram_for(ts.bm, ts.bn, ts.bk, 2u) > max_sram) {
    ts.double_buffer = false;
    if (flags != nullptr) flags->disabled_double_buffer = true;
  }

  // Single-buffer SRAM overflow guard: halve bk down to >= 8, then bn/bm down to >= 32.
  if (total_sram_for(ts.bm, ts.bn, ts.bk, 1u) > max_sram) {
    while (total_sram_for(ts.bm, ts.bn, ts.bk, 1u) > max_sram && ts.bk > 8u) {
      const uint32_t next_bk = std::max<uint32_t>(8u, ts.bk / 2u);
      if (next_bk == ts.bk) break;
      ts.bk = next_bk;
      if (flags != nullptr) flags->clamped_tile = true;
    }
    while (total_sram_for(ts.bm, ts.bn, ts.bk, 1u) > max_sram && (ts.bn > 32u || ts.bm > 32u)) {
      if (ts.bn >= ts.bm && ts.bn > 32u) {
        const uint32_t next_bn = std::max<uint32_t>(32u, ts.bn / 2u);
        if (next_bn == ts.bn) break;
        ts.bn = next_bn;
        if (flags != nullptr) flags->clamped_tile = true;
      } else if (ts.bm > 32u) {
        const uint32_t next_bm = std::max<uint32_t>(32u, ts.bm / 2u);
        if (next_bm == ts.bm) break;
        ts.bm = next_bm;
        if (flags != nullptr) flags->clamped_tile = true;
      } else {
        break;
      }
    }
  }

  // Register-pressure occupancy guard for deep epilogues (>16 8x8 fragments/simdgroup).
  if (epilogue.is_deep_epilogue() && ts.num_simdgroups > 0u) {
    const uint64_t frag_unit = static_cast<uint64_t>(sgmm_m) * sgmm_n;
    const uint64_t tile_area = static_cast<uint64_t>(ts.bm) * ts.bn;
    while (tile_area > static_cast<uint64_t>(kMaxFragsPerSimdgroup) * ts.num_simdgroups * frag_unit) {
      const uint64_t next_sg_64 = static_cast<uint64_t>(ts.num_simdgroups) * 2u;
      if (next_sg_64 * simd_width > max_tg) break;
      const uint32_t next_sg = static_cast<uint32_t>(next_sg_64);
      uint32_t sg_m = 1u;
      uint32_t sg_n = 1u;
      if (!find_sg_grid(ts.bm, ts.bn, next_sg, sgmm_m, sgmm_n, sg_m, sg_n)) break;
      ts.num_simdgroups = next_sg;
      if (flags != nullptr) flags->clamped_simdgroups = true;
    }
    if (tile_area > static_cast<uint64_t>(kMaxFragsPerSimdgroup) * ts.num_simdgroups * frag_unit &&
        ts.double_buffer) {
      ts.double_buffer = false;
      if (flags != nullptr) flags->disabled_double_buffer = true;
    }
  }

  return !(ts == before);
}

}  // namespace

FusionCostReport FusionOccupancyCostModelPass::analyze(const ir::Kernel& kernel, const ir::TargetSpec& target) {
  FusionCostReport report;
  const uint32_t simd_width = std::max<uint32_t>(1u, target.simd_width);
  const uint32_t sgmm_m = std::max<uint32_t>(1u, target.sgmm_m);
  const uint32_t sgmm_n = std::max<uint32_t>(1u, target.sgmm_n);
  const uint32_t max_tg = std::max<uint32_t>(simd_width, target.max_threads_per_threadgroup);
  const uint64_t max_sram = target.max_threadgroup_memory_bytes;

  KernelScanInfo scan;
  if (kernel.entry_block != kInvalidId && kernel.entry_block < kernel.blocks.size()) {
    scan_block_for_cost(kernel, kernel.entry_block, target, scan);
  }

  const EpilogueSummary epilogue = collect_kernel_epilogue(kernel);
  report.fused_epilogue_ops = epilogue.num_steps;
  report.epilogue_saved_regs = epilogue.distinct_save_regs;
  report.has_fused_tail = epilogue.has_fused_tail;

  if (scan.compute_ops.empty() && !scan.has_flash_attn && !scan.has_depthwise_conv &&
      kernel.hints.tile_spec.has_value()) {
    const auto& ts = *kernel.hints.tile_spec;
    const bool is_coop_row_scan = scan.has_scan_op && ts.bm <= 1u;
    if (!is_coop_row_scan && ts.bm > 0u && ts.bn > 0u && ts.bk > 0u) {
      ComputeTileInfo ci;
      ci.bm = ts.bm;
      ci.bn = ts.bn;
      ci.bk = ts.bk;
      ci.num_simdgroups = std::max<uint32_t>(1u, ts.num_simdgroups);
      ci.double_buffer = ts.double_buffer;
      const ScalarDType dt = infer_kernel_float_dtype(kernel);
      ci.dt_a = dt;
      ci.dt_b = dt;
      scan.compute_ops.push_back(ci);
    }
  }

  if (!scan.compute_ops.empty()) {
    const ComputeTileInfo* primary = &scan.compute_ops[0];
    uint64_t staging_sram = 0u;
    for (const auto& c : scan.compute_ops) {
      const uint32_t pad_a = ThreadgroupStagingAndSwizzlePass::recommended_pad_elems(c.dt_a);
      const uint32_t pad_b = ThreadgroupStagingAndSwizzlePass::recommended_pad_elems(c.dt_b);
      const uint32_t stages = c.double_buffer ? 2u : 1u;
      staging_sram +=
          ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(c.bm, c.bn, c.bk, c.dt_a, c.dt_b, pad_a, pad_b, stages);
      if (static_cast<uint64_t>(c.bm) * c.bn > static_cast<uint64_t>(primary->bm) * primary->bn) {
        primary = &c;
      }
    }

    const uint64_t tail_sram = epilogue.has_fused_tail ? static_cast<uint64_t>(primary->bm) * 4u : 0u;
    report.estimated_sram_bytes = scan.existing_alloc_bytes + staging_sram + tail_sram;

    const uint32_t num_sg = std::max<uint32_t>(1u, primary->num_simdgroups);
    report.threads_per_threadgroup = num_sg * simd_width;

    uint32_t sg_m = 1u;
    uint32_t sg_n = 1u;
    if (!find_sg_grid(primary->bm, primary->bn, num_sg, sgmm_m, sgmm_n, sg_m, sg_n)) {
      fallback_sg_grid(primary->bm, primary->bn, num_sg, sg_m, sg_n);
    }
    const uint32_t unit_m = std::max<uint32_t>(1u, sg_m * sgmm_m);
    const uint32_t unit_n = std::max<uint32_t>(1u, sg_n * sgmm_n);
    const uint32_t frags_m = std::max<uint32_t>(1u, (primary->bm + unit_m - 1u) / unit_m);
    const uint32_t frags_n = std::max<uint32_t>(1u, (primary->bn + unit_n - 1u) / unit_n);
    report.estimated_registers_per_thread =
        kBaseAddressingRegs + (2u * frags_m * frags_n + kSgmmScratchRegs) + epilogue.extra_regs();

    const double elem_bytes =
        0.5 * static_cast<double>(dtype_bytes(primary->dt_a) + dtype_bytes(primary->dt_b));
    const double denom = (static_cast<double>(primary->bm) * primary->bk +
                          static_cast<double>(primary->bk) * primary->bn) *
                         elem_bytes;
    if (denom > 0.0) {
      report.arithmetic_intensity =
          (2.0 * static_cast<double>(primary->bm) * primary->bn * primary->bk) / denom;
    }
  } else if (scan.has_flash_attn) {
    const uint64_t tail_sram = epilogue.has_fused_tail ? static_cast<uint64_t>(scan.flash_attn_bm) * 4u : 0u;
    report.estimated_sram_bytes = scan.existing_alloc_bytes + scan.flash_attn_sram_bytes + tail_sram;
    report.threads_per_threadgroup = std::max<uint32_t>(simd_width, scan.flash_attn_tg_threads);
    report.estimated_registers_per_thread = kBaseAddressingRegs + 32u + epilogue.extra_regs();
    const double elem_bytes = static_cast<double>(dtype_bytes(scan.flash_attn_dt));
    const double denom = (static_cast<double>(scan.flash_attn_bm) * scan.flash_attn_bd +
                          2.0 * static_cast<double>(scan.flash_attn_bn) * scan.flash_attn_bd) *
                         elem_bytes;
    if (denom > 0.0) {
      report.arithmetic_intensity =
          (4.0 * static_cast<double>(scan.flash_attn_bm) * scan.flash_attn_bn * scan.flash_attn_bd) / denom;
    }
  } else {
    const uint64_t tail_sram = epilogue.has_fused_tail ? 64u * 4u : 0u;
    report.estimated_sram_bytes = scan.existing_alloc_bytes + tail_sram;
    report.threads_per_threadgroup =
        scan.fallback_tg_threads > 0u ? scan.fallback_tg_threads : std::min<uint32_t>(256u, max_tg);
    report.estimated_registers_per_thread = kBaseAddressingRegs + epilogue.extra_regs();
    const double pw_flops =
        epilogue.epilogue_flops > 0.0 ? epilogue.epilogue_flops : static_cast<double>(scan.pointwise_op_count);
    report.arithmetic_intensity = pw_flops / 8.0;
  }

  const uint32_t tg_threads = std::max<uint32_t>(1u, report.threads_per_threadgroup);
  const uint32_t thread_cap = kMaxThreadsPerCore / tg_threads;
  const uint32_t sram_cap =
      (report.estimated_sram_bytes == 0u)
          ? thread_cap
          : static_cast<uint32_t>(max_sram / report.estimated_sram_bytes);
  const uint64_t regs_per_tg =
      static_cast<uint64_t>(tg_threads) * std::max<uint32_t>(1u, report.estimated_registers_per_thread);
  const uint32_t reg_cap = static_cast<uint32_t>(kRegsPerCore / regs_per_tg);

  report.max_resident_tgs_per_core = std::max<uint32_t>(1u, std::min({thread_cap, sram_cap, reg_cap}));
  report.estimated_occupancy =
      std::min(1.0, static_cast<double>(report.max_resident_tgs_per_core) * static_cast<double>(tg_threads) /
                        static_cast<double>(kMaxThreadsPerCore));

  return report;
}

PassResult FusionOccupancyCostModelPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  FusionCostReport flags;

  KernelScanInfo scan;
  if (kernel.entry_block != kInvalidId && kernel.entry_block < kernel.blocks.size()) {
    scan_block_for_cost(kernel, kernel.entry_block, ctx.target, scan);
  }
  const EpilogueSummary kernel_epilogue = collect_kernel_epilogue(kernel);

  ScalarDType primary_dt_a = infer_kernel_float_dtype(kernel);
  ScalarDType primary_dt_b = primary_dt_a;
  if (!scan.compute_ops.empty()) {
    const ComputeTileInfo* primary = &scan.compute_ops[0];
    for (const auto& c : scan.compute_ops) {
      if (static_cast<uint64_t>(c.bm) * c.bn > static_cast<uint64_t>(primary->bm) * primary->bn) {
        primary = &c;
      }
    }
    primary_dt_a = primary->dt_a;
    primary_dt_b = primary->dt_b;
  }

  if (kernel.hints.tile_spec.has_value()) {
    const bool is_non_sgmm_only =
        scan.compute_ops.empty() &&
        (scan.has_flash_attn || scan.has_depthwise_conv || (scan.has_scan_op && kernel.hints.tile_spec->bm <= 1u));
    if (clamp_tile_spec(*kernel.hints.tile_spec, primary_dt_a, primary_dt_b, scan.existing_alloc_bytes,
                        kernel_epilogue, ctx.target, !is_non_sgmm_only, &flags)) {
      res.changed = true;
    }
  }

  for (auto& op_ptr : kernel.ops) {
    if (!op_ptr || op_ptr->erased || !op_ptr->hints.tile_spec.has_value()) continue;

    if (op_ptr->kind == OpKind::Dot) {
      ScalarDType dt_a = primary_dt_a;
      ScalarDType dt_b = primary_dt_b;
      if (op_ptr->operands.size() >= 2) {
        dt_a = infer_ptr_operand_dtype(kernel, *op_ptr, 0, primary_dt_a);
        dt_b = infer_ptr_operand_dtype(kernel, *op_ptr, 1, primary_dt_b);
      }
      const auto& eff_dag =
          op_ptr->hints.epilogue_dag.has_value() ? op_ptr->hints.epilogue_dag : kernel.hints.epilogue_dag;
      const auto& eff_tail =
          (op_ptr->hints.fused_tail.has_value() && op_ptr->hints.fused_tail->kind != TailKind::None)
              ? op_ptr->hints.fused_tail
              : kernel.hints.fused_tail;
      const EpilogueSummary op_epilogue = summarize_epilogue_hints(eff_dag, eff_tail);
      if (clamp_tile_spec(*op_ptr->hints.tile_spec, dt_a, dt_b, scan.existing_alloc_bytes, op_epilogue, ctx.target,
                          true, &flags)) {
        res.changed = true;
      }
    } else if (op_ptr->kind == OpKind::ImplicitGemmConv && std::holds_alternative<ConvAttr>(op_ptr->attr)) {
      const auto& ca = std::get<ConvAttr>(op_ptr->attr);
      const ScalarDType dt = infer_ptr_operand_dtype(kernel, *op_ptr, 0, primary_dt_a);
      const auto& eff_dag =
          op_ptr->hints.epilogue_dag.has_value() ? op_ptr->hints.epilogue_dag : kernel.hints.epilogue_dag;
      const auto& eff_tail =
          (op_ptr->hints.fused_tail.has_value() && op_ptr->hints.fused_tail->kind != TailKind::None)
              ? op_ptr->hints.fused_tail
              : kernel.hints.fused_tail;
      const EpilogueSummary op_epilogue = summarize_epilogue_hints(eff_dag, eff_tail);
      if (clamp_tile_spec(*op_ptr->hints.tile_spec, dt, dt, scan.existing_alloc_bytes, op_epilogue, ctx.target,
                          !ca.is_depthwise, &flags)) {
        res.changed = true;
      }
    }
  }

  last_report_ = analyze(kernel, ctx.target);
  last_report_.clamped_tile = flags.clamped_tile;
  last_report_.disabled_double_buffer = flags.disabled_double_buffer;
  last_report_.clamped_simdgroups = flags.clamped_simdgroups;
  return res;
}

}  // namespace m1triton::passes

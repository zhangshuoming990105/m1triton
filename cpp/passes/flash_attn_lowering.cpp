#include "passes/flash_attn_lowering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace m1triton::passes {

namespace {

std::optional<uint32_t> param_index_of(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Value& v = kernel.value(vid);
  if (v.def_op != ir::kInvalidId || v.def_block != kernel.entry_block) return std::nullopt;
  if (v.arg_idx >= kernel.params.size()) return std::nullopt;
  return v.arg_idx;
}

std::optional<int64_t> find_constexpr_param(const ir::Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name && kp.constexpr_val > 0) {
      return kp.constexpr_val;
    }
  }
  return std::nullopt;
}

uint32_t round_down_pow2(uint32_t v) {
  if (v == 0u) return 1u;
  uint32_t p = 1u;
  while ((p << 1u) <= v) p <<= 1u;
  return p;
}

uint32_t align_down(uint32_t v, uint32_t step) {
  if (v < step) return step;
  return (v / step) * step;
}

uint32_t required_smem_bytes(uint32_t bm, uint32_t bn, uint32_t num_sg, ir::ScalarDType elem_dt) {
  if (elem_dt == ir::ScalarDType::F32 && bm == 16u && bn == 512u && num_sg == 16u) {
    return bm * bn * 4u;
  }
  const uint32_t ldp = bn + 4u;
  return (bm * ldp + 3u * bm) * 4u;
}

}  // namespace

PassResult FlashAttentionLoweringPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  const uint32_t simd = std::max<uint32_t>(1u, ctx.target.simd_width);
  const uint32_t max_sg = std::max<uint32_t>(1u, ctx.target.max_threads_per_threadgroup / simd);
  const uint32_t max_smem = ctx.target.max_threadgroup_memory_bytes;

  for (auto& op_ptr : kernel.ops) {
    ir::Op& op = *op_ptr;
    if (op.erased || op.kind != ir::OpKind::FlashAttentionTile) continue;

    auto* fa = std::get_if<ir::FlashAttnAttr>(&op.attr);
    if (fa == nullptr) {
      res.ok = false;
      res.diagnostic = "flash_attention_tile op missing FlashAttnAttr";
      return res;
    }
    if (op.operands.size() != 8 || !op.results.empty()) {
      res.ok = false;
      res.diagnostic = "flash_attention_tile expects 8 operands (q, k, v, out, BH, L_q, L_kv, D) and 0 results";
      return res;
    }
    if (!std::isfinite(fa->scale)) {
      res.ok = false;
      res.diagnostic = "flash_attention_tile requires a finite scale";
      return res;
    }

    ir::ScalarDType elem_dt = ir::ScalarDType::F32;
    for (size_t i = 0; i < 4; ++i) {
      const ir::Type& ty = kernel.value(op.operands[i]).type;
      if (!ty.is_ptr()) {
        res.ok = false;
        res.diagnostic = "flash_attention_tile operand " + std::to_string(i) + " must be a device pointer";
        return res;
      }
      const ir::ScalarDType pt = std::get<ir::PtrType>(ty.value).pointee;
      if (pt != ir::ScalarDType::F32 && pt != ir::ScalarDType::F16) {
        res.ok = false;
        res.diagnostic = "flash_attention_tile supports f32 or f16 pointers";
        return res;
      }
      if (i == 0) {
        elem_dt = pt;
      } else if (pt != elem_dt) {
        res.ok = false;
        res.diagnostic = "flash_attention_tile requires matching Q/K/V/O pointer dtypes";
        return res;
      }
    }

    if (const auto out_pidx = param_index_of(kernel, op.operands[3])) {
      if (!kernel.params[*out_pidx].writable) {
        kernel.params[*out_pidx].writable = true;
        res.changed = true;
      }
    }

    const ir::FlashAttnAttr before = *fa;

    uint32_t bm = 16u;
    uint32_t bn = 256u;
    uint32_t bd = 16u;
    uint32_t num_sg = std::min<uint32_t>(16u, max_sg);

    const std::optional<ir::TileSpecAttr>& hint_ts =
        op.hints.tile_spec.has_value() ? op.hints.tile_spec : kernel.hints.tile_spec;
    if (hint_ts.has_value()) {
      if (hint_ts->bm > 0u) bm = hint_ts->bm;
      if (hint_ts->bn > 0u) bn = hint_ts->bn;
      if (hint_ts->bk > 0u) bd = hint_ts->bk;
      if (hint_ts->num_simdgroups > 0u) num_sg = hint_ts->num_simdgroups;
    }

    if (const auto v = find_constexpr_param(kernel, "BLOCK_M")) bm = static_cast<uint32_t>(*v);
    if (const auto v = find_constexpr_param(kernel, "BLOCK_N")) bn = static_cast<uint32_t>(*v);
    if (const auto v = find_constexpr_param(kernel, "BLOCK_D")) bd = static_cast<uint32_t>(*v);
    if (const auto v = find_constexpr_param(kernel, "NUM_SG")) num_sg = static_cast<uint32_t>(*v);

    num_sg = round_down_pow2(std::min<uint32_t>(std::max<uint32_t>(1u, num_sg), max_sg));
    bm = std::min<uint32_t>(32u, align_down(bm, 8u));
    bn = std::min<uint32_t>(512u, align_down(bn, 32u));
    bd = std::min<uint32_t>(64u, align_down(bd, 8u));

    if (bn == 512u && (elem_dt != ir::ScalarDType::F32 || bm != 16u || num_sg != 16u)) {
      bn = 256u;
    }

    while (required_smem_bytes(bm, bn, num_sg, elem_dt) > max_smem && bn > 32u) {
      bn = align_down(bn / 2u, 32u);
    }
    while (required_smem_bytes(bm, bn, num_sg, elem_dt) > max_smem && bm > 8u) {
      bm = align_down(bm / 2u, 8u);
    }

    if (required_smem_bytes(bm, bn, num_sg, elem_dt) > max_smem) {
      res.ok = false;
      res.diagnostic = "flash_attention_tile exceeds target threadgroup memory budget";
      return res;
    }

    fa->bm = bm;
    fa->bn = bn;
    fa->bd = bd;
    fa->num_simdgroups = num_sg;

    if (!(*fa == before)) {
      res.changed = true;
    }
  }

  return res;
}

}  // namespace m1triton::passes

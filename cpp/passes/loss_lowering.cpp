#include "passes/loss_lowering.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <variant>

namespace m1triton::passes {

namespace {

constexpr uint32_t kPreferredSimdgroupsPerThreadgroup = 8;

uint32_t select_tg_threads(const ir::TargetSpec& target, const ir::OpHints& op_hints,
                           const ir::OpHints& kernel_hints) {
  const uint32_t simd = std::max<uint32_t>(target.simd_width, 1u);
  uint32_t simdgroups = kPreferredSimdgroupsPerThreadgroup;
  if (op_hints.tile_spec.has_value() && op_hints.tile_spec->num_simdgroups > 0u) {
    simdgroups = op_hints.tile_spec->num_simdgroups;
  } else if (kernel_hints.tile_spec.has_value() && kernel_hints.tile_spec->num_simdgroups > 0u) {
    simdgroups = kernel_hints.tile_spec->num_simdgroups;
  }
  uint32_t threads = simd * simdgroups;
  threads = std::min(threads, target.max_threads_per_threadgroup);
  threads = std::max(simd, (threads / simd) * simd);
  while (threads > simd && !target.fits_threadgroup_memory((threads / simd) * 4u * 4u)) {
    threads -= simd;
  }
  return threads;
}

float canonical_zero(float v) { return v == 0.0f ? 0.0f : v; }

bool is_f32_ptr(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Type& ty = kernel.value(vid).type;
  return ty.is_ptr() && ty.as_ptr().pointee == ir::ScalarDType::F32;
}

bool is_int_index_ptr(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Type& ty = kernel.value(vid).type;
  if (!ty.is_ptr()) return false;
  const ir::ScalarDType dt = ty.as_ptr().pointee;
  return dt == ir::ScalarDType::I64 || dt == ir::ScalarDType::I32;
}

bool is_int_scalar(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Type& ty = kernel.value(vid).type;
  return ty.is_scalar() && ty.is_int_elem();
}

bool is_numeric_scalar(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Type& ty = kernel.value(vid).type;
  return ty.is_scalar() && (ty.is_float_elem() || ty.is_int_elem());
}

bool validate_operands(const ir::Kernel& kernel, const ir::Op& op, const ir::LossAttr& attr, std::string& err) {
  const auto& ops = op.operands;
  if (attr.stage == 2) {
    if (ops.size() != 4) {
      err = "fused_loss stage 2 expects 4 operands (partial_ptr, out_ptr, num_partials, denom)";
      return false;
    }
    if (!is_f32_ptr(kernel, ops[0]) || !is_f32_ptr(kernel, ops[1])) {
      err = "fused_loss stage 2 operands 0 and 1 must be f32 device pointers";
      return false;
    }
    if (!is_int_scalar(kernel, ops[2]) || !is_numeric_scalar(kernel, ops[3])) {
      err = "fused_loss stage 2 expects integer num_partials and numeric denom";
      return false;
    }
    return true;
  }

  switch (attr.kind) {
    case ir::LossKind::MSE:
    case ir::LossKind::Huber:
    case ir::LossKind::KLDiv:
      if (ops.size() != 4) {
        err = "fused_loss stage 1 (mse/huber/kl_div) expects 4 operands (pred_ptr, tgt_ptr, partial_ptr, total)";
        return false;
      }
      if (!is_f32_ptr(kernel, ops[0]) || !is_f32_ptr(kernel, ops[1]) || !is_f32_ptr(kernel, ops[2])) {
        err = "fused_loss stage 1 (mse/huber/kl_div) operands 0..2 must be f32 device pointers";
        return false;
      }
      if (!is_int_scalar(kernel, ops[3])) {
        err = "fused_loss stage 1 (mse/huber/kl_div) operand 3 (total) must be an integer scalar";
        return false;
      }
      return true;

    case ir::LossKind::Hinge:
      if (ops.size() != 5) {
        err = "fused_loss stage 1 (hinge) expects 5 operands (pred_ptr, tgt_ptr, partial_ptr, rows, cols)";
        return false;
      }
      if (!is_f32_ptr(kernel, ops[0]) || !is_f32_ptr(kernel, ops[1]) || !is_f32_ptr(kernel, ops[2])) {
        err = "fused_loss stage 1 (hinge) operands 0..2 must be f32 device pointers";
        return false;
      }
      if (!is_int_scalar(kernel, ops[3]) || !is_int_scalar(kernel, ops[4])) {
        err = "fused_loss stage 1 (hinge) operands 3..4 (rows, cols) must be integer scalars";
        return false;
      }
      return true;

    case ir::LossKind::CrossEntropy:
      if (ops.size() != 5) {
        err = "fused_loss stage 1 (cross_entropy) expects 5 operands (pred_ptr, tgt_ptr, partial_ptr, B, C)";
        return false;
      }
      if (!is_f32_ptr(kernel, ops[0]) || !is_int_index_ptr(kernel, ops[1]) || !is_f32_ptr(kernel, ops[2])) {
        err = "fused_loss stage 1 (cross_entropy) expects f32 pred/partial pointers and i64/i32 target pointer";
        return false;
      }
      if (!is_int_scalar(kernel, ops[3]) || !is_int_scalar(kernel, ops[4])) {
        err = "fused_loss stage 1 (cross_entropy) operands 3..4 (B, C) must be integer scalars";
        return false;
      }
      return true;

    case ir::LossKind::TripletMargin:
      if (ops.size() != 6) {
        err = "fused_loss stage 1 (triplet_margin) expects 6 operands (anchor, pos, neg, partial, B, D)";
        return false;
      }
      for (size_t i = 0; i < 4; ++i) {
        if (!is_f32_ptr(kernel, ops[i])) {
          err = "fused_loss stage 1 (triplet_margin) operands 0..3 must be f32 device pointers";
          return false;
        }
      }
      if (!is_int_scalar(kernel, ops[4]) || !is_int_scalar(kernel, ops[5])) {
        err = "fused_loss stage 1 (triplet_margin) operands 4..5 (B, D) must be integer scalars";
        return false;
      }
      return true;
  }
  err = "unknown fused_loss kind";
  return false;
}

}  // namespace

PassResult LossLoweringPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  res.ok = true;
  res.changed = false;

  for (auto& op_ptr : kernel.ops) {
    ir::Op& op = *op_ptr;
    if (op.erased || op.kind != ir::OpKind::FusedLoss) continue;

    auto* attr = std::get_if<ir::LossAttr>(&op.attr);
    if (attr == nullptr) {
      res.ok = false;
      res.diagnostic = "fused_loss op is missing LossAttr";
      return res;
    }
    if (attr->stage != 1 && attr->stage != 2) {
      res.ok = false;
      res.diagnostic = "fused_loss stage must be 1 or 2";
      return res;
    }
    if (!std::isfinite(attr->margin) || !std::isfinite(attr->beta) || !std::isfinite(attr->eps)) {
      res.ok = false;
      res.diagnostic = "fused_loss hyperparameters (margin, beta, eps) must be finite";
      return res;
    }
    if (attr->kind == ir::LossKind::Huber && attr->beta < 0.0f) {
      res.ok = false;
      res.diagnostic = "fused_loss huber beta must be non-negative";
      return res;
    }
    if (attr->kind == ir::LossKind::TripletMargin && attr->eps < 0.0f) {
      res.ok = false;
      res.diagnostic = "fused_loss triplet_margin eps must be non-negative";
      return res;
    }
    if (!validate_operands(kernel, op, *attr, res.diagnostic)) {
      res.ok = false;
      return res;
    }

    ir::LossAttr next = *attr;
    next.margin = canonical_zero(next.margin);
    next.beta = canonical_zero(next.beta);
    next.eps = canonical_zero(next.eps);
    next.tg_threads = select_tg_threads(ctx.target, op.hints, kernel.hints);

    if (!(next == *attr)) {
      *attr = next;
      res.changed = true;
    }

    const size_t out_idx = (next.stage == 2) ? 1u : (next.kind == ir::LossKind::TripletMargin ? 3u : 2u);
    const ir::Value& ov = kernel.value(op.operands[out_idx]);
    if (ov.def_op == ir::kInvalidId && ov.def_block == kernel.entry_block && ov.arg_idx < kernel.params.size()) {
      if (!kernel.params[ov.arg_idx].writable) {
        kernel.params[ov.arg_idx].writable = true;
        res.changed = true;
      }
    }
  }

  return res;
}

}  // namespace m1triton::passes

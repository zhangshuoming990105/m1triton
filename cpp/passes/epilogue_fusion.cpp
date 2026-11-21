#include "passes/epilogue_fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

OpId insert_op_before(Kernel& k, OpId anchor_id, OpKind kind, std::vector<ValueId> operands, const Type& result_type,
                      OpPayload attr = EmptyAttr{}) {
  const BlockId blk_id = k.op(anchor_id).parent_block;
  const OpId new_id = k.create_op(kind, kInvalidId, std::move(operands), {result_type}, std::move(attr));
  k.op(new_id).parent_block = blk_id;
  auto& ops = k.block(blk_id).ops;
  auto it = std::find(ops.begin(), ops.end(), anchor_id);
  ops.insert(it, new_id);
  return new_id;
}

const Op* defining_op(const Kernel& k, ValueId v) {
  const Value& val = k.value(v);
  if (val.def_op == kInvalidId) return nullptr;
  const Op& op = k.op(val.def_op);
  return op.erased ? nullptr : &op;
}

const ConstAttr* scalar_const(const Kernel& k, ValueId v) {
  const Op* op = defining_op(k, v);
  if (op == nullptr || op->kind != OpKind::Constant) return nullptr;
  return std::get_if<ConstAttr>(&op->attr);
}

const ConstAttr* const_or_splat_const(const Kernel& k, ValueId v) {
  if (const auto* c = scalar_const(k, v)) return c;
  const Op* op = defining_op(k, v);
  if (op == nullptr || op->kind != OpKind::Splat || op->operands.size() != 1) return nullptr;
  return scalar_const(k, op->operands[0]);
}

std::optional<double> const_as_double(const ConstAttr* c) {
  if (c == nullptr) return std::nullopt;
  if (const auto* d = std::get_if<double>(&c->val)) return *d;
  if (const auto* i = std::get_if<int64_t>(&c->val)) return static_cast<double>(*i);
  if (const auto* u = std::get_if<uint64_t>(&c->val)) return static_cast<double>(*u);
  return std::nullopt;
}

std::optional<double> const_value(const Kernel& k, ValueId v) { return const_as_double(const_or_splat_const(k, v)); }

const UnaryAttr* unary_of(const Op* op, UnaryOp which) {
  if (op == nullptr || op->kind != OpKind::Unary) return nullptr;
  const auto* attr = std::get_if<UnaryAttr>(&op->attr);
  return (attr != nullptr && attr->op == which) ? attr : nullptr;
}

const BinaryAttr* binary_attr(const Op* op) {
  if (op == nullptr || op->kind != OpKind::Binary) return nullptr;
  return std::get_if<BinaryAttr>(&op->attr);
}

size_t use_count(const Kernel& k, ValueId v) {
  size_t n = 0;
  for (const auto& node : k.ops) {
    if (node->erased) continue;
    n += static_cast<size_t>(std::count(node->operands.begin(), node->operands.end(), v));
  }
  return n;
}

void erase_if_dead(Kernel& k, OpId id) {
  const Op& op = k.op(id);
  if (op.erased || !is_pure_op(op.kind)) return;
  for (ValueId r : op.results) {
    if (use_count(k, r) != 0) return;
  }
  const std::vector<ValueId> operands = op.operands;
  k.erase_op(id);
  for (ValueId operand : operands) {
    const Value& val = k.value(operand);
    if (val.def_op != kInvalidId) erase_if_dead(k, val.def_op);
  }
}

void replace_and_erase(Kernel& k, OpId old_op, ValueId new_value) {
  k.replace_all_uses(k.op(old_op).results[0], new_value);
  const std::vector<ValueId> operands = k.op(old_op).operands;
  k.erase_op(old_op);
  for (ValueId operand : operands) {
    const Value& val = k.value(operand);
    if (val.def_op != kInvalidId) erase_if_dead(k, val.def_op);
  }
}

// x ** 2 -> x * x, x ** 3 -> (x * x) * x.
bool fold_integer_pow(Kernel& k, OpId id) {
  const Op& op = k.op(id);
  const auto* bin = binary_attr(&op);
  if (bin == nullptr || bin->op != BinaryOp::Pow || op.results.empty()) return false;
  const Type res_ty = k.value(op.results[0]).type;
  if (!res_ty.is_float_elem()) return false;
  const ValueId x = op.operands[0];
  if (!k.value(x).type.same_shape_and_elem(res_ty)) return false;
  const auto exponent = const_value(k, op.operands[1]);
  if (!exponent.has_value() || (*exponent != 2.0 && *exponent != 3.0)) return false;

  const BinaryAttr mul{BinaryOp::Mul};
  const OpId sq = insert_op_before(k, id, OpKind::Binary, {x, x}, res_ty, mul);
  ValueId out = k.op(sq).results[0];
  if (*exponent == 3.0) {
    const OpId cube = insert_op_before(k, id, OpKind::Binary, {out, x}, res_ty, mul);
    out = k.op(cube).results[0];
  }
  replace_and_erase(k, id, out);
  return true;
}

// 1 / (1 + exp(-x)) -> sigmoid(x).
bool fold_sigmoid(Kernel& k, OpId id) {
  const Op& op = k.op(id);
  const auto* bin = binary_attr(&op);
  if (bin == nullptr || bin->op != BinaryOp::Div || op.results.empty()) return false;
  const Type res_ty = k.value(op.results[0]).type;
  if (!res_ty.is_float_elem()) return false;
  const auto numer = const_value(k, op.operands[0]);
  if (!numer.has_value() || *numer != 1.0) return false;

  const Op* denom = defining_op(k, op.operands[1]);
  const auto* add = binary_attr(denom);
  if (add == nullptr || add->op != BinaryOp::Add) return false;

  ValueId exp_val = kInvalidId;
  for (int side = 0; side < 2; ++side) {
    const auto one = const_value(k, denom->operands[side]);
    if (one.has_value() && *one == 1.0) {
      exp_val = denom->operands[1 - side];
      break;
    }
  }
  if (exp_val == kInvalidId) return false;
  const Op* exp_op = defining_op(k, exp_val);
  if (unary_of(exp_op, UnaryOp::Exp) == nullptr) return false;
  const Op* neg_op = defining_op(k, exp_op->operands[0]);
  if (unary_of(neg_op, UnaryOp::Neg) == nullptr) return false;

  const ValueId x = neg_op->operands[0];
  if (!k.value(x).type.same_shape_and_elem(res_ty)) return false;
  const OpId sig = insert_op_before(k, id, OpKind::Unary, {x}, res_ty, UnaryAttr{UnaryOp::Sigmoid});
  replace_and_erase(k, id, k.op(sig).results[0]);
  return true;
}

// select(x > c, x, c) and select(x >= c, x, c) -> max(x, c).
bool fold_select_to_max(Kernel& k, OpId id) {
  const Op& op = k.op(id);
  if (op.kind != OpKind::Select || op.operands.size() != 3 || op.results.empty()) return false;
  const Type res_ty = k.value(op.results[0]).type;
  if (!res_ty.is_float_elem() && !res_ty.is_int_elem()) return false;
  const ValueId x = op.operands[1];
  const ValueId fallback = op.operands[2];
  if (!k.value(x).type.same_shape_and_elem(res_ty) || !k.value(fallback).type.same_shape_and_elem(res_ty)) {
    return false;
  }
  const auto bound = const_value(k, fallback);
  if (!bound.has_value()) return false;

  const Op* cmp = defining_op(k, op.operands[0]);
  const auto* cmp_attr = binary_attr(cmp);
  if (cmp_attr == nullptr || cmp->operands.size() != 2) return false;

  bool matches = false;
  const auto lhs_const = const_value(k, cmp->operands[0]);
  const auto rhs_const = const_value(k, cmp->operands[1]);
  switch (cmp_attr->op) {
    case BinaryOp::CmpGt:
    case BinaryOp::CmpGe:
      matches = cmp->operands[0] == x && rhs_const.has_value() && *rhs_const == *bound;
      break;
    case BinaryOp::CmpLt:
    case BinaryOp::CmpLe:
      matches = cmp->operands[1] == x && lhs_const.has_value() && *lhs_const == *bound;
      break;
    default:
      break;
  }
  if (!matches) return false;

  const OpId mx = insert_op_before(k, id, OpKind::Binary, {x, fallback}, res_ty, BinaryAttr{BinaryOp::Max});
  replace_and_erase(k, id, k.op(mx).results[0]);
  return true;
}

enum class Stride : uint8_t { Uniform, Unit, Other };

Stride classify_stride(const Kernel& k, ValueId v, int depth = 0) {
  if (depth > 32) return Stride::Other;
  if (!k.value(v).type.is_tile()) return Stride::Uniform;
  const Op* op = defining_op(k, v);
  if (op == nullptr) return Stride::Other;
  switch (op->kind) {
    case OpKind::Constant:
    case OpKind::Splat:
    case OpKind::ProgramId:
    case OpKind::NumPrograms:
      return Stride::Uniform;
    case OpKind::Arange:
      return Stride::Unit;
    case OpKind::Broadcast:
    case OpKind::Cast:
      return classify_stride(k, op->operands[0], depth + 1);
    case OpKind::PtrAdd: {
      const Stride base = classify_stride(k, op->operands[0], depth + 1);
      const Stride off = classify_stride(k, op->operands[1], depth + 1);
      if (base == Stride::Uniform && off == Stride::Uniform) return Stride::Uniform;
      if ((base == Stride::Uniform && off == Stride::Unit) || (base == Stride::Unit && off == Stride::Uniform)) {
        return Stride::Unit;
      }
      return Stride::Other;
    }
    case OpKind::Binary: {
      const auto* bin = binary_attr(op);
      if (bin == nullptr) return Stride::Other;
      const Stride lhs = classify_stride(k, op->operands[0], depth + 1);
      const Stride rhs = classify_stride(k, op->operands[1], depth + 1);
      if (lhs == Stride::Uniform && rhs == Stride::Uniform) return Stride::Uniform;
      if (bin->op == BinaryOp::Add &&
          ((lhs == Stride::Uniform && rhs == Stride::Unit) || (lhs == Stride::Unit && rhs == Stride::Uniform))) {
        return Stride::Unit;
      }
      if (bin->op == BinaryOp::Sub && lhs == Stride::Unit && rhs == Stride::Uniform) return Stride::Unit;
      return Stride::Other;
    }
    default:
      return Stride::Other;
  }
}

bool is_vectorizable_pointee(ScalarDType dt) {
  return dt == ScalarDType::F32 || dt == ScalarDType::F16 || dt == ScalarDType::I32 || dt == ScalarDType::U32;
}

bool is_ptr_aligned_for_vec(const Kernel& k, ValueId v, uint32_t req_bytes, int depth = 0) {
  if (depth > 32) return false;
  const Value& val = k.value(v);
  if (val.def_block == k.entry_block) {
    uint32_t non_ce = 0;
    for (const auto& p : k.params) {
      if (p.is_constexpr) continue;
      if (non_ce == val.arg_idx) {
        return p.divisibility >= req_bytes && (p.divisibility % req_bytes) == 0;
      }
      ++non_ce;
    }
    if (val.arg_idx < k.params.size()) {
      const uint32_t div = k.params[val.arg_idx].divisibility;
      return div >= req_bytes && (div % req_bytes) == 0;
    }
    return false;
  }
  const Op* op = defining_op(k, v);
  if (op == nullptr) return false;
  if (op->kind == OpKind::Splat || op->kind == OpKind::Broadcast || op->kind == OpKind::PtrAdd) {
    return is_ptr_aligned_for_vec(k, op->operands[0], req_bytes, depth + 1);
  }
  return false;
}

bool is_elementwise_op(OpKind kind) {
  switch (kind) {
    case OpKind::Constant:
    case OpKind::ProgramId:
    case OpKind::NumPrograms:
    case OpKind::Arange:
    case OpKind::Splat:
    case OpKind::Broadcast:
    case OpKind::Cast:
    case OpKind::Unary:
    case OpKind::Binary:
    case OpKind::Select:
    case OpKind::PtrAdd:
    case OpKind::Load:
    case OpKind::Store:
    case OpKind::Return:
      return true;
    default:
      return false;
  }
}

// Marks every 1D unit-stride tile Load/Store of a straight-line elementwise kernel with vec_width=4 so the
// 1D float4 emitter engages without a hand-written hint.
bool propagate_vec_width(Kernel& k, const TargetSpec& target) {
  std::vector<OpId> mem_ops;
  int64_t block_len = 0;
  for (const auto& node : k.ops) {
    if (node->erased) continue;
    if (!is_elementwise_op(node->kind) || node->parent_block != k.entry_block) return false;
    if (node->kind != OpKind::Load && node->kind != OpKind::Store) continue;
    const Type& ptr_ty = k.value(node->operands[0]).type;
    if (!ptr_ty.is_tile() || !ptr_ty.is_ptr_elem()) return false;
    const auto& shape = ptr_ty.as_tile().shape;
    if (shape.size() != 1 || shape[0] < 4 || shape[0] % 4 != 0) return false;
    if (block_len != 0 && block_len != shape[0]) return false;
    block_len = shape[0];
    const ScalarDType pointee = ptr_ty.ptr_elem().pointee;
    const uint32_t vec_bytes = dtype_bytes(pointee) * 4;
    if (!is_vectorizable_pointee(pointee) || vec_bytes > target.vector_bytes) return false;
    if (!is_ptr_aligned_for_vec(k, node->operands[0], vec_bytes)) return false;
    if (classify_stride(k, node->operands[0]) != Stride::Unit) return false;
    mem_ops.push_back(node->id);
  }
  if (mem_ops.empty()) return false;

  bool changed = false;
  for (OpId id : mem_ops) {
    Op& op = k.op(id);
    if (op.hints.vec_width.has_value()) continue;
    if (const auto* ls = std::get_if<LoadStoreAttr>(&op.attr); ls != nullptr && ls->vec_width > 1) continue;
    op.hints.vec_width = VecWidthAttr{4};
    changed = true;
  }
  return changed;
}

}  // namespace

PassResult EpilogueFusionPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  res.ok = true;
  res.changed = false;

  bool progress = true;
  for (int round = 0; progress && round < 8; ++round) {
    progress = false;
    for (size_t b = 0; b < kernel.blocks.size(); ++b) {
      const std::vector<OpId> snapshot = kernel.block(static_cast<BlockId>(b)).ops;
      for (OpId id : snapshot) {
        if (kernel.op(id).erased) continue;
        if (fold_integer_pow(kernel, id) || fold_sigmoid(kernel, id) || fold_select_to_max(kernel, id)) {
          progress = true;
        }
      }
    }
    res.changed = res.changed || progress;
  }

  if (propagate_vec_width(kernel, ctx.target)) res.changed = true;
  return res;
}

}  // namespace m1triton::passes

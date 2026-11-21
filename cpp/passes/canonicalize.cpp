#include "passes/canonicalize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

OpId insert_op_before(Kernel& k, BlockId blk_id, OpId anchor_id, OpKind kind, std::vector<ValueId> operands,
                      const std::vector<Type>& res_types, OpPayload attr = EmptyAttr{}) {
  const OpId new_id = k.create_op(kind, kInvalidId, std::move(operands), res_types, std::move(attr));
  k.op(new_id).parent_block = blk_id;
  auto& ops = k.block(blk_id).ops;
  auto it = std::find(ops.begin(), ops.end(), anchor_id);
  ops.insert(it, new_id);
  return new_id;
}

const ConstAttr* get_scalar_const(const Kernel& k, ValueId v) {
  const Value& val = k.value(v);
  if (val.def_op == kInvalidId) return nullptr;
  const Op& op = k.op(val.def_op);
  if (op.erased || op.kind != OpKind::Constant) return nullptr;
  return std::get_if<ConstAttr>(&op.attr);
}

const ConstAttr* get_const_or_splat_const(const Kernel& k, ValueId v) {
  if (const auto* c = get_scalar_const(k, v)) return c;
  const Value& val = k.value(v);
  if (val.def_op == kInvalidId) return nullptr;
  const Op& op = k.op(val.def_op);
  if (op.erased || op.kind != OpKind::Splat || op.operands.size() != 1) return nullptr;
  return get_scalar_const(k, op.operands[0]);
}

bool const_is_zero(const ConstAttr& c) {
  if (const auto* i = std::get_if<int64_t>(&c.val)) return *i == 0;
  if (const auto* u = std::get_if<uint64_t>(&c.val)) return *u == 0;
  if (const auto* d = std::get_if<double>(&c.val)) return *d == 0.0;
  if (const auto* b = std::get_if<bool>(&c.val)) return !*b;
  return false;
}

bool const_is_one(const ConstAttr& c) {
  if (const auto* i = std::get_if<int64_t>(&c.val)) return *i == 1;
  if (const auto* u = std::get_if<uint64_t>(&c.val)) return *u == 1;
  if (const auto* d = std::get_if<double>(&c.val)) return *d == 1.0;
  if (const auto* b = std::get_if<bool>(&c.val)) return *b;
  return false;
}

std::optional<bool> const_as_bool(const ConstAttr& c) {
  if (const auto* b = std::get_if<bool>(&c.val)) return *b;
  return std::nullopt;
}

double const_to_double(const ConstAttr& c) {
  if (const auto* d = std::get_if<double>(&c.val)) return *d;
  if (const auto* i = std::get_if<int64_t>(&c.val)) return static_cast<double>(*i);
  if (const auto* u = std::get_if<uint64_t>(&c.val)) return static_cast<double>(*u);
  return std::get<bool>(c.val) ? 1.0 : 0.0;
}

int64_t const_to_i64(const ConstAttr& c) {
  if (const auto* i = std::get_if<int64_t>(&c.val)) return *i;
  if (const auto* u = std::get_if<uint64_t>(&c.val)) return static_cast<int64_t>(*u);
  if (const auto* d = std::get_if<double>(&c.val)) return static_cast<int64_t>(*d);
  return std::get<bool>(c.val) ? 1 : 0;
}

uint64_t const_to_u64(const ConstAttr& c) {
  if (const auto* u = std::get_if<uint64_t>(&c.val)) return *u;
  if (const auto* i = std::get_if<int64_t>(&c.val)) return static_cast<uint64_t>(*i);
  if (const auto* d = std::get_if<double>(&c.val)) return static_cast<uint64_t>(*d);
  return std::get<bool>(c.val) ? 1ULL : 0ULL;
}

ConstAttr make_typed_const_from_double(double v, ScalarDType dt) {
  if (dt == ScalarDType::F32 || dt == ScalarDType::F16) {
    return ConstAttr{static_cast<double>(static_cast<float>(v))};
  }
  if (dt == ScalarDType::I32) {
    return ConstAttr{static_cast<int64_t>(static_cast<int32_t>(v))};
  }
  if (dt == ScalarDType::U32) {
    return ConstAttr{static_cast<uint64_t>(static_cast<uint32_t>(v))};
  }
  if (dt == ScalarDType::I64) {
    return ConstAttr{static_cast<int64_t>(v)};
  }
  return ConstAttr{v != 0.0};
}

std::optional<ConstAttr> fold_unary_const(UnaryOp op, const ConstAttr& c, ScalarDType dt) {
  if (dt == ScalarDType::F32 || dt == ScalarDType::F16) {
    const double x = const_to_double(c);
    double r = 0.0;
    switch (op) {
      case UnaryOp::Exp:
        r = std::exp(x);
        break;
      case UnaryOp::Log:
        if (x <= 0.0) return std::nullopt;
        r = std::log(x);
        break;
      case UnaryOp::Sqrt:
        if (x < 0.0) return std::nullopt;
        r = std::sqrt(x);
        break;
      case UnaryOp::Rsqrt:
        if (x <= 0.0) return std::nullopt;
        r = 1.0 / std::sqrt(x);
        break;
      case UnaryOp::Abs:
        r = std::fabs(x);
        break;
      case UnaryOp::Neg:
        r = -x;
        break;
      case UnaryOp::Tanh:
        r = std::tanh(x);
        break;
      case UnaryOp::Erf:
        r = std::erf(x);
        break;
      case UnaryOp::Floor:
        r = std::floor(x);
        break;
      case UnaryOp::Ceil:
        r = std::ceil(x);
        break;
      case UnaryOp::Sin:
        r = std::sin(x);
        break;
      case UnaryOp::Cos:
        r = std::cos(x);
        break;
      case UnaryOp::Sigmoid:
        r = 1.0 / (1.0 + std::exp(-x));
        break;
      case UnaryOp::Not:
        return std::nullopt;
    }
    return ConstAttr{static_cast<double>(static_cast<float>(r))};
  }

  if (dt == ScalarDType::Bool) {
    if (op == UnaryOp::Not) {
      return ConstAttr{!std::get<bool>(c.val)};
    }
    return std::nullopt;
  }

  if (dt == ScalarDType::I32) {
    const int32_t x = static_cast<int32_t>(const_to_i64(c));
    if (op == UnaryOp::Neg) return ConstAttr{static_cast<int64_t>(static_cast<int32_t>(0u - static_cast<uint32_t>(x)))};
    if (op == UnaryOp::Abs) {
      if (x == std::numeric_limits<int32_t>::min()) return std::nullopt;
      return ConstAttr{static_cast<int64_t>(x < 0 ? -x : x)};
    }
    if (op == UnaryOp::Not) return ConstAttr{static_cast<int64_t>(~x)};
    return std::nullopt;
  }

  if (dt == ScalarDType::I64) {
    const int64_t x = const_to_i64(c);
    if (op == UnaryOp::Neg) return ConstAttr{static_cast<int64_t>(0ULL - static_cast<uint64_t>(x))};
    if (op == UnaryOp::Abs) {
      if (x == std::numeric_limits<int64_t>::min()) return std::nullopt;
      return ConstAttr{x < 0 ? -x : x};
    }
    if (op == UnaryOp::Not) return ConstAttr{~x};
    return std::nullopt;
  }

  if (dt == ScalarDType::U32) {
    const uint32_t x = static_cast<uint32_t>(const_to_u64(c));
    if (op == UnaryOp::Not) return ConstAttr{static_cast<uint64_t>(~x)};
    if (op == UnaryOp::Abs) return ConstAttr{static_cast<uint64_t>(x)};
    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<ConstAttr> fold_binary_const(BinaryOp op, const ConstAttr& ca, const ConstAttr& cb, ScalarDType dt) {
  if (dt == ScalarDType::F32 || dt == ScalarDType::F16) {
    const double a = const_to_double(ca);
    const double b = const_to_double(cb);
    switch (op) {
      case BinaryOp::Add:
        return make_typed_const_from_double(a + b, dt);
      case BinaryOp::Sub:
        return make_typed_const_from_double(a - b, dt);
      case BinaryOp::Mul:
        return make_typed_const_from_double(a * b, dt);
      case BinaryOp::Div:
        if (b == 0.0) return std::nullopt;
        return make_typed_const_from_double(a / b, dt);
      case BinaryOp::Rem:
        if (b == 0.0) return std::nullopt;
        return make_typed_const_from_double(std::fmod(a, b), dt);
      case BinaryOp::Max:
        return make_typed_const_from_double(std::fmax(a, b), dt);
      case BinaryOp::Min:
        return make_typed_const_from_double(std::fmin(a, b), dt);
      case BinaryOp::Pow:
        return make_typed_const_from_double(std::pow(a, b), dt);
      case BinaryOp::CmpEq:
        return ConstAttr{a == b};
      case BinaryOp::CmpNe:
        return ConstAttr{a != b};
      case BinaryOp::CmpLt:
        return ConstAttr{a < b};
      case BinaryOp::CmpLe:
        return ConstAttr{a <= b};
      case BinaryOp::CmpGt:
        return ConstAttr{a > b};
      case BinaryOp::CmpGe:
        return ConstAttr{a >= b};
      default:
        return std::nullopt;
    }
  }

  if (dt == ScalarDType::Bool) {
    const bool a = std::get<bool>(ca.val);
    const bool b = std::get<bool>(cb.val);
    switch (op) {
      case BinaryOp::And:
        return ConstAttr{a && b};
      case BinaryOp::Or:
        return ConstAttr{a || b};
      case BinaryOp::Xor:
        return ConstAttr{a != b};
      case BinaryOp::CmpEq:
        return ConstAttr{a == b};
      case BinaryOp::CmpNe:
        return ConstAttr{a != b};
      default:
        return std::nullopt;
    }
  }

  if (dt == ScalarDType::U32) {
    const uint32_t a = static_cast<uint32_t>(const_to_u64(ca));
    const uint32_t b = static_cast<uint32_t>(const_to_u64(cb));
    switch (op) {
      case BinaryOp::Add:
        return ConstAttr{static_cast<uint64_t>(a + b)};
      case BinaryOp::Sub:
        return ConstAttr{static_cast<uint64_t>(a - b)};
      case BinaryOp::Mul:
        return ConstAttr{static_cast<uint64_t>(a * b)};
      case BinaryOp::Div:
        if (b == 0) return std::nullopt;
        return ConstAttr{static_cast<uint64_t>(a / b)};
      case BinaryOp::Rem:
        if (b == 0) return std::nullopt;
        return ConstAttr{static_cast<uint64_t>(a % b)};
      case BinaryOp::Max:
        return ConstAttr{static_cast<uint64_t>(std::max(a, b))};
      case BinaryOp::Min:
        return ConstAttr{static_cast<uint64_t>(std::min(a, b))};
      case BinaryOp::And:
        return ConstAttr{static_cast<uint64_t>(a & b)};
      case BinaryOp::Or:
        return ConstAttr{static_cast<uint64_t>(a | b)};
      case BinaryOp::Xor:
        return ConstAttr{static_cast<uint64_t>(a ^ b)};
      case BinaryOp::Shl:
        if (b >= 32) return std::nullopt;
        return ConstAttr{static_cast<uint64_t>(a << b)};
      case BinaryOp::Shr:
        if (b >= 32) return std::nullopt;
        return ConstAttr{static_cast<uint64_t>(a >> b)};
      case BinaryOp::CmpEq:
        return ConstAttr{a == b};
      case BinaryOp::CmpNe:
        return ConstAttr{a != b};
      case BinaryOp::CmpLt:
        return ConstAttr{a < b};
      case BinaryOp::CmpLe:
        return ConstAttr{a <= b};
      case BinaryOp::CmpGt:
        return ConstAttr{a > b};
      case BinaryOp::CmpGe:
        return ConstAttr{a >= b};
      default:
        return std::nullopt;
    }
  }

  const bool is_i32 = (dt == ScalarDType::I32);
  const int64_t a = is_i32 ? static_cast<int64_t>(static_cast<int32_t>(const_to_i64(ca))) : const_to_i64(ca);
  const int64_t b = is_i32 ? static_cast<int64_t>(static_cast<int32_t>(const_to_i64(cb))) : const_to_i64(cb);
  const auto wrap = [is_i32](uint64_t u) -> ConstAttr {
    if (is_i32) return ConstAttr{static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(u)))};
    return ConstAttr{static_cast<int64_t>(u)};
  };
  switch (op) {
    case BinaryOp::Add:
      return wrap(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
    case BinaryOp::Sub:
      return wrap(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
    case BinaryOp::Mul:
      return wrap(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
    case BinaryOp::Div:
      if (b == 0) return std::nullopt;
      if (is_i32 && a == std::numeric_limits<int32_t>::min() && b == -1) return std::nullopt;
      if (!is_i32 && a == std::numeric_limits<int64_t>::min() && b == -1) return std::nullopt;
      return ConstAttr{a / b};
    case BinaryOp::Rem:
      if (b == 0) return std::nullopt;
      if (is_i32 && a == std::numeric_limits<int32_t>::min() && b == -1) return std::nullopt;
      if (!is_i32 && a == std::numeric_limits<int64_t>::min() && b == -1) return std::nullopt;
      return ConstAttr{a % b};
    case BinaryOp::Max:
      return ConstAttr{std::max(a, b)};
    case BinaryOp::Min:
      return ConstAttr{std::min(a, b)};
    case BinaryOp::And:
      return wrap(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
    case BinaryOp::Or:
      return wrap(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
    case BinaryOp::Xor:
      return wrap(static_cast<uint64_t>(a) ^ static_cast<uint64_t>(b));
    case BinaryOp::Shl:
      if (b < 0 || b >= (is_i32 ? 32 : 64)) return std::nullopt;
      return wrap(static_cast<uint64_t>(a) << static_cast<unsigned>(b));
    case BinaryOp::Shr:
      if (b < 0 || b >= (is_i32 ? 32 : 64)) return std::nullopt;
      return ConstAttr{a >> static_cast<unsigned>(b)};
    case BinaryOp::CmpEq:
      return ConstAttr{a == b};
    case BinaryOp::CmpNe:
      return ConstAttr{a != b};
    case BinaryOp::CmpLt:
      return ConstAttr{a < b};
    case BinaryOp::CmpLe:
      return ConstAttr{a <= b};
    case BinaryOp::CmpGt:
      return ConstAttr{a > b};
    case BinaryOp::CmpGe:
      return ConstAttr{a >= b};
    default:
      return std::nullopt;
  }
}

ConstAttr fold_cast_const(const ConstAttr& c, ScalarDType target_dt) {
  switch (target_dt) {
    case ScalarDType::F32:
    case ScalarDType::F16:
      return ConstAttr{static_cast<double>(static_cast<float>(const_to_double(c)))};
    case ScalarDType::I32:
      return ConstAttr{static_cast<int64_t>(static_cast<int32_t>(const_to_i64(c)))};
    case ScalarDType::U32:
      return ConstAttr{static_cast<uint64_t>(static_cast<uint32_t>(const_to_u64(c)))};
    case ScalarDType::I64:
      return ConstAttr{const_to_i64(c)};
    case ScalarDType::Bool:
      return ConstAttr{!const_is_zero(c)};
  }
  return c;
}

bool canonicalize_block(Kernel& k, BlockId blk_id) {
  bool changed = false;
  const auto op_snapshot = k.block(blk_id).ops;

  for (OpId op_id : op_snapshot) {
    if (k.op(op_id).erased) continue;

    // Recurse into regions first.
    const auto regions = k.op(op_id).regions;
    for (BlockId reg_id : regions) {
      if (canonicalize_block(k, reg_id)) {
        changed = true;
      }
    }

    const OpKind kind = k.op(op_id).kind;

    if (kind == OpKind::Cast) {
      const ValueId in_v = k.op(op_id).operands[0];
      const ValueId out_v = k.op(op_id).results[0];
      const Type in_ty = k.value(in_v).type;
      const Type out_ty = k.value(out_v).type;
      if (in_ty.same_shape_and_elem(out_ty)) {
        k.replace_all_uses(out_v, in_v);
        k.erase_op(op_id);
        changed = true;
        continue;
      }
      if (out_ty.is_scalar()) {
        if (const auto* c = get_scalar_const(k, in_v)) {
          const OpId new_c =
              insert_op_before(k, blk_id, op_id, OpKind::Constant, {}, {out_ty}, fold_cast_const(*c, out_ty.as_scalar()));
          k.replace_all_uses(out_v, k.op(new_c).results[0]);
          k.erase_op(op_id);
          changed = true;
          continue;
        }
      }
    }

    if (kind == OpKind::Broadcast) {
      const ValueId in_v = k.op(op_id).operands[0];
      const ValueId out_v = k.op(op_id).results[0];
      const Type in_ty = k.value(in_v).type;
      const Type out_ty = k.value(out_v).type;
      if (in_ty.same_shape_and_elem(out_ty)) {
        k.replace_all_uses(out_v, in_v);
        k.erase_op(op_id);
        changed = true;
        continue;
      }
      const OpId def_id = k.value(in_v).def_op;
      if (def_id != kInvalidId && !k.op(def_id).erased && k.op(def_id).kind == OpKind::Splat) {
        const ValueId scalar_src = k.op(def_id).operands[0];
        const OpId new_splat = insert_op_before(k, blk_id, op_id, OpKind::Splat, {scalar_src}, {out_ty});
        k.replace_all_uses(out_v, k.op(new_splat).results[0]);
        k.erase_op(op_id);
        changed = true;
        continue;
      }
    }

    if (kind == OpKind::Unary) {
      const ValueId in_v = k.op(op_id).operands[0];
      const ValueId out_v = k.op(op_id).results[0];
      const Type out_ty = k.value(out_v).type;
      const UnaryOp uop = std::get<UnaryAttr>(k.op(op_id).attr).op;
      if (out_ty.is_scalar()) {
        if (const auto* c = get_scalar_const(k, in_v)) {
          if (auto folded = fold_unary_const(uop, *c, out_ty.as_scalar())) {
            const OpId new_c = insert_op_before(k, blk_id, op_id, OpKind::Constant, {}, {out_ty}, *folded);
            k.replace_all_uses(out_v, k.op(new_c).results[0]);
            k.erase_op(op_id);
            changed = true;
            continue;
          }
        }
      }
    }

    if (kind == OpKind::Binary) {
      const BinaryOp bop = std::get<BinaryAttr>(k.op(op_id).attr).op;
      ValueId lhs = k.op(op_id).operands[0];
      ValueId rhs = k.op(op_id).operands[1];
      const ValueId out_v = k.op(op_id).results[0];
      const Type lhs_ty = k.value(lhs).type;
      const Type out_ty = k.value(out_v).type;

      // Put constant on RHS for commutative binary ops.
      if (is_commutative_op(bop) && get_const_or_splat_const(k, lhs) != nullptr &&
          get_const_or_splat_const(k, rhs) == nullptr) {
        std::swap(k.op(op_id).operands[0], k.op(op_id).operands[1]);
        std::swap(lhs, rhs);
        changed = true;
      }

      // Scalar constant folding.
      if (lhs_ty.is_scalar()) {
        const auto* cl = get_scalar_const(k, lhs);
        const auto* cr = get_scalar_const(k, rhs);
        if (cl && cr) {
          if (auto folded = fold_binary_const(bop, *cl, *cr, lhs_ty.as_scalar())) {
            const OpId new_c = insert_op_before(k, blk_id, op_id, OpKind::Constant, {}, {out_ty}, *folded);
            k.replace_all_uses(out_v, k.op(new_c).results[0]);
            k.erase_op(op_id);
            changed = true;
            continue;
          }
        }
      }

      // Algebraic identities (both scalar and splat constants).
      const auto* cl = get_const_or_splat_const(k, lhs);
      const auto* cr = get_const_or_splat_const(k, rhs);
      std::optional<ValueId> replacement;

      if (bop == BinaryOp::Add) {
        if (cr && const_is_zero(*cr)) replacement = lhs;
        else if (cl && const_is_zero(*cl)) replacement = rhs;
      } else if (bop == BinaryOp::Sub) {
        if (cr && const_is_zero(*cr)) replacement = lhs;
      } else if (bop == BinaryOp::Mul) {
        if (cr && const_is_one(*cr)) replacement = lhs;
        else if (cl && const_is_one(*cl)) replacement = rhs;
        else if (cr && const_is_zero(*cr)) replacement = rhs;
        else if (cl && const_is_zero(*cl)) replacement = lhs;
      } else if (bop == BinaryOp::Div) {
        if (cr && const_is_one(*cr)) replacement = lhs;
      } else if (bop == BinaryOp::And) {
        if (cr && const_is_zero(*cr)) replacement = rhs;
        else if (cl && const_is_zero(*cl)) replacement = lhs;
      } else if (bop == BinaryOp::Or || bop == BinaryOp::Xor) {
        if (cr && const_is_zero(*cr)) replacement = lhs;
        else if (cl && const_is_zero(*cl)) replacement = rhs;
      } else if (bop == BinaryOp::Shl || bop == BinaryOp::Shr) {
        if (cr && const_is_zero(*cr)) replacement = lhs;
      }

      if (replacement) {
        k.replace_all_uses(out_v, *replacement);
        k.erase_op(op_id);
        changed = true;
        continue;
      }
    }

    if (kind == OpKind::Select) {
      const ValueId cond = k.op(op_id).operands[0];
      const ValueId tval = k.op(op_id).operands[1];
      const ValueId fval = k.op(op_id).operands[2];
      const ValueId out_v = k.op(op_id).results[0];

      if (tval == fval) {
        k.replace_all_uses(out_v, tval);
        k.erase_op(op_id);
        changed = true;
        continue;
      }
      if (const auto* cc = get_const_or_splat_const(k, cond)) {
        if (auto b = const_as_bool(*cc)) {
          k.replace_all_uses(out_v, *b ? tval : fval);
          k.erase_op(op_id);
          changed = true;
          continue;
        }
      }
    }

    if (kind == OpKind::PtrAdd) {
      const ValueId ptr_v = k.op(op_id).operands[0];
      const ValueId off_v = k.op(op_id).operands[1];
      const ValueId out_v = k.op(op_id).results[0];
      const Type ptr_ty = k.value(ptr_v).type;
      const Type out_ty = k.value(out_v).type;

      // PtrAdd(base, 0) -> base when shape is unchanged.
      if (ptr_ty.same_shape_and_elem(out_ty)) {
        if (const auto* c_off = get_const_or_splat_const(k, off_v); c_off && const_is_zero(*c_off)) {
          k.replace_all_uses(out_v, ptr_v);
          k.erase_op(op_id);
          changed = true;
          continue;
        }
      }

      // Unwrap Splat(base_ptr) inside PtrAdd when offset already provides the tile shape.
      const OpId ptr_def = k.value(ptr_v).def_op;
      if (ptr_def != kInvalidId && !k.op(ptr_def).erased && k.op(ptr_def).kind == OpKind::Splat &&
          k.value(off_v).type.is_tile()) {
        k.op(op_id).operands[0] = k.op(ptr_def).operands[0];
        changed = true;
        continue;
      }

      // Fold nested PtrAdd(PtrAdd(base, off1), off2) -> PtrAdd(base, Add(off1, off2)).
      if (ptr_def != kInvalidId && !k.op(ptr_def).erased && k.op(ptr_def).kind == OpKind::PtrAdd) {
        const ValueId base = k.op(ptr_def).operands[0];
        ValueId off1 = k.op(ptr_def).operands[1];
        ValueId off2 = off_v;

        const Type o1_ty = k.value(off1).type;
        const Type o2_ty = k.value(off2).type;
        const ScalarDType dt1 = o1_ty.scalar_elem();
        const ScalarDType dt2 = o2_ty.scalar_elem();
        const ScalarDType common_dt =
            (dt1 == ScalarDType::I64 || dt2 == ScalarDType::I64) ? ScalarDType::I64 : dt1;

        Type add_ty;
        if (o1_ty.is_tile() && o2_ty.is_tile()) {
          if (o1_ty.as_tile().shape != o2_ty.as_tile().shape) {
            continue;
          }
          add_ty = Type::tile(o1_ty.as_tile().shape, common_dt);
        } else if (o1_ty.is_tile()) {
          add_ty = Type::tile(o1_ty.as_tile().shape, common_dt);
        } else if (o2_ty.is_tile()) {
          add_ty = Type::tile(o2_ty.as_tile().shape, common_dt);
        } else {
          add_ty = Type::scalar(common_dt);
        }

        if (k.value(off1).type.scalar_elem() != common_dt) {
          const Type cast_ty = o1_ty.is_tile() ? Type::tile(o1_ty.as_tile().shape, common_dt) : Type::scalar(common_dt);
          const OpId c1 = insert_op_before(k, blk_id, op_id, OpKind::Cast, {off1}, {cast_ty});
          off1 = k.op(c1).results[0];
        }
        if (k.value(off2).type.scalar_elem() != common_dt) {
          const Type cast_ty = o2_ty.is_tile() ? Type::tile(o2_ty.as_tile().shape, common_dt) : Type::scalar(common_dt);
          const OpId c2 = insert_op_before(k, blk_id, op_id, OpKind::Cast, {off2}, {cast_ty});
          off2 = k.op(c2).results[0];
        }
        if (add_ty.is_tile() && !k.value(off1).type.is_tile()) {
          const OpId s1 = insert_op_before(k, blk_id, op_id, OpKind::Splat, {off1}, {add_ty});
          off1 = k.op(s1).results[0];
        }
        if (add_ty.is_tile() && !k.value(off2).type.is_tile()) {
          const OpId s2 = insert_op_before(k, blk_id, op_id, OpKind::Splat, {off2}, {add_ty});
          off2 = k.op(s2).results[0];
        }

        // Only fold if the resulting PtrAdd(base, add_ty) preserves out_ty's shape.
        const Type base_ty = k.value(base).type;
        const bool result_shape_ok =
            out_ty.is_tile() ? ((base_ty.is_tile() && base_ty.as_tile().shape == out_ty.as_tile().shape) ||
                                (add_ty.is_tile() && add_ty.as_tile().shape == out_ty.as_tile().shape))
                             : (!base_ty.is_tile() && !add_ty.is_tile());
        if (result_shape_ok) {
          const OpId add_op =
              insert_op_before(k, blk_id, op_id, OpKind::Binary, {off1, off2}, {add_ty}, BinaryAttr{BinaryOp::Add});
          k.op(op_id).operands[0] = base;
          k.op(op_id).operands[1] = k.op(add_op).results[0];
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace

PassResult CanonicalizePass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  if (kernel.entry_block == ir::kInvalidId) {
    return res;
  }
  constexpr int kMaxIterations = 8;
  for (int iter = 0; iter < kMaxIterations; ++iter) {
    if (!canonicalize_block(kernel, kernel.entry_block)) {
      break;
    }
    res.changed = true;
  }
  return res;
}

}  // namespace m1triton::passes

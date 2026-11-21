#include "codegen/metal/reduce_scan_emitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::ArangeAttr;
using ir::AxisAttr;
using ir::BarrierAttr;
using ir::BarrierScope;
using ir::BinaryAttr;
using ir::BinaryOp;
using ir::Block;
using ir::BlockId;
using ir::ConstAttr;
using ir::Kernel;
using ir::KernelParam;
using ir::LayoutAttr;
using ir::LoadStoreAttr;
using ir::Op;
using ir::OpId;
using ir::OpKind;
using ir::ReduceAttr;
using ir::ReduceKind;
using ir::ReduceResultMode;
using ir::ScalarDType;
using ir::ScanAttr;
using ir::ScanKind;
using ir::TargetSpec;
using ir::TileType;
using ir::Type;
using ir::UnaryAttr;
using ir::UnaryOp;
using ir::Value;
using ir::ValueId;
using ir::kInvalidId;

bool is_ptr_type(const Type& ty) {
  return ty.is_ptr_elem();
}

ScalarDType scalar_dtype_of(const Type& ty) {
  return ty.is_ptr_elem() ? ty.ptr_elem().pointee : ty.scalar_elem();
}

bool is_float_dtype(ScalarDType dt) {
  return dt == ScalarDType::F32 || dt == ScalarDType::F16;
}

bool is_int_dtype(ScalarDType dt) {
  return dt == ScalarDType::I32 || dt == ScalarDType::U32 || dt == ScalarDType::I64;
}

bool can_vec4_mem(ScalarDType dt) {
  return dt == ScalarDType::F32 || dt == ScalarDType::F16 || dt == ScalarDType::I32 || dt == ScalarDType::U32;
}

bool is_unit_tile(const Type& ty) {
  if (!ty.is_tile()) return false;
  const auto& sh = ty.as_tile().shape;
  for (int64_t d : sh) {
    if (d != 1) return false;
  }
  return !sh.empty();
}

bool is_scalar_like(const Type& ty) {
  return !ty.is_tile() || is_unit_tile(ty);
}

bool is_reduce_op_kind(OpKind k) {
  return k == OpKind::Reduce || k == OpKind::SimdReduce;
}

bool is_scan_op_kind(OpKind k) {
  return k == OpKind::Scan;
}

uint32_t round_up_32(uint32_t v) {
  if (v == 0) return 32u;
  return ((v + 31u) / 32u) * 32u;
}

uint32_t align_up_16(uint32_t v) {
  if (v == 0) return 0u;
  return ((v + 15u) / 16u) * 16u;
}

uint32_t next_pow2_u32(uint32_t v) {
  if (v <= 1u) return 1u;
  uint32_t p = 1u;
  while (p < v) p <<= 1u;
  return p;
}

std::string indent_str(int depth) {
  return std::string(static_cast<size_t>(depth) * 2, ' ');
}

std::string ssa_var(ValueId id) {
  return "m1_v" + std::to_string(id);
}

std::string axis_component(uint32_t axis) {
  switch (axis) {
    case 0:
      return "x";
    case 1:
      return "y";
    default:
      return "z";
  }
}

std::string msl_scalar_type(ScalarDType dt) {
  switch (dt) {
    case ScalarDType::F32:
      return "float";
    case ScalarDType::F16:
      return "half";
    case ScalarDType::I32:
      return "int";
    case ScalarDType::U32:
      return "uint";
    case ScalarDType::I64:
      return "long";
    case ScalarDType::Bool:
      return "bool";
  }
  return "float";
}

std::string msl_vec4_type(ScalarDType dt) {
  return msl_scalar_type(dt) + "4";
}

std::string msl_value_type(ScalarDType dt, bool vec4) {
  return vec4 ? msl_vec4_type(dt) : msl_scalar_type(dt);
}

std::string format_float_literal(double d) {
  if (std::isnan(d)) return "NAN";
  if (std::isinf(d)) return d < 0.0 ? "(-INFINITY)" : "INFINITY";
  std::ostringstream oss;
  oss << std::setprecision(9) << static_cast<float>(d);
  std::string s = oss.str();
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
    s += ".0";
  }
  return s + "f";
}

std::string format_scalar_literal_from_i64(int64_t v, ScalarDType dt) {
  switch (dt) {
    case ScalarDType::F32:
      return format_float_literal(static_cast<double>(v));
    case ScalarDType::F16:
      return "half(" + format_float_literal(static_cast<double>(v)) + ")";
    case ScalarDType::I32: {
      int32_t i32 = static_cast<int32_t>(v);
      if (i32 == std::numeric_limits<int32_t>::min()) return "(-2147483647 - 1)";
      return std::to_string(i32);
    }
    case ScalarDType::U32:
      return std::to_string(static_cast<uint32_t>(v)) + "u";
    case ScalarDType::I64:
      if (v == std::numeric_limits<int64_t>::min()) return "(-9223372036854775807ll - 1ll)";
      return std::to_string(v) + "ll";
    case ScalarDType::Bool:
      return v != 0 ? "true" : "false";
  }
  return "0";
}

std::string msl_const_literal(const ConstAttr& attr, ScalarDType dt) {
  return std::visit(
      [dt](const auto& raw) -> std::string {
        using T = std::decay_t<decltype(raw)>;
        if constexpr (std::is_same_v<T, bool>) {
          if (dt == ScalarDType::Bool) return raw ? "true" : "false";
          return format_scalar_literal_from_i64(raw ? 1 : 0, dt);
        } else if constexpr (std::is_same_v<T, double>) {
          if (dt == ScalarDType::F32) return format_float_literal(raw);
          if (dt == ScalarDType::F16) return "half(" + format_float_literal(raw) + ")";
          if (dt == ScalarDType::Bool) return raw != 0.0 ? "true" : "false";
          return format_scalar_literal_from_i64(static_cast<int64_t>(raw), dt);
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          if (dt == ScalarDType::U32) return std::to_string(static_cast<uint32_t>(raw)) + "u";
          if (dt == ScalarDType::F32) return format_float_literal(static_cast<double>(raw));
          if (dt == ScalarDType::F16) return "half(" + format_float_literal(static_cast<double>(raw)) + ")";
          if (dt == ScalarDType::Bool) return raw != 0 ? "true" : "false";
          return format_scalar_literal_from_i64(static_cast<int64_t>(raw), dt);
        } else {
          return format_scalar_literal_from_i64(raw, dt);
        }
      },
      attr.val);
}

std::optional<int64_t> const_as_i64(const ConstAttr& attr) {
  return std::visit(
      [](const auto& raw) -> std::optional<int64_t> {
        using T = std::decay_t<decltype(raw)>;
        if constexpr (std::is_same_v<T, int64_t>) {
          return raw;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          if (raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return static_cast<int64_t>(raw);
          }
          return std::nullopt;
        } else if constexpr (std::is_same_v<T, bool>) {
          return raw ? int64_t{1} : int64_t{0};
        } else {
          return std::nullopt;
        }
      },
      attr.val);
}

std::optional<double> const_as_f64(const ConstAttr& attr) {
  return std::visit(
      [](const auto& raw) -> std::optional<double> {
        using T = std::decay_t<decltype(raw)>;
        if constexpr (std::is_same_v<T, double>) {
          return raw;
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return static_cast<double>(raw);
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          return static_cast<double>(raw);
        } else {
          return raw ? 1.0 : 0.0;
        }
      },
      attr.val);
}

std::string msl_zero_literal(ScalarDType dt, bool vec4) {
  std::string s = format_scalar_literal_from_i64(0, dt);
  return vec4 ? (msl_vec4_type(dt) + "(" + s + ")") : s;
}

std::string msl_one_literal(ScalarDType dt, bool vec4) {
  std::string s = format_scalar_literal_from_i64(1, dt);
  return vec4 ? (msl_vec4_type(dt) + "(" + s + ")") : s;
}

std::string msl_reduce_identity(ReduceKind kind, ScalarDType dt) {
  if (kind == ReduceKind::Sum) {
    return msl_zero_literal(dt, false);
  }
  if (kind == ReduceKind::Max || kind == ReduceKind::Argmax) {
    switch (dt) {
      case ScalarDType::F32:
        return "(-INFINITY)";
      case ScalarDType::F16:
        return "half(-INFINITY)";
      case ScalarDType::I32:
        return "(-2147483647 - 1)";
      case ScalarDType::U32:
        return "0u";
      case ScalarDType::I64:
        return "(-9223372036854775807ll - 1ll)";
      case ScalarDType::Bool:
        return "false";
    }
  }
  switch (dt) {
    case ScalarDType::F32:
      return "INFINITY";
    case ScalarDType::F16:
      return "half(INFINITY)";
    case ScalarDType::I32:
      return "2147483647";
    case ScalarDType::U32:
      return "4294967295u";
    case ScalarDType::I64:
      return "9223372036854775807ll";
    case ScalarDType::Bool:
      return "true";
  }
  return "0.0f";
}

bool const_matches_reduce_identity(const ConstAttr& attr, ReduceKind kind, ScalarDType dt) {
  if (kind == ReduceKind::Sum) {
    if (is_float_dtype(dt)) {
      auto d = const_as_f64(attr);
      return d.has_value() && *d == 0.0;
    }
    auto i = const_as_i64(attr);
    return i.has_value() && *i == 0;
  }
  if (kind == ReduceKind::Max) {
    if (is_float_dtype(dt)) {
      auto d = const_as_f64(attr);
      return d.has_value() && std::isinf(*d) && *d < 0.0;
    }
    if (dt == ScalarDType::I32) {
      auto i = const_as_i64(attr);
      return i.has_value() && *i == std::numeric_limits<int32_t>::min();
    }
    if (dt == ScalarDType::U32) {
      auto i = const_as_i64(attr);
      return i.has_value() && *i == 0;
    }
    if (dt == ScalarDType::I64) {
      auto i = const_as_i64(attr);
      return i.has_value() && *i == std::numeric_limits<int64_t>::min();
    }
    return false;
  }
  if (kind == ReduceKind::Min) {
    if (is_float_dtype(dt)) {
      auto d = const_as_f64(attr);
      return d.has_value() && std::isinf(*d) && *d > 0.0;
    }
    if (dt == ScalarDType::I32) {
      auto i = const_as_i64(attr);
      return i.has_value() && *i == std::numeric_limits<int32_t>::max();
    }
    if (dt == ScalarDType::U32) {
      return std::holds_alternative<uint64_t>(attr.val) && std::get<uint64_t>(attr.val) == UINT32_MAX;
    }
    if (dt == ScalarDType::I64) {
      auto i = const_as_i64(attr);
      return i.has_value() && *i == std::numeric_limits<int64_t>::max();
    }
    return false;
  }
  return false;
}

std::string msl_cast_expr(const std::string& expr, ScalarDType src_dt, ScalarDType dst_dt, bool vec4) {
  if (src_dt == dst_dt) return expr;
  const std::string dst_ty = msl_value_type(dst_dt, vec4);
  if (dst_dt == ScalarDType::Bool) {
    return "(" + expr + " != " + msl_zero_literal(src_dt, vec4) + ")";
  }
  if (src_dt == ScalarDType::Bool && vec4) {
    std::string one = dst_ty + "(" + format_scalar_literal_from_i64(1, dst_dt) + ")";
    return "metal::select(" + msl_zero_literal(dst_dt, true) + ", " + one + ", " + expr + ")";
  }
  return dst_ty + "(" + expr + ")";
}

std::string promote_to_vec4(const std::string& expr, ScalarDType dt, bool is_already_vec4) {
  if (is_already_vec4) return expr;
  return msl_vec4_type(dt) + "(" + expr + ")";
}

struct HelperFlags {
  bool needs_erf = false;
  bool needs_tanh = false;
};

std::string msl_unary_expr(UnaryOp op, const std::string& x, ScalarDType dt, bool vec4, HelperFlags& helpers) {
  const std::string ty = msl_value_type(dt, vec4);
  switch (op) {
    case UnaryOp::Exp:
      return "metal::exp(" + x + ")";
    case UnaryOp::Log:
      return "metal::log(" + x + ")";
    case UnaryOp::Sqrt:
      return "metal::sqrt(" + x + ")";
    case UnaryOp::Rsqrt:
      return "metal::rsqrt(" + x + ")";
    case UnaryOp::Abs:
      return "metal::abs(" + x + ")";
    case UnaryOp::Neg:
      return "(-" + x + ")";
    case UnaryOp::Tanh:
      helpers.needs_tanh = true;
      return "m1_tanh(" + x + ")";
    case UnaryOp::Erf:
      helpers.needs_erf = true;
      return "m1_erf(" + x + ")";
    case UnaryOp::Floor:
      return "metal::floor(" + x + ")";
    case UnaryOp::Ceil:
      return "metal::ceil(" + x + ")";
    case UnaryOp::Sin:
      return "metal::sin(" + x + ")";
    case UnaryOp::Cos:
      return "metal::cos(" + x + ")";
    case UnaryOp::Sigmoid: {
      std::string one = vec4 ? (ty + "(" + format_scalar_literal_from_i64(1, dt) + ")")
                             : format_scalar_literal_from_i64(1, dt);
      return "(" + one + " / (" + one + " + metal::exp(-" + x + ")))";
    }
    case UnaryOp::Not:
      if (dt == ScalarDType::Bool) return "(!" + x + ")";
      return "(~" + x + ")";
  }
  return x;
}

ScalarDType common_cmp_dtype(ScalarDType a, ScalarDType b) {
  if (a == b) return a;
  if (a == ScalarDType::F32 || b == ScalarDType::F32) return ScalarDType::F32;
  if (a == ScalarDType::F16 || b == ScalarDType::F16) return ScalarDType::F16;
  if (a == ScalarDType::I64 || b == ScalarDType::I64) return ScalarDType::I64;
  if (a == ScalarDType::U32 || b == ScalarDType::U32) return ScalarDType::U32;
  return ScalarDType::I32;
}

std::string msl_binary_expr(BinaryOp op, const std::string& lhs_in, const std::string& rhs_in, ScalarDType lhs_dt,
                            ScalarDType rhs_dt, ScalarDType res_dt, bool vec4) {
  bool is_cmp = (op == BinaryOp::CmpEq || op == BinaryOp::CmpNe || op == BinaryOp::CmpLt || op == BinaryOp::CmpLe ||
                 op == BinaryOp::CmpGt || op == BinaryOp::CmpGe);
  ScalarDType op_dt = is_cmp ? common_cmp_dtype(lhs_dt, rhs_dt) : res_dt;
  std::string a = msl_cast_expr(lhs_in, lhs_dt, op_dt, vec4);
  std::string b = msl_cast_expr(rhs_in, rhs_dt, op_dt, vec4);

  switch (op) {
    case BinaryOp::Add:
      return "(" + a + " + " + b + ")";
    case BinaryOp::Sub:
      return "(" + a + " - " + b + ")";
    case BinaryOp::Mul:
      return "(" + a + " * " + b + ")";
    case BinaryOp::Div:
      return "(" + a + " / " + b + ")";
    case BinaryOp::Rem:
      if (is_float_dtype(op_dt)) return "metal::fmod(" + a + ", " + b + ")";
      return "(" + a + " % " + b + ")";
    case BinaryOp::Max:
      return "metal::max(" + a + ", " + b + ")";
    case BinaryOp::Min:
      return "metal::min(" + a + ", " + b + ")";
    case BinaryOp::Pow:
      if (is_float_dtype(op_dt)) return "metal::pow(" + a + ", " + b + ")";
      return msl_cast_expr("metal::pow(" + msl_cast_expr(a, op_dt, ScalarDType::F32, vec4) + ", " +
                               msl_cast_expr(b, op_dt, ScalarDType::F32, vec4) + ")",
                           ScalarDType::F32, op_dt, vec4);
    case BinaryOp::And:
      return (op_dt == ScalarDType::Bool) ? "(" + a + " && " + b + ")" : "(" + a + " & " + b + ")";
    case BinaryOp::Or:
      return (op_dt == ScalarDType::Bool) ? "(" + a + " || " + b + ")" : "(" + a + " | " + b + ")";
    case BinaryOp::Xor:
      return (op_dt == ScalarDType::Bool) ? "(" + a + " != " + b + ")" : "(" + a + " ^ " + b + ")";
    case BinaryOp::Shl:
      return "(" + a + " << " + b + ")";
    case BinaryOp::Shr:
      return "(" + a + " >> " + b + ")";
    case BinaryOp::CmpEq:
      return "(" + a + " == " + b + ")";
    case BinaryOp::CmpNe:
      return "(" + a + " != " + b + ")";
    case BinaryOp::CmpLt:
      return "(" + a + " < " + b + ")";
    case BinaryOp::CmpLe:
      return "(" + a + " <= " + b + ")";
    case BinaryOp::CmpGt:
      return "(" + a + " > " + b + ")";
    case BinaryOp::CmpGe:
      return "(" + a + " >= " + b + ")";
  }
  return "(" + a + " + " + b + ")";
}

std::string msl_select_expr(const std::string& cond, const std::string& t_val, const std::string& f_val,
                            ScalarDType t_dt, ScalarDType f_dt, ScalarDType res_dt, bool vec4) {
  std::string tv = msl_cast_expr(t_val, t_dt, res_dt, vec4);
  std::string fv = msl_cast_expr(f_val, f_dt, res_dt, vec4);
  if (!vec4) {
    return "(" + cond + " ? " + tv + " : " + fv + ")";
  }
  return "metal::select(" + fv + ", " + tv + ", " + cond + ")";
}

struct LoadOperands {
  ValueId ptrs = kInvalidId;
  ValueId mask = kInvalidId;
  ValueId other = kInvalidId;
  LoadStoreAttr attr;
};

LoadOperands extract_load_operands(const Op& op, const Kernel& kernel) {
  LoadOperands out;
  if (std::holds_alternative<LoadStoreAttr>(op.attr)) {
    out.attr = std::get<LoadStoreAttr>(op.attr);
  }
  if (!op.operands.empty()) out.ptrs = op.operands[0];

  if (out.attr.has_mask && out.attr.has_other) {
    if (op.operands.size() >= 2) out.mask = op.operands[1];
    if (op.operands.size() >= 3) out.other = op.operands[2];
  } else if (out.attr.has_mask) {
    if (op.operands.size() >= 2) out.mask = op.operands[1];
  } else if (out.attr.has_other) {
    if (op.operands.size() >= 3 && op.operands[1] == kInvalidId) {
      out.other = op.operands[2];
    } else if (op.operands.size() >= 2) {
      out.other = op.operands[1];
    }
  } else {
    if (op.operands.size() >= 2 && op.operands[1] != kInvalidId) {
      const Type& ty1 = kernel.value(op.operands[1]).type;
      if (!is_ptr_type(ty1) && scalar_dtype_of(ty1) == ScalarDType::Bool) {
        out.mask = op.operands[1];
        if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) {
          out.other = op.operands[2];
        }
      } else {
        out.other = op.operands[1];
      }
    }
  }
  return out;
}

struct StoreOperands {
  ValueId ptrs = kInvalidId;
  ValueId val = kInvalidId;
  ValueId mask = kInvalidId;
  LoadStoreAttr attr;
};

StoreOperands extract_store_operands(const Op& op) {
  StoreOperands out;
  if (std::holds_alternative<LoadStoreAttr>(op.attr)) {
    out.attr = std::get<LoadStoreAttr>(op.attr);
  }
  if (op.operands.size() >= 1) out.ptrs = op.operands[0];
  if (op.operands.size() >= 2) out.val = op.operands[1];
  if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) out.mask = op.operands[2];
  return out;
}

void collect_blocks_dfs(const Kernel& kernel, BlockId bid, std::vector<BlockId>& order,
                        std::unordered_set<BlockId>& seen) {
  if (bid == kInvalidId || bid >= kernel.blocks.size() || !seen.insert(bid).second) return;
  order.push_back(bid);
  const Block& blk = kernel.block(bid);
  for (OpId oid : blk.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    for (BlockId r : op.regions) {
      collect_blocks_dfs(kernel, r, order, seen);
    }
  }
}

bool block_has_reduce_or_scan(const Kernel& kernel, BlockId bid) {
  if (bid == kInvalidId || bid >= kernel.blocks.size()) return false;
  const Block& blk = kernel.block(bid);
  for (OpId oid : blk.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    if (is_reduce_op_kind(op.kind) || is_scan_op_kind(op.kind)) return true;
    for (BlockId rb : op.regions) {
      if (block_has_reduce_or_scan(kernel, rb)) return true;
    }
  }
  return false;
}

struct ReduceKernelAnalysis {
  std::vector<ParamSlot> param_slots;
  std::unordered_map<uint32_t, std::string> param_msl_name;
  std::unordered_map<ValueId, uint32_t> val_to_param_idx;
  std::unordered_map<ValueId, std::set<uint32_t>> ptr_origins;
  std::unordered_set<uint32_t> written_params;
  std::unordered_map<ValueId, int64_t> const_i64;
  std::unordered_set<ValueId> div4_values;
  std::unordered_set<ValueId> nonneg_values;
  std::unordered_map<ValueId, std::vector<OpId>> value_uses;
  std::unordered_set<OpId> broadcast_reduces;

  std::vector<BlockId> all_blocks;
  std::vector<OpId> reduce_ops;
  std::vector<OpId> scan_ops;

  size_t max_rank = 0;
  bool has_dot = false;
  bool has_reduce = false;
  bool has_scan = false;
  bool has_tile_reshape_or_permute = false;
  bool has_multiple_1d_sizes = false;
  int64_t block_size_1d = 0;
  std::pair<int64_t, int64_t> primary_2d_shape = {0, 0};
  std::optional<LayoutAttr> hint_layout;
};

bool analyze_reduce_kernel(const Kernel& kernel, ReduceKernelAnalysis& out, std::string& err) {
  if (kernel.name.empty()) {
    err = "Kernel name must not be empty";
    return false;
  }
  if (kernel.entry_block == kInvalidId || kernel.entry_block >= kernel.blocks.size()) {
    err = "Kernel has invalid entry block";
    return false;
  }

  std::unordered_set<BlockId> seen_blocks;
  collect_blocks_dfs(kernel, kernel.entry_block, out.all_blocks, seen_blocks);

  for (BlockId bid : out.all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if (is_reduce_op_kind(op.kind)) {
        out.has_reduce = true;
        out.reduce_ops.push_back(oid);
      } else if (is_scan_op_kind(op.kind)) {
        out.has_scan = true;
        out.scan_ops.push_back(oid);
      } else if (op.kind == OpKind::Dot) {
        out.has_dot = true;
      }
      for (ValueId ov : op.operands) {
        if (ov != kInvalidId) out.value_uses[ov].push_back(oid);
      }
    }
  }

  if (!out.has_reduce && !out.has_scan) {
    return true;
  }

  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    if (kernel.params[i].type.is_tile()) {
      err = "Kernel parameter '" + kernel.params[i].name + "' cannot have tile type";
      return false;
    }
  }

  const Block& entry = kernel.block(kernel.entry_block);
  std::vector<uint32_t> non_constexpr_indices;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    if (!kernel.params[i].is_constexpr) non_constexpr_indices.push_back(i);
  }

  if (entry.args.size() == kernel.params.size()) {
    for (uint32_t i = 0; i < kernel.params.size(); ++i) {
      out.val_to_param_idx[entry.args[i]] = i;
    }
  } else if (entry.args.size() == non_constexpr_indices.size()) {
    for (size_t k = 0; k < non_constexpr_indices.size(); ++k) {
      out.val_to_param_idx[entry.args[k]] = non_constexpr_indices[k];
    }
  } else {
    for (ValueId vid : entry.args) {
      const Value& v = kernel.value(vid);
      if (v.arg_idx < kernel.params.size()) {
        out.val_to_param_idx[vid] = v.arg_idx;
      } else {
        for (uint32_t i = 0; i < kernel.params.size(); ++i) {
          if (!v.debug_name.empty() && kernel.params[i].name == v.debug_name) {
            out.val_to_param_idx[vid] = i;
            break;
          }
        }
      }
    }
  }

  for (const auto& [vid, pidx] : out.val_to_param_idx) {
    const KernelParam& kp = kernel.params[pidx];
    if (kp.is_constexpr) {
      out.const_i64[vid] = kp.constexpr_val;
      if (kp.constexpr_val % 4 == 0) out.div4_values.insert(vid);
      if (kp.constexpr_val >= 0) out.nonneg_values.insert(vid);
    } else if (kp.type.is_ptr()) {
      out.ptr_origins[vid].insert(pidx);
    } else if (kp.divisibility >= 4 && (kp.divisibility % 4 == 0)) {
      out.div4_values.insert(vid);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (BlockId bid : out.all_blocks) {
      for (OpId oid : kernel.block(bid).ops) {
        const Op& op = kernel.op(oid);
        if (op.erased) continue;

        auto merge_origins = [&](ValueId dst, const std::set<uint32_t>& src) {
          auto& d = out.ptr_origins[dst];
          size_t before = d.size();
          d.insert(src.begin(), src.end());
          if (d.size() != before) changed = true;
        };

        switch (op.kind) {
          case OpKind::Constant: {
            if (!op.results.empty() && std::holds_alternative<ConstAttr>(op.attr)) {
              ValueId r = op.results[0];
              if (auto c = const_as_i64(std::get<ConstAttr>(op.attr))) {
                if (!out.const_i64.count(r)) {
                  out.const_i64[r] = *c;
                  changed = true;
                }
                if ((*c % 4 == 0) && out.div4_values.insert(r).second) changed = true;
                if (*c >= 0 && out.nonneg_values.insert(r).second) changed = true;
              }
            }
            break;
          }
          case OpKind::ProgramId:
          case OpKind::NumPrograms: {
            if (!op.results.empty() && out.nonneg_values.insert(op.results[0]).second) changed = true;
            break;
          }
          case OpKind::Cast: {
            if (!op.results.empty() && !op.operands.empty()) {
              ValueId src = op.operands[0];
              ValueId dst = op.results[0];
              if (out.const_i64.count(src) && !out.const_i64.count(dst)) {
                out.const_i64[dst] = out.const_i64[src];
                changed = true;
              }
              if (out.div4_values.count(src) && out.div4_values.insert(dst).second) changed = true;
              if (out.nonneg_values.count(src) && out.nonneg_values.insert(dst).second) changed = true;
            }
            break;
          }
          case OpKind::Binary: {
            if (!op.results.empty() && op.operands.size() >= 2 && std::holds_alternative<BinaryAttr>(op.attr)) {
              ValueId a = op.operands[0];
              ValueId b = op.operands[1];
              ValueId dst = op.results[0];
              BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
              if (bop == BinaryOp::Mul) {
                if ((out.div4_values.count(a) || out.div4_values.count(b)) && out.div4_values.insert(dst).second) {
                  changed = true;
                }
                if (out.nonneg_values.count(a) && out.nonneg_values.count(b) && out.nonneg_values.insert(dst).second) {
                  changed = true;
                }
                if (out.const_i64.count(a) && out.const_i64.count(b) && !out.const_i64.count(dst)) {
                  out.const_i64[dst] = out.const_i64[a] * out.const_i64[b];
                  changed = true;
                }
              } else if (bop == BinaryOp::Add || bop == BinaryOp::Sub) {
                if (out.div4_values.count(a) && out.div4_values.count(b) && out.div4_values.insert(dst).second) {
                  changed = true;
                }
                if (bop == BinaryOp::Add && out.nonneg_values.count(a) && out.nonneg_values.count(b) &&
                    out.nonneg_values.insert(dst).second) {
                  changed = true;
                }
                if (out.const_i64.count(a) && out.const_i64.count(b) && !out.const_i64.count(dst)) {
                  out.const_i64[dst] =
                      (bop == BinaryOp::Add) ? (out.const_i64[a] + out.const_i64[b]) : (out.const_i64[a] - out.const_i64[b]);
                  changed = true;
                }
              }
            }
            break;
          }
          case OpKind::PtrAdd:
          case OpKind::Splat:
          case OpKind::ExpandDims:
          case OpKind::Broadcast:
          case OpKind::Reshape:
          case OpKind::Permute: {
            if (!op.results.empty() && !op.operands.empty()) {
              ValueId src = op.operands[0];
              ValueId dst = op.results[0];
              if (auto it = out.ptr_origins.find(src); it != out.ptr_origins.end()) {
                merge_origins(dst, it->second);
              }
              if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast) {
                if (out.const_i64.count(src) && !out.const_i64.count(dst)) {
                  out.const_i64[dst] = out.const_i64[src];
                  changed = true;
                }
                if (out.div4_values.count(src) && out.div4_values.insert(dst).second) changed = true;
                if (out.nonneg_values.count(src) && out.nonneg_values.insert(dst).second) changed = true;
              }
            }
            break;
          }
          case OpKind::Select: {
            if (!op.results.empty() && op.operands.size() >= 3) {
              for (size_t idx : {size_t{1}, size_t{2}}) {
                if (auto it = out.ptr_origins.find(op.operands[idx]); it != out.ptr_origins.end()) {
                  merge_origins(op.results[0], it->second);
                }
              }
            }
            break;
          }
          case OpKind::ForLoop: {
            if (!op.regions.empty() && op.operands.size() >= 3) {
              const Block& body = kernel.block(op.regions[0]);
              if (!body.args.empty()) {
                ValueId iv = body.args[0];
                if (out.div4_values.count(op.operands[0]) && out.div4_values.count(op.operands[2]) &&
                    out.div4_values.insert(iv).second) {
                  changed = true;
                }
                if (out.nonneg_values.count(op.operands[0]) && out.nonneg_values.count(op.operands[2]) &&
                    out.nonneg_values.insert(iv).second) {
                  changed = true;
                }
              }
              size_t num_iter = op.operands.size() - 3;
              for (size_t i = 0; i < num_iter; ++i) {
                if (auto it = out.ptr_origins.find(op.operands[3 + i]); it != out.ptr_origins.end()) {
                  if (1 + i < body.args.size()) merge_origins(body.args[1 + i], it->second);
                  if (i < op.results.size()) merge_origins(op.results[i], it->second);
                }
              }
              for (OpId boid : body.ops) {
                const Op& bop = kernel.op(boid);
                if (!bop.erased && bop.kind == OpKind::Yield) {
                  for (size_t i = 0; i < bop.operands.size(); ++i) {
                    if (auto it = out.ptr_origins.find(bop.operands[i]); it != out.ptr_origins.end()) {
                      if (1 + i < body.args.size()) merge_origins(body.args[1 + i], it->second);
                      if (i < op.results.size()) merge_origins(op.results[i], it->second);
                    }
                  }
                }
              }
            }
            break;
          }
          case OpKind::If: {
            for (BlockId rb : op.regions) {
              if (rb == kInvalidId) continue;
              for (OpId roid : kernel.block(rb).ops) {
                const Op& rop = kernel.op(roid);
                if (!rop.erased && rop.kind == OpKind::Yield) {
                  for (size_t i = 0; i < rop.operands.size() && i < op.results.size(); ++i) {
                    if (auto it = out.ptr_origins.find(rop.operands[i]); it != out.ptr_origins.end()) {
                      merge_origins(op.results[i], it->second);
                    }
                  }
                }
              }
            }
            break;
          }
          default:
            break;
        }
      }
    }
  }

  for (BlockId bid : out.all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;

      if ((op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) && !op.operands.empty()) {
        if (auto it = out.ptr_origins.find(op.operands[0]); it != out.ptr_origins.end()) {
          out.written_params.insert(it->second.begin(), it->second.end());
        }
      }
      if ((op.kind == OpKind::Reshape || op.kind == OpKind::Permute) && !op.results.empty() &&
          !is_scalar_like(kernel.value(op.results[0]).type)) {
        out.has_tile_reshape_or_permute = true;
      }

      for (ValueId rid : op.results) {
        const Type& ty = kernel.value(rid).type;
        if (!ty.is_tile()) continue;
        const auto& tt = ty.as_tile();
        if (!is_unit_tile(ty)) {
          out.max_rank = std::max(out.max_rank, tt.shape.size());
        } else if (out.max_rank == 0) {
          out.max_rank = 1;
        }
        if (tt.shape.size() == 1 && tt.shape[0] > 1) {
          if (out.block_size_1d == 0) {
            out.block_size_1d = tt.shape[0];
          } else if (out.block_size_1d != tt.shape[0]) {
            out.has_multiple_1d_sizes = true;
            out.block_size_1d = std::max(out.block_size_1d, tt.shape[0]);
          }
        } else if (tt.shape.size() == 2) {
          if (out.primary_2d_shape.first == 0 ||
              (tt.shape[0] * tt.shape[1] > out.primary_2d_shape.first * out.primary_2d_shape.second)) {
            out.primary_2d_shape = {tt.shape[0], tt.shape[1]};
          }
        }
        if (!out.hint_layout.has_value() && tt.layout.kind != ir::LayoutKind::Unassigned) {
          out.hint_layout = tt.layout;
        }
      }
    }
  }
  if (out.block_size_1d == 0 && out.max_rank == 1) {
    out.block_size_1d = 1;
  }

  for (OpId roid : out.reduce_ops) {
    const Op& rop = kernel.op(roid);
    bool needs_bcast = false;
    if (rop.parent_block != kernel.entry_block) {
      needs_bcast = true;
    }
    std::vector<ValueId> wl(rop.results.begin(), rop.results.end());
    std::unordered_set<ValueId> visited(wl.begin(), wl.end());
    while (!wl.empty() && !needs_bcast) {
      ValueId v = wl.back();
      wl.pop_back();
      auto uit = out.value_uses.find(v);
      if (uit == out.value_uses.end()) continue;
      for (OpId uoid : uit->second) {
        const Op& uop = kernel.op(uoid);
        if (uop.erased) continue;
        if (uop.kind == OpKind::Yield || uop.kind == OpKind::ForLoop || uop.kind == OpKind::If) {
          needs_bcast = true;
          break;
        }
        if (uop.kind == OpKind::Splat || uop.kind == OpKind::Broadcast || uop.kind == OpKind::ExpandDims) {
          if (!uop.results.empty() && !is_scalar_like(kernel.value(uop.results[0]).type)) {
            needs_bcast = true;
            break;
          }
        }
        for (ValueId o : uop.operands) {
          if (o != kInvalidId && !is_scalar_like(kernel.value(o).type)) {
            needs_bcast = true;
            break;
          }
        }
        for (ValueId r : uop.results) {
          if (!is_scalar_like(kernel.value(r).type)) {
            needs_bcast = true;
            break;
          }
          if (visited.insert(r).second) wl.push_back(r);
        }
      }
    }
    if (needs_bcast) out.broadcast_reduces.insert(roid);
  }

  uint32_t next_slot = 0;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const KernelParam& kp = kernel.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot slot;
    slot.param_index = i;
    slot.buffer_index = next_slot++;
    slot.name = kp.name.empty() ? ("m1_arg" + std::to_string(slot.buffer_index)) : kp.name;
    slot.is_pointer = kp.type.is_ptr();
    slot.writable = slot.is_pointer && (kp.writable || out.written_params.count(i) > 0);
    slot.dtype = scalar_dtype_of(kp.type);
    out.param_msl_name[i] = slot.name;
    out.param_slots.push_back(std::move(slot));
  }

  return true;
}

bool ptr_value_is_writable(const ReduceKernelAnalysis& analysis, ValueId vid) {
  auto it = analysis.ptr_origins.find(vid);
  if (it == analysis.ptr_origins.end() || it->second.empty()) return false;
  for (uint32_t pidx : it->second) {
    bool slot_writable = false;
    for (const auto& slot : analysis.param_slots) {
      if (slot.param_index == pidx) {
        slot_writable = slot.writable;
        break;
      }
    }
    if (!slot_writable) return false;
  }
  return true;
}

std::string msl_ptr_type_str(const ReduceKernelAnalysis& analysis, ValueId vid, ScalarDType pointee) {
  bool writable = ptr_value_is_writable(analysis, vid);
  return std::string("device ") + (writable ? "" : "const ") + msl_scalar_type(pointee) + "*";
}

void populate_entry_param_exprs(const Kernel& kernel, const ReduceKernelAnalysis& analysis,
                                std::unordered_map<ValueId, std::string>& expr_map) {
  for (const auto& [vid, pidx] : analysis.val_to_param_idx) {
    const KernelParam& kp = kernel.params[pidx];
    if (kp.is_constexpr) {
      expr_map[vid] = format_scalar_literal_from_i64(kp.constexpr_val, scalar_dtype_of(kp.type));
    } else {
      expr_map[vid] = analysis.param_msl_name.at(pidx);
    }
  }
}

std::string build_msl_preamble_and_sig(const Kernel& kernel, const ReduceKernelAnalysis& analysis,
                                       const HelperFlags& helpers) {
  std::ostringstream os;
  os << "#include <metal_stdlib>\n";
  os << "using namespace metal;\n\n";

  if (helpers.needs_erf) {
    os << "template <typename T>\n"
       << "inline T m1_erf(T x) {\n"
       << "  T sign = metal::select(T(1.0f), T(-1.0f), x < T(0.0f));\n"
       << "  T ax = metal::abs(x);\n"
       << "  T t = T(1.0f) / (T(1.0f) + T(0.3275911f) * ax);\n"
       << "  T y = T(1.0f) - (((((T(1.061405429f) * t + T(-1.453152027f)) * t) + T(1.421413741f)) * t + "
          "T(-0.284496736f)) * t + T(0.254829592f)) * t * metal::exp(-ax * ax);\n"
       << "  return sign * y;\n"
       << "}\n\n";
  }

  if (helpers.needs_tanh) {
    os << "template <typename T>\n"
       << "inline T m1_tanh(T x) {\n"
       << "  T clamped = metal::select(x, metal::select(T(-10.0f), T(10.0f), x > T(0.0f)), metal::abs(x) > T(10.0f));\n"
       << "  return metal::tanh(clamped);\n"
       << "}\n\n";
  }

  os << "kernel void " << kernel.name << "(\n";
  for (const auto& slot : analysis.param_slots) {
    if (slot.is_pointer) {
      os << "    device " << (slot.writable ? "" : "const ") << msl_scalar_type(slot.dtype) << "* " << slot.name
         << " [[buffer(" << slot.buffer_index << ")]],\n";
    } else {
      os << "    constant " << msl_scalar_type(slot.dtype) << "& " << slot.name << " [[buffer(" << slot.buffer_index
         << ")]],\n";
    }
  }
  os << "    uint3 m1_tg_pos [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint3 m1_tid [[thread_position_in_threadgroup]],\n"
     << "    uint3 m1_tg_size [[threads_per_threadgroup]],\n"
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
     << "    uint m1_simd_lane [[thread_index_in_simdgroup]],\n"
     << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]]) {\n";
  return os.str();
}

std::string emit_scalar_op_expr(const Kernel& kernel, const ReduceKernelAnalysis& analysis, const Op& op,
                                const std::unordered_map<ValueId, std::string>& expr_map, HelperFlags& helpers,
                                std::string& err) {
  auto get_e = [&](ValueId id) -> std::string {
    auto it = expr_map.find(id);
    if (it != expr_map.end()) return it->second;
    return ssa_var(id);
  };

  ScalarDType res_dt = op.results.empty() ? ScalarDType::F32 : scalar_dtype_of(kernel.value(op.results[0]).type);
  switch (op.kind) {
    case OpKind::Constant:
      return msl_const_literal(std::get<ConstAttr>(op.attr), res_dt);
    case OpKind::ProgramId: {
      uint32_t axis = std::get<AxisAttr>(op.attr).axis;
      return msl_scalar_type(res_dt) + "(m1_tg_pos." + axis_component(axis) + ")";
    }
    case OpKind::NumPrograms: {
      uint32_t axis = std::get<AxisAttr>(op.attr).axis;
      return msl_scalar_type(res_dt) + "(m1_grid_dim." + axis_component(axis) + ")";
    }
    case OpKind::Arange: {
      const auto& attr = std::get<ArangeAttr>(op.attr);
      return format_scalar_literal_from_i64(attr.start, res_dt);
    }
    case OpKind::Splat:
    case OpKind::Broadcast:
    case OpKind::ExpandDims:
    case OpKind::Reshape:
      return get_e(op.operands[0]);
    case OpKind::Cast: {
      ScalarDType src_dt = scalar_dtype_of(kernel.value(op.operands[0]).type);
      return msl_cast_expr(get_e(op.operands[0]), src_dt, res_dt, false);
    }
    case OpKind::Unary: {
      UnaryOp uop = std::get<UnaryAttr>(op.attr).op;
      return msl_unary_expr(uop, get_e(op.operands[0]), res_dt, false, helpers);
    }
    case OpKind::Binary: {
      BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
      ScalarDType ldt = scalar_dtype_of(kernel.value(op.operands[0]).type);
      ScalarDType rdt = scalar_dtype_of(kernel.value(op.operands[1]).type);
      return msl_binary_expr(bop, get_e(op.operands[0]), get_e(op.operands[1]), ldt, rdt, res_dt, false);
    }
    case OpKind::Select: {
      ScalarDType tdt = scalar_dtype_of(kernel.value(op.operands[1]).type);
      ScalarDType fdt = scalar_dtype_of(kernel.value(op.operands[2]).type);
      return msl_select_expr(get_e(op.operands[0]), get_e(op.operands[1]), get_e(op.operands[2]), tdt, fdt, res_dt,
                             false);
    }
    case OpKind::PtrAdd:
      return "(" + get_e(op.operands[0]) + " + " + get_e(op.operands[1]) + ")";
    case OpKind::Load: {
      auto lo = extract_load_operands(op, kernel);
      std::string ptr_e = get_e(lo.ptrs);
      if (lo.mask == kInvalidId) return "(*(" + ptr_e + "))";
      std::string other_e = (lo.other != kInvalidId)
                                ? msl_cast_expr(get_e(lo.other), scalar_dtype_of(kernel.value(lo.other).type), res_dt,
                                                false)
                                : msl_zero_literal(res_dt, false);
      return "(" + get_e(lo.mask) + " ? (*(" + ptr_e + ")) : " + other_e + ")";
    }
    case OpKind::AtomicAdd: {
      std::string ptr_e = get_e(op.operands[0]);
      std::string val_e = msl_cast_expr(get_e(op.operands[1]), scalar_dtype_of(kernel.value(op.operands[1]).type),
                                        res_dt, false);
      std::string atomic_ty;
      if (res_dt == ScalarDType::F32) {
        atomic_ty = "atomic_float";
      } else if (res_dt == ScalarDType::I32) {
        atomic_ty = "atomic_int";
      } else if (res_dt == ScalarDType::U32) {
        atomic_ty = "atomic_uint";
      } else {
        err = "Unsupported AtomicAdd dtype: " + msl_scalar_type(res_dt);
        return "";
      }
      std::string call = "atomic_fetch_add_explicit(reinterpret_cast<device " + atomic_ty + "*>(" + ptr_e + "), " +
                         val_e + ", memory_order_relaxed)";
      if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) {
        return "(" + get_e(op.operands[2]) + " ? " + call + " : " + msl_zero_literal(res_dt, false) + ")";
      }
      return call;
    }
    default:
      (void)analysis;
      err = "Unsupported scalar op kind in reduce/scan emitter";
      return "";
  }
}

struct SmemRegistry {
  uint32_t num_sg = 1;
  bool need_f32 = false;
  bool need_f16 = false;
  bool need_i32 = false;
  bool need_u32 = false;
  bool need_i64 = false;
  bool need_idx = false;

  std::string array_for(ScalarDType dt) {
    switch (dt) {
      case ScalarDType::F32:
        need_f32 = true;
        return "m1_smem";
      case ScalarDType::F16:
        need_f16 = true;
        return "m1_smem_f16";
      case ScalarDType::I32:
      case ScalarDType::Bool:
        need_i32 = true;
        return "m1_smem_i32";
      case ScalarDType::U32:
        need_u32 = true;
        return "m1_smem_u32";
      case ScalarDType::I64:
        need_i64 = true;
        return "m1_smem_i64";
    }
    need_f32 = true;
    return "m1_smem";
  }

  std::string idx_array() {
    need_idx = true;
    return "m1_smem_idx";
  }

  std::pair<std::string, uint32_t> emit_decls() const {
    if (num_sg <= 1) return {"", 0u};
    std::ostringstream os;
    uint32_t total_bytes = 0;
    if (need_f32) {
      os << "  threadgroup float m1_smem[" << num_sg << "];\n";
      total_bytes += num_sg * 4u;
    }
    if (need_f16) {
      os << "  threadgroup half m1_smem_f16[" << num_sg << "];\n";
      total_bytes += num_sg * 2u;
    }
    if (need_i32) {
      os << "  threadgroup int m1_smem_i32[" << num_sg << "];\n";
      total_bytes += num_sg * 4u;
    }
    if (need_u32) {
      os << "  threadgroup uint m1_smem_u32[" << num_sg << "];\n";
      total_bytes += num_sg * 4u;
    }
    if (need_i64) {
      os << "  threadgroup long m1_smem_i64[" << num_sg << "];\n";
      total_bytes += num_sg * 8u;
    }
    if (need_idx) {
      os << "  threadgroup int m1_smem_idx[" << num_sg << "];\n";
      total_bytes += num_sg * 4u;
    }
    return {os.str(), align_up_16(total_bytes)};
  }
};

std::string simd_op_name(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::Sum:
      return "simd_sum";
    case ReduceKind::Max:
      return "simd_max";
    case ReduceKind::Min:
      return "simd_min";
    default:
      return "simd_sum";
  }
}

std::string scalar_combine_expr(ReduceKind kind, const std::string& a, const std::string& b) {
  if (kind == ReduceKind::Sum) return "(" + a + " + " + b + ")";
  if (kind == ReduceKind::Max || kind == ReduceKind::Argmax) return "metal::max(" + a + ", " + b + ")";
  return "metal::min(" + a + ", " + b + ")";
}

void emit_simd_reduce_i64_butterfly(ReduceKind kind, const std::string& var, uint32_t start_off, int ind,
                                    std::ostringstream& out) {
  std::string lo_v = var + "_lo";
  std::string hi_v = var + "_hi";
  std::string o_v = var + "_ov";
  out << indent_str(ind) << "for (ushort m1_off = " << start_off << "u; m1_off > 0u; m1_off >>= 1u) {\n";
  out << indent_str(ind + 1) << "uint2 m1_bits = as_type<uint2>(" << var << ");\n";
  out << indent_str(ind + 1) << "uint " << lo_v << " = simd_shuffle_xor(m1_bits.x, m1_off);\n";
  out << indent_str(ind + 1) << "uint " << hi_v << " = simd_shuffle_xor(m1_bits.y, m1_off);\n";
  out << indent_str(ind + 1) << "long " << o_v << " = as_type<long>(uint2(" << lo_v << ", " << hi_v << "));\n";
  out << indent_str(ind + 1) << var << " = " << scalar_combine_expr(kind, var, o_v) << ";\n";
  out << indent_str(ind) << "}\n";
}

void emit_simd_arg_butterfly(bool is_max, ScalarDType dt, const std::string& val_var, const std::string& idx_var,
                             uint32_t start_off, int ind, std::ostringstream& out) {
  const std::string cmp = is_max ? " > " : " < ";
  const std::string ty = msl_scalar_type(dt);
  out << indent_str(ind) << "for (ushort m1_off = " << start_off << "u; m1_off > 0u; m1_off >>= 1u) {\n";
  if (dt == ScalarDType::I64) {
    out << indent_str(ind + 1) << "uint2 m1_ab = as_type<uint2>(" << val_var << ");\n";
    out << indent_str(ind + 1) << "long m1_ov = as_type<long>(uint2(simd_shuffle_xor(m1_ab.x, m1_off), "
                                  "simd_shuffle_xor(m1_ab.y, m1_off)));\n";
  } else {
    out << indent_str(ind + 1) << ty << " m1_ov = simd_shuffle_xor(" << val_var << ", m1_off);\n";
  }
  out << indent_str(ind + 1) << "int m1_oi = simd_shuffle_xor(" << idx_var << ", m1_off);\n";
  out << indent_str(ind + 1) << "bool m1_take = (m1_ov" << cmp << val_var << ") || ((m1_ov == " << val_var
      << ") && (m1_oi < " << idx_var << "));\n";
  out << indent_str(ind + 1) << val_var << " = m1_take ? m1_ov : " << val_var << ";\n";
  out << indent_str(ind + 1) << idx_var << " = m1_take ? m1_oi : " << idx_var << ";\n";
  out << indent_str(ind) << "}\n";
}

void emit_cross_simd_value_reduce(ReduceKind kind, ScalarDType dt, const std::string& local_var,
                                  const std::string& out_var, uint32_t num_sg, bool broadcast_all,
                                  bool emit_trailing_barrier, SmemRegistry& smem, int ind, std::ostringstream& out) {
  ScalarDType red_dt = (dt == ScalarDType::Bool) ? ScalarDType::I32 : dt;
  std::string ty = msl_scalar_type(red_dt);
  std::string ident = msl_reduce_identity(kind, red_dt);
  std::string work_var = local_var;

  if (dt == ScalarDType::Bool) {
    work_var = out_var + "_b";
    out << indent_str(ind) << "int " << work_var << " = int(" << local_var << ");\n";
  }

  if (red_dt == ScalarDType::I64) {
    emit_simd_reduce_i64_butterfly(kind, work_var, 16u, ind, out);
  } else {
    out << indent_str(ind) << work_var << " = " << simd_op_name(kind) << "(" << work_var << ");\n";
  }

  if (num_sg <= 1u) {
    std::string final_e = (dt == ScalarDType::Bool) ? ("(" + work_var + " != 0)") : work_var;
    out << indent_str(ind) << msl_scalar_type(dt) << " " << out_var << " = " << final_e << ";\n";
    return;
  }

  std::string sarr = smem.array_for(red_dt);
  out << indent_str(ind) << "if (m1_simd_lane == 0u) { " << sarr << "[m1_simd_id] = " << work_var << "; }\n";
  out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  if (broadcast_all) {
    out << indent_str(ind) << "if (m1_simd_id == 0u) {\n";
    out << indent_str(ind + 1) << ty << " m1_sgv = (m1_simd_lane < " << num_sg << "u) ? " << sarr
        << "[m1_simd_lane] : " << ident << ";\n";
    if (red_dt == ScalarDType::I64) {
      uint32_t p2 = next_pow2_u32(num_sg);
      emit_simd_reduce_i64_butterfly(kind, "m1_sgv", std::max<uint32_t>(1u, p2 >> 1u), ind + 1, out);
    } else {
      out << indent_str(ind + 1) << "m1_sgv = " << simd_op_name(kind) << "(m1_sgv);\n";
    }
    out << indent_str(ind + 1) << "if (m1_simd_lane == 0u) { " << sarr << "[0] = m1_sgv; }\n";
    out << indent_str(ind) << "}\n";
    out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    std::string read_e = (dt == ScalarDType::Bool) ? ("(" + sarr + "[0] != 0)") : (sarr + "[0]");
    out << indent_str(ind) << msl_scalar_type(dt) << " " << out_var << " = " << read_e << ";\n";
    if (emit_trailing_barrier) {
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
  } else {
    out << indent_str(ind) << msl_scalar_type(dt) << " " << out_var << " = " << msl_reduce_identity(kind, dt) << ";\n";
    out << indent_str(ind) << "if (m1_simd_id == 0u) {\n";
    out << indent_str(ind + 1) << ty << " m1_sgv = (m1_simd_lane < " << num_sg << "u) ? " << sarr
        << "[m1_simd_lane] : " << ident << ";\n";
    if (red_dt == ScalarDType::I64) {
      uint32_t p2 = next_pow2_u32(num_sg);
      emit_simd_reduce_i64_butterfly(kind, "m1_sgv", std::max<uint32_t>(1u, p2 >> 1u), ind + 1, out);
    } else {
      out << indent_str(ind + 1) << "m1_sgv = " << simd_op_name(kind) << "(m1_sgv);\n";
    }
    std::string fin = (dt == ScalarDType::Bool) ? "(m1_sgv != 0)" : "m1_sgv";
    out << indent_str(ind + 1) << out_var << " = " << fin << ";\n";
    out << indent_str(ind) << "}\n";
    if (emit_trailing_barrier) {
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
  }
}

void emit_cross_simd_arg_reduce(bool is_max, ScalarDType dt, const std::string& local_val, const std::string& local_idx,
                                const std::string& out_val, const std::string& out_idx, uint32_t num_sg,
                                bool broadcast_all, bool emit_trailing_barrier, SmemRegistry& smem, int ind,
                                std::ostringstream& out) {
  ScalarDType red_dt = (dt == ScalarDType::Bool) ? ScalarDType::I32 : dt;
  std::string ty = msl_scalar_type(red_dt);
  ReduceKind rk = is_max ? ReduceKind::Argmax : ReduceKind::Argmin;
  std::string ident = msl_reduce_identity(rk, red_dt);
  std::string work_val = local_val;

  if (dt == ScalarDType::Bool) {
    work_val = out_val + "_b";
    out << indent_str(ind) << "int " << work_val << " = int(" << local_val << ");\n";
  }

  emit_simd_arg_butterfly(is_max, red_dt, work_val, local_idx, 16u, ind, out);

  if (num_sg <= 1u) {
    if (!out_val.empty()) {
      std::string fv = (dt == ScalarDType::Bool) ? ("(" + work_val + " != 0)") : work_val;
      out << indent_str(ind) << msl_scalar_type(dt) << " " << out_val << " = " << fv << ";\n";
    }
    if (!out_idx.empty()) {
      out << indent_str(ind) << "int " << out_idx << " = " << local_idx << ";\n";
    }
    return;
  }

  std::string svarr = smem.array_for(red_dt);
  std::string siarr = smem.idx_array();
  uint32_t p2 = next_pow2_u32(num_sg);
  uint32_t start_off = std::max<uint32_t>(1u, p2 >> 1u);

  out << indent_str(ind) << "if (m1_simd_lane == 0u) { " << svarr << "[m1_simd_id] = " << work_val << "; " << siarr
      << "[m1_simd_id] = " << local_idx << "; }\n";
  out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  if (broadcast_all) {
    out << indent_str(ind) << "if (m1_simd_id == 0u) {\n";
    out << indent_str(ind + 1) << "bool m1_act = (m1_simd_lane < " << num_sg << "u);\n";
    out << indent_str(ind + 1) << ty << " m1_sgv = m1_act ? " << svarr << "[m1_simd_lane] : " << ident << ";\n";
    out << indent_str(ind + 1) << "int m1_sgi = m1_act ? " << siarr << "[m1_simd_lane] : 2147483647;\n";
    emit_simd_arg_butterfly(is_max, red_dt, "m1_sgv", "m1_sgi", start_off, ind + 1, out);
    out << indent_str(ind + 1) << "if (m1_simd_lane == 0u) { " << svarr << "[0] = m1_sgv; " << siarr
        << "[0] = m1_sgi; }\n";
    out << indent_str(ind) << "}\n";
    out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    if (!out_val.empty()) {
      std::string fv = (dt == ScalarDType::Bool) ? ("(" + svarr + "[0] != 0)") : (svarr + "[0]");
      out << indent_str(ind) << msl_scalar_type(dt) << " " << out_val << " = " << fv << ";\n";
    }
    if (!out_idx.empty()) {
      out << indent_str(ind) << "int " << out_idx << " = " << siarr << "[0];\n";
    }
    if (emit_trailing_barrier) {
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
  } else {
    if (!out_val.empty()) {
      out << indent_str(ind) << msl_scalar_type(dt) << " " << out_val << " = " << msl_reduce_identity(rk, dt) << ";\n";
    }
    if (!out_idx.empty()) {
      out << indent_str(ind) << "int " << out_idx << " = 0;\n";
    }
    out << indent_str(ind) << "if (m1_simd_id == 0u) {\n";
    out << indent_str(ind + 1) << "bool m1_act = (m1_simd_lane < " << num_sg << "u);\n";
    out << indent_str(ind + 1) << ty << " m1_sgv = m1_act ? " << svarr << "[m1_simd_lane] : " << ident << ";\n";
    out << indent_str(ind + 1) << "int m1_sgi = m1_act ? " << siarr << "[m1_simd_lane] : 2147483647;\n";
    emit_simd_arg_butterfly(is_max, red_dt, "m1_sgv", "m1_sgi", start_off, ind + 1, out);
    if (!out_val.empty()) {
      std::string fv = (dt == ScalarDType::Bool) ? "(m1_sgv != 0)" : "m1_sgv";
      out << indent_str(ind + 1) << out_val << " = " << fv << ";\n";
    }
    if (!out_idx.empty()) {
      out << indent_str(ind + 1) << out_idx << " = m1_sgi;\n";
    }
    out << indent_str(ind) << "}\n";
    if (emit_trailing_barrier) {
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
  }
}

struct LinearInt {
  bool valid = false;
  int64_t elem_stride = 0;
  std::string dyn_stride_expr;
  int64_t iv_coeff = 0;
  std::string inv_expr;
  bool inv_is_zero = false;
  bool inv_div4 = false;
  ScalarDType dtype = ScalarDType::I32;
};

struct LinearPtr {
  bool valid = false;
  int64_t elem_stride = 0;
  std::string dyn_stride_expr;
  int64_t iv_coeff = 0;
  std::string base_ptr;
  std::string inv_offset;
  bool inv_is_zero = false;
  bool inv_div4 = false;
  ScalarDType offset_dtype = ScalarDType::I32;
  std::string poff_var;
  std::string poff_delta_expr;
  bool poff_delta_div4 = true;
};

std::string combine_add_expr(const std::string& a, bool a_zero, const std::string& b, bool b_zero, ScalarDType dt) {
  if (a_zero && b_zero) return format_scalar_literal_from_i64(0, dt);
  if (a_zero) return b;
  if (b_zero) return a;
  return "(" + a + " + " + b + ")";
}

std::string combine_sub_expr(const std::string& a, bool a_zero, const std::string& b, bool b_zero, ScalarDType dt) {
  if (b_zero) return a_zero ? format_scalar_literal_from_i64(0, dt) : a;
  if (a_zero) return "(-" + b + ")";
  return "(" + a + " - " + b + ")";
}

struct RampBound {
  bool is_upper = true;
  bool strict = true;
  LinearInt ramp;
  std::string bound_expr;
  ScalarDType bound_dtype = ScalarDType::I32;
};

struct DecomposedMask {
  bool ok = true;
  std::vector<std::string> inv_scalar_guards;
  std::vector<std::string> var_scalar_guards;
  std::vector<RampBound> ramp_bounds;
};

bool decompose_mask_rec(const Kernel& kernel, const std::unordered_map<ValueId, LinearInt>& lin_ints,
                        const std::unordered_map<ValueId, std::string>& scalar_exprs,
                        const std::unordered_set<ValueId>& loop_inv_vals, ValueId mask_id, DecomposedMask& out) {
  if (mask_id == kInvalidId) return true;
  const Value& mv = kernel.value(mask_id);
  if (is_scalar_like(mv.type)) {
    auto it = scalar_exprs.find(mask_id);
    if (it == scalar_exprs.end()) return false;
    if (loop_inv_vals.count(mask_id) > 0) {
      out.inv_scalar_guards.push_back(it->second);
    } else {
      out.var_scalar_guards.push_back(it->second);
    }
    return true;
  }
  if (mv.def_op == kInvalidId) return false;
  const Op& op = kernel.op(mv.def_op);
  if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) {
    return decompose_mask_rec(kernel, lin_ints, scalar_exprs, loop_inv_vals, op.operands[0], out);
  }
  if (op.kind == OpKind::Binary && std::holds_alternative<BinaryAttr>(op.attr)) {
    BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
    if (bop == BinaryOp::And) {
      return decompose_mask_rec(kernel, lin_ints, scalar_exprs, loop_inv_vals, op.operands[0], out) &&
             decompose_mask_rec(kernel, lin_ints, scalar_exprs, loop_inv_vals, op.operands[1], out);
    }
    auto it_l = lin_ints.find(op.operands[0]);
    auto it_r = lin_ints.find(op.operands[1]);
    if (it_l == lin_ints.end() || it_r == lin_ints.end() || !it_l->second.valid || !it_r->second.valid) {
      return false;
    }
    const LinearInt& L = it_l->second;
    const LinearInt& R = it_r->second;
    if (!L.dyn_stride_expr.empty() || !R.dyn_stride_expr.empty()) return false;

    if (L.elem_stride == 1 && R.elem_stride == 0 && R.iv_coeff == 0) {
      if (bop == BinaryOp::CmpLt) {
        out.ramp_bounds.push_back(RampBound{true, true, L, R.inv_expr, R.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpLe) {
        out.ramp_bounds.push_back(RampBound{true, false, L, R.inv_expr, R.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpGe) {
        out.ramp_bounds.push_back(RampBound{false, false, L, R.inv_expr, R.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpGt) {
        out.ramp_bounds.push_back(RampBound{false, true, L, R.inv_expr, R.dtype});
        return true;
      }
    } else if (L.elem_stride == 0 && L.iv_coeff == 0 && R.elem_stride == 1) {
      if (bop == BinaryOp::CmpGt) {
        out.ramp_bounds.push_back(RampBound{true, true, R, L.inv_expr, L.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpGe) {
        out.ramp_bounds.push_back(RampBound{true, false, R, L.inv_expr, L.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpLe) {
        out.ramp_bounds.push_back(RampBound{false, false, R, L.inv_expr, L.dtype});
        return true;
      }
      if (bop == BinaryOp::CmpLt) {
        out.ramp_bounds.push_back(RampBound{false, true, R, L.inv_expr, L.dtype});
        return true;
      }
    } else if (L.elem_stride == 0 && R.elem_stride == 0 && L.iv_coeff == 0 && R.iv_coeff == 0) {
      std::string e = msl_binary_expr(bop, L.inv_expr, R.inv_expr, L.dtype, R.dtype, ScalarDType::Bool, false);
      out.inv_scalar_guards.push_back(e);
      return true;
    }
  }
  return false;
}

bool detect_ptr_iter_delta(const Kernel& kernel, ValueId arg_id, ValueId yield_val, ValueId& delta_out) {
  const Value& yv = kernel.value(yield_val);
  if (yv.def_op == kInvalidId) return false;
  const Op& yop = kernel.op(yv.def_op);
  if (yop.kind != OpKind::PtrAdd || yop.operands.size() < 2) return false;
  if (yop.operands[0] != arg_id) return false;
  ValueId off = yop.operands[1];
  while (off != kInvalidId && kernel.value(off).type.is_tile()) {
    const Value& ov = kernel.value(off);
    if (ov.def_op == kInvalidId) return false;
    const Op& oop = kernel.op(ov.def_op);
    if ((oop.kind == OpKind::Splat || oop.kind == OpKind::Broadcast) && !oop.operands.empty()) {
      off = oop.operands[0];
    } else {
      return false;
    }
  }
  if (off == kInvalidId) return false;
  delta_out = off;
  return true;
}

std::optional<ValueId> unwrap_scalar_splat(const Kernel& kernel, ValueId vid) {
  ValueId cur = vid;
  while (cur != kInvalidId && !is_scalar_like(kernel.value(cur).type)) {
    const Value& v = kernel.value(cur);
    if (v.def_op == kInvalidId) return std::nullopt;
    const Op& op = kernel.op(v.def_op);
    if ((op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) &&
        !op.operands.empty()) {
      cur = op.operands[0];
    } else {
      return std::nullopt;
    }
  }
  if (cur == kInvalidId) return std::nullopt;
  return cur;
}

std::optional<ConstAttr> unwrap_const_splat(const Kernel& kernel, ValueId vid) {
  auto sc = unwrap_scalar_splat(kernel, vid);
  if (!sc.has_value()) return std::nullopt;
  ValueId cur = *sc;
  const Value& v = kernel.value(cur);
  if (v.def_op != kInvalidId) {
    const Op& op = kernel.op(v.def_op);
    if (op.kind == OpKind::Constant && std::holds_alternative<ConstAttr>(op.attr)) {
      return std::get<ConstAttr>(op.attr);
    }
    if (op.kind == OpKind::Cast && !op.operands.empty()) {
      return unwrap_const_splat(kernel, op.operands[0]);
    }
  }
  return std::nullopt;
}

struct CollapsibleAccInfo {
  ReduceKind kind = ReduceKind::Sum;
  ScalarDType dtype = ScalarDType::F32;
};

std::optional<CollapsibleAccInfo> detect_collapsible_acc(const Kernel& kernel, const ReduceKernelAnalysis& analysis,
                                                         const Op& loop_op, size_t iter_idx) {
  if (loop_op.regions.empty()) return std::nullopt;
  const Block& body = kernel.block(loop_op.regions[0]);
  if (1 + iter_idx >= body.args.size() || iter_idx >= loop_op.results.size()) return std::nullopt;
  ValueId init_v = loop_op.operands[3 + iter_idx];
  ValueId arg_v = body.args[1 + iter_idx];
  ValueId res_v = loop_op.results[iter_idx];
  const Type& ty = kernel.value(arg_v).type;
  if (is_scalar_like(ty) || is_ptr_type(ty)) return std::nullopt;
  ScalarDType dt = scalar_dtype_of(ty);

  auto uit = analysis.value_uses.find(res_v);
  if (uit == analysis.value_uses.end() || uit->second.empty()) return std::nullopt;
  std::optional<ReduceKind> target_kind;
  for (OpId uoid : uit->second) {
    const Op& uop = kernel.op(uoid);
    if (uop.erased) continue;
    if (!is_reduce_op_kind(uop.kind) || !std::holds_alternative<ReduceAttr>(uop.attr)) return std::nullopt;
    const auto& ra = std::get<ReduceAttr>(uop.attr);
    if (ra.result_mode != ReduceResultMode::ValueOnly) return std::nullopt;
    if (ra.kind != ReduceKind::Sum && ra.kind != ReduceKind::Max && ra.kind != ReduceKind::Min) return std::nullopt;
    if (!target_kind.has_value()) {
      target_kind = ra.kind;
    } else if (*target_kind != ra.kind) {
      return std::nullopt;
    }
  }
  if (!target_kind.has_value()) return std::nullopt;

  auto c_attr = unwrap_const_splat(kernel, init_v);
  if (!c_attr.has_value() || !const_matches_reduce_identity(*c_attr, *target_kind, dt)) {
    return std::nullopt;
  }

  BinaryOp expected_bop = (*target_kind == ReduceKind::Sum)
                              ? BinaryOp::Add
                              : ((*target_kind == ReduceKind::Max) ? BinaryOp::Max : BinaryOp::Min);

  const Op* yield_op = nullptr;
  for (OpId boid : body.ops) {
    const Op& bop = kernel.op(boid);
    if (!bop.erased && bop.kind == OpKind::Yield) {
      yield_op = &bop;
      break;
    }
  }
  if (!yield_op || iter_idx >= yield_op->operands.size()) return std::nullopt;
  ValueId yield_v = yield_op->operands[iter_idx];

  ValueId cur = arg_v;
  std::unordered_set<ValueId> chain{cur};
  while (cur != yield_v) {
    auto cur_uses = analysis.value_uses.find(cur);
    if (cur_uses == analysis.value_uses.end() || cur_uses->second.size() != 1) return std::nullopt;
    const Op& uop = kernel.op(cur_uses->second[0]);
    if (uop.erased || uop.kind != OpKind::Binary || !std::holds_alternative<BinaryAttr>(uop.attr)) {
      return std::nullopt;
    }
    if (std::get<BinaryAttr>(uop.attr).op != expected_bop) return std::nullopt;
    if (uop.operands.size() < 2 || uop.results.empty()) return std::nullopt;
    if (uop.operands[0] == cur && uop.operands[1] == cur) return std::nullopt;
    cur = uop.results[0];
    if (!chain.insert(cur).second) return std::nullopt;
  }

  return CollapsibleAccInfo{*target_kind, dt};
}

// ---------------------------------------------------------------------------
// 1D Vectorized (vec4 + loop peeling) Reduction Emitter
// ---------------------------------------------------------------------------

struct Vec4ReduceEmitter {
  const Kernel& kernel;
  const ReduceKernelAnalysis& analysis;
  const TargetSpec& target;
  int64_t block_size = 1;
  uint32_t tg_x = 256;
  uint32_t num_sg = 8;
  uint32_t chunks = 1;
  bool exact_cover = true;

  HelperFlags helpers;
  SmemRegistry smem;
  std::string err;
  uint32_t tmp_seq = 0;

  std::unordered_map<ValueId, std::string> scalar_exprs;
  std::unordered_map<ValueId, LinearInt> lin_ints;
  std::unordered_map<ValueId, LinearPtr> lin_ptrs;
  std::unordered_set<ValueId> loop_inv_vals;

  struct MatTile {
    std::string var_name;
    ScalarDType dtype = ScalarDType::F32;
    bool collapsed = false;
    ReduceKind collapse_kind = ReduceKind::Sum;
  };
  std::unordered_map<ValueId, MatTile> mat_tiles;
  std::unordered_map<ValueId, std::pair<ValueId, std::string>> ptr_iter_map;

  std::string next_tmp(const std::string& pfx) { return pfx + "_" + std::to_string(tmp_seq++); }

  std::string chunk_elem(const MatTile& mt, const std::string& c_var) const {
    if (chunks == 1 || mt.collapsed) return mt.var_name;
    return mt.var_name + "[" + c_var + "]";
  }

  void seed_entry_params() {
    populate_entry_param_exprs(kernel, analysis, scalar_exprs);
    for (const auto& [vid, pidx] : analysis.val_to_param_idx) {
      loop_inv_vals.insert(vid);
      const KernelParam& kp = kernel.params[pidx];
      if (kp.type.is_ptr()) {
        LinearPtr lp;
        lp.valid = true;
        lp.elem_stride = 0;
        lp.base_ptr = scalar_exprs[vid];
        lp.inv_offset = "0";
        lp.inv_is_zero = true;
        lp.inv_div4 = true;
        lin_ptrs[vid] = lp;
      } else if (is_int_dtype(scalar_dtype_of(kp.type))) {
        LinearInt li;
        li.valid = true;
        li.elem_stride = 0;
        li.inv_expr = scalar_exprs[vid];
        li.dtype = scalar_dtype_of(kp.type);
        li.inv_is_zero = (kp.is_constexpr && kp.constexpr_val == 0);
        li.inv_div4 = analysis.div4_values.count(vid) > 0;
        lin_ints[vid] = li;
      }
    }
  }

  void track_linear_info(const Op& op, bool in_loop) {
    if (op.results.empty()) return;
    ValueId rid = op.results[0];
    const Type& rty = kernel.value(rid).type;

    if (op.kind == OpKind::Constant) {
      ScalarDType rdt = scalar_dtype_of(rty);
      if (is_int_dtype(rdt)) {
        LinearInt li;
        li.valid = true;
        li.dtype = rdt;
        li.inv_expr = msl_const_literal(std::get<ConstAttr>(op.attr), rdt);
        auto c = const_as_i64(std::get<ConstAttr>(op.attr));
        li.inv_is_zero = (c.has_value() && *c == 0);
        li.inv_div4 = analysis.div4_values.count(rid) > 0;
        lin_ints[rid] = li;
      }
      if (!in_loop) loop_inv_vals.insert(rid);
      return;
    }

    if (is_scalar_like(rty)) {
      if (!in_loop) loop_inv_vals.insert(rid);
      if (is_ptr_type(rty)) {
        if (op.kind == OpKind::PtrAdd && op.operands.size() >= 2 && lin_ptrs.count(op.operands[0]) &&
            lin_ints.count(op.operands[1])) {
          const LinearPtr& P = lin_ptrs[op.operands[0]];
          const LinearInt& O = lin_ints[op.operands[1]];
          if (P.valid && O.valid && P.elem_stride == 0 && O.elem_stride == 0 && O.dyn_stride_expr.empty() &&
              P.dyn_stride_expr.empty()) {
            LinearPtr res = P;
            res.iv_coeff = P.iv_coeff + O.iv_coeff;
            res.offset_dtype = P.inv_is_zero ? O.dtype : common_cmp_dtype(P.offset_dtype, O.dtype);
            std::string po = P.inv_is_zero ? "" : msl_cast_expr(P.inv_offset, P.offset_dtype, res.offset_dtype, false);
            std::string oo = O.inv_is_zero ? "" : msl_cast_expr(O.inv_expr, O.dtype, res.offset_dtype, false);
            res.inv_is_zero = P.inv_is_zero && O.inv_is_zero;
            res.inv_offset = combine_add_expr(po, P.inv_is_zero, oo, O.inv_is_zero, res.offset_dtype);
            res.inv_div4 = P.inv_div4 && O.inv_div4;
            lin_ptrs[rid] = res;
            return;
          }
        }
        if ((op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) &&
            !op.operands.empty() && lin_ptrs.count(op.operands[0])) {
          lin_ptrs[rid] = lin_ptrs[op.operands[0]];
          return;
        }
        LinearPtr lp;
        lp.valid = true;
        lp.base_ptr = scalar_exprs.count(rid) ? scalar_exprs[rid] : ssa_var(rid);
        lp.inv_offset = "0";
        lp.inv_is_zero = true;
        lp.inv_div4 = false;
        lin_ptrs[rid] = lp;
      } else if (is_int_dtype(scalar_dtype_of(rty))) {
        if (!in_loop) {
          LinearInt li;
          li.valid = true;
          li.dtype = scalar_dtype_of(rty);
          li.inv_expr = scalar_exprs.count(rid) ? scalar_exprs[rid] : ssa_var(rid);
          auto cit = analysis.const_i64.find(rid);
          li.inv_is_zero = (cit != analysis.const_i64.end() && cit->second == 0);
          li.inv_div4 = analysis.div4_values.count(rid) > 0;
          lin_ints[rid] = li;
        } else {
          track_linear_tile_int(op);
        }
      }
      return;
    }

    if (op.kind == OpKind::Arange) {
      const auto& attr = std::get<ArangeAttr>(op.attr);
      ScalarDType rdt = scalar_dtype_of(rty);
      LinearInt li;
      li.valid = true;
      li.elem_stride = 1;
      li.iv_coeff = 0;
      li.dtype = rdt;
      li.inv_expr = format_scalar_literal_from_i64(attr.start, rdt);
      li.inv_is_zero = (attr.start == 0);
      li.inv_div4 = (attr.start % 4 == 0);
      lin_ints[rid] = li;
      return;
    }

    if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) {
      ValueId src = op.operands[0];
      if (lin_ints.count(src)) lin_ints[rid] = lin_ints[src];
      if (lin_ptrs.count(src)) lin_ptrs[rid] = lin_ptrs[src];
      return;
    }

    if (op.kind == OpKind::PtrAdd) {
      ValueId pid = op.operands[0];
      ValueId offid = op.operands[1];
      if (lin_ptrs.count(pid) && lin_ints.count(offid)) {
        const LinearPtr& P = lin_ptrs[pid];
        const LinearInt& O = lin_ints[offid];
        if (P.valid && O.valid && (P.elem_stride + O.elem_stride <= 1)) {
          if (!P.dyn_stride_expr.empty() && !O.dyn_stride_expr.empty() && P.dyn_stride_expr != O.dyn_stride_expr) {
            return;
          }
          LinearPtr res = P;
          res.elem_stride = P.elem_stride + O.elem_stride;
          res.dyn_stride_expr = P.dyn_stride_expr.empty() ? O.dyn_stride_expr : P.dyn_stride_expr;
          res.iv_coeff = P.iv_coeff + O.iv_coeff;
          res.offset_dtype = P.inv_is_zero ? O.dtype : common_cmp_dtype(P.offset_dtype, O.dtype);
          std::string po = P.inv_is_zero ? "" : msl_cast_expr(P.inv_offset, P.offset_dtype, res.offset_dtype, false);
          std::string oo = O.inv_is_zero ? "" : msl_cast_expr(O.inv_expr, O.dtype, res.offset_dtype, false);
          res.inv_is_zero = P.inv_is_zero && O.inv_is_zero;
          res.inv_offset = combine_add_expr(po, P.inv_is_zero, oo, O.inv_is_zero, res.offset_dtype);
          res.inv_div4 = P.inv_div4 && O.inv_div4;
          lin_ptrs[rid] = res;
        }
      }
      return;
    }

    track_linear_tile_int(op);
  }

  void track_linear_tile_int(const Op& op) {
    if (op.results.empty()) return;
    ValueId rid = op.results[0];
    ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
    if (!is_int_dtype(rdt)) return;

    if (op.kind == OpKind::Cast && !op.operands.empty()) {
      ValueId src = op.operands[0];
      if (lin_ints.count(src) && lin_ints[src].valid) {
        LinearInt li = lin_ints[src];
        if (!li.inv_is_zero) {
          li.inv_expr = msl_cast_expr(li.inv_expr, li.dtype, rdt, false);
        } else {
          li.inv_expr = format_scalar_literal_from_i64(0, rdt);
        }
        li.dtype = rdt;
        lin_ints[rid] = li;
      }
      return;
    }

    if (op.kind == OpKind::Binary && op.operands.size() >= 2 && std::holds_alternative<BinaryAttr>(op.attr)) {
      BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
      auto it_l = lin_ints.find(op.operands[0]);
      auto it_r = lin_ints.find(op.operands[1]);
      if (it_l == lin_ints.end() || it_r == lin_ints.end() || !it_l->second.valid || !it_r->second.valid) return;
      const LinearInt& L = it_l->second;
      const LinearInt& R = it_r->second;

      if (bop == BinaryOp::Add && (L.elem_stride + R.elem_stride <= 1)) {
        if (!L.dyn_stride_expr.empty() && !R.dyn_stride_expr.empty() && L.dyn_stride_expr != R.dyn_stride_expr) return;
        LinearInt res;
        res.valid = true;
        res.elem_stride = L.elem_stride + R.elem_stride;
        res.dyn_stride_expr = L.dyn_stride_expr.empty() ? R.dyn_stride_expr : L.dyn_stride_expr;
        res.iv_coeff = L.iv_coeff + R.iv_coeff;
        res.dtype = rdt;
        std::string lb = L.inv_is_zero ? "" : msl_cast_expr(L.inv_expr, L.dtype, rdt, false);
        std::string rb = R.inv_is_zero ? "" : msl_cast_expr(R.inv_expr, R.dtype, rdt, false);
        res.inv_is_zero = L.inv_is_zero && R.inv_is_zero;
        res.inv_expr = combine_add_expr(lb, L.inv_is_zero, rb, R.inv_is_zero, rdt);
        res.inv_div4 = L.inv_div4 && R.inv_div4;
        lin_ints[rid] = res;
      } else if (bop == BinaryOp::Sub && R.elem_stride == 0 && L.elem_stride <= 1 && R.dyn_stride_expr.empty()) {
        LinearInt res;
        res.valid = true;
        res.elem_stride = L.elem_stride;
        res.dyn_stride_expr = L.dyn_stride_expr;
        res.iv_coeff = L.iv_coeff - R.iv_coeff;
        res.dtype = rdt;
        std::string lb = L.inv_is_zero ? "" : msl_cast_expr(L.inv_expr, L.dtype, rdt, false);
        std::string rb = R.inv_is_zero ? "" : msl_cast_expr(R.inv_expr, R.dtype, rdt, false);
        res.inv_is_zero = L.inv_is_zero && R.inv_is_zero;
        res.inv_expr = combine_sub_expr(lb, L.inv_is_zero, rb, R.inv_is_zero, rdt);
        res.inv_div4 = L.inv_div4 && R.inv_div4;
        lin_ints[rid] = res;
      } else if (bop == BinaryOp::Mul) {
        auto try_scale = [&](const LinearInt& ramp, const LinearInt& sc, ValueId sc_vid) -> bool {
          if (sc.elem_stride != 0 || sc.iv_coeff != 0 || !sc.dyn_stride_expr.empty()) return false;
          auto cit = analysis.const_i64.find(sc_vid);
          if (cit != analysis.const_i64.end()) {
            int64_t k = cit->second;
            if (k == 1) {
              LinearInt res = ramp;
              if (!res.inv_is_zero) res.inv_expr = msl_cast_expr(res.inv_expr, res.dtype, rdt, false);
              res.dtype = rdt;
              lin_ints[rid] = res;
              return true;
            }
            if (ramp.elem_stride == 0) {
              LinearInt res = ramp;
              res.iv_coeff = ramp.iv_coeff * k;
              res.dtype = rdt;
              if (!ramp.inv_is_zero) {
                res.inv_expr = "(" + msl_cast_expr(ramp.inv_expr, ramp.dtype, rdt, false) + " * " +
                               format_scalar_literal_from_i64(k, rdt) + ")";
              }
              res.inv_div4 = ramp.inv_div4 || (k % 4 == 0);
              lin_ints[rid] = res;
              return true;
            }
            return false;
          }
          if (ramp.elem_stride == 1 && ramp.dyn_stride_expr.empty()) {
            LinearInt res = ramp;
            res.dtype = rdt;
            res.dyn_stride_expr = msl_cast_expr(sc.inv_expr, sc.dtype, ScalarDType::I64, false);
            if (!res.inv_is_zero) {
              res.inv_expr = msl_cast_expr(res.inv_expr, ramp.dtype, rdt, false);
            }
            lin_ints[rid] = res;
            return true;
          }
          return false;
        };
        if (!try_scale(L, R, op.operands[1])) {
          try_scale(R, L, op.operands[0]);
        }
      }
    }
  }

  std::string eval_scalar_at_lane(ValueId vid, const std::string& lane_expr, const std::string& c_var,
                                  const std::string& e_var, std::unordered_map<ValueId, std::string>& local_memo,
                                  const std::unordered_map<ValueId, std::string>* extra_scalars,
                                  const std::unordered_set<ValueId>* skip_mat = nullptr) {
    if (vid == kInvalidId) return "0";
    if (auto mit = local_memo.find(vid); mit != local_memo.end()) return mit->second;
    if (extra_scalars) {
      if (auto eit = extra_scalars->find(vid); eit != extra_scalars->end()) return eit->second;
    }
    const Value& v = kernel.value(vid);
    if (is_scalar_like(v.type)) {
      if (auto sit = scalar_exprs.find(vid); sit != scalar_exprs.end()) return sit->second;
      if (v.def_op != kInvalidId && kernel.op(v.def_op).kind == OpKind::Constant) {
        return msl_const_literal(std::get<ConstAttr>(kernel.op(v.def_op).attr), scalar_dtype_of(v.type));
      }
      return ssa_var(vid);
    }
    if (auto pit = ptr_iter_map.find(vid); pit != ptr_iter_map.end()) {
      std::string init_p =
          eval_scalar_at_lane(pit->second.first, lane_expr, c_var, e_var, local_memo, extra_scalars, skip_mat);
      std::string e = "(" + init_p + " + " + pit->second.second + ")";
      local_memo[vid] = e;
      return e;
    }
    if (skip_mat == nullptr || skip_mat->count(vid) == 0) {
      if (auto mt_it = mat_tiles.find(vid); mt_it != mat_tiles.end()) {
        std::string e = chunk_elem(mt_it->second, c_var) + "[" + e_var + "]";
        local_memo[vid] = e;
        return e;
      }
    }
    if (v.def_op == kInvalidId) return ssa_var(vid);
    const Op& op = kernel.op(v.def_op);
    ScalarDType rdt = scalar_dtype_of(v.type);
    std::string res;
    switch (op.kind) {
      case OpKind::Arange: {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        res = (attr.start == 0)
                  ? (msl_scalar_type(rdt) + "(" + lane_expr + ")")
                  : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) + "(" +
                     lane_expr + "))");
        break;
      }
      case OpKind::Splat:
      case OpKind::Broadcast:
      case OpKind::ExpandDims:
      case OpKind::Reshape:
        res = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        break;
      case OpKind::Cast: {
        std::string se = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        res = msl_cast_expr(se, scalar_dtype_of(kernel.value(op.operands[0]).type), rdt, false);
        break;
      }
      case OpKind::Unary: {
        std::string se = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        res = msl_unary_expr(std::get<UnaryAttr>(op.attr).op, se, rdt, false, helpers);
        break;
      }
      case OpKind::Binary: {
        std::string le = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        std::string re = eval_scalar_at_lane(op.operands[1], lane_expr, c_var, e_var, local_memo, extra_scalars);
        res = msl_binary_expr(std::get<BinaryAttr>(op.attr).op, le, re, scalar_dtype_of(kernel.value(op.operands[0]).type),
                              scalar_dtype_of(kernel.value(op.operands[1]).type), rdt, false);
        break;
      }
      case OpKind::Select: {
        std::string ce = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        std::string te = eval_scalar_at_lane(op.operands[1], lane_expr, c_var, e_var, local_memo, extra_scalars);
        std::string fe = eval_scalar_at_lane(op.operands[2], lane_expr, c_var, e_var, local_memo, extra_scalars);
        res = msl_select_expr(ce, te, fe, scalar_dtype_of(kernel.value(op.operands[1]).type),
                              scalar_dtype_of(kernel.value(op.operands[2]).type), rdt, false);
        break;
      }
      case OpKind::PtrAdd: {
        std::string pe = eval_scalar_at_lane(op.operands[0], lane_expr, c_var, e_var, local_memo, extra_scalars);
        std::string oe = eval_scalar_at_lane(op.operands[1], lane_expr, c_var, e_var, local_memo, extra_scalars);
        res = "(" + pe + " + " + oe + ")";
        break;
      }
      case OpKind::Load: {
        auto lo = extract_load_operands(op, kernel);
        std::string pe = eval_scalar_at_lane(lo.ptrs, lane_expr, c_var, e_var, local_memo, extra_scalars);
        if (lo.mask == kInvalidId) {
          res = "(*(" + pe + "))";
        } else {
          std::string me = eval_scalar_at_lane(lo.mask, lane_expr, c_var, e_var, local_memo, extra_scalars);
          std::string oe =
              (lo.other != kInvalidId)
                  ? msl_cast_expr(eval_scalar_at_lane(lo.other, lane_expr, c_var, e_var, local_memo, extra_scalars),
                                  scalar_dtype_of(kernel.value(lo.other).type), rdt, false)
                  : msl_zero_literal(rdt, false);
          res = "(" + me + " ? (*(" + pe + ")) : " + oe + ")";
        }
        break;
      }
      default:
        err = "Unsupported op in scalar tail evaluation";
        return "0";
    }
    local_memo[vid] = res;
    return res;
  }

  struct MemOpPlan {
    LinearPtr lp;
    DecomposedMask dmask;
    std::string hoisted_ptr;
    bool is_store = false;
  };

  bool build_mem_plans(const std::vector<OpId>& ops, std::unordered_map<OpId, MemOpPlan>& plans) {
    for (OpId oid : ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if (op.kind == OpKind::Load && !op.results.empty() && !is_scalar_like(kernel.value(op.results[0]).type)) {
        auto lo = extract_load_operands(op, kernel);
        ScalarDType dt = scalar_dtype_of(kernel.value(op.results[0]).type);
        if (!can_vec4_mem(dt)) return false;
        auto pit = lin_ptrs.find(lo.ptrs);
        if (pit == lin_ptrs.end() || !pit->second.valid || pit->second.elem_stride != 1) return false;
        DecomposedMask dm;
        if (!decompose_mask_rec(kernel, lin_ints, scalar_exprs, loop_inv_vals, lo.mask, dm)) return false;
        plans[oid] = MemOpPlan{pit->second, std::move(dm), "", false};
      } else if (op.kind == OpKind::Store && op.operands.size() >= 2 &&
                 !is_scalar_like(kernel.value(op.operands[1]).type)) {
        auto so = extract_store_operands(op);
        ScalarDType dt = scalar_dtype_of(kernel.value(so.val).type);
        if (!can_vec4_mem(dt)) return false;
        auto pit = lin_ptrs.find(so.ptrs);
        if (pit == lin_ptrs.end() || !pit->second.valid || pit->second.elem_stride != 1) return false;
        DecomposedMask dm;
        if (!decompose_mask_rec(kernel, lin_ints, scalar_exprs, loop_inv_vals, so.mask, dm)) return false;
        plans[oid] = MemOpPlan{pit->second, std::move(dm), "", true};
      }
    }
    return true;
  }

  std::string build_ptr_addr_expr(const MemOpPlan& plan, const std::string& iv_var, const std::string& elem_base_var) {
    std::string ptr = plan.hoisted_ptr.empty() ? plan.lp.base_ptr : plan.hoisted_ptr;
    std::vector<std::string> terms;
    if (plan.hoisted_ptr.empty() && !plan.lp.inv_is_zero) {
      terms.push_back("long(" + plan.lp.inv_offset + ")");
    }
    if (!plan.lp.poff_var.empty()) {
      terms.push_back(plan.lp.poff_var);
    }
    if (plan.lp.iv_coeff != 0 && !iv_var.empty()) {
      if (plan.lp.iv_coeff == 1) {
        terms.push_back("long(" + iv_var + ")");
      } else {
        terms.push_back("(long(" + iv_var + ") * " + std::to_string(plan.lp.iv_coeff) + "ll)");
      }
    }
    terms.push_back("long(" + elem_base_var + ")");
    std::string off = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
      off = "(" + off + " + " + terms[i] + ")";
    }
    return "(" + ptr + " + " + off + ")";
  }

  std::string build_ramp_all4_guard(const RampBound& rb, const std::string& iv_var, const std::string& elem_base_var) {
    std::vector<std::string> terms;
    if (!rb.ramp.inv_is_zero) terms.push_back("long(" + rb.ramp.inv_expr + ")");
    if (rb.ramp.iv_coeff != 0 && !iv_var.empty()) {
      if (rb.ramp.iv_coeff == 1) {
        terms.push_back("long(" + iv_var + ")");
      } else {
        terms.push_back("(long(" + iv_var + ") * " + std::to_string(rb.ramp.iv_coeff) + "ll)");
      }
    }
    terms.push_back("long(" + elem_base_var + ")");
    std::string base = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) base = "(" + base + " + " + terms[i] + ")";
    std::string bnd = "long(" + rb.bound_expr + ")";
    if (rb.is_upper) {
      return "(" + base + " + 3ll" + (rb.strict ? " < " : " <= ") + bnd + ")";
    }
    return "(" + base + (rb.strict ? " > " : " >= ") + bnd + ")";
  }

  void emit_vec4_ops_body(const std::vector<OpId>& ops, const std::unordered_set<ValueId>& needed_fast,
                          const std::unordered_map<OpId, MemOpPlan>& plans,
                          const std::unordered_map<ValueId, MatTile>& extra_mat,
                          const std::vector<std::pair<ValueId, MatTile>>* tile_yields, const std::string& iv_var,
                          const std::string& c_var, const std::string& elem_base_var, bool allow_inv_mask_ternary,
                          int ind, std::ostringstream& out) {
    std::unordered_map<ValueId, std::string> v4_exprs = scalar_exprs;
    std::unordered_set<ValueId> is_v4;

    for (const auto& [vid, mt] : mat_tiles) {
      v4_exprs[vid] = chunk_elem(mt, c_var);
      is_v4.insert(vid);
    }
    for (const auto& [vid, mt] : extra_mat) {
      v4_exprs[vid] = chunk_elem(mt, c_var);
      is_v4.insert(vid);
    }

    auto get_v4 = [&](ValueId id, ScalarDType expected_dt) -> std::string {
      std::string e = v4_exprs.at(id);
      bool already = is_v4.count(id) > 0;
      ScalarDType cur_dt = scalar_dtype_of(kernel.value(id).type);
      if (!already) {
        e = msl_cast_expr(e, cur_dt, expected_dt, false);
        return promote_to_vec4(e, expected_dt, false);
      }
      return msl_cast_expr(e, cur_dt, expected_dt, true);
    };

    for (OpId oid : ops) {
      const Op& op = kernel.op(oid);
      if (op.erased || op.kind == OpKind::Return || op.kind == OpKind::Yield) continue;

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        if (is_scalar_like(kernel.value(so.ptrs).type)) continue;
        const auto& plan = plans.at(oid);
        ScalarDType dt = scalar_dtype_of(kernel.value(so.val).type);
        std::string addr = build_ptr_addr_expr(plan, iv_var, elem_base_var);
        std::string ve = get_v4(so.val, dt);
        if (allow_inv_mask_ternary && !plan.dmask.inv_scalar_guards.empty()) {
          std::string g = plan.dmask.inv_scalar_guards[0];
          for (size_t i = 1; i < plan.dmask.inv_scalar_guards.size(); ++i) {
            g = "(" + g + " && " + plan.dmask.inv_scalar_guards[i] + ")";
          }
          out << indent_str(ind) << "if (" << g << ") { *(device " << msl_vec4_type(dt) << "*)" << addr << " = " << ve
              << "; }\n";
        } else {
          out << indent_str(ind) << "*(device " << msl_vec4_type(dt) << "*)" << addr << " = " << ve << ";\n";
        }
        continue;
      }

      if (op.results.empty() || !needed_fast.count(op.results[0])) continue;
      ValueId rid = op.results[0];
      if (is_scalar_like(kernel.value(rid).type)) continue;
      ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
      std::string vty = msl_vec4_type(rdt);

      bool is_predeclared = mat_tiles.count(rid) > 0;
      std::string lhs_var = is_predeclared ? chunk_elem(mat_tiles.at(rid), c_var) : ssa_var(rid);
      std::string decl_pfx = is_predeclared ? "" : (vty + " ");

      std::string rhs;
      if (op.kind == OpKind::Load) {
        auto lo = extract_load_operands(op, kernel);
        const auto& plan = plans.at(oid);
        std::string addr = build_ptr_addr_expr(plan, iv_var, elem_base_var);
        std::string load_e = "*(const device " + vty + "*)" + addr;
        if (allow_inv_mask_ternary && !plan.dmask.inv_scalar_guards.empty()) {
          std::string g = plan.dmask.inv_scalar_guards[0];
          for (size_t i = 1; i < plan.dmask.inv_scalar_guards.size(); ++i) {
            g = "(" + g + " && " + plan.dmask.inv_scalar_guards[i] + ")";
          }
          std::string other_e = msl_zero_literal(rdt, true);
          if (lo.other != kInvalidId) {
            if (auto sc = unwrap_scalar_splat(kernel, lo.other)) {
              ScalarDType odt = scalar_dtype_of(kernel.value(*sc).type);
              std::string se = scalar_exprs.count(*sc) ? scalar_exprs.at(*sc) : ssa_var(*sc);
              other_e = vty + "(" + msl_cast_expr(se, odt, rdt, false) + ")";
            } else if (v4_exprs.count(lo.other)) {
              other_e = get_v4(lo.other, rdt);
            }
          }
          rhs = "(" + g + " ? " + load_e + " : " + other_e + ")";
        } else {
          rhs = load_e;
        }
      } else if (op.kind == OpKind::Arange) {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        std::string base = (attr.start == 0)
                               ? (msl_scalar_type(rdt) + "(" + elem_base_var + ")")
                               : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) +
                                  "(" + elem_base_var + "))");
        rhs = "(" + vty + "(" + base + ") + " + vty + "(" + format_scalar_literal_from_i64(0, rdt) + ", " +
              format_scalar_literal_from_i64(1, rdt) + ", " + format_scalar_literal_from_i64(2, rdt) + ", " +
              format_scalar_literal_from_i64(3, rdt) + "))";
      } else if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) {
        rhs = get_v4(op.operands[0], rdt);
      } else if (op.kind == OpKind::Cast) {
        ScalarDType sdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        rhs = msl_cast_expr(get_v4(op.operands[0], sdt), sdt, rdt, true);
      } else if (op.kind == OpKind::Unary) {
        UnaryOp uop = std::get<UnaryAttr>(op.attr).op;
        ScalarDType sdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        rhs = msl_unary_expr(uop, get_v4(op.operands[0], sdt), rdt, true, helpers);
      } else if (op.kind == OpKind::Binary) {
        BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
        ScalarDType ldt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        ScalarDType rdt_in = scalar_dtype_of(kernel.value(op.operands[1]).type);
        rhs = msl_binary_expr(bop, get_v4(op.operands[0], ldt), get_v4(op.operands[1], rdt_in), ldt, rdt_in, rdt, true);
      } else if (op.kind == OpKind::Select) {
        ScalarDType tdt = scalar_dtype_of(kernel.value(op.operands[1]).type);
        ScalarDType fdt = scalar_dtype_of(kernel.value(op.operands[2]).type);
        rhs = msl_select_expr(get_v4(op.operands[0], ScalarDType::Bool), get_v4(op.operands[1], tdt),
                              get_v4(op.operands[2], fdt), tdt, fdt, rdt, true);
      } else {
        err = "Unsupported tile op in vec4 fast path";
        return;
      }

      out << indent_str(ind) << decl_pfx << lhs_var << " = " << rhs << ";\n";
      v4_exprs[rid] = lhs_var;
      is_v4.insert(rid);
    }

    if (tile_yields != nullptr) {
      std::vector<std::string> tmps;
      tmps.reserve(tile_yields->size());
      for (const auto& [yval, mt] : *tile_yields) {
        std::string tmp = next_tmp("m1_yv");
        out << indent_str(ind) << msl_vec4_type(mt.dtype) << " " << tmp << " = " << get_v4(yval, mt.dtype) << ";\n";
        tmps.push_back(tmp);
      }
      for (size_t i = 0; i < tile_yields->size(); ++i) {
        out << indent_str(ind) << chunk_elem((*tile_yields)[i].second, c_var) << " = " << tmps[i] << ";\n";
      }
    }
  }

  void emit_scalar_4lane_tail(const std::vector<OpId>& ops, const std::unordered_set<ValueId>& mat_outputs,
                              const std::vector<std::pair<ValueId, MatTile>>* tile_yields, const std::string& c_var,
                              const std::string& elem_base_var, int ind, std::ostringstream& out) {
    out << indent_str(ind) << "for (uint m1_e = 0u; m1_e < 4u; ++m1_e) {\n";
    out << indent_str(ind + 1) << "uint m1_lane = " << elem_base_var << " + m1_e;\n";
    std::unordered_map<ValueId, std::string> local_memo;

    for (OpId oid : ops) {
      const Op& op = kernel.op(oid);
      if (op.erased || op.kind == OpKind::Return || op.kind == OpKind::Yield) continue;

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        if (is_scalar_like(kernel.value(so.ptrs).type)) continue;
        ScalarDType pdt = scalar_dtype_of(kernel.value(so.ptrs).type);
        std::string pe = eval_scalar_at_lane(so.ptrs, "m1_lane", c_var, "m1_e", local_memo, nullptr);
        std::string ve = msl_cast_expr(eval_scalar_at_lane(so.val, "m1_lane", c_var, "m1_e", local_memo, nullptr),
                                       scalar_dtype_of(kernel.value(so.val).type), pdt, false);
        if (so.mask != kInvalidId) {
          std::string me = eval_scalar_at_lane(so.mask, "m1_lane", c_var, "m1_e", local_memo, nullptr);
          out << indent_str(ind + 1) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
        } else {
          out << indent_str(ind + 1) << "*(" << pe << ") = " << ve << ";\n";
        }
        continue;
      }

      if (op.results.empty()) continue;
      ValueId rid = op.results[0];
      if (is_scalar_like(kernel.value(rid).type)) continue;
      if (mat_outputs.count(rid) > 0) {
        std::string e = eval_scalar_at_lane(rid, "m1_lane", c_var, "m1_e", local_memo, nullptr, &mat_outputs);
        out << indent_str(ind + 1) << chunk_elem(mat_tiles.at(rid), c_var) << "[m1_e] = " << e << ";\n";
        local_memo[rid] = chunk_elem(mat_tiles.at(rid), c_var) + "[m1_e]";
      }
    }

    if (tile_yields != nullptr) {
      std::vector<std::string> tmps;
      tmps.reserve(tile_yields->size());
      for (const auto& [yval, mt] : *tile_yields) {
        std::string ev = msl_cast_expr(eval_scalar_at_lane(yval, "m1_lane", c_var, "m1_e", local_memo, nullptr),
                                       scalar_dtype_of(kernel.value(yval).type), mt.dtype, false);
        std::string tmp = next_tmp("m1_ys");
        out << indent_str(ind + 1) << msl_scalar_type(mt.dtype) << " " << tmp << " = " << ev << ";\n";
        tmps.push_back(tmp);
      }
      for (size_t i = 0; i < tile_yields->size(); ++i) {
        out << indent_str(ind + 1) << chunk_elem((*tile_yields)[i].second, c_var) << "[m1_e] = " << tmps[i] << ";\n";
      }
    }
    out << indent_str(ind) << "}\n";
  }

  void emit_reduce_op(const Op& op, bool emit_trailing_barrier, int ind, std::ostringstream& out) {
    const auto& ra = std::get<ReduceAttr>(op.attr);
    ValueId in_v = op.operands[0];
    ScalarDType dt = scalar_dtype_of(kernel.value(in_v).type);
    const std::string ty = msl_scalar_type(dt);
    bool is_arg = (ra.kind == ReduceKind::Argmax || ra.kind == ReduceKind::Argmin ||
                   ra.result_mode == ReduceResultMode::ValueAndIndex || ra.result_mode == ReduceResultMode::IndexOnly);
    bool is_max = (ra.kind == ReduceKind::Max || ra.kind == ReduceKind::Argmax);
    bool broadcast_all = analysis.broadcast_reduces.count(op.id) > 0;

    std::string loc_val = next_tmp("m1_rv");
    std::string loc_idx = is_arg ? next_tmp("m1_ri") : "";

    auto emit_chunk_reduce = [&](const std::string& v4_expr, const std::string& elem_base_expr, int cind) {
      if (!is_arg) {
        std::string c4;
        if (ra.kind == ReduceKind::Sum) {
          c4 = "((" + v4_expr + ".x + " + v4_expr + ".y) + (" + v4_expr + ".z + " + v4_expr + ".w))";
        } else if (ra.kind == ReduceKind::Max) {
          c4 = "metal::max(metal::max(" + v4_expr + ".x, " + v4_expr + ".y), metal::max(" + v4_expr + ".z, " + v4_expr +
               ".w))";
        } else {
          c4 = "metal::min(metal::min(" + v4_expr + ".x, " + v4_expr + ".y), metal::min(" + v4_expr + ".z, " + v4_expr +
               ".w))";
        }
        out << indent_str(cind) << loc_val << " = " << scalar_combine_expr(ra.kind, loc_val, c4) << ";\n";
      } else {
        const std::string cmp = is_max ? " > " : " < ";
        out << indent_str(cind) << "for (uint m1_e = 0u; m1_e < 4u; ++m1_e) {\n";
        out << indent_str(cind + 1) << ty << " m1_lv = " << v4_expr << "[m1_e];\n";
        out << indent_str(cind + 1) << "int m1_li = int(" << elem_base_expr << " + m1_e);\n";
        out << indent_str(cind + 1) << "bool m1_bt = (m1_lv" << cmp << loc_val << ") || ((m1_lv == " << loc_val
            << ") && (m1_li < " << loc_idx << "));\n";
        out << indent_str(cind + 1) << loc_val << " = m1_bt ? m1_lv : " << loc_val << ";\n";
        out << indent_str(cind + 1) << loc_idx << " = m1_bt ? m1_li : " << loc_idx << ";\n";
        out << indent_str(cind) << "}\n";
      }
    };

    out << indent_str(ind) << ty << " " << loc_val << " = " << msl_reduce_identity(ra.kind, dt) << ";\n";
    if (is_arg) {
      out << indent_str(ind) << "int " << loc_idx << " = 2147483647;\n";
    }

    auto mt_it = mat_tiles.find(in_v);
    if (mt_it != mat_tiles.end() && mt_it->second.collapsed) {
      emit_chunk_reduce(mt_it->second.var_name, "(m1_tid_linear * 4u)", ind);
    } else if (mt_it != mat_tiles.end()) {
      if (chunks == 1) {
        if (!exact_cover) {
          out << indent_str(ind) << "if ((m1_tid_linear * 4u) < " << block_size << "u) {\n";
          emit_chunk_reduce(mt_it->second.var_name, "(m1_tid_linear * 4u)", ind + 1);
          out << indent_str(ind) << "}\n";
        } else {
          emit_chunk_reduce(mt_it->second.var_name, "(m1_tid_linear * 4u)", ind);
        }
      } else {
        if (chunks <= 4) out << indent_str(ind) << "#pragma unroll\n";
        out << indent_str(ind) << "for (uint m1_c = 0u; m1_c < " << chunks << "u; ++m1_c) {\n";
        out << indent_str(ind + 1) << "uint m1_eb = (m1_tid_linear + m1_c * " << tg_x << "u) * 4u;\n";
        if (!exact_cover) {
          out << indent_str(ind + 1) << "if (m1_eb < " << block_size << "u) {\n";
          emit_chunk_reduce(mt_it->second.var_name + "[m1_c]", "m1_eb", ind + 2);
          out << indent_str(ind + 1) << "}\n";
        } else {
          emit_chunk_reduce(mt_it->second.var_name + "[m1_c]", "m1_eb", ind + 1);
        }
        out << indent_str(ind) << "}\n";
      }
    } else {
      out << indent_str(ind) << "for (uint m1_lane = m1_tid_linear; m1_lane < " << block_size << "u; m1_lane += " << tg_x
          << "u) {\n";
      std::unordered_map<ValueId, std::string> lm;
      std::string ev = eval_scalar_at_lane(in_v, "m1_lane", "0u", "0u", lm, nullptr);
      if (!is_arg) {
        out << indent_str(ind + 1) << loc_val << " = " << scalar_combine_expr(ra.kind, loc_val, ev) << ";\n";
      } else {
        const std::string cmp = is_max ? " > " : " < ";
        out << indent_str(ind + 1) << ty << " m1_lv = " << ev << ";\n";
        out << indent_str(ind + 1) << "int m1_li = int(m1_lane);\n";
        out << indent_str(ind + 1) << "bool m1_bt = (m1_lv" << cmp << loc_val << ") || ((m1_lv == " << loc_val
            << ") && (m1_li < " << loc_idx << "));\n";
        out << indent_str(ind + 1) << loc_val << " = m1_bt ? m1_lv : " << loc_val << ";\n";
        out << indent_str(ind + 1) << loc_idx << " = m1_bt ? m1_li : " << loc_idx << ";\n";
      }
      out << indent_str(ind) << "}\n";
    }

    if (!is_arg) {
      ValueId rid = op.results[0];
      std::string rv = ssa_var(rid);
      emit_cross_simd_value_reduce(ra.kind, dt, loc_val, rv, num_sg, broadcast_all, emit_trailing_barrier, smem, ind,
                                   out);
      scalar_exprs[rid] = rv;
    } else if (ra.result_mode == ReduceResultMode::ValueAndIndex && op.results.size() >= 2) {
      std::string rv = ssa_var(op.results[0]);
      std::string ri = ssa_var(op.results[1]);
      emit_cross_simd_arg_reduce(is_max, dt, loc_val, loc_idx, rv, ri, num_sg, broadcast_all, emit_trailing_barrier,
                                 smem, ind, out);
      scalar_exprs[op.results[0]] = rv;
      scalar_exprs[op.results[1]] = ri;
    } else {
      std::string ri = ssa_var(op.results[0]);
      emit_cross_simd_arg_reduce(is_max, dt, loc_val, loc_idx, "", ri, num_sg, broadcast_all, emit_trailing_barrier,
                                 smem, ind, out);
      scalar_exprs[op.results[0]] = ri;
    }
  }

  bool emit_for_loop_no_reduce(const Op& op, int ind, std::ostringstream& out) {
    if (op.operands.size() < 3 || op.regions.empty()) {
      err = "Malformed ForLoop";
      return false;
    }
    const Block& body = kernel.block(op.regions[0]);
    if (body.args.empty()) {
      err = "ForLoop missing iv";
      return false;
    }

    ValueId iv_id = body.args[0];
    ScalarDType iv_dt = scalar_dtype_of(kernel.value(iv_id).type);
    std::string iv_var = ssa_var(iv_id);
    std::string lo_e =
        msl_cast_expr(scalar_exprs.at(op.operands[0]), scalar_dtype_of(kernel.value(op.operands[0]).type), iv_dt, false);
    std::string hi_e =
        msl_cast_expr(scalar_exprs.at(op.operands[1]), scalar_dtype_of(kernel.value(op.operands[1]).type), iv_dt, false);
    std::string step_e =
        msl_cast_expr(scalar_exprs.at(op.operands[2]), scalar_dtype_of(kernel.value(op.operands[2]).type), iv_dt, false);

    auto step_c = analysis.const_i64.find(op.operands[2]);
    bool has_pos_const_step = (step_c != analysis.const_i64.end() && step_c->second > 0);
    bool neg_step = (step_c != analysis.const_i64.end() && step_c->second < 0);

    const Op* yield_op = nullptr;
    for (OpId boid : body.ops) {
      const Op& bop = kernel.op(boid);
      if (!bop.erased && bop.kind == OpKind::Yield) {
        yield_op = &bop;
        break;
      }
    }
    if (!yield_op) {
      err = "ForLoop missing Yield";
      return false;
    }

    size_t num_iter = op.operands.size() - 3;
    std::vector<std::pair<ValueId, MatTile>> tile_yields;
    std::vector<std::tuple<std::string, ValueId, ScalarDType>> scalar_yields;
    std::vector<std::pair<std::string, ValueId>> ptr_yields;
    std::unordered_map<ValueId, MatTile> loop_mats;

    for (size_t i = 0; i < num_iter; ++i) {
      ValueId arg_id = body.args[1 + i];
      ValueId init_id = op.operands[3 + i];
      const Type& aty = kernel.value(arg_id).type;
      std::string avar = ssa_var(arg_id);

      if (is_scalar_like(aty)) {
        ScalarDType adt = scalar_dtype_of(aty);
        if (is_ptr_type(aty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, arg_id, adt) << " " << avar << " = "
              << scalar_exprs.at(init_id) << ";\n";
        } else {
          out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
              << msl_cast_expr(scalar_exprs.at(init_id), scalar_dtype_of(kernel.value(init_id).type), adt, false)
              << ";\n";
        }
        scalar_exprs[arg_id] = avar;
        if (i < op.results.size()) scalar_exprs[op.results[i]] = avar;
        scalar_yields.emplace_back(avar, yield_op->operands[i], adt);
      } else if (is_ptr_type(aty)) {
        ValueId delta_v = kInvalidId;
        if (!detect_ptr_iter_delta(kernel, arg_id, yield_op->operands[i], delta_v)) return false;
        std::string poff = "m1_poff_" + std::to_string(arg_id);
        out << indent_str(ind) << "long " << poff << " = 0ll;\n";
        ptr_iter_map[arg_id] = {init_id, poff};
        if (i < op.results.size()) ptr_iter_map[op.results[i]] = {init_id, poff};
        ptr_yields.emplace_back(poff, delta_v);

        auto init_pit = lin_ptrs.find(init_id);
        if (init_pit != lin_ptrs.end() && init_pit->second.valid) {
          LinearPtr lp = init_pit->second;
          lp.poff_var = poff;
          lp.poff_delta_expr = scalar_exprs.count(delta_v) ? scalar_exprs.at(delta_v) : ssa_var(delta_v);
          lp.poff_delta_div4 = analysis.div4_values.count(delta_v) > 0;
          lin_ptrs[arg_id] = lp;
        }
      } else {
        ScalarDType adt = scalar_dtype_of(aty);
        auto coll = detect_collapsible_acc(kernel, analysis, op, i);
        bool is_coll = coll.has_value();
        if (!is_coll && chunks > 8) return false;

        MatTile mt{avar, adt, is_coll, is_coll ? coll->kind : ReduceKind::Sum};
        if (is_coll) {
          out << indent_str(ind) << msl_vec4_type(adt) << " " << avar << " = " << msl_vec4_type(adt) << "("
              << msl_reduce_identity(coll->kind, adt) << ");\n";
        } else if (chunks == 1) {
          std::string init_e;
          if (auto sc = unwrap_scalar_splat(kernel, init_id)) {
            std::string se = scalar_exprs.count(*sc) ? scalar_exprs.at(*sc) : ssa_var(*sc);
            init_e = msl_vec4_type(adt) + "(" + msl_cast_expr(se, scalar_dtype_of(kernel.value(*sc).type), adt, false) +
                     ")";
          } else if (mat_tiles.count(init_id)) {
            init_e = mat_tiles.at(init_id).var_name;
          } else {
            return false;
          }
          out << indent_str(ind) << msl_vec4_type(adt) << " " << avar << " = " << init_e << ";\n";
        } else {
          out << indent_str(ind) << msl_vec4_type(adt) << " " << avar << "[" << chunks << "];\n";
          if (auto sc = unwrap_scalar_splat(kernel, init_id)) {
            std::string se = scalar_exprs.count(*sc) ? scalar_exprs.at(*sc) : ssa_var(*sc);
            std::string ve =
                msl_vec4_type(adt) + "(" + msl_cast_expr(se, scalar_dtype_of(kernel.value(*sc).type), adt, false) + ")";
            out << indent_str(ind) << "for (uint m1_ci = 0u; m1_ci < " << chunks << "u; ++m1_ci) { " << avar
                << "[m1_ci] = " << ve << "; }\n";
          } else if (mat_tiles.count(init_id) && !mat_tiles.at(init_id).collapsed) {
            out << indent_str(ind) << "for (uint m1_ci = 0u; m1_ci < " << chunks << "u; ++m1_ci) { " << avar
                << "[m1_ci] = " << mat_tiles.at(init_id).var_name << "[m1_ci]; }\n";
          } else {
            return false;
          }
        }
        mat_tiles[arg_id] = mt;
        if (i < op.results.size()) mat_tiles[op.results[i]] = mt;
        loop_mats[arg_id] = mt;
        tile_yields.emplace_back(yield_op->operands[i], mt);
      }
    }

    scalar_exprs[iv_id] = iv_var;
    {
      LinearInt iv_li;
      iv_li.valid = true;
      iv_li.elem_stride = 0;
      iv_li.iv_coeff = 1;
      iv_li.inv_expr = "0";
      iv_li.inv_is_zero = true;
      iv_li.inv_div4 = true;
      iv_li.dtype = iv_dt;
      lin_ints[iv_id] = iv_li;
    }

    std::vector<OpId> body_scalar_ops;
    std::vector<OpId> body_tile_ops;
    for (OpId boid : body.ops) {
      const Op& bop = kernel.op(boid);
      if (bop.erased || bop.kind == OpKind::Yield) continue;
      if (bop.kind == OpKind::ForLoop || bop.kind == OpKind::If || bop.kind == OpKind::Barrier ||
          bop.kind == OpKind::AtomicAdd) {
        return false;
      }
      bool is_scalar_op = (!bop.results.empty() && is_scalar_like(kernel.value(bop.results[0]).type)) ||
                          (bop.kind == OpKind::Store && is_scalar_like(kernel.value(bop.operands[0]).type));
      if (is_scalar_op) {
        if (bop.kind == OpKind::Store) return false;
        if (bop.kind != OpKind::Constant) {
          body_scalar_ops.push_back(boid);
        }
      } else {
        body_tile_ops.push_back(boid);
      }
    }

    for (OpId boid : body.ops) {
      const Op& bop = kernel.op(boid);
      if (bop.erased || bop.kind == OpKind::Yield) continue;
      if (!bop.results.empty() && is_scalar_like(kernel.value(bop.results[0]).type)) {
        if (bop.kind == OpKind::Constant) {
          scalar_exprs[bop.results[0]] =
              msl_const_literal(std::get<ConstAttr>(bop.attr), scalar_dtype_of(kernel.value(bop.results[0]).type));
        } else {
          scalar_exprs[bop.results[0]] = ssa_var(bop.results[0]);
        }
      }
      track_linear_info(bop, true);
    }

    std::unordered_map<OpId, MemOpPlan> plans;
    if (!build_mem_plans(body_tile_ops, plans)) return false;

    std::unordered_set<ValueId> needed_fast;
    std::vector<ValueId> wl;
    auto mark_need = [&](ValueId v) {
      if (v != kInvalidId && !is_scalar_like(kernel.value(v).type) && needed_fast.insert(v).second) wl.push_back(v);
    };
    for (const auto& [yv, _] : tile_yields) mark_need(yv);
    for (OpId boid : body_tile_ops) {
      const Op& bop = kernel.op(boid);
      if (bop.kind == OpKind::Store && bop.operands.size() >= 2) mark_need(bop.operands[1]);
    }
    while (!wl.empty()) {
      ValueId v = wl.back();
      wl.pop_back();
      const Value& val = kernel.value(v);
      if (val.def_op == kInvalidId) continue;
      const Op& dop = kernel.op(val.def_op);
      if (dop.kind == OpKind::Load || dop.parent_block != op.regions[0]) continue;
      for (ValueId o : dop.operands) mark_need(o);
    }

    std::vector<std::string> align_conds;
    std::unordered_set<std::string> seen_conds;
    auto add_align = [&](const std::string& c) {
      if (!c.empty() && seen_conds.insert(c).second) align_conds.push_back(c);
    };

    bool can_peel = has_pos_const_step && !neg_step && body_scalar_ops.empty();
    std::vector<std::string> peel_upper_limits;
    std::unordered_set<std::string> seen_upper_limits;
    std::vector<std::string> peel_extra_conds;
    std::unordered_set<std::string> seen_extra_conds;
    auto add_peel_upper = [&](const std::string& s) {
      if (!s.empty() && seen_upper_limits.insert(s).second) peel_upper_limits.push_back(s);
    };
    auto add_peel_extra = [&](const std::string& s) {
      if (!s.empty() && seen_extra_conds.insert(s).second) peel_extra_conds.push_back(s);
    };

    std::unordered_map<std::string, std::string> hoisted_ptr_cache;
    for (OpId boid : body_tile_ops) {
      auto pit = plans.find(boid);
      if (pit == plans.end()) continue;
      auto& plan = pit->second;
      const Op& bop = kernel.op(boid);
      ScalarDType pdt =
          plan.is_store ? scalar_dtype_of(kernel.value(bop.operands[1]).type) : scalar_dtype_of(kernel.value(bop.results[0]).type);

      if (!plan.lp.inv_is_zero) {
        std::string key = plan.lp.base_ptr + "|" + plan.lp.inv_offset + "|" + std::to_string(plan.is_store);
        auto hit = hoisted_ptr_cache.find(key);
        if (hit != hoisted_ptr_cache.end()) {
          plan.hoisted_ptr = hit->second;
        } else {
          std::string hname = next_tmp("m1_rptr");
          out << indent_str(ind) << "device " << (plan.is_store ? "" : "const ") << msl_scalar_type(pdt) << "* "
              << hname << " = (" << plan.lp.base_ptr << " + long(" << plan.lp.inv_offset << "));\n";
          hoisted_ptr_cache[key] = hname;
          plan.hoisted_ptr = hname;
        }
      }

      if (!plan.lp.dyn_stride_expr.empty()) {
        add_align("(" + plan.lp.dyn_stride_expr + " == 1ll)");
      }
      if (!plan.lp.inv_div4) {
        std::string al = "((long(" + plan.lp.inv_offset + ") & 3ll) == 0ll)";
        if (!plan.dmask.inv_scalar_guards.empty()) {
          std::string g = plan.dmask.inv_scalar_guards[0];
          for (size_t i = 1; i < plan.dmask.inv_scalar_guards.size(); ++i) {
            g = "(" + g + " && " + plan.dmask.inv_scalar_guards[i] + ")";
          }
          al = "(!" + g + " || " + al + ")";
        }
        add_align(al);
      }
      if (plan.lp.iv_coeff != 0) {
        if (!analysis.div4_values.count(op.operands[0])) {
          add_align("((long(" + lo_e + ") & 3ll) == 0ll)");
        }
        if (!analysis.div4_values.count(op.operands[2])) {
          add_align("((long(" + step_e + ") & 3ll) == 0ll)");
        }
      }
      if (!plan.lp.poff_var.empty() && !plan.lp.poff_delta_div4) {
        add_align("((long(" + plan.lp.poff_delta_expr + ") & 3ll) == 0ll)");
      }

      if (!plan.dmask.var_scalar_guards.empty()) {
        can_peel = false;
      }
      for (const auto& ig : plan.dmask.inv_scalar_guards) {
        add_peel_extra("(" + ig + ")");
      }
      for (const auto& rb : plan.dmask.ramp_bounds) {
        if (rb.ramp.iv_coeff != 1) {
          if (rb.ramp.iv_coeff == 0) {
            std::string base = rb.ramp.inv_is_zero ? "0ll" : ("long(" + rb.ramp.inv_expr + ")");
            std::string bnd = "long(" + rb.bound_expr + ")";
            if (rb.is_upper) {
              add_peel_extra("(" + base + " + " + std::to_string(block_size - 1) + "ll" +
                             (rb.strict ? " < " : " <= ") + bnd + ")");
            } else {
              add_peel_extra("(" + base + (rb.strict ? " > " : " >= ") + bnd + ")");
            }
          } else {
            can_peel = false;
          }
        } else {
          std::string inv_sub = rb.ramp.inv_is_zero ? "" : (" - long(" + rb.ramp.inv_expr + ")");
          if (rb.is_upper) {
            int64_t span = rb.strict ? block_size : (block_size - 1);
            add_peel_upper("(long(" + rb.bound_expr + ")" + inv_sub + " - " + std::to_string(span) + "ll)");
          } else {
            std::string base_lo = rb.ramp.inv_is_zero ? ("long(" + lo_e + ")")
                                                      : ("(long(" + rb.ramp.inv_expr + ") + long(" + lo_e + "))");
            add_peel_extra("(" + base_lo + (rb.strict ? " > " : " >= ") + "long(" + rb.bound_expr + "))");
          }
        }
      }
    }

    std::string align_var = next_tmp("m1_vec4_ok");
    if (align_conds.empty()) {
      out << indent_str(ind) << "bool " << align_var << " = true;\n";
    } else {
      out << indent_str(ind) << "bool " << align_var << " = ";
      for (size_t i = 0; i < align_conds.size(); ++i) {
        if (i > 0) out << " && ";
        out << align_conds[i];
      }
      out << ";\n";
    }

    auto emit_iter_updates = [&](int uind) {
      for (const auto& [poff, dv] : ptr_yields) {
        std::string de = scalar_exprs.count(dv) ? scalar_exprs.at(dv) : ssa_var(dv);
        out << indent_str(uind) << poff << " += long(" << de << ");\n";
      }
      if (!scalar_yields.empty()) {
        std::vector<std::string> tmps;
        for (const auto& [avar, yv, adt] : scalar_yields) {
          std::string t = next_tmp("m1_sy");
          std::string ye = scalar_exprs.count(yv) ? scalar_exprs.at(yv) : ssa_var(yv);
          out << indent_str(uind) << msl_scalar_type(adt) << " " << t << " = "
              << msl_cast_expr(ye, scalar_dtype_of(kernel.value(yv).type), adt, false) << ";\n";
          tmps.push_back(t);
        }
        for (size_t i = 0; i < scalar_yields.size(); ++i) {
          out << indent_str(uind) << std::get<0>(scalar_yields[i]) << " = " << tmps[i] << ";\n";
        }
      }
    };

    out << indent_str(ind) << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << ";\n";

    if (can_peel) {
      std::string fast_cond = align_var;
      for (const auto& ec : peel_extra_conds) fast_cond += " && " + ec;

      out << indent_str(ind) << "if (" << fast_cond << ") {\n";
      std::string fast_max_var = next_tmp("m1_fast_max");
      if (peel_upper_limits.empty()) {
        out << indent_str(ind + 1) << "long " << fast_max_var << " = long(" << hi_e << ");\n";
      } else {
        std::string lim = peel_upper_limits[0];
        for (size_t i = 1; i < peel_upper_limits.size(); ++i) {
          lim = "metal::min(" + lim + ", " + peel_upper_limits[i] + ")";
        }
        out << indent_str(ind + 1) << "long " << fast_max_var << " = metal::min(long(" << hi_e << "), (" << lim
            << ") + 1ll);\n";
      }
      if (chunks == 1) {
        out << indent_str(ind + 1) << "#pragma unroll 4\n";
      }
      out << indent_str(ind + 1) << "for (; long(" << iv_var << ") < " << fast_max_var << "; " << iv_var
          << " += " << step_e << ") {\n";
      int bind = ind + 2;
      if (chunks > 1) {
        if (chunks <= 4) out << indent_str(bind) << "#pragma unroll\n";
        out << indent_str(bind) << "for (uint m1_c = 0u; m1_c < " << chunks << "u; ++m1_c) {\n";
        ++bind;
        out << indent_str(bind) << "uint m1_elem_base = (m1_tid_linear + m1_c * " << tg_x << "u) * 4u;\n";
      } else {
        out << indent_str(bind) << "uint m1_elem_base = m1_tid_linear * 4u;\n";
      }
      if (!exact_cover) {
        out << indent_str(bind) << "if (m1_elem_base < " << block_size << "u) {\n";
        ++bind;
      }
      emit_vec4_ops_body(body_tile_ops, needed_fast, plans, loop_mats, &tile_yields, iv_var,
                         chunks > 1 ? "m1_c" : "0u", "m1_elem_base", false, bind, out);
      if (!err.empty()) return false;
      if (!exact_cover) {
        --bind;
        out << indent_str(bind) << "}\n";
      }
      if (chunks > 1) {
        --bind;
        out << indent_str(bind) << "}\n";
      }
      emit_iter_updates(ind + 2);
      out << indent_str(ind + 1) << "}\n";
      out << indent_str(ind) << "}\n";
    }

    out << indent_str(ind) << "for (; " << iv_var << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var
        << " += " << step_e << ") {\n";
    for (OpId soid : body_scalar_ops) {
      const Op& sop = kernel.op(soid);
      if (sop.kind == OpKind::Constant) continue;
      std::string expr = emit_scalar_op_expr(kernel, analysis, sop, scalar_exprs, helpers, err);
      if (!err.empty()) return false;
      ValueId rid = sop.results[0];
      const Type& rty = kernel.value(rid).type;
      std::string vn = ssa_var(rid);
      if (is_ptr_type(rty)) {
        out << indent_str(ind + 1) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vn << " = "
            << expr << ";\n";
      } else {
        out << indent_str(ind + 1) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vn << " = " << expr << ";\n";
      }
      scalar_exprs[rid] = vn;
    }

    int tind = ind + 1;
    if (chunks > 1) {
      out << indent_str(tind) << "for (uint m1_c = 0u; m1_c < " << chunks << "u; ++m1_c) {\n";
      ++tind;
      out << indent_str(tind) << "uint m1_elem_base = (m1_tid_linear + m1_c * " << tg_x << "u) * 4u;\n";
    } else {
      out << indent_str(tind) << "uint m1_elem_base = m1_tid_linear * 4u;\n";
    }
    if (!exact_cover) {
      out << indent_str(tind) << "if (m1_elem_base < " << block_size << "u) {\n";
      ++tind;
    }

    std::vector<std::string> chunk_guards;
    std::unordered_set<std::string> seen_cg;
    auto add_cg = [&](const std::string& g) {
      if (!g.empty() && seen_cg.insert(g).second) chunk_guards.push_back(g);
    };
    add_cg(align_var);
    for (OpId boid : body_tile_ops) {
      auto pit = plans.find(boid);
      if (pit == plans.end()) continue;
      for (const auto& vg : pit->second.dmask.var_scalar_guards) add_cg("(" + vg + ")");
      for (const auto& rb : pit->second.dmask.ramp_bounds) {
        add_cg(build_ramp_all4_guard(rb, iv_var, "m1_elem_base"));
      }
    }

    out << indent_str(tind) << "if (";
    for (size_t i = 0; i < chunk_guards.size(); ++i) {
      if (i > 0) out << " && ";
      out << chunk_guards[i];
    }
    out << ") {\n";
    emit_vec4_ops_body(body_tile_ops, needed_fast, plans, loop_mats, &tile_yields, iv_var, chunks > 1 ? "m1_c" : "0u",
                       "m1_elem_base", true, tind + 1, out);
    if (!err.empty()) return false;
    out << indent_str(tind) << "} else {\n";
    std::unordered_set<ValueId> empty_outs;
    emit_scalar_4lane_tail(body_tile_ops, empty_outs, &tile_yields, chunks > 1 ? "m1_c" : "0u", "m1_elem_base",
                           tind + 1, out);
    if (!err.empty()) return false;
    out << indent_str(tind) << "}\n";

    if (!exact_cover) {
      --tind;
      out << indent_str(tind) << "}\n";
    }
    if (chunks > 1) {
      --tind;
      out << indent_str(tind) << "}\n";
    }
    emit_iter_updates(ind + 1);
    out << indent_str(ind) << "}\n";
    return true;
  }

  bool emit_tile_segment(const std::vector<OpId>& seg_ops, const std::unordered_set<ValueId>& seg_live_outs,
                         const std::string& iv_var, int ind, std::ostringstream& out) {
    if (seg_ops.empty()) return true;

    std::unordered_map<OpId, MemOpPlan> plans;
    if (!build_mem_plans(seg_ops, plans)) return false;

    std::unordered_set<ValueId> needed_fast;
    std::vector<ValueId> wl;
    auto mark_need = [&](ValueId v) {
      if (v != kInvalidId && !is_scalar_like(kernel.value(v).type) && needed_fast.insert(v).second) wl.push_back(v);
    };
    for (ValueId lv : seg_live_outs) mark_need(lv);
    for (OpId oid : seg_ops) {
      const Op& op = kernel.op(oid);
      if (op.kind == OpKind::Store && op.operands.size() >= 2) mark_need(op.operands[1]);
    }
    std::unordered_set<OpId> seg_op_set(seg_ops.begin(), seg_ops.end());
    while (!wl.empty()) {
      ValueId v = wl.back();
      wl.pop_back();
      const Value& val = kernel.value(v);
      if (val.def_op == kInvalidId || !seg_op_set.count(val.def_op)) continue;
      const Op& dop = kernel.op(val.def_op);
      if (dop.kind == OpKind::Load) continue;
      for (ValueId o : dop.operands) mark_need(o);
    }

    bool has_any_work = !seg_live_outs.empty();
    for (OpId oid : seg_ops) {
      if (kernel.op(oid).kind == OpKind::Store) has_any_work = true;
    }
    if (!has_any_work) return true;

    for (OpId oid : seg_ops) {
      const Op& op = kernel.op(oid);
      if (op.results.empty()) continue;
      ValueId rid = op.results[0];
      if (!seg_live_outs.count(rid)) continue;
      if (is_ptr_type(kernel.value(rid).type)) continue;
      ScalarDType dt = scalar_dtype_of(kernel.value(rid).type);
      std::string vn = ssa_var(rid);
      if (chunks == 1) {
        out << indent_str(ind) << msl_vec4_type(dt) << " " << vn << " = " << msl_zero_literal(dt, true) << ";\n";
      } else {
        out << indent_str(ind) << msl_vec4_type(dt) << " " << vn << "[" << chunks << "];\n";
      }
      mat_tiles[rid] = MatTile{vn, dt, false, ReduceKind::Sum};
    }

    int cind = ind;
    std::string eb_var = next_tmp("m1_eb");
    if (chunks > 1) {
      if (chunks <= 4) out << indent_str(cind) << "#pragma unroll\n";
      out << indent_str(cind) << "for (uint m1_c = 0u; m1_c < " << chunks << "u; ++m1_c) {\n";
      ++cind;
      out << indent_str(cind) << "uint " << eb_var << " = (m1_tid_linear + m1_c * " << tg_x << "u) * 4u;\n";
    } else {
      out << indent_str(cind) << "uint " << eb_var << " = m1_tid_linear * 4u;\n";
    }
    if (!exact_cover) {
      out << indent_str(cind) << "if (" << eb_var << " < " << block_size << "u) {\n";
      ++cind;
    }

    std::vector<std::string> fast_guards;
    std::unordered_set<std::string> seen_fg;
    auto add_fg = [&](const std::string& g) {
      if (!g.empty() && g != "true" && seen_fg.insert(g).second) fast_guards.push_back(g);
    };

    for (OpId oid : seg_ops) {
      auto pit = plans.find(oid);
      if (pit == plans.end()) continue;
      const auto& plan = pit->second;
      if (!plan.lp.dyn_stride_expr.empty()) {
        add_fg("(" + plan.lp.dyn_stride_expr + " == 1ll)");
      }
      if (!plan.lp.inv_div4) {
        add_fg("((long(" + plan.lp.inv_offset + ") & 3ll) == 0ll)");
      }
      if (plan.lp.iv_coeff != 0 && !iv_var.empty()) {
        add_fg("((long(" + iv_var + ") & 3ll) == 0ll)");
      }
      for (const auto& vg : plan.dmask.var_scalar_guards) add_fg("(" + vg + ")");
      for (const auto& rb : plan.dmask.ramp_bounds) {
        add_fg(build_ramp_all4_guard(rb, iv_var, eb_var));
      }
    }

    std::unordered_map<ValueId, MatTile> empty_extra;
    if (fast_guards.empty()) {
      emit_vec4_ops_body(seg_ops, needed_fast, plans, empty_extra, nullptr, iv_var, chunks > 1 ? "m1_c" : "0u",
                         eb_var, true, cind, out);
      if (!err.empty()) return false;
    } else {
      out << indent_str(cind) << "if (";
      for (size_t i = 0; i < fast_guards.size(); ++i) {
        if (i > 0) out << " && ";
        out << fast_guards[i];
      }
      out << ") {\n";
      emit_vec4_ops_body(seg_ops, needed_fast, plans, empty_extra, nullptr, iv_var, chunks > 1 ? "m1_c" : "0u",
                         eb_var, true, cind + 1, out);
      if (!err.empty()) return false;
      out << indent_str(cind) << "} else {\n";
      emit_scalar_4lane_tail(seg_ops, seg_live_outs, nullptr, chunks > 1 ? "m1_c" : "0u", eb_var, cind + 1,
                             out);
      if (!err.empty()) return false;
      out << indent_str(cind) << "}\n";
    }

    if (!exact_cover) {
      --cind;
      out << indent_str(cind) << "}\n";
    }
    if (chunks > 1) {
      --cind;
      out << indent_str(cind) << "}\n";
    }
    return true;
  }

  bool emit_block(BlockId bid, const std::string& iv_var, bool in_loop, int ind,
                  const std::vector<std::tuple<std::string, ScalarDType>>* scalar_yield_targets,
                  std::ostringstream& out) {
    const Block& blk = kernel.block(bid);

    std::vector<OpId> live_ops;
    for (OpId oid : blk.ops) {
      if (!kernel.op(oid).erased) live_ops.push_back(oid);
    }

    auto has_later_smem_user = [&](size_t pos) -> bool {
      if (in_loop) return true;
      for (size_t j = pos + 1; j < live_ops.size(); ++j) {
        const Op& fop = kernel.op(live_ops[j]);
        if (is_reduce_op_kind(fop.kind) || is_scan_op_kind(fop.kind)) return true;
        for (BlockId rb : fop.regions) {
          if (block_has_reduce_or_scan(kernel, rb)) return true;
        }
      }
      return false;
    };

    std::vector<OpId> cur_seg;
    auto flush_seg = [&](size_t next_idx) -> bool {
      if (cur_seg.empty()) return true;
      std::unordered_set<OpId> seg_set(cur_seg.begin(), cur_seg.end());
      std::unordered_set<ValueId> seg_live_outs;
      for (OpId soid : cur_seg) {
        const Op& sop = kernel.op(soid);
        for (ValueId rid : sop.results) {
          if (is_scalar_like(kernel.value(rid).type) || is_ptr_type(kernel.value(rid).type)) continue;
          if (unwrap_scalar_splat(kernel, rid).has_value()) continue;
          auto uit = analysis.value_uses.find(rid);
          if (uit == analysis.value_uses.end()) continue;
          for (OpId uoid : uit->second) {
            const Op& uop = kernel.op(uoid);
            if (uop.erased) continue;
            if (!seg_set.count(uoid)) {
              if (uop.kind == OpKind::Load || uop.kind == OpKind::Store || uop.kind == OpKind::PtrAdd) {
                if (uop.kind == OpKind::Store && uop.operands.size() >= 2 && uop.operands[1] == rid) {
                  seg_live_outs.insert(rid);
                }
                continue;
              }
              if (uop.kind == OpKind::Binary && is_int_dtype(scalar_dtype_of(kernel.value(rid).type)) &&
                  lin_ints.count(rid) && lin_ints[rid].valid && !is_reduce_op_kind(uop.kind)) {
                bool only_addr_or_mask = true;
                for (OpId u2 : uit->second) {
                  const Op& op2 = kernel.op(u2);
                  if (is_reduce_op_kind(op2.kind) || op2.kind == OpKind::Yield || op2.kind == OpKind::ForLoop) {
                    only_addr_or_mask = false;
                  }
                }
                if (only_addr_or_mask) continue;
              }
              if (uop.kind == OpKind::ForLoop) {
                bool is_init = false;
                for (size_t k = 3; k < uop.operands.size(); ++k) {
                  if (uop.operands[k] == rid) {
                    if (unwrap_scalar_splat(kernel, rid).has_value()) {
                      is_init = true;
                    }
                  }
                }
                if (is_init) continue;
              }
              seg_live_outs.insert(rid);
              break;
            }
          }
        }
      }
      (void)next_idx;
      bool ok = emit_tile_segment(cur_seg, seg_live_outs, iv_var, ind, out);
      cur_seg.clear();
      return ok;
    };

    for (size_t i = 0; i < live_ops.size(); ++i) {
      OpId oid = live_ops[i];
      const Op& op = kernel.op(oid);

      if (op.kind == OpKind::Return) {
        if (!flush_seg(i)) return false;
        continue;
      }

      if (op.kind == OpKind::Yield) {
        if (!flush_seg(i)) return false;
        if (scalar_yield_targets != nullptr) {
          std::vector<std::string> tmps;
          for (size_t k = 0; k < op.operands.size() && k < scalar_yield_targets->size(); ++k) {
            ValueId ov = op.operands[k];
            if (!is_scalar_like(kernel.value(ov).type)) return false;
            ScalarDType dst_dt = std::get<1>((*scalar_yield_targets)[k]);
            std::string t = next_tmp("m1_y");
            out << indent_str(ind) << msl_scalar_type(dst_dt) << " " << t << " = "
                << msl_cast_expr(scalar_exprs.at(ov), scalar_dtype_of(kernel.value(ov).type), dst_dt, false) << ";\n";
            tmps.push_back(t);
          }
          for (size_t k = 0; k < tmps.size(); ++k) {
            out << indent_str(ind) << std::get<0>((*scalar_yield_targets)[k]) << " = " << tmps[k] << ";\n";
          }
        }
        continue;
      }

      if (op.kind == OpKind::Barrier) {
        if (!flush_seg(i)) return false;
        BarrierScope scope = std::holds_alternative<BarrierAttr>(op.attr) ? std::get<BarrierAttr>(op.attr).scope
                                                                          : BarrierScope::Threadgroup;
        if (scope == BarrierScope::Simdgroup) {
          out << indent_str(ind) << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
        } else if (scope == BarrierScope::DeviceMem) {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_device);\n";
        } else {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        }
        continue;
      }

      if (is_reduce_op_kind(op.kind)) {
        if (!flush_seg(i)) return false;
        emit_reduce_op(op, has_later_smem_user(i), ind, out);
        if (!err.empty()) return false;
        continue;
      }

      if (op.kind == OpKind::ForLoop) {
        if (!flush_seg(i)) return false;
        if (op.regions.empty()) return false;
        if (!block_has_reduce_or_scan(kernel, op.regions[0])) {
          if (!emit_for_loop_no_reduce(op, ind, out)) return false;
        } else {
          const Block& body = kernel.block(op.regions[0]);
          if (body.args.empty()) return false;
          ValueId iv_id = body.args[0];
          ScalarDType iv_dt = scalar_dtype_of(kernel.value(iv_id).type);
          std::string loop_iv = ssa_var(iv_id);
          std::string lo_e = msl_cast_expr(scalar_exprs.at(op.operands[0]),
                                           scalar_dtype_of(kernel.value(op.operands[0]).type), iv_dt, false);
          std::string hi_e = msl_cast_expr(scalar_exprs.at(op.operands[1]),
                                           scalar_dtype_of(kernel.value(op.operands[1]).type), iv_dt, false);
          std::string step_e = msl_cast_expr(scalar_exprs.at(op.operands[2]),
                                             scalar_dtype_of(kernel.value(op.operands[2]).type), iv_dt, false);
          bool neg_step = (analysis.const_i64.count(op.operands[2]) && analysis.const_i64.at(op.operands[2]) < 0);

          size_t num_iter = op.operands.size() - 3;
          std::vector<std::tuple<std::string, ScalarDType>> ytargets;
          for (size_t k = 0; k < num_iter; ++k) {
            ValueId arg_id = body.args[1 + k];
            ValueId init_id = op.operands[3 + k];
            const Type& aty = kernel.value(arg_id).type;
            if (!is_scalar_like(aty) || is_ptr_type(aty)) return false;
            ScalarDType adt = scalar_dtype_of(aty);
            std::string avar = ssa_var(arg_id);
            out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
                << msl_cast_expr(scalar_exprs.at(init_id), scalar_dtype_of(kernel.value(init_id).type), adt, false)
                << ";\n";
            scalar_exprs[arg_id] = avar;
            if (k < op.results.size()) scalar_exprs[op.results[k]] = avar;
            ytargets.emplace_back(avar, adt);
          }

          scalar_exprs[iv_id] = loop_iv;
          LinearInt iv_li;
          iv_li.valid = true;
          iv_li.iv_coeff = 1;
          iv_li.inv_expr = "0";
          iv_li.inv_is_zero = true;
          iv_li.inv_div4 = true;
          iv_li.dtype = iv_dt;
          lin_ints[iv_id] = iv_li;

          out << indent_str(ind) << "for (" << msl_scalar_type(iv_dt) << " " << loop_iv << " = " << lo_e << "; "
              << loop_iv << (neg_step ? " > " : " < ") << hi_e << "; " << loop_iv << " += " << step_e << ") {\n";
          if (!emit_block(op.regions[0], loop_iv, true, ind + 1, &ytargets, out)) return false;
          out << indent_str(ind) << "}\n";
        }
        continue;
      }

      if (op.kind == OpKind::If) {
        if (!flush_seg(i)) return false;
        if (op.operands.empty() || op.regions.empty()) return false;
        if (!is_scalar_like(kernel.value(op.operands[0]).type)) return false;
        std::vector<std::tuple<std::string, ScalarDType>> ytargets;
        for (ValueId rid : op.results) {
          const Type& rty = kernel.value(rid).type;
          if (!is_scalar_like(rty) || is_ptr_type(rty)) return false;
          ScalarDType rdt = scalar_dtype_of(rty);
          std::string rv = ssa_var(rid);
          out << indent_str(ind) << msl_scalar_type(rdt) << " " << rv << " = " << msl_zero_literal(rdt, false)
              << ";\n";
          scalar_exprs[rid] = rv;
          ytargets.emplace_back(rv, rdt);
        }
        out << indent_str(ind) << "if (" << scalar_exprs.at(op.operands[0]) << ") {\n";
        if (!emit_block(op.regions[0], iv_var, in_loop, ind + 1, &ytargets, out)) return false;
        if (op.regions.size() >= 2 && op.regions[1] != kInvalidId) {
          out << indent_str(ind) << "} else {\n";
          if (!emit_block(op.regions[1], iv_var, in_loop, ind + 1, &ytargets, out)) return false;
        }
        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.kind == OpKind::Store && is_scalar_like(kernel.value(op.operands[0]).type)) {
        if (!flush_seg(i)) return false;
        auto so = extract_store_operands(op);
        ScalarDType pdt = scalar_dtype_of(kernel.value(so.ptrs).type);
        std::string pe = scalar_exprs.at(so.ptrs);
        std::string ve =
            msl_cast_expr(scalar_exprs.at(so.val), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
        std::string guard = "(m1_tid_linear == 0u)";
        if (so.mask != kInvalidId) guard = "(" + guard + " && (" + scalar_exprs.at(so.mask) + "))";
        out << indent_str(ind) << "if (" << guard << ") { *(" << pe << ") = " << ve << "; }\n";
        continue;
      }

      if (op.kind == OpKind::AtomicAdd && is_scalar_like(kernel.value(op.operands[0]).type)) {
        if (!flush_seg(i)) return false;
        ValueId rid = op.results.empty() ? kInvalidId : op.results[0];
        ScalarDType pdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
        if (!err.empty()) return false;
        std::string vn = (rid != kInvalidId) ? ssa_var(rid) : next_tmp("m1_atom");
        out << indent_str(ind) << msl_scalar_type(pdt) << " " << vn << " = " << msl_zero_literal(pdt, false) << ";\n";
        out << indent_str(ind) << "if (m1_tid_linear == 0u) { " << vn << " = " << expr << "; }\n";
        if (rid != kInvalidId) scalar_exprs[rid] = vn;
        continue;
      }

      if (op.kind == OpKind::AtomicAdd) {
        return false;
      }

      bool is_scalar_def = !op.results.empty() && is_scalar_like(kernel.value(op.results[0]).type);
      if (is_scalar_def) {
        ValueId rid = op.results[0];
        const Type& rty = kernel.value(rid).type;
        if (op.kind == OpKind::Constant) {
          scalar_exprs[rid] = msl_const_literal(std::get<ConstAttr>(op.attr), scalar_dtype_of(rty));
          track_linear_info(op, in_loop);
          continue;
        }
        if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims ||
            op.kind == OpKind::Reshape) {
          scalar_exprs[rid] = scalar_exprs.at(op.operands[0]);
          track_linear_info(op, in_loop);
          continue;
        }
        if (!flush_seg(i)) return false;
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
        if (!err.empty()) return false;
        std::string vn = ssa_var(rid);
        if (is_ptr_type(rty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vn << " = " << expr
              << ";\n";
        } else {
          out << indent_str(ind) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vn << " = " << expr << ";\n";
        }
        scalar_exprs[rid] = vn;
        track_linear_info(op, in_loop);
        continue;
      }

      track_linear_info(op, in_loop);
      cur_seg.push_back(oid);
    }

    if (!flush_seg(live_ops.size())) return false;
    return true;
  }
};

std::optional<CompiledMSL> try_emit_1d_reduce_vec4(const Kernel& kernel, const TargetSpec& target,
                                                   const ReduceKernelAnalysis& analysis) {
  if (analysis.max_rank != 1 || analysis.has_scan || analysis.has_tile_reshape_or_permute ||
      analysis.has_multiple_1d_sizes) {
    return std::nullopt;
  }
  int64_t block_size = analysis.block_size_1d;
  if (block_size < 128 || (block_size % 4) != 0) {
    return std::nullopt;
  }

  uint32_t vec_elems = static_cast<uint32_t>(block_size / 4);
  uint32_t tg_x = std::clamp<uint32_t>(round_up_32(vec_elems), 32u, 256u);
  if (tg_x > target.max_threads_per_threadgroup) {
    tg_x = (target.max_threads_per_threadgroup / 32u) * 32u;
  }
  uint32_t chunks = (vec_elems + tg_x - 1u) / tg_x;
  uint32_t num_sg = tg_x / 32u;

  Vec4ReduceEmitter em{
      .kernel = kernel,
      .analysis = analysis,
      .target = target,
      .block_size = block_size,
      .tg_x = tg_x,
      .num_sg = num_sg,
      .chunks = chunks,
      .exact_cover = (chunks * tg_x == vec_elems),
  };
  em.smem.num_sg = num_sg;
  em.seed_entry_params();

  std::ostringstream body_ops;
  if (!em.emit_block(kernel.entry_block, "", false, 1, nullptr, body_ops)) {
    return std::nullopt;
  }

  auto [smem_decls, smem_bytes] = em.smem.emit_decls();
  if (!target.fits_threadgroup_memory(smem_bytes)) {
    return std::nullopt;
  }

  std::ostringstream full_body;
  full_body << "  (void)m1_grid_dim; (void)m1_tid; (void)m1_tg_size;\n";
  if (num_sg <= 1u) full_body << "  (void)m1_simd_lane; (void)m1_simd_id;\n";
  full_body << smem_decls;
  full_body << body_ops.str();
  full_body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, em.helpers) + full_body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = smem_bytes;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

// ---------------------------------------------------------------------------
// 1D General Scalar Hardware-SIMD Reduction Emitter
// ---------------------------------------------------------------------------

struct Scalar1DReduceEmitter {
  const Kernel& kernel;
  const ReduceKernelAnalysis& analysis;
  const TargetSpec& target;
  int64_t block_size = 1;
  uint32_t tg_x = 32;
  uint32_t num_sg = 1;
  uint32_t ept = 1;

  HelperFlags helpers;
  SmemRegistry smem;
  std::string err;
  uint32_t tmp_seq = 0;

  std::unordered_map<ValueId, std::string> scalar_exprs;
  struct MatScalarTile {
    std::string var_name;
    ScalarDType dtype = ScalarDType::F32;
    bool collapsed = false;
    ReduceKind collapse_kind = ReduceKind::Sum;
  };
  std::unordered_map<ValueId, MatScalarTile> mat_tiles;
  std::unordered_map<ValueId, std::pair<ValueId, std::string>> ptr_iter_map;

  std::string next_tmp(const std::string& pfx) { return pfx + "_" + std::to_string(tmp_seq++); }

  std::string tile_slot(const MatScalarTile& mt, const std::string& slot_var) const {
    if (mt.collapsed || ept == 1) return mt.var_name;
    return mt.var_name + "[" + slot_var + "]";
  }

  std::string eval_at(ValueId vid, const std::string& lane_expr, const std::string& slot_expr,
                      std::unordered_map<ValueId, std::string>& memo) {
    if (vid == kInvalidId) return "0";
    if (auto mit = memo.find(vid); mit != memo.end()) return mit->second;
    const Value& v = kernel.value(vid);
    if (is_scalar_like(v.type)) {
      if (auto sit = scalar_exprs.find(vid); sit != scalar_exprs.end()) return sit->second;
      if (v.def_op != kInvalidId && kernel.op(v.def_op).kind == OpKind::Constant) {
        return msl_const_literal(std::get<ConstAttr>(kernel.op(v.def_op).attr), scalar_dtype_of(v.type));
      }
      return ssa_var(vid);
    }
    if (auto pit = ptr_iter_map.find(vid); pit != ptr_iter_map.end()) {
      std::string init_p = eval_at(pit->second.first, lane_expr, slot_expr, memo);
      std::string e = "(" + init_p + " + " + pit->second.second + ")";
      memo[vid] = e;
      return e;
    }
    if (auto sit = scan_contig_tiles.find(vid); sit != scan_contig_tiles.end()) {
      std::string e = sit->second.buf_var + "[" + slot_expr + "]";
      memo[vid] = e;
      return e;
    }
    if (auto mt_it = mat_tiles.find(vid); mt_it != mat_tiles.end()) {
      std::string e = tile_slot(mt_it->second, slot_expr);
      memo[vid] = e;
      return e;
    }
    if (v.def_op == kInvalidId) return ssa_var(vid);
    const Op& op = kernel.op(v.def_op);
    ScalarDType rdt = scalar_dtype_of(v.type);
    std::string res;
    switch (op.kind) {
      case OpKind::Arange: {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        res = (attr.start == 0)
                  ? (msl_scalar_type(rdt) + "(" + lane_expr + ")")
                  : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) + "(" +
                     lane_expr + "))");
        break;
      }
      case OpKind::Splat:
      case OpKind::Broadcast:
      case OpKind::ExpandDims:
      case OpKind::Reshape:
        res = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        break;
      case OpKind::Cast: {
        std::string se = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        res = msl_cast_expr(se, scalar_dtype_of(kernel.value(op.operands[0]).type), rdt, false);
        break;
      }
      case OpKind::Unary: {
        std::string se = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        res = msl_unary_expr(std::get<UnaryAttr>(op.attr).op, se, rdt, false, helpers);
        break;
      }
      case OpKind::Binary: {
        std::string le = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        std::string re = eval_at(op.operands[1], lane_expr, slot_expr, memo);
        res = msl_binary_expr(std::get<BinaryAttr>(op.attr).op, le, re, scalar_dtype_of(kernel.value(op.operands[0]).type),
                              scalar_dtype_of(kernel.value(op.operands[1]).type), rdt, false);
        break;
      }
      case OpKind::Select: {
        std::string ce = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        std::string te = eval_at(op.operands[1], lane_expr, slot_expr, memo);
        std::string fe = eval_at(op.operands[2], lane_expr, slot_expr, memo);
        res = msl_select_expr(ce, te, fe, scalar_dtype_of(kernel.value(op.operands[1]).type),
                              scalar_dtype_of(kernel.value(op.operands[2]).type), rdt, false);
        break;
      }
      case OpKind::PtrAdd: {
        std::string pe = eval_at(op.operands[0], lane_expr, slot_expr, memo);
        std::string oe = eval_at(op.operands[1], lane_expr, slot_expr, memo);
        res = "(" + pe + " + " + oe + ")";
        break;
      }
      case OpKind::Load: {
        auto lo = extract_load_operands(op, kernel);
        std::string pe = eval_at(lo.ptrs, lane_expr, slot_expr, memo);
        if (lo.mask == kInvalidId) {
          res = "(*(" + pe + "))";
        } else {
          std::string me = eval_at(lo.mask, lane_expr, slot_expr, memo);
          std::string oe = (lo.other != kInvalidId)
                               ? msl_cast_expr(eval_at(lo.other, lane_expr, slot_expr, memo),
                                               scalar_dtype_of(kernel.value(lo.other).type), rdt, false)
                               : msl_zero_literal(rdt, false);
          res = "(" + me + " ? (*(" + pe + ")) : " + oe + ")";
        }
        break;
      }
      default:
        err = "Unsupported tile op in 1D scalar reduction emitter";
        return "0";
    }
    memo[vid] = res;
    return res;
  }

  void emit_reduce_op(const Op& op, bool emit_trailing_barrier, int ind, std::ostringstream& out) {
    const auto& ra = std::get<ReduceAttr>(op.attr);
    ValueId in_v = op.operands[0];
    int64_t len = kernel.value(in_v).type.as_tile().shape[0];
    ScalarDType dt = scalar_dtype_of(kernel.value(in_v).type);
    const std::string ty = msl_scalar_type(dt);
    bool is_arg = (ra.kind == ReduceKind::Argmax || ra.kind == ReduceKind::Argmin ||
                   ra.result_mode == ReduceResultMode::ValueAndIndex || ra.result_mode == ReduceResultMode::IndexOnly);
    bool is_max = (ra.kind == ReduceKind::Max || ra.kind == ReduceKind::Argmax);
    bool broadcast_all = analysis.broadcast_reduces.count(op.id) > 0;

    std::string loc_val = next_tmp("m1_rv");
    std::string loc_idx = is_arg ? next_tmp("m1_ri") : "";

    out << indent_str(ind) << ty << " " << loc_val << " = " << msl_reduce_identity(ra.kind, dt) << ";\n";
    if (is_arg) out << indent_str(ind) << "int " << loc_idx << " = 2147483647;\n";

    auto mt_it = mat_tiles.find(in_v);
    if (mt_it != mat_tiles.end() && mt_it->second.collapsed) {
      out << indent_str(ind) << loc_val << " = " << mt_it->second.var_name << ";\n";
    } else if (mt_it != mat_tiles.end()) {
      out << indent_str(ind) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
      out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
      out << indent_str(ind + 1) << "if (m1_lane < " << len << "u) {\n";
      std::string ev = tile_slot(mt_it->second, "m1_s");
      if (!is_arg) {
        out << indent_str(ind + 2) << loc_val << " = " << scalar_combine_expr(ra.kind, loc_val, ev) << ";\n";
      } else {
        const std::string cmp = is_max ? " > " : " < ";
        out << indent_str(ind + 2) << ty << " m1_lv = " << ev << ";\n";
        out << indent_str(ind + 2) << "int m1_li = int(m1_lane);\n";
        out << indent_str(ind + 2) << "bool m1_bt = (m1_lv" << cmp << loc_val << ") || ((m1_lv == " << loc_val
            << ") && (m1_li < " << loc_idx << "));\n";
        out << indent_str(ind + 2) << loc_val << " = m1_bt ? m1_lv : " << loc_val << ";\n";
        out << indent_str(ind + 2) << loc_idx << " = m1_bt ? m1_li : " << loc_idx << ";\n";
      }
      out << indent_str(ind + 1) << "}\n";
      out << indent_str(ind) << "}\n";
    } else {
      out << indent_str(ind) << "for (uint m1_s = 0u; ; ++m1_s) {\n";
      out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
      out << indent_str(ind + 1) << "if (m1_lane >= " << len << "u) break;\n";
      std::unordered_map<ValueId, std::string> memo;
      std::string ev = eval_at(in_v, "m1_lane", "m1_s", memo);
      if (!err.empty()) return;
      if (!is_arg) {
        out << indent_str(ind + 1) << loc_val << " = " << scalar_combine_expr(ra.kind, loc_val, ev) << ";\n";
      } else {
        const std::string cmp = is_max ? " > " : " < ";
        out << indent_str(ind + 1) << ty << " m1_lv = " << ev << ";\n";
        out << indent_str(ind + 1) << "int m1_li = int(m1_lane);\n";
        out << indent_str(ind + 1) << "bool m1_bt = (m1_lv" << cmp << loc_val << ") || ((m1_lv == " << loc_val
            << ") && (m1_li < " << loc_idx << "));\n";
        out << indent_str(ind + 1) << loc_val << " = m1_bt ? m1_lv : " << loc_val << ";\n";
        out << indent_str(ind + 1) << loc_idx << " = m1_bt ? m1_li : " << loc_idx << ";\n";
      }
      out << indent_str(ind) << "}\n";
    }

    if (!is_arg) {
      ValueId rid = op.results[0];
      std::string rv = ssa_var(rid);
      emit_cross_simd_value_reduce(ra.kind, dt, loc_val, rv, num_sg, broadcast_all, emit_trailing_barrier, smem, ind,
                                   out);
      scalar_exprs[rid] = rv;
    } else if (ra.result_mode == ReduceResultMode::ValueAndIndex && op.results.size() >= 2) {
      std::string rv = ssa_var(op.results[0]);
      std::string ri = ssa_var(op.results[1]);
      emit_cross_simd_arg_reduce(is_max, dt, loc_val, loc_idx, rv, ri, num_sg, broadcast_all, emit_trailing_barrier,
                                 smem, ind, out);
      scalar_exprs[op.results[0]] = rv;
      scalar_exprs[op.results[1]] = ri;
    } else {
      std::string ri = ssa_var(op.results[0]);
      emit_cross_simd_arg_reduce(is_max, dt, loc_val, loc_idx, "", ri, num_sg, broadcast_all, emit_trailing_barrier,
                                 smem, ind, out);
      scalar_exprs[op.results[0]] = ri;
    }
  }

  void emit_scan_op(const Op& op, bool emit_trailing_barrier, int ind, std::ostringstream& out) {
    const auto& sa = std::get<ScanAttr>(op.attr);
    ValueId in_v = op.operands[0];
    ValueId mask_v = (sa.has_mask && op.operands.size() >= 2) ? op.operands[1] : kInvalidId;
    ValueId out_v = op.results[0];
    int64_t len = kernel.value(in_v).type.as_tile().shape[0];
    ScalarDType dt = scalar_dtype_of(kernel.value(in_v).type);
    const std::string ty = msl_scalar_type(dt);
    const bool is_sum = (sa.kind == ScanKind::Cumsum);
    const std::string ident = is_sum ? msl_zero_literal(dt, false) : msl_one_literal(dt, false);
    const std::string comb_op = is_sum ? " + " : " * ";

    uint32_t seg_ept = static_cast<uint32_t>((len + tg_x - 1) / tg_x);
    if (seg_ept > 32u) {
      err = "1D Scan tile exceeds per-thread register budget (len / tg_x > 32)";
      return;
    }

    std::string loc_buf = ssa_var(out_v);
    out << indent_str(ind) << ty << " " << loc_buf << "[" << seg_ept << "];\n";
    std::string loc_tot = next_tmp("m1_stot");
    out << indent_str(ind) << ty << " " << loc_tot << " = " << ident << ";\n";
    std::string base_var = next_tmp("m1_sbase");
    out << indent_str(ind) << "uint " << base_var << " = m1_tid_linear * " << seg_ept << "u;\n";

    out << indent_str(ind) << "for (uint m1_k = 0u; m1_k < " << seg_ept << "u; ++m1_k) {\n";
    out << indent_str(ind + 1) << "uint m1_pos = " << base_var << " + m1_k;\n";
    out << indent_str(ind + 1) << "if (m1_pos < " << len << "u) {\n";
    std::string phys_lane = sa.reverse ? ("(" + std::to_string(len - 1) + "u - m1_pos)") : "m1_pos";
    out << indent_str(ind + 2) << "uint m1_plane = " << phys_lane << ";\n";
    std::unordered_map<ValueId, std::string> memo;
    std::string in_e;
    if (mat_tiles.count(in_v)) {
      err = "Scan input must be evaluated directly or unmaterialized";
      return;
    } else {
      in_e = eval_at(in_v, "m1_plane", "0u", memo);
      if (!err.empty()) return;
    }
    if (mask_v != kInvalidId) {
      std::string me = eval_at(mask_v, "m1_plane", "0u", memo);
      if (!err.empty()) return;
      in_e = "(" + me + " ? " + in_e + " : " + ident + ")";
    }
    out << indent_str(ind + 2) << ty << " m1_sv = " << in_e << ";\n";
    if (sa.exclusive) {
      out << indent_str(ind + 2) << loc_buf << "[m1_k] = " << loc_tot << ";\n";
      out << indent_str(ind + 2) << loc_tot << " = (" << loc_tot << comb_op << "m1_sv);\n";
    } else {
      out << indent_str(ind + 2) << loc_tot << " = (" << loc_tot << comb_op << "m1_sv);\n";
      out << indent_str(ind + 2) << loc_buf << "[m1_k] = " << loc_tot << ";\n";
    }
    out << indent_str(ind + 1) << "} else {\n";
    out << indent_str(ind + 2) << loc_buf << "[m1_k] = " << ident << ";\n";
    out << indent_str(ind + 1) << "}\n";
    out << indent_str(ind) << "}\n";

    std::string sg_pref = next_tmp("m1_sgp");
    std::string sg_tot = next_tmp("m1_sgt");
    if (is_sum && dt != ScalarDType::I64 && dt != ScalarDType::Bool) {
      std::string sg_inc = next_tmp("m1_sgi");
      out << indent_str(ind) << ty << " " << sg_inc << " = simd_prefix_inclusive_sum(" << loc_tot << ");\n";
      out << indent_str(ind) << ty << " " << sg_pref << " = (" << sg_inc << " - " << loc_tot << ");\n";
      out << indent_str(ind) << ty << " " << sg_tot << " = simd_shuffle(" << sg_inc << ", 31u);\n";
    } else {
      std::string sg_inc = next_tmp("m1_sgi");
      out << indent_str(ind) << ty << " " << sg_inc << " = " << loc_tot << ";\n";
      out << indent_str(ind) << "for (ushort m1_d = 1u; m1_d < 32u; m1_d <<= 1u) {\n";
      if (dt == ScalarDType::I64) {
        out << indent_str(ind + 1) << "uint2 m1_b = as_type<uint2>(" << sg_inc << ");\n";
        out << indent_str(ind + 1) << "long m1_up = as_type<long>(uint2(simd_shuffle_up(m1_b.x, m1_d), "
                                      "simd_shuffle_up(m1_b.y, m1_d)));\n";
      } else {
        out << indent_str(ind + 1) << ty << " m1_up = simd_shuffle_up(" << sg_inc << ", m1_d);\n";
      }
      out << indent_str(ind + 1) << "if (m1_simd_lane >= m1_d) { " << sg_inc << " = (" << sg_inc << comb_op
          << "m1_up); }\n";
      out << indent_str(ind) << "}\n";
      out << indent_str(ind) << ty << " " << sg_pref << " = " << ident << ";\n";
      if (dt == ScalarDType::I64) {
        out << indent_str(ind) << "uint2 m1_pb = as_type<uint2>(" << sg_inc << ");\n";
        out << indent_str(ind) << "long m1_pup = as_type<long>(uint2(simd_shuffle_up(m1_pb.x, 1u), "
                                  "simd_shuffle_up(m1_pb.y, 1u)));\n";
        out << indent_str(ind) << "if (m1_simd_lane > 0u) { " << sg_pref << " = m1_pup; }\n";
        out << indent_str(ind) << "long " << sg_tot
            << " = as_type<long>(uint2(simd_shuffle(m1_pb.x, 31u), simd_shuffle(m1_pb.y, 31u)));\n";
      } else {
        out << indent_str(ind) << ty << " m1_pup = simd_shuffle_up(" << sg_inc << ", 1u);\n";
        out << indent_str(ind) << "if (m1_simd_lane > 0u) { " << sg_pref << " = m1_pup; }\n";
        out << indent_str(ind) << ty << " " << sg_tot << " = simd_shuffle(" << sg_inc << ", 31u);\n";
      }
    }

    std::string tg_pref = next_tmp("m1_tgp");
    out << indent_str(ind) << ty << " " << tg_pref << " = " << sg_pref << ";\n";
    if (num_sg > 1u) {
      std::string sarr = smem.array_for(dt);
      out << indent_str(ind) << "if (m1_simd_lane == 0u) { " << sarr << "[m1_simd_id] = " << sg_tot << "; }\n";
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      out << indent_str(ind) << "if (m1_simd_id == 0u) {\n";
      out << indent_str(ind + 1) << ty << " m1_sv = (m1_simd_lane < " << num_sg << "u) ? " << sarr
          << "[m1_simd_lane] : " << ident << ";\n";
      if (is_sum && dt != ScalarDType::I64 && dt != ScalarDType::Bool) {
        out << indent_str(ind + 1) << ty << " m1_sp = simd_prefix_inclusive_sum(m1_sv) - m1_sv;\n";
      } else {
        out << indent_str(ind + 1) << ty << " m1_sc = m1_sv;\n";
        out << indent_str(ind + 1) << "for (ushort m1_d = 1u; m1_d < " << next_pow2_u32(num_sg)
            << "u; m1_d <<= 1u) {\n";
        if (dt == ScalarDType::I64) {
          out << indent_str(ind + 2) << "uint2 m1_b = as_type<uint2>(m1_sc);\n";
          out << indent_str(ind + 2) << "long m1_up = as_type<long>(uint2(simd_shuffle_up(m1_b.x, m1_d), "
                                        "simd_shuffle_up(m1_b.y, m1_d)));\n";
        } else {
          out << indent_str(ind + 2) << ty << " m1_up = simd_shuffle_up(m1_sc, m1_d);\n";
        }
        out << indent_str(ind + 2) << "if (m1_simd_lane >= m1_d) { m1_sc = (m1_sc" << comb_op << "m1_up); }\n";
        out << indent_str(ind + 1) << "}\n";
        out << indent_str(ind + 1) << ty << " m1_sp = " << ident << ";\n";
        if (dt == ScalarDType::I64) {
          out << indent_str(ind + 1) << "uint2 m1_b2 = as_type<uint2>(m1_sc);\n";
          out << indent_str(ind + 1) << "long m1_p2 = as_type<long>(uint2(simd_shuffle_up(m1_b2.x, 1u), "
                                        "simd_shuffle_up(m1_b2.y, 1u)));\n";
          out << indent_str(ind + 1) << "if (m1_simd_lane > 0u) { m1_sp = m1_p2; }\n";
        } else {
          out << indent_str(ind + 1) << ty << " m1_p2 = simd_shuffle_up(m1_sc, 1u);\n";
          out << indent_str(ind + 1) << "if (m1_simd_lane > 0u) { m1_sp = m1_p2; }\n";
        }
      }
      out << indent_str(ind + 1) << "if (m1_simd_lane < " << num_sg << "u) { " << sarr << "[m1_simd_lane] = m1_sp; }\n";
      out << indent_str(ind) << "}\n";
      out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      out << indent_str(ind) << tg_pref << " = (" << sarr << "[m1_simd_id]" << comb_op << sg_pref << ");\n";
      if (emit_trailing_barrier) {
        out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      }
    }

    out << indent_str(ind) << "for (uint m1_k = 0u; m1_k < " << seg_ept << "u; ++m1_k) {\n";
    out << indent_str(ind + 1) << loc_buf << "[m1_k] = (" << tg_pref << comb_op << loc_buf << "[m1_k]);\n";
    out << indent_str(ind) << "}\n";

    mat_tiles[out_v] = MatScalarTile{loc_buf, dt, false, ReduceKind::Sum};
    scan_contig_tiles[out_v] = ScanContigInfo{loc_buf, base_var, seg_ept, len, sa.reverse};
  }

  struct ScanContigInfo {
    std::string buf_var;
    std::string base_var;
    uint32_t seg_ept = 1;
    int64_t len = 1;
    bool reverse = false;
  };
  std::unordered_map<ValueId, ScanContigInfo> scan_contig_tiles;

  bool emit_block(BlockId bid, bool in_loop, int ind,
                  const std::vector<std::tuple<std::string, ScalarDType>>* scalar_yield_targets,
                  std::ostringstream& out) {
    const Block& blk = kernel.block(bid);
    std::vector<OpId> live_ops;
    for (OpId oid : blk.ops) {
      if (!kernel.op(oid).erased) live_ops.push_back(oid);
    }

    auto has_later_smem_user = [&](size_t pos) -> bool {
      if (in_loop) return true;
      for (size_t j = pos + 1; j < live_ops.size(); ++j) {
        const Op& fop = kernel.op(live_ops[j]);
        if (is_reduce_op_kind(fop.kind) || is_scan_op_kind(fop.kind)) return true;
        for (BlockId rb : fop.regions) {
          if (block_has_reduce_or_scan(kernel, rb)) return true;
        }
      }
      return false;
    };

    std::unordered_set<ValueId> must_materialize;
    for (size_t i = 0; i < live_ops.size(); ++i) {
      const Op& op = kernel.op(live_ops[i]);
      if (op.results.empty()) continue;
      ValueId rid = op.results[0];
      if (is_scalar_like(kernel.value(rid).type) || is_ptr_type(kernel.value(rid).type)) continue;
      if (is_scan_op_kind(op.kind)) continue;
      bool used_after_reduce_or_multi = false;
      int reduce_uses = 0;
      int non_reduce_uses = 0;
      if (auto uit = analysis.value_uses.find(rid); uit != analysis.value_uses.end()) {
        for (OpId uoid : uit->second) {
          const Op& uop = kernel.op(uoid);
          if (uop.erased) continue;
          if (is_reduce_op_kind(uop.kind)) {
            ++reduce_uses;
          } else {
            ++non_reduce_uses;
          }
        }
      }
      if ((reduce_uses >= 1 && non_reduce_uses >= 1) || reduce_uses >= 2) {
        used_after_reduce_or_multi = true;
      }
      if (used_after_reduce_or_multi && ept <= 16 &&
          (op.kind == OpKind::Load || op.kind == OpKind::Unary || op.kind == OpKind::Binary ||
           op.kind == OpKind::Select || op.kind == OpKind::Cast)) {
        must_materialize.insert(rid);
      }
    }

    for (size_t i = 0; i < live_ops.size(); ++i) {
      OpId oid = live_ops[i];
      const Op& op = kernel.op(oid);

      if (op.kind == OpKind::Return) continue;

      if (op.kind == OpKind::Yield) {
        if (scalar_yield_targets != nullptr) {
          std::vector<std::string> tmps;
          for (size_t k = 0; k < op.operands.size() && k < scalar_yield_targets->size(); ++k) {
            ValueId ov = op.operands[k];
            ScalarDType dst_dt = std::get<1>((*scalar_yield_targets)[k]);
            std::string t = next_tmp("m1_y");
            out << indent_str(ind) << msl_scalar_type(dst_dt) << " " << t << " = "
                << msl_cast_expr(scalar_exprs.at(ov), scalar_dtype_of(kernel.value(ov).type), dst_dt, false) << ";\n";
            tmps.push_back(t);
          }
          for (size_t k = 0; k < tmps.size(); ++k) {
            out << indent_str(ind) << std::get<0>((*scalar_yield_targets)[k]) << " = " << tmps[k] << ";\n";
          }
        }
        continue;
      }

      if (op.kind == OpKind::Barrier) {
        BarrierScope scope = std::holds_alternative<BarrierAttr>(op.attr) ? std::get<BarrierAttr>(op.attr).scope
                                                                          : BarrierScope::Threadgroup;
        if (scope == BarrierScope::Simdgroup) {
          out << indent_str(ind) << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
        } else if (scope == BarrierScope::DeviceMem) {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_device);\n";
        } else {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        }
        continue;
      }

      if (is_reduce_op_kind(op.kind)) {
        emit_reduce_op(op, has_later_smem_user(i), ind, out);
        if (!err.empty()) return false;
        continue;
      }

      if (is_scan_op_kind(op.kind)) {
        emit_scan_op(op, has_later_smem_user(i), ind, out);
        if (!err.empty()) return false;
        continue;
      }

      if (op.kind == OpKind::ForLoop) {
        if (op.operands.size() < 3 || op.regions.empty()) {
          err = "Malformed ForLoop";
          return false;
        }
        const Block& body = kernel.block(op.regions[0]);
        if (body.args.empty()) {
          err = "ForLoop missing iv";
          return false;
        }
        ValueId iv_id = body.args[0];
        ScalarDType iv_dt = scalar_dtype_of(kernel.value(iv_id).type);
        std::string iv_var = ssa_var(iv_id);
        std::string lo_e = msl_cast_expr(scalar_exprs.at(op.operands[0]),
                                         scalar_dtype_of(kernel.value(op.operands[0]).type), iv_dt, false);
        std::string hi_e = msl_cast_expr(scalar_exprs.at(op.operands[1]),
                                         scalar_dtype_of(kernel.value(op.operands[1]).type), iv_dt, false);
        std::string step_e = msl_cast_expr(scalar_exprs.at(op.operands[2]),
                                           scalar_dtype_of(kernel.value(op.operands[2]).type), iv_dt, false);
        bool neg_step = (analysis.const_i64.count(op.operands[2]) && analysis.const_i64.at(op.operands[2]) < 0);

        if (block_has_reduce_or_scan(kernel, op.regions[0])) {
          size_t num_iter = op.operands.size() - 3;
          std::vector<std::tuple<std::string, ScalarDType>> ytargets;
          for (size_t k = 0; k < num_iter; ++k) {
            ValueId arg_id = body.args[1 + k];
            ValueId init_id = op.operands[3 + k];
            const Type& aty = kernel.value(arg_id).type;
            if (!is_scalar_like(aty)) {
              err = "ForLoop containing Reduce/Scan requires scalar iter_args";
              return false;
            }
            ScalarDType adt = scalar_dtype_of(aty);
            std::string avar = ssa_var(arg_id);
            out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
                << msl_cast_expr(scalar_exprs.at(init_id), scalar_dtype_of(kernel.value(init_id).type), adt, false)
                << ";\n";
            scalar_exprs[arg_id] = avar;
            if (k < op.results.size()) scalar_exprs[op.results[k]] = avar;
            ytargets.emplace_back(avar, adt);
          }
          scalar_exprs[iv_id] = iv_var;
          out << indent_str(ind) << "for (" << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << "; "
              << iv_var << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var << " += " << step_e << ") {\n";
          if (!emit_block(op.regions[0], true, ind + 1, &ytargets, out)) return false;
          out << indent_str(ind) << "}\n";
          continue;
        }

        const Op* yield_op = nullptr;
        for (OpId boid : body.ops) {
          const Op& bop = kernel.op(boid);
          if (!bop.erased && bop.kind == OpKind::Yield) {
            yield_op = &bop;
            break;
          }
        }
        if (!yield_op) {
          err = "ForLoop missing Yield";
          return false;
        }

        size_t num_iter = op.operands.size() - 3;
        std::vector<std::tuple<std::string, ValueId, ScalarDType>> scalar_yields;
        std::vector<std::pair<std::string, ValueId>> ptr_yields;
        std::vector<std::pair<ValueId, MatScalarTile>> tile_yields;

        for (size_t k = 0; k < num_iter; ++k) {
          ValueId arg_id = body.args[1 + k];
          ValueId init_id = op.operands[3 + k];
          const Type& aty = kernel.value(arg_id).type;
          std::string avar = ssa_var(arg_id);
          if (is_scalar_like(aty)) {
            ScalarDType adt = scalar_dtype_of(aty);
            if (is_ptr_type(aty)) {
              out << indent_str(ind) << msl_ptr_type_str(analysis, arg_id, adt) << " " << avar << " = "
                  << scalar_exprs.at(init_id) << ";\n";
            } else {
              out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
                  << msl_cast_expr(scalar_exprs.at(init_id), scalar_dtype_of(kernel.value(init_id).type), adt, false)
                  << ";\n";
            }
            scalar_exprs[arg_id] = avar;
            if (k < op.results.size()) scalar_exprs[op.results[k]] = avar;
            scalar_yields.emplace_back(avar, yield_op->operands[k], adt);
          } else if (is_ptr_type(aty)) {
            ValueId delta_v = kInvalidId;
            if (!detect_ptr_iter_delta(kernel, arg_id, yield_op->operands[k], delta_v)) {
              err = "Unsupported pointer-tile ForLoop iter_arg update";
              return false;
            }
            std::string poff = "m1_poff_" + std::to_string(arg_id);
            out << indent_str(ind) << "long " << poff << " = 0ll;\n";
            ptr_iter_map[arg_id] = {init_id, poff};
            if (k < op.results.size()) ptr_iter_map[op.results[k]] = {init_id, poff};
            ptr_yields.emplace_back(poff, delta_v);
          } else {
            ScalarDType adt = scalar_dtype_of(aty);
            auto coll = detect_collapsible_acc(kernel, analysis, op, k);
            bool is_coll = coll.has_value();
            if (!is_coll && ept > 32u) {
              err = "Non-collapsible 1D tile iter_arg exceeds per-thread register limit";
              return false;
            }
            MatScalarTile mt{avar, adt, is_coll, is_coll ? coll->kind : ReduceKind::Sum};
            if (is_coll) {
              out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
                  << msl_reduce_identity(coll->kind, adt) << ";\n";
            } else if (ept == 1) {
              std::unordered_map<ValueId, std::string> lm;
              std::string ie = eval_at(init_id, "m1_tid_linear", "0u", lm);
              out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = (m1_tid_linear < " << block_size
                  << "u) ? " << msl_cast_expr(ie, scalar_dtype_of(kernel.value(init_id).type), adt, false) << " : "
                  << msl_zero_literal(adt, false) << ";\n";
            } else {
              out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << "[" << ept << "];\n";
              out << indent_str(ind) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
              out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
              std::unordered_map<ValueId, std::string> lm;
              std::string ie = eval_at(init_id, "m1_lane", "m1_s", lm);
              out << indent_str(ind + 1) << avar << "[m1_s] = (m1_lane < " << block_size << "u) ? "
                  << msl_cast_expr(ie, scalar_dtype_of(kernel.value(init_id).type), adt, false) << " : "
                  << msl_zero_literal(adt, false) << ";\n";
              out << indent_str(ind) << "}\n";
            }
            mat_tiles[arg_id] = mt;
            if (k < op.results.size()) mat_tiles[op.results[k]] = mt;
            tile_yields.emplace_back(yield_op->operands[k], mt);
          }
        }

        scalar_exprs[iv_id] = iv_var;
        out << indent_str(ind) << "for (" << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << "; " << iv_var
            << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var << " += " << step_e << ") {\n";

        for (OpId boid : body.ops) {
          const Op& bop = kernel.op(boid);
          if (bop.erased || bop.kind == OpKind::Yield) continue;
          if (!bop.results.empty() && is_scalar_like(kernel.value(bop.results[0]).type)) {
            ValueId rid = bop.results[0];
            if (bop.kind == OpKind::Constant) {
              scalar_exprs[rid] =
                  msl_const_literal(std::get<ConstAttr>(bop.attr), scalar_dtype_of(kernel.value(rid).type));
            } else {
              std::string expr = emit_scalar_op_expr(kernel, analysis, bop, scalar_exprs, helpers, err);
              if (!err.empty()) return false;
              std::string vn = ssa_var(rid);
              const Type& rty = kernel.value(rid).type;
              if (is_ptr_type(rty)) {
                out << indent_str(ind + 1) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vn
                    << " = " << expr << ";\n";
              } else {
                out << indent_str(ind + 1) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vn << " = " << expr
                    << ";\n";
              }
              scalar_exprs[rid] = vn;
            }
          }
        }

        bool body_has_tile_work = !tile_yields.empty();
        for (OpId boid : body.ops) {
          const Op& bop = kernel.op(boid);
          if (!bop.erased && (bop.kind == OpKind::Store || bop.kind == OpKind::AtomicAdd)) {
            body_has_tile_work = true;
          }
        }
        if (body_has_tile_work) {
          out << indent_str(ind + 1) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
          out << indent_str(ind + 2) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
          out << indent_str(ind + 2) << "if (m1_lane < " << block_size << "u) {\n";
          std::unordered_map<ValueId, std::string> memo;
          for (OpId boid : body.ops) {
            const Op& bop = kernel.op(boid);
            if (bop.erased || bop.kind == OpKind::Yield) continue;
            if (bop.kind == OpKind::Store && !is_scalar_like(kernel.value(bop.operands[0]).type)) {
              auto so = extract_store_operands(bop);
              ScalarDType pdt = scalar_dtype_of(kernel.value(so.ptrs).type);
              std::string pe = eval_at(so.ptrs, "m1_lane", "m1_s", memo);
              std::string ve = msl_cast_expr(eval_at(so.val, "m1_lane", "m1_s", memo),
                                             scalar_dtype_of(kernel.value(so.val).type), pdt, false);
              if (!err.empty()) return false;
              if (so.mask != kInvalidId) {
                std::string me = eval_at(so.mask, "m1_lane", "m1_s", memo);
                out << indent_str(ind + 3) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
              } else {
                out << indent_str(ind + 3) << "*(" << pe << ") = " << ve << ";\n";
              }
            }
          }
          std::vector<std::string> ytmps;
          for (const auto& [yv, mt] : tile_yields) {
            std::string ev =
                msl_cast_expr(eval_at(yv, "m1_lane", "m1_s", memo), scalar_dtype_of(kernel.value(yv).type), mt.dtype,
                              false);
            if (!err.empty()) return false;
            std::string t = next_tmp("m1_ty");
            out << indent_str(ind + 3) << msl_scalar_type(mt.dtype) << " " << t << " = " << ev << ";\n";
            ytmps.push_back(t);
          }
          for (size_t k = 0; k < tile_yields.size(); ++k) {
            out << indent_str(ind + 3) << tile_slot(tile_yields[k].second, "m1_s") << " = " << ytmps[k] << ";\n";
          }
          out << indent_str(ind + 2) << "}\n";
          out << indent_str(ind + 1) << "}\n";
        }

        for (const auto& [poff, dv] : ptr_yields) {
          std::string de = scalar_exprs.count(dv) ? scalar_exprs.at(dv) : ssa_var(dv);
          out << indent_str(ind + 1) << poff << " += long(" << de << ");\n";
        }
        if (!scalar_yields.empty()) {
          std::vector<std::string> tmps;
          for (const auto& [avar, yv, adt] : scalar_yields) {
            std::string t = next_tmp("m1_sy");
            out << indent_str(ind + 1) << msl_scalar_type(adt) << " " << t << " = "
                << msl_cast_expr(scalar_exprs.at(yv), scalar_dtype_of(kernel.value(yv).type), adt, false) << ";\n";
            tmps.push_back(t);
          }
          for (size_t k = 0; k < scalar_yields.size(); ++k) {
            out << indent_str(ind + 1) << std::get<0>(scalar_yields[k]) << " = " << tmps[k] << ";\n";
          }
        }

        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.kind == OpKind::If) {
        if (op.operands.empty() || op.regions.empty()) {
          err = "Malformed If op";
          return false;
        }
        std::vector<std::tuple<std::string, ScalarDType>> ytargets;
        for (ValueId rid : op.results) {
          const Type& rty = kernel.value(rid).type;
          if (!is_scalar_like(rty) || is_ptr_type(rty)) {
            err = "Unsupported tile result on If in reduce/scan emitter";
            return false;
          }
          ScalarDType rdt = scalar_dtype_of(rty);
          std::string rv = ssa_var(rid);
          out << indent_str(ind) << msl_scalar_type(rdt) << " " << rv << " = " << msl_zero_literal(rdt, false)
              << ";\n";
          scalar_exprs[rid] = rv;
          ytargets.emplace_back(rv, rdt);
        }
        out << indent_str(ind) << "if (" << scalar_exprs.at(op.operands[0]) << ") {\n";
        if (!emit_block(op.regions[0], in_loop, ind + 1, &ytargets, out)) return false;
        if (op.regions.size() >= 2 && op.regions[1] != kInvalidId) {
          out << indent_str(ind) << "} else {\n";
          if (!emit_block(op.regions[1], in_loop, ind + 1, &ytargets, out)) return false;
        }
        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        const Type& pty = kernel.value(so.ptrs).type;
        ScalarDType pdt = scalar_dtype_of(pty);
        if (is_scalar_like(pty)) {
          std::string pe = scalar_exprs.at(so.ptrs);
          std::string ve =
              msl_cast_expr(scalar_exprs.at(so.val), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          std::string guard = "(m1_tid_linear == 0u)";
          if (so.mask != kInvalidId) guard = "(" + guard + " && (" + scalar_exprs.at(so.mask) + "))";
          out << indent_str(ind) << "if (" << guard << ") { *(" << pe << ") = " << ve << "; }\n";
          continue;
        }
        auto find_scan_dep = [&](ValueId root) -> const ScanContigInfo* {
          std::vector<ValueId> wl{root};
          std::unordered_set<ValueId> seen{root};
          while (!wl.empty()) {
            ValueId cur = wl.back();
            wl.pop_back();
            if (auto sit = scan_contig_tiles.find(cur); sit != scan_contig_tiles.end()) {
              return &sit->second;
            }
            const Value& cv = kernel.value(cur);
            if (cv.def_op == kInvalidId) continue;
            const Op& cop = kernel.op(cv.def_op);
            for (ValueId o : cop.operands) {
              if (o != kInvalidId && seen.insert(o).second) wl.push_back(o);
            }
          }
          return nullptr;
        };
        if (const ScanContigInfo* sc = find_scan_dep(so.val)) {
          out << indent_str(ind) << "for (uint m1_k = 0u; m1_k < " << sc->seg_ept << "u; ++m1_k) {\n";
          out << indent_str(ind + 1) << "uint m1_pos = " << sc->base_var << " + m1_k;\n";
          out << indent_str(ind + 1) << "if (m1_pos < " << sc->len << "u) {\n";
          std::string phys_lane = sc->reverse ? ("(" + std::to_string(sc->len - 1) + "u - m1_pos)") : "m1_pos";
          out << indent_str(ind + 2) << "uint m1_plane = " << phys_lane << ";\n";
          std::unordered_map<ValueId, std::string> memo;
          std::string pe = eval_at(so.ptrs, "m1_plane", "m1_k", memo);
          std::string ve = msl_cast_expr(eval_at(so.val, "m1_plane", "m1_k", memo),
                                         scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_at(so.mask, "m1_plane", "m1_k", memo);
            out << indent_str(ind + 2) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
          } else {
            out << indent_str(ind + 2) << "*(" << pe << ") = " << ve << ";\n";
          }
          out << indent_str(ind + 1) << "}\n";
          out << indent_str(ind) << "}\n";
          continue;
        }
        int64_t len = pty.as_tile().shape[0];
        out << indent_str(ind) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
        out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
        out << indent_str(ind + 1) << "if (m1_lane < " << len << "u) {\n";
        std::unordered_map<ValueId, std::string> memo;
        std::string pe = eval_at(so.ptrs, "m1_lane", "m1_s", memo);
        std::string ve =
            msl_cast_expr(eval_at(so.val, "m1_lane", "m1_s", memo), scalar_dtype_of(kernel.value(so.val).type), pdt,
                          false);
        if (!err.empty()) return false;
        if (so.mask != kInvalidId) {
          std::string me = eval_at(so.mask, "m1_lane", "m1_s", memo);
          out << indent_str(ind + 2) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
        } else {
          out << indent_str(ind + 2) << "*(" << pe << ") = " << ve << ";\n";
        }
        out << indent_str(ind + 1) << "}\n";
        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.kind == OpKind::AtomicAdd && !op.operands.empty()) {
        const Type& pty = kernel.value(op.operands[0]).type;
        ScalarDType pdt = scalar_dtype_of(pty);
        ValueId rid = op.results.empty() ? kInvalidId : op.results[0];
        if (is_scalar_like(pty)) {
          std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
          if (!err.empty()) return false;
          std::string vn = (rid != kInvalidId) ? ssa_var(rid) : next_tmp("m1_atom");
          out << indent_str(ind) << msl_scalar_type(pdt) << " " << vn << " = " << msl_zero_literal(pdt, false) << ";\n";
          out << indent_str(ind) << "if (m1_tid_linear == 0u) { " << vn << " = " << expr << "; }\n";
          if (rid != kInvalidId) scalar_exprs[rid] = vn;
          continue;
        }
        std::string atomic_ty = (pdt == ScalarDType::F32) ? "atomic_float"
                                : (pdt == ScalarDType::I32) ? "atomic_int"
                                : (pdt == ScalarDType::U32) ? "atomic_uint"
                                                            : "";
        if (atomic_ty.empty()) {
          err = "Unsupported AtomicAdd dtype";
          return false;
        }
        int64_t len = pty.as_tile().shape[0];
        out << indent_str(ind) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
        out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
        out << indent_str(ind + 1) << "if (m1_lane < " << len << "u) {\n";
        std::unordered_map<ValueId, std::string> memo;
        std::string pe = eval_at(op.operands[0], "m1_lane", "m1_s", memo);
        std::string ve = msl_cast_expr(eval_at(op.operands[1], "m1_lane", "m1_s", memo),
                                       scalar_dtype_of(kernel.value(op.operands[1]).type), pdt, false);
        std::string call = "atomic_fetch_add_explicit(reinterpret_cast<device " + atomic_ty + "*>(" + pe + "), " + ve +
                           ", memory_order_relaxed)";
        if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) {
          std::string me = eval_at(op.operands[2], "m1_lane", "m1_s", memo);
          out << indent_str(ind + 2) << "if (" << me << ") { (void)" << call << "; }\n";
        } else {
          out << indent_str(ind + 2) << "(void)" << call << ";\n";
        }
        out << indent_str(ind + 1) << "}\n";
        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.results.empty()) continue;
      ValueId rid = op.results[0];
      const Type& rty = kernel.value(rid).type;

      if (is_scalar_like(rty)) {
        if (op.kind == OpKind::Constant) {
          scalar_exprs[rid] = msl_const_literal(std::get<ConstAttr>(op.attr), scalar_dtype_of(rty));
          continue;
        }
        if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims ||
            op.kind == OpKind::Reshape) {
          scalar_exprs[rid] = scalar_exprs.at(op.operands[0]);
          continue;
        }
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
        if (!err.empty()) return false;
        std::string vn = ssa_var(rid);
        if (is_ptr_type(rty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vn << " = " << expr
              << ";\n";
        } else {
          out << indent_str(ind) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vn << " = " << expr << ";\n";
        }
        scalar_exprs[rid] = vn;
        continue;
      }

      if (must_materialize.count(rid) > 0) {
        ScalarDType rdt = scalar_dtype_of(rty);
        int64_t len = rty.as_tile().shape[0];
        std::string vn = ssa_var(rid);
        if (ept == 1) {
          out << indent_str(ind) << msl_scalar_type(rdt) << " " << vn << " = " << msl_zero_literal(rdt, false) << ";\n";
        } else {
          out << indent_str(ind) << msl_scalar_type(rdt) << " " << vn << "[" << ept << "];\n";
        }
        out << indent_str(ind) << "for (uint m1_s = 0u; m1_s < " << ept << "u; ++m1_s) {\n";
        out << indent_str(ind + 1) << "uint m1_lane = m1_tid_linear + m1_s * " << tg_x << "u;\n";
        out << indent_str(ind + 1) << "if (m1_lane < " << len << "u) {\n";
        std::unordered_map<ValueId, std::string> memo;
        std::string ev = eval_at(rid, "m1_lane", "m1_s", memo);
        if (!err.empty()) return false;
        MatScalarTile mt{vn, rdt, false, ReduceKind::Sum};
        out << indent_str(ind + 2) << tile_slot(mt, "m1_s") << " = " << ev << ";\n";
        out << indent_str(ind + 1) << "}\n";
        out << indent_str(ind) << "}\n";
        mat_tiles[rid] = MatScalarTile{vn, rdt, false, ReduceKind::Sum};
      }
    }
    return true;
  }
};

CompiledMSL emit_1d_scalar_reduce_scan(const Kernel& kernel, const TargetSpec& target,
                                       const ReduceKernelAnalysis& analysis) {
  int64_t block_size = std::max<int64_t>(1, analysis.block_size_1d);
  uint32_t tg_x = std::clamp<uint32_t>(round_up_32(static_cast<uint32_t>(block_size)), 32u, 256u);
  if (tg_x > target.max_threads_per_threadgroup) {
    tg_x = (target.max_threads_per_threadgroup / 32u) * 32u;
  }
  uint32_t num_sg = tg_x / 32u;
  uint32_t ept = static_cast<uint32_t>((block_size + tg_x - 1) / tg_x);

  Scalar1DReduceEmitter em{
      .kernel = kernel,
      .analysis = analysis,
      .target = target,
      .block_size = block_size,
      .tg_x = tg_x,
      .num_sg = num_sg,
      .ept = ept,
  };
  em.smem.num_sg = num_sg;
  populate_entry_param_exprs(kernel, analysis, em.scalar_exprs);

  std::ostringstream body_ops;
  if (!em.emit_block(kernel.entry_block, false, 1, nullptr, body_ops)) {
    return CompiledMSL{false, "", {}, em.err};
  }

  auto [smem_decls, smem_bytes] = em.smem.emit_decls();
  if (!target.fits_threadgroup_memory(smem_bytes)) {
    return CompiledMSL{false, "", {}, "Threadgroup memory exceeds Apple M1 Max budget"};
  }

  std::ostringstream full_body;
  full_body << "  (void)m1_grid_dim; (void)m1_tid; (void)m1_tg_size;\n";
  if (num_sg <= 1u) full_body << "  (void)m1_simd_lane; (void)m1_simd_id;\n";
  full_body << smem_decls;
  full_body << body_ops.str();
  full_body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, em.helpers) + full_body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = smem_bytes;
  res.launch_spec.vec_width = 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

// ---------------------------------------------------------------------------
// 2D Row-Wise Hardware SIMD Reduction Emitter ([BM, BN] reduced on axis=1)
// ---------------------------------------------------------------------------

struct Row2DReduceEmitter {
  const Kernel& kernel;
  const ReduceKernelAnalysis& analysis;
  int64_t bm = 1;
  int64_t bn = 1;
  uint32_t tg_x = 32;
  uint32_t tg_y = 1;
  uint32_t rm = 1;

  HelperFlags helpers;
  std::string err;
  uint32_t tmp_seq = 0;

  std::unordered_map<ValueId, std::string> scalar_exprs;
  std::unordered_map<ValueId, std::string> row_vec_var;
  std::unordered_map<ValueId, std::pair<ValueId, std::string>> ptr_iter_map;

  struct Mat2DColAcc {
    std::string var_name;
    ScalarDType dtype = ScalarDType::F32;
    ReduceKind kind = ReduceKind::Sum;
  };
  std::unordered_map<ValueId, Mat2DColAcc> col_accs;

  std::string next_tmp(const std::string& pfx) { return pfx + "_" + std::to_string(tmp_seq++); }

  std::string eval_2d(ValueId vid, const std::string& r_expr, const std::string& c_expr, const std::string& rm_idx,
                      std::unordered_map<ValueId, std::string>& memo) {
    if (vid == kInvalidId) return "0";
    std::string mkey = std::to_string(vid);
    if (auto mit = memo.find(vid); mit != memo.end()) return mit->second;
    const Value& v = kernel.value(vid);
    if (is_scalar_like(v.type)) {
      if (auto sit = scalar_exprs.find(vid); sit != scalar_exprs.end()) return sit->second;
      if (v.def_op != kInvalidId && kernel.op(v.def_op).kind == OpKind::Constant) {
        return msl_const_literal(std::get<ConstAttr>(kernel.op(v.def_op).attr), scalar_dtype_of(v.type));
      }
      return ssa_var(vid);
    }
    if (auto pit = ptr_iter_map.find(vid); pit != ptr_iter_map.end()) {
      std::string bp = eval_2d(pit->second.first, r_expr, c_expr, rm_idx, memo);
      std::string e = "(" + bp + " + " + pit->second.second + ")";
      memo[vid] = e;
      return e;
    }
    if (auto cit = col_accs.find(vid); cit != col_accs.end()) {
      std::string e = (rm == 1) ? cit->second.var_name : (cit->second.var_name + "[" + rm_idx + "]");
      memo[vid] = e;
      return e;
    }
    if (auto rvit = row_vec_var.find(vid); rvit != row_vec_var.end()) {
      std::string e = (rm == 1) ? rvit->second : (rvit->second + "[" + rm_idx + "]");
      memo[vid] = e;
      return e;
    }
    if (v.def_op == kInvalidId) return ssa_var(vid);
    const Op& op = kernel.op(v.def_op);
    const auto& shape = v.type.as_tile().shape;
    ScalarDType rdt = scalar_dtype_of(v.type);
    std::string res;

    switch (op.kind) {
      case OpKind::Arange: {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        std::string idx_e = (shape.size() == 1 && shape[0] == bm && shape[0] != bn) ? r_expr : c_expr;
        res = (attr.start == 0)
                  ? (msl_scalar_type(rdt) + "(" + idx_e + ")")
                  : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) + "(" +
                     idx_e + "))");
        break;
      }
      case OpKind::ExpandDims: {
        uint32_t ax = std::get<AxisAttr>(op.attr).axis;
        std::string coord = (ax == 0) ? c_expr : r_expr;
        res = eval_2d_1d(op.operands[0], coord, rm_idx, memo);
        break;
      }
      case OpKind::Splat:
      case OpKind::Broadcast: {
        const Type& src_ty = kernel.value(op.operands[0]).type;
        if (!src_ty.is_tile()) {
          res = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        } else {
          const auto& ssh = src_ty.as_tile().shape;
          if (ssh.size() == 2) {
            std::string nr = (ssh[0] == 1) ? "0u" : r_expr;
            std::string nc = (ssh[1] == 1) ? "0u" : c_expr;
            res = eval_2d(op.operands[0], nr, nc, rm_idx, memo);
          } else {
            res = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
          }
        }
        break;
      }
      case OpKind::Cast: {
        std::string se = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        res = msl_cast_expr(se, scalar_dtype_of(kernel.value(op.operands[0]).type), rdt, false);
        break;
      }
      case OpKind::Unary: {
        std::string se = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        res = msl_unary_expr(std::get<UnaryAttr>(op.attr).op, se, rdt, false, helpers);
        break;
      }
      case OpKind::Binary: {
        std::string le = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        std::string re = eval_2d(op.operands[1], r_expr, c_expr, rm_idx, memo);
        res = msl_binary_expr(std::get<BinaryAttr>(op.attr).op, le, re, scalar_dtype_of(kernel.value(op.operands[0]).type),
                              scalar_dtype_of(kernel.value(op.operands[1]).type), rdt, false);
        break;
      }
      case OpKind::Select: {
        std::string ce = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        std::string te = eval_2d(op.operands[1], r_expr, c_expr, rm_idx, memo);
        std::string fe = eval_2d(op.operands[2], r_expr, c_expr, rm_idx, memo);
        res = msl_select_expr(ce, te, fe, scalar_dtype_of(kernel.value(op.operands[1]).type),
                              scalar_dtype_of(kernel.value(op.operands[2]).type), rdt, false);
        break;
      }
      case OpKind::PtrAdd: {
        std::string pe = eval_2d(op.operands[0], r_expr, c_expr, rm_idx, memo);
        std::string oe = eval_2d(op.operands[1], r_expr, c_expr, rm_idx, memo);
        res = "(" + pe + " + " + oe + ")";
        break;
      }
      case OpKind::Load: {
        auto lo = extract_load_operands(op, kernel);
        std::string pe = eval_2d(lo.ptrs, r_expr, c_expr, rm_idx, memo);
        if (lo.mask == kInvalidId) {
          res = "(*(" + pe + "))";
        } else {
          std::string me = eval_2d(lo.mask, r_expr, c_expr, rm_idx, memo);
          std::string oe = (lo.other != kInvalidId)
                               ? msl_cast_expr(eval_2d(lo.other, r_expr, c_expr, rm_idx, memo),
                                               scalar_dtype_of(kernel.value(lo.other).type), rdt, false)
                               : msl_zero_literal(rdt, false);
          res = "(" + me + " ? (*(" + pe + ")) : " + oe + ")";
        }
        break;
      }
      default:
        err = "Unsupported op in 2D row-wise reduction emitter";
        return "0";
    }
    memo[vid] = res;
    return res;
  }

  std::string eval_2d_1d(ValueId vid, const std::string& coord, const std::string& rm_idx,
                         std::unordered_map<ValueId, std::string>& memo) {
    if (auto rvit = row_vec_var.find(vid); rvit != row_vec_var.end()) {
      return (rm == 1) ? rvit->second : (rvit->second + "[" + rm_idx + "]");
    }
    const Value& v = kernel.value(vid);
    if (is_scalar_like(v.type)) return eval_2d(vid, coord, coord, rm_idx, memo);
    if (v.def_op == kInvalidId) return ssa_var(vid);
    const Op& op = kernel.op(v.def_op);
    ScalarDType rdt = scalar_dtype_of(v.type);
    if (op.kind == OpKind::Arange) {
      const auto& attr = std::get<ArangeAttr>(op.attr);
      return (attr.start == 0)
                 ? (msl_scalar_type(rdt) + "(" + coord + ")")
                 : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) + "(" +
                    coord + "))");
    }
    return eval_2d(vid, coord, coord, rm_idx, memo);
  }

  bool emit_entry(std::ostringstream& out) {
    const Block& entry = kernel.block(kernel.entry_block);
    for (OpId oid : entry.ops) {
      const Op& op = kernel.op(oid);
      if (op.erased || op.kind == OpKind::Return) continue;

      if (op.kind == OpKind::ForLoop) {
        if (op.operands.size() < 3 || op.regions.empty()) return false;
        const Block& body = kernel.block(op.regions[0]);
        if (body.args.empty() || block_has_reduce_or_scan(kernel, op.regions[0])) return false;
        const Op* yield_op = nullptr;
        for (OpId boid : body.ops) {
          const Op& bop = kernel.op(boid);
          if (!bop.erased && bop.kind == OpKind::Yield) {
            yield_op = &bop;
            break;
          }
        }
        if (!yield_op) return false;

        ValueId iv_id = body.args[0];
        ScalarDType iv_dt = scalar_dtype_of(kernel.value(iv_id).type);
        std::string iv_var = ssa_var(iv_id);
        std::string lo_e = msl_cast_expr(scalar_exprs.at(op.operands[0]),
                                         scalar_dtype_of(kernel.value(op.operands[0]).type), iv_dt, false);
        std::string hi_e = msl_cast_expr(scalar_exprs.at(op.operands[1]),
                                         scalar_dtype_of(kernel.value(op.operands[1]).type), iv_dt, false);
        std::string step_e = msl_cast_expr(scalar_exprs.at(op.operands[2]),
                                           scalar_dtype_of(kernel.value(op.operands[2]).type), iv_dt, false);
        bool neg_step = (analysis.const_i64.count(op.operands[2]) && analysis.const_i64.at(op.operands[2]) < 0);

        size_t num_iter = op.operands.size() - 3;
        std::vector<std::pair<ValueId, Mat2DColAcc>> acc_yields;
        std::vector<std::pair<std::string, ValueId>> ptr_yields;
        for (size_t k = 0; k < num_iter; ++k) {
          ValueId arg_id = body.args[1 + k];
          ValueId init_id = op.operands[3 + k];
          const Type& aty = kernel.value(arg_id).type;
          if (is_ptr_type(aty)) {
            ValueId dv = kInvalidId;
            if (!detect_ptr_iter_delta(kernel, arg_id, yield_op->operands[k], dv)) return false;
            std::string poff = "m1_poff_" + std::to_string(arg_id);
            out << "  long " << poff << " = 0ll;\n";
            ptr_iter_map[arg_id] = {init_id, poff};
            ptr_yields.emplace_back(poff, dv);
            continue;
          }
          auto coll = detect_collapsible_acc(kernel, analysis, op, k);
          if (!coll.has_value()) return false;
          std::string avar = ssa_var(arg_id);
          std::string id_lit = msl_reduce_identity(coll->kind, coll->dtype);
          if (rm == 1) {
            out << "  " << msl_scalar_type(coll->dtype) << " " << avar << " = " << id_lit << ";\n";
          } else {
            out << "  " << msl_scalar_type(coll->dtype) << " " << avar << "[" << rm << "];\n";
            out << "  for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) { " << avar << "[m1_rm] = " << id_lit
                << "; }\n";
          }
          Mat2DColAcc mc{avar, coll->dtype, coll->kind};
          col_accs[arg_id] = mc;
          if (k < op.results.size()) col_accs[op.results[k]] = mc;
          acc_yields.emplace_back(yield_op->operands[k], mc);
        }

        scalar_exprs[iv_id] = iv_var;
        out << "  for (" << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << "; " << iv_var
            << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var << " += " << step_e << ") {\n";
        out << "    for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) {\n";
        out << "      uint m1_row = m1_tid.y + m1_rm * " << tg_y << "u;\n";
        out << "      if (m1_row < " << bm << "u) {\n";
        out << "        for (uint m1_col = m1_tid.x; m1_col < " << bn << "u; m1_col += 32u) {\n";
        std::unordered_map<ValueId, std::string> memo;
        for (const auto& [yv, mc] : acc_yields) {
          std::string ev = msl_cast_expr(eval_2d(yv, "m1_row", "m1_col", "m1_rm", memo),
                                         scalar_dtype_of(kernel.value(yv).type), mc.dtype, false);
          if (!err.empty()) return false;
          std::string dst = (rm == 1) ? mc.var_name : (mc.var_name + "[m1_rm]");
          out << "          " << dst << " = " << ev << ";\n";
        }
        out << "        }\n";
        out << "      }\n";
        out << "    }\n";
        for (const auto& [poff, dv] : ptr_yields) {
          out << "    " << poff << " += long(" << scalar_exprs.at(dv) << ");\n";
        }
        out << "  }\n";
        continue;
      }

      if (is_reduce_op_kind(op.kind)) {
        const auto& ra = std::get<ReduceAttr>(op.attr);
        if (ra.axis != 1) return false;
        ValueId in_v = op.operands[0];
        ScalarDType dt = scalar_dtype_of(kernel.value(in_v).type);
        std::string ty = msl_scalar_type(dt);
        bool is_arg = (ra.kind == ReduceKind::Argmax || ra.kind == ReduceKind::Argmin ||
                       ra.result_mode == ReduceResultMode::ValueAndIndex ||
                       ra.result_mode == ReduceResultMode::IndexOnly);
        bool is_max = (ra.kind == ReduceKind::Max || ra.kind == ReduceKind::Argmax);

        std::string r_val_var;
        std::string r_idx_var;
        if (!is_arg) {
          r_val_var = ssa_var(op.results[0]);
          row_vec_var[op.results[0]] = r_val_var;
          if (rm == 1) {
            out << "  " << ty << " " << r_val_var << " = " << msl_reduce_identity(ra.kind, dt) << ";\n";
          } else {
            out << "  " << ty << " " << r_val_var << "[" << rm << "];\n";
          }
        } else if (ra.result_mode == ReduceResultMode::ValueAndIndex && op.results.size() >= 2) {
          r_val_var = ssa_var(op.results[0]);
          r_idx_var = ssa_var(op.results[1]);
          row_vec_var[op.results[0]] = r_val_var;
          row_vec_var[op.results[1]] = r_idx_var;
          if (rm == 1) {
            out << "  " << ty << " " << r_val_var << " = " << msl_reduce_identity(ra.kind, dt) << ";\n";
            out << "  int " << r_idx_var << " = 0;\n";
          } else {
            out << "  " << ty << " " << r_val_var << "[" << rm << "];\n";
            out << "  int " << r_idx_var << "[" << rm << "];\n";
          }
        } else {
          r_idx_var = ssa_var(op.results[0]);
          row_vec_var[op.results[0]] = r_idx_var;
          if (rm == 1) {
            out << "  int " << r_idx_var << " = 0;\n";
          } else {
            out << "  int " << r_idx_var << "[" << rm << "];\n";
          }
        }

        out << "  for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) {\n";
        out << "    uint m1_row = m1_tid.y + m1_rm * " << tg_y << "u;\n";
        out << "    " << ty << " m1_rv = " << msl_reduce_identity(ra.kind, dt) << ";\n";
        if (is_arg) out << "    int m1_ri = 2147483647;\n";
        out << "    if (m1_row < " << bm << "u) {\n";
        if (auto cit = col_accs.find(in_v); cit != col_accs.end()) {
          std::string src = (rm == 1) ? cit->second.var_name : (cit->second.var_name + "[m1_rm]");
          out << "      m1_rv = " << src << ";\n";
        } else {
          out << "      for (uint m1_col = m1_tid.x; m1_col < " << bn << "u; m1_col += 32u) {\n";
          std::unordered_map<ValueId, std::string> memo;
          std::string ev = eval_2d(in_v, "m1_row", "m1_col", "m1_rm", memo);
          if (!err.empty()) return false;
          if (!is_arg) {
            out << "        m1_rv = " << scalar_combine_expr(ra.kind, "m1_rv", ev) << ";\n";
          } else {
            const std::string cmp = is_max ? " > " : " < ";
            out << "        " << ty << " m1_lv = " << ev << ";\n";
            out << "        int m1_li = int(m1_col);\n";
            out << "        bool m1_bt = (m1_lv" << cmp << "m1_rv) || ((m1_lv == m1_rv) && (m1_li < m1_ri));\n";
            out << "        m1_rv = m1_bt ? m1_lv : m1_rv;\n";
            out << "        m1_ri = m1_bt ? m1_li : m1_ri;\n";
          }
          out << "      }\n";
        }
        out << "    }\n";
        if (!is_arg) {
          if (dt == ScalarDType::I64) {
            emit_simd_reduce_i64_butterfly(ra.kind, "m1_rv", 16u, 2, out);
          } else {
            out << "    m1_rv = " << simd_op_name(ra.kind) << "(m1_rv);\n";
          }
          std::string dst = (rm == 1) ? r_val_var : (r_val_var + "[m1_rm]");
          out << "    " << dst << " = m1_rv;\n";
        } else {
          emit_simd_arg_butterfly(is_max, dt, "m1_rv", "m1_ri", 16u, 2, out);
          if (!r_val_var.empty()) {
            std::string dstv = (rm == 1) ? r_val_var : (r_val_var + "[m1_rm]");
            out << "    " << dstv << " = m1_rv;\n";
          }
          if (!r_idx_var.empty()) {
            std::string dsti = (rm == 1) ? r_idx_var : (r_idx_var + "[m1_rm]");
            out << "    " << dsti << " = m1_ri;\n";
          }
        }
        out << "  }\n";
        continue;
      }

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        const Type& pty = kernel.value(so.ptrs).type;
        ScalarDType pdt = scalar_dtype_of(pty);
        if (pty.is_tile() && pty.as_tile().shape.size() == 1 && pty.as_tile().shape[0] == bm) {
          out << "  if (m1_tid.x == 0u) {\n";
          out << "    for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) {\n";
          out << "      uint m1_row = m1_tid.y + m1_rm * " << tg_y << "u;\n";
          out << "      if (m1_row < " << bm << "u) {\n";
          std::unordered_map<ValueId, std::string> memo;
          std::string pe = eval_2d_1d(so.ptrs, "m1_row", "m1_rm", memo);
          std::string ve = msl_cast_expr(eval_2d_1d(so.val, "m1_row", "m1_rm", memo),
                                         scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_2d_1d(so.mask, "m1_row", "m1_rm", memo);
            out << "        if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
          } else {
            out << "        *(" << pe << ") = " << ve << ";\n";
          }
          out << "      }\n";
          out << "    }\n";
          out << "  }\n";
          continue;
        }
        if (pty.is_tile() && pty.as_tile().shape.size() == 2 && pty.as_tile().shape[0] == bm &&
            pty.as_tile().shape[1] == bn) {
          out << "  for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) {\n";
          out << "    uint m1_row = m1_tid.y + m1_rm * " << tg_y << "u;\n";
          out << "    if (m1_row < " << bm << "u) {\n";
          out << "      for (uint m1_col = m1_tid.x; m1_col < " << bn << "u; m1_col += 32u) {\n";
          std::unordered_map<ValueId, std::string> memo;
          std::string pe = eval_2d(so.ptrs, "m1_row", "m1_col", "m1_rm", memo);
          std::string ve = msl_cast_expr(eval_2d(so.val, "m1_row", "m1_col", "m1_rm", memo),
                                         scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_2d(so.mask, "m1_row", "m1_col", "m1_rm", memo);
            out << "        if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
          } else {
            out << "        *(" << pe << ") = " << ve << ";\n";
          }
          out << "      }\n";
          out << "    }\n";
          out << "  }\n";
          continue;
        }
        return false;
      }

      if (!op.results.empty() && is_scalar_like(kernel.value(op.results[0]).type)) {
        ValueId rid = op.results[0];
        const Type& rty = kernel.value(rid).type;
        if (op.kind == OpKind::Constant) {
          scalar_exprs[rid] = msl_const_literal(std::get<ConstAttr>(op.attr), scalar_dtype_of(rty));
          continue;
        }
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
        if (!err.empty()) return false;
        std::string vn = ssa_var(rid);
        if (is_ptr_type(rty)) {
          out << "  " << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vn << " = " << expr << ";\n";
        } else {
          out << "  " << msl_scalar_type(scalar_dtype_of(rty)) << " " << vn << " = " << expr << ";\n";
        }
        scalar_exprs[rid] = vn;
      }
    }
    return true;
  }
};

std::optional<CompiledMSL> try_emit_2d_row_reduce(const Kernel& kernel, const TargetSpec& target,
                                                  const ReduceKernelAnalysis& analysis) {
  if (analysis.max_rank != 2 || analysis.has_scan) return std::nullopt;
  int64_t bm = analysis.primary_2d_shape.first;
  int64_t bn = analysis.primary_2d_shape.second;
  if (bm <= 0 || bn <= 0) return std::nullopt;
  for (OpId roid : analysis.reduce_ops) {
    const Op& rop = kernel.op(roid);
    if (!std::holds_alternative<ReduceAttr>(rop.attr)) return std::nullopt;
    const auto& ra = std::get<ReduceAttr>(rop.attr);
    if (ra.axis != 1) return std::nullopt;
    const Type& ity = kernel.value(rop.operands[0]).type;
    if (!ity.is_tile() || ity.as_tile().shape.size() != 2 || ity.as_tile().shape[0] != bm ||
        ity.as_tile().shape[1] != bn) {
      return std::nullopt;
    }
  }

  uint32_t tg_x = 32u;
  uint32_t max_rows = std::min<uint32_t>(8u, target.max_threads_per_threadgroup / 32u);
  uint32_t tg_y = std::clamp<uint32_t>(static_cast<uint32_t>(bm), 1u, std::max<uint32_t>(1u, max_rows));
  uint32_t rm = static_cast<uint32_t>((bm + tg_y - 1) / tg_y);
  if (rm > 16u) return std::nullopt;

  Row2DReduceEmitter em{
      .kernel = kernel,
      .analysis = analysis,
      .bm = bm,
      .bn = bn,
      .tg_x = tg_x,
      .tg_y = tg_y,
      .rm = rm,
  };
  populate_entry_param_exprs(kernel, analysis, em.scalar_exprs);

  std::ostringstream body_ops;
  if (!em.emit_entry(body_ops)) return std::nullopt;

  std::ostringstream full_body;
  full_body << "  (void)m1_grid_dim; (void)m1_tg_size; (void)m1_tid_linear; (void)m1_simd_lane; (void)m1_simd_id;\n";
  full_body << body_ops.str();
  full_body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, em.helpers) + full_body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = tg_y;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

}  // namespace

CompiledMSL try_emit_reduce_scan_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  if (matched_out != nullptr) *matched_out = false;

  ReduceKernelAnalysis analysis;
  std::string err;
  if (!analyze_reduce_kernel(kernel, analysis, err)) {
    return CompiledMSL{false, "", {}, err};
  }

  if (!analysis.has_reduce && !analysis.has_scan) {
    return CompiledMSL{false, "", {}, ""};
  }

  if (analysis.has_dot || analysis.max_rank == 0 || analysis.max_rank > 2) {
    return CompiledMSL{false, "", {}, ""};
  }

  if (analysis.max_rank == 2) {
    if (auto res2d = try_emit_2d_row_reduce(kernel, target, analysis)) {
      if (matched_out != nullptr) *matched_out = true;
      return *res2d;
    }
    return CompiledMSL{false, "", {}, ""};
  }

  if (matched_out != nullptr) *matched_out = true;

  if (!analysis.has_scan) {
    if (auto vec4_res = try_emit_1d_reduce_vec4(kernel, target, analysis)) {
      return *vec4_res;
    }
  }

  return emit_1d_scalar_reduce_scan(kernel, target, analysis);
}

}  // namespace m1triton::codegen::metal

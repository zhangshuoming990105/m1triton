#include "codegen/metal/metal_codegen.hpp"

#include "codegen/metal/norm_softmax_emitter.hpp"
#include "codegen/metal/pool_emitter.hpp"
#include "codegen/metal/reduce_scan_emitter.hpp"
#include "codegen/metal/row_scan_emitter.hpp"
#include "codegen/metal/sgmm_emitter.hpp"

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
using ir::DotAttr;
using ir::Kernel;
using ir::KernelParam;
using ir::LayoutAttr;
using ir::LoadStoreAttr;
using ir::Op;
using ir::OpId;
using ir::OpKind;
using ir::PermuteAttr;
using ir::PtrType;
using ir::ScalarDType;
using ir::TargetSpec;
using ir::TileType;
using ir::Type;
using ir::UnaryAttr;
using ir::UnaryOp;
using ir::Value;
using ir::ValueId;
using ir::kInvalidId;

uint32_t local_dtype_bytes(ScalarDType dt) {
  return ir::dtype_bytes(dt);
}

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

std::string msl_zero_literal(ScalarDType dt, bool vec4) {
  std::string scalar_z;
  switch (dt) {
    case ScalarDType::F32:
      scalar_z = "0.0f";
      break;
    case ScalarDType::F16:
      scalar_z = "half(0.0f)";
      break;
    case ScalarDType::I32:
      scalar_z = "0";
      break;
    case ScalarDType::U32:
      scalar_z = "0u";
      break;
    case ScalarDType::I64:
      scalar_z = "0ll";
      break;
    case ScalarDType::Bool:
      scalar_z = "false";
      break;
  }
  if (!vec4) return scalar_z;
  return msl_vec4_type(dt) + "(" + scalar_z + ")";
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
  // MSL select(a, b, c) returns b when c is true, a when c is false.
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

struct DotOperands {
  ValueId lhs = kInvalidId;
  ValueId rhs = kInvalidId;
  ValueId acc = kInvalidId;
  DotAttr attr;
};

DotOperands extract_dot_operands(const Op& op) {
  DotOperands out;
  if (std::holds_alternative<DotAttr>(op.attr)) {
    out.attr = std::get<DotAttr>(op.attr);
  }
  if (op.operands.size() >= 1) out.lhs = op.operands[0];
  if (op.operands.size() >= 2) out.rhs = op.operands[1];
  if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) out.acc = op.operands[2];
  return out;
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

uint32_t round_up_32(uint32_t v) {
  if (v == 0) return 32;
  return ((v + 31u) / 32u) * 32u;
}

struct KernelAnalysis {
  std::vector<ParamSlot> param_slots;
  std::unordered_map<uint32_t, std::string> param_msl_name;
  std::unordered_map<ValueId, uint32_t> val_to_param_idx;
  std::unordered_map<ValueId, std::set<uint32_t>> ptr_origins;
  std::unordered_set<uint32_t> written_params;
  std::unordered_map<ValueId, int64_t> const_i64;
  std::unordered_set<ValueId> div4_values;
  std::unordered_set<ValueId> nonneg_values;

  size_t max_rank = 0;
  bool has_dot = false;
  bool has_atomic = false;
  bool has_reduce = false;
  bool has_scan = false;
  bool has_norm_softmax = false;
  bool has_pool = false;
  bool has_row_scan = false;
  bool has_tile_cf = false;
  bool has_tile_reshape_or_permute = false;
  int64_t block_size_1d = 0;
  std::pair<int64_t, int64_t> primary_2d_shape = {0, 0};
  std::optional<LayoutAttr> hint_layout;
  uint32_t kernel_vec_width_hint = 1;
};

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

bool block_has_tile_op(const Kernel& kernel, BlockId bid) {
  if (bid == kInvalidId || bid >= kernel.blocks.size()) return false;
  const Block& blk = kernel.block(bid);
  for (ValueId arg : blk.args) {
    if (kernel.value(arg).type.is_tile()) return true;
  }
  for (OpId oid : blk.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    for (ValueId r : op.results) {
      if (kernel.value(r).type.is_tile()) return true;
    }
    for (ValueId o : op.operands) {
      if (o != kInvalidId && kernel.value(o).type.is_tile()) return true;
    }
    for (BlockId rb : op.regions) {
      if (block_has_tile_op(kernel, rb)) return true;
    }
  }
  return false;
}

bool analyze_kernel(const Kernel& kernel, KernelAnalysis& out, std::string& err) {
  if (kernel.name.empty()) {
    err = "Kernel name must not be empty";
    return false;
  }
  if (kernel.entry_block == kInvalidId || kernel.entry_block >= kernel.blocks.size()) {
    err = "Kernel has invalid entry block";
    return false;
  }

  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const auto& p = kernel.params[i];
    if (p.type.is_tile()) {
      err = "Kernel parameter '" + p.name + "' cannot have tile type";
      return false;
    }
  }

  const Block& entry = kernel.block(kernel.entry_block);
  std::vector<uint32_t> non_constexpr_indices;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    if (!kernel.params[i].is_constexpr) {
      non_constexpr_indices.push_back(i);
    }
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
    } else {
      if (kp.divisibility >= 4 && (kp.divisibility % 4 == 0)) {
        out.div4_values.insert(vid);
      }
    }
  }

  std::vector<BlockId> all_blocks;
  std::unordered_set<BlockId> seen_blocks;
  collect_blocks_dfs(kernel, kernel.entry_block, all_blocks, seen_blocks);

  // Propagate constants, divisibility, non-negativity, and pointer origins to fixed point.
  bool changed = true;
  while (changed) {
    changed = false;
    for (BlockId bid : all_blocks) {
      const Block& blk = kernel.block(bid);
      for (OpId oid : blk.ops) {
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
            if (!op.results.empty() && out.nonneg_values.insert(op.results[0]).second) {
              changed = true;
            }
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
              } else if (bop == BinaryOp::Add || bop == BinaryOp::Sub) {
                if (out.div4_values.count(a) && out.div4_values.count(b) && out.div4_values.insert(dst).second) {
                  changed = true;
                }
                if (bop == BinaryOp::Add && out.nonneg_values.count(a) && out.nonneg_values.count(b) &&
                    out.nonneg_values.insert(dst).second) {
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
              auto it = out.ptr_origins.find(op.operands[0]);
              if (it != out.ptr_origins.end()) {
                merge_origins(op.results[0], it->second);
              }
            }
            break;
          }
          case OpKind::Select: {
            if (!op.results.empty() && op.operands.size() >= 3) {
              for (size_t idx : {size_t{1}, size_t{2}}) {
                auto it = out.ptr_origins.find(op.operands[idx]);
                if (it != out.ptr_origins.end()) {
                  merge_origins(op.results[0], it->second);
                }
              }
            }
            break;
          }
          case OpKind::ForLoop: {
            if (!op.regions.empty()) {
              const Block& body = kernel.block(op.regions[0]);
              size_t num_iter = op.operands.size() > 3 ? op.operands.size() - 3 : 0;
              for (size_t i = 0; i < num_iter; ++i) {
                ValueId init_v = op.operands[3 + i];
                auto it = out.ptr_origins.find(init_v);
                if (it != out.ptr_origins.end()) {
                  if (1 + i < body.args.size()) merge_origins(body.args[1 + i], it->second);
                  if (i < op.results.size()) merge_origins(op.results[i], it->second);
                }
              }
              for (OpId boid : body.ops) {
                const Op& bop = kernel.op(boid);
                if (!bop.erased && bop.kind == OpKind::Yield) {
                  for (size_t i = 0; i < bop.operands.size(); ++i) {
                    auto it = out.ptr_origins.find(bop.operands[i]);
                    if (it != out.ptr_origins.end()) {
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
              const Block& rblk = kernel.block(rb);
              for (OpId roid : rblk.ops) {
                const Op& rop = kernel.op(roid);
                if (!rop.erased && rop.kind == OpKind::Yield) {
                  for (size_t i = 0; i < rop.operands.size() && i < op.results.size(); ++i) {
                    auto it = out.ptr_origins.find(rop.operands[i]);
                    if (it != out.ptr_origins.end()) {
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

  for (BlockId bid : all_blocks) {
    const Block& blk = kernel.block(bid);
    for (OpId oid : blk.ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;

      if (op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) {
        if (!op.operands.empty()) {
          auto it = out.ptr_origins.find(op.operands[0]);
          if (it != out.ptr_origins.end()) {
            out.written_params.insert(it->second.begin(), it->second.end());
          }
        }
      }
      if (op.kind == OpKind::Dot) out.has_dot = true;
      if (op.kind == OpKind::AtomicAdd) out.has_atomic = true;
      if (op.kind == OpKind::Reduce || op.kind == OpKind::SimdReduce) out.has_reduce = true;
      if (op.kind == OpKind::Scan) out.has_scan = true;
      if (op.kind == OpKind::OnlineSoftmax || op.kind == OpKind::WelfordReduce) {
        out.has_norm_softmax = true;
        if (op.operands.size() >= 2) {
          auto it = out.ptr_origins.find(op.operands[1]);
          if (it != out.ptr_origins.end()) out.written_params.insert(it->second.begin(), it->second.end());
        }
      }
      if (op.kind == OpKind::Pool) {
        out.has_pool = true;
        if (op.operands.size() >= 2) {
          auto it = out.ptr_origins.find(op.operands[1]);
          if (it != out.ptr_origins.end()) out.written_params.insert(it->second.begin(), it->second.end());
        }
      }
      if (op.kind == OpKind::SimdScan) {
        out.has_row_scan = true;
        if (op.operands.size() >= 2) {
          auto it = out.ptr_origins.find(op.operands[1]);
          if (it != out.ptr_origins.end()) out.written_params.insert(it->second.begin(), it->second.end());
        }
      }
      if ((op.kind == OpKind::ForLoop || op.kind == OpKind::If)) {
        for (BlockId rb : op.regions) {
          if (block_has_tile_op(kernel, rb)) out.has_tile_cf = true;
        }
      }
      if ((op.kind == OpKind::Reshape || op.kind == OpKind::Permute) && !op.results.empty() &&
          kernel.value(op.results[0]).type.is_tile()) {
        out.has_tile_reshape_or_permute = true;
      }

      for (ValueId rid : op.results) {
        const Type& ty = kernel.value(rid).type;
        if (ty.is_tile()) {
          const auto& tt = ty.as_tile();
          out.max_rank = std::max(out.max_rank, tt.shape.size());
          if (tt.shape.size() == 1 && tt.shape[0] > out.block_size_1d) {
            out.block_size_1d = tt.shape[0];
          }
          if (tt.shape.size() == 2) {
            if (op.kind == OpKind::Dot) {
              out.primary_2d_shape = {tt.shape[0], tt.shape[1]};
            } else if (!out.has_dot && out.primary_2d_shape.first == 0 && tt.shape[0] > 1 && tt.shape[1] > 1) {
              out.primary_2d_shape = {tt.shape[0], tt.shape[1]};
            }
          }
          if (!out.hint_layout.has_value() && tt.layout.kind != ir::LayoutKind::Unassigned) {
            out.hint_layout = tt.layout;
          }
        }
      }
      if (op.kind == OpKind::Store && op.operands.size() >= 2) {
        const Type& val_ty = kernel.value(op.operands[1]).type;
        if (val_ty.is_tile() && val_ty.as_tile().shape.size() == 2 && !out.has_dot) {
          out.primary_2d_shape = {val_ty.as_tile().shape[0], val_ty.as_tile().shape[1]};
        }
      }
    }
  }

  if (out.max_rank == 2 && out.primary_2d_shape.first == 0) {
    for (const auto& vptr : kernel.values) {
      if (vptr && vptr->type.is_tile() && vptr->type.as_tile().shape.size() == 2) {
        out.primary_2d_shape = {vptr->type.as_tile().shape[0], vptr->type.as_tile().shape[1]};
        break;
      }
    }
  }

  if (kernel.hints.vec_width.has_value()) {
    out.kernel_vec_width_hint = kernel.hints.vec_width->width;
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

bool ptr_value_is_writable(const KernelAnalysis& analysis, ValueId vid) {
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

std::string msl_ptr_type_str(const KernelAnalysis& analysis, ValueId vid, ScalarDType pointee) {
  bool writable = ptr_value_is_writable(analysis, vid);
  return std::string("device ") + (writable ? "" : "const ") + msl_scalar_type(pointee) + "*";
}

std::string build_msl_preamble_and_sig(const Kernel& kernel, const KernelAnalysis& analysis,
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
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]]) {\n";
  return os.str();
}

uint32_t effective_op_vec_width(const Op& op, const Kernel& kernel, const KernelAnalysis& analysis) {
  if (std::holds_alternative<LoadStoreAttr>(op.attr)) {
    uint32_t w = std::get<LoadStoreAttr>(op.attr).vec_width;
    if (w > 1) return w;
  }
  if (op.hints.vec_width.has_value() && op.hints.vec_width->width > 1) {
    return op.hints.vec_width->width;
  }
  if (!op.results.empty() && kernel.value(op.results[0]).type.is_tile()) {
    uint32_t w = kernel.value(op.results[0]).type.as_tile().layout.vec_width;
    if (w > 1) return w;
  }
  if (!op.operands.empty() && kernel.value(op.operands[0]).type.is_tile()) {
    uint32_t w = kernel.value(op.operands[0]).type.as_tile().layout.vec_width;
    if (w > 1) return w;
  }
  if (op.kind == OpKind::Store && op.operands.size() >= 2 && kernel.value(op.operands[1]).type.is_tile()) {
    uint32_t w = kernel.value(op.operands[1]).type.as_tile().layout.vec_width;
    if (w > 1) return w;
  }
  return analysis.kernel_vec_width_hint;
}

std::string emit_scalar_op_expr(const Kernel& kernel, const KernelAnalysis& analysis, const Op& op,
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
    case OpKind::PtrAdd: {
      return "(" + get_e(op.operands[0]) + " + " + get_e(op.operands[1]) + ")";
    }
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
      err = "Unsupported scalar op kind in MetalCodegen";
      return "";
  }
}

void populate_entry_param_exprs(const Kernel& kernel, const KernelAnalysis& analysis,
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

// ---------------------------------------------------------------------------
// 1D Vectorized Dual-Path Emitter (vec_width == 4)
// ---------------------------------------------------------------------------

struct LinearInt {
  bool valid = false;
  int64_t stride = 0;  // 0 = scalar/uniform, 1 = unit-stride ramp
  std::string base_expr;
  ScalarDType dtype = ScalarDType::I32;
  bool base_is_zero = false;
  bool base_div4 = false;
  bool base_nonneg = false;
};

struct LinearPtr {
  bool valid = false;
  int64_t stride = 0;
  std::string base_ptr;
  std::string base_offset;
  ScalarDType offset_dtype = ScalarDType::I32;
  bool offset_is_zero = false;
  bool offset_div4 = false;
  bool offset_nonneg = false;
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

bool try_build_all4_guard(const Kernel& kernel, const std::unordered_map<ValueId, LinearInt>& lin_ints,
                          const std::unordered_map<ValueId, std::string>& scalar_exprs, ValueId mask_id,
                          const std::string& elem_base_var,
                          std::unordered_map<std::string, std::string>& base_idx4_vars,
                          std::vector<std::tuple<std::string, ScalarDType, std::string>>& idx4_decls,
                          std::string& guard_out) {
  if (mask_id == kInvalidId) {
    guard_out = "true";
    return true;
  }
  const Value& mv = kernel.value(mask_id);
  if (!mv.type.is_tile()) {
    auto it = scalar_exprs.find(mask_id);
    if (it == scalar_exprs.end()) return false;
    guard_out = "(" + it->second + ")";
    return true;
  }
  if (mv.def_op == kInvalidId) return false;
  const Op& op = kernel.op(mv.def_op);
  if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast) {
    return try_build_all4_guard(kernel, lin_ints, scalar_exprs, op.operands[0], elem_base_var, base_idx4_vars,
                                idx4_decls, guard_out);
  }
  if (op.kind == OpKind::Binary && std::holds_alternative<BinaryAttr>(op.attr)) {
    BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
    if (bop == BinaryOp::And) {
      std::string g1, g2;
      if (!try_build_all4_guard(kernel, lin_ints, scalar_exprs, op.operands[0], elem_base_var, base_idx4_vars,
                                idx4_decls, g1)) {
        return false;
      }
      if (!try_build_all4_guard(kernel, lin_ints, scalar_exprs, op.operands[1], elem_base_var, base_idx4_vars,
                                idx4_decls, g2)) {
        return false;
      }
      guard_out = "(" + g1 + " && " + g2 + ")";
      return true;
    }

    auto it_l = lin_ints.find(op.operands[0]);
    auto it_r = lin_ints.find(op.operands[1]);
    if (it_l == lin_ints.end() || it_r == lin_ints.end() || !it_l->second.valid || !it_r->second.valid) {
      return false;
    }
    const LinearInt& L = it_l->second;
    const LinearInt& R = it_r->second;

    auto get_idx4_var = [&](const LinearInt& lin) -> std::string {
      std::string key = msl_scalar_type(lin.dtype) + "|" + (lin.base_is_zero ? "0" : lin.base_expr);
      auto it = base_idx4_vars.find(key);
      if (it != base_idx4_vars.end()) return it->second;
      std::string vname = base_idx4_vars.empty() ? "m1_idx4" : ("m1_idx4_" + std::to_string(base_idx4_vars.size()));
      std::string rhs = lin.base_is_zero
                            ? (msl_scalar_type(lin.dtype) + "(" + elem_base_var + ")")
                            : ("(" + lin.base_expr + " + " + msl_scalar_type(lin.dtype) + "(" + elem_base_var + "))");
      base_idx4_vars[key] = vname;
      idx4_decls.emplace_back(vname, lin.dtype, rhs);
      return vname;
    };

    // Upper-bound checks: lin < bound, lin <= bound, bound > lin, bound >= lin.
    auto make_upper = [&](const LinearInt& lin, const LinearInt& bnd, bool strict) -> std::string {
      std::string idx4 = get_idx4_var(lin);
      ScalarDType cmp_dt = common_cmp_dtype(lin.dtype, bnd.dtype);
      std::string idx_c = msl_cast_expr(idx4, lin.dtype, cmp_dt, false);
      std::string bnd_c = msl_cast_expr(bnd.base_expr, bnd.dtype, cmp_dt, false);
      std::string three = format_scalar_literal_from_i64(3, cmp_dt);
      std::string op_s = strict ? " < " : " <= ";
      return "(" + idx_c + " + " + three + op_s + bnd_c + ")";
    };

    // Lower-bound checks: lin >= lo, lin > lo, lo <= lin, lo < lin.
    auto make_lower = [&](const LinearInt& lin, const LinearInt& lo, bool strict) -> std::string {
      std::string idx4 = get_idx4_var(lin);
      ScalarDType cmp_dt = common_cmp_dtype(lin.dtype, lo.dtype);
      std::string idx_c = msl_cast_expr(idx4, lin.dtype, cmp_dt, false);
      std::string lo_c = msl_cast_expr(lo.base_expr, lo.dtype, cmp_dt, false);
      std::string op_s = strict ? " > " : " >= ";
      return "(" + idx_c + op_s + lo_c + ")";
    };

    if (L.stride == 1 && R.stride == 0) {
      if (bop == BinaryOp::CmpLt) {
        guard_out = make_upper(L, R, true);
        return true;
      }
      if (bop == BinaryOp::CmpLe) {
        guard_out = make_upper(L, R, false);
        return true;
      }
      if (bop == BinaryOp::CmpGe) {
        guard_out = make_lower(L, R, false);
        return true;
      }
      if (bop == BinaryOp::CmpGt) {
        guard_out = make_lower(L, R, true);
        return true;
      }
    } else if (L.stride == 0 && R.stride == 1) {
      if (bop == BinaryOp::CmpGt) {
        guard_out = make_upper(R, L, true);
        return true;
      }
      if (bop == BinaryOp::CmpGe) {
        guard_out = make_upper(R, L, false);
        return true;
      }
      if (bop == BinaryOp::CmpLe) {
        guard_out = make_lower(R, L, false);
        return true;
      }
      if (bop == BinaryOp::CmpLt) {
        guard_out = make_lower(R, L, true);
        return true;
      }
    } else if (L.stride == 0 && R.stride == 0) {
      guard_out = msl_binary_expr(bop, L.base_expr, R.base_expr, L.dtype, R.dtype, ScalarDType::Bool, false);
      return true;
    }
  }
  return false;
}

std::optional<CompiledMSL> try_emit_1d_vec4(const Kernel& kernel, const TargetSpec& target,
                                            const KernelAnalysis& analysis) {
  if (analysis.max_rank != 1 || analysis.has_dot || analysis.has_atomic || analysis.has_reduce ||
      analysis.has_tile_cf || analysis.has_tile_reshape_or_permute) {
    return std::nullopt;
  }
  if (analysis.block_size_1d < 4 || (analysis.block_size_1d % 4) != 0) {
    return std::nullopt;
  }

  const Block& entry = kernel.block(kernel.entry_block);

  auto ptr_origins_aligned = [&](ValueId ptr_val, ScalarDType pointee) -> bool {
    const uint32_t req_bytes = local_dtype_bytes(pointee) * 4u;
    auto it = analysis.ptr_origins.find(ptr_val);
    if (it == analysis.ptr_origins.end() || it->second.empty()) return false;
    for (uint32_t pidx : it->second) {
      const uint32_t div = kernel.params[pidx].divisibility;
      if (div < req_bytes || (div % req_bytes) != 0) return false;
    }
    return true;
  };

  // Verify every tile load/store requests vec_width == 4 and a supported 4-wide pointee dtype.
  size_t tile_mem_count = 0;
  for (OpId oid : entry.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    if (op.kind == OpKind::ForLoop || op.kind == OpKind::If || op.kind == OpKind::Barrier) {
      return std::nullopt;
    }
    if (op.kind == OpKind::Load && !op.results.empty() && kernel.value(op.results[0]).type.is_tile()) {
      ++tile_mem_count;
      if (effective_op_vec_width(op, kernel, analysis) != 4) return std::nullopt;
      const ScalarDType dt = scalar_dtype_of(kernel.value(op.results[0]).type);
      if (!can_vec4_mem(dt)) return std::nullopt;
      if (!ptr_origins_aligned(op.operands[0], dt)) return std::nullopt;
      if (kernel.value(op.results[0]).type.as_tile().shape[0] != analysis.block_size_1d) return std::nullopt;
    } else if (op.kind == OpKind::Store && op.operands.size() >= 2 && kernel.value(op.operands[1]).type.is_tile()) {
      ++tile_mem_count;
      if (effective_op_vec_width(op, kernel, analysis) != 4) return std::nullopt;
      const ScalarDType dt = scalar_dtype_of(kernel.value(op.operands[1]).type);
      if (!can_vec4_mem(dt)) return std::nullopt;
      if (!ptr_origins_aligned(op.operands[0], dt)) return std::nullopt;
      if (kernel.value(op.operands[1]).type.as_tile().shape[0] != analysis.block_size_1d) return std::nullopt;
    }
  }
  if (tile_mem_count == 0) return std::nullopt;

  // Determine scalar expressions and linear address info across entry_block.
  std::unordered_map<ValueId, std::string> scalar_exprs;
  populate_entry_param_exprs(kernel, analysis, scalar_exprs);

  std::unordered_map<ValueId, LinearInt> lin_ints;
  std::unordered_map<ValueId, LinearPtr> lin_ptrs;
  for (const auto& [vid, pidx] : analysis.val_to_param_idx) {
    const KernelParam& kp = kernel.params[pidx];
    if (kp.type.is_ptr()) {
      LinearPtr lp;
      lp.valid = true;
      lp.stride = 0;
      lp.base_ptr = scalar_exprs[vid];
      lp.offset_is_zero = true;
      lp.offset_div4 = true;
      lp.offset_nonneg = true;
      lin_ptrs[vid] = lp;
    } else if (is_int_dtype(scalar_dtype_of(kp.type))) {
      LinearInt li;
      li.valid = true;
      li.stride = 0;
      li.base_expr = scalar_exprs[vid];
      li.dtype = scalar_dtype_of(kp.type);
      li.base_is_zero = (kp.is_constexpr && kp.constexpr_val == 0);
      li.base_div4 = analysis.div4_values.count(vid) > 0;
      li.base_nonneg = analysis.nonneg_values.count(vid) > 0;
      lin_ints[vid] = li;
    }
  }

  HelperFlags helpers;
  std::ostringstream scalar_prologue;
  std::string scalar_err;

  for (OpId oid : entry.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased || op.kind == OpKind::Return) continue;

    bool has_tile_result = (!op.results.empty() && kernel.value(op.results[0]).type.is_tile());
    bool is_scalar_op = !has_tile_result && op.kind != OpKind::Store;
    if (op.kind == OpKind::Store && !kernel.value(op.operands[0]).type.is_tile()) {
      return std::nullopt;
    }

    if (is_scalar_op) {
      if (op.kind == OpKind::Constant) {
        ValueId rid = op.results[0];
        ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
        std::string lit = msl_const_literal(std::get<ConstAttr>(op.attr), rdt);
        scalar_exprs[rid] = lit;
        if (is_int_dtype(rdt)) {
          LinearInt li;
          li.valid = true;
          li.stride = 0;
          li.base_expr = lit;
          li.dtype = rdt;
          auto c = const_as_i64(std::get<ConstAttr>(op.attr));
          li.base_is_zero = (c.has_value() && *c == 0);
          li.base_div4 = analysis.div4_values.count(rid) > 0;
          li.base_nonneg = analysis.nonneg_values.count(rid) > 0;
          lin_ints[rid] = li;
        }
        continue;
      }

      std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, scalar_err);
      if (!scalar_err.empty()) return std::nullopt;
      ValueId rid = op.results[0];
      const Type& rty = kernel.value(rid).type;
      std::string vname = ssa_var(rid);
      if (rty.is_ptr()) {
        scalar_prologue << "  " << msl_ptr_type_str(analysis, rid, rty.as_ptr().pointee) << " " << vname << " = "
                        << expr << ";\n";
        LinearPtr lp;
        lp.valid = true;
        lp.stride = 0;
        lp.base_ptr = vname;
        lp.offset_is_zero = true;
        lp.offset_div4 = true;
        lp.offset_nonneg = true;
        lin_ptrs[rid] = lp;
      } else {
        ScalarDType rdt = rty.as_scalar();
        scalar_prologue << "  " << msl_scalar_type(rdt) << " " << vname << " = " << expr << ";\n";
        if (is_int_dtype(rdt)) {
          LinearInt li;
          li.valid = true;
          li.stride = 0;
          li.base_expr = vname;
          li.dtype = rdt;
          li.base_div4 = analysis.div4_values.count(rid) > 0;
          li.base_nonneg = analysis.nonneg_values.count(rid) > 0;
          lin_ints[rid] = li;
        }
      }
      scalar_exprs[rid] = vname;
      continue;
    }

    // Track linear integer and pointer expressions on 1D tiles.
    if (op.kind == OpKind::Arange && !op.results.empty()) {
      ValueId rid = op.results[0];
      const auto& attr = std::get<ArangeAttr>(op.attr);
      ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
      LinearInt li;
      li.valid = true;
      li.stride = 1;
      li.dtype = rdt;
      li.base_expr = format_scalar_literal_from_i64(attr.start, rdt);
      li.base_is_zero = (attr.start == 0);
      li.base_div4 = (attr.start % 4 == 0);
      li.base_nonneg = (attr.start >= 0);
      lin_ints[rid] = li;
    } else if ((op.kind == OpKind::Splat || op.kind == OpKind::Broadcast) && !op.results.empty()) {
      ValueId src = op.operands[0];
      ValueId rid = op.results[0];
      if (lin_ints.count(src)) lin_ints[rid] = lin_ints[src];
      if (lin_ptrs.count(src)) lin_ptrs[rid] = lin_ptrs[src];
    } else if (op.kind == OpKind::Cast && !op.results.empty()) {
      ValueId src = op.operands[0];
      ValueId rid = op.results[0];
      ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
      if (lin_ints.count(src) && is_int_dtype(rdt)) {
        LinearInt li = lin_ints[src];
        if (!li.base_is_zero) {
          li.base_expr = msl_cast_expr(li.base_expr, li.dtype, rdt, false);
        } else {
          li.base_expr = format_scalar_literal_from_i64(0, rdt);
        }
        li.dtype = rdt;
        lin_ints[rid] = li;
      }
    } else if (op.kind == OpKind::Binary && !op.results.empty() && std::holds_alternative<BinaryAttr>(op.attr)) {
      BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
      ValueId rid = op.results[0];
      ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
      if (is_int_dtype(rdt) && lin_ints.count(op.operands[0]) && lin_ints.count(op.operands[1])) {
        const LinearInt& L = lin_ints[op.operands[0]];
        const LinearInt& R = lin_ints[op.operands[1]];
        if (bop == BinaryOp::Add && (L.stride + R.stride <= 1)) {
          LinearInt res;
          res.valid = true;
          res.stride = L.stride + R.stride;
          res.dtype = rdt;
          std::string lb = L.base_is_zero ? "" : msl_cast_expr(L.base_expr, L.dtype, rdt, false);
          std::string rb = R.base_is_zero ? "" : msl_cast_expr(R.base_expr, R.dtype, rdt, false);
          res.base_is_zero = L.base_is_zero && R.base_is_zero;
          res.base_expr = combine_add_expr(lb, L.base_is_zero, rb, R.base_is_zero, rdt);
          res.base_div4 = L.base_div4 && R.base_div4;
          res.base_nonneg = L.base_nonneg && R.base_nonneg;
          lin_ints[rid] = res;
        } else if (bop == BinaryOp::Sub && R.stride == 0 && L.stride <= 1) {
          LinearInt res;
          res.valid = true;
          res.stride = L.stride;
          res.dtype = rdt;
          std::string lb = L.base_is_zero ? "" : msl_cast_expr(L.base_expr, L.dtype, rdt, false);
          std::string rb = R.base_is_zero ? "" : msl_cast_expr(R.base_expr, R.dtype, rdt, false);
          res.base_is_zero = L.base_is_zero && R.base_is_zero;
          res.base_expr = combine_sub_expr(lb, L.base_is_zero, rb, R.base_is_zero, rdt);
          res.base_div4 = L.base_div4 && R.base_div4;
          res.base_nonneg = false;
          lin_ints[rid] = res;
        }
      }
    } else if (op.kind == OpKind::PtrAdd && !op.results.empty()) {
      ValueId pid = op.operands[0];
      ValueId offid = op.operands[1];
      ValueId rid = op.results[0];
      if (lin_ptrs.count(pid) && lin_ints.count(offid)) {
        const LinearPtr& P = lin_ptrs[pid];
        const LinearInt& O = lin_ints[offid];
        if (P.stride + O.stride <= 1) {
          LinearPtr res;
          res.valid = true;
          res.stride = P.stride + O.stride;
          res.base_ptr = P.base_ptr;
          res.offset_dtype = P.offset_is_zero ? O.dtype : common_cmp_dtype(P.offset_dtype, O.dtype);
          std::string po = P.offset_is_zero ? "" : msl_cast_expr(P.base_offset, P.offset_dtype, res.offset_dtype, false);
          std::string oo = O.base_is_zero ? "" : msl_cast_expr(O.base_expr, O.dtype, res.offset_dtype, false);
          res.offset_is_zero = P.offset_is_zero && O.base_is_zero;
          res.base_offset = combine_add_expr(po, P.offset_is_zero, oo, O.base_is_zero, res.offset_dtype);
          res.offset_div4 = P.offset_div4 && O.base_div4;
          res.offset_nonneg = P.offset_nonneg && O.base_nonneg;
          lin_ptrs[rid] = res;
        }
      }
    }
  }

  // Validate all tile loads and stores have stride == 1 linear pointers and analyzable guards.
  std::unordered_map<std::string, std::string> base_idx4_vars;
  std::vector<std::tuple<std::string, ScalarDType, std::string>> idx4_decls;
  std::vector<std::string> fast_guards;
  std::unordered_set<std::string> seen_guards;
  std::unordered_map<OpId, std::pair<std::string, std::string>> mem_ptr_and_idx4;

  auto get_ptr_idx4 = [&](const LinearPtr& lp) -> std::string {
    std::string key = msl_scalar_type(lp.offset_dtype) + "|" + (lp.offset_is_zero ? "0" : lp.base_offset);
    auto it = base_idx4_vars.find(key);
    if (it != base_idx4_vars.end()) return it->second;
    std::string vname = base_idx4_vars.empty() ? "m1_idx4" : ("m1_idx4_" + std::to_string(base_idx4_vars.size()));
    std::string rhs = lp.offset_is_zero
                          ? (msl_scalar_type(lp.offset_dtype) + "(m1_elem_base)")
                          : ("(" + lp.base_offset + " + " + msl_scalar_type(lp.offset_dtype) + "(m1_elem_base))");
    base_idx4_vars[key] = vname;
    idx4_decls.emplace_back(vname, lp.offset_dtype, rhs);
    return vname;
  };

  auto add_guard = [&](const std::string& g) {
    if (g == "true" || g.empty()) return;
    if (seen_guards.insert(g).second) fast_guards.push_back(g);
  };

  for (OpId oid : entry.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    if (op.kind == OpKind::Load && !op.results.empty() && kernel.value(op.results[0]).type.is_tile()) {
      auto lo = extract_load_operands(op, kernel);
      auto pit = lin_ptrs.find(lo.ptrs);
      if (pit == lin_ptrs.end() || !pit->second.valid || pit->second.stride != 1) return std::nullopt;
      std::string idx4 = get_ptr_idx4(pit->second);
      mem_ptr_and_idx4[oid] = {pit->second.base_ptr, idx4};
      if (!pit->second.offset_div4) {
        add_guard("((" + idx4 + " & " + format_scalar_literal_from_i64(3, pit->second.offset_dtype) + ") == " +
                  format_scalar_literal_from_i64(0, pit->second.offset_dtype) + ")");
      }
      std::string g;
      if (!try_build_all4_guard(kernel, lin_ints, scalar_exprs, lo.mask, "m1_elem_base", base_idx4_vars, idx4_decls,
                                g)) {
        return std::nullopt;
      }
      add_guard(g);
    } else if (op.kind == OpKind::Store && op.operands.size() >= 2 && kernel.value(op.operands[1]).type.is_tile()) {
      auto so = extract_store_operands(op);
      auto pit = lin_ptrs.find(so.ptrs);
      if (pit == lin_ptrs.end() || !pit->second.valid || pit->second.stride != 1) return std::nullopt;
      std::string idx4 = get_ptr_idx4(pit->second);
      mem_ptr_and_idx4[oid] = {pit->second.base_ptr, idx4};
      if (!pit->second.offset_div4) {
        add_guard("((" + idx4 + " & " + format_scalar_literal_from_i64(3, pit->second.offset_dtype) + ") == " +
                  format_scalar_literal_from_i64(0, pit->second.offset_dtype) + ")");
      }
      std::string g;
      if (!try_build_all4_guard(kernel, lin_ints, scalar_exprs, so.mask, "m1_elem_base", base_idx4_vars, idx4_decls,
                                g)) {
        return std::nullopt;
      }
      add_guard(g);
    }
  }

  // Choose tg_x: honor LayoutAttr::tg_threads[0] if set, else clamp(BLOCK_SIZE / 4, 32, 256) multiple of 32.
  uint32_t vec_elems = static_cast<uint32_t>(analysis.block_size_1d / 4);
  uint32_t tg_x = std::clamp<uint32_t>(round_up_32(vec_elems), 32u, 256u);
  if (analysis.hint_layout.has_value() && analysis.hint_layout->tg_threads[0] > 1) {
    tg_x = std::min<uint32_t>(analysis.hint_layout->tg_threads[0], target.max_threads_per_threadgroup);
  }
  uint32_t chunks = (vec_elems + tg_x - 1u) / tg_x;
  bool exact_cover = (chunks * tg_x == vec_elems);

  // Determine which tile values are live in the vec4 fast path (skip dead address/mask ops).
  std::unordered_set<ValueId> needed_fast;
  std::vector<ValueId> worklist;
  auto mark_needed = [&](ValueId v) {
    if (v != kInvalidId && kernel.value(v).type.is_tile() && needed_fast.insert(v).second) {
      worklist.push_back(v);
    }
  };
  for (OpId oid : entry.ops) {
    const Op& op = kernel.op(oid);
    if (!op.erased && op.kind == OpKind::Store && op.operands.size() >= 2) {
      mark_needed(op.operands[1]);
    }
  }
  while (!worklist.empty()) {
    ValueId vid = worklist.back();
    worklist.pop_back();
    const Value& v = kernel.value(vid);
    if (v.def_op == kInvalidId) continue;
    const Op& op = kernel.op(v.def_op);
    if (op.kind == OpKind::Load) continue;  // ptrs/mask/other handled by fast path directly
    for (ValueId o : op.operands) mark_needed(o);
  }

  std::ostringstream body;
  body << "  (void)m1_grid_dim; (void)m1_tid; (void)m1_tg_size;\n";
  body << scalar_prologue.str();

  int ind = 1;
  if (chunks > 1) {
    if (chunks <= 4) body << indent_str(ind) << "#pragma unroll\n";
    body << indent_str(ind) << "for (uint m1_c = 0u; m1_c < " << chunks << "u; ++m1_c) {\n";
    ++ind;
    body << indent_str(ind) << "uint m1_elem_base = (m1_tid_linear + m1_c * " << tg_x << "u) * 4u;\n";
  } else {
    body << indent_str(ind) << "uint m1_elem_base = m1_tid_linear * 4u;\n";
  }

  if (!exact_cover) {
    body << indent_str(ind) << "if (m1_elem_base < " << analysis.block_size_1d << "u) {\n";
    ++ind;
  }

  for (const auto& [vname, dt, rhs] : idx4_decls) {
    body << indent_str(ind) << msl_scalar_type(dt) << " " << vname << " = " << rhs << ";\n";
  }

  bool has_guard = !fast_guards.empty();
  if (has_guard) {
    body << indent_str(ind) << "if (";
    for (size_t i = 0; i < fast_guards.size(); ++i) {
      if (i > 0) body << " && ";
      body << fast_guards[i];
    }
    body << ") {\n";
    ++ind;
  }

  // Emit vec4 fast path body.
  std::unordered_map<ValueId, std::string> vec4_exprs = scalar_exprs;
  std::unordered_set<ValueId> is_vec4_val;
  auto get_v4 = [&](ValueId id, ScalarDType expected_dt) -> std::string {
    std::string e = vec4_exprs.at(id);
    bool already = is_vec4_val.count(id) > 0;
    ScalarDType cur_dt = scalar_dtype_of(kernel.value(id).type);
    if (!already) {
      e = msl_cast_expr(e, cur_dt, expected_dt, false);
      return promote_to_vec4(e, expected_dt, false);
    }
    return msl_cast_expr(e, cur_dt, expected_dt, true);
  };

  for (OpId oid : entry.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased || op.kind == OpKind::Return) continue;

    if (op.kind == OpKind::Store) {
      auto so = extract_store_operands(op);
      const auto& [bptr, idx4] = mem_ptr_and_idx4.at(oid);
      ScalarDType dt = scalar_dtype_of(kernel.value(so.val).type);
      std::string vexpr = get_v4(so.val, dt);
      body << indent_str(ind) << "*(device " << msl_vec4_type(dt) << "*)(" << bptr << " + " << idx4 << ") = " << vexpr
           << ";\n";
      continue;
    }

    if (op.results.empty() || !needed_fast.count(op.results[0])) continue;
    ValueId rid = op.results[0];
    ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
    std::string vname = ssa_var(rid);
    std::string vty = msl_vec4_type(rdt);

    if (op.kind == OpKind::Load) {
      const auto& [bptr, idx4] = mem_ptr_and_idx4.at(oid);
      body << indent_str(ind) << vty << " " << vname << " = *(const device " << vty << "*)(" << bptr << " + " << idx4
           << ");\n";
    } else if (op.kind == OpKind::Arange) {
      const auto& attr = std::get<ArangeAttr>(op.attr);
      std::string base = (attr.start == 0)
                             ? (msl_scalar_type(rdt) + "(m1_elem_base)")
                             : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) +
                                "(m1_elem_base))");
      body << indent_str(ind) << vty << " " << vname << " = (" << vty << "(" << base << ") + " << vty << "("
           << format_scalar_literal_from_i64(0, rdt) << ", " << format_scalar_literal_from_i64(1, rdt) << ", "
           << format_scalar_literal_from_i64(2, rdt) << ", " << format_scalar_literal_from_i64(3, rdt) << "));\n";
    } else if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast) {
      std::string se = get_v4(op.operands[0], rdt);
      body << indent_str(ind) << vty << " " << vname << " = " << se << ";\n";
    } else if (op.kind == OpKind::Cast) {
      ScalarDType sdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
      std::string se = get_v4(op.operands[0], sdt);
      body << indent_str(ind) << vty << " " << vname << " = " << msl_cast_expr(se, sdt, rdt, true) << ";\n";
    } else if (op.kind == OpKind::Unary) {
      UnaryOp uop = std::get<UnaryAttr>(op.attr).op;
      ScalarDType sdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
      std::string se = get_v4(op.operands[0], sdt);
      body << indent_str(ind) << vty << " " << vname << " = " << msl_unary_expr(uop, se, rdt, true, helpers) << ";\n";
    } else if (op.kind == OpKind::Binary) {
      BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
      ScalarDType ldt = scalar_dtype_of(kernel.value(op.operands[0]).type);
      ScalarDType rdt_in = scalar_dtype_of(kernel.value(op.operands[1]).type);
      std::string le = get_v4(op.operands[0], ldt);
      std::string re = get_v4(op.operands[1], rdt_in);
      body << indent_str(ind) << vty << " " << vname << " = " << msl_binary_expr(bop, le, re, ldt, rdt_in, rdt, true)
           << ";\n";
    } else if (op.kind == OpKind::Select) {
      std::string ce = get_v4(op.operands[0], ScalarDType::Bool);
      ScalarDType tdt = scalar_dtype_of(kernel.value(op.operands[1]).type);
      ScalarDType fdt = scalar_dtype_of(kernel.value(op.operands[2]).type);
      std::string te = get_v4(op.operands[1], tdt);
      std::string fe = get_v4(op.operands[2], fdt);
      body << indent_str(ind) << vty << " " << vname << " = " << msl_select_expr(ce, te, fe, tdt, fdt, rdt, true)
           << ";\n";
    } else {
      return std::nullopt;
    }

    vec4_exprs[rid] = vname;
    is_vec4_val.insert(rid);
  }

  // Emit scalar tail fallback when fast path is guarded.
  if (has_guard) {
    --ind;
    body << indent_str(ind) << "} else {\n";
    ++ind;
    body << indent_str(ind) << "for (uint m1_e = 0u; m1_e < 4u; ++m1_e) {\n";
    ++ind;
    body << indent_str(ind) << "uint m1_lane = m1_elem_base + m1_e;\n";

    std::unordered_map<ValueId, std::string> tail_exprs = scalar_exprs;
    for (OpId oid : entry.ops) {
      const Op& op = kernel.op(oid);
      if (op.erased || op.kind == OpKind::Return) continue;

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        std::string pe = tail_exprs.at(so.ptrs);
        ScalarDType pdt = scalar_dtype_of(kernel.value(so.ptrs).type);
        std::string ve = msl_cast_expr(tail_exprs.at(so.val), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
        if (so.mask != kInvalidId) {
          body << indent_str(ind) << "if (" << tail_exprs.at(so.mask) << ") {\n";
          body << indent_str(ind + 1) << "*(" << pe << ") = " << ve << ";\n";
          body << indent_str(ind) << "}\n";
        } else {
          body << indent_str(ind) << "*(" << pe << ") = " << ve << ";\n";
        }
        continue;
      }

      if (op.results.empty() || !kernel.value(op.results[0]).type.is_tile()) continue;
      ValueId rid = op.results[0];
      const Type& rty = kernel.value(rid).type;
      std::string vname = ssa_var(rid);

      if (op.kind == OpKind::Arange) {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        ScalarDType rdt = scalar_dtype_of(rty);
        std::string expr = (attr.start == 0)
                               ? (msl_scalar_type(rdt) + "(m1_lane)")
                               : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) +
                                  "(m1_lane))");
        body << indent_str(ind) << msl_scalar_type(rdt) << " " << vname << " = " << expr << ";\n";
        tail_exprs[rid] = vname;
      } else if (op.kind == OpKind::Splat || op.kind == OpKind::Broadcast) {
        tail_exprs[rid] = tail_exprs.at(op.operands[0]);
      } else if (op.kind == OpKind::PtrAdd) {
        std::string expr = "(" + tail_exprs.at(op.operands[0]) + " + " + tail_exprs.at(op.operands[1]) + ")";
        body << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vname << " = "
             << expr << ";\n";
        tail_exprs[rid] = vname;
      } else {
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, tail_exprs, helpers, scalar_err);
        if (!scalar_err.empty()) return std::nullopt;
        ScalarDType rdt = scalar_dtype_of(rty);
        body << indent_str(ind) << msl_scalar_type(rdt) << " " << vname << " = " << expr << ";\n";
        tail_exprs[rid] = vname;
      }
    }

    --ind;
    body << indent_str(ind) << "}\n";
    --ind;
    body << indent_str(ind) << "}\n";
  }

  if (!exact_cover) {
    --ind;
    body << indent_str(ind) << "}\n";
  }
  if (chunks > 1) {
    --ind;
    body << indent_str(ind) << "}\n";
  }
  body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, helpers) + body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

// ---------------------------------------------------------------------------
// General 1D Scalar / Masked Emitter
// ---------------------------------------------------------------------------

struct General1DEmitter {
  const Kernel& kernel;
  const KernelAnalysis& analysis;
  uint32_t tg_x = 32;
  int64_t block_size = 1;
  HelperFlags helpers;
  std::string err;

  bool emit_block(BlockId bid, int ind, bool in_lane_loop, std::unordered_map<ValueId, std::string>& exprs,
                  const std::vector<std::string>* yield_targets, std::ostringstream& out) {
    const Block& blk = kernel.block(bid);
    for (OpId oid : blk.ops) {
      if (!emit_op(oid, ind, in_lane_loop, exprs, yield_targets, out)) return false;
    }
    return true;
  }

  bool emit_op(OpId oid, int ind, bool in_lane_loop, std::unordered_map<ValueId, std::string>& exprs,
               const std::vector<std::string>* yield_targets, std::ostringstream& out) {
    const Op& op = kernel.op(oid);
    if (op.erased) return true;

    if (op.kind == OpKind::Return) {
      return true;
    }

    if (op.kind == OpKind::Yield) {
      if (yield_targets != nullptr) {
        std::vector<std::string> temps;
        temps.reserve(op.operands.size());
        for (size_t i = 0; i < op.operands.size() && i < yield_targets->size(); ++i) {
          ValueId ov = op.operands[i];
          std::string tmp = "m1_y_" + std::to_string(oid) + "_" + std::to_string(i);
          const Type& ty = kernel.value(ov).type;
          if (is_ptr_type(ty)) {
            out << indent_str(ind) << msl_ptr_type_str(analysis, ov, scalar_dtype_of(ty)) << " " << tmp << " = "
                << exprs.at(ov) << ";\n";
          } else {
            out << indent_str(ind) << msl_scalar_type(scalar_dtype_of(ty)) << " " << tmp << " = " << exprs.at(ov)
                << ";\n";
          }
          temps.push_back(tmp);
        }
        for (size_t i = 0; i < temps.size(); ++i) {
          out << indent_str(ind) << (*yield_targets)[i] << " = " << temps[i] << ";\n";
        }
      }
      return true;
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
      return true;
    }

    if (op.kind == OpKind::ForLoop) {
      if (op.operands.size() < 3 || op.regions.empty()) {
        err = "Malformed ForLoop op";
        return false;
      }
      const Block& body = kernel.block(op.regions[0]);
      if (body.args.empty()) {
        err = "ForLoop body block missing induction variable arg";
        return false;
      }
      ValueId iv_id = body.args[0];
      ScalarDType iv_dt = scalar_dtype_of(kernel.value(iv_id).type);
      std::string iv_var = ssa_var(iv_id);
      std::string lo_e =
          msl_cast_expr(exprs.at(op.operands[0]), scalar_dtype_of(kernel.value(op.operands[0]).type), iv_dt, false);
      std::string hi_e =
          msl_cast_expr(exprs.at(op.operands[1]), scalar_dtype_of(kernel.value(op.operands[1]).type), iv_dt, false);
      std::string step_e =
          msl_cast_expr(exprs.at(op.operands[2]), scalar_dtype_of(kernel.value(op.operands[2]).type), iv_dt, false);

      bool neg_step = false;
      if (auto it = analysis.const_i64.find(op.operands[2]); it != analysis.const_i64.end() && it->second < 0) {
        neg_step = true;
      }

      size_t num_iter = op.operands.size() - 3;
      std::vector<std::string> iter_vars;
      iter_vars.reserve(num_iter);
      for (size_t i = 0; i < num_iter; ++i) {
        ValueId arg_id = body.args[1 + i];
        ValueId init_id = op.operands[3 + i];
        std::string avar = ssa_var(arg_id);
        const Type& aty = kernel.value(arg_id).type;
        if (is_ptr_type(aty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, arg_id, scalar_dtype_of(aty)) << " " << avar << " = "
              << exprs.at(init_id) << ";\n";
        } else {
          ScalarDType adt = scalar_dtype_of(aty);
          out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << " = "
              << msl_cast_expr(exprs.at(init_id), scalar_dtype_of(kernel.value(init_id).type), adt, false) << ";\n";
        }
        exprs[arg_id] = avar;
        iter_vars.push_back(avar);
      }

      out << indent_str(ind) << "for (" << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << "; " << iv_var
          << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var << " += " << step_e << ") {\n";
      exprs[iv_id] = iv_var;
      if (!emit_block(op.regions[0], ind + 1, in_lane_loop, exprs, &iter_vars, out)) return false;
      out << indent_str(ind) << "}\n";

      for (size_t i = 0; i < op.results.size() && i < iter_vars.size(); ++i) {
        exprs[op.results[i]] = iter_vars[i];
      }
      return true;
    }

    if (op.kind == OpKind::If) {
      if (op.operands.empty() || op.regions.empty()) {
        err = "Malformed If op";
        return false;
      }
      std::vector<std::string> res_vars;
      for (ValueId rid : op.results) {
        const Type& rty = kernel.value(rid).type;
        std::string rv = ssa_var(rid);
        if (is_ptr_type(rty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << rv
              << " = nullptr;\n";
        } else {
          ScalarDType rdt = scalar_dtype_of(rty);
          out << indent_str(ind) << msl_scalar_type(rdt) << " " << rv << " = " << msl_zero_literal(rdt, false)
              << ";\n";
        }
        exprs[rid] = rv;
        res_vars.push_back(rv);
      }
      out << indent_str(ind) << "if (" << exprs.at(op.operands[0]) << ") {\n";
      if (!emit_block(op.regions[0], ind + 1, in_lane_loop, exprs, &res_vars, out)) return false;
      if (op.regions.size() >= 2 && op.regions[1] != kInvalidId) {
        out << indent_str(ind) << "} else {\n";
        if (!emit_block(op.regions[1], ind + 1, in_lane_loop, exprs, &res_vars, out)) return false;
      }
      out << indent_str(ind) << "}\n";
      return true;
    }

    if (op.kind == OpKind::Store) {
      auto so = extract_store_operands(op);
      const Type& pty = kernel.value(so.ptrs).type;
      bool is_scalar_store = !pty.is_tile();
      std::string pe = exprs.at(so.ptrs);
      ScalarDType pdt = scalar_dtype_of(pty);
      std::string ve = msl_cast_expr(exprs.at(so.val), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
      std::string guard;
      if (is_scalar_store) {
        guard = in_lane_loop ? "(m1_lane == 0u)" : "(m1_tid_linear == 0u)";
      } else if (in_lane_loop && !pty.as_tile().shape.empty() && pty.as_tile().shape[0] < block_size) {
        guard = "(m1_lane < " + std::to_string(pty.as_tile().shape[0]) + "u)";
      }
      if (so.mask != kInvalidId) {
        guard = guard.empty() ? exprs.at(so.mask) : ("(" + guard + " && (" + exprs.at(so.mask) + "))");
      }
      if (!guard.empty()) {
        out << indent_str(ind) << "if (" << guard << ") {\n";
        out << indent_str(ind + 1) << "*(" << pe << ") = " << ve << ";\n";
        out << indent_str(ind) << "}\n";
      } else {
        out << indent_str(ind) << "*(" << pe << ") = " << ve << ";\n";
      }
      return true;
    }

    if (op.kind == OpKind::AtomicAdd && !op.operands.empty()) {
      const Type& pty = kernel.value(op.operands[0]).type;
      bool is_scalar_atomic = !pty.is_tile();
      ValueId rid = op.results.empty() ? kInvalidId : op.results[0];
      ScalarDType rdt = scalar_dtype_of(pty);
      std::string expr = emit_scalar_op_expr(kernel, analysis, op, exprs, helpers, err);
      if (!err.empty()) return false;
      std::string vname = (rid != kInvalidId) ? ssa_var(rid) : ("m1_atom_" + std::to_string(oid));
      std::string cond;
      if (is_scalar_atomic) {
        cond = in_lane_loop ? "(m1_lane == 0u)" : "(m1_tid_linear == 0u)";
      } else if (in_lane_loop && !pty.as_tile().shape.empty() && pty.as_tile().shape[0] < block_size) {
        cond = "(m1_lane < " + std::to_string(pty.as_tile().shape[0]) + "u)";
      }
      if (!cond.empty()) {
        out << indent_str(ind) << msl_scalar_type(rdt) << " " << vname << " = " << msl_zero_literal(rdt, false)
            << ";\n";
        out << indent_str(ind) << "if (" << cond << ") { " << vname << " = " << expr << "; }\n";
      } else {
        out << indent_str(ind) << msl_scalar_type(rdt) << " " << vname << " = " << expr << ";\n";
      }
      if (rid != kInvalidId) exprs[rid] = vname;
      return true;
    }

    if (op.kind == OpKind::Arange && !op.results.empty()) {
      ValueId rid = op.results[0];
      const auto& attr = std::get<ArangeAttr>(op.attr);
      ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
      std::string vname = ssa_var(rid);
      std::string expr = (attr.start == 0)
                             ? (msl_scalar_type(rdt) + "(m1_lane)")
                             : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) +
                                "(m1_lane))");
      out << indent_str(ind) << msl_scalar_type(rdt) << " " << vname << " = " << expr << ";\n";
      exprs[rid] = vname;
      return true;
    }

    if ((op.kind == OpKind::Splat || op.kind == OpKind::Broadcast || op.kind == OpKind::Reshape) &&
        !op.results.empty()) {
      exprs[op.results[0]] = exprs.at(op.operands[0]);
      return true;
    }

    if (op.kind == OpKind::Constant && !op.results.empty()) {
      ValueId rid = op.results[0];
      exprs[rid] = msl_const_literal(std::get<ConstAttr>(op.attr), scalar_dtype_of(kernel.value(rid).type));
      return true;
    }

    if (op.results.empty()) {
      err = "Unsupported void op in 1D MetalCodegen";
      return false;
    }

    ValueId rid = op.results[0];
    const Type& rty = kernel.value(rid).type;
    std::string expr = emit_scalar_op_expr(kernel, analysis, op, exprs, helpers, err);
    if (!err.empty()) return false;
    std::string vname = ssa_var(rid);
    if (is_ptr_type(rty)) {
      out << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vname << " = " << expr
          << ";\n";
    } else {
      out << indent_str(ind) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vname << " = " << expr << ";\n";
    }
    exprs[rid] = vname;
    return true;
  }
};

CompiledMSL emit_1d_general(const Kernel& kernel, const TargetSpec& target, const KernelAnalysis& analysis) {
  if (analysis.has_reduce) {
    return CompiledMSL{false, "", {}, "Reduce op requires dedicated reduction lowering pass before MetalCodegen"};
  }

  int64_t block_size = std::max<int64_t>(1, analysis.block_size_1d);
  uint32_t tg_x = std::clamp<uint32_t>(round_up_32(static_cast<uint32_t>(block_size)), 32u, 256u);
  if (analysis.hint_layout.has_value() && analysis.hint_layout->tg_threads[0] > 1) {
    tg_x = std::min<uint32_t>(analysis.hint_layout->tg_threads[0], target.max_threads_per_threadgroup);
  }

  General1DEmitter em{kernel, analysis, tg_x, block_size, {}, {}};
  std::unordered_map<ValueId, std::string> exprs;
  populate_entry_param_exprs(kernel, analysis, exprs);

  std::ostringstream body;
  body << "  (void)m1_grid_dim; (void)m1_tid; (void)m1_tg_size;\n";

  const Block& entry = kernel.block(kernel.entry_block);

  if (analysis.max_rank == 0) {
    if (!em.emit_block(kernel.entry_block, 1, false, exprs, nullptr, body)) {
      return CompiledMSL{false, "", {}, em.err};
    }
  } else {
    size_t split_idx = entry.ops.size();
    for (size_t i = 0; i < entry.ops.size(); ++i) {
      const Op& op = kernel.op(entry.ops[i]);
      if (op.erased || op.kind == OpKind::Return) continue;
      bool touches_tile = false;
      for (ValueId r : op.results) {
        if (kernel.value(r).type.is_tile()) touches_tile = true;
      }
      for (ValueId o : op.operands) {
        if (o != kInvalidId && kernel.value(o).type.is_tile()) touches_tile = true;
      }
      for (BlockId rb : op.regions) {
        if (block_has_tile_op(kernel, rb)) touches_tile = true;
      }
      if (touches_tile || op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd || op.kind == OpKind::Barrier ||
          op.kind == OpKind::ForLoop || op.kind == OpKind::If) {
        split_idx = i;
        break;
      }
    }

    for (size_t i = 0; i < split_idx; ++i) {
      if (!em.emit_op(entry.ops[i], 1, false, exprs, nullptr, body)) {
        return CompiledMSL{false, "", {}, em.err};
      }
    }

    if (split_idx < entry.ops.size()) {
      body << "  for (uint m1_lane = m1_tid_linear; m1_lane < " << block_size << "u; m1_lane += " << tg_x << "u) {\n";
      for (size_t i = split_idx; i < entry.ops.size(); ++i) {
        if (!em.emit_op(entry.ops[i], 2, true, exprs, nullptr, body)) {
          return CompiledMSL{false, "", {}, em.err};
        }
      }
      body << "  }\n";
    }
  }
  body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, em.helpers) + body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

// ---------------------------------------------------------------------------
// 2D Tiled & Dot Path Emitter
// ---------------------------------------------------------------------------

struct DotBufferInfo {
  OpId dot_op = kInvalidId;
  std::string tg_a_name;
  std::string tg_b_name;
  ScalarDType lhs_dt = ScalarDType::F32;
  ScalarDType rhs_dt = ScalarDType::F32;
  int64_t lhs_rows = 0;
  int64_t lhs_cols = 0;
  int64_t rhs_rows = 0;
  int64_t rhs_cols = 0;
  int64_t bm = 0;
  int64_t bn = 0;
  int64_t bk = 0;
  bool transpose_a = false;
  bool transpose_b = false;
};

struct PtrIterInfo {
  ValueId init_val = kInvalidId;
  std::string poff_var;
  ValueId delta_val = kInvalidId;
};

struct Tiled2DEmitter {
  const Kernel& kernel;
  const KernelAnalysis& analysis;
  int64_t bm = 0;
  int64_t bn = 0;
  uint32_t tg_x = 16;
  uint32_t tg_y = 16;
  uint32_t rm = 1;
  uint32_t rn = 1;
  bool exact_m = true;
  bool exact_n = true;

  HelperFlags helpers;
  std::string err;

  std::unordered_map<OpId, DotBufferInfo> dot_infos;
  std::unordered_set<ValueId> materialized_2d;
  std::unordered_map<ValueId, PtrIterInfo> ptr_iters;
  std::unordered_map<ValueId, std::string> scalar_exprs;
  std::unordered_map<ValueId, std::string> mat_tile_var;

  uint32_t eval_seq = 0;

  struct CoordScope {
    std::string row_expr;
    std::string col_expr;
    bool in_owned_tile_loop = false;
    std::string owned_idx_expr;
    int ind = 1;
    std::ostringstream* out = nullptr;
    std::unordered_map<std::string, std::string> memo;
  };

  std::string eval_at(ValueId vid, const std::string& r_expr, const std::string& c_expr, CoordScope& scope) {
    if (vid == kInvalidId) return "0";
    const Value& v = kernel.value(vid);

    // Scalar or scalar pointer already computed in enclosing scope.
    if (!v.type.is_tile()) {
      auto it = scalar_exprs.find(vid);
      if (it != scalar_exprs.end()) return it->second;
      if (v.def_op != kInvalidId && kernel.op(v.def_op).kind == OpKind::Constant) {
        return msl_const_literal(std::get<ConstAttr>(kernel.op(v.def_op).attr), scalar_dtype_of(v.type));
      }
      return ssa_var(vid);
    }

    const TileType& tt = v.type.as_tile();

    // Pointer-tile loop iter_arg with scalar increment.
    if (auto pit = ptr_iters.find(vid); pit != ptr_iters.end()) {
      std::string base_p = eval_at(pit->second.init_val, r_expr, c_expr, scope);
      return "(" + base_p + " + " + pit->second.poff_var + ")";
    }

    // Materialized 2D tile [BM, BN] accessed inside thread-owned (m1_rm, m1_rn) loop.
    if (materialized_2d.count(vid) > 0) {
      if (scope.in_owned_tile_loop && r_expr == scope.row_expr && c_expr == scope.col_expr) {
        return mat_tile_var.at(vid) + "[" + scope.owned_idx_expr + "]";
      }
      err = "Cannot lazily re-index a materialized 2D tile at arbitrary coordinates";
      return "0";
    }

    // Normalize coordinates for broadcasting dimensions of size 1.
    std::string nr = r_expr;
    std::string nc = c_expr;
    if (tt.shape.size() == 1) {
      nr = "0u";
      if (tt.shape[0] == 1) nc = "0u";
    } else if (tt.shape.size() == 2) {
      if (tt.shape[0] == 1) nr = "0u";
      if (tt.shape[1] == 1) nc = "0u";
    }

    std::string mkey = std::to_string(vid) + "|" + nr + "|" + nc;
    if (auto mit = scope.memo.find(mkey); mit != scope.memo.end()) {
      return mit->second;
    }

    if (v.def_op == kInvalidId) {
      err = "Tile block argument without materialization info in 2D MetalCodegen";
      return "0";
    }

    const Op& op = kernel.op(v.def_op);
    ScalarDType rdt = scalar_dtype_of(v.type);
    std::string expr;

    switch (op.kind) {
      case OpKind::Arange: {
        const auto& attr = std::get<ArangeAttr>(op.attr);
        std::string idx_c = (tt.shape.size() == 1) ? nc : nc;
        expr = (attr.start == 0)
                   ? (msl_scalar_type(rdt) + "(" + idx_c + ")")
                   : ("(" + format_scalar_literal_from_i64(attr.start, rdt) + " + " + msl_scalar_type(rdt) + "(" +
                      idx_c + "))");
        break;
      }
      case OpKind::Splat: {
        expr = eval_at(op.operands[0], nr, nc, scope);
        scope.memo[mkey] = expr;
        return expr;
      }
      case OpKind::ExpandDims: {
        uint32_t axis = std::get<AxisAttr>(op.attr).axis;
        const Type& src_ty = kernel.value(op.operands[0]).type;
        if (!src_ty.is_tile()) {
          expr = eval_at(op.operands[0], "0u", "0u", scope);
        } else if (src_ty.as_tile().shape.size() == 1) {
          // 1D [L] -> 2D: axis == 1 gives [L, 1] (indexed by r), axis == 0 gives [1, L] (indexed by c).
          std::string coord_1d = (axis == 1) ? nr : nc;
          expr = eval_at(op.operands[0], "0u", coord_1d, scope);
        } else {
          err = "Unsupported ExpandDims rank > 2 in 2D MetalCodegen";
          return "0";
        }
        scope.memo[mkey] = expr;
        return expr;
      }
      case OpKind::Broadcast: {
        const Type& src_ty = kernel.value(op.operands[0]).type;
        if (!src_ty.is_tile()) {
          expr = eval_at(op.operands[0], "0u", "0u", scope);
        } else if (src_ty.as_tile().shape.size() == 1) {
          expr = eval_at(op.operands[0], "0u", nc, scope);
        } else if (src_ty.as_tile().shape.size() == 2) {
          expr = eval_at(op.operands[0], nr, nc, scope);
        } else {
          err = "Unsupported Broadcast rank > 2 in 2D MetalCodegen";
          return "0";
        }
        scope.memo[mkey] = expr;
        return expr;
      }
      case OpKind::Permute: {
        const auto& perm = std::get<PermuteAttr>(op.attr).perm;
        if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
          expr = eval_at(op.operands[0], nc, nr, scope);
        } else if (perm.size() == 2 && perm[0] == 0 && perm[1] == 1) {
          expr = eval_at(op.operands[0], nr, nc, scope);
        } else {
          err = "Unsupported Permute pattern in 2D MetalCodegen";
          return "0";
        }
        scope.memo[mkey] = expr;
        return expr;
      }
      case OpKind::Reshape: {
        const Type& src_ty = kernel.value(op.operands[0]).type;
        if (!src_ty.is_tile()) {
          expr = eval_at(op.operands[0], "0u", "0u", scope);
        } else {
          std::string lin = (tt.shape.size() == 2) ? ("(" + nr + " * " + std::to_string(tt.shape[1]) + "u + " + nc + ")")
                                                   : nc;
          const auto& sshape = src_ty.as_tile().shape;
          if (sshape.size() == 1) {
            expr = eval_at(op.operands[0], "0u", lin, scope);
          } else if (sshape.size() == 2) {
            std::string sr = "(" + lin + " / " + std::to_string(sshape[1]) + "u)";
            std::string sc = "(" + lin + " % " + std::to_string(sshape[1]) + "u)";
            expr = eval_at(op.operands[0], sr, sc, scope);
          } else {
            err = "Unsupported Reshape rank > 2 in 2D MetalCodegen";
            return "0";
          }
        }
        scope.memo[mkey] = expr;
        return expr;
      }
      case OpKind::Cast: {
        std::string se = eval_at(op.operands[0], nr, nc, scope);
        ScalarDType sdt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        expr = msl_cast_expr(se, sdt, rdt, false);
        break;
      }
      case OpKind::Unary: {
        std::string se = eval_at(op.operands[0], nr, nc, scope);
        expr = msl_unary_expr(std::get<UnaryAttr>(op.attr).op, se, rdt, false, helpers);
        break;
      }
      case OpKind::Binary: {
        std::string le = eval_at(op.operands[0], nr, nc, scope);
        std::string re = eval_at(op.operands[1], nr, nc, scope);
        ScalarDType ldt = scalar_dtype_of(kernel.value(op.operands[0]).type);
        ScalarDType rdt_in = scalar_dtype_of(kernel.value(op.operands[1]).type);
        expr = msl_binary_expr(std::get<BinaryAttr>(op.attr).op, le, re, ldt, rdt_in, rdt, false);
        break;
      }
      case OpKind::Select: {
        std::string ce = eval_at(op.operands[0], nr, nc, scope);
        std::string te = eval_at(op.operands[1], nr, nc, scope);
        std::string fe = eval_at(op.operands[2], nr, nc, scope);
        ScalarDType tdt = scalar_dtype_of(kernel.value(op.operands[1]).type);
        ScalarDType fdt = scalar_dtype_of(kernel.value(op.operands[2]).type);
        expr = msl_select_expr(ce, te, fe, tdt, fdt, rdt, false);
        break;
      }
      case OpKind::PtrAdd: {
        std::string pe = eval_at(op.operands[0], nr, nc, scope);
        std::string oe = eval_at(op.operands[1], nr, nc, scope);
        expr = "(" + pe + " + " + oe + ")";
        break;
      }
      case OpKind::Load: {
        auto lo = extract_load_operands(op, kernel);
        std::string pe = eval_at(lo.ptrs, nr, nc, scope);
        if (lo.mask == kInvalidId) {
          expr = "(*(" + pe + "))";
        } else {
          std::string me = eval_at(lo.mask, nr, nc, scope);
          std::string oe = (lo.other != kInvalidId)
                               ? msl_cast_expr(eval_at(lo.other, nr, nc, scope),
                                               scalar_dtype_of(kernel.value(lo.other).type), rdt, false)
                               : msl_zero_literal(rdt, false);
          expr = "(" + me + " ? (*(" + pe + ")) : " + oe + ")";
        }
        break;
      }
      default:
        err = "Unsupported lazy tile op in 2D MetalCodegen";
        return "0";
    }

    if (!err.empty()) return "0";
    std::string tmp = "m1_e" + std::to_string(eval_seq++);
    if (is_ptr_type(v.type)) {
      (*scope.out) << indent_str(scope.ind) << msl_ptr_type_str(analysis, vid, rdt) << " " << tmp << " = " << expr
                   << ";\n";
    } else {
      (*scope.out) << indent_str(scope.ind) << msl_scalar_type(rdt) << " " << tmp << " = " << expr << ";\n";
    }
    scope.memo[mkey] = tmp;
    return tmp;
  }

  void emit_owned_tile_loop_header(int ind, std::ostringstream& out, int& inner_ind) {
    out << indent_str(ind) << "for (uint m1_rm = 0u; m1_rm < " << rm << "u; ++m1_rm) {\n";
    out << indent_str(ind + 1) << "uint m1_row = m1_tid.y + m1_rm * " << tg_y << "u;\n";
    out << indent_str(ind + 1) << "for (uint m1_rn = 0u; m1_rn < " << rn << "u; ++m1_rn) {\n";
    out << indent_str(ind + 2) << "uint m1_col = m1_tid.x + m1_rn * " << tg_x << "u;\n";
    inner_ind = ind + 2;
    if (!exact_m || !exact_n) {
      out << indent_str(inner_ind) << "if (";
      if (!exact_m && !exact_n) {
        out << "m1_row < " << bm << "u && m1_col < " << bn << "u";
      } else if (!exact_m) {
        out << "m1_row < " << bm << "u";
      } else {
        out << "m1_col < " << bn << "u";
      }
      out << ") {\n";
      ++inner_ind;
    }
  }

  void emit_owned_tile_loop_footer(int ind, std::ostringstream& out) {
    if (!exact_m || !exact_n) {
      out << indent_str(ind + 2) << "}\n";
    }
    out << indent_str(ind + 1) << "}\n";
    out << indent_str(ind) << "}\n";
  }

  bool emit_dot_op(const Op& op, int ind, std::ostringstream& out) {
    auto dit = dot_infos.find(op.id);
    if (dit == dot_infos.end()) {
      err = "Internal error: missing DotBufferInfo";
      return false;
    }
    const DotBufferInfo& di = dit->second;
    auto dop = extract_dot_operands(op);
    ValueId rid = op.results[0];
    ScalarDType acc_dt = scalar_dtype_of(kernel.value(rid).type);
    uint32_t total_threads = tg_x * tg_y;

    // Stage lhs into threadgroup memory tg_a.
    int64_t lhs_numel = di.lhs_rows * di.lhs_cols;
    if (materialized_2d.count(dop.lhs) > 0) {
      int inner_ind = ind;
      emit_owned_tile_loop_header(ind, out, inner_ind);
      out << indent_str(inner_ind) << di.tg_a_name << "[m1_row * " << di.lhs_cols << "u + m1_col] = "
          << mat_tile_var.at(dop.lhs) << "[m1_rm * " << rn << "u + m1_rn];\n";
      emit_owned_tile_loop_footer(ind, out);
    } else {
      out << indent_str(ind) << "for (uint m1_s = m1_tid_linear; m1_s < " << lhs_numel << "u; m1_s += " << total_threads
          << "u) {\n";
      out << indent_str(ind + 1) << "uint m1_sr = m1_s / " << di.lhs_cols << "u;\n";
      out << indent_str(ind + 1) << "uint m1_sc = m1_s % " << di.lhs_cols << "u;\n";
      CoordScope sc{"m1_sr", "m1_sc", false, "", ind + 1, &out, {}};
      std::string a_expr = eval_at(dop.lhs, "m1_sr", "m1_sc", sc);
      if (!err.empty()) return false;
      out << indent_str(ind + 1) << di.tg_a_name << "[m1_s] = "
          << msl_cast_expr(a_expr, scalar_dtype_of(kernel.value(dop.lhs).type), di.lhs_dt, false) << ";\n";
      out << indent_str(ind) << "}\n";
    }

    // Stage rhs into threadgroup memory tg_b.
    int64_t rhs_numel = di.rhs_rows * di.rhs_cols;
    if (materialized_2d.count(dop.rhs) > 0) {
      int inner_ind = ind;
      emit_owned_tile_loop_header(ind, out, inner_ind);
      out << indent_str(inner_ind) << di.tg_b_name << "[m1_row * " << di.rhs_cols << "u + m1_col] = "
          << mat_tile_var.at(dop.rhs) << "[m1_rm * " << rn << "u + m1_rn];\n";
      emit_owned_tile_loop_footer(ind, out);
    } else {
      out << indent_str(ind) << "for (uint m1_s = m1_tid_linear; m1_s < " << rhs_numel << "u; m1_s += " << total_threads
          << "u) {\n";
      out << indent_str(ind + 1) << "uint m1_sr = m1_s / " << di.rhs_cols << "u;\n";
      out << indent_str(ind + 1) << "uint m1_sc = m1_s % " << di.rhs_cols << "u;\n";
      CoordScope sc{"m1_sr", "m1_sc", false, "", ind + 1, &out, {}};
      std::string b_expr = eval_at(dop.rhs, "m1_sr", "m1_sc", sc);
      if (!err.empty()) return false;
      out << indent_str(ind + 1) << di.tg_b_name << "[m1_s] = "
          << msl_cast_expr(b_expr, scalar_dtype_of(kernel.value(dop.rhs).type), di.rhs_dt, false) << ";\n";
      out << indent_str(ind) << "}\n";
    }

    out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

    // Accumulate thread-owned [RM, RN] output entries over k = 0..BK-1.
    std::string rvar = ssa_var(rid);
    mat_tile_var[rid] = rvar;
    out << indent_str(ind) << msl_scalar_type(acc_dt) << " " << rvar << "[" << (rm * rn) << "];\n";

    int inner_ind = ind;
    emit_owned_tile_loop_header(ind, out, inner_ind);
    std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
    std::string init_acc = msl_zero_literal(acc_dt, false);
    if (dop.acc != kInvalidId) {
      CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};
      init_acc =
          msl_cast_expr(eval_at(dop.acc, "m1_row", "m1_col", sc), scalar_dtype_of(kernel.value(dop.acc).type), acc_dt,
                        false);
      if (!err.empty()) return false;
    }
    out << indent_str(inner_ind) << msl_scalar_type(acc_dt) << " m1_acc = " << init_acc << ";\n";
    out << indent_str(inner_ind) << "for (uint m1_k = 0u; m1_k < " << di.bk << "u; ++m1_k) {\n";

    std::string a_elem = di.transpose_a ? (di.tg_a_name + "[m1_k * " + std::to_string(di.bm) + "u + m1_row]")
                                        : (di.tg_a_name + "[m1_row * " + std::to_string(di.bk) + "u + m1_k]");
    std::string b_elem = di.transpose_b ? (di.tg_b_name + "[m1_col * " + std::to_string(di.bk) + "u + m1_k]")
                                        : (di.tg_b_name + "[m1_k * " + std::to_string(di.bn) + "u + m1_col]");
    a_elem = msl_cast_expr(a_elem, di.lhs_dt, acc_dt, false);
    b_elem = msl_cast_expr(b_elem, di.rhs_dt, acc_dt, false);
    out << indent_str(inner_ind + 1) << "m1_acc += " << a_elem << " * " << b_elem << ";\n";
    out << indent_str(inner_ind) << "}\n";
    out << indent_str(inner_ind) << rvar << "[" << owned_idx << "] = m1_acc;\n";
    emit_owned_tile_loop_footer(ind, out);

    out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    return true;
  }

  bool emit_materialized_tile_op(const Op& op, int ind, std::ostringstream& out) {
    ValueId rid = op.results[0];
    ScalarDType rdt = scalar_dtype_of(kernel.value(rid).type);
    std::string rvar = ssa_var(rid);
    mat_tile_var[rid] = rvar;
    out << indent_str(ind) << msl_scalar_type(rdt) << " " << rvar << "[" << (rm * rn) << "];\n";

    int inner_ind = ind;
    emit_owned_tile_loop_header(ind, out, inner_ind);
    std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
    CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};

    std::string val_expr;
    switch (op.kind) {
      case OpKind::Splat:
      case OpKind::Broadcast:
        val_expr = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        break;
      case OpKind::Cast: {
        std::string se = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        val_expr = msl_cast_expr(se, scalar_dtype_of(kernel.value(op.operands[0]).type), rdt, false);
        break;
      }
      case OpKind::Unary: {
        std::string se = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        val_expr = msl_unary_expr(std::get<UnaryAttr>(op.attr).op, se, rdt, false, helpers);
        break;
      }
      case OpKind::Binary: {
        std::string le = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        std::string re = eval_at(op.operands[1], "m1_row", "m1_col", sc);
        val_expr = msl_binary_expr(std::get<BinaryAttr>(op.attr).op, le, re,
                                   scalar_dtype_of(kernel.value(op.operands[0]).type),
                                   scalar_dtype_of(kernel.value(op.operands[1]).type), rdt, false);
        break;
      }
      case OpKind::Select: {
        std::string ce = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        std::string te = eval_at(op.operands[1], "m1_row", "m1_col", sc);
        std::string fe = eval_at(op.operands[2], "m1_row", "m1_col", sc);
        val_expr = msl_select_expr(ce, te, fe, scalar_dtype_of(kernel.value(op.operands[1]).type),
                                   scalar_dtype_of(kernel.value(op.operands[2]).type), rdt, false);
        break;
      }
      case OpKind::Load: {
        auto lo = extract_load_operands(op, kernel);
        std::string pe = eval_at(lo.ptrs, "m1_row", "m1_col", sc);
        if (lo.mask == kInvalidId) {
          val_expr = "(*(" + pe + "))";
        } else {
          std::string me = eval_at(lo.mask, "m1_row", "m1_col", sc);
          std::string oe = (lo.other != kInvalidId)
                               ? msl_cast_expr(eval_at(lo.other, "m1_row", "m1_col", sc),
                                               scalar_dtype_of(kernel.value(lo.other).type), rdt, false)
                               : msl_zero_literal(rdt, false);
          val_expr = "(" + me + " ? (*(" + pe + ")) : " + oe + ")";
        }
        break;
      }
      default:
        err = "Unsupported materialized 2D op in MetalCodegen";
        return false;
    }

    if (!err.empty()) return false;
    out << indent_str(inner_ind) << rvar << "[" << owned_idx << "] = " << val_expr << ";\n";
    emit_owned_tile_loop_footer(ind, out);
    return true;
  }

  struct YieldTarget {
    enum class Kind { Scalar, Tile2D, PtrIter };
    Kind kind = Kind::Scalar;
    std::string var_name;
    ScalarDType dtype = ScalarDType::F32;
    ValueId arg_id = kInvalidId;
  };

  // Detect pointer-tile loop update pattern: yield_val = PtrAdd(iter_arg, scalar_delta).
  bool detect_ptr_iter_delta(ValueId arg_id, ValueId yield_id, ValueId& delta_out) const {
    if (yield_id == arg_id) {
      delta_out = kInvalidId;
      return true;
    }
    const Value& yv = kernel.value(yield_id);
    if (yv.def_op == kInvalidId) return false;
    const Op& yop = kernel.op(yv.def_op);
    if (yop.kind != OpKind::PtrAdd || yop.operands.size() < 2) return false;
    if (yop.operands[0] != arg_id) return false;
    ValueId off = yop.operands[1];
    // Unwrap Splat/Broadcast to find scalar delta.
    while (kernel.value(off).type.is_tile()) {
      const Value& ov = kernel.value(off);
      if (ov.def_op == kInvalidId) return false;
      const Op& oop = kernel.op(ov.def_op);
      if ((oop.kind == OpKind::Splat || oop.kind == OpKind::Broadcast) && !oop.operands.empty()) {
        off = oop.operands[0];
      } else {
        return false;
      }
    }
    delta_out = off;
    return true;
  }

  bool emit_block(BlockId bid, int ind, const std::vector<YieldTarget>* yield_targets, std::ostringstream& out) {
    const Block& blk = kernel.block(bid);
    for (OpId oid : blk.ops) {
      const Op& op = kernel.op(oid);
      if (op.erased || op.kind == OpKind::Return) continue;

      if (op.kind == OpKind::Yield) {
        if (yield_targets == nullptr) continue;
        // Stage scalar and tile yields into temporaries first for simultaneous assignment safety.
        std::vector<std::string> scalar_tmps(yield_targets->size());
        std::vector<std::string> tile_tmps(yield_targets->size());
        for (size_t i = 0; i < op.operands.size() && i < yield_targets->size(); ++i) {
          const auto& yt = (*yield_targets)[i];
          ValueId yv = op.operands[i];
          if (yt.kind == YieldTarget::Kind::Scalar) {
            std::string tmp = "m1_sy_" + std::to_string(oid) + "_" + std::to_string(i);
            const Type& ty = kernel.value(yv).type;
            if (is_ptr_type(ty)) {
              out << indent_str(ind) << msl_ptr_type_str(analysis, yv, scalar_dtype_of(ty)) << " " << tmp << " = "
                  << scalar_exprs.at(yv) << ";\n";
            } else {
              out << indent_str(ind) << msl_scalar_type(yt.dtype) << " " << tmp << " = "
                  << msl_cast_expr(scalar_exprs.at(yv), scalar_dtype_of(ty), yt.dtype, false) << ";\n";
            }
            scalar_tmps[i] = tmp;
          } else if (yt.kind == YieldTarget::Kind::Tile2D) {
            if (materialized_2d.count(yv) > 0 && mat_tile_var.count(yv) > 0) {
              tile_tmps[i] = mat_tile_var.at(yv);
            } else {
              std::string tmp = "m1_ty_" + std::to_string(oid) + "_" + std::to_string(i);
              out << indent_str(ind) << msl_scalar_type(yt.dtype) << " " << tmp << "[" << (rm * rn) << "];\n";
              int inner_ind = ind;
              emit_owned_tile_loop_header(ind, out, inner_ind);
              std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
              CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};
              std::string ev =
                  msl_cast_expr(eval_at(yv, "m1_row", "m1_col", sc), scalar_dtype_of(kernel.value(yv).type), yt.dtype,
                                false);
              if (!err.empty()) return false;
              out << indent_str(inner_ind) << tmp << "[" << owned_idx << "] = " << ev << ";\n";
              emit_owned_tile_loop_footer(ind, out);
              tile_tmps[i] = tmp;
            }
          }
        }
        for (size_t i = 0; i < op.operands.size() && i < yield_targets->size(); ++i) {
          const auto& yt = (*yield_targets)[i];
          if (yt.kind == YieldTarget::Kind::Scalar) {
            out << indent_str(ind) << yt.var_name << " = " << scalar_tmps[i] << ";\n";
          } else if (yt.kind == YieldTarget::Kind::Tile2D) {
            if (tile_tmps[i] != yt.var_name) {
              out << indent_str(ind) << "for (uint m1_i = 0u; m1_i < " << (rm * rn) << "u; ++m1_i) { " << yt.var_name
                  << "[m1_i] = " << tile_tmps[i] << "[m1_i]; }\n";
            }
          } else if (yt.kind == YieldTarget::Kind::PtrIter) {
            const auto& pi = ptr_iters.at(yt.arg_id);
            if (pi.delta_val != kInvalidId) {
              out << indent_str(ind) << pi.poff_var << " += long(" << scalar_exprs.at(pi.delta_val) << ");\n";
            }
          }
        }
        continue;
      }

      if (op.kind == OpKind::Barrier) {
        BarrierScope scope =
            std::holds_alternative<BarrierAttr>(op.attr) ? std::get<BarrierAttr>(op.attr).scope : BarrierScope::Threadgroup;
        if (scope == BarrierScope::Simdgroup) {
          out << indent_str(ind) << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
        } else if (scope == BarrierScope::DeviceMem) {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_device);\n";
        } else {
          out << indent_str(ind) << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        }
        continue;
      }

      if (op.kind == OpKind::Dot) {
        if (!emit_dot_op(op, ind, out)) return false;
        continue;
      }

      if (op.kind == OpKind::ForLoop) {
        if (op.operands.size() < 3 || op.regions.empty()) {
          err = "Malformed ForLoop op in 2D MetalCodegen";
          return false;
        }
        const Block& body = kernel.block(op.regions[0]);
        if (body.args.empty()) {
          err = "ForLoop body block missing iv arg";
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
        bool neg_step = false;
        if (auto it = analysis.const_i64.find(op.operands[2]); it != analysis.const_i64.end() && it->second < 0) {
          neg_step = true;
        }

        // Locate Yield op in body to detect pointer-tile increment patterns.
        const Op* yield_op = nullptr;
        for (OpId boid : body.ops) {
          const Op& bop = kernel.op(boid);
          if (!bop.erased && bop.kind == OpKind::Yield) {
            yield_op = &bop;
            break;
          }
        }

        size_t num_iter = op.operands.size() - 3;
        std::vector<YieldTarget> ytargets;
        ytargets.reserve(num_iter);

        for (size_t i = 0; i < num_iter; ++i) {
          ValueId arg_id = body.args[1 + i];
          ValueId init_id = op.operands[3 + i];
          const Type& aty = kernel.value(arg_id).type;
          std::string avar = ssa_var(arg_id);

          if (!aty.is_tile()) {
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
            ytargets.push_back(YieldTarget{YieldTarget::Kind::Scalar, avar, adt, arg_id});
          } else if (is_ptr_type(aty)) {
            ValueId delta_v = kInvalidId;
            if (yield_op == nullptr || i >= yield_op->operands.size() ||
                !detect_ptr_iter_delta(arg_id, yield_op->operands[i], delta_v)) {
              err = "Pointer-tile ForLoop iter_arg must be updated by a scalar PtrAdd increment";
              return false;
            }
            std::string poff = "m1_poff_" + std::to_string(arg_id);
            out << indent_str(ind) << "long " << poff << " = 0ll;\n";
            ptr_iters[arg_id] = PtrIterInfo{init_id, poff, delta_v};
            if (i < op.results.size()) {
              ptr_iters[op.results[i]] = PtrIterInfo{init_id, poff, kInvalidId};
            }
            ytargets.push_back(YieldTarget{YieldTarget::Kind::PtrIter, poff, ScalarDType::I64, arg_id});
          } else {
            const auto& ashape = aty.as_tile().shape;
            if (ashape.size() != 2 || ashape[0] != bm || ashape[1] != bn) {
              err = "ForLoop 2D tile iter_arg shape must match output tile shape [BM, BN]";
              return false;
            }
            ScalarDType adt = scalar_dtype_of(aty);
            out << indent_str(ind) << msl_scalar_type(adt) << " " << avar << "[" << (rm * rn) << "];\n";
            if (materialized_2d.count(init_id) > 0 && mat_tile_var.count(init_id) > 0) {
              out << indent_str(ind) << "for (uint m1_i = 0u; m1_i < " << (rm * rn) << "u; ++m1_i) { " << avar
                  << "[m1_i] = " << mat_tile_var.at(init_id) << "[m1_i]; }\n";
            } else {
              int inner_ind = ind;
              emit_owned_tile_loop_header(ind, out, inner_ind);
              std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
              CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};
              std::string ev =
                  msl_cast_expr(eval_at(init_id, "m1_row", "m1_col", sc), scalar_dtype_of(kernel.value(init_id).type),
                                adt, false);
              if (!err.empty()) return false;
              out << indent_str(inner_ind) << avar << "[" << owned_idx << "] = " << ev << ";\n";
              emit_owned_tile_loop_footer(ind, out);
            }
            mat_tile_var[arg_id] = avar;
            if (i < op.results.size()) mat_tile_var[op.results[i]] = avar;
            ytargets.push_back(YieldTarget{YieldTarget::Kind::Tile2D, avar, adt, arg_id});
          }
        }

        out << indent_str(ind) << "for (" << msl_scalar_type(iv_dt) << " " << iv_var << " = " << lo_e << "; " << iv_var
            << (neg_step ? " > " : " < ") << hi_e << "; " << iv_var << " += " << step_e << ") {\n";
        scalar_exprs[iv_id] = iv_var;
        if (!emit_block(op.regions[0], ind + 1, &ytargets, out)) return false;
        out << indent_str(ind) << "}\n";

        for (size_t i = 0; i < op.results.size() && i < ytargets.size(); ++i) {
          if (ytargets[i].kind == YieldTarget::Kind::Scalar) {
            scalar_exprs[op.results[i]] = ytargets[i].var_name;
          } else if (ytargets[i].kind == YieldTarget::Kind::Tile2D) {
            mat_tile_var[op.results[i]] = ytargets[i].var_name;
          }
        }
        continue;
      }

      if (op.kind == OpKind::If) {
        if (op.operands.empty() || op.regions.empty()) {
          err = "Malformed If op in 2D MetalCodegen";
          return false;
        }
        std::vector<YieldTarget> ytargets;
        for (ValueId rid : op.results) {
          const Type& rty = kernel.value(rid).type;
          std::string rv = ssa_var(rid);
          ScalarDType rdt = scalar_dtype_of(rty);
          if (!rty.is_tile()) {
            if (is_ptr_type(rty)) {
              out << indent_str(ind) << msl_ptr_type_str(analysis, rid, rdt) << " " << rv << " = nullptr;\n";
            } else {
              out << indent_str(ind) << msl_scalar_type(rdt) << " " << rv << " = " << msl_zero_literal(rdt, false)
                  << ";\n";
            }
            scalar_exprs[rid] = rv;
            ytargets.push_back(YieldTarget{YieldTarget::Kind::Scalar, rv, rdt, rid});
          } else {
            const auto& sh = rty.as_tile().shape;
            if (is_ptr_type(rty) || sh.size() != 2 || sh[0] != bm || sh[1] != bn) {
              err = "Unsupported tile result type on 2D If in MetalCodegen";
              return false;
            }
            out << indent_str(ind) << msl_scalar_type(rdt) << " " << rv << "[" << (rm * rn) << "];\n";
            mat_tile_var[rid] = rv;
            ytargets.push_back(YieldTarget{YieldTarget::Kind::Tile2D, rv, rdt, rid});
          }
        }
        out << indent_str(ind) << "if (" << scalar_exprs.at(op.operands[0]) << ") {\n";
        if (!emit_block(op.regions[0], ind + 1, &ytargets, out)) return false;
        if (op.regions.size() >= 2 && op.regions[1] != kInvalidId) {
          out << indent_str(ind) << "} else {\n";
          if (!emit_block(op.regions[1], ind + 1, &ytargets, out)) return false;
        }
        out << indent_str(ind) << "}\n";
        continue;
      }

      if (op.kind == OpKind::Store) {
        auto so = extract_store_operands(op);
        const Type& pty = kernel.value(so.ptrs).type;
        ScalarDType pdt = scalar_dtype_of(pty);
        if (!pty.is_tile()) {
          std::string pe = scalar_exprs.at(so.ptrs);
          std::string ve =
              msl_cast_expr(scalar_exprs.at(so.val), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          std::string guard = "(m1_tid_linear == 0u)";
          if (so.mask != kInvalidId) guard = "(" + guard + " && (" + scalar_exprs.at(so.mask) + "))";
          out << indent_str(ind) << "if (" << guard << ") { *(" << pe << ") = " << ve << "; }\n";
          continue;
        }
        const auto& sshape = pty.as_tile().shape;
        if (sshape.size() == 2 && sshape[0] == bm && sshape[1] == bn) {
          int inner_ind = ind;
          emit_owned_tile_loop_header(ind, out, inner_ind);
          std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
          CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};
          std::string pe = eval_at(so.ptrs, "m1_row", "m1_col", sc);
          std::string ve = msl_cast_expr(eval_at(so.val, "m1_row", "m1_col", sc),
                                         scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_at(so.mask, "m1_row", "m1_col", sc);
            if (!err.empty()) return false;
            out << indent_str(inner_ind) << "if (" << me << ") {\n";
            out << indent_str(inner_ind + 1) << "*(" << pe << ") = " << ve << ";\n";
            out << indent_str(inner_ind) << "}\n";
          } else {
            out << indent_str(inner_ind) << "*(" << pe << ") = " << ve << ";\n";
          }
          emit_owned_tile_loop_footer(ind, out);
        } else if (sshape.size() == 1) {
          uint32_t total_threads = tg_x * tg_y;
          out << indent_str(ind) << "for (uint m1_s = m1_tid_linear; m1_s < " << sshape[0]
              << "u; m1_s += " << total_threads << "u) {\n";
          CoordScope sc{"0u", "m1_s", false, "", ind + 1, &out, {}};
          std::string pe = eval_at(so.ptrs, "0u", "m1_s", sc);
          std::string ve =
              msl_cast_expr(eval_at(so.val, "0u", "m1_s", sc), scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_at(so.mask, "0u", "m1_s", sc);
            if (!err.empty()) return false;
            out << indent_str(ind + 1) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
          } else {
            out << indent_str(ind + 1) << "*(" << pe << ") = " << ve << ";\n";
          }
          out << indent_str(ind) << "}\n";
        } else if (sshape.size() == 2 && materialized_2d.count(so.val) == 0) {
          uint32_t total_threads = tg_x * tg_y;
          int64_t total_elems = sshape[0] * sshape[1];
          out << indent_str(ind) << "for (uint m1_s = m1_tid_linear; m1_s < " << total_elems
              << "u; m1_s += " << total_threads << "u) {\n";
          out << indent_str(ind + 1) << "uint m1_sr = m1_s / " << sshape[1] << "u;\n";
          out << indent_str(ind + 1) << "uint m1_sc = m1_s % " << sshape[1] << "u;\n";
          CoordScope sc{"m1_sr", "m1_sc", false, "", ind + 1, &out, {}};
          std::string pe = eval_at(so.ptrs, "m1_sr", "m1_sc", sc);
          std::string ve = msl_cast_expr(eval_at(so.val, "m1_sr", "m1_sc", sc),
                                         scalar_dtype_of(kernel.value(so.val).type), pdt, false);
          if (!err.empty()) return false;
          if (so.mask != kInvalidId) {
            std::string me = eval_at(so.mask, "m1_sr", "m1_sc", sc);
            if (!err.empty()) return false;
            out << indent_str(ind + 1) << "if (" << me << ") { *(" << pe << ") = " << ve << "; }\n";
          } else {
            out << indent_str(ind + 1) << "*(" << pe << ") = " << ve << ";\n";
          }
          out << indent_str(ind) << "}\n";
        } else {
          err = "Store 2D tile shape does not match primary output tile shape [BM, BN]";
          return false;
        }
        continue;
      }

      if (op.kind == OpKind::AtomicAdd && !op.operands.empty()) {
        const Type& pty = kernel.value(op.operands[0]).type;
        ScalarDType pdt = scalar_dtype_of(pty);
        std::string atomic_ty = (pdt == ScalarDType::F32) ? "atomic_float"
                                : (pdt == ScalarDType::I32) ? "atomic_int"
                                : (pdt == ScalarDType::U32) ? "atomic_uint"
                                                            : "";
        if (atomic_ty.empty()) {
          err = "Unsupported AtomicAdd dtype in 2D MetalCodegen";
          return false;
        }
        if (!pty.is_tile()) {
          ValueId rid = op.results.empty() ? kInvalidId : op.results[0];
          std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
          if (!err.empty()) return false;
          std::string vname = (rid != kInvalidId) ? ssa_var(rid) : ("m1_atom_" + std::to_string(oid));
          out << indent_str(ind) << msl_scalar_type(pdt) << " " << vname << " = " << msl_zero_literal(pdt, false)
              << ";\n";
          out << indent_str(ind) << "if (m1_tid_linear == 0u) { " << vname << " = " << expr << "; }\n";
          if (rid != kInvalidId) scalar_exprs[rid] = vname;
          continue;
        }
        ValueId rid = op.results.empty() ? kInvalidId : op.results[0];
        std::string rvar = (rid != kInvalidId) ? ssa_var(rid) : "";
        if (rid != kInvalidId && materialized_2d.count(rid) > 0) {
          mat_tile_var[rid] = rvar;
          out << indent_str(ind) << msl_scalar_type(pdt) << " " << rvar << "[" << (rm * rn) << "];\n";
        }
        int inner_ind = ind;
        emit_owned_tile_loop_header(ind, out, inner_ind);
        std::string owned_idx = "m1_rm * " + std::to_string(rn) + "u + m1_rn";
        CoordScope sc{"m1_row", "m1_col", true, owned_idx, inner_ind, &out, {}};
        std::string pe = eval_at(op.operands[0], "m1_row", "m1_col", sc);
        std::string ve = msl_cast_expr(eval_at(op.operands[1], "m1_row", "m1_col", sc),
                                       scalar_dtype_of(kernel.value(op.operands[1]).type), pdt, false);
        std::string call = "atomic_fetch_add_explicit(reinterpret_cast<device " + atomic_ty + "*>(" + pe + "), " + ve +
                           ", memory_order_relaxed)";
        std::string full_e = call;
        if (op.operands.size() >= 3 && op.operands[2] != kInvalidId) {
          std::string me = eval_at(op.operands[2], "m1_row", "m1_col", sc);
          full_e = "(" + me + " ? " + call + " : " + msl_zero_literal(pdt, false) + ")";
        }
        if (!err.empty()) return false;
        if (rid != kInvalidId && materialized_2d.count(rid) > 0) {
          out << indent_str(inner_ind) << rvar << "[" << owned_idx << "] = " << full_e << ";\n";
        } else {
          out << indent_str(inner_ind) << "(void)" << full_e << ";\n";
        }
        emit_owned_tile_loop_footer(ind, out);
        continue;
      }

      if (op.results.empty()) {
        err = "Unsupported void op in 2D MetalCodegen";
        return false;
      }

      ValueId rid = op.results[0];
      const Type& rty = kernel.value(rid).type;

      // Pure scalar op: emit directly into current block scope.
      if (!rty.is_tile()) {
        if (op.kind == OpKind::Constant) {
          scalar_exprs[rid] = msl_const_literal(std::get<ConstAttr>(op.attr), scalar_dtype_of(rty));
          continue;
        }
        std::string expr = emit_scalar_op_expr(kernel, analysis, op, scalar_exprs, helpers, err);
        if (!err.empty()) return false;
        std::string vname = ssa_var(rid);
        if (is_ptr_type(rty)) {
          out << indent_str(ind) << msl_ptr_type_str(analysis, rid, scalar_dtype_of(rty)) << " " << vname << " = "
              << expr << ";\n";
        } else {
          out << indent_str(ind) << msl_scalar_type(scalar_dtype_of(rty)) << " " << vname << " = " << expr << ";\n";
        }
        scalar_exprs[rid] = vname;
        continue;
      }

      // Materialized 2D tile op: emit per-thread register tile computation.
      if (materialized_2d.count(rid) > 0) {
        if (!emit_materialized_tile_op(op, ind, out)) return false;
        continue;
      }

      // Lazy 1D/2D tile expression: evaluated on demand at exact coordinates.
    }
    return true;
  }
};

uint32_t largest_pow2_le(uint32_t n) {
  uint32_t p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

std::pair<uint32_t, uint32_t> select_2d_tg_dims(int64_t bm, int64_t bn, const KernelAnalysis& analysis,
                                                const TargetSpec& target) {
  if (analysis.hint_layout.has_value() &&
      (analysis.hint_layout->tg_threads[0] > 1 || analysis.hint_layout->tg_threads[1] > 1)) {
    uint32_t tx = std::max<uint32_t>(1u, analysis.hint_layout->tg_threads[0]);
    uint32_t ty = std::max<uint32_t>(1u, analysis.hint_layout->tg_threads[1]);
    if (tx * ty <= target.max_threads_per_threadgroup) {
      return {tx, ty};
    }
  }

  uint32_t max_x = std::min<uint32_t>(static_cast<uint32_t>(std::max<int64_t>(1, bn)), 16u);
  uint32_t max_y = std::min<uint32_t>(static_cast<uint32_t>(std::max<int64_t>(1, bm)), 16u);
  uint32_t tx = largest_pow2_le(max_x);
  uint32_t ty = largest_pow2_le(max_y);

  // Ensure tg_x * tg_y is a multiple of 32 and <= 256.
  while (tx * ty < 32u) {
    if (tx < 32u && (tx <= ty || ty >= static_cast<uint32_t>(bm))) {
      tx <<= 1;
    } else {
      ty <<= 1;
    }
  }
  while (tx * ty > 256u) {
    if (ty > tx) {
      ty >>= 1;
    } else {
      tx >>= 1;
    }
  }
  return {tx, ty};
}

CompiledMSL emit_2d_tiled(const Kernel& kernel, const TargetSpec& target, const KernelAnalysis& analysis) {
  int64_t bm = analysis.primary_2d_shape.first;
  int64_t bn = analysis.primary_2d_shape.second;
  if (bm <= 0 || bn <= 0) {
    return CompiledMSL{false, "", {}, "Invalid 2D tile dimensions in MetalCodegen"};
  }

  auto [tg_x, tg_y] = select_2d_tg_dims(bm, bn, analysis, target);
  uint32_t rm = static_cast<uint32_t>((bm + tg_y - 1) / tg_y);
  uint32_t rn = static_cast<uint32_t>((bn + tg_x - 1) / tg_x);

  Tiled2DEmitter em{
      .kernel = kernel,
      .analysis = analysis,
      .bm = bm,
      .bn = bn,
      .tg_x = tg_x,
      .tg_y = tg_y,
      .rm = rm,
      .rn = rn,
      .exact_m = (static_cast<int64_t>(rm) * tg_y == bm),
      .exact_n = (static_cast<int64_t>(rn) * tg_x == bn),
  };

  std::vector<BlockId> all_blocks;
  std::unordered_set<BlockId> seen_blocks;
  collect_blocks_dfs(kernel, kernel.entry_block, all_blocks, seen_blocks);

  // Collect all Dot ops, validate shapes, and check threadgroup SRAM budget.
  std::vector<OpId> dot_ops;
  for (BlockId bid : all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (!op.erased && op.kind == OpKind::Dot) dot_ops.push_back(oid);
    }
  }

  uint32_t total_tg_bytes = 0;
  std::ostringstream tg_decls;
  for (size_t idx = 0; idx < dot_ops.size(); ++idx) {
    OpId oid = dot_ops[idx];
    const Op& op = kernel.op(oid);
    auto dop = extract_dot_operands(op);
    if (dop.lhs == kInvalidId || dop.rhs == kInvalidId || op.results.empty()) {
      return CompiledMSL{false, "", {}, "Malformed Dot op"};
    }
    const Type& lty = kernel.value(dop.lhs).type;
    const Type& rty = kernel.value(dop.rhs).type;
    const Type& oty = kernel.value(op.results[0]).type;
    if (!lty.is_tile() || !rty.is_tile() || !oty.is_tile() || lty.as_tile().shape.size() != 2 ||
        rty.as_tile().shape.size() != 2 || oty.as_tile().shape.size() != 2) {
      return CompiledMSL{false, "", {}, "Dot operands and result must be rank-2 tiles"};
    }

    DotBufferInfo di;
    di.dot_op = oid;
    di.lhs_dt = scalar_dtype_of(lty);
    di.rhs_dt = scalar_dtype_of(rty);
    di.lhs_rows = lty.as_tile().shape[0];
    di.lhs_cols = lty.as_tile().shape[1];
    di.rhs_rows = rty.as_tile().shape[0];
    di.rhs_cols = rty.as_tile().shape[1];
    di.transpose_a = dop.attr.transpose_a;
    di.transpose_b = dop.attr.transpose_b;
    di.bm = di.transpose_a ? di.lhs_cols : di.lhs_rows;
    int64_t bk_a = di.transpose_a ? di.lhs_rows : di.lhs_cols;
    int64_t bk_b = di.transpose_b ? di.rhs_cols : di.rhs_rows;
    di.bn = di.transpose_b ? di.rhs_rows : di.rhs_cols;

    if (bk_a != bk_b || di.bm != oty.as_tile().shape[0] || di.bn != oty.as_tile().shape[1]) {
      return CompiledMSL{false, "", {}, "Dot tile dimensions mismatch"};
    }
    if (di.bm != bm || di.bn != bn) {
      return CompiledMSL{false, "", {}, "Multiple distinct Dot output tile shapes are not supported in one kernel"};
    }
    di.bk = bk_a;
    di.tg_a_name = (dot_ops.size() == 1) ? "tg_a" : ("tg_a_" + std::to_string(idx));
    di.tg_b_name = (dot_ops.size() == 1) ? "tg_b" : ("tg_b_" + std::to_string(idx));

    uint64_t bytes_a = static_cast<uint64_t>(di.lhs_rows * di.lhs_cols) * local_dtype_bytes(di.lhs_dt);
    uint64_t bytes_b = static_cast<uint64_t>(di.rhs_rows * di.rhs_cols) * local_dtype_bytes(di.rhs_dt);
    uint64_t pair_bytes = bytes_a + bytes_b;
    if (pair_bytes > target.max_threadgroup_memory_bytes ||
        total_tg_bytes + pair_bytes > target.max_threadgroup_memory_bytes) {
      return CompiledMSL{
          false,
          "",
          {},
          "Threadgroup memory requirement (" + std::to_string(total_tg_bytes + pair_bytes) +
              " bytes) exceeds target limit (" + std::to_string(target.max_threadgroup_memory_bytes) + " bytes)"};
    }
    total_tg_bytes += static_cast<uint32_t>(pair_bytes);
    tg_decls << "  threadgroup " << msl_scalar_type(di.lhs_dt) << " " << di.tg_a_name << "["
             << (di.lhs_rows * di.lhs_cols) << "];\n";
    tg_decls << "  threadgroup " << msl_scalar_type(di.rhs_dt) << " " << di.tg_b_name << "["
             << (di.rhs_rows * di.rhs_cols) << "];\n";
    em.dot_infos[oid] = std::move(di);
  }

  // Mark materialized 2D [BM, BN] tiles via fixed-point propagation.
  auto is_primary_data_tile = [&](ValueId vid) -> bool {
    const Type& ty = kernel.value(vid).type;
    if (!ty.is_tile() || is_ptr_type(ty)) return false;
    const auto& sh = ty.as_tile().shape;
    return sh.size() == 2 && sh[0] == bm && sh[1] == bn;
  };

  size_t mem_write_ops = 0;
  for (BlockId bid : all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if (op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) {
        ++mem_write_ops;
      }
    }
  }

  for (BlockId bid : all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if (op.kind == OpKind::Dot && !op.results.empty()) {
        em.materialized_2d.insert(op.results[0]);
      } else if (op.kind == OpKind::Load && mem_write_ops >= 2 && !op.results.empty() &&
                 is_primary_data_tile(op.results[0])) {
        em.materialized_2d.insert(op.results[0]);
      } else if (op.kind == OpKind::ForLoop && !op.regions.empty()) {
        const Block& body = kernel.block(op.regions[0]);
        for (size_t i = 1; i < body.args.size(); ++i) {
          if (is_primary_data_tile(body.args[i])) em.materialized_2d.insert(body.args[i]);
        }
        for (ValueId r : op.results) {
          if (is_primary_data_tile(r)) em.materialized_2d.insert(r);
        }
      } else if (op.kind == OpKind::If) {
        for (ValueId r : op.results) {
          if (is_primary_data_tile(r)) em.materialized_2d.insert(r);
        }
      } else if (op.kind == OpKind::AtomicAdd && !op.results.empty() && is_primary_data_tile(op.results[0])) {
        em.materialized_2d.insert(op.results[0]);
      }
    }
  }

  bool mat_changed = true;
  while (mat_changed) {
    mat_changed = false;
    for (BlockId bid : all_blocks) {
      for (OpId oid : kernel.block(bid).ops) {
        const Op& op = kernel.op(oid);
        if (op.erased || op.results.empty()) continue;
        ValueId rid = op.results[0];
        if (!is_primary_data_tile(rid) || em.materialized_2d.count(rid) > 0) continue;
        for (ValueId o : op.operands) {
          if (o != kInvalidId && em.materialized_2d.count(o) > 0) {
            em.materialized_2d.insert(rid);
            mat_changed = true;
            break;
          }
        }
      }
    }
  }

  populate_entry_param_exprs(kernel, analysis, em.scalar_exprs);

  std::ostringstream body;
  body << "  (void)m1_grid_dim; (void)m1_tg_size;\n";
  if (dot_ops.empty()) body << "  (void)m1_tid_linear;\n";
  body << tg_decls.str();

  if (!em.emit_block(kernel.entry_block, 1, nullptr, body)) {
    return CompiledMSL{false, "", {}, em.err};
  }
  body << "}\n";

  CompiledMSL res;
  res.ok = true;
  res.msl_source = build_msl_preamble_and_sig(kernel, analysis, em.helpers) + body.str();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = tg_x;
  res.launch_spec.tg_y = tg_y;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = total_tg_bytes;
  res.launch_spec.vec_width = 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = analysis.param_slots;
  return res;
}

}  // namespace

CompiledMSL MetalCodegen::emit(const ir::Kernel& kernel, const ir::TargetSpec& target) {
  KernelAnalysis analysis;
  std::string err;
  if (!analyze_kernel(kernel, analysis, err)) {
    return CompiledMSL{false, "", {}, err};
  }

  if (analysis.max_rank > 2) {
    return CompiledMSL{false, "", {}, "Tiles with rank > 2 are not supported in MetalCodegen"};
  }

  if (analysis.has_norm_softmax) {
    bool matched = false;
    auto ns_res = try_emit_norm_softmax_kernel(kernel, target, &matched);
    if (matched) {
      return ns_res;
    }
  }

  if (analysis.has_pool) {
    bool matched = false;
    auto pool_res = try_emit_pool_kernel(kernel, target, &matched);
    if (matched) {
      return pool_res;
    }
  }

  if (analysis.has_row_scan) {
    bool matched = false;
    auto scan_res = try_emit_row_scan_kernel(kernel, target, &matched);
    if (matched) {
      return scan_res;
    }
  }

  if (analysis.has_reduce || analysis.has_scan) {
    bool matched = false;
    auto red_res = try_emit_reduce_scan_kernel(kernel, target, &matched);
    if (matched) {
      return red_res;
    }
  }

  if (analysis.has_dot) {
    bool use_sgmm = kernel.hints.tile_spec.has_value();
    if (!use_sgmm) {
      for (const auto& op : kernel.ops) {
        if (!op->erased && op->kind == OpKind::Dot && !op->results.empty()) {
          const Type& rty = kernel.value(op->results[0]).type;
          if (rty.is_tile() && rty.as_tile().layout.kind == ir::LayoutKind::SimdgroupFragment) {
            use_sgmm = true;
            break;
          }
        }
      }
    }
    if (use_sgmm) {
      bool matched = false;
      auto sgmm_res = try_emit_sgmm_kernel(kernel, target, &matched);
      if (matched) {
        return sgmm_res;
      }
    }
  }

  if (analysis.has_dot || analysis.max_rank == 2) {
    return emit_2d_tiled(kernel, target, analysis);
  }

  if (auto vec4_res = try_emit_1d_vec4(kernel, target, analysis)) {
    return *vec4_res;
  }

  return emit_1d_general(kernel, target, analysis);
}

}  // namespace m1triton::codegen::metal

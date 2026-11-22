#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

namespace m1triton::codegen::metal {

using ir::ArangeAttr;
using ir::AxisAttr;
using ir::BinaryAttr;
using ir::BinaryOp;
using ir::Block;
using ir::BlockId;
using ir::ConstAttr;
using ir::Kernel;
using ir::KernelParam;
using ir::LoadStoreAttr;
using ir::Op;
using ir::OpId;
using ir::OpKind;
using ir::ScalarDType;
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

ScalarDType common_cmp_dtype(ScalarDType a, ScalarDType b) {
  if (a == b) return a;
  if (a == ScalarDType::F32 || b == ScalarDType::F32) return ScalarDType::F32;
  if (a == ScalarDType::F16 || b == ScalarDType::F16) return ScalarDType::F16;
  if (a == ScalarDType::I64 || b == ScalarDType::I64) return ScalarDType::I64;
  if (a == ScalarDType::U32 || b == ScalarDType::U32) return ScalarDType::U32;
  return ScalarDType::I32;
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

std::string msl_int_literal(int64_t v) {
  if (v >= std::numeric_limits<int32_t>::min() && v <= std::numeric_limits<int32_t>::max()) {
    return std::to_string(v);
  }
  return std::to_string(v) + "L";
}

std::string msl_exact_float_literal(double v) {
  const float f = static_cast<float>(v);
  char buf[64];
  for (int prec = 6; prec <= 9; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*e", prec, static_cast<double>(f));
    if (std::strtof(buf, nullptr) == f) break;
  }
  return std::string(buf) + "f";
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
  if (v == 0) return 32u;
  return ((v + 31u) / 32u) * 32u;
}

uint32_t fit_divisor(uint32_t threads, uint32_t block) {
  threads = std::max<uint32_t>(1, std::min(threads, block));
  while (block % threads != 0) --threads;
  return threads;
}

void emit_helper_functions(std::ostringstream& os, const HelperFlags& helpers) {
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
}

void emit_param_slot_decls(std::ostringstream& os, const std::vector<ParamSlot>& param_slots) {
  for (const auto& slot : param_slots) {
    if (slot.is_pointer) {
      os << "    device " << (slot.writable ? "" : "const ") << msl_scalar_type(slot.dtype) << "* " << slot.name
         << " [[buffer(" << slot.buffer_index << ")]],\n";
    } else {
      os << "    constant " << msl_scalar_type(slot.dtype) << "& " << slot.name << " [[buffer(" << slot.buffer_index
         << ")]],\n";
    }
  }
}

std::string build_msl_preamble_and_sig(const Kernel& kernel, const std::vector<ParamSlot>& param_slots,
                                       const HelperFlags& helpers, bool include_simd_ids) {
  std::ostringstream os;
  os << "#include <metal_stdlib>\n";
  os << "using namespace metal;\n\n";
  emit_helper_functions(os, helpers);

  os << "kernel void " << kernel.name << "(\n";
  emit_param_slot_decls(os, param_slots);
  os << "    uint3 m1_tg_pos [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint3 m1_tid [[thread_position_in_threadgroup]],\n"
     << "    uint3 m1_tg_size [[threads_per_threadgroup]],\n";
  if (include_simd_ids) {
    os << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
       << "    uint m1_simd_lane [[thread_index_in_simdgroup]],\n"
       << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]]) {\n";
  } else {
    os << "    uint m1_tid_linear [[thread_index_in_threadgroup]]) {\n";
  }
  return os.str();
}

void emit_kernel_param_decls(std::ostringstream& os, const Kernel& kernel, std::string_view out_name) {
  os << "kernel void " << kernel.name << "(\n";
  uint32_t slot = 0;
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr) continue;
    if (kp.type.is_ptr()) {
      const auto pt = std::get<ir::PtrType>(kp.type.value);
      const bool is_out = kp.name == out_name;
      os << "    device " << (is_out ? "" : "const ") << msl_scalar_type(pt.pointee) << "* " << kp.name;
    } else {
      os << "    constant " << msl_scalar_type(std::get<ScalarDType>(kp.type.value)) << "& " << kp.name;
    }
    os << " [[buffer(" << slot++ << ")]],\n";
  }
}

std::vector<ParamSlot> build_single_out_param_slots(const Kernel& kernel, std::string_view out_name) {
  std::vector<ParamSlot> slots;
  uint32_t slot = 0;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const KernelParam& kp = kernel.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot ps;
    ps.param_index = i;
    ps.buffer_index = slot++;
    ps.name = kp.name;
    ps.is_pointer = kp.type.is_ptr();
    ps.writable = ps.is_pointer && (kp.writable || kp.name == out_name);
    ps.dtype = ps.is_pointer ? std::get<ir::PtrType>(kp.type.value).pointee : std::get<ScalarDType>(kp.type.value);
    slots.push_back(std::move(ps));
  }
  return slots;
}

std::string msl_prefixed_param_name(const KernelParam& kp, uint32_t idx) {
  return "m1p_" + (kp.name.empty() ? ("arg" + std::to_string(idx)) : kp.name);
}

std::string emit_prefixed_kernel_sig(const Kernel& kernel, const std::vector<ParamSlot>& slots) {
  std::ostringstream os;
  os << "kernel void " << kernel.name << "(\n";
  for (const auto& slot : slots) {
    const std::string name = msl_prefixed_param_name(kernel.params[slot.param_index], slot.param_index);
    if (slot.is_pointer) {
      os << "    device " << (slot.writable ? "" : "const ") << msl_scalar_type(slot.dtype) << "* " << name;
    } else {
      os << "    constant " << msl_scalar_type(slot.dtype) << "& " << name;
    }
    os << " [[buffer(" << slot.buffer_index << ")]],\n";
  }
  os << "    uint3 m1_tgp [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_ntg [[threadgroups_per_grid]],\n"
     << "    uint m1_tid [[thread_index_in_threadgroup]],\n"
     << "    uint m1_lane [[thread_index_in_simdgroup]],\n"
     << "    uint m1_sg [[simdgroup_index_in_threadgroup]]) {\n";
  return os.str();
}

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

void map_entry_args_to_params(const Kernel& kernel, std::unordered_map<ValueId, uint32_t>& val_to_param_idx) {
  const Block& entry = kernel.block(kernel.entry_block);
  std::vector<uint32_t> non_constexpr_indices;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    if (!kernel.params[i].is_constexpr) {
      non_constexpr_indices.push_back(i);
    }
  }

  if (entry.args.size() == kernel.params.size()) {
    for (uint32_t i = 0; i < kernel.params.size(); ++i) {
      val_to_param_idx[entry.args[i]] = i;
    }
  } else if (entry.args.size() == non_constexpr_indices.size()) {
    for (size_t k = 0; k < non_constexpr_indices.size(); ++k) {
      val_to_param_idx[entry.args[k]] = non_constexpr_indices[k];
    }
  } else {
    for (ValueId vid : entry.args) {
      const Value& v = kernel.value(vid);
      if (v.arg_idx < kernel.params.size()) {
        val_to_param_idx[vid] = v.arg_idx;
      } else {
        for (uint32_t i = 0; i < kernel.params.size(); ++i) {
          if (!v.debug_name.empty() && kernel.params[i].name == v.debug_name) {
            val_to_param_idx[vid] = i;
            break;
          }
        }
      }
    }
  }
}

void build_param_slots(const Kernel& kernel, const std::unordered_set<uint32_t>& written_params,
                       std::vector<ParamSlot>& param_slots,
                       std::unordered_map<uint32_t, std::string>& param_msl_name) {
  uint32_t next_slot = 0;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const KernelParam& kp = kernel.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot slot;
    slot.param_index = i;
    slot.buffer_index = next_slot++;
    slot.name = kp.name.empty() ? ("m1_arg" + std::to_string(slot.buffer_index)) : kp.name;
    slot.is_pointer = kp.type.is_ptr();
    slot.writable = slot.is_pointer && (kp.writable || written_params.count(i) > 0);
    slot.dtype = scalar_dtype_of(kp.type);
    param_msl_name[i] = slot.name;
    param_slots.push_back(std::move(slot));
  }
}

bool ptr_value_is_writable(const std::unordered_map<ValueId, std::set<uint32_t>>& ptr_origins,
                           const std::vector<ParamSlot>& param_slots, ValueId vid) {
  auto it = ptr_origins.find(vid);
  if (it == ptr_origins.end() || it->second.empty()) return false;
  for (uint32_t pidx : it->second) {
    bool slot_writable = false;
    for (const auto& slot : param_slots) {
      if (slot.param_index == pidx) {
        slot_writable = slot.writable;
        break;
      }
    }
    if (!slot_writable) return false;
  }
  return true;
}

std::string msl_ptr_type_str(const std::unordered_map<ValueId, std::set<uint32_t>>& ptr_origins,
                             const std::vector<ParamSlot>& param_slots, ValueId vid, ScalarDType pointee) {
  bool writable = ptr_value_is_writable(ptr_origins, param_slots, vid);
  return std::string("device ") + (writable ? "" : "const ") + msl_scalar_type(pointee) + "*";
}

void populate_entry_param_exprs(const Kernel& kernel, const std::unordered_map<ValueId, uint32_t>& val_to_param_idx,
                                const std::unordered_map<uint32_t, std::string>& param_msl_name,
                                std::unordered_map<ValueId, std::string>& expr_map) {
  for (const auto& [vid, pidx] : val_to_param_idx) {
    const KernelParam& kp = kernel.params[pidx];
    if (kp.is_constexpr) {
      expr_map[vid] = format_scalar_literal_from_i64(kp.constexpr_val, scalar_dtype_of(kp.type));
    } else {
      expr_map[vid] = param_msl_name.at(pidx);
    }
  }
}

std::string emit_scalar_op_expr(const Kernel& kernel, const Op& op,
                                const std::unordered_map<ValueId, std::string>& expr_map, HelperFlags& helpers,
                                std::string& err, bool allow_unit_tile_ops, const char* unsupported_err) {
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
      if (!allow_unit_tile_ops) {
        err = unsupported_err;
        return "";
      }
      const auto& attr = std::get<ArangeAttr>(op.attr);
      return format_scalar_literal_from_i64(attr.start, res_dt);
    }
    case OpKind::Splat:
    case OpKind::Broadcast:
    case OpKind::ExpandDims:
    case OpKind::Reshape:
      if (!allow_unit_tile_ops) {
        err = unsupported_err;
        return "";
      }
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
      err = unsupported_err;
      return "";
  }
}

std::optional<uint32_t> param_index_of(const Kernel& kernel, ValueId vid) {
  const Value& v = kernel.value(vid);
  if (v.def_op != kInvalidId || v.def_block != kernel.entry_block) return std::nullopt;
  if (v.arg_idx >= kernel.params.size()) return std::nullopt;
  return v.arg_idx;
}

std::optional<std::string> scalar_long_expr(const Kernel& kernel, ValueId vid, bool allow_bool) {
  if (auto pidx = param_index_of(kernel, vid)) {
    const KernelParam& kp = kernel.params[*pidx];
    if (kp.type.is_ptr()) return std::nullopt;
    if (kp.is_constexpr) return std::to_string(kp.constexpr_val) + "L";
    return "((long)" + kp.name + ")";
  }
  const Value& v = kernel.value(vid);
  if (v.def_op == kInvalidId) return std::nullopt;
  const Op& def = kernel.op(v.def_op);
  if (def.kind == OpKind::Constant) {
    const auto* ca = std::get_if<ConstAttr>(&def.attr);
    if (ca == nullptr) return std::nullopt;
    if (const auto* i = std::get_if<int64_t>(&ca->val)) return std::to_string(*i) + "L";
    if (const auto* u = std::get_if<uint64_t>(&ca->val)) return std::to_string(*u) + "L";
    if (allow_bool) {
      if (const auto* b = std::get_if<bool>(&ca->val)) return *b ? "1L" : "0L";
    }
    return std::nullopt;
  }
  if (def.kind == OpKind::Cast && def.operands.size() == 1) return scalar_long_expr(kernel, def.operands[0], allow_bool);
  return std::nullopt;
}

std::optional<int64_t> positive_constexpr_param(const Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name && kp.constexpr_val > 0) return kp.constexpr_val;
  }
  return std::nullopt;
}

int64_t positive_constexpr_param(const Kernel& kernel, const char* name, int64_t fallback) {
  return positive_constexpr_param(kernel, name).value_or(fallback);
}

void collect_epilogue_dag_helpers(const ir::EpilogueDAGAttr& dag, HelperFlags& helpers) {
  for (const auto& st : dag.steps) {
    switch (st.op) {
      case ir::EpilogueOpKind::GELU:
        helpers.needs_erf = true;
        break;
      case ir::EpilogueOpKind::Tanh:
      case ir::EpilogueOpKind::GELUTanh:
      case ir::EpilogueOpKind::Mish:
        helpers.needs_tanh = true;
        break;
      default:
        break;
    }
  }
}

void collect_epilogue_dag_param_indices(const ir::EpilogueDAGAttr& dag, std::set<uint32_t>& read_params) {
  for (const auto& st : dag.steps) {
    if ((ir::epilogue_op_needs_ptr_arg(st.op) || ir::epilogue_op_needs_scalar_arg(st.op)) && st.arg_idx >= 0) {
      read_params.insert(static_cast<uint32_t>(st.arg_idx));
    }
  }
}

void emit_epilogue_dag_msl(std::ostringstream& os, const Kernel&, const ir::EpilogueDAGAttr& dag,
                           const std::string& v, const std::string& ch_expr, const std::string& elem_idx_expr,
                           const std::function<std::string(uint32_t)>& param_name_fn, HelperFlags& helpers,
                           const std::string& indent) {
  collect_epilogue_dag_helpers(dag, helpers);
  bool declared_reg[4] = {false, false, false, false};
  for (const auto& st : dag.steps) {
    const std::string p0 = msl_exact_float_literal(st.param0);
    const std::string p1 = msl_exact_float_literal(st.param1);
    switch (st.op) {
      case ir::EpilogueOpKind::AddConst:
        os << indent << v << " = " << v << " + " << p0 << ";\n";
        break;
      case ir::EpilogueOpKind::SubConst:
        os << indent << v << " = " << v << " - " << p0 << ";\n";
        break;
      case ir::EpilogueOpKind::RSubConst:
        os << indent << v << " = " << p0 << " - " << v << ";\n";
        break;
      case ir::EpilogueOpKind::MulConst:
        os << indent << v << " = " << v << " * " << p0 << ";\n";
        break;
      case ir::EpilogueOpKind::DivConst:
        os << indent << v << " = " << v << " / " << p0 << ";\n";
        break;
      case ir::EpilogueOpKind::MinConst:
        os << indent << v << " = (" << v << " > " << p0 << ") ? " << p0 << " : " << v << ";\n";
        break;
      case ir::EpilogueOpKind::MaxConst:
        os << indent << v << " = (" << v << " < " << p0 << ") ? " << p0 << " : " << v << ";\n";
        break;
      case ir::EpilogueOpKind::Clamp:
      case ir::EpilogueOpKind::HardTanh:
        os << indent << v << " = (" << v << " < " << p0 << ") ? " << p0 << " : ((" << v << " > " << p1 << ") ? "
           << p1 << " : " << v << ");\n";
        break;
      case ir::EpilogueOpKind::AddBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " + " << pn << "[" << ch_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::SubBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " - " << pn << "[" << ch_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::MulBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " * " << pn << "[" << ch_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::DivBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " / " << pn << "[" << ch_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::MinBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << "{ float m1_eb = " << pn << "[" << ch_expr << "]; " << v << " = (" << v
           << " > m1_eb) ? m1_eb : " << v << "; }\n";
        break;
      }
      case ir::EpilogueOpKind::MaxBias: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << "{ float m1_eb = " << pn << "[" << ch_expr << "]; " << v << " = (" << v
           << " < m1_eb) ? m1_eb : " << v << "; }\n";
        break;
      }
      case ir::EpilogueOpKind::AddScalar: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " + float(" << pn << ");\n";
        break;
      }
      case ir::EpilogueOpKind::SubScalar: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " - float(" << pn << ");\n";
        break;
      }
      case ir::EpilogueOpKind::MulScalar: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " * float(" << pn << ");\n";
        break;
      }
      case ir::EpilogueOpKind::DivScalar: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " / float(" << pn << ");\n";
        break;
      }
      case ir::EpilogueOpKind::AddResidual: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " + " << pn << "[" << elem_idx_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::SubResidual: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " - " << pn << "[" << elem_idx_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::MulResidual: {
        const std::string pn = param_name_fn(static_cast<uint32_t>(st.arg_idx));
        os << indent << v << " = " << v << " * " << pn << "[" << elem_idx_expr << "];\n";
        break;
      }
      case ir::EpilogueOpKind::SaveReg: {
        const int s = st.arg_idx;
        if (!declared_reg[s]) {
          os << indent << "float m1_esav" << s << " = " << v << ";\n";
          declared_reg[s] = true;
        } else {
          os << indent << "m1_esav" << s << " = " << v << ";\n";
        }
        break;
      }
      case ir::EpilogueOpKind::AddSaved:
        os << indent << v << " = " << v << " + m1_esav" << st.arg_idx << ";\n";
        break;
      case ir::EpilogueOpKind::SubSaved:
        os << indent << v << " = " << v << " - m1_esav" << st.arg_idx << ";\n";
        break;
      case ir::EpilogueOpKind::MulSaved:
        os << indent << v << " = " << v << " * m1_esav" << st.arg_idx << ";\n";
        break;
      case ir::EpilogueOpKind::ReLU:
        os << indent << v << " = (" << v << " < 0.0f) ? 0.0f : " << v << ";\n";
        break;
      case ir::EpilogueOpKind::LeakyReLU:
        os << indent << v << " = (" << v << " > 0.0f) ? " << v << " : (" << v << " * " << p0 << ");\n";
        break;
      case ir::EpilogueOpKind::Sigmoid:
        os << indent << v << " = 1.0f / (1.0f + exp(-" << v << "));\n";
        break;
      case ir::EpilogueOpKind::Tanh:
        os << indent << v << " = m1_tanh(" << v << ");\n";
        break;
      case ir::EpilogueOpKind::GELU:
        os << indent << v << " = 0.5f * " << v << " * (1.0f + m1_erf(" << v << " * 0.7071067811865476f));\n";
        break;
      case ir::EpilogueOpKind::GELUTanh:
        os << indent << v << " = 0.5f * " << v << " * (1.0f + m1_tanh(0.7978845608028654f * (" << v
           << " + 0.044715f * " << v << " * " << v << " * " << v << ")));\n";
        break;
      case ir::EpilogueOpKind::Swish:
        os << indent << v << " = " << v << " / (1.0f + exp(-" << v << "));\n";
        break;
      case ir::EpilogueOpKind::Mish:
        os << indent << "{ float m1_sp = (" << v << " > 20.0f) ? " << v << " : log(1.0f + exp(" << v << ")); " << v
           << " = " << v << " * m1_tanh(m1_sp); }\n";
        break;
      case ir::EpilogueOpKind::HardSwish:
        os << indent << v << " = " << v << " * ((" << v << " <= -3.0f) ? 0.0f : ((" << v << " >= 3.0f) ? 1.0f : (("
           << v << " + 3.0f) * 0.1666666716337204f)));\n";
        break;
      case ir::EpilogueOpKind::HardSigmoid:
        os << indent << v << " = (" << v << " <= -3.0f) ? 0.0f : ((" << v << " >= 3.0f) ? 1.0f : ((" << v
           << " + 3.0f) * 0.1666666716337204f));\n";
        break;
      case ir::EpilogueOpKind::ELU:
        os << indent << v << " = (" << v << " > 0.0f) ? " << v << " : (" << p0 << " * (exp((" << v
           << " > 0.0f) ? 0.0f : " << v << ") - 1.0f));\n";
        break;
      case ir::EpilogueOpKind::SELU:
        os << indent << v << " = (" << v << " > 0.0f) ? (1.0507009873554805f * " << v
           << ") : (1.7580993408473768f * (exp((" << v << " > 0.0f) ? 0.0f : " << v << ") - 1.0f));\n";
        break;
      case ir::EpilogueOpKind::Softplus: {
        const float beta = (st.param0 == 0.0f) ? 1.0f : st.param0;
        const float thr = (st.param1 == 0.0f) ? 20.0f : st.param1;
        const std::string bs = msl_exact_float_literal(beta);
        const std::string ts = msl_exact_float_literal(thr);
        os << indent << v << " = (" << v << " * " << bs << " > " << ts << ") ? " << v << " : (log(1.0f + exp(" << v
           << " * " << bs << ")) / " << bs << ");\n";
        break;
      }
      case ir::EpilogueOpKind::Softsign:
        os << indent << v << " = " << v << " / (1.0f + fabs(" << v << "));\n";
        break;
      case ir::EpilogueOpKind::Log:
        os << indent << v << " = log(" << v << ");\n";
        break;
      case ir::EpilogueOpKind::Exp:
        os << indent << v << " = exp(" << v << ");\n";
        break;
      case ir::EpilogueOpKind::Abs:
        os << indent << v << " = fabs(" << v << ");\n";
        break;
      case ir::EpilogueOpKind::Neg:
        os << indent << v << " = -" << v << ";\n";
        break;
      case ir::EpilogueOpKind::Sqrt:
        os << indent << v << " = sqrt(" << v << ");\n";
        break;
    }
  }
}

}  // namespace m1triton::codegen::metal

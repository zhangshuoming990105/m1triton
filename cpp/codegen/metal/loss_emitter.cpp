#include "codegen/metal/loss_emitter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::Kernel;
using ir::LossAttr;
using ir::LossKind;
using ir::Op;
using ir::OpKind;
using ir::ScalarDType;
using ir::ValueId;

struct Operand {
  std::string expr;
  std::optional<int64_t> konst;
  bool is_ptr = false;
  bool aligned16 = false;
  ScalarDType dtype = ScalarDType::F32;
};

struct Emitter {
  explicit Emitter(const Kernel& k) : kernel(k) {}

  const Kernel& kernel;
  uint32_t tg = 256;
  std::vector<Operand> ops;
  std::vector<ParamSlot> slots;
  std::ostringstream body;
  std::string err;

  uint32_t nsg() const { return tg / 32u; }
};

std::string msl_scalar(ScalarDType dt) {
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
  return "int";
}

std::string int_literal(int64_t v) {
  if (v >= std::numeric_limits<int32_t>::min() && v <= std::numeric_limits<int32_t>::max()) {
    return std::to_string(v);
  }
  return std::to_string(v) + "L";
}

std::string float_literal(double v) {
  const float f = static_cast<float>(v);
  char buf[64];
  for (int prec = 6; prec <= 9; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*e", prec, static_cast<double>(f));
    if (std::strtof(buf, nullptr) == f) break;
  }
  return std::string(buf) + "f";
}

std::string msl_param_name(const ir::KernelParam& kp, uint32_t idx) {
  return "m1p_" + (kp.name.empty() ? ("arg" + std::to_string(idx)) : kp.name);
}

bool resolve_operand(const Kernel& k, ValueId vid, Operand& out, std::string& err) {
  const ir::Value& val = k.value(vid);
  if (val.def_op == ir::kInvalidId) {
    if (val.def_block != k.entry_block || val.arg_idx >= k.params.size()) {
      err = "loss operand is a block argument that is not a kernel parameter";
      return false;
    }
    const ir::KernelParam& kp = k.params[val.arg_idx];
    out.is_ptr = kp.type.is_ptr();
    out.aligned16 = out.is_ptr && kp.divisibility >= 16;
    if (out.is_ptr) {
      out.dtype = kp.type.as_ptr().pointee;
    } else if (kp.type.is_scalar()) {
      out.dtype = kp.type.as_scalar();
    } else {
      err = "loss emitter does not support tile parameters";
      return false;
    }
    if (kp.is_constexpr) {
      out.konst = kp.constexpr_val;
      out.expr = int_literal(kp.constexpr_val);
    } else {
      out.expr = msl_param_name(kp, val.arg_idx);
    }
    return true;
  }
  const Op& op = k.op(val.def_op);
  if (op.kind == OpKind::Constant) {
    const auto* c = std::get_if<ir::ConstAttr>(&op.attr);
    if (c == nullptr) {
      err = "constant loss operand without ConstAttr";
      return false;
    }
    out.is_ptr = false;
    out.dtype = val.type.is_scalar() ? val.type.as_scalar() : ScalarDType::F32;
    if (const auto* i = std::get_if<int64_t>(&c->val)) {
      out.konst = *i;
      out.expr = int_literal(*i);
    } else if (const auto* u = std::get_if<uint64_t>(&c->val)) {
      out.konst = static_cast<int64_t>(*u);
      out.expr = int_literal(static_cast<int64_t>(*u));
    } else if (const auto* b = std::get_if<bool>(&c->val)) {
      out.konst = *b ? 1 : 0;
      out.expr = *b ? "1" : "0";
    } else {
      out.expr = float_literal(std::get<double>(c->val));
    }
    return true;
  }
  if (op.kind == OpKind::Cast && op.operands.size() == 1 && val.type.is_scalar()) {
    Operand inner;
    if (!resolve_operand(k, op.operands[0], inner, err)) return false;
    out = inner;
    out.dtype = val.type.as_scalar();
    out.expr = msl_scalar(out.dtype) + "(" + inner.expr + ")";
    return true;
  }
  err = "unsupported loss operand producer '" + std::string(ir::op_kind_name(op.kind)) + "'";
  return false;
}

bool build_slots(Emitter& e, ValueId out_ptr) {
  const Kernel& k = e.kernel;
  uint32_t written_param = UINT32_MAX;
  const ir::Value& ov = k.value(out_ptr);
  if (ov.def_op == ir::kInvalidId && ov.def_block == k.entry_block) written_param = ov.arg_idx;
  uint32_t next = 0;
  for (uint32_t i = 0; i < k.params.size(); ++i) {
    const ir::KernelParam& kp = k.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot slot;
    slot.param_index = i;
    slot.buffer_index = next++;
    slot.name = kp.name.empty() ? ("m1_arg" + std::to_string(slot.buffer_index)) : kp.name;
    slot.is_pointer = kp.type.is_ptr();
    slot.writable = slot.is_pointer && (kp.writable || i == written_param);
    if (slot.is_pointer) {
      const ScalarDType dt = kp.type.as_ptr().pointee;
      if (dt != ScalarDType::F32 && dt != ScalarDType::I64 && dt != ScalarDType::I32) {
        e.err = "loss emitter supports f32, i64, and i32 device pointers only";
        return false;
      }
      slot.dtype = dt;
    } else if (kp.type.is_scalar()) {
      slot.dtype = kp.type.as_scalar();
    } else {
      e.err = "loss emitter does not accept tile parameters";
      return false;
    }
    e.slots.push_back(std::move(slot));
  }
  return true;
}

std::string signature(const Emitter& e) {
  std::ostringstream os;
  os << "kernel void " << e.kernel.name << "(\n";
  for (const auto& slot : e.slots) {
    const std::string name = msl_param_name(e.kernel.params[slot.param_index], slot.param_index);
    if (slot.is_pointer) {
      os << "    device " << (slot.writable ? "" : "const ") << msl_scalar(slot.dtype) << "* " << name;
    } else {
      os << "    constant " << msl_scalar(slot.dtype) << "& " << name;
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

constexpr const char* kPreamble = R"MSL(#include <metal_stdlib>
using namespace metal;

inline float m1_sum_tg(float v, threadgroup float* sh, uint lane, uint sg, uint nsg) {
  v = simd_sum(v);
  if (lane == 0u) sh[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0.0f;
  if (sg == 0u && lane == 0u) {
    for (uint i = 0u; i < nsg; ++i) r += sh[i];
  }
  return r;
}

)MSL";

bool all_aligned(const Emitter& e, std::initializer_list<size_t> idxs) {
  for (size_t i : idxs) {
    if (!e.ops[i].aligned16) return false;
  }
  return true;
}

bool konst_mult4(const Operand& o) { return o.konst.has_value() && (*o.konst % 4) == 0; }

bool check_f32_ptrs(Emitter& e, std::initializer_list<size_t> idxs, const char* what) {
  for (size_t i : idxs) {
    if (!e.ops[i].is_ptr || e.ops[i].dtype != ScalarDType::F32) {
      e.err = std::string(what) + ": operand " + std::to_string(i) + " must be an f32 device pointer";
      return false;
    }
  }
  return true;
}

bool emit_stage2(Emitter& e) {
  if (e.ops.size() != 4) {
    e.err = "fused_loss stage 2 expects 4 operands (partial_ptr, out_ptr, num_partials, denom)";
    return false;
  }
  if (!check_f32_ptrs(e, {0, 1}, "fused_loss stage 2")) return false;
  if (e.ops[2].is_ptr || e.ops[3].is_ptr) {
    e.err = "fused_loss stage 2 operands 2 and 3 must be scalars";
    return false;
  }

  const char* vt = e.ops[0].aligned16 ? "float4" : "packed_float4";
  const std::string p = e.ops[0].expr;
  const std::string o = e.ops[1].expr;
  auto& os = e.body;
  os << "  (void)m1_tgp;\n"
     << "  (void)m1_ntg;\n"
     << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  const long m1_n = long(" << e.ops[2].expr << ");\n"
     << "  const long m1_nv = m1_n >> 2;\n"
     << "  device const " << vt << "* m1_pv = (device const " << vt << "*)(" << p << ");\n"
     << "  float4 m1_acc4 = float4(0.0f);\n"
     << "  for (long m1_i = long(m1_tid); m1_i < m1_nv; m1_i += " << e.tg << "L) {\n"
     << "    m1_acc4 += float4(m1_pv[m1_i]);\n"
     << "  }\n"
     << "  float m1_acc = (m1_acc4.x + m1_acc4.y) + (m1_acc4.z + m1_acc4.w);\n"
     << "  const long m1_tail = m1_nv << 2;\n"
     << "  const long m1_rem = m1_n - m1_tail;\n"
     << "  for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "    m1_acc += " << p << "[m1_tail + m1_r];\n"
     << "  }\n"
     << "  const float m1_tot = m1_sum_tg(m1_acc, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u);\n"
     << "  if (m1_tid == 0u) {\n"
     << "    " << o << "[0] = m1_tot / float(" << e.ops[3].expr << ");\n"
     << "  }\n";
  return true;
}

bool emit_flat_stage1(Emitter& e, const LossAttr& a) {
  if (e.ops.size() != 4) {
    e.err = "fused_loss flat stage 1 expects 4 operands (pred_ptr, tgt_ptr, partial_ptr, total)";
    return false;
  }
  if (!check_f32_ptrs(e, {0, 1, 2}, "fused_loss flat stage 1")) return false;
  if (e.ops[3].is_ptr) {
    e.err = "fused_loss flat stage 1 operand 3 (total) must be an integer scalar";
    return false;
  }

  const char* vt = all_aligned(e, {0, 1}) ? "float4" : "packed_float4";
  const std::string p = e.ops[0].expr;
  const std::string t = e.ops[1].expr;
  const std::string out = e.ops[2].expr;
  auto& os = e.body;

  auto emit_vec4_accum = [&](const std::string& indent, const std::string& p_var, const std::string& t_var,
                             const std::string& acc_var) {
    if (a.kind == LossKind::MSE) {
      os << indent << "const float4 m1_d = " << p_var << " - " << t_var << ";\n"
         << indent << acc_var << " = fma(m1_d, m1_d, " << acc_var << ");\n";
    } else if (a.kind == LossKind::Huber) {
      os << indent << "const float4 m1_d = abs(" << p_var << " - " << t_var << ");\n";
      if (a.beta == 0.0f) {
        os << indent << acc_var << " += m1_d;\n";
      } else {
        os << indent << acc_var
           << " += select(m1_d - m1_half_beta, m1_half_inv_beta * (m1_d * m1_d), m1_d < float4(m1_beta));\n";
      }
    } else {
      os << indent << "const bool4 m1_pos = " << t_var << " > float4(0.0f);\n"
         << indent << "const float4 m1_st = select(float4(1.0f), " << t_var << ", m1_pos);\n"
         << indent << "const float4 m1_sp = select(float4(1.0f), " << p_var << ", m1_pos);\n"
         << indent << acc_var << " += select(float4(0.0f), " << t_var
         << " * (log(m1_st) - log(m1_sp)), m1_pos);\n";
    }
  };

  const long tile_v = 2L * static_cast<long>(e.tg);
  os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  const long m1_total = long(" << e.ops[3].expr << ");\n"
     << "  const long m1_nv = m1_total >> 2;\n"
     << "  device const " << vt << "* m1_pv = (device const " << vt << "*)(" << p << ");\n"
     << "  device const " << vt << "* m1_tv = (device const " << vt << "*)(" << t << ");\n";

  if (a.kind == LossKind::Huber && a.beta > 0.0f) {
    os << "  const float m1_beta = " << float_literal(a.beta) << ";\n"
       << "  const float m1_half_inv_beta = " << float_literal(0.5 / static_cast<double>(a.beta)) << ";\n"
       << "  const float m1_half_beta = " << float_literal(0.5 * static_cast<double>(a.beta)) << ";\n";
  }

  os << "  float4 m1_acc4 = float4(0.0f);\n"
     << "  float4 m1_acc4_1 = float4(0.0f);\n"
     << "  const long m1_tile = " << tile_v << "L;\n"
     << "  const long m1_gstride = long(m1_ntg.x) * m1_tile;\n"
     << "  for (long m1_t0 = long(m1_tgp.x) * m1_tile; m1_t0 < m1_nv; m1_t0 += m1_gstride) {\n"
     << "    const long m1_b = m1_t0 + long(m1_tid);\n"
     << "    if (m1_t0 + m1_tile <= m1_nv) {\n"
     << "      const float4 m1_p0 = float4(m1_pv[m1_b]);\n"
     << "      const float4 m1_t0v = float4(m1_tv[m1_b]);\n"
     << "      const float4 m1_p1 = float4(m1_pv[m1_b + " << e.tg << "L]);\n"
     << "      const float4 m1_t1v = float4(m1_tv[m1_b + " << e.tg << "L]);\n"
     << "      {\n";
  emit_vec4_accum("        ", "m1_p0", "m1_t0v", "m1_acc4");
  os << "      }\n"
     << "      {\n";
  emit_vec4_accum("        ", "m1_p1", "m1_t1v", "m1_acc4_1");
  os << "      }\n"
     << "    } else {\n"
     << "      const long m1_end = min(m1_t0 + m1_tile, m1_nv);\n"
     << "      for (long m1_i = m1_b; m1_i < m1_end; m1_i += " << e.tg << "L) {\n"
     << "        const float4 m1_p = float4(m1_pv[m1_i]);\n"
     << "        const float4 m1_t = float4(m1_tv[m1_i]);\n";
  emit_vec4_accum("        ", "m1_p", "m1_t", "m1_acc4");
  os << "      }\n"
     << "    }\n"
     << "  }\n"
     << "  m1_acc4 += m1_acc4_1;\n"
     << "  float m1_acc = (m1_acc4.x + m1_acc4.y) + (m1_acc4.z + m1_acc4.w);\n"
     << "  if (m1_tgp.x == 0u) {\n"
     << "    const long m1_tail = m1_nv << 2;\n"
     << "    const long m1_rem = m1_total - m1_tail;\n"
     << "    for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "      const float m1_p = " << p << "[m1_tail + m1_r];\n"
     << "      const float m1_t = " << t << "[m1_tail + m1_r];\n";
  if (a.kind == LossKind::MSE) {
    os << "      const float m1_d = m1_p - m1_t;\n"
       << "      m1_acc = fma(m1_d, m1_d, m1_acc);\n";
  } else if (a.kind == LossKind::Huber) {
    os << "      const float m1_d = abs(m1_p - m1_t);\n";
    if (a.beta == 0.0f) {
      os << "      m1_acc += m1_d;\n";
    } else {
      os << "      m1_acc += (m1_d < m1_beta) ? (m1_half_inv_beta * m1_d * m1_d) : (m1_d - m1_half_beta);\n";
    }
  } else {
    os << "      if (m1_t > 0.0f) {\n"
       << "        m1_acc += m1_t * (log(m1_t) - log(m1_p));\n"
       << "      }\n";
  }
  os << "    }\n"
     << "  }\n"
     << "  const float m1_tot = m1_sum_tg(m1_acc, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u);\n"
     << "  if (m1_tid == 0u) {\n"
     << "    " << out << "[m1_tgp.x] = m1_tot;\n"
     << "  }\n";
  return true;
}

void emit_hinge_slice(Emitter& e, const std::string& indent, const char* vt, const std::string& p_base,
                      const std::string& t_base, const std::string& len_expr) {
  const std::string p = e.ops[0].expr;
  const std::string t = e.ops[1].expr;
  auto& os = e.body;
  os << indent << "const long m1_nv = (" << len_expr << ") >> 2;\n"
     << indent << "device const " << vt << "* m1_pv = (device const " << vt << "*)(" << p << " + " << p_base << ");\n"
     << indent << "device const " << vt << "* m1_tv = (device const " << vt << "*)(" << t << " + " << t_base << ");\n"
     << indent << "for (long m1_i = long(m1_tid); m1_i < m1_nv; m1_i += " << e.tg << "L) {\n"
     << indent << "  const float4 m1_p = float4(m1_pv[m1_i]);\n"
     << indent << "  const float4 m1_t = float4(m1_tv[m1_i]);\n"
     << indent << "  m1_acc4 += max(1.0f - m1_p * m1_t, float4(0.0f));\n"
     << indent << "}\n"
     << indent << "const long m1_tail = m1_nv << 2;\n"
     << indent << "const long m1_rem = (" << len_expr << ") - m1_tail;\n"
     << indent << "for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << indent << "  const float m1_p = " << p << "[" << p_base << " + m1_tail + m1_r];\n"
     << indent << "  const float m1_t = " << t << "[" << t_base << " + m1_tail + m1_r];\n"
     << indent << "  m1_acc += max(1.0f - m1_p * m1_t, 0.0f);\n"
     << indent << "}\n";
}

bool emit_hinge_stage1(Emitter& e) {
  if (e.ops.size() != 5) {
    e.err = "fused_loss hinge stage 1 expects 5 operands (pred_ptr, tgt_ptr, partial_ptr, rows, cols)";
    return false;
  }
  if (!check_f32_ptrs(e, {0, 1, 2}, "fused_loss hinge stage 1")) return false;
  if (e.ops[3].is_ptr || e.ops[4].is_ptr) {
    e.err = "fused_loss hinge stage 1 operands 3 and 4 (rows, cols) must be integer scalars";
    return false;
  }

  const bool ptrs_aligned = all_aligned(e, {0, 1});
  const bool static_aligned = ptrs_aligned && konst_mult4(e.ops[4]);
  const std::string out = e.ops[2].expr;
  auto& os = e.body;

  os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  const long m1_rows = long(" << e.ops[3].expr << ");\n"
     << "  const long m1_cols = long(" << e.ops[4].expr << ");\n"
     << "  float4 m1_acc4 = float4(0.0f);\n"
     << "  float m1_acc = 0.0f;\n"
     << "  if (m1_rows >= long(m1_ntg.x)) {\n";

  if (static_aligned) {
    os << "    for (long m1_row = long(m1_tgp.x); m1_row < m1_rows; m1_row += long(m1_ntg.x)) {\n"
       << "      const long m1_base = m1_row * m1_cols;\n";
    emit_hinge_slice(e, "      ", "float4", "m1_base", "0L", "m1_cols");
    os << "    }\n";
  } else if (ptrs_aligned) {
    os << "    if ((m1_cols & 3L) == 0L) {\n"
       << "      for (long m1_row = long(m1_tgp.x); m1_row < m1_rows; m1_row += long(m1_ntg.x)) {\n"
       << "        const long m1_base = m1_row * m1_cols;\n";
    emit_hinge_slice(e, "        ", "float4", "m1_base", "0L", "m1_cols");
    os << "      }\n"
       << "    } else {\n"
       << "      for (long m1_row = long(m1_tgp.x); m1_row < m1_rows; m1_row += long(m1_ntg.x)) {\n"
       << "        const long m1_base = m1_row * m1_cols;\n";
    emit_hinge_slice(e, "        ", "packed_float4", "m1_base", "0L", "m1_cols");
    os << "      }\n"
       << "    }\n";
  } else {
    os << "    for (long m1_row = long(m1_tgp.x); m1_row < m1_rows; m1_row += long(m1_ntg.x)) {\n"
       << "      const long m1_base = m1_row * m1_cols;\n";
    emit_hinge_slice(e, "      ", "packed_float4", "m1_base", "0L", "m1_cols");
    os << "    }\n";
  }

  os << "  } else if (m1_rows > 0L && m1_cols > 0L) {\n"
     << "    const long m1_cchunk = ((m1_cols + long(m1_ntg.x) - 1L) / long(m1_ntg.x) + 3L) & ~3L;\n"
     << "    const long m1_c0 = min(long(m1_tgp.x) * m1_cchunk, m1_cols);\n"
     << "    const long m1_cn = min(m1_cchunk, m1_cols - m1_c0);\n";

  if (static_aligned) {
    os << "    for (long m1_row = 0L; m1_row < m1_rows; ++m1_row) {\n"
       << "      const long m1_pbase = m1_row * m1_cols + m1_c0;\n";
    emit_hinge_slice(e, "      ", "float4", "m1_pbase", "m1_c0", "m1_cn");
    os << "    }\n";
  } else if (ptrs_aligned) {
    os << "    if ((m1_cols & 3L) == 0L || m1_rows == 1L) {\n"
       << "      for (long m1_row = 0L; m1_row < m1_rows; ++m1_row) {\n"
       << "        const long m1_pbase = m1_row * m1_cols + m1_c0;\n";
    emit_hinge_slice(e, "        ", "float4", "m1_pbase", "m1_c0", "m1_cn");
    os << "      }\n"
       << "    } else {\n"
       << "      for (long m1_row = 0L; m1_row < m1_rows; ++m1_row) {\n"
       << "        const long m1_pbase = m1_row * m1_cols + m1_c0;\n";
    emit_hinge_slice(e, "        ", "packed_float4", "m1_pbase", "m1_c0", "m1_cn");
    os << "      }\n"
       << "    }\n";
  } else {
    os << "    for (long m1_row = 0L; m1_row < m1_rows; ++m1_row) {\n"
       << "      const long m1_pbase = m1_row * m1_cols + m1_c0;\n";
    emit_hinge_slice(e, "      ", "packed_float4", "m1_pbase", "m1_c0", "m1_cn");
    os << "    }\n";
  }

  os << "  }\n"
     << "  m1_acc += (m1_acc4.x + m1_acc4.y) + (m1_acc4.z + m1_acc4.w);\n"
     << "  const float m1_tot = m1_sum_tg(m1_acc, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u);\n"
     << "  if (m1_tid == 0u) {\n"
     << "    " << out << "[m1_tgp.x] = m1_tot;\n"
     << "  }\n";
  return true;
}

bool emit_cross_entropy_stage1(Emitter& e) {
  if (e.ops.size() != 5) {
    e.err = "fused_loss cross_entropy stage 1 expects 5 operands (pred_ptr, tgt_ptr, partial_ptr, B, C)";
    return false;
  }
  if (!check_f32_ptrs(e, {0, 2}, "fused_loss cross_entropy stage 1")) return false;
  if (!e.ops[1].is_ptr || (e.ops[1].dtype != ScalarDType::I64 && e.ops[1].dtype != ScalarDType::I32)) {
    e.err = "fused_loss cross_entropy stage 1 operand 1 must be an i64 or i32 device pointer";
    return false;
  }
  if (e.ops[3].is_ptr || e.ops[4].is_ptr) {
    e.err = "fused_loss cross_entropy stage 1 operands 3 and 4 (B, C) must be integer scalars";
    return false;
  }

  const bool ptr_aligned = e.ops[0].aligned16;
  const bool need_peel = ptr_aligned && !konst_mult4(e.ops[4]);
  const char* vt = ptr_aligned ? "float4" : "packed_float4";
  const std::string p = e.ops[0].expr;
  const std::string t = e.ops[1].expr;
  const std::string out = e.ops[2].expr;
  auto& os = e.body;

  os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  threadgroup float m1_sh2[" << e.nsg() << "];\n"
     << "  const long m1_B = long(" << e.ops[3].expr << ");\n"
     << "  const long m1_C = long(" << e.ops[4].expr << ");\n"
     << "  float m1_row_sum = 0.0f;\n"
     << "  if (m1_C > 0L) {\n"
     << "    for (long m1_row = long(m1_tgp.x); m1_row < m1_B; m1_row += long(m1_ntg.x)) {\n"
     << "      const long m1_base = m1_row * m1_C;\n";
  if (need_peel) {
    os << "      const long m1_hd = min((4L - (m1_base & 3L)) & 3L, m1_C);\n";
  } else {
    os << "      const long m1_hd = 0L;\n";
  }
  os << "      const long m1_nv = (m1_C - m1_hd) >> 2;\n"
     << "      const long m1_rem = m1_C - (m1_nv << 2);\n"
     << "      device const " << vt << "* m1_pv = (device const " << vt << "*)(" << p << " + m1_base + m1_hd);\n"
     << "      if (m1_nv <= " << (4L * static_cast<long>(e.tg)) << "L && m1_rem == 0L) {\n"
     << "        const long m1_i0 = long(m1_tid);\n"
     << "        const long m1_i1 = m1_i0 + " << e.tg << "L;\n"
     << "        const long m1_i2 = m1_i0 + " << (2u * e.tg) << "L;\n"
     << "        const long m1_i3 = m1_i0 + " << (3u * e.tg) << "L;\n"
     << "        const bool m1_h0 = m1_i0 < m1_nv;\n"
     << "        const bool m1_h1 = m1_i1 < m1_nv;\n"
     << "        const bool m1_h2 = m1_i2 < m1_nv;\n"
     << "        const bool m1_h3 = m1_i3 < m1_nv;\n"
     << "        const float4 m1_v0 = m1_h0 ? float4(m1_pv[m1_i0]) : float4(-3.402823466e+38f);\n"
     << "        const float4 m1_v1 = m1_h1 ? float4(m1_pv[m1_i1]) : float4(-3.402823466e+38f);\n"
     << "        const float4 m1_v2 = m1_h2 ? float4(m1_pv[m1_i2]) : float4(-3.402823466e+38f);\n"
     << "        const float4 m1_v3 = m1_h3 ? float4(m1_pv[m1_i3]) : float4(-3.402823466e+38f);\n"
     << "        const float4 m1_mv = max(max(m1_v0, m1_v1), max(m1_v2, m1_v3));\n"
     << "        float m1_m = simd_max(max(max(m1_mv.x, m1_mv.y), max(m1_mv.z, m1_mv.w)));\n"
     << "        if (m1_lane == 0u) m1_sh[m1_sg] = m1_m;\n"
     << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "        float m1_rm = m1_sh[0];\n"
     << "        for (uint m1_g = 1u; m1_g < " << e.nsg() << "u; ++m1_g) m1_rm = max(m1_rm, m1_sh[m1_g]);\n"
     << "        float4 m1_d4 = float4(0.0f);\n"
     << "        if (m1_h0) m1_d4 += exp(m1_v0 - float4(m1_rm));\n"
     << "        if (m1_h1) m1_d4 += exp(m1_v1 - float4(m1_rm));\n"
     << "        if (m1_h2) m1_d4 += exp(m1_v2 - float4(m1_rm));\n"
     << "        if (m1_h3) m1_d4 += exp(m1_v3 - float4(m1_rm));\n"
     << "        const float m1_d = simd_sum((m1_d4.x + m1_d4.y) + (m1_d4.z + m1_d4.w));\n"
     << "        if (m1_lane == 0u) m1_sh2[m1_sg] = m1_d;\n"
     << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "        if (m1_tid == 0u) {\n"
     << "          float m1_rd = 0.0f;\n"
     << "          for (uint m1_g = 0u; m1_g < " << e.nsg() << "u; ++m1_g) m1_rd += m1_sh2[m1_g];\n"
     << "          const long m1_t = long(" << t << "[m1_row]);\n"
     << "          if (m1_t >= 0L && m1_t < m1_C) {\n"
     << "            m1_row_sum += (m1_rm + log(m1_rd)) - " << p << "[m1_base + m1_t];\n"
     << "          }\n"
     << "        }\n"
     << "      } else {\n"
     << "        float m1_m = -3.402823466e+38f;\n"
     << "        for (long m1_i = long(m1_tid); m1_i < m1_nv; m1_i += " << e.tg << "L) {\n"
     << "          const float4 m1_v = float4(m1_pv[m1_i]);\n"
     << "          m1_m = max(m1_m, max(max(m1_v.x, m1_v.y), max(m1_v.z, m1_v.w)));\n"
     << "        }\n"
     << "        for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "          const long m1_j = (m1_r < m1_hd) ? m1_r : (m1_r + (m1_nv << 2));\n"
     << "          m1_m = max(m1_m, " << p << "[m1_base + m1_j]);\n"
     << "        }\n"
     << "        m1_m = simd_max(m1_m);\n"
     << "        if (m1_lane == 0u) m1_sh[m1_sg] = m1_m;\n"
     << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "        float m1_rm = m1_sh[0];\n"
     << "        for (uint m1_g = 1u; m1_g < " << e.nsg() << "u; ++m1_g) m1_rm = max(m1_rm, m1_sh[m1_g]);\n"
     << "        float4 m1_d4 = float4(0.0f);\n"
     << "        for (long m1_i = long(m1_tid); m1_i < m1_nv; m1_i += " << e.tg << "L) {\n"
     << "          m1_d4 += exp(float4(m1_pv[m1_i]) - float4(m1_rm));\n"
     << "        }\n"
     << "        float m1_d = (m1_d4.x + m1_d4.y) + (m1_d4.z + m1_d4.w);\n"
     << "        for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "          const long m1_j = (m1_r < m1_hd) ? m1_r : (m1_r + (m1_nv << 2));\n"
     << "          m1_d += exp(" << p << "[m1_base + m1_j] - m1_rm);\n"
     << "        }\n"
     << "        m1_d = simd_sum(m1_d);\n"
     << "        if (m1_lane == 0u) m1_sh2[m1_sg] = m1_d;\n"
     << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "        if (m1_tid == 0u) {\n"
     << "          float m1_rd = 0.0f;\n"
     << "          for (uint m1_g = 0u; m1_g < " << e.nsg() << "u; ++m1_g) m1_rd += m1_sh2[m1_g];\n"
     << "          const long m1_t = long(" << t << "[m1_row]);\n"
     << "          if (m1_t >= 0L && m1_t < m1_C) {\n"
     << "            m1_row_sum += (m1_rm + log(m1_rd)) - " << p << "[m1_base + m1_t];\n"
     << "          }\n"
     << "        }\n"
     << "      }\n"
     << "    }\n"
     << "  }\n"
     << "  if (m1_tid == 0u) {\n"
     << "    " << out << "[m1_tgp.x] = m1_row_sum;\n"
     << "  }\n";
  return true;
}

bool emit_triplet_margin_stage1(Emitter& e, const LossAttr& a) {
  if (e.ops.size() != 6) {
    e.err = "fused_loss triplet_margin stage 1 expects 6 operands (anchor, pos, neg, partial, B, D)";
    return false;
  }
  if (!check_f32_ptrs(e, {0, 1, 2, 3}, "fused_loss triplet_margin stage 1")) return false;
  if (e.ops[4].is_ptr || e.ops[5].is_ptr) {
    e.err = "fused_loss triplet_margin stage 1 operands 4 and 5 (B, D) must be integer scalars";
    return false;
  }

  const bool ptrs_aligned = all_aligned(e, {0, 1, 2});
  const bool need_peel = ptrs_aligned && !konst_mult4(e.ops[5]);
  const char* vt = ptrs_aligned ? "float4" : "packed_float4";
  const std::string anc = e.ops[0].expr;
  const std::string pos = e.ops[1].expr;
  const std::string neg = e.ops[2].expr;
  const std::string out = e.ops[3].expr;
  auto& os = e.body;

  os << "  threadgroup float m1_shap[2][" << e.nsg() << "];\n"
     << "  threadgroup float m1_shan[2][" << e.nsg() << "];\n"
     << "  const long m1_B = long(" << e.ops[4].expr << ");\n"
     << "  const long m1_D = long(" << e.ops[5].expr << ");\n"
     << "  const float m1_eps = " << float_literal(a.eps) << ";\n"
     << "  const float m1_margin = " << float_literal(a.margin) << ";\n"
     << "  float m1_row_sum = 0.0f;\n"
     << "  uint m1_buf = 0u;\n"
     << "  for (long m1_row = long(m1_tgp.x); m1_row < m1_B; m1_row += long(m1_ntg.x)) {\n"
     << "    const long m1_base = m1_row * m1_D;\n";
  if (need_peel) {
    os << "    const long m1_hd = min((4L - (m1_base & 3L)) & 3L, m1_D);\n";
  } else {
    os << "    const long m1_hd = 0L;\n";
  }
  os << "    const long m1_nv = (m1_D - m1_hd) >> 2;\n"
     << "    device const " << vt << "* m1_av = (device const " << vt << "*)(" << anc << " + m1_base + m1_hd);\n"
     << "    device const " << vt << "* m1_pv = (device const " << vt << "*)(" << pos << " + m1_base + m1_hd);\n"
     << "    device const " << vt << "* m1_nvp = (device const " << vt << "*)(" << neg << " + m1_base + m1_hd);\n"
     << "    float4 m1_ap4 = float4(0.0f);\n"
     << "    float4 m1_an4 = float4(0.0f);\n"
     << "    for (long m1_i = long(m1_tid); m1_i < m1_nv; m1_i += " << e.tg << "L) {\n"
     << "      const float4 m1_va = float4(m1_av[m1_i]);\n"
     << "      const float4 m1_vp = float4(m1_pv[m1_i]);\n"
     << "      const float4 m1_vn = float4(m1_nvp[m1_i]);\n"
     << "      const float4 m1_dap = (m1_va - m1_vp) + m1_eps;\n"
     << "      const float4 m1_dan = (m1_va - m1_vn) + m1_eps;\n"
     << "      m1_ap4 = fma(m1_dap, m1_dap, m1_ap4);\n"
     << "      m1_an4 = fma(m1_dan, m1_dan, m1_an4);\n"
     << "    }\n"
     << "    float m1_ap = (m1_ap4.x + m1_ap4.y) + (m1_ap4.z + m1_ap4.w);\n"
     << "    float m1_an = (m1_an4.x + m1_an4.y) + (m1_an4.z + m1_an4.w);\n"
     << "    const long m1_rem = m1_D - (m1_nv << 2);\n"
     << "    for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "      const long m1_j = (m1_r < m1_hd) ? m1_r : (m1_r + (m1_nv << 2));\n"
     << "      const float m1_va = " << anc << "[m1_base + m1_j];\n"
     << "      const float m1_dap = (m1_va - " << pos << "[m1_base + m1_j]) + m1_eps;\n"
     << "      const float m1_dan = (m1_va - " << neg << "[m1_base + m1_j]) + m1_eps;\n"
     << "      m1_ap = fma(m1_dap, m1_dap, m1_ap);\n"
     << "      m1_an = fma(m1_dan, m1_dan, m1_an);\n"
     << "    }\n"
     << "    m1_ap = simd_sum(m1_ap);\n"
     << "    m1_an = simd_sum(m1_an);\n"
     << "    if (m1_lane == 0u) {\n"
     << "      m1_shap[m1_buf][m1_sg] = m1_ap;\n"
     << "      m1_shan[m1_buf][m1_sg] = m1_an;\n"
     << "    }\n"
     << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "    if (m1_tid == 0u) {\n"
     << "      float m1_sap = 0.0f;\n"
     << "      float m1_san = 0.0f;\n"
     << "      for (uint m1_g = 0u; m1_g < " << e.nsg() << "u; ++m1_g) {\n"
     << "        m1_sap += m1_shap[m1_buf][m1_g];\n"
     << "        m1_san += m1_shan[m1_buf][m1_g];\n"
     << "      }\n"
     << "      m1_row_sum += max(sqrt(m1_sap) - sqrt(m1_san) + m1_margin, 0.0f);\n"
     << "    }\n"
     << "    m1_buf ^= 1u;\n"
     << "  }\n"
     << "  if (m1_tid == 0u) {\n"
     << "    " << out << "[m1_tgp.x] = m1_row_sum;\n"
     << "  }\n";
  return true;
}

bool emit_loss(Emitter& e, const LossAttr& a) {
  if (a.stage == 2) return emit_stage2(e);
  if (a.stage != 1) {
    e.err = "fused_loss stage must be 1 or 2";
    return false;
  }
  switch (a.kind) {
    case LossKind::MSE:
    case LossKind::Huber:
    case LossKind::KLDiv:
      return emit_flat_stage1(e, a);
    case LossKind::Hinge:
      return emit_hinge_stage1(e);
    case LossKind::CrossEntropy:
      return emit_cross_entropy_stage1(e);
    case LossKind::TripletMargin:
      return emit_triplet_margin_stage1(e, a);
  }
  e.err = "unknown fused_loss kind";
  return false;
}

}  // namespace

CompiledMSL try_emit_loss_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  if (matched_out != nullptr) *matched_out = false;
  const Op* fused = nullptr;
  bool foreign = false;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    if (op.kind == OpKind::FusedLoss) {
      if (fused != nullptr) foreign = true;
      fused = &op;
    } else if (op.kind != OpKind::Constant && op.kind != OpKind::Cast && op.kind != OpKind::Return) {
      foreign = true;
    }
  }
  if (fused == nullptr) return CompiledMSL{false, "", {}, ""};
  if (matched_out != nullptr) *matched_out = true;
  if (foreign) {
    return CompiledMSL{false, "", {}, "fused_loss kernels must contain exactly one FusedLoss op and no other compute"};
  }

  const auto* attr = std::get_if<LossAttr>(&fused->attr);
  if (attr == nullptr) {
    return CompiledMSL{false, "", {}, "fused_loss op is missing LossAttr"};
  }

  const uint32_t tg = attr->tg_threads;
  if (tg == 0u || target.simd_width != 32u || (tg % 32u) != 0u || tg > target.max_threads_per_threadgroup) {
    return CompiledMSL{false, "", {}, "fused_loss tg_threads must be a non-zero multiple of 32 within target limit"};
  }

  Emitter e{kernel};
  e.tg = tg;
  for (ValueId vid : fused->operands) {
    Operand o;
    if (!resolve_operand(kernel, vid, o, e.err)) return CompiledMSL{false, "", {}, e.err};
    e.ops.push_back(std::move(o));
  }

  const size_t out_idx = (attr->stage == 2) ? 1u : (attr->kind == LossKind::TripletMargin ? 3u : 2u);
  if (out_idx >= fused->operands.size() || !build_slots(e, fused->operands[out_idx])) {
    return CompiledMSL{false, "", {}, e.err.empty() ? "fused_loss op has insufficient operands" : e.err};
  }

  if (!emit_loss(e, *attr)) {
    return CompiledMSL{false, "", {}, e.err};
  }

  CompiledMSL res;
  res.ok = true;
  res.msl_source = std::string(kPreamble) + signature(e) + e.body.str() + "}\n";
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = e.tg;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = e.slots;
  return res;
}

}  // namespace m1triton::codegen::metal

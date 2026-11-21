#include "codegen/metal/norm_softmax_emitter.hpp"

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
using ir::NormKind;
using ir::Op;
using ir::OpKind;
using ir::ValueId;

enum class VecMode { Aligned, Peel, Packed };

struct Operand {
  std::string expr;
  std::optional<int64_t> konst;
  bool is_ptr = false;
  bool aligned16 = false;
};

struct Aux {
  std::string name;
  std::string ptr;
  std::string base;
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

std::string msl_scalar(ir::ScalarDType dt) {
  switch (dt) {
    case ir::ScalarDType::F32:
      return "float";
    case ir::ScalarDType::F16:
      return "half";
    case ir::ScalarDType::I32:
      return "int";
    case ir::ScalarDType::U32:
      return "uint";
    case ir::ScalarDType::I64:
      return "long";
    case ir::ScalarDType::Bool:
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
      err = "operand is a block argument that is not a kernel parameter";
      return false;
    }
    const ir::KernelParam& kp = k.params[val.arg_idx];
    out.is_ptr = kp.type.is_ptr();
    out.aligned16 = out.is_ptr && kp.divisibility >= 16;
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
      err = "constant operand without ConstAttr";
      return false;
    }
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
    out.expr = msl_scalar(val.type.as_scalar()) + "(" + inner.expr + ")";
    return true;
  }
  err = "unsupported operand producer '" + std::string(ir::op_kind_name(op.kind)) + "'";
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
      if (kp.type.as_ptr().pointee != ir::ScalarDType::F32) {
        e.err = "norm/softmax emitter supports f32 tensors only";
        return false;
      }
      slot.dtype = ir::ScalarDType::F32;
    } else if (kp.type.is_scalar()) {
      slot.dtype = kp.type.as_scalar();
    } else {
      e.err = "norm/softmax emitter does not accept tile parameters";
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
      os << "    device " << (slot.writable ? "" : "const ") << "float* " << name;
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

struct m1_wf {
  float n;
  float mean;
  float m2;
};

inline m1_wf m1_wf_merge(m1_wf a, m1_wf b) {
  if (b.n <= 0.0f) return a;
  if (a.n <= 0.0f) return b;
  const float n = a.n + b.n;
  const float d = b.mean - a.mean;
  const float rb = b.n / n;
  return m1_wf{n, a.mean + d * rb, a.m2 + b.m2 + d * d * a.n * rb};
}

inline m1_wf m1_wf_shifted(uint cnt, float s, float ss, float k) {
  if (cnt == 0u) return m1_wf{0.0f, 0.0f, 0.0f};
  const float n = float(cnt);
  const float mu = s / n;
  return m1_wf{n, k + mu, max(ss - s * mu, 0.0f)};
}

inline m1_wf m1_wf_simd(m1_wf w) {
  for (ushort o = 16; o > 0; o >>= 1) {
    const m1_wf t = m1_wf{simd_shuffle_xor(w.n, o), simd_shuffle_xor(w.mean, o), simd_shuffle_xor(w.m2, o)};
    w = m1_wf_merge(w, t);
  }
  return m1_wf{simd_broadcast(w.n, 0), simd_broadcast(w.mean, 0), simd_broadcast(w.m2, 0)};
}

inline m1_wf m1_wf_tg(m1_wf w, threadgroup m1_wf* sh, uint lane, uint sg, uint nsg) {
  w = m1_wf_simd(w);
  if (lane == 0) sh[sg] = w;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m1_wf r = sh[0];
  for (uint i = 1; i < nsg; ++i) r = m1_wf_merge(r, sh[i]);
  return r;
}

inline float m1_sum_tg(float v, threadgroup float* sh, uint lane, uint sg, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) sh[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = 0.0f;
  for (uint i = 0; i < nsg; ++i) r += sh[i];
  return r;
}

inline m1_wf m1_wf_partials(device const float* p, uint first, uint count, uint lane) {
  m1_wf w = m1_wf{0.0f, 0.0f, 0.0f};
  for (uint i = lane; i < count; i += 32u) {
    device const float* q = p + 3ul * ulong(first + i);
    w = m1_wf_merge(w, m1_wf{q[0], q[1], q[2]});
  }
  return m1_wf_simd(w);
}

)MSL";

const char* vec_type(VecMode mode) { return mode == VecMode::Packed ? "packed_float4" : "float4"; }

std::string head_expr(VecMode mode, const std::string& base, const std::string& len) {
  if (mode != VecMode::Peel) return "0u";
  return "min(uint((4L - (" + base + " & 3L)) & 3L), " + len + ")";
}

void emit_reduce_loop(Emitter& e, VecMode mode, const std::string& xptr, const std::string& base,
                      const std::string& len, const std::string& vstmt, const std::string& sstmt) {
  const char* vt = vec_type(mode);
  auto& os = e.body;
  os << "  {\n"
     << "    const uint m1_len = " << len << ";\n"
     << "    const uint m1_hd = " << head_expr(mode, base, "m1_len") << ";\n"
     << "    const uint m1_nv = (m1_len - m1_hd) >> 2;\n"
     << "    device const " << vt << "* m1_xv = (device const " << vt << "*)(" << xptr << " + " << base
     << " + m1_hd);\n"
     << "    for (uint m1_i = m1_tid; m1_i < m1_nv; m1_i += " << e.tg << "u) {\n"
     << "      const float4 v = float4(m1_xv[m1_i]);\n"
     << "      " << vstmt << "\n"
     << "    }\n"
     << "    const uint m1_rem = m1_len - (m1_nv << 2);\n"
     << "    for (uint m1_r = m1_tid; m1_r < m1_rem; m1_r += " << e.tg << "u) {\n"
     << "      const uint m1_j = m1_r < m1_hd ? m1_r : m1_r + (m1_nv << 2);\n"
     << "      const float s = " << xptr << "[" << base << " + long(m1_j)];\n"
     << "      " << sstmt << "\n"
     << "    }\n"
     << "  }\n";
}

void emit_map_loop(Emitter& e, VecMode mode, const std::string& xptr, const std::string& optr,
                   const std::string& base, const std::string& len, const std::string& vexpr,
                   const std::string& sexpr, bool reverse, const std::vector<Aux>& aux = {}) {
  const char* vt = vec_type(mode);
  auto& os = e.body;
  os << "  {\n"
     << "    const uint m1_len = " << len << ";\n"
     << "    const uint m1_hd = " << head_expr(mode, base, "m1_len") << ";\n"
     << "    const uint m1_nv = (m1_len - m1_hd) >> 2;\n"
     << "    device const " << vt << "* m1_xv = (device const " << vt << "*)(" << xptr << " + " << base
     << " + m1_hd);\n"
     << "    device " << vt << "* m1_ov = (device " << vt << "*)(" << optr << " + " << base << " + m1_hd);\n";
  for (const Aux& a : aux) {
    os << "    device const " << vt << "* m1_" << a.name << "v = (device const " << vt << "*)(" << a.ptr
       << " + " << a.base << " + m1_hd);\n";
  }
  if (reverse) {
    os << "    for (int m1_i = int(m1_nv) - 1 - int(m1_tid); m1_i >= 0; m1_i -= " << e.tg << ") {\n";
  } else {
    os << "    for (uint m1_i = m1_tid; m1_i < m1_nv; m1_i += " << e.tg << "u) {\n";
  }
  os << "      const float4 v = float4(m1_xv[m1_i]);\n";
  for (const Aux& a : aux) os << "      const float4 " << a.name << " = float4(m1_" << a.name << "v[m1_i]);\n";
  os << "      m1_ov[m1_i] = " << vt << "(" << vexpr << ");\n"
     << "    }\n"
     << "    const uint m1_rem = m1_len - (m1_nv << 2);\n"
     << "    for (uint m1_r = m1_tid; m1_r < m1_rem; m1_r += " << e.tg << "u) {\n"
     << "      const uint m1_j = m1_r < m1_hd ? m1_r : m1_r + (m1_nv << 2);\n"
     << "      const float s = " << xptr << "[" << base << " + long(m1_j)];\n";
  for (const Aux& a : aux) {
    os << "      const float " << a.name << "s = " << a.ptr << "[" << a.base << " + long(m1_j)];\n";
  }
  os << "      " << optr << "[" << base << " + long(m1_j)] = " << sexpr << ";\n"
     << "    }\n"
     << "  }\n";
}

bool all_aligned(const Emitter& e, std::initializer_list<size_t> idxs) {
  for (size_t i : idxs) {
    if (!e.ops[i].aligned16) return false;
  }
  return true;
}

VecMode pick_mode(const Emitter& e, std::initializer_list<size_t> ptr_idxs, bool bases_mult4) {
  if (!all_aligned(e, ptr_idxs)) return VecMode::Packed;
  return bases_mult4 ? VecMode::Aligned : VecMode::Peel;
}

bool konst_mult(const Operand& o, int64_t m) { return o.konst.has_value() && (*o.konst % m) == 0; }

bool expect_operands(Emitter& e, size_t n, const char* what) {
  if (e.ops.size() != n) {
    e.err = std::string(what) + " expects " + std::to_string(n) + " operands, got " + std::to_string(e.ops.size());
    return false;
  }
  if (!e.ops[0].is_ptr || !e.ops[1].is_ptr) {
    e.err = std::string(what) + ": operands 0 and 1 must be device pointers";
    return false;
  }
  return true;
}

void emit_wf_accumulators(Emitter& e, const std::string& shift_expr) {
  e.body << "  threadgroup m1_wf m1_shw[" << e.nsg() << "];\n"
         << "  const float m1_k = " << shift_expr << ";\n"
         << "  uint m1_cnt = 0u;\n"
         << "  float m1_s = 0.0f;\n"
         << "  float m1_ss = 0.0f;\n";
}

const char* kWfVec = "const float4 d = v - m1_k; m1_s += (d.x + d.y) + (d.z + d.w); m1_ss += dot(d, d); m1_cnt += 4u;";
const char* kWfScalar = "const float d = s - m1_k; m1_s += d; m1_ss = fma(d, d, m1_ss); m1_cnt += 1u;";

std::string tg_wf(const Emitter& e) {
  return "m1_wf_tg(m1_wf_shifted(m1_cnt, m1_s, m1_ss, m1_k), m1_shw, m1_lane, m1_sg, " + std::to_string(e.nsg()) +
         "u)";
}

bool emit_softmax(Emitter& e, const ir::OnlineSoftmaxAttr& a) {
  if (!expect_operands(e, 4, "online_softmax")) return false;
  const std::string x = e.ops[0].expr, o = e.ops[1].expr;
  const VecMode mode = pick_mode(e, {0, 1}, konst_mult(e.ops[3], 4));
  auto& os = e.body;
  os << "  threadgroup float m1_sm[" << e.nsg() << "];\n"
     << "  threadgroup float m1_sd[" << e.nsg() << "];\n"
     << "  const uint m1_L = uint(" << e.ops[3].expr << ");\n"
     << "  if (m1_tgp.x >= uint(" << e.ops[2].expr << ")) return;\n"
     << "  const long m1_base = long(m1_tgp.x) * long(m1_L);\n"
     << "  float m1_m = -3.402823466e+38f;\n"
     << "  float m1_d = 0.0f;\n";
  emit_reduce_loop(e, mode, x, "m1_base", "m1_L",
                   "const float m1_mn = max(m1_m, max(max(v.x, v.y), max(v.z, v.w))); "
                   "m1_d = fma(m1_d, exp(m1_m - m1_mn), (exp(v.x - m1_mn) + exp(v.y - m1_mn)) + "
                   "(exp(v.z - m1_mn) + exp(v.w - m1_mn))); m1_m = m1_mn;",
                   "const float m1_mn = max(m1_m, s); m1_d = fma(m1_d, exp(m1_m - m1_mn), exp(s - m1_mn)); "
                   "m1_m = m1_mn;");
  os << "  for (ushort m1_o = 16; m1_o > 0; m1_o >>= 1) {\n"
     << "    const float m1_om = simd_shuffle_xor(m1_m, m1_o);\n"
     << "    const float m1_od = simd_shuffle_xor(m1_d, m1_o);\n"
     << "    const float m1_mn = max(m1_m, m1_om);\n"
     << "    m1_d = m1_d * exp(m1_m - m1_mn) + m1_od * exp(m1_om - m1_mn);\n"
     << "    m1_m = m1_mn;\n"
     << "  }\n"
     << "  if (m1_lane == 0) {\n"
     << "    m1_sm[m1_sg] = m1_m;\n"
     << "    m1_sd[m1_sg] = m1_d;\n"
     << "  }\n"
     << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "  float m1_rm = m1_sm[0];\n"
     << "  for (uint m1_g = 1; m1_g < " << e.nsg() << "u; ++m1_g) m1_rm = max(m1_rm, m1_sm[m1_g]);\n"
     << "  float m1_rd = 0.0f;\n"
     << "  for (uint m1_g = 0; m1_g < " << e.nsg() << "u; ++m1_g) m1_rd += m1_sd[m1_g] * exp(m1_sm[m1_g] - m1_rm);\n"
     << "  const float m1_inv = 1.0f / m1_rd;\n"
     << "  const float m1_lse = m1_rm + log(m1_rd);\n";
  if (a.log_softmax) {
    emit_map_loop(e, mode, x, o, "m1_base", "m1_L", "v - m1_lse", "s - m1_lse", true);
  } else {
    emit_map_loop(e, mode, x, o, "m1_base", "m1_L", "exp(v - m1_rm) * m1_inv", "exp(s - m1_rm) * m1_inv", true);
  }
  return true;
}

bool emit_row_norm(Emitter& e, const ir::WelfordNormAttr& a) {
  if (!expect_operands(e, 4, "row norm")) return false;
  const std::string x = e.ops[0].expr, o = e.ops[1].expr;
  const VecMode mode = pick_mode(e, {0, 1}, konst_mult(e.ops[3], 4));
  auto& os = e.body;
  os << "  const uint m1_L = uint(" << e.ops[3].expr << ");\n"
     << "  if (m1_tgp.x >= uint(" << e.ops[2].expr << ")) return;\n"
     << "  const long m1_base = long(m1_tgp.x) * long(m1_L);\n";
  if (a.kind == NormKind::InstanceNorm) {
    emit_wf_accumulators(e, "m1_L > 0u ? " + x + "[m1_base] : 0.0f");
    emit_reduce_loop(e, mode, x, "m1_base", "m1_L", kWfVec, kWfScalar);
    os << "  const m1_wf m1_w = " << tg_wf(e) << ";\n"
       << "  const float m1_inv = 1.0f / sqrt(m1_w.m2 / m1_w.n + " << float_literal(a.eps) << ");\n"
       << "  const float m1_sh = -m1_w.mean * m1_inv;\n";
    emit_map_loop(e, mode, x, o, "m1_base", "m1_L", "fma(v, m1_inv, m1_sh)", "fma(s, m1_inv, m1_sh)", true);
    return true;
  }
  const bool l1 = a.kind == NormKind::L1Norm;
  os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  float m1_acc = 0.0f;\n";
  emit_reduce_loop(e, mode, x, "m1_base", "m1_L",
                   l1 ? "m1_acc += (abs(v.x) + abs(v.y)) + (abs(v.z) + abs(v.w));" : "m1_acc += dot(v, v);",
                   l1 ? "m1_acc += abs(s);" : "m1_acc = fma(s, s, m1_acc);");
  os << "  const float m1_tot = m1_sum_tg(m1_acc, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u);\n"
     << "  const float m1_scale = " << (l1 ? "float(m1_L) / m1_tot" : "1.0f / sqrt(m1_tot)") << ";\n";
  emit_map_loop(e, mode, x, o, "m1_base", "m1_L", "v * m1_scale", "s * m1_scale", true);
  return true;
}

bool emit_frobenius(Emitter& e, const ir::WelfordNormAttr& a) {
  auto& os = e.body;
  const long tile_v = 2L * static_cast<long>(e.tg);
  if (a.stage == 1) {
    if (!expect_operands(e, 3, "frobenius stage 1")) return false;
    const char* vt = vec_type(pick_mode(e, {0}, true));
    const std::string x = e.ops[0].expr;
    os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
       << "  const long m1_total = long(" << e.ops[2].expr << ");\n"
       << "  const long m1_nv = m1_total >> 2;\n"
       << "  device const " << vt << "* m1_xv = (device const " << vt << "*)(" << x << ");\n"
       << "  float4 m1_a0 = float4(0.0f);\n"
       << "  float4 m1_a1 = float4(0.0f);\n"
       << "  float m1_acc = 0.0f;\n"
       << "  const long m1_tile = " << tile_v << "L;\n"
       << "  const long m1_gstride = long(m1_ntg.x) * m1_tile;\n"
       << "  for (long m1_t0 = long(m1_tgp.x) * m1_tile; m1_t0 < m1_nv; m1_t0 += m1_gstride) {\n"
       << "    const long m1_b = m1_t0 + long(m1_tid);\n"
       << "    if (m1_t0 + m1_tile <= m1_nv) {\n"
       << "      const float4 v0 = float4(m1_xv[m1_b]);\n"
       << "      const float4 v1 = float4(m1_xv[m1_b + " << e.tg << "L]);\n"
       << "      m1_a0 = fma(v0, v0, m1_a0);\n"
       << "      m1_a1 = fma(v1, v1, m1_a1);\n"
       << "    } else {\n"
       << "      const long m1_end = min(m1_t0 + m1_tile, m1_nv);\n"
       << "      for (long m1_i = m1_b; m1_i < m1_end; m1_i += " << e.tg << "L) {\n"
       << "        const float4 v = float4(m1_xv[m1_i]);\n"
       << "        m1_acc += dot(v, v);\n"
       << "      }\n"
       << "    }\n"
       << "  }\n"
       << "  const float4 m1_a = m1_a0 + m1_a1;\n"
       << "  m1_acc += (m1_a.x + m1_a.y) + (m1_a.z + m1_a.w);\n"
       << "  if (m1_tgp.x == 0u) {\n"
       << "    const long m1_tail = m1_nv << 2;\n"
       << "    const long m1_rem = m1_total - m1_tail;\n"
       << "    for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
       << "      const float s = " << x << "[m1_tail + m1_r];\n"
       << "      m1_acc = fma(s, s, m1_acc);\n"
       << "    }\n"
       << "  }\n"
       << "  const float m1_tot = m1_sum_tg(m1_acc, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u);\n"
       << "  if (m1_tid == 0) " << e.ops[1].expr << "[m1_tgp.x] = m1_tot;\n";
    return true;
  }
  if (a.stage != 2) {
    e.err = "frobenius norm requires stage=1 or stage=2";
    return false;
  }
  if (!expect_operands(e, 5, "frobenius stage 2")) return false;
  const char* vt = vec_type(pick_mode(e, {0, 1}, true));
  const char* pvt = vec_type(pick_mode(e, {2}, true));
  const std::string x = e.ops[0].expr;
  const std::string o = e.ops[1].expr;
  const std::string p = e.ops[2].expr;
  os << "  threadgroup float m1_sh[" << e.nsg() << "];\n"
     << "  const uint m1_np = uint(" << e.ops[4].expr << ");\n"
     << "  const uint m1_npv = m1_np >> 2;\n"
     << "  device const " << pvt << "* m1_pv = (device const " << pvt << "*)(" << p << ");\n"
     << "  float4 m1_p4 = float4(0.0f);\n"
     << "  for (uint m1_i = m1_tid; m1_i < m1_npv; m1_i += " << e.tg << "u) m1_p4 += float4(m1_pv[m1_i]);\n"
     << "  float m1_p = (m1_p4.x + m1_p4.y) + (m1_p4.z + m1_p4.w);\n"
     << "  for (uint m1_i = (m1_npv << 2) + m1_tid; m1_i < m1_np; m1_i += " << e.tg << "u) m1_p += " << p << "[m1_i];\n"
     << "  const float m1_inv = rsqrt(m1_sum_tg(m1_p, m1_sh, m1_lane, m1_sg, " << e.nsg() << "u));\n"
     << "  const long m1_total = long(" << e.ops[3].expr << ");\n"
     << "  const long m1_nv = m1_total >> 2;\n"
     << "  device const " << vt << "* m1_xv = (device const " << vt << "*)(" << x << ");\n"
     << "  device " << vt << "* m1_ov = (device " << vt << "*)(" << o << ");\n"
     << "  const long m1_tile = " << tile_v << "L;\n"
     << "  const long m1_gstride = long(m1_ntg.x) * m1_tile;\n"
     << "  for (long m1_t0 = long(m1_tgp.x) * m1_tile; m1_t0 < m1_nv; m1_t0 += m1_gstride) {\n"
     << "    const long m1_b = m1_t0 + long(m1_tid);\n"
     << "    if (m1_t0 + m1_tile <= m1_nv) {\n"
     << "      const float4 v0 = float4(m1_xv[m1_b]);\n"
     << "      const float4 v1 = float4(m1_xv[m1_b + " << e.tg << "L]);\n"
     << "      m1_ov[m1_b] = " << vt << "(v0 * m1_inv);\n"
     << "      m1_ov[m1_b + " << e.tg << "L] = " << vt << "(v1 * m1_inv);\n"
     << "    } else {\n"
     << "      const long m1_end = min(m1_t0 + m1_tile, m1_nv);\n"
     << "      for (long m1_i = m1_b; m1_i < m1_end; m1_i += " << e.tg << "L) {\n"
     << "        const float4 v = float4(m1_xv[m1_i]);\n"
     << "        m1_ov[m1_i] = " << vt << "(v * m1_inv);\n"
     << "      }\n"
     << "    }\n"
     << "  }\n"
     << "  if (m1_tgp.x == 0u) {\n"
     << "    const long m1_tail = m1_nv << 2;\n"
     << "    const long m1_rem = m1_total - m1_tail;\n"
     << "    for (long m1_r = long(m1_tid); m1_r < m1_rem; m1_r += " << e.tg << "L) {\n"
     << "      " << o << "[m1_tail + m1_r] = " << x << "[m1_tail + m1_r] * m1_inv;\n"
     << "    }\n"
     << "  }\n";
  return true;
}

bool emit_stats_stage1(Emitter& e, const ir::WelfordNormAttr& a) {
  const std::string x = e.ops[0].expr, part = e.ops[1].expr;
  auto& os = e.body;
  if (a.kind == NormKind::BatchNorm) {
    if (!expect_operands(e, 6, "batch_norm stage 1")) return false;
    const VecMode mode = pick_mode(e, {0}, konst_mult(e.ops[4], 4));
    os << "  const uint m1_N = uint(" << e.ops[2].expr << ");\n"
       << "  const uint m1_C = uint(" << e.ops[3].expr << ");\n"
       << "  const uint m1_HW = uint(" << e.ops[4].expr << ");\n"
       << "  const uint m1_CH = uint(" << e.ops[5].expr << ");\n"
       << "  const uint m1_c = m1_tgp.y;\n"
       << "  const uint m1_n0 = (m1_tgp.x * m1_N) / m1_CH;\n"
       << "  const uint m1_n1 = ((m1_tgp.x + 1u) * m1_N) / m1_CH;\n";
    emit_wf_accumulators(e, "(m1_N > 0u && m1_HW > 0u) ? " + x + "[long(m1_c) * long(m1_HW)] : 0.0f");
    os << "  for (uint m1_b = m1_n0; m1_b < m1_n1; ++m1_b) {\n"
       << "  const long m1_base = (long(m1_b) * long(m1_C) + long(m1_c)) * long(m1_HW);\n";
    emit_reduce_loop(e, mode, x, "m1_base", "m1_HW", kWfVec, kWfScalar);
    os << "  }\n"
       << "  const long m1_slot = long(m1_c) * long(m1_CH) + long(m1_tgp.x);\n";
  } else {
    const bool gn = a.kind == NormKind::GroupNorm;
    if (!expect_operands(e, gn ? 6 : 5, gn ? "group_norm stage 1" : "layer_norm stage 1")) return false;
    if (gn) {
      os << "  const uint m1_C = uint(" << e.ops[3].expr << ");\n"
         << "  const long m1_S = long(m1_C / " << a.num_groups << "u) * long(" << e.ops[4].expr << ");\n"
         << "  const uint m1_CH = uint(" << e.ops[5].expr << ");\n";
    } else {
      os << "  const long m1_S = long(" << e.ops[3].expr << ");\n"
         << "  const uint m1_CH = uint(" << e.ops[4].expr << ");\n";
    }
    const bool seg4 = gn ? konst_mult(e.ops[4], 4) : konst_mult(e.ops[3], 4);
    const VecMode mode = pick_mode(e, {0}, seg4);
    os << "  const long m1_cl = ((m1_S + long(m1_CH) - 1L) / long(m1_CH) + 3L) & ~3L;\n"
       << "  const long m1_t0 = min(long(m1_tgp.x) * m1_cl, m1_S);\n"
       << "  const uint m1_n = uint(min(m1_cl, m1_S - m1_t0));\n"
       << "  const long m1_base = long(m1_tgp.y) * m1_S + m1_t0;\n";
    emit_wf_accumulators(e, "m1_S > 0L ? " + x + "[long(m1_tgp.y) * m1_S] : 0.0f");
    emit_reduce_loop(e, mode, x, "m1_base", "m1_n", kWfVec, kWfScalar);
    os << "  const long m1_slot = long(m1_tgp.y) * long(m1_CH) + long(m1_tgp.x);\n";
  }
  os << "  const m1_wf m1_w = " << tg_wf(e) << ";\n"
     << "  if (m1_tid == 0) {\n"
     << "    device float* m1_q = " << part << " + 3L * m1_slot;\n"
     << "    m1_q[0] = m1_w.n;\n"
     << "    m1_q[1] = m1_w.mean;\n"
     << "    m1_q[2] = m1_w.m2;\n"
     << "  }\n";
  return true;
}

bool emit_stats_stage2(Emitter& e, const ir::WelfordNormAttr& a) {
  const size_t nw = a.has_weight ? 1 : 0;
  const size_t nb = a.has_bias ? 1 : 0;
  const size_t ip = 2 + nw + nb;
  const bool ln = a.kind == NormKind::LayerNorm;
  const size_t expected = ip + 1 + (ln ? 3 : 4);
  if (!expect_operands(e, expected, "norm stage 2")) return false;
  const std::string x = e.ops[0].expr, o = e.ops[1].expr, part = e.ops[ip].expr;
  const std::string w = a.has_weight ? e.ops[2].expr : "", b = a.has_bias ? e.ops[2 + nw].expr : "";
  auto& os = e.body;
  const float eps = a.eps;
  if (ln) {
    const bool ns4 = konst_mult(e.ops[ip + 2], 4);
    std::vector<size_t> ptrs = {0, 1};
    if (a.has_weight) ptrs.push_back(2);
    if (a.has_bias) ptrs.push_back(2 + nw);
    bool aligned = true;
    for (size_t pi : ptrs) aligned = aligned && e.ops[pi].aligned16;
    const VecMode mode = (aligned && ns4) ? VecMode::Aligned : VecMode::Packed;
    os << "  const uint m1_L = uint(" << e.ops[ip + 2].expr << ");\n"
       << "  const uint m1_CH = uint(" << e.ops[ip + 3].expr << ");\n"
       << "  const m1_wf m1_w = m1_wf_partials(" << part << ", m1_tgp.y * m1_CH, m1_CH, m1_lane);\n"
       << "  const float m1_inv = 1.0f / sqrt(m1_w.m2 / m1_w.n + " << float_literal(eps) << ");\n"
       << "  const float m1_mu = m1_w.mean;\n"
       << "  const uint m1_tl = ((m1_L + m1_ntg.x - 1u) / m1_ntg.x + 3u) & ~3u;\n"
       << "  const uint m1_t0 = min(m1_tgp.x * m1_tl, m1_L);\n"
       << "  const long m1_base = long(m1_tgp.y) * long(m1_L) + long(m1_t0);\n";
    std::vector<Aux> aux;
    std::string vexpr = "(v - m1_mu) * m1_inv", sexpr = "(s - m1_mu) * m1_inv";
    if (a.has_weight) {
      aux.push_back({"w", w, "long(m1_t0)"});
      vexpr = "(" + vexpr + ") * w";
      sexpr = "(" + sexpr + ") * ws";
    }
    if (a.has_bias) {
      aux.push_back({"b", b, "long(m1_t0)"});
      vexpr = "(" + vexpr + ") + b";
      sexpr = "(" + sexpr + ") + bs";
    }
    emit_map_loop(e, mode, x, o, "m1_base", "min(m1_tl, m1_L - m1_t0)", vexpr, sexpr, true, aux);
    return true;
  }
  const VecMode mode = pick_mode(e, {0, 1}, konst_mult(e.ops[ip + 3], 4));
  os << "  const uint m1_C = uint(" << e.ops[ip + 2].expr << ");\n"
     << "  const uint m1_L = uint(" << e.ops[ip + 3].expr << ");\n"
     << "  const uint m1_CH = uint(" << e.ops[ip + 4].expr << ");\n"
     << "  const uint m1_row = m1_tgp.y;\n"
     << "  const uint m1_c = m1_row % m1_C;\n";
  if (a.kind == NormKind::BatchNorm) {
    os << "  const uint m1_first = m1_c * m1_CH;\n";
  } else {
    const int32_t g = a.num_groups;
    os << "  const uint m1_first = ((m1_row / m1_C) * " << g << "u + m1_c / (m1_C / " << g << "u)) * m1_CH;\n";
  }
  os << "  const m1_wf m1_w = m1_wf_partials(" << part << ", m1_first, m1_CH, m1_lane);\n"
     << "  const float m1_inv = 1.0f / sqrt(m1_w.m2 / m1_w.n + " << float_literal(eps) << ");\n"
     << "  const float m1_scale = m1_inv" << (a.has_weight ? " * " + w + "[m1_c]" : "") << ";\n"
     << "  const float m1_shift = " << (a.has_bias ? b + "[m1_c]" : "0.0f") << " - m1_w.mean * m1_scale;\n"
     << "  const uint m1_tl = ((m1_L + m1_ntg.x - 1u) / m1_ntg.x + 3u) & ~3u;\n"
     << "  const uint m1_t0 = min(m1_tgp.x * m1_tl, m1_L);\n"
     << "  const long m1_base = long(m1_row) * long(m1_L) + long(m1_t0);\n";
  emit_map_loop(e, mode, x, o, "m1_base", "min(m1_tl, m1_L - m1_t0)", "fma(v, m1_scale, m1_shift)",
                "fma(s, m1_scale, m1_shift)", true);
  return true;
}

bool emit_rms(Emitter& e, const ir::WelfordNormAttr& a) {
  if (!expect_operands(e, 5, "rms_norm")) return false;
  const std::string x = e.ops[0].expr, o = e.ops[1].expr;
  const uint32_t chunks = e.tg / 64u;
  if (chunks == 0 || e.tg % 64u != 0) {
    e.err = "rms_norm requires tg_threads to be a multiple of 64";
    return false;
  }
  const Operand& c_op = e.ops[3];
  const bool sram = a.rms_sram_cache && c_op.konst.has_value() && *c_op.konst > 0 && *c_op.konst <= 64 &&
                    konst_mult(e.ops[4], 64);
  auto& os = e.body;
  os << "  threadgroup float m1_part[" << e.tg << "];\n"
     << "  threadgroup float4 m1_scl4[16];\n"
     << "  threadgroup float* m1_scl = (threadgroup float*)m1_scl4;\n"
     << "  const uint m1_C = uint(" << c_op.expr << ");\n"
     << "  const uint m1_HW = uint(" << e.ops[4].expr << ");\n"
     << "  const uint m1_pos = m1_tid & 63u;\n"
     << "  const uint m1_cc = m1_tid >> 6;\n"
     << "  const uint m1_s0 = m1_tgp.x * 64u;\n"
     << "  const long m1_nb = long(m1_tgp.y) * long(m1_C) * long(m1_HW) + long(m1_s0);\n"
     << "  float m1_acc = 0.0f;\n";
  if (sram) {
    const char* vt = vec_type(pick_mode(e, {0, 1}, true));
    const int64_t c = *c_op.konst;
    os << "  threadgroup float4 m1_tile[" << c * 16 << "];\n"
       << "  for (uint m1_i = m1_tid; m1_i < " << c * 16 << "u; m1_i += " << e.tg << "u) {\n"
       << "    m1_tile[m1_i] = float4(((device const " << vt << "*)(" << x
       << " + m1_nb + long(m1_i >> 4) * long(m1_HW)))[m1_i & 15u]);\n"
       << "  }\n"
       << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "  threadgroup const float* m1_tf = (threadgroup const float*)m1_tile;\n"
       << "  for (uint m1_c = m1_cc; m1_c < " << c << "u; m1_c += " << chunks << "u) {\n"
       << "    const float v = m1_tf[m1_c * 64u + m1_pos];\n"
       << "    m1_acc = fma(v, v, m1_acc);\n"
       << "  }\n";
  } else {
    os << "  const bool m1_ok = m1_s0 + m1_pos < m1_HW;\n"
       << "  if (m1_ok) {\n"
       << "    for (uint m1_c = m1_cc; m1_c < m1_C; m1_c += " << chunks << "u) {\n"
       << "      const float v = " << x << "[m1_nb + long(m1_c) * long(m1_HW) + long(m1_pos)];\n"
       << "      m1_acc = fma(v, v, m1_acc);\n"
       << "    }\n"
       << "  }\n";
  }
  os << "  m1_part[m1_tid] = m1_acc;\n"
     << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "  if (m1_tid < 64u) {\n"
     << "    float m1_t = 0.0f;\n"
     << "    for (uint m1_j = 0; m1_j < " << chunks << "u; ++m1_j) m1_t += m1_part[m1_j * 64u + m1_tid];\n"
     << "    m1_scl[m1_tid] = 1.0f / sqrt(m1_t / float(m1_C) + " << float_literal(a.eps) << ");\n"
     << "  }\n"
     << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  if (sram) {
    const char* vt = vec_type(pick_mode(e, {0, 1}, true));
    const int64_t c = *c_op.konst;
    os << "  for (uint m1_i = m1_tid; m1_i < " << c * 16 << "u; m1_i += " << e.tg << "u) {\n"
       << "    ((device " << vt << "*)(" << o << " + m1_nb + long(m1_i >> 4) * long(m1_HW)))[m1_i & 15u] = " << vt
       << "(m1_tile[m1_i] * m1_scl4[m1_i & 15u]);\n"
       << "  }\n";
  } else {
    os << "  if (m1_ok) {\n"
       << "    const float m1_sc = m1_scl[m1_pos];\n"
       << "    for (uint m1_c = m1_cc; m1_c < m1_C; m1_c += " << chunks << "u) {\n"
       << "      const long m1_off = m1_nb + long(m1_c) * long(m1_HW) + long(m1_pos);\n"
       << "      " << o << "[m1_off] = " << x << "[m1_off] * m1_sc;\n"
       << "    }\n"
       << "  }\n";
  }
  return true;
}

bool emit_welford(Emitter& e, const ir::WelfordNormAttr& a) {
  switch (a.kind) {
    case NormKind::InstanceNorm:
    case NormKind::L1Norm:
    case NormKind::L2Norm:
      return emit_row_norm(e, a);
    case NormKind::RMSNorm:
      return emit_rms(e, a);
    case NormKind::FrobeniusNorm:
      return emit_frobenius(e, a);
    case NormKind::BatchNorm:
    case NormKind::GroupNorm:
    case NormKind::LayerNorm:
      if (a.kind == NormKind::GroupNorm && a.num_groups <= 0) {
        e.err = "group_norm requires num_groups > 0";
        return false;
      }
      if (a.stage == 1) return emit_stats_stage1(e, a);
      if (a.stage == 2) return emit_stats_stage2(e, a);
      e.err = "batch/group/layer norm require stage=1 (statistics) or stage=2 (normalize)";
      return false;
  }
  e.err = "unknown norm kind";
  return false;
}

}  // namespace

CompiledMSL try_emit_norm_softmax_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  if (matched_out != nullptr) *matched_out = false;
  const Op* fused = nullptr;
  bool foreign = false;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    if (op.kind == OpKind::OnlineSoftmax || op.kind == OpKind::WelfordReduce) {
      if (fused != nullptr) {
        foreign = true;
      }
      fused = &op;
    } else if (op.kind != OpKind::Constant && op.kind != OpKind::Cast && op.kind != OpKind::Return) {
      foreign = true;
    }
  }
  if (fused == nullptr) return CompiledMSL{false, "", {}, ""};
  if (matched_out != nullptr) *matched_out = true;
  if (foreign) {
    return CompiledMSL{false, "", {}, "norm/softmax kernels must contain exactly one fused op and no other compute"};
  }

  Emitter e{kernel};
  uint32_t tg = 256;
  if (const auto* sa = std::get_if<ir::OnlineSoftmaxAttr>(&fused->attr)) tg = sa->tg_threads;
  if (const auto* wa = std::get_if<ir::WelfordNormAttr>(&fused->attr)) tg = wa->tg_threads;
  if (tg == 0 || tg % target.simd_width != 0 || tg > target.max_threads_per_threadgroup || target.simd_width != 32) {
    return CompiledMSL{false, "", {}, "norm/softmax tg_threads must be a multiple of the 32-wide SIMD group"};
  }
  e.tg = tg;

  for (ValueId vid : fused->operands) {
    Operand o;
    if (!resolve_operand(kernel, vid, o, e.err)) return CompiledMSL{false, "", {}, e.err};
    e.ops.push_back(std::move(o));
  }
  if (e.ops.size() < 2 || !build_slots(e, fused->operands[1])) {
    return CompiledMSL{false, "", {}, e.err.empty() ? "norm/softmax op needs x_ptr and out_ptr" : e.err};
  }

  bool ok = false;
  if (const auto* sa = std::get_if<ir::OnlineSoftmaxAttr>(&fused->attr)) {
    ok = emit_softmax(e, *sa);
  } else if (const auto* wa = std::get_if<ir::WelfordNormAttr>(&fused->attr)) {
    ok = emit_welford(e, *wa);
  } else {
    e.err = "fused norm/softmax op is missing its attribute payload";
  }
  if (!ok) return CompiledMSL{false, "", {}, e.err};

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

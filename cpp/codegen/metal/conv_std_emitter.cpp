#include "codegen/metal/conv_std_emitter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::ConvAttr;
using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::ScalarDType;
using ir::ValueId;

constexpr uint32_t kMaxFragsPerSimdgroup = 16;

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
  return "float";
}

std::string msl_packed_vec4(const std::string& elem) { return elem == "half" ? "packed_half4" : "packed_float4"; }
std::string msl_vec4(const std::string& elem) { return elem == "half" ? "half4" : "float4"; }

struct ConvDim {
  std::string in = "1L";
  std::string out = "1L";
  int32_t k = 1;
  int32_t s = 1;
  int32_t p = 0;
  int32_t d = 1;
};

struct ConvPlan {
  const Op* op = nullptr;
  ConvAttr attr;
  std::string x_name;
  std::string w_name;
  std::string out_name;
  std::string bias_name;
  std::string x_elem;
  std::string w_elem;
  std::string out_elem;
  std::string bias_elem;
  std::string n_expr;
  std::string cin_expr;
  std::string cout_expr;
  std::array<ConvDim, 3> dims;
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t num_sgs = 4;
  uint32_t sg_m = 2;
  uint32_t sg_n = 2;
  uint32_t num_threads = 128;
  uint32_t smem_floats = 0;
  bool pointwise = false;
  bool idx32 = false;
  bool k_mult_bk = false;
};

std::optional<uint32_t> param_index_of(const Kernel& kernel, ValueId vid) {
  const ir::Value& v = kernel.value(vid);
  if (v.def_op != ir::kInvalidId || v.def_block != kernel.entry_block) return std::nullopt;
  if (v.arg_idx >= kernel.params.size()) return std::nullopt;
  return v.arg_idx;
}

std::optional<std::string> scalar_expr(const Kernel& kernel, ValueId vid) {
  if (auto pidx = param_index_of(kernel, vid)) {
    const ir::KernelParam& kp = kernel.params[*pidx];
    if (kp.type.is_ptr()) return std::nullopt;
    if (kp.is_constexpr) return std::to_string(kp.constexpr_val) + "L";
    return "((long)" + kp.name + ")";
  }
  const ir::Value& v = kernel.value(vid);
  if (v.def_op == ir::kInvalidId) return std::nullopt;
  const Op& def = kernel.op(v.def_op);
  if (def.kind == OpKind::Constant) {
    const auto* ca = std::get_if<ir::ConstAttr>(&def.attr);
    if (ca == nullptr) return std::nullopt;
    if (const auto* i = std::get_if<int64_t>(&ca->val)) return std::to_string(*i) + "L";
    if (const auto* u = std::get_if<uint64_t>(&ca->val)) return std::to_string(*u) + "L";
    return std::nullopt;
  }
  if (def.kind == OpKind::Cast && def.operands.size() == 1) return scalar_expr(kernel, def.operands[0]);
  return std::nullopt;
}

std::optional<int64_t> scalar_const(const Kernel& kernel, ValueId vid) {
  if (auto pidx = param_index_of(kernel, vid)) {
    const ir::KernelParam& kp = kernel.params[*pidx];
    if (kp.is_constexpr && !kp.type.is_ptr()) return kp.constexpr_val;
    return std::nullopt;
  }
  const ir::Value& v = kernel.value(vid);
  if (v.def_op == ir::kInvalidId) return std::nullopt;
  const Op& def = kernel.op(v.def_op);
  if (def.kind == OpKind::Cast && def.operands.size() == 1) return scalar_const(kernel, def.operands[0]);
  if (def.kind != OpKind::Constant) return std::nullopt;
  const auto* ca = std::get_if<ir::ConstAttr>(&def.attr);
  if (ca == nullptr) return std::nullopt;
  if (const auto* i = std::get_if<int64_t>(&ca->val)) return *i;
  if (const auto* u = std::get_if<uint64_t>(&ca->val)) return static_cast<int64_t>(*u);
  return std::nullopt;
}

std::optional<int64_t> constexpr_param(const Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name && kp.constexpr_val > 0) return kp.constexpr_val;
  }
  return std::nullopt;
}

bool valid_tile(uint32_t v) { return v == 16 || v == 32 || v == 64 || v == 128; }
bool valid_bk(uint32_t v) { return v == 8 || v == 16 || v == 32; }

std::optional<std::string> float_ptr(const Kernel& kernel, ValueId vid, std::string& name) {
  const auto idx = param_index_of(kernel, vid);
  if (!idx || !kernel.params[*idx].type.is_ptr()) return std::nullopt;
  const ScalarDType dt = std::get<ir::PtrType>(kernel.params[*idx].type.value).pointee;
  if (dt != ScalarDType::F32 && dt != ScalarDType::F16) return std::nullopt;
  name = kernel.params[*idx].name;
  return msl_scalar(dt);
}

std::optional<ConvPlan> build_plan(const Kernel& kernel, const ir::TargetSpec& target, std::string& err) {
  ConvPlan plan;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::ImplicitGemmConv:
        if (plan.op != nullptr) {
          err = "conv_std emitter supports exactly one implicit_gemm_conv op per kernel";
          return std::nullopt;
        }
        plan.op = &op;
        break;
      case OpKind::Constant:
      case OpKind::Cast:
      case OpKind::Return:
        break;
      default:
        return std::nullopt;
    }
  }
  if (plan.op == nullptr) return std::nullopt;
  const auto* ca = std::get_if<ConvAttr>(&plan.op->attr);
  if (ca == nullptr || ca->transposed || ca->is_depthwise) return std::nullopt;
  plan.attr = *ca;

  const uint32_t nd = ca->ndim;
  const auto& ops = plan.op->operands;
  const size_t ptr_count = ca->has_bias ? 4 : 3;
  if (nd < 1 || nd > 3 || ops.size() != ptr_count + 3 + 2 * nd || ca->kernel_size.size() != nd ||
      ca->stride.size() != nd || ca->padding.size() != nd || ca->dilation.size() != nd || ca->groups <= 0) {
    err = "implicit_gemm_conv has malformed operands or attributes; run ConvLoweringPass first";
    return std::nullopt;
  }

  auto x_elem = float_ptr(kernel, ops[0], plan.x_name);
  auto w_elem = float_ptr(kernel, ops[1], plan.w_name);
  auto o_elem = float_ptr(kernel, ops[2], plan.out_name);
  if (!x_elem || !w_elem || !o_elem) {
    err = "implicit_gemm_conv x/w/out operands must be f32 or f16 pointer kernel parameters";
    return std::nullopt;
  }
  plan.x_elem = *x_elem;
  plan.w_elem = *w_elem;
  plan.out_elem = *o_elem;
  if (ca->has_bias) {
    auto b_elem = float_ptr(kernel, ops[3], plan.bias_name);
    if (!b_elem) {
      err = "implicit_gemm_conv bias operand must be an f32 or f16 pointer kernel parameter";
      return std::nullopt;
    }
    plan.bias_elem = *b_elem;
  }

  std::vector<std::string> exprs;
  for (size_t i = ptr_count; i < ops.size(); ++i) {
    auto e = scalar_expr(kernel, ops[i]);
    if (!e) {
      err = "implicit_gemm_conv shape operands must be scalar kernel parameters or constants";
      return std::nullopt;
    }
    exprs.push_back(*e);
  }
  plan.n_expr = exprs[0];
  plan.cin_expr = exprs[1];
  plan.cout_expr = exprs[2];
  bool unit = true;
  int64_t ks = 1;
  for (uint32_t i = 0; i < nd; ++i) {
    ConvDim& dim = plan.dims[3 - nd + i];
    dim.in = exprs[3 + i];
    dim.out = exprs[3 + nd + i];
    dim.k = ca->kernel_size[i];
    dim.s = ca->stride[i];
    dim.p = ca->padding[i];
    dim.d = ca->dilation[i];
    if (dim.k <= 0 || dim.s <= 0 || dim.d <= 0 || dim.p < 0) {
      err = "implicit_gemm_conv requires positive kernel/stride/dilation and non-negative padding";
      return std::nullopt;
    }
    unit = unit && dim.k == 1 && dim.s == 1 && dim.p == 0;
    ks *= dim.k;
  }
  plan.pointwise = unit;

  plan.bm = ca->bm;
  plan.bn = ca->bn;
  plan.bk = ca->bk;
  plan.num_sgs = ca->num_simdgroups == 8 ? 8u : 4u;
  if (auto v = constexpr_param(kernel, "BLOCK_M")) plan.bm = static_cast<uint32_t>(*v);
  if (auto v = constexpr_param(kernel, "BLOCK_N")) plan.bn = static_cast<uint32_t>(*v);
  if (auto v = constexpr_param(kernel, "BLOCK_K")) plan.bk = static_cast<uint32_t>(*v);
  if (auto v = constexpr_param(kernel, "NUM_SG")) plan.num_sgs = (*v >= 8) ? 8u : 4u;
  if (plan.bm == 16 && plan.bn == 16) plan.num_sgs = 4;

  if (plan.num_sgs == 8) {
    if (plan.bm >= plan.bn && plan.bm >= 32) {
      plan.sg_m = 4;
      plan.sg_n = 2;
    } else {
      plan.sg_m = 2;
      plan.sg_n = 4;
    }
  } else {
    plan.sg_m = 2;
    plan.sg_n = 2;
  }
  plan.num_threads = plan.num_sgs * 32;

  if (!valid_tile(plan.bm) || !valid_tile(plan.bn) || !valid_bk(plan.bk) ||
      plan.bm < plan.sg_m * 8 || plan.bn < plan.sg_n * 8 ||
      (plan.bm / (plan.sg_m * 8)) * (plan.bn / (plan.sg_n * 8)) > kMaxFragsPerSimdgroup) {
    err = "implicit_gemm_conv tile (bm, bn, bk) unsupported; expected bm/bn in {16,32,64,128}, bk in {8,16,32}";
    return std::nullopt;
  }

  const auto cin = scalar_const(kernel, ops[ptr_count + 1]);
  const auto cout = scalar_const(kernel, ops[ptr_count + 2]);
  if (cin) {
    if (*cin % ca->groups != 0) {
      err = "implicit_gemm_conv C_in must be divisible by groups";
      return std::nullopt;
    }
    const int64_t k_total = (*cin / ca->groups) * ks;
    plan.k_mult_bk = k_total > 0 && (k_total % plan.bk == 0);
  }
  {
    int64_t pin = 1;
    bool known = cin.has_value() && cout.has_value();
    for (uint32_t i = 0; i < nd; ++i) {
      const auto e = scalar_const(kernel, ops[ptr_count + 3 + i]);
      known = known && e.has_value();
      if (e) pin *= *e;
    }
    constexpr int64_t kInt32Limit = int64_t{1} << 31;
    plan.idx32 = known && *cin * pin < kInt32Limit && *cout * (*cin / ca->groups) * ks < kInt32Limit;
  }

  const uint32_t sg_elems = (plan.bm / plan.sg_m) * (plan.bn / plan.sg_n);
  plan.smem_floats = std::max(plan.bm * (plan.bk + 4) + plan.bk * (plan.bn + 4), sg_elems);
  if (plan.smem_floats * 4 > target.max_threadgroup_memory_bytes ||
      plan.num_threads > target.max_threads_per_threadgroup) {
    err = "implicit_gemm_conv tile exceeds threadgroup limits";
    return std::nullopt;
  }
  return plan;
}

class ConvStdWriter {
 public:
  ConvStdWriter(const Kernel& kernel, const ConvPlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    emit_signature();
    emit_prologue();
    if (plan_.pointwise) {
      emit_pointwise_loop();
    } else {
      emit_conv_loop();
    }
    emit_epilogue();
    os_ << "}\n";
    return os_.str();
  }

 private:
  uint32_t fm() const { return plan_.bm / (plan_.sg_m * 8); }
  uint32_t fn() const { return plan_.bn / (plan_.sg_n * 8); }
  uint32_t w_vpr() const { return plan_.bk / 4; }
  uint32_t x_vpr() const { return plan_.bn / 4; }
  uint32_t w_rstep() const { return plan_.num_threads / w_vpr(); }
  uint32_t x_rstep() const { return plan_.num_threads / x_vpr(); }
  uint32_t w_per() const { return (plan_.bm * w_vpr() + plan_.num_threads - 1) / plan_.num_threads; }
  uint32_t x_per() const { return (plan_.bk * x_vpr() + plan_.num_threads - 1) / plan_.num_threads; }

  void emit_signature() {
    os_ << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    os_ << "kernel void " << kernel_.name << "(\n";
    uint32_t slot = 0;
    for (const auto& kp : kernel_.params) {
      if (kp.is_constexpr) continue;
      if (kp.type.is_ptr()) {
        const auto pt = std::get<ir::PtrType>(kp.type.value);
        const bool is_out = kp.name == plan_.out_name;
        os_ << "    device " << (is_out ? "" : "const ") << msl_scalar(pt.pointee) << "* " << kp.name;
      } else {
        os_ << "    constant " << msl_scalar(std::get<ScalarDType>(kp.type.value)) << "& " << kp.name;
      }
      os_ << " [[buffer(" << slot++ << ")]],\n";
    }
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    os_ << "    uint m1_tid [[thread_index_in_threadgroup]],\n";
    os_ << "    uint m1_sg [[simdgroup_index_in_threadgroup]]) {\n";
  }

  void emit_prologue() {
    const ConvDim& D = plan_.dims[0];
    const ConvDim& H = plan_.dims[1];
    const ConvDim& W = plan_.dims[2];
    os_ << "  constexpr int m1_BM = " << plan_.bm << ";\n";
    os_ << "  constexpr int m1_BN = " << plan_.bn << ";\n";
    os_ << "  constexpr int m1_BK = " << plan_.bk << ";\n";
    os_ << "  constexpr int m1_LDW = m1_BK + 4;\n";
    os_ << "  constexpr int m1_LDX = m1_BN + 4;\n";
    os_ << "  constexpr int m1_G = " << plan_.attr.groups << ";\n";
    os_ << "  constexpr int m1_KD = " << D.k << ", m1_KH = " << H.k << ", m1_KW = " << W.k << ";\n";
    os_ << "  constexpr int m1_KHW = m1_KH * m1_KW;\n";
    os_ << "  constexpr int m1_KS = m1_KD * m1_KHW;\n";
    os_ << "  const long m1_N = " << plan_.n_expr << ";\n";
    os_ << "  const long m1_Cin = " << plan_.cin_expr << ";\n";
    os_ << "  const long m1_Cout = " << plan_.cout_expr << ";\n";
    os_ << "  const int m1_Din = (int)" << D.in << ", m1_Hin = (int)" << H.in << ", m1_Win = (int)" << W.in << ";\n";
    os_ << "  const int m1_Dout = (int)" << D.out << ", m1_Hout = (int)" << H.out << ", m1_Wout = (int)" << W.out
        << ";\n";
    os_ << "  const int m1_Cig = (int)(m1_Cin / m1_G);\n";
    os_ << "  const int m1_Cog = (int)(m1_Cout / m1_G);\n";
    os_ << "  const int m1_K = m1_Cig * m1_KS;\n";
    os_ << "  const int m1_KFull = (m1_K / m1_BK) * m1_BK;\n";
    os_ << "  const uint m1_HWo = (uint)m1_Hout * (uint)m1_Wout;\n";
    os_ << "  const uint m1_P = (uint)m1_Dout * m1_HWo;\n";
    os_ << "  const uint m1_M = (uint)m1_N * m1_P;\n";
    os_ << "  const long m1_Pin = (long)m1_Din * (long)m1_Hin * (long)m1_Win;\n";
    os_ << "  typedef " << (plan_.idx32 ? "int" : "long") << " m1_idx;\n";
    os_ << "  const m1_idx m1_PinI = (m1_idx)m1_Pin;\n";
    os_ << "  const int m1_g = (int)m1_tgid.z;\n";
    os_ << "  const int m1_co0 = (int)m1_tgid.y * m1_BM;\n";
    os_ << "  const uint m1_m0 = m1_tgid.x * (uint)m1_BN;\n";
    os_ << "  if (m1_g >= m1_G || m1_co0 >= m1_Cog || m1_m0 >= m1_M) return;\n";
    os_ << "  threadgroup float m1_smem[" << plan_.smem_floats << "];\n";
    os_ << "  threadgroup float* m1_sw = m1_smem;\n";
    os_ << "  threadgroup float* m1_sx = m1_smem + m1_BM * m1_LDW;\n";
    os_ << "  const int m1_wc = (int)(m1_tid % " << w_vpr() << "u) * 4;\n";
    os_ << "  const int m1_wr0 = (int)(m1_tid / " << w_vpr() << "u);\n";
    os_ << "  const int m1_xc = (int)(m1_tid % " << x_vpr() << "u) * 4;\n";
    os_ << "  const int m1_xr0 = (int)(m1_tid / " << x_vpr() << "u);\n";
    os_ << "  device const " << plan_.w_elem << "* m1_wb = " << plan_.w_name
        << " + (long)(m1_g * m1_Cog + m1_co0) * (long)m1_K;\n";
    os_ << "  const uint m1_mx = m1_m0 + (uint)m1_xc;\n";
    os_ << "  const uint m1_nx = (m1_mx < m1_M) ? (m1_mx / m1_P) : 0u;\n";
    os_ << "  const uint m1_px = (m1_mx < m1_M) ? (m1_mx - m1_nx * m1_P) : 0u;\n";
    os_ << "  device const " << plan_.x_elem << "* m1_xb = " << plan_.x_name
        << " + ((long)m1_nx * m1_Cin + (long)m1_g * (long)m1_Cig) * m1_Pin;\n";
    os_ << "  const int m1_sgr = (int)(m1_sg / " << plan_.sg_n << "u) * " << (fm() * 8) << ";\n";
    os_ << "  const int m1_sgc = (int)(m1_sg % " << plan_.sg_n << "u) * " << (fn() * 8) << ";\n";
    os_ << "  simdgroup_float8x8 m1_acc[" << fm() << "][" << fn() << "];\n";
    for (uint32_t i = 0; i < fm(); ++i) {
      for (uint32_t j = 0; j < fn(); ++j) {
        os_ << "  m1_acc[" << i << "][" << j << "] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << "  (void)m1_KFull; (void)m1_PinI; (void)m1_KS; (void)m1_KHW;\n";
  }

  void emit_mma(const std::string& ind) {
    for (uint32_t kk = 0; kk < plan_.bk; kk += 8) {
      os_ << ind << "{\n";
      os_ << ind << "  simdgroup_float8x8 m1_fa[" << fm() << "];\n";
      os_ << ind << "  simdgroup_float8x8 m1_fb[" << fn() << "];\n";
      os_ << ind << "  simdgroup_load(m1_fa[0], m1_sw + m1_sgr * m1_LDW + " << kk << ", m1_LDW);\n";
      for (uint32_t j = 0; j < fn(); ++j) {
        os_ << ind << "  simdgroup_load(m1_fb[" << j << "], m1_sx + " << kk << " * m1_LDX + m1_sgc + " << (j * 8)
            << ", m1_LDX);\n";
      }
      for (uint32_t i = 0; i < fm(); ++i) {
        if (i + 1 < fm()) {
          os_ << ind << "  simdgroup_load(m1_fa[" << (i + 1) << "], m1_sw + (m1_sgr + " << ((i + 1) * 8)
              << ") * m1_LDW + " << kk << ", m1_LDW);\n";
        }
        for (uint32_t j = 0; j < fn(); ++j) {
          os_ << ind << "  simdgroup_multiply_accumulate(m1_acc[" << i << "][" << j << "], m1_fa[" << i << "], m1_fb["
              << j << "], m1_acc[" << i << "][" << j << "]);\n";
        }
      }
      os_ << ind << "}\n";
    }
  }

  void emit_safe_w_load(const std::string& k0, const std::string& ind) {
    const std::string wpv4 = msl_packed_vec4(plan_.w_elem);
    const bool need_row_guard = w_per() * w_rstep() > plan_.bm;
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_row_guard && ((p + 1) * w_rstep() > plan_.bm);
      os_ << ind << "{\n";
      os_ << ind << "  const int m1_r = m1_wr0 + " << (p * w_rstep()) << ";\n";
      if (guard) os_ << ind << "  if (m1_r < m1_BM) {\n";
      os_ << ind << "  const int m1_k = (" << k0 << ") + m1_wc;\n";
      os_ << ind << "  float4 m1_v = float4(0.0f);\n";
      os_ << ind << "  if (m1_co0 + m1_r < m1_Cog) {\n";
      os_ << ind << "    if (m1_k + 3 < m1_K) m1_v = float4(*(const device " << wpv4
          << "*)(m1_wb + (m1_idx)m1_r * (m1_idx)m1_K + m1_k));\n";
      os_ << ind << "    else { for (int m1_e = 0; m1_e < 4; ++m1_e) if (m1_k + m1_e < m1_K) m1_v[m1_e] = "
          << "(float)m1_wb[(m1_idx)m1_r * (m1_idx)m1_K + m1_k + m1_e]; }\n";
      os_ << ind << "  }\n";
      os_ << ind << "  *(threadgroup float4*)(m1_sw + m1_r * m1_LDW + m1_wc) = m1_v;\n";
      if (guard) os_ << ind << "  }\n";
      os_ << ind << "}\n";
    }
  }

  void emit_pointwise_loop() {
    const std::string xpv4 = msl_packed_vec4(plan_.x_elem);
    const bool need_x_guard = x_per() * x_rstep() > plan_.bk;
    os_ << "  const bool m1_x4 = (m1_mx + 3u < m1_M) && (m1_px + 3u < m1_P);\n";
    os_ << "  device const " << plan_.x_elem << "* m1_xp = m1_xb + (long)m1_px;\n";
    os_ << "  for (int m1_k0 = 0; m1_k0 < m1_K; m1_k0 += m1_BK) {\n";
    emit_safe_w_load("m1_k0", "    ");
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      os_ << "    {\n";
      os_ << "      const int m1_rr = m1_xr0 + " << (p * x_rstep()) << ";\n";
      if (guard) os_ << "      if (m1_rr < m1_BK) {\n";
      os_ << "      const int m1_k = m1_k0 + m1_rr;\n";
      os_ << "      float4 m1_v = float4(0.0f);\n";
      os_ << "      if (m1_k < m1_K) {\n";
      os_ << "        if (m1_x4) {\n";
      os_ << "          m1_v = float4(*(const device " << xpv4 << "*)(m1_xp + (m1_idx)m1_k * m1_PinI));\n";
      os_ << "        } else {\n";
      os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "            const uint m1_me = m1_mx + (uint)m1_e;\n";
      os_ << "            if (m1_me < m1_M) {\n";
      os_ << "              const uint m1_ne = m1_me / m1_P;\n";
      os_ << "              const uint m1_pe = m1_me - m1_ne * m1_P;\n";
      os_ << "              m1_v[m1_e] = (float)" << plan_.x_name
          << "[((long)m1_ne * m1_Cin + (long)(m1_g * m1_Cig + m1_k)) * m1_Pin + (long)m1_pe];\n";
      os_ << "            }\n";
      os_ << "          }\n";
      os_ << "        }\n";
      os_ << "      }\n";
      os_ << "      *(threadgroup float4*)(m1_sx + m1_rr * m1_LDX + m1_xc) = m1_v;\n";
      if (guard) os_ << "      }\n";
      os_ << "    }\n";
    }
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma("    ");
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
  }

  void emit_conv_loop() {
    const ConvDim& D = plan_.dims[0];
    const ConvDim& H = plan_.dims[1];
    const ConvDim& W = plan_.dims[2];
    const std::string wpv4 = msl_packed_vec4(plan_.w_elem);
    const std::string xpv4 = msl_packed_vec4(plan_.x_elem);
    const bool need_w_guard = w_per() * w_rstep() > plan_.bm;
    const bool need_x_guard = x_per() * x_rstep() > plan_.bk;
    const bool use_step = x_per() <= 2;

    const int32_t khw = H.k * W.k;
    const int32_t ks = D.k * khw;
    const int32_t d_ci = static_cast<int32_t>(plan_.bk) / ks;
    const int32_t d_rem = static_cast<int32_t>(plan_.bk) % ks;
    const int32_t d_kd = d_rem / khw;
    const int32_t d_rem2 = d_rem % khw;
    const int32_t d_kh = d_rem2 / W.k;
    const int32_t d_kw = d_rem2 % W.k;

    os_ << "  const uint m1_m1 = m1_m0 + (uint)(m1_BN - 1);\n";
    os_ << "  const uint m1_n0 = m1_m0 / m1_P;\n";
    os_ << "  const uint m1_p0 = m1_m0 - m1_n0 * m1_P;\n";
    os_ << "  const uint m1_n1 = (m1_m1 < m1_M) ? (m1_m1 / m1_P) : (m1_n0 + 1u);\n";
    os_ << "  const uint m1_p1 = (m1_m1 < m1_M) ? (m1_m1 - m1_n1 * m1_P) : 0u;\n";
    os_ << "  const int m1_od0 = (int)(m1_p0 / m1_HWo);\n";
    os_ << "  const uint m1_pr0 = m1_p0 - (uint)m1_od0 * m1_HWo;\n";
    os_ << "  const int m1_oh0 = (int)(m1_pr0 / (uint)m1_Wout);\n";
    os_ << "  const int m1_ow0 = (int)(m1_pr0 - (uint)m1_oh0 * (uint)m1_Wout);\n";
    os_ << "  const int m1_id_lo = m1_od0 * " << D.s << " - " << D.p << ";\n";
    os_ << "  const int m1_ih_lo = m1_oh0 * " << H.s << " - " << H.p << ";\n";
    os_ << "  const int m1_iw_lo = m1_ow0 * " << W.s << " - " << W.p << ";\n";
    os_ << "  const bool m1_tg_fast = (m1_co0 + m1_BM <= m1_Cog) && (m1_m1 < m1_M) && (m1_n0 == m1_n1) &&\n";
    os_ << "      (m1_p0 / (uint)m1_Wout == m1_p1 / (uint)m1_Wout) &&\n";
    os_ << "      (m1_id_lo >= 0) && (m1_id_lo + " << ((D.k - 1) * D.d) << " < m1_Din) &&\n";
    os_ << "      (m1_ih_lo >= 0) && (m1_ih_lo + " << ((H.k - 1) * H.d) << " < m1_Hin) &&\n";
    os_ << "      (m1_iw_lo >= 0) && (m1_iw_lo + " << ((static_cast<int32_t>(plan_.bn) - 1) * W.s + (W.k - 1) * W.d)
        << " < m1_Win);\n";

    os_ << "  if (m1_tg_fast) {\n";
    os_ << "    device const " << plan_.x_elem << "* m1_xpix = " << plan_.x_name
        << " + ((long)m1_n0 * m1_Cin + (long)m1_g * (long)m1_Cig) * m1_Pin"
        << " + ((long)m1_id_lo * (long)m1_Hin + (long)m1_ih_lo) * (long)m1_Win + (long)(m1_iw_lo + m1_xc * " << W.s
        << ");\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      os_ << "    device const " << plan_.w_elem << "* m1_wp" << p << " = m1_wb + (long)(m1_wr0 + " << (p * w_rstep())
          << ") * (long)m1_K + m1_wc;\n";
    }
    if (use_step) {
      for (uint32_t p = 0; p < x_per(); ++p) {
        os_ << "    int m1_k_init" << p << " = m1_xr0 + " << (p * x_rstep()) << ";\n";
        os_ << "    int m1_ci" << p << " = m1_k_init" << p << " / m1_KS;\n";
        os_ << "    int m1_rt" << p << " = m1_k_init" << p << " - m1_ci" << p << " * m1_KS;\n";
        if (D.k > 1) {
          os_ << "    int m1_kd" << p << " = m1_rt" << p << " / m1_KHW;\n";
          os_ << "    m1_rt" << p << " -= m1_kd" << p << " * m1_KHW;\n";
        }
        if (H.k > 1) {
          os_ << "    int m1_kh" << p << " = m1_rt" << p << " / m1_KW;\n";
          os_ << "    int m1_kw" << p << " = m1_rt" << p << " - m1_kh" << p << " * m1_KW;\n";
        } else {
          os_ << "    int m1_kw" << p << " = m1_rt" << p << ";\n";
        }
        os_ << "    m1_idx m1_off" << p << " = (m1_idx)m1_ci" << p << " * m1_PinI";
        if (D.k > 1) os_ << " + (m1_idx)(m1_kd" << p << " * " << D.d << ") * (m1_idx)m1_Hin * (m1_idx)m1_Win";
        if (H.k > 1) os_ << " + (m1_idx)(m1_kh" << p << " * " << H.d << ") * (m1_idx)m1_Win";
        if (W.k > 1) os_ << " + (m1_idx)(m1_kw" << p << " * " << W.d << ")";
        os_ << ";\n";
      }
    }

    os_ << "    for (int m1_k0 = 0; m1_k0 < m1_KFull; m1_k0 += m1_BK) {\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      if (guard) os_ << "      if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
      os_ << "      *(threadgroup float4*)(m1_sw + (m1_wr0 + " << (p * w_rstep()) << ") * m1_LDW + m1_wc) = "
          << "float4(*(const device " << wpv4 << "*)(m1_wp" << p << " + m1_k0));\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      os_ << "      {\n";
      if (guard) os_ << "        if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK) {\n";
      std::string off_expr;
      if (use_step) {
        off_expr = "m1_off" + std::to_string(p);
      } else {
        os_ << "        const int m1_k = m1_k0 + m1_xr0 + " << (p * x_rstep()) << ";\n";
        os_ << "        const int m1_ci = m1_k / m1_KS;\n";
        os_ << "        const int m1_rt = m1_k - m1_ci * m1_KS;\n";
        os_ << "        const int m1_kd = m1_rt / m1_KHW;\n";
        os_ << "        const int m1_rt2 = m1_rt - m1_kd * m1_KHW;\n";
        os_ << "        const int m1_kh = m1_rt2 / m1_KW;\n";
        os_ << "        const int m1_kw = m1_rt2 - m1_kh * m1_KW;\n";
        os_ << "        const m1_idx m1_off = (m1_idx)m1_ci * m1_PinI + ((m1_idx)(m1_kd * " << D.d
            << ") * (m1_idx)m1_Hin + (m1_idx)(m1_kh * " << H.d << ")) * (m1_idx)m1_Win + (m1_idx)(m1_kw * " << W.d
            << ");\n";
        off_expr = "m1_off";
      }
      if (W.s == 1) {
        os_ << "        *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep())
            << ") * m1_LDX + m1_xc) = float4(*(const device " << xpv4 << "*)(m1_xpix + " << off_expr << "));\n";
      } else {
        os_ << "        device const " << plan_.x_elem << "* m1_xs = m1_xpix + " << off_expr << ";\n";
        os_ << "        *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep())
            << ") * m1_LDX + m1_xc) = float4((float)m1_xs[0], (float)m1_xs[" << W.s << "], (float)m1_xs[" << (2 * W.s)
            << "], (float)m1_xs[" << (3 * W.s) << "]);\n";
      }
      if (use_step) {
        if (d_kw != 0) os_ << "        m1_kw" << p << " += " << d_kw << ";\n";
        if (H.k > 1 && d_kh != 0) os_ << "        m1_kh" << p << " += " << d_kh << ";\n";
        if (D.k > 1 && d_kd != 0) os_ << "        m1_kd" << p << " += " << d_kd << ";\n";
        if (d_ci != 0) os_ << "        m1_ci" << p << " += " << d_ci << ";\n";
        if (W.k > 1) {
          os_ << "        if (m1_kw" << p << " >= m1_KW) { m1_kw" << p << " -= m1_KW; ";
          if (H.k > 1) {
            os_ << "++m1_kh" << p << "; }\n";
          } else if (D.k > 1) {
            os_ << "++m1_kd" << p << "; }\n";
          } else {
            os_ << "++m1_ci" << p << "; }\n";
          }
        }
        if (H.k > 1) {
          os_ << "        if (m1_kh" << p << " >= m1_KH) { m1_kh" << p << " -= m1_KH; ";
          if (D.k > 1) {
            os_ << "++m1_kd" << p << "; }\n";
          } else {
            os_ << "++m1_ci" << p << "; }\n";
          }
        }
        if (D.k > 1) {
          os_ << "        if (m1_kd" << p << " >= m1_KD) { m1_kd" << p << " -= m1_KD; ++m1_ci" << p << "; }\n";
        }
        os_ << "        m1_off" << p << " = (m1_idx)m1_ci" << p << " * m1_PinI";
        if (D.k > 1) os_ << " + (m1_idx)(m1_kd" << p << " * " << D.d << ") * (m1_idx)m1_Hin * (m1_idx)m1_Win";
        if (H.k > 1) os_ << " + (m1_idx)(m1_kh" << p << " * " << H.d << ") * (m1_idx)m1_Win";
        if (W.k > 1) os_ << " + (m1_idx)(m1_kw" << p << " * " << W.d << ")";
        os_ << ";\n";
      }
      if (guard) os_ << "        }\n";
      os_ << "      }\n";
    }
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma("      ");
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";

    if (!plan_.k_mult_bk) {
      os_ << "    if (m1_KFull < m1_K) {\n";
      emit_safe_w_load("m1_KFull", "      ");
      for (uint32_t p = 0; p < x_per(); ++p) {
        const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
        os_ << "      {\n";
        os_ << "        const int m1_rr = m1_xr0 + " << (p * x_rstep()) << ";\n";
        if (guard) os_ << "        if (m1_rr < m1_BK) {\n";
        os_ << "        const int m1_k = m1_KFull + m1_rr;\n";
        os_ << "        float4 m1_v = float4(0.0f);\n";
        os_ << "        if (m1_k < m1_K) {\n";
        std::string off_expr;
        if (use_step) {
          off_expr = "m1_off" + std::to_string(p);
        } else {
          os_ << "          const int m1_ci = m1_k / m1_KS;\n";
          os_ << "          const int m1_rt = m1_k - m1_ci * m1_KS;\n";
          os_ << "          const int m1_kd = m1_rt / m1_KHW;\n";
          os_ << "          const int m1_rt2 = m1_rt - m1_kd * m1_KHW;\n";
          os_ << "          const int m1_kh = m1_rt2 / m1_KW;\n";
          os_ << "          const int m1_kw = m1_rt2 - m1_kh * m1_KW;\n";
          os_ << "          const m1_idx m1_off = (m1_idx)m1_ci * m1_PinI + ((m1_idx)(m1_kd * " << D.d
              << ") * (m1_idx)m1_Hin + (m1_idx)(m1_kh * " << H.d << ")) * (m1_idx)m1_Win + (m1_idx)(m1_kw * " << W.d
              << ");\n";
          off_expr = "m1_off";
        }
        if (W.s == 1) {
          os_ << "          m1_v = float4(*(const device " << xpv4 << "*)(m1_xpix + " << off_expr << "));\n";
        } else {
          os_ << "          device const " << plan_.x_elem << "* m1_xs = m1_xpix + " << off_expr << ";\n";
          os_ << "          m1_v = float4((float)m1_xs[0], (float)m1_xs[" << W.s << "], (float)m1_xs[" << (2 * W.s)
              << "], (float)m1_xs[" << (3 * W.s) << "]);\n";
        }
        os_ << "        }\n";
        os_ << "        *(threadgroup float4*)(m1_sx + m1_rr * m1_LDX + m1_xc) = m1_v;\n";
        if (guard) os_ << "        }\n";
        os_ << "      }\n";
      }
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      emit_mma("      ");
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "    }\n";
    }

    os_ << "  } else {\n";
    os_ << "    const int m1_odx = (int)(m1_px / m1_HWo);\n";
    os_ << "    const uint m1_prx = m1_px - (uint)m1_odx * m1_HWo;\n";
    os_ << "    const int m1_ohx = (int)(m1_prx / (uint)m1_Wout);\n";
    os_ << "    const int m1_owx = (int)(m1_prx - (uint)m1_ohx * (uint)m1_Wout);\n";
    os_ << "    const int m1_idx0 = m1_odx * " << D.s << " - " << D.p << ";\n";
    os_ << "    const int m1_ihx0 = m1_ohx * " << H.s << " - " << H.p << ";\n";
    os_ << "    const int m1_iwx0 = m1_owx * " << W.s << " - " << W.p << ";\n";
    os_ << "    const bool m1_x4 = (m1_mx + 3u < m1_M) && (m1_owx + 3 < m1_Wout) &&\n";
    os_ << "        (m1_idx0 >= 0) && (m1_idx0 + " << ((D.k - 1) * D.d) << " < m1_Din) &&\n";
    os_ << "        (m1_ihx0 >= 0) && (m1_ihx0 + " << ((H.k - 1) * H.d) << " < m1_Hin) &&\n";
    os_ << "        (m1_iwx0 >= 0) && (m1_iwx0 + " << (3 * W.s + (W.k - 1) * W.d) << " < m1_Win);\n";
    os_ << "    const long m1_xoff4 = m1_x4 ? (((long)m1_idx0 * (long)m1_Hin + (long)m1_ihx0) * (long)m1_Win + "
        << "(long)m1_iwx0) : 0L;\n";
    os_ << "    for (int m1_k0 = 0; m1_k0 < m1_K; m1_k0 += m1_BK) {\n";
    emit_safe_w_load("m1_k0", "      ");
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      os_ << "      {\n";
      os_ << "        const int m1_rr = m1_xr0 + " << (p * x_rstep()) << ";\n";
      if (guard) os_ << "        if (m1_rr < m1_BK) {\n";
      os_ << "        const int m1_k = m1_k0 + m1_rr;\n";
      os_ << "        float4 m1_v = float4(0.0f);\n";
      os_ << "        if (m1_k < m1_K) {\n";
      os_ << "          const int m1_ci = m1_k / m1_KS;\n";
      os_ << "          const int m1_rt = m1_k - m1_ci * m1_KS;\n";
      os_ << "          const int m1_kd = m1_rt / m1_KHW;\n";
      os_ << "          const int m1_rt2 = m1_rt - m1_kd * m1_KHW;\n";
      os_ << "          const int m1_kh = m1_rt2 / m1_KW;\n";
      os_ << "          const int m1_kw = m1_rt2 - m1_kh * m1_KW;\n";
      os_ << "          if (m1_x4) {\n";
      os_ << "            const long m1_o = m1_xoff4 + (long)m1_ci * m1_Pin + ((long)(m1_kd * " << D.d
          << ") * (long)m1_Hin + (long)(m1_kh * " << H.d << ")) * (long)m1_Win + (long)(m1_kw * " << W.d << ");\n";
      if (W.s == 1) {
        os_ << "            m1_v = float4(*(const device " << xpv4 << "*)(m1_xb + m1_o));\n";
      } else {
        os_ << "            device const " << plan_.x_elem << "* m1_xs = m1_xb + m1_o;\n";
        os_ << "            m1_v = float4((float)m1_xs[0], (float)m1_xs[" << W.s << "], (float)m1_xs[" << (2 * W.s)
            << "], (float)m1_xs[" << (3 * W.s) << "]);\n";
      }
      os_ << "          } else {\n";
      os_ << "            for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "              const uint m1_me = m1_mx + (uint)m1_e;\n";
      os_ << "              if (m1_me < m1_M) {\n";
      os_ << "                const uint m1_ne = m1_me / m1_P;\n";
      os_ << "                const uint m1_pe = m1_me - m1_ne * m1_P;\n";
      os_ << "                const int m1_ode = (int)(m1_pe / m1_HWo);\n";
      os_ << "                const uint m1_pre = m1_pe - (uint)m1_ode * m1_HWo;\n";
      os_ << "                const int m1_ohe = (int)(m1_pre / (uint)m1_Wout);\n";
      os_ << "                const int m1_owe = (int)(m1_pre - (uint)m1_ohe * (uint)m1_Wout);\n";
      os_ << "                const int m1_id = m1_ode * " << D.s << " - " << D.p << " + m1_kd * " << D.d << ";\n";
      os_ << "                const int m1_ih = m1_ohe * " << H.s << " - " << H.p << " + m1_kh * " << H.d << ";\n";
      os_ << "                const int m1_iw = m1_owe * " << W.s << " - " << W.p << " + m1_kw * " << W.d << ";\n";
      os_ << "                if ((uint)m1_id < (uint)m1_Din && (uint)m1_ih < (uint)m1_Hin && (uint)m1_iw < "
          << "(uint)m1_Win) {\n";
      os_ << "                  m1_v[m1_e] = (float)" << plan_.x_name
          << "[((long)m1_ne * m1_Cin + (long)(m1_g * m1_Cig + m1_ci)) * m1_Pin + ((long)m1_id * (long)m1_Hin + "
          << "(long)m1_ih) * (long)m1_Win + (long)m1_iw];\n";
      os_ << "                }\n";
      os_ << "              }\n";
      os_ << "            }\n";
      os_ << "          }\n";
      os_ << "        }\n";
      os_ << "        *(threadgroup float4*)(m1_sx + m1_rr * m1_LDX + m1_xc) = m1_v;\n";
      if (guard) os_ << "        }\n";
      os_ << "      }\n";
    }
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma("      ");
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";
    os_ << "  }\n";
  }

  void emit_epilogue() {
    const std::string& oe = plan_.out_elem;
    const std::string opv4 = msl_packed_vec4(oe);
    const std::string ov4 = msl_vec4(oe);
    const uint32_t sg_rows = fm() * 8;
    const uint32_t sg_ld = fn() * 8;
    const uint32_t sg_elems = sg_rows * sg_ld;
    const uint32_t max_sgs = std::max<uint32_t>(1, plan_.smem_floats / sg_elems);
    const uint32_t v4_per_sg = sg_elems / 4;
    const uint32_t v4_per_row = sg_ld / 4;

    os_ << "  device " << oe << "* m1_ob = " << plan_.out_name << " + (long)m1_g * (long)m1_Cog * (long)m1_P;\n";
    os_ << "  for (uint m1_sb = 0u; m1_sb < " << plan_.num_sgs << "u; m1_sb += " << max_sgs << "u) {\n";
    os_ << "    if (m1_sg >= m1_sb && m1_sg < m1_sb + " << max_sgs << "u) {\n";
    os_ << "      threadgroup float* m1_sgs = m1_smem + (m1_sg - m1_sb) * " << sg_elems << "u;\n";
    for (uint32_t i = 0; i < fm(); ++i) {
      for (uint32_t j = 0; j < fn(); ++j) {
        os_ << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_sgs + " << (i * 8) << "u * " << sg_ld
            << "u + " << (j * 8) << "u, " << sg_ld << ");\n";
      }
    }
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    const uint m1_act = min(" << max_sgs << "u, " << plan_.num_sgs << "u - m1_sb);\n";
    os_ << "    for (uint m1_vi = m1_tid; m1_vi < m1_act * " << v4_per_sg << "u; m1_vi += " << plan_.num_threads
        << "u) {\n";
    os_ << "      const uint m1_lsg = m1_vi / " << v4_per_sg << "u;\n";
    os_ << "      const uint m1_rem = m1_vi - m1_lsg * " << v4_per_sg << "u;\n";
    os_ << "      const uint m1_lr = m1_rem / " << v4_per_row << "u;\n";
    os_ << "      const uint m1_lc = (m1_rem - m1_lr * " << v4_per_row << "u) * 4u;\n";
    os_ << "      const uint m1_gsg = m1_sb + m1_lsg;\n";
    os_ << "      const int m1_co = m1_co0 + (int)(m1_gsg / " << plan_.sg_n << "u) * " << sg_rows << " + (int)m1_lr;\n";
    os_ << "      const uint m1_mm = m1_m0 + (m1_gsg % " << plan_.sg_n << "u) * " << sg_ld << "u + m1_lc;\n";
    os_ << "      if (m1_co < m1_Cog && m1_mm < m1_M) {\n";
    os_ << "        float4 m1_ov = *(const threadgroup float4*)(m1_smem + m1_lsg * " << sg_elems << "u + m1_lr * "
        << sg_ld << "u + m1_lc);\n";
    if (plan_.attr.has_bias) {
      os_ << "        m1_ov += float4((float)" << plan_.bias_name << "[m1_g * m1_Cog + m1_co]);\n";
    }
    os_ << "        const uint m1_nn = m1_mm / m1_P;\n";
    os_ << "        const uint m1_pp = m1_mm - m1_nn * m1_P;\n";
    os_ << "        if (m1_mm + 3u < m1_M && m1_pp + 3u < m1_P) {\n";
    os_ << "          *(device " << opv4 << "*)(m1_ob + ((long)m1_nn * m1_Cout + (long)m1_co) * (long)m1_P + "
        << "(long)m1_pp) = " << opv4 << "(" << ov4 << "(m1_ov));\n";
    os_ << "        } else {\n";
    os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
    os_ << "            const uint m1_me = m1_mm + (uint)m1_e;\n";
    os_ << "            if (m1_me < m1_M) {\n";
    os_ << "              const uint m1_ne = m1_me / m1_P;\n";
    os_ << "              const uint m1_pe = m1_me - m1_ne * m1_P;\n";
    os_ << "              m1_ob[((long)m1_ne * m1_Cout + (long)m1_co) * (long)m1_P + (long)m1_pe] = " << oe
        << "(m1_ov[m1_e]);\n";
    os_ << "            }\n";
    os_ << "          }\n";
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
  }

  const Kernel& kernel_;
  const ConvPlan& plan_;
  std::ostringstream os_;
};

}  // namespace

CompiledMSL try_emit_conv_std_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  if (matched_out != nullptr) *matched_out = false;
  std::string err;
  auto plan = build_plan(kernel, target, err);
  if (!plan) {
    if (err.empty()) return CompiledMSL{false, "", {}, ""};
    if (matched_out != nullptr) *matched_out = true;
    return CompiledMSL{false, "", {}, err};
  }
  if (matched_out != nullptr) *matched_out = true;

  CompiledMSL res;
  res.ok = true;
  res.msl_source = ConvStdWriter(kernel, *plan).emit();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan->num_threads;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  uint32_t slot = 0;
  for (uint32_t i = 0; i < kernel.params.size(); ++i) {
    const ir::KernelParam& kp = kernel.params[i];
    if (kp.is_constexpr) continue;
    ParamSlot ps;
    ps.param_index = i;
    ps.buffer_index = slot++;
    ps.name = kp.name;
    ps.is_pointer = kp.type.is_ptr();
    ps.writable = ps.is_pointer && (kp.writable || kp.name == plan->out_name);
    ps.dtype = ps.is_pointer ? std::get<ir::PtrType>(kp.type.value).pointee : std::get<ScalarDType>(kp.type.value);
    res.launch_spec.param_slots.push_back(std::move(ps));
  }
  return res;
}

}  // namespace m1triton::codegen::metal

#include "codegen/metal/flash_attn_emitter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <variant>

namespace m1triton::codegen::metal {

namespace {

using ir::FlashAttnAttr;
using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::ScalarDType;
using ir::ValueId;

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

std::string format_float_literal(float v) {
  std::ostringstream ss;
  ss << std::setprecision(9) << v;
  std::string s = ss.str();
  if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
    s += ".0";
  }
  s += "f";
  return s;
}

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
    if (const auto* b = std::get_if<bool>(&ca->val)) return *b ? "1L" : "0L";
    return std::nullopt;
  }
  if (def.kind == OpKind::Cast && def.operands.size() == 1) return scalar_expr(kernel, def.operands[0]);
  return std::nullopt;
}

enum class FlashAttnMode {
  SinglePassSgmm,
  MultiTileSgmm,
  GeneralRowWise,
};

struct FlashAttnPlan {
  const Op* op = nullptr;
  FlashAttnAttr attr;
  std::string q_name;
  std::string k_name;
  std::string v_name;
  std::string out_name;
  ScalarDType elem_dt = ScalarDType::F32;
  std::string elem_type = "float";
  std::string bh_expr;
  std::string lq_expr;
  std::string lkv_expr;
  std::string d_expr;
  uint32_t bm = 16;
  uint32_t bn = 256;
  uint32_t num_sg = 16;
  uint32_t tg_threads = 512;
  FlashAttnMode mode = FlashAttnMode::GeneralRowWise;
};

std::optional<FlashAttnPlan> build_plan(const Kernel& kernel, const ir::TargetSpec& target, std::string& err) {
  FlashAttnPlan plan;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::FlashAttentionTile:
        if (plan.op != nullptr) {
          err = "flash_attn emitter supports exactly one flash_attention_tile op per kernel";
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

  const auto* fa = std::get_if<FlashAttnAttr>(&plan.op->attr);
  if (fa == nullptr) {
    err = "flash_attention_tile op missing FlashAttnAttr";
    return std::nullopt;
  }
  if (plan.op->operands.size() != 8 || !plan.op->results.empty()) {
    err = "flash_attention_tile expects 8 operands and 0 results";
    return std::nullopt;
  }
  if (!std::isfinite(fa->scale)) {
    err = "flash_attention_tile requires finite scale";
    return std::nullopt;
  }
  plan.attr = *fa;

  const auto& ops = plan.op->operands;
  std::string* ptr_names[4] = {&plan.q_name, &plan.k_name, &plan.v_name, &plan.out_name};
  for (size_t i = 0; i < 4; ++i) {
    const auto pidx = param_index_of(kernel, ops[i]);
    if (!pidx || !kernel.params[*pidx].type.is_ptr()) {
      err = "flash_attention_tile pointer operands 0..3 must be kernel pointer parameters";
      return std::nullopt;
    }
    const ScalarDType pt = std::get<ir::PtrType>(kernel.params[*pidx].type.value).pointee;
    if (pt != ScalarDType::F32 && pt != ScalarDType::F16) {
      err = "flash_attention_tile supports f32 or f16 pointers";
      return std::nullopt;
    }
    if (i == 0) {
      plan.elem_dt = pt;
      plan.elem_type = msl_scalar(pt);
    } else if (pt != plan.elem_dt) {
      err = "flash_attention_tile requires matching Q/K/V/O pointer dtypes";
      return std::nullopt;
    }
    *ptr_names[i] = kernel.params[*pidx].name;
  }

  std::string* shape_exprs[4] = {&plan.bh_expr, &plan.lq_expr, &plan.lkv_expr, &plan.d_expr};
  for (size_t i = 0; i < 4; ++i) {
    auto e = scalar_expr(kernel, ops[4 + i]);
    if (!e) {
      err = "flash_attention_tile shape operands must be scalar parameters or integer constants";
      return std::nullopt;
    }
    *shape_exprs[i] = *e;
  }

  const uint32_t simd = std::max<uint32_t>(1u, target.simd_width);
  plan.bm = fa->bm > 0u ? fa->bm : 16u;
  plan.bn = fa->bn > 0u ? fa->bn : 256u;
  plan.num_sg = fa->num_simdgroups > 0u ? fa->num_simdgroups : 16u;
  plan.tg_threads = plan.num_sg * simd;

  if (plan.tg_threads > target.max_threads_per_threadgroup) {
    err = "flash_attention_tile threadgroup exceeds max_threads_per_threadgroup";
    return std::nullopt;
  }

  const uint32_t max_smem = target.max_threadgroup_memory_bytes;
  if (plan.elem_dt == ScalarDType::F32 && plan.bm == 16u && plan.bn == 512u && plan.num_sg == 16u &&
      plan.bm * plan.bn * 4u <= max_smem) {
    plan.mode = FlashAttnMode::SinglePassSgmm;
  } else if (plan.elem_dt == ScalarDType::F32 && plan.bm == 16u && plan.num_sg == 16u &&
             (plan.bn == 256u || plan.bn == 128u || plan.bn == 64u) &&
             (plan.bm * (plan.bn + 4u) + 3u * plan.bm) * 4u <= max_smem) {
    plan.mode = FlashAttnMode::MultiTileSgmm;
  } else {
    if (plan.bm * std::min<uint32_t>(plan.bn, 256u) * 4u > max_smem) {
      err = "flash_attention_tile exceeds target threadgroup memory budget";
      return std::nullopt;
    }
    plan.mode = FlashAttnMode::GeneralRowWise;
  }

  return plan;
}

class FlashAttnWriter {
 public:
  FlashAttnWriter(const Kernel& kernel, const FlashAttnPlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    emit_header();
    emit_signature();
    emit_prologue();
    switch (plan_.mode) {
      case FlashAttnMode::SinglePassSgmm:
        emit_single_pass_sgmm();
        break;
      case FlashAttnMode::MultiTileSgmm:
        emit_multitile_sgmm();
        break;
      case FlashAttnMode::GeneralRowWise:
        emit_general_rowwise(std::min<uint32_t>(plan_.bn, 256u), "  ");
        break;
    }
    os_ << "}\n";
    return os_.str();
  }

 private:
  void emit_header() {
    os_ << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    if (plan_.mode == FlashAttnMode::MultiTileSgmm) {
      os_ << "static inline float2 m1_mma2(float2 a, float2 b, float2 c) {\n"
          << "  simdgroup_float8x8 am, bm, cm;\n"
          << "  reinterpret_cast<thread float2&>(am.thread_elements()) = a;\n"
          << "  reinterpret_cast<thread float2&>(bm.thread_elements()) = b;\n"
          << "  reinterpret_cast<thread float2&>(cm.thread_elements()) = c;\n"
          << "  simdgroup_multiply_accumulate(cm, am, bm, cm);\n"
          << "  return reinterpret_cast<thread float2&>(cm.thread_elements());\n"
          << "}\n\n";
    }
  }

  void emit_signature() {
    os_ << "[[max_total_threads_per_threadgroup(" << plan_.tg_threads << ")]]\n";
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
    os_ << "    uint m1_sg [[simdgroup_index_in_threadgroup]],\n";
    os_ << "    uint m1_lane [[thread_index_in_simdgroup]]) {\n";
  }

  void emit_prologue() {
    const uint32_t shm_floats = compute_shm_floats();
    os_ << "  threadgroup float m1_shm[" << shm_floats << "];\n";
    if (plan_.mode == FlashAttnMode::MultiTileSgmm) {
      os_ << "  threadgroup float m1_ms[" << plan_.bm << "];\n";
      os_ << "  threadgroup float m1_ls[" << plan_.bm << "];\n";
      os_ << "  threadgroup float m1_as[" << plan_.bm << "];\n";
    }
    os_ << "  const long m1_BH = " << plan_.bh_expr << ";\n";
    os_ << "  const long m1_Lq = " << plan_.lq_expr << ";\n";
    os_ << "  const long m1_Lkv = " << plan_.lkv_expr << ";\n";
    os_ << "  const long m1_D = " << plan_.d_expr << ";\n";
    os_ << "  const long m1_bh = (long)m1_tgid.y;\n";
    os_ << "  if (m1_bh >= m1_BH) return;\n";
    os_ << "  const long m1_q0 = (long)m1_tgid.x * " << plan_.bm << "L;\n";
    os_ << "  if (m1_q0 >= m1_Lq) return;\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Qb = " << plan_.q_name
        << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Kb = " << plan_.k_name
        << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Vb = " << plan_.v_name
        << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device " << plan_.elem_type << "* m1_Ob = " << plan_.out_name
        << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  const float m1_scale_log2 = " << format_float_literal(plan_.attr.scale)
        << " * 1.4426950408889634f;\n";
    os_ << "  (void)m1_tid;\n";
  }

  uint32_t compute_shm_floats() const {
    if (plan_.mode == FlashAttnMode::SinglePassSgmm) {
      return plan_.bm * plan_.bn;
    }
    if (plan_.mode == FlashAttnMode::MultiTileSgmm) {
      return plan_.bm * (plan_.bn + 4u);
    }
    return plan_.bm * std::min<uint32_t>(plan_.bn, 256u);
  }

  std::string kmax_expr(const std::string& row_expr) const {
    if (!plan_.attr.causal) return "m1_Lkv";
    return "min(m1_Lkv, max(0L, m1_q0 + (long)(" + row_expr + ") + 1L + " +
           std::to_string(plan_.attr.causal_offset) + "L))";
  }

  void emit_single_pass_sgmm() {
    constexpr uint32_t kLdp = 512u;
    os_ << "  if (m1_q0 + 16L <= m1_Lq && m1_Lkv > 0L && m1_Lkv <= 512L && (m1_Lkv & 31L) == 0L && "
           "m1_D > 0L && m1_D <= 1024L && (m1_D & 63L) == 0L) {\n";

    os_ << "    {\n";
    os_ << "      const int m1_sc0 = (int)m1_sg * 32;\n";
    os_ << "      if ((long)m1_sc0 < m1_Lkv) {\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
        os_ << "        simdgroup_float8x8 s_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << "        device const float* qp_0 = m1_Qb + m1_q0 * m1_D;\n";
    os_ << "        device const float* qp_1 = qp_0 + 8L * m1_D;\n";
    for (int j = 0; j < 4; ++j) {
      os_ << "        device const float* kp_" << j << " = m1_Kb + (long)(m1_sc0 + " << (j * 8) << ") * m1_D;\n";
    }
    os_ << "        for (long m1_d0 = 0L; m1_d0 < m1_D; m1_d0 += 16L) {\n";
    for (int ks = 0; ks < 16; ks += 8) {
      os_ << "          {\n";
      os_ << "            simdgroup_float8x8 fa_0; simdgroup_load(fa_0, qp_0 + " << ks << ", m1_D);\n";
      os_ << "            simdgroup_float8x8 fa_1; simdgroup_load(fa_1, qp_1 + " << ks << ", m1_D);\n";
      os_ << "            simdgroup_float8x8 fb_cur, fb_nxt;\n";
      os_ << "            simdgroup_load(fb_cur, kp_0 + " << ks << ", m1_D, ulong2(0, 0), true);\n";
      for (int j = 0; j < 4; ++j) {
        if (j + 1 < 4) {
          os_ << "            simdgroup_load(fb_nxt, kp_" << (j + 1) << " + " << ks
              << ", m1_D, ulong2(0, 0), true);\n";
        }
        os_ << "            simdgroup_multiply_accumulate(s_0_" << j << ", fa_0, fb_cur, s_0_" << j << ");\n";
        os_ << "            simdgroup_multiply_accumulate(s_1_" << j << ", fa_1, fb_cur, s_1_" << j << ");\n";
        if (j + 1 < 4) {
          os_ << "            fb_cur = fb_nxt;\n";
        }
      }
      os_ << "          }\n";
    }
    os_ << "          qp_0 += 16L; qp_1 += 16L;\n";
    os_ << "          kp_0 += 16L; kp_1 += 16L; kp_2 += 16L; kp_3 += 16L;\n";
    os_ << "        }\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
        os_ << "        simdgroup_store(s_" << i << "_" << j << ", m1_shm + " << (i * 8 * kLdp)
            << " + (m1_sc0 + " << (j * 8) << "), " << kLdp << ");\n";
      }
    }
    os_ << "      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";

    os_ << "    {\n";
    os_ << "      const int m1_row = (int)m1_sg;\n";
    os_ << "      const long m1_kmax = " << kmax_expr("m1_row") << ";\n";
    os_ << "      float m1_vals[16];\n";
    os_ << "      float m1_mx = -1e30f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << "      { const int c = (int)m1_lane + " << (j * 32) << "; "
          << "const float v = ((long)c < m1_kmax) ? (m1_shm[m1_row * " << kLdp
          << " + c] * m1_scale_log2) : -1e30f; m1_vals[" << j << "] = v; m1_mx = max(m1_mx, v); }\n";
    }
    os_ << "      m1_mx = simd_max(m1_mx);\n";
    os_ << "      const bool m1_valid = (m1_kmax > 0L) && (m1_mx > -1e29f);\n";
    os_ << "      float m1_rs = 0.0f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << "      { const int c = (int)m1_lane + " << (j * 32) << "; "
          << "const float p = (m1_valid && (long)c < m1_kmax) ? exp2(m1_vals[" << j << "] - m1_mx) : 0.0f; "
          << "m1_vals[" << j << "] = p; m1_rs += p; }\n";
    }
    os_ << "      m1_rs = simd_sum(m1_rs);\n";
    os_ << "      const float m1_inv = (m1_valid && m1_rs > 0.0f) ? (1.0f / m1_rs) : 0.0f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << "      m1_shm[m1_row * " << kLdp << " + (int)m1_lane + " << (j * 32) << "] = m1_vals[" << j
          << "] * m1_inv;\n";
    }
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";

    os_ << "    {\n";
    os_ << "      const int m1_sgd0 = (int)m1_sg * 64;\n";
    os_ << "      if ((long)m1_sgd0 < m1_D) {\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        os_ << "        simdgroup_float8x8 o_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << "        device const float* vp = m1_Vb + (long)m1_sgd0;\n";
    os_ << "        for (int m1_k0 = 0; m1_k0 < (int)m1_Lkv; m1_k0 += 16) {\n";
    for (int ku = 0; ku < 2; ++ku) {
      os_ << "          {\n";
      os_ << "            const int kk = m1_k0 + " << (ku * 8) << ";\n";
      os_ << "            simdgroup_float8x8 pf_0; simdgroup_load(pf_0, m1_shm + kk, " << kLdp << ");\n";
      os_ << "            simdgroup_float8x8 pf_1; simdgroup_load(pf_1, m1_shm + " << (8 * kLdp) << " + kk, " << kLdp
          << ");\n";
      os_ << "            simdgroup_float8x8 vf_cur, vf_nxt;\n";
      os_ << "            simdgroup_load(vf_cur, vp + " << (ku * 8) << "L * m1_D, m1_D);\n";
      for (int j = 0; j < 8; ++j) {
        if (j + 1 < 8) {
          os_ << "            simdgroup_load(vf_nxt, vp + " << (ku * 8) << "L * m1_D + " << ((j + 1) * 8)
              << ", m1_D);\n";
        }
        os_ << "            simdgroup_multiply_accumulate(o_0_" << j << ", pf_0, vf_cur, o_0_" << j << ");\n";
        os_ << "            simdgroup_multiply_accumulate(o_1_" << j << ", pf_1, vf_cur, o_1_" << j << ");\n";
        if (j + 1 < 8) {
          os_ << "            vf_cur = vf_nxt;\n";
        }
      }
      os_ << "          }\n";
    }
    os_ << "          vp += 16L * m1_D;\n";
    os_ << "        }\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        os_ << "        simdgroup_store(o_" << i << "_" << j << ", m1_Ob + (m1_q0 + " << (i * 8)
            << "L) * m1_D + (long)(m1_sgd0 + " << (j * 8) << "), m1_D);\n";
      }
    }
    os_ << "      }\n";
    os_ << "    }\n";
    os_ << "    return;\n";
    os_ << "  }\n";

    emit_general_rowwise(256u, "  ");
  }

  void emit_multitile_sgmm() {
    const uint32_t bn = plan_.bn;
    const uint32_t ldp = bn + 4u;
    const uint32_t s_tn = (bn == 256u) ? 4u : 2u;
    const uint32_t s_sgn = bn / (s_tn * 8u);
    const uint32_t d_grp_count = 16u / s_sgn;
    const uint32_t bd_sg = (d_grp_count == 2u) ? 32u : 16u;
    const uint32_t bd_step = bd_sg * d_grp_count;
    const uint32_t n_soft = bn / 32u;

    os_ << "  if (m1_q0 + 16L <= m1_Lq && m1_Lkv > 0L && (m1_Lkv & 31L) == 0L && "
           "m1_D > 0L && m1_D <= 1024L && (m1_D & 63L) == 0L) {\n";
    os_ << "    if (m1_tid < 16u) { m1_ms[m1_tid] = -1e30f; m1_ls[m1_tid] = 0.0f; m1_as[m1_tid] = 1.0f; }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        os_ << "    float2 o_" << i << "_" << j << " = float2(0.0f);\n";
      }
    }
    os_ << "    const int m1_qid = (int)m1_lane / 4;\n";
    os_ << "    const int m1_fr = (m1_qid & 4) + ((int)m1_lane / 2) % 4;\n";
    os_ << "    const int m1_fc = (m1_qid & 2) * 2 + ((int)m1_lane % 2) * 2;\n";
    os_ << "    const int m1_dgrp = (int)m1_sg / " << s_sgn << ";\n";
    os_ << "    const int m1_sc0 = ((int)m1_sg % " << s_sgn << ") * " << (s_tn * 8u) << ";\n";
    os_ << "    const int m1_sgd0 = (int)m1_sg * 64;\n";
    if (plan_.attr.causal) {
      os_ << "    const long m1_kv_lim = min(m1_Lkv, max(0L, m1_q0 + 16L + " << plan_.attr.causal_offset << "L));\n";
    } else {
      os_ << "    const long m1_kv_lim = m1_Lkv;\n";
    }
    os_ << "    for (long m1_kv0 = 0L; m1_kv0 < m1_kv_lim; m1_kv0 += " << bn << "L) {\n";
    os_ << "      {\n";
    for (int i = 0; i < 2; ++i) {
      for (uint32_t j = 0; j < s_tn; ++j) {
        os_ << "        simdgroup_float8x8 s_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << "        if (m1_kv0 + (long)m1_sc0 < m1_Lkv) {\n";
    os_ << "          for (long m1_d0 = 0L; m1_d0 < m1_D; m1_d0 += " << bd_step << "L) {\n";
    for (uint32_t ks = 0; ks < bd_sg; ks += 8u) {
      os_ << "            {\n";
      os_ << "              const long m1_dcol = m1_d0 + (long)(m1_dgrp * " << bd_sg << " + " << ks << ");\n";
      os_ << "              simdgroup_float8x8 fa_0; simdgroup_load(fa_0, m1_Qb + m1_q0 * m1_D + m1_dcol, m1_D);\n";
      os_ << "              simdgroup_float8x8 fa_1; simdgroup_load(fa_1, m1_Qb + (m1_q0 + 8L) * m1_D + m1_dcol, "
             "m1_D);\n";
      os_ << "              simdgroup_float8x8 fb_cur, fb_nxt;\n";
      os_ << "              simdgroup_load(fb_cur, m1_Kb + (m1_kv0 + (long)m1_sc0) * m1_D + m1_dcol, m1_D, "
             "ulong2(0, 0), true);\n";
      for (uint32_t j = 0; j < s_tn; ++j) {
        if (j + 1u < s_tn) {
          os_ << "              simdgroup_load(fb_nxt, m1_Kb + (m1_kv0 + (long)(m1_sc0 + " << ((j + 1u) * 8u)
              << ")) * m1_D + m1_dcol, m1_D, ulong2(0, 0), true);\n";
        }
        os_ << "              simdgroup_multiply_accumulate(s_0_" << j << ", fa_0, fb_cur, s_0_" << j << ");\n";
        os_ << "              simdgroup_multiply_accumulate(s_1_" << j << ", fa_1, fb_cur, s_1_" << j << ");\n";
        if (j + 1u < s_tn) {
          os_ << "              fb_cur = fb_nxt;\n";
        }
      }
      os_ << "            }\n";
    }
    os_ << "          }\n";
    os_ << "        }\n";
    for (uint32_t g = 0; g < d_grp_count; ++g) {
      os_ << "        if (m1_dgrp == " << g << ") {\n";
      for (int i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < s_tn; ++j) {
          if (g == 0u) {
            os_ << "          simdgroup_store(s_" << i << "_" << j << ", m1_shm + " << (i * 8u * ldp)
                << " + (m1_sc0 + " << (j * 8u) << "), " << ldp << ");\n";
          } else {
            os_ << "          *(threadgroup float2*)(m1_shm + (" << (i * 8) << " + m1_fr) * " << ldp
                << " + m1_sc0 + " << (j * 8u) << " + m1_fc) += reinterpret_cast<thread float2&>(s_" << i << "_" << j
                << ".thread_elements());\n";
          }
        }
      }
      os_ << "        }\n";
      os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    os_ << "      }\n";

    os_ << "      {\n";
    os_ << "        const int m1_row = (int)m1_sg;\n";
    os_ << "        const long m1_kmax = " << kmax_expr("m1_row") << ";\n";
    os_ << "        float m1_vals[" << n_soft << "];\n";
    os_ << "        float m1_mx = -1e30f;\n";
    for (uint32_t j = 0; j < n_soft; ++j) {
      os_ << "        { const int c = (int)m1_lane + " << (j * 32u) << "; "
          << "const float v = (m1_kv0 + (long)c < m1_kmax) ? (m1_shm[m1_row * " << ldp
          << " + c] * m1_scale_log2) : -1e30f; m1_vals[" << j << "] = v; m1_mx = max(m1_mx, v); }\n";
    }
    os_ << "        m1_mx = simd_max(m1_mx);\n";
    os_ << "        const float m1_mold = m1_ms[m1_row];\n";
    os_ << "        const float m1_mnew = max(m1_mold, m1_mx);\n";
    os_ << "        const bool m1_valid = (m1_mnew > -1e29f);\n";
    os_ << "        const float m1_alpha = (m1_valid && m1_mold > -1e29f) ? exp2(m1_mold - m1_mnew) : 0.0f;\n";
    os_ << "        float m1_rs = 0.0f;\n";
    for (uint32_t j = 0; j < n_soft; ++j) {
      os_ << "        { const int c = (int)m1_lane + " << (j * 32u) << "; "
          << "const float p = (m1_valid && (m1_kv0 + (long)c < m1_kmax)) ? exp2(m1_vals[" << j
          << "] - m1_mnew) : 0.0f; m1_rs += p; m1_shm[m1_row * " << ldp << " + c] = p; }\n";
    }
    os_ << "        m1_rs = simd_sum(m1_rs);\n";
    os_ << "        if (m1_lane == 0u) { m1_ms[m1_row] = m1_mnew; m1_ls[m1_row] = m1_ls[m1_row] * m1_alpha + m1_rs; "
           "m1_as[m1_row] = m1_alpha; }\n";
    os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "      }\n";

    os_ << "      if ((long)m1_sgd0 < m1_D) {\n";
    for (int i = 0; i < 2; ++i) {
      os_ << "        { const float a = m1_as[" << (i * 8) << " + m1_fr];\n";
      for (int j = 0; j < 8; ++j) {
        os_ << "          o_" << i << "_" << j << " *= a;\n";
      }
      os_ << "        }\n";
    }
    os_ << "        const int m1_ktile = (int)min(" << bn << "L, m1_Lkv - m1_kv0);\n";
    os_ << "        for (int m1_k0 = 0; m1_k0 < m1_ktile; m1_k0 += 16) {\n";
    for (int ku = 0; ku < 2; ++ku) {
      os_ << "          {\n";
      os_ << "            const int kk = m1_k0 + " << (ku * 8) << ";\n";
      os_ << "            float2 pf_0 = *(threadgroup const float2*)(m1_shm + m1_fr * " << ldp << " + kk + m1_fc);\n";
      os_ << "            float2 pf_1 = *(threadgroup const float2*)(m1_shm + (8 + m1_fr) * " << ldp
          << " + kk + m1_fc);\n";
      os_ << "            device const float* vr = m1_Vb + (m1_kv0 + (long)(kk + m1_fr)) * m1_D + (long)(m1_sgd0 + "
             "m1_fc);\n";
      os_ << "            float2 vf_nxt; float2 vf_cur = float2(*(device const packed_float2*)(vr));\n";
      for (int j = 0; j < 8; ++j) {
        if (j + 1 < 8) {
          os_ << "            vf_nxt = float2(*(device const packed_float2*)(vr + " << ((j + 1) * 8) << "));\n";
        }
        os_ << "            o_0_" << j << " = m1_mma2(pf_0, vf_cur, o_0_" << j << ");\n";
        os_ << "            o_1_" << j << " = m1_mma2(pf_1, vf_cur, o_1_" << j << ");\n";
        if (j + 1 < 8) {
          os_ << "            vf_cur = vf_nxt;\n";
        }
      }
      os_ << "          }\n";
    }
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";

    os_ << "    if ((long)m1_sgd0 < m1_D) {\n";
    for (int i = 0; i < 2; ++i) {
      os_ << "      {\n";
      os_ << "        const long qr = m1_q0 + (long)(" << (i * 8) << " + m1_fr);\n";
      os_ << "        const float lval = m1_ls[" << (i * 8) << " + m1_fr];\n";
      os_ << "        const float inv = (lval > 0.0f) ? (1.0f / lval) : 0.0f;\n";
      for (int j = 0; j < 8; ++j) {
        os_ << "        *(device packed_float2*)(m1_Ob + qr * m1_D + (long)(m1_sgd0 + " << (j * 8)
            << " + m1_fc)) = packed_float2(o_" << i << "_" << j << " * inv);\n";
      }
      os_ << "      }\n";
    }
    os_ << "    }\n";
    os_ << "    return;\n";
    os_ << "  }\n";

    emit_general_rowwise(std::min<uint32_t>(bn, 256u), "  ");
  }

  void emit_general_rowwise(uint32_t tile_n, const std::string& ind) {
    const uint32_t per_lane = std::max<uint32_t>(1u, tile_n / 32u);
    os_ << ind << "for (int m1_r = (int)m1_sg; m1_r < " << plan_.bm << "; m1_r += " << plan_.num_sg << ") {\n";
    os_ << ind << "  const long m1_qr = m1_q0 + (long)m1_r;\n";
    os_ << ind << "  if (m1_qr >= m1_Lq) continue;\n";
    os_ << ind << "  const long m1_kmax = " << kmax_expr("m1_r") << ";\n";
    os_ << ind << "  if (m1_kmax <= 0L) {\n";
    os_ << ind << "    for (long m1_d = (long)m1_lane; m1_d < m1_D; m1_d += 32L) {\n";
    os_ << ind << "      m1_Ob[m1_qr * m1_D + m1_d] = " << plan_.elem_type << "(0.0f);\n";
    os_ << ind << "    }\n";
    os_ << ind << "    continue;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup float* m1_rs_buf = m1_shm + m1_r * " << tile_n << ";\n";
    os_ << ind << "  float m1_mi = -1e30f;\n";
    os_ << ind << "  float m1_li = 0.0f;\n";
    os_ << ind << "  for (long m1_kv0 = 0L; m1_kv0 < m1_kmax; m1_kv0 += " << tile_n << "L) {\n";
    os_ << ind << "    const int m1_tlen = (int)min(" << tile_n << "L, m1_kmax - m1_kv0);\n";
    os_ << ind << "    for (int m1_c = 0; m1_c < m1_tlen; ++m1_c) {\n";
    os_ << ind << "      float m1_dot = 0.0f;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_qrow = m1_Qb + m1_qr * m1_D;\n";
    os_ << ind << "      device const " << plan_.elem_type
        << "* m1_krow = m1_Kb + (m1_kv0 + (long)m1_c) * m1_D;\n";
    os_ << ind << "      for (long m1_d = (long)m1_lane; m1_d < m1_D; m1_d += 32L) {\n";
    os_ << ind << "        m1_dot = fma((float)m1_qrow[m1_d], (float)m1_krow[m1_d], m1_dot);\n";
    os_ << ind << "      }\n";
    os_ << ind << "      m1_dot = simd_sum(m1_dot);\n";
    os_ << ind << "      if (m1_lane == 0u) m1_rs_buf[m1_c] = m1_dot * m1_scale_log2;\n";
    os_ << ind << "    }\n";
    os_ << ind << "    simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "    float m1_mx = -1e30f;\n";
    os_ << ind << "    float m1_sv[" << per_lane << "];\n";
    for (uint32_t j = 0; j < per_lane; ++j) {
      os_ << ind << "    { const int c = (int)m1_lane + " << (j * 32u)
          << "; const float v = (c < m1_tlen) ? m1_rs_buf[c] : -1e30f; m1_sv[" << j
          << "] = v; m1_mx = max(m1_mx, v); }\n";
    }
    os_ << ind << "    m1_mx = simd_max(m1_mx);\n";
    os_ << ind << "    const float m1_mnew = max(m1_mi, m1_mx);\n";
    os_ << ind << "    const float m1_alpha = (m1_mi > -1e29f && m1_mnew > -1e29f) ? exp2(m1_mi - m1_mnew) : 0.0f;\n";
    os_ << ind << "    float m1_psum = 0.0f;\n";
    for (uint32_t j = 0; j < per_lane; ++j) {
      os_ << ind << "    { const int c = (int)m1_lane + " << (j * 32u)
          << "; const float p = (c < m1_tlen && m1_mnew > -1e29f) ? exp2(m1_sv[" << j
          << "] - m1_mnew) : 0.0f; m1_psum += p; if (c < m1_tlen) m1_rs_buf[c] = p; }\n";
    }
    os_ << ind << "    m1_psum = simd_sum(m1_psum);\n";
    os_ << ind << "    m1_li = m1_li * m1_alpha + m1_psum;\n";
    os_ << ind << "    m1_mi = m1_mnew;\n";
    os_ << ind << "    simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "    const bool m1_last = (m1_kv0 + " << tile_n << "L >= m1_kmax);\n";
    os_ << ind << "    const float m1_inv = (m1_last && m1_li > 0.0f) ? (1.0f / m1_li) : 1.0f;\n";
    os_ << ind << "    for (long m1_d = (long)m1_lane; m1_d < m1_D; m1_d += 32L) {\n";
    os_ << ind << "      float m1_acc = (m1_kv0 == 0L) ? 0.0f : ((float)m1_Ob[m1_qr * m1_D + m1_d] * m1_alpha);\n";
    os_ << ind << "      for (int m1_c = 0; m1_c < m1_tlen; ++m1_c) {\n";
    os_ << ind << "        m1_acc = fma(m1_rs_buf[m1_c], (float)m1_Vb[(m1_kv0 + (long)m1_c) * m1_D + m1_d], m1_acc);\n";
    os_ << ind << "      }\n";
    os_ << ind << "      if (m1_last) m1_acc *= m1_inv;\n";
    os_ << ind << "      m1_Ob[m1_qr * m1_D + m1_d] = " << plan_.elem_type << "(m1_acc);\n";
    os_ << ind << "    }\n";
    os_ << ind << "    simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  const Kernel& kernel_;
  const FlashAttnPlan& plan_;
  std::ostringstream os_;
};

}  // namespace

CompiledMSL try_emit_flash_attn_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
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
  res.msl_source = FlashAttnWriter(kernel, *plan).emit();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan->tg_threads;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 2;
  res.launch_spec.fast_math = true;

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

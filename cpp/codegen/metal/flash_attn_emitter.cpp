#include "codegen/metal/flash_attn_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

enum class FlashAttnMode {
  TwoWaveSgmm,
  SinglePassSgmm,
  MultiTileSgmm,
  GeneralRowWise,
};

std::optional<int64_t> find_constexpr_param(const Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name) {
      return kp.constexpr_val;
    }
  }
  return std::nullopt;
}

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
  bool fast_only = false;
  uint32_t d_static = 0;
  uint32_t lkv_static = 0;
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
      plan.elem_type = msl_scalar_type(pt);
    } else if (pt != plan.elem_dt) {
      err = "flash_attention_tile requires matching Q/K/V/O pointer dtypes";
      return std::nullopt;
    }
    *ptr_names[i] = kernel.params[*pidx].name;
  }

  std::string* shape_exprs[4] = {&plan.bh_expr, &plan.lq_expr, &plan.lkv_expr, &plan.d_expr};
  for (size_t i = 0; i < 4; ++i) {
    auto e = scalar_long_expr(kernel, ops[4 + i]);
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
  if (const auto v = find_constexpr_param(kernel, "FAST_ONLY")) {
    plan.fast_only = (*v != 0);
  }
  if (plan.fast_only) {
    if (const auto v = find_constexpr_param(kernel, "D_STATIC")) {
      if (*v > 0 && *v <= 1024 && ((*v) & 63) == 0) {
        plan.d_static = static_cast<uint32_t>(*v);
      }
    }
    if (const auto v = find_constexpr_param(kernel, "LKV_STATIC")) {
      if (*v > 0 && *v <= 512 && ((*v) & 31) == 0) {
        plan.lkv_static = static_cast<uint32_t>(*v);
      }
    }
  }

  if (plan.elem_dt == ScalarDType::F32 && plan.bm == 32u && plan.bn == 512u && plan.num_sg == 16u &&
      16u * plan.bn * 4u <= max_smem) {
    plan.mode = FlashAttnMode::TwoWaveSgmm;
  } else if (plan.elem_dt == ScalarDType::F32 && plan.bm == 16u && plan.bn == 512u && plan.num_sg == 16u &&
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
      case FlashAttnMode::TwoWaveSgmm:
        emit_two_wave_sgmm();
        break;
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
    emit_kernel_param_decls(os_, kernel_, plan_.out_name);
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    os_ << "    uint m1_tid [[thread_index_in_threadgroup]],\n";
    os_ << "    uint m1_sg [[simdgroup_index_in_threadgroup]],\n";
    os_ << "    uint m1_lane [[thread_index_in_simdgroup]]) {\n";
  }

  std::string d_stride_long() const {
    return (plan_.d_static > 0u) ? (std::to_string(plan_.d_static) + "L") : "m1_D";
  }

  std::string d_stride_arg() const {
    return (plan_.d_static > 0u) ? std::to_string(plan_.d_static) : "m1_D";
  }

  std::string lkv_int_expr() const {
    return (plan_.lkv_static > 0u) ? std::to_string(plan_.lkv_static) : "(int)m1_Lkv";
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
    const std::string ds = d_stride_long();
    os_ << "  device const " << plan_.elem_type << "* m1_Qb = " << plan_.q_name
        << " + m1_bh * (m1_Lq * " << ds << ");\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Kb = " << plan_.k_name
        << " + m1_bh * (m1_Lkv * " << ds << ");\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Vb = " << plan_.v_name
        << " + m1_bh * (m1_Lkv * " << ds << ");\n";
    os_ << "  device " << plan_.elem_type << "* m1_Ob = " << plan_.out_name
        << " + m1_bh * (m1_Lq * " << ds << ");\n";
    os_ << "  const float m1_scale_log2 = " << format_float_literal(plan_.attr.scale)
        << " * 1.4426950408889634f;\n";
    os_ << "  (void)m1_tid; (void)m1_D;\n";
  }

  uint32_t compute_shm_floats() const {
    if (plan_.mode == FlashAttnMode::TwoWaveSgmm) {
      return 16u * 512u;
    }
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

  void emit_two_wave_sgmm() {
    constexpr uint32_t kLdp = 512u;
    constexpr uint32_t kBq = 64u;
    constexpr uint32_t kLdqt = 36u;
    constexpr uint32_t kSlot = kBq * kLdqt;
    const std::string ds_l = d_stride_long();
    const std::string ds_a = d_stride_arg();
    const std::string lkv_i = lkv_int_expr();
    const bool full_lkv = (plan_.lkv_static == 512u);
    const bool full_d = (plan_.d_static == 1024u);
    const bool fast_softmax = plan_.fast_only && !plan_.attr.causal && full_lkv;
    const std::string ind = plan_.fast_only ? "  " : "    ";

    if (!plan_.fast_only) {
      os_ << "  if (m1_q0 + 32L <= m1_Lq && m1_Lkv > 0L && m1_Lkv <= 512L && (m1_Lkv & 31L) == 0L && "
             "m1_D > 0L && m1_D <= 1024L && (m1_D & 63L) == 0L) {\n";
    }

    for (int i = 0; i < 4; ++i) {
      for (int j = 2; j < 4; ++j) {
        os_ << ind << "simdgroup_float8x8 s_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }

    auto emit_kq_comp = [&](const std::string& q_ptr_s, const std::string& d_expr, const std::string& ci) {
      if (!full_lkv) {
        os_ << ci << "if ((long)m1_sc0 < m1_Lkv) {\n";
      }
      const std::string ii = full_lkv ? ci : (ci + "  ");
      os_ << ii << "device const float* kb = m1_Kb + (long)m1_sc0 * " << ds_l << " + " << d_expr << ";\n";
      os_ << ii << "for (int ks = 0; ks < " << kBq << "; ks += 16) {\n";
      os_ << ii << "  simdgroup_float8x8 fa_cur, fa_nxt;\n";
      os_ << ii << "  simdgroup_load(fa_cur, kb + ks, " << ds_a << ");\n";
      for (int sub = 0; sub < 16; sub += 8) {
        os_ << ii << "  {\n";
        for (int j = 0; j < 4; ++j) {
          os_ << ii << "    simdgroup_float8x8 fb_" << j << "; simdgroup_load(fb_" << j << ", " << q_ptr_s
              << " + (ks + " << sub << ") * " << kLdqt << " + " << (j * 8) << ", " << kLdqt << ");\n";
        }
        for (int i = 0; i < 4; ++i) {
          if (i + 1 < 4) {
            os_ << ii << "    simdgroup_load(fa_nxt, kb + " << ((i + 1) * 8) << "L * " << ds_l << " + ks + " << sub
                << ", " << ds_a << ");\n";
          } else if (sub == 0) {
            os_ << ii << "    simdgroup_load(fa_nxt, kb + ks + 8, " << ds_a << ");\n";
          }
          for (int j = 0; j < 4; ++j) {
            os_ << ii << "    simdgroup_multiply_accumulate(s_" << i << "_" << j << ", fa_cur, fb_" << j << ", s_" << i
                << "_" << j << ");\n";
          }
          if (i + 1 < 4 || sub == 0) {
            os_ << ii << "    fa_cur = fa_nxt;\n";
          }
        }
        os_ << ii << "  }\n";
      }
      os_ << ii << "}\n";
      if (!full_lkv) {
        os_ << ci << "}\n";
      }
    };

    os_ << ind << "{\n";
    os_ << ind << "  const int m1_sc0 = (int)m1_sg * 32;\n";
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        os_ << ind << "  simdgroup_float8x8 s_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "  const uint m1_qr = m1_tid >> 4;\n";
    os_ << ind << "  const uint m1_qc = (m1_tid & 15u) << 2;\n";
    os_ << ind << "  device const float* m1_qload = m1_Qb + (m1_q0 + (long)m1_qr) * " << ds_l << " + (long)m1_qc;\n";
    os_ << ind << "  uint m1_buf = 0u;\n";
    os_ << ind << "  {\n";
    os_ << ind << "    const float4 rq0 = float4(*(device const packed_float4*)(m1_qload));\n";
    os_ << ind << "    m1_shm[(m1_qc + 0u) * " << kLdqt << "u + m1_qr] = rq0.x;\n";
    os_ << ind << "    m1_shm[(m1_qc + 1u) * " << kLdqt << "u + m1_qr] = rq0.y;\n";
    os_ << ind << "    m1_shm[(m1_qc + 2u) * " << kLdqt << "u + m1_qr] = rq0.z;\n";
    os_ << ind << "    m1_shm[(m1_qc + 3u) * " << kLdqt << "u + m1_qr] = rq0.w;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "  for (long m1_d0 = 0L; m1_d0 + " << kBq << "L < " << ds_l << "; m1_d0 += " << kBq << "L) {\n";
    os_ << ind << "    const uint m1_nb = 1u - m1_buf;\n";
    os_ << ind << "    threadgroup float* m1_qcur = m1_shm + m1_buf * " << kSlot << "u;\n";
    os_ << ind << "    threadgroup float* m1_qnxt = m1_shm + m1_nb * " << kSlot << "u;\n";
    os_ << ind << "    const float4 rqn = float4(*(device const packed_float4*)(m1_qload + m1_d0 + " << kBq << "L));\n";
    emit_kq_comp("m1_qcur", "m1_d0", ind + "    ");
    os_ << ind << "    m1_qnxt[(m1_qc + 0u) * " << kLdqt << "u + m1_qr] = rqn.x;\n";
    os_ << ind << "    m1_qnxt[(m1_qc + 1u) * " << kLdqt << "u + m1_qr] = rqn.y;\n";
    os_ << ind << "    m1_qnxt[(m1_qc + 2u) * " << kLdqt << "u + m1_qr] = rqn.z;\n";
    os_ << ind << "    m1_qnxt[(m1_qc + 3u) * " << kLdqt << "u + m1_qr] = rqn.w;\n";
    os_ << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "    m1_buf = m1_nb;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  {\n";
    os_ << ind << "    threadgroup float* m1_qcur = m1_shm + m1_buf * " << kSlot << "u;\n";
    emit_kq_comp("m1_qcur", "(" + ds_l + " - " + std::to_string(kBq) + "L)", ind + "    ");
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    if (!full_lkv) {
      os_ << ind << "  if ((long)m1_sc0 < m1_Lkv) {\n";
    }
    const std::string si = full_lkv ? (ind + "  ") : (ind + "    ");
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        os_ << si << "simdgroup_store(s_" << i << "_" << j << ", m1_shm + " << (j * 8 * kLdp)
            << " + (m1_sc0 + " << (i * 8) << "), " << kLdp << ", ulong2(0, 0), true);\n";
      }
    }
    if (!full_lkv) {
      os_ << ind << "  }\n";
    }
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";

    auto emit_wave_softmax_pv = [&](uint32_t row_off) {
      os_ << ind << "{\n";
      os_ << ind << "  const int m1_row = (int)m1_sg;\n";
      if (!fast_softmax) {
        const std::string row_abs = (row_off == 0u) ? "m1_row" : ("(" + std::to_string(row_off) + " + m1_row)");
        os_ << ind << "  const long m1_kmax = " << kmax_expr(row_abs) << ";\n";
      }
      os_ << ind << "  float m1_vals[16];\n";
      os_ << ind << "  float m1_mx = -1e30f;\n";
      for (int j = 0; j < 16; ++j) {
        if (fast_softmax) {
          os_ << ind << "  { const int c = (int)m1_lane + " << (j * 32) << "; "
              << "const float v = m1_shm[m1_row * " << kLdp << " + c] * m1_scale_log2; "
              << "m1_vals[" << j << "] = v; m1_mx = max(m1_mx, v); }\n";
        } else {
          os_ << ind << "  { const int c = (int)m1_lane + " << (j * 32) << "; "
              << "const float v = ((long)c < m1_kmax) ? (m1_shm[m1_row * " << kLdp
              << " + c] * m1_scale_log2) : -1e30f; m1_vals[" << j << "] = v; m1_mx = max(m1_mx, v); }\n";
        }
      }
      os_ << ind << "  m1_mx = simd_max(m1_mx);\n";
      if (!fast_softmax) {
        os_ << ind << "  const bool m1_valid = (m1_kmax > 0L) && (m1_mx > -1e29f);\n";
      }
      os_ << ind << "  float m1_rs = 0.0f;\n";
      for (int j = 0; j < 16; ++j) {
        if (fast_softmax) {
          os_ << ind << "  { const float p = exp2(m1_vals[" << j << "] - m1_mx); "
              << "m1_vals[" << j << "] = p; m1_rs += p; }\n";
        } else {
          os_ << ind << "  { const int c = (int)m1_lane + " << (j * 32) << "; "
              << "const float p = (m1_valid && (long)c < m1_kmax) ? exp2(m1_vals[" << j
              << "] - m1_mx) : 0.0f; m1_vals[" << j << "] = p; m1_rs += p; }\n";
        }
      }
      os_ << ind << "  m1_rs = simd_sum(m1_rs);\n";
      if (fast_softmax) {
        os_ << ind << "  const float m1_inv = 1.0f / m1_rs;\n";
      } else {
        os_ << ind << "  const float m1_inv = (m1_valid && m1_rs > 0.0f) ? (1.0f / m1_rs) : 0.0f;\n";
      }
      for (int j = 0; j < 16; ++j) {
        os_ << ind << "  m1_shm[m1_row * " << kLdp << " + (int)m1_lane + " << (j * 32) << "] = m1_vals[" << j
            << "] * m1_inv;\n";
      }
      os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << ind << "}\n";

      os_ << ind << "{\n";
      os_ << ind << "  const long m1_sgd0 = (long)m1_sg * 64L;\n";
      if (!full_d) {
        os_ << ind << "  if (m1_sgd0 < " << ds_l << ") {\n";
      }
      const std::string pi = full_d ? (ind + "  ") : (ind + "    ");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
          os_ << pi << "simdgroup_float8x8 o_" << i << "_" << j
              << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        }
      }
      os_ << pi << "device const float* vp = m1_Vb + m1_sgd0;\n";
      os_ << pi << "for (int m1_k0 = 0; m1_k0 < " << lkv_i << "; m1_k0 += 16) {\n";
      os_ << pi << "  simdgroup_float8x8 vf_cur, vf_nxt;\n";
      os_ << pi << "  simdgroup_load(vf_cur, vp, " << ds_a << ");\n";
      for (int ku = 0; ku < 2; ++ku) {
        os_ << pi << "  {\n";
        os_ << pi << "    const int kk = m1_k0 + " << (ku * 8) << ";\n";
        os_ << pi << "    simdgroup_float8x8 pf_0; simdgroup_load(pf_0, m1_shm + kk, " << kLdp << ");\n";
        os_ << pi << "    simdgroup_float8x8 pf_1; simdgroup_load(pf_1, m1_shm + " << (8 * kLdp) << " + kk, " << kLdp
            << ");\n";
        for (int j = 0; j < 8; ++j) {
          if (j + 1 < 8) {
            os_ << pi << "    simdgroup_load(vf_nxt, vp + " << (ku * 8) << "L * " << ds_l << " + " << ((j + 1) * 8)
                << ", " << ds_a << ");\n";
          } else if (ku == 0) {
            os_ << pi << "    simdgroup_load(vf_nxt, vp + 8L * " << ds_l << ", " << ds_a << ");\n";
          }
          os_ << pi << "    simdgroup_multiply_accumulate(o_0_" << j << ", pf_0, vf_cur, o_0_" << j << ");\n";
          os_ << pi << "    simdgroup_multiply_accumulate(o_1_" << j << ", pf_1, vf_cur, o_1_" << j << ");\n";
          if (j + 1 < 8 || ku == 0) {
            os_ << pi << "    vf_cur = vf_nxt;\n";
          }
        }
        os_ << pi << "  }\n";
      }
      os_ << pi << "  vp += 16L * " << ds_l << ";\n";
      os_ << pi << "}\n";
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
          os_ << pi << "simdgroup_store(o_" << i << "_" << j << ", m1_Ob + (m1_q0 + " << (row_off + i * 8) << "L) * "
              << ds_l << " + m1_sgd0 + " << (j * 8) << "L, " << ds_a << ");\n";
        }
      }
      if (!full_d) {
        os_ << ind << "  }\n";
      }
      os_ << ind << "}\n";
    };

    emit_wave_softmax_pv(0u);
    os_ << ind << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "{\n";
    os_ << ind << "  const int m1_sc0 = (int)m1_sg * 32;\n";
    if (!full_lkv) {
      os_ << ind << "  if ((long)m1_sc0 < m1_Lkv) {\n";
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 2; j < 4; ++j) {
        os_ << si << "simdgroup_store(s_" << i << "_" << j << ", m1_shm + " << ((j - 2) * 8 * kLdp)
            << " + (m1_sc0 + " << (i * 8) << "), " << kLdp << ", ulong2(0, 0), true);\n";
      }
    }
    if (!full_lkv) {
      os_ << ind << "  }\n";
    }
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";
    emit_wave_softmax_pv(16u);

    if (!plan_.fast_only) {
      os_ << "    return;\n";
      os_ << "  }\n";
      emit_general_rowwise(256u, "  ");
    }
  }

  void emit_single_pass_sgmm() {
    constexpr uint32_t kLdp = 512u;
    const std::string ds_l = d_stride_long();
    const std::string ds_a = d_stride_arg();
    const std::string lkv_i = lkv_int_expr();
    const std::string ind = plan_.fast_only ? "  " : "    ";
    if (!plan_.fast_only) {
      os_ << "  if (m1_q0 + 16L <= m1_Lq && m1_Lkv > 0L && m1_Lkv <= 512L && (m1_Lkv & 31L) == 0L && "
             "m1_D > 0L && m1_D <= 1024L && (m1_D & 63L) == 0L) {\n";
    }

    os_ << ind << "{\n";
    os_ << ind << "  const int m1_sc0 = (int)m1_sg * 32;\n";
    os_ << ind << "  if ((long)m1_sc0 < m1_Lkv) {\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
        os_ << ind << "    simdgroup_float8x8 s_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "    device const float* qp_0 = m1_Qb + m1_q0 * " << ds_l << ";\n";
    os_ << ind << "    device const float* qp_1 = qp_0 + 8L * " << ds_l << ";\n";
    for (int j = 0; j < 4; ++j) {
      os_ << ind << "    device const float* kp_" << j << " = m1_Kb + (long)(m1_sc0 + " << (j * 8) << ") * " << ds_l
          << ";\n";
    }
    os_ << ind << "    for (long m1_d0 = 0L; m1_d0 < " << ds_l << "; m1_d0 += 16L) {\n";
    for (int ks = 0; ks < 16; ks += 8) {
      os_ << ind << "      {\n";
      os_ << ind << "        simdgroup_float8x8 fa_0; simdgroup_load(fa_0, qp_0 + " << ks << ", " << ds_a << ");\n";
      os_ << ind << "        simdgroup_float8x8 fa_1; simdgroup_load(fa_1, qp_1 + " << ks << ", " << ds_a << ");\n";
      os_ << ind << "        simdgroup_float8x8 fb_cur, fb_nxt;\n";
      os_ << ind << "        simdgroup_load(fb_cur, kp_0 + " << ks << ", " << ds_a << ", ulong2(0, 0), true);\n";
      for (int j = 0; j < 4; ++j) {
        if (j + 1 < 4) {
          os_ << ind << "        simdgroup_load(fb_nxt, kp_" << (j + 1) << " + " << ks << ", " << ds_a
              << ", ulong2(0, 0), true);\n";
        }
        os_ << ind << "        simdgroup_multiply_accumulate(s_0_" << j << ", fa_0, fb_cur, s_0_" << j << ");\n";
        os_ << ind << "        simdgroup_multiply_accumulate(s_1_" << j << ", fa_1, fb_cur, s_1_" << j << ");\n";
        if (j + 1 < 4) {
          os_ << ind << "        fb_cur = fb_nxt;\n";
        }
      }
      os_ << ind << "      }\n";
    }
    os_ << ind << "      qp_0 += 16L; qp_1 += 16L;\n";
    os_ << ind << "      kp_0 += 16L; kp_1 += 16L; kp_2 += 16L; kp_3 += 16L;\n";
    os_ << ind << "    }\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
        os_ << ind << "    simdgroup_store(s_" << i << "_" << j << ", m1_shm + " << (i * 8 * kLdp)
            << " + (m1_sc0 + " << (j * 8) << "), " << kLdp << ");\n";
      }
    }
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";

    os_ << ind << "{\n";
    os_ << ind << "  const int m1_row = (int)m1_sg;\n";
    os_ << ind << "  const long m1_kmax = " << kmax_expr("m1_row") << ";\n";
    os_ << ind << "  float m1_vals[16];\n";
    os_ << ind << "  float m1_mx = -1e30f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << ind << "  { const int c = (int)m1_lane + " << (j * 32) << "; "
          << "const float v = ((long)c < m1_kmax) ? (m1_shm[m1_row * " << kLdp
          << " + c] * m1_scale_log2) : -1e30f; m1_vals[" << j << "] = v; m1_mx = max(m1_mx, v); }\n";
    }
    os_ << ind << "  m1_mx = simd_max(m1_mx);\n";
    os_ << ind << "  const bool m1_valid = (m1_kmax > 0L) && (m1_mx > -1e29f);\n";
    os_ << ind << "  float m1_rs = 0.0f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << ind << "  { const int c = (int)m1_lane + " << (j * 32) << "; "
          << "const float p = (m1_valid && (long)c < m1_kmax) ? exp2(m1_vals[" << j << "] - m1_mx) : 0.0f; "
          << "m1_vals[" << j << "] = p; m1_rs += p; }\n";
    }
    os_ << ind << "  m1_rs = simd_sum(m1_rs);\n";
    os_ << ind << "  const float m1_inv = (m1_valid && m1_rs > 0.0f) ? (1.0f / m1_rs) : 0.0f;\n";
    for (int j = 0; j < 16; ++j) {
      os_ << ind << "  m1_shm[m1_row * " << kLdp << " + (int)m1_lane + " << (j * 32) << "] = m1_vals[" << j
          << "] * m1_inv;\n";
    }
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";

    os_ << ind << "{\n";
    os_ << ind << "  const int m1_sgd0 = (int)m1_sg * 64;\n";
    os_ << ind << "  if ((long)m1_sgd0 < " << ds_l << ") {\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        os_ << ind << "    simdgroup_float8x8 o_" << i << "_" << j
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "    device const float* vp = m1_Vb + (long)m1_sgd0;\n";
    os_ << ind << "    for (int m1_k0 = 0; m1_k0 < " << lkv_i << "; m1_k0 += 16) {\n";
    for (int ku = 0; ku < 2; ++ku) {
      os_ << ind << "      {\n";
      os_ << ind << "        const int kk = m1_k0 + " << (ku * 8) << ";\n";
      os_ << ind << "        simdgroup_float8x8 pf_0; simdgroup_load(pf_0, m1_shm + kk, " << kLdp << ");\n";
      os_ << ind << "        simdgroup_float8x8 pf_1; simdgroup_load(pf_1, m1_shm + " << (8 * kLdp) << " + kk, " << kLdp
          << ");\n";
      os_ << ind << "        simdgroup_float8x8 vf_cur, vf_nxt;\n";
      os_ << ind << "        simdgroup_load(vf_cur, vp + " << (ku * 8) << "L * " << ds_l << ", " << ds_a << ");\n";
      for (int j = 0; j < 8; ++j) {
        if (j + 1 < 8) {
          os_ << ind << "        simdgroup_load(vf_nxt, vp + " << (ku * 8) << "L * " << ds_l << " + " << ((j + 1) * 8)
              << ", " << ds_a << ");\n";
        }
        os_ << ind << "        simdgroup_multiply_accumulate(o_0_" << j << ", pf_0, vf_cur, o_0_" << j << ");\n";
        os_ << ind << "        simdgroup_multiply_accumulate(o_1_" << j << ", pf_1, vf_cur, o_1_" << j << ");\n";
        if (j + 1 < 8) {
          os_ << ind << "        vf_cur = vf_nxt;\n";
        }
      }
      os_ << ind << "      }\n";
    }
    os_ << ind << "      vp += 16L * " << ds_l << ";\n";
    os_ << ind << "    }\n";
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 8; ++j) {
        os_ << ind << "    simdgroup_store(o_" << i << "_" << j << ", m1_Ob + (m1_q0 + " << (i * 8) << "L) * " << ds_l
            << " + (long)(m1_sgd0 + " << (j * 8) << "), " << ds_a << ");\n";
      }
    }
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
    if (!plan_.fast_only) {
      os_ << "    return;\n";
      os_ << "  }\n";
      emit_general_rowwise(256u, "  ");
    }
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

    if (!plan_.fast_only) {
      os_ << "  if (m1_q0 + 16L <= m1_Lq && m1_Lkv > 0L && (m1_Lkv & 31L) == 0L && "
             "m1_D > 0L && m1_D <= 1024L && (m1_D & 63L) == 0L) {\n";
    } else {
      os_ << "  {\n";
    }
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
    if (!plan_.fast_only) {
      os_ << "    return;\n";
      os_ << "  }\n";
      emit_general_rowwise(std::min<uint32_t>(bn, 256u), "  ");
    } else {
      os_ << "  }\n";
    }
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
  res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan->out_name);
  return res;
}

}  // namespace m1triton::codegen::metal

#include "codegen/metal/l3_gemm_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

namespace m1triton::codegen::metal {

namespace {

using ir::GroupedGemmAttr;
using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::QuantDotAttr;
using ir::ScalarDType;
using ir::TargetSpec;

struct SgTileLayout {
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t num_sg = 4;
  uint32_t tg_threads = 128;
  uint32_t sg_m = 2;
  uint32_t sg_n = 2;
  uint32_t tm = 4;
  uint32_t tn = 4;
  uint32_t pad_a = 4;
  uint32_t pad_b = 4;
  uint32_t a_ld = 20;
  uint32_t b_ld = 68;
  uint32_t sb_elems = 0;
  bool b_native_t = false;
  uint32_t smem_bytes = 0;
};

bool compute_sg_tile_layout(uint32_t bm, uint32_t bn, uint32_t bk, uint32_t num_sg, uint32_t sram_elem_bytes,
                            bool b_native_t, uint32_t extra_smem_bytes, const TargetSpec& target, SgTileLayout& out,
                            std::string& err) {
  if (bm < 8u || (bm % 8u) != 0u || bn < 8u || (bn % 8u) != 0u || bk < 8u || (bk % 8u) != 0u) {
    err = "L3 GEMM tile dimensions (BM, BN, BK) must be positive multiples of 8";
    return false;
  }
  if (num_sg == 0u) {
    err = "L3 GEMM num_simdgroups must be positive";
    return false;
  }
  const uint32_t simd = std::max<uint32_t>(1u, target.simd_width);
  const uint32_t tg_threads = num_sg * simd;
  if (tg_threads > target.max_threads_per_threadgroup) {
    err = "L3 GEMM threadgroup exceeds target.max_threads_per_threadgroup";
    return false;
  }

  const uint32_t m8 = bm / 8u;
  const uint32_t n8 = bn / 8u;
  uint32_t best_sg_m = 0u;
  uint32_t best_sg_n = 0u;
  uint32_t best_score = UINT32_MAX;
  for (uint32_t sm = 1u; sm <= num_sg; ++sm) {
    if ((num_sg % sm) != 0u) continue;
    const uint32_t sn = num_sg / sm;
    if ((m8 % sm) != 0u || (n8 % sn) != 0u) continue;
    const uint32_t tm = m8 / sm;
    const uint32_t tn = n8 / sn;
    const uint32_t score = (tm + tn) * 16u + (tm > tn ? (tm - tn) : (tn - tm));
    if (score < best_score) {
      best_score = score;
      best_sg_m = sm;
      best_sg_n = sn;
    }
  }
  if (best_sg_m == 0u) {
    err = "L3 GEMM cannot factor num_simdgroups across (BM/8, BN/8) simdgroup tiles";
    return false;
  }

  const uint32_t pad = (sram_elem_bytes == 2u) ? 8u : 4u;
  const uint32_t a_ld = bk + pad;
  const uint32_t b_ld = b_native_t ? (bk + pad) : (bn + pad);
  const uint32_t sb_elems = b_native_t ? (bn * b_ld) : (bk * b_ld);
  const uint32_t smem_bytes = (bm * a_ld + sb_elems) * sram_elem_bytes + extra_smem_bytes;
  if (!target.fits_threadgroup_memory(smem_bytes)) {
    err = "L3 GEMM tile exceeds target 32 KB threadgroup SRAM budget";
    return false;
  }

  out.bm = bm;
  out.bn = bn;
  out.bk = bk;
  out.num_sg = num_sg;
  out.tg_threads = tg_threads;
  out.sg_m = best_sg_m;
  out.sg_n = best_sg_n;
  out.tm = m8 / best_sg_m;
  out.tn = n8 / best_sg_n;
  out.pad_a = pad;
  out.pad_b = pad;
  out.a_ld = a_ld;
  out.b_ld = b_ld;
  out.sb_elems = sb_elems;
  out.b_native_t = b_native_t;
  out.smem_bytes = smem_bytes;
  return true;
}

std::optional<int64_t> find_constexpr_i64(const Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name) {
      return kp.constexpr_val;
    }
  }
  return std::nullopt;
}

uint32_t resolve_pos_override(const Kernel& kernel, const char* primary, const char* secondary, uint32_t fallback) {
  if (auto v = positive_constexpr_param(kernel, primary)) return static_cast<uint32_t>(*v);
  if (secondary != nullptr) {
    if (auto v = positive_constexpr_param(kernel, secondary)) return static_cast<uint32_t>(*v);
  }
  return fallback;
}

void emit_mma2_helpers(std::ostringstream& os, bool is_fp16) {
  if (is_fp16) {
    os << "static inline float2 m1_mma2_hf(half2 a, half2 b, float2 c) {\n"
       << "  simdgroup_half8x8 am, bm;\n"
       << "  simdgroup_float8x8 cm;\n"
       << "  reinterpret_cast<thread half2&>(am.thread_elements()) = a;\n"
       << "  reinterpret_cast<thread half2&>(bm.thread_elements()) = b;\n"
       << "  reinterpret_cast<thread float2&>(cm.thread_elements()) = c;\n"
       << "  simdgroup_multiply_accumulate(cm, am, bm, cm);\n"
       << "  return reinterpret_cast<thread float2&>(cm.thread_elements());\n"
       << "}\n\n";
  } else {
    os << "static inline float2 m1_mma2(float2 a, float2 b, float2 c) {\n"
       << "  simdgroup_float8x8 am, bm, cm;\n"
       << "  reinterpret_cast<thread float2&>(am.thread_elements()) = a;\n"
       << "  reinterpret_cast<thread float2&>(bm.thread_elements()) = b;\n"
       << "  reinterpret_cast<thread float2&>(cm.thread_elements()) = c;\n"
       << "  simdgroup_multiply_accumulate(cm, am, bm, cm);\n"
       << "  return reinterpret_cast<thread float2&>(cm.thread_elements());\n"
       << "}\n\n";
  }
}

void emit_mma2_tile_compute(std::ostringstream& os, const SgTileLayout& s, bool is_fp16, const std::string& ind) {
  const std::string v2 = is_fp16 ? "half2" : "float2";
  const std::string fn = is_fp16 ? "m1_mma2_hf" : "m1_mma2";
  for (uint32_t ks = 0u; ks < s.bk; ks += 8u) {
    os << ind << "{\n";
    for (uint32_t j = 0u; j < s.tn; ++j) {
      if (s.b_native_t) {
        os << ind << "  " << v2 << " m1_fb_" << j << " = " << v2 << "(m1_sb[(m1_sg_col + " << (j * 8u)
           << "u + m1_fc) * " << s.b_ld << "u + (" << ks << "u + m1_fr)], m1_sb[(m1_sg_col + " << (j * 8u)
           << "u + m1_fc + 1u) * " << s.b_ld << "u + (" << ks << "u + m1_fr)]);\n";
      } else {
        os << ind << "  " << v2 << " m1_fb_" << j << " = *(threadgroup const " << v2 << "*)(m1_sb + (" << ks
           << "u + m1_fr) * " << s.b_ld << "u + (m1_sg_col + " << (j * 8u) << "u + m1_fc));\n";
      }
    }
    os << ind << "  " << v2 << " m1_fa_c = *(threadgroup const " << v2 << "*)(m1_sa + (m1_sg_row + m1_fr) * "
       << s.a_ld << "u + (" << ks << "u + m1_fc));\n"
       << ind << "  " << v2 << " m1_fa_n;\n";
    for (uint32_t i = 0u; i < s.tm; ++i) {
      if (i + 1u < s.tm) {
        os << ind << "  m1_fa_n = *(threadgroup const " << v2 << "*)(m1_sa + (m1_sg_row + " << ((i + 1u) * 8u)
           << "u + m1_fr) * " << s.a_ld << "u + (" << ks << "u + m1_fc));\n";
      }
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << ind << "  m1_c_" << i << "_" << j << " = " << fn << "(m1_fa_c, m1_fb_" << j << ", m1_c_" << i << "_"
           << j << ");\n";
      }
      if (i + 1u < s.tm) {
        os << ind << "  m1_fa_c = m1_fa_n;\n";
      }
    }
    os << ind << "}\n";
  }
}

void emit_vec4_load_a(std::ostringstream& os, const SgTileLayout& s, const std::string& a_ptr,
                      const std::string& k_expr, const std::string& dst_prefix, bool is_fp16,
                      const std::string& m_guard_limit, const std::string& ind) {
  const std::string v4 = is_fp16 ? "half4" : "float4";
  const std::string pv4 = is_fp16 ? "packed_half4" : "packed_float4";
  const std::string zero = is_fp16 ? "half4(0.0h)" : "float4(0.0f)";
  const uint32_t nvec = (s.bm * s.bk) / 4u;
  const uint32_t vpr = s.bk / 4u;
  const uint32_t per = (nvec + s.tg_threads - 1u) / s.tg_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    const std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    const std::string off = "ulong(m1_row0 + " + r + ") * ulong(m1_K) + ulong(" + k_expr + " + " + c4 + ")";
    std::string guard;
    if ((p + 1u) * s.tg_threads > nvec) {
      guard = "(" + vi + " < " + std::to_string(nvec) + "u)";
    }
    if (!m_guard_limit.empty()) {
      const std::string mg = "((m1_row0 + " + r + ") < " + m_guard_limit + ")";
      guard = guard.empty() ? mg : ("(" + guard + " && " + mg + ")");
    }
    if (!guard.empty()) {
      os << ind << v4 << " " << dst_prefix << "_" << p << " = " << zero << ";\n"
         << ind << "if (" << guard << ") " << dst_prefix << "_" << p << " = " << v4 << "(*(const device " << pv4
         << "*)(" << a_ptr << " + " << off << "));\n";
    } else {
      os << ind << v4 << " " << dst_prefix << "_" << p << " = " << v4 << "(*(const device " << pv4 << "*)(" << a_ptr
         << " + " << off << "));\n";
    }
  }
}

void emit_vec4_store_a(std::ostringstream& os, const SgTileLayout& s, const std::string& src_prefix, bool is_fp16,
                       const std::string& ind) {
  const std::string v4 = is_fp16 ? "half4" : "float4";
  const uint32_t nvec = (s.bm * s.bk) / 4u;
  const uint32_t vpr = s.bk / 4u;
  const uint32_t per = (nvec + s.tg_threads - 1u) / s.tg_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    const std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    const std::string cond = ((p + 1u) * s.tg_threads > nvec) ? ("if (" + vi + " < " + std::to_string(nvec) + "u) ") : "";
    os << ind << cond << "*(threadgroup " << v4 << "*)(m1_sa + " << r << " * " << s.a_ld << "u + " << c4 << ") = "
       << src_prefix << "_" << p << ";\n";
  }
}

void emit_vec4_load_b(std::ostringstream& os, const SgTileLayout& s, const std::string& b_ptr,
                      const std::string& k_expr, const std::string& dst_prefix, bool transpose_b, bool is_fp16,
                      const std::string& ind) {
  const std::string v4 = is_fp16 ? "half4" : "float4";
  const std::string pv4 = is_fp16 ? "packed_half4" : "packed_float4";
  const std::string zero = is_fp16 ? "half4(0.0h)" : "float4(0.0f)";
  const uint32_t nvec = (s.bk * s.bn) / 4u;
  const uint32_t vpr = (transpose_b ? s.bk : s.bn) / 4u;
  const uint32_t per = (nvec + s.tg_threads - 1u) / s.tg_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    const std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    const std::string off = transpose_b
                                ? ("ulong(m1_col0 + " + r + ") * ulong(m1_K) + ulong(" + k_expr + " + " + c4 + ")")
                                : ("ulong(" + k_expr + " + " + r + ") * ulong(m1_N) + ulong(m1_col0 + " + c4 + ")");
    if ((p + 1u) * s.tg_threads > nvec) {
      os << ind << v4 << " " << dst_prefix << "_" << p << " = " << zero << ";\n"
         << ind << "if (" << vi << " < " << nvec << "u) " << dst_prefix << "_" << p << " = " << v4
         << "(*(const device " << pv4 << "*)(" << b_ptr << " + " << off << "));\n";
    } else {
      os << ind << v4 << " " << dst_prefix << "_" << p << " = " << v4 << "(*(const device " << pv4 << "*)(" << b_ptr
         << " + " << off << "));\n";
    }
  }
}

void emit_vec4_store_b(std::ostringstream& os, const SgTileLayout& s, const std::string& src_prefix, bool transpose_b,
                       bool is_fp16, const std::string& ind) {
  const std::string v4 = is_fp16 ? "half4" : "float4";
  const uint32_t nvec = (s.bk * s.bn) / 4u;
  const uint32_t vpr = (transpose_b ? s.bk : s.bn) / 4u;
  const uint32_t per = (nvec + s.tg_threads - 1u) / s.tg_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    const std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    const std::string cond = ((p + 1u) * s.tg_threads > nvec) ? ("if (" + vi + " < " + std::to_string(nvec) + "u) ") : "";
    os << ind << cond << "*(threadgroup " << v4 << "*)(m1_sb + " << r << " * " << s.b_ld << "u + " << c4 << ") = "
       << src_prefix << "_" << p << ";\n";
  }
}

void emit_store_out_tile(std::ostringstream& os, const SgTileLayout& s, const std::string& out_ptr,
                         const std::string& m_limit, ScalarDType out_dt, const std::string& ind) {
  const bool out_fp16 = (out_dt == ScalarDType::F16);
  const std::string out_t = out_fp16 ? "half" : "float";
  os << ind << "if (m1_full_m && m1_full_n) {\n";
  if (!out_fp16) {
    for (uint32_t i = 0u; i < s.tm; ++i) {
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << ind << "  { simdgroup_float8x8 m1_cm; reinterpret_cast<thread float2&>(m1_cm.thread_elements()) = m1_c_"
           << i << "_" << j << "; simdgroup_store(m1_cm, " << out_ptr << " + ulong(m1_row0 + m1_sg_row + " << (i * 8u)
           << "u) * ulong(m1_N) + ulong(m1_col0 + m1_sg_col + " << (j * 8u) << "u), ulong(m1_N)); }\n";
      }
    }
  } else {
    for (uint32_t i = 0u; i < s.tm; ++i) {
      os << ind << "  { device half* m1_or = " << out_ptr << " + ulong(m1_row0 + m1_sg_row + " << (i * 8u)
         << "u + m1_fr) * ulong(m1_N);\n";
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << ind << "    *(device packed_half2*)(m1_or + ulong(m1_col0 + m1_sg_col + " << (j * 8u)
           << "u + m1_fc)) = half2(m1_c_" << i << "_" << j << ");\n";
      }
      os << ind << "  }\n";
    }
  }
  os << ind << "} else {\n";
  for (uint32_t i = 0u; i < s.tm; ++i) {
    os << ind << "  {\n"
       << ind << "    uint m1_gr = m1_row0 + m1_sg_row + " << (i * 8u) << "u + m1_fr;\n"
       << ind << "    if (m1_gr < " << m_limit << ") {\n"
       << ind << "      device " << out_t << "* m1_or = " << out_ptr << " + ulong(m1_gr) * ulong(m1_N);\n";
    for (uint32_t j = 0u; j < s.tn; ++j) {
      os << ind << "      { uint m1_gc = m1_col0 + m1_sg_col + " << (j * 8u)
         << "u + m1_fc; if (m1_gc < m1_N) m1_or[m1_gc] = " << out_t << "(m1_c_" << i << "_" << j
         << ".x); if (m1_gc + 1u < m1_N) m1_or[m1_gc + 1u] = " << out_t << "(m1_c_" << i << "_" << j << ".y); }\n";
    }
    os << ind << "    }\n"
       << ind << "  }\n";
  }
  os << ind << "}\n";
}

struct GroupedGemmPlan {
  const Op* op = nullptr;
  GroupedGemmAttr attr;
  SgTileLayout sched;
  ScalarDType elem_dt = ScalarDType::F32;
  std::string a_name;
  std::string w_name;
  std::string offsets_name;
  std::string out_name;
  std::string e_expr;
  std::string n_expr;
  std::string k_expr;
};

bool build_grouped_gemm_plan(const Kernel& kernel, const Op& op, const TargetSpec& target, GroupedGemmPlan& plan,
                             std::string& err) {
  const auto* gg = std::get_if<GroupedGemmAttr>(&op.attr);
  if (gg == nullptr) {
    err = "grouped_gemm op missing GroupedGemmAttr";
    return false;
  }
  if (op.operands.size() != 7u || !op.results.empty()) {
    err = "grouped_gemm expects 7 operands (A, W, Offsets, Out, E, N, K) and 0 results";
    return false;
  }

  plan.op = &op;
  plan.attr = *gg;

  const auto a_pidx = param_index_of(kernel, op.operands[0]);
  const auto w_pidx = param_index_of(kernel, op.operands[1]);
  const auto off_pidx = param_index_of(kernel, op.operands[2]);
  const auto out_pidx = param_index_of(kernel, op.operands[3]);
  if (!a_pidx || !w_pidx || !off_pidx || !out_pidx) {
    err = "grouped_gemm pointer operands 0..3 must be kernel pointer parameters";
    return false;
  }
  if (!kernel.params[*a_pidx].type.is_ptr() || !kernel.params[*w_pidx].type.is_ptr() ||
      !kernel.params[*off_pidx].type.is_ptr() || !kernel.params[*out_pidx].type.is_ptr()) {
    err = "grouped_gemm operands 0..3 must be device pointers";
    return false;
  }
  if (*out_pidx == *a_pidx || *out_pidx == *w_pidx || *out_pidx == *off_pidx) {
    err = "grouped_gemm Out pointer must not alias A, W, or Offsets";
    return false;
  }

  const ScalarDType a_dt = std::get<ir::PtrType>(kernel.params[*a_pidx].type.value).pointee;
  const ScalarDType w_dt = std::get<ir::PtrType>(kernel.params[*w_pidx].type.value).pointee;
  const ScalarDType off_dt = std::get<ir::PtrType>(kernel.params[*off_pidx].type.value).pointee;
  const ScalarDType out_dt = std::get<ir::PtrType>(kernel.params[*out_pidx].type.value).pointee;
  if ((a_dt != ScalarDType::F32 && a_dt != ScalarDType::F16) || w_dt != a_dt || out_dt != a_dt) {
    err = "grouped_gemm requires matching F32 or F16 pointer dtypes for A, W, and Out";
    return false;
  }
  if (off_dt != ScalarDType::I32 && off_dt != ScalarDType::U32) {
    err = "grouped_gemm Offsets pointer must be I32 or U32";
    return false;
  }

  auto e_expr = scalar_long_expr(kernel, op.operands[4]);
  auto n_expr = scalar_long_expr(kernel, op.operands[5]);
  auto k_expr = scalar_long_expr(kernel, op.operands[6]);
  if (!e_expr || !n_expr || !k_expr) {
    err = "grouped_gemm E, N, K operands must be scalar parameters or integer constants";
    return false;
  }

  const uint32_t bm = resolve_pos_override(kernel, "BLOCK_M", "BM", gg->bm > 0u ? gg->bm : 64u);
  const uint32_t bn = resolve_pos_override(kernel, "BLOCK_N", "BN", gg->bn > 0u ? gg->bn : 64u);
  const uint32_t bk = resolve_pos_override(kernel, "BLOCK_K", "BK", gg->bk > 0u ? gg->bk : 16u);
  const uint32_t num_sg =
      resolve_pos_override(kernel, "NUM_SG", "NUM_SIMDGROUPS", gg->num_simdgroups > 0u ? gg->num_simdgroups : 4u);
  if (auto p_ovr = find_constexpr_i64(kernel, "PERSISTENT")) {
    plan.attr.persistent = (*p_ovr != 0);
  }
  if (auto tb_ovr = find_constexpr_i64(kernel, "TRANSPOSE_B")) {
    plan.attr.transpose_b = (*tb_ovr != 0);
  }
  plan.attr.max_tgs_per_core =
      resolve_pos_override(kernel, "MAX_TGS_PER_CORE", nullptr, gg->max_tgs_per_core > 0u ? gg->max_tgs_per_core : 2u);

  const uint32_t elem_bytes = (a_dt == ScalarDType::F16) ? 2u : 4u;
  const uint32_t extra_smem = plan.attr.persistent ? 48u : 0u;
  if (!compute_sg_tile_layout(bm, bn, bk, num_sg, elem_bytes, plan.attr.transpose_b, extra_smem, target, plan.sched,
                              err)) {
    return false;
  }

  plan.elem_dt = a_dt;
  plan.a_name = kernel.params[*a_pidx].name;
  plan.w_name = kernel.params[*w_pidx].name;
  plan.offsets_name = kernel.params[*off_pidx].name;
  plan.out_name = kernel.params[*out_pidx].name;
  plan.e_expr = *e_expr;
  plan.n_expr = *n_expr;
  plan.k_expr = *k_expr;
  return true;
}

void emit_grouped_gemm_tile_body(std::ostringstream& os, const GroupedGemmPlan& plan, const std::string& ind) {
  const auto& s = plan.sched;
  const bool is_fp16 = (plan.elem_dt == ScalarDType::F16);
  const std::string sram_t = is_fp16 ? "half" : "float";
  const std::string zero_s = is_fp16 ? "0.0h" : "0.0f";
  const bool tb = plan.attr.transpose_b;

  const uint32_t a_per = ((s.bm * s.bk) / 4u + s.tg_threads - 1u) / s.tg_threads;
  const uint32_t b_per = ((s.bk * s.bn) / 4u + s.tg_threads - 1u) / s.tg_threads;
  const bool use_reg_prefetch = (s.tm * s.tn <= 16u) && (a_per + b_per <= 6u);

  os << ind << "device const " << sram_t << "* m1_A_e = " << plan.a_name << " + ulong(m1_s_e) * ulong(m1_K);\n"
     << ind << "device const " << sram_t << "* m1_W_e = " << plan.w_name
     << " + ulong(m1_expert) * ulong(m1_N) * ulong(m1_K);\n"
     << ind << "device " << sram_t << "* m1_Out_e = " << plan.out_name << " + ulong(m1_s_e) * ulong(m1_N);\n";
  for (uint32_t i = 0u; i < s.tm; ++i) {
    for (uint32_t j = 0u; j < s.tn; ++j) {
      os << ind << "float2 m1_c_" << i << "_" << j << " = float2(0.0f);\n";
    }
  }
  os << ind << "bool m1_full_m = (m1_row0 + " << s.bm << "u <= m1_m_e);\n"
     << ind << "bool m1_full_n = (m1_col0 + " << s.bn << "u <= m1_N);\n"
     << ind << "uint m1_k0 = 0u;\n"
     << ind << "if (m1_full_n) {\n"
     << ind << "  uint m1_k_full_end = m1_K & ~" << (s.bk - 1u) << "u;\n"
     << ind << "  if (m1_full_m) {\n";

  if (use_reg_prefetch) {
    os << ind << "    if (m1_k0 < m1_k_full_end) {\n"
       << ind << "      {\n";
    emit_vec4_load_a(os, s, "m1_A_e", "m1_k0", "m1_pa", is_fp16, "", ind + "        ");
    emit_vec4_load_b(os, s, "m1_W_e", "m1_k0", "m1_pb", tb, is_fp16, ind + "        ");
    emit_vec4_store_a(os, s, "m1_pa", is_fp16, ind + "        ");
    emit_vec4_store_b(os, s, "m1_pb", tb, is_fp16, ind + "        ");
    os << ind << "      }\n"
       << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "      for (m1_k0 += " << s.bk << "u; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_vec4_load_a(os, s, "m1_A_e", "m1_k0", "m1_ra", is_fp16, "", ind + "        ");
    emit_vec4_load_b(os, s, "m1_W_e", "m1_k0", "m1_rb", tb, is_fp16, ind + "        ");
    emit_mma2_tile_compute(os, s, is_fp16, ind + "        ");
    os << ind << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_vec4_store_a(os, s, "m1_ra", is_fp16, ind + "        ");
    emit_vec4_store_b(os, s, "m1_rb", tb, is_fp16, ind + "        ");
    os << ind << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "      }\n";
    emit_mma2_tile_compute(os, s, is_fp16, ind + "      ");
    os << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "    }\n";
  } else {
    os << ind << "    for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_vec4_load_a(os, s, "m1_A_e", "m1_k0", "m1_ra", is_fp16, "", ind + "      ");
    emit_vec4_load_b(os, s, "m1_W_e", "m1_k0", "m1_rb", tb, is_fp16, ind + "      ");
    emit_vec4_store_a(os, s, "m1_ra", is_fp16, ind + "      ");
    emit_vec4_store_b(os, s, "m1_rb", tb, is_fp16, ind + "      ");
    os << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma2_tile_compute(os, s, is_fp16, ind + "      ");
    os << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "    }\n";
  }

  os << ind << "  } else {\n"
     << ind << "    for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
  emit_vec4_load_a(os, s, "m1_A_e", "m1_k0", "m1_ra", is_fp16, "m1_m_e", ind + "      ");
  emit_vec4_load_b(os, s, "m1_W_e", "m1_k0", "m1_rb", tb, is_fp16, ind + "      ");
  emit_vec4_store_a(os, s, "m1_ra", is_fp16, ind + "      ");
  emit_vec4_store_b(os, s, "m1_rb", tb, is_fp16, ind + "      ");
  os << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_mma2_tile_compute(os, s, is_fp16, ind + "      ");
  os << ind << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << ind << "    }\n"
     << ind << "  }\n"
     << ind << "}\n";

  os << ind << "if (m1_k0 < m1_K) {\n"
     << ind << "  for (; m1_k0 < m1_K; m1_k0 += " << s.bk << "u) {\n"
     << ind << "    for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bm * s.bk) << "u; m1_idx += " << s.tg_threads
     << "u) {\n"
     << ind << "      uint m1_r = m1_idx / " << s.bk << "u;\n"
     << ind << "      uint m1_c = m1_idx % " << s.bk << "u;\n"
     << ind << "      uint m1_gr = m1_row0 + m1_r;\n"
     << ind << "      uint m1_gk = m1_k0 + m1_c;\n"
     << ind << "      m1_sa[m1_r * " << s.a_ld
     << "u + m1_c] = (m1_gr < m1_m_e && m1_gk < m1_K) ? m1_A_e[ulong(m1_gr) * ulong(m1_K) + ulong(m1_gk)] : " << zero_s
     << ";\n"
     << ind << "    }\n"
     << ind << "    for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bk * s.bn) << "u; m1_idx += " << s.tg_threads
     << "u) {\n";
  if (tb) {
    os << ind << "      uint m1_r = m1_idx / " << s.bk << "u;\n"
       << ind << "      uint m1_c = m1_idx % " << s.bk << "u;\n"
       << ind << "      uint m1_gc = m1_col0 + m1_r;\n"
       << ind << "      uint m1_gk = m1_k0 + m1_c;\n"
       << ind << "      m1_sb[m1_r * " << s.b_ld
       << "u + m1_c] = (m1_gk < m1_K && m1_gc < m1_N) ? m1_W_e[ulong(m1_gc) * ulong(m1_K) + ulong(m1_gk)] : "
       << zero_s << ";\n";
  } else {
    os << ind << "      uint m1_r = m1_idx / " << s.bn << "u;\n"
       << ind << "      uint m1_c = m1_idx % " << s.bn << "u;\n"
       << ind << "      uint m1_gk = m1_k0 + m1_r;\n"
       << ind << "      uint m1_gc = m1_col0 + m1_c;\n"
       << ind << "      m1_sb[m1_r * " << s.b_ld
       << "u + m1_c] = (m1_gk < m1_K && m1_gc < m1_N) ? m1_W_e[ulong(m1_gk) * ulong(m1_N) + ulong(m1_gc)] : "
       << zero_s << ";\n";
  }
  os << ind << "    }\n"
     << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_mma2_tile_compute(os, s, is_fp16, ind + "    ");
  os << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << ind << "  }\n"
     << ind << "}\n";

  emit_store_out_tile(os, s, "m1_Out_e", "m1_m_e", plan.elem_dt, ind);
}

std::string emit_grouped_gemm_msl(const Kernel& kernel, const GroupedGemmPlan& plan) {
  const auto& s = plan.sched;
  const bool is_fp16 = (plan.elem_dt == ScalarDType::F16);
  const std::string sram_t = is_fp16 ? "half" : "float";

  std::ostringstream os;
  os << "#include <metal_stdlib>\n"
     << "#include <metal_simdgroup_matrix>\n"
     << "using namespace metal;\n\n";
  emit_mma2_helpers(os, is_fp16);
  os << "[[max_total_threads_per_threadgroup(" << s.tg_threads << ")]]\n";
  emit_kernel_param_decls(os, kernel, plan.out_name);
  os << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
     << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]],\n"
     << "    uint m1_lane [[thread_index_in_simdgroup]]) {\n"
     << "  (void)m1_grid_dim;\n"
     << "  long m1_E_l = " << plan.e_expr << ";\n"
     << "  long m1_N_l = " << plan.n_expr << ";\n"
     << "  long m1_K_l = " << plan.k_expr << ";\n"
     << "  if (m1_E_l <= 0L || m1_N_l <= 0L || m1_K_l < 0L) return;\n"
     << "  uint m1_E = uint(m1_E_l);\n"
     << "  uint m1_N = uint(m1_N_l);\n"
     << "  uint m1_K = uint(m1_K_l);\n"
     << "  uint m1_tiles_n = (m1_N + " << (s.bn - 1u) << "u) / " << s.bn << "u;\n"
     << "  threadgroup " << sram_t << " m1_sa[" << (s.bm * s.a_ld) << "];\n"
     << "  threadgroup " << sram_t << " m1_sb[" << s.sb_elems << "];\n"
     << "  uint m1_sg_row = (m1_simd_id / " << s.sg_n << "u) * " << (s.tm * 8u) << "u;\n"
     << "  uint m1_sg_col = (m1_simd_id % " << s.sg_n << "u) * " << (s.tn * 8u) << "u;\n"
     << "  uint m1_qid = m1_lane >> 2u;\n"
     << "  uint m1_fr = (m1_qid & 4u) + ((m1_lane >> 1u) & 3u);\n"
     << "  uint m1_fc = ((m1_qid & 2u) << 1u) + ((m1_lane & 1u) << 1u);\n";

  if (plan.attr.persistent) {
    os << "  threadgroup uint m1_sched[12];\n"
       << "  if (m1_tid_linear == 0u) {\n"
       << "    m1_sched[0] = 0u;\n"
       << "    m1_sched[1] = 0u;\n"
       << "    m1_sched[2] = 0u;\n"
       << "    m1_sched[3] = 0u;\n"
       << "    m1_sched[4] = 0u;\n"
       << "    m1_sched[5] = 0u;\n"
       << "    m1_sched[11] = 0u;\n"
       << "  }\n"
       << "  for (uint m1_tile_idx = m1_tgid.x; ; m1_tile_idx += m1_grid_dim.x) {\n"
       << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "    if (m1_tid_linear == 0u) {\n"
       << "      uint m1_e = m1_sched[0];\n"
       << "      uint m1_e_tile_start = m1_sched[1];\n"
       << "      uint m1_e_tile_end = m1_sched[2];\n"
       << "      uint m1_s_e = m1_sched[3];\n"
       << "      uint m1_m_e = m1_sched[4];\n"
       << "      uint m1_tiles_m = m1_sched[5];\n"
       << "      while (m1_tile_idx >= m1_e_tile_end) {\n"
       << "        if (m1_e >= m1_E) break;\n"
       << "        long m1_off0 = long(" << plan.offsets_name << "[m1_e]);\n"
       << "        long m1_off1 = long(" << plan.offsets_name << "[m1_e + 1u]);\n"
       << "        uint m1_me = (m1_off1 > m1_off0) ? uint(m1_off1 - m1_off0) : 0u;\n"
       << "        uint m1_tm_e = (m1_me + " << (s.bm - 1u) << "u) / " << s.bm << "u;\n"
       << "        uint m1_et = m1_tm_e * m1_tiles_n;\n"
       << "        m1_e_tile_start = m1_e_tile_end;\n"
       << "        m1_e_tile_end += m1_et;\n"
       << "        m1_s_e = (m1_off0 > 0L) ? uint(m1_off0) : 0u;\n"
       << "        m1_m_e = m1_me;\n"
       << "        m1_tiles_m = m1_tm_e;\n"
       << "        m1_e += 1u;\n"
       << "      }\n"
       << "      m1_sched[0] = m1_e;\n"
       << "      m1_sched[1] = m1_e_tile_start;\n"
       << "      m1_sched[2] = m1_e_tile_end;\n"
       << "      m1_sched[3] = m1_s_e;\n"
       << "      m1_sched[4] = m1_m_e;\n"
       << "      m1_sched[5] = m1_tiles_m;\n"
       << "      if (m1_tile_idx >= m1_e_tile_end) {\n"
       << "        m1_sched[11] = 1u;\n"
       << "      } else {\n"
       << "        uint m1_local_tile = m1_tile_idx - m1_e_tile_start;\n"
       << "        uint m1_gw = 8u;\n"
       << "        uint m1_npg = m1_gw * m1_tiles_n;\n"
       << "        uint m1_gid = m1_local_tile / m1_npg;\n"
       << "        uint m1_fm = m1_gid * m1_gw;\n"
       << "        uint m1_gsm = min(m1_tiles_m - m1_fm, m1_gw);\n"
       << "        uint m1_rem = m1_local_tile % m1_npg;\n"
       << "        uint m1_pid_m = m1_fm + (m1_rem % m1_gsm);\n"
       << "        uint m1_pid_n = m1_rem / m1_gsm;\n"
       << "        if (m1_gid & 1u) m1_pid_n = m1_tiles_n - 1u - m1_pid_n;\n"
       << "        m1_sched[6] = m1_e - 1u;\n"
       << "        m1_sched[7] = m1_s_e;\n"
       << "        m1_sched[8] = m1_m_e;\n"
       << "        m1_sched[9] = m1_pid_m * " << s.bm << "u;\n"
       << "        m1_sched[10] = m1_pid_n * " << s.bn << "u;\n"
       << "        m1_sched[11] = 0u;\n"
       << "      }\n"
       << "    }\n"
       << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "    if (m1_sched[11] != 0u) break;\n"
       << "    uint m1_expert = m1_sched[6];\n"
       << "    uint m1_s_e = m1_sched[7];\n"
       << "    uint m1_m_e = m1_sched[8];\n"
       << "    uint m1_row0 = m1_sched[9];\n"
       << "    uint m1_col0 = m1_sched[10];\n";
    emit_grouped_gemm_tile_body(os, plan, "    ");
    os << "  }\n";
  } else {
    os << "  uint m1_expert = 0u;\n"
       << "  uint m1_s_e = 0u;\n"
       << "  uint m1_m_e = 0u;\n"
       << "  uint m1_row0 = 0u;\n"
       << "  uint m1_col0 = 0u;\n"
       << "  if (m1_grid_dim.z > 1u || m1_grid_dim.y > 1u) {\n"
       << "    m1_expert = m1_tgid.z;\n"
       << "    if (m1_expert >= m1_E) return;\n"
       << "    long m1_off0 = long(" << plan.offsets_name << "[m1_expert]);\n"
       << "    long m1_off1 = long(" << plan.offsets_name << "[m1_expert + 1u]);\n"
       << "    if (m1_off1 <= m1_off0) return;\n"
       << "    m1_s_e = (m1_off0 > 0L) ? uint(m1_off0) : 0u;\n"
       << "    m1_m_e = uint(m1_off1 - m1_off0);\n"
       << "    m1_row0 = m1_tgid.y * " << s.bm << "u;\n"
       << "    m1_col0 = m1_tgid.x * " << s.bn << "u;\n"
       << "    if (m1_row0 >= m1_m_e || m1_col0 >= m1_N) return;\n"
       << "  } else {\n"
       << "    uint m1_tile_idx = m1_tgid.x;\n"
       << "    uint m1_e_tile_start = 0u;\n"
       << "    uint m1_e_tile_end = 0u;\n"
       << "    uint m1_tiles_m = 0u;\n"
       << "    bool m1_found = false;\n"
       << "    for (uint m1_e = 0u; m1_e < m1_E; ++m1_e) {\n"
       << "      long m1_off0 = long(" << plan.offsets_name << "[m1_e]);\n"
       << "      long m1_off1 = long(" << plan.offsets_name << "[m1_e + 1u]);\n"
       << "      uint m1_me = (m1_off1 > m1_off0) ? uint(m1_off1 - m1_off0) : 0u;\n"
       << "      uint m1_tm_e = (m1_me + " << (s.bm - 1u) << "u) / " << s.bm << "u;\n"
       << "      uint m1_et = m1_tm_e * m1_tiles_n;\n"
       << "      m1_e_tile_start = m1_e_tile_end;\n"
       << "      m1_e_tile_end += m1_et;\n"
       << "      if (m1_tile_idx < m1_e_tile_end) {\n"
       << "        m1_expert = m1_e;\n"
       << "        m1_s_e = (m1_off0 > 0L) ? uint(m1_off0) : 0u;\n"
       << "        m1_m_e = m1_me;\n"
       << "        m1_tiles_m = m1_tm_e;\n"
       << "        m1_found = true;\n"
       << "        break;\n"
       << "      }\n"
       << "    }\n"
       << "    if (!m1_found) return;\n"
       << "    uint m1_local_tile = m1_tile_idx - m1_e_tile_start;\n"
       << "    uint m1_gw = 8u;\n"
       << "    uint m1_npg = m1_gw * m1_tiles_n;\n"
       << "    uint m1_gid = m1_local_tile / m1_npg;\n"
       << "    uint m1_fm = m1_gid * m1_gw;\n"
       << "    uint m1_gsm = min(m1_tiles_m - m1_fm, m1_gw);\n"
       << "    uint m1_rem = m1_local_tile % m1_npg;\n"
       << "    uint m1_pid_m = m1_fm + (m1_rem % m1_gsm);\n"
       << "    uint m1_pid_n = m1_rem / m1_gsm;\n"
       << "    if (m1_gid & 1u) m1_pid_n = m1_tiles_n - 1u - m1_pid_n;\n"
       << "    m1_row0 = m1_pid_m * " << s.bm << "u;\n"
       << "    m1_col0 = m1_pid_n * " << s.bn << "u;\n"
       << "  }\n";
    emit_grouped_gemm_tile_body(os, plan, "  ");
  }
  os << "}\n";
  return os.str();
}

struct QuantDotPlan {
  const Op* op = nullptr;
  QuantDotAttr attr;
  SgTileLayout sched;
  ScalarDType a_dt = ScalarDType::F16;
  ScalarDType scale_dt = ScalarDType::F16;
  ScalarDType zero_dt = ScalarDType::F16;
  ScalarDType out_dt = ScalarDType::F16;
  std::string a_name;
  std::string qw_name;
  std::string scales_name;
  std::string zeros_name;
  std::string out_name;
  std::string m_expr;
  std::string n_expr;
  std::string k_expr;
};

bool build_quant_dot_plan(const Kernel& kernel, const Op& op, const TargetSpec& target, QuantDotPlan& plan,
                          std::string& err) {
  const auto* qd = std::get_if<QuantDotAttr>(&op.attr);
  if (qd == nullptr) {
    err = "quant_dot op missing QuantDotAttr";
    return false;
  }
  if (op.operands.size() != 8u || !op.results.empty()) {
    err = "quant_dot expects 8 operands (A, QWeight, Scales, Zeros, Out, M, N, K) and 0 results";
    return false;
  }

  plan.op = &op;
  plan.attr = *qd;
  plan.attr.group_size =
      resolve_pos_override(kernel, "GROUP_SIZE", nullptr, qd->group_size > 0u ? qd->group_size : 128u);
  if (plan.attr.bits != 4u) {
    err = "quant_dot currently supports 4-bit quantization (bits=4)";
    return false;
  }
  if (plan.attr.group_size == 0u || (plan.attr.group_size % 8u) != 0u) {
    err = "quant_dot group_size must be a positive multiple of 8";
    return false;
  }

  const auto a_pidx = param_index_of(kernel, op.operands[0]);
  const auto qw_pidx = param_index_of(kernel, op.operands[1]);
  const auto sc_pidx = param_index_of(kernel, op.operands[2]);
  const auto zp_pidx = param_index_of(kernel, op.operands[3]);
  const auto out_pidx = param_index_of(kernel, op.operands[4]);
  if (!a_pidx || !qw_pidx || !sc_pidx || !zp_pidx || !out_pidx) {
    err = "quant_dot pointer operands 0..4 must be kernel pointer parameters";
    return false;
  }
  if (!kernel.params[*a_pidx].type.is_ptr() || !kernel.params[*qw_pidx].type.is_ptr() ||
      !kernel.params[*sc_pidx].type.is_ptr() || !kernel.params[*zp_pidx].type.is_ptr() ||
      !kernel.params[*out_pidx].type.is_ptr()) {
    err = "quant_dot operands 0..4 must be device pointers";
    return false;
  }
  if (*out_pidx == *a_pidx || *out_pidx == *qw_pidx || *out_pidx == *sc_pidx || *out_pidx == *zp_pidx) {
    err = "quant_dot Out pointer must not alias input pointers";
    return false;
  }

  const ScalarDType a_dt = std::get<ir::PtrType>(kernel.params[*a_pidx].type.value).pointee;
  const ScalarDType qw_dt = std::get<ir::PtrType>(kernel.params[*qw_pidx].type.value).pointee;
  const ScalarDType sc_dt = std::get<ir::PtrType>(kernel.params[*sc_pidx].type.value).pointee;
  const ScalarDType zp_dt = std::get<ir::PtrType>(kernel.params[*zp_pidx].type.value).pointee;
  const ScalarDType out_dt = std::get<ir::PtrType>(kernel.params[*out_pidx].type.value).pointee;
  if ((a_dt != ScalarDType::F16 && a_dt != ScalarDType::F32) ||
      (sc_dt != ScalarDType::F16 && sc_dt != ScalarDType::F32) ||
      (zp_dt != ScalarDType::F16 && zp_dt != ScalarDType::F32) ||
      (out_dt != ScalarDType::F16 && out_dt != ScalarDType::F32)) {
    err = "quant_dot requires F16 or F32 pointers for A, Scales, Zeros, and Out";
    return false;
  }
  if (qw_dt != ScalarDType::I32 && qw_dt != ScalarDType::U32) {
    err = "quant_dot QWeight pointer must be I32 or U32";
    return false;
  }

  auto m_expr = scalar_long_expr(kernel, op.operands[5]);
  auto n_expr = scalar_long_expr(kernel, op.operands[6]);
  auto k_expr = scalar_long_expr(kernel, op.operands[7]);
  if (!m_expr || !n_expr || !k_expr) {
    err = "quant_dot M, N, K operands must be scalar parameters or integer constants";
    return false;
  }

  const uint32_t bm = resolve_pos_override(kernel, "BLOCK_M", "BM", qd->bm > 0u ? qd->bm : 64u);
  const uint32_t bn = resolve_pos_override(kernel, "BLOCK_N", "BN", qd->bn > 0u ? qd->bn : 64u);
  const uint32_t bk = resolve_pos_override(kernel, "BLOCK_K", "BK", qd->bk > 0u ? qd->bk : 32u);
  const uint32_t num_sg =
      resolve_pos_override(kernel, "NUM_SG", "NUM_SIMDGROUPS", qd->num_simdgroups > 0u ? qd->num_simdgroups : 4u);
  if (auto sym_ovr = find_constexpr_i64(kernel, "SYMMETRIC")) {
    plan.attr.symmetric = (*sym_ovr != 0);
  }

  const uint32_t sram_bytes = (a_dt == ScalarDType::F16) ? 2u : 4u;
  if (!compute_sg_tile_layout(bm, bn, bk, num_sg, sram_bytes, false, 0u, target, plan.sched, err)) {
    return false;
  }

  plan.a_dt = a_dt;
  plan.scale_dt = sc_dt;
  plan.zero_dt = zp_dt;
  plan.out_dt = out_dt;
  plan.a_name = kernel.params[*a_pidx].name;
  plan.qw_name = kernel.params[*qw_pidx].name;
  plan.scales_name = kernel.params[*sc_pidx].name;
  plan.zeros_name = kernel.params[*zp_pidx].name;
  plan.out_name = kernel.params[*out_pidx].name;
  plan.m_expr = *m_expr;
  plan.n_expr = *n_expr;
  plan.k_expr = *k_expr;
  return true;
}

void emit_quant_load_b_regs(std::ostringstream& os, const QuantDotPlan& plan, const std::string& k_expr,
                            const std::string& prefix, const std::string& ind) {
  const auto& s = plan.sched;
  const bool dequant_f16 = (plan.a_dt == ScalarDType::F16);
  const std::string dq2 = dequant_f16 ? "half2" : "float2";
  const std::string dq_zero = dequant_f16 ? "half2(0.0h)" : "float2(0.0f)";
  const std::string sym_z = dequant_f16 ? "half2(8.0h)" : "float2(8.0f)";
  const uint32_t k_words = s.bk / 8u;
  const uint32_t n_pairs = s.bn / 2u;
  const uint32_t total_pairs = k_words * n_pairs;
  const uint32_t per = (total_pairs + s.tg_threads - 1u) / s.tg_threads;
  const bool same_group = (plan.attr.group_size % s.bk) == 0u;
  const std::string sc_p2 = (plan.scale_dt == ScalarDType::F16) ? "packed_half2" : "packed_float2";
  const std::string zp_p2 = (plan.zero_dt == ScalarDType::F16) ? "packed_half2" : "packed_float2";

  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string rw = "(" + vi + " / " + std::to_string(n_pairs) + "u)";
    const std::string c2 = "((" + vi + " % " + std::to_string(n_pairs) + "u) * 2u)";
    const std::string g_expr =
        same_group ? ("((" + k_expr + ") / " + std::to_string(plan.attr.group_size) + "u)")
                   : ("((" + k_expr + " + " + rw + " * 8u) / " + std::to_string(plan.attr.group_size) + "u)");
    const std::string qw_off =
        "ulong(((" + k_expr + ") >> 3u) + " + rw + ") * ulong(m1_N) + ulong(m1_col0 + " + c2 + ")";
    const std::string g_off = "ulong(" + g_expr + ") * ulong(m1_N) + ulong(m1_col0 + " + c2 + ")";
    const bool guarded = ((p + 1u) * s.tg_threads > total_pairs);

    if (guarded) {
      os << ind << "uint2 " << prefix << "_qw_" << p << " = uint2(0u);\n"
         << ind << dq2 << " " << prefix << "_sc_" << p << " = " << dq_zero << ";\n"
         << ind << dq2 << " " << prefix << "_zp_" << p << " = " << dq_zero << ";\n"
         << ind << "if (" << vi << " < " << total_pairs << "u) {\n"
         << ind << "  " << prefix << "_qw_" << p << " = uint2(*(const device packed_uint2*)(m1_qw_u32 + " << qw_off
         << "));\n"
         << ind << "  " << prefix << "_sc_" << p << " = " << dq2 << "(*(const device " << sc_p2 << "*)("
         << plan.scales_name << " + " << g_off << "));\n";
      if (plan.attr.symmetric) {
        os << ind << "  " << prefix << "_zp_" << p << " = " << sym_z << ";\n";
      } else {
        os << ind << "  " << prefix << "_zp_" << p << " = " << dq2 << "(*(const device " << zp_p2 << "*)("
           << plan.zeros_name << " + " << g_off << "));\n";
      }
      os << ind << "}\n";
    } else {
      os << ind << "uint2 " << prefix << "_qw_" << p << " = uint2(*(const device packed_uint2*)(m1_qw_u32 + " << qw_off
         << "));\n"
         << ind << dq2 << " " << prefix << "_sc_" << p << " = " << dq2 << "(*(const device " << sc_p2 << "*)("
         << plan.scales_name << " + " << g_off << "));\n";
      if (plan.attr.symmetric) {
        os << ind << dq2 << " " << prefix << "_zp_" << p << " = " << sym_z << ";\n";
      } else {
        os << ind << dq2 << " " << prefix << "_zp_" << p << " = " << dq2 << "(*(const device " << zp_p2 << "*)("
           << plan.zeros_name << " + " << g_off << "));\n";
      }
    }
  }
}

void emit_quant_unpack_store_b(std::ostringstream& os, const QuantDotPlan& plan, const std::string& prefix,
                               const std::string& ind) {
  const auto& s = plan.sched;
  const bool is_fp16 = (plan.a_dt == ScalarDType::F16);
  const std::string v2 = is_fp16 ? "half2" : "float2";
  const uint32_t k_words = s.bk / 8u;
  const uint32_t n_pairs = s.bn / 2u;
  const uint32_t total_pairs = k_words * n_pairs;
  const uint32_t per = (total_pairs + s.tg_threads - 1u) / s.tg_threads;

  for (uint32_t p = 0u; p < per; ++p) {
    const std::string vi = "(m1_tid_linear + " + std::to_string(p * s.tg_threads) + "u)";
    const std::string rw = "(" + vi + " / " + std::to_string(n_pairs) + "u)";
    const std::string c2 = "((" + vi + " % " + std::to_string(n_pairs) + "u) * 2u)";
    const bool guarded = ((p + 1u) * s.tg_threads > total_pairs);
    if (guarded) {
      os << ind << "if (" << vi << " < " << total_pairs << "u) {\n";
    }
    const std::string sub = guarded ? (ind + "  ") : ind;
    os << sub << "{\n"
       << sub << "  uint m1_r0 = " << rw << " * 8u;\n"
       << sub << "  uint m1_c2 = " << c2 << ";\n"
       << sub << "  #pragma unroll\n"
       << sub << "  for (uint m1_j = 0u; m1_j < 8u; ++m1_j) {\n"
       << sub << "    " << v2 << " m1_q = " << v2 << "((" << prefix << "_qw_" << p << ".x >> (4u * m1_j)) & 0xFu, ("
       << prefix << "_qw_" << p << ".y >> (4u * m1_j)) & 0xFu);\n"
       << sub << "    *(threadgroup " << v2 << "*)(m1_sb + (m1_r0 + m1_j) * " << s.b_ld << "u + m1_c2) = (m1_q - "
       << prefix << "_zp_" << p << ") * " << prefix << "_sc_" << p << ";\n"
       << sub << "  }\n"
       << sub << "}\n";
    if (guarded) {
      os << ind << "}\n";
    }
  }
}

std::string emit_quant_dot_msl(const Kernel& kernel, const QuantDotPlan& plan) {
  const auto& s = plan.sched;
  const bool is_fp16 = (plan.a_dt == ScalarDType::F16);
  const std::string sram_t = is_fp16 ? "half" : "float";
  const std::string zero_s = is_fp16 ? "0.0h" : "0.0f";

  const uint32_t a_per = ((s.bm * s.bk) / 4u + s.tg_threads - 1u) / s.tg_threads;
  const uint32_t b_per = (((s.bk / 8u) * (s.bn / 2u)) + s.tg_threads - 1u) / s.tg_threads;
  const bool use_reg_prefetch = (s.tm * s.tn <= 16u) && (a_per + b_per * 2u <= 10u);

  std::ostringstream os;
  os << "#include <metal_stdlib>\n"
     << "#include <metal_simdgroup_matrix>\n"
     << "using namespace metal;\n\n";
  emit_mma2_helpers(os, is_fp16);
  os << "[[max_total_threads_per_threadgroup(" << s.tg_threads << ")]]\n";
  emit_kernel_param_decls(os, kernel, plan.out_name);
  os << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
     << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]],\n"
     << "    uint m1_lane [[thread_index_in_simdgroup]]) {\n"
     << "  long m1_M_l = " << plan.m_expr << ";\n"
     << "  long m1_N_l = " << plan.n_expr << ";\n"
     << "  long m1_K_l = " << plan.k_expr << ";\n"
     << "  if (m1_M_l <= 0L || m1_N_l <= 0L || m1_K_l < 0L) return;\n"
     << "  uint m1_M = uint(m1_M_l);\n"
     << "  uint m1_N = uint(m1_N_l);\n"
     << "  uint m1_K = uint(m1_K_l);\n"
     << "  uint m1_tiles_m = (m1_M + " << (s.bm - 1u) << "u) / " << s.bm << "u;\n"
     << "  uint m1_tiles_n = (m1_N + " << (s.bn - 1u) << "u) / " << s.bn << "u;\n"
     << "  uint m1_pid_m;\n"
     << "  uint m1_pid_n;\n"
     << "  if (m1_grid_dim.y == 1u) {\n"
     << "    uint m1_pid = m1_tgid.x;\n"
     << "    if (m1_pid >= m1_tiles_m * m1_tiles_n) return;\n"
     << "    uint m1_gw = 8u;\n"
     << "    uint m1_npg = m1_gw * m1_tiles_n;\n"
     << "    uint m1_gid = m1_pid / m1_npg;\n"
     << "    uint m1_fm = m1_gid * m1_gw;\n"
     << "    uint m1_gsm = min(m1_tiles_m - m1_fm, m1_gw);\n"
     << "    uint m1_rem = m1_pid % m1_npg;\n"
     << "    m1_pid_m = m1_fm + (m1_rem % m1_gsm);\n"
     << "    m1_pid_n = m1_rem / m1_gsm;\n"
     << "    if (m1_gid & 1u) m1_pid_n = m1_tiles_n - 1u - m1_pid_n;\n"
     << "  } else {\n"
     << "    if (m1_tgid.x >= m1_tiles_n || m1_tgid.y >= m1_tiles_m) return;\n"
     << "    uint m1_pid = m1_tgid.x * m1_tiles_m + m1_tgid.y;\n"
     << "    uint m1_gw = 8u;\n"
     << "    uint m1_npg = m1_gw * m1_tiles_n;\n"
     << "    uint m1_gid = m1_pid / m1_npg;\n"
     << "    uint m1_fm = m1_gid * m1_gw;\n"
     << "    uint m1_gsm = min(m1_tiles_m - m1_fm, m1_gw);\n"
     << "    uint m1_rem = m1_pid % m1_npg;\n"
     << "    m1_pid_m = m1_fm + (m1_rem % m1_gsm);\n"
     << "    m1_pid_n = m1_rem / m1_gsm;\n"
     << "    if (m1_gid & 1u) m1_pid_n = m1_tiles_n - 1u - m1_pid_n;\n"
     << "  }\n"
     << "  uint m1_row0 = m1_pid_m * " << s.bm << "u;\n"
     << "  uint m1_col0 = m1_pid_n * " << s.bn << "u;\n"
     << "  device const uint* m1_qw_u32 = reinterpret_cast<device const uint*>(" << plan.qw_name << ");\n"
     << "  threadgroup " << sram_t << " m1_sa[" << (s.bm * s.a_ld) << "];\n"
     << "  threadgroup " << sram_t << " m1_sb[" << s.sb_elems << "];\n"
     << "  uint m1_sg_row = (m1_simd_id / " << s.sg_n << "u) * " << (s.tm * 8u) << "u;\n"
     << "  uint m1_sg_col = (m1_simd_id % " << s.sg_n << "u) * " << (s.tn * 8u) << "u;\n"
     << "  uint m1_qid = m1_lane >> 2u;\n"
     << "  uint m1_fr = (m1_qid & 4u) + ((m1_lane >> 1u) & 3u);\n"
     << "  uint m1_fc = ((m1_qid & 2u) << 1u) + ((m1_lane & 1u) << 1u);\n";

  for (uint32_t i = 0u; i < s.tm; ++i) {
    for (uint32_t j = 0u; j < s.tn; ++j) {
      os << "  float2 m1_c_" << i << "_" << j << " = float2(0.0f);\n";
    }
  }

  os << "  bool m1_full_m = (m1_row0 + " << s.bm << "u <= m1_M);\n"
     << "  bool m1_full_n = (m1_col0 + " << s.bn << "u <= m1_N);\n"
     << "  uint m1_k0 = 0u;\n"
     << "  if (m1_full_n) {\n"
     << "    uint m1_k_full_end = m1_K & ~" << (s.bk - 1u) << "u;\n"
     << "    if (m1_full_m) {\n";

  if (use_reg_prefetch) {
    os << "      if (m1_k0 < m1_k_full_end) {\n"
       << "        {\n";
    emit_vec4_load_a(os, s, plan.a_name, "m1_k0", "m1_pa", is_fp16, "", "          ");
    emit_quant_load_b_regs(os, plan, "m1_k0", "m1_pb", "          ");
    emit_vec4_store_a(os, s, "m1_pa", is_fp16, "          ");
    emit_quant_unpack_store_b(os, plan, "m1_pb", "          ");
    os << "        }\n"
       << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "        for (m1_k0 += " << s.bk << "u; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_vec4_load_a(os, s, plan.a_name, "m1_k0", "m1_ra", is_fp16, "", "          ");
    emit_quant_load_b_regs(os, plan, "m1_k0", "m1_rb", "          ");
    emit_mma2_tile_compute(os, s, is_fp16, "          ");
    os << "          threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_vec4_store_a(os, s, "m1_ra", is_fp16, "          ");
    emit_quant_unpack_store_b(os, plan, "m1_rb", "          ");
    os << "          threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "        }\n";
    emit_mma2_tile_compute(os, s, is_fp16, "        ");
    os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "      }\n";
  } else {
    os << "      for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_vec4_load_a(os, s, plan.a_name, "m1_k0", "m1_ra", is_fp16, "", "        ");
    emit_quant_load_b_regs(os, plan, "m1_k0", "m1_rb", "        ");
    emit_vec4_store_a(os, s, "m1_ra", is_fp16, "        ");
    emit_quant_unpack_store_b(os, plan, "m1_rb", "        ");
    os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma2_tile_compute(os, s, is_fp16, "        ");
    os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "      }\n";
  }
  os << "    } else {\n"
     << "      for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
  emit_vec4_load_a(os, s, plan.a_name, "m1_k0", "m1_ra", is_fp16, "m1_M", "        ");
  emit_quant_load_b_regs(os, plan, "m1_k0", "m1_rb", "        ");
  emit_vec4_store_a(os, s, "m1_ra", is_fp16, "        ");
  emit_quant_unpack_store_b(os, plan, "m1_rb", "        ");
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_mma2_tile_compute(os, s, is_fp16, "        ");
  os << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "      }\n"
     << "    }\n"
     << "  }\n";

  const uint32_t k_words = s.bk / 8u;
  os << "  if (m1_k0 < m1_K) {\n"
     << "    for (; m1_k0 < m1_K; m1_k0 += " << s.bk << "u) {\n"
     << "      for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bm * s.bk) << "u; m1_idx += " << s.tg_threads
     << "u) {\n"
     << "        uint m1_r = m1_idx / " << s.bk << "u;\n"
     << "        uint m1_c = m1_idx % " << s.bk << "u;\n"
     << "        uint m1_gr = m1_row0 + m1_r;\n"
     << "        uint m1_gk = m1_k0 + m1_c;\n"
     << "        m1_sa[m1_r * " << s.a_ld << "u + m1_c] = (m1_gr < m1_M && m1_gk < m1_K) ? " << plan.a_name
     << "[ulong(m1_gr) * ulong(m1_K) + ulong(m1_gk)] : " << zero_s << ";\n"
     << "      }\n"
     << "      for (uint m1_idx = m1_tid_linear; m1_idx < " << (k_words * s.bn) << "u; m1_idx += " << s.tg_threads
     << "u) {\n"
     << "        uint m1_rw = m1_idx / " << s.bn << "u;\n"
     << "        uint m1_c = m1_idx % " << s.bn << "u;\n"
     << "        uint m1_gc = m1_col0 + m1_c;\n"
     << "        uint m1_kb = m1_k0 + m1_rw * 8u;\n"
     << "        if (m1_gc < m1_N && m1_kb < m1_K) {\n"
     << "          uint m1_qw = m1_qw_u32[ulong((m1_k0 >> 3u) + m1_rw) * ulong(m1_N) + ulong(m1_gc)];\n"
     << "          uint m1_g = m1_kb / " << plan.attr.group_size << "u;\n"
     << "          float m1_sc = float(" << plan.scales_name << "[ulong(m1_g) * ulong(m1_N) + ulong(m1_gc)]);\n";
  if (plan.attr.symmetric) {
    os << "          float m1_zp = 8.0f;\n";
  } else {
    os << "          float m1_zp = float(" << plan.zeros_name << "[ulong(m1_g) * ulong(m1_N) + ulong(m1_gc)]);\n";
  }
  os << "          #pragma unroll\n"
     << "          for (uint m1_j = 0u; m1_j < 8u; ++m1_j) {\n"
     << "            float m1_w = (m1_kb + m1_j < m1_K) ? ((float((m1_qw >> (4u * m1_j)) & 0xFu) - m1_zp) * m1_sc) : "
        "0.0f;\n"
     << "            m1_sb[(m1_rw * 8u + m1_j) * " << s.b_ld << "u + m1_c] = " << sram_t << "(m1_w);\n"
     << "          }\n"
     << "        } else {\n"
     << "          #pragma unroll\n"
     << "          for (uint m1_j = 0u; m1_j < 8u; ++m1_j) {\n"
     << "            m1_sb[(m1_rw * 8u + m1_j) * " << s.b_ld << "u + m1_c] = " << zero_s << ";\n"
     << "          }\n"
     << "        }\n"
     << "      }\n"
     << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_mma2_tile_compute(os, s, is_fp16, "      ");
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "    }\n"
     << "  }\n";

  emit_store_out_tile(os, s, plan.out_name, "m1_M", plan.out_dt, "  ");
  os << "}\n";
  return os.str();
}

}  // namespace

CompiledMSL try_emit_l3_gemm_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched) {
  if (matched != nullptr) *matched = false;

  const Op* target_op = nullptr;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::GroupedGemm:
      case OpKind::QuantDot:
        if (target_op != nullptr) {
          if (matched != nullptr) *matched = true;
          return CompiledMSL{false, "", {}, "l3_gemm_emitter supports at most one GroupedGemm or QuantDot op per kernel"};
        }
        target_op = &op;
        break;
      case OpKind::Constant:
      case OpKind::Cast:
      case OpKind::Return:
        break;
      default:
        return CompiledMSL{false, "", {}, ""};
    }
  }
  if (target_op == nullptr) {
    return CompiledMSL{false, "", {}, ""};
  }
  if (matched != nullptr) *matched = true;

  std::string err;
  CompiledMSL res;
  if (target_op->kind == OpKind::GroupedGemm) {
    GroupedGemmPlan plan;
    if (!build_grouped_gemm_plan(kernel, *target_op, target, plan, err)) {
      return CompiledMSL{false, "", {}, err};
    }
    res.ok = true;
    res.msl_source = emit_grouped_gemm_msl(kernel, plan);
    res.launch_spec.entry_name = kernel.name;
    res.launch_spec.tg_x = plan.sched.tg_threads;
    res.launch_spec.tg_y = 1;
    res.launch_spec.tg_z = 1;
    res.launch_spec.threadgroup_mem_bytes = 0;
    res.launch_spec.vec_width = 4;
    res.launch_spec.fast_math = false;
    res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan.out_name);
    return res;
  }

  QuantDotPlan plan;
  if (!build_quant_dot_plan(kernel, *target_op, target, plan, err)) {
    return CompiledMSL{false, "", {}, err};
  }
  res.ok = true;
  res.msl_source = emit_quant_dot_msl(kernel, plan);
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan.sched.tg_threads;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan.out_name);
  return res;
}

}  // namespace m1triton::codegen::metal

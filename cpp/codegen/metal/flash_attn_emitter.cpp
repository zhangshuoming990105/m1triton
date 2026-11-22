#include "codegen/metal/flash_attn_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::FlashAttnAttr;
using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::ScalarDType;

enum class FwdSchedule {
  TwoWaveSgmm,
  SinglePassSgmm,
  MultiTileSgmm,
  SplitMSgmm,
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

std::optional<std::string> find_scalar_param_long_expr(const Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.name != name || kp.type.is_ptr()) continue;
    if (kp.is_constexpr) {
      if (kp.constexpr_val > 0) return std::to_string(kp.constexpr_val) + "L";
      return std::nullopt;
    }
    return "((long)" + kp.name + ")";
  }
  return std::nullopt;
}

struct FlashAttnPlan {
  const Op* op = nullptr;
  FlashAttnAttr attr;
  ir::FlashAttnMode ir_mode = ir::FlashAttnMode::Fwd;
  std::vector<ParamSlot> param_slots;

  ScalarDType elem_dt = ScalarDType::F32;
  std::string elem_type = "float";

  // Forward / Backward tensor names and shape expressions
  std::string q_name;
  std::string k_name;
  std::string v_name;
  std::string out_name;
  std::string lse_name;
  std::string alibi_name;
  std::string do_name;
  std::string delta_name;
  std::string dk_name;
  std::string dv_name;
  std::string dq_name;
  std::string bh_expr;
  std::string lq_expr;
  std::string lkv_expr;
  std::string d_expr;
  std::string hq_expr;
  std::string alibi_heads_expr;
  bool is_v1_plain = false;

  // PagedDecode tensor names and shape expressions
  std::string block_tables_name;
  std::string context_lens_name;
  std::string partial_o_name;
  std::string partial_lse_name;
  std::string b_expr;
  std::string hkv_expr;
  std::string max_blocks_expr;
  uint32_t paged_block_size = 16;
  uint32_t paged_num_splits = 0;

  uint32_t bm = 16;
  uint32_t bn = 256;
  uint32_t num_sg = 16;
  uint32_t tg_threads = 512;
  uint32_t max_smem = 32768;
  uint32_t general_tile_n = 256;
  bool fast_only = false;
  uint32_t d_static = 0;
  uint32_t lkv_static = 0;
  FwdSchedule fwd_sched = FwdSchedule::GeneralRowWise;
};

bool expect_ptr_param(const Kernel& kernel, ir::ValueId vid, std::string& out_name, ScalarDType& out_dt,
                      uint32_t& out_pidx) {
  const auto pidx = param_index_of(kernel, vid);
  if (!pidx || !kernel.params[*pidx].type.is_ptr()) return false;
  out_pidx = *pidx;
  out_name = kernel.params[*pidx].name;
  out_dt = std::get<ir::PtrType>(kernel.params[*pidx].type.value).pointee;
  return true;
}

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
  if (!plan.op->results.empty()) {
    err = "flash_attention_tile expects 0 results";
    return std::nullopt;
  }
  if (!std::isfinite(fa->scale)) {
    err = "flash_attention_tile requires finite scale";
    return std::nullopt;
  }
  if (fa->kv_group_size < 1) {
    err = "flash_attention_tile requires kv_group_size >= 1";
    return std::nullopt;
  }

  plan.attr = *fa;
  plan.ir_mode = fa->mode;
  const uint32_t simd = std::max<uint32_t>(1u, target.simd_width);
  plan.max_smem = target.max_threadgroup_memory_bytes;
  const auto& ops = plan.op->operands;
  std::unordered_set<uint32_t> written_params;

  auto finalize_slots = [&]() {
    std::unordered_map<uint32_t, std::string> param_msl_name;
    build_param_slots(kernel, written_params, plan.param_slots, param_msl_name);
  };

  if (fa->mode == ir::FlashAttnMode::Fwd) {
    const size_t expected =
        8u + (fa->return_lse ? 1u : 0u) + (fa->has_alibi ? 1u : 0u) + (fa->kv_group_size > 1 ? 1u : 0u);
    if (ops.size() != expected) {
      err = "flash_attention_tile fwd expects " + std::to_string(expected) + " operands";
      return std::nullopt;
    }

    std::string* ptr_names[4] = {&plan.q_name, &plan.k_name, &plan.v_name, &plan.out_name};
    for (size_t i = 0; i < 4; ++i) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[i], *ptr_names[i], pt, pidx)) {
        err = "flash_attention_tile pointer operands 0..3 must be kernel pointer parameters";
        return std::nullopt;
      }
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
      if (i == 3) written_params.insert(pidx);
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

    size_t ext_idx = 8u;
    if (fa->return_lse) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[ext_idx++], plan.lse_name, pt, pidx) || pt != ScalarDType::F32) {
        err = "flash_attention_tile lse_ptr operand must be an f32 pointer parameter";
        return std::nullopt;
      }
      written_params.insert(pidx);
    }
    if (fa->has_alibi) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[ext_idx++], plan.alibi_name, pt, pidx) ||
          (pt != ScalarDType::F32 && pt != ScalarDType::F16)) {
        err = "flash_attention_tile alibi_ptr operand must be an f32/f16 pointer parameter";
        return std::nullopt;
      }
    }
    if (fa->kv_group_size > 1) {
      auto e = scalar_long_expr(kernel, ops[ext_idx++]);
      if (!e) {
        err = "flash_attention_tile num_q_heads operand must be a scalar parameter or integer constant";
        return std::nullopt;
      }
      plan.hq_expr = *e;
    }

    if (fa->has_alibi) {
      if (fa->kv_group_size > 1) {
        plan.alibi_heads_expr = plan.hq_expr;
      } else if (auto ah = find_scalar_param_long_expr(kernel, "ALIBI_HEADS")) {
        plan.alibi_heads_expr = *ah;
      } else if (auto hq = find_scalar_param_long_expr(kernel, "H_q")) {
        plan.alibi_heads_expr = *hq;
      } else if (auto nh = find_scalar_param_long_expr(kernel, "NUM_HEADS")) {
        plan.alibi_heads_expr = *nh;
      } else {
        plan.alibi_heads_expr = "m1_BH";
      }
    }

    plan.is_v1_plain = (!fa->return_lse && !fa->has_alibi && fa->window_size < 0 && fa->softcap <= 0.0f &&
                        fa->kv_group_size == 1);

    plan.bm = fa->bm > 0u ? fa->bm : 16u;
    plan.bn = fa->bn > 0u ? fa->bn : 256u;
    plan.num_sg = fa->num_simdgroups > 0u ? fa->num_simdgroups : 16u;
    plan.tg_threads = plan.num_sg * simd;

    if (plan.tg_threads > target.max_threads_per_threadgroup) {
      err = "flash_attention_tile threadgroup exceeds max_threads_per_threadgroup";
      return std::nullopt;
    }

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

    const uint32_t max_smem = plan.max_smem;
    const bool split_m_ok =
        (plan.elem_dt == ScalarDType::F32 && plan.num_sg <= 8u && plan.bm >= 8u * plan.num_sg &&
         (plan.bm % (8u * plan.num_sg)) == 0u && (plan.bm / (8u * plan.num_sg)) <= 2u && plan.bn >= 32u &&
         (plan.bn % 32u) == 0u && plan.bm * 32u * 4u <= max_smem);

    if (plan.is_v1_plain && plan.elem_dt == ScalarDType::F32 && plan.bm == 32u && plan.bn == 512u &&
        plan.num_sg == 16u && 16u * plan.bn * 4u <= max_smem) {
      plan.fwd_sched = FwdSchedule::TwoWaveSgmm;
    } else if (plan.is_v1_plain && plan.elem_dt == ScalarDType::F32 && plan.bm == 16u && plan.bn == 512u &&
               plan.num_sg == 16u && plan.bm * plan.bn * 4u <= max_smem) {
      plan.fwd_sched = FwdSchedule::SinglePassSgmm;
    } else if (plan.elem_dt == ScalarDType::F32 && plan.bm == 16u && plan.num_sg == 16u &&
               (plan.bn == 256u || plan.bn == 128u || plan.bn == 64u) &&
               (plan.bm * (plan.bn + 4u) + 3u * plan.bm) * 4u <= max_smem) {
      plan.fwd_sched = FwdSchedule::MultiTileSgmm;
    } else if (split_m_ok) {
      plan.fwd_sched = FwdSchedule::SplitMSgmm;
      plan.general_tile_n = 32u;
    } else {
      uint32_t tn = std::min<uint32_t>(plan.bn, 256u);
      while (plan.bm * tn * 4u > max_smem && tn > 32u) {
        tn /= 2u;
      }
      if (plan.bm * tn * 4u > max_smem) {
        err = "flash_attention_tile exceeds target threadgroup memory budget";
        return std::nullopt;
      }
      plan.general_tile_n = tn;
      plan.fwd_sched = FwdSchedule::GeneralRowWise;
    }

    finalize_slots();
    return plan;
  }

  if (fa->mode == ir::FlashAttnMode::BwdDKDV) {
    if (ops.size() != 12u) {
      err = "flash_attention_bwd_dkdv expects 12 operands";
      return std::nullopt;
    }
    std::string* ptrs[8] = {&plan.q_name,   &plan.k_name,     &plan.v_name,  &plan.do_name,
                            &plan.lse_name, &plan.delta_name, &plan.dk_name, &plan.dv_name};
    for (size_t i = 0; i < 8u; ++i) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[i], *ptrs[i], pt, pidx)) {
        err = "flash_attention_bwd_dkdv operands 0..7 must be kernel pointer parameters";
        return std::nullopt;
      }
      if (i == 4u || i == 5u) {
        if (pt != ScalarDType::F32) {
          err = "flash_attention_bwd_dkdv L and Delta pointers must be f32";
          return std::nullopt;
        }
      } else {
        if (pt != ScalarDType::F32 && pt != ScalarDType::F16) {
          err = "flash_attention_bwd_dkdv supports f32 or f16 pointers";
          return std::nullopt;
        }
        if (i == 0u) {
          plan.elem_dt = pt;
          plan.elem_type = msl_scalar_type(pt);
        } else if (pt != plan.elem_dt) {
          err = "flash_attention_bwd_dkdv requires matching Q/K/V/dO/dK/dV pointer dtypes";
          return std::nullopt;
        }
      }
      if (i == 6u || i == 7u) written_params.insert(pidx);
    }

    std::string* shapes[4] = {&plan.bh_expr, &plan.lq_expr, &plan.lkv_expr, &plan.d_expr};
    for (size_t i = 0; i < 4u; ++i) {
      auto e = scalar_long_expr(kernel, ops[8u + i]);
      if (!e) {
        err = "flash_attention_bwd_dkdv shape operands must be scalars or integer constants";
        return std::nullopt;
      }
      *shapes[i] = *e;
    }

    const bool has_num_sg = find_constexpr_param(kernel, "NUM_SG").has_value();
    const bool has_bm = find_constexpr_param(kernel, "BLOCK_M").has_value();
    const bool has_bn = find_constexpr_param(kernel, "BLOCK_N").has_value();
    plan.bm = has_bm ? fa->bm : 32u;
    plan.bn = has_bn ? fa->bn : 32u;
    plan.num_sg = has_num_sg ? fa->num_simdgroups : 4u;
    plan.num_sg = std::max<uint32_t>(1u, std::min<uint32_t>(plan.num_sg, target.max_threads_per_threadgroup / simd));
    plan.tg_threads = plan.num_sg * simd;
    if (plan.tg_threads > target.max_threads_per_threadgroup) {
      err = "flash_attention_bwd_dkdv threadgroup exceeds max_threads_per_threadgroup";
      return std::nullopt;
    }
    finalize_slots();
    return plan;
  }

  if (fa->mode == ir::FlashAttnMode::BwdDQ) {
    if (ops.size() != 11u) {
      err = "flash_attention_bwd_dq expects 11 operands";
      return std::nullopt;
    }
    std::string* ptrs[7] = {&plan.q_name,   &plan.k_name,     &plan.v_name, &plan.do_name,
                            &plan.lse_name, &plan.delta_name, &plan.dq_name};
    for (size_t i = 0; i < 7u; ++i) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[i], *ptrs[i], pt, pidx)) {
        err = "flash_attention_bwd_dq operands 0..6 must be kernel pointer parameters";
        return std::nullopt;
      }
      if (i == 4u || i == 5u) {
        if (pt != ScalarDType::F32) {
          err = "flash_attention_bwd_dq L and Delta pointers must be f32";
          return std::nullopt;
        }
      } else {
        if (pt != ScalarDType::F32 && pt != ScalarDType::F16) {
          err = "flash_attention_bwd_dq supports f32 or f16 pointers";
          return std::nullopt;
        }
        if (i == 0u) {
          plan.elem_dt = pt;
          plan.elem_type = msl_scalar_type(pt);
        } else if (pt != plan.elem_dt) {
          err = "flash_attention_bwd_dq requires matching Q/K/V/dO/dQ pointer dtypes";
          return std::nullopt;
        }
      }
      if (i == 6u) written_params.insert(pidx);
    }

    std::string* shapes[4] = {&plan.bh_expr, &plan.lq_expr, &plan.lkv_expr, &plan.d_expr};
    for (size_t i = 0; i < 4u; ++i) {
      auto e = scalar_long_expr(kernel, ops[7u + i]);
      if (!e) {
        err = "flash_attention_bwd_dq shape operands must be scalars or integer constants";
        return std::nullopt;
      }
      *shapes[i] = *e;
    }

    const bool has_num_sg = find_constexpr_param(kernel, "NUM_SG").has_value();
    const bool has_bm = find_constexpr_param(kernel, "BLOCK_M").has_value();
    const bool has_bn = find_constexpr_param(kernel, "BLOCK_N").has_value();
    plan.bm = has_bm ? fa->bm : 32u;
    plan.bn = has_bn ? fa->bn : 32u;
    plan.num_sg = has_num_sg ? fa->num_simdgroups : 4u;
    plan.num_sg = std::max<uint32_t>(1u, std::min<uint32_t>(plan.num_sg, target.max_threads_per_threadgroup / simd));
    plan.tg_threads = plan.num_sg * simd;
    if (plan.tg_threads > target.max_threads_per_threadgroup) {
      err = "flash_attention_bwd_dq threadgroup exceeds max_threads_per_threadgroup";
      return std::nullopt;
    }
    finalize_slots();
    return plan;
  }

  if (fa->mode == ir::FlashAttnMode::PagedDecode) {
    if (ops.size() != 12u) {
      err = "paged_attention_decode expects 12 operands";
      return std::nullopt;
    }
    std::string* ptrs[7] = {&plan.q_name,           &plan.k_name,         &plan.v_name,
                            &plan.block_tables_name, &plan.context_lens_name, &plan.partial_o_name,
                            &plan.partial_lse_name};
    for (size_t i = 0; i < 7u; ++i) {
      ScalarDType pt = ScalarDType::F32;
      uint32_t pidx = 0;
      if (!expect_ptr_param(kernel, ops[i], *ptrs[i], pt, pidx)) {
        err = "paged_attention_decode operands 0..6 must be kernel pointer parameters";
        return std::nullopt;
      }
      if (i < 3u) {
        if (pt != ScalarDType::F32 && pt != ScalarDType::F16) {
          err = "paged_attention_decode Q/K_cache/V_cache must be f32 or f16 pointers";
          return std::nullopt;
        }
        if (i == 0u) {
          plan.elem_dt = pt;
          plan.elem_type = msl_scalar_type(pt);
        } else if (pt != plan.elem_dt) {
          err = "paged_attention_decode requires matching Q/K_cache/V_cache pointer dtypes";
          return std::nullopt;
        }
      } else if (i == 3u || i == 4u) {
        if (pt != ScalarDType::I32 && pt != ScalarDType::I64) {
          err = "paged_attention_decode BlockTables and ContextLens must be i32 or i64 pointers";
          return std::nullopt;
        }
      } else {
        if (pt != ScalarDType::F32) {
          err = "paged_attention_decode PartialO and PartialLSE must be f32 pointers";
          return std::nullopt;
        }
        written_params.insert(pidx);
      }
    }

    std::string* shapes[5] = {&plan.b_expr, &plan.hq_expr, &plan.hkv_expr, &plan.max_blocks_expr, &plan.d_expr};
    for (size_t i = 0; i < 5u; ++i) {
      auto e = scalar_long_expr(kernel, ops[7u + i]);
      if (!e) {
        err = "paged_attention_decode shape operands must be scalars or integer constants";
        return std::nullopt;
      }
      *shapes[i] = *e;
    }

    plan.paged_block_size = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_SIZE", 16));
    if (const auto ns = positive_constexpr_param(kernel, "NUM_SPLITS")) {
      plan.paged_num_splits = static_cast<uint32_t>(*ns);
    }
    const bool has_num_sg = find_constexpr_param(kernel, "NUM_SG").has_value();
    plan.num_sg = has_num_sg ? fa->num_simdgroups : 4u;
    plan.num_sg = std::max<uint32_t>(1u, std::min<uint32_t>(plan.num_sg, 16u));
    plan.tg_threads = plan.num_sg * simd;
    if (plan.tg_threads > target.max_threads_per_threadgroup) {
      err = "paged_attention_decode threadgroup exceeds max_threads_per_threadgroup";
      return std::nullopt;
    }
    finalize_slots();
    return plan;
  }

  err = "Unsupported FlashAttnMode";
  return std::nullopt;
}

class FlashAttnWriter {
 public:
  FlashAttnWriter(const Kernel& kernel, const FlashAttnPlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    emit_header();
    emit_signature();
    switch (plan_.ir_mode) {
      case ir::FlashAttnMode::Fwd:
        emit_fwd_body();
        break;
      case ir::FlashAttnMode::BwdDKDV:
        emit_bwd_dkdv_body();
        break;
      case ir::FlashAttnMode::BwdDQ:
        emit_bwd_dq_body();
        break;
      case ir::FlashAttnMode::PagedDecode:
        emit_paged_decode_body();
        break;
    }
    os_ << "}\n";
    return os_.str();
  }

 private:
  void emit_header() {
    os_ << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    if (plan_.attr.softcap > 0.0f) {
      HelperFlags helpers;
      helpers.needs_tanh = true;
      emit_helper_functions(os_, helpers);
    }
    if (plan_.ir_mode == ir::FlashAttnMode::Fwd && plan_.fwd_sched == FwdSchedule::MultiTileSgmm) {
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
    emit_param_slot_decls(os_, plan_.param_slots);
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    if (plan_.ir_mode == ir::FlashAttnMode::PagedDecode && plan_.paged_num_splits == 0u) {
      os_ << "    uint3 m1_ntg [[threadgroups_per_grid]],\n";
    }
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

  std::string kmax_expr(const std::string& row_expr) const {
    if (!plan_.attr.causal) return "m1_Lkv";
    return "min(m1_Lkv, max(0L, m1_q0 + (long)(" + row_expr + ") + 1L + " +
           std::to_string(plan_.attr.causal_offset) + "L))";
  }

  std::string kmin_expr(const std::string& row_expr) const {
    if (plan_.attr.window_size < 0) return "0L";
    return "max(0L, m1_q0 + (long)(" + row_expr + ") - " + std::to_string(plan_.attr.window_size) + "L)";
  }

  std::string score_to_log2_expr(const std::string& raw_dot, const std::string& q_pos_expr,
                                 const std::string& k_pos_expr) const {
    if (!plan_.attr.has_alibi && plan_.attr.softcap <= 0.0f) {
      return "((" + raw_dot + ") * m1_scale_log2)";
    }
    std::string s = "((" + raw_dot + ") * m1_scale";
    if (plan_.attr.has_alibi) {
      s += " + m1_alibi_slope * float((" + k_pos_expr + ") - (" + q_pos_expr + "))";
    }
    s += ")";
    if (plan_.attr.softcap > 0.0f) {
      s = "(m1_softcap * m1_tanh((" + s + ") * m1_inv_softcap))";
    }
    return "((" + s + ") * 1.4426950408889634f)";
  }

  uint32_t compute_fwd_shm_floats() const {
    if (plan_.fwd_sched == FwdSchedule::TwoWaveSgmm) return 16u * 512u;
    if (plan_.fwd_sched == FwdSchedule::SinglePassSgmm) return plan_.bm * plan_.bn;
    if (plan_.fwd_sched == FwdSchedule::MultiTileSgmm) return plan_.bm * (plan_.bn + 4u);
    if (plan_.fwd_sched == FwdSchedule::SplitMSgmm) return plan_.fast_only ? 0u : (plan_.bm * plan_.general_tile_n);
    return plan_.bm * plan_.general_tile_n;
  }

  void emit_fwd_body() {
    emit_fwd_prologue();
    switch (plan_.fwd_sched) {
      case FwdSchedule::TwoWaveSgmm:
        emit_two_wave_sgmm();
        break;
      case FwdSchedule::SinglePassSgmm:
        emit_single_pass_sgmm();
        break;
      case FwdSchedule::MultiTileSgmm:
        emit_multitile_sgmm();
        break;
      case FwdSchedule::SplitMSgmm:
        emit_split_m_sgmm();
        break;
      case FwdSchedule::GeneralRowWise:
        emit_general_rowwise(plan_.general_tile_n, "  ");
        break;
    }
  }

  void emit_fwd_prologue() {
    const uint32_t shm_floats = compute_fwd_shm_floats();
    if (shm_floats > 0u) {
      os_ << "  threadgroup float m1_shm[" << shm_floats << "];\n";
    }
    if (plan_.fwd_sched == FwdSchedule::MultiTileSgmm) {
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

    std::string kv_bh_expr = "m1_bh";
    if (plan_.attr.kv_group_size > 1) {
      os_ << "  const long m1_Hq = " << plan_.hq_expr << ";\n";
      os_ << "  const long m1_b = m1_bh / m1_Hq;\n";
      os_ << "  const long m1_hq = m1_bh % m1_Hq;\n";
      os_ << "  const long m1_kv_bh = m1_b * (m1_Hq / " << plan_.attr.kv_group_size << "L) + (m1_hq / "
          << plan_.attr.kv_group_size << "L);\n";
      kv_bh_expr = "m1_kv_bh";
    }

    os_ << "  device const " << plan_.elem_type << "* m1_Qb = " << plan_.q_name
        << " + m1_bh * (m1_Lq * " << ds << ");\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Kb = " << plan_.k_name
        << " + " << kv_bh_expr << " * (m1_Lkv * " << ds << ");\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Vb = " << plan_.v_name
        << " + " << kv_bh_expr << " * (m1_Lkv * " << ds << ");\n";
    os_ << "  device " << plan_.elem_type << "* m1_Ob = " << plan_.out_name
        << " + m1_bh * (m1_Lq * " << ds << ");\n";
    if (plan_.attr.return_lse) {
      os_ << "  device float* m1_Lb = " << plan_.lse_name << " + m1_bh * m1_Lq;\n";
    }
    if (plan_.attr.has_alibi) {
      if (plan_.attr.kv_group_size > 1) {
        os_ << "  const float m1_alibi_slope = (float)" << plan_.alibi_name << "[m1_hq];\n";
      } else {
        os_ << "  const long m1_alibi_h = " << plan_.alibi_heads_expr << ";\n";
        os_ << "  const float m1_alibi_slope = (float)" << plan_.alibi_name
            << "[(m1_alibi_h > 0L) ? (m1_bh % m1_alibi_h) : m1_bh];\n";
      }
    }
    if (plan_.attr.has_alibi || plan_.attr.softcap > 0.0f) {
      os_ << "  const float m1_scale = " << format_float_literal(plan_.attr.scale) << ";\n";
    }
    if (plan_.attr.softcap > 0.0f) {
      os_ << "  const float m1_softcap = " << format_float_literal(plan_.attr.softcap) << ";\n";
      os_ << "  const float m1_inv_softcap = " << format_float_literal(1.0 / static_cast<double>(plan_.attr.softcap))
          << ";\n";
    }
    os_ << "  const float m1_scale_log2 = " << format_float_literal(plan_.attr.scale)
        << " * 1.4426950408889634f;\n";
    os_ << "  (void)m1_tid; (void)m1_D; (void)m1_scale_log2;\n";
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
    if (plan_.attr.window_size >= 0) {
      os_ << "    const long m1_kv_start = max(0L, ((m1_q0 - " << plan_.attr.window_size << "L) & ~31L));\n";
      os_ << "    for (long m1_kv0 = m1_kv_start; m1_kv0 < m1_kv_lim; m1_kv0 += " << bn << "L) {\n";
    } else {
      os_ << "    for (long m1_kv0 = 0L; m1_kv0 < m1_kv_lim; m1_kv0 += " << bn << "L) {\n";
    }
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
    os_ << "        const long m1_qr = m1_q0 + (long)m1_row;\n";
    os_ << "        const long m1_kmin = " << kmin_expr("m1_row") << ";\n";
    os_ << "        const long m1_kmax = " << kmax_expr("m1_row") << ";\n";
    os_ << "        (void)m1_qr; (void)m1_kmin;\n";
    os_ << "        float m1_vals[" << n_soft << "];\n";
    os_ << "        float m1_mx = -1e30f;\n";
    for (uint32_t j = 0; j < n_soft; ++j) {
      const std::string raw = "m1_shm[m1_row * " + std::to_string(ldp) + " + c]";
      const std::string scored = score_to_log2_expr(raw, "m1_qr", "m1_kpos");
      const std::string cond =
          (plan_.attr.window_size >= 0) ? "(m1_kpos >= m1_kmin && m1_kpos < m1_kmax)" : "(m1_kpos < m1_kmax)";
      os_ << "        { const int c = (int)m1_lane + " << (j * 32u) << "; const long m1_kpos = m1_kv0 + (long)c; "
          << "const float v = " << cond << " ? " << scored << " : -1e30f; m1_vals[" << j
          << "] = v; m1_mx = max(m1_mx, v); }\n";
    }
    os_ << "        m1_mx = simd_max(m1_mx);\n";
    os_ << "        const float m1_mold = m1_ms[m1_row];\n";
    os_ << "        const float m1_mnew = max(m1_mold, m1_mx);\n";
    os_ << "        const bool m1_valid = (m1_mnew > -1e29f);\n";
    os_ << "        const float m1_alpha = (m1_valid && m1_mold > -1e29f) ? exp2(m1_mold - m1_mnew) : 0.0f;\n";
    os_ << "        float m1_rs = 0.0f;\n";
    for (uint32_t j = 0; j < n_soft; ++j) {
      const std::string cond = (plan_.attr.window_size >= 0)
                                   ? "(m1_valid && m1_kpos >= m1_kmin && m1_kpos < m1_kmax)"
                                   : "(m1_valid && m1_kpos < m1_kmax)";
      os_ << "        { const int c = (int)m1_lane + " << (j * 32u) << "; const long m1_kpos = m1_kv0 + (long)c; "
          << "const float p = " << cond << " ? exp2(m1_vals[" << j
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
    if (plan_.attr.return_lse) {
      os_ << "    if (m1_tid < 16u) {\n";
      os_ << "      const long qr = m1_q0 + (long)m1_tid;\n";
      os_ << "      const float mv = m1_ms[m1_tid];\n";
      os_ << "      const float lv = m1_ls[m1_tid];\n";
      os_ << "      m1_Lb[qr] = (lv > 0.0f && mv > -1e29f) ? (mv * 0.6931471805599453f + log(lv)) "
             ": as_type<float>(0xff800000u);\n";
      os_ << "    }\n";
    }
    if (!plan_.fast_only) {
      os_ << "    return;\n";
      os_ << "  }\n";
      emit_general_rowwise(std::min<uint32_t>(bn, 256u), "  ");
    } else {
      os_ << "  }\n";
    }
  }

  void emit_direct_fwd_step(uint32_t d_val, uint32_t m_tiles, uint32_t n_tiles, bool unmasked_fast,
                            const std::string& ind) {
    const uint32_t d_tiles = d_val / 8u;
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        os_ << ind << "simdgroup_float8x8 s_" << mi << "_" << nj
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "{\n";
    os_ << ind << "  device const float* kp = m1_Kb + m1_kv0 * " << d_val << "L;\n";
    os_ << ind << "  simdgroup_float8x8 kt_cur, kt_nxt;\n";
    os_ << ind << "  simdgroup_load(kt_cur, kp, " << d_val << ", ulong2(0, 0), true);\n";
    for (uint32_t di = 0; di < d_tiles; ++di) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        const uint32_t next_nj = (nj + 1u < n_tiles) ? (nj + 1u) : 0u;
        const uint32_t next_di = (nj + 1u < n_tiles) ? di : (di + 1u);
        if (next_di < d_tiles) {
          os_ << ind << "  simdgroup_load(kt_nxt, kp + " << (next_nj * 8u * d_val + next_di * 8u) << "L, " << d_val
              << ", ulong2(0, 0), true);\n";
        }
        for (uint32_t mi = 0; mi < m_tiles; ++mi) {
          os_ << ind << "  simdgroup_multiply_accumulate(s_" << mi << "_" << nj << ", q_" << mi << "_" << di
              << ", kt_cur, s_" << mi << "_" << nj << ");\n";
        }
        if (next_di < d_tiles) {
          os_ << ind << "  kt_cur = kt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      os_ << ind << "{\n";
      os_ << ind << "  float m1_tmx = -1e30f;\n";
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        os_ << ind << "  {\n";
        os_ << ind << "    thread float2& sv = reinterpret_cast<thread float2&>(s_" << mi << "_" << nj
            << ".thread_elements());\n";
        if (!unmasked_fast || plan_.attr.has_alibi) {
          os_ << ind << "    const long k0 = m1_kv0 + (long)(" << (nj * 8u) << " + m1_fc);\n";
          os_ << ind << "    const long k1 = k0 + 1L;\n";
        }
        if (!plan_.attr.has_alibi && plan_.attr.softcap <= 0.0f) {
          os_ << ind << "    sv *= m1_scale_log2;\n";
        } else {
          os_ << ind << "    sv *= m1_scale;\n";
          if (plan_.attr.has_alibi) {
            os_ << ind << "    sv.x += m1_alibi_slope * float(k0 - m1_qr_" << mi << ");\n";
            os_ << ind << "    sv.y += m1_alibi_slope * float(k1 - m1_qr_" << mi << ");\n";
          }
          if (plan_.attr.softcap > 0.0f) {
            os_ << ind << "    sv.x = m1_softcap * m1_tanh(sv.x * m1_inv_softcap);\n";
            os_ << ind << "    sv.y = m1_softcap * m1_tanh(sv.y * m1_inv_softcap);\n";
          }
          os_ << ind << "    sv *= 1.4426950408889634f;\n";
        }
        if (!unmasked_fast) {
          const std::string cond0 = (plan_.attr.window_size >= 0)
                                        ? ("(k0 >= m1_kmin_" + std::to_string(mi) + " && k0 < m1_kmax_" +
                                           std::to_string(mi) + ")")
                                        : ("(k0 < m1_kmax_" + std::to_string(mi) + ")");
          const std::string cond1 = (plan_.attr.window_size >= 0)
                                        ? ("(k1 >= m1_kmin_" + std::to_string(mi) + " && k1 < m1_kmax_" +
                                           std::to_string(mi) + ")")
                                        : ("(k1 < m1_kmax_" + std::to_string(mi) + ")");
          os_ << ind << "    sv.x = " << cond0 << " ? sv.x : -1e30f;\n";
          os_ << ind << "    sv.y = " << cond1 << " ? sv.y : -1e30f;\n";
        }
        os_ << ind << "    m1_tmx = max(m1_tmx, max(sv.x, sv.y));\n";
        os_ << ind << "  }\n";
      }
      os_ << ind << "  m1_tmx = max(m1_tmx, simd_shuffle_xor(m1_tmx, 1));\n";
      os_ << ind << "  m1_tmx = max(m1_tmx, simd_shuffle_xor(m1_tmx, 8));\n";
      os_ << ind << "  const float m1_mnew = max(m1_m_" << mi << ", m1_tmx);\n";
      if (unmasked_fast) {
        os_ << ind << "  const float m1_alpha = (m1_m_" << mi << " > -1e29f) ? exp2(m1_m_" << mi
            << " - m1_mnew) : 0.0f;\n";
      } else {
        os_ << ind << "  const bool m1_rvalid = (m1_mnew > -1e29f);\n";
        os_ << ind << "  const float m1_alpha = (m1_rvalid && m1_m_" << mi << " > -1e29f) ? exp2(m1_m_" << mi
            << " - m1_mnew) : 0.0f;\n";
      }
      for (uint32_t di = 0; di < d_tiles; ++di) {
        os_ << ind << "  reinterpret_cast<thread float2&>(o_" << mi << "_" << di
            << ".thread_elements()) *= m1_alpha;\n";
      }
      os_ << ind << "  float m1_trs = 0.0f;\n";
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        os_ << ind << "  {\n";
        os_ << ind << "    thread float2& sv = reinterpret_cast<thread float2&>(s_" << mi << "_" << nj
            << ".thread_elements());\n";
        if (unmasked_fast) {
          os_ << ind << "    sv = exp2(sv - m1_mnew);\n";
        } else {
          os_ << ind << "    const long k0 = m1_kv0 + (long)(" << (nj * 8u) << " + m1_fc);\n";
          os_ << ind << "    const long k1 = k0 + 1L;\n";
          const std::string cond0 = (plan_.attr.window_size >= 0)
                                        ? ("(m1_rvalid && k0 >= m1_kmin_" + std::to_string(mi) + " && k0 < m1_kmax_" +
                                           std::to_string(mi) + ")")
                                        : ("(m1_rvalid && k0 < m1_kmax_" + std::to_string(mi) + ")");
          const std::string cond1 = (plan_.attr.window_size >= 0)
                                        ? ("(m1_rvalid && k1 >= m1_kmin_" + std::to_string(mi) + " && k1 < m1_kmax_" +
                                           std::to_string(mi) + ")")
                                        : ("(m1_rvalid && k1 < m1_kmax_" + std::to_string(mi) + ")");
          os_ << ind << "    sv.x = " << cond0 << " ? exp2(sv.x - m1_mnew) : 0.0f;\n";
          os_ << ind << "    sv.y = " << cond1 << " ? exp2(sv.y - m1_mnew) : 0.0f;\n";
        }
        os_ << ind << "    m1_trs += sv.x + sv.y;\n";
        os_ << ind << "  }\n";
      }
      os_ << ind << "  m1_trs += simd_shuffle_xor(m1_trs, 1);\n";
      os_ << ind << "  m1_trs += simd_shuffle_xor(m1_trs, 8);\n";
      os_ << ind << "  m1_l_" << mi << " = m1_l_" << mi << " * m1_alpha + m1_trs;\n";
      os_ << ind << "  m1_m_" << mi << " = m1_mnew;\n";
      os_ << ind << "}\n";
    }

    os_ << ind << "{\n";
    os_ << ind << "  device const float* vp = m1_Vb + m1_kv0 * " << d_val << "L;\n";
    os_ << ind << "  simdgroup_float8x8 vt_cur, vt_nxt;\n";
    os_ << ind << "  simdgroup_load(vt_cur, vp, " << d_val << ");\n";
    for (uint32_t nj = 0; nj < n_tiles; ++nj) {
      for (uint32_t di = 0; di < d_tiles; ++di) {
        const uint32_t next_di = (di + 1u < d_tiles) ? (di + 1u) : 0u;
        const uint32_t next_nj = (di + 1u < d_tiles) ? nj : (nj + 1u);
        if (next_nj < n_tiles) {
          os_ << ind << "  simdgroup_load(vt_nxt, vp + " << (next_nj * 8u * d_val + next_di * 8u) << "L, " << d_val
              << ");\n";
        }
        for (uint32_t mi = 0; mi < m_tiles; ++mi) {
          os_ << ind << "  simdgroup_multiply_accumulate(o_" << mi << "_" << di << ", s_" << mi << "_" << nj
              << ", vt_cur, o_" << mi << "_" << di << ");\n";
        }
        if (next_nj < n_tiles) {
          os_ << ind << "  vt_cur = vt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";
  }

  void emit_split_m_d_spec(uint32_t d_val, uint32_t tile_n, const std::string& ind) {
    const uint32_t d_tiles = d_val / 8u;
    const uint32_t n_tiles = tile_n / 8u;
    const uint32_t rows_per_sg = plan_.bm / plan_.num_sg;
    const uint32_t m_tiles = rows_per_sg / 8u;

    os_ << ind << "const int m1_qid = (int)m1_lane / 4;\n";
    os_ << ind << "const int m1_fr = (m1_qid & 4) + ((int)m1_lane / 2) % 4;\n";
    os_ << ind << "const int m1_fc = (m1_qid & 2) * 2 + ((int)m1_lane % 2) * 2;\n";
    os_ << ind << "const long m1_sg_q0 = m1_q0 + (long)(m1_sg * " << rows_per_sg << "u);\n";
    os_ << ind << "if (m1_sg_q0 >= m1_Lq) return;\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t di = 0; di < d_tiles; ++di) {
        os_ << ind << "simdgroup_float8x8 q_" << mi << "_" << di << ";\n";
        os_ << ind << "simdgroup_load(q_" << mi << "_" << di << ", m1_Qb + (m1_sg_q0 + " << (mi * 8u) << "L) * "
            << d_val << "L + " << (di * 8u) << "L, " << d_val << ");\n";
      }
      os_ << ind << "float m1_m_" << mi << " = -1e30f;\n";
      os_ << ind << "float m1_l_" << mi << " = 0.0f;\n";
      for (uint32_t di = 0; di < d_tiles; ++di) {
        os_ << ind << "simdgroup_float8x8 o_" << mi << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
      os_ << ind << "const long m1_qr_" << mi << " = m1_sg_q0 + (long)(" << (mi * 8u) << " + m1_fr);\n";
      if (plan_.attr.causal) {
        os_ << ind << "const long m1_kmax_" << mi << " = min(m1_Lkv, max(0L, m1_qr_" << mi << " + 1L + "
            << plan_.attr.causal_offset << "L));\n";
      } else {
        os_ << ind << "const long m1_kmax_" << mi << " = m1_Lkv;\n";
      }
      if (plan_.attr.window_size >= 0) {
        os_ << ind << "const long m1_kmin_" << mi << " = max(0L, m1_qr_" << mi << " - " << plan_.attr.window_size
            << "L);\n";
      } else {
        os_ << ind << "const long m1_kmin_" << mi << " = 0L;\n";
      }
    }

    if (plan_.attr.window_size >= 0) {
      os_ << ind << "const long m1_kv_start = max(0L, (m1_sg_q0 - " << plan_.attr.window_size << "L) & ~7L);\n";
      os_ << ind << "const long m1_full_start = max(0L, m1_sg_q0 + " << (rows_per_sg - 1u) << "L - "
          << plan_.attr.window_size << "L);\n";
    } else {
      os_ << ind << "const long m1_kv_start = 0L;\n";
      os_ << ind << "const long m1_full_start = 0L;\n";
    }
    if (plan_.attr.causal) {
      os_ << ind << "const long m1_kv_end_raw = min(m1_Lkv, max(0L, m1_sg_q0 + " << rows_per_sg << "L + "
          << plan_.attr.causal_offset << "L));\n";
      os_ << ind << "const long m1_kv_end = min(m1_Lkv, (m1_kv_end_raw + 7L) & ~7L);\n";
      os_ << ind << "const long m1_full_end = min(m1_Lkv, max(0L, m1_sg_q0 + 1L + " << plan_.attr.causal_offset
          << "L));\n";
    } else {
      os_ << ind << "const long m1_kv_end = m1_Lkv;\n";
      os_ << ind << "const long m1_full_end = m1_Lkv;\n";
    }

    os_ << ind << "long m1_kv0 = m1_kv_start;\n";
    os_ << ind << "for (; m1_kv0 + " << tile_n << "L <= m1_kv_end; m1_kv0 += " << tile_n << "L) {\n";
    if (!plan_.attr.causal && plan_.attr.window_size < 0) {
      emit_direct_fwd_step(d_val, m_tiles, n_tiles, true, ind + "  ");
    } else {
      os_ << ind << "  if (m1_kv0 >= m1_full_start && m1_kv0 + " << tile_n << "L <= m1_full_end) {\n";
      emit_direct_fwd_step(d_val, m_tiles, n_tiles, true, ind + "    ");
      os_ << ind << "  } else {\n";
      emit_direct_fwd_step(d_val, m_tiles, n_tiles, false, ind + "    ");
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";
    os_ << ind << "for (; m1_kv0 < m1_kv_end; m1_kv0 += 8L) {\n";
    emit_direct_fwd_step(d_val, m_tiles, 1u, false, ind + "  ");
    os_ << ind << "}\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      os_ << ind << "{\n";
      os_ << ind << "  const bool m1_ok = (m1_l_" << mi << " > 0.0f) && (m1_m_" << mi << " > -1e29f);\n";
      os_ << ind << "  const float m1_inv = m1_ok ? (1.0f / m1_l_" << mi << ") : 0.0f;\n";
      for (uint32_t di = 0; di < d_tiles; ++di) {
        os_ << ind << "  reinterpret_cast<thread float2&>(o_" << mi << "_" << di
            << ".thread_elements()) *= m1_inv;\n";
        os_ << ind << "  simdgroup_store(o_" << mi << "_" << di << ", m1_Ob + (m1_sg_q0 + " << (mi * 8u) << "L) * "
            << d_val << "L + " << (di * 8u) << "L, " << d_val << ");\n";
      }
      if (plan_.attr.return_lse) {
        os_ << ind << "  if (m1_fc == 0) {\n";
        os_ << ind << "    m1_Lb[m1_qr_" << mi << "] = m1_ok ? (m1_m_" << mi << " * 0.6931471805599453f + log(m1_l_"
            << mi << ")) : as_type<float>(0xff800000u);\n";
        os_ << ind << "  }\n";
      }
      os_ << ind << "}\n";
    }
  }

  void emit_split_m_sgmm() {
    const uint32_t tile_n_64 = (plan_.bn == 32u) ? 32u : 64u;
    const uint32_t m_tiles = (plan_.bm / plan_.num_sg) / 8u;

    if (plan_.fast_only) {
      os_ << "  if (m1_D == 64L) {\n";
      emit_split_m_d_spec(64u, tile_n_64, "    ");
      os_ << "    return;\n";
      os_ << "  }\n";
      if (m_tiles == 1u) {
        os_ << "  if (m1_D == 128L) {\n";
        emit_split_m_d_spec(128u, 32u, "    ");
        os_ << "    return;\n";
        os_ << "  }\n";
      }
      return;
    }

    const uint32_t lq_mask = m_tiles * 8u - 1u;
    os_ << "  if (m1_D == 64L && m1_Lq > 0L && (m1_Lq & " << lq_mask
        << "L) == 0L && m1_Lkv > 0L && (m1_Lkv & 7L) == 0L) {\n";
    emit_split_m_d_spec(64u, tile_n_64, "    ");
    os_ << "    return;\n";
    os_ << "  }\n";

    if (m_tiles == 1u) {
      os_ << "  if (m1_D == 128L && m1_Lq > 0L && (m1_Lq & 7L) == 0L && m1_Lkv > 0L && (m1_Lkv & 7L) == 0L) {\n";
      emit_split_m_d_spec(128u, 32u, "    ");
      os_ << "    return;\n";
      os_ << "  }\n";
    }

    emit_general_rowwise(plan_.general_tile_n, "  ");
  }

  void emit_general_rowwise(uint32_t tile_n, const std::string& ind) {
    const uint32_t per_lane = std::max<uint32_t>(1u, tile_n / 32u);
    os_ << ind << "for (int m1_r = (int)m1_sg; m1_r < " << plan_.bm << "; m1_r += " << plan_.num_sg << ") {\n";
    os_ << ind << "  const long m1_qr = m1_q0 + (long)m1_r;\n";
    os_ << ind << "  if (m1_qr >= m1_Lq) continue;\n";
    os_ << ind << "  const long m1_kmin = " << kmin_expr("m1_r") << ";\n";
    os_ << ind << "  const long m1_kmax = " << kmax_expr("m1_r") << ";\n";
    os_ << ind << "  if (m1_kmax <= m1_kmin) {\n";
    os_ << ind << "    for (long m1_d = (long)m1_lane; m1_d < m1_D; m1_d += 32L) {\n";
    os_ << ind << "      m1_Ob[m1_qr * m1_D + m1_d] = " << plan_.elem_type << "(0.0f);\n";
    os_ << ind << "    }\n";
    if (plan_.attr.return_lse) {
      os_ << ind << "    if (m1_lane == 0u) m1_Lb[m1_qr] = as_type<float>(0xff800000u);\n";
    }
    os_ << ind << "    continue;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup float* m1_rs_buf = m1_shm + m1_r * " << tile_n << ";\n";
    os_ << ind << "  float m1_mi = -1e30f;\n";
    os_ << ind << "  float m1_li = 0.0f;\n";
    os_ << ind << "  for (long m1_kv0 = m1_kmin; m1_kv0 < m1_kmax; m1_kv0 += " << tile_n << "L) {\n";
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
    os_ << ind << "      if (m1_lane == 0u) m1_rs_buf[m1_c] = "
        << score_to_log2_expr("m1_dot", "m1_qr", "m1_kv0 + (long)m1_c") << ";\n";
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
    os_ << ind << "      float m1_acc = (m1_kv0 == m1_kmin) ? 0.0f : ((float)m1_Ob[m1_qr * m1_D + m1_d] * m1_alpha);\n";
    os_ << ind << "      for (int m1_c = 0; m1_c < m1_tlen; ++m1_c) {\n";
    os_ << ind << "        m1_acc = fma(m1_rs_buf[m1_c], (float)m1_Vb[(m1_kv0 + (long)m1_c) * m1_D + m1_d], m1_acc);\n";
    os_ << ind << "      }\n";
    os_ << ind << "      if (m1_last) m1_acc *= m1_inv;\n";
    os_ << ind << "      m1_Ob[m1_qr * m1_D + m1_d] = " << plan_.elem_type << "(m1_acc);\n";
    os_ << ind << "    }\n";
    os_ << ind << "    simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "  }\n";
    if (plan_.attr.return_lse) {
      os_ << ind << "  if (m1_lane == 0u) {\n";
      os_ << ind << "    m1_Lb[m1_qr] = (m1_li > 0.0f && m1_mi > -1e29f) ? (m1_mi * 0.6931471805599453f + log(m1_li)) "
                    ": as_type<float>(0xff800000u);\n";
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";
  }

  void emit_stage_tile_to_shm(const std::string& ind, const std::string& src_ptr, const std::string& row_base,
                              const std::string& row_limit, const std::string& dst_shm, uint32_t rows, uint32_t d_val,
                              uint32_t ld) {
    const uint32_t vecs_per_row = d_val / 4u;
    const uint32_t total_vecs = rows * vecs_per_row;
    os_ << ind << "if (" << row_base << " + " << rows << "L <= " << row_limit << ") {\n";
    os_ << ind << "  for (uint m1_idx = m1_tid; m1_idx < " << total_vecs << "u; m1_idx += " << plan_.tg_threads
        << "u) {\n";
    os_ << ind << "    const uint r = m1_idx / " << vecs_per_row << "u;\n";
    os_ << ind << "    const uint c = (m1_idx % " << vecs_per_row << "u) << 2;\n";
    os_ << ind << "    *(threadgroup float4*)(" << dst_shm << " + r * " << ld << "u + c) = "
        << "float4(*(device const packed_float4*)(" << src_ptr << " + (" << row_base << " + (long)r) * " << d_val
        << "L + (long)c));\n";
    os_ << ind << "  }\n";
    os_ << ind << "} else {\n";
    os_ << ind << "  for (uint m1_idx = m1_tid; m1_idx < " << total_vecs << "u; m1_idx += " << plan_.tg_threads
        << "u) {\n";
    os_ << ind << "    const uint r = m1_idx / " << vecs_per_row << "u;\n";
    os_ << ind << "    const uint c = (m1_idx % " << vecs_per_row << "u) << 2;\n";
    os_ << ind << "    const long gr = " << row_base << " + (long)r;\n";
    os_ << ind << "    *(threadgroup float4*)(" << dst_shm << " + r * " << ld << "u + c) = "
        << "(gr < " << row_limit << ") ? float4(*(device const packed_float4*)(" << src_ptr << " + gr * " << d_val
        << "L + (long)c)) : float4(0.0f);\n";
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  void emit_bwd_dkdv_body() {
    const uint32_t bn = plan_.bn;
    const uint32_t q_tile = (plan_.bm >= 32u) ? 32u : 16u;
    constexpr uint32_t kLd = 68u;
    const uint32_t shm_floats = 2u * q_tile * kLd + 2u * q_tile;
    const bool can_sgmm_64 =
        (plan_.elem_dt == ScalarDType::F32 && bn >= 8u * plan_.num_sg && (bn % (8u * plan_.num_sg)) == 0u &&
         (bn / (8u * plan_.num_sg)) <= 2u && shm_floats * 4u <= plan_.max_smem);

    if (can_sgmm_64) {
      os_ << "  threadgroup float m1_shm[" << shm_floats << "];\n";
    }
    os_ << "  const long m1_BH = " << plan_.bh_expr << ";\n";
    os_ << "  const long m1_Lq = " << plan_.lq_expr << ";\n";
    os_ << "  const long m1_Lkv = " << plan_.lkv_expr << ";\n";
    os_ << "  const long m1_D = " << plan_.d_expr << ";\n";
    os_ << "  const long m1_bh = (long)m1_tgid.y;\n";
    os_ << "  if (m1_bh >= m1_BH) return;\n";
    os_ << "  const long m1_k0 = (long)m1_tgid.x * " << bn << "L;\n";
    os_ << "  if (m1_k0 >= m1_Lkv) return;\n";

    os_ << "  device const " << plan_.elem_type << "* m1_Qb = " << plan_.q_name << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Kb = " << plan_.k_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Vb = " << plan_.v_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_dOb = " << plan_.do_name << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  device const float* m1_Lb = " << plan_.lse_name << " + m1_bh * m1_Lq;\n";
    os_ << "  device const float* m1_Deltab = " << plan_.delta_name << " + m1_bh * m1_Lq;\n";
    os_ << "  device " << plan_.elem_type << "* m1_dKb = " << plan_.dk_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device " << plan_.elem_type << "* m1_dVb = " << plan_.dv_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  const float m1_scale = " << format_float_literal(plan_.attr.scale) << ";\n";
    os_ << "  const float m1_scale_log2 = "
        << format_float_literal(static_cast<double>(plan_.attr.scale) * 1.44269504088896340736) << ";\n";
    os_ << "  (void)m1_scale_log2;\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << "  const float m1_softcap = " << format_float_literal(plan_.attr.softcap) << ";\n";
      os_ << "  const float m1_inv_softcap = " << format_float_literal(1.0 / static_cast<double>(plan_.attr.softcap))
          << ";\n";
    }
    if (plan_.attr.causal) {
      os_ << "  const long m1_q_start = max(0L, m1_k0 - " << plan_.attr.causal_offset << "L);\n";
    } else {
      os_ << "  const long m1_q_start = 0L;\n";
    }
    if (plan_.attr.window_size >= 0) {
      os_ << "  const long m1_q_end = min(m1_Lq, max(0L, m1_k0 + " << bn << "L + " << plan_.attr.window_size
          << "L));\n";
    } else {
      os_ << "  const long m1_q_end = m1_Lq;\n";
    }
    os_ << "  (void)m1_tid;\n";

    if (can_sgmm_64) {
      os_ << "  if (m1_D == 64L && m1_Lq > 0L && (m1_Lq & 7L) == 0L && m1_Lkv > 0L && (m1_Lkv & 7L) == 0L) {\n";
      emit_bwd_dkdv_sgmm64(q_tile, "    ");
      os_ << "    return;\n";
      os_ << "  }\n";
    }

    emit_bwd_dkdv_general("  ");
  }

  void emit_shm_bwd_dkdv_step(uint32_t n_tiles, uint32_t m_tiles, const std::string& q_shm, const std::string& do_shm,
                              const std::string& l_shm, const std::string& delta_shm, const std::string& q_base_expr,
                              bool unmasked_fast, const std::string& ind) {
    constexpr uint32_t kLd = 68u;
    constexpr uint32_t kDt = 8u;
    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
      for (uint32_t mi = 0; mi < m_tiles; ++mi) {
        os_ << ind << "simdgroup_float8x8 st_" << ni << "_" << mi
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 dpt_" << ni << "_" << mi
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 qt_cur, qt_nxt;\n";
    os_ << ind << "  simdgroup_load(qt_cur, " << q_shm << ", " << kLd << ", ulong2(0, 0), true);\n";
    for (uint32_t di = 0; di < kDt; ++di) {
      for (uint32_t mi = 0; mi < m_tiles; ++mi) {
        const uint32_t next_mi = (mi + 1u < m_tiles) ? (mi + 1u) : 0u;
        const uint32_t next_di = (mi + 1u < m_tiles) ? di : (di + 1u);
        if (next_di < kDt) {
          os_ << ind << "  simdgroup_load(qt_nxt, " << q_shm << " + " << (next_mi * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ", ulong2(0, 0), true);\n";
        }
        for (uint32_t ni = 0; ni < n_tiles; ++ni) {
          os_ << ind << "  simdgroup_multiply_accumulate(st_" << ni << "_" << mi << ", k_" << ni << "_" << di
              << ", qt_cur, st_" << ni << "_" << mi << ");\n";
        }
        if (next_di < kDt) {
          os_ << ind << "  qt_cur = qt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 dot_cur, dot_nxt;\n";
    os_ << ind << "  simdgroup_load(dot_cur, " << do_shm << ", " << kLd << ", ulong2(0, 0), true);\n";
    for (uint32_t di = 0; di < kDt; ++di) {
      for (uint32_t mi = 0; mi < m_tiles; ++mi) {
        const uint32_t next_mi = (mi + 1u < m_tiles) ? (mi + 1u) : 0u;
        const uint32_t next_di = (mi + 1u < m_tiles) ? di : (di + 1u);
        if (next_di < kDt) {
          os_ << ind << "  simdgroup_load(dot_nxt, " << do_shm << " + " << (next_mi * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ", ulong2(0, 0), true);\n";
        }
        for (uint32_t ni = 0; ni < n_tiles; ++ni) {
          os_ << ind << "  simdgroup_multiply_accumulate(dpt_" << ni << "_" << mi << ", v_" << ni << "_" << di
              << ", dot_cur, dpt_" << ni << "_" << mi << ");\n";
        }
        if (next_di < kDt) {
          os_ << ind << "  dot_cur = dot_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      os_ << ind << "{\n";
      os_ << ind << "  const int c0 = " << (mi * 8u) << " + m1_fc;\n";
      os_ << ind << "  const float2 lv = *(threadgroup const float2*)(" << l_shm << " + c0);\n";
      os_ << ind << "  const float2 dv = *(threadgroup const float2*)(" << delta_shm << " + c0);\n";
      for (uint32_t ni = 0; ni < n_tiles; ++ni) {
        os_ << ind << "  {\n";
        os_ << ind << "    thread float2& sv = reinterpret_cast<thread float2&>(st_" << ni << "_" << mi
            << ".thread_elements());\n";
        os_ << ind << "    thread float2& dpv = reinterpret_cast<thread float2&>(dpt_" << ni << "_" << mi
            << ".thread_elements());\n";
        if (unmasked_fast) {
          if (plan_.attr.softcap > 0.0f) {
            os_ << ind << "    float2 sc = sv * m1_scale;\n";
            os_ << ind << "    const float2 tv = float2(m1_tanh(sc.x * m1_inv_softcap), m1_tanh(sc.y * m1_inv_softcap));\n";
            os_ << ind << "    sc = m1_softcap * tv;\n";
            os_ << ind << "    const float2 deriv = 1.0f - tv * tv;\n";
            os_ << ind << "    const float2 pv = exp2((sc - lv) * 1.4426950408889634f);\n";
            os_ << ind << "    sv = pv;\n";
            os_ << ind << "    dpv = pv * (dpv - dv) * deriv * m1_scale;\n";
          } else {
            os_ << ind << "    const float2 pv = exp2(fma(sv, m1_scale_log2, -lv));\n";
            os_ << ind << "    sv = pv;\n";
            os_ << ind << "    dpv = pv * (dpv - dv) * m1_scale;\n";
          }
        } else {
          os_ << ind << "    const long q0 = (" << q_base_expr << ") + (long)c0;\n";
          os_ << ind << "    const long q1 = q0 + 1L;\n";
          os_ << ind << "    const long kr = m1_kr_" << ni << ";\n";
          auto emit_elem = [&](const char* comp, const char* qv) {
            std::string valid = "(kr < m1_Lkv && " + std::string(qv) + " < m1_Lq";
            if (plan_.attr.causal) {
              valid += " && kr <= " + std::string(qv) + " + " + std::to_string(plan_.attr.causal_offset) + "L";
            }
            if (plan_.attr.window_size >= 0) {
              valid += " && kr >= " + std::string(qv) + " - " + std::to_string(plan_.attr.window_size) + "L";
            }
            valid += ")";
            os_ << ind << "    {\n";
            os_ << ind << "      const bool ok = " << valid << " && (lv." << comp << " > -1e29f);\n";
            if (plan_.attr.softcap > 0.0f) {
              os_ << ind << "      float sc = sv." << comp << " * m1_scale;\n";
              os_ << ind << "      const float tv = m1_tanh(sc * m1_inv_softcap);\n";
              os_ << ind << "      sc = m1_softcap * tv;\n";
              os_ << ind << "      const float deriv = 1.0f - tv * tv;\n";
              os_ << ind << "      const float pv = ok ? exp2((sc - lv." << comp << ") * 1.4426950408889634f) : 0.0f;\n";
              os_ << ind << "      sv." << comp << " = pv;\n";
              os_ << ind << "      dpv." << comp << " = ok ? (pv * (dpv." << comp << " - dv." << comp
                  << ") * deriv * m1_scale) : 0.0f;\n";
            } else {
              os_ << ind << "      const float pv = ok ? exp2(fma(sv." << comp << ", m1_scale_log2, -lv." << comp
                  << ")) : 0.0f;\n";
              os_ << ind << "      sv." << comp << " = pv;\n";
              os_ << ind << "      dpv." << comp << " = ok ? (pv * (dpv." << comp << " - dv." << comp
                  << ") * m1_scale) : 0.0f;\n";
            }
            os_ << ind << "    }\n";
          };
          emit_elem("x", "q0");
          emit_elem("y", "q1");
        }
        os_ << ind << "  }\n";
      }
      os_ << ind << "}\n";
    }

    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 dot_cur, dot_nxt;\n";
    os_ << ind << "  simdgroup_load(dot_cur, " << do_shm << ", " << kLd << ");\n";
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t di = 0; di < kDt; ++di) {
        const uint32_t next_di = (di + 1u < kDt) ? (di + 1u) : 0u;
        const uint32_t next_mi = (di + 1u < kDt) ? mi : (mi + 1u);
        if (next_mi < m_tiles) {
          os_ << ind << "  simdgroup_load(dot_nxt, " << do_shm << " + " << (next_mi * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ");\n";
        }
        for (uint32_t ni = 0; ni < n_tiles; ++ni) {
          os_ << ind << "  simdgroup_multiply_accumulate(dv_" << ni << "_" << di << ", st_" << ni << "_" << mi
              << ", dot_cur, dv_" << ni << "_" << di << ");\n";
        }
        if (next_mi < m_tiles) {
          os_ << ind << "  dot_cur = dot_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 qt_cur, qt_nxt;\n";
    os_ << ind << "  simdgroup_load(qt_cur, " << q_shm << ", " << kLd << ");\n";
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t di = 0; di < kDt; ++di) {
        const uint32_t next_di = (di + 1u < kDt) ? (di + 1u) : 0u;
        const uint32_t next_mi = (di + 1u < kDt) ? mi : (mi + 1u);
        if (next_mi < m_tiles) {
          os_ << ind << "  simdgroup_load(qt_nxt, " << q_shm << " + " << (next_mi * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ");\n";
        }
        for (uint32_t ni = 0; ni < n_tiles; ++ni) {
          os_ << ind << "  simdgroup_multiply_accumulate(dk_" << ni << "_" << di << ", dpt_" << ni << "_" << mi
              << ", qt_cur, dk_" << ni << "_" << di << ");\n";
        }
        if (next_mi < m_tiles) {
          os_ << ind << "  qt_cur = qt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";
  }

  void emit_bwd_dkdv_sgmm64(uint32_t q_tile, const std::string& ind) {
    constexpr uint32_t kD = 64u;
    constexpr uint32_t kDt = 8u;
    constexpr uint32_t kLd = 68u;
    const uint32_t bn = plan_.bn;
    const uint32_t rows_per_sg = bn / plan_.num_sg;
    const uint32_t n_tiles = rows_per_sg / 8u;
    const std::string q_shm = "m1_shm";
    const std::string do_shm = "(m1_shm + " + std::to_string(q_tile * kLd) + "u)";
    const std::string l_shm = "(m1_shm + " + std::to_string(2u * q_tile * kLd) + "u)";
    const std::string delta_shm = "(m1_shm + " + std::to_string(2u * q_tile * kLd + q_tile) + "u)";

    os_ << ind << "const int m1_qid = (int)m1_lane / 4;\n";
    os_ << ind << "const int m1_fr = (m1_qid & 4) + ((int)m1_lane / 2) % 4;\n";
    os_ << ind << "const int m1_fc = (m1_qid & 2) * 2 + ((int)m1_lane % 2) * 2;\n";
    os_ << ind << "const long m1_sg_k0 = m1_k0 + (long)(m1_sg * " << rows_per_sg << "u);\n";
    os_ << ind << "const bool m1_sg_active = (m1_sg_k0 < m1_Lkv);\n";

    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "simdgroup_float8x8 k_" << ni << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 v_" << ni << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 dk_" << ni << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 dv_" << ni << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
      os_ << ind << "const long m1_kr_" << ni << " = m1_sg_k0 + (long)(" << (ni * 8u) << " + m1_fr);\n";
    }
    os_ << ind << "if (m1_sg_active) {\n";
    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
      os_ << ind << "  if (m1_sg_k0 + " << (ni * 8u) << "L < m1_Lkv) {\n";
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "    simdgroup_load(k_" << ni << "_" << di << ", m1_Kb + (m1_sg_k0 + " << (ni * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
        os_ << ind << "    simdgroup_load(v_" << ni << "_" << di << ", m1_Vb + (m1_sg_k0 + " << (ni * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
      }
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";

    if (plan_.attr.causal) {
      os_ << ind << "const long m1_sg_qs = max(0L, m1_sg_k0 - " << plan_.attr.causal_offset << "L);\n";
      os_ << ind << "const long m1_full_qs = max(0L, m1_sg_k0 + " << (rows_per_sg - 1u) << "L - "
          << plan_.attr.causal_offset << "L);\n";
    } else {
      os_ << ind << "const long m1_sg_qs = 0L;\n";
      os_ << ind << "const long m1_full_qs = 0L;\n";
    }
    if (plan_.attr.window_size >= 0) {
      os_ << ind << "const long m1_sg_qe = min(m1_Lq, max(0L, m1_sg_k0 + " << rows_per_sg << "L + "
          << plan_.attr.window_size << "L));\n";
      os_ << ind << "const long m1_full_qe = min(m1_Lq, max(0L, m1_sg_k0 + 1L + " << plan_.attr.window_size
          << "L));\n";
    } else {
      os_ << ind << "const long m1_sg_qe = m1_Lq;\n";
      os_ << ind << "const long m1_full_qe = m1_Lq;\n";
    }

    os_ << ind << "for (long m1_q0 = m1_q_start; m1_q0 < m1_q_end; m1_q0 += " << q_tile << "L) {\n";
    emit_stage_tile_to_shm(ind + "  ", "m1_Qb", "m1_q0", "m1_Lq", q_shm, q_tile, kD, kLd);
    emit_stage_tile_to_shm(ind + "  ", "m1_dOb", "m1_q0", "m1_Lq", do_shm, q_tile, kD, kLd);
    os_ << ind << "  if (m1_tid < " << q_tile << "u) {\n";
    os_ << ind << "    const long gq = m1_q0 + (long)m1_tid;\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "    " << l_shm << "[m1_tid] = (gq < m1_Lq) ? m1_Lb[gq] : -1e30f;\n";
    } else {
      os_ << ind << "    const float lraw = (gq < m1_Lq) ? m1_Lb[gq] : -1e30f;\n";
      os_ << ind << "    " << l_shm << "[m1_tid] = (lraw > -1e29f) ? (lraw * 1.4426950408889634f) : -1e30f;\n";
    }
    os_ << ind << "    " << delta_shm << "[m1_tid] = (gq < m1_Lq) ? m1_Deltab[gq] : 0.0f;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "  if (m1_sg_active && m1_q0 + " << q_tile << "L > m1_sg_qs && m1_q0 < m1_sg_qe) {\n";
    const uint32_t sub_rows = (q_tile >= 32u) ? 16u : q_tile;
    const uint32_t sub_count = q_tile / sub_rows;
    const uint32_t sub_m_tiles = sub_rows / 8u;
    for (uint32_t sub = 0; sub < sub_count; ++sub) {
      const uint32_t row_off = sub * sub_rows;
      const std::string sub_q_shm =
          (row_off == 0u) ? q_shm : ("(" + q_shm + " + " + std::to_string(row_off * kLd) + "u)");
      const std::string sub_do_shm =
          (row_off == 0u) ? do_shm : ("(" + do_shm + " + " + std::to_string(row_off * kLd) + "u)");
      const std::string sub_l_shm =
          (row_off == 0u) ? l_shm : ("(" + l_shm + " + " + std::to_string(row_off) + "u)");
      const std::string sub_d_shm =
          (row_off == 0u) ? delta_shm : ("(" + delta_shm + " + " + std::to_string(row_off) + "u)");
      const std::string sub_q0 = (row_off == 0u) ? "m1_q0" : ("m1_q0 + " + std::to_string(row_off) + "L");
      const std::string inner_ind = (sub_count > 1u) ? (ind + "      ") : (ind + "    ");
      if (sub_count > 1u) {
        os_ << ind << "    if (" << sub_q0 << " + " << sub_rows << "L > m1_sg_qs && " << sub_q0 << " < m1_sg_qe) {\n";
      }
      if (!plan_.attr.causal && plan_.attr.window_size < 0) {
        os_ << inner_ind << "if (" << sub_q0 << " + " << sub_rows << "L <= m1_Lq) {\n";
        emit_shm_bwd_dkdv_step(n_tiles, sub_m_tiles, sub_q_shm, sub_do_shm, sub_l_shm, sub_d_shm, sub_q0, true,
                               inner_ind + "  ");
        os_ << inner_ind << "} else {\n";
        emit_shm_bwd_dkdv_step(n_tiles, sub_m_tiles, sub_q_shm, sub_do_shm, sub_l_shm, sub_d_shm, sub_q0, false,
                               inner_ind + "  ");
        os_ << inner_ind << "}\n";
      } else {
        os_ << inner_ind << "if (" << sub_q0 << " >= m1_full_qs && " << sub_q0 << " + " << sub_rows
            << "L <= m1_full_qe) {\n";
        emit_shm_bwd_dkdv_step(n_tiles, sub_m_tiles, sub_q_shm, sub_do_shm, sub_l_shm, sub_d_shm, sub_q0, true,
                               inner_ind + "  ");
        os_ << inner_ind << "} else {\n";
        emit_shm_bwd_dkdv_step(n_tiles, sub_m_tiles, sub_q_shm, sub_do_shm, sub_l_shm, sub_d_shm, sub_q0, false,
                               inner_ind + "  ");
        os_ << inner_ind << "}\n";
      }
      if (sub_count > 1u) {
        os_ << ind << "    }\n";
      }
    }
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";

    os_ << ind << "if (m1_sg_active) {\n";
    for (uint32_t ni = 0; ni < n_tiles; ++ni) {
      os_ << ind << "  if (m1_sg_k0 + " << (ni * 8u) << "L < m1_Lkv) {\n";
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "    simdgroup_store(dk_" << ni << "_" << di << ", m1_dKb + (m1_sg_k0 + " << (ni * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
        os_ << ind << "    simdgroup_store(dv_" << ni << "_" << di << ", m1_dVb + (m1_sg_k0 + " << (ni * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
      }
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";
  }

  void emit_bwd_dkdv_general(const std::string& ind) {
    os_ << ind << "for (int m1_r = (int)m1_sg; m1_r < " << plan_.bn << "; m1_r += " << plan_.num_sg << ") {\n";
    os_ << ind << "  const long m1_kr = m1_k0 + (long)m1_r;\n";
    os_ << ind << "  if (m1_kr >= m1_Lkv) continue;\n";
    os_ << ind << "  device const " << plan_.elem_type << "* m1_krow = m1_Kb + m1_kr * m1_D;\n";
    os_ << ind << "  device const " << plan_.elem_type << "* m1_vrow = m1_Vb + m1_kr * m1_D;\n";
    os_ << ind << "  device " << plan_.elem_type << "* m1_dkrow = m1_dKb + m1_kr * m1_D;\n";
    os_ << ind << "  device " << plan_.elem_type << "* m1_dvrow = m1_dVb + m1_kr * m1_D;\n";
    os_ << ind << "  if (m1_D <= 256L) {\n";
    os_ << ind << "    float kv[8]; float vv[8]; float dka[8]; float dva[8];\n";
    os_ << ind << "    for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "      const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "      kv[i] = (d < m1_D) ? (float)m1_krow[d] : 0.0f;\n";
    os_ << ind << "      vv[i] = (d < m1_D) ? (float)m1_vrow[d] : 0.0f;\n";
    os_ << ind << "      dka[i] = 0.0f; dva[i] = 0.0f;\n";
    os_ << ind << "    }\n";
    os_ << ind << "    for (long m1_q = m1_q_start; m1_q < m1_q_end; ++m1_q) {\n";
    if (plan_.attr.causal) {
      os_ << ind << "      if (m1_kr > m1_q + " << plan_.attr.causal_offset << "L) continue;\n";
    }
    if (plan_.attr.window_size >= 0) {
      os_ << ind << "      if (m1_kr < m1_q - " << plan_.attr.window_size << "L) continue;\n";
    }
    os_ << ind << "      const float lval = m1_Lb[m1_q];\n";
    os_ << ind << "      if (lval <= -1e29f) continue;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_qrow = m1_Qb + m1_q * m1_D;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_dorow = m1_dOb + m1_q * m1_D;\n";
    os_ << ind << "      float sdot = 0.0f; float dpdot = 0.0f;\n";
    os_ << ind << "      for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "        const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "        if (d < m1_D) {\n";
    os_ << ind << "          sdot = fma((float)m1_qrow[d], kv[i], sdot);\n";
    os_ << ind << "          dpdot = fma((float)m1_dorow[d], vv[i], dpdot);\n";
    os_ << ind << "        }\n";
    os_ << ind << "      }\n";
    os_ << ind << "      float sc = simd_sum(sdot) * m1_scale;\n";
    os_ << ind << "      const float dp = simd_sum(dpdot);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float tv = m1_tanh(sc * m1_inv_softcap);\n";
      os_ << ind << "      sc = m1_softcap * tv;\n";
      os_ << ind << "      const float deriv = 1.0f - tv * tv;\n";
    }
    os_ << ind << "      const float p = exp2((sc - lval) * 1.4426950408889634f);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float ds = p * (dp - m1_Deltab[m1_q]) * deriv * m1_scale;\n";
    } else {
      os_ << ind << "      const float ds = p * (dp - m1_Deltab[m1_q]) * m1_scale;\n";
    }
    os_ << ind << "      for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "        const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "        if (d < m1_D) {\n";
    os_ << ind << "          dva[i] = fma(p, (float)m1_dorow[d], dva[i]);\n";
    os_ << ind << "          dka[i] = fma(ds, (float)m1_qrow[d], dka[i]);\n";
    os_ << ind << "        }\n";
    os_ << ind << "      }\n";
    os_ << ind << "    }\n";
    os_ << ind << "    for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "      const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "      if (d < m1_D) { m1_dkrow[d] = " << plan_.elem_type << "(dka[i]); m1_dvrow[d] = "
        << plan_.elem_type << "(dva[i]); }\n";
    os_ << ind << "    }\n";
    os_ << ind << "  } else {\n";
    os_ << ind << "    for (long d = (long)m1_lane; d < m1_D; d += 32L) { m1_dkrow[d] = " << plan_.elem_type
        << "(0.0f); m1_dvrow[d] = " << plan_.elem_type << "(0.0f); }\n";
    os_ << ind << "    for (long m1_q = m1_q_start; m1_q < m1_q_end; ++m1_q) {\n";
    if (plan_.attr.causal) {
      os_ << ind << "      if (m1_kr > m1_q + " << plan_.attr.causal_offset << "L) continue;\n";
    }
    if (plan_.attr.window_size >= 0) {
      os_ << ind << "      if (m1_kr < m1_q - " << plan_.attr.window_size << "L) continue;\n";
    }
    os_ << ind << "      const float lval = m1_Lb[m1_q];\n";
    os_ << ind << "      if (lval <= -1e29f) continue;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_qrow = m1_Qb + m1_q * m1_D;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_dorow = m1_dOb + m1_q * m1_D;\n";
    os_ << ind << "      float sdot = 0.0f; float dpdot = 0.0f;\n";
    os_ << ind << "      for (long d = (long)m1_lane; d < m1_D; d += 32L) {\n";
    os_ << ind << "        sdot = fma((float)m1_qrow[d], (float)m1_krow[d], sdot);\n";
    os_ << ind << "        dpdot = fma((float)m1_dorow[d], (float)m1_vrow[d], dpdot);\n";
    os_ << ind << "      }\n";
    os_ << ind << "      float sc = simd_sum(sdot) * m1_scale;\n";
    os_ << ind << "      const float dp = simd_sum(dpdot);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float tv = m1_tanh(sc * m1_inv_softcap);\n";
      os_ << ind << "      sc = m1_softcap * tv;\n";
      os_ << ind << "      const float deriv = 1.0f - tv * tv;\n";
    }
    os_ << ind << "      const float p = exp2((sc - lval) * 1.4426950408889634f);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float ds = p * (dp - m1_Deltab[m1_q]) * deriv * m1_scale;\n";
    } else {
      os_ << ind << "      const float ds = p * (dp - m1_Deltab[m1_q]) * m1_scale;\n";
    }
    os_ << ind << "      for (long d = (long)m1_lane; d < m1_D; d += 32L) {\n";
    os_ << ind << "        m1_dvrow[d] = " << plan_.elem_type << "(fma(p, (float)m1_dorow[d], (float)m1_dvrow[d]));\n";
    os_ << ind << "        m1_dkrow[d] = " << plan_.elem_type << "(fma(ds, (float)m1_qrow[d], (float)m1_dkrow[d]));\n";
    os_ << ind << "      }\n";
    os_ << ind << "    }\n";
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  void emit_bwd_dq_body() {
    const uint32_t bm = plan_.bm;
    const uint32_t kv_tile = (plan_.bn >= 32u) ? 32u : 16u;
    constexpr uint32_t kLd = 68u;
    const bool can_sgmm_64 =
        (plan_.elem_dt == ScalarDType::F32 && bm >= 8u * plan_.num_sg && (bm % (8u * plan_.num_sg)) == 0u &&
         (bm / (8u * plan_.num_sg)) <= 2u && (2u * kv_tile * kLd) * 4u <= plan_.max_smem);

    if (can_sgmm_64) {
      os_ << "  threadgroup float m1_shm[" << (2u * kv_tile * kLd) << "];\n";
    }
    os_ << "  const long m1_BH = " << plan_.bh_expr << ";\n";
    os_ << "  const long m1_Lq = " << plan_.lq_expr << ";\n";
    os_ << "  const long m1_Lkv = " << plan_.lkv_expr << ";\n";
    os_ << "  const long m1_D = " << plan_.d_expr << ";\n";
    os_ << "  const long m1_bh = (long)m1_tgid.y;\n";
    os_ << "  if (m1_bh >= m1_BH) return;\n";
    if (plan_.attr.causal) {
      os_ << "  const long m1_num_qblk = (m1_Lq + " << (bm - 1u) << "L) / " << bm << "L;\n";
      os_ << "  if ((long)m1_tgid.x >= m1_num_qblk) return;\n";
      os_ << "  const long m1_q0 = (m1_num_qblk - 1L - (long)m1_tgid.x) * " << bm << "L;\n";
    } else {
      os_ << "  const long m1_q0 = (long)m1_tgid.x * " << bm << "L;\n";
      os_ << "  if (m1_q0 >= m1_Lq) return;\n";
    }

    os_ << "  device const " << plan_.elem_type << "* m1_Qb = " << plan_.q_name << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Kb = " << plan_.k_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_Vb = " << plan_.v_name << " + m1_bh * (m1_Lkv * m1_D);\n";
    os_ << "  device const " << plan_.elem_type << "* m1_dOb = " << plan_.do_name << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  device const float* m1_Lb = " << plan_.lse_name << " + m1_bh * m1_Lq;\n";
    os_ << "  device const float* m1_Deltab = " << plan_.delta_name << " + m1_bh * m1_Lq;\n";
    os_ << "  device " << plan_.elem_type << "* m1_dQb = " << plan_.dq_name << " + m1_bh * (m1_Lq * m1_D);\n";
    os_ << "  const float m1_scale = " << format_float_literal(plan_.attr.scale) << ";\n";
    os_ << "  const float m1_scale_log2 = "
        << format_float_literal(static_cast<double>(plan_.attr.scale) * 1.44269504088896340736) << ";\n";
    os_ << "  (void)m1_scale_log2;\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << "  const float m1_softcap = " << format_float_literal(plan_.attr.softcap) << ";\n";
      os_ << "  const float m1_inv_softcap = " << format_float_literal(1.0 / static_cast<double>(plan_.attr.softcap))
          << ";\n";
    }
    if (plan_.attr.window_size >= 0) {
      os_ << "  const long m1_kv_start = max(0L, m1_q0 - " << plan_.attr.window_size << "L);\n";
    } else {
      os_ << "  const long m1_kv_start = 0L;\n";
    }
    if (plan_.attr.causal) {
      os_ << "  const long m1_kv_end = min(m1_Lkv, max(0L, m1_q0 + " << bm << "L + " << plan_.attr.causal_offset
          << "L));\n";
    } else {
      os_ << "  const long m1_kv_end = m1_Lkv;\n";
    }
    os_ << "  (void)m1_tid;\n";

    if (can_sgmm_64) {
      os_ << "  if (m1_D == 64L && m1_Lq > 0L && (m1_Lq & 7L) == 0L && m1_Lkv > 0L && (m1_Lkv & 7L) == 0L) {\n";
      emit_bwd_dq_sgmm64(kv_tile, "    ");
      os_ << "    return;\n";
      os_ << "  }\n";
    }

    emit_bwd_dq_general("  ");
  }

  void emit_shm_bwd_dq_step(uint32_t m_tiles, uint32_t n_tiles, const std::string& k_shm, const std::string& v_shm,
                            bool unmasked_fast, const std::string& ind) {
    constexpr uint32_t kLd = 68u;
    constexpr uint32_t kDt = 8u;
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        os_ << ind << "simdgroup_float8x8 s_" << mi << "_" << nj
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 dp_" << mi << "_" << nj
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 kt_cur, kt_nxt;\n";
    os_ << ind << "  simdgroup_load(kt_cur, " << k_shm << ", " << kLd << ", ulong2(0, 0), true);\n";
    for (uint32_t di = 0; di < kDt; ++di) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        const uint32_t next_nj = (nj + 1u < n_tiles) ? (nj + 1u) : 0u;
        const uint32_t next_di = (nj + 1u < n_tiles) ? di : (di + 1u);
        if (next_di < kDt) {
          os_ << ind << "  simdgroup_load(kt_nxt, " << k_shm << " + " << (next_nj * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ", ulong2(0, 0), true);\n";
        }
        for (uint32_t mi = 0; mi < m_tiles; ++mi) {
          os_ << ind << "  simdgroup_multiply_accumulate(s_" << mi << "_" << nj << ", q_" << mi << "_" << di
              << ", kt_cur, s_" << mi << "_" << nj << ");\n";
        }
        if (next_di < kDt) {
          os_ << ind << "  kt_cur = kt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 vt_cur, vt_nxt;\n";
    os_ << ind << "  simdgroup_load(vt_cur, " << v_shm << ", " << kLd << ", ulong2(0, 0), true);\n";
    for (uint32_t di = 0; di < kDt; ++di) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        const uint32_t next_nj = (nj + 1u < n_tiles) ? (nj + 1u) : 0u;
        const uint32_t next_di = (nj + 1u < n_tiles) ? di : (di + 1u);
        if (next_di < kDt) {
          os_ << ind << "  simdgroup_load(vt_nxt, " << v_shm << " + " << (next_nj * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ", ulong2(0, 0), true);\n";
        }
        for (uint32_t mi = 0; mi < m_tiles; ++mi) {
          os_ << ind << "  simdgroup_multiply_accumulate(dp_" << mi << "_" << nj << ", do_" << mi << "_" << di
              << ", vt_cur, dp_" << mi << "_" << nj << ");\n";
        }
        if (next_di < kDt) {
          os_ << ind << "  vt_cur = vt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t nj = 0; nj < n_tiles; ++nj) {
        os_ << ind << "{\n";
        os_ << ind << "  thread float2& sv = reinterpret_cast<thread float2&>(s_" << mi << "_" << nj
            << ".thread_elements());\n";
        os_ << ind << "  const float2 dpv = reinterpret_cast<thread float2&>(dp_" << mi << "_" << nj
            << ".thread_elements());\n";
        if (unmasked_fast) {
          if (plan_.attr.softcap > 0.0f) {
            os_ << ind << "  float2 sc = sv * m1_scale;\n";
            os_ << ind << "  const float2 tv = float2(m1_tanh(sc.x * m1_inv_softcap), m1_tanh(sc.y * m1_inv_softcap));\n";
            os_ << ind << "  sc = m1_softcap * tv;\n";
            os_ << ind << "  const float2 deriv = 1.0f - tv * tv;\n";
            os_ << ind << "  const float2 pv = (m1_lv_" << mi << " > -1e29f) ? exp2((sc - m1_lv_" << mi
                << ") * 1.4426950408889634f) : float2(0.0f);\n";
            os_ << ind << "  sv = pv * (dpv - m1_dv_" << mi << ") * deriv * m1_scale;\n";
          } else {
            os_ << ind << "  const float2 pv = exp2(fma(sv, m1_scale_log2, -m1_lv_log2_" << mi << "));\n";
            os_ << ind << "  sv = pv * (dpv - m1_dv_" << mi << ") * m1_scale;\n";
          }
        } else {
          os_ << ind << "  const long k0 = m1_kv0 + (long)(" << (nj * 8u) << " + m1_fc);\n";
          os_ << ind << "  const long k1 = k0 + 1L;\n";
          os_ << ind << "  const bool row_ok = (m1_qr_" << mi << " < m1_Lq) && (m1_lv_" << mi << " > -1e29f);\n";
          auto emit_e = [&](const char* comp, const char* kv) {
            std::string cond = "(row_ok && " + std::string(kv) + " < m1_kmax_" + std::to_string(mi);
            if (plan_.attr.window_size >= 0) {
              cond += " && " + std::string(kv) + " >= m1_kmin_" + std::to_string(mi);
            }
            cond += ")";
            os_ << ind << "  {\n";
            os_ << ind << "    const bool ok = " << cond << ";\n";
            if (plan_.attr.softcap > 0.0f) {
              os_ << ind << "    float sc = sv." << comp << " * m1_scale;\n";
              os_ << ind << "    const float tv = m1_tanh(sc * m1_inv_softcap);\n";
              os_ << ind << "    sc = m1_softcap * tv;\n";
              os_ << ind << "    const float deriv = 1.0f - tv * tv;\n";
              os_ << ind << "    const float pv = ok ? exp2((sc - m1_lv_" << mi << ") * 1.4426950408889634f) : 0.0f;\n";
              os_ << ind << "    sv." << comp << " = ok ? (pv * (dpv." << comp << " - m1_dv_" << mi
                  << ") * deriv * m1_scale) : 0.0f;\n";
            } else {
              os_ << ind << "    const float pv = ok ? exp2(fma(sv." << comp << ", m1_scale_log2, -m1_lv_log2_" << mi
                  << ")) : 0.0f;\n";
              os_ << ind << "    sv." << comp << " = ok ? (pv * (dpv." << comp << " - m1_dv_" << mi
                  << ") * m1_scale) : 0.0f;\n";
            }
            os_ << ind << "  }\n";
          };
          emit_e("x", "k0");
          emit_e("y", "k1");
        }
        os_ << ind << "}\n";
      }
    }

    os_ << ind << "{\n";
    os_ << ind << "  simdgroup_float8x8 kt_cur, kt_nxt;\n";
    os_ << ind << "  simdgroup_load(kt_cur, " << k_shm << ", " << kLd << ");\n";
    for (uint32_t nj = 0; nj < n_tiles; ++nj) {
      for (uint32_t di = 0; di < kDt; ++di) {
        const uint32_t next_di = (di + 1u < kDt) ? (di + 1u) : 0u;
        const uint32_t next_nj = (di + 1u < kDt) ? nj : (nj + 1u);
        if (next_nj < n_tiles) {
          os_ << ind << "  simdgroup_load(kt_nxt, " << k_shm << " + " << (next_nj * 8u * kLd + next_di * 8u) << "u, "
              << kLd << ");\n";
        }
        for (uint32_t mi = 0; mi < m_tiles; ++mi) {
          os_ << ind << "  simdgroup_multiply_accumulate(dq_" << mi << "_" << di << ", s_" << mi << "_" << nj
              << ", kt_cur, dq_" << mi << "_" << di << ");\n";
        }
        if (next_nj < n_tiles) {
          os_ << ind << "  kt_cur = kt_nxt;\n";
        }
      }
    }
    os_ << ind << "}\n";
  }

  void emit_bwd_dq_sgmm64(uint32_t kv_tile, const std::string& ind) {
    constexpr uint32_t kD = 64u;
    constexpr uint32_t kDt = 8u;
    constexpr uint32_t kLd = 68u;
    const uint32_t bm = plan_.bm;
    const uint32_t rows_per_sg = bm / plan_.num_sg;
    const uint32_t m_tiles = rows_per_sg / 8u;
    const uint32_t n_tiles = kv_tile / 8u;
    const std::string k_shm = "m1_shm";
    const std::string v_shm = "(m1_shm + " + std::to_string(kv_tile * kLd) + "u)";

    os_ << ind << "const int m1_qid = (int)m1_lane / 4;\n";
    os_ << ind << "const int m1_fr = (m1_qid & 4) + ((int)m1_lane / 2) % 4;\n";
    os_ << ind << "const int m1_fc = (m1_qid & 2) * 2 + ((int)m1_lane % 2) * 2;\n";
    os_ << ind << "const long m1_sg_q0 = m1_q0 + (long)(m1_sg * " << rows_per_sg << "u);\n";
    os_ << ind << "const bool m1_sg_active = (m1_sg_q0 < m1_Lq);\n";

    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "simdgroup_float8x8 q_" << mi << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 do_" << mi << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        os_ << ind << "simdgroup_float8x8 dq_" << mi << "_" << di
            << " = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
      os_ << ind << "const long m1_qr_" << mi << " = m1_sg_q0 + (long)(" << (mi * 8u) << " + m1_fr);\n";
      os_ << ind << "const float m1_lv_" << mi << " = (m1_qr_" << mi << " < m1_Lq) ? m1_Lb[m1_qr_" << mi
          << "] : -1e30f;\n";
      if (plan_.attr.softcap <= 0.0f) {
        os_ << ind << "const float m1_lv_log2_" << mi << " = (m1_lv_" << mi << " > -1e29f) ? (m1_lv_" << mi
            << " * 1.4426950408889634f) : 1e30f;\n";
      }
      os_ << ind << "const float m1_dv_" << mi << " = (m1_qr_" << mi << " < m1_Lq) ? m1_Deltab[m1_qr_" << mi
          << "] : 0.0f;\n";
      if (plan_.attr.causal) {
        os_ << ind << "const long m1_kmax_" << mi << " = min(m1_Lkv, max(0L, m1_qr_" << mi << " + 1L + "
            << plan_.attr.causal_offset << "L));\n";
      } else {
        os_ << ind << "const long m1_kmax_" << mi << " = m1_Lkv;\n";
      }
      if (plan_.attr.window_size >= 0) {
        os_ << ind << "const long m1_kmin_" << mi << " = max(0L, m1_qr_" << mi << " - " << plan_.attr.window_size
            << "L);\n";
      } else {
        os_ << ind << "const long m1_kmin_" << mi << " = 0L;\n";
      }
    }
    os_ << ind << "if (m1_sg_active) {\n";
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      os_ << ind << "  if (m1_sg_q0 + " << (mi * 8u) << "L < m1_Lq) {\n";
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "    simdgroup_load(q_" << mi << "_" << di << ", m1_Qb + (m1_sg_q0 + " << (mi * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
        os_ << ind << "    simdgroup_load(do_" << mi << "_" << di << ", m1_dOb + (m1_sg_q0 + " << (mi * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
      }
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";

    if (plan_.attr.window_size >= 0) {
      os_ << ind << "const long m1_sg_ks = max(0L, m1_sg_q0 - " << plan_.attr.window_size << "L);\n";
      os_ << ind << "const long m1_full_ks = max(0L, m1_sg_q0 + " << (rows_per_sg - 1u) << "L - "
          << plan_.attr.window_size << "L);\n";
    } else {
      os_ << ind << "const long m1_sg_ks = 0L;\n";
      os_ << ind << "const long m1_full_ks = 0L;\n";
    }
    if (plan_.attr.causal) {
      os_ << ind << "const long m1_sg_ke = min(m1_Lkv, max(0L, m1_sg_q0 + " << rows_per_sg << "L + "
          << plan_.attr.causal_offset << "L));\n";
      os_ << ind << "const long m1_full_ke = min(m1_Lkv, max(0L, m1_sg_q0 + 1L + " << plan_.attr.causal_offset
          << "L));\n";
    } else {
      os_ << ind << "const long m1_sg_ke = m1_Lkv;\n";
      os_ << ind << "const long m1_full_ke = m1_Lkv;\n";
    }

    os_ << ind << "for (long m1_kv0 = m1_kv_start; m1_kv0 < m1_kv_end; m1_kv0 += " << kv_tile << "L) {\n";
    emit_stage_tile_to_shm(ind + "  ", "m1_Kb", "m1_kv0", "m1_Lkv", k_shm, kv_tile, kD, kLd);
    emit_stage_tile_to_shm(ind + "  ", "m1_Vb", "m1_kv0", "m1_Lkv", v_shm, kv_tile, kD, kLd);
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "  if (m1_sg_active && m1_kv0 + " << kv_tile << "L > m1_sg_ks && m1_kv0 < m1_sg_ke) {\n";
    if (!plan_.attr.causal && plan_.attr.window_size < 0) {
      os_ << ind << "    if (m1_kv0 + " << kv_tile << "L <= m1_Lkv) {\n";
      emit_shm_bwd_dq_step(m_tiles, n_tiles, k_shm, v_shm, true, ind + "      ");
      os_ << ind << "    } else {\n";
      emit_shm_bwd_dq_step(m_tiles, n_tiles, k_shm, v_shm, false, ind + "      ");
      os_ << ind << "    }\n";
    } else {
      os_ << ind << "    if (m1_kv0 >= m1_full_ks && m1_kv0 + " << kv_tile << "L <= m1_full_ke) {\n";
      emit_shm_bwd_dq_step(m_tiles, n_tiles, k_shm, v_shm, true, ind + "      ");
      os_ << ind << "    } else {\n";
      emit_shm_bwd_dq_step(m_tiles, n_tiles, k_shm, v_shm, false, ind + "      ");
      os_ << ind << "    }\n";
    }
    os_ << ind << "  }\n";
    os_ << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << ind << "}\n";

    os_ << ind << "if (m1_sg_active) {\n";
    for (uint32_t mi = 0; mi < m_tiles; ++mi) {
      os_ << ind << "  if (m1_sg_q0 + " << (mi * 8u) << "L < m1_Lq) {\n";
      for (uint32_t di = 0; di < kDt; ++di) {
        os_ << ind << "    simdgroup_store(dq_" << mi << "_" << di << ", m1_dQb + (m1_sg_q0 + " << (mi * 8u)
            << "L) * 64L + " << (di * 8u) << "L, 64);\n";
      }
      os_ << ind << "  }\n";
    }
    os_ << ind << "}\n";
  }

  void emit_bwd_dq_general(const std::string& ind) {
    os_ << ind << "for (int m1_r = (int)m1_sg; m1_r < " << plan_.bm << "; m1_r += " << plan_.num_sg << ") {\n";
    os_ << ind << "  const long m1_qr = m1_q0 + (long)m1_r;\n";
    os_ << ind << "  if (m1_qr >= m1_Lq) continue;\n";
    os_ << ind << "  const long m1_kmin = " << kmin_expr("m1_r") << ";\n";
    os_ << ind << "  const long m1_kmax = " << kmax_expr("m1_r") << ";\n";
    os_ << ind << "  const float lval = m1_Lb[m1_qr];\n";
    os_ << ind << "  const float dval = m1_Deltab[m1_qr];\n";
    os_ << ind << "  device const " << plan_.elem_type << "* m1_qrow = m1_Qb + m1_qr * m1_D;\n";
    os_ << ind << "  device const " << plan_.elem_type << "* m1_dorow = m1_dOb + m1_qr * m1_D;\n";
    os_ << ind << "  device " << plan_.elem_type << "* m1_dqrow = m1_dQb + m1_qr * m1_D;\n";
    os_ << ind << "  if (m1_kmax <= m1_kmin || lval <= -1e29f) {\n";
    os_ << ind << "    for (long d = (long)m1_lane; d < m1_D; d += 32L) m1_dqrow[d] = " << plan_.elem_type
        << "(0.0f);\n";
    os_ << ind << "    continue;\n";
    os_ << ind << "  }\n";
    os_ << ind << "  if (m1_D <= 256L) {\n";
    os_ << ind << "    float qv[8]; float dov[8]; float dqa[8];\n";
    os_ << ind << "    for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "      const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "      qv[i] = (d < m1_D) ? (float)m1_qrow[d] : 0.0f;\n";
    os_ << ind << "      dov[i] = (d < m1_D) ? (float)m1_dorow[d] : 0.0f;\n";
    os_ << ind << "      dqa[i] = 0.0f;\n";
    os_ << ind << "    }\n";
    os_ << ind << "    for (long m1_k = m1_kmin; m1_k < m1_kmax; ++m1_k) {\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_krow = m1_Kb + m1_k * m1_D;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_vrow = m1_Vb + m1_k * m1_D;\n";
    os_ << ind << "      float sdot = 0.0f; float dpdot = 0.0f;\n";
    os_ << ind << "      for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "        const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "        if (d < m1_D) {\n";
    os_ << ind << "          sdot = fma(qv[i], (float)m1_krow[d], sdot);\n";
    os_ << ind << "          dpdot = fma(dov[i], (float)m1_vrow[d], dpdot);\n";
    os_ << ind << "        }\n";
    os_ << ind << "      }\n";
    os_ << ind << "      float sc = simd_sum(sdot) * m1_scale;\n";
    os_ << ind << "      const float dp = simd_sum(dpdot);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float tv = m1_tanh(sc * m1_inv_softcap);\n";
      os_ << ind << "      sc = m1_softcap * tv;\n";
      os_ << ind << "      const float deriv = 1.0f - tv * tv;\n";
    }
    os_ << ind << "      const float p = exp2((sc - lval) * 1.4426950408889634f);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float ds = p * (dp - dval) * deriv * m1_scale;\n";
    } else {
      os_ << ind << "      const float ds = p * (dp - dval) * m1_scale;\n";
    }
    os_ << ind << "      for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "        const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "        if (d < m1_D) dqa[i] = fma(ds, (float)m1_krow[d], dqa[i]);\n";
    os_ << ind << "      }\n";
    os_ << ind << "    }\n";
    os_ << ind << "    for (int i = 0; i < 8; ++i) {\n";
    os_ << ind << "      const long d = (long)m1_lane + (long)i * 32L;\n";
    os_ << ind << "      if (d < m1_D) m1_dqrow[d] = " << plan_.elem_type << "(dqa[i]);\n";
    os_ << ind << "    }\n";
    os_ << ind << "  } else {\n";
    os_ << ind << "    for (long d = (long)m1_lane; d < m1_D; d += 32L) m1_dqrow[d] = " << plan_.elem_type
        << "(0.0f);\n";
    os_ << ind << "    for (long m1_k = m1_kmin; m1_k < m1_kmax; ++m1_k) {\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_krow = m1_Kb + m1_k * m1_D;\n";
    os_ << ind << "      device const " << plan_.elem_type << "* m1_vrow = m1_Vb + m1_k * m1_D;\n";
    os_ << ind << "      float sdot = 0.0f; float dpdot = 0.0f;\n";
    os_ << ind << "      for (long d = (long)m1_lane; d < m1_D; d += 32L) {\n";
    os_ << ind << "        sdot = fma((float)m1_qrow[d], (float)m1_krow[d], sdot);\n";
    os_ << ind << "        dpdot = fma((float)m1_dorow[d], (float)m1_vrow[d], dpdot);\n";
    os_ << ind << "      }\n";
    os_ << ind << "      float sc = simd_sum(sdot) * m1_scale;\n";
    os_ << ind << "      const float dp = simd_sum(dpdot);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float tv = m1_tanh(sc * m1_inv_softcap);\n";
      os_ << ind << "      sc = m1_softcap * tv;\n";
      os_ << ind << "      const float deriv = 1.0f - tv * tv;\n";
    }
    os_ << ind << "      const float p = exp2((sc - lval) * 1.4426950408889634f);\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << ind << "      const float ds = p * (dp - dval) * deriv * m1_scale;\n";
    } else {
      os_ << ind << "      const float ds = p * (dp - dval) * m1_scale;\n";
    }
    os_ << ind << "      for (long d = (long)m1_lane; d < m1_D; d += 32L) {\n";
    os_ << ind << "        m1_dqrow[d] = " << plan_.elem_type << "(fma(ds, (float)m1_krow[d], (float)m1_dqrow[d]));\n";
    os_ << ind << "      }\n";
    os_ << ind << "    }\n";
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  void emit_paged_decode_body() {
    const uint32_t num_sg = plan_.num_sg;
    const uint32_t blk_sz = plan_.paged_block_size;

    os_ << "  threadgroup float m1_m_shm[" << num_sg << "];\n";
    os_ << "  threadgroup float m1_l_shm[" << num_sg << "];\n";
    os_ << "  threadgroup float m1_o_shm[" << (num_sg * 128u) << "];\n";
    os_ << "  const long m1_B = " << plan_.b_expr << ";\n";
    os_ << "  const long m1_Hq = " << plan_.hq_expr << ";\n";
    os_ << "  const long m1_Hkv = " << plan_.hkv_expr << ";\n";
    os_ << "  const long m1_MaxBlocks = " << plan_.max_blocks_expr << ";\n";
    os_ << "  const long m1_D = " << plan_.d_expr << ";\n";
    if (plan_.paged_num_splits > 0u) {
      os_ << "  const long m1_num_splits = " << plan_.paged_num_splits << "L;\n";
    } else {
      os_ << "  const long m1_num_splits = (long)m1_ntg.x;\n";
    }
    os_ << "  const long m1_split = (long)m1_tgid.x;\n";
    os_ << "  const long m1_bh = (long)m1_tgid.y;\n";
    os_ << "  if (m1_split >= m1_num_splits || m1_bh >= m1_B * m1_Hq) return;\n";
    os_ << "  const long m1_b = m1_bh / m1_Hq;\n";
    os_ << "  const long m1_hq = m1_bh % m1_Hq;\n";
    os_ << "  const long m1_gqa = max(1L, m1_Hq / m1_Hkv);\n";
    os_ << "  const long m1_hkv = m1_hq / m1_gqa;\n";
    os_ << "  const long m1_ctx_len = (long)" << plan_.context_lens_name << "[m1_b];\n";
    os_ << "  const long m1_tps = (m1_ctx_len + m1_num_splits - 1L) / m1_num_splits;\n";
    os_ << "  const long m1_t0 = min(m1_ctx_len, m1_split * m1_tps);\n";
    os_ << "  const long m1_t1 = min(m1_ctx_len, m1_t0 + m1_tps);\n";
    os_ << "  device float* m1_po = " << plan_.partial_o_name << " + (m1_bh * m1_num_splits + m1_split) * m1_D;\n";
    os_ << "  device float* m1_plse = " << plan_.partial_lse_name << " + (m1_bh * m1_num_splits + m1_split) * 2L;\n";
    os_ << "  if (m1_t0 >= m1_t1) {\n";
    os_ << "    for (long d = (long)m1_tid; d < m1_D; d += " << plan_.tg_threads << "L) m1_po[d] = 0.0f;\n";
    os_ << "    if (m1_tid == 0u) { m1_plse[0] = -1e30f; m1_plse[1] = 0.0f; }\n";
    os_ << "    return;\n";
    os_ << "  }\n";
    os_ << "  device const " << plan_.elem_type << "* m1_qrow = " << plan_.q_name << " + m1_bh * m1_D;\n";
    os_ << "  const float m1_scale = " << format_float_literal(plan_.attr.scale) << ";\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << "  const float m1_softcap = " << format_float_literal(plan_.attr.softcap) << ";\n";
      os_ << "  const float m1_inv_softcap = " << format_float_literal(1.0 / static_cast<double>(plan_.attr.softcap))
          << ";\n";
    }

    auto emit_paged_vec_path = [&](uint32_t d_val, uint32_t vec_w) {
      const char* vtype = (vec_w == 2u) ? "float2" : "float4";
      const char* ptype = (vec_w == 2u) ? "packed_float2" : "packed_float4";
      os_ << "  if (m1_D == " << d_val << "L) {\n";
      if (plan_.elem_dt == ScalarDType::F32) {
        os_ << "    const " << vtype << " qv = " << vtype << "(*(device const " << ptype
            << "*)(m1_qrow + (long)(m1_lane * " << vec_w << "u)));\n";
      } else if (vec_w == 2u) {
        os_ << "    const float2 qv = float2((float)m1_qrow[m1_lane * 2u], (float)m1_qrow[m1_lane * 2u + 1u]);\n";
      } else {
        os_ << "    const float4 qv = float4((float)m1_qrow[m1_lane * 4u], (float)m1_qrow[m1_lane * 4u + 1u], "
               "(float)m1_qrow[m1_lane * 4u + 2u], (float)m1_qrow[m1_lane * 4u + 3u]);\n";
      }
      os_ << "    " << vtype << " acc = " << vtype << "(0.0f);\n";
      os_ << "    float m_sg = -1e30f;\n";
      os_ << "    float l_sg = 0.0f;\n";
      os_ << "    for (long tok = m1_t0 + (long)m1_sg; tok < m1_t1; tok += " << num_sg << "L) {\n";
      os_ << "      const long blk_idx = tok / " << blk_sz << "L;\n";
      os_ << "      const long blk_off = tok % " << blk_sz << "L;\n";
      os_ << "      const long phys_blk = (long)" << plan_.block_tables_name << "[m1_b * m1_MaxBlocks + blk_idx];\n";
      os_ << "      const long base = ((phys_blk * m1_Hkv + m1_hkv) * " << blk_sz << "L + blk_off) * " << d_val
          << "L + (long)(m1_lane * " << vec_w << "u);\n";
      if (plan_.elem_dt == ScalarDType::F32) {
        os_ << "      const " << vtype << " kv = " << vtype << "(*(device const " << ptype << "*)(" << plan_.k_name
            << " + base));\n";
        os_ << "      const " << vtype << " vv = " << vtype << "(*(device const " << ptype << "*)(" << plan_.v_name
            << " + base));\n";
      } else if (vec_w == 2u) {
        os_ << "      const float2 kv = float2((float)" << plan_.k_name << "[base], (float)" << plan_.k_name
            << "[base + 1L]);\n";
        os_ << "      const float2 vv = float2((float)" << plan_.v_name << "[base], (float)" << plan_.v_name
            << "[base + 1L]);\n";
      } else {
        os_ << "      const float4 kv = float4((float)" << plan_.k_name << "[base], (float)" << plan_.k_name
            << "[base + 1L], (float)" << plan_.k_name << "[base + 2L], (float)" << plan_.k_name << "[base + 3L]);\n";
        os_ << "      const float4 vv = float4((float)" << plan_.v_name << "[base], (float)" << plan_.v_name
            << "[base + 1L], (float)" << plan_.v_name << "[base + 2L], (float)" << plan_.v_name << "[base + 3L]);\n";
      }
      if (vec_w == 2u) {
        os_ << "      float sc = simd_sum(fma(qv.x, kv.x, qv.y * kv.y)) * m1_scale;\n";
      } else {
        os_ << "      float sc = simd_sum(fma(qv.x, kv.x, fma(qv.y, kv.y, fma(qv.z, kv.z, qv.w * kv.w)))) * m1_scale;\n";
      }
      if (plan_.attr.softcap > 0.0f) {
        os_ << "      sc = m1_softcap * m1_tanh(sc * m1_inv_softcap);\n";
      }
      os_ << "      const float m_new = max(m_sg, sc);\n";
      os_ << "      const float alpha = (m_sg > -1e29f) ? exp(m_sg - m_new) : 0.0f;\n";
      os_ << "      const float p = exp(sc - m_new);\n";
      os_ << "      l_sg = l_sg * alpha + p;\n";
      os_ << "      m_sg = m_new;\n";
      os_ << "      acc = fma(" << vtype << "(p), vv, acc * alpha);\n";
      os_ << "    }\n";
      os_ << "    if (m1_lane == 0u) { m1_m_shm[m1_sg] = m_sg; m1_l_shm[m1_sg] = l_sg; }\n";
      os_ << "    *(threadgroup " << vtype << "*)(m1_o_shm + m1_sg * " << d_val << "u + m1_lane * " << vec_w
          << "u) = acc;\n";
      os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "    if (m1_sg == 0u) {\n";
      os_ << "      float m_tot = -1e30f;\n";
      os_ << "      for (uint s = 0u; s < " << num_sg << "u; ++s) {\n";
      os_ << "        if (m1_l_shm[s] > 0.0f) m_tot = max(m_tot, m1_m_shm[s]);\n";
      os_ << "      }\n";
      os_ << "      float l_tot = 0.0f;\n";
      os_ << "      " << vtype << " o_tot = " << vtype << "(0.0f);\n";
      os_ << "      for (uint s = 0u; s < " << num_sg << "u; ++s) {\n";
      os_ << "        if (m1_l_shm[s] > 0.0f) {\n";
      os_ << "          const float w = exp(m1_m_shm[s] - m_tot);\n";
      os_ << "          l_tot += m1_l_shm[s] * w;\n";
      os_ << "          o_tot = fma(" << vtype << "(w), *(threadgroup const " << vtype << "*)(m1_o_shm + s * " << d_val
          << "u + m1_lane * " << vec_w << "u), o_tot);\n";
      os_ << "        }\n";
      os_ << "      }\n";
      os_ << "      const float inv = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;\n";
      os_ << "      *(device " << ptype << "*)(m1_po + (long)(m1_lane * " << vec_w << "u)) = " << ptype
          << "(o_tot * inv);\n";
      os_ << "      if (m1_lane == 0u) { m1_plse[0] = m_tot; m1_plse[1] = l_tot; }\n";
      os_ << "    }\n";
      os_ << "    return;\n";
      os_ << "  }\n";
    };

    emit_paged_vec_path(64u, 2u);
    emit_paged_vec_path(128u, 4u);

    // General D fallback (2-pass exact online softmax across arbitrary head dimension D)
    os_ << "  {\n";
    os_ << "    float m_sg = -1e30f;\n";
    os_ << "    float l_sg = 0.0f;\n";
    os_ << "    for (long tok = m1_t0 + (long)m1_sg; tok < m1_t1; tok += " << num_sg << "L) {\n";
    os_ << "      const long blk_idx = tok / " << blk_sz << "L;\n";
    os_ << "      const long blk_off = tok % " << blk_sz << "L;\n";
    os_ << "      const long phys_blk = (long)" << plan_.block_tables_name << "[m1_b * m1_MaxBlocks + blk_idx];\n";
    os_ << "      device const " << plan_.elem_type << "* kptr = " << plan_.k_name
        << " + ((phys_blk * m1_Hkv + m1_hkv) * " << blk_sz << "L + blk_off) * m1_D;\n";
    os_ << "      float dot = 0.0f;\n";
    os_ << "      for (long d = (long)m1_lane; d < m1_D; d += 32L) dot = fma((float)m1_qrow[d], (float)kptr[d], dot);\n";
    os_ << "      float sc = simd_sum(dot) * m1_scale;\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << "      sc = m1_softcap * m1_tanh(sc * m1_inv_softcap);\n";
    }
    os_ << "      const float m_new = max(m_sg, sc);\n";
    os_ << "      const float alpha = (m_sg > -1e29f) ? exp(m_sg - m_new) : 0.0f;\n";
    os_ << "      l_sg = l_sg * alpha + exp(sc - m_new);\n";
    os_ << "      m_sg = m_new;\n";
    os_ << "    }\n";
    os_ << "    if (m1_lane == 0u) { m1_m_shm[m1_sg] = m_sg; m1_l_shm[m1_sg] = l_sg; }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    float m_tot = -1e30f;\n";
    os_ << "    for (uint s = 0u; s < " << num_sg << "u; ++s) { if (m1_l_shm[s] > 0.0f) m_tot = max(m_tot, m1_m_shm[s]); }\n";
    os_ << "    float l_tot = 0.0f;\n";
    os_ << "    for (uint s = 0u; s < " << num_sg << "u; ++s) { if (m1_l_shm[s] > 0.0f) l_tot += m1_l_shm[s] * exp(m1_m_shm[s] - m_tot); }\n";
    os_ << "    const float inv = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;\n";
    os_ << "    for (long d0 = 0L; d0 < m1_D; d0 += 128L) {\n";
    os_ << "      const long dlen = min(128L, m1_D - d0);\n";
    os_ << "      for (long i = (long)m1_lane; i < dlen; i += 32L) m1_o_shm[m1_sg * 128u + (uint)i] = 0.0f;\n";
    os_ << "      for (long tok = m1_t0 + (long)m1_sg; tok < m1_t1; tok += " << num_sg << "L) {\n";
    os_ << "        const long blk_idx = tok / " << blk_sz << "L;\n";
    os_ << "        const long blk_off = tok % " << blk_sz << "L;\n";
    os_ << "        const long phys_blk = (long)" << plan_.block_tables_name << "[m1_b * m1_MaxBlocks + blk_idx];\n";
    os_ << "        const long tok_base = ((phys_blk * m1_Hkv + m1_hkv) * " << blk_sz << "L + blk_off) * m1_D;\n";
    os_ << "        device const " << plan_.elem_type << "* kptr = " << plan_.k_name << " + tok_base;\n";
    os_ << "        device const " << plan_.elem_type << "* vptr = " << plan_.v_name << " + tok_base + d0;\n";
    os_ << "        float dot = 0.0f;\n";
    os_ << "        for (long d = (long)m1_lane; d < m1_D; d += 32L) dot = fma((float)m1_qrow[d], (float)kptr[d], dot);\n";
    os_ << "        float sc = simd_sum(dot) * m1_scale;\n";
    if (plan_.attr.softcap > 0.0f) {
      os_ << "        sc = m1_softcap * m1_tanh(sc * m1_inv_softcap);\n";
    }
    os_ << "        const float p = exp(sc - m_tot);\n";
    os_ << "        for (long i = (long)m1_lane; i < dlen; i += 32L) {\n";
    os_ << "          m1_o_shm[m1_sg * 128u + (uint)i] = fma(p, (float)vptr[i], m1_o_shm[m1_sg * 128u + (uint)i]);\n";
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "      if (m1_sg == 0u) {\n";
    os_ << "        for (long i = (long)m1_lane; i < dlen; i += 32L) {\n";
    os_ << "          float s_acc = 0.0f;\n";
    os_ << "          for (uint s = 0u; s < " << num_sg << "u; ++s) s_acc += m1_o_shm[s * 128u + (uint)i];\n";
    os_ << "          m1_po[d0 + i] = s_acc * inv;\n";
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";
    os_ << "    if (m1_tid == 0u) { m1_plse[0] = m_tot; m1_plse[1] = l_tot; }\n";
    os_ << "  }\n";
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
  res.launch_spec.param_slots = plan->param_slots;
  return res;
}

}  // namespace m1triton::codegen::metal

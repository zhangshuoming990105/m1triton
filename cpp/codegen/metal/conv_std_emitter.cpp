#include "codegen/metal/conv_std_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
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
  int winograd_mode = 0;
  uint32_t wino_th = 4;
  uint32_t wino_tw = 8;
  uint32_t wino_vld = 36;
  uint32_t wino_eld = 36;
  std::optional<ir::EpilogueDAGAttr> epilogue_dag;
  std::optional<ir::FusedTailAttr> fused_tail;
  HelperFlags helpers;
};

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

bool valid_tile(uint32_t v) { return v == 16 || v == 32 || v == 64 || v == 128; }
bool valid_bk(uint32_t v) { return v == 8 || v == 16 || v == 32; }

std::optional<std::string> float_ptr(const Kernel& kernel, ValueId vid, std::string& name) {
  const auto idx = param_index_of(kernel, vid);
  if (!idx || !kernel.params[*idx].type.is_ptr()) return std::nullopt;
  const ScalarDType dt = std::get<ir::PtrType>(kernel.params[*idx].type.value).pointee;
  if (dt != ScalarDType::F32 && dt != ScalarDType::F16) return std::nullopt;
  name = kernel.params[*idx].name;
  return msl_scalar_type(dt);
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
    auto e = scalar_long_expr(kernel, ops[i], false);
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

  if (plan.op->hints.epilogue_dag && !plan.op->hints.epilogue_dag->steps.empty()) {
    plan.epilogue_dag = *plan.op->hints.epilogue_dag;
  } else if (kernel.hints.epilogue_dag && !kernel.hints.epilogue_dag->steps.empty()) {
    plan.epilogue_dag = *kernel.hints.epilogue_dag;
  }
  if (plan.op->hints.fused_tail && plan.op->hints.fused_tail->kind != ir::TailKind::None) {
    plan.fused_tail = *plan.op->hints.fused_tail;
  } else if (kernel.hints.fused_tail && kernel.hints.fused_tail->kind != ir::TailKind::None) {
    plan.fused_tail = *kernel.hints.fused_tail;
  }

  if (plan.epilogue_dag) {
    std::set<uint32_t> read_params;
    collect_epilogue_dag_param_indices(*plan.epilogue_dag, read_params);
    for (uint32_t pidx : read_params) {
      if (pidx >= kernel.params.size()) {
        err = "implicit_gemm_conv epilogue_dag references out-of-range param index";
        return std::nullopt;
      }
    }
  }
  if (plan.fused_tail) {
    const auto tk = plan.fused_tail->kind;
    if (tk != ir::TailKind::SpatialGlobalAvgPool && tk != ir::TailKind::SpatialGlobalSum &&
        tk != ir::TailKind::SpatialGlobalMax) {
      err = "implicit_gemm_conv fused_tail only supports spatial_global_{avg_pool,sum,max}";
      return std::nullopt;
    }
  }

  if (auto wv = positive_constexpr_param(kernel, "WINOGRAD"); wv && (*wv == 10 || *wv == 11 || *wv == 12)) {
    if (nd != 2 || plan.attr.groups != 1 || plan.dims[1].k != 3 || plan.dims[2].k != 3 ||
        plan.dims[1].s != 1 || plan.dims[2].s != 1 || plan.dims[1].d != 1 || plan.dims[2].d != 1 ||
        plan.x_elem != "float" || plan.w_elem != "float" || plan.out_elem != "float") {
      err = "implicit_gemm_conv WINOGRAD 3-stage requires 2D f32 3x3 s=1 d=1 groups=1";
      return std::nullopt;
    }
    if (plan.fused_tail) {
      err = "implicit_gemm_conv fused_tail is not supported on 3-stage Winograd F(4x4, 3x3)";
      return std::nullopt;
    }
    plan.winograd_mode = static_cast<int>(*wv);
    if (plan.winograd_mode == 12 && plan.epilogue_dag) {
      collect_epilogue_dag_helpers(*plan.epilogue_dag, plan.helpers);
    }
    if (plan.winograd_mode == 10) {
      plan.num_threads = 64;
      plan.num_sgs = 2;
      plan.smem_floats = 0;
    } else if (plan.winograd_mode == 12) {
      plan.num_threads = 64;
      plan.num_sgs = 2;
      plan.smem_floats = 0;
    } else {
      plan.bm = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_M").value_or(128));
      plan.bn = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_N").value_or(64));
      plan.bk = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_K").value_or(32));
      plan.num_sgs = static_cast<uint32_t>(positive_constexpr_param(kernel, "NUM_SG").value_or(8));
      if (plan.num_sgs == 8) {
        if (plan.bm >= plan.bn && plan.bm >= 32) {
          plan.sg_m = 4;
          plan.sg_n = 2;
        } else {
          plan.sg_m = 2;
          plan.sg_n = 4;
        }
      } else {
        plan.num_sgs = 4;
        if (plan.bm == 32 && plan.bn == 128) {
          plan.sg_m = 1;
          plan.sg_n = 4;
        } else if (plan.bm == 128 && plan.bn == 32) {
          plan.sg_m = 4;
          plan.sg_n = 1;
        } else {
          plan.sg_m = 2;
          plan.sg_n = 2;
        }
      }
      plan.num_threads = plan.num_sgs * 32u;
      if (!valid_tile(plan.bm) || !valid_tile(plan.bn) || !valid_bk(plan.bk) ||
          plan.bm < plan.sg_m * 8 || plan.bn < plan.sg_n * 8 ||
          (plan.bm / (plan.sg_m * 8)) * (plan.bn / (plan.sg_n * 8)) > kMaxFragsPerSimdgroup) {
        err = "implicit_gemm_conv WINOGRAD=11 BMM tile unsupported";
        return std::nullopt;
      }
      plan.smem_floats = plan.bm * (plan.bk + 4u) + plan.bk * (plan.bn + 4u);
      if (plan.smem_floats * 4u > target.max_threadgroup_memory_bytes) {
        err = "implicit_gemm_conv WINOGRAD=11 BMM exceeds SRAM limit";
        return std::nullopt;
      }
    }
    return plan;
  }

  if (auto wv = positive_constexpr_param(kernel, "WINOGRAD"); wv && *wv == 1) {
    if (nd != 2 || plan.attr.groups != 1 || plan.dims[1].k != 3 || plan.dims[2].k != 3 ||
        plan.dims[1].s != 1 || plan.dims[2].s != 1 || plan.dims[1].d != 1 || plan.dims[2].d != 1 ||
        plan.x_elem != "float" || plan.w_elem != "float" || plan.out_elem != "float") {
      err = "implicit_gemm_conv WINOGRAD requires 2D f32 3x3 s=1 d=1 groups=1";
      return std::nullopt;
    }
    plan.winograd_mode = 1;
    constexpr uint32_t n_alpha = 16u;
    plan.bm = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_M").value_or(32));
    plan.bn = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_N").value_or(32));
    plan.bk = static_cast<uint32_t>(positive_constexpr_param(kernel, "BLOCK_K").value_or(16));
    plan.wino_th = static_cast<uint32_t>(positive_constexpr_param(kernel, "TILE_TH").value_or(2));
    plan.wino_tw = static_cast<uint32_t>(positive_constexpr_param(kernel, "TILE_TW").value_or(16));
    plan.num_sgs = static_cast<uint32_t>(positive_constexpr_param(kernel, "NUM_SG").value_or(8));
    plan.num_threads = plan.num_sgs * 32u;
    if (plan.bm == 0 || plan.bn == 0 || plan.bk == 0 || (plan.bm % 8u) != 0 || (plan.bn % 8u) != 0 ||
        (plan.bk % 8u) != 0 || plan.wino_th * plan.wino_tw != plan.bn || (n_alpha % plan.num_sgs) != 0) {
      err = "implicit_gemm_conv WINOGRAD tile configuration unsupported";
      return std::nullopt;
    }
    const uint32_t alphas_per_sg = n_alpha / plan.num_sgs;
    const uint32_t max_accs = (plan.num_threads >= 512u) ? 16u : 32u;
    if (alphas_per_sg * (plan.bm / 8u) * (plan.bn / 8u) > max_accs) {
      err = "implicit_gemm_conv WINOGRAD exceeds register accumulator limit";
      return std::nullopt;
    }
    const bool pad_smem = positive_constexpr_param(kernel, "WINO_PAD").value_or(1) > 0;
    const uint32_t pad_add = (plan.bn % 8u == 0u) ? 4u : 0u;
    plan.wino_vld =
        (pad_smem && n_alpha * plan.bk * (plan.bn + pad_add) * 4u <= target.max_threadgroup_memory_bytes)
            ? (plan.bn + pad_add)
            : plan.bn;
    plan.wino_eld =
        (pad_smem && n_alpha * 8u * (plan.bn + pad_add) * 4u <= target.max_threadgroup_memory_bytes)
            ? (plan.bn + pad_add)
            : plan.bn;
    plan.smem_floats = std::max(n_alpha * plan.bk * plan.wino_vld, n_alpha * 8u * plan.wino_eld);
    if (plan.fused_tail && 8u * plan.bn > plan.num_threads) {
      err = "implicit_gemm_conv WINOGRAD fused_tail requires 8 * BLOCK_N <= num_threads";
      return std::nullopt;
    }
    if (plan.smem_floats * 4u > target.max_threadgroup_memory_bytes ||
        plan.num_threads > target.max_threads_per_threadgroup) {
      err = "implicit_gemm_conv WINOGRAD exceeds threadgroup limits";
      return std::nullopt;
    }
    if (plan.epilogue_dag) {
      collect_epilogue_dag_helpers(*plan.epilogue_dag, plan.helpers);
    }
    return plan;
  }

  plan.bm = ca->bm;
  plan.bn = ca->bn;
  plan.bk = ca->bk;
  plan.num_sgs = ca->num_simdgroups == 8 ? 8u : 4u;
  if (auto v = positive_constexpr_param(kernel, "BLOCK_M")) plan.bm = static_cast<uint32_t>(*v);
  if (auto v = positive_constexpr_param(kernel, "BLOCK_N")) plan.bn = static_cast<uint32_t>(*v);
  if (auto v = positive_constexpr_param(kernel, "BLOCK_K")) plan.bk = static_cast<uint32_t>(*v);
  if (auto v = positive_constexpr_param(kernel, "NUM_SG")) plan.num_sgs = (*v >= 8) ? 8u : 4u;
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
  const uint32_t tail_smem = plan.fused_tail ? (plan.bm * plan.sg_n) : 0u;
  if ((plan.smem_floats + tail_smem) * 4u > target.max_threadgroup_memory_bytes ||
      plan.num_threads > target.max_threads_per_threadgroup) {
    err = "implicit_gemm_conv tile exceeds threadgroup limits";
    return std::nullopt;
  }
  if (plan.epilogue_dag) {
    collect_epilogue_dag_helpers(*plan.epilogue_dag, plan.helpers);
  }
  return plan;
}

class ConvStdWriter {
 public:
  ConvStdWriter(const Kernel& kernel, const ConvPlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    if (plan_.winograd_mode == 1) {
      return emit_winograd_2x3();
    }
    if (plan_.winograd_mode == 10) {
      return emit_wino4x3_input();
    }
    if (plan_.winograd_mode == 11) {
      return emit_wino4x3_bmm();
    }
    if (plan_.winograd_mode == 12) {
      return emit_wino4x3_output();
    }
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
  std::string emit_wino4x3_input() {
    const ConvDim& H = plan_.dims[1];
    const ConvDim& W = plan_.dims[2];
    emit_signature();
    os_ << "  (void)m1_sg; (void)" << plan_.w_name << ";\n";
    os_ << "  constexpr int m1_PH = " << H.p << ", m1_PW = " << W.p << ";\n";
    os_ << "  const int m1_Cin = (int)" << plan_.cin_expr << ";\n";
    os_ << "  const int m1_Hin = (int)" << H.in << ", m1_Win = (int)" << W.in << ";\n";
    os_ << "  const int m1_Hout = (int)" << H.out << ", m1_Wout = (int)" << W.out << ";\n";
    os_ << "  const int m1_Th = (m1_Hout + 3) >> 2;\n";
    os_ << "  const int m1_Tw = (m1_Wout + 3) >> 2;\n";
    os_ << "  const int m1_tx = (int)m1_tgid.x * 64 + (int)m1_tid;\n";
    os_ << "  const int m1_rloc = (int)m1_tgid.y;\n";
    os_ << "  const int m1_ci = (int)m1_tgid.z;\n";
    os_ << "  if (m1_ci >= m1_Cin || m1_rloc >= chunk_rows || m1_tx >= m1_Tw) return;\n";
    os_ << "  const int m1_rglob = row_start + m1_rloc;\n";
    os_ << "  const int m1_n = m1_rglob / m1_Th;\n";
    os_ << "  const int m1_ty = m1_rglob - m1_n * m1_Th;\n";
    os_ << "  const int m1_ih0 = m1_ty * 4 - m1_PH;\n";
    os_ << "  const int m1_iw0 = m1_tx * 4 - m1_PW;\n";
    os_ << "  const long m1_Pin = (long)m1_Hin * (long)m1_Win;\n";
    os_ << "  device const float* m1_xc = " << plan_.x_name << " + ((long)m1_n * (long)m1_Cin + (long)m1_ci) * m1_Pin;\n";
    os_ << "  float m1_t[6][6];\n";
    os_ << "  if (m1_ih0 >= 0 && m1_ih0 + 5 < m1_Hin && m1_iw0 >= 0 && m1_iw0 + 5 < m1_Win) {\n";
    os_ << "    device const float* m1_xp = m1_xc + (long)m1_ih0 * (long)m1_Win + (long)m1_iw0;\n";
    for (int r = 0; r < 6; ++r) {
      os_ << "    { float4 m1_a = float4(*(const device packed_float4*)(m1_xp + " << r << "L * (long)m1_Win));\n";
      os_ << "      float2 m1_b = float2(*(const device packed_float2*)(m1_xp + " << r << "L * (long)m1_Win + 4L));\n";
      os_ << "      m1_t[" << r << "][0] = m1_a.x; m1_t[" << r << "][1] = m1_a.y; m1_t[" << r
          << "][2] = m1_a.z; m1_t[" << r << "][3] = m1_a.w; m1_t[" << r << "][4] = m1_b.x; m1_t[" << r
          << "][5] = m1_b.y; }\n";
    }
    os_ << "  } else {\n";
    for (int r = 0; r < 6; ++r) {
      os_ << "    if ((uint)(m1_ih0 + " << r << ") < (uint)m1_Hin) {\n";
      os_ << "      device const float* m1_xr = m1_xc + (long)(m1_ih0 + " << r << ") * (long)m1_Win;\n";
      for (int c = 0; c < 6; ++c) {
        os_ << "      m1_t[" << r << "][" << c << "] = ((uint)(m1_iw0 + " << c << ") < (uint)m1_Win) ? m1_xr[m1_iw0 + "
            << c << "] : 0.0f;\n";
      }
      os_ << "    } else {\n";
      os_ << "      m1_t[" << r << "][0] = 0.0f; m1_t[" << r << "][1] = 0.0f; m1_t[" << r
          << "][2] = 0.0f; m1_t[" << r << "][3] = 0.0f; m1_t[" << r << "][4] = 0.0f; m1_t[" << r << "][5] = 0.0f;\n";
      os_ << "    }\n";
    }
    os_ << "  }\n";
    os_ << "  #pragma unroll\n";
    os_ << "  for (int m1_c = 0; m1_c < 6; ++m1_c) {\n";
    os_ << "    const float m1_x0 = m1_t[0][m1_c], m1_x1 = m1_t[1][m1_c], m1_x2 = m1_t[2][m1_c];\n";
    os_ << "    const float m1_x3 = m1_t[3][m1_c], m1_x4 = m1_t[4][m1_c], m1_x5 = m1_t[5][m1_c];\n";
    os_ << "    m1_t[0][m1_c] = 4.0f * m1_x0 - 5.0f * m1_x2 + m1_x4;\n";
    os_ << "    const float m1_a1 = m1_x4 - 4.0f * m1_x2, m1_b1 = m1_x3 - 4.0f * m1_x1;\n";
    os_ << "    m1_t[1][m1_c] = m1_a1 + m1_b1;\n";
    os_ << "    m1_t[2][m1_c] = m1_a1 - m1_b1;\n";
    os_ << "    const float m1_a2 = m1_x4 - m1_x2, m1_b2 = 2.0f * (m1_x3 - m1_x1);\n";
    os_ << "    m1_t[3][m1_c] = m1_a2 + m1_b2;\n";
    os_ << "    m1_t[4][m1_c] = m1_a2 - m1_b2;\n";
    os_ << "    m1_t[5][m1_c] = 4.0f * m1_x1 - 5.0f * m1_x3 + m1_x5;\n";
    os_ << "  }\n";
    os_ << "  const long m1_p = (long)m1_rloc * (long)m1_Tw + (long)m1_tx;\n";
    os_ << "  const long m1_pld = (long)p_ld;\n";
    os_ << "  const long m1_astride = (long)m1_Cin * m1_pld + 64L;\n";
    os_ << "  device float* m1_vp = " << plan_.out_name << " + (long)m1_ci * m1_pld + m1_p;\n";
    os_ << "  #pragma unroll\n";
    os_ << "  for (int m1_r = 0; m1_r < 6; ++m1_r) {\n";
    os_ << "    const float m1_x0 = m1_t[m1_r][0], m1_x1 = m1_t[m1_r][1], m1_x2 = m1_t[m1_r][2];\n";
    os_ << "    const float m1_x3 = m1_t[m1_r][3], m1_x4 = m1_t[m1_r][4], m1_x5 = m1_t[m1_r][5];\n";
    os_ << "    const long m1_rb = (long)(m1_r * 6) * m1_astride;\n";
    os_ << "    m1_vp[m1_rb + 0L * m1_astride] = 4.0f * m1_x0 - 5.0f * m1_x2 + m1_x4;\n";
    os_ << "    const float m1_a1 = m1_x4 - 4.0f * m1_x2, m1_b1 = m1_x3 - 4.0f * m1_x1;\n";
    os_ << "    m1_vp[m1_rb + 1L * m1_astride] = m1_a1 + m1_b1;\n";
    os_ << "    m1_vp[m1_rb + 2L * m1_astride] = m1_a1 - m1_b1;\n";
    os_ << "    const float m1_a2 = m1_x4 - m1_x2, m1_b2 = 2.0f * (m1_x3 - m1_x1);\n";
    os_ << "    m1_vp[m1_rb + 3L * m1_astride] = m1_a2 + m1_b2;\n";
    os_ << "    m1_vp[m1_rb + 4L * m1_astride] = m1_a2 - m1_b2;\n";
    os_ << "    m1_vp[m1_rb + 5L * m1_astride] = 4.0f * m1_x1 - 5.0f * m1_x3 + m1_x5;\n";
    os_ << "  }\n";
    os_ << "}\n";
    return os_.str();
  }

  std::string emit_wino4x3_bmm() {
    const uint32_t bfm = fm();
    const uint32_t bfn = fn();
    const bool need_w_guard = w_per() * w_rstep() > plan_.bm;
    const bool need_x_guard = x_per() * x_rstep() > plan_.bk;
    emit_signature();
    os_ << "  constexpr int m1_BM = " << plan_.bm << ", m1_BN = " << plan_.bn << ", m1_BK = " << plan_.bk << ";\n";
    os_ << "  constexpr int m1_LDW = m1_BK + 4, m1_LDX = m1_BN + 4;\n";
    os_ << "  const int m1_M = (int)" << plan_.cout_expr << ";\n";
    os_ << "  const int m1_K = (int)" << plan_.cin_expr << ";\n";
    os_ << "  const int m1_N = p_ld;\n";
    os_ << "  const int m1_alpha = (int)m1_tgid.z;\n";
    os_ << "  const int m1_co0 = (int)m1_tgid.y * m1_BM;\n";
    os_ << "  const int m1_p0 = (int)m1_tgid.x * m1_BN;\n";
    os_ << "  if (m1_alpha >= 36 || m1_co0 >= m1_M || m1_p0 >= m1_N) return;\n";
    os_ << "  device const float* m1_Ua = " << plan_.w_name << " + (long)m1_alpha * (long)m1_M * (long)m1_K;\n";
    os_ << "  device const float* m1_Va = " << plan_.x_name << " + (long)m1_alpha * ((long)m1_K * (long)m1_N + 64L);\n";
    os_ << "  device float* m1_Ma = " << plan_.out_name << " + (long)m1_alpha * ((long)m1_M * (long)m1_N + 64L);\n";
    os_ << "  threadgroup float m1_smem[" << plan_.smem_floats << "];\n";
    os_ << "  threadgroup float* m1_sw = m1_smem;\n";
    os_ << "  threadgroup float* m1_sx = m1_smem + m1_BM * m1_LDW;\n";
    os_ << "  const int m1_wc = (int)(m1_tid % " << w_vpr() << "u) * 4;\n";
    os_ << "  const int m1_wr0 = (int)(m1_tid / " << w_vpr() << "u);\n";
    os_ << "  const int m1_xc = (int)(m1_tid % " << x_vpr() << "u) * 4;\n";
    os_ << "  const int m1_xr0 = (int)(m1_tid / " << x_vpr() << "u);\n";
    os_ << "  const int m1_sgr = (int)(m1_sg / " << plan_.sg_n << "u) * " << (bfm * 8) << ";\n";
    os_ << "  const int m1_sgc = (int)(m1_sg % " << plan_.sg_n << "u) * " << (bfn * 8) << ";\n";
    os_ << "  simdgroup_float8x8 m1_acc[" << bfm << "][" << bfn << "];\n";
    for (uint32_t i = 0; i < bfm; ++i) {
      for (uint32_t j = 0; j < bfn; ++j) {
        os_ << "  m1_acc[" << i << "][" << j << "] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      }
    }
    os_ << "  const bool m1_full_m = (m1_co0 + m1_BM <= m1_M);\n";
    os_ << "  const bool m1_full_n = (m1_p0 + m1_BN <= m1_N);\n";
    os_ << "  const int m1_KFull = (m1_K / m1_BK) * m1_BK;\n";
    os_ << "  if (m1_full_m && m1_full_n) {\n";
    os_ << "    if (m1_KFull > 0) {\n";
    os_ << "      float4 m1_rw[" << w_per() << "];\n";
    os_ << "      float4 m1_rx[" << x_per() << "];\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      if (guard) os_ << "      if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
      os_ << "      *(threadgroup float4*)(m1_sw + (m1_wr0 + " << (p * w_rstep()) << ") * m1_LDW + m1_wc) = "
          << "float4(*(const device packed_float4*)(m1_Ua + (long)(m1_co0 + m1_wr0 + " << (p * w_rstep())
          << ") * (long)m1_K + m1_wc));\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      if (guard) os_ << "      if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK)\n  ";
      os_ << "      *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep()) << ") * m1_LDX + m1_xc) = "
          << "float4(*(const device packed_float4*)(m1_Va + (long)(m1_xr0 + " << (p * x_rstep())
          << ") * (long)m1_N + m1_p0 + m1_xc));\n";
    }
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "      for (int m1_k0 = 0; m1_k0 + m1_BK < m1_KFull; m1_k0 += m1_BK) {\n";
    os_ << "        const int m1_kn = m1_k0 + m1_BK;\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      if (guard) os_ << "        if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
      os_ << "        m1_rw[" << p << "] = float4(*(const device packed_float4*)(m1_Ua + (long)(m1_co0 + m1_wr0 + "
          << (p * w_rstep()) << ") * (long)m1_K + m1_kn + m1_wc));\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      if (guard) os_ << "        if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK)\n  ";
      os_ << "        m1_rx[" << p << "] = float4(*(const device packed_float4*)(m1_Va + (long)(m1_kn + m1_xr0 + "
          << (p * x_rstep()) << ") * (long)m1_N + m1_p0 + m1_xc));\n";
    }
    emit_mma("        ");
    os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      if (guard) os_ << "        if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
      os_ << "        *(threadgroup float4*)(m1_sw + (m1_wr0 + " << (p * w_rstep()) << ") * m1_LDW + m1_wc) = m1_rw["
          << p << "];\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      if (guard) os_ << "        if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK)\n  ";
      os_ << "        *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep()) << ") * m1_LDX + m1_xc) = m1_rx["
          << p << "];\n";
    }
    os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "      }\n";
    emit_mma("      ");
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";
    os_ << "    if (m1_KFull < m1_K) {\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      os_ << "      {\n";
      os_ << "        const int m1_r = m1_wr0 + " << (p * w_rstep()) << ";\n";
      if (guard) os_ << "        if (m1_r < m1_BM) {\n";
      os_ << "        const int m1_k = m1_KFull + m1_wc;\n";
      os_ << "        float4 m1_v = float4(0.0f);\n";
      os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) if (m1_k + m1_e < m1_K) m1_v[m1_e] = m1_Ua[(long)(m1_co0 + m1_r) * (long)m1_K + m1_k + m1_e];\n";
      os_ << "        *(threadgroup float4*)(m1_sw + m1_r * m1_LDW + m1_wc) = m1_v;\n";
      if (guard) os_ << "        }\n";
      os_ << "      }\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      os_ << "      {\n";
      os_ << "        const int m1_rr = m1_xr0 + " << (p * x_rstep()) << ";\n";
      if (guard) os_ << "        if (m1_rr < m1_BK) {\n";
      os_ << "        const int m1_k = m1_KFull + m1_rr;\n";
      os_ << "        float4 m1_v = (m1_k < m1_K) ? float4(*(const device packed_float4*)(m1_Va + (long)m1_k * (long)m1_N + m1_p0 + m1_xc)) : float4(0.0f);\n";
      os_ << "        *(threadgroup float4*)(m1_sx + m1_rr * m1_LDX + m1_xc) = m1_v;\n";
      if (guard) os_ << "        }\n";
      os_ << "      }\n";
    }
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma("      ");
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";
    for (uint32_t i = 0; i < bfm; ++i) {
      for (uint32_t j = 0; j < bfn; ++j) {
        os_ << "    simdgroup_store(m1_acc[" << i << "][" << j << "], m1_Ma + (long)(m1_co0 + m1_sgr + " << (i * 8)
            << ") * (long)m1_N + (long)(m1_p0 + m1_sgc + " << (j * 8) << "), (ulong)m1_N);\n";
      }
    }
    os_ << "    return;\n";
    os_ << "  }\n";

    os_ << "  for (int m1_k0 = 0; m1_k0 < m1_K; m1_k0 += m1_BK) {\n";
    for (uint32_t p = 0; p < w_per(); ++p) {
      const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
      os_ << "    {\n";
      os_ << "      const int m1_r = m1_wr0 + " << (p * w_rstep()) << ";\n";
      if (guard) os_ << "      if (m1_r < m1_BM) {\n";
      os_ << "      const int m1_k = m1_k0 + m1_wc;\n";
      os_ << "      float4 m1_v = float4(0.0f);\n";
      os_ << "      if (m1_co0 + m1_r < m1_M) {\n";
      os_ << "        if (m1_k + 3 < m1_K) m1_v = float4(*(const device packed_float4*)(m1_Ua + (long)(m1_co0 + m1_r) * (long)m1_K + m1_k));\n";
      os_ << "        else { for (int m1_e = 0; m1_e < 4; ++m1_e) if (m1_k + m1_e < m1_K) m1_v[m1_e] = m1_Ua[(long)(m1_co0 + m1_r) * (long)m1_K + m1_k + m1_e]; }\n";
      os_ << "      }\n";
      os_ << "      *(threadgroup float4*)(m1_sw + m1_r * m1_LDW + m1_wc) = m1_v;\n";
      if (guard) os_ << "      }\n";
      os_ << "    }\n";
    }
    for (uint32_t p = 0; p < x_per(); ++p) {
      const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
      os_ << "    {\n";
      os_ << "      const int m1_rr = m1_xr0 + " << (p * x_rstep()) << ";\n";
      if (guard) os_ << "      if (m1_rr < m1_BK) {\n";
      os_ << "      const int m1_k = m1_k0 + m1_rr;\n";
      os_ << "      float4 m1_v = float4(0.0f);\n";
      os_ << "      if (m1_k < m1_K) {\n";
      os_ << "        if (m1_p0 + m1_xc + 3 < m1_N) m1_v = float4(*(const device packed_float4*)(m1_Va + (long)m1_k * (long)m1_N + m1_p0 + m1_xc));\n";
      os_ << "        else { for (int m1_e = 0; m1_e < 4; ++m1_e) if (m1_p0 + m1_xc + m1_e < m1_N) m1_v[m1_e] = m1_Va[(long)m1_k * (long)m1_N + m1_p0 + m1_xc + m1_e]; }\n";
      os_ << "      }\n";
      os_ << "      *(threadgroup float4*)(m1_sx + m1_rr * m1_LDX + m1_xc) = m1_v;\n";
      if (guard) os_ << "      }\n";
      os_ << "    }\n";
    }
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma("    ");
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
    const uint32_t sg_rows = bfm * 8u;
    const uint32_t sg_ld = bfn * 8u;
    const uint32_t sg_elems = sg_rows * sg_ld;
    const uint32_t max_sgs = std::max<uint32_t>(1u, plan_.smem_floats / sg_elems);
    os_ << "  for (uint m1_sb = 0u; m1_sb < " << plan_.num_sgs << "u; m1_sb += " << max_sgs << "u) {\n";
    os_ << "    if (m1_sg >= m1_sb && m1_sg < m1_sb + " << max_sgs << "u) {\n";
    os_ << "      threadgroup float* m1_sgs = m1_smem + (m1_sg - m1_sb) * " << sg_elems << "u;\n";
    for (uint32_t i = 0; i < bfm; ++i) {
      for (uint32_t j = 0; j < bfn; ++j) {
        os_ << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_sgs + " << (i * 8u) << "u * " << sg_ld
            << "u + " << (j * 8u) << "u, " << sg_ld << ");\n";
      }
    }
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    const uint m1_act = min(" << max_sgs << "u, " << plan_.num_sgs << "u - m1_sb);\n";
    os_ << "    for (uint m1_idx = m1_tid; m1_idx < m1_act * " << sg_elems << "u; m1_idx += " << plan_.num_threads
        << "u) {\n";
    os_ << "      const uint m1_lsg = m1_idx / " << sg_elems << "u;\n";
    os_ << "      const uint m1_rem = m1_idx - m1_lsg * " << sg_elems << "u;\n";
    os_ << "      const uint m1_lr = m1_rem / " << sg_ld << "u;\n";
    os_ << "      const uint m1_lc = m1_rem - m1_lr * " << sg_ld << "u;\n";
    os_ << "      const uint m1_gsg = m1_sb + m1_lsg;\n";
    os_ << "      const int m1_gr = m1_co0 + (int)(m1_gsg / " << plan_.sg_n << "u) * " << sg_rows << " + (int)m1_lr;\n";
    os_ << "      const int m1_gc = m1_p0 + (int)(m1_gsg % " << plan_.sg_n << "u) * " << sg_ld << " + (int)m1_lc;\n";
    os_ << "      if (m1_gr < m1_M && m1_gc < m1_N) m1_Ma[(long)m1_gr * (long)m1_N + (long)m1_gc] = m1_smem[m1_idx];\n";
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
    os_ << "}\n";
    return os_.str();
  }

  void emit_epilogue_steps(const std::string& val_var, const std::string& ch_expr, const std::string& elem_idx_expr,
                           const std::string& ind) {
    if (!plan_.epilogue_dag) return;
    HelperFlags tmp_helpers = plan_.helpers;
    emit_epilogue_dag_msl(
        os_, kernel_, *plan_.epilogue_dag, val_var, ch_expr, elem_idx_expr,
        [this](uint32_t idx) -> std::string {
          const auto& kp = kernel_.params[idx];
          if (kp.is_constexpr) return std::to_string(kp.constexpr_val);
          return kp.name;
        },
        tmp_helpers, ind);
  }

  std::string emit_wino4x3_output() {
    const ConvDim& H = plan_.dims[1];
    const ConvDim& W = plan_.dims[2];
    emit_signature();
    os_ << "  (void)m1_sg; (void)" << plan_.w_name << ";\n";
    os_ << "  const int m1_Cout = (int)" << plan_.cout_expr << ";\n";
    os_ << "  const int m1_Hout = (int)" << H.out << ", m1_Wout = (int)" << W.out << ";\n";
    os_ << "  const int m1_Th = (m1_Hout + 3) >> 2;\n";
    os_ << "  const int m1_Tw = (m1_Wout + 3) >> 2;\n";
    os_ << "  const int m1_tx = (int)m1_tgid.x * 64 + (int)m1_tid;\n";
    os_ << "  const int m1_rloc = (int)m1_tgid.y;\n";
    os_ << "  const int m1_co = (int)m1_tgid.z;\n";
    os_ << "  if (m1_co >= m1_Cout || m1_rloc >= chunk_rows || m1_tx >= m1_Tw) return;\n";
    os_ << "  const int m1_rglob = row_start + m1_rloc;\n";
    os_ << "  const int m1_n = m1_rglob / m1_Th;\n";
    os_ << "  const int m1_ty = m1_rglob - m1_n * m1_Th;\n";
    os_ << "  const int m1_oh0 = m1_ty * 4;\n";
    os_ << "  const int m1_ow0 = m1_tx * 4;\n";
    os_ << "  const long m1_p = (long)m1_rloc * (long)m1_Tw + (long)m1_tx;\n";
    os_ << "  const long m1_pld = (long)p_ld;\n";
    os_ << "  const long m1_astride = (long)m1_Cout * m1_pld + 64L;\n";
    os_ << "  device const float* m1_mp = " << plan_.x_name << " + (long)m1_co * m1_pld + m1_p;\n";
    os_ << "  float m1_s[4][6];\n";
    os_ << "  #pragma unroll\n";
    os_ << "  for (int m1_c = 0; m1_c < 6; ++m1_c) {\n";
    os_ << "    const float m1_m0 = m1_mp[(long)(0 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_m1 = m1_mp[(long)(6 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_m2 = m1_mp[(long)(12 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_m3 = m1_mp[(long)(18 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_m4 = m1_mp[(long)(24 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_m5 = m1_mp[(long)(30 + m1_c) * m1_astride];\n";
    os_ << "    const float m1_p12 = m1_m1 + m1_m2, m1_m12 = m1_m1 - m1_m2;\n";
    os_ << "    const float m1_p34 = m1_m3 + m1_m4, m1_m34 = m1_m3 - m1_m4;\n";
    os_ << "    m1_s[0][m1_c] = m1_m0 + m1_p12 + m1_p34;\n";
    os_ << "    m1_s[1][m1_c] = m1_m12 + 2.0f * m1_m34;\n";
    os_ << "    m1_s[2][m1_c] = m1_p12 + 4.0f * m1_p34;\n";
    os_ << "    m1_s[3][m1_c] = m1_m12 + 8.0f * m1_m34 + m1_m5;\n";
    os_ << "  }\n";
    if (plan_.attr.has_bias) {
      os_ << "  const float m1_bv = (float)" << plan_.bias_name << "[m1_co];\n";
    }
    os_ << "  const long m1_Pout = (long)m1_Hout * (long)m1_Wout;\n";
    os_ << "  const long m1_oc_base = ((long)m1_n * (long)m1_Cout + (long)m1_co) * m1_Pout;\n";
    os_ << "  device float* m1_op = " << plan_.out_name << " + m1_oc_base;\n";
    os_ << "  const bool m1_w4 = (m1_ow0 + 3 < m1_Wout);\n";
    os_ << "  #pragma unroll\n";
    os_ << "  for (int m1_r = 0; m1_r < 4; ++m1_r) {\n";
    os_ << "    if (m1_oh0 + m1_r < m1_Hout) {\n";
    os_ << "      const float m1_p12 = m1_s[m1_r][1] + m1_s[m1_r][2], m1_m12 = m1_s[m1_r][1] - m1_s[m1_r][2];\n";
    os_ << "      const float m1_p34 = m1_s[m1_r][3] + m1_s[m1_r][4], m1_m34 = m1_s[m1_r][3] - m1_s[m1_r][4];\n";
    os_ << "      float4 m1_ov = float4(m1_s[m1_r][0] + m1_p12 + m1_p34, m1_m12 + 2.0f * m1_m34, "
           "m1_p12 + 4.0f * m1_p34, m1_m12 + 8.0f * m1_m34 + m1_s[m1_r][5]);\n";
    if (plan_.attr.has_bias) {
      os_ << "      m1_ov += float4(m1_bv);\n";
    }
    os_ << "      const long m1_row_off = (long)(m1_oh0 + m1_r) * (long)m1_Wout + (long)m1_ow0;\n";
    os_ << "      device float* m1_or = m1_op + m1_row_off;\n";
    if (plan_.epilogue_dag) {
      os_ << "      const long m1_row_idx = m1_oc_base + m1_row_off;\n";
      os_ << "      if (m1_w4) {\n";
      os_ << "        #pragma unroll\n";
      os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "          float m1_ev = m1_ov[m1_e];\n";
      emit_epilogue_steps("m1_ev", "m1_co", "(m1_row_idx + (long)m1_e)", "          ");
      os_ << "          m1_ov[m1_e] = m1_ev;\n";
      os_ << "        }\n";
      os_ << "        *(device packed_float4*)(m1_or) = packed_float4(m1_ov);\n";
      os_ << "      } else {\n";
      os_ << "        #pragma unroll\n";
      os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "          if (m1_ow0 + m1_e < m1_Wout) {\n";
      os_ << "            float m1_ev = m1_ov[m1_e];\n";
      emit_epilogue_steps("m1_ev", "m1_co", "(m1_row_idx + (long)m1_e)", "            ");
      os_ << "            m1_or[m1_e] = m1_ev;\n";
      os_ << "          }\n";
      os_ << "        }\n";
      os_ << "      }\n";
    } else {
      os_ << "      if (m1_w4) {\n";
      os_ << "        *(device packed_float4*)(m1_or) = packed_float4(m1_ov);\n";
      os_ << "      } else {\n";
      os_ << "        m1_or[0] = m1_ov.x;\n";
      os_ << "        if (m1_ow0 + 1 < m1_Wout) m1_or[1] = m1_ov.y;\n";
      os_ << "        if (m1_ow0 + 2 < m1_Wout) m1_or[2] = m1_ov.z;\n";
      os_ << "      }\n";
    }
    os_ << "    }\n";
    os_ << "  }\n";
    os_ << "}\n";
    return os_.str();
  }

  void emit_wino2x3_load_patch(const std::string& ind, const std::string& ci_expr) {
    os_ << ind << "m1_r0 = float4(0.0f); m1_r1 = float4(0.0f); m1_r2 = float4(0.0f); m1_r3 = float4(0.0f);\n";
    os_ << ind << "if (" << ci_expr << " < m1_Cin) {\n";
    os_ << ind << "  device const float* m1_xc = m1_xn + (long)(" << ci_expr << ") * m1_Pin;\n";
    os_ << ind << "  if (m1_in_fast) {\n";
    os_ << ind << "    device const float* m1_xp = m1_xc + (long)m1_ih0 * (long)m1_Win + (long)m1_iw0;\n";
    os_ << ind << "    m1_r0 = float4(*(const device packed_float4*)(m1_xp));\n";
    os_ << ind << "    m1_r1 = float4(*(const device packed_float4*)(m1_xp + m1_Win));\n";
    os_ << ind << "    m1_r2 = float4(*(const device packed_float4*)(m1_xp + 2 * m1_Win));\n";
    os_ << ind << "    m1_r3 = float4(*(const device packed_float4*)(m1_xp + 3 * m1_Win));\n";
    os_ << ind << "  } else {\n";
    for (int rr = 0; rr < 4; ++rr) {
      os_ << ind << "    if ((uint)(m1_ih0 + " << rr << ") < (uint)m1_Hin) {\n";
      os_ << ind << "      device const float* m1_xr = m1_xc + (long)(m1_ih0 + " << rr << ") * (long)m1_Win;\n";
      os_ << ind << "      if ((uint)(m1_iw0 + 0) < (uint)m1_Win) m1_r" << rr << ".x = m1_xr[m1_iw0 + 0];\n";
      os_ << ind << "      if ((uint)(m1_iw0 + 1) < (uint)m1_Win) m1_r" << rr << ".y = m1_xr[m1_iw0 + 1];\n";
      os_ << ind << "      if ((uint)(m1_iw0 + 2) < (uint)m1_Win) m1_r" << rr << ".z = m1_xr[m1_iw0 + 2];\n";
      os_ << ind << "      if ((uint)(m1_iw0 + 3) < (uint)m1_Win) m1_r" << rr << ".w = m1_xr[m1_iw0 + 3];\n";
      os_ << ind << "    }\n";
    }
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  void emit_wino2x3_store_v(const std::string& ind) {
    os_ << ind << "const float4 m1_t0 = m1_r0 - m1_r2;\n";
    os_ << ind << "const float4 m1_t1 = m1_r1 + m1_r2;\n";
    os_ << ind << "const float4 m1_t2 = m1_r2 - m1_r1;\n";
    os_ << ind << "const float4 m1_t3 = m1_r1 - m1_r3;\n";
    for (int rr = 0; rr < 4; ++rr) {
      os_ << ind << "m1_smem[" << (rr * 4 + 0) << "u * m1_VPLANE + m1_voff] = m1_t" << rr << ".x - m1_t" << rr
          << ".z;\n";
      os_ << ind << "m1_smem[" << (rr * 4 + 1) << "u * m1_VPLANE + m1_voff] = m1_t" << rr << ".y + m1_t" << rr
          << ".z;\n";
      os_ << ind << "m1_smem[" << (rr * 4 + 2) << "u * m1_VPLANE + m1_voff] = m1_t" << rr << ".z - m1_t" << rr
          << ".y;\n";
      os_ << ind << "m1_smem[" << (rr * 4 + 3) << "u * m1_VPLANE + m1_voff] = m1_t" << rr << ".y - m1_t" << rr
          << ".w;\n";
    }
  }

  void emit_wino_mma_block(uint32_t aps, uint32_t wfm, uint32_t wfn, uint32_t bk, const std::string& ind) {
    for (uint32_t ks = 0; ks < bk; ks += 8u) {
      os_ << ind << "{\n";
      os_ << ind << "  const ulong m1_kb = (ulong)((m1_c0 + " << ks << ") >> 3);\n";
      for (uint32_t a = 0; a < aps; ++a) {
        os_ << ind << "  {\n";
        os_ << ind << "    const uint m1_al = (uint)m1_sg + " << (a * plan_.num_sgs) << "u;\n";
        os_ << ind << "    device const float* m1_ub = " << plan_.w_name
            << " + (((ulong)m1_al * m1_cin_blk + m1_kb) * m1_cout_blk + m1_co_blk0) * 64ul;\n";
        os_ << ind << "    threadgroup const float* m1_vb = m1_smem + m1_al * m1_VPLANE + " << ks << "u * m1_VLD;\n";
        os_ << ind << "    simdgroup_float8x8 m1_fa[2];\n";
        os_ << ind << "    simdgroup_float8x8 m1_fb[" << wfn << "];\n";
        os_ << ind << "    simdgroup_load(m1_fa[0], m1_ub, 8u);\n";
        for (uint32_t j = 0; j < wfn; ++j) {
          os_ << ind << "    simdgroup_load(m1_fb[" << j << "], m1_vb + " << (j * 8u) << "u, m1_VLD);\n";
        }
        for (uint32_t i = 0; i < wfm; ++i) {
          if (i + 1u < wfm) {
            os_ << ind << "    simdgroup_load(m1_fa[" << ((i + 1u) & 1u) << "], m1_ub + " << ((i + 1u) * 64u)
                << "u, 8u);\n";
          }
          for (uint32_t j = 0; j < wfn; ++j) {
            os_ << ind << "    simdgroup_multiply_accumulate(m1_acc[" << a << "][" << i << "][" << j << "], m1_fa["
                << (i & 1u) << "], m1_fb[" << j << "], m1_acc[" << a << "][" << i << "][" << j << "]);\n";
          }
        }
        os_ << ind << "  }\n";
      }
      os_ << ind << "}\n";
    }
  }

  std::string emit_winograd_2x3() {
    const ConvDim& H = plan_.dims[1];
    const ConvDim& W = plan_.dims[2];
    const uint32_t wfm = plan_.bm / 8u;
    const uint32_t wfn = plan_.bn / 8u;
    const uint32_t aps = 16u / plan_.num_sgs;
    const uint32_t in_pairs = plan_.bk * plan_.bn;
    const uint32_t epi_pairs = 8u * plan_.bn;
    const bool pipeline_x = (in_pairs == plan_.num_threads);
    emit_signature();
    os_ << "  constexpr int m1_BM = " << plan_.bm << ", m1_BN = " << plan_.bn << ", m1_BK = " << plan_.bk << ";\n";
    os_ << "  constexpr int m1_TTH = " << plan_.wino_th << ", m1_TTW = " << plan_.wino_tw << ";\n";
    os_ << "  constexpr uint m1_VLD = " << plan_.wino_vld << "u, m1_ELD = " << plan_.wino_eld << "u;\n";
    os_ << "  constexpr uint m1_VPLANE = (uint)m1_BK * m1_VLD;\n";
    os_ << "  constexpr uint m1_EPLANE = 8u * m1_ELD;\n";
    os_ << "  constexpr int m1_PH = " << H.p << ", m1_PW = " << W.p << ";\n";
    os_ << "  const int m1_N = (int)" << plan_.n_expr << ";\n";
    os_ << "  const int m1_Cin = (int)" << plan_.cin_expr << ";\n";
    os_ << "  const int m1_Cout = (int)" << plan_.cout_expr << ";\n";
    os_ << "  const int m1_Hin = (int)" << H.in << ", m1_Win = (int)" << W.in << ";\n";
    os_ << "  const int m1_Hout = (int)" << H.out << ", m1_Wout = (int)" << W.out << ";\n";
    os_ << "  const int m1_Th = (m1_Hout + 1) >> 1;\n";
    os_ << "  const int m1_Tw = (m1_Wout + 1) >> 1;\n";
    os_ << "  const int m1_Gtw = (m1_Tw + m1_TTW - 1) / m1_TTW;\n";
    os_ << "  const int m1_n = (int)m1_tgid.z;\n";
    os_ << "  const int m1_co0 = (int)m1_tgid.y * m1_BM;\n";
    os_ << "  const int m1_tb = (int)m1_tgid.x;\n";
    os_ << "  const int m1_tby = m1_tb / m1_Gtw;\n";
    os_ << "  const int m1_tbx = m1_tb - m1_tby * m1_Gtw;\n";
    os_ << "  const int m1_ty0 = m1_tby * m1_TTH;\n";
    os_ << "  const int m1_tx0 = m1_tbx * m1_TTW;\n";
    os_ << "  if (m1_n >= m1_N || m1_co0 >= m1_Cout || m1_ty0 >= m1_Th) return;\n";
    os_ << "  const long m1_Pin = (long)m1_Hin * (long)m1_Win;\n";
    os_ << "  const long m1_Pout = (long)m1_Hout * (long)m1_Wout;\n";
    os_ << "  device const float* m1_xn = " << plan_.x_name << " + (long)m1_n * (long)m1_Cin * m1_Pin;\n";
    os_ << "  const ulong m1_cin_blk = (ulong)((m1_Cin + m1_BK - 1) / m1_BK) * (ulong)(m1_BK >> 3);\n";
    os_ << "  const ulong m1_cout_blk = (ulong)((m1_Cout + m1_BM - 1) / m1_BM) * (ulong)(m1_BM >> 3);\n";
    os_ << "  const ulong m1_co_blk0 = (ulong)(m1_co0 >> 3);\n";
    os_ << "  threadgroup float m1_smem[" << plan_.smem_floats << "];\n";
    os_ << "  simdgroup_float8x8 m1_acc[" << aps << "][" << wfm << "][" << wfn << "];\n";
    for (uint32_t a = 0; a < aps; ++a) {
      for (uint32_t i = 0; i < wfm; ++i) {
        for (uint32_t j = 0; j < wfn; ++j) {
          os_ << "  m1_acc[" << a << "][" << i << "][" << j << "] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
        }
      }
    }
    os_ << "  const int m1_icl = (int)(m1_tid / (uint)m1_BN);\n";
    os_ << "  const int m1_itl = (int)(m1_tid % (uint)m1_BN);\n";
    os_ << "  const int m1_ilty = m1_itl / m1_TTW;\n";
    os_ << "  const int m1_iltx = m1_itl - m1_ilty * m1_TTW;\n";
    os_ << "  const int m1_ih0 = (m1_ty0 + m1_ilty) * 2 - m1_PH;\n";
    os_ << "  const int m1_iw0 = (m1_tx0 + m1_iltx) * 2 - m1_PW;\n";
    os_ << "  const bool m1_in_fast = (m1_ih0 >= 0 && m1_ih0 + 3 < m1_Hin && m1_iw0 >= 0 && m1_iw0 + 3 < m1_Win);\n";
    os_ << "  const uint m1_voff = (uint)m1_icl * m1_VLD + (uint)m1_itl;\n";
    if (pipeline_x) {
      os_ << "  if (m1_Cin > 0) {\n";
      os_ << "    float4 m1_r0, m1_r1, m1_r2, m1_r3;\n";
      emit_wino2x3_load_patch("    ", "m1_icl");
      emit_wino2x3_store_v("    ");
      os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "    for (int m1_c0 = 0; m1_c0 < m1_Cin; m1_c0 += m1_BK) {\n";
      os_ << "      const bool m1_has_next = (m1_c0 + m1_BK < m1_Cin);\n";
      os_ << "      if (m1_has_next) {\n";
      emit_wino2x3_load_patch("        ", "m1_c0 + m1_BK + m1_icl");
      os_ << "      }\n";
      emit_wino_mma_block(aps, wfm, wfn, plan_.bk, "      ");
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "      if (m1_has_next) {\n";
      emit_wino2x3_store_v("        ");
      os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "      }\n";
      os_ << "    }\n";
      os_ << "  }\n";
    } else {
      os_ << "  for (int m1_c0 = 0; m1_c0 < m1_Cin; m1_c0 += m1_BK) {\n";
      os_ << "    for (uint m1_ii = m1_tid; m1_ii < " << in_pairs << "u; m1_ii += " << plan_.num_threads << "u) {\n";
      os_ << "      const int m1_cl = (int)(m1_ii / (uint)m1_BN);\n";
      os_ << "      const int m1_tl = (int)(m1_ii % (uint)m1_BN);\n";
      os_ << "      const int m1_lty = m1_tl / m1_TTW;\n";
      os_ << "      const int m1_ltx = m1_tl - m1_lty * m1_TTW;\n";
      os_ << "      const int m1_ih = (m1_ty0 + m1_lty) * 2 - m1_PH;\n";
      os_ << "      const int m1_iw = (m1_tx0 + m1_ltx) * 2 - m1_PW;\n";
      os_ << "      const int m1_ci = m1_c0 + m1_cl;\n";
      os_ << "      float4 m1_r0 = float4(0.0f), m1_r1 = float4(0.0f), m1_r2 = float4(0.0f), m1_r3 = float4(0.0f);\n";
      os_ << "      if (m1_ci < m1_Cin) {\n";
      os_ << "        device const float* m1_xc = m1_xn + (long)m1_ci * m1_Pin;\n";
      os_ << "        if (m1_ih >= 0 && m1_ih + 3 < m1_Hin && m1_iw >= 0 && m1_iw + 3 < m1_Win) {\n";
      os_ << "          device const float* m1_xp = m1_xc + (long)m1_ih * (long)m1_Win + (long)m1_iw;\n";
      os_ << "          m1_r0 = float4(*(const device packed_float4*)(m1_xp));\n";
      os_ << "          m1_r1 = float4(*(const device packed_float4*)(m1_xp + m1_Win));\n";
      os_ << "          m1_r2 = float4(*(const device packed_float4*)(m1_xp + 2 * m1_Win));\n";
      os_ << "          m1_r3 = float4(*(const device packed_float4*)(m1_xp + 3 * m1_Win));\n";
      os_ << "        } else {\n";
      for (int rr = 0; rr < 4; ++rr) {
        os_ << "          if ((uint)(m1_ih + " << rr << ") < (uint)m1_Hin) {\n";
        os_ << "            device const float* m1_xr = m1_xc + (long)(m1_ih + " << rr << ") * (long)m1_Win;\n";
        os_ << "            if ((uint)(m1_iw + 0) < (uint)m1_Win) m1_r" << rr << ".x = m1_xr[m1_iw + 0];\n";
        os_ << "            if ((uint)(m1_iw + 1) < (uint)m1_Win) m1_r" << rr << ".y = m1_xr[m1_iw + 1];\n";
        os_ << "            if ((uint)(m1_iw + 2) < (uint)m1_Win) m1_r" << rr << ".z = m1_xr[m1_iw + 2];\n";
        os_ << "            if ((uint)(m1_iw + 3) < (uint)m1_Win) m1_r" << rr << ".w = m1_xr[m1_iw + 3];\n";
        os_ << "          }\n";
      }
      os_ << "        }\n";
      os_ << "      }\n";
      os_ << "      const uint m1_vo = (uint)m1_cl * m1_VLD + (uint)m1_tl;\n";
      os_ << "      const float4 m1_t0 = m1_r0 - m1_r2;\n";
      os_ << "      const float4 m1_t1 = m1_r1 + m1_r2;\n";
      os_ << "      const float4 m1_t2 = m1_r2 - m1_r1;\n";
      os_ << "      const float4 m1_t3 = m1_r1 - m1_r3;\n";
      for (int rr = 0; rr < 4; ++rr) {
        os_ << "      m1_smem[" << (rr * 4 + 0) << "u * m1_VPLANE + m1_vo] = m1_t" << rr << ".x - m1_t" << rr << ".z;\n";
        os_ << "      m1_smem[" << (rr * 4 + 1) << "u * m1_VPLANE + m1_vo] = m1_t" << rr << ".y + m1_t" << rr << ".z;\n";
        os_ << "      m1_smem[" << (rr * 4 + 2) << "u * m1_VPLANE + m1_vo] = m1_t" << rr << ".z - m1_t" << rr << ".y;\n";
        os_ << "      m1_smem[" << (rr * 4 + 3) << "u * m1_VPLANE + m1_vo] = m1_t" << rr << ".y - m1_t" << rr << ".w;\n";
      }
      os_ << "    }\n";
      os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      emit_wino_mma_block(aps, wfm, wfn, plan_.bk, "    ");
      os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "  }\n";
    }

    for (uint32_t i = 0; i < wfm; ++i) {
      for (uint32_t a = 0; a < aps; ++a) {
        os_ << "  {\n";
        os_ << "    threadgroup float* m1_es = m1_smem + ((uint)m1_sg + " << (a * plan_.num_sgs) << "u) * m1_EPLANE;\n";
        for (uint32_t j = 0; j < wfn; ++j) {
          os_ << "    simdgroup_store(m1_acc[" << a << "][" << i << "][" << j << "], m1_es + " << (j * 8u)
              << "u, m1_ELD);\n";
        }
        os_ << "  }\n";
      }
      os_ << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      if (plan_.fused_tail) {
        const auto tk = plan_.fused_tail->kind;
        const bool is_max = (tk == ir::TailKind::SpatialGlobalMax);
        const std::string ps_lit = msl_exact_float_literal(plan_.fused_tail->post_scale);
        os_ << "  {\n";
        os_ << "    float m1_pv = " << (is_max ? "-INFINITY" : "0.0f") << ";\n";
        os_ << "    if (m1_tid < " << epi_pairs << "u) {\n";
        os_ << "      const int m1_ecl = (int)(m1_tid / (uint)m1_BN);\n";
        os_ << "      const int m1_etl = (int)(m1_tid % (uint)m1_BN);\n";
        os_ << "      const int m1_co = m1_co0 + " << (i * 8u) << " + m1_ecl;\n";
        os_ << "      const int m1_elty = m1_etl / m1_TTW;\n";
        os_ << "      const int m1_eltx = m1_etl - m1_elty * m1_TTW;\n";
        os_ << "      const int m1_oh0 = (m1_ty0 + m1_elty) * 2;\n";
        os_ << "      const int m1_ow0 = (m1_tx0 + m1_eltx) * 2;\n";
        os_ << "      if (m1_co < m1_Cout && m1_oh0 < m1_Hout && m1_ow0 < m1_Wout) {\n";
        os_ << "        const uint m1_eoff = (uint)m1_ecl * m1_ELD + (uint)m1_etl;\n";
        for (int a = 0; a < 16; ++a) {
          os_ << "        const float m1_m" << a << " = m1_smem[" << a << "u * m1_EPLANE + m1_eoff];\n";
        }
        for (int c = 0; c < 4; ++c) {
          os_ << "        const float m1_s0" << c << " = m1_m" << c << " + m1_m" << (4 + c) << " + m1_m" << (8 + c)
              << ";\n";
          os_ << "        const float m1_s1" << c << " = m1_m" << (4 + c) << " - m1_m" << (8 + c) << " - m1_m"
              << (12 + c) << ";\n";
        }
        os_ << "        float m1_y00 = m1_s00 + m1_s01 + m1_s02;\n";
        os_ << "        float m1_y01 = m1_s01 - m1_s02 - m1_s03;\n";
        os_ << "        float m1_y10 = m1_s10 + m1_s11 + m1_s12;\n";
        os_ << "        float m1_y11 = m1_s11 - m1_s12 - m1_s13;\n";
        if (plan_.attr.has_bias) {
          os_ << "        const float m1_bv = (float)" << plan_.bias_name << "[m1_co];\n";
          os_ << "        m1_y00 += m1_bv; m1_y01 += m1_bv; m1_y10 += m1_bv; m1_y11 += m1_bv;\n";
        }
        os_ << "        const long m1_oc_base = ((long)m1_n * (long)m1_Cout + (long)m1_co) * m1_Pout;\n";
        os_ << "        const long m1_r0_idx = m1_oc_base + (long)m1_oh0 * (long)m1_Wout + (long)m1_ow0;\n";
        os_ << "        const long m1_r1_idx = m1_oc_base + (long)(m1_oh0 + 1) * (long)m1_Wout + (long)m1_ow0;\n";
        os_ << "        (void)m1_r0_idx; (void)m1_r1_idx;\n";
        if (plan_.epilogue_dag) {
          os_ << "        {\n";
          emit_epilogue_steps("m1_y00", "m1_co", "m1_r0_idx", "          ");
          os_ << "        }\n";
        }
        os_ << "        m1_pv = m1_y00;\n";
        os_ << "        if (m1_ow0 + 1 < m1_Wout) {\n";
        if (plan_.epilogue_dag) {
          os_ << "          {\n";
          emit_epilogue_steps("m1_y01", "m1_co", "(m1_r0_idx + 1L)", "            ");
          os_ << "          }\n";
        }
        if (is_max) {
          os_ << "          m1_pv = (m1_y01 > m1_pv || isnan(m1_y01)) ? m1_y01 : m1_pv;\n";
        } else {
          os_ << "          m1_pv += m1_y01;\n";
        }
        os_ << "        }\n";
        os_ << "        if (m1_oh0 + 1 < m1_Hout) {\n";
        if (plan_.epilogue_dag) {
          os_ << "          {\n";
          emit_epilogue_steps("m1_y10", "m1_co", "m1_r1_idx", "            ");
          os_ << "          }\n";
        }
        if (is_max) {
          os_ << "          m1_pv = (m1_y10 > m1_pv || isnan(m1_y10)) ? m1_y10 : m1_pv;\n";
        } else {
          os_ << "          m1_pv += m1_y10;\n";
        }
        os_ << "          if (m1_ow0 + 1 < m1_Wout) {\n";
        if (plan_.epilogue_dag) {
          os_ << "            {\n";
          emit_epilogue_steps("m1_y11", "m1_co", "(m1_r1_idx + 1L)", "              ");
          os_ << "            }\n";
        }
        if (is_max) {
          os_ << "            m1_pv = (m1_y11 > m1_pv || isnan(m1_y11)) ? m1_y11 : m1_pv;\n";
        } else {
          os_ << "            m1_pv += m1_y11;\n";
        }
        os_ << "          }\n";
        os_ << "        }\n";
        os_ << "      }\n";
        os_ << "    }\n";
        os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        os_ << "    if (m1_tid < " << epi_pairs << "u) m1_smem[m1_tid] = m1_pv;\n";
        os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        os_ << "    if (m1_tid < 8u) {\n";
        os_ << "      const int m1_co = m1_co0 + " << (i * 8u) << " + (int)m1_tid;\n";
        os_ << "      if (m1_co < m1_Cout) {\n";
        os_ << "        float m1_fin = m1_smem[m1_tid * (uint)m1_BN];\n";
        os_ << "        for (uint m1_t = 1u; m1_t < (uint)m1_BN; ++m1_t) {\n";
        os_ << "          const float m1_tv = m1_smem[m1_tid * (uint)m1_BN + m1_t];\n";
        if (is_max) {
          os_ << "          m1_fin = (m1_tv > m1_fin || isnan(m1_tv)) ? m1_tv : m1_fin;\n";
        } else {
          os_ << "          m1_fin += m1_tv;\n";
        }
        os_ << "        }\n";
        if (tk == ir::TailKind::SpatialGlobalAvgPool) {
          os_ << "        m1_fin *= (" << ps_lit << " / (float)m1_Pout);\n";
        } else if (tk == ir::TailKind::SpatialGlobalSum) {
          if (plan_.fused_tail->post_scale != 1.0f) {
            os_ << "        m1_fin *= " << ps_lit << ";\n";
          }
        } else if (plan_.fused_tail->post_scale != 1.0f) {
          os_ << "        if (m1_num_tgs.x == 1u) m1_fin *= " << ps_lit << ";\n";
        }
        os_ << "        " << plan_.out_name
            << "[((long)m1_n * (long)m1_Cout + (long)m1_co) * (long)m1_num_tgs.x + (long)m1_tgid.x] = m1_fin;\n";
        os_ << "      }\n";
        os_ << "    }\n";
        os_ << "  }\n";
      } else {
        os_ << "  for (uint m1_ei = m1_tid; m1_ei < " << epi_pairs << "u; m1_ei += " << plan_.num_threads << "u) {\n";
        os_ << "    const int m1_ecl = (int)(m1_ei / (uint)m1_BN);\n";
        os_ << "    const int m1_etl = (int)(m1_ei % (uint)m1_BN);\n";
        os_ << "    const int m1_co = m1_co0 + " << (i * 8u) << " + m1_ecl;\n";
        os_ << "    const int m1_elty = m1_etl / m1_TTW;\n";
        os_ << "    const int m1_eltx = m1_etl - m1_elty * m1_TTW;\n";
        os_ << "    const int m1_oh0 = (m1_ty0 + m1_elty) * 2;\n";
        os_ << "    const int m1_ow0 = (m1_tx0 + m1_eltx) * 2;\n";
        os_ << "    if (m1_co < m1_Cout && m1_oh0 < m1_Hout && m1_ow0 < m1_Wout) {\n";
        os_ << "      const uint m1_eoff = (uint)m1_ecl * m1_ELD + (uint)m1_etl;\n";
        for (int a = 0; a < 16; ++a) {
          os_ << "      const float m1_m" << a << " = m1_smem[" << a << "u * m1_EPLANE + m1_eoff];\n";
        }
        for (int c = 0; c < 4; ++c) {
          os_ << "      const float m1_s0" << c << " = m1_m" << c << " + m1_m" << (4 + c) << " + m1_m" << (8 + c)
              << ";\n";
          os_ << "      const float m1_s1" << c << " = m1_m" << (4 + c) << " - m1_m" << (8 + c) << " - m1_m" << (12 + c)
              << ";\n";
        }
        os_ << "      float m1_y00 = m1_s00 + m1_s01 + m1_s02;\n";
        os_ << "      float m1_y01 = m1_s01 - m1_s02 - m1_s03;\n";
        os_ << "      float m1_y10 = m1_s10 + m1_s11 + m1_s12;\n";
        os_ << "      float m1_y11 = m1_s11 - m1_s12 - m1_s13;\n";
        if (plan_.attr.has_bias) {
          os_ << "      const float m1_bv = (float)" << plan_.bias_name << "[m1_co];\n";
          os_ << "      m1_y00 += m1_bv; m1_y01 += m1_bv; m1_y10 += m1_bv; m1_y11 += m1_bv;\n";
        }
        if (plan_.epilogue_dag) {
          os_ << "      const long m1_oc_base = ((long)m1_n * (long)m1_Cout + (long)m1_co) * m1_Pout;\n";
          os_ << "      const long m1_r0_off = (long)m1_oh0 * (long)m1_Wout + (long)m1_ow0;\n";
          os_ << "      const long m1_r1_off = (long)(m1_oh0 + 1) * (long)m1_Wout + (long)m1_ow0;\n";
          os_ << "      device float* m1_op = " << plan_.out_name << " + m1_oc_base;\n";
          os_ << "      if (m1_oh0 + 1 < m1_Hout && m1_ow0 + 1 < m1_Wout) {\n";
          os_ << "        {\n";
          emit_epilogue_steps("m1_y00", "m1_co", "(m1_oc_base + m1_r0_off)", "          ");
          os_ << "        }\n";
          os_ << "        {\n";
          emit_epilogue_steps("m1_y01", "m1_co", "(m1_oc_base + m1_r0_off + 1L)", "          ");
          os_ << "        }\n";
          os_ << "        {\n";
          emit_epilogue_steps("m1_y10", "m1_co", "(m1_oc_base + m1_r1_off)", "          ");
          os_ << "        }\n";
          os_ << "        {\n";
          emit_epilogue_steps("m1_y11", "m1_co", "(m1_oc_base + m1_r1_off + 1L)", "          ");
          os_ << "        }\n";
          os_ << "        *(device packed_float2*)(m1_op + m1_r0_off) = packed_float2(m1_y00, m1_y01);\n";
          os_ << "        *(device packed_float2*)(m1_op + m1_r1_off) = packed_float2(m1_y10, m1_y11);\n";
          os_ << "      } else {\n";
          os_ << "        {\n";
          emit_epilogue_steps("m1_y00", "m1_co", "(m1_oc_base + m1_r0_off)", "          ");
          os_ << "        }\n";
          os_ << "        m1_op[m1_r0_off] = m1_y00;\n";
          os_ << "        if (m1_ow0 + 1 < m1_Wout) {\n";
          os_ << "          {\n";
          emit_epilogue_steps("m1_y01", "m1_co", "(m1_oc_base + m1_r0_off + 1L)", "            ");
          os_ << "          }\n";
          os_ << "          m1_op[m1_r0_off + 1L] = m1_y01;\n";
          os_ << "        }\n";
          os_ << "        if (m1_oh0 + 1 < m1_Hout) {\n";
          os_ << "          {\n";
          emit_epilogue_steps("m1_y10", "m1_co", "(m1_oc_base + m1_r1_off)", "            ");
          os_ << "          }\n";
          os_ << "          m1_op[m1_r1_off] = m1_y10;\n";
          os_ << "          if (m1_ow0 + 1 < m1_Wout) {\n";
          os_ << "            {\n";
          emit_epilogue_steps("m1_y11", "m1_co", "(m1_oc_base + m1_r1_off + 1L)", "              ");
          os_ << "            }\n";
          os_ << "            m1_op[m1_r1_off + 1L] = m1_y11;\n";
          os_ << "          }\n";
          os_ << "        }\n";
          os_ << "      }\n";
        } else {
          os_ << "      device float* m1_op = " << plan_.out_name
              << " + ((long)m1_n * (long)m1_Cout + (long)m1_co) * m1_Pout;\n";
          os_ << "      if (m1_oh0 + 1 < m1_Hout && m1_ow0 + 1 < m1_Wout) {\n";
          os_ << "        *(device packed_float2*)(m1_op + (long)m1_oh0 * (long)m1_Wout + (long)m1_ow0) = "
                 "packed_float2(m1_y00, m1_y01);\n";
          os_ << "        *(device packed_float2*)(m1_op + (long)(m1_oh0 + 1) * (long)m1_Wout + (long)m1_ow0) = "
                 "packed_float2(m1_y10, m1_y11);\n";
          os_ << "      } else {\n";
          os_ << "        m1_op[(long)m1_oh0 * (long)m1_Wout + (long)m1_ow0] = m1_y00;\n";
          os_ << "        if (m1_ow0 + 1 < m1_Wout) m1_op[(long)m1_oh0 * (long)m1_Wout + (long)(m1_ow0 + 1)] = m1_y01;\n";
          os_ << "        if (m1_oh0 + 1 < m1_Hout) {\n";
          os_ << "          m1_op[(long)(m1_oh0 + 1) * (long)m1_Wout + (long)m1_ow0] = m1_y10;\n";
          os_ << "          if (m1_ow0 + 1 < m1_Wout) m1_op[(long)(m1_oh0 + 1) * (long)m1_Wout + (long)(m1_ow0 + 1)] = "
                 "m1_y11;\n";
          os_ << "        }\n";
          os_ << "      }\n";
        }
        os_ << "    }\n";
        os_ << "  }\n";
      }
      if (i + 1u < wfm) {
        os_ << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      }
    }
    os_ << "}\n";
    return os_.str();
  }

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
    emit_helper_functions(os_, plan_.helpers);
    emit_kernel_param_decls(os_, kernel_, plan_.out_name);
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    if (plan_.fused_tail) {
      os_ << "    uint3 m1_num_tgs [[threadgroups_per_grid]],\n";
    }
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
    os_ << "  const ulong m1_HWo64 = (ulong)m1_Hout * (ulong)m1_Wout;\n";
    os_ << "  const ulong m1_P64 = (ulong)m1_Dout * m1_HWo64;\n";
    if (plan_.fused_tail) {
      os_ << "  const bool m1_m32 = ((ulong)m1_N * m1_P64 <= 0xfffffffful);\n";
      os_ << "  const uint m1_HWo = (uint)m1_HWo64;\n";
      os_ << "  const uint m1_P = (uint)m1_P64;\n";
      os_ << "  const long m1_Pin = (long)m1_Din * (long)m1_Hin * (long)m1_Win;\n";
      os_ << "  typedef " << (plan_.idx32 ? "int" : "long") << " m1_idx;\n";
      os_ << "  const m1_idx m1_PinI = (m1_idx)m1_Pin;\n";
      os_ << "  const int m1_n_tg = (int)m1_tgid.z / m1_G;\n";
      os_ << "  const int m1_g = (int)m1_tgid.z - m1_n_tg * m1_G;\n";
      os_ << "  const int m1_co0 = (int)m1_tgid.y * m1_BM;\n";
      os_ << "  const ulong m1_p_tile0 = (ulong)m1_tgid.x * (ulong)m1_BN;\n";
      os_ << "  if (m1_n_tg >= (int)m1_N || m1_co0 >= m1_Cog || m1_p_tile0 >= m1_P64) return;\n";
      os_ << "  const ulong m1_m0 = (ulong)m1_n_tg * m1_P64 + m1_p_tile0;\n";
      os_ << "  const ulong m1_M = (ulong)(m1_n_tg + 1) * m1_P64;\n";
    } else {
      os_ << "  const ulong m1_M = (ulong)m1_N * m1_P64;\n";
      os_ << "  const bool m1_m32 = (m1_M <= 0xfffffffful);\n";
      os_ << "  const uint m1_HWo = (uint)m1_HWo64;\n";
      os_ << "  const uint m1_P = (uint)m1_P64;\n";
      os_ << "  const long m1_Pin = (long)m1_Din * (long)m1_Hin * (long)m1_Win;\n";
      os_ << "  typedef " << (plan_.idx32 ? "int" : "long") << " m1_idx;\n";
      os_ << "  const m1_idx m1_PinI = (m1_idx)m1_Pin;\n";
      os_ << "  const int m1_g = (int)m1_tgid.z;\n";
      os_ << "  const int m1_co0 = (int)m1_tgid.y * m1_BM;\n";
      os_ << "  const ulong m1_m0 = (ulong)m1_tgid.x * (ulong)m1_BN;\n";
      os_ << "  if (m1_g >= m1_G || m1_co0 >= m1_Cog || m1_m0 >= m1_M) return;\n";
    }
    os_ << "  threadgroup float m1_smem[" << plan_.smem_floats << "];\n";
    os_ << "  threadgroup float* m1_sw = m1_smem;\n";
    os_ << "  threadgroup float* m1_sx = m1_smem + m1_BM * m1_LDW;\n";
    os_ << "  const int m1_wc = (int)(m1_tid % " << w_vpr() << "u) * 4;\n";
    os_ << "  const int m1_wr0 = (int)(m1_tid / " << w_vpr() << "u);\n";
    os_ << "  const int m1_xc = (int)(m1_tid % " << x_vpr() << "u) * 4;\n";
    os_ << "  const int m1_xr0 = (int)(m1_tid / " << x_vpr() << "u);\n";
    os_ << "  device const " << plan_.w_elem << "* m1_wb = " << plan_.w_name
        << " + (long)(m1_g * m1_Cog + m1_co0) * (long)m1_K;\n";
    os_ << "  const ulong m1_mx = m1_m0 + (ulong)m1_xc;\n";
    os_ << "  const ulong m1_nx = (m1_mx < m1_M) ? (m1_m32 ? (ulong)((uint)m1_mx / m1_P) : (m1_mx / m1_P64)) : 0ul;\n";
    os_ << "  const ulong m1_px = (m1_mx < m1_M) ? (m1_mx - m1_nx * m1_P64) : 0ul;\n";
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
    os_ << "  const bool m1_x4 = (m1_mx + 3ul < m1_M) && (m1_px + 3ul < m1_P64);\n";
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
      os_ << "            const ulong m1_me = m1_mx + (ulong)m1_e;\n";
      os_ << "            if (m1_me < m1_M) {\n";
      os_ << "              const ulong m1_ne = m1_m32 ? (ulong)((uint)m1_me / m1_P) : (m1_me / m1_P64);\n";
      os_ << "              const ulong m1_pe = m1_me - m1_ne * m1_P64;\n";
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

    os_ << "  const ulong m1_m1 = m1_m0 + (ulong)(m1_BN - 1);\n";
    os_ << "  const ulong m1_n0 = m1_m32 ? (ulong)((uint)m1_m0 / m1_P) : (m1_m0 / m1_P64);\n";
    os_ << "  const ulong m1_p0 = m1_m0 - m1_n0 * m1_P64;\n";
    os_ << "  const ulong m1_n1 = (m1_m1 < m1_M) ? (m1_m32 ? (ulong)((uint)m1_m1 / m1_P) : (m1_m1 / m1_P64)) : (m1_n0 + 1ul);\n";
    os_ << "  const ulong m1_p1 = (m1_m1 < m1_M) ? (m1_m1 - m1_n1 * m1_P64) : 0ul;\n";
    os_ << "  const int m1_od0 = (int)(m1_m32 ? ((uint)m1_p0 / m1_HWo) : (uint)(m1_p0 / m1_HWo64));\n";
    os_ << "  const ulong m1_pr0 = m1_p0 - (ulong)m1_od0 * m1_HWo64;\n";
    os_ << "  const int m1_oh0 = (int)(m1_m32 ? ((uint)m1_pr0 / (uint)m1_Wout) : (uint)(m1_pr0 / (ulong)m1_Wout));\n";
    os_ << "  const int m1_ow0 = (int)(m1_pr0 - (ulong)m1_oh0 * (ulong)m1_Wout);\n";
    os_ << "  const int m1_id_lo = m1_od0 * " << D.s << " - " << D.p << ";\n";
    os_ << "  const int m1_ih_lo = m1_oh0 * " << H.s << " - " << H.p << ";\n";
    os_ << "  const int m1_iw_lo = m1_ow0 * " << W.s << " - " << W.p << ";\n";
    os_ << "  const bool m1_tg_fast = (m1_co0 + m1_BM <= m1_Cog) && (m1_m1 < m1_M) && (m1_n0 == m1_n1) &&\n";
    os_ << "      (m1_m32 ? ((uint)m1_p0 / (uint)m1_Wout == (uint)m1_p1 / (uint)m1_Wout) : (m1_p0 / (ulong)m1_Wout == m1_p1 / (ulong)m1_Wout)) &&\n";
    os_ << "      (m1_id_lo >= 0) && (m1_id_lo + " << ((D.k - 1) * D.d) << " < m1_Din) &&\n";
    os_ << "      (m1_ih_lo >= 0) && (m1_ih_lo + " << ((H.k - 1) * H.d) << " < m1_Hin) &&\n";
    os_ << "      (m1_iw_lo >= 0) && (m1_iw_lo + " << ((static_cast<int32_t>(plan_.bn) - 1) * W.s + (W.k - 1) * W.d)
        << " < m1_Win);\n";

    const auto emit_step_advance = [&](uint32_t p, const std::string& ind) {
      if (d_kw != 0) os_ << ind << "m1_kw" << p << " += " << d_kw << ";\n";
      if (H.k > 1 && d_kh != 0) os_ << ind << "m1_kh" << p << " += " << d_kh << ";\n";
      if (D.k > 1 && d_kd != 0) os_ << ind << "m1_kd" << p << " += " << d_kd << ";\n";
      if (d_ci != 0) os_ << ind << "m1_ci" << p << " += " << d_ci << ";\n";
      if (W.k > 1) {
        os_ << ind << "if (m1_kw" << p << " >= m1_KW) { m1_kw" << p << " -= m1_KW; ";
        if (H.k > 1) {
          os_ << "++m1_kh" << p << "; }\n";
        } else if (D.k > 1) {
          os_ << "++m1_kd" << p << "; }\n";
        } else {
          os_ << "++m1_ci" << p << "; }\n";
        }
      }
      if (H.k > 1) {
        os_ << ind << "if (m1_kh" << p << " >= m1_KH) { m1_kh" << p << " -= m1_KH; ";
        if (D.k > 1) {
          os_ << "++m1_kd" << p << "; }\n";
        } else {
          os_ << "++m1_ci" << p << "; }\n";
        }
      }
      if (D.k > 1) {
        os_ << ind << "if (m1_kd" << p << " >= m1_KD) { m1_kd" << p << " -= m1_KD; ++m1_ci" << p << "; }\n";
      }
      os_ << ind << "m1_off" << p << " = (m1_idx)m1_ci" << p << " * m1_PinI";
      if (D.k > 1) os_ << " + (m1_idx)(m1_kd" << p << " * " << D.d << ") * (m1_idx)m1_Hin * (m1_idx)m1_Win";
      if (H.k > 1) os_ << " + (m1_idx)(m1_kh" << p << " * " << H.d << ") * (m1_idx)m1_Win";
      if (W.k > 1) os_ << " + (m1_idx)(m1_kw" << p << " * " << W.d << ")";
      os_ << ";\n";
    };

    const bool pipe_fast = (W.s == 1) && use_step && (w_per() <= 2) && (x_per() <= 2);

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

    if (pipe_fast) {
      os_ << "    if (m1_KFull > 0) {\n";
      os_ << "      float4 m1_rw[" << w_per() << "];\n";
      os_ << "      float4 m1_rx[" << x_per() << "];\n";
      for (uint32_t p = 0; p < w_per(); ++p) {
        const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
        if (guard) os_ << "      if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
        os_ << "      *(threadgroup float4*)(m1_sw + (m1_wr0 + " << (p * w_rstep()) << ") * m1_LDW + m1_wc) = "
            << "float4(*(const device " << wpv4 << "*)(m1_wp" << p << "));\n";
      }
      for (uint32_t p = 0; p < x_per(); ++p) {
        const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
        if (guard) os_ << "      if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK) {\n";
        os_ << "      *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep())
            << ") * m1_LDX + m1_xc) = float4(*(const device " << xpv4 << "*)(m1_xpix + m1_off" << p << "));\n";
        emit_step_advance(p, "      ");
        if (guard) os_ << "      }\n";
      }
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "      for (int m1_k0 = 0; m1_k0 + m1_BK < m1_KFull; m1_k0 += m1_BK) {\n";
      os_ << "        const int m1_kn = m1_k0 + m1_BK;\n";
      for (uint32_t p = 0; p < w_per(); ++p) {
        const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
        if (guard) os_ << "        if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
        os_ << "        m1_rw[" << p << "] = float4(*(const device " << wpv4 << "*)(m1_wp" << p << " + m1_kn));\n";
      }
      for (uint32_t p = 0; p < x_per(); ++p) {
        const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
        if (guard) os_ << "        if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK) {\n";
        os_ << "        m1_rx[" << p << "] = float4(*(const device " << xpv4 << "*)(m1_xpix + m1_off" << p << "));\n";
        emit_step_advance(p, "        ");
        if (guard) os_ << "        }\n";
      }
      emit_mma("        ");
      os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      for (uint32_t p = 0; p < w_per(); ++p) {
        const bool guard = need_w_guard && ((p + 1) * w_rstep() > plan_.bm);
        if (guard) os_ << "        if (m1_wr0 + " << (p * w_rstep()) << " < m1_BM)\n  ";
        os_ << "        *(threadgroup float4*)(m1_sw + (m1_wr0 + " << (p * w_rstep()) << ") * m1_LDW + m1_wc) = m1_rw["
            << p << "];\n";
      }
      for (uint32_t p = 0; p < x_per(); ++p) {
        const bool guard = need_x_guard && ((p + 1) * x_rstep() > plan_.bk);
        if (guard) os_ << "        if (m1_xr0 + " << (p * x_rstep()) << " < m1_BK)\n  ";
        os_ << "        *(threadgroup float4*)(m1_sx + (m1_xr0 + " << (p * x_rstep()) << ") * m1_LDX + m1_xc) = m1_rx["
            << p << "];\n";
      }
      os_ << "        threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "      }\n";
      emit_mma("      ");
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "    }\n";
    } else {
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
          emit_step_advance(p, "        ");
        }
        if (guard) os_ << "        }\n";
        os_ << "      }\n";
      }
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      emit_mma("      ");
      os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
      os_ << "    }\n";
    }

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

    const bool no_pad = (D.p == 0 && H.p == 0 && W.p == 0);
    os_ << "  } else {\n";
    os_ << "    const int m1_odx = (int)(m1_m32 ? ((uint)m1_px / m1_HWo) : (uint)(m1_px / m1_HWo64));\n";
    os_ << "    const ulong m1_prx = m1_px - (ulong)m1_odx * m1_HWo64;\n";
    os_ << "    const int m1_ohx = (int)(m1_m32 ? ((uint)m1_prx / (uint)m1_Wout) : (uint)(m1_prx / (ulong)m1_Wout));\n";
    os_ << "    const int m1_owx = (int)(m1_prx - (ulong)m1_ohx * (ulong)m1_Wout);\n";
    os_ << "    const int m1_idx0 = m1_odx * " << D.s << " - " << D.p << ";\n";
    os_ << "    const int m1_ihx0 = m1_ohx * " << H.s << " - " << H.p << ";\n";
    os_ << "    const int m1_iwx0 = m1_owx * " << W.s << " - " << W.p << ";\n";
    os_ << "    const bool m1_x4 = (m1_mx + 3ul < m1_M) && (m1_owx + 3 < m1_Wout) &&\n";
    os_ << "        (m1_idx0 >= 0) && (m1_idx0 + " << ((D.k - 1) * D.d) << " < m1_Din) &&\n";
    os_ << "        (m1_ihx0 >= 0) && (m1_ihx0 + " << ((H.k - 1) * H.d) << " < m1_Hin) &&\n";
    os_ << "        (m1_iwx0 >= 0) && (m1_iwx0 + " << (3 * W.s + (W.k - 1) * W.d) << " < m1_Win);\n";
    os_ << "    const long m1_xoff4 = m1_x4 ? (((long)m1_idx0 * (long)m1_Hin + (long)m1_ihx0) * (long)m1_Win + "
        << "(long)m1_iwx0) : 0L;\n";
    os_ << "    device const " << plan_.x_elem << "* m1_ep[4];\n";
    if (!no_pad) {
      os_ << "    int m1_eid0[4], m1_eih0[4], m1_eiw0[4];\n";
    }
    os_ << "    if (!m1_x4) {\n";
    os_ << "      for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
    os_ << "        const ulong m1_me = m1_mx + (ulong)m1_e;\n";
    os_ << "        if (m1_me < m1_M) {\n";
    os_ << "          const ulong m1_ne = m1_m32 ? (ulong)((uint)m1_me / m1_P) : (m1_me / m1_P64);\n";
    os_ << "          const ulong m1_pe = m1_me - m1_ne * m1_P64;\n";
    os_ << "          const int m1_ode = (int)(m1_m32 ? ((uint)m1_pe / m1_HWo) : (uint)(m1_pe / m1_HWo64));\n";
    os_ << "          const ulong m1_pre = m1_pe - (ulong)m1_ode * m1_HWo64;\n";
    os_ << "          const int m1_ohe = (int)(m1_m32 ? ((uint)m1_pre / (uint)m1_Wout) : (uint)(m1_pre / (ulong)m1_Wout));\n";
    os_ << "          const int m1_owe = (int)(m1_pre - (ulong)m1_ohe * (ulong)m1_Wout);\n";
    if (no_pad) {
      os_ << "          m1_ep[m1_e] = " << plan_.x_name
          << " + ((long)m1_ne * m1_Cin + (long)m1_g * (long)m1_Cig) * m1_Pin"
          << " + ((long)(m1_ode * " << D.s << ") * (long)m1_Hin + (long)(m1_ohe * " << H.s
          << ")) * (long)m1_Win + (long)(m1_owe * " << W.s << ");\n";
    } else {
      os_ << "          m1_ep[m1_e] = " << plan_.x_name
          << " + ((long)m1_ne * m1_Cin + (long)m1_g * (long)m1_Cig) * m1_Pin;\n";
      os_ << "          m1_eid0[m1_e] = m1_ode * " << D.s << " - " << D.p << ";\n";
      os_ << "          m1_eih0[m1_e] = m1_ohe * " << H.s << " - " << H.p << ";\n";
      os_ << "          m1_eiw0[m1_e] = m1_owe * " << W.s << " - " << W.p << ";\n";
    }
    os_ << "        } else {\n";
    os_ << "          m1_ep[m1_e] = nullptr;\n";
    if (!no_pad) {
      os_ << "          m1_eid0[m1_e] = -1; m1_eih0[m1_e] = -1; m1_eiw0[m1_e] = -1;\n";
    }
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "    }\n";
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
      std::string ci_e, kd_e, kh_e, kw_e, off_e;
      if (use_step) {
        ci_e = "m1_ci" + std::to_string(p);
        kd_e = (D.k > 1) ? ("m1_kd" + std::to_string(p)) : "0";
        kh_e = (H.k > 1) ? ("m1_kh" + std::to_string(p)) : "0";
        kw_e = (W.k > 1) ? ("m1_kw" + std::to_string(p)) : "0";
        off_e = "m1_off" + std::to_string(p);
      } else {
        os_ << "          const int m1_ci = m1_k / m1_KS;\n";
        os_ << "          const int m1_rt = m1_k - m1_ci * m1_KS;\n";
        os_ << "          const int m1_kd = m1_rt / m1_KHW;\n";
        os_ << "          const int m1_rt2 = m1_rt - m1_kd * m1_KHW;\n";
        os_ << "          const int m1_kh = m1_rt2 / m1_KW;\n";
        os_ << "          const int m1_kw = m1_rt2 - m1_kh * m1_KW;\n";
        os_ << "          const long m1_off = (long)m1_ci * m1_Pin + ((long)(m1_kd * " << D.d
            << ") * (long)m1_Hin + (long)(m1_kh * " << H.d << ")) * (long)m1_Win + (long)(m1_kw * " << W.d << ");\n";
        ci_e = "m1_ci";
        kd_e = "m1_kd";
        kh_e = "m1_kh";
        kw_e = "m1_kw";
        off_e = "m1_off";
      }
      os_ << "          if (m1_x4) {\n";
      os_ << "            const long m1_o = m1_xoff4 + (long)" << off_e << ";\n";
      if (W.s == 1) {
        os_ << "            m1_v = float4(*(const device " << xpv4 << "*)(m1_xb + m1_o));\n";
      } else {
        os_ << "            device const " << plan_.x_elem << "* m1_xs = m1_xb + m1_o;\n";
        os_ << "            m1_v = float4((float)m1_xs[0], (float)m1_xs[" << W.s << "], (float)m1_xs[" << (2 * W.s)
            << "], (float)m1_xs[" << (3 * W.s) << "]);\n";
      }
      os_ << "          } else {\n";
      if (no_pad) {
        os_ << "            #pragma unroll\n";
        os_ << "            for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
        os_ << "              if (m1_ep[m1_e] != nullptr) m1_v[m1_e] = (float)m1_ep[m1_e][" << off_e << "];\n";
        os_ << "            }\n";
      } else {
        os_ << "            #pragma unroll\n";
        os_ << "            for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
        os_ << "              if (m1_ep[m1_e] != nullptr) {\n";
        os_ << "                const int m1_id = m1_eid0[m1_e] + (" << kd_e << ") * " << D.d << ";\n";
        os_ << "                const int m1_ih = m1_eih0[m1_e] + (" << kh_e << ") * " << H.d << ";\n";
        os_ << "                const int m1_iw = m1_eiw0[m1_e] + (" << kw_e << ") * " << W.d << ";\n";
        os_ << "                if ((uint)m1_id < (uint)m1_Din && (uint)m1_ih < (uint)m1_Hin && (uint)m1_iw < "
            << "(uint)m1_Win) {\n";
        os_ << "                  m1_v[m1_e] = (float)m1_ep[m1_e][(long)" << ci_e
            << " * m1_Pin + ((long)m1_id * (long)m1_Hin + (long)m1_ih) * (long)m1_Win + (long)m1_iw];\n";
        os_ << "                }\n";
        os_ << "              }\n";
        os_ << "            }\n";
      }
      os_ << "          }\n";
      os_ << "        }\n";
      if (use_step) {
        emit_step_advance(p, "        ");
      }
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

  void emit_fused_tail_epilogue() {
    const std::string& oe = plan_.out_elem;
    const uint32_t sg_rows = fm() * 8;
    const uint32_t sg_ld = fn() * 8;
    const uint32_t sg_elems = sg_rows * sg_ld;
    const uint32_t max_sgs = std::max<uint32_t>(1, plan_.smem_floats / sg_elems);
    const auto tk = plan_.fused_tail->kind;
    const bool is_max = (tk == ir::TailKind::SpatialGlobalMax);
    const std::string ps_lit = msl_exact_float_literal(plan_.fused_tail->post_scale);

    os_ << "  threadgroup float m1_tacc[" << (plan_.bm * plan_.sg_n) << "];\n";
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
    os_ << "    for (uint m1_ri = m1_tid; m1_ri < m1_act * " << sg_rows << "u; m1_ri += " << plan_.num_threads
        << "u) {\n";
    os_ << "      const uint m1_lsg = m1_ri / " << sg_rows << "u;\n";
    os_ << "      const uint m1_lr = m1_ri - m1_lsg * " << sg_rows << "u;\n";
    os_ << "      const uint m1_gsg = m1_sb + m1_lsg;\n";
    os_ << "      const uint m1_sgy = m1_gsg / " << plan_.sg_n << "u;\n";
    os_ << "      const uint m1_sgx = m1_gsg - m1_sgy * " << plan_.sg_n << "u;\n";
    os_ << "      const uint m1_cl = m1_sgy * " << sg_rows << "u + m1_lr;\n";
    os_ << "      const int m1_co = m1_co0 + (int)m1_cl;\n";
    os_ << "      const ulong m1_pbase = m1_p_tile0 + (ulong)(m1_sgx * " << sg_ld << "u);\n";
    os_ << "      float m1_racc = " << (is_max ? "-INFINITY" : "0.0f") << ";\n";
    os_ << "      if (m1_co < m1_Cog) {\n";
    os_ << "        const int m1_cg = m1_g * m1_Cog + m1_co;\n";
    os_ << "        const long m1_oc_base = ((long)m1_n_tg * m1_Cout + (long)m1_cg) * (long)m1_P64;\n";
    os_ << "        (void)m1_cg; (void)m1_oc_base;\n";
    if (plan_.attr.has_bias) {
      os_ << "        const float m1_bv = (float)" << plan_.bias_name << "[m1_cg];\n";
    }
    os_ << "        threadgroup const float* m1_row = m1_smem + m1_lsg * " << sg_elems << "u + m1_lr * " << sg_ld
        << "u;\n";
    os_ << "        for (uint m1_lc = 0u; m1_lc < " << sg_ld << "u; ++m1_lc) {\n";
    os_ << "          const ulong m1_p = m1_pbase + (ulong)m1_lc;\n";
    os_ << "          if (m1_p < m1_P64) {\n";
    os_ << "            float m1_ev = m1_row[m1_lc];\n";
    if (plan_.attr.has_bias) {
      os_ << "            m1_ev += m1_bv;\n";
    }
    emit_epilogue_steps("m1_ev", "m1_cg", "(m1_oc_base + (long)m1_p)", "            ");
    if (is_max) {
      os_ << "            m1_racc = (m1_ev > m1_racc || isnan(m1_ev)) ? m1_ev : m1_racc;\n";
    } else {
      os_ << "            m1_racc += m1_ev;\n";
    }
    os_ << "          }\n";
    os_ << "        }\n";
    os_ << "      }\n";
    os_ << "      m1_tacc[m1_cl * " << plan_.sg_n << "u + m1_sgx] = m1_racc;\n";
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
    os_ << "  for (uint m1_cl = m1_tid; m1_cl < (uint)m1_BM; m1_cl += " << plan_.num_threads << "u) {\n";
    os_ << "    const int m1_co = m1_co0 + (int)m1_cl;\n";
    os_ << "    if (m1_co < m1_Cog) {\n";
    os_ << "      float m1_fin = m1_tacc[m1_cl * " << plan_.sg_n << "u];\n";
    os_ << "      for (uint m1_s = 1u; m1_s < " << plan_.sg_n << "u; ++m1_s) {\n";
    os_ << "        const float m1_tv = m1_tacc[m1_cl * " << plan_.sg_n << "u + m1_s];\n";
    if (is_max) {
      os_ << "        m1_fin = (m1_tv > m1_fin || isnan(m1_tv)) ? m1_tv : m1_fin;\n";
    } else {
      os_ << "        m1_fin += m1_tv;\n";
    }
    os_ << "      }\n";
    if (tk == ir::TailKind::SpatialGlobalAvgPool) {
      os_ << "      m1_fin *= (" << ps_lit << " / (float)m1_P64);\n";
    } else if (tk == ir::TailKind::SpatialGlobalSum) {
      if (plan_.fused_tail->post_scale != 1.0f) {
        os_ << "      m1_fin *= " << ps_lit << ";\n";
      }
    } else if (plan_.fused_tail->post_scale != 1.0f) {
      os_ << "      if (m1_num_tgs.x == 1u) m1_fin *= " << ps_lit << ";\n";
    }
    os_ << "      " << plan_.out_name
        << "[((long)m1_n_tg * m1_Cout + (long)(m1_g * m1_Cog + m1_co)) * (long)m1_num_tgs.x + (long)m1_tgid.x] = "
        << oe << "(m1_fin);\n";
    os_ << "    }\n";
    os_ << "  }\n";
  }

  void emit_epilogue() {
    if (plan_.fused_tail) {
      emit_fused_tail_epilogue();
      return;
    }
    const std::string& oe = plan_.out_elem;
    const std::string opv4 = msl_packed_vec4(oe);
    const std::string ov4 = msl_vec4(oe);
    const uint32_t sg_rows = fm() * 8;
    const uint32_t sg_ld = fn() * 8;
    const uint32_t sg_elems = sg_rows * sg_ld;
    const uint32_t max_sgs = std::max<uint32_t>(1, plan_.smem_floats / sg_elems);
    const uint32_t v4_per_sg = sg_elems / 4;
    const uint32_t v4_per_row = sg_ld / 4;

    os_ << "  device " << oe << "* m1_ob = " << plan_.out_name << " + (long)m1_g * (long)m1_Cog * (long)m1_P64;\n";
    if (!plan_.attr.has_bias && !plan_.epilogue_dag && oe == "float") {
      os_ << "  if (m1_co0 + m1_BM <= m1_Cog && m1_m0 + (ulong)(m1_BN - 1) < m1_M && (m1_P64 & 1023ul) != 0ul) {\n";
      os_ << "    const ulong m1_en0 = m1_m32 ? (ulong)((uint)m1_m0 / m1_P) : (m1_m0 / m1_P64);\n";
      os_ << "    const ulong m1_ep0 = m1_m0 - m1_en0 * m1_P64;\n";
      os_ << "    if (m1_ep0 + (ulong)m1_BN <= m1_P64) {\n";
      os_ << "      device float* m1_od = m1_ob + ((long)m1_en0 * m1_Cout + (long)(m1_co0 + m1_sgr)) * (long)m1_P64 + "
          << "(long)(m1_ep0 + (ulong)m1_sgc);\n";
      for (uint32_t i = 0; i < fm(); ++i) {
        for (uint32_t j = 0; j < fn(); ++j) {
          os_ << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_od + " << (i * 8)
              << "L * (long)m1_P64 + " << (j * 8) << "L, (ulong)m1_P64);\n";
        }
      }
      os_ << "      return;\n";
      os_ << "    }\n";
      os_ << "  }\n";
    }
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
    os_ << "      const ulong m1_mm = m1_m0 + (ulong)((m1_gsg % " << plan_.sg_n << "u) * " << sg_ld << "u + m1_lc);\n";
    os_ << "      if (m1_co < m1_Cog && m1_mm < m1_M) {\n";
    os_ << "        float4 m1_ov = *(const threadgroup float4*)(m1_smem + m1_lsg * " << sg_elems << "u + m1_lr * "
        << sg_ld << "u + m1_lc);\n";
    if (plan_.attr.has_bias) {
      os_ << "        m1_ov += float4((float)" << plan_.bias_name << "[m1_g * m1_Cog + m1_co]);\n";
    }
    os_ << "        const ulong m1_nn = m1_m32 ? (ulong)((uint)m1_mm / m1_P) : (m1_mm / m1_P64);\n";
    os_ << "        const ulong m1_pp = m1_mm - m1_nn * m1_P64;\n";
    if (plan_.epilogue_dag) {
      os_ << "        const int m1_cg = m1_g * m1_Cog + m1_co;\n";
      os_ << "        if (m1_mm + 3ul < m1_M && m1_pp + 3ul < m1_P64) {\n";
      os_ << "          const long m1_base_idx = ((long)m1_nn * m1_Cout + (long)m1_cg) * (long)m1_P64 + (long)m1_pp;\n";
      os_ << "          #pragma unroll\n";
      os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "            float m1_ev = m1_ov[m1_e];\n";
      emit_epilogue_steps("m1_ev", "m1_cg", "(m1_base_idx + (long)m1_e)", "            ");
      os_ << "            m1_ov[m1_e] = m1_ev;\n";
      os_ << "          }\n";
      os_ << "          *(device " << opv4 << "*)(" << plan_.out_name << " + m1_base_idx) = " << opv4 << "(" << ov4
          << "(m1_ov));\n";
      os_ << "        } else {\n";
      os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "            const ulong m1_me = m1_mm + (ulong)m1_e;\n";
      os_ << "            if (m1_me < m1_M) {\n";
      os_ << "              const ulong m1_ne = m1_m32 ? (ulong)((uint)m1_me / m1_P) : (m1_me / m1_P64);\n";
      os_ << "              const ulong m1_pe = m1_me - m1_ne * m1_P64;\n";
      os_ << "              const long m1_eidx = ((long)m1_ne * m1_Cout + (long)m1_cg) * (long)m1_P64 + (long)m1_pe;\n";
      os_ << "              float m1_ev = m1_ov[m1_e];\n";
      emit_epilogue_steps("m1_ev", "m1_cg", "m1_eidx", "              ");
      os_ << "              " << plan_.out_name << "[m1_eidx] = " << oe << "(m1_ev);\n";
      os_ << "            }\n";
      os_ << "          }\n";
      os_ << "        }\n";
    } else {
      os_ << "        if (m1_mm + 3ul < m1_M && m1_pp + 3ul < m1_P64) {\n";
      os_ << "          *(device " << opv4 << "*)(m1_ob + ((long)m1_nn * m1_Cout + (long)m1_co) * (long)m1_P64 + "
          << "(long)m1_pp) = " << opv4 << "(" << ov4 << "(m1_ov));\n";
      os_ << "        } else {\n";
      os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
      os_ << "            const ulong m1_me = m1_mm + (ulong)m1_e;\n";
      os_ << "            if (m1_me < m1_M) {\n";
      os_ << "              const ulong m1_ne = m1_m32 ? (ulong)((uint)m1_me / m1_P) : (m1_me / m1_P64);\n";
      os_ << "              const ulong m1_pe = m1_me - m1_ne * m1_P64;\n";
      os_ << "              m1_ob[((long)m1_ne * m1_Cout + (long)m1_co) * (long)m1_P64 + (long)m1_pe] = " << oe
          << "(m1_ov[m1_e]);\n";
      os_ << "            }\n";
      os_ << "          }\n";
      os_ << "        }\n";
    }
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
  res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan->out_name);
  return res;
}

}  // namespace m1triton::codegen::metal

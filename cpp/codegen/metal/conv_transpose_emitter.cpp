#include "codegen/metal/conv_transpose_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
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

constexpr const char* kBlockMParam = "BLOCK_M";
constexpr const char* kBlockNParam = "BLOCK_N";
constexpr const char* kBlockKParam = "BLOCK_K";
constexpr const char* kNumSgParam = "NUM_SG";
constexpr const char* kWBlk8Param = "W_BLK8";
constexpr const char* kHaloParam = "HALO";
constexpr const char* kFuseSwParam = "FUSE_SW";
constexpr std::array<const char*, 3> kTileParams = {"TILE_D", "TILE_H", "TILE_W"};

struct ConvDim {
  std::string in = "1L";
  std::string out = "1L";
  int32_t k = 1;
  int32_t s = 1;
  int32_t p = 0;
  int32_t d = 1;
  int32_t step = 1;
  int32_t delta = 1;
  int32_t nmax = 1;
  uint32_t tile = 1;
  uint32_t ext = 1;
};

struct ActiveWPhase {
  int32_t aw = 0;
  std::vector<std::pair<int32_t, int32_t>> taps;
};

struct ConvTransposePlan {
  const Op* op = nullptr;
  ConvAttr attr;
  uint32_t nd = 2;
  std::string x_name;
  std::string w_name;
  std::string out_name;
  std::string bias_name;
  std::string x_elem;
  std::string w_elem;
  std::string out_elem;
  std::string n_expr;
  std::string cin_expr;
  std::string cout_expr;
  std::array<ConvDim, 3> dims;
  int32_t groups = 1;
  bool w_blk8 = false;
  bool halo = false;
  bool fuse_sw = false;
  bool ci_exact = false;
  bool co_exact = false;
  uint32_t threads = 128;
  uint32_t num_sg = 4;
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t sg_m = 2;
  uint32_t sg_n = 2;
  uint32_t fm = 4;
  uint32_t fn = 4;
  uint32_t wm = 32;
  uint32_t wn = 32;
  uint32_t a_ld = 68;
  uint32_t b_ld = 68;
  uint32_t hplane = 0;
  uint32_t hplane_ld = 0;
  uint32_t smem_floats = 0;
  int32_t fuse_shift_min_w = 0;
  uint32_t fuse_epi_rows = 8;
  std::vector<ActiveWPhase> active_w;
  std::optional<ir::EpilogueDAGAttr> epilogue_dag;
  std::optional<ir::FusedTailAttr> fused_tail;
};

bool is_supported_spatial_tail(ir::TailKind kind) {
  return kind == ir::TailKind::SpatialGlobalAvgPool || kind == ir::TailKind::SpatialGlobalSum ||
         kind == ir::TailKind::SpatialGlobalMax;
}

std::optional<int64_t> literal_value(const std::string& expr) {
  if (expr.size() < 2 || expr.back() != 'L') return std::nullopt;
  const std::string digits = expr.substr(0, expr.size() - 1);
  if (digits.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
  return std::strtoll(digits.c_str(), nullptr, 10);
}

std::optional<ScalarDType> float_ptr_dtype(const Kernel& kernel, uint32_t pidx) {
  const ir::KernelParam& kp = kernel.params[pidx];
  if (!kp.type.is_ptr()) return std::nullopt;
  const ScalarDType dt = std::get<ir::PtrType>(kp.type.value).pointee;
  if (dt != ScalarDType::F32 && dt != ScalarDType::F16) return std::nullopt;
  return dt;
}

int32_t vec_at(const std::vector<int32_t>& v, uint32_t i, int32_t fallback) {
  return i < v.size() ? v[i] : fallback;
}

bool vec_ok(const std::vector<int32_t>& v, uint32_t nd) { return v.empty() || v.size() == nd; }

bool choose_tile_layout(ConvTransposePlan& plan) {
  if (plan.bm == 0 || plan.bn == 0 || plan.bk == 0) return false;
  if (plan.bm % 8 != 0 || plan.bk % 8 != 0 || plan.bn % 8 != 0) return false;
  if (plan.num_sg == 0 || plan.num_sg > 8) return false;
  plan.threads = plan.num_sg * 32;
  if (plan.bn > plan.threads || plan.threads % plan.bn != 0) return false;
  bool found = false;
  int64_t best = -1;
  for (uint32_t sm = 1; sm <= plan.num_sg; ++sm) {
    if (plan.num_sg % sm != 0) continue;
    const uint32_t sn = plan.num_sg / sm;
    if (plan.bm % (sm * 8) != 0 || plan.bn % (sn * 8) != 0) continue;
    const uint32_t wm = plan.bm / sm;
    const uint32_t wn = plan.bn / sn;
    if ((wm / 8) * (wn / 8) > 32) continue;
    const int64_t score = static_cast<int64_t>(std::min(wm, wn)) * 1000 - std::abs(static_cast<int64_t>(wm) - wn);
    if (score > best) {
      best = score;
      plan.sg_m = sm;
      plan.sg_n = sn;
      found = true;
    }
  }
  if (!found) return false;
  plan.wm = plan.bm / plan.sg_m;
  plan.wn = plan.bn / plan.sg_n;
  plan.fm = plan.wm / 8;
  plan.fn = plan.wn / 8;
  plan.a_ld = plan.bm + 4;
  plan.b_ld = plan.bn + 4;
  return true;
}

bool configure_halo(const Kernel& kernel, ConvTransposePlan& plan, std::string& err) {
  if (!plan.w_blk8) {
    err = "conv-transpose HALO mode requires W_BLK8=1, f32 weights, and tile-aligned channels";
    return false;
  }
  uint32_t cols = 1;
  plan.hplane = 1;
  for (uint32_t i = 0; i < 3; ++i) {
    ConvDim& dim = plan.dims[i];
    const bool active = i >= 3 - plan.nd;
    const int64_t fallback = i == 2 ? plan.bn : 1;
    dim.tile = active ? static_cast<uint32_t>(positive_constexpr_param(kernel, kTileParams[i], fallback)) : 1u;
    dim.ext = dim.tile + static_cast<uint32_t>((dim.nmax - 1) * dim.delta);
    cols *= dim.tile;
    plan.hplane *= dim.ext;
  }
  if (cols != plan.bn || plan.dims[2].tile % 8 != 0) {
    err = "conv-transpose HALO tile requires TILE_D*TILE_H*TILE_W == BLOCK_N and TILE_W % 8 == 0";
    return false;
  }
  plan.halo = true;
  plan.hplane_ld = plan.hplane + (plan.hplane % 16 == 0 ? 4u : 0u);
  plan.smem_floats = std::max(plan.bk * plan.hplane_ld, plan.wm * plan.wn);
  return true;
}

bool configure_fuse_sw(ConvTransposePlan& plan, const ir::TargetSpec& target, std::string& err) {
  const ConvDim& wdim = plan.dims[2];
  if (!plan.halo || wdim.s <= 1 || plan.out_elem != "float") {
    err = "conv-transpose FUSE_SW=1 requires HALO=1, f32 output, and stride_w > 1";
    return false;
  }
  bool have_shift = false;
  int32_t shift_min = 0;
  int32_t shift_max = 0;
  for (int32_t aw = 0; aw < wdim.s; ++aw) {
    ActiveWPhase ph;
    ph.aw = aw;
    for (int32_t rw = 0; rw < wdim.k; ++rw) {
      const int32_t num = aw + wdim.p - rw * wdim.d;
      if (num % wdim.s == 0) {
        const int32_t sh = num / wdim.s;
        ph.taps.emplace_back(rw, sh);
        if (!have_shift) {
          shift_min = sh;
          shift_max = sh;
          have_shift = true;
        } else {
          shift_min = std::min(shift_min, sh);
          shift_max = std::max(shift_max, sh);
        }
      }
    }
    if (!ph.taps.empty()) {
      plan.active_w.push_back(std::move(ph));
    }
  }
  const uint32_t n_act = static_cast<uint32_t>(plan.active_w.size());
  if (n_act == 0 || n_act * plan.fm * plan.fn > 24u) {
    err = "conv-transpose FUSE_SW=1 exceeds active W-phase accumulator limit";
    return false;
  }
  plan.fuse_sw = true;
  plan.fuse_shift_min_w = shift_min;
  plan.dims[2].ext = plan.dims[2].tile + static_cast<uint32_t>(shift_max - shift_min);
  plan.hplane = plan.dims[0].ext * plan.dims[1].ext * plan.dims[2].ext;
  plan.hplane_ld = plan.hplane + (plan.hplane % 16 == 0 ? 4u : 0u);
  const uint32_t max_epi_floats =
      std::min<uint32_t>(2048u, static_cast<uint32_t>(target.max_threadgroup_memory_bytes / 4u));
  uint32_t epi_rows = 8u;
  for (uint32_t cand = plan.wm; cand >= 8u; cand -= 8u) {
    if (plan.wm % cand == 0 && cand * n_act * plan.bn <= max_epi_floats) {
      epi_rows = cand;
      break;
    }
  }
  plan.fuse_epi_rows = epi_rows;
  plan.smem_floats = std::max(plan.bk * plan.hplane_ld, plan.fuse_epi_rows * n_act * plan.bn);
  return true;
}

std::optional<ConvTransposePlan> build_plan(const Kernel& kernel, const ir::TargetSpec& target, std::string& err) {
  ConvTransposePlan plan;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::ImplicitGemmConv: {
        const auto* ca = std::get_if<ConvAttr>(&op.attr);
        if (ca == nullptr || !ca->transposed) return std::nullopt;
        if (plan.op != nullptr) {
          err = "conv-transpose emitter supports exactly one ImplicitGemmConv op per kernel";
          return std::nullopt;
        }
        plan.op = &op;
        break;
      }
      case OpKind::Constant:
      case OpKind::Cast:
      case OpKind::Return:
        break;
      default:
        return std::nullopt;
    }
  }
  if (plan.op == nullptr) return std::nullopt;

  plan.attr = std::get<ConvAttr>(plan.op->attr);
  const ConvAttr& ca = plan.attr;
  const uint32_t nd = ca.ndim;
  plan.nd = nd;
  const auto& ops = plan.op->operands;
  const size_t bias_slots = ca.has_bias ? 1 : 0;
  if (nd < 1 || nd > 3 || ops.size() != 3 + bias_slots + 3 + 2 * nd || ca.kernel_size.size() != nd ||
      !vec_ok(ca.stride, nd) || !vec_ok(ca.padding, nd) || !vec_ok(ca.dilation, nd) ||
      !vec_ok(ca.output_padding, nd)) {
    err = "conv-transpose op has malformed operands or ConvAttr vectors";
    return std::nullopt;
  }
  if (ca.groups < 1) {
    err = "conv-transpose groups must be >= 1";
    return std::nullopt;
  }
  plan.groups = ca.groups;

  std::array<std::optional<uint32_t>, 4> ptr_idx;
  for (size_t i = 0; i < 3 + bias_slots; ++i) {
    ptr_idx[i] = param_index_of(kernel, ops[i]);
    if (!ptr_idx[i] || !float_ptr_dtype(kernel, *ptr_idx[i])) {
      err = "conv-transpose x/w/out/bias operands must be f32 or f16 pointer kernel parameters";
      return std::nullopt;
    }
  }
  plan.x_name = kernel.params[*ptr_idx[0]].name;
  plan.w_name = kernel.params[*ptr_idx[1]].name;
  plan.out_name = kernel.params[*ptr_idx[2]].name;
  plan.x_elem = msl_scalar_type(*float_ptr_dtype(kernel, *ptr_idx[0]));
  plan.w_elem = msl_scalar_type(*float_ptr_dtype(kernel, *ptr_idx[1]));
  plan.out_elem = msl_scalar_type(*float_ptr_dtype(kernel, *ptr_idx[2]));
  if (ca.has_bias) plan.bias_name = kernel.params[*ptr_idx[3]].name;

  std::vector<std::string> exprs;
  for (size_t i = 3 + bias_slots; i < ops.size(); ++i) {
    auto e = scalar_long_expr(kernel, ops[i], false);
    if (!e) {
      err = "conv-transpose shape operands must be scalar kernel parameters or constants";
      return std::nullopt;
    }
    exprs.push_back(*e);
  }
  plan.n_expr = exprs[0];
  plan.cin_expr = exprs[1];
  plan.cout_expr = exprs[2];
  for (uint32_t i = 0; i < nd; ++i) {
    ConvDim& dim = plan.dims[3 - nd + i];
    dim.in = exprs[3 + i];
    dim.out = exprs[3 + nd + i];
    dim.k = ca.kernel_size[i];
    dim.s = vec_at(ca.stride, i, 1);
    dim.p = vec_at(ca.padding, i, 0);
    dim.d = vec_at(ca.dilation, i, 1);
    if (dim.k < 1 || dim.s < 1 || dim.d < 1 || dim.p < 0) {
      err = "conv-transpose requires kernel_size/stride/dilation >= 1 and padding >= 0";
      return std::nullopt;
    }
    const int32_t g = std::gcd(dim.s, dim.d);
    dim.step = dim.s / g;
    dim.delta = dim.d / g;
    dim.nmax = (dim.k + dim.step - 1) / dim.step;
  }

  plan.bm = static_cast<uint32_t>(positive_constexpr_param(kernel, kBlockMParam, ca.bm > 0 ? ca.bm : 64));
  plan.bn = static_cast<uint32_t>(positive_constexpr_param(kernel, kBlockNParam, ca.bn > 0 ? ca.bn : 64));
  plan.bk = static_cast<uint32_t>(positive_constexpr_param(kernel, kBlockKParam, ca.bk > 0 ? ca.bk : 16));
  plan.num_sg = static_cast<uint32_t>(
      positive_constexpr_param(kernel, kNumSgParam, ca.num_simdgroups > 0 ? ca.num_simdgroups : 4));
  if (!choose_tile_layout(plan)) {
    err = "conv-transpose tile must use multiples of 8 with BLOCK_N dividing the threadgroup size";
    return std::nullopt;
  }
  if (plan.threads > target.max_threads_per_threadgroup) {
    err = "conv-transpose threadgroup exceeds max_threads_per_threadgroup";
    return std::nullopt;
  }

  const auto cin = literal_value(plan.cin_expr);
  const auto cout = literal_value(plan.cout_expr);
  if (cin && cout) {
    if (*cin % plan.groups != 0 || *cout % plan.groups != 0) {
      err = "conv-transpose channels must be divisible by groups";
      return std::nullopt;
    }
    plan.ci_exact = (*cin / plan.groups) % plan.bk == 0;
    plan.co_exact = (*cout / plan.groups) % plan.bm == 0;
  }

  const bool req_blk8 = positive_constexpr_param(kernel, kWBlk8Param, 0) > 0;
  if (req_blk8) {
    if (plan.w_elem != "float" || !plan.co_exact) {
      err = "conv-transpose W_BLK8=1 requires f32 weights and output channels divisible by BLOCK_M";
      return std::nullopt;
    }
    plan.w_blk8 = true;
  }

  const std::optional<ir::EpilogueDAGAttr>* dag_src =
      kernel.hints.epilogue_dag.has_value() ? &kernel.hints.epilogue_dag : &plan.op->hints.epilogue_dag;
  if (dag_src->has_value() && !(*dag_src)->steps.empty()) {
    std::set<uint32_t> read_params;
    collect_epilogue_dag_param_indices(**dag_src, read_params);
    for (uint32_t pidx : read_params) {
      if (pidx >= kernel.params.size()) {
        err = "conv-transpose epilogue_dag references out-of-range kernel parameter";
        return std::nullopt;
      }
    }
    plan.epilogue_dag = **dag_src;
  }

  const std::optional<ir::FusedTailAttr>* tail_src =
      kernel.hints.fused_tail.has_value() ? &kernel.hints.fused_tail : &plan.op->hints.fused_tail;
  if (tail_src->has_value() && (*tail_src)->kind != ir::TailKind::None) {
    if (!is_supported_spatial_tail((*tail_src)->kind)) {
      err = "conv-transpose emitter supports only spatial global fused tails";
      return std::nullopt;
    }
    plan.fused_tail = **tail_src;
  }

  if (positive_constexpr_param(kernel, kHaloParam, 0) > 0) {
    if (!configure_halo(kernel, plan, err)) return std::nullopt;
    if (positive_constexpr_param(kernel, kFuseSwParam, 0) > 0) {
      if (!configure_fuse_sw(plan, target, err)) return std::nullopt;
    }
  } else {
    const uint32_t ab = plan.w_blk8 ? (plan.bk * plan.b_ld) : (plan.bk * plan.a_ld + plan.bk * plan.b_ld);
    plan.smem_floats = std::max(ab, plan.wm * plan.wn);
  }
  const uint32_t tail_smem_floats = (plan.fused_tail && !plan.fuse_sw) ? plan.bm : 0u;
  if (static_cast<uint64_t>(plan.smem_floats + tail_smem_floats) * 4u > target.max_threadgroup_memory_bytes) {
    err = "conv-transpose tile exceeds max_threadgroup_memory_bytes";
    return std::nullopt;
  }
  return plan;
}

class ConvTransposeWriter {
 public:
  ConvTransposeWriter(const Kernel& kernel, const ConvTransposePlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    emit_signature();
    emit_prologue();
    if (plan_.fuse_sw) {
      emit_fuse_sw_mainloop();
      emit_fuse_sw_epilogue();
    } else if (plan_.halo) {
      emit_halo_mainloop();
      emit_epilogue();
    } else {
      emit_general_mainloop();
      emit_epilogue();
    }
    os_ << "}\n";
    return os_.str();
  }

 private:
  const ConvDim& D() const { return plan_.dims[0]; }
  const ConvDim& H() const { return plan_.dims[1]; }
  const ConvDim& W() const { return plan_.dims[2]; }

  void emit_signature() {
    os_ << "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n\n";
    if (plan_.epilogue_dag) {
      HelperFlags helpers;
      collect_epilogue_dag_helpers(*plan_.epilogue_dag, helpers);
      emit_helper_functions(os_, helpers);
    }
    if (tail_is_max()) {
      os_ << "static inline float m1_nanmax(float a, float b) {\n"
          << "  return isnan(a) ? a : (isnan(b) ? b : ((a > b) ? a : b));\n"
          << "}\n"
          << "static inline float m1_simd_nanmax(float v) {\n"
          << "  v = m1_nanmax(v, simd_shuffle_xor(v, 16));\n"
          << "  v = m1_nanmax(v, simd_shuffle_xor(v, 8));\n"
          << "  v = m1_nanmax(v, simd_shuffle_xor(v, 4));\n"
          << "  v = m1_nanmax(v, simd_shuffle_xor(v, 2));\n"
          << "  v = m1_nanmax(v, simd_shuffle_xor(v, 1));\n"
          << "  return v;\n"
          << "}\n\n";
    }
    emit_kernel_param_decls(os_, kernel_, plan_.out_name);
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    if (plan_.fused_tail) {
      os_ << "    uint3 m1_ntg [[threadgroups_per_grid]],\n";
    }
    os_ << "    uint m1_lid [[thread_index_in_threadgroup]],\n";
    if (plan_.fused_tail) {
      os_ << "    uint m1_lane [[thread_index_in_simdgroup]],\n";
    }
    os_ << "    uint m1_sgid [[simdgroup_index_in_threadgroup]]) {\n";
  }

  void emit_dag(const std::string& val_var, const std::string& ch_expr, const std::string& elem_idx_expr,
                const std::string& ind) {
    if (!plan_.epilogue_dag) return;
    HelperFlags unused_helpers;
    emit_epilogue_dag_msl(
        os_, kernel_, *plan_.epilogue_dag, val_var, ch_expr, elem_idx_expr,
        [&](uint32_t idx) -> std::string {
          const auto& kp = kernel_.params[idx];
          if (kp.is_constexpr) {
            return format_scalar_literal_from_i64(kp.constexpr_val, scalar_dtype_of(kp.type));
          }
          return kp.name;
        },
        unused_helpers, ind);
  }

  bool tail_is_max() const {
    return plan_.fused_tail && plan_.fused_tail->kind == ir::TailKind::SpatialGlobalMax;
  }

  std::string tail_init() const { return tail_is_max() ? "(-INFINITY)" : "0.0f"; }

  std::string tail_combine(const std::string& a, const std::string& b) const {
    return tail_is_max() ? ("m1_nanmax(" + a + ", " + b + ")") : ("(" + a + " + " + b + ")");
  }

  std::string tail_simd_reduce(const std::string& v) const {
    return tail_is_max() ? ("m1_simd_nanmax(" + v + ")") : ("simd_sum(" + v + ")");
  }

  void emit_tail_finalize(const std::string& var, const std::string& ind) {
    if (!plan_.fused_tail) return;
    const auto& ft = *plan_.fused_tail;
    if (ft.kind == ir::TailKind::SpatialGlobalAvgPool) {
      if (ft.post_scale != 1.0f) {
        os_ << ind << var << " = " << var << " * (" << msl_exact_float_literal(ft.post_scale)
            << " / (float)m1_out_plane);\n";
      } else {
        os_ << ind << var << " = " << var << " / (float)m1_out_plane;\n";
      }
    } else if (ft.post_scale != 1.0f) {
      os_ << ind << var << " = " << var << " * " << msl_exact_float_literal(ft.post_scale) << ";\n";
    }
  }

  void emit_dim_phase(const char* tag, const ConvDim& dim) {
    const std::string t(tag);
    os_ << "  int m1_r0" << t << " = 0;\n";
    os_ << "  while (m1_r0" << t << " < m1_S" << t << " && (m1_a" << t << " + m1_P" << t << " - m1_r0" << t
        << " * m1_D" << t << ") % m1_S" << t << " != 0) ++m1_r0" << t << ";\n";
    os_ << "  const int m1_n" << t << " = (m1_r0" << t << " < m1_S" << t << " && m1_r0" << t << " < m1_K" << t
        << ") ? ((m1_K" << t << " - 1 - m1_r0" << t << ") / " << dim.step << " + 1) : 0;\n";
  }

  void emit_prologue() {
    const auto& p = plan_;
    os_ << "  constexpr int m1_BM = " << p.bm << ", m1_BN = " << p.bn << ", m1_BK = " << p.bk << ";\n";
    os_ << "  constexpr int m1_NT = " << p.threads << ", m1_ROWS = m1_NT / m1_BN;\n";
    os_ << "  constexpr long m1_G = " << p.groups << ";\n";
    os_ << "  constexpr int m1_KD = " << D().k << ", m1_KH = " << H().k << ", m1_KW = " << W().k << ";\n";
    os_ << "  constexpr int m1_SD = " << D().s << ", m1_SH = " << H().s << ", m1_SW = " << W().s << ";\n";
    os_ << "  constexpr int m1_PD = " << D().p << ", m1_PH = " << H().p << ", m1_PW = " << W().p << ";\n";
    os_ << "  constexpr int m1_DD = " << D().d << ", m1_DH = " << H().d << ", m1_DW = " << W().d << ";\n";
    os_ << "  constexpr int m1_KT = m1_KD * m1_KH * m1_KW;\n";
    if (p.fuse_sw) {
      os_ << "  constexpr int m1_NPH = m1_SD * m1_SH;\n";
    } else {
      os_ << "  constexpr int m1_NPH = m1_SD * m1_SH * m1_SW;\n";
    }
    os_ << "  const long m1_N = " << p.n_expr << ", m1_Cin = " << p.cin_expr << ", m1_Cout = " << p.cout_expr
        << ";\n";
    os_ << "  const long m1_Cig = m1_Cin / m1_G, m1_Cog = m1_Cout / m1_G;\n";
    os_ << "  const long m1_Din = " << D().in << ", m1_Hin = " << H().in << ", m1_Win = " << W().in << ";\n";
    os_ << "  const long m1_Dout = " << D().out << ", m1_Hout = " << H().out << ", m1_Wout = " << W().out
        << ";\n";
    os_ << "  const long m1_z = (long)m1_tgid.z;\n";
    os_ << "  const int m1_ph = (int)(m1_z % m1_NPH);\n";
    os_ << "  const long m1_ng = m1_z / m1_NPH;\n";
    os_ << "  const long m1_n = m1_ng / m1_G;\n";
    os_ << "  const long m1_g = m1_ng - m1_n * m1_G;\n";
    os_ << "  if (m1_n >= m1_N) return;\n";
    if (p.fuse_sw) {
      os_ << "  const int m1_aH = m1_ph % m1_SH;\n";
      os_ << "  const int m1_aD = m1_ph / m1_SH;\n";
    } else {
      os_ << "  const int m1_aW = m1_ph % m1_SW;\n";
      os_ << "  const int m1_aH = (m1_ph / m1_SW) % m1_SH;\n";
      os_ << "  const int m1_aD = m1_ph / (m1_SW * m1_SH);\n";
    }
    os_ << "  const long m1_QDm = (m1_Dout + m1_SD - 1) / m1_SD;\n";
    os_ << "  const long m1_QHm = (m1_Hout + m1_SH - 1) / m1_SH;\n";
    os_ << "  const long m1_QWm = (m1_Wout + m1_SW - 1) / m1_SW;\n";
    os_ << "  const long m1_QD = (m1_Dout - m1_aD + m1_SD - 1) / m1_SD;\n";
    os_ << "  const long m1_QH = (m1_Hout - m1_aH + m1_SH - 1) / m1_SH;\n";
    if (!p.fuse_sw) {
      os_ << "  const long m1_QW = (m1_Wout - m1_aW + m1_SW - 1) / m1_SW;\n";
    }
    os_ << "  const long m1_m0 = (long)m1_tgid.y * m1_BM;\n";
    if (p.halo) {
      os_ << "  constexpr int m1_TD = " << D().tile << ", m1_TH = " << H().tile << ", m1_TW = " << W().tile << ";\n";
      os_ << "  constexpr int m1_XD = " << D().ext << ", m1_XH = " << H().ext << ", m1_XW = " << W().ext << ";\n";
      os_ << "  constexpr int m1_XP = " << p.hplane << ", m1_XPL = " << p.hplane_ld << ";\n";
      os_ << "  const long m1_TQW = (m1_QWm + m1_TW - 1) / m1_TW;\n";
      os_ << "  const long m1_TQH = (m1_QHm + m1_TH - 1) / m1_TH;\n";
      os_ << "  const long m1_tq = (long)m1_tgid.x;\n";
      os_ << "  const long m1_qw0 = (m1_tq % m1_TQW) * m1_TW;\n";
      os_ << "  const long m1_tq2 = m1_tq / m1_TQW;\n";
      os_ << "  const long m1_qh0 = (m1_tq2 % m1_TQH) * m1_TH;\n";
      os_ << "  const long m1_qd0 = (m1_tq2 / m1_TQH) * m1_TD;\n";
      os_ << "  if (m1_m0 >= m1_Cog || m1_qd0 >= m1_QDm) return;\n";
    } else {
      os_ << "  const long m1_q0 = (long)m1_tgid.x * m1_BN;\n";
      os_ << "  if (m1_m0 >= m1_Cog || m1_q0 >= m1_QDm * m1_QHm * m1_QWm) return;\n";
      os_ << "  const int m1_col = (int)(m1_lid % (uint)m1_BN);\n";
      os_ << "  const int m1_row = (int)(m1_lid / (uint)m1_BN);\n";
      os_ << "  const long m1_q = m1_q0 + m1_col;\n";
      os_ << "  const long m1_qw = m1_q % m1_QWm;\n";
      os_ << "  const long m1_qt = m1_q / m1_QWm;\n";
      os_ << "  const long m1_qh = m1_qt % m1_QHm;\n";
      os_ << "  const long m1_qd = m1_qt / m1_QHm;\n";
      os_ << "  const bool m1_col_ok = m1_qd < m1_QD && m1_qh < m1_QH && m1_qw < m1_QW;\n";
    }
    os_ << "  const long m1_in_plane = m1_Din * m1_Hin * m1_Win;\n";
    os_ << "  const long m1_out_plane = m1_Dout * m1_Hout * m1_Wout;\n";
    os_ << "  device const " << p.x_elem << "* m1_xg = " << p.x_name
        << " + (m1_n * m1_Cin + m1_g * m1_Cig) * m1_in_plane;\n";
    os_ << "  device const " << p.w_elem << "* m1_wg = " << p.w_name << " + m1_g * m1_Cig * m1_Cog * m1_KT;\n";
    emit_dim_phase("D", D());
    emit_dim_phase("H", H());
    if (p.fuse_sw) {
      os_ << "  if (m1_nD == 0 || m1_nH == 0) {\n";
      if (p.fused_tail) {
        os_ << "    const long m1_pstride = (long)m1_NPH * (long)m1_ntg.x;\n";
        os_ << "    const long m1_pcol = (long)m1_ph * (long)m1_ntg.x + (long)m1_tgid.x;\n";
        os_ << "    constexpr uint m1_TILE_COLS = (uint)(m1_TD * m1_TH * m1_TW * m1_SW);\n";
        os_ << "    for (uint m1_lr = m1_sgid; m1_lr < (uint)m1_BM; m1_lr += " << p.num_sg << "u) {\n";
        os_ << "      const long m1_co = m1_m0 + (long)m1_lr;\n";
        if (!p.co_exact) os_ << "      if (m1_co >= m1_Cog) continue;\n";
        os_ << "      const long m1_ch = m1_g * m1_Cog + m1_co;\n";
        if (p.attr.has_bias) {
          os_ << "      const float m1_zv = (float)" << p.bias_name << "[m1_ch];\n";
        } else {
          os_ << "      const float m1_zv = 0.0f;\n";
        }
        os_ << "      float m1_ra = " << tail_init() << ";\n";
        os_ << "      for (uint m1_c = m1_lane; m1_c < m1_TILE_COLS; m1_c += 32u) {\n";
        os_ << "        const uint m1_ow_loc = m1_c % (uint)(m1_TW * m1_SW);\n";
        os_ << "        const uint m1_r1 = m1_c / (uint)(m1_TW * m1_SW);\n";
        os_ << "        const uint m1_th = m1_r1 % (uint)m1_TH;\n";
        os_ << "        const uint m1_td = m1_r1 / (uint)m1_TH;\n";
        os_ << "        const long m1_od = m1_aD + (long)m1_SD * (m1_qd0 + (long)m1_td);\n";
        os_ << "        const long m1_oh = m1_aH + (long)m1_SH * (m1_qh0 + (long)m1_th);\n";
        os_ << "        const long m1_ow = m1_qw0 * (long)m1_SW + (long)m1_ow_loc;\n";
        os_ << "        if (m1_od < m1_Dout && m1_oh < m1_Hout && m1_ow < m1_Wout) {\n";
        os_ << "          float m1_v = m1_zv;\n";
        if (p.epilogue_dag) {
          os_ << "          const long m1_eidx = (m1_n * m1_Cout + m1_ch) * m1_out_plane"
              << " + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow;\n";
          emit_dag("m1_v", "m1_ch", "m1_eidx", "          ");
        }
        os_ << "          m1_ra = " << tail_combine("m1_ra", "m1_v") << ";\n";
        os_ << "        }\n";
        os_ << "      }\n";
        os_ << "      m1_ra = " << tail_simd_reduce("m1_ra") << ";\n";
        os_ << "      if (m1_lane == 0u) {\n";
        emit_tail_finalize("m1_ra", "        ");
        os_ << "        " << p.out_name << "[(m1_n * m1_Cout + m1_ch) * m1_pstride + m1_pcol] = (" << p.out_elem
            << ")m1_ra;\n";
        os_ << "      }\n";
        os_ << "    }\n";
      } else {
        os_ << "    constexpr uint m1_ROW_V4 = (uint)(m1_TW * m1_SW) / 4u;\n";
        os_ << "    constexpr uint m1_TILE_V4 = (uint)(m1_BM * m1_TD * m1_TH) * m1_ROW_V4;\n";
        os_ << "    for (uint m1_vi = m1_lid; m1_vi < m1_TILE_V4; m1_vi += (uint)m1_NT) {\n";
        os_ << "      const uint m1_c4 = m1_vi % m1_ROW_V4;\n";
        os_ << "      const uint m1_r1 = m1_vi / m1_ROW_V4;\n";
        os_ << "      const uint m1_th = m1_r1 % (uint)m1_TH;\n";
        os_ << "      const uint m1_r2 = m1_r1 / (uint)m1_TH;\n";
        os_ << "      const uint m1_td = m1_r2 % (uint)m1_TD;\n";
        os_ << "      const uint m1_lr = m1_r2 / (uint)m1_TD;\n";
        os_ << "      const long m1_co = m1_m0 + (long)m1_lr;\n";
        os_ << "      const long m1_od = m1_aD + (long)m1_SD * (m1_qd0 + (long)m1_td);\n";
        os_ << "      const long m1_oh = m1_aH + (long)m1_SH * (m1_qh0 + (long)m1_th);\n";
        os_ << "      const long m1_ow = m1_qw0 * (long)m1_SW + (long)(m1_c4 * 4u);\n";
        os_ << "      if (m1_od < m1_Dout && m1_oh < m1_Hout && m1_ow < m1_Wout) {\n";
        os_ << "        const long m1_ch = m1_g * m1_Cog + m1_co;\n";
        if (p.attr.has_bias) {
          os_ << "        const float m1_zv = (float)" << p.bias_name << "[m1_ch];\n";
        } else {
          os_ << "        const float m1_zv = 0.0f;\n";
        }
        os_ << "        const long m1_out_base = (m1_n * m1_Cout + m1_ch) * m1_out_plane"
            << " + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow;\n";
        os_ << "        device float* m1_op = " << p.out_name << " + m1_out_base;\n";
        if (p.epilogue_dag) {
          os_ << "        float4 m1_ov = float4(m1_zv);\n";
          os_ << "        #pragma unroll\n";
          os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
          os_ << "          if (m1_ow + m1_e < m1_Wout) {\n";
          os_ << "            float m1_ev = m1_ov[m1_e];\n";
          os_ << "            const long m1_eidx = m1_out_base + (long)m1_e;\n";
          emit_dag("m1_ev", "m1_ch", "m1_eidx", "            ");
          os_ << "            m1_ov[m1_e] = m1_ev;\n";
          os_ << "          }\n";
          os_ << "        }\n";
          os_ << "        if (m1_ow + 3 < m1_Wout) {\n";
          os_ << "          *(device packed_float4*)(m1_op) = packed_float4(m1_ov);\n";
          os_ << "        } else {\n";
          os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
          os_ << "            if (m1_ow + m1_e < m1_Wout) m1_op[m1_e] = m1_ov[m1_e];\n";
          os_ << "          }\n";
          os_ << "        }\n";
        } else {
          os_ << "        if (m1_ow + 3 < m1_Wout) {\n";
          os_ << "          *(device packed_float4*)(m1_op) = packed_float4(float4(m1_zv));\n";
          os_ << "        } else {\n";
          os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
          os_ << "            if (m1_ow + m1_e < m1_Wout) m1_op[m1_e] = m1_zv;\n";
          os_ << "          }\n";
          os_ << "        }\n";
        }
        os_ << "      }\n";
        os_ << "    }\n";
      }
      os_ << "    return;\n";
      os_ << "  }\n";
      os_ << "  threadgroup float m1_smem[" << p.smem_floats << "];\n";
      os_ << "  const uint m1_sg_row = (m1_sgid / " << p.sg_n << "u) * " << p.wm << "u;\n";
      os_ << "  const uint m1_sg_col = (m1_sgid % " << p.sg_n << "u) * " << p.wn << "u;\n";
      os_ << "  simdgroup_float8x8 m1_acc[" << p.active_w.size() << "][" << p.fm << "][" << p.fn << "];\n";
      os_ << "  for (int m1_a = 0; m1_a < " << p.active_w.size() << "; ++m1_a) {\n";
      os_ << "    for (int m1_i = 0; m1_i < " << p.fm << "; ++m1_i) {\n";
      os_ << "      for (int m1_j = 0; m1_j < " << p.fn << "; ++m1_j) {\n";
      os_ << "        m1_acc[m1_a][m1_i][m1_j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
      os_ << "      }\n    }\n  }\n";
      os_ << "  (void)m1_ROWS;\n";
      return;
    }
    emit_dim_phase("W", W());
    os_ << "  threadgroup float m1_smem[" << p.smem_floats << "];\n";
    os_ << "  const uint m1_sg_row = (m1_sgid / " << p.sg_n << "u) * " << p.wm << "u;\n";
    os_ << "  const uint m1_sg_col = (m1_sgid % " << p.sg_n << "u) * " << p.wn << "u;\n";
    os_ << "  simdgroup_float8x8 m1_acc[" << p.fm << "][" << p.fn << "];\n";
    os_ << "  for (int m1_i = 0; m1_i < " << p.fm << "; ++m1_i) {\n";
    os_ << "    for (int m1_j = 0; m1_j < " << p.fn << "; ++m1_j) {\n";
    os_ << "      m1_acc[m1_i][m1_j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n";
    os_ << "    }\n  }\n";
    os_ << "  (void)m1_ROWS;\n";
  }

  void emit_interleaved_mma_blk8(const std::string& wb_expr, const std::string& b_base_expr,
                                 const std::string& b_ld_expr, bool indexed_fb, const std::string& ind) {
    const auto& p = plan_;
    os_ << ind << "simdgroup_float8x8 m1_fa[" << p.fm << "];\n";
    os_ << ind << "simdgroup_float8x8 m1_fb[" << p.fn << "];\n";
    os_ << ind << "simdgroup_load(m1_fa[0], " << wb_expr << ", 8u);\n";
    for (uint32_t j = 0; j < p.fn; ++j) {
      if (indexed_fb) {
        os_ << ind << "simdgroup_load(m1_fb[" << j << "], " << b_base_expr << " + m1_fbo" << j << ", " << b_ld_expr
            << ");\n";
      } else {
        os_ << ind << "simdgroup_load(m1_fb[" << j << "], " << b_base_expr << " + " << (j * 8u) << "u, " << b_ld_expr
            << ");\n";
      }
    }
    for (uint32_t i = 0; i < p.fm; ++i) {
      if (i + 1u < p.fm) {
        os_ << ind << "simdgroup_load(m1_fa[" << (i + 1u) << "], " << wb_expr << " + " << ((i + 1u) * 64u)
            << "u, 8u);\n";
      }
      for (uint32_t j = 0; j < p.fn; ++j) {
        os_ << ind << "simdgroup_multiply_accumulate(m1_acc[" << i << "][" << j << "], m1_fa[" << i << "], m1_fb[" << j
            << "], m1_acc[" << i << "][" << j << "]);\n";
      }
    }
  }

  void emit_fuse_sw_mainloop() {
    const auto& p = plan_;
    const bool unroll_dh = static_cast<int64_t>(D().nmax) * H().nmax <= 6;
    const char* tags[2] = {"D", "H"};
    const char* q0s[2] = {"m1_qd0", "m1_qh0"};
    for (uint32_t i = 0; i < 2; ++i) {
      const std::string t(tags[i]);
      os_ << "  const long m1_i0" << t << " = " << q0s[i] << " + (m1_a" << t << " + m1_P" << t << " - m1_r0" << t
          << " * m1_D" << t << ") / m1_S" << t << " - (long)(m1_n" << t << " - 1) * " << p.dims[i].delta << ";\n";
    }
    os_ << "  const long m1_i0W = m1_qw0 + (" << p.fuse_shift_min_w << "L);\n";
    for (uint32_t j = 0; j < p.fn; ++j) {
      os_ << "  const uint m1_c" << j << " = m1_sg_col + " << (j * 8u) << "u;\n";
      os_ << "  const uint m1_fbo" << j << " = ((m1_c" << j
          << " / (uint)(m1_TW * m1_TH)) * (uint)m1_XH + (m1_c" << j
          << " / (uint)m1_TW) % (uint)m1_TH) * (uint)m1_XW + m1_c" << j << " % (uint)m1_TW;\n";
    }
    os_ << "  const ulong m1_cog_blk = (ulong)(m1_Cog >> 3);\n";
    os_ << "  const ulong m1_row_blk = (ulong)((m1_m0 + (long)m1_sg_row) >> 3);\n";
    os_ << "  const bool m1_x_in = (m1_i0D >= 0 && m1_i0D + m1_XD <= m1_Din && m1_i0H >= 0 && m1_i0H + m1_XH <= m1_Hin"
        << " && m1_i0W >= 0 && m1_i0W + m1_XW <= m1_Win);\n";
    os_ << "  threadgroup float* m1_Bs = m1_smem;\n";

    std::vector<std::pair<int32_t, std::vector<std::pair<uint32_t, int32_t>>>> bw_groups;
    for (uint32_t ai = 0; ai < static_cast<uint32_t>(p.active_w.size()); ++ai) {
      for (const auto& [rw, sh] : p.active_w[ai].taps) {
        const int32_t bw = sh - p.fuse_shift_min_w;
        auto it = std::find_if(bw_groups.begin(), bw_groups.end(), [bw](const auto& g) { return g.first == bw; });
        if (it == bw_groups.end()) {
          bw_groups.push_back({bw, {{ai, rw}}});
        } else {
          it->second.emplace_back(ai, rw);
        }
      }
    }

    os_ << "  for (long m1_c0 = 0; m1_c0 < m1_Cig; m1_c0 += m1_BK) {\n";
    os_ << "    device const " << p.x_elem << "* m1_xc = m1_xg + m1_c0 * m1_in_plane;\n";
    os_ << "    if (m1_x_in" << (p.ci_exact ? "" : " && m1_c0 + m1_BK <= m1_Cig") << ") {\n";
    os_ << "      device const " << p.x_elem
        << "* m1_xb = m1_xc + (m1_i0D * m1_Hin + m1_i0H) * m1_Win + m1_i0W;\n";
    os_ << "      for (int m1_idx = (int)m1_lid; m1_idx < m1_BK * m1_XP; m1_idx += m1_NT) {\n";
    os_ << "        const int m1_ci = m1_idx / m1_XP;\n";
    os_ << "        const int m1_rem = m1_idx - m1_ci * m1_XP;\n";
    os_ << "        const int m1_rd_l = m1_rem / (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rem2 = m1_rem - m1_rd_l * (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rh_l = m1_rem2 / m1_XW;\n";
    os_ << "        const int m1_rw_l = m1_rem2 - m1_rh_l * m1_XW;\n";
    os_ << "        m1_Bs[m1_ci * m1_XPL + m1_rem] = (float)m1_xb[(long)m1_ci * m1_in_plane + ((long)m1_rd_l * m1_Hin + "
        << "(long)m1_rh_l) * m1_Win + (long)m1_rw_l];\n";
    os_ << "      }\n";
    os_ << "    } else {\n";
    os_ << "      for (int m1_idx = (int)m1_lid; m1_idx < m1_BK * m1_XP; m1_idx += m1_NT) {\n";
    os_ << "        const int m1_ci = m1_idx / m1_XP;\n";
    os_ << "        const int m1_rem = m1_idx - m1_ci * m1_XP;\n";
    os_ << "        const int m1_rd_l = m1_rem / (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rem2 = m1_rem - m1_rd_l * (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rh_l = m1_rem2 / m1_XW;\n";
    os_ << "        const int m1_rw_l = m1_rem2 - m1_rh_l * m1_XW;\n";
    os_ << "        const long m1_hd = m1_i0D + m1_rd_l;\n";
    os_ << "        const long m1_hh = m1_i0H + m1_rh_l;\n";
    os_ << "        const long m1_hw = m1_i0W + m1_rw_l;\n";
    os_ << "        float m1_v = 0.0f;\n";
    os_ << "        if (" << (p.ci_exact ? "" : "m1_c0 + m1_ci < m1_Cig && ")
        << "m1_hd >= 0 && m1_hd < m1_Din && m1_hh >= 0 && m1_hh < m1_Hin && m1_hw >= 0 && m1_hw < m1_Win)"
        << " m1_v = (float)m1_xc[(long)m1_ci * m1_in_plane + (m1_hd * m1_Hin + m1_hh) * m1_Win + m1_hw];\n";
    os_ << "        m1_Bs[m1_ci * m1_XPL + m1_rem] = m1_v;\n";
    os_ << "      }\n";
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    #pragma unroll\n";
    os_ << "    for (uint m1_ks = 0u; m1_ks < (uint)m1_BK; m1_ks += 8u) {\n";
    os_ << "      const ulong m1_kb = (ulong)((m1_c0 + (long)m1_ks) >> 3);\n";
    if (unroll_dh) os_ << "      #pragma unroll\n";
    os_ << "      for (int m1_jD = 0; m1_jD < m1_nD; ++m1_jD) {\n";
    os_ << "        const int m1_rD = m1_r0D + m1_jD * " << D().step << ";\n";
    os_ << "        const int m1_bD = (m1_nD - 1 - m1_jD) * " << D().delta << ";\n";
    if (unroll_dh) os_ << "        #pragma unroll\n";
    os_ << "        for (int m1_jH = 0; m1_jH < m1_nH; ++m1_jH) {\n";
    os_ << "          const int m1_rH = m1_r0H + m1_jH * " << H().step << ";\n";
    os_ << "          const int m1_bH = (m1_nH - 1 - m1_jH) * " << H().delta << ";\n";
    os_ << "          const uint m1_bh_off = m1_ks * (uint)m1_XPL + (uint)((m1_bD * m1_XH + m1_bH) * m1_XW);\n";
    os_ << "          const int m1_tap_dh = (m1_rD * m1_KH + m1_rH) * m1_KW;\n";
    for (const auto& [bw, pairs] : bw_groups) {
      os_ << "          {\n";
      os_ << "            threadgroup const float* m1_bt = m1_Bs + m1_bh_off + " << bw << "u;\n";
      os_ << "            simdgroup_float8x8 m1_fb[" << p.fn << "];\n";
      for (uint32_t j = 0; j < p.fn; ++j) {
        os_ << "            simdgroup_load(m1_fb[" << j << "], m1_bt + m1_fbo" << j << ", (ulong)m1_XPL);\n";
      }
      for (const auto& [ai, rw] : pairs) {
        os_ << "            {\n";
        os_ << "              const ulong m1_tap = (ulong)(m1_tap_dh + " << rw << ");\n";
        os_ << "              device const float* m1_wb = m1_wg + ((m1_kb * (ulong)m1_KT + m1_tap) * m1_cog_blk + "
            << "m1_row_blk) * 64ul;\n";
        os_ << "              simdgroup_float8x8 m1_fa[" << p.fm << "];\n";
        os_ << "              simdgroup_load(m1_fa[0], m1_wb, 8u);\n";
        for (uint32_t i = 0; i < p.fm; ++i) {
          if (i + 1u < p.fm) {
            os_ << "              simdgroup_load(m1_fa[" << (i + 1u) << "], m1_wb + " << ((i + 1u) * 64u)
                << "u, 8u);\n";
          }
          for (uint32_t j = 0; j < p.fn; ++j) {
            os_ << "              simdgroup_multiply_accumulate(m1_acc[" << ai << "][" << i << "][" << j
                << "], m1_fa[" << i << "], m1_fb[" << j << "], m1_acc[" << ai << "][" << i << "][" << j << "]);\n";
          }
        }
        os_ << "            }\n";
      }
      os_ << "          }\n";
    }
    os_ << "        }\n      }\n    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "  }\n";
  }

  void emit_fuse_sw_epilogue() {
    const auto& p = plan_;
    const uint32_t n_act = static_cast<uint32_t>(p.active_w.size());
    const uint32_t epi_rows = p.fuse_epi_rows;
    const uint32_t epi_f = epi_rows / 8u;
    os_ << "  threadgroup float* m1_Cs = m1_smem;\n";
    os_ << "  constexpr uint m1_EPI_ROWS = " << epi_rows << "u;\n";
    if (p.fused_tail) {
      os_ << "  const long m1_pstride = (long)m1_NPH * (long)m1_ntg.x;\n";
      os_ << "  const long m1_pcol = (long)m1_ph * (long)m1_ntg.x + (long)m1_tgid.x;\n";
      os_ << "  constexpr uint m1_TILE_COLS = (uint)(m1_TD * m1_TH * m1_TW * m1_SW);\n";
    } else {
      os_ << "  constexpr uint m1_ROW_V4 = (uint)(m1_TW * m1_SW) / 4u;\n";
      os_ << "  constexpr uint m1_EPI_V4 = m1_EPI_ROWS * (uint)(m1_TD * m1_TH) * m1_ROW_V4;\n";
    }
    os_ << "  const uint m1_sg_m_idx = m1_sgid / " << p.sg_n << "u;\n";
    const uint32_t total_strips = p.sg_m * (p.fm / epi_f);
    uint32_t strip_idx = 0;
    for (uint32_t sr = 0; sr < p.sg_m; ++sr) {
      for (uint32_t i0 = 0; i0 < p.fm; i0 += epi_f) {
        ++strip_idx;
        os_ << "  {\n";
        os_ << "    if (m1_sg_m_idx == " << sr << "u) {\n";
        for (uint32_t ai = 0; ai < n_act; ++ai) {
          for (uint32_t fi = 0; fi < epi_f; ++fi) {
            for (uint32_t j = 0; j < p.fn; ++j) {
              os_ << "      simdgroup_store(m1_acc[" << ai << "][" << (i0 + fi) << "][" << j
                  << "], m1_Cs + (" << (ai * epi_rows + fi * 8u) << "u) * (uint)m1_BN + m1_sg_col + " << (j * 8u)
                  << "u, (uint)m1_BN);\n";
            }
          }
        }
        os_ << "    }\n";
        os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        os_ << "    const long m1_co_base = m1_m0 + " << (sr * p.wm + i0 * 8u) << "L;\n";
        if (p.fused_tail) {
          os_ << "    for (uint m1_lr = m1_sgid; m1_lr < m1_EPI_ROWS; m1_lr += " << p.num_sg << "u) {\n";
          os_ << "      const long m1_co = m1_co_base + (long)m1_lr;\n";
          if (!p.co_exact) os_ << "      if (m1_co >= m1_Cog) continue;\n";
          os_ << "      const long m1_ch = m1_g * m1_Cog + m1_co;\n";
          os_ << "      float m1_ra = " << tail_init() << ";\n";
          os_ << "      for (uint m1_c = m1_lane; m1_c < m1_TILE_COLS; m1_c += 32u) {\n";
          os_ << "        const uint m1_ow_loc = m1_c % (uint)(m1_TW * m1_SW);\n";
          os_ << "        const uint m1_r1 = m1_c / (uint)(m1_TW * m1_SW);\n";
          os_ << "        const uint m1_th = m1_r1 % (uint)m1_TH;\n";
          os_ << "        const uint m1_td = m1_r1 / (uint)m1_TH;\n";
          os_ << "        const long m1_od = m1_aD + (long)m1_SD * (m1_qd0 + (long)m1_td);\n";
          os_ << "        const long m1_oh = m1_aH + (long)m1_SH * (m1_qh0 + (long)m1_th);\n";
          os_ << "        const long m1_ow = m1_qw0 * (long)m1_SW + (long)m1_ow_loc;\n";
          os_ << "        if (m1_od < m1_Dout && m1_oh < m1_Hout && m1_ow < m1_Wout) {\n";
          os_ << "          const uint m1_base_col = (m1_td * (uint)m1_TH + m1_th) * (uint)m1_TW;\n";
          os_ << "          const uint m1_tw = m1_ow_loc / (uint)m1_SW;\n";
          os_ << "          const uint m1_aw = m1_ow_loc - m1_tw * (uint)m1_SW;\n";
          os_ << "          const int m1_ai = ";
          for (uint32_t ai = 0; ai < n_act; ++ai) {
            os_ << "(m1_aw == " << p.active_w[ai].aw << "u) ? " << ai << " : ";
          }
          os_ << "-1;\n";
          os_ << "          float m1_v = (m1_ai >= 0) ? m1_Cs[((uint)m1_ai * m1_EPI_ROWS + m1_lr) * (uint)m1_BN + "
              << "m1_base_col + m1_tw] : 0.0f;\n";
          if (p.attr.has_bias) {
            os_ << "          m1_v += (float)" << p.bias_name << "[m1_ch];\n";
          }
          if (p.epilogue_dag) {
            os_ << "          const long m1_eidx = (m1_n * m1_Cout + m1_ch) * m1_out_plane"
                << " + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow;\n";
            emit_dag("m1_v", "m1_ch", "m1_eidx", "          ");
          }
          os_ << "          m1_ra = " << tail_combine("m1_ra", "m1_v") << ";\n";
          os_ << "        }\n";
          os_ << "      }\n";
          os_ << "      m1_ra = " << tail_simd_reduce("m1_ra") << ";\n";
          os_ << "      if (m1_lane == 0u) {\n";
          emit_tail_finalize("m1_ra", "        ");
          os_ << "        " << p.out_name << "[(m1_n * m1_Cout + m1_ch) * m1_pstride + m1_pcol] = (" << p.out_elem
              << ")m1_ra;\n";
          os_ << "      }\n";
          os_ << "    }\n";
        } else {
          os_ << "    for (uint m1_vi = m1_lid; m1_vi < m1_EPI_V4; m1_vi += (uint)m1_NT) {\n";
          os_ << "      const uint m1_c4 = m1_vi % m1_ROW_V4;\n";
          os_ << "      const uint m1_r1 = m1_vi / m1_ROW_V4;\n";
          os_ << "      const uint m1_th = m1_r1 % (uint)m1_TH;\n";
          os_ << "      const uint m1_r2 = m1_r1 / (uint)m1_TH;\n";
          os_ << "      const uint m1_td = m1_r2 % (uint)m1_TD;\n";
          os_ << "      const uint m1_lr = m1_r2 / (uint)m1_TD;\n";
          os_ << "      const long m1_co = m1_co_base + (long)m1_lr;\n";
          os_ << "      const long m1_od = m1_aD + (long)m1_SD * (m1_qd0 + (long)m1_td);\n";
          os_ << "      const long m1_oh = m1_aH + (long)m1_SH * (m1_qh0 + (long)m1_th);\n";
          os_ << "      const uint m1_ow_loc = m1_c4 * 4u;\n";
          os_ << "      const long m1_ow = m1_qw0 * (long)m1_SW + (long)m1_ow_loc;\n";
          os_ << "      if (m1_od < m1_Dout && m1_oh < m1_Hout && m1_ow < m1_Wout) {\n";
          os_ << "        const long m1_ch = m1_g * m1_Cog + m1_co;\n";
          os_ << "        const uint m1_base_col = (m1_td * (uint)m1_TH + m1_th) * (uint)m1_TW;\n";
          os_ << "        uint m1_tw = m1_ow_loc / (uint)m1_SW;\n";
          os_ << "        uint m1_aw = m1_ow_loc - m1_tw * (uint)m1_SW;\n";
          os_ << "        float4 m1_ov;\n";
          os_ << "        #pragma unroll\n";
          os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
          os_ << "          const int m1_ai = ";
          for (uint32_t ai = 0; ai < n_act; ++ai) {
            os_ << "(m1_aw == " << p.active_w[ai].aw << "u) ? " << ai << " : ";
          }
          os_ << "-1;\n";
          os_ << "          m1_ov[m1_e] = (m1_ai >= 0) ? m1_Cs[((uint)m1_ai * m1_EPI_ROWS + m1_lr) * (uint)m1_BN + "
              << "m1_base_col + m1_tw] : 0.0f;\n";
          os_ << "          if (++m1_aw == (uint)m1_SW) { m1_aw = 0u; ++m1_tw; }\n";
          os_ << "        }\n";
          if (p.attr.has_bias) {
            os_ << "        m1_ov += float4((float)" << p.bias_name << "[m1_ch]);\n";
          }
          os_ << "        const long m1_out_base = (m1_n * m1_Cout + m1_ch) * m1_out_plane"
              << " + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow;\n";
          if (p.epilogue_dag) {
            os_ << "        #pragma unroll\n";
            os_ << "        for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
            os_ << "          if (m1_ow + m1_e < m1_Wout) {\n";
            os_ << "            float m1_ev = m1_ov[m1_e];\n";
            os_ << "            const long m1_eidx = m1_out_base + (long)m1_e;\n";
            emit_dag("m1_ev", "m1_ch", "m1_eidx", "            ");
            os_ << "            m1_ov[m1_e] = m1_ev;\n";
            os_ << "          }\n";
            os_ << "        }\n";
          }
          os_ << "        device float* m1_op = " << p.out_name << " + m1_out_base;\n";
          os_ << "        if (m1_ow + 3 < m1_Wout) {\n";
          os_ << "          *(device packed_float4*)(m1_op) = packed_float4(m1_ov);\n";
          os_ << "        } else {\n";
          os_ << "          for (int m1_e = 0; m1_e < 4; ++m1_e) {\n";
          os_ << "            if (m1_ow + m1_e < m1_Wout) m1_op[m1_e] = m1_ov[m1_e];\n";
          os_ << "          }\n";
          os_ << "        }\n";
          os_ << "      }\n";
          os_ << "    }\n";
        }
        if (strip_idx < total_strips) {
          os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        }
        os_ << "  }\n";
      }
    }
  }

  void emit_halo_mainloop() {
    const auto& p = plan_;
    const bool unroll_taps = static_cast<int64_t>(D().nmax) * H().nmax * W().nmax <= 15;
    os_ << "  if (m1_nD > 0 && m1_nH > 0 && m1_nW > 0) {\n";
    const char* tags[3] = {"D", "H", "W"};
    const char* q0s[3] = {"m1_qd0", "m1_qh0", "m1_qw0"};
    for (uint32_t i = 0; i < 3; ++i) {
      const std::string t(tags[i]);
      os_ << "    const long m1_i0" << t << " = " << q0s[i] << " + (m1_a" << t << " + m1_P" << t << " - m1_r0" << t
          << " * m1_D" << t << ") / m1_S" << t << " - (long)(m1_n" << t << " - 1) * " << p.dims[i].delta << ";\n";
    }
    for (uint32_t j = 0; j < p.fn; ++j) {
      os_ << "    const uint m1_c" << j << " = m1_sg_col + " << (j * 8u) << "u;\n";
      os_ << "    const uint m1_fbo" << j << " = ((m1_c" << j
          << " / (uint)(m1_TW * m1_TH)) * (uint)m1_XH + (m1_c" << j
          << " / (uint)m1_TW) % (uint)m1_TH) * (uint)m1_XW + m1_c" << j << " % (uint)m1_TW;\n";
    }
    os_ << "    const ulong m1_cog_blk = (ulong)(m1_Cog >> 3);\n";
    os_ << "    const ulong m1_row_blk = (ulong)((m1_m0 + (long)m1_sg_row) >> 3);\n";
    os_ << "    threadgroup float* m1_Bs = m1_smem;\n";
    os_ << "    for (long m1_c0 = 0; m1_c0 < m1_Cig; m1_c0 += m1_BK) {\n";
    os_ << "      device const " << p.x_elem << "* m1_xc = m1_xg + m1_c0 * m1_in_plane;\n";
    os_ << "      for (int m1_idx = (int)m1_lid; m1_idx < m1_BK * m1_XP; m1_idx += m1_NT) {\n";
    os_ << "        const int m1_ci = m1_idx / m1_XP;\n";
    os_ << "        const int m1_rem = m1_idx - m1_ci * m1_XP;\n";
    os_ << "        const int m1_rd_l = m1_rem / (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rem2 = m1_rem - m1_rd_l * (m1_XH * m1_XW);\n";
    os_ << "        const int m1_rh_l = m1_rem2 / m1_XW;\n";
    os_ << "        const int m1_rw_l = m1_rem2 - m1_rh_l * m1_XW;\n";
    os_ << "        const long m1_hd = m1_i0D + m1_rd_l;\n";
    os_ << "        const long m1_hh = m1_i0H + m1_rh_l;\n";
    os_ << "        const long m1_hw = m1_i0W + m1_rw_l;\n";
    os_ << "        float m1_v = 0.0f;\n";
    os_ << "        if (" << (p.ci_exact ? "" : "m1_c0 + m1_ci < m1_Cig && ")
        << "m1_hd >= 0 && m1_hd < m1_Din && m1_hh >= 0 && m1_hh < m1_Hin && m1_hw >= 0 && m1_hw < m1_Win)"
        << " m1_v = (float)m1_xc[(long)m1_ci * m1_in_plane + (m1_hd * m1_Hin + m1_hh) * m1_Win + m1_hw];\n";
    os_ << "        m1_Bs[m1_ci * m1_XPL + m1_rem] = m1_v;\n";
    os_ << "      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "      #pragma unroll\n";
    os_ << "      for (uint m1_ks = 0u; m1_ks < (uint)m1_BK; m1_ks += 8u) {\n";
    os_ << "        const ulong m1_kb = (ulong)((m1_c0 + (long)m1_ks) >> 3);\n";
    if (unroll_taps) os_ << "        #pragma unroll\n";
    os_ << "        for (int m1_jD = 0; m1_jD < m1_nD; ++m1_jD) {\n";
    os_ << "          const int m1_rD = m1_r0D + m1_jD * " << D().step << ";\n";
    os_ << "          const int m1_bD = (m1_nD - 1 - m1_jD) * " << D().delta << ";\n";
    if (unroll_taps) os_ << "          #pragma unroll\n";
    os_ << "          for (int m1_jH = 0; m1_jH < m1_nH; ++m1_jH) {\n";
    os_ << "            const int m1_rH = m1_r0H + m1_jH * " << H().step << ";\n";
    os_ << "            const int m1_bH = (m1_nH - 1 - m1_jH) * " << H().delta << ";\n";
    if (unroll_taps) os_ << "            #pragma unroll\n";
    os_ << "            for (int m1_jW = 0; m1_jW < m1_nW; ++m1_jW) {\n";
    os_ << "              const int m1_rW = m1_r0W + m1_jW * " << W().step << ";\n";
    os_ << "              const int m1_bW = (m1_nW - 1 - m1_jW) * " << W().delta << ";\n";
    os_ << "              const ulong m1_tap = (ulong)((m1_rD * m1_KH + m1_rH) * m1_KW + m1_rW);\n";
    os_ << "              device const float* m1_wb = m1_wg + ((m1_kb * (ulong)m1_KT + m1_tap) * m1_cog_blk + "
        << "m1_row_blk) * 64ul;\n";
    os_ << "              threadgroup const float* m1_bt = m1_Bs + m1_ks * (uint)m1_XPL + (uint)((m1_bD * m1_XH + "
        << "m1_bH) * m1_XW + m1_bW);\n";
    emit_interleaved_mma_blk8("m1_wb", "m1_bt", "(ulong)m1_XPL", true, "              ");
    os_ << "            }\n          }\n        }\n      }\n";
    os_ << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    }\n";
    os_ << "  }\n";
  }

  void emit_general_mainloop() {
    const auto& p = plan_;
    os_ << "  constexpr uint m1_ALD = " << p.a_ld << "u, m1_BLD = " << p.b_ld << "u;\n";
    if (p.w_blk8) {
      os_ << "  threadgroup float* m1_Bs = m1_smem;\n";
      os_ << "  const ulong m1_cog_blk = (ulong)(m1_Cog >> 3);\n";
      os_ << "  const ulong m1_row_blk = (ulong)((m1_m0 + (long)m1_sg_row) >> 3);\n";
    } else {
      os_ << "  threadgroup float* m1_As = m1_smem;\n";
      os_ << "  threadgroup float* m1_Bs = m1_smem + (uint)m1_BK * m1_ALD;\n";
    }
    os_ << "  for (int m1_jD = 0; m1_jD < m1_nD; ++m1_jD) {\n";
    os_ << "    const int m1_rD = m1_r0D + m1_jD * " << D().step << ";\n";
    os_ << "    const long m1_id = m1_qd + (m1_aD + m1_PD - m1_rD * m1_DD) / m1_SD;\n";
    os_ << "    for (int m1_jH = 0; m1_jH < m1_nH; ++m1_jH) {\n";
    os_ << "      const int m1_rH = m1_r0H + m1_jH * " << H().step << ";\n";
    os_ << "      const long m1_ih = m1_qh + (m1_aH + m1_PH - m1_rH * m1_DH) / m1_SH;\n";
    os_ << "      for (int m1_jW = 0; m1_jW < m1_nW; ++m1_jW) {\n";
    os_ << "        const int m1_rW = m1_r0W + m1_jW * " << W().step << ";\n";
    os_ << "        const long m1_iw = m1_qw + (m1_aW + m1_PW - m1_rW * m1_DW) / m1_SW;\n";
    os_ << "        const bool m1_ok = m1_col_ok && m1_id >= 0 && m1_id < m1_Din && m1_ih >= 0 && m1_ih < m1_Hin &&"
        << " m1_iw >= 0 && m1_iw < m1_Win;\n";
    os_ << "        const long m1_ioff = m1_ok ? ((m1_id * m1_Hin + m1_ih) * m1_Win + m1_iw) : 0L;\n";
    os_ << "        const long m1_tap = (long)((m1_rD * m1_KH + m1_rH) * m1_KW + m1_rW);\n";
    os_ << "        for (long m1_c0 = 0; m1_c0 < m1_Cig; m1_c0 += m1_BK) {\n";
    if (!p.w_blk8) {
      os_ << "          for (int m1_i = (int)m1_lid; m1_i < m1_BM * m1_BK; m1_i += m1_NT) {\n";
      os_ << "            const int m1_ak = m1_i / m1_BM;\n";
      os_ << "            const int m1_am = m1_i - m1_ak * m1_BM;\n";
      os_ << "            const long m1_ci = m1_c0 + m1_ak;\n";
      os_ << "            const long m1_co = m1_m0 + m1_am;\n";
      os_ << "            float m1_v = 0.0f;\n";
      os_ << "            if (m1_ci < m1_Cig && m1_co < m1_Cog) m1_v = (float)m1_wg[(m1_ci * m1_Cog + m1_co) * m1_KT "
          << "+ m1_tap];\n";
      os_ << "            m1_As[(uint)m1_ak * m1_ALD + (uint)m1_am] = m1_v;\n";
      os_ << "          }\n";
    }
    os_ << "          for (int m1_r = m1_row; m1_r < m1_BK; m1_r += m1_ROWS) {\n";
    os_ << "            const long m1_ci = m1_c0 + m1_r;\n";
    os_ << "            float m1_v = 0.0f;\n";
    os_ << "            if (m1_ok && m1_ci < m1_Cig) m1_v = (float)m1_xg[m1_ci * m1_in_plane + m1_ioff];\n";
    os_ << "            m1_Bs[(uint)m1_r * m1_BLD + (uint)m1_col] = m1_v;\n";
    os_ << "          }\n";
    os_ << "          threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "          #pragma unroll\n";
    os_ << "          for (uint m1_ks = 0u; m1_ks < (uint)m1_BK; m1_ks += 8u) {\n";
    if (p.w_blk8) {
      os_ << "            const ulong m1_kb = (ulong)((m1_c0 + (long)m1_ks) >> 3);\n";
      os_ << "            device const float* m1_wb = m1_wg + ((m1_kb * (ulong)m1_KT + (ulong)m1_tap) * m1_cog_blk + "
          << "m1_row_blk) * 64ul;\n";
      emit_interleaved_mma_blk8("m1_wb", "m1_Bs + m1_ks * m1_BLD + m1_sg_col", "(ulong)m1_BLD", false, "            ");
    } else {
      os_ << "            simdgroup_float8x8 m1_fa[" << p.fm << "];\n";
      os_ << "            simdgroup_float8x8 m1_fb[" << p.fn << "];\n";
      os_ << "            simdgroup_load(m1_fa[0], m1_As + m1_ks * m1_ALD + m1_sg_row, m1_ALD, ulong2(0, 0), true);\n";
      for (uint32_t j = 0; j < p.fn; ++j) {
        os_ << "            simdgroup_load(m1_fb[" << j << "], m1_Bs + m1_ks * m1_BLD + m1_sg_col + " << (j * 8u)
            << "u, m1_BLD);\n";
      }
      for (uint32_t i = 0; i < p.fm; ++i) {
        if (i + 1u < p.fm) {
          os_ << "            simdgroup_load(m1_fa[" << (i + 1u) << "], m1_As + m1_ks * m1_ALD + m1_sg_row + "
              << ((i + 1u) * 8u) << "u, m1_ALD, ulong2(0, 0), true);\n";
        }
        for (uint32_t j = 0; j < p.fn; ++j) {
          os_ << "            simdgroup_multiply_accumulate(m1_acc[" << i << "][" << j << "], m1_fa[" << i
              << "], m1_fb[" << j << "], m1_acc[" << i << "][" << j << "]);\n";
        }
      }
    }
    os_ << "          }\n";
    os_ << "          threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "        }\n      }\n    }\n  }\n";
    os_ << "  (void)m1_ALD;\n";
  }

  void emit_epilogue() {
    const auto& p = plan_;
    if (p.halo && !p.attr.has_bias && !p.epilogue_dag && !p.fused_tail && p.out_elem == "float" && W().s == 1) {
      os_ << "  if (m1_qd0 + m1_TD <= m1_QD && m1_qh0 + m1_TH <= m1_QH && m1_qw0 + m1_TW <= m1_QW"
          << (p.co_exact ? "" : " && m1_m0 + m1_BM <= m1_Cog") << " && (m1_out_plane & 1023L) != 0) {\n";
      os_ << "    device float* m1_og = " << p.out_name
          << " + (m1_n * m1_Cout + m1_g * m1_Cog + m1_m0 + (long)m1_sg_row) * m1_out_plane;\n";
      for (uint32_t j = 0; j < p.fn; ++j) {
        os_ << "    const uint m1_ec" << j << " = m1_sg_col + " << (j * 8u) << "u;\n";
        os_ << "    const long m1_ed" << j << " = m1_aD + (long)m1_SD * (m1_qd0 + (long)(m1_ec" << j
            << " / (uint)(m1_TW * m1_TH)));\n";
        os_ << "    const long m1_eh" << j << " = m1_aH + (long)m1_SH * (m1_qh0 + (long)((m1_ec" << j
            << " / (uint)m1_TW) % (uint)m1_TH));\n";
        os_ << "    const long m1_ew" << j << " = m1_qw0 + (long)(m1_ec" << j << " % (uint)m1_TW);\n";
        os_ << "    const long m1_eoff" << j << " = (m1_ed" << j << " * m1_Hout + m1_eh" << j << ") * m1_Wout + m1_ew"
            << j << ";\n";
      }
      for (uint32_t i = 0; i < p.fm; ++i) {
        for (uint32_t j = 0; j < p.fn; ++j) {
          os_ << "    simdgroup_store(m1_acc[" << i << "][" << j << "], m1_og + " << (i * 8u)
              << "L * m1_out_plane + m1_eoff" << j << ", (ulong)m1_out_plane);\n";
        }
      }
      os_ << "    return;\n";
      os_ << "  }\n";
    }
    os_ << "  threadgroup float* m1_Cs = m1_smem;\n";
    os_ << "  constexpr uint m1_WM = " << p.wm << "u, m1_WN = " << p.wn << "u;\n";
    if (p.fused_tail) {
      os_ << "  threadgroup float m1_tail[" << p.bm << "];\n";
      os_ << "  for (uint m1_r = m1_lid; m1_r < (uint)m1_BM; m1_r += (uint)m1_NT) m1_tail[m1_r] = " << tail_init()
          << ";\n";
      os_ << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    os_ << "  for (uint m1_sg = 0u; m1_sg < " << p.num_sg << "u; ++m1_sg) {\n";
    os_ << "    if (m1_sgid == m1_sg) {\n";
    for (uint32_t i = 0; i < p.fm; ++i) {
      for (uint32_t j = 0; j < p.fn; ++j) {
        os_ << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_Cs + " << (i * 8u) << "u * m1_WN + "
            << (j * 8u) << "u, m1_WN);\n";
      }
    }
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    os_ << "    const long m1_gr0 = m1_m0 + (long)((m1_sg / " << p.sg_n << "u) * m1_WM);\n";
    os_ << "    const uint m1_gc0 = (m1_sg % " << p.sg_n << "u) * m1_WN;\n";
    os_ << "    for (uint m1_idx = m1_lid; m1_idx < m1_WM * m1_WN; m1_idx += (uint)m1_NT) {\n";
    os_ << "      const uint m1_lr = m1_idx / m1_WN;\n";
    os_ << "      const uint m1_lc = m1_gc0 + (m1_idx - m1_lr * m1_WN);\n";
    os_ << "      const long m1_co = m1_gr0 + (long)m1_lr;\n";
    if (!p.co_exact && !p.fused_tail) os_ << "      if (m1_co >= m1_Cog) continue;\n";
    if (p.halo) {
      os_ << "      const long m1_eqw = m1_qw0 + (long)(m1_lc % (uint)m1_TW);\n";
      os_ << "      const long m1_eqh = m1_qh0 + (long)((m1_lc / (uint)m1_TW) % (uint)m1_TH);\n";
      os_ << "      const long m1_eqd = m1_qd0 + (long)(m1_lc / (uint)(m1_TW * m1_TH));\n";
    } else {
      os_ << "      const long m1_eq = m1_q0 + (long)m1_lc;\n";
      os_ << "      const long m1_eqw = m1_eq % m1_QWm;\n";
      os_ << "      const long m1_eqt = m1_eq / m1_QWm;\n";
      os_ << "      const long m1_eqh = m1_eqt % m1_QHm;\n";
      os_ << "      const long m1_eqd = m1_eqt / m1_QHm;\n";
    }
    const std::string co_guard = (!p.co_exact && p.fused_tail) ? "m1_co < m1_Cog && " : "";
    os_ << "      if (" << co_guard << "m1_eqd < m1_QD && m1_eqh < m1_QH && m1_eqw < m1_QW) {\n";
    os_ << "        const long m1_od = m1_aD + (long)m1_SD * m1_eqd;\n";
    os_ << "        const long m1_oh = m1_aH + (long)m1_SH * m1_eqh;\n";
    os_ << "        const long m1_ow = m1_aW + (long)m1_SW * m1_eqw;\n";
    os_ << "        const long m1_ch = m1_g * m1_Cog + m1_co;\n";
    os_ << "        float m1_v = m1_Cs[m1_idx];\n";
    if (p.attr.has_bias) {
      os_ << "        m1_v += (float)" << p.bias_name << "[m1_ch];\n";
    }
    if (p.epilogue_dag) {
      os_ << "        const long m1_eidx = (m1_n * m1_Cout + m1_ch) * m1_out_plane"
          << " + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow;\n";
      emit_dag("m1_v", "m1_ch", "m1_eidx", "        ");
    }
    if (p.fused_tail) {
      os_ << "        m1_Cs[m1_idx] = m1_v;\n";
      os_ << "      } else {\n";
      os_ << "        m1_Cs[m1_idx] = " << tail_init() << ";\n";
      os_ << "      }\n";
    } else {
      if (p.epilogue_dag) {
        os_ << "        " << p.out_name << "[m1_eidx] = (" << p.out_elem << ")m1_v;\n";
      } else {
        os_ << "        " << p.out_name
            << "[(m1_n * m1_Cout + m1_ch) * m1_out_plane + (m1_od * m1_Hout + m1_oh) * m1_Wout + m1_ow] = ("
            << p.out_elem << ")m1_v;\n";
      }
      os_ << "      }\n";
    }
    os_ << "    }\n";
    os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    if (p.fused_tail) {
      os_ << "    const uint m1_tr0 = (m1_sg / " << p.sg_n << "u) * m1_WM;\n";
      os_ << "    for (uint m1_lr = m1_sgid; m1_lr < m1_WM; m1_lr += " << p.num_sg << "u) {\n";
      os_ << "      float m1_ra = " << tail_init() << ";\n";
      os_ << "      for (uint m1_c = m1_lane; m1_c < m1_WN; m1_c += 32u) {\n";
      os_ << "        m1_ra = " << tail_combine("m1_ra", "m1_Cs[m1_lr * m1_WN + m1_c]") << ";\n";
      os_ << "      }\n";
      os_ << "      m1_ra = " << tail_simd_reduce("m1_ra") << ";\n";
      os_ << "      if (m1_lane == 0u) {\n";
      os_ << "        m1_tail[m1_tr0 + m1_lr] = " << tail_combine("m1_tail[m1_tr0 + m1_lr]", "m1_ra") << ";\n";
      os_ << "      }\n";
      os_ << "    }\n";
      os_ << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    os_ << "  }\n";
    if (p.fused_tail) {
      os_ << "  const long m1_pstride = (long)m1_NPH * (long)m1_ntg.x;\n";
      os_ << "  const long m1_pcol = (long)m1_ph * (long)m1_ntg.x + (long)m1_tgid.x;\n";
      os_ << "  for (uint m1_r = m1_lid; m1_r < (uint)m1_BM; m1_r += (uint)m1_NT) {\n";
      os_ << "    const long m1_co = m1_m0 + (long)m1_r;\n";
      if (!p.co_exact) os_ << "    if (m1_co >= m1_Cog) continue;\n";
      os_ << "    const long m1_ch = m1_g * m1_Cog + m1_co;\n";
      os_ << "    float m1_res = m1_tail[m1_r];\n";
      emit_tail_finalize("m1_res", "    ");
      os_ << "    " << p.out_name << "[(m1_n * m1_Cout + m1_ch) * m1_pstride + m1_pcol] = (" << p.out_elem
          << ")m1_res;\n";
      os_ << "  }\n";
    }
  }

  const Kernel& kernel_;
  const ConvTransposePlan& plan_;
  std::ostringstream os_;
};

}  // namespace

CompiledMSL try_emit_conv_transpose_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
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
  res.msl_source = ConvTransposeWriter(kernel, *plan).emit();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan->threads;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan->out_name);
  return res;
}

}  // namespace m1triton::codegen::metal

#include "codegen/metal/conv_dw_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
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

constexpr const char* kBlockWParam = "BLOCK_W";
constexpr const char* kBlockHParam = "BLOCK_H";
constexpr const char* kVecWParam = "VEC_W";
constexpr const char* kRowsPerThreadParam = "ROWS_PER_THREAD";
constexpr uint32_t kDefaultBlockW = 64;
constexpr uint32_t kDefaultBlockH = 32;
constexpr uint32_t kDefaultThreadsX = 32;
constexpr uint32_t kDefaultThreadsY = 8;
constexpr int64_t kHaloBudgetBytes = 16 * 1024;
constexpr int64_t kMaxKernelTaps = 256;
constexpr int32_t kMaxRegSwKernel = 15;

struct ConvDim {
  int32_t k = 1;
  int32_t s = 1;
  int32_t p = 0;
  int32_t d = 1;
};

struct ConvPlan {
  const Op* op = nullptr;
  std::string x_name;
  std::string w_name;
  std::string out_name;
  std::string bias_name;
  std::string elem;
  bool has_bias = false;
  std::string n_expr;
  std::string cin_expr;
  std::string cout_expr;
  std::string hin_expr;
  std::string win_expr;
  std::string hout_expr;
  std::string wout_expr;
  ConvDim h;
  ConvDim w;
  uint32_t tx = kDefaultThreadsX;
  uint32_t ty = kDefaultThreadsY;
  uint32_t bw = kDefaultBlockW;
  uint32_t bh = kDefaultBlockH;
  uint32_t rh = 1;
  bool reg_sw = false;
  bool sram = false;
  int64_t halo_w = 0;
  int64_t halo_h = 0;
};

int64_t halo_extent(int64_t tile, const ConvDim& dim) {
  return (tile - 1) * dim.s + static_cast<int64_t>(dim.d) * (dim.k - 1) + 1;
}

bool read_dim(const ConvAttr& a, size_t axis, ConvDim& out) {
  auto pick = [axis](const std::vector<int32_t>& v, int32_t fallback) {
    if (v.empty()) return fallback;
    return v.size() == 1 ? v[0] : v[axis];
  };
  out.k = pick(a.kernel_size, 1);
  out.s = pick(a.stride, 1);
  out.p = pick(a.padding, 0);
  out.d = pick(a.dilation, 1);
  return out.k >= 1 && out.s >= 1 && out.p >= 0 && out.d >= 1;
}

std::optional<ConvPlan> build_plan(const Kernel& kernel, const ir::TargetSpec& target, std::string& err) {
  ConvPlan plan;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::ImplicitGemmConv:
        if (plan.op != nullptr) {
          err = "depthwise conv emitter supports exactly one conv op per kernel";
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
  if (ca == nullptr || ca->transposed || !ca->is_depthwise) return std::nullopt;
  if (ca->ndim != 2 || ca->kernel_size.empty()) {
    err = "depthwise conv emitter supports 2D convolutions with an explicit kernel_size";
    return std::nullopt;
  }
  if (!read_dim(*ca, 0, plan.h) || !read_dim(*ca, 1, plan.w)) {
    err = "depthwise conv has invalid kernel/stride/padding/dilation attributes";
    return std::nullopt;
  }
  if (static_cast<int64_t>(plan.h.k) * plan.w.k > kMaxKernelTaps) {
    err = "depthwise conv kernel exceeds the register-cached tap budget";
    return std::nullopt;
  }

  plan.has_bias = ca->has_bias;
  const size_t num_ptrs = plan.has_bias ? 4 : 3;
  const auto& ops = plan.op->operands;
  if (ops.size() != num_ptrs + 6 && ops.size() != num_ptrs + 7) {
    err = "depthwise conv expects operands [x, w, out, (bias), N, C_in, (C_out), H_in, W_in, H_out, W_out]";
    return std::nullopt;
  }

  std::vector<uint32_t> ptr_idx;
  for (size_t i = 0; i < num_ptrs; ++i) {
    const auto idx = param_index_of(kernel, ops[i]);
    if (!idx || !kernel.params[*idx].type.is_ptr()) {
      err = "depthwise conv pointer operands must be pointer kernel parameters";
      return std::nullopt;
    }
    ptr_idx.push_back(*idx);
  }
  const ScalarDType dt = std::get<ir::PtrType>(kernel.params[ptr_idx[0]].type.value).pointee;
  for (uint32_t idx : ptr_idx) {
    if (std::get<ir::PtrType>(kernel.params[idx].type.value).pointee != dt) {
      err = "depthwise conv requires matching element types across x, w, out and bias";
      return std::nullopt;
    }
  }
  if (dt != ScalarDType::F32 && dt != ScalarDType::F16) {
    err = "depthwise conv emitter requires f32 or f16 pointers";
    return std::nullopt;
  }
  plan.elem = msl_scalar_type(dt);
  plan.x_name = kernel.params[ptr_idx[0]].name;
  plan.w_name = kernel.params[ptr_idx[1]].name;
  plan.out_name = kernel.params[ptr_idx[2]].name;
  if (plan.has_bias) plan.bias_name = kernel.params[ptr_idx[3]].name;

  std::vector<std::string> dims;
  for (size_t i = num_ptrs; i < ops.size(); ++i) {
    auto e = scalar_long_expr(kernel, ops[i]);
    if (!e) {
      err = "depthwise conv shape operands must be scalar kernel parameters or constants";
      return std::nullopt;
    }
    dims.push_back(*e);
  }
  const bool has_cout = dims.size() == 7;
  size_t at = 0;
  plan.n_expr = dims[at++];
  plan.cin_expr = dims[at++];
  plan.cout_expr = has_cout ? dims[at++] : plan.cin_expr;
  plan.hin_expr = dims[at++];
  plan.win_expr = dims[at++];
  plan.hout_expr = dims[at++];
  plan.wout_expr = dims[at++];

  const int64_t def_bw = (ca->bk == 1 && ca->bn > 0) ? static_cast<int64_t>(ca->bn) : kDefaultBlockW;
  const int64_t def_bh = (ca->bk == 1 && ca->bm > 0) ? static_cast<int64_t>(ca->bm) : kDefaultBlockH;
  plan.bw = static_cast<uint32_t>(positive_constexpr_param(kernel, kBlockWParam, def_bw));
  plan.bh = static_cast<uint32_t>(positive_constexpr_param(kernel, kBlockHParam, def_bh));
  const int64_t vec_w = positive_constexpr_param(kernel, kVecWParam, 0);
  const int64_t rh_hint = positive_constexpr_param(kernel, kRowsPerThreadParam, 0);

  if ((vec_w == 4 || plan.bw >= 128) && plan.w.s == 1 && plan.w.d == 1 && plan.h.s == 1 && plan.h.d == 1 &&
      plan.w.k <= kMaxRegSwKernel && plan.h.k <= kMaxRegSwKernel &&
      plan.bw >= 4 && plan.bw <= 128 && (plan.bw % 4 == 0)) {
    const uint32_t tx = plan.bw / 4;
    uint32_t rh = rh_hint > 0 ? static_cast<uint32_t>(rh_hint)
                              : (plan.bh >= 64 ? 16u : (plan.bh >= 32 ? 8u : (plan.bh >= 8 ? 4u : 1u)));
    rh = std::max<uint32_t>(1u, std::min(rh, plan.bh));
    while (plan.bh % rh != 0) --rh;
    const uint32_t ty = plan.bh / rh;
    if (tx * ty <= target.max_threads_per_threadgroup) {
      plan.reg_sw = true;
      plan.tx = tx;
      plan.ty = ty;
      plan.rh = rh;
      return plan;
    }
  }

  plan.tx = fit_divisor(kDefaultThreadsX, plan.bw);
  plan.ty = fit_divisor(kDefaultThreadsY, plan.bh);
  if (plan.tx * plan.ty > target.max_threads_per_threadgroup) {
    err = "depthwise conv threadgroup exceeds max_threads_per_threadgroup";
    return std::nullopt;
  }

  plan.halo_w = halo_extent(plan.bw, plan.w);
  plan.halo_h = halo_extent(plan.bh, plan.h);
  const int64_t elem_bytes = dt == ScalarDType::F16 ? 2 : 4;
  const int64_t smem_bytes = plan.halo_w * plan.halo_h * elem_bytes;
  const int64_t budget =
      std::min<int64_t>(kHaloBudgetBytes, static_cast<int64_t>(target.max_threadgroup_memory_bytes));
  plan.sram = smem_bytes <= budget;
  return plan;
}

class ConvDwWriter {
 public:
  ConvDwWriter(const Kernel& kernel, const ConvPlan& plan) : kernel_(kernel), plan_(plan) {}

  std::string emit() {
    emit_signature();
    emit_prologue();
    if (plan_.reg_sw) {
      emit_reg_sw_body();
    } else if (plan_.sram) {
      emit_sram_body();
    } else {
      emit_direct_body();
    }
    os_ << "}\n";
    return os_.str();
  }

 private:
  bool is_f16() const { return plan_.elem == "half"; }

  std::string load_v4_expr(const std::string& ptr) const {
    if (is_f16()) return "float4(half4(*(device const packed_half4*)(" + ptr + ")))";
    return "float4(*(device const packed_float4*)(" + ptr + "))";
  }

  std::string load_v2_expr(const std::string& ptr) const {
    if (is_f16()) return "float2(half2(*(device const packed_half2*)(" + ptr + ")))";
    return "float2(*(device const packed_float2*)(" + ptr + "))";
  }

  std::string store_v4_expr(const std::string& val) const {
    if (is_f16()) return "packed_half4(half4(" + val + "))";
    return "packed_float4(" + val + ")";
  }

  std::string packed4_type() const { return is_f16() ? "packed_half4" : "packed_float4"; }

  void emit_signature() {
    os_ << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    emit_kernel_param_decls(os_, kernel_, plan_.out_name);
    os_ << "    uint3 m1_tgid [[threadgroup_position_in_grid]],\n";
    os_ << "    uint3 m1_tid [[thread_position_in_threadgroup]]) {\n";
  }

  void emit_prologue() {
    os_ << "  constexpr int m1_TX = " << plan_.tx << ";\n";
    os_ << "  constexpr int m1_TY = " << plan_.ty << ";\n";
    os_ << "  constexpr int m1_BW = " << plan_.bw << ";\n";
    os_ << "  constexpr int m1_BH = " << plan_.bh << ";\n";
    os_ << "  constexpr int m1_KH = " << plan_.h.k << ", m1_KW = " << plan_.w.k << ";\n";
    os_ << "  constexpr int m1_SH = " << plan_.h.s << ", m1_SWD = " << plan_.w.s << ";\n";
    os_ << "  constexpr int m1_PH = " << plan_.h.p << ", m1_PW = " << plan_.w.p << ";\n";
    os_ << "  constexpr int m1_DH = " << plan_.h.d << ", m1_DW = " << plan_.w.d << ";\n";
    os_ << "  const long m1_N = " << plan_.n_expr << ", m1_Cin = " << plan_.cin_expr << ", m1_Cout = " << plan_.cout_expr
        << ";\n";
    os_ << "  const long m1_Hin = " << plan_.hin_expr << ", m1_Win = " << plan_.win_expr << ";\n";
    os_ << "  const long m1_Hout = " << plan_.hout_expr << ", m1_Wout = " << plan_.wout_expr << ";\n";
    os_ << "  const long m1_z = (long)m1_tgid.z;\n";
    os_ << "  if (m1_z >= m1_N * m1_Cout) return;\n";
    os_ << "  const long m1_n = m1_z / m1_Cout;\n";
    os_ << "  const long m1_co = m1_z - m1_n * m1_Cout;\n";
    os_ << "  const long m1_ci = m1_co / max(m1_Cout / m1_Cin, 1L);\n";
    os_ << "  const long m1_ow0 = (long)m1_tgid.x * m1_BW;\n";
    os_ << "  const long m1_oh0 = (long)m1_tgid.y * m1_BH;\n";
    os_ << "  device const " << plan_.elem << "* m1_xb = " << plan_.x_name
        << " + (m1_n * m1_Cin + m1_ci) * (m1_Hin * m1_Win);\n";
    os_ << "  device " << plan_.elem << "* m1_ob = " << plan_.out_name << " + m1_z * (m1_Hout * m1_Wout);\n";
    os_ << "  float m1_wr[m1_KH * m1_KW];\n";
    if (plan_.reg_sw) os_ << "  #pragma unroll\n";
    os_ << "  for (int m1_i = 0; m1_i < m1_KH * m1_KW; ++m1_i) {\n";
    os_ << "    m1_wr[m1_i] = (float)" << plan_.w_name << "[m1_co * (m1_KH * m1_KW) + m1_i];\n";
    os_ << "  }\n";
    if (plan_.has_bias) {
      os_ << "  const float m1_bv = (float)" << plan_.bias_name << "[m1_co];\n";
    } else {
      os_ << "  const float m1_bv = 0.0f;\n";
    }
  }

  void emit_reg_sw_load_fast(const std::string& ptr_expr, const std::string& ind) {
    const int nw = 4 + plan_.w.k - 1;
    os_ << ind << "float m1_r[" << nw << "];\n";
    int j = 0;
    for (; j + 4 <= nw; j += 4) {
      const std::string ptr = j == 0 ? ptr_expr : (ptr_expr + " + " + std::to_string(j));
      os_ << ind << "{ const float4 m1_v = " << load_v4_expr(ptr) << ";\n";
      os_ << ind << "  m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; m1_r[" << (j + 2)
          << "] = m1_v.z; m1_r[" << (j + 3) << "] = m1_v.w; }\n";
    }
    if (j + 2 <= nw) {
      const std::string ptr = j == 0 ? ptr_expr : (ptr_expr + " + " + std::to_string(j));
      os_ << ind << "{ const float2 m1_v = " << load_v2_expr(ptr) << ";\n";
      os_ << ind << "  m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; }\n";
      j += 2;
    }
    if (j < nw) {
      os_ << ind << "m1_r[" << j << "] = (float)(" << ptr_expr << ")[" << j << "];\n";
    }
  }

  void emit_reg_sw_load_row(const std::string& ih_expr, const std::string& ind) {
    const int nw = 4 + plan_.w.k - 1;
    os_ << ind << "float m1_r[" << nw << "];\n";
    os_ << ind << "device const " << plan_.elem << "* m1_rp = m1_xb + (" << ih_expr << ") * m1_Win;\n";
    os_ << ind << "if (m1_x_in) {\n";
    os_ << ind << "  device const " << plan_.elem << "* m1_rs = m1_rp + m1_iw0;\n";
    int j = 0;
    for (; j + 4 <= nw; j += 4) {
      const std::string ptr = j == 0 ? "m1_rs" : ("m1_rs + " + std::to_string(j));
      os_ << ind << "  { const float4 m1_v = " << load_v4_expr(ptr) << ";\n";
      os_ << ind << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; m1_r[" << (j + 2)
          << "] = m1_v.z; m1_r[" << (j + 3) << "] = m1_v.w; }\n";
    }
    if (j + 2 <= nw) {
      const std::string ptr = j == 0 ? "m1_rs" : ("m1_rs + " + std::to_string(j));
      os_ << ind << "  { const float2 m1_v = " << load_v2_expr(ptr) << ";\n";
      os_ << ind << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; }\n";
      j += 2;
    }
    if (j < nw) {
      os_ << ind << "  m1_r[" << j << "] = (float)m1_rs[" << j << "];\n";
    }
    os_ << ind << "} else {\n";
    os_ << ind << "  #pragma unroll\n";
    os_ << ind << "  for (int m1_k = 0; m1_k < " << nw << "; ++m1_k) {\n";
    os_ << ind << "    const long m1_iw = m1_iw0 + m1_k;\n";
    os_ << ind << "    m1_r[m1_k] = (m1_iw >= 0 && m1_iw < m1_Win) ? (float)m1_rp[m1_iw] : 0.0f;\n";
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
  }

  void emit_reg_sw_conv1d(const std::string& acc, int kh_idx, const std::string& ind) {
    for (int kw = 0; kw < plan_.w.k; ++kw) {
      const int widx = kh_idx * plan_.w.k + kw;
      os_ << ind << acc << " = fma(float4(m1_r[" << kw << "], m1_r[" << (kw + 1) << "], m1_r[" << (kw + 2)
          << "], m1_r[" << (kw + 3) << "]), float4(m1_wr[" << widx << "]), " << acc << ");\n";
    }
  }

  void emit_reg_sw_store(const std::string& val, const std::string& oh_expr, const std::string& ind) {
    os_ << ind << "device " << plan_.elem << "* m1_op = m1_ob + (" << oh_expr << ") * m1_Wout + m1_ow;\n";
    os_ << ind << "if (m1_o_full) {\n";
    os_ << ind << "  *(device " << packed4_type() << "*)m1_op = " << store_v4_expr(val) << ";\n";
    os_ << ind << "} else {\n";
    os_ << ind << "  m1_op[0] = " << plan_.elem << "(" << val << ".x);\n";
    os_ << ind << "  if (m1_ow + 1 < m1_Wout) m1_op[1] = " << plan_.elem << "(" << val << ".y);\n";
    os_ << ind << "  if (m1_ow + 2 < m1_Wout) m1_op[2] = " << plan_.elem << "(" << val << ".z);\n";
    os_ << ind << "}\n";
  }

  void emit_reg_sw_body() {
    const int kh = plan_.h.k;
    const int nw = 4 + plan_.w.k - 1;
    const bool check_ih = plan_.h.p > 0;
    os_ << "  (void)m1_TX; (void)m1_TY; (void)m1_SH; (void)m1_SWD; (void)m1_DH; (void)m1_DW;\n";
    os_ << "  constexpr int m1_RH = " << plan_.rh << ";\n";
    os_ << "  constexpr int m1_NW = " << nw << ";\n";
    os_ << "  const long m1_ow = m1_ow0 + (long)m1_tid.x * 4;\n";
    os_ << "  const long m1_oh_base = m1_oh0 + (long)m1_tid.y * m1_RH;\n";
    os_ << "  if (m1_ow >= m1_Wout || m1_oh_base >= m1_Hout) return;\n";
    os_ << "  const int m1_rows = (int)min((long)m1_RH, m1_Hout - m1_oh_base);\n";
    os_ << "  const long m1_iw0 = m1_ow - m1_PW;\n";
    os_ << "  const bool m1_x_in = (m1_iw0 >= 0) && (m1_iw0 + m1_NW <= m1_Win);\n";
    os_ << "  const bool m1_o_full = (m1_ow + 4 <= m1_Wout);\n";
    if (check_ih) {
      os_ << "  const bool m1_fast = m1_x_in && m1_o_full && (m1_oh_base - m1_PH >= 0) && "
          << "(m1_oh_base + m1_rows - 1 - m1_PH + (m1_KH - 1) < m1_Hin);\n";
    } else {
      os_ << "  const bool m1_fast = m1_x_in && m1_o_full;\n";
    }

    os_ << "  if (m1_fast) {\n";
    os_ << "    device const " << plan_.elem << "* m1_rs = m1_xb + (m1_oh_base - m1_PH) * m1_Win + m1_iw0;\n";
    os_ << "    device " << plan_.elem << "* m1_op = m1_ob + m1_oh_base * m1_Wout + m1_ow;\n";
    if (kh == 1) {
      os_ << "    for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      os_ << "      float4 m1_out = float4(m1_bv);\n";
      emit_reg_sw_load_fast("m1_rs", "      ");
      emit_reg_sw_conv1d("m1_out", 0, "      ");
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_out") << ";\n";
      os_ << "      m1_rs += m1_Win;\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "    }\n";
    } else {
      for (int s = 0; s < kh - 1; ++s) {
        os_ << "    float4 m1_acc" << s << " = float4(m1_bv);\n";
      }
      for (int pr = 0; pr < kh - 1; ++pr) {
        os_ << "    {\n";
        emit_reg_sw_load_fast("m1_rs", "      ");
        for (int k = 0; k <= pr; ++k) {
          emit_reg_sw_conv1d("m1_acc" + std::to_string(pr - k), k, "      ");
        }
        os_ << "      m1_rs += m1_Win;\n";
        os_ << "    }\n";
      }
      os_ << "    for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      os_ << "      float4 m1_out = m1_acc0;\n";
      emit_reg_sw_load_fast("m1_rs", "      ");
      emit_reg_sw_conv1d("m1_out", kh - 1, "      ");
      for (int j = 0; j < kh - 2; ++j) {
        os_ << "      m1_acc" << j << " = m1_acc" << (j + 1) << ";\n";
        emit_reg_sw_conv1d("m1_acc" + std::to_string(j), kh - 2 - j, "      ");
      }
      os_ << "      m1_acc" << (kh - 2) << " = float4(m1_bv);\n";
      emit_reg_sw_conv1d("m1_acc" + std::to_string(kh - 2), 0, "      ");
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_out") << ";\n";
      os_ << "      m1_rs += m1_Win;\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "    }\n";
    }
    os_ << "    return;\n";
    os_ << "  }\n";

    if (kh == 1) {
      os_ << "  for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      os_ << "    const long m1_oh = m1_oh_base + m1_ry;\n";
      os_ << "    const long m1_ih = m1_oh - m1_PH;\n";
      os_ << "    float4 m1_out = float4(m1_bv);\n";
      if (check_ih) {
        os_ << "    if (m1_ih >= 0 && m1_ih < m1_Hin) {\n";
        emit_reg_sw_load_row("m1_ih", "      ");
        emit_reg_sw_conv1d("m1_out", 0, "      ");
        os_ << "    }\n";
      } else {
        emit_reg_sw_load_row("m1_ih", "    ");
        emit_reg_sw_conv1d("m1_out", 0, "    ");
      }
      emit_reg_sw_store("m1_out", "m1_oh", "    ");
      os_ << "  }\n";
      return;
    }

    for (int s = 0; s < kh - 1; ++s) {
      os_ << "  float4 m1_acc" << s << " = float4(m1_bv);\n";
    }
    for (int pr = 0; pr < kh - 1; ++pr) {
      os_ << "  {\n";
      os_ << "    const long m1_ih = m1_oh_base - m1_PH + " << pr << ";\n";
      if (check_ih) {
        os_ << "    if (m1_ih >= 0 && m1_ih < m1_Hin) {\n";
        emit_reg_sw_load_row("m1_ih", "      ");
        for (int k = 0; k <= pr; ++k) {
          emit_reg_sw_conv1d("m1_acc" + std::to_string(pr - k), k, "      ");
        }
        os_ << "    }\n";
      } else {
        emit_reg_sw_load_row("m1_ih", "    ");
        for (int k = 0; k <= pr; ++k) {
          emit_reg_sw_conv1d("m1_acc" + std::to_string(pr - k), k, "    ");
        }
      }
      os_ << "  }\n";
    }

    os_ << "  for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
    os_ << "    const long m1_oh = m1_oh_base + m1_ry;\n";
    os_ << "    const long m1_ih = m1_oh - m1_PH + " << (kh - 1) << ";\n";
    os_ << "    float4 m1_out = m1_acc0;\n";
    if (check_ih) {
      os_ << "    if (m1_ih >= 0 && m1_ih < m1_Hin) {\n";
      emit_reg_sw_load_row("m1_ih", "      ");
      emit_reg_sw_conv1d("m1_out", kh - 1, "      ");
      for (int j = 0; j < kh - 2; ++j) {
        os_ << "      m1_acc" << j << " = m1_acc" << (j + 1) << ";\n";
        emit_reg_sw_conv1d("m1_acc" + std::to_string(j), kh - 2 - j, "      ");
      }
      os_ << "      m1_acc" << (kh - 2) << " = float4(m1_bv);\n";
      emit_reg_sw_conv1d("m1_acc" + std::to_string(kh - 2), 0, "      ");
      os_ << "    } else {\n";
      for (int j = 0; j < kh - 2; ++j) {
        os_ << "      m1_acc" << j << " = m1_acc" << (j + 1) << ";\n";
      }
      os_ << "      m1_acc" << (kh - 2) << " = float4(m1_bv);\n";
      os_ << "    }\n";
    } else {
      emit_reg_sw_load_row("m1_ih", "    ");
      emit_reg_sw_conv1d("m1_out", kh - 1, "    ");
      for (int j = 0; j < kh - 2; ++j) {
        os_ << "    m1_acc" << j << " = m1_acc" << (j + 1) << ";\n";
        emit_reg_sw_conv1d("m1_acc" + std::to_string(j), kh - 2 - j, "    ");
      }
      os_ << "    m1_acc" << (kh - 2) << " = float4(m1_bv);\n";
      emit_reg_sw_conv1d("m1_acc" + std::to_string(kh - 2), 0, "    ");
    }
    emit_reg_sw_store("m1_out", "m1_oh", "    ");
    os_ << "  }\n";
  }

  void emit_loop_open() {
    os_ << "  for (int m1_vy = 0; m1_vy < m1_BH / m1_TY; ++m1_vy) {\n";
    os_ << "    const int m1_ly = (int)m1_tid.y + m1_vy * m1_TY;\n";
    os_ << "    const long m1_oh = m1_oh0 + m1_ly;\n";
    os_ << "    if (m1_oh >= m1_Hout) break;\n";
    os_ << "    for (int m1_vx = 0; m1_vx < m1_BW / m1_TX; ++m1_vx) {\n";
    os_ << "      const int m1_lx = (int)m1_tid.x + m1_vx * m1_TX;\n";
    os_ << "      const long m1_ow = m1_ow0 + m1_lx;\n";
    os_ << "      if (m1_ow >= m1_Wout) break;\n";
    os_ << "      float m1_acc = m1_bv;\n";
  }

  void emit_loop_close() {
    os_ << "      m1_ob[m1_oh * m1_Wout + m1_ow] = " << plan_.elem << "(m1_acc);\n";
    os_ << "    }\n  }\n";
  }

  void emit_sram_body() {
    os_ << "  constexpr int m1_HW = " << plan_.halo_w << ";\n";
    os_ << "  constexpr int m1_HH = " << plan_.halo_h << ";\n";
    os_ << "  threadgroup " << plan_.elem << " m1_smem[m1_HH * m1_HW];\n";
    os_ << "  const long m1_ih0 = m1_oh0 * m1_SH - m1_PH;\n";
    os_ << "  const long m1_iw0 = m1_ow0 * m1_SWD - m1_PW;\n";
    os_ << "  const int m1_lin = (int)m1_tid.y * m1_TX + (int)m1_tid.x;\n";
    os_ << "  for (int m1_i = m1_lin; m1_i < m1_HH * m1_HW; m1_i += m1_TX * m1_TY) {\n";
    os_ << "    const int m1_r = m1_i / m1_HW;\n";
    os_ << "    const int m1_c = m1_i - m1_r * m1_HW;\n";
    os_ << "    const long m1_ih = m1_ih0 + m1_r;\n";
    os_ << "    const long m1_iw = m1_iw0 + m1_c;\n";
    os_ << "    " << plan_.elem << " m1_v = " << plan_.elem << "(0);\n";
    os_ << "    if (m1_ih >= 0 && m1_ih < m1_Hin && m1_iw >= 0 && m1_iw < m1_Win) {\n";
    os_ << "      m1_v = m1_xb[m1_ih * m1_Win + m1_iw];\n";
    os_ << "    }\n";
    os_ << "    m1_smem[m1_i] = m1_v;\n";
    os_ << "  }\n";
    os_ << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_loop_open();
    os_ << "      const int m1_base = m1_ly * m1_SH * m1_HW + m1_lx * m1_SWD;\n";
    os_ << "      #pragma unroll\n";
    os_ << "      for (int m1_kh = 0; m1_kh < m1_KH; ++m1_kh) {\n";
    os_ << "        #pragma unroll\n";
    os_ << "        for (int m1_kw = 0; m1_kw < m1_KW; ++m1_kw) {\n";
    os_ << "          m1_acc = fma(m1_wr[m1_kh * m1_KW + m1_kw],\n";
    os_ << "                       (float)m1_smem[m1_base + m1_kh * m1_DH * m1_HW + m1_kw * m1_DW], m1_acc);\n";
    os_ << "        }\n";
    os_ << "      }\n";
    emit_loop_close();
  }

  void emit_direct_body() {
    emit_loop_open();
    os_ << "      const long m1_ih0 = m1_oh * m1_SH - m1_PH;\n";
    os_ << "      const long m1_iw0 = m1_ow * m1_SWD - m1_PW;\n";
    os_ << "      for (int m1_kh = 0; m1_kh < m1_KH; ++m1_kh) {\n";
    os_ << "        const long m1_ih = m1_ih0 + m1_kh * m1_DH;\n";
    os_ << "        if (m1_ih < 0 || m1_ih >= m1_Hin) continue;\n";
    os_ << "        device const " << plan_.elem << "* m1_row = m1_xb + m1_ih * m1_Win;\n";
    os_ << "        for (int m1_kw = 0; m1_kw < m1_KW; ++m1_kw) {\n";
    os_ << "          const long m1_iw = m1_iw0 + m1_kw * m1_DW;\n";
    os_ << "          if (m1_iw >= 0 && m1_iw < m1_Win) {\n";
    os_ << "            m1_acc = fma(m1_wr[m1_kh * m1_KW + m1_kw], (float)m1_row[m1_iw], m1_acc);\n";
    os_ << "          }\n";
    os_ << "        }\n";
    os_ << "      }\n";
    emit_loop_close();
  }

  const Kernel& kernel_;
  const ConvPlan& plan_;
  std::ostringstream os_;
};

}  // namespace

CompiledMSL try_emit_conv_dw_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
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
  res.msl_source = ConvDwWriter(kernel, *plan).emit();
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan->tx;
  res.launch_spec.tg_y = plan->ty;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = plan->reg_sw ? 4 : 1;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = build_single_out_param_slots(kernel, plan->out_name);
  return res;
}

}  // namespace m1triton::codegen::metal

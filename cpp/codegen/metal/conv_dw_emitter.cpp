#include "codegen/metal/conv_dw_emitter.hpp"

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
using ir::ValueId;

constexpr const char* kBlockWParam = "BLOCK_W";
constexpr const char* kBlockHParam = "BLOCK_H";
constexpr uint32_t kDefaultBlockW = 64;
constexpr uint32_t kDefaultBlockH = 32;
constexpr uint32_t kDefaultThreadsX = 32;
constexpr uint32_t kDefaultThreadsY = 8;
constexpr int64_t kHaloBudgetBytes = 16 * 1024;
constexpr int64_t kMaxKernelTaps = 256;

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
  bool sram = false;
  int64_t halo_w = 0;
  int64_t halo_h = 0;
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
    if (const auto* b = std::get_if<bool>(&ca->val)) return *b ? "1L" : "0L";
    return std::nullopt;
  }
  if (def.kind == OpKind::Cast && def.operands.size() == 1) return scalar_expr(kernel, def.operands[0]);
  return std::nullopt;
}

int64_t constexpr_param(const Kernel& kernel, const char* name, int64_t fallback) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name && kp.constexpr_val > 0) return kp.constexpr_val;
  }
  return fallback;
}

uint32_t fit_divisor(uint32_t threads, uint32_t block) {
  threads = std::max<uint32_t>(1, std::min(threads, block));
  while (block % threads != 0) --threads;
  return threads;
}

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
  plan.elem = msl_scalar(dt);
  plan.x_name = kernel.params[ptr_idx[0]].name;
  plan.w_name = kernel.params[ptr_idx[1]].name;
  plan.out_name = kernel.params[ptr_idx[2]].name;
  if (plan.has_bias) plan.bias_name = kernel.params[ptr_idx[3]].name;

  std::vector<std::string> dims;
  for (size_t i = num_ptrs; i < ops.size(); ++i) {
    auto e = scalar_expr(kernel, ops[i]);
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
  plan.bw = static_cast<uint32_t>(constexpr_param(kernel, kBlockWParam, def_bw));
  plan.bh = static_cast<uint32_t>(constexpr_param(kernel, kBlockHParam, def_bh));
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
    if (plan_.sram) {
      emit_sram_body();
    } else {
      emit_direct_body();
    }
    os_ << "}\n";
    return os_.str();
  }

 private:
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
    os_ << "  for (int m1_i = 0; m1_i < m1_KH * m1_KW; ++m1_i) {\n";
    os_ << "    m1_wr[m1_i] = (float)" << plan_.w_name << "[m1_co * (m1_KH * m1_KW) + m1_i];\n";
    os_ << "  }\n";
    if (plan_.has_bias) {
      os_ << "  const float m1_bv = (float)" << plan_.bias_name << "[m1_co];\n";
    } else {
      os_ << "  const float m1_bv = 0.0f;\n";
    }
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
  res.launch_spec.vec_width = 1;
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

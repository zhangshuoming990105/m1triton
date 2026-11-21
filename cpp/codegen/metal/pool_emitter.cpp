#include "codegen/metal/pool_emitter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::PoolAttr;
using ir::PoolKind;
using ir::ScalarDType;
using ir::ValueId;

constexpr const char* kBlockWParam = "BLOCK_W";
constexpr const char* kBlockHParam = "BLOCK_H";

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

struct PoolDim {
  std::string in;
  std::string out;
  int32_t k = 1;
  int32_t s = 1;
  int32_t p = 0;
  int32_t d = 1;
  bool trivial = true;
};

struct PoolPlan {
  const Op* op = nullptr;
  PoolAttr attr;
  std::string x_name;
  std::string out_name;
  std::string elem;
  std::string n_expr;
  std::string c_expr;
  std::array<PoolDim, 3> dims;
  uint32_t tx = 1;
  uint32_t ty = 1;
  uint32_t bw = 1;
  uint32_t bh = 1;
  bool sram = false;
  int64_t smem_w = 0;
  int64_t smem_h = 0;
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

int64_t halo(int64_t tile, const PoolDim& dim) {
  return (tile - 1) * dim.s + static_cast<int64_t>(dim.d) * (dim.k - 1) + 1;
}

std::optional<PoolPlan> build_plan(const Kernel& kernel, const ir::TargetSpec& target, std::string& err) {
  PoolPlan plan;
  for (const auto& op_ptr : kernel.ops) {
    const Op& op = *op_ptr;
    if (op.erased) continue;
    switch (op.kind) {
      case OpKind::Pool:
        if (plan.op != nullptr) {
          err = "pool emitter supports exactly one pool op per kernel";
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

  const auto* pa = std::get_if<PoolAttr>(&plan.op->attr);
  if (pa == nullptr) {
    err = "pool op missing PoolAttr";
    return std::nullopt;
  }
  plan.attr = *pa;
  const uint32_t nd = pa->ndim;
  const auto& ops = plan.op->operands;
  if (nd < 1 || nd > 3 || ops.size() != 4 + 2 * nd || pa->kernel_size.size() != nd || pa->stride.size() != nd ||
      pa->padding.size() != nd || pa->dilation.size() != nd) {
    err = "pool op has malformed operands or attributes; run PoolLoweringPass first";
    return std::nullopt;
  }

  const auto x_idx = param_index_of(kernel, ops[0]);
  const auto o_idx = param_index_of(kernel, ops[1]);
  if (!x_idx || !o_idx || !kernel.params[*x_idx].type.is_ptr() || !kernel.params[*o_idx].type.is_ptr()) {
    err = "pool operands 0 and 1 must be pointer kernel parameters";
    return std::nullopt;
  }
  const ScalarDType x_dt = std::get<ir::PtrType>(kernel.params[*x_idx].type.value).pointee;
  const ScalarDType o_dt = std::get<ir::PtrType>(kernel.params[*o_idx].type.value).pointee;
  if (x_dt != o_dt || (x_dt != ScalarDType::F32 && x_dt != ScalarDType::F16)) {
    err = "pool emitter requires matching f32 or f16 input/output pointers";
    return std::nullopt;
  }
  plan.x_name = kernel.params[*x_idx].name;
  plan.out_name = kernel.params[*o_idx].name;
  plan.elem = msl_scalar(x_dt);

  std::vector<std::string> exprs;
  for (size_t i = 2; i < ops.size(); ++i) {
    auto e = scalar_expr(kernel, ops[i]);
    if (!e) {
      err = "pool shape operands must be scalar kernel parameters or constants";
      return std::nullopt;
    }
    exprs.push_back(*e);
  }
  plan.n_expr = exprs[0];
  plan.c_expr = exprs[1];
  for (uint32_t i = 0; i < nd; ++i) {
    PoolDim& dim = plan.dims[3 - nd + i];
    dim.in = exprs[2 + i];
    dim.out = exprs[2 + nd + i];
    dim.k = pa->kernel_size[i];
    dim.s = pa->stride[i];
    dim.p = pa->padding[i];
    dim.d = pa->dilation[i];
    dim.trivial = false;
  }
  for (uint32_t i = 0; i < 3 - nd; ++i) {
    plan.dims[i].in = "1L";
    plan.dims[i].out = "1L";
  }

  const uint32_t tg_x = pa->tg_x > 0 ? pa->tg_x : (nd == 1 ? 256u : 16u);
  const uint32_t tg_y = nd == 1 ? 1u : (pa->tg_y > 0 ? pa->tg_y : 16u);
  plan.bw = static_cast<uint32_t>(constexpr_param(kernel, kBlockWParam, tg_x));
  plan.bh = nd == 1 ? 1u : static_cast<uint32_t>(constexpr_param(kernel, kBlockHParam, tg_y));
  plan.tx = fit_divisor(tg_x, plan.bw);
  plan.ty = fit_divisor(tg_y, plan.bh);
  if (plan.tx * plan.ty > target.max_threads_per_threadgroup) {
    err = "pool threadgroup exceeds max_threads_per_threadgroup";
    return std::nullopt;
  }

  plan.smem_w = halo(plan.bw, plan.dims[2]);
  plan.smem_h = halo(plan.bh, plan.dims[1]);
  const int64_t smem_bytes = plan.smem_w * plan.smem_h * 4;
  plan.sram = pa->use_sram_halo && plan.dims[0].k == 1 &&
              smem_bytes <= static_cast<int64_t>(target.max_threadgroup_memory_bytes);
  return plan;
}

class PoolWriter {
 public:
  PoolWriter(const Kernel& kernel, const PoolPlan& plan) : kernel_(kernel), plan_(plan) {}

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
  const PoolDim& D() const { return plan_.dims[0]; }
  const PoolDim& H() const { return plan_.dims[1]; }
  const PoolDim& W() const { return plan_.dims[2]; }
  bool is_max() const { return plan_.attr.kind == PoolKind::Max; }
  std::string pad_value() const { return is_max() ? "-INFINITY" : "0.0f"; }

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
    os_ << "  const long m1_Din = " << D().in << ", m1_Hin = " << H().in << ", m1_Win = " << W().in << ";\n";
    os_ << "  const long m1_Dout = " << D().out << ", m1_Hout = " << H().out << ", m1_Wout = " << W().out
        << ";\n";
    os_ << "  const long m1_z = (long)m1_tgid.z;\n";
    if (D().trivial) {
      os_ << "  const long m1_nc = m1_z;\n";
      os_ << "  const long m1_od = 0;\n";
    } else {
      os_ << "  const long m1_nc = m1_z / m1_Dout;\n";
      os_ << "  const long m1_od = m1_z - m1_nc * m1_Dout;\n";
    }
    os_ << "  if (m1_nc >= " << plan_.n_expr << " * " << plan_.c_expr << ") return;\n";
    os_ << "  const long m1_ow0 = (long)m1_tgid.x * m1_BW;\n";
    os_ << "  const long m1_oh0 = (long)m1_tgid.y * m1_BH;\n";
    os_ << "  const long m1_in_plane = m1_Hin * m1_Win;\n";
    os_ << "  const long m1_out_plane = m1_Hout * m1_Wout;\n";
    os_ << "  device const " << plan_.elem << "* m1_xb = " << plan_.x_name << " + m1_nc * (m1_Din * m1_in_plane);\n";
    os_ << "  device " << plan_.elem << "* m1_ob = " << plan_.out_name
        << " + (m1_nc * m1_Dout + m1_od) * m1_out_plane;\n";
    os_ << "  (void)m1_in_plane; (void)m1_oh0; (void)m1_od;\n";
  }

  void emit_combine(const std::string& v, int depth) {
    const std::string ind(static_cast<size_t>(depth) * 2, ' ');
    if (is_max()) {
      os_ << ind << "{ const float m1_v = " << v << "; m1_acc = (m1_v > m1_acc || isnan(m1_v)) ? m1_v : m1_acc; }\n";
    } else {
      os_ << ind << "m1_acc += " << v << ";\n";
    }
  }

  void emit_store(int depth) {
    const std::string ind(static_cast<size_t>(depth) * 2, ' ');
    std::string value = "m1_acc";
    if (!is_max()) {
      if (!plan_.attr.ceil_mode && plan_.attr.count_include_pad) {
        const int64_t denom = static_cast<int64_t>(D().k) * H().k * W().k;
        value = "m1_acc / " + std::to_string(denom) + ".0f";
      } else {
        emit_divisor(depth);
        value = "m1_acc / m1_div";
      }
    }
    os_ << ind << "m1_ob[m1_oh * m1_Wout + m1_ow] = " << plan_.elem << "(" << value << ");\n";
  }

  void emit_divisor(int depth) {
    const std::string ind(static_cast<size_t>(depth) * 2, ' ');
    const char* names[3] = {"d", "h", "w"};
    const char* outs[3] = {"m1_od", "m1_oh", "m1_ow"};
    const char* ins[3] = {"m1_Din", "m1_Hin", "m1_Win"};
    os_ << ind << "float m1_div = 1.0f;\n";
    for (int i = 0; i < 3; ++i) {
      const PoolDim& dim = plan_.dims[static_cast<size_t>(i)];
      if (dim.trivial) continue;
      const std::string st = std::string("m1_s") + names[i];
      const std::string en = std::string("m1_e") + names[i];
      os_ << ind << "{ long " << st << " = " << outs[i] << " * " << dim.s << " - " << dim.p << ";\n";
      os_ << ind << "  long " << en << " = min(" << st << " + " << dim.k << ", " << ins[i] << " + " << dim.p
          << ");\n";
      if (plan_.attr.count_include_pad) {
        os_ << ind << "  m1_div *= (float)(" << en << " - " << st << "); }\n";
      } else {
        os_ << ind << "  " << st << " = max(" << st << ", 0L); " << en << " = min(" << en << ", " << ins[i]
            << ");\n";
        os_ << ind << "  m1_div *= (float)(" << en << " - " << st << "); }\n";
      }
    }
  }

  void emit_output_loop_open() {
    os_ << "  for (int m1_vy = 0; m1_vy < m1_BH / m1_TY; ++m1_vy) {\n";
    os_ << "    const int m1_ly = (int)m1_tid.y + m1_vy * m1_TY;\n";
    os_ << "    const long m1_oh = m1_oh0 + m1_ly;\n";
    os_ << "    if (m1_oh >= m1_Hout) break;\n";
    os_ << "    for (int m1_vx = 0; m1_vx < m1_BW / m1_TX; ++m1_vx) {\n";
    os_ << "      const int m1_lx = (int)m1_tid.x + m1_vx * m1_TX;\n";
    os_ << "      const long m1_ow = m1_ow0 + m1_lx;\n";
    os_ << "      if (m1_ow >= m1_Wout) break;\n";
    os_ << "      float m1_acc = " << pad_value() << ";\n";
  }

  void emit_output_loop_close() {
    emit_store(3);
    os_ << "    }\n  }\n";
  }

  void emit_sram_body() {
    const int64_t sw = plan_.smem_w;
    const int64_t sh = plan_.smem_h;
    os_ << "  constexpr int m1_SW = " << sw << ";\n";
    os_ << "  constexpr int m1_SH = " << sh << ";\n";
    os_ << "  threadgroup float m1_smem[m1_SH * m1_SW];\n";
    os_ << "  const long m1_ih0 = m1_oh0 * " << H().s << " - " << H().p << ";\n";
    os_ << "  const long m1_iw0 = m1_ow0 * " << W().s << " - " << W().p << ";\n";
    os_ << "  (void)m1_ih0;\n";
    os_ << "  const int m1_lid = (int)(m1_tid.y * m1_TX + m1_tid.x);\n";
    os_ << "  for (int m1_i = m1_lid; m1_i < m1_SH * m1_SW; m1_i += m1_TX * m1_TY) {\n";
    if (sh == 1) {
      os_ << "    const long m1_iw = m1_iw0 + m1_i;\n";
      os_ << "    float m1_v = " << pad_value() << ";\n";
      os_ << "    if (m1_iw >= 0 && m1_iw < m1_Win) m1_v = (float)m1_xb[m1_iw];\n";
    } else {
      os_ << "    const int m1_r = m1_i / m1_SW;\n";
      os_ << "    const long m1_ih = m1_ih0 + m1_r;\n";
      os_ << "    const long m1_iw = m1_iw0 + (m1_i - m1_r * m1_SW);\n";
      os_ << "    float m1_v = " << pad_value() << ";\n";
      os_ << "    if (m1_ih >= 0 && m1_ih < m1_Hin && m1_iw >= 0 && m1_iw < m1_Win) m1_v = (float)m1_xb[m1_ih * "
             "m1_Win + m1_iw];\n";
    }
    os_ << "    m1_smem[m1_i] = m1_v;\n";
    os_ << "  }\n";
    os_ << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_output_loop_open();
    os_ << "      #pragma unroll\n";
    os_ << "      for (int m1_kh = 0; m1_kh < " << H().k << "; ++m1_kh) {\n";
    os_ << "        threadgroup const float* m1_row = m1_smem + (m1_ly * " << H().s << " + m1_kh * " << H().d
        << ") * m1_SW + m1_lx * " << W().s << ";\n";
    os_ << "        #pragma unroll\n";
    os_ << "        for (int m1_kw = 0; m1_kw < " << W().k << "; ++m1_kw) {\n";
    emit_combine("m1_row[m1_kw * " + std::to_string(W().d) + "]", 5);
    os_ << "        }\n      }\n";
    emit_output_loop_close();
  }

  void emit_direct_body() {
    emit_output_loop_open();
    os_ << "      const long m1_iw0 = m1_ow * " << W().s << " - " << W().p << ";\n";
    os_ << "      const long m1_ih0 = m1_oh * " << H().s << " - " << H().p << ";\n";
    os_ << "      (void)m1_ih0;\n";
    int depth = 3;
    std::string plane = "m1_xb";
    if (!D().trivial) {
      os_ << "      const long m1_id0 = m1_od * " << D().s << " - " << D().p << ";\n";
      os_ << "      #pragma unroll\n";
      os_ << "      for (int m1_kd = 0; m1_kd < " << D().k << "; ++m1_kd) {\n";
      os_ << "        const long m1_id = m1_id0 + m1_kd * " << D().d << ";\n";
      os_ << "        if (m1_id < 0 || m1_id >= m1_Din) continue;\n";
      os_ << "        device const " << plan_.elem << "* m1_xd = m1_xb + m1_id * m1_in_plane;\n";
      plane = "m1_xd";
      depth = 4;
    }
    std::string ind(static_cast<size_t>(depth) * 2, ' ');
    std::string row = plane;
    if (!H().trivial) {
      os_ << ind << "#pragma unroll\n";
      os_ << ind << "for (int m1_kh = 0; m1_kh < " << H().k << "; ++m1_kh) {\n";
      os_ << ind << "  const long m1_ih = m1_ih0 + m1_kh * " << H().d << ";\n";
      os_ << ind << "  if (m1_ih < 0 || m1_ih >= m1_Hin) continue;\n";
      os_ << ind << "  device const " << plan_.elem << "* m1_xr = " << plane << " + m1_ih * m1_Win;\n";
      row = "m1_xr";
      ++depth;
      ind = std::string(static_cast<size_t>(depth) * 2, ' ');
    }
    os_ << ind << "#pragma unroll\n";
    os_ << ind << "for (int m1_kw = 0; m1_kw < " << W().k << "; ++m1_kw) {\n";
    os_ << ind << "  const long m1_iw = m1_iw0 + m1_kw * " << W().d << ";\n";
    os_ << ind << "  if (m1_iw >= 0 && m1_iw < m1_Win) {\n";
    emit_combine("(float)" + row + "[m1_iw]", depth + 2);
    os_ << ind << "  }\n";
    os_ << ind << "}\n";
    while (depth > 3) {
      --depth;
      os_ << std::string(static_cast<size_t>(depth) * 2, ' ') << "}\n";
    }
    emit_output_loop_close();
  }

  const Kernel& kernel_;
  const PoolPlan& plan_;
  std::ostringstream os_;
};

}  // namespace

CompiledMSL try_emit_pool_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
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
  res.msl_source = PoolWriter(kernel, *plan).emit();
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

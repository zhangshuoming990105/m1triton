#include "codegen/metal/pool_emitter.hpp"

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

using ir::Kernel;
using ir::Op;
using ir::OpKind;
using ir::PoolAttr;
using ir::PoolKind;
using ir::ScalarDType;
using ir::ValueId;

constexpr const char* kBlockWParam = "BLOCK_W";
constexpr const char* kBlockHParam = "BLOCK_H";
constexpr const char* kVecWParam = "VEC_W";
constexpr const char* kRowsPerThreadParam = "ROWS_PER_THREAD";
constexpr int32_t kMaxRegSwKernel = 8;

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
  uint32_t rh = 1;
  bool reg_sw = false;
  bool vec_direct = false;
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
  const int64_t vec_w = constexpr_param(kernel, kVecWParam, 0);
  const int64_t rh_hint = constexpr_param(kernel, kRowsPerThreadParam, 0);

  if (nd == 2 && plan.attr.kind == PoolKind::Max && (vec_w == 4 || plan.bw >= 128) &&
      plan.dims[1].s == 1 && plan.dims[2].s == 1 && plan.dims[1].d == 1 && plan.dims[2].d == 1 &&
      plan.dims[1].k <= kMaxRegSwKernel && plan.dims[2].k <= kMaxRegSwKernel &&
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

  if (nd == 2 && plan.dims[2].d == 1 && plan.dims[2].s >= plan.dims[2].k &&
      (plan.dims[2].k == 2 || plan.dims[2].k == 4) && (vec_w == 4 || plan.bw >= 64)) {
    const uint32_t tx = fit_divisor(32u, plan.bw);
    const uint32_t ty = fit_divisor(8u, plan.bh);
    if (tx * ty <= target.max_threads_per_threadgroup) {
      plan.vec_direct = true;
      plan.tx = tx;
      plan.ty = ty;
      return plan;
    }
  }

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
    if (plan_.reg_sw) {
      emit_reg_sw_max_body();
    } else if (plan_.vec_direct) {
      emit_vec_direct_body();
    } else if (plan_.sram) {
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
  bool is_f16() const { return plan_.elem == "half"; }
  std::string pad_value() const { return is_max() ? "-INFINITY" : "0.0f"; }

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
    if (plan_.reg_sw) {
      os_ << "static inline float4 m1_max4(float4 a, float4 b) {\n";
      os_ << "  return select(b, a, (a > b) || isnan(a));\n";
      os_ << "}\n\n";
    } else if (plan_.vec_direct && is_max()) {
      os_ << "static inline float m1_max1(float a, float b) {\n";
      os_ << "  return (a > b || isnan(a)) ? a : b;\n";
      os_ << "}\n\n";
    }
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

  void emit_reg_sw_row_hmax_fast(const std::string& dst, const std::string& ptr_expr, const std::string& ind) {
    const int kw = W().k;
    const int nw = 4 + kw - 1;
    if (kw == 4) {
      os_ << ind << "float4 " << dst << ";\n";
      os_ << ind << "{\n";
      os_ << ind << "  const float4 m1_v0 = " << load_v4_expr(ptr_expr) << ";\n";
      os_ << ind << "  const float4 m1_v1 = " << load_v4_expr(ptr_expr + " + 3") << ";\n";
      os_ << ind << "  const float4 m1_m0 = m1_max4(m1_v0, float4(m1_v0.yzw, m1_v1.y));\n";
      os_ << ind << "  const float4 m1_m1 = m1_max4(float4(m1_v0.zw, m1_v1.yz), m1_v1);\n";
      os_ << ind << "  " << dst << " = m1_max4(m1_m0, m1_m1);\n";
      os_ << ind << "}\n";
      return;
    }
    os_ << ind << "float4 " << dst << ";\n";
    os_ << ind << "{\n";
    os_ << ind << "  float m1_r[" << nw << "];\n";
    int j = 0;
    for (; j + 4 <= nw; j += 4) {
      const std::string ptr = j == 0 ? ptr_expr : (ptr_expr + " + " + std::to_string(j));
      os_ << ind << "  { const float4 m1_v = " << load_v4_expr(ptr) << ";\n";
      os_ << ind << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; m1_r[" << (j + 2)
          << "] = m1_v.z; m1_r[" << (j + 3) << "] = m1_v.w; }\n";
    }
    if (j + 2 <= nw) {
      const std::string ptr = j == 0 ? ptr_expr : (ptr_expr + " + " + std::to_string(j));
      os_ << ind << "  { const float2 m1_v = " << load_v2_expr(ptr) << ";\n";
      os_ << ind << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; }\n";
      j += 2;
    }
    if (j < nw) {
      os_ << ind << "  m1_r[" << j << "] = (float)(" << ptr_expr << ")[" << j << "];\n";
    }
    os_ << ind << "  " << dst << " = float4(m1_r[0], m1_r[1], m1_r[2], m1_r[3]);\n";
    for (int c = 1; c < kw; ++c) {
      os_ << ind << "  " << dst << " = m1_max4(" << dst << ", float4(m1_r[" << c << "], m1_r[" << (c + 1)
          << "], m1_r[" << (c + 2) << "], m1_r[" << (c + 3) << "]));\n";
    }
    os_ << ind << "}\n";
  }

  void emit_reg_sw_row_hmax(const std::string& dst, const std::string& ih_expr, const std::string& ind) {
    const int kw = W().k;
    const int nw = 4 + kw - 1;
    const bool check_ih = (H().p > 0) || plan_.attr.ceil_mode;
    os_ << ind << "float4 " << dst << " = float4(-INFINITY);\n";
    if (check_ih) {
      os_ << ind << "if ((" << ih_expr << ") >= 0 && (" << ih_expr << ") < m1_Hin) {\n";
    }
    const std::string bi = check_ih ? (ind + "  ") : ind;
    os_ << bi << "device const " << plan_.elem << "* m1_rp = m1_xb + (" << ih_expr << ") * m1_Win;\n";
    os_ << bi << "if (m1_x_in) {\n";
    os_ << bi << "  device const " << plan_.elem << "* m1_rs = m1_rp + m1_iw0;\n";
    if (kw == 4) {
      os_ << bi << "  const float4 m1_v0 = " << load_v4_expr("m1_rs") << ";\n";
      os_ << bi << "  const float4 m1_v1 = " << load_v4_expr("m1_rs + 3") << ";\n";
      os_ << bi << "  const float4 m1_m0 = m1_max4(m1_v0, float4(m1_v0.yzw, m1_v1.y));\n";
      os_ << bi << "  const float4 m1_m1 = m1_max4(float4(m1_v0.zw, m1_v1.yz), m1_v1);\n";
      os_ << bi << "  " << dst << " = m1_max4(m1_m0, m1_m1);\n";
    } else {
      os_ << bi << "  float m1_r[" << nw << "];\n";
      int j = 0;
      for (; j + 4 <= nw; j += 4) {
        const std::string ptr = j == 0 ? "m1_rs" : ("m1_rs + " + std::to_string(j));
        os_ << bi << "  { const float4 m1_v = " << load_v4_expr(ptr) << ";\n";
        os_ << bi << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; m1_r[" << (j + 2)
            << "] = m1_v.z; m1_r[" << (j + 3) << "] = m1_v.w; }\n";
      }
      if (j + 2 <= nw) {
        const std::string ptr = j == 0 ? "m1_rs" : ("m1_rs + " + std::to_string(j));
        os_ << bi << "  { const float2 m1_v = " << load_v2_expr(ptr) << ";\n";
        os_ << bi << "    m1_r[" << j << "] = m1_v.x; m1_r[" << (j + 1) << "] = m1_v.y; }\n";
        j += 2;
      }
      if (j < nw) {
        os_ << bi << "  m1_r[" << j << "] = (float)m1_rs[" << j << "];\n";
      }
      os_ << bi << "  " << dst << " = float4(m1_r[0], m1_r[1], m1_r[2], m1_r[3]);\n";
      for (int c = 1; c < kw; ++c) {
        os_ << bi << "  " << dst << " = m1_max4(" << dst << ", float4(m1_r[" << c << "], m1_r[" << (c + 1)
            << "], m1_r[" << (c + 2) << "], m1_r[" << (c + 3) << "]));\n";
      }
    }
    os_ << bi << "} else {\n";
    os_ << bi << "  float m1_r[" << nw << "];\n";
    os_ << bi << "  #pragma unroll\n";
    os_ << bi << "  for (int m1_k = 0; m1_k < " << nw << "; ++m1_k) {\n";
    os_ << bi << "    const long m1_iw = m1_iw0 + m1_k;\n";
    os_ << bi << "    m1_r[m1_k] = (m1_iw >= 0 && m1_iw < m1_Win) ? (float)m1_rp[m1_iw] : -INFINITY;\n";
    os_ << bi << "  }\n";
    os_ << bi << "  " << dst << " = float4(m1_r[0], m1_r[1], m1_r[2], m1_r[3]);\n";
    for (int c = 1; c < kw; ++c) {
      os_ << bi << "  " << dst << " = m1_max4(" << dst << ", float4(m1_r[" << c << "], m1_r[" << (c + 1)
          << "], m1_r[" << (c + 2) << "], m1_r[" << (c + 3) << "]));\n";
    }
    os_ << bi << "}\n";
    if (check_ih) {
      os_ << ind << "}\n";
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

  std::string balanced_max4_expr(std::vector<std::string> items) const {
    while (items.size() > 1) {
      std::vector<std::string> next;
      for (size_t i = 0; i < items.size(); i += 2) {
        if (i + 1 < items.size()) {
          next.push_back("m1_max4(" + items[i] + ", " + items[i + 1] + ")");
        } else {
          next.push_back(items[i]);
        }
      }
      items = std::move(next);
    }
    return items[0];
  }

  void emit_reg_sw_max_body() {
    const int kh = H().k;
    const int nw = 4 + W().k - 1;
    os_ << "  (void)m1_TX; (void)m1_TY; (void)m1_Din; (void)m1_Dout;\n";
    os_ << "  constexpr int m1_RH = " << plan_.rh << ";\n";
    os_ << "  constexpr int m1_NW = " << nw << ";\n";
    os_ << "  const long m1_ow = m1_ow0 + (long)m1_tid.x * 4;\n";
    os_ << "  const long m1_oh_base = m1_oh0 + (long)m1_tid.y * m1_RH;\n";
    os_ << "  if (m1_ow >= m1_Wout || m1_oh_base >= m1_Hout) return;\n";
    os_ << "  const int m1_rows = (int)min((long)m1_RH, m1_Hout - m1_oh_base);\n";
    os_ << "  const long m1_iw0 = m1_ow - " << W().p << ";\n";
    os_ << "  const bool m1_x_in = (m1_iw0 >= 0) && (m1_iw0 + m1_NW <= m1_Win);\n";
    os_ << "  const bool m1_o_full = (m1_ow + 4 <= m1_Wout);\n";
    os_ << "  const bool m1_fast = m1_x_in && m1_o_full && (m1_oh_base - " << H().p
        << " >= 0) && (m1_oh_base + m1_rows - 1 - " << H().p << " + " << (kh - 1) << " < m1_Hin);\n";

    std::vector<std::string> tree_vars;
    for (int i = 0; i < kh - 1; ++i) tree_vars.push_back("m1_h" + std::to_string(i));
    tree_vars.push_back("m1_hnew");
    const std::string vmax_expr = kh >= 2 ? balanced_max4_expr(std::move(tree_vars)) : "m1_hnew";

    os_ << "  if (m1_fast) {\n";
    os_ << "    device const " << plan_.elem << "* m1_rs = m1_xb + (m1_oh_base - " << H().p
        << ") * m1_Win + m1_iw0;\n";
    os_ << "    device " << plan_.elem << "* m1_op = m1_ob + m1_oh_base * m1_Wout + m1_ow;\n";
    if (kh == 1) {
      os_ << "    for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      emit_reg_sw_row_hmax_fast("m1_hnew", "m1_rs", "      ");
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_hnew") << ";\n";
      os_ << "      m1_rs += m1_Win;\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "    }\n";
    } else if (kh == 4) {
      for (int pr = 0; pr < 3; ++pr) {
        emit_reg_sw_row_hmax_fast("m1_h" + std::to_string(pr), "m1_rs", "    ");
        os_ << "    m1_rs += m1_Win;\n";
      }
      os_ << "    int m1_ry = 0;\n";
      os_ << "    for (; m1_ry + 1 < m1_rows; m1_ry += 2) {\n";
      emit_reg_sw_row_hmax_fast("m1_h3", "m1_rs", "      ");
      os_ << "      m1_rs += m1_Win;\n";
      emit_reg_sw_row_hmax_fast("m1_h4", "m1_rs", "      ");
      os_ << "      m1_rs += m1_Win;\n";
      os_ << "      const float4 m1_m123 = m1_max4(m1_max4(m1_h1, m1_h2), m1_h3);\n";
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_max4(m1_h0, m1_m123)")
          << ";\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_max4(m1_m123, m1_h4)")
          << ";\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "      m1_h0 = m1_h2; m1_h1 = m1_h3; m1_h2 = m1_h4;\n";
      os_ << "    }\n";
      os_ << "    if (m1_ry < m1_rows) {\n";
      emit_reg_sw_row_hmax_fast("m1_hnew", "m1_rs", "      ");
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr(vmax_expr) << ";\n";
      os_ << "    }\n";
    } else {
      for (int pr = 0; pr < kh - 1; ++pr) {
        emit_reg_sw_row_hmax_fast("m1_h" + std::to_string(pr), "m1_rs", "    ");
        os_ << "    m1_rs += m1_Win;\n";
      }
      os_ << "    for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      emit_reg_sw_row_hmax_fast("m1_hnew", "m1_rs", "      ");
      os_ << "      const float4 m1_out = " << vmax_expr << ";\n";
      os_ << "      *(device " << packed4_type() << "*)m1_op = " << store_v4_expr("m1_out") << ";\n";
      for (int i = 0; i < kh - 2; ++i) {
        os_ << "      m1_h" << i << " = m1_h" << (i + 1) << ";\n";
      }
      os_ << "      m1_h" << (kh - 2) << " = m1_hnew;\n";
      os_ << "      m1_rs += m1_Win;\n";
      os_ << "      m1_op += m1_Wout;\n";
      os_ << "    }\n";
    }
    os_ << "    return;\n";
    os_ << "  }\n";

    if (kh == 1) {
      os_ << "  for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
      os_ << "    const long m1_oh = m1_oh_base + m1_ry;\n";
      emit_reg_sw_row_hmax("m1_hnew", "m1_oh - " + std::to_string(H().p), "    ");
      emit_reg_sw_store("m1_hnew", "m1_oh", "    ");
      os_ << "  }\n";
      return;
    }

    for (int pr = 0; pr < kh - 1; ++pr) {
      emit_reg_sw_row_hmax("m1_h" + std::to_string(pr),
                           "m1_oh_base - " + std::to_string(H().p) + " + " + std::to_string(pr), "  ");
    }

    os_ << "  for (int m1_ry = 0; m1_ry < m1_rows; ++m1_ry) {\n";
    os_ << "    const long m1_oh = m1_oh_base + m1_ry;\n";
    emit_reg_sw_row_hmax("m1_hnew", "m1_oh - " + std::to_string(H().p) + " + " + std::to_string(kh - 1), "    ");
    os_ << "    const float4 m1_out = " << vmax_expr << ";\n";
    emit_reg_sw_store("m1_out", "m1_oh", "    ");
    for (int i = 0; i < kh - 2; ++i) {
      os_ << "    m1_h" << i << " = m1_h" << (i + 1) << ";\n";
    }
    os_ << "    m1_h" << (kh - 2) << " = m1_hnew;\n";
    os_ << "  }\n";
  }

  void emit_vec_direct_body() {
    const int kw = W().k;
    emit_output_loop_open();
    os_ << "      const long m1_iw0 = m1_ow * " << W().s << " - " << W().p << ";\n";
    os_ << "      const long m1_ih0 = m1_oh * " << H().s << " - " << H().p << ";\n";
    os_ << "      const bool m1_x_in = (m1_iw0 >= 0) && (m1_iw0 + " << kw << " <= m1_Win);\n";
    os_ << "      #pragma unroll\n";
    os_ << "      for (int m1_kh = 0; m1_kh < " << H().k << "; ++m1_kh) {\n";
    os_ << "        const long m1_ih = m1_ih0 + m1_kh * " << H().d << ";\n";
    os_ << "        if (m1_ih < 0 || m1_ih >= m1_Hin) continue;\n";
    os_ << "        device const " << plan_.elem << "* m1_xr = m1_xb + m1_ih * m1_Win;\n";
    os_ << "        if (m1_x_in) {\n";
    if (kw == 4) {
      os_ << "          const float4 m1_v4 = " << load_v4_expr("m1_xr + m1_iw0") << ";\n";
      if (is_max()) {
        os_ << "          m1_acc = m1_max1(m1_acc, m1_max1(m1_max1(m1_v4.x, m1_v4.y), m1_max1(m1_v4.z, m1_v4.w)));\n";
      } else {
        os_ << "          m1_acc += (m1_v4.x + m1_v4.y) + (m1_v4.z + m1_v4.w);\n";
      }
    } else {
      os_ << "          const float2 m1_v2 = " << load_v2_expr("m1_xr + m1_iw0") << ";\n";
      if (is_max()) {
        os_ << "          m1_acc = m1_max1(m1_acc, m1_max1(m1_v2.x, m1_v2.y));\n";
      } else {
        os_ << "          m1_acc += m1_v2.x + m1_v2.y;\n";
      }
    }
    os_ << "        } else {\n";
    os_ << "          #pragma unroll\n";
    os_ << "          for (int m1_kw = 0; m1_kw < " << kw << "; ++m1_kw) {\n";
    os_ << "            const long m1_iw = m1_iw0 + m1_kw;\n";
    os_ << "            if (m1_iw >= 0 && m1_iw < m1_Win) {\n";
    emit_combine("(float)m1_xr[m1_iw]", 7);
    os_ << "            }\n";
    os_ << "          }\n";
    os_ << "        }\n";
    os_ << "      }\n";
    emit_output_loop_close();
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
  res.launch_spec.vec_width = plan->reg_sw ? 4 : (plan->vec_direct ? static_cast<uint32_t>(plan->dims[2].k) : 1);
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

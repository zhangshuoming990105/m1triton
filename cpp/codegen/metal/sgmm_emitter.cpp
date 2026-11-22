#include "codegen/metal/sgmm_emitter.hpp"

#include "codegen/metal/msl_common.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace m1triton::codegen::metal {

namespace {

using ir::ArangeAttr;
using ir::AxisAttr;
using ir::BinaryAttr;
using ir::BinaryOp;
using ir::Block;
using ir::BlockId;
using ir::ConstAttr;
using ir::DotAttr;
using ir::Kernel;
using ir::KernelParam;
using ir::MatrixStructure;
using ir::Op;
using ir::OpId;
using ir::OpKind;
using ir::PermuteAttr;
using ir::ScalarDType;
using ir::StructureAttr;
using ir::TargetSpec;
using ir::TileSpecAttr;
using ir::Type;
using ir::UnaryAttr;
using ir::UnaryOp;
using ir::ValueId;
using ir::kInvalidId;

bool is_unsupported_op(OpKind kind) {
  switch (kind) {
    case OpKind::Reduce:
    case OpKind::Scan:
    case OpKind::WelfordReduce:
    case OpKind::WelfordCombine:
    case OpKind::OnlineSoftmax:
    case OpKind::OnlineSoftmaxStep:
    case OpKind::ImplicitGemmConv:
    case OpKind::ImplicitGemmLoad:
    case OpKind::FlashAttentionTile:
    case OpKind::ThreadgroupAlloc:
    case OpKind::ThreadgroupStage:
    case OpKind::Barrier:
    case OpKind::SimdgroupMatmul:
    case OpKind::SimdReduce:
    case OpKind::SimdScan:
      return true;
    default:
      return false;
  }
}

struct ScalarInfo {
  bool valid = false;
  std::string expr;
  std::optional<int64_t> const_i64;
  std::optional<double> const_f64;
  bool uses_iv = false;
  bool uses_pid0 = false;
  bool uses_pid1 = false;
  bool uses_pid2 = false;
  std::optional<uint32_t> linear_pid_axis;
  int64_t linear_pid_scale = 0;
  int64_t linear_pid_offset = 0;
  bool is_iv_linear = false;
  int64_t iv_scale = 0;
  int64_t iv_offset = 0;
};

struct ParamMapping {
  std::vector<ParamSlot> slots;
  std::unordered_map<uint32_t, std::string> param_msl_name;
  std::unordered_map<ValueId, uint32_t> val_to_param_idx;
  std::unordered_map<ValueId, std::set<uint32_t>> ptr_origins;
  std::unordered_set<uint32_t> written_params;
};

bool build_param_mapping(const Kernel& kernel, const std::vector<BlockId>& all_blocks, ParamMapping& out) {
  const Block& entry = kernel.block(kernel.entry_block);
  if (entry.args.size() != kernel.params.size()) return false;
  for (size_t i = 0; i < kernel.params.size(); ++i) {
    ValueId vid = entry.args[i];
    out.val_to_param_idx[vid] = static_cast<uint32_t>(i);
    if (kernel.params[i].type.is_ptr()) {
      out.ptr_origins[vid] = {static_cast<uint32_t>(i)};
    }
  }

  auto merge_origins = [&](ValueId dst, const std::set<uint32_t>& src) {
    auto& d = out.ptr_origins[dst];
    d.insert(src.begin(), src.end());
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (BlockId bid : all_blocks) {
      for (OpId oid : kernel.block(bid).ops) {
        const Op& op = kernel.op(oid);
        if (op.erased) continue;
        if ((op.kind == OpKind::PtrAdd || op.kind == OpKind::Splat || op.kind == OpKind::ExpandDims ||
             op.kind == OpKind::Broadcast || op.kind == OpKind::Reshape || op.kind == OpKind::Permute) &&
            !op.results.empty() && !op.operands.empty()) {
          auto it = out.ptr_origins.find(op.operands[0]);
          if (it != out.ptr_origins.end()) {
            size_t prev = out.ptr_origins[op.results[0]].size();
            merge_origins(op.results[0], it->second);
            if (out.ptr_origins[op.results[0]].size() != prev) changed = true;
          }
        } else if (op.kind == OpKind::Select && !op.results.empty() && op.operands.size() >= 3) {
          for (size_t idx : {size_t{1}, size_t{2}}) {
            auto it = out.ptr_origins.find(op.operands[idx]);
            if (it != out.ptr_origins.end()) {
              size_t prev = out.ptr_origins[op.results[0]].size();
              merge_origins(op.results[0], it->second);
              if (out.ptr_origins[op.results[0]].size() != prev) changed = true;
            }
          }
        } else if (op.kind == OpKind::ForLoop && !op.regions.empty()) {
          const Block& body = kernel.block(op.regions[0]);
          size_t num_iter = op.operands.size() > 3 ? op.operands.size() - 3 : 0;
          for (size_t i = 0; i < num_iter; ++i) {
            auto it = out.ptr_origins.find(op.operands[3 + i]);
            if (it != out.ptr_origins.end()) {
              if (1 + i < body.args.size()) {
                size_t prev = out.ptr_origins[body.args[1 + i]].size();
                merge_origins(body.args[1 + i], it->second);
                if (out.ptr_origins[body.args[1 + i]].size() != prev) changed = true;
              }
              if (i < op.results.size()) {
                size_t prev = out.ptr_origins[op.results[i]].size();
                merge_origins(op.results[i], it->second);
                if (out.ptr_origins[op.results[i]].size() != prev) changed = true;
              }
            }
          }
          for (OpId boid : body.ops) {
            const Op& bop = kernel.op(boid);
            if (!bop.erased && bop.kind == OpKind::Yield) {
              for (size_t i = 0; i < bop.operands.size(); ++i) {
                auto it = out.ptr_origins.find(bop.operands[i]);
                if (it != out.ptr_origins.end()) {
                  if (1 + i < body.args.size()) {
                    size_t prev = out.ptr_origins[body.args[1 + i]].size();
                    merge_origins(body.args[1 + i], it->second);
                    if (out.ptr_origins[body.args[1 + i]].size() != prev) changed = true;
                  }
                  if (i < op.results.size()) {
                    size_t prev = out.ptr_origins[op.results[i]].size();
                    merge_origins(op.results[i], it->second);
                    if (out.ptr_origins[op.results[i]].size() != prev) changed = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  for (BlockId bid : all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if ((op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) && !op.operands.empty()) {
        auto it = out.ptr_origins.find(op.operands[0]);
        if (it != out.ptr_origins.end()) {
          out.written_params.insert(it->second.begin(), it->second.end());
        }
      }
    }
  }

  build_param_slots(kernel, out.written_params, out.slots, out.param_msl_name);
  return true;
}

class ScalarAnalyzer {
 public:
  ScalarAnalyzer(const Kernel& kernel, const ParamMapping& params, ValueId loop_iv)
      : kernel_(kernel), params_(params), loop_iv_(loop_iv) {
    const Block& entry = kernel_.block(kernel_.entry_block);
    for (size_t i = 0; i < kernel_.params.size() && i < entry.args.size(); ++i) {
      ValueId vid = entry.args[i];
      const KernelParam& kp = kernel_.params[i];
      ScalarInfo info;
      if (kp.type.is_scalar()) {
        info.valid = true;
        ScalarDType dt = scalar_dtype_of(kp.type);
        if (kp.is_constexpr) {
          info.expr = format_scalar_literal_from_i64(kp.constexpr_val, dt);
          if (dt != ScalarDType::F32 && dt != ScalarDType::F16) {
            info.const_i64 = kp.constexpr_val;
          }
          info.const_f64 = static_cast<double>(kp.constexpr_val);
        } else {
          info.expr = params_.param_msl_name.at(static_cast<uint32_t>(i));
        }
      }
      cache_[vid] = std::move(info);
    }
    if (loop_iv_ != kInvalidId) {
      ScalarInfo iv_info;
      iv_info.valid = true;
      iv_info.expr = "m1_iv";
      iv_info.uses_iv = true;
      iv_info.is_iv_linear = true;
      iv_info.iv_scale = 1;
      iv_info.iv_offset = 0;
      cache_[loop_iv_] = std::move(iv_info);
    }
  }

  ScalarInfo analyze(ValueId vid) {
    if (vid == kInvalidId) return {};
    auto it = cache_.find(vid);
    if (it != cache_.end()) return it->second;

    const auto& val = kernel_.value(vid);
    if (val.type.is_tile() || val.type.is_ptr() || val.def_op == kInvalidId) {
      return cache_[vid] = ScalarInfo{};
    }

    const Op& op = kernel_.op(val.def_op);
    ScalarInfo res;
    ScalarDType dt = scalar_dtype_of(val.type);

    switch (op.kind) {
      case OpKind::Constant: {
        const auto& attr = std::get<ConstAttr>(op.attr);
        res.valid = true;
        res.expr = msl_const_literal(attr, dt);
        res.const_i64 = const_as_i64(attr);
        res.const_f64 = const_as_f64(attr);
        break;
      }
      case OpKind::ProgramId: {
        uint32_t axis = std::get<AxisAttr>(op.attr).axis;
        res.valid = true;
        res.expr = msl_scalar_type(dt) + "(m1_tg_pos." + axis_component(axis) + ")";
        res.uses_pid0 = (axis == 0);
        res.uses_pid1 = (axis == 1);
        res.uses_pid2 = (axis == 2);
        res.linear_pid_axis = axis;
        res.linear_pid_scale = 1;
        res.linear_pid_offset = 0;
        break;
      }
      case OpKind::NumPrograms: {
        uint32_t axis = std::get<AxisAttr>(op.attr).axis;
        res.valid = true;
        res.expr = msl_scalar_type(dt) + "(m1_grid_dim." + axis_component(axis) + ")";
        break;
      }
      case OpKind::Cast: {
        ScalarInfo src = analyze(op.operands[0]);
        if (!src.valid) break;
        res.valid = true;
        res.expr = msl_scalar_type(dt) + "(" + src.expr + ")";
        if (src.const_i64.has_value() && !is_float_dtype(dt)) res.const_i64 = src.const_i64;
        res.const_f64 = src.const_f64;
        res.uses_iv = src.uses_iv;
        res.uses_pid0 = src.uses_pid0;
        res.uses_pid1 = src.uses_pid1;
        res.uses_pid2 = src.uses_pid2;
        if (!is_float_dtype(dt)) {
          res.linear_pid_axis = src.linear_pid_axis;
          res.linear_pid_scale = src.linear_pid_scale;
          res.linear_pid_offset = src.linear_pid_offset;
          res.is_iv_linear = src.is_iv_linear;
          res.iv_scale = src.iv_scale;
          res.iv_offset = src.iv_offset;
        }
        break;
      }
      case OpKind::Unary: {
        ScalarInfo src = analyze(op.operands[0]);
        if (!src.valid) break;
        UnaryOp uop = std::get<UnaryAttr>(op.attr).op;
        if (uop == UnaryOp::Neg) {
          res.valid = true;
          res.expr = "(-(" + src.expr + "))";
          if (src.const_i64.has_value()) res.const_i64 = -(*src.const_i64);
          if (src.const_f64.has_value()) res.const_f64 = -(*src.const_f64);
        } else if (uop == UnaryOp::Not) {
          res.valid = true;
          res.expr = "(!" + src.expr + ")";
          if (src.const_i64.has_value()) res.const_i64 = (*src.const_i64 == 0) ? 1 : 0;
        }
        res.uses_iv = src.uses_iv;
        res.uses_pid0 = src.uses_pid0;
        res.uses_pid1 = src.uses_pid1;
        res.uses_pid2 = src.uses_pid2;
        break;
      }
      case OpKind::Binary: {
        ScalarInfo l = analyze(op.operands[0]);
        ScalarInfo r = analyze(op.operands[1]);
        if (!l.valid || !r.valid) break;
        BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
        res.valid = true;
        res.uses_iv = l.uses_iv || r.uses_iv;
        res.uses_pid0 = l.uses_pid0 || r.uses_pid0;
        res.uses_pid1 = l.uses_pid1 || r.uses_pid1;
        res.uses_pid2 = l.uses_pid2 || r.uses_pid2;

        if (bop == BinaryOp::Max) {
          res.expr = "metal::max(" + l.expr + ", " + r.expr + ")";
          if (l.const_i64 && r.const_i64) res.const_i64 = std::max(*l.const_i64, *r.const_i64);
        } else if (bop == BinaryOp::Min) {
          res.expr = "metal::min(" + l.expr + ", " + r.expr + ")";
          if (l.const_i64 && r.const_i64) res.const_i64 = std::min(*l.const_i64, *r.const_i64);
        } else if (bop == BinaryOp::Pow) {
          res.expr = "metal::pow(float(" + l.expr + "), float(" + r.expr + "))";
        } else {
          const char* sym = nullptr;
          switch (bop) {
            case BinaryOp::Add: sym = " + "; break;
            case BinaryOp::Sub: sym = " - "; break;
            case BinaryOp::Mul: sym = " * "; break;
            case BinaryOp::Div: sym = " / "; break;
            case BinaryOp::Rem: sym = " % "; break;
            case BinaryOp::And: sym = " & "; break;
            case BinaryOp::Or: sym = " | "; break;
            case BinaryOp::Xor: sym = " ^ "; break;
            case BinaryOp::Shl: sym = " << "; break;
            case BinaryOp::Shr: sym = " >> "; break;
            case BinaryOp::CmpEq: sym = " == "; break;
            case BinaryOp::CmpNe: sym = " != "; break;
            case BinaryOp::CmpLt: sym = " < "; break;
            case BinaryOp::CmpLe: sym = " <= "; break;
            case BinaryOp::CmpGt: sym = " > "; break;
            case BinaryOp::CmpGe: sym = " >= "; break;
            default: sym = nullptr; break;
          }
          if (!sym) {
            res.valid = false;
            break;
          }
          res.expr = "(" + l.expr + sym + r.expr + ")";
          if (l.const_i64 && r.const_i64) {
            int64_t a = *l.const_i64;
            int64_t b = *r.const_i64;
            switch (bop) {
              case BinaryOp::Add: res.const_i64 = a + b; break;
              case BinaryOp::Sub: res.const_i64 = a - b; break;
              case BinaryOp::Mul: res.const_i64 = a * b; break;
              case BinaryOp::Div: if (b != 0) res.const_i64 = a / b; break;
              case BinaryOp::Rem: if (b != 0) res.const_i64 = a % b; break;
              case BinaryOp::CmpEq: res.const_i64 = (a == b) ? 1 : 0; break;
              case BinaryOp::CmpNe: res.const_i64 = (a != b) ? 1 : 0; break;
              case BinaryOp::CmpLt: res.const_i64 = (a < b) ? 1 : 0; break;
              case BinaryOp::CmpLe: res.const_i64 = (a <= b) ? 1 : 0; break;
              case BinaryOp::CmpGt: res.const_i64 = (a > b) ? 1 : 0; break;
              case BinaryOp::CmpGe: res.const_i64 = (a >= b) ? 1 : 0; break;
              default: break;
            }
          }
        }

        if (bop == BinaryOp::Mul) {
          if (l.linear_pid_axis && r.const_i64) {
            res.linear_pid_axis = l.linear_pid_axis;
            res.linear_pid_scale = l.linear_pid_scale * (*r.const_i64);
            res.linear_pid_offset = l.linear_pid_offset * (*r.const_i64);
          } else if (r.linear_pid_axis && l.const_i64) {
            res.linear_pid_axis = r.linear_pid_axis;
            res.linear_pid_scale = r.linear_pid_scale * (*l.const_i64);
            res.linear_pid_offset = r.linear_pid_offset * (*l.const_i64);
          }
          if (l.is_iv_linear && r.const_i64) {
            res.is_iv_linear = true;
            res.iv_scale = l.iv_scale * (*r.const_i64);
            res.iv_offset = l.iv_offset * (*r.const_i64);
          } else if (r.is_iv_linear && l.const_i64) {
            res.is_iv_linear = true;
            res.iv_scale = r.iv_scale * (*l.const_i64);
            res.iv_offset = r.iv_offset * (*l.const_i64);
          }
        } else if (bop == BinaryOp::Add) {
          if (l.linear_pid_axis && r.const_i64) {
            res.linear_pid_axis = l.linear_pid_axis;
            res.linear_pid_scale = l.linear_pid_scale;
            res.linear_pid_offset = l.linear_pid_offset + (*r.const_i64);
          } else if (r.linear_pid_axis && l.const_i64) {
            res.linear_pid_axis = r.linear_pid_axis;
            res.linear_pid_scale = r.linear_pid_scale;
            res.linear_pid_offset = r.linear_pid_offset + (*l.const_i64);
          }
          if (l.is_iv_linear && r.const_i64) {
            res.is_iv_linear = true;
            res.iv_scale = l.iv_scale;
            res.iv_offset = l.iv_offset + (*r.const_i64);
          } else if (r.is_iv_linear && l.const_i64) {
            res.is_iv_linear = true;
            res.iv_scale = r.iv_scale;
            res.iv_offset = r.iv_offset + (*l.const_i64);
          }
        }
        break;
      }
      case OpKind::Select: {
        if (op.operands.size() >= 3) {
          ScalarInfo c = analyze(op.operands[0]);
          ScalarInfo t = analyze(op.operands[1]);
          ScalarInfo f = analyze(op.operands[2]);
          if (c.valid && t.valid && f.valid) {
            res.valid = true;
            res.expr = "(" + c.expr + " ? " + t.expr + " : " + f.expr + ")";
            res.uses_iv = c.uses_iv || t.uses_iv || f.uses_iv;
            res.uses_pid0 = c.uses_pid0 || t.uses_pid0 || f.uses_pid0;
            res.uses_pid1 = c.uses_pid1 || t.uses_pid1 || f.uses_pid1;
            res.uses_pid2 = c.uses_pid2 || t.uses_pid2 || f.uses_pid2;
          }
        }
        break;
      }
      default:
        break;
    }

    cache_[vid] = res;
    return res;
  }

 private:
  const Kernel& kernel_;
  const ParamMapping& params_;
  ValueId loop_iv_;
  std::unordered_map<ValueId, ScalarInfo> cache_;
};

struct Affine1D {
  bool valid = false;
  bool has_arange = false;
  int64_t arange_start = 0;
  int64_t arange_end = 0;
  std::string base_expr = "0";
  std::string stride_expr = "1";
  std::optional<int64_t> stride_const;
  bool has_outer_scale = false;
  std::string unscaled_base_expr = "0";
  std::string outer_scale_expr = "1";
  std::optional<int64_t> outer_scale_const;
  bool uses_iv = false;
  bool is_iv_linear = false;
  int64_t iv_scale = 0;
  int64_t iv_offset = 0;
  bool uses_pid0 = false;
  bool uses_pid1 = false;
  bool uses_pid2 = false;
  std::optional<uint32_t> linear_pid_axis;
  int64_t linear_pid_scale = 0;
  int64_t linear_pid_offset = 0;
};

struct Affine2D {
  bool valid = false;
  uint32_t base_param_idx = UINT32_MAX;
  std::string ptr_scalar_offset = "0";
  bool ptr_scalar_uses_pid2 = false;
  Affine1D row;
  Affine1D col;
};

class TileAffineAnalyzer {
 public:
  TileAffineAnalyzer(const Kernel& kernel, const ParamMapping& params, ScalarAnalyzer& scalar, const Op* for_op)
      : kernel_(kernel), params_(params), scalar_(scalar), for_op_(for_op) {}

  Affine1D analyze_1d(ValueId vid) {
    if (vid == kInvalidId) return {};
    auto it = cache_1d_.find(vid);
    if (it != cache_1d_.end()) return it->second;

    const auto& val = kernel_.value(vid);
    Affine1D out;
    if (!val.type.is_tile()) {
      ScalarInfo s = scalar_.analyze(vid);
      if (s.valid) {
        out.valid = true;
        out.has_arange = false;
        out.base_expr = s.expr;
        out.stride_expr = "0";
        out.stride_const = 0;
        out.uses_iv = s.uses_iv;
        out.is_iv_linear = s.is_iv_linear;
        out.iv_scale = s.iv_scale;
        out.iv_offset = s.iv_offset;
        out.uses_pid0 = s.uses_pid0;
        out.uses_pid1 = s.uses_pid1;
        out.uses_pid2 = s.uses_pid2;
        out.linear_pid_axis = s.linear_pid_axis;
        out.linear_pid_scale = s.linear_pid_scale;
        out.linear_pid_offset = s.linear_pid_offset;
      }
      return cache_1d_[vid] = out;
    }

    if (val.type.as_tile().shape.size() != 1 || val.def_op == kInvalidId) {
      return cache_1d_[vid] = out;
    }

    const Op& op = kernel_.op(val.def_op);
    switch (op.kind) {
      case OpKind::Arange: {
        const auto& a = std::get<ArangeAttr>(op.attr);
        out.valid = true;
        out.has_arange = true;
        out.arange_start = a.start;
        out.arange_end = a.end;
        out.base_expr = "0";
        out.stride_expr = "1";
        out.stride_const = 1;
        break;
      }
      case OpKind::Splat: {
        ScalarInfo s = scalar_.analyze(op.operands[0]);
        if (s.valid) {
          out.valid = true;
          out.has_arange = false;
          out.base_expr = s.expr;
          out.stride_expr = "0";
          out.stride_const = 0;
          out.uses_iv = s.uses_iv;
          out.is_iv_linear = s.is_iv_linear;
          out.iv_scale = s.iv_scale;
          out.iv_offset = s.iv_offset;
          out.uses_pid0 = s.uses_pid0;
          out.uses_pid1 = s.uses_pid1;
          out.uses_pid2 = s.uses_pid2;
          out.linear_pid_axis = s.linear_pid_axis;
          out.linear_pid_scale = s.linear_pid_scale;
          out.linear_pid_offset = s.linear_pid_offset;
        }
        break;
      }
      case OpKind::Cast: {
        Affine1D src = analyze_1d(op.operands[0]);
        if (src.valid && !is_float_dtype(scalar_dtype_of(val.type))) {
          out = src;
        }
        break;
      }
      case OpKind::Binary: {
        Affine1D l = analyze_1d(op.operands[0]);
        Affine1D r = analyze_1d(op.operands[1]);
        if (!l.valid || !r.valid) break;
        BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
        if (bop == BinaryOp::Add || bop == BinaryOp::Sub) {
          if (l.has_arange && r.has_arange) break;
          if (bop == BinaryOp::Sub && r.has_arange) break;
          const Affine1D& ar = l.has_arange ? l : r;
          const Affine1D& sc = l.has_arange ? r : l;
          out = ar;
          out.uses_iv = l.uses_iv || r.uses_iv;
          out.uses_pid0 = l.uses_pid0 || r.uses_pid0;
          out.uses_pid1 = l.uses_pid1 || r.uses_pid1;
          out.uses_pid2 = l.uses_pid2 || r.uses_pid2;
          const char* sym = (bop == BinaryOp::Add) ? " + " : " - ";
          if (l.base_expr == "0" && bop == BinaryOp::Add) {
            out.base_expr = r.base_expr;
          } else if (r.base_expr == "0") {
            out.base_expr = l.base_expr;
          } else {
            out.base_expr = "(" + l.base_expr + sym + r.base_expr + ")";
          }
          if (!ar.has_outer_scale && ar.has_arange && sc.base_expr != "0") {
            out.unscaled_base_expr = out.base_expr;
            if (!ar.is_iv_linear && sc.is_iv_linear && bop == BinaryOp::Add) {
              out.is_iv_linear = true;
              out.iv_scale = sc.iv_scale;
              out.iv_offset = sc.iv_offset;
            } else {
              out.is_iv_linear = false;
            }
            if (!ar.linear_pid_axis && sc.linear_pid_axis && bop == BinaryOp::Add) {
              out.linear_pid_axis = sc.linear_pid_axis;
              out.linear_pid_scale = sc.linear_pid_scale;
              out.linear_pid_offset = sc.linear_pid_offset;
            } else {
              out.linear_pid_axis.reset();
            }
          }
        } else if (bop == BinaryOp::Mul) {
          if (l.has_arange && r.has_arange) break;
          const Affine1D& ar = l.has_arange ? l : r;
          const Affine1D& sc = l.has_arange ? r : l;
          if (!ar.has_arange || sc.uses_iv) break;
          out = ar;
          out.uses_iv = l.uses_iv || r.uses_iv;
          out.uses_pid0 = l.uses_pid0 || r.uses_pid0;
          out.uses_pid1 = l.uses_pid1 || r.uses_pid1;
          out.uses_pid2 = l.uses_pid2 || r.uses_pid2;
          std::optional<int64_t> sc_const;
          if (!kernel_.value(op.operands[l.has_arange ? 1 : 0]).type.is_tile()) {
            sc_const = scalar_.analyze(op.operands[l.has_arange ? 1 : 0]).const_i64;
          } else if (kernel_.op(kernel_.value(op.operands[l.has_arange ? 1 : 0]).def_op).kind == OpKind::Splat) {
            sc_const = scalar_.analyze(kernel_.op(kernel_.value(op.operands[l.has_arange ? 1 : 0]).def_op).operands[0]).const_i64;
          }
          if (!ar.has_outer_scale && ar.stride_const == 1) {
            out.has_outer_scale = true;
            out.unscaled_base_expr = ar.base_expr;
            out.outer_scale_expr = sc.base_expr;
            out.outer_scale_const = sc_const;
            out.stride_expr = sc.base_expr;
            out.stride_const = sc_const;
            out.base_expr = (ar.base_expr == "0") ? "0" : ("(" + ar.base_expr + " * " + sc.base_expr + ")");
          } else {
            out.outer_scale_expr = "(" + out.outer_scale_expr + " * " + sc.base_expr + ")";
            if (out.outer_scale_const && sc_const) {
              out.outer_scale_const = (*out.outer_scale_const) * (*sc_const);
            } else {
              out.outer_scale_const.reset();
            }
            out.stride_expr = "(" + out.stride_expr + " * " + sc.base_expr + ")";
            if (out.stride_const && sc_const) {
              out.stride_const = (*out.stride_const) * (*sc_const);
            } else {
              out.stride_const.reset();
            }
            out.base_expr = (ar.base_expr == "0") ? "0" : ("(" + ar.base_expr + " * " + sc.base_expr + ")");
          }
        }
        break;
      }
      default:
        break;
    }
    return cache_1d_[vid] = out;
  }

  Affine2D analyze_2d_ptr(ValueId vid) {
    if (vid == kInvalidId) return {};
    auto it = cache_2d_.find(vid);
    if (it != cache_2d_.end()) return it->second;

    const auto& val = kernel_.value(vid);
    Affine2D out;

    if (val.def_block == kernel_.entry_block && params_.val_to_param_idx.count(vid)) {
      uint32_t pidx = params_.val_to_param_idx.at(vid);
      if (kernel_.params[pidx].type.is_ptr()) {
        out.valid = true;
        out.base_param_idx = pidx;
      }
      return cache_2d_[vid] = out;
    }

    if (for_op_ && !for_op_->regions.empty()) {
      const Block& body = kernel_.block(for_op_->regions[0]);
      for (size_t i = 1; i < body.args.size(); ++i) {
        if (body.args[i] == vid) {
          ValueId init_v = for_op_->operands[3 + (i - 1)];
          Affine2D init_aff = analyze_2d_ptr(init_v);
          if (!init_aff.valid) return cache_2d_[vid] = out;

          ValueId yield_v = kInvalidId;
          for (OpId oid : body.ops) {
            if (kernel_.op(oid).kind == OpKind::Yield) {
              if (i - 1 < kernel_.op(oid).operands.size()) {
                yield_v = kernel_.op(oid).operands[i - 1];
              }
            }
          }
          if (yield_v == kInvalidId || kernel_.value(yield_v).def_op == kInvalidId) {
            return cache_2d_[vid] = out;
          }
          const Op& yop = kernel_.op(kernel_.value(yield_v).def_op);
          if (yop.kind != OpKind::PtrAdd || yop.operands.size() < 2 || yop.operands[0] != vid) {
            return cache_2d_[vid] = out;
          }
          ValueId inc_v = yop.operands[1];
          std::string inc_expr;
          if (!kernel_.value(inc_v).type.is_tile()) {
            ScalarInfo s = scalar_.analyze(inc_v);
            if (!s.valid) return cache_2d_[vid] = out;
            inc_expr = s.expr;
          } else if (kernel_.value(inc_v).def_op != kInvalidId &&
                     kernel_.op(kernel_.value(inc_v).def_op).kind == OpKind::Splat) {
            ScalarInfo s = scalar_.analyze(kernel_.op(kernel_.value(inc_v).def_op).operands[0]);
            if (!s.valid) return cache_2d_[vid] = out;
            inc_expr = s.expr;
          } else {
            return cache_2d_[vid] = out;
          }
          out = init_aff;
          ScalarInfo step_info = scalar_.analyze(for_op_->operands[2]);
          if (!step_info.valid || !step_info.const_i64.has_value() || *step_info.const_i64 <= 0) {
            return cache_2d_[vid] = out;
          }
          bool pick_col = false;
          bool pick_row = false;
          if (out.col.has_arange && !out.col.uses_iv && !out.col.uses_pid0 && !out.col.uses_pid1) {
            pick_col = true;
          } else if (out.row.has_arange && !out.row.uses_iv && !out.row.uses_pid0 && !out.row.uses_pid1) {
            pick_row = true;
          } else if (out.col.has_arange && !out.col.uses_iv) {
            pick_col = true;
          } else if (out.row.has_arange && !out.row.uses_iv) {
            pick_row = true;
          }
          if (pick_col) {
            int64_t scale = (*step_info.const_i64 == 1) ? out.col.arange_end : 1;
            out.col.uses_iv = true;
            out.col.is_iv_linear = true;
            out.col.iv_scale = scale;
            out.col.iv_offset = 0;
            out.col.unscaled_base_expr = (scale == 1) ? "m1_iv" : ("(m1_iv * " + std::to_string(scale) + ")");
            out.col.base_expr = "(" + out.col.unscaled_base_expr + " * " + out.col.stride_expr + ")";
          } else if (pick_row) {
            int64_t scale = (*step_info.const_i64 == 1) ? out.row.arange_end : 1;
            out.row.uses_iv = true;
            out.row.is_iv_linear = true;
            out.row.iv_scale = scale;
            out.row.iv_offset = 0;
            out.row.unscaled_base_expr = (scale == 1) ? "m1_iv" : ("(m1_iv * " + std::to_string(scale) + ")");
            out.row.base_expr = "(" + out.row.unscaled_base_expr + " * " + out.row.stride_expr + ")";
          } else {
            out.valid = false;
          }
          (void)inc_expr;
          return cache_2d_[vid] = out;
        }
      }
    }

    if (val.def_op == kInvalidId) return cache_2d_[vid] = out;
    const Op& op = kernel_.op(val.def_op);

    switch (op.kind) {
      case OpKind::Cast: {
        if (!op.operands.empty() && !is_float_dtype(scalar_dtype_of(val.type)) &&
            !is_float_dtype(scalar_dtype_of(kernel_.value(op.operands[0]).type))) {
          out = analyze_2d_ptr(op.operands[0]);
        }
        break;
      }
      case OpKind::Splat: {
        ScalarInfo s = scalar_.analyze(op.operands[0]);
        if (s.valid) {
          out.valid = true;
          out.ptr_scalar_offset = s.expr;
          out.ptr_scalar_uses_pid2 = s.uses_pid2;
        }
        break;
      }
      case OpKind::ExpandDims: {
        uint32_t axis = std::get<AxisAttr>(op.attr).axis;
        const Type& src_ty = kernel_.value(op.operands[0]).type;
        if (!src_ty.is_tile() || src_ty.as_tile().shape.size() != 1) break;
        Affine1D a1 = analyze_1d(op.operands[0]);
        if (!a1.valid) break;
        out.valid = true;
        if (axis == 1) {
          out.row = a1;
        } else if (axis == 0) {
          out.col = a1;
        } else {
          out.valid = false;
        }
        break;
      }
      case OpKind::Broadcast: {
        const Type& src_ty = kernel_.value(op.operands[0]).type;
        if (!src_ty.is_tile()) {
          ScalarInfo s = scalar_.analyze(op.operands[0]);
          if (s.valid) {
            out.valid = true;
            out.ptr_scalar_offset = s.expr;
            out.ptr_scalar_uses_pid2 = s.uses_pid2;
          }
          break;
        }
        const auto& src_shape = src_ty.as_tile().shape;
        if (src_shape.size() == 2) {
          out = analyze_2d_ptr(op.operands[0]);
        }
        break;
      }
      case OpKind::PtrAdd: {
        Affine2D base = analyze_2d_ptr(op.operands[0]);
        if (!base.valid || base.base_param_idx == UINT32_MAX) break;
        const Type& off_ty = kernel_.value(op.operands[1]).type;
        if (!off_ty.is_tile()) {
          ScalarInfo s = scalar_.analyze(op.operands[1]);
          if (!s.valid) break;
          out = base;
          out.ptr_scalar_offset =
              (out.ptr_scalar_offset == "0") ? s.expr : ("(" + out.ptr_scalar_offset + " + " + s.expr + ")");
          out.ptr_scalar_uses_pid2 = out.ptr_scalar_uses_pid2 || s.uses_pid2;
        } else {
          Affine2D off = analyze_2d_ptr(op.operands[1]);
          if (!off.valid || off.base_param_idx != UINT32_MAX) break;
          out = base;
          if (off.ptr_scalar_offset != "0") {
            out.ptr_scalar_offset = (out.ptr_scalar_offset == "0")
                                        ? off.ptr_scalar_offset
                                        : ("(" + out.ptr_scalar_offset + " + " + off.ptr_scalar_offset + ")");
            out.ptr_scalar_uses_pid2 = out.ptr_scalar_uses_pid2 || off.ptr_scalar_uses_pid2;
          }
          if (off.row.valid) {
            if (out.row.valid && out.row.has_arange) break;
            out.row = off.row;
          }
          if (off.col.valid) {
            if (out.col.valid && out.col.has_arange) break;
            out.col = off.col;
          }
        }
        break;
      }
      case OpKind::Binary: {
        const BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
        if (bop == BinaryOp::Mul) {
          Affine2D l = analyze_2d_ptr(op.operands[0]);
          Affine2D r = analyze_2d_ptr(op.operands[1]);
          if (!l.valid || !r.valid) break;
          if (l.base_param_idx != UINT32_MAX || r.base_param_idx != UINT32_MAX) break;
          const bool l_has_axis = l.row.has_arange || l.col.has_arange;
          const bool r_has_axis = r.row.has_arange || r.col.has_arange;
          if (l_has_axis == r_has_axis) break;
          const Affine2D& ax = l_has_axis ? l : r;
          const Affine2D& sc = l_has_axis ? r : l;
          if (sc.row.valid || sc.col.valid || sc.ptr_scalar_offset == "0") break;
          const ValueId sc_vid = op.operands[l_has_axis ? 1 : 0];
          std::optional<int64_t> sc_const;
          if (!kernel_.value(sc_vid).type.is_tile()) {
            sc_const = scalar_.analyze(sc_vid).const_i64;
          } else if (kernel_.value(sc_vid).def_op != kInvalidId &&
                     kernel_.op(kernel_.value(sc_vid).def_op).kind == OpKind::Splat) {
            sc_const = scalar_.analyze(kernel_.op(kernel_.value(sc_vid).def_op).operands[0]).const_i64;
          }
          auto scale_1d = [&](Affine1D a) -> Affine1D {
            if (!a.has_arange) return a;
            if (!a.has_outer_scale && a.stride_const == 1) {
              a.has_outer_scale = true;
              a.unscaled_base_expr = a.base_expr;
              a.outer_scale_expr = sc.ptr_scalar_offset;
              a.outer_scale_const = sc_const;
              a.stride_expr = sc.ptr_scalar_offset;
              a.stride_const = sc_const;
              a.base_expr = (a.base_expr == "0") ? "0" : ("(" + a.base_expr + " * " + sc.ptr_scalar_offset + ")");
            } else {
              a.outer_scale_expr = "(" + a.outer_scale_expr + " * " + sc.ptr_scalar_offset + ")";
              if (a.outer_scale_const && sc_const) {
                a.outer_scale_const = (*a.outer_scale_const) * (*sc_const);
              } else {
                a.outer_scale_const.reset();
              }
              a.stride_expr = "(" + a.stride_expr + " * " + sc.ptr_scalar_offset + ")";
              if (a.stride_const && sc_const) {
                a.stride_const = (*a.stride_const) * (*sc_const);
              } else {
                a.stride_const.reset();
              }
              a.base_expr = (a.base_expr == "0") ? "0" : ("(" + a.base_expr + " * " + sc.ptr_scalar_offset + ")");
            }
            return a;
          };
          out = ax;
          if (out.row.has_arange) out.row = scale_1d(out.row);
          if (out.col.has_arange) out.col = scale_1d(out.col);
          if (out.ptr_scalar_offset != "0") {
            out.ptr_scalar_offset = "(" + out.ptr_scalar_offset + " * " + sc.ptr_scalar_offset + ")";
          }
          break;
        }
        if (bop != BinaryOp::Add) break;
        Affine2D l = analyze_2d_ptr(op.operands[0]);
        Affine2D r = analyze_2d_ptr(op.operands[1]);
        if (!l.valid || !r.valid) break;
        if (l.base_param_idx != UINT32_MAX || r.base_param_idx != UINT32_MAX) break;
        out.valid = true;
        out.ptr_scalar_uses_pid2 = l.ptr_scalar_uses_pid2 || r.ptr_scalar_uses_pid2;
        if (l.ptr_scalar_offset == "0") out.ptr_scalar_offset = r.ptr_scalar_offset;
        else if (r.ptr_scalar_offset == "0") out.ptr_scalar_offset = l.ptr_scalar_offset;
        else out.ptr_scalar_offset = "(" + l.ptr_scalar_offset + " + " + r.ptr_scalar_offset + ")";

        if (l.row.valid && r.row.valid) {
          if (l.row.has_arange && r.row.has_arange) {
            out.valid = false;
            break;
          }
          out.row = l.row.has_arange ? l.row : r.row;
        } else {
          out.row = l.row.valid ? l.row : r.row;
        }

        if (l.col.valid && r.col.valid) {
          if (l.col.has_arange && r.col.has_arange) {
            out.valid = false;
            break;
          }
          out.col = l.col.has_arange ? l.col : r.col;
        } else {
          out.col = l.col.valid ? l.col : r.col;
        }
        break;
      }
      default:
        break;
    }

    return cache_2d_[vid] = out;
  }

 private:
  const Kernel& kernel_;
  const ParamMapping& params_;
  ScalarAnalyzer& scalar_;
  const Op* for_op_;
  std::unordered_map<ValueId, Affine1D> cache_1d_;
  std::unordered_map<ValueId, Affine2D> cache_2d_;
};

struct MaskBounds {
  bool valid = true;
  std::optional<std::string> m_upper;
  std::optional<std::string> n_upper;
  std::optional<std::string> k_upper;
  std::optional<std::string> k_lower;
  std::optional<MatrixStructure> tri_struct;
  int32_t tri_diag = 0;
};

enum class AxisTag { Unknown, RowM, ColN, AxisK };

AxisTag classify_axis_expr(const Kernel& kernel, TileAffineAnalyzer& taa, ValueId vid,
                           const Affine1D& m_ref, const Affine1D& n_ref, const Affine1D& k_ref,
                           bool* out_uses_iv = nullptr) {
  ValueId cur = vid;
  while (cur != kInvalidId && kernel.value(cur).type.is_tile() && kernel.value(cur).def_op != kInvalidId) {
    const Op& op = kernel.op(kernel.value(cur).def_op);
    if (op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) {
      cur = op.operands[0];
    } else {
      break;
    }
  }
  Affine1D a = taa.analyze_1d(cur);
  if (!a.valid || !a.has_arange) return AxisTag::Unknown;
  if (out_uses_iv) *out_uses_iv = a.uses_iv;
  if (a.arange_end == m_ref.arange_end && a.unscaled_base_expr == m_ref.unscaled_base_expr &&
      a.uses_iv == m_ref.uses_iv && a.uses_pid0 == m_ref.uses_pid0 && a.uses_pid1 == m_ref.uses_pid1) {
    if (a.base_expr == m_ref.unscaled_base_expr || a.base_expr == m_ref.base_expr) {
      return AxisTag::RowM;
    }
  }
  if (a.arange_end == n_ref.arange_end && a.unscaled_base_expr == n_ref.unscaled_base_expr &&
      a.uses_iv == n_ref.uses_iv && a.uses_pid0 == n_ref.uses_pid0 && a.uses_pid1 == n_ref.uses_pid1) {
    if (a.base_expr == n_ref.unscaled_base_expr || a.base_expr == n_ref.base_expr) {
      return AxisTag::ColN;
    }
  }
  if (a.arange_end == k_ref.arange_end &&
      (a.uses_iv == k_ref.uses_iv || (k_ref.uses_iv && !a.uses_iv && !a.uses_pid0 && !a.uses_pid1))) {
    return AxisTag::AxisK;
  }
  return AxisTag::Unknown;
}

bool extract_scalar_bound(const Kernel& kernel, ScalarAnalyzer& sa, ValueId vid, std::string& expr_out,
                          bool add_one, int64_t expected_iv_sub_scale = 0) {
  ValueId cur = vid;
  while (cur != kInvalidId && kernel.value(cur).type.is_tile() && kernel.value(cur).def_op != kInvalidId) {
    const Op& op = kernel.op(kernel.value(cur).def_op);
    if (op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims || op.kind == OpKind::Splat) {
      cur = op.operands[0];
    } else {
      break;
    }
  }
  ScalarInfo s = sa.analyze(cur);
  if (s.valid && !s.uses_iv) {
    expr_out = add_one ? ("(" + s.expr + " + 1)") : s.expr;
    return true;
  }
  if (expected_iv_sub_scale > 0 && cur != kInvalidId && kernel.value(cur).def_op != kInvalidId) {
    const Op& bop_node = kernel.op(kernel.value(cur).def_op);
    if (bop_node.kind == OpKind::Binary && std::get<BinaryAttr>(bop_node.attr).op == BinaryOp::Sub &&
        bop_node.operands.size() >= 2) {
      ScalarInfo base_s = sa.analyze(bop_node.operands[0]);
      ScalarInfo iv_s = sa.analyze(bop_node.operands[1]);
      if (base_s.valid && !base_s.uses_iv && iv_s.valid && iv_s.is_iv_linear &&
          iv_s.iv_scale == expected_iv_sub_scale && iv_s.iv_offset == 0) {
        expr_out = add_one ? ("(" + base_s.expr + " + 1)") : base_s.expr;
        return true;
      }
    }
  }
  return false;
}

bool parse_mask_bounds(const Kernel& kernel, ScalarAnalyzer& sa, TileAffineAnalyzer& taa, ValueId mask_vid,
                       const Affine1D& m_ref, const Affine1D& n_ref, const Affine1D& k_ref,
                       int64_t k_step_to_elems, MaskBounds& out) {
  if (mask_vid == kInvalidId) return true;
  if (kernel.value(mask_vid).def_op == kInvalidId) return false;
  const Op& op = kernel.op(kernel.value(mask_vid).def_op);

  if (op.kind == OpKind::Broadcast || op.kind == OpKind::ExpandDims) {
    return parse_mask_bounds(kernel, sa, taa, op.operands[0], m_ref, n_ref, k_ref, k_step_to_elems, out);
  }
  if (op.kind == OpKind::Binary && std::holds_alternative<BinaryAttr>(op.attr)) {
    BinaryOp bop = std::get<BinaryAttr>(op.attr).op;
    if (bop == BinaryOp::And) {
      return parse_mask_bounds(kernel, sa, taa, op.operands[0], m_ref, n_ref, k_ref, k_step_to_elems, out) &&
             parse_mask_bounds(kernel, sa, taa, op.operands[1], m_ref, n_ref, k_ref, k_step_to_elems, out);
    }
    if (bop == BinaryOp::CmpLt || bop == BinaryOp::CmpLe || bop == BinaryOp::CmpGt || bop == BinaryOp::CmpGe) {
      bool l_uses_iv = false;
      bool r_uses_iv = false;
      AxisTag ltag = classify_axis_expr(kernel, taa, op.operands[0], m_ref, n_ref, k_ref, &l_uses_iv);
      AxisTag rtag = classify_axis_expr(kernel, taa, op.operands[1], m_ref, n_ref, k_ref, &r_uses_iv);

      if (ltag == AxisTag::RowM && rtag == AxisTag::ColN) {
        if (bop == BinaryOp::CmpLe) {
          out.tri_struct = MatrixStructure::UpperTriangular;
          return true;
        }
        if (bop == BinaryOp::CmpGe) {
          out.tri_struct = MatrixStructure::LowerTriangular;
          return true;
        }
        return false;
      }
      if (ltag == AxisTag::ColN && rtag == AxisTag::RowM) {
        if (bop == BinaryOp::CmpGe) {
          out.tri_struct = MatrixStructure::UpperTriangular;
          return true;
        }
        if (bop == BinaryOp::CmpLe) {
          out.tri_struct = MatrixStructure::LowerTriangular;
          return true;
        }
        return false;
      }

      if (ltag != AxisTag::Unknown && rtag == AxisTag::Unknown) {
        std::string b_expr;
        int64_t iv_sub = (ltag == AxisTag::AxisK && !l_uses_iv) ? k_step_to_elems : 0;
        if (bop == BinaryOp::CmpLt || bop == BinaryOp::CmpLe) {
          if (!extract_scalar_bound(kernel, sa, op.operands[1], b_expr, bop == BinaryOp::CmpLe, iv_sub)) return false;
          if (ltag == AxisTag::RowM) out.m_upper = b_expr;
          else if (ltag == AxisTag::ColN) out.n_upper = b_expr;
          else if (ltag == AxisTag::AxisK) out.k_upper = b_expr;
          return true;
        }
        if (bop == BinaryOp::CmpGe || bop == BinaryOp::CmpGt) {
          if (!extract_scalar_bound(kernel, sa, op.operands[1], b_expr, bop == BinaryOp::CmpGt, iv_sub)) return false;
          if (ltag == AxisTag::AxisK) {
            out.k_lower = b_expr;
            return true;
          }
        }
      } else if (rtag != AxisTag::Unknown && ltag == AxisTag::Unknown) {
        std::string b_expr;
        int64_t iv_sub = (rtag == AxisTag::AxisK && !r_uses_iv) ? k_step_to_elems : 0;
        if (bop == BinaryOp::CmpGt || bop == BinaryOp::CmpGe) {
          if (!extract_scalar_bound(kernel, sa, op.operands[0], b_expr, bop == BinaryOp::CmpGe, iv_sub)) return false;
          if (rtag == AxisTag::RowM) out.m_upper = b_expr;
          else if (rtag == AxisTag::ColN) out.n_upper = b_expr;
          else if (rtag == AxisTag::AxisK) out.k_upper = b_expr;
          return true;
        }
        if (bop == BinaryOp::CmpLe || bop == BinaryOp::CmpLt) {
          if (!extract_scalar_bound(kernel, sa, op.operands[0], b_expr, bop == BinaryOp::CmpLt, iv_sub)) return false;
          if (rtag == AxisTag::AxisK) {
            out.k_lower = b_expr;
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool is_zero_constant_tile(const Kernel& kernel, ValueId vid) {
  if (vid == kInvalidId || kernel.value(vid).def_op == kInvalidId) return false;
  const Op& op = kernel.op(kernel.value(vid).def_op);
  ValueId c_vid = kInvalidId;
  if (op.kind == OpKind::Constant) {
    c_vid = vid;
  } else if (op.kind == OpKind::Splat && !op.operands.empty()) {
    c_vid = op.operands[0];
  } else {
    return false;
  }
  if (c_vid == kInvalidId || kernel.value(c_vid).def_op == kInvalidId) return false;
  const Op& cop = kernel.op(kernel.value(c_vid).def_op);
  if (cop.kind != OpKind::Constant) return false;
  auto f = const_as_f64(std::get<ConstAttr>(cop.attr));
  return f.has_value() && (*f == 0.0);
}

ValueId strip_tile_cast_permute(const Kernel& kernel, ValueId vid, bool& did_permute) {
  did_permute = false;
  ValueId cur = vid;
  for (int depth = 0; depth < 4; ++depth) {
    if (cur == kInvalidId || kernel.value(cur).def_op == kInvalidId) break;
    const Op& op = kernel.op(kernel.value(cur).def_op);
    if (op.kind == OpKind::Cast && !op.operands.empty()) {
      cur = op.operands[0];
      continue;
    }
    if (op.kind == OpKind::Permute && !op.operands.empty()) {
      const auto& perm = std::get<PermuteAttr>(op.attr).perm;
      if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
        did_permute = !did_permute;
        cur = op.operands[0];
        continue;
      }
      return kInvalidId;
    }
    break;
  }
  return cur;
}

struct SgmmSchedule {
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 32;
  uint32_t sg_m = 2;
  uint32_t sg_n = 2;
  uint32_t tm = 4;
  uint32_t tn = 4;
  uint32_t num_sgs = 4;
  uint32_t num_threads = 128;
  uint32_t pad_a = 4;
  uint32_t pad_b = 4;
  uint32_t num_bufs = 1;
  uint32_t swizzle_w = 8;
  bool swizzle_along_n = true;
  bool use_tm1 = true;
  bool interleave_sgmm = true;
  bool use_mma2 = false;
  uint32_t smem_bytes = 0;
};

struct SgmmPlan {
  uint32_t bm = 0;
  uint32_t bn = 0;
  uint32_t bk = 0;

  ScalarDType a_elem_dt = ScalarDType::F32;
  ScalarDType b_elem_dt = ScalarDType::F32;
  ScalarDType c_elem_dt = ScalarDType::F32;
  ScalarDType mma_dt = ScalarDType::F32;
  ScalarDType acc_dt = ScalarDType::F32;

  ParamMapping params;
  std::string a_param;
  std::string b_param;
  std::string c_param;
  std::string a_batch_off = "0";
  std::string b_batch_off = "0";
  std::string c_batch_off = "0";

  std::string stride_am_expr = "1";
  std::string stride_ak_expr = "1";
  std::string stride_bk_expr = "1";
  std::string stride_bn_expr = "1";
  std::string stride_cm_expr = "1";
  std::string stride_cn_expr = "1";
  std::optional<int64_t> stride_am_const;
  std::optional<int64_t> stride_ak_const;
  std::optional<int64_t> stride_bk_const;
  std::optional<int64_t> stride_bn_const;
  std::optional<int64_t> stride_cm_const;
  std::optional<int64_t> stride_cn_const;

  bool simple_2d_pid_grid = false;
  uint32_t pid_m_axis = 0;
  uint32_t pid_n_axis = 1;
  std::string row0_expr = "0";
  std::string col0_expr = "0";

  std::optional<std::string> m_bound;
  std::optional<std::string> n_bound;
  std::optional<std::string> k_bound;
  std::optional<std::string> k_lower_mask;

  std::string k_begin_ir = "0";
  std::string k_end_ir = "0";
  int64_t k_iv_extra_offset = 0;

  uint32_t split_k = 1;
  bool is_atomic = false;

  MatrixStructure structure = MatrixStructure::Dense;
  int32_t diagonal_offset = 0;

  SgmmSchedule sched;
};

std::optional<StructureAttr> resolve_structure(const Kernel& kernel, const Op& dot_op, const DotAttr& dot_attr,
                                               const MaskBounds& mask_b) {
  if (kernel.hints.structure.has_value() && kernel.hints.structure->kind != MatrixStructure::Dense) {
    return kernel.hints.structure;
  }
  if (dot_op.hints.structure.has_value() && dot_op.hints.structure->kind != MatrixStructure::Dense) {
    return dot_op.hints.structure;
  }
  if (dot_attr.out_struct.kind != MatrixStructure::Dense) {
    return dot_attr.out_struct;
  }
  if (!dot_op.results.empty() && kernel.value(dot_op.results[0]).type.is_tile()) {
    const auto& st = kernel.value(dot_op.results[0]).type.as_tile().structure;
    if (st.kind != MatrixStructure::Dense) return st;
  }
  if (mask_b.tri_struct.has_value()) {
    return StructureAttr{*mask_b.tri_struct, mask_b.tri_diag, false};
  }
  return std::nullopt;
}

bool select_sg_grid(uint32_t bm, uint32_t bn, const std::optional<TileSpecAttr>& ts, const TargetSpec& target,
                    uint32_t& sg_m_out, uint32_t& sg_n_out, uint32_t& tm_out, uint32_t& tn_out) {
  uint32_t m8 = bm / 8u;
  uint32_t n8 = bn / 8u;

  if (ts.has_value() && ts->num_simdgroups > 0) {
    uint32_t req = ts->num_simdgroups;
    if (req * 32u <= target.max_threads_per_threadgroup) {
      for (uint32_t max_frags : {16u, 32u}) {
        uint32_t best_diff = UINT32_MAX;
        bool best_tm_ge = false;
        bool found = false;
        for (uint32_t gm = 1; gm <= req; ++gm) {
          if (req % gm != 0) continue;
          uint32_t gn = req / gm;
          if (m8 % gm != 0 || n8 % gn != 0) continue;
          uint32_t tm = m8 / gm;
          uint32_t tn = n8 / gn;
          if (tm * tn > max_frags) continue;
          uint32_t diff = (tm > tn) ? (tm - tn) : (tn - tm);
          bool tm_ge = (tm >= tn);
          if (!found || diff < best_diff || (diff == best_diff && tm_ge && !best_tm_ge)) {
            best_diff = diff;
            best_tm_ge = tm_ge;
            sg_m_out = gm;
            sg_n_out = gn;
            tm_out = tm;
            tn_out = tn;
            found = true;
          }
        }
        if (found) return true;
      }
    }
  }

  sg_m_out = bm / 32u;
  sg_n_out = bn / 32u;
  tm_out = 4u;
  tn_out = 4u;
  return (sg_m_out * sg_n_out * 32u <= target.max_threads_per_threadgroup);
}

bool select_schedule(const Kernel& kernel, const Op& dot_op, const TargetSpec& target, SgmmPlan& plan) {
  std::optional<TileSpecAttr> ts = kernel.hints.tile_spec;
  if (!ts.has_value() && dot_op.hints.tile_spec.has_value()) ts = dot_op.hints.tile_spec;

  SgmmSchedule s;
  s.bm = plan.bm;
  s.bn = plan.bn;
  s.bk = plan.bk;
  if (!select_sg_grid(s.bm, s.bn, ts, target, s.sg_m, s.sg_n, s.tm, s.tn)) {
    return false;
  }
  s.num_sgs = s.sg_m * s.sg_n;
  s.num_threads = s.num_sgs * 32u;

  bool a_can_cm = !plan.stride_am_const.has_value() || *plan.stride_am_const == 1;
  bool a_always_rm = plan.stride_ak_const.has_value() && *plan.stride_ak_const == 1;
  if (a_always_rm) a_can_cm = false;

  bool b_can_cm = !plan.stride_bk_const.has_value() || *plan.stride_bk_const == 1;
  bool b_always_rm = plan.stride_bn_const.has_value() && *plan.stride_bn_const == 1;
  if (b_always_rm) b_can_cm = false;

  bool f32_all = (plan.a_elem_dt == ScalarDType::F32 && plan.b_elem_dt == ScalarDType::F32 &&
                  plan.c_elem_dt == ScalarDType::F32 && plan.acc_dt == ScalarDType::F32);
  bool eligible_mma2 =
      f32_all && !plan.is_atomic && plan.structure == MatrixStructure::Dense && (s.bk == 16u) &&
      ((s.bm == 64u && s.bn == 64u && s.num_sgs == 4u && s.tm == 4u && s.tn == 4u) ||
       (s.bm == 64u && s.bn == 128u && s.num_sgs == 8u && s.tm == 4u && s.tn == 4u) ||
       (s.bm == 128u && s.bn == 64u && s.num_sgs == 8u && s.tm == 4u && s.tn == 4u) ||
       (s.bm == 128u && s.bn == 128u && s.num_sgs == 8u && s.tm == 8u && s.tn == 4u));

  uint32_t mma_bytes = ir::dtype_bytes(plan.mma_dt);
  uint32_t acc_bytes = ir::dtype_bytes(plan.acc_dt);
  uint32_t sg_tile_bytes = (s.tm * 8u) * (s.tn * 8u + 4u) * acc_bytes;

  auto calc_smem_bytes = [&](uint32_t nbuf, uint32_t pad, bool tm1) -> uint32_t {
    uint32_t a_rm = s.bm * (s.bk + pad);
    uint32_t a_cm = (tm1 && a_can_cm) ? s.bk * (s.bm + pad) : a_rm;
    uint32_t b_rm = s.bk * (s.bn + pad);
    uint32_t b_cm = (tm1 && b_can_cm) ? s.bn * (s.bk + pad) : b_rm;
    uint32_t ab = nbuf * (std::max(a_rm, a_cm) + std::max(b_rm, b_cm)) * mma_bytes;
    return std::max(ab, sg_tile_bytes);
  };

  bool want_db = ts.has_value() ? ts->double_buffer
                                : (s.bk <= 16u && calc_smem_bytes(2u, 4u, true) <= 20480u);

  if (eligible_mma2 && !want_db) {
    uint32_t mma2_bytes = (s.bm * (s.bk + 4u) + s.bk * (s.bn + 4u)) * mma_bytes;
    if (target.fits_threadgroup_memory(mma2_bytes)) {
      s.num_bufs = 1u;
      s.pad_a = 4u;
      s.pad_b = 4u;
      s.use_tm1 = false;
      s.use_mma2 = true;
      s.smem_bytes = mma2_bytes;
      if (plan.simple_2d_pid_grid) {
        uint32_t req_sw = ts.has_value() ? ts->swizzle_w : 8u;
        s.swizzle_w = (req_sw <= 1u) ? 1u : req_sw;
        s.swizzle_along_n = true;
      } else {
        s.swizzle_w = 1u;
      }
      s.interleave_sgmm = false;
      plan.sched = s;
      return true;
    }
  }

  bool fitted = false;
  struct Attempt {
    uint32_t nbuf;
    uint32_t pad;
    bool tm1;
  };
  std::vector<Attempt> attempts;
  if (want_db) {
    attempts.push_back({2u, 4u, true});
    attempts.push_back({2u, 4u, false});
    attempts.push_back({2u, 0u, true});
    attempts.push_back({2u, 0u, false});
  }
  attempts.push_back({1u, 4u, true});
  attempts.push_back({1u, 4u, false});
  attempts.push_back({1u, 0u, true});
  attempts.push_back({1u, 0u, false});

  for (const auto& att : attempts) {
    uint32_t bytes = calc_smem_bytes(att.nbuf, att.pad, att.tm1);
    if (target.fits_threadgroup_memory(bytes)) {
      s.num_bufs = att.nbuf;
      s.pad_a = att.pad;
      s.pad_b = att.pad;
      s.use_tm1 = att.tm1;
      s.smem_bytes = bytes;
      fitted = true;
      break;
    }
  }
  if (!fitted) return false;

  if (plan.simple_2d_pid_grid) {
    uint32_t req_sw = ts.has_value() ? ts->swizzle_w : 8u;
    s.swizzle_w = (req_sw <= 1u) ? 1u : req_sw;
    s.swizzle_along_n = (s.bn <= s.bm);
  } else {
    s.swizzle_w = 1u;
  }

  s.interleave_sgmm = (s.num_bufs == 1u);
  plan.sched = s;
  return true;
}

bool match_sgmm_plan(const Kernel& kernel, const TargetSpec& target, SgmmPlan& plan) {
  if (kernel.entry_block == kInvalidId || kernel.entry_block >= kernel.blocks.size()) return false;

  std::vector<BlockId> all_blocks;
  std::unordered_set<BlockId> seen_blocks;
  collect_blocks_dfs(kernel, kernel.entry_block, all_blocks, seen_blocks);

  OpId dot_oid = kInvalidId;
  OpId for_oid = kInvalidId;
  OpId epilogue_oid = kInvalidId;
  uint32_t dot_count = 0;
  uint32_t for_count = 0;
  uint32_t mem_write_count = 0;

  for (BlockId bid : all_blocks) {
    for (OpId oid : kernel.block(bid).ops) {
      const Op& op = kernel.op(oid);
      if (op.erased) continue;
      if (is_unsupported_op(op.kind)) return false;
      if (op.kind == OpKind::Dot) {
        dot_oid = oid;
        ++dot_count;
      } else if (op.kind == OpKind::ForLoop) {
        if (bid != kernel.entry_block) return false;
        for_oid = oid;
        ++for_count;
      } else if (op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) {
        if (bid != kernel.entry_block) return false;
        epilogue_oid = oid;
        ++mem_write_count;
      } else if (op.kind == OpKind::If) {
        return false;
      }
    }
  }

  if (dot_count != 1u || for_count != 1u || mem_write_count != 1u) return false;

  const Op& for_op = kernel.op(for_oid);
  const Op& dot_op = kernel.op(dot_oid);
  const Op& epi_op = kernel.op(epilogue_oid);
  if (for_op.regions.size() != 1u || dot_op.parent_block != for_op.regions[0]) return false;

  const Block& body = kernel.block(for_op.regions[0]);
  if (body.args.empty() || for_op.operands.size() < 4u) return false;

  OpId yield_oid = kInvalidId;
  for (OpId oid : body.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    if (op.kind == OpKind::Load) {
      continue;
    }
    if (op.kind == OpKind::Yield) {
      yield_oid = oid;
    }
  }
  if (yield_oid == kInvalidId) return false;
  const Op& yield_op = kernel.op(yield_oid);

  ValueId dot_res = dot_op.results[0];
  std::optional<size_t> acc_iter_idx;
  const auto& dot_attr = std::holds_alternative<DotAttr>(dot_op.attr) ? std::get<DotAttr>(dot_op.attr) : DotAttr{};
  ValueId dot_acc_in = (dot_op.operands.size() >= 3 && dot_op.operands[2] != kInvalidId) ? dot_op.operands[2] : kInvalidId;

  for (size_t i = 0; i < yield_op.operands.size() && (1 + i) < body.args.size(); ++i) {
    ValueId arg_v = body.args[1 + i];
    ValueId y_v = yield_op.operands[i];
    if (dot_acc_in != kInvalidId) {
      if (arg_v == dot_acc_in && y_v == dot_res) {
        acc_iter_idx = i;
      }
    } else if (kernel.value(y_v).def_op != kInvalidId) {
      const Op& yop = kernel.op(kernel.value(y_v).def_op);
      if (yop.kind == OpKind::Binary && std::get<BinaryAttr>(yop.attr).op == BinaryOp::Add && yop.operands.size() >= 2) {
        if ((yop.operands[0] == arg_v && yop.operands[1] == dot_res) ||
            (yop.operands[1] == arg_v && yop.operands[0] == dot_res)) {
          acc_iter_idx = i;
        }
      }
    }
  }
  if (!acc_iter_idx.has_value()) return false;
  if (!is_zero_constant_tile(kernel, for_op.operands[3 + *acc_iter_idx])) return false;

  for (size_t i = 0; i < yield_op.operands.size() && (1 + i) < body.args.size(); ++i) {
    if (i == *acc_iter_idx) continue;
    ValueId y_v = yield_op.operands[i];
    if (kernel.value(y_v).def_op == kInvalidId || kernel.op(kernel.value(y_v).def_op).kind != OpKind::PtrAdd) {
      return false;
    }
  }

  ValueId acc_out = for_op.results[*acc_iter_idx];
  if (epi_op.operands.size() < 2) return false;
  ValueId epi_val = epi_op.operands[1];
  ValueId epi_mask = (epi_op.operands.size() >= 3 && epi_op.operands[2] != kInvalidId) ? epi_op.operands[2] : kInvalidId;
  std::optional<MatrixStructure> epi_select_tri;
  ValueId cur_val = epi_val;
  for (int depth = 0; depth < 3; ++depth) {
    if (cur_val == acc_out) break;
    if (kernel.value(cur_val).def_op == kInvalidId) return false;
    const Op& vop = kernel.op(kernel.value(cur_val).def_op);
    if (vop.kind == OpKind::Cast && !vop.operands.empty()) {
      cur_val = vop.operands[0];
      continue;
    }
    if (vop.kind == OpKind::Select && vop.operands.size() >= 3 && is_zero_constant_tile(kernel, vop.operands[2])) {
      cur_val = vop.operands[1];
      epi_select_tri = MatrixStructure::Dense;
      continue;
    }
    return false;
  }
  if (cur_val != acc_out) return false;

  bool permute_a = false;
  bool permute_b = false;
  ValueId load_a_res = strip_tile_cast_permute(kernel, dot_op.operands[0], permute_a);
  ValueId load_b_res = strip_tile_cast_permute(kernel, dot_op.operands[1], permute_b);
  if (load_a_res == kInvalidId || load_b_res == kInvalidId) return false;
  if (kernel.value(load_a_res).def_op == kInvalidId || kernel.value(load_b_res).def_op == kInvalidId) return false;

  const Op& load_a = kernel.op(kernel.value(load_a_res).def_op);
  const Op& load_b = kernel.op(kernel.value(load_b_res).def_op);
  if (load_a.kind != OpKind::Load || load_b.kind != OpKind::Load) return false;
  if (load_a.parent_block != for_op.regions[0] || load_b.parent_block != for_op.regions[0]) return false;

  std::unordered_set<ValueId> allowed_tile_ops;
  allowed_tile_ops.insert(load_a.results[0]);
  allowed_tile_ops.insert(load_b.results[0]);
  allowed_tile_ops.insert(dot_op.operands[0]);
  allowed_tile_ops.insert(dot_op.operands[1]);
  allowed_tile_ops.insert(dot_res);
  allowed_tile_ops.insert(yield_op.operands[*acc_iter_idx]);
  for (OpId oid : body.ops) {
    const Op& op = kernel.op(oid);
    if (op.erased) continue;
    if (op.kind == OpKind::Load && oid != kernel.value(load_a_res).def_op && oid != kernel.value(load_b_res).def_op) {
      return false;
    }
    if (op.kind == OpKind::Store || op.kind == OpKind::AtomicAdd) return false;
    if (op.kind == OpKind::Unary && !op.results.empty() && kernel.value(op.results[0]).type.is_tile()) {
      return false;
    }
    if (op.kind == OpKind::Binary && !op.results.empty() && kernel.value(op.results[0]).type.is_tile()) {
      const Type& rty = kernel.value(op.results[0]).type;
      if (is_float_dtype(scalar_dtype_of(rty)) && oid != kernel.value(yield_op.operands[*acc_iter_idx]).def_op) {
        return false;
      }
    }
    if (op.kind == OpKind::Select && !op.results.empty() && kernel.value(op.results[0]).type.is_tile() &&
        is_float_dtype(scalar_dtype_of(kernel.value(op.results[0]).type))) {
      return false;
    }
  }

  const Type& dot_lty = kernel.value(dot_op.operands[0]).type;
  const Type& dot_rty = kernel.value(dot_op.operands[1]).type;
  const Type& dot_oty = kernel.value(dot_res).type;
  if (!dot_lty.is_tile() || !dot_rty.is_tile() || !dot_oty.is_tile()) return false;
  if (dot_lty.as_tile().shape.size() != 2 || dot_rty.as_tile().shape.size() != 2 ||
      dot_oty.as_tile().shape.size() != 2) {
    return false;
  }

  const auto& lsh = dot_lty.as_tile().shape;
  const auto& rsh = dot_rty.as_tile().shape;
  const auto& osh = dot_oty.as_tile().shape;
  bool dot_ta = dot_attr.transpose_a;
  bool dot_tb = dot_attr.transpose_b;
  int64_t bm = dot_ta ? lsh[1] : lsh[0];
  int64_t bka = dot_ta ? lsh[0] : lsh[1];
  int64_t bkb = dot_tb ? rsh[1] : rsh[0];
  int64_t bn = dot_tb ? rsh[0] : rsh[1];
  if (bka != bkb || bm != osh[0] || bn != osh[1]) return false;
  if (bm < 32 || (bm % 32) != 0 || bn < 32 || (bn % 32) != 0 || bka < 8 || (bka % 8) != 0) return false;

  plan.bm = static_cast<uint32_t>(bm);
  plan.bn = static_cast<uint32_t>(bn);
  plan.bk = static_cast<uint32_t>(bka);

  plan.a_elem_dt = scalar_dtype_of(kernel.value(load_a.operands[0]).type);
  plan.b_elem_dt = scalar_dtype_of(kernel.value(load_b.operands[0]).type);
  plan.c_elem_dt = scalar_dtype_of(kernel.value(epi_op.operands[0]).type);
  plan.acc_dt = scalar_dtype_of(dot_oty);
  if (!is_float_dtype(plan.a_elem_dt) || !is_float_dtype(plan.b_elem_dt) || !is_float_dtype(plan.c_elem_dt) ||
      !is_float_dtype(plan.acc_dt)) {
    return false;
  }
  if (epi_op.kind == OpKind::AtomicAdd && plan.c_elem_dt != ScalarDType::F32) return false;
  plan.mma_dt = (plan.a_elem_dt == ScalarDType::F16 && plan.b_elem_dt == ScalarDType::F16)
                    ? ScalarDType::F16
                    : ScalarDType::F32;

  if (load_a.operands.size() >= 3 && load_a.operands[2] != kInvalidId && !is_zero_constant_tile(kernel, load_a.operands[2])) {
    return false;
  }
  if (load_b.operands.size() >= 3 && load_b.operands[2] != kInvalidId && !is_zero_constant_tile(kernel, load_b.operands[2])) {
    return false;
  }

  if (!build_param_mapping(kernel, all_blocks, plan.params)) return false;

  ScalarAnalyzer sa(kernel, plan.params, body.args[0]);
  ScalarInfo lo_s = sa.analyze(for_op.operands[0]);
  ScalarInfo hi_s = sa.analyze(for_op.operands[1]);
  ScalarInfo step_s = sa.analyze(for_op.operands[2]);
  if (!lo_s.valid || !hi_s.valid || !step_s.valid || !step_s.const_i64.has_value() || *step_s.const_i64 <= 0) {
    return false;
  }

  TileAffineAnalyzer taa(kernel, plan.params, sa, &for_op);
  Affine2D a_aff = taa.analyze_2d_ptr(load_a.operands[0]);
  Affine2D b_aff = taa.analyze_2d_ptr(load_b.operands[0]);
  Affine2D c_aff = taa.analyze_2d_ptr(epi_op.operands[0]);
  if (!a_aff.valid || !b_aff.valid || !c_aff.valid) return false;
  if (a_aff.base_param_idx == UINT32_MAX || b_aff.base_param_idx == UINT32_MAX ||
      c_aff.base_param_idx == UINT32_MAX) {
    return false;
  }
  if (a_aff.base_param_idx == c_aff.base_param_idx || b_aff.base_param_idx == c_aff.base_param_idx) {
    return false;
  }

  plan.a_param = plan.params.param_msl_name.at(a_aff.base_param_idx);
  plan.b_param = plan.params.param_msl_name.at(b_aff.base_param_idx);
  plan.c_param = plan.params.param_msl_name.at(c_aff.base_param_idx);
  plan.a_batch_off = a_aff.ptr_scalar_offset;
  plan.b_batch_off = b_aff.ptr_scalar_offset;
  plan.c_batch_off = c_aff.ptr_scalar_offset;

  bool swap_a = (dot_ta != permute_a);
  bool swap_b = (dot_tb != permute_b);
  const Affine1D& a_m = swap_a ? a_aff.col : a_aff.row;
  const Affine1D& a_k = swap_a ? a_aff.row : a_aff.col;
  const Affine1D& b_k = swap_b ? b_aff.col : b_aff.row;
  const Affine1D& b_n = swap_b ? b_aff.row : b_aff.col;
  const Affine1D& c_m = c_aff.row;
  const Affine1D& c_n = c_aff.col;

  if (!a_m.valid || !a_k.valid || !b_k.valid || !b_n.valid || !c_m.valid || !c_n.valid) return false;
  if (!a_m.has_arange || !a_k.has_arange || !b_k.has_arange || !b_n.has_arange || !c_m.has_arange || !c_n.has_arange) {
    return false;
  }
  if (a_m.arange_start != 0 || a_k.arange_start != 0 || b_k.arange_start != 0 || b_n.arange_start != 0 ||
      c_m.arange_start != 0 || c_n.arange_start != 0) {
    return false;
  }
  if (a_m.arange_end != bm || c_m.arange_end != bm || b_n.arange_end != bn || c_n.arange_end != bn ||
      a_k.arange_end != bka || b_k.arange_end != bka) {
    return false;
  }
  if (a_m.unscaled_base_expr != c_m.unscaled_base_expr || b_n.unscaled_base_expr != c_n.unscaled_base_expr) {
    return false;
  }
  if (!a_k.uses_iv || !b_k.uses_iv) return false;

  int64_t k_step = *step_s.const_i64;
  if (k_step == bka && a_k.is_iv_linear && a_k.iv_scale == 1 && b_k.is_iv_linear && b_k.iv_scale == 1 &&
      a_k.iv_offset == b_k.iv_offset) {
    plan.k_iv_extra_offset = a_k.iv_offset;
    plan.k_begin_ir = (a_k.iv_offset == 0) ? lo_s.expr : ("(" + lo_s.expr + " + " + std::to_string(a_k.iv_offset) + ")");
    plan.k_end_ir = (a_k.iv_offset == 0) ? hi_s.expr : ("(" + hi_s.expr + " + " + std::to_string(a_k.iv_offset) + ")");
  } else if (k_step == 1 && a_k.is_iv_linear && a_k.iv_scale == bka && b_k.is_iv_linear && b_k.iv_scale == bka &&
             a_k.iv_offset == b_k.iv_offset) {
    plan.k_iv_extra_offset = a_k.iv_offset;
    plan.k_begin_ir = "(" + lo_s.expr + " * " + std::to_string(bka) +
                      (a_k.iv_offset != 0 ? (" + " + std::to_string(a_k.iv_offset)) : "") + ")";
    plan.k_end_ir = "(" + hi_s.expr + " * " + std::to_string(bka) +
                    (a_k.iv_offset != 0 ? (" + " + std::to_string(a_k.iv_offset)) : "") + ")";
  } else {
    return false;
  }

  plan.stride_am_expr = a_m.stride_expr;
  plan.stride_am_const = a_m.stride_const;
  plan.stride_ak_expr = a_k.stride_expr;
  plan.stride_ak_const = a_k.stride_const;
  plan.stride_bk_expr = b_k.stride_expr;
  plan.stride_bk_const = b_k.stride_const;
  plan.stride_bn_expr = b_n.stride_expr;
  plan.stride_bn_const = b_n.stride_const;
  plan.stride_cm_expr = c_m.stride_expr;
  plan.stride_cm_const = c_m.stride_const;
  plan.stride_cn_expr = c_n.stride_expr;
  plan.stride_cn_const = c_n.stride_const;

  plan.row0_expr = c_m.has_outer_scale ? c_m.unscaled_base_expr : c_m.base_expr;
  plan.col0_expr = c_n.has_outer_scale ? c_n.unscaled_base_expr : c_n.base_expr;

  if (c_m.linear_pid_axis && c_n.linear_pid_axis && *c_m.linear_pid_axis != *c_n.linear_pid_axis &&
      (*c_m.linear_pid_axis <= 1u) && (*c_n.linear_pid_axis <= 1u) && c_m.linear_pid_scale == bm &&
      c_n.linear_pid_scale == bn && c_m.linear_pid_offset == 0 && c_n.linear_pid_offset == 0) {
    plan.simple_2d_pid_grid = true;
    plan.pid_m_axis = *c_m.linear_pid_axis;
    plan.pid_n_axis = *c_n.linear_pid_axis;
  }

  MaskBounds mb_a, mb_b, mb_c;
  ValueId a_mask = (load_a.operands.size() >= 2) ? load_a.operands[1] : kInvalidId;
  ValueId b_mask = (load_b.operands.size() >= 2) ? load_b.operands[1] : kInvalidId;
  int64_t k_step_to_elems = (k_step == 1) ? bka : 1;
  if (!parse_mask_bounds(kernel, sa, taa, a_mask, a_m, b_n, a_k, k_step_to_elems, mb_a)) return false;
  if (!parse_mask_bounds(kernel, sa, taa, b_mask, a_m, b_n, b_k, k_step_to_elems, mb_b)) return false;
  if (!parse_mask_bounds(kernel, sa, taa, epi_mask, c_m, c_n, a_k, k_step_to_elems, mb_c)) return false;

  if (epi_select_tri.has_value()) {
    const Op& sel_op = kernel.op(kernel.value(epi_val).def_op);
    if (sel_op.kind == OpKind::Select &&
        !parse_mask_bounds(kernel, sa, taa, sel_op.operands[0], c_m, c_n, a_k, k_step_to_elems, mb_c)) {
      return false;
    }
  }

  if (mb_a.m_upper && mb_c.m_upper && *mb_a.m_upper != *mb_c.m_upper) return false;
  if (mb_b.n_upper && mb_c.n_upper && *mb_b.n_upper != *mb_c.n_upper) return false;
  if (mb_a.k_upper && mb_b.k_upper && *mb_a.k_upper != *mb_b.k_upper) return false;

  plan.m_bound = mb_c.m_upper ? mb_c.m_upper : mb_a.m_upper;
  plan.n_bound = mb_c.n_upper ? mb_c.n_upper : mb_b.n_upper;
  plan.k_bound = mb_a.k_upper ? mb_a.k_upper : mb_b.k_upper;
  plan.k_lower_mask = mb_a.k_lower ? mb_a.k_lower : mb_b.k_lower;

  if (auto st = resolve_structure(kernel, dot_op, dot_attr, mb_c)) {
    if (st->kind == MatrixStructure::Diagonal || st->kind == MatrixStructure::CausalMask) {
      return false;
    }
    if (st->kind == MatrixStructure::UpperTriangular || st->kind == MatrixStructure::LowerTriangular) {
      plan.structure = st->kind;
      plan.diagonal_offset = st->diagonal_offset;
    }
  }

  bool ir_uses_pid2_in_k = lo_s.uses_pid2 || hi_s.uses_pid2;
  bool ptr_uses_pid2 =
      a_aff.ptr_scalar_uses_pid2 || b_aff.ptr_scalar_uses_pid2 || c_aff.ptr_scalar_uses_pid2;

  std::optional<TileSpecAttr> ts = kernel.hints.tile_spec;
  if (!ts.has_value() && dot_op.hints.tile_spec.has_value()) ts = dot_op.hints.tile_spec;
  if (!ir_uses_pid2_in_k && !ptr_uses_pid2 && ts.has_value() && ts->split_k > 1u) {
    plan.split_k = ts->split_k;
  }
  plan.is_atomic = (epi_op.kind == OpKind::AtomicAdd) || (plan.split_k > 1u);

  if (!select_schedule(kernel, dot_op, target, plan)) return false;
  return true;
}

std::string msl_packed_vec4_type(ScalarDType dt) {
  return (dt == ScalarDType::F16) ? "packed_half4" : "packed_float4";
}

std::string msl_sgmm_type(ScalarDType dt) {
  return (dt == ScalarDType::F16) ? "simdgroup_half8x8" : "simdgroup_float8x8";
}

void emit_tile_compute(std::ostringstream& os, const SgmmPlan& plan, const std::string& sa_ptr,
                       const std::string& sb_ptr, const std::string& a_ld_expr, const std::string& b_ld_expr,
                       bool a_cm, bool b_cm, const std::string& ind) {
  const auto& s = plan.sched;
  std::string mma_t = msl_sgmm_type(plan.mma_dt);
  for (uint32_t ks = 0; ks < s.bk; ks += 8u) {
    os << ind << "{\n"
       << ind << "  " << mma_t << " m1_fa[" << s.tm << "];\n"
       << ind << "  " << mma_t << " m1_fb[" << s.tn << "];\n";
    if (s.interleave_sgmm) {
      if (a_cm) {
        os << ind << "  simdgroup_load(m1_fa[0], " << sa_ptr << " + " << ks << "u * " << a_ld_expr
           << " + m1_sg_row, " << a_ld_expr << ", ulong2(0, 0), true);\n";
      } else {
        os << ind << "  simdgroup_load(m1_fa[0], " << sa_ptr << " + m1_sg_row * " << a_ld_expr << " + " << ks
           << "u, " << a_ld_expr << ");\n";
      }
      for (uint32_t j = 0; j < s.tn; ++j) {
        if (b_cm) {
          os << ind << "  simdgroup_load(m1_fb[" << j << "], " << sb_ptr << " + (m1_sg_col + " << (j * 8u)
             << "u) * " << b_ld_expr << " + " << ks << "u, " << b_ld_expr << ", ulong2(0, 0), true);\n";
        } else {
          os << ind << "  simdgroup_load(m1_fb[" << j << "], " << sb_ptr << " + " << ks << "u * " << b_ld_expr
             << " + (m1_sg_col + " << (j * 8u) << "u), " << b_ld_expr << ");\n";
        }
      }
      for (uint32_t i = 0; i < s.tm; ++i) {
        if (i + 1u < s.tm) {
          if (a_cm) {
            os << ind << "  simdgroup_load(m1_fa[" << (i + 1u) << "], " << sa_ptr << " + " << ks << "u * "
               << a_ld_expr << " + (m1_sg_row + " << ((i + 1u) * 8u) << "u), " << a_ld_expr
               << ", ulong2(0, 0), true);\n";
          } else {
            os << ind << "  simdgroup_load(m1_fa[" << (i + 1u) << "], " << sa_ptr << " + (m1_sg_row + "
               << ((i + 1u) * 8u) << "u) * " << a_ld_expr << " + " << ks << "u, " << a_ld_expr << ");\n";
          }
        }
        for (uint32_t j = 0; j < s.tn; ++j) {
          os << ind << "  simdgroup_multiply_accumulate(m1_acc[" << i << "][" << j << "], m1_fa[" << i
             << "], m1_fb[" << j << "], m1_acc[" << i << "][" << j << "]);\n";
        }
      }
    } else {
      for (uint32_t i = 0; i < s.tm; ++i) {
        if (a_cm) {
          os << ind << "  simdgroup_load(m1_fa[" << i << "], " << sa_ptr << " + " << ks << "u * " << a_ld_expr
             << " + (m1_sg_row + " << (i * 8u) << "u), " << a_ld_expr << ", ulong2(0, 0), true);\n";
        } else {
          os << ind << "  simdgroup_load(m1_fa[" << i << "], " << sa_ptr << " + (m1_sg_row + " << (i * 8u)
             << "u) * " << a_ld_expr << " + " << ks << "u, " << a_ld_expr << ");\n";
        }
      }
      for (uint32_t j = 0; j < s.tn; ++j) {
        if (b_cm) {
          os << ind << "  simdgroup_load(m1_fb[" << j << "], " << sb_ptr << " + (m1_sg_col + " << (j * 8u)
             << "u) * " << b_ld_expr << " + " << ks << "u, " << b_ld_expr << ", ulong2(0, 0), true);\n";
        } else {
          os << ind << "  simdgroup_load(m1_fb[" << j << "], " << sb_ptr << " + " << ks << "u * " << b_ld_expr
             << " + (m1_sg_col + " << (j * 8u) << "u), " << b_ld_expr << ");\n";
        }
      }
      for (uint32_t i = 0; i < s.tm; ++i) {
        for (uint32_t j = 0; j < s.tn; ++j) {
          os << ind << "  simdgroup_multiply_accumulate(m1_acc[" << i << "][" << j << "], m1_fa[" << i
             << "], m1_fb[" << j << "], m1_acc[" << i << "][" << j << "]);\n";
        }
      }
    }
    os << ind << "}\n";
  }
}

void emit_vec_load_a(std::ostringstream& os, const SgmmPlan& plan, const std::string& sa_ptr,
                     const std::string& k_expr, const std::string& a_ld_expr, bool a_cm, bool use_tm1,
                     const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bm * s.bk) / 4u;
  uint32_t vpr = (!a_cm ? s.bk : s.bm) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string mma_v4 = msl_vec4_type(plan.mma_dt);
  std::string src_pv4 = msl_packed_vec4_type(plan.a_elem_dt);

  for (uint32_t p = 0; p < per; ++p) {
    os << ind << "{\n"
       << ind << "  uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
    if ((p + 1u) * s.num_threads > nvec) os << ind << "  if (m1_vi < " << nvec << "u) {\n";
    os << ind << "  uint m1_r = m1_vi / " << vpr << "u;\n"
       << ind << "  uint m1_c4 = (m1_vi % " << vpr << "u) * 4u;\n";
    if (!a_cm) {
      os << ind << "  " << mma_v4 << " m1_va = " << mma_v4 << "(*(const device " << src_pv4
         << "*)(m1_A + ulong(m1_row0 + m1_r) * ulong(m1_sam) + ulong(" << k_expr << " + m1_c4)));\n"
         << ind << "  *(threadgroup " << mma_v4 << "*)(" << sa_ptr << " + m1_r * " << a_ld_expr
         << " + m1_c4) = m1_va;\n";
    } else {
      os << ind << "  " << mma_v4 << " m1_va = " << mma_v4 << "(*(const device " << src_pv4
         << "*)(m1_A + ulong(" << k_expr << " + m1_r) * ulong(m1_sak) + ulong(m1_row0 + m1_c4)));\n";
      if (use_tm1) {
        os << ind << "  *(threadgroup " << mma_v4 << "*)(" << sa_ptr << " + m1_r * " << a_ld_expr
           << " + m1_c4) = m1_va;\n";
      } else {
        os << ind << "  " << sa_ptr << "[(m1_c4 + 0u) * " << a_ld_expr << " + m1_r] = " << mma_t << "(m1_va.x);\n"
           << ind << "  " << sa_ptr << "[(m1_c4 + 1u) * " << a_ld_expr << " + m1_r] = " << mma_t << "(m1_va.y);\n"
           << ind << "  " << sa_ptr << "[(m1_c4 + 2u) * " << a_ld_expr << " + m1_r] = " << mma_t << "(m1_va.z);\n"
           << ind << "  " << sa_ptr << "[(m1_c4 + 3u) * " << a_ld_expr << " + m1_r] = " << mma_t << "(m1_va.w);\n";
      }
    }
    if ((p + 1u) * s.num_threads > nvec) os << ind << "  }\n";
    os << ind << "}\n";
  }
}

void emit_vec_load_b(std::ostringstream& os, const SgmmPlan& plan, const std::string& sb_ptr,
                     const std::string& k_expr, const std::string& b_ld_expr, bool b_cm, bool use_tm1,
                     const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bk * s.bn) / 4u;
  uint32_t vpr = (!b_cm ? s.bn : s.bk) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string mma_v4 = msl_vec4_type(plan.mma_dt);
  std::string src_pv4 = msl_packed_vec4_type(plan.b_elem_dt);

  for (uint32_t p = 0; p < per; ++p) {
    os << ind << "{\n"
       << ind << "  uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
    if ((p + 1u) * s.num_threads > nvec) os << ind << "  if (m1_vi < " << nvec << "u) {\n";
    os << ind << "  uint m1_r = m1_vi / " << vpr << "u;\n"
       << ind << "  uint m1_c4 = (m1_vi % " << vpr << "u) * 4u;\n";
    if (!b_cm) {
      os << ind << "  " << mma_v4 << " m1_vb = " << mma_v4 << "(*(const device " << src_pv4
         << "*)(m1_B + ulong(" << k_expr << " + m1_r) * ulong(m1_sbk) + ulong(m1_col0 + m1_c4)));\n"
         << ind << "  *(threadgroup " << mma_v4 << "*)(" << sb_ptr << " + m1_r * " << b_ld_expr
         << " + m1_c4) = m1_vb;\n";
    } else {
      os << ind << "  " << mma_v4 << " m1_vb = " << mma_v4 << "(*(const device " << src_pv4
         << "*)(m1_B + ulong(m1_col0 + m1_r) * ulong(m1_sbn) + ulong(" << k_expr << " + m1_c4)));\n";
      if (use_tm1) {
        os << ind << "  *(threadgroup " << mma_v4 << "*)(" << sb_ptr << " + m1_r * " << b_ld_expr
           << " + m1_c4) = m1_vb;\n";
      } else {
        os << ind << "  " << sb_ptr << "[(m1_c4 + 0u) * " << b_ld_expr << " + m1_r] = " << mma_t << "(m1_vb.x);\n"
           << ind << "  " << sb_ptr << "[(m1_c4 + 1u) * " << b_ld_expr << " + m1_r] = " << mma_t << "(m1_vb.y);\n"
           << ind << "  " << sb_ptr << "[(m1_c4 + 2u) * " << b_ld_expr << " + m1_r] = " << mma_t << "(m1_vb.z);\n"
           << ind << "  " << sb_ptr << "[(m1_c4 + 3u) * " << b_ld_expr << " + m1_r] = " << mma_t << "(m1_vb.w);\n";
      }
    }
    if ((p + 1u) * s.num_threads > nvec) os << ind << "  }\n";
    os << ind << "}\n";
  }
}

void emit_safe_load_a(std::ostringstream& os, const SgmmPlan& plan, const std::string& sa_ptr,
                      const std::string& k_expr, const std::string& a_ld_expr, bool store_cm,
                      const std::string& ind) {
  const auto& s = plan.sched;
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string m_check = plan.m_bound.has_value() ? "(m1_gr < m1_M)" : "true";
  std::string k_check = plan.k_lower_mask.has_value()
                            ? "(m1_gk >= m1_k_mask_lo && m1_gk < m1_k_end)"
                            : "(m1_gk < m1_k_end)";
  os << ind << "for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bm * s.bk) << "u; m1_idx += " << s.num_threads
     << "u) {\n";
  if (!store_cm) {
    os << ind << "  uint m1_r = m1_idx / " << s.bk << "u;\n"
       << ind << "  uint m1_c = m1_idx % " << s.bk << "u;\n"
       << ind << "  uint m1_gr = m1_row0 + m1_r;\n"
       << ind << "  uint m1_gk = " << k_expr << " + m1_c;\n"
       << ind << "  " << mma_t << " m1_av = (" << m_check << " && " << k_check << ") ? " << mma_t
       << "(m1_A[long(m1_gr) * long(m1_sam) + long(m1_gk) * long(m1_sak)]) : " << mma_t << "(0.0f);\n"
       << ind << "  " << sa_ptr << "[m1_r * " << a_ld_expr << " + m1_c] = m1_av;\n";
  } else {
    os << ind << "  uint m1_kr = m1_idx / " << s.bm << "u;\n"
       << ind << "  uint m1_mc = m1_idx % " << s.bm << "u;\n"
       << ind << "  uint m1_gk = " << k_expr << " + m1_kr;\n"
       << ind << "  uint m1_gr = m1_row0 + m1_mc;\n"
       << ind << "  " << mma_t << " m1_av = (" << m_check << " && " << k_check << ") ? " << mma_t
       << "(m1_A[long(m1_gr) * long(m1_sam) + long(m1_gk) * long(m1_sak)]) : " << mma_t << "(0.0f);\n"
       << ind << "  " << sa_ptr << "[m1_kr * " << a_ld_expr << " + m1_mc] = m1_av;\n";
  }
  os << ind << "}\n";
}

void emit_safe_load_b(std::ostringstream& os, const SgmmPlan& plan, const std::string& sb_ptr,
                      const std::string& k_expr, const std::string& b_ld_expr, bool store_cm,
                      const std::string& ind) {
  const auto& s = plan.sched;
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string n_check = plan.n_bound.has_value() ? "(m1_gc < m1_N)" : "true";
  std::string k_check = plan.k_lower_mask.has_value()
                            ? "(m1_gk >= m1_k_mask_lo && m1_gk < m1_k_end)"
                            : "(m1_gk < m1_k_end)";
  os << ind << "for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bk * s.bn) << "u; m1_idx += " << s.num_threads
     << "u) {\n";
  if (!store_cm) {
    os << ind << "  uint m1_r = m1_idx / " << s.bn << "u;\n"
       << ind << "  uint m1_c = m1_idx % " << s.bn << "u;\n"
       << ind << "  uint m1_gk = " << k_expr << " + m1_r;\n"
       << ind << "  uint m1_gc = m1_col0 + m1_c;\n"
       << ind << "  " << mma_t << " m1_bv = (" << k_check << " && " << n_check << ") ? " << mma_t
       << "(m1_B[long(m1_gk) * long(m1_sbk) + long(m1_gc) * long(m1_sbn)]) : " << mma_t << "(0.0f);\n"
       << ind << "  " << sb_ptr << "[m1_r * " << b_ld_expr << " + m1_c] = m1_bv;\n";
  } else {
    os << ind << "  uint m1_nr = m1_idx / " << s.bk << "u;\n"
       << ind << "  uint m1_kc = m1_idx % " << s.bk << "u;\n"
       << ind << "  uint m1_gc = m1_col0 + m1_nr;\n"
       << ind << "  uint m1_gk = " << k_expr << " + m1_kc;\n"
       << ind << "  " << mma_t << " m1_bv = (" << k_check << " && " << n_check << ") ? " << mma_t
       << "(m1_B[long(m1_gk) * long(m1_sbk) + long(m1_gc) * long(m1_sbn)]) : " << mma_t << "(0.0f);\n"
       << ind << "  " << sb_ptr << "[m1_nr * " << b_ld_expr << " + m1_kc] = m1_bv;\n";
  }
  os << ind << "}\n";
}

void emit_fast_k_loop(std::ostringstream& os, const SgmmPlan& plan, bool a_cm, bool b_cm, const std::string& ind) {
  const auto& s = plan.sched;
  bool store_a_cm = a_cm && s.use_tm1;
  bool store_b_cm = b_cm && s.use_tm1;
  uint32_t a_ld = (store_a_cm ? s.bm : s.bk) + s.pad_a;
  uint32_t b_ld = (store_b_cm ? s.bk : s.bn) + s.pad_b;
  uint32_t a_buf_elems = (store_a_cm ? s.bk : s.bm) * a_ld;
  uint32_t b_buf_elems = (store_b_cm ? s.bn : s.bk) * b_ld;
  std::string a_ld_s = std::to_string(a_ld) + "u";
  std::string b_ld_s = std::to_string(b_ld) + "u";
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string mma_v4 = msl_vec4_type(plan.mma_dt);
  std::string a_pv4 = msl_packed_vec4_type(plan.a_elem_dt);
  std::string b_pv4 = msl_packed_vec4_type(plan.b_elem_dt);

  os << ind << "uint m1_k0 = m1_k_loop_begin;\n"
     << ind << "uint m1_k_full_end = m1_k_loop_begin + ((m1_k_loop_end - m1_k_loop_begin) / " << s.bk << "u) * "
     << s.bk << "u;\n";

  if (s.num_bufs == 1u) {
    os << ind << "for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_vec_load_a(os, plan, "m1_tg_a", "m1_k0", a_ld_s, a_cm, s.use_tm1, ind + "  ");
    emit_vec_load_b(os, plan, "m1_tg_b", "m1_k0", b_ld_s, b_cm, s.use_tm1, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_tile_compute(os, plan, "m1_tg_a", "m1_tg_b", a_ld_s, b_ld_s, store_a_cm, store_b_cm, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "}\n";
  } else {
    uint32_t a_nvec = (s.bm * s.bk) / 4u;
    uint32_t a_vpr = (!a_cm ? s.bk : s.bm) / 4u;
    uint32_t a_per = (a_nvec + s.num_threads - 1u) / s.num_threads;
    uint32_t b_nvec = (s.bk * s.bn) / 4u;
    uint32_t b_vpr = (!b_cm ? s.bn : s.bk) / 4u;
    uint32_t b_per = (b_nvec + s.num_threads - 1u) / s.num_threads;

    os << ind << "if (m1_k0 < m1_k_full_end) {\n"
       << ind << "  uint m1_buf = 0u;\n";
    emit_vec_load_a(os, plan, "m1_tg_a", "m1_k0", a_ld_s, a_cm, s.use_tm1, ind + "  ");
    emit_vec_load_b(os, plan, "m1_tg_b", "m1_k0", b_ld_s, b_cm, s.use_tm1, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "  for (; m1_k0 + " << s.bk << "u < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n"
       << ind << "    uint m1_kn = m1_k0 + " << s.bk << "u;\n"
       << ind << "    uint m1_nb = 1u - m1_buf;\n"
       << ind << "    threadgroup " << mma_t << "* m1_sa_cur = m1_tg_a + m1_buf * " << a_buf_elems << "u;\n"
       << ind << "    threadgroup " << mma_t << "* m1_sb_cur = m1_tg_b + m1_buf * " << b_buf_elems << "u;\n"
       << ind << "    threadgroup " << mma_t << "* m1_sa_nxt = m1_tg_a + m1_nb * " << a_buf_elems << "u;\n"
       << ind << "    threadgroup " << mma_t << "* m1_sb_nxt = m1_tg_b + m1_nb * " << b_buf_elems << "u;\n"
       << ind << "    " << mma_v4 << " m1_ra[" << a_per << "];\n"
       << ind << "    " << mma_v4 << " m1_rb[" << b_per << "];\n";

    for (uint32_t p = 0; p < a_per; ++p) {
      os << ind << "    {\n"
         << ind << "      uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
      if ((p + 1u) * s.num_threads > a_nvec) os << ind << "      if (m1_vi < " << a_nvec << "u) {\n";
      os << ind << "      uint m1_r = m1_vi / " << a_vpr << "u;\n"
         << ind << "      uint m1_c4 = (m1_vi % " << a_vpr << "u) * 4u;\n";
      if (!a_cm) {
        os << ind << "      m1_ra[" << p << "] = " << mma_v4 << "(*(const device " << a_pv4
           << "*)(m1_A + ulong(m1_row0 + m1_r) * ulong(m1_sam) + ulong(m1_kn + m1_c4)));\n";
      } else {
        os << ind << "      m1_ra[" << p << "] = " << mma_v4 << "(*(const device " << a_pv4
           << "*)(m1_A + ulong(m1_kn + m1_r) * ulong(m1_sak) + ulong(m1_row0 + m1_c4)));\n";
      }
      if ((p + 1u) * s.num_threads > a_nvec) os << ind << "      }\n";
      os << ind << "    }\n";
    }
    for (uint32_t p = 0; p < b_per; ++p) {
      os << ind << "    {\n"
         << ind << "      uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
      if ((p + 1u) * s.num_threads > b_nvec) os << ind << "      if (m1_vi < " << b_nvec << "u) {\n";
      os << ind << "      uint m1_r = m1_vi / " << b_vpr << "u;\n"
         << ind << "      uint m1_c4 = (m1_vi % " << b_vpr << "u) * 4u;\n";
      if (!b_cm) {
        os << ind << "      m1_rb[" << p << "] = " << mma_v4 << "(*(const device " << b_pv4
           << "*)(m1_B + ulong(m1_kn + m1_r) * ulong(m1_sbk) + ulong(m1_col0 + m1_c4)));\n";
      } else {
        os << ind << "      m1_rb[" << p << "] = " << mma_v4 << "(*(const device " << b_pv4
           << "*)(m1_B + ulong(m1_col0 + m1_r) * ulong(m1_sbn) + ulong(m1_kn + m1_c4)));\n";
      }
      if ((p + 1u) * s.num_threads > b_nvec) os << ind << "      }\n";
      os << ind << "    }\n";
    }

    emit_tile_compute(os, plan, "m1_sa_cur", "m1_sb_cur", a_ld_s, b_ld_s, store_a_cm, store_b_cm, ind + "    ");

    for (uint32_t p = 0; p < a_per; ++p) {
      os << ind << "    {\n"
         << ind << "      uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
      if ((p + 1u) * s.num_threads > a_nvec) os << ind << "      if (m1_vi < " << a_nvec << "u) {\n";
      os << ind << "      uint m1_r = m1_vi / " << a_vpr << "u;\n"
         << ind << "      uint m1_c4 = (m1_vi % " << a_vpr << "u) * 4u;\n";
      if (!a_cm || s.use_tm1) {
        os << ind << "      *(threadgroup " << mma_v4 << "*)(m1_sa_nxt + m1_r * " << a_ld_s << " + m1_c4) = m1_ra["
           << p << "];\n";
      } else {
        os << ind << "      m1_sa_nxt[(m1_c4 + 0u) * " << a_ld_s << " + m1_r] = " << mma_t << "(m1_ra[" << p
           << "].x);\n"
           << ind << "      m1_sa_nxt[(m1_c4 + 1u) * " << a_ld_s << " + m1_r] = " << mma_t << "(m1_ra[" << p
           << "].y);\n"
           << ind << "      m1_sa_nxt[(m1_c4 + 2u) * " << a_ld_s << " + m1_r] = " << mma_t << "(m1_ra[" << p
           << "].z);\n"
           << ind << "      m1_sa_nxt[(m1_c4 + 3u) * " << a_ld_s << " + m1_r] = " << mma_t << "(m1_ra[" << p
           << "].w);\n";
      }
      if ((p + 1u) * s.num_threads > a_nvec) os << ind << "      }\n";
      os << ind << "    }\n";
    }
    for (uint32_t p = 0; p < b_per; ++p) {
      os << ind << "    {\n"
         << ind << "      uint m1_vi = m1_tid_linear + " << (p * s.num_threads) << "u;\n";
      if ((p + 1u) * s.num_threads > b_nvec) os << ind << "      if (m1_vi < " << b_nvec << "u) {\n";
      os << ind << "      uint m1_r = m1_vi / " << b_vpr << "u;\n"
         << ind << "      uint m1_c4 = (m1_vi % " << b_vpr << "u) * 4u;\n";
      if (!b_cm || s.use_tm1) {
        os << ind << "      *(threadgroup " << mma_v4 << "*)(m1_sb_nxt + m1_r * " << b_ld_s << " + m1_c4) = m1_rb["
           << p << "];\n";
      } else {
        os << ind << "      m1_sb_nxt[(m1_c4 + 0u) * " << b_ld_s << " + m1_r] = " << mma_t << "(m1_rb[" << p
           << "].x);\n"
           << ind << "      m1_sb_nxt[(m1_c4 + 1u) * " << b_ld_s << " + m1_r] = " << mma_t << "(m1_rb[" << p
           << "].y);\n"
           << ind << "      m1_sb_nxt[(m1_c4 + 2u) * " << b_ld_s << " + m1_r] = " << mma_t << "(m1_rb[" << p
           << "].z);\n"
           << ind << "      m1_sb_nxt[(m1_c4 + 3u) * " << b_ld_s << " + m1_r] = " << mma_t << "(m1_rb[" << p
           << "].w);\n";
      }
      if ((p + 1u) * s.num_threads > b_nvec) os << ind << "      }\n";
      os << ind << "    }\n";
    }

    os << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "    m1_buf = m1_nb;\n"
       << ind << "  }\n"
       << ind << "  threadgroup " << mma_t << "* m1_sa_cur = m1_tg_a + m1_buf * " << a_buf_elems << "u;\n"
       << ind << "  threadgroup " << mma_t << "* m1_sb_cur = m1_tg_b + m1_buf * " << b_buf_elems << "u;\n";
    emit_tile_compute(os, plan, "m1_sa_cur", "m1_sb_cur", a_ld_s, b_ld_s, store_a_cm, store_b_cm, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "  m1_k0 = m1_k_full_end;\n"
       << ind << "}\n";
  }

  os << ind << "if (m1_k0 < m1_k_end) {\n";
  emit_safe_load_a(os, plan, "m1_tg_a", "m1_k0", a_ld_s, store_a_cm, ind + "  ");
  emit_safe_load_b(os, plan, "m1_tg_b", "m1_k0", b_ld_s, store_b_cm, ind + "  ");
  os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_tile_compute(os, plan, "m1_tg_a", "m1_tg_b", a_ld_s, b_ld_s, store_a_cm, store_b_cm, ind + "  ");
  os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << ind << "}\n";
}

void emit_safe_k_loop(std::ostringstream& os, const SgmmPlan& plan, const std::string& ind) {
  const auto& s = plan.sched;
  std::string a_ld_s = std::to_string(s.bk + s.pad_a) + "u";
  std::string b_ld_s = std::to_string(s.bn + s.pad_b) + "u";
  os << ind << "for (uint m1_k0 = m1_k_begin; m1_k0 < m1_k_end; m1_k0 += " << s.bk << "u) {\n";
  emit_safe_load_a(os, plan, "m1_tg_a", "m1_k0", a_ld_s, false, ind + "  ");
  emit_safe_load_b(os, plan, "m1_tg_b", "m1_k0", b_ld_s, false, ind + "  ");
  os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_tile_compute(os, plan, "m1_tg_a", "m1_tg_b", a_ld_s, b_ld_s, false, false, ind + "  ");
  os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << ind << "}\n";
}

void emit_mma2_gmem_load_a(std::ostringstream& os, const SgmmPlan& plan, const std::string& a_ptr_expr,
                           const std::string& sam_u, const std::string& sak_u, const std::string& k_expr,
                           const std::string& dst_prefix, bool a_cm, const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bm * s.bk) / 4u;
  uint32_t vpr = (!a_cm ? s.bk : s.bm) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    std::string vi = "(m1_tid_linear + " + std::to_string(p * s.num_threads) + "u)";
    std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    std::string ptr_off = !a_cm ? ("ulong(m1_row0 + " + r + ") * " + sam_u + " + ulong(" + k_expr + " + " + c4 + ")")
                                : ("ulong(" + k_expr + " + " + r + ") * " + sak_u + " + ulong(m1_row0 + " + c4 + ")");
    if ((p + 1u) * s.num_threads > nvec) {
      os << ind << "float4 " << dst_prefix << "_" << p << " = float4(0.0f);\n"
         << ind << "if (" << vi << " < " << nvec << "u) " << dst_prefix << "_" << p
         << " = float4(*(const device packed_float4*)(" << a_ptr_expr << " + " << ptr_off << "));\n";
    } else {
      os << ind << "float4 " << dst_prefix << "_" << p << " = float4(*(const device packed_float4*)(" << a_ptr_expr
         << " + " << ptr_off << "));\n";
    }
  }
}

void emit_mma2_sram_store_a(std::ostringstream& os, const SgmmPlan& plan, uint32_t a_ld,
                            const std::string& src_prefix, bool a_cm, const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bm * s.bk) / 4u;
  uint32_t vpr = (!a_cm ? s.bk : s.bm) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    std::string vi = "(m1_tid_linear + " + std::to_string(p * s.num_threads) + "u)";
    std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    std::string cond = ((p + 1u) * s.num_threads > nvec) ? ("if (" + vi + " < " + std::to_string(nvec) + "u) ") : "";
    if (!a_cm) {
      os << ind << cond << "*(threadgroup float4*)(m1_sa + " << r << " * " << a_ld << "u + " << c4 << ") = "
         << src_prefix << "_" << p << ";\n";
    } else {
      os << ind << cond << "{ m1_sa[(" << c4 << " + 0u) * " << a_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".x; m1_sa[(" << c4 << " + 1u) * " << a_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".y; m1_sa[(" << c4 << " + 2u) * " << a_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".z; m1_sa[(" << c4 << " + 3u) * " << a_ld << "u + " << r << "] = " << src_prefix << "_" << p << ".w; }\n";
    }
  }
}

void emit_mma2_gmem_load_b(std::ostringstream& os, const SgmmPlan& plan, const std::string& b_ptr_expr,
                           const std::string& sbk_u, const std::string& sbn_u, const std::string& k_expr,
                           const std::string& dst_prefix, bool b_cm, const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bk * s.bn) / 4u;
  uint32_t vpr = (!b_cm ? s.bn : s.bk) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    std::string vi = "(m1_tid_linear + " + std::to_string(p * s.num_threads) + "u)";
    std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    std::string ptr_off = !b_cm ? ("ulong(" + k_expr + " + " + r + ") * " + sbk_u + " + ulong(m1_col0 + " + c4 + ")")
                                : ("ulong(m1_col0 + " + r + ") * " + sbn_u + " + ulong(" + k_expr + " + " + c4 + ")");
    if ((p + 1u) * s.num_threads > nvec) {
      os << ind << "float4 " << dst_prefix << "_" << p << " = float4(0.0f);\n"
         << ind << "if (" << vi << " < " << nvec << "u) " << dst_prefix << "_" << p
         << " = float4(*(const device packed_float4*)(" << b_ptr_expr << " + " << ptr_off << "));\n";
    } else {
      os << ind << "float4 " << dst_prefix << "_" << p << " = float4(*(const device packed_float4*)(" << b_ptr_expr
         << " + " << ptr_off << "));\n";
    }
  }
}

void emit_mma2_sram_store_b(std::ostringstream& os, const SgmmPlan& plan, uint32_t b_ld,
                            const std::string& src_prefix, bool b_cm, const std::string& ind) {
  const auto& s = plan.sched;
  uint32_t nvec = (s.bk * s.bn) / 4u;
  uint32_t vpr = (!b_cm ? s.bn : s.bk) / 4u;
  uint32_t per = (nvec + s.num_threads - 1u) / s.num_threads;
  for (uint32_t p = 0u; p < per; ++p) {
    std::string vi = "(m1_tid_linear + " + std::to_string(p * s.num_threads) + "u)";
    std::string r = "(" + vi + " / " + std::to_string(vpr) + "u)";
    std::string c4 = "((" + vi + " % " + std::to_string(vpr) + "u) * 4u)";
    std::string cond = ((p + 1u) * s.num_threads > nvec) ? ("if (" + vi + " < " + std::to_string(nvec) + "u) ") : "";
    if (!b_cm) {
      os << ind << cond << "*(threadgroup float4*)(m1_sb + " << r << " * " << b_ld << "u + " << c4 << ") = "
         << src_prefix << "_" << p << ";\n";
    } else {
      os << ind << cond << "{ m1_sb[(" << c4 << " + 0u) * " << b_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".x; m1_sb[(" << c4 << " + 1u) * " << b_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".y; m1_sb[(" << c4 << " + 2u) * " << b_ld << "u + " << r << "] = " << src_prefix << "_" << p
         << ".z; m1_sb[(" << c4 << " + 3u) * " << b_ld << "u + " << r << "] = " << src_prefix << "_" << p << ".w; }\n";
    }
  }
}

void emit_mma2_safe_loads(std::ostringstream& os, const SgmmPlan& plan, const std::string& a_ptr_expr,
                          const std::string& b_ptr_expr, uint32_t a_ld, uint32_t b_ld, const std::string& ind) {
  const auto& s = plan.sched;
  std::string m_cond = plan.m_bound.has_value() ? "(m1_gr < m1_M)" : "true";
  std::string n_cond = plan.n_bound.has_value() ? "(m1_gc < m1_N)" : "true";
  std::string k_cond = "(m1_gk < m1_k_end)";
  if (plan.k_lower_mask.has_value()) {
    k_cond = "(" + k_cond + " && (m1_gk >= m1_k_mask_lo))";
  }
  std::string sam_off = (plan.stride_am_const == 1) ? "long(m1_gr)" : ("long(m1_gr) * long(" + plan.stride_am_expr + ")");
  std::string sak_off = (plan.stride_ak_const == 1) ? "long(m1_gk)" : ("long(m1_gk) * long(" + plan.stride_ak_expr + ")");
  std::string sbk_off = (plan.stride_bk_const == 1) ? "long(m1_gk)" : ("long(m1_gk) * long(" + plan.stride_bk_expr + ")");
  std::string sbn_off = (plan.stride_bn_const == 1) ? "long(m1_gc)" : ("long(m1_gc) * long(" + plan.stride_bn_expr + ")");

  os << ind << "for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bm * s.bk) << "u; m1_idx += " << s.num_threads
     << "u) {\n"
     << ind << "  uint m1_r = m1_idx / " << s.bk << "u;\n"
     << ind << "  uint m1_c = m1_idx % " << s.bk << "u;\n"
     << ind << "  uint m1_gr = m1_row0 + m1_r;\n"
     << ind << "  uint m1_gk = m1_k0 + m1_c;\n"
     << ind << "  float m1_av = (" << m_cond << " && " << k_cond << ") ? (" << a_ptr_expr << ")[" << sam_off << " + "
     << sak_off << "] : 0.0f;\n"
     << ind << "  m1_sa[m1_r * " << a_ld << "u + m1_c] = m1_av;\n"
     << ind << "}\n"
     << ind << "for (uint m1_idx = m1_tid_linear; m1_idx < " << (s.bk * s.bn) << "u; m1_idx += " << s.num_threads
     << "u) {\n"
     << ind << "  uint m1_r = m1_idx / " << s.bn << "u;\n"
     << ind << "  uint m1_c = m1_idx % " << s.bn << "u;\n"
     << ind << "  uint m1_gk = m1_k0 + m1_r;\n"
     << ind << "  uint m1_gc = m1_col0 + m1_c;\n"
     << ind << "  float m1_bv = (" << k_cond << " && " << n_cond << ") ? (" << b_ptr_expr << ")[" << sbk_off << " + "
     << sbn_off << "] : 0.0f;\n"
     << ind << "  m1_sb[m1_r * " << b_ld << "u + m1_c] = m1_bv;\n"
     << ind << "}\n";
}

void emit_mma2_compute(std::ostringstream& os, const SgmmPlan& plan, uint32_t a_ld, uint32_t b_ld,
                       const std::string& ind) {
  const auto& s = plan.sched;
  for (uint32_t ks = 0u; ks < s.bk; ks += 8u) {
    os << ind << "{\n";
    for (uint32_t j = 0u; j < s.tn; ++j) {
      os << ind << "  float2 m1_fb_" << j << " = *(threadgroup const float2*)(m1_sb + (" << ks
         << "u + m1_fr) * " << b_ld << "u + (m1_sg_col + " << (j * 8u) << "u + m1_fc));\n";
    }
    os << ind << "  float2 m1_fa_c = *(threadgroup const float2*)(m1_sa + (m1_sg_row + m1_fr) * " << a_ld
       << "u + (" << ks << "u + m1_fc));\n"
       << ind << "  float2 m1_fa_n;\n";
    for (uint32_t i = 0u; i < s.tm; ++i) {
      if (i + 1u < s.tm) {
        os << ind << "  m1_fa_n = *(threadgroup const float2*)(m1_sa + (m1_sg_row + " << ((i + 1u) * 8u)
           << "u + m1_fr) * " << a_ld << "u + (" << ks << "u + m1_fc));\n";
      }
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << ind << "  m1_c_" << i << "_" << j << " = m1_mma2(m1_fa_c, m1_fb_" << j << ", m1_c_" << i << "_" << j
           << ");\n";
      }
      if (i + 1u < s.tm) {
        os << ind << "  m1_fa_c = m1_fa_n;\n";
      }
    }
    os << ind << "}\n";
  }
}

void emit_mma2_fast_loop_body(std::ostringstream& os, const SgmmPlan& plan, const std::string& a_ptr_expr,
                              const std::string& b_ptr_expr, const std::string& sam_u, const std::string& sak_u,
                              const std::string& sbk_u, const std::string& sbn_u, uint32_t a_ld, uint32_t b_ld,
                              bool a_cm, bool b_cm, bool use_reg_prefetch, bool needs_trailing_barrier,
                              const std::string& ind) {
  const auto& s = plan.sched;
  if (use_reg_prefetch) {
    os << ind << "if (m1_k0 < m1_k_full_end) {\n"
       << ind << "  {\n";
    emit_mma2_gmem_load_a(os, plan, a_ptr_expr, sam_u, sak_u, "m1_k0", "m1_pa", a_cm, ind + "    ");
    emit_mma2_gmem_load_b(os, plan, b_ptr_expr, sbk_u, sbn_u, "m1_k0", "m1_pb", b_cm, ind + "    ");
    emit_mma2_sram_store_a(os, plan, a_ld, "m1_pa", a_cm, ind + "    ");
    emit_mma2_sram_store_b(os, plan, b_ld, "m1_pb", b_cm, ind + "    ");
    os << ind << "  }\n"
       << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "  for (m1_k0 += " << s.bk << "u; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_mma2_gmem_load_a(os, plan, a_ptr_expr, sam_u, sak_u, "m1_k0", "m1_ra", a_cm, ind + "    ");
    emit_mma2_gmem_load_b(os, plan, b_ptr_expr, sbk_u, sbn_u, "m1_k0", "m1_rb", b_cm, ind + "    ");
    emit_mma2_compute(os, plan, a_ld, b_ld, ind + "    ");
    os << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma2_sram_store_a(os, plan, a_ld, "m1_ra", a_cm, ind + "    ");
    emit_mma2_sram_store_b(os, plan, b_ld, "m1_rb", b_cm, ind + "    ");
    os << ind << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "  }\n";
    emit_mma2_compute(os, plan, a_ld, b_ld, ind + "  ");
    if (needs_trailing_barrier) {
      os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    }
    os << ind << "}\n";
  } else {
    os << ind << "for (; m1_k0 < m1_k_full_end; m1_k0 += " << s.bk << "u) {\n";
    emit_mma2_gmem_load_a(os, plan, a_ptr_expr, sam_u, sak_u, "m1_k0", "m1_ra", a_cm, ind + "  ");
    emit_mma2_gmem_load_b(os, plan, b_ptr_expr, sbk_u, sbn_u, "m1_k0", "m1_rb", b_cm, ind + "  ");
    emit_mma2_sram_store_a(os, plan, a_ld, "m1_ra", a_cm, ind + "  ");
    emit_mma2_sram_store_b(os, plan, b_ld, "m1_rb", b_cm, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    emit_mma2_compute(os, plan, a_ld, b_ld, ind + "  ");
    os << ind << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << ind << "}\n";
  }
}

std::string emit_mma2_sgmm_msl(const Kernel& kernel, const SgmmPlan& plan) {
  const auto& s = plan.sched;
  const uint32_t a_ld = s.bk + s.pad_a;
  const uint32_t b_ld = s.bn + s.pad_b;
  std::ostringstream os;
  os << "#include <metal_stdlib>\n"
     << "#include <metal_simdgroup_matrix>\n"
     << "using namespace metal;\n\n"
     << "static inline float2 m1_mma2(float2 a, float2 b, float2 c) {\n"
     << "  simdgroup_float8x8 am, bm, cm;\n"
     << "  reinterpret_cast<thread float2&>(am.thread_elements()) = a;\n"
     << "  reinterpret_cast<thread float2&>(bm.thread_elements()) = b;\n"
     << "  reinterpret_cast<thread float2&>(cm.thread_elements()) = c;\n"
     << "  simdgroup_multiply_accumulate(cm, am, bm, cm);\n"
     << "  return reinterpret_cast<thread float2&>(cm.thread_elements());\n"
     << "}\n\n"
     << "[[max_total_threads_per_threadgroup(" << s.num_threads << ")]]\n"
     << "kernel void " << kernel.name << "(\n";
  emit_param_slot_decls(os, plan.params.slots);
  os << "    uint3 m1_tg_pos [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint3 m1_tid [[thread_position_in_threadgroup]],\n"
     << "    uint3 m1_tg_size [[threads_per_threadgroup]],\n"
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
     << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]],\n"
     << "    uint m1_lane [[thread_index_in_simdgroup]]) {\n"
     << "  (void)m1_tid; (void)m1_tg_size; (void)m1_grid_dim;\n";

  if (plan.simple_2d_pid_grid && s.swizzle_w > 1u) {
    std::string raw_m = "m1_tg_pos." + axis_component(plan.pid_m_axis);
    std::string raw_n = "m1_tg_pos." + axis_component(plan.pid_n_axis);
    std::string dim_m = "m1_grid_dim." + axis_component(plan.pid_m_axis);
    std::string dim_n = "m1_grid_dim." + axis_component(plan.pid_n_axis);
    os << "  uint m1_num_m = " << dim_m << ";\n"
       << "  uint m1_num_n = " << dim_n << ";\n"
       << "  uint m1_pid = " << raw_n << " * m1_num_m + " << raw_m << ";\n"
       << "  uint m1_gw = " << s.swizzle_w << "u;\n";
    if (s.swizzle_along_n) {
      os << "  uint m1_npg = m1_gw * m1_num_m;\n"
         << "  uint m1_gid = m1_pid / m1_npg;\n"
         << "  uint m1_fn = m1_gid * m1_gw;\n"
         << "  uint m1_gsn = min(m1_num_n - m1_fn, m1_gw);\n"
         << "  uint m1_rem = m1_pid % m1_npg;\n"
         << "  uint m1_pid_n = m1_fn + (m1_rem % m1_gsn);\n"
         << "  uint m1_pid_m = m1_rem / m1_gsn;\n"
         << "  if (m1_gid & 1u) m1_pid_m = m1_num_m - 1u - m1_pid_m;\n";
    } else {
      os << "  uint m1_npg = m1_gw * m1_num_n;\n"
         << "  uint m1_gid = m1_pid / m1_npg;\n"
         << "  uint m1_fm = m1_gid * m1_gw;\n"
         << "  uint m1_gsm = min(m1_num_m - m1_fm, m1_gw);\n"
         << "  uint m1_rem = m1_pid % m1_npg;\n"
         << "  uint m1_pid_m = m1_fm + (m1_rem % m1_gsm);\n"
         << "  uint m1_pid_n = m1_rem / m1_gsm;\n"
         << "  if (m1_gid & 1u) m1_pid_n = m1_num_n - 1u - m1_pid_n;\n";
    }
    os << "  uint m1_row0 = m1_pid_m * " << s.bm << "u;\n"
       << "  uint m1_col0 = m1_pid_n * " << s.bn << "u;\n";
  } else {
    os << "  uint m1_row0 = uint(" << plan.row0_expr << ");\n"
       << "  uint m1_col0 = uint(" << plan.col0_expr << ");\n";
  }

  if (plan.m_bound.has_value()) {
    os << "  uint m1_M = uint(max(0l, long(" << *plan.m_bound << ")));\n"
       << "  if (m1_row0 >= m1_M) return;\n";
  }
  if (plan.n_bound.has_value()) {
    os << "  uint m1_N = uint(max(0l, long(" << *plan.n_bound << ")));\n"
       << "  if (m1_col0 >= m1_N) return;\n";
  }

  const bool k_begin_zero = (plan.k_begin_ir == "0");
  if (!k_begin_zero) {
    os << "  uint m1_k_begin = uint(max(0l, long(" << plan.k_begin_ir << ")));\n";
  }
  os << "  uint m1_k_end = uint(max(0l, long(" << plan.k_end_ir << ")));\n";
  if (plan.k_bound.has_value() && *plan.k_bound != plan.k_end_ir) {
    os << "  m1_k_end = min(m1_k_end, uint(max(0l, long(" << *plan.k_bound << "))));\n";
  }
  if (plan.k_lower_mask.has_value()) {
    os << "  uint m1_k_mask_lo = uint(max(0l, long(" << *plan.k_lower_mask << ")));\n";
  }

  std::string a_ptr_expr = (plan.a_batch_off == "0")
                               ? plan.a_param
                               : ("(" + plan.a_param + " + long(" + plan.a_batch_off + "))");
  std::string b_ptr_expr = (plan.b_batch_off == "0")
                               ? plan.b_param
                               : ("(" + plan.b_param + " + long(" + plan.b_batch_off + "))");
  std::string c_ptr_expr = (plan.c_batch_off == "0")
                               ? plan.c_param
                               : ("(" + plan.c_param + " + long(" + plan.c_batch_off + "))");

  std::string sam_u = (plan.stride_am_const == 1) ? "1ul" : ("ulong(" + plan.stride_am_expr + ")");
  std::string sak_u = (plan.stride_ak_const == 1) ? "1ul" : ("ulong(" + plan.stride_ak_expr + ")");
  std::string sbk_u = (plan.stride_bk_const == 1) ? "1ul" : ("ulong(" + plan.stride_bk_expr + ")");
  std::string sbn_u = (plan.stride_bn_const == 1) ? "1ul" : ("ulong(" + plan.stride_bn_expr + ")");
  std::string scm_u = (plan.stride_cm_const == 1) ? "1ul" : ("ulong(" + plan.stride_cm_expr + ")");

  os << "  threadgroup float m1_sa[" << (s.bm * a_ld) << "];\n"
     << "  threadgroup float m1_sb[" << (s.bk * b_ld) << "];\n"
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

  std::string full_m_expr = plan.m_bound ? ("(m1_row0 + " + std::to_string(s.bm) + "u <= m1_M)") : "true";
  std::string full_n_expr = plan.n_bound ? ("(m1_col0 + " + std::to_string(s.bn) + "u <= m1_N)") : "true";
  std::string k_begin_s = k_begin_zero ? "0u" : "m1_k_begin";
  std::string k_mask_guard = plan.k_lower_mask.has_value() ? ("(m1_k_mask_lo <= " + k_begin_s + ")") : "true";

  bool a_always_rm = plan.stride_ak_const.has_value() && *plan.stride_ak_const == 1;
  bool a_always_cm = !a_always_rm && plan.stride_am_const.has_value() && *plan.stride_am_const == 1;
  bool b_always_rm = plan.stride_bn_const.has_value() && *plan.stride_bn_const == 1;
  bool b_always_cm = !b_always_rm && plan.stride_bk_const.has_value() && *plan.stride_bk_const == 1;
  bool c_always_rm = plan.stride_cn_const.has_value() && *plan.stride_cn_const == 1;
  const bool layout_compile_time_known = (a_always_rm || a_always_cm) && (b_always_rm || b_always_cm);
  const bool use_reg_prefetch = (s.tm * s.tn <= 16u) && (s.swizzle_w > 1u);
  const bool aligned_fast_only = !plan.m_bound.has_value() && !plan.n_bound.has_value() && !plan.k_bound.has_value() &&
                                 !plan.k_lower_mask.has_value() && layout_compile_time_known && c_always_rm &&
                                 k_begin_zero;

  if (aligned_fast_only) {
    os << "  uint m1_k0 = 0u;\n"
       << "  uint m1_k_full_end = m1_k_end;\n";
    emit_mma2_fast_loop_body(os, plan, a_ptr_expr, b_ptr_expr, sam_u, sak_u, sbk_u, sbn_u, a_ld, b_ld, a_always_cm,
                             b_always_cm, use_reg_prefetch, false, "  ");
    for (uint32_t i = 0u; i < s.tm; ++i) {
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << "  { simdgroup_float8x8 m1_cm; reinterpret_cast<thread float2&>(m1_cm.thread_elements()) = m1_c_" << i
           << "_" << j << "; simdgroup_store(m1_cm, " << c_ptr_expr << " + ulong(m1_row0 + m1_sg_row + " << (i * 8u)
           << "u) * " << scm_u << " + ulong(m1_col0 + m1_sg_col + " << (j * 8u) << "u), " << scm_u << "); }\n";
      }
    }
    os << "}\n";
    return os.str();
  }

  os << "  uint m1_k0 = " << k_begin_s << ";\n";
  if (layout_compile_time_known) {
    os << "  if (" << full_m_expr << " && " << full_n_expr << " && " << k_mask_guard << ") {\n";
    if (k_begin_zero) {
      os << "    uint m1_k_full_end = m1_k_end & ~" << (s.bk - 1u) << "u;\n";
    } else {
      os << "    uint m1_k_full_end = (m1_k_end > m1_k_begin) ? (m1_k_begin + ((m1_k_end - m1_k_begin) & ~"
         << (s.bk - 1u) << "u)) : m1_k_begin;\n";
    }
    emit_mma2_fast_loop_body(os, plan, a_ptr_expr, b_ptr_expr, sam_u, sak_u, sbk_u, sbn_u, a_ld, b_ld, a_always_cm,
                             b_always_cm, use_reg_prefetch, true, "    ");
    os << "  }\n";
  } else {
    bool a_can_rm = !a_always_cm && (!plan.stride_ak_const.has_value() || *plan.stride_ak_const == 1);
    bool b_can_rm = !b_always_cm && (!plan.stride_bn_const.has_value() || *plan.stride_bn_const == 1);
    if (a_can_rm && b_can_rm) {
      os << "  if (" << full_m_expr << " && " << full_n_expr << " && " << k_mask_guard << " && (long("
         << plan.stride_ak_expr << ") == 1l) && (long(" << plan.stride_bn_expr << ") == 1l)) {\n";
      if (k_begin_zero) {
        os << "    uint m1_k_full_end = m1_k_end & ~" << (s.bk - 1u) << "u;\n";
      } else {
        os << "    uint m1_k_full_end = (m1_k_end > m1_k_begin) ? (m1_k_begin + ((m1_k_end - m1_k_begin) & ~"
           << (s.bk - 1u) << "u)) : m1_k_begin;\n";
      }
      emit_mma2_fast_loop_body(os, plan, a_ptr_expr, b_ptr_expr, sam_u, sak_u, sbk_u, sbn_u, a_ld, b_ld, false, false,
                               use_reg_prefetch, true, "    ");
      os << "  }\n";
    }
  }

  os << "  if (m1_k0 < m1_k_end) {\n"
     << "    for (; m1_k0 < m1_k_end; m1_k0 += " << s.bk << "u) {\n";
  emit_mma2_safe_loads(os, plan, a_ptr_expr, b_ptr_expr, a_ld, b_ld, "      ");
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  emit_mma2_compute(os, plan, a_ld, b_ld, "      ");
  os << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "    }\n"
     << "  }\n";

  std::string c_rm_cond = c_always_rm ? "" : (" && (long(" + plan.stride_cn_expr + ") == 1l)");
  os << "  if (" << full_m_expr << " && " << full_n_expr << c_rm_cond << ") {\n";
  for (uint32_t i = 0u; i < s.tm; ++i) {
    for (uint32_t j = 0u; j < s.tn; ++j) {
      os << "    { simdgroup_float8x8 m1_cm; reinterpret_cast<thread float2&>(m1_cm.thread_elements()) = m1_c_" << i
         << "_" << j << "; simdgroup_store(m1_cm, " << c_ptr_expr << " + ulong(m1_row0 + m1_sg_row + " << (i * 8u)
         << "u) * " << scm_u << " + ulong(m1_col0 + m1_sg_col + " << (j * 8u) << "u), " << scm_u << "); }\n";
    }
  }
  os << "  } else {\n";
  for (uint32_t i = 0u; i < s.tm; ++i) {
    os << "    {\n"
       << "      uint m1_gr = m1_row0 + m1_sg_row + " << (i * 8u) << "u + m1_fr;\n";
    if (plan.m_bound.has_value()) {
      os << "      if (m1_gr < m1_M) {\n";
    }
    if (c_always_rm) {
      os << "        device float* m1_cr = " << c_ptr_expr << " + ulong(m1_gr) * " << scm_u << ";\n";
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << "        { uint m1_gc = m1_col0 + m1_sg_col + " << (j * 8u) << "u + m1_fc;";
        if (plan.n_bound.has_value()) {
          os << " if (m1_gc < m1_N) m1_cr[m1_gc] = m1_c_" << i << "_" << j
             << ".x; if (m1_gc + 1u < m1_N) m1_cr[m1_gc + 1u] = m1_c_" << i << "_" << j << ".y; }\n";
        } else {
          os << " m1_cr[m1_gc] = m1_c_" << i << "_" << j << ".x; m1_cr[m1_gc + 1u] = m1_c_" << i << "_" << j
             << ".y; }\n";
        }
      }
    } else {
      for (uint32_t j = 0u; j < s.tn; ++j) {
        os << "        { uint m1_gc = m1_col0 + m1_sg_col + " << (j * 8u) << "u + m1_fc;";
        if (plan.n_bound.has_value()) {
          os << " if (m1_gc < m1_N) (" << c_ptr_expr << ")[long(m1_gr) * long(" << plan.stride_cm_expr
             << ") + long(m1_gc) * long(" << plan.stride_cn_expr << ")] = m1_c_" << i << "_" << j
             << ".x; if (m1_gc + 1u < m1_N) (" << c_ptr_expr << ")[long(m1_gr) * long(" << plan.stride_cm_expr
             << ") + long(m1_gc + 1u) * long(" << plan.stride_cn_expr << ")] = m1_c_" << i << "_" << j << ".y; }\n";
        } else {
          os << " (" << c_ptr_expr << ")[long(m1_gr) * long(" << plan.stride_cm_expr << ") + long(m1_gc) * long("
             << plan.stride_cn_expr << ")] = m1_c_" << i << "_" << j << ".x; (" << c_ptr_expr << ")[long(m1_gr) * long("
             << plan.stride_cm_expr << ") + long(m1_gc + 1u) * long(" << plan.stride_cn_expr << ")] = m1_c_" << i
             << "_" << j << ".y; }\n";
        }
      }
    }
    if (plan.m_bound.has_value()) {
      os << "      }\n";
    }
    os << "    }\n";
  }
  os << "  }\n"
     << "}\n";
  return os.str();
}

std::string emit_sgmm_msl(const Kernel& kernel, const SgmmPlan& plan) {
  if (plan.sched.use_mma2) {
    return emit_mma2_sgmm_msl(kernel, plan);
  }
  const auto& s = plan.sched;
  std::ostringstream os;
  os << "#include <metal_stdlib>\n"
     << "#include <metal_simdgroup_matrix>\n"
     << "using namespace metal;\n\n"
     << "[[max_total_threads_per_threadgroup(" << s.num_threads << ")]]\n";

  os << "kernel void " << kernel.name << "(\n";
  emit_param_slot_decls(os, plan.params.slots);
  os << "    uint3 m1_tg_pos [[threadgroup_position_in_grid]],\n"
     << "    uint3 m1_grid_dim [[threadgroups_per_grid]],\n"
     << "    uint3 m1_tid [[thread_position_in_threadgroup]],\n"
     << "    uint3 m1_tg_size [[threads_per_threadgroup]],\n"
     << "    uint m1_tid_linear [[thread_index_in_threadgroup]],\n"
     << "    uint m1_simd_id [[simdgroup_index_in_threadgroup]]) {\n"
     << "  (void)m1_tid; (void)m1_tg_size; (void)m1_grid_dim;\n";

  if (plan.simple_2d_pid_grid && s.swizzle_w > 1u) {
    std::string raw_m = "m1_tg_pos." + axis_component(plan.pid_m_axis);
    std::string raw_n = "m1_tg_pos." + axis_component(plan.pid_n_axis);
    std::string dim_m = "m1_grid_dim." + axis_component(plan.pid_m_axis);
    std::string dim_n = "m1_grid_dim." + axis_component(plan.pid_n_axis);
    os << "  uint m1_num_m = " << dim_m << ";\n"
       << "  uint m1_num_n = " << dim_n << ";\n"
       << "  uint m1_pid = " << raw_n << " * m1_num_m + " << raw_m << ";\n"
       << "  uint m1_gw = " << s.swizzle_w << "u;\n";
    if (s.swizzle_along_n) {
      os << "  uint m1_npg = m1_gw * m1_num_m;\n"
         << "  uint m1_gid = m1_pid / m1_npg;\n"
         << "  uint m1_fn = m1_gid * m1_gw;\n"
         << "  uint m1_gsn = min(m1_num_n - m1_fn, m1_gw);\n"
         << "  uint m1_pid_n = m1_fn + ((m1_pid % m1_npg) % m1_gsn);\n"
         << "  uint m1_pid_m = (m1_pid % m1_npg) / m1_gsn;\n";
    } else {
      os << "  uint m1_npg = m1_gw * m1_num_n;\n"
         << "  uint m1_gid = m1_pid / m1_npg;\n"
         << "  uint m1_fm = m1_gid * m1_gw;\n"
         << "  uint m1_gsm = min(m1_num_m - m1_fm, m1_gw);\n"
         << "  uint m1_pid_m = m1_fm + ((m1_pid % m1_npg) % m1_gsm);\n"
         << "  uint m1_pid_n = (m1_pid % m1_npg) / m1_gsm;\n";
    }
    os << "  uint m1_row0 = m1_pid_m * " << s.bm << "u;\n"
       << "  uint m1_col0 = m1_pid_n * " << s.bn << "u;\n";
  } else {
    os << "  uint m1_row0 = uint(" << plan.row0_expr << ");\n"
       << "  uint m1_col0 = uint(" << plan.col0_expr << ");\n";
  }

  if (plan.m_bound.has_value()) {
    os << "  uint m1_M = uint(max(0l, long(" << *plan.m_bound << ")));\n"
       << "  if (m1_row0 >= m1_M) return;\n";
  }
  if (plan.n_bound.has_value()) {
    os << "  uint m1_N = uint(max(0l, long(" << *plan.n_bound << ")));\n"
       << "  if (m1_col0 >= m1_N) return;\n";
  }

  os << "  long m1_sam = long(" << plan.stride_am_expr << ");\n"
     << "  long m1_sak = long(" << plan.stride_ak_expr << ");\n"
     << "  long m1_sbk = long(" << plan.stride_bk_expr << ");\n"
     << "  long m1_sbn = long(" << plan.stride_bn_expr << ");\n"
     << "  long m1_scm = long(" << plan.stride_cm_expr << ");\n"
     << "  long m1_scn = long(" << plan.stride_cn_expr << ");\n";

  os << "  device const " << msl_scalar_type(plan.a_elem_dt) << "* m1_A = " << plan.a_param << " + long("
     << plan.a_batch_off << ");\n"
     << "  device const " << msl_scalar_type(plan.b_elem_dt) << "* m1_B = " << plan.b_param << " + long("
     << plan.b_batch_off << ");\n"
     << "  device " << msl_scalar_type(plan.c_elem_dt) << "* m1_C = " << plan.c_param << " + long(" << plan.c_batch_off
     << ");\n";

  if (plan.structure == MatrixStructure::UpperTriangular) {
    os << "  if (int(m1_row0) >= int(m1_col0) + " << static_cast<int32_t>(s.bn) - plan.diagonal_offset << ") {\n";
    if (!plan.is_atomic) {
      std::string m_end = plan.m_bound ? ("min(m1_M, m1_row0 + " + std::to_string(s.bm) + "u)")
                                       : ("m1_row0 + " + std::to_string(s.bm) + "u");
      std::string n_end = plan.n_bound ? ("min(m1_N, m1_col0 + " + std::to_string(s.bn) + "u)")
                                       : ("m1_col0 + " + std::to_string(s.bn) + "u");
      os << "    uint m1_rows = " << m_end << " - m1_row0;\n"
         << "    uint m1_cols = " << n_end << " - m1_col0;\n"
         << "    uint m1_tot = m1_rows * m1_cols;\n"
         << "    for (uint m1_idx = m1_tid_linear; m1_idx < m1_tot; m1_idx += " << s.num_threads << "u) {\n"
         << "      uint m1_r = m1_idx / m1_cols;\n"
         << "      uint m1_c = m1_idx - m1_r * m1_cols;\n"
         << "      m1_C[long(m1_row0 + m1_r) * m1_scm + long(m1_col0 + m1_c) * m1_scn] = "
         << msl_scalar_type(plan.c_elem_dt) << "(0.0f);\n"
         << "    }\n";
    }
    os << "    return;\n"
       << "  }\n";
  } else if (plan.structure == MatrixStructure::LowerTriangular) {
    os << "  if (int(m1_col0) >= int(m1_row0) + " << static_cast<int32_t>(s.bm) + plan.diagonal_offset << ") {\n";
    if (!plan.is_atomic) {
      std::string m_end = plan.m_bound ? ("min(m1_M, m1_row0 + " + std::to_string(s.bm) + "u)")
                                       : ("m1_row0 + " + std::to_string(s.bm) + "u");
      std::string n_end = plan.n_bound ? ("min(m1_N, m1_col0 + " + std::to_string(s.bn) + "u)")
                                       : ("m1_col0 + " + std::to_string(s.bn) + "u");
      os << "    uint m1_rows = " << m_end << " - m1_row0;\n"
         << "    uint m1_cols = " << n_end << " - m1_col0;\n"
         << "    uint m1_tot = m1_rows * m1_cols;\n"
         << "    for (uint m1_idx = m1_tid_linear; m1_idx < m1_tot; m1_idx += " << s.num_threads << "u) {\n"
         << "      uint m1_r = m1_idx / m1_cols;\n"
         << "      uint m1_c = m1_idx - m1_r * m1_cols;\n"
         << "      m1_C[long(m1_row0 + m1_r) * m1_scm + long(m1_col0 + m1_c) * m1_scn] = "
         << msl_scalar_type(plan.c_elem_dt) << "(0.0f);\n"
         << "    }\n";
    }
    os << "    return;\n"
       << "  }\n";
  }

  os << "  uint m1_k_begin = uint(max(0l, long(" << plan.k_begin_ir << ")));\n"
     << "  uint m1_k_end = uint(max(0l, long(" << plan.k_end_ir << ")));\n";
  if (plan.k_bound.has_value()) {
    os << "  uint m1_K_bound = uint(max(0l, long(" << *plan.k_bound << ")));\n"
       << "  m1_k_end = min(m1_k_end, m1_K_bound);\n";
  }
  if (plan.k_lower_mask.has_value()) {
    os << "  uint m1_k_mask_lo = uint(max(0l, long(" << *plan.k_lower_mask << ")));\n";
  }

  if (plan.structure == MatrixStructure::UpperTriangular && plan.diagonal_offset >= 0) {
    os << "  uint m1_tri_lo = (uint(max(0, int(m1_row0) + " << plan.diagonal_offset << ")) / " << s.bk << "u) * "
       << s.bk << "u;\n"
       << "  uint m1_tri_hi = uint(max(0, int(m1_col0) + " << static_cast<int32_t>(s.bn) - plan.diagonal_offset
       << "));\n"
       << "  m1_k_begin = max(m1_k_begin, m1_tri_lo);\n"
       << "  m1_k_end = min(m1_k_end, m1_tri_hi);\n";
  } else if (plan.structure == MatrixStructure::LowerTriangular && plan.diagonal_offset <= 0) {
    os << "  uint m1_tri_lo = (uint(max(0, int(m1_col0) - " << plan.diagonal_offset << ")) / " << s.bk << "u) * "
       << s.bk << "u;\n"
       << "  uint m1_tri_hi = uint(max(0, int(m1_row0) + " << static_cast<int32_t>(s.bm) + plan.diagonal_offset
       << "));\n"
       << "  m1_k_begin = max(m1_k_begin, m1_tri_lo);\n"
       << "  m1_k_end = min(m1_k_end, m1_tri_hi);\n";
  }

  if (plan.split_k > 1u) {
    os << "  uint m1_sk_span = (m1_k_end > m1_k_begin) ? (m1_k_end - m1_k_begin) : 0u;\n"
       << "  uint m1_sk_step = ((m1_sk_span + " << (plan.split_k - 1u) << "u) / " << plan.split_k << "u + "
       << (s.bk - 1u) << "u) & ~" << (s.bk - 1u) << "u;\n"
       << "  uint m1_sk_lo = min(m1_k_end, m1_k_begin + m1_tg_pos.z * m1_sk_step);\n"
       << "  uint m1_sk_hi = min(m1_k_end, m1_sk_lo + m1_sk_step);\n"
       << "  m1_k_begin = m1_sk_lo;\n"
       << "  m1_k_end = m1_sk_hi;\n";
  }

  uint32_t smem_u32 = (s.smem_bytes + 3u) / 4u;
  std::string mma_t = msl_scalar_type(plan.mma_dt);
  std::string acc_t = msl_scalar_type(plan.acc_dt);
  std::string acc_sg_t = msl_sgmm_type(plan.acc_dt);
  uint32_t mma_bytes = ir::dtype_bytes(plan.mma_dt);

  uint32_t a_rm_elems = s.bm * (s.bk + s.pad_a);
  uint32_t a_cm_elems = (s.use_tm1 ? s.bk * (s.bm + s.pad_a) : a_rm_elems);
  uint32_t a_slot_elems = std::max(a_rm_elems, a_cm_elems);
  uint32_t tg_b_offset_bytes = s.num_bufs * a_slot_elems * mma_bytes;

  os << "  threadgroup float m1_smem[" << smem_u32 << "];\n"
     << "  threadgroup " << mma_t << "* m1_tg_a = reinterpret_cast<threadgroup " << mma_t << "*>(m1_smem);\n"
     << "  threadgroup " << mma_t << "* m1_tg_b = reinterpret_cast<threadgroup " << mma_t
     << "*>(reinterpret_cast<threadgroup uchar*>(m1_smem) + " << tg_b_offset_bytes << "u);\n"
     << "  uint m1_sg_row = (m1_simd_id / " << s.sg_n << "u) * " << (s.tm * 8u) << "u;\n"
     << "  uint m1_sg_col = (m1_simd_id % " << s.sg_n << "u) * " << (s.tn * 8u) << "u;\n"
     << "  " << acc_sg_t << " m1_acc[" << s.tm << "][" << s.tn << "];\n"
     << "  for (uint m1_i = 0u; m1_i < " << s.tm << "u; ++m1_i)\n"
     << "    for (uint m1_j = 0u; m1_j < " << s.tn << "u; ++m1_j)\n"
     << "      m1_acc[m1_i][m1_j] = " << acc_sg_t << "(0.0f);\n";

  std::string full_m_expr = plan.m_bound ? ("(m1_row0 + " + std::to_string(s.bm) + "u <= m1_M)") : "true";
  std::string full_n_expr = plan.n_bound ? ("(m1_col0 + " + std::to_string(s.bn) + "u <= m1_N)") : "true";
  os << "  bool m1_full_m = " << full_m_expr << ";\n"
     << "  bool m1_full_n = " << full_n_expr << ";\n"
     << "  uint m1_k_loop_begin = m1_k_begin;\n"
     << "  uint m1_k_loop_end = m1_k_end;\n";

  bool a_always_rm = plan.stride_ak_const.has_value() && *plan.stride_ak_const == 1;
  bool a_always_cm = !a_always_rm && plan.stride_am_const.has_value() && *plan.stride_am_const == 1;
  bool b_always_rm = plan.stride_bn_const.has_value() && *plan.stride_bn_const == 1;
  bool b_always_cm = !b_always_rm && plan.stride_bk_const.has_value() && *plan.stride_bk_const == 1;

  bool a_can_rm = !a_always_cm && (!plan.stride_ak_const.has_value() || *plan.stride_ak_const == 1);
  bool a_can_cm = !a_always_rm && (!plan.stride_am_const.has_value() || *plan.stride_am_const == 1);
  bool b_can_rm = !b_always_cm && (!plan.stride_bn_const.has_value() || *plan.stride_bn_const == 1);
  bool b_can_cm = !b_always_rm && (!plan.stride_bk_const.has_value() || *plan.stride_bk_const == 1);

  if (plan.is_atomic) {
    os << "  if (m1_k_begin >= m1_k_end) return;\n";
  }
  os << "  bool m1_a_rm = (m1_sak == 1l);\n"
     << "  bool m1_a_cm = (!m1_a_rm && m1_sam == 1l);\n"
     << "  bool m1_b_rm = (m1_sbn == 1l);\n"
     << "  bool m1_b_cm = (!m1_b_rm && m1_sbk == 1l);\n"
     << "  if (m1_k_begin < m1_k_end) {\n"
     << "    if (m1_full_m && m1_full_n && !m1_k_mask_guard_placeholder) {\n";

  std::string body_str_guard = plan.k_lower_mask.has_value() ? "(m1_k_mask_lo <= m1_k_begin)" : "true";
  std::string full_code = os.str();
  size_t pos = full_code.find("!m1_k_mask_guard_placeholder");
  full_code.replace(pos, std::string("!m1_k_mask_guard_placeholder").size(), body_str_guard);
  os.str("");
  os.clear();
  os << full_code;

  const bool layout_compile_time_known =
      (a_always_rm || a_always_cm) && (b_always_rm || b_always_cm);

  bool first_branch = true;
  auto emit_mode_branch = [&](bool acm, bool bcm, const std::string& cond) {
    if (layout_compile_time_known) {
      first_branch = false;
      emit_fast_k_loop(os, plan, acm, bcm, "      ");
      return;
    }
    os << "      " << (first_branch ? "if" : "} else if") << " (" << cond << ") {\n";
    first_branch = false;
    emit_fast_k_loop(os, plan, acm, bcm, "        ");
  };

  if (a_can_rm && b_can_rm) emit_mode_branch(false, false, "m1_a_rm && m1_b_rm");
  if (a_can_cm && b_can_rm) emit_mode_branch(true, false, "m1_a_cm && m1_b_rm");
  if (a_can_rm && b_can_cm) emit_mode_branch(false, true, "m1_a_rm && m1_b_cm");
  if (a_can_cm && b_can_cm) emit_mode_branch(true, true, "m1_a_cm && m1_b_cm");

  if (!layout_compile_time_known) {
    if (!first_branch) {
      os << "      } else {\n";
      emit_safe_k_loop(os, plan, "        ");
      os << "      }\n";
    } else {
      emit_safe_k_loop(os, plan, "      ");
    }
  }
  os << "    } else {\n";
  emit_safe_k_loop(os, plan, "      ");
  os << "    }\n"
     << "  }\n";

  if (plan.is_atomic) {
    uint32_t sg_ld = s.tn * 8u + 4u;
    uint32_t sg_tile_elems = (s.tm * 8u) * sg_ld;
    uint32_t max_sgs = std::max<uint32_t>(1u, (s.smem_bytes / ir::dtype_bytes(plan.acc_dt)) / sg_tile_elems);
    uint32_t epsg = (s.tm * 8u) * (s.tn * 8u);
    std::string m_cond = plan.m_bound.has_value() ? "(m1_gr < m1_M)" : "true";
    std::string n_cond = plan.n_bound.has_value() ? "(m1_gc < m1_N)" : "true";

    os << "  device atomic_float* m1_C_atom = reinterpret_cast<device atomic_float*>(m1_C);\n"
       << "  threadgroup float* m1_epi = reinterpret_cast<threadgroup float*>(m1_smem);\n"
       << "  for (uint m1_sb = 0u; m1_sb < " << s.num_sgs << "u; m1_sb += " << max_sgs << "u) {\n"
       << "    if (m1_simd_id >= m1_sb && m1_simd_id < m1_sb + " << max_sgs << "u) {\n"
       << "      threadgroup float* m1_sg_s = m1_epi + (m1_simd_id - m1_sb) * " << sg_tile_elems << "u;\n";
    for (uint32_t i = 0; i < s.tm; ++i) {
      for (uint32_t j = 0; j < s.tn; ++j) {
        os << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_sg_s + " << (i * 8u) << "u * " << sg_ld
           << "u + " << (j * 8u) << "u, " << sg_ld << ");\n";
      }
    }
    os << "    }\n"
       << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "    uint m1_act = min(" << max_sgs << "u, " << s.num_sgs << "u - m1_sb);\n"
       << "    for (uint m1_idx = m1_tid_linear; m1_idx < m1_act * " << epsg << "u; m1_idx += " << s.num_threads
       << "u) {\n"
       << "      uint m1_lsg = m1_idx / " << epsg << "u;\n"
       << "      uint m1_rem = m1_idx % " << epsg << "u;\n"
       << "      uint m1_lr = m1_rem / " << (s.tn * 8u) << "u;\n"
       << "      uint m1_lc = m1_rem % " << (s.tn * 8u) << "u;\n"
       << "      uint m1_gsg = m1_sb + m1_lsg;\n"
       << "      uint m1_gr = m1_row0 + (m1_gsg / " << s.sg_n << "u) * " << (s.tm * 8u) << "u + m1_lr;\n"
       << "      uint m1_gc = m1_col0 + (m1_gsg % " << s.sg_n << "u) * " << (s.tn * 8u) << "u + m1_lc;\n"
       << "      if (" << m_cond << " && " << n_cond << ") {\n"
       << "        float m1_val = m1_epi[m1_lsg * " << sg_tile_elems << "u + m1_lr * " << sg_ld << "u + m1_lc];\n";
    if (plan.structure == MatrixStructure::UpperTriangular) {
      os << "        if (int(m1_gr) + " << plan.diagonal_offset << " <= int(m1_gc))\n  ";
    } else if (plan.structure == MatrixStructure::LowerTriangular) {
      os << "        if (int(m1_gc) <= int(m1_gr) + " << plan.diagonal_offset << ")\n  ";
    }
    os << "        atomic_fetch_add_explicit(m1_C_atom + long(m1_gr) * m1_scm + long(m1_gc) * m1_scn, m1_val, "
          "memory_order_relaxed);\n"
       << "      }\n"
       << "    }\n"
       << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
       << "  }\n"
       << "}\n";
    return os.str();
  }

  std::string diag_cond = "";
  if (plan.structure == MatrixStructure::UpperTriangular) {
    diag_cond = " && (int(m1_row0 + " + std::to_string(s.bm - 1u) + "u) + " + std::to_string(plan.diagonal_offset) +
                " <= int(m1_col0))";
  } else if (plan.structure == MatrixStructure::LowerTriangular) {
    diag_cond = " && (int(m1_col0 + " + std::to_string(s.bn - 1u) + "u) <= int(m1_row0) + " +
                std::to_string(plan.diagonal_offset) + ")";
  }

  if (plan.acc_dt == plan.c_elem_dt) {
    os << "  if (m1_full_m && m1_full_n && (m1_scn == 1l)" << diag_cond << ") {\n";
    for (uint32_t i = 0; i < s.tm; ++i) {
      for (uint32_t j = 0; j < s.tn; ++j) {
        os << "    simdgroup_store(m1_acc[" << i << "][" << j << "], m1_C + ulong(m1_row0 + m1_sg_row + " << (i * 8u)
           << "u) * ulong(m1_scm) + ulong(m1_col0 + m1_sg_col + " << (j * 8u) << "u), ulong(m1_scm));\n";
      }
    }
    os << "    return;\n"
       << "  }\n";
  }

  uint32_t sg_ld = s.tn * 8u + 4u;
  uint32_t sg_tile_elems = (s.tm * 8u) * sg_ld;
  uint32_t max_sgs = std::max<uint32_t>(1u, (s.smem_bytes / ir::dtype_bytes(plan.acc_dt)) / sg_tile_elems);
  uint32_t epsg = (s.tm * 8u) * (s.tn * 8u);
  std::string m_cond = plan.m_bound.has_value() ? "(m1_gr < m1_M)" : "true";
  std::string n_cond = plan.n_bound.has_value() ? "(m1_gc < m1_N)" : "true";
  std::string c_t = msl_scalar_type(plan.c_elem_dt);

  os << "  threadgroup " << acc_t << "* m1_epi = reinterpret_cast<threadgroup " << acc_t << "*>(m1_smem);\n"
     << "  for (uint m1_sb = 0u; m1_sb < " << s.num_sgs << "u; m1_sb += " << max_sgs << "u) {\n"
     << "    if (m1_simd_id >= m1_sb && m1_simd_id < m1_sb + " << max_sgs << "u) {\n"
     << "      threadgroup " << acc_t << "* m1_sg_s = m1_epi + (m1_simd_id - m1_sb) * " << sg_tile_elems << "u;\n";
  for (uint32_t i = 0; i < s.tm; ++i) {
    for (uint32_t j = 0; j < s.tn; ++j) {
      os << "      simdgroup_store(m1_acc[" << i << "][" << j << "], m1_sg_s + " << (i * 8u) << "u * " << sg_ld
         << "u + " << (j * 8u) << "u, " << sg_ld << ");\n";
    }
  }
  os << "    }\n"
     << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "    uint m1_act = min(" << max_sgs << "u, " << s.num_sgs << "u - m1_sb);\n"
     << "    for (uint m1_idx = m1_tid_linear; m1_idx < m1_act * " << epsg << "u; m1_idx += " << s.num_threads
     << "u) {\n"
     << "      uint m1_lsg = m1_idx / " << epsg << "u;\n"
     << "      uint m1_rem = m1_idx % " << epsg << "u;\n"
     << "      uint m1_lr = m1_rem / " << (s.tn * 8u) << "u;\n"
     << "      uint m1_lc = m1_rem % " << (s.tn * 8u) << "u;\n"
     << "      uint m1_gsg = m1_sb + m1_lsg;\n"
     << "      uint m1_gr = m1_row0 + (m1_gsg / " << s.sg_n << "u) * " << (s.tm * 8u) << "u + m1_lr;\n"
     << "      uint m1_gc = m1_col0 + (m1_gsg % " << s.sg_n << "u) * " << (s.tn * 8u) << "u + m1_lc;\n"
     << "      if (" << m_cond << " && " << n_cond << ") {\n"
     << "        " << acc_t << " m1_val = m1_epi[m1_lsg * " << sg_tile_elems << "u + m1_lr * " << sg_ld
     << "u + m1_lc];\n";
  if (plan.structure == MatrixStructure::UpperTriangular) {
    os << "        m1_val = (int(m1_gr) + " << plan.diagonal_offset << " <= int(m1_gc)) ? m1_val : " << acc_t
       << "(0.0f);\n";
  } else if (plan.structure == MatrixStructure::LowerTriangular) {
    os << "        m1_val = (int(m1_gc) <= int(m1_gr) + " << plan.diagonal_offset << ") ? m1_val : " << acc_t
       << "(0.0f);\n";
  }
  os << "        m1_C[long(m1_gr) * m1_scm + long(m1_gc) * m1_scn] = " << c_t << "(m1_val);\n"
     << "      }\n"
     << "    }\n"
     << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     << "  }\n"
     << "}\n";
  return os.str();
}

}  // namespace

CompiledMSL try_emit_sgmm_kernel(const ir::Kernel& kernel, const ir::TargetSpec& target, bool* matched_out) {
  if (matched_out) *matched_out = false;

  SgmmPlan plan;
  if (!match_sgmm_plan(kernel, target, plan)) {
    return CompiledMSL{false, "", {}, ""};
  }

  std::string msl = emit_sgmm_msl(kernel, plan);
  if (msl.empty()) {
    return CompiledMSL{false, "", {}, ""};
  }

  CompiledMSL res;
  res.ok = true;
  res.msl_source = std::move(msl);
  res.launch_spec.entry_name = kernel.name;
  res.launch_spec.tg_x = plan.sched.num_threads;
  res.launch_spec.tg_y = 1;
  res.launch_spec.tg_z = 1;
  res.launch_spec.threadgroup_mem_bytes = 0;
  res.launch_spec.vec_width = 4;
  res.launch_spec.fast_math = false;
  res.launch_spec.param_slots = std::move(plan.params.slots);
  if (matched_out) *matched_out = true;
  return res;
}

}  // namespace m1triton::codegen::metal

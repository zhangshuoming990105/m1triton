#include "passes/conv_lowering.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace m1triton::passes {

namespace {

constexpr uint32_t kDefaultSimdgroups = 4;
constexpr uint32_t kMaxFragsPerSimdgroup = 16;
constexpr uint32_t kDepthwiseTileH = 32;
constexpr uint32_t kDepthwiseTileW = 64;
constexpr uint32_t kDepthwiseHaloBudget = 16384;

bool normalize_vec(std::vector<int32_t>& v, uint32_t ndim, int32_t fill) {
  if (v.size() == ndim) return true;
  if (v.empty()) {
    v.assign(ndim, fill);
    return true;
  }
  if (v.size() == 1) {
    const int32_t scalar = v[0];
    v.assign(ndim, scalar);
    return true;
  }
  return false;
}

std::string validate(const ir::ConvAttr& ca) {
  for (uint32_t i = 0; i < ca.ndim; ++i) {
    if (ca.kernel_size[i] <= 0) return "kernel_size must be positive";
    if (ca.stride[i] <= 0) return "stride must be positive";
    if (ca.dilation[i] <= 0) return "dilation must be positive";
    if (ca.padding[i] < 0) return "padding must be non-negative";
    if (ca.output_padding[i] < 0) return "output_padding must be non-negative";
  }
  if (ca.groups <= 0) return "groups must be positive";
  return {};
}

std::optional<int64_t> constexpr_operand(const ir::Kernel& kernel, ir::ValueId vid) {
  const ir::Value& v = kernel.value(vid);
  if (v.def_op != ir::kInvalidId) {
    const ir::Op& def = kernel.op(v.def_op);
    if (def.kind == ir::OpKind::Cast && def.operands.size() == 1) return constexpr_operand(kernel, def.operands[0]);
    if (def.kind != ir::OpKind::Constant) return std::nullopt;
    const auto* ca = std::get_if<ir::ConstAttr>(&def.attr);
    if (ca == nullptr) return std::nullopt;
    if (const auto* i = std::get_if<int64_t>(&ca->val)) return *i;
    if (const auto* u = std::get_if<uint64_t>(&ca->val)) return static_cast<int64_t>(*u);
    return std::nullopt;
  }
  if (v.def_block != kernel.entry_block || v.arg_idx >= kernel.params.size()) return std::nullopt;
  const ir::KernelParam& kp = kernel.params[v.arg_idx];
  if (!kp.is_constexpr || kp.type.is_ptr()) return std::nullopt;
  return kp.constexpr_val;
}

std::optional<int64_t> constexpr_param(const ir::Kernel& kernel, const char* name) {
  for (const auto& kp : kernel.params) {
    if (kp.is_constexpr && kp.name == name && kp.constexpr_val > 0) return kp.constexpr_val;
  }
  return std::nullopt;
}

uint32_t snap_tile(uint32_t v) {
  if (v <= 16) return 16;
  if (v <= 32) return 32;
  if (v <= 64) return 64;
  return 128;
}

uint32_t snap_bk(uint32_t v) {
  if (v <= 8) return 8;
  if (v <= 16) return 16;
  return 32;
}

uint32_t snap_simdgroups(uint32_t v) { return v >= 8 ? 8 : 4; }

uint32_t gemm_smem_bytes(uint32_t bm, uint32_t bn, uint32_t bk) {
  return (bm * (bk + 4) + bk * (bn + 4)) * 4;
}

void sg_grid(uint32_t bm, uint32_t bn, uint32_t sgs, uint32_t& sg_m, uint32_t& sg_n) {
  if (sgs == 8 && (bm >= 32 || bn >= 32)) {
    if (bm >= bn && bm >= 32) {
      sg_m = 4;
      sg_n = 2;
    } else {
      sg_m = 2;
      sg_n = 4;
    }
  } else {
    sg_m = 2;
    sg_n = 2;
  }
}

void schedule_gemm(ir::ConvAttr& ca, const ir::Kernel& kernel, const ir::Op& op, const PassContext& ctx) {
  const std::optional<ir::TileSpecAttr>& hint =
      op.hints.tile_spec.has_value() ? op.hints.tile_spec : kernel.hints.tile_spec;
  const size_t dim_base = ca.has_bias ? 4 : 3;
  std::optional<int64_t> cog;
  std::optional<int64_t> cig;
  if (op.operands.size() > dim_base + 2) {
    if (auto cin = constexpr_operand(kernel, op.operands[dim_base + 1])) cig = *cin / std::max(1, ca.groups);
    if (auto cout = constexpr_operand(kernel, op.operands[dim_base + 2])) cog = *cout / std::max(1, ca.groups);
  }
  int64_t ks = 1;
  for (uint32_t i = 0; i < ca.ndim; ++i) ks *= ca.kernel_size[i];
  const int64_t k_total = cig && *cig > 0 ? (*cig * ks) : 0;

  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t sgs = kDefaultSimdgroups;

  if (ca.is_pointwise) {
    if (cog && *cog <= 16) {
      bm = 16;
    } else if (cog && *cog <= 32) {
      bm = 32;
    }
    bk = (k_total > 0 && k_total % 32 == 0) ? 32 : 16;
  } else if (cog && *cog >= 128 && (*cog % 128 == 0) && k_total >= 192 && (k_total % 32 == 0)) {
    bm = 128;
    bn = 64;
    bk = 32;
    sgs = 8;
  } else if (cog && *cog <= 16) {
    bm = 16;
  } else if (cog && (*cog <= 32 || (*cog % 32 == 0 && *cog % 64 != 0))) {
    bm = 32;
  }

  if (hint.has_value()) {
    if (hint->bm > 0) bm = snap_tile(hint->bm);
    if (hint->bn > 0) bn = snap_tile(hint->bn);
    if (hint->bk > 0) bk = snap_bk(hint->bk);
    if (hint->num_simdgroups > 0) sgs = snap_simdgroups(hint->num_simdgroups);
  }
  if (auto v = constexpr_param(kernel, "BLOCK_M")) bm = snap_tile(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "BLOCK_N")) bn = snap_tile(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "BLOCK_K")) bk = snap_bk(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "NUM_SG")) sgs = snap_simdgroups(static_cast<uint32_t>(*v));

  if (bm == 16 && bn == 16) sgs = 4;
  if (bm * bn > kMaxFragsPerSimdgroup * 64 * sgs && sgs < 8 && (bm >= 32 || bn >= 32)) sgs = 8;

  uint32_t sg_m = 2;
  uint32_t sg_n = 2;
  sg_grid(bm, bn, sgs, sg_m, sg_n);
  while ((bm / (sg_m * 8)) * (bn / (sg_n * 8)) > kMaxFragsPerSimdgroup) {
    if (bn >= bm && bn > sg_n * 8) {
      bn /= 2;
    } else if (bm > sg_m * 8) {
      bm /= 2;
    } else {
      break;
    }
    sg_grid(bm, bn, sgs, sg_m, sg_n);
  }
  while (gemm_smem_bytes(bm, bn, bk) > ctx.target.max_threadgroup_memory_bytes && bk > 8) bk /= 2;
  while (gemm_smem_bytes(bm, bn, bk) > ctx.target.max_threadgroup_memory_bytes && bn > 16) bn /= 2;

  ca.bm = bm;
  ca.bn = bn;
  ca.bk = bk;
  ca.num_simdgroups = sgs;
  ca.use_sram_halo = false;
}

void schedule_depthwise(ir::ConvAttr& ca, const ir::Kernel& kernel, const ir::Op& op) {
  const std::optional<ir::TileSpecAttr>& hint =
      op.hints.tile_spec.has_value() ? op.hints.tile_spec : kernel.hints.tile_spec;
  uint32_t th = kDepthwiseTileH;
  uint32_t tw = kDepthwiseTileW;
  if (hint.has_value()) {
    if (hint->bm > 0) th = hint->bm;
    if (hint->bn > 0) tw = hint->bn;
  }
  if (auto v = constexpr_param(kernel, "BLOCK_H")) th = static_cast<uint32_t>(*v);
  if (auto v = constexpr_param(kernel, "BLOCK_W")) tw = static_cast<uint32_t>(*v);
  const uint32_t nd = ca.ndim;
  const int32_t kh = nd >= 2 ? ca.kernel_size[nd - 2] : 1;
  const int32_t sh = nd >= 2 ? ca.stride[nd - 2] : 1;
  const int32_t dh = nd >= 2 ? ca.dilation[nd - 2] : 1;
  const int64_t halo_h = static_cast<int64_t>(th - 1) * sh + static_cast<int64_t>(dh) * (kh - 1) + 1;
  const int64_t halo_w = static_cast<int64_t>(tw - 1) * ca.stride[nd - 1] +
                         static_cast<int64_t>(ca.dilation[nd - 1]) * (ca.kernel_size[nd - 1] - 1) + 1;
  ca.bm = th;
  ca.bn = tw;
  ca.bk = 1;
  ca.num_simdgroups = std::max<uint32_t>(1, (th * tw) / 128);
  ca.num_simdgroups = std::min<uint32_t>(ca.num_simdgroups, 32);
  ca.use_sram_halo = halo_h * halo_w * 4 <= static_cast<int64_t>(kDepthwiseHaloBudget);
}

void schedule_transposed(ir::ConvAttr& ca, const ir::Kernel& kernel, const ir::Op& op, const PassContext& ctx) {
  const std::optional<ir::TileSpecAttr>& hint =
      op.hints.tile_spec.has_value() ? op.hints.tile_spec : kernel.hints.tile_spec;
  uint32_t bm = 64;
  uint32_t bn = 64;
  uint32_t bk = 16;
  uint32_t sgs = kDefaultSimdgroups;
  if (hint.has_value()) {
    if (hint->bm > 0) bm = snap_tile(hint->bm);
    if (hint->bn > 0) bn = snap_tile(hint->bn);
    if (hint->bk > 0) bk = snap_bk(hint->bk);
    if (hint->num_simdgroups > 0) sgs = std::min<uint32_t>(hint->num_simdgroups, 32);
  }
  if (auto v = constexpr_param(kernel, "BLOCK_M")) bm = snap_tile(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "BLOCK_N")) bn = snap_tile(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "BLOCK_K")) bk = snap_bk(static_cast<uint32_t>(*v));
  if (auto v = constexpr_param(kernel, "NUM_SG")) sgs = std::min<uint32_t>(static_cast<uint32_t>(*v), 32);
  while (gemm_smem_bytes(bm, bn, bk) > ctx.target.max_threadgroup_memory_bytes && bk > 8) bk /= 2;
  while (gemm_smem_bytes(bm, bn, bk) > ctx.target.max_threadgroup_memory_bytes && bn > 16) bn /= 2;
  ca.bm = bm;
  ca.bn = bn;
  ca.bk = bk;
  ca.num_simdgroups = sgs;
  ca.use_sram_halo = false;
}

}  // namespace

PassResult ConvLoweringPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  for (auto& op_ptr : kernel.ops) {
    ir::Op& op = *op_ptr;
    if (op.erased || op.kind != ir::OpKind::ImplicitGemmConv) continue;
    auto* ca = std::get_if<ir::ConvAttr>(&op.attr);
    if (ca == nullptr || ca->ndim < 1 || ca->ndim > 3) {
      res.ok = false;
      res.diagnostic = "implicit_gemm_conv requires ConvAttr with ndim in [1, 3]";
      return res;
    }
    const ir::ConvAttr before = *ca;
    const uint32_t nd = ca->ndim;
    if (!normalize_vec(ca->kernel_size, nd, 1) || !normalize_vec(ca->stride, nd, 1) ||
        !normalize_vec(ca->padding, nd, 0) || !normalize_vec(ca->dilation, nd, 1) ||
        !normalize_vec(ca->output_padding, nd, 0)) {
      res.ok = false;
      res.diagnostic = "implicit_gemm_conv: vector attribute length mismatch with ndim";
      return res;
    }
    if (const std::string err = validate(*ca); !err.empty()) {
      res.ok = false;
      res.diagnostic = "implicit_gemm_conv: " + err;
      return res;
    }

    if (ca->transposed) {
      schedule_transposed(*ca, kernel, op, ctx);
    } else if (ca->is_depthwise) {
      schedule_depthwise(*ca, kernel, op);
    } else {
      bool unit = true;
      for (uint32_t i = 0; i < nd; ++i) {
        unit = unit && ca->kernel_size[i] == 1 && ca->stride[i] == 1 && ca->padding[i] == 0;
      }
      ca->is_pointwise = unit;
      schedule_gemm(*ca, kernel, op, ctx);
    }
    if (!(*ca == before)) res.changed = true;

    if (op.operands.size() > 2) {
      const ir::Value& ov = kernel.value(op.operands[2]);
      if (ov.def_op == ir::kInvalidId && ov.def_block == kernel.entry_block && ov.arg_idx < kernel.params.size()) {
        if (!kernel.params[ov.arg_idx].writable) {
          kernel.params[ov.arg_idx].writable = true;
          res.changed = true;
        }
      }
    }
  }
  return res;
}

}  // namespace m1triton::passes

#include "passes/simdgroup_matmul.hpp"

#include <cstdint>
#include <string>

#include "passes/threadgroup_staging.hpp"

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

bool validate_swizzle_and_pad(const LayoutAttr& layout, std::string_view side, std::string& err) {
  switch (layout.swizzle) {
    case SwizzleMode::None:
      if (layout.pad_elems != 0) {
        err = std::string(side) + ": SwizzleMode::None requires pad_elems == 0";
        return false;
      }
      return true;
    case SwizzleMode::PadBank4:
      if (layout.pad_elems != 4) {
        err = std::string(side) + ": SwizzleMode::PadBank4 requires pad_elems == 4";
        return false;
      }
      return true;
    case SwizzleMode::PadBank8:
      if (layout.pad_elems != 8) {
        err = std::string(side) + ": SwizzleMode::PadBank8 requires pad_elems == 8";
        return false;
      }
      return true;
    case SwizzleMode::Xor8:
      return true;
  }
  return true;
}

bool same_tg_threads(const LayoutAttr& a, const LayoutAttr& b) {
  return a.tg_threads[0] == b.tg_threads[0] && a.tg_threads[1] == b.tg_threads[1] &&
         a.tg_threads[2] == b.tg_threads[2];
}

uint64_t total_tg_threads(const LayoutAttr& l) {
  return static_cast<uint64_t>(l.tg_threads[0]) * l.tg_threads[1] * l.tg_threads[2];
}

void collect_existing_alloc_bytes(const Kernel& kernel, BlockId blk_id, uint64_t& bytes) {
  for (OpId op_id : kernel.block(blk_id).ops) {
    const Op& node = kernel.op(op_id);
    if (node.erased) continue;
    if (node.kind == OpKind::ThreadgroupAlloc && std::holds_alternative<ThreadgroupAllocAttr>(node.attr)) {
      const auto& alloc = std::get<ThreadgroupAllocAttr>(node.attr);
      if (!alloc.shape.empty() && alloc.stages > 0) {
        uint64_t elems = 1;
        for (size_t i = 0; i + 1 < alloc.shape.size(); ++i) {
          elems *= static_cast<uint64_t>(alloc.shape[i]);
        }
        elems *= static_cast<uint64_t>(alloc.shape.back()) + alloc.pad_elems;
        bytes += elems * dtype_bytes(alloc.elem) * alloc.stages;
      }
    }
    for (BlockId reg_id : node.regions) {
      collect_existing_alloc_bytes(kernel, reg_id, bytes);
    }
  }
}

bool validate_block(const Kernel& kernel, BlockId blk_id, const TargetSpec& target, uint64_t existing_alloc_bytes,
                    uint64_t& total_sgmm_sram, std::string& err) {
  for (OpId op_id : kernel.block(blk_id).ops) {
    const Op& node = kernel.op(op_id);
    if (node.erased) continue;

    if (node.kind == OpKind::SimdgroupMatmul) {
      if (!std::holds_alternative<SimdgroupMatmulAttr>(node.attr)) {
        err = "op " + std::to_string(op_id) + " (simdgroup_matmul): missing SimdgroupMatmulAttr";
        return false;
      }
      const auto& sm = std::get<SimdgroupMatmulAttr>(node.attr);
      if (sm.m_tiles == 0 || sm.n_tiles == 0 || sm.k_step == 0 || target.sgmm_k == 0 ||
          (sm.k_step % target.sgmm_k) != 0 || sm.ldm_a == 0 || sm.ldm_b == 0) {
        err = "op " + std::to_string(op_id) + " (simdgroup_matmul): invalid SGMM tiling parameters";
        return false;
      }
    } else if (node.kind == OpKind::Dot) {
      const std::string tag = "op " + std::to_string(op_id) + " (dot)";
      if (node.operands.size() < 2 || node.results.empty() || !std::holds_alternative<DotAttr>(node.attr)) {
        err = tag + ": malformed Dot op";
        return false;
      }
      const auto& da = std::get<DotAttr>(node.attr);
      const Type& lty = kernel.value(node.operands[0]).type;
      const Type& rty = kernel.value(node.operands[1]).type;
      const Type& oty = kernel.value(node.results[0]).type;
      if (!lty.is_tile() || !rty.is_tile() || !oty.is_tile()) {
        err = tag + ": operands and result must be tiles";
        return false;
      }

      const LayoutAttr& out_layout = oty.as_tile().layout;
      if (out_layout.kind == LayoutKind::Unassigned) {
        err = tag + ": result layout is Unassigned (run tile-layout before simdgroup-matmul)";
        return false;
      }

      const uint64_t threads = total_tg_threads(out_layout);
      if (threads == 0 || threads > target.max_threads_per_threadgroup ||
          (target.simd_width > 0 && (threads % target.simd_width) != 0)) {
        err = tag + ": invalid threadgroup thread count (" + std::to_string(threads) + ")";
        return false;
      }

      if (out_layout.kind == LayoutKind::Blocked) {
        // Non-SGMM fallback Dot (e.g. small tile shapes): thread budget verified above.
      } else if (out_layout.kind == LayoutKind::SimdgroupFragment) {
        if (target.sgmm_m == 0 || target.sgmm_n == 0 || target.sgmm_k == 0 || target.simd_width == 0) {
          err = tag + ": TargetSpec has zero SGMM dimensions or simd_width";
          return false;
        }
        if (!lty.is_float_elem() || !rty.is_float_elem() || !oty.is_float_elem()) {
          err = tag + ": SimdgroupFragment Dot requires float operands and result";
          return false;
        }

        const auto& ls = lty.as_tile().shape;
        const auto& rs = rty.as_tile().shape;
        const auto& os = oty.as_tile().shape;
        if ((os.size() != 2 && os.size() != 3) || ls.size() != os.size() || rs.size() != os.size()) {
          err = tag + ": SimdgroupFragment Dot requires 2D or 3D tile shapes";
          return false;
        }

        const size_t off = (os.size() == 3) ? 1 : 0;
        const int64_t bm = da.transpose_a ? ls[off + 1] : ls[off];
        const int64_t bk_lhs = da.transpose_a ? ls[off] : ls[off + 1];
        const int64_t bk_rhs = da.transpose_b ? rs[off + 1] : rs[off];
        const int64_t bn = da.transpose_b ? rs[off] : rs[off + 1];
        if (bk_lhs != bk_rhs || bm != os[off] || bn != os[off + 1]) {
          err = tag + ": Dot shape mismatch";
          return false;
        }

        if ((bm % target.sgmm_m) != 0 || (bn % target.sgmm_n) != 0 || (bk_lhs % target.sgmm_k) != 0) {
          err = tag + ": Dot shape [" + std::to_string(bm) + ", " + std::to_string(bn) + ", " +
                std::to_string(bk_lhs) + "] is not divisible by TargetSpec SGMM unit (" +
                std::to_string(target.sgmm_m) + "x" + std::to_string(target.sgmm_n) + "x" +
                std::to_string(target.sgmm_k) + ")";
          return false;
        }

        if (out_layout.frag_role != FragRole::Accumulator) {
          err = tag + ": SimdgroupFragment Dot result must have frag_role == Accumulator";
          return false;
        }

        const uint32_t sg_m = out_layout.simd_grid[0];
        const uint32_t sg_n = out_layout.simd_grid[1];
        if (sg_m == 0 || sg_n == 0) {
          err = tag + ": SimdgroupFragment simd_grid dimensions must be positive";
          return false;
        }
        const uint64_t expected_threads = static_cast<uint64_t>(sg_m) * sg_n * target.simd_width;
        if (threads != expected_threads) {
          err = tag + ": tg_threads (" + std::to_string(threads) + ") does not match simd_grid (" +
                std::to_string(sg_m) + "x" + std::to_string(sg_n) + ") * simd_width (" +
                std::to_string(target.simd_width) + ") = " + std::to_string(expected_threads);
          return false;
        }

        const int64_t unit_m = static_cast<int64_t>(sg_m) * target.sgmm_m;
        const int64_t unit_n = static_cast<int64_t>(sg_n) * target.sgmm_n;
        if ((bm % unit_m) != 0 || (bn % unit_n) != 0) {
          err = tag + ": Dot [BM, BN] is not evenly partitioned across simd_grid * sgmm unit";
          return false;
        }
        const uint32_t expected_ft_m = static_cast<uint32_t>(bm / unit_m);
        const uint32_t expected_ft_n = static_cast<uint32_t>(bn / unit_n);
        if (out_layout.frag_tiles_m != expected_ft_m || out_layout.frag_tiles_n != expected_ft_n ||
            out_layout.frag_tiles_m == 0 || out_layout.frag_tiles_n == 0) {
          err = tag + ": frag_tiles (" + std::to_string(out_layout.frag_tiles_m) + "x" +
                std::to_string(out_layout.frag_tiles_n) + ") does not match expected (" +
                std::to_string(expected_ft_m) + "x" + std::to_string(expected_ft_n) + ")";
          return false;
        }

        const LayoutAttr& lhs_layout = lty.as_tile().layout;
        const LayoutAttr& rhs_layout = rty.as_tile().layout;
        if (lhs_layout.kind != LayoutKind::ThreadgroupStaged || rhs_layout.kind != LayoutKind::ThreadgroupStaged) {
          err = tag + ": SimdgroupFragment Dot operands must have ThreadgroupStaged layout";
          return false;
        }
        if (lhs_layout.stages == 0 || rhs_layout.stages == 0 || lhs_layout.stages != rhs_layout.stages) {
          err = tag + ": ThreadgroupStaged operands must have matching positive stages";
          return false;
        }
        if (!same_tg_threads(lhs_layout, out_layout) || !same_tg_threads(rhs_layout, out_layout)) {
          err = tag + ": ThreadgroupStaged operand tg_threads must match Dot result tg_threads";
          return false;
        }
        if (!validate_swizzle_and_pad(lhs_layout, tag + " lhs", err) ||
            !validate_swizzle_and_pad(rhs_layout, tag + " rhs", err)) {
          return false;
        }

        const uint64_t dot_sram = ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(
            bm, bn, bk_lhs, lty.scalar_elem(), rty.scalar_elem(), lhs_layout.pad_elems, rhs_layout.pad_elems,
            lhs_layout.stages);
        total_sgmm_sram += dot_sram;
        if (existing_alloc_bytes + total_sgmm_sram > target.max_threadgroup_memory_bytes) {
          err = tag + ": total threadgroup SRAM (" + std::to_string(existing_alloc_bytes + total_sgmm_sram) +
                " bytes) exceeds target max_threadgroup_memory_bytes (" +
                std::to_string(target.max_threadgroup_memory_bytes) + " bytes)";
          return false;
        }
      } else {
        err = tag + ": unsupported LayoutKind on Dot result";
        return false;
      }
    }

    for (BlockId reg_id : node.regions) {
      if (!validate_block(kernel, reg_id, target, existing_alloc_bytes, total_sgmm_sram, err)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

std::optional<ir::SimdgroupMatmulAttr> SimdgroupMatrixLoweringPass::compute_sgmm_attr(const ir::Kernel& kernel,
                                                                                      const ir::Op& dot_op,
                                                                                      const ir::TargetSpec& target) {
  if (dot_op.erased || dot_op.kind != OpKind::Dot || dot_op.operands.size() < 2 || dot_op.results.empty() ||
      !std::holds_alternative<DotAttr>(dot_op.attr)) {
    return std::nullopt;
  }
  const auto& da = std::get<DotAttr>(dot_op.attr);
  const Type& lty = kernel.value(dot_op.operands[0]).type;
  const Type& rty = kernel.value(dot_op.operands[1]).type;
  const Type& oty = kernel.value(dot_op.results[0]).type;
  if (!lty.is_tile() || !rty.is_tile() || !oty.is_tile()) return std::nullopt;

  const LayoutAttr& out_layout = oty.as_tile().layout;
  const LayoutAttr& lhs_layout = lty.as_tile().layout;
  const LayoutAttr& rhs_layout = rty.as_tile().layout;
  if (out_layout.kind != LayoutKind::SimdgroupFragment || lhs_layout.kind != LayoutKind::ThreadgroupStaged ||
      rhs_layout.kind != LayoutKind::ThreadgroupStaged) {
    return std::nullopt;
  }

  const auto& ls = lty.as_tile().shape;
  const auto& rs = rty.as_tile().shape;
  const auto& os = oty.as_tile().shape;
  if ((os.size() != 2 && os.size() != 3) || ls.size() != os.size() || rs.size() != os.size()) {
    return std::nullopt;
  }
  const size_t off = (os.size() == 3) ? 1 : 0;
  const int64_t bk = da.transpose_a ? ls[off] : ls[off + 1];
  const int64_t bn = da.transpose_b ? rs[off] : rs[off + 1];

  SimdgroupMatmulAttr attr;
  attr.m_tiles = out_layout.frag_tiles_m;
  attr.n_tiles = out_layout.frag_tiles_n;
  attr.k_step = target.sgmm_k;
  attr.ldm_a = static_cast<uint32_t>(bk) + lhs_layout.pad_elems;
  attr.ldm_b = static_cast<uint32_t>(bn) + rhs_layout.pad_elems;
  attr.acc_dtype = da.acc_dtype;
  return attr;
}

PassResult SimdgroupMatrixLoweringPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  if (kernel.entry_block == kInvalidId) {
    return res;
  }

  uint64_t existing_alloc_bytes = 0;
  collect_existing_alloc_bytes(kernel, kernel.entry_block, existing_alloc_bytes);

  uint64_t total_sgmm_sram = 0;
  std::string err;
  if (!validate_block(kernel, kernel.entry_block, ctx.target, existing_alloc_bytes, total_sgmm_sram, err)) {
    res.ok = false;
    res.diagnostic = std::move(err);
  }
  return res;
}

}  // namespace m1triton::passes

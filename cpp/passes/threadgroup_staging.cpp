#include "passes/threadgroup_staging.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

void collect_sram_and_sgmm_dots(const Kernel& kernel, BlockId blk_id, uint64_t& existing_alloc_bytes,
                                std::vector<OpId>& sgmm_dots) {
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
        existing_alloc_bytes += elems * dtype_bytes(alloc.elem) * alloc.stages;
      }
    } else if (node.kind == OpKind::Dot && node.operands.size() >= 2 && !node.results.empty() &&
               std::holds_alternative<DotAttr>(node.attr)) {
      const Type& out_ty = kernel.value(node.results[0]).type;
      if (out_ty.is_tile() && out_ty.as_tile().layout.kind == LayoutKind::SimdgroupFragment) {
        sgmm_dots.push_back(op_id);
      }
    }

    for (BlockId reg_id : node.regions) {
      collect_sram_and_sgmm_dots(kernel, reg_id, existing_alloc_bytes, sgmm_dots);
    }
  }
}

bool apply_staged_layout(TileType& tile, const LayoutAttr& out_layout, SwizzleMode swizzle, uint32_t pad_elems,
                         uint32_t stages) {
  LayoutAttr layout = tile.layout;
  layout.kind = LayoutKind::ThreadgroupStaged;
  layout.tg_threads[0] = out_layout.tg_threads[0];
  layout.tg_threads[1] = out_layout.tg_threads[1];
  layout.tg_threads[2] = out_layout.tg_threads[2];
  layout.swizzle = swizzle;
  layout.pad_elems = pad_elems;
  layout.stages = stages;
  layout.frag_role = FragRole::Accumulator;
  layout.frag_tiles_m = 1;
  layout.frag_tiles_n = 1;
  layout.simd_grid[0] = 1;
  layout.simd_grid[1] = 1;

  if (tile.layout != layout) {
    tile.layout = layout;
    return true;
  }
  return false;
}

}  // namespace

uint32_t ThreadgroupStagingAndSwizzlePass::recommended_pad_elems(ir::ScalarDType dt) {
  return (dtype_bytes(dt) <= 2) ? 8u : 4u;
}

ir::SwizzleMode ThreadgroupStagingAndSwizzlePass::recommended_swizzle_mode(ir::ScalarDType dt) {
  return (recommended_pad_elems(dt) == 8u) ? ir::SwizzleMode::PadBank8 : ir::SwizzleMode::PadBank4;
}

uint64_t ThreadgroupStagingAndSwizzlePass::compute_dot_sram_bytes(int64_t bm, int64_t bn, int64_t bk,
                                                                  ir::ScalarDType dt_a, ir::ScalarDType dt_b,
                                                                  uint32_t pad_a, uint32_t pad_b, uint32_t stages) {
  if (bm <= 0 || bn <= 0 || bk <= 0 || stages == 0) return 0;
  const uint64_t bytes_a =
      static_cast<uint64_t>(bm) * (static_cast<uint64_t>(bk) + pad_a) * dtype_bytes(dt_a);
  const uint64_t bytes_b =
      static_cast<uint64_t>(bk) * (static_cast<uint64_t>(bn) + pad_b) * dtype_bytes(dt_b);
  return (bytes_a + bytes_b) * stages;
}

PassResult ThreadgroupStagingAndSwizzlePass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  if (kernel.entry_block == kInvalidId) {
    return res;
  }

  uint64_t existing_alloc_bytes = 0;
  std::vector<OpId> sgmm_dots;
  collect_sram_and_sgmm_dots(kernel, kernel.entry_block, existing_alloc_bytes, sgmm_dots);
  if (sgmm_dots.empty()) {
    return res;
  }

  const uint64_t sram_limit = ctx.target.max_threadgroup_memory_bytes;
  if (existing_alloc_bytes > sram_limit) {
    res.ok = false;
    res.diagnostic = "Existing threadgroup allocations (" + std::to_string(existing_alloc_bytes) +
                     " bytes) exceed target max_threadgroup_memory_bytes (" + std::to_string(sram_limit) + " bytes)";
    return res;
  }

  uint64_t cumulative_dot_sram = 0;

  for (OpId op_id : sgmm_dots) {
    const Op& node = kernel.op(op_id);
    const auto& da = std::get<DotAttr>(node.attr);
    const ValueId lhs_id = node.operands[0];
    const ValueId rhs_id = node.operands[1];
    const ValueId res_id = node.results[0];

    Type& lhs_ty = kernel.value(lhs_id).type;
    Type& rhs_ty = kernel.value(rhs_id).type;
    const Type& out_ty = kernel.value(res_id).type;

    if (!lhs_ty.is_tile() || !rhs_ty.is_tile() || !out_ty.is_tile() || lhs_ty.is_ptr_elem() ||
        rhs_ty.is_ptr_elem() || out_ty.is_ptr_elem()) {
      res.ok = false;
      res.diagnostic = "SGMM Dot op " + std::to_string(op_id) + " requires non-pointer tile operands and result";
      return res;
    }

    const auto& ls = lhs_ty.as_tile().shape;
    const auto& rs = rhs_ty.as_tile().shape;
    const auto& os = out_ty.as_tile().shape;
    if ((os.size() != 2 && os.size() != 3) || ls.size() != os.size() || rs.size() != os.size()) {
      res.ok = false;
      res.diagnostic = "SGMM Dot op " + std::to_string(op_id) + " requires 2D or 3D tile operands";
      return res;
    }

    const size_t off = (os.size() == 3) ? 1 : 0;
    const int64_t bm = da.transpose_a ? ls[off + 1] : ls[off];
    const int64_t bk_lhs = da.transpose_a ? ls[off] : ls[off + 1];
    const int64_t bk_rhs = da.transpose_b ? rs[off + 1] : rs[off];
    const int64_t bn = da.transpose_b ? rs[off] : rs[off + 1];
    if (bk_lhs != bk_rhs || bm != os[off] || bn != os[off + 1]) {
      res.ok = false;
      res.diagnostic = "SGMM Dot op " + std::to_string(op_id) + " has mismatched matrix dimensions";
      return res;
    }

    const ScalarDType dt_a = lhs_ty.scalar_elem();
    const ScalarDType dt_b = rhs_ty.scalar_elem();
    const uint32_t pad_a = recommended_pad_elems(dt_a);
    const uint32_t pad_b = recommended_pad_elems(dt_b);
    const SwizzleMode swizzle_a = recommended_swizzle_mode(dt_a);
    const SwizzleMode swizzle_b = recommended_swizzle_mode(dt_b);

    const uint64_t single_stage_bytes = compute_dot_sram_bytes(bm, bn, bk_lhs, dt_a, dt_b, pad_a, pad_b, 1);
    const std::optional<TileSpecAttr> ts =
        node.hints.tile_spec.has_value() ? node.hints.tile_spec : kernel.hints.tile_spec;
    const bool wants_double_buffer = ts.has_value() && ts->double_buffer;

    uint32_t stages = 1;
    if (wants_double_buffer) {
      const uint64_t double_stage_bytes = single_stage_bytes * 2;
      if (existing_alloc_bytes + cumulative_dot_sram + double_stage_bytes <= sram_limit) {
        stages = 2;
      } else {
        stages = 1;
      }
    }

    const uint64_t total_after_dot = existing_alloc_bytes + cumulative_dot_sram + (single_stage_bytes * stages);
    if (total_after_dot > sram_limit) {
      res.ok = false;
      res.diagnostic = "SGMM Dot op " + std::to_string(op_id) + " threadgroup staging requires " +
                       std::to_string(total_after_dot) + " bytes (stages=" + std::to_string(stages) +
                       "), exceeding target max_threadgroup_memory_bytes (" + std::to_string(sram_limit) + " bytes)";
      return res;
    }
    cumulative_dot_sram += single_stage_bytes * stages;

    const LayoutAttr& out_layout = out_ty.as_tile().layout;
    if (apply_staged_layout(lhs_ty.as_tile_mut(), out_layout, swizzle_a, pad_a, stages)) {
      res.changed = true;
    }
    if (apply_staged_layout(rhs_ty.as_tile_mut(), out_layout, swizzle_b, pad_b, stages)) {
      res.changed = true;
    }
  }

  // Keep kernel.params in sync with entry block arg types if any param is a tile.
  const auto& entry_args = kernel.block(kernel.entry_block).args;
  for (size_t i = 0; i < entry_args.size() && i < kernel.params.size(); ++i) {
    kernel.params[i].type = kernel.value(entry_args[i]).type;
  }

  return res;
}

}  // namespace m1triton::passes

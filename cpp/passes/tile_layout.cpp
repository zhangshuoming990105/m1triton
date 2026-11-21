#include "passes/tile_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "passes/contiguity_analysis.hpp"

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

struct KernelTileSummary {
  bool has_tiles = false;
  bool has_dot = false;
  bool has_2d = false;
  int64_t dot_m = 0;
  int64_t dot_n = 0;
  int64_t max_2d_m = 0;
  int64_t max_2d_n = 0;
  int64_t max_1d_len = 0;
  uint32_t max_vec_width = 1;
  std::unordered_map<ValueId, uint32_t> value_vec_width;
  std::vector<ValueId> tile_values;
};

void record_tile_value(const Kernel& k, ValueId vid, KernelTileSummary& summary) {
  const Type& ty = k.value(vid).type;
  if (!ty.is_tile()) return;
  summary.has_tiles = true;
  summary.tile_values.push_back(vid);
  const auto& shape = ty.as_tile().shape;
  if (shape.size() == 1) {
    summary.max_1d_len = std::max(summary.max_1d_len, shape[0]);
  } else if (shape.size() >= 2) {
    summary.has_2d = true;
    const int64_t m = shape[shape.size() - 2];
    const int64_t n = shape[shape.size() - 1];
    if (m * n > summary.max_2d_m * summary.max_2d_n) {
      summary.max_2d_m = m;
      summary.max_2d_n = n;
    }
  }
}

void scan_block(const Kernel& k, BlockId blk_id, KernelTileSummary& summary) {
  const Block& blk = k.block(blk_id);
  for (ValueId arg : blk.args) {
    record_tile_value(k, arg, summary);
  }
  for (OpId op_id : blk.ops) {
    const Op& node = k.op(op_id);
    if (node.erased) continue;

    for (ValueId rid : node.results) {
      record_tile_value(k, rid, summary);
    }

    if ((node.kind == OpKind::Load || node.kind == OpKind::Store) &&
        std::holds_alternative<LoadStoreAttr>(node.attr)) {
      const uint32_t vw = std::get<LoadStoreAttr>(node.attr).vec_width;
      summary.max_vec_width = std::max(summary.max_vec_width, vw);
      if (!node.operands.empty()) {
        summary.value_vec_width[node.operands[0]] = std::max(summary.value_vec_width[node.operands[0]], vw);
      }
      if (node.kind == OpKind::Load && !node.results.empty()) {
        summary.value_vec_width[node.results[0]] = std::max(summary.value_vec_width[node.results[0]], vw);
      } else if (node.kind == OpKind::Store && node.operands.size() >= 2) {
        summary.value_vec_width[node.operands[1]] = std::max(summary.value_vec_width[node.operands[1]], vw);
      }
    } else if (node.kind == OpKind::Dot && !node.results.empty()) {
      const Type& out_ty = k.value(node.results[0]).type;
      if (out_ty.is_tile() && out_ty.as_tile().shape.size() >= 2) {
        const auto& shape = out_ty.as_tile().shape;
        summary.has_dot = true;
        const int64_t m = shape[shape.size() - 2];
        const int64_t n = shape[shape.size() - 1];
        if (m * n > summary.dot_m * summary.dot_n) {
          summary.dot_m = m;
          summary.dot_n = n;
        }
      }
    }

    for (BlockId reg_id : node.regions) {
      scan_block(k, reg_id, summary);
    }
  }
}

uint32_t clamp_valid_vec_width(uint32_t desired, uint32_t limit) {
  const uint32_t v = std::min(desired, limit);
  if (v >= 4) return 4;
  if (v >= 2) return 2;
  return 1;
}

uint32_t scalar_bytes_of_tile(const Type& ty) {
  if (ty.is_ptr_elem()) {
    return dtype_bytes(ty.ptr_elem().pointee);
  }
  return dtype_bytes(ty.scalar_elem());
}

}  // namespace

PassResult TileAndLayoutAssignmentPass::run(ir::Kernel& kernel, const PassContext& ctx) {
  PassResult res;
  if (kernel.entry_block == kInvalidId) {
    return res;
  }

  KernelTileSummary summary;
  scan_block(kernel, kernel.entry_block, summary);
  if (!summary.has_tiles) {
    return res;
  }

  const uint32_t simd = std::max<uint32_t>(1, ctx.target.simd_width);
  const uint32_t max_tg_hw = std::max(simd, ctx.target.max_threads_per_threadgroup);

  std::array<uint32_t, 3> tg = {simd, 1, 1};
  const bool is_2d_kernel = summary.has_dot || summary.has_2d;

  if (is_2d_kernel) {
    const int64_t bm = summary.has_dot ? summary.dot_m : summary.max_2d_m;
    const int64_t bn = summary.has_dot ? summary.dot_n : summary.max_2d_n;
    if (bm * bn <= 256 && 64 <= max_tg_hw && (64 % simd == 0)) {
      tg = {8, 8, 1};
    } else if (256 <= max_tg_hw && (256 % simd == 0)) {
      tg = {16, 16, 1};
    } else {
      tg = {simd, 1, 1};
    }
  } else {
    const uint32_t max_1d_tg = std::min<uint32_t>(256, max_tg_hw);
    const uint32_t vec = std::max<uint32_t>(1, summary.max_vec_width);
    const uint32_t work_items = std::max<uint32_t>(1, static_cast<uint32_t>(summary.max_1d_len / vec));
    uint32_t tg_x = std::clamp(work_items, simd, max_1d_tg);
    tg_x = ((tg_x + simd - 1) / simd) * simd;
    if (tg_x > max_1d_tg) {
      tg_x = (max_1d_tg / simd) * simd;
    }
    tg = {std::max(simd, tg_x), 1, 1};
  }

  const auto facts = ContiguityAndAlignmentAnalysis::analyze(kernel);

  for (ValueId vid : summary.tile_values) {
    Type& ty = kernel.value(vid).type;
    if (!ty.is_tile()) continue;

    TileType& tile = ty.as_tile_mut();
    LayoutAttr layout = tile.layout;
    layout.kind = LayoutKind::Blocked;
    layout.tg_threads[0] = tg[0];
    layout.tg_threads[1] = tg[1];
    layout.tg_threads[2] = tg[2];

    if (auto fit = facts.find(vid); fit != facts.end()) {
      layout.contiguity_elems = std::max<uint32_t>(1, fit->second.contiguity);
    }

    const uint32_t desired_vec =
        summary.value_vec_width.contains(vid) ? summary.value_vec_width[vid]
                                              : (is_2d_kernel ? 1u : summary.max_vec_width);
    const uint32_t elem_bytes = scalar_bytes_of_tile(ty);

    if (tile.shape.size() == 1) {
      const int64_t len = tile.shape[0];
      const uint32_t ept0 = std::max<uint32_t>(1, static_cast<uint32_t>((len + tg[0] - 1) / tg[0]));
      layout.elems_per_thread[0] = ept0;
      layout.elems_per_thread[1] = 1;
      layout.vec_width = is_2d_kernel ? 1u : clamp_valid_vec_width(desired_vec, ept0);
      layout.alignment_bytes = layout.vec_width * elem_bytes;
    } else {
      const int64_t m = tile.shape[tile.shape.size() - 2];
      const int64_t n = tile.shape[tile.shape.size() - 1];
      const uint32_t ept_m = std::max<uint32_t>(1, static_cast<uint32_t>((m + tg[1] - 1) / tg[1]));
      const uint32_t ept_n = std::max<uint32_t>(1, static_cast<uint32_t>((n + tg[0] - 1) / tg[0]));
      layout.elems_per_thread[0] = ept_m;
      layout.elems_per_thread[1] = ept_n;
      layout.vec_width = clamp_valid_vec_width(desired_vec, ept_n);
      layout.alignment_bytes = layout.vec_width * elem_bytes;
    }

    if (tile.layout != layout) {
      tile.layout = layout;
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

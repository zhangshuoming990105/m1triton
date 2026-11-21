#include "passes/tile_layout.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "passes/contiguity_analysis.hpp"

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

struct DotInfo {
  OpId op_id = kInvalidId;
  ValueId result_id = kInvalidId;
  int64_t bm = 0;
  int64_t bn = 0;
  int64_t bk = 0;
  bool is_float = false;
  std::optional<TileSpecAttr> tile_spec;
};

struct SgmmAssignment {
  uint32_t simd_grid_m = 2;
  uint32_t simd_grid_n = 2;
  uint32_t frag_tiles_m = 1;
  uint32_t frag_tiles_n = 1;
  uint32_t tg_threads_x = 128;
};

struct KernelTileSummary {
  bool has_tiles = false;
  bool has_dot = false;
  bool has_reduce = false;
  bool has_2d = false;
  int64_t dot_m = 0;
  int64_t dot_n = 0;
  int64_t max_reduce_dim = 0;
  int64_t max_2d_m = 0;
  int64_t max_2d_n = 0;
  int64_t max_1d_len = 0;
  uint32_t max_vec_width = 1;
  std::unordered_map<ValueId, uint32_t> value_vec_width;
  std::vector<ValueId> tile_values;
  std::vector<DotInfo> dots;
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
    } else if (node.kind == OpKind::Dot && node.operands.size() >= 2 && !node.results.empty() &&
               std::holds_alternative<DotAttr>(node.attr)) {
      const auto& da = std::get<DotAttr>(node.attr);
      const Type& lty = k.value(node.operands[0]).type;
      const Type& rty = k.value(node.operands[1]).type;
      const Type& out_ty = k.value(node.results[0]).type;
      if (lty.is_tile() && rty.is_tile() && out_ty.is_tile() && out_ty.as_tile().shape.size() >= 2) {
        const auto& ls = lty.as_tile().shape;
        const auto& rs = rty.as_tile().shape;
        const auto& os = out_ty.as_tile().shape;
        const size_t off = (os.size() == 3) ? 1 : 0;
        const int64_t bm = da.transpose_a ? ls[off + 1] : ls[off];
        const int64_t bk = da.transpose_a ? ls[off] : ls[off + 1];
        const int64_t bn = da.transpose_b ? rs[off] : rs[off + 1];

        summary.has_dot = true;
        if (bm * bn > summary.dot_m * summary.dot_n) {
          summary.dot_m = bm;
          summary.dot_n = bn;
        }

        DotInfo info;
        info.op_id = op_id;
        info.result_id = node.results[0];
        info.bm = bm;
        info.bn = bn;
        info.bk = bk;
        info.is_float = lty.is_float_elem() && rty.is_float_elem() && out_ty.is_float_elem();
        info.tile_spec = node.hints.tile_spec.has_value() ? node.hints.tile_spec : k.hints.tile_spec;
        summary.dots.push_back(info);
      }
    } else if (node.kind == OpKind::Reduce && !node.operands.empty() &&
               std::holds_alternative<ReduceAttr>(node.attr)) {
      const auto& ra = std::get<ReduceAttr>(node.attr);
      const Type& ity = k.value(node.operands[0]).type;
      if (ity.is_tile()) {
        const auto& shape = ity.as_tile().shape;
        if (ra.axis >= 0 && static_cast<size_t>(ra.axis) < shape.size()) {
          summary.has_reduce = true;
          summary.max_reduce_dim = std::max(summary.max_reduce_dim, shape[static_cast<size_t>(ra.axis)]);
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

bool is_dot_sgmm_eligible(const DotInfo& d, const TargetSpec& target, SgmmSchedulePolicy policy) {
  if (!d.is_float) return false;
  if (d.bm <= 0 || d.bn <= 0 || d.bk <= 0) return false;
  if (target.sgmm_m == 0 || target.sgmm_n == 0 || target.sgmm_k == 0) return false;
  if ((d.bm % 32) != 0 || (d.bn % 32) != 0 || (d.bk % 8) != 0) return false;
  if ((d.bm % target.sgmm_m) != 0 || (d.bn % target.sgmm_n) != 0 || (d.bk % target.sgmm_k) != 0) return false;
  if (policy == SgmmSchedulePolicy::HintGuided && !d.tile_spec.has_value()) return false;
  return true;
}

std::optional<SgmmAssignment> choose_sgmm_assignment(const DotInfo& d, const TargetSpec& target,
                                                     std::optional<uint32_t> forced_num_sg) {
  const uint32_t simd = std::max<uint32_t>(1, target.simd_width);
  const uint32_t max_tg = target.max_threads_per_threadgroup;

  uint32_t req_sg = 4;
  if (forced_num_sg.has_value()) {
    req_sg = *forced_num_sg;
  } else if (d.tile_spec.has_value() && d.tile_spec->num_simdgroups > 0) {
    req_sg = d.tile_spec->num_simdgroups;
  } else {
    req_sg = (d.bm >= 128 && d.bn >= 64) ? 8u : 4u;
  }

  std::vector<std::pair<uint32_t, uint32_t>> candidates;
  if (req_sg >= 16) {
    candidates.push_back({4, 4});
  }
  if (req_sg >= 8) {
    candidates.push_back({4, 2});
    candidates.push_back({2, 4});
  }
  if (req_sg >= 4) {
    candidates.push_back({2, 2});
  }
  if (req_sg >= 2) {
    candidates.push_back({2, 1});
    candidates.push_back({1, 2});
  }
  candidates.push_back({1, 1});

  for (const auto& [sg_m, sg_n] : candidates) {
    const uint64_t num_sg = static_cast<uint64_t>(sg_m) * sg_n;
    const uint64_t threads = num_sg * simd;
    if (threads == 0 || threads > max_tg || (threads % simd) != 0) continue;
    if (forced_num_sg.has_value() && num_sg != *forced_num_sg) continue;
    const int64_t unit_m = static_cast<int64_t>(sg_m) * target.sgmm_m;
    const int64_t unit_n = static_cast<int64_t>(sg_n) * target.sgmm_n;
    if ((d.bm % unit_m) != 0 || (d.bn % unit_n) != 0) continue;

    SgmmAssignment a;
    a.simd_grid_m = sg_m;
    a.simd_grid_n = sg_n;
    a.frag_tiles_m = static_cast<uint32_t>(d.bm / unit_m);
    a.frag_tiles_n = static_cast<uint32_t>(d.bn / unit_n);
    a.tg_threads_x = static_cast<uint32_t>(threads);
    return a;
  }
  return std::nullopt;
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

  std::unordered_map<ValueId, SgmmAssignment> sgmm_results;
  bool use_sgmm = false;
  if (!summary.dots.empty()) {
    bool all_eligible = true;
    const DotInfo* primary = &summary.dots[0];
    for (const auto& d : summary.dots) {
      if (!is_dot_sgmm_eligible(d, ctx.target, sgmm_policy_)) {
        all_eligible = false;
        break;
      }
      if (d.bm * d.bn > primary->bm * primary->bn) {
        primary = &d;
      }
    }
    if (all_eligible) {
      auto primary_assign = choose_sgmm_assignment(*primary, ctx.target, std::nullopt);
      if (primary_assign.has_value()) {
        const uint32_t kernel_sg = primary_assign->simd_grid_m * primary_assign->simd_grid_n;
        bool all_assigned = true;
        for (const auto& d : summary.dots) {
          auto da = choose_sgmm_assignment(d, ctx.target, kernel_sg);
          if (!da.has_value()) {
            all_assigned = false;
            break;
          }
          sgmm_results[d.result_id] = *da;
        }
        use_sgmm = all_assigned;
        if (!use_sgmm) {
          sgmm_results.clear();
        }
      }
    }
  }

  std::array<uint32_t, 3> tg = {simd, 1, 1};
  const bool is_reduce_kernel = !summary.has_dot && summary.has_reduce;
  const bool is_2d_kernel = !is_reduce_kernel && (summary.has_dot || summary.has_2d);

  if (use_sgmm) {
    const auto& first_assign = sgmm_results.begin()->second;
    tg = {first_assign.tg_threads_x, 1, 1};
  } else if (is_reduce_kernel) {
    uint32_t max_red_tg = std::min<uint32_t>(256u, max_tg_hw);
    max_red_tg = std::max(simd, (max_red_tg / simd) * simd);
    const uint32_t block_k = static_cast<uint32_t>(std::max<int64_t>(1, summary.max_reduce_dim));
    uint32_t tg_x = std::clamp(block_k, simd, max_red_tg);
    tg_x = ((tg_x + simd - 1) / simd) * simd;
    if (tg_x > max_red_tg) {
      tg_x = max_red_tg;
    }
    tg = {std::max(simd, tg_x), 1, 1};
  } else if (is_2d_kernel) {
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
    LayoutAttr layout;
    layout.contiguity_elems = tile.layout.contiguity_elems;
    if (auto fit = facts.find(vid); fit != facts.end()) {
      layout.contiguity_elems = std::max<uint32_t>(1, fit->second.contiguity);
    }

    const uint32_t elem_bytes = scalar_bytes_of_tile(ty);

    if (auto sit = sgmm_results.find(vid); sit != sgmm_results.end()) {
      const SgmmAssignment& sa = sit->second;
      layout.kind = LayoutKind::SimdgroupFragment;
      layout.tg_threads[0] = sa.tg_threads_x;
      layout.tg_threads[1] = 1;
      layout.tg_threads[2] = 1;
      layout.frag_role = FragRole::Accumulator;
      layout.simd_grid[0] = sa.simd_grid_m;
      layout.simd_grid[1] = sa.simd_grid_n;
      layout.frag_tiles_m = sa.frag_tiles_m;
      layout.frag_tiles_n = sa.frag_tiles_n;
      layout.elems_per_thread[0] = std::max<uint32_t>(1, sa.frag_tiles_m * 2);
      layout.elems_per_thread[1] = std::max<uint32_t>(1, sa.frag_tiles_n);
      layout.vec_width = 1;
      layout.alignment_bytes = elem_bytes;
    } else {
      layout.kind = LayoutKind::Blocked;
      layout.tg_threads[0] = tg[0];
      layout.tg_threads[1] = tg[1];
      layout.tg_threads[2] = tg[2];

      const uint32_t desired_vec =
          summary.value_vec_width.contains(vid)
              ? summary.value_vec_width[vid]
              : ((is_2d_kernel || use_sgmm) ? 1u : summary.max_vec_width);

      if (tile.shape.size() == 1) {
        const int64_t len = tile.shape[0];
        const uint32_t ept0 = std::max<uint32_t>(1, static_cast<uint32_t>((len + tg[0] - 1) / tg[0]));
        layout.elems_per_thread[0] = ept0;
        layout.elems_per_thread[1] = 1;
        layout.vec_width = (is_2d_kernel || use_sgmm) ? 1u : clamp_valid_vec_width(desired_vec, ept0);
        layout.alignment_bytes = layout.vec_width * elem_bytes;
      } else {
        const int64_t m = tile.shape[tile.shape.size() - 2];
        const int64_t n = tile.shape[tile.shape.size() - 1];
        const uint32_t ept_m = std::max<uint32_t>(1, static_cast<uint32_t>((m + tg[1] - 1) / tg[1]));
        const uint32_t ept_n = std::max<uint32_t>(1, static_cast<uint32_t>((n + tg[0] - 1) / tg[0]));
        layout.elems_per_thread[0] = ept_m;
        layout.elems_per_thread[1] = ept_n;
        uint32_t vec_limit = ept_n;
        if (use_sgmm && tg[1] == 1 && tg[2] == 1) {
          const uint32_t lin_ept =
              std::max<uint32_t>(1, static_cast<uint32_t>(((m * n) + tg[0] - 1) / tg[0]));
          vec_limit = std::min<uint32_t>(lin_ept, static_cast<uint32_t>(n));
        }
        layout.vec_width = clamp_valid_vec_width(desired_vec, vec_limit);
        layout.alignment_bytes = layout.vec_width * elem_bytes;
      }
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

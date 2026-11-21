#include "passes/structure_pruning.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

bool validate_structure_attr(const StructureAttr& s, std::string_view where, std::string& err) {
  switch (s.kind) {
    case MatrixStructure::Dense:
      if (s.diagonal_offset != 0 || s.is_unit_diagonal) {
        err = std::string(where) + ": Dense structure must not set diagonal_offset or is_unit_diagonal";
        return false;
      }
      return true;
    case MatrixStructure::Symmetric:
      if (s.diagonal_offset != 0 || s.is_unit_diagonal) {
        err = std::string(where) + ": Symmetric structure must not set diagonal_offset or is_unit_diagonal";
        return false;
      }
      return true;
    case MatrixStructure::CausalMask:
      if (s.is_unit_diagonal) {
        err = std::string(where) + ": CausalMask structure must not set is_unit_diagonal";
        return false;
      }
      return true;
    case MatrixStructure::UpperTriangular:
    case MatrixStructure::LowerTriangular:
    case MatrixStructure::Diagonal:
      if (s.is_unit_diagonal && s.diagonal_offset != 0) {
        err = std::string(where) + ": unit diagonal requires diagonal_offset == 0";
        return false;
      }
      return true;
  }
  return true;
}

StructureAttr effective_structure(const StructureAttr& s, bool transposed) {
  if (!transposed) return s;
  StructureAttr out = s;
  switch (s.kind) {
    case MatrixStructure::UpperTriangular:
      out.kind = MatrixStructure::LowerTriangular;
      out.diagonal_offset = -s.diagonal_offset;
      return out;
    case MatrixStructure::LowerTriangular:
      out.kind = MatrixStructure::UpperTriangular;
      out.diagonal_offset = -s.diagonal_offset;
      return out;
    case MatrixStructure::Diagonal:
      out.diagonal_offset = -s.diagonal_offset;
      return out;
    case MatrixStructure::Dense:
    case MatrixStructure::Symmetric:
    case MatrixStructure::CausalMask:
      return out;
  }
  return out;
}

bool resolve_operand_structure(const Kernel& k, ValueId vid, const StructureAttr& existing_attr,
                               std::string_view side, StructureAttr& out, std::string& err) {
  if (!validate_structure_attr(existing_attr, side, err)) {
    return false;
  }

  StructureAttr tile_struct{};
  const Value& val = k.value(vid);
  if (val.type.is_tile()) {
    tile_struct = val.type.as_tile().structure;
    if (!validate_structure_attr(tile_struct, side, err)) {
      return false;
    }
  }

  if (existing_attr.kind != MatrixStructure::Dense && tile_struct.kind != MatrixStructure::Dense &&
      existing_attr != tile_struct) {
    err = std::string(side) + ": conflicting StructureAttr between DotAttr and operand TileType";
    return false;
  }

  out = (existing_attr.kind != MatrixStructure::Dense) ? existing_attr : tile_struct;

  if (out.kind == MatrixStructure::Dense && val.def_op != kInvalidId && val.def_op < k.ops.size()) {
    const Op& def = k.op(val.def_op);
    if (!def.erased && def.hints.structure.has_value()) {
      if (!validate_structure_attr(*def.hints.structure, side, err)) {
        return false;
      }
      if (def.hints.structure->kind != MatrixStructure::Dense) {
        out = *def.hints.structure;
      }
    }
  }
  return true;
}

StructureAttr infer_dot_output_structure(const StructureAttr& eff_lhs, const StructureAttr& eff_rhs,
                                         bool same_ssa_operands) {
  if (eff_lhs.kind == MatrixStructure::UpperTriangular && eff_rhs.kind == MatrixStructure::UpperTriangular) {
    const int32_t off = eff_lhs.diagonal_offset + eff_rhs.diagonal_offset;
    const bool unit =
        (eff_lhs.diagonal_offset == 0 && eff_rhs.diagonal_offset == 0 && eff_lhs.is_unit_diagonal &&
         eff_rhs.is_unit_diagonal);
    return StructureAttr{MatrixStructure::UpperTriangular, off, unit};
  }

  if (eff_lhs.kind == MatrixStructure::LowerTriangular && eff_rhs.kind == MatrixStructure::LowerTriangular) {
    const int32_t off = eff_lhs.diagonal_offset + eff_rhs.diagonal_offset;
    const bool unit =
        (eff_lhs.diagonal_offset == 0 && eff_rhs.diagonal_offset == 0 && eff_lhs.is_unit_diagonal &&
         eff_rhs.is_unit_diagonal);
    return StructureAttr{MatrixStructure::LowerTriangular, off, unit};
  }

  if (eff_lhs.kind == MatrixStructure::Diagonal && eff_rhs.kind == MatrixStructure::Diagonal) {
    const int32_t off = eff_lhs.diagonal_offset + eff_rhs.diagonal_offset;
    const bool unit =
        (eff_lhs.diagonal_offset == 0 && eff_rhs.diagonal_offset == 0 && eff_lhs.is_unit_diagonal &&
         eff_rhs.is_unit_diagonal);
    return StructureAttr{MatrixStructure::Diagonal, off, unit};
  }

  if (eff_lhs.kind == MatrixStructure::Diagonal && eff_lhs.diagonal_offset == 0 &&
      (eff_rhs.kind == MatrixStructure::UpperTriangular || eff_rhs.kind == MatrixStructure::LowerTriangular)) {
    const bool unit =
        (eff_rhs.diagonal_offset == 0 && eff_lhs.is_unit_diagonal && eff_rhs.is_unit_diagonal);
    return StructureAttr{eff_rhs.kind, eff_rhs.diagonal_offset, unit};
  }

  if (eff_rhs.kind == MatrixStructure::Diagonal && eff_rhs.diagonal_offset == 0 &&
      (eff_lhs.kind == MatrixStructure::UpperTriangular || eff_lhs.kind == MatrixStructure::LowerTriangular)) {
    const bool unit =
        (eff_lhs.diagonal_offset == 0 && eff_lhs.is_unit_diagonal && eff_rhs.is_unit_diagonal);
    return StructureAttr{eff_lhs.kind, eff_lhs.diagonal_offset, unit};
  }

  // (A @ B)^T = B^T @ A^T = B @ A != A @ B for distinct symmetric matrices; only A @ A is guaranteed symmetric.
  if (eff_lhs.kind == MatrixStructure::Symmetric && eff_rhs.kind == MatrixStructure::Symmetric &&
      same_ssa_operands) {
    return StructureAttr{MatrixStructure::Symmetric, 0, false};
  }

  return StructureAttr{};
}

bool process_block(Kernel& kernel, BlockId blk_id, const std::optional<StructureAttr>& kernel_hint,
                   PassResult& res) {
  const auto op_ids = kernel.block(blk_id).ops;
  for (OpId op_id : op_ids) {
    Op& node = kernel.op(op_id);
    if (node.erased) continue;

    if (node.kind == OpKind::Dot && node.operands.size() >= 2 && !node.results.empty() &&
        std::holds_alternative<DotAttr>(node.attr)) {
      DotAttr da = std::get<DotAttr>(node.attr);
      const ValueId lhs_id = node.operands[0];
      const ValueId rhs_id = node.operands[1];
      const ValueId res_id = node.results[0];

      std::string err;
      StructureAttr lhs_s;
      StructureAttr rhs_s;
      if (!resolve_operand_structure(kernel, lhs_id, da.lhs_struct, "dot lhs", lhs_s, err) ||
          !resolve_operand_structure(kernel, rhs_id, da.rhs_struct, "dot rhs", rhs_s, err)) {
        res.ok = false;
        res.diagnostic = err;
        return false;
      }

      const std::optional<StructureAttr> active_hint =
          node.hints.structure.has_value() ? node.hints.structure : kernel_hint;
      if (active_hint.has_value()) {
        if (!validate_structure_attr(*active_hint, "dot hint", err)) {
          res.ok = false;
          res.diagnostic = err;
          return false;
        }
        const StructureAttr& hs = *active_hint;
        if (hs.kind == MatrixStructure::UpperTriangular || hs.kind == MatrixStructure::LowerTriangular ||
            hs.kind == MatrixStructure::Symmetric) {
          if (lhs_s.kind == MatrixStructure::Dense) lhs_s = hs;
          if (rhs_s.kind == MatrixStructure::Dense) rhs_s = hs;
        } else if (hs.kind == MatrixStructure::Diagonal) {
          if (lhs_s.kind == MatrixStructure::Dense && rhs_s.kind == MatrixStructure::Dense) {
            lhs_s = hs;
          }
        }
      }

      if (!validate_structure_attr(da.out_struct, "dot out_struct", err)) {
        res.ok = false;
        res.diagnostic = err;
        return false;
      }

      StructureAttr res_tile_struct{};
      if (kernel.value(res_id).type.is_tile()) {
        res_tile_struct = kernel.value(res_id).type.as_tile().structure;
        if (!validate_structure_attr(res_tile_struct, "dot result tile", err)) {
          res.ok = false;
          res.diagnostic = err;
          return false;
        }
      }

      if (da.out_struct.kind != MatrixStructure::Dense && res_tile_struct.kind != MatrixStructure::Dense &&
          da.out_struct != res_tile_struct) {
        res.ok = false;
        res.diagnostic = "dot out: conflicting StructureAttr between DotAttr and result TileType";
        return false;
      }

      StructureAttr out_s =
          (da.out_struct.kind != MatrixStructure::Dense) ? da.out_struct : res_tile_struct;
      if (out_s.kind == MatrixStructure::Dense) {
        const StructureAttr eff_lhs = effective_structure(lhs_s, da.transpose_a);
        const StructureAttr eff_rhs = effective_structure(rhs_s, da.transpose_b);
        out_s = infer_dot_output_structure(eff_lhs, eff_rhs, lhs_id == rhs_id);
      }

      if (!validate_structure_attr(out_s, "dot inferred out_struct", err)) {
        res.ok = false;
        res.diagnostic = err;
        return false;
      }

      da.lhs_struct = lhs_s;
      da.rhs_struct = rhs_s;
      da.out_struct = out_s;

      if (std::get<DotAttr>(node.attr) != da) {
        node.attr = da;
        res.changed = true;
      }

      if (out_s.kind != MatrixStructure::Dense && kernel.value(res_id).type.is_tile()) {
        TileType& res_tile = kernel.value(res_id).type.as_tile_mut();
        if (res_tile.structure != out_s) {
          res_tile.structure = out_s;
          res.changed = true;
        }
      }
    }

    const auto regions = node.regions;
    for (BlockId reg_id : regions) {
      if (!process_block(kernel, reg_id, kernel_hint, res)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

PassResult StructureAwareDeadTilePruningPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  if (kernel.entry_block == kInvalidId) {
    return res;
  }

  if (kernel.hints.structure.has_value()) {
    std::string err;
    if (!validate_structure_attr(*kernel.hints.structure, "kernel.hints.structure", err)) {
      res.ok = false;
      res.diagnostic = std::move(err);
      return res;
    }
  }

  process_block(kernel, kernel.entry_block, kernel.hints.structure, res);
  return res;
}

}  // namespace m1triton::passes

#include "ir/verifier.hpp"

#include <cstdint>
#include <sstream>
#include <unordered_set>

namespace m1triton::ir {

namespace {

bool is_positive_shape(const std::vector<int64_t>& shape) {
  if (shape.empty()) return false;
  for (int64_t d : shape) {
    if (d <= 0) return false;
  }
  return true;
}

int64_t shape_numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

bool same_shape(const Type& a, const Type& b) {
  if (a.is_tile() != b.is_tile()) {
    return false;
  }
  if (!a.is_tile()) {
    return true;
  }
  return a.as_tile().shape == b.as_tile().shape;
}

bool is_float_only_unary(UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp:
    case UnaryOp::Log:
    case UnaryOp::Sqrt:
    case UnaryOp::Rsqrt:
    case UnaryOp::Tanh:
    case UnaryOp::Erf:
    case UnaryOp::Floor:
    case UnaryOp::Ceil:
    case UnaryOp::Sin:
    case UnaryOp::Cos:
    case UnaryOp::Sigmoid:
      return true;
    default:
      return false;
  }
}

bool is_bitwise_binary(BinaryOp op) {
  switch (op) {
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Xor:
    case BinaryOp::Shl:
    case BinaryOp::Shr:
      return true;
    default:
      return false;
  }
}

bool is_valid_vec_width(uint32_t w) {
  return w == 1 || w == 2 || w == 4 || w == 8;
}

class KernelVerifier {
 public:
  KernelVerifier(const Kernel& kernel, const TargetSpec& target, VerifyStage stage, VerifyResult& out)
      : kernel_(kernel), target_(target), stage_(stage), out_(out) {}

  void run() {
    if (kernel_.name.empty()) {
      error("Kernel name must not be empty");
    }
    if (kernel_.entry_block == kInvalidId || kernel_.entry_block >= kernel_.blocks.size()) {
      error("Kernel has invalid entry_block");
      return;
    }

    const Block& entry = kernel_.block(kernel_.entry_block);
    if (entry.parent_op != kInvalidId) {
      error("Entry block must not have a parent_op");
    }
    if (entry.args.size() != kernel_.params.size()) {
      error("Entry block argument count (" + std::to_string(entry.args.size()) +
            ") does not match kernel param count (" + std::to_string(kernel_.params.size()) + ")");
    } else {
      for (size_t i = 0; i < kernel_.params.size(); ++i) {
        const ValueId arg_id = entry.args[i];
        if (!valid_value_id(arg_id)) {
          error("Entry block arg " + std::to_string(i) + " has invalid ValueId");
          continue;
        }
        if (kernel_.value(arg_id).type != kernel_.params[i].type) {
          error("Entry block arg " + std::to_string(i) + " type mismatch with kernel param '" +
                kernel_.params[i].name + "'");
        }
      }
    }

    verify_op_hints(kernel_.hints, "kernel @" + kernel_.name);

    std::unordered_set<ValueId> visible;
    verify_block(kernel_.entry_block, visible);

    if (stage_ == VerifyStage::PostTile || stage_ == VerifyStage::PreCodegen) {
      if (total_tg_bytes_ > target_.max_threadgroup_memory_bytes) {
        error("Total threadgroup memory (" + std::to_string(total_tg_bytes_) +
              " bytes) exceeds target limit (" + std::to_string(target_.max_threadgroup_memory_bytes) + " bytes)");
      }
    }
  }

 private:
  void error(std::string msg) {
    out_.ok = false;
    out_.errors.push_back(std::move(msg));
  }

  void verify_op_hints(const OpHints& hints, const std::string& where) {
    if (hints.epilogue_dag.has_value()) {
      bool saved_reg[4] = {false, false, false, false};
      for (size_t i = 0; i < hints.epilogue_dag->steps.size(); ++i) {
        const auto& st = hints.epilogue_dag->steps[i];
        const std::string stag = where + " epilogue_dag[" + std::to_string(i) + "]";
        if (epilogue_op_needs_ptr_arg(st.op)) {
          if (st.arg_idx < 0 || static_cast<size_t>(st.arg_idx) >= kernel_.params.size()) {
            error(stag + ": buffer arg_idx out of range");
          } else if (!kernel_.params[static_cast<size_t>(st.arg_idx)].type.is_ptr()) {
            error(stag + ": buffer step expects pointer kernel param");
          }
        } else if (epilogue_op_needs_scalar_arg(st.op)) {
          if (st.arg_idx < 0 || static_cast<size_t>(st.arg_idx) >= kernel_.params.size()) {
            error(stag + ": scalar arg_idx out of range");
          } else if (!kernel_.params[static_cast<size_t>(st.arg_idx)].type.is_scalar()) {
            error(stag + ": scalar step expects scalar kernel param");
          }
        } else if (epilogue_op_uses_reg_slot(st.op)) {
          if (st.arg_idx < 0 || st.arg_idx >= 4) {
            error(stag + ": register slot must be in [0, 3]");
          } else if (st.op == EpilogueOpKind::SaveReg) {
            saved_reg[st.arg_idx] = true;
          } else if (!saved_reg[st.arg_idx]) {
            error(stag + ": reads unsaved register slot " + std::to_string(st.arg_idx));
          }
        }
        if ((st.op == EpilogueOpKind::Clamp || st.op == EpilogueOpKind::HardTanh) && st.param0 > st.param1) {
          error(stag + ": clamp/hardtanh min exceeds max");
        }
      }
    }
    if (hints.fused_tail.has_value() && hints.fused_tail->kind != TailKind::None) {
      const auto& ft = *hints.fused_tail;
      if (ft.kernel_size < 1 || ft.stride < 1) {
        error(where + " fused_tail: kernel_size and stride must be >= 1");
      }
    }
  }

  bool valid_value_id(ValueId id) const { return id < kernel_.values.size(); }
  bool valid_op_id(OpId id) const { return id < kernel_.ops.size(); }
  bool valid_block_id(BlockId id) const { return id < kernel_.blocks.size(); }

  void check_tile_layout(const Type& ty, const std::string& where) {
    if (!ty.is_tile()) {
      return;
    }
    const auto& tile = ty.as_tile();
    if (!is_positive_shape(tile.shape)) {
      error(where + ": tile shape must be non-empty with positive dimensions");
    }

    if (stage_ == VerifyStage::PostTile || stage_ == VerifyStage::PreCodegen) {
      const auto& layout = tile.layout;
      if (stage_ == VerifyStage::PreCodegen && layout.kind == LayoutKind::Unassigned) {
        error(where + ": tile layout must be assigned before PreCodegen");
      }
      if (layout.kind != LayoutKind::Unassigned) {
        const uint64_t total_threads =
            static_cast<uint64_t>(layout.tg_threads[0]) * layout.tg_threads[1] * layout.tg_threads[2];
        if (total_threads == 0) {
          error(where + ": layout tg_threads must be non-zero");
        }
        if (total_threads > target_.max_threads_per_threadgroup) {
          error(where + ": layout thread count (" + std::to_string(total_threads) +
                ") exceeds target max_threads_per_threadgroup (" +
                std::to_string(target_.max_threads_per_threadgroup) + ")");
        }
        if (target_.simd_width > 0 && (total_threads % target_.simd_width) != 0) {
          error(where + ": layout thread count (" + std::to_string(total_threads) +
                ") must be divisible by target simd_width (" + std::to_string(target_.simd_width) + ")");
        }
        if (!is_valid_vec_width(layout.vec_width)) {
          error(where + ": invalid layout vec_width " + std::to_string(layout.vec_width));
        }
      }
    }
  }

  void verify_block(BlockId block_id, std::unordered_set<ValueId> visible) {
    if (!valid_block_id(block_id)) {
      error("Invalid BlockId " + std::to_string(block_id));
      return;
    }
    if (!visited_blocks_.insert(block_id).second) {
      error("Block " + std::to_string(block_id) + " visited more than once (non-tree region graph)");
      return;
    }

    const Block& blk = kernel_.block(block_id);
    for (uint32_t i = 0; i < blk.args.size(); ++i) {
      const ValueId arg_id = blk.args[i];
      if (!valid_value_id(arg_id)) {
        error("Block " + std::to_string(block_id) + " arg " + std::to_string(i) + " has invalid ValueId");
        continue;
      }
      const Value& val = kernel_.value(arg_id);
      if (val.def_block != block_id || val.def_op != kInvalidId) {
        error("Block arg %" + std::to_string(arg_id) + " has inconsistent def_block/def_op");
      }
      if (!visible.insert(arg_id).second) {
        error("Duplicate definition of block arg %" + std::to_string(arg_id));
      }
      check_tile_layout(val.type, "block arg %" + std::to_string(arg_id));
    }

    if (blk.ops.empty()) {
      error("Block " + std::to_string(block_id) + " is empty (missing terminator)");
      return;
    }

    for (size_t idx = 0; idx < blk.ops.size(); ++idx) {
      const OpId op_id = blk.ops[idx];
      if (!valid_op_id(op_id)) {
        error("Block " + std::to_string(block_id) + " references invalid OpId " + std::to_string(op_id));
        continue;
      }
      const Op& node = kernel_.op(op_id);
      if (node.erased) {
        error("Block " + std::to_string(block_id) + " contains erased OpId " + std::to_string(op_id));
        continue;
      }
      if (node.parent_block != block_id) {
        error("Op " + std::to_string(op_id) + " has mismatched parent_block");
      }

      const bool is_last = (idx + 1 == blk.ops.size());
      if (is_terminator_op(node.kind) && !is_last) {
        error("Terminator op " + std::string(op_kind_name(node.kind)) + " must be the last op in block " +
              std::to_string(block_id));
      }
      if (is_last) {
        if (block_id == kernel_.entry_block && node.kind != OpKind::Return) {
          error("Entry block must terminate with Return");
        } else if (block_id != kernel_.entry_block && node.kind != OpKind::Yield) {
          error("Region block " + std::to_string(block_id) + " must terminate with Yield");
        }
      }

      bool operands_valid = true;
      for (size_t o = 0; o < node.operands.size(); ++o) {
        const ValueId vid = node.operands[o];
        if (!valid_value_id(vid)) {
          error("Op " + std::to_string(op_id) + " (" + std::string(op_kind_name(node.kind)) + ") operand " +
                std::to_string(o) + " has invalid ValueId");
          operands_valid = false;
        } else if (!visible.contains(vid)) {
          error("Op " + std::to_string(op_id) + " (" + std::string(op_kind_name(node.kind)) +
                ") uses value %" + std::to_string(vid) + " which does not dominate it");
          operands_valid = false;
        }
      }

      bool results_valid = true;
      for (uint32_t r = 0; r < node.results.size(); ++r) {
        const ValueId rid = node.results[r];
        if (!valid_value_id(rid)) {
          error("Op " + std::to_string(op_id) + " result " + std::to_string(r) + " has invalid ValueId");
          results_valid = false;
          continue;
        }
        const Value& rv = kernel_.value(rid);
        if (rv.def_op != op_id || rv.def_block != kInvalidId || rv.result_idx != r) {
          error("Op " + std::to_string(op_id) + " result %" + std::to_string(rid) +
                " has inconsistent def metadata");
        }
        check_tile_layout(rv.type, "op " + std::to_string(op_id) + " result %" + std::to_string(rid));
      }

      if (operands_valid && results_valid) {
        verify_op(node, visible);
      }

      for (ValueId rid : node.results) {
        if (valid_value_id(rid) && !visible.insert(rid).second) {
          error("Duplicate SSA value definition %" + std::to_string(rid));
        }
      }
    }
  }

  void verify_op(const Op& node, const std::unordered_set<ValueId>& visible) {
    const std::string tag = "op " + std::to_string(node.id) + " (" + std::string(op_kind_name(node.kind)) + ")";

    switch (node.kind) {
      case OpKind::Constant: {
        if (!node.operands.empty() || node.results.size() != 1) {
          error(tag + ": expects 0 operands and 1 result");
          break;
        }
        if (!std::holds_alternative<ConstAttr>(node.attr)) {
          error(tag + ": missing ConstAttr");
        }
        if (!kernel_.value(node.results[0]).type.is_scalar()) {
          error(tag + ": result must be scalar");
        }
        break;
      }

      case OpKind::ProgramId:
      case OpKind::NumPrograms: {
        if (!node.operands.empty() || node.results.size() != 1) {
          error(tag + ": expects 0 operands and 1 result");
          break;
        }
        if (!std::holds_alternative<AxisAttr>(node.attr)) {
          error(tag + ": missing AxisAttr");
        } else if (std::get<AxisAttr>(node.attr).axis > 2) {
          error(tag + ": axis must be in [0, 2]");
        }
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!rty.is_scalar() || !rty.is_int_elem()) {
          error(tag + ": result must be scalar integer");
        }
        break;
      }

      case OpKind::Arange: {
        if (!node.operands.empty() || node.results.size() != 1) {
          error(tag + ": expects 0 operands and 1 result");
          break;
        }
        if (!std::holds_alternative<ArangeAttr>(node.attr)) {
          error(tag + ": missing ArangeAttr");
          break;
        }
        const auto& a = std::get<ArangeAttr>(node.attr);
        if (a.end <= a.start) {
          error(tag + ": end must be greater than start");
        }
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!rty.is_tile() || !rty.is_int_elem() || rty.as_tile().shape.size() != 1 ||
            rty.as_tile().shape[0] != (a.end - a.start)) {
          error(tag + ": result must be 1D int tile of length (end - start)");
        }
        break;
      }

      case OpKind::Splat: {
        if (node.operands.size() != 1 || node.results.size() != 1) {
          error(tag + ": expects 1 operand and 1 result");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (ity.is_tile() || !rty.is_tile()) {
          error(tag + ": expects scalar/ptr operand and tile result");
        } else if (ity.elem_type() != rty.elem_type()) {
          error(tag + ": element type mismatch");
        }
        break;
      }

      case OpKind::ExpandDims: {
        if (node.operands.size() != 1 || node.results.size() != 1 || !std::holds_alternative<AxisAttr>(node.attr)) {
          error(tag + ": expects 1 operand, 1 result, and AxisAttr");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!rty.is_tile() || ity.elem_type() != rty.elem_type()) {
          error(tag + ": result must be tile with matching elem type");
          break;
        }
        std::vector<int64_t> expected = ity.is_tile() ? ity.as_tile().shape : std::vector<int64_t>{};
        const uint32_t axis = std::get<AxisAttr>(node.attr).axis;
        if (axis > expected.size()) {
          error(tag + ": axis out of range");
          break;
        }
        expected.insert(expected.begin() + static_cast<std::ptrdiff_t>(axis), 1);
        if (rty.as_tile().shape != expected) {
          error(tag + ": output shape does not match ExpandDims");
        }
        break;
      }

      case OpKind::Broadcast: {
        if (node.operands.size() != 1 || node.results.size() != 1) {
          error(tag + ": expects 1 operand and 1 result");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!ity.is_tile() || !rty.is_tile() || ity.elem_type() != rty.elem_type()) {
          error(tag + ": expects tile operand and tile result with matching elem type");
          break;
        }
        const auto& src = ity.as_tile().shape;
        const auto& dst = rty.as_tile().shape;
        if (src.size() > dst.size()) {
          error(tag + ": cannot broadcast higher rank to lower rank");
          break;
        }
        const size_t diff = dst.size() - src.size();
        for (size_t i = 0; i < src.size(); ++i) {
          if (src[i] != dst[diff + i] && src[i] != 1) {
            error(tag + ": incompatible broadcast dimension at axis " + std::to_string(i));
            break;
          }
        }
        break;
      }

      case OpKind::Reshape: {
        if (node.operands.size() != 1 || node.results.size() != 1) {
          error(tag + ": expects 1 operand and 1 result");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!ity.is_tile() || !rty.is_tile() || ity.elem_type() != rty.elem_type()) {
          error(tag + ": expects tile operand and tile result with matching elem type");
        } else if (shape_numel(ity.as_tile().shape) != shape_numel(rty.as_tile().shape)) {
          error(tag + ": element count mismatch");
        }
        break;
      }

      case OpKind::Permute: {
        if (node.operands.size() != 1 || node.results.size() != 1 ||
            !std::holds_alternative<PermuteAttr>(node.attr)) {
          error(tag + ": expects 1 operand, 1 result, and PermuteAttr");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!ity.is_tile() || !rty.is_tile() || ity.elem_type() != rty.elem_type()) {
          error(tag + ": expects tile operand and result with matching elem type");
          break;
        }
        const auto& perm = std::get<PermuteAttr>(node.attr).perm;
        const auto& src = ity.as_tile().shape;
        if (perm.size() != src.size() || rty.as_tile().shape.size() != src.size()) {
          error(tag + ": permutation rank mismatch");
          break;
        }
        std::vector<bool> seen(src.size(), false);
        for (size_t i = 0; i < perm.size(); ++i) {
          if (perm[i] >= src.size() || seen[perm[i]]) {
            error(tag + ": invalid permutation");
            break;
          }
          seen[perm[i]] = true;
          if (rty.as_tile().shape[i] != src[perm[i]]) {
            error(tag + ": result shape does not match permutation");
            break;
          }
        }
        break;
      }

      case OpKind::Cast: {
        if (node.operands.size() != 1 || node.results.size() != 1) {
          error(tag + ": expects 1 operand and 1 result");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (ity.is_ptr_elem() || rty.is_ptr_elem()) {
          error(tag + ": pointer cast is not supported");
        } else if (!same_shape(ity, rty)) {
          error(tag + ": operand and result shapes must match");
        }
        break;
      }

      case OpKind::Unary: {
        if (node.operands.size() != 1 || node.results.size() != 1 || !std::holds_alternative<UnaryAttr>(node.attr)) {
          error(tag + ": expects 1 operand, 1 result, and UnaryAttr");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        const UnaryOp uop = std::get<UnaryAttr>(node.attr).op;
        if (ity.is_ptr_elem() || rty.is_ptr_elem()) {
          error(tag + ": cannot operate on pointer types");
        } else if (!ity.same_shape_and_elem(rty)) {
          error(tag + ": operand and result types must match");
        } else if (is_float_only_unary(uop) && !ity.is_float_elem()) {
          error(tag + ": transcendental/float unary op requires float operand");
        } else if (uop == UnaryOp::Not && !ity.is_bool_elem() && !ity.is_int_elem()) {
          error(tag + ": UnaryOp::Not requires bool or int operand");
        }
        break;
      }

      case OpKind::Binary: {
        if (node.operands.size() != 2 || node.results.size() != 1 || !std::holds_alternative<BinaryAttr>(node.attr)) {
          error(tag + ": expects 2 operands, 1 result, and BinaryAttr");
          break;
        }
        const Type& lty = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.operands[1]).type;
        const Type& out_ty = kernel_.value(node.results[0]).type;
        const BinaryOp bop = std::get<BinaryAttr>(node.attr).op;
        if (lty.is_ptr_elem() || rty.is_ptr_elem()) {
          error(tag + ": binary op cannot operate on pointers (use PtrAdd)");
          break;
        }
        if (!lty.same_shape_and_elem(rty)) {
          error(tag + ": lhs and rhs types must match (" + type_to_string(lty) + " vs " + type_to_string(rty) + ")");
          break;
        }
        if (is_bitwise_binary(bop)) {
          if ((bop == BinaryOp::Shl || bop == BinaryOp::Shr) && !lty.is_int_elem()) {
            error(tag + ": shift op requires integer operands");
          } else if (!lty.is_int_elem() && !lty.is_bool_elem()) {
            error(tag + ": bitwise op requires integer or bool operands");
          }
        }
        if (is_comparison_op(bop)) {
          if (!same_shape(lty, out_ty) || !out_ty.is_bool_elem()) {
            error(tag + ": comparison result must have same shape and bool element type");
          }
        } else if (!lty.same_shape_and_elem(out_ty)) {
          error(tag + ": result type must match operand type");
        }
        break;
      }

      case OpKind::Select: {
        if (node.operands.size() != 3 || node.results.size() != 1) {
          error(tag + ": expects 3 operands and 1 result");
          break;
        }
        const Type& cty = kernel_.value(node.operands[0]).type;
        const Type& tty = kernel_.value(node.operands[1]).type;
        const Type& fty = kernel_.value(node.operands[2]).type;
        const Type& out_ty = kernel_.value(node.results[0]).type;
        if (!cty.is_bool_elem()) {
          error(tag + ": condition must have bool element type");
        }
        if (!tty.same_shape_and_elem(fty) || !tty.same_shape_and_elem(out_ty)) {
          error(tag + ": true, false, and result types must match");
        }
        if (!same_shape(cty, out_ty)) {
          error(tag + ": condition shape must match result shape");
        }
        break;
      }

      case OpKind::PtrAdd: {
        if (node.operands.size() != 2 || node.results.size() != 1) {
          error(tag + ": expects 2 operands and 1 result");
          break;
        }
        const Type& pty = kernel_.value(node.operands[0]).type;
        const Type& oty = kernel_.value(node.operands[1]).type;
        const Type& out_ty = kernel_.value(node.results[0]).type;
        if (!pty.is_ptr_elem()) {
          error(tag + ": first operand must be pointer or tile of pointers");
          break;
        }
        if (!oty.is_int_elem()) {
          error(tag + ": offset operand must have integer element type");
          break;
        }
        if (!out_ty.is_ptr_elem() || out_ty.ptr_elem() != pty.ptr_elem()) {
          error(tag + ": result must have same pointer element type as base pointer");
          break;
        }
        if (pty.is_tile() && oty.is_tile()) {
          if (!same_shape(pty, oty) || !same_shape(pty, out_ty)) {
            error(tag + ": tile ptr and tile offset shapes must match result shape");
          }
        } else if (pty.is_tile()) {
          if (!same_shape(pty, out_ty)) {
            error(tag + ": result shape must match tile pointer shape");
          }
        } else if (oty.is_tile()) {
          if (!same_shape(oty, out_ty)) {
            error(tag + ": result shape must match tile offset shape");
          }
        } else if (out_ty.is_tile()) {
          error(tag + ": scalar ptr + scalar offset must produce scalar ptr");
        }
        break;
      }

      case OpKind::Load: {
        if (!std::holds_alternative<LoadStoreAttr>(node.attr)) {
          error(tag + ": missing LoadStoreAttr");
          break;
        }
        const auto& ls = std::get<LoadStoreAttr>(node.attr);
        const size_t expected_ops = 1 + (ls.has_mask ? 1 : 0) + (ls.has_other ? 1 : 0);
        if (node.operands.size() != expected_ops || node.results.size() != 1) {
          error(tag + ": operand count (" + std::to_string(node.operands.size()) +
                ") does not match LoadStoreAttr (expected " + std::to_string(expected_ops) + ")");
          break;
        }
        const Type& pty = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!pty.is_ptr_elem()) {
          error(tag + ": address operand must be pointer or tile of pointers");
          break;
        }
        if (rty.is_ptr_elem() || rty.scalar_elem() != pty.ptr_elem().pointee || !same_shape(pty, rty)) {
          error(tag + ": result type must match pointer pointee dtype and shape");
        }
        size_t idx = 1;
        if (ls.has_mask) {
          const Type& mty = kernel_.value(node.operands[idx++]).type;
          if (!mty.is_bool_elem() || (!mty.is_scalar() && !same_shape(mty, rty))) {
            error(tag + ": mask must be bool with matching shape");
          }
        }
        if (ls.has_other) {
          const Type& oty = kernel_.value(node.operands[idx++]).type;
          if (!oty.same_shape_and_elem(rty)) {
            error(tag + ": other value type must match load result type");
          }
        }
        if (!is_valid_vec_width(ls.vec_width)) {
          error(tag + ": invalid vec_width " + std::to_string(ls.vec_width));
        }
        if (ls.alignment_bytes == 0) {
          error(tag + ": alignment_bytes must be positive");
        }
        break;
      }

      case OpKind::Store: {
        if (!std::holds_alternative<LoadStoreAttr>(node.attr)) {
          error(tag + ": missing LoadStoreAttr");
          break;
        }
        const auto& ls = std::get<LoadStoreAttr>(node.attr);
        const size_t expected_ops = 2 + (ls.has_mask ? 1 : 0);
        if (node.operands.size() != expected_ops || !node.results.empty()) {
          error(tag + ": expects " + std::to_string(expected_ops) + " operands and 0 results");
          break;
        }
        const Type& pty = kernel_.value(node.operands[0]).type;
        const Type& vty = kernel_.value(node.operands[1]).type;
        if (!pty.is_ptr_elem()) {
          error(tag + ": address operand must be pointer or tile of pointers");
          break;
        }
        if (vty.is_ptr_elem() || vty.scalar_elem() != pty.ptr_elem().pointee || !same_shape(pty, vty)) {
          error(tag + ": value type must match pointer pointee dtype and shape");
        }
        if (ls.has_mask) {
          const Type& mty = kernel_.value(node.operands[2]).type;
          if (!mty.is_bool_elem() || (!mty.is_scalar() && !same_shape(mty, vty))) {
            error(tag + ": mask must be bool with matching shape");
          }
        }
        if (!is_valid_vec_width(ls.vec_width)) {
          error(tag + ": invalid vec_width " + std::to_string(ls.vec_width));
        }
        if (ls.alignment_bytes == 0) {
          error(tag + ": alignment_bytes must be positive");
        }
        break;
      }

      case OpKind::AtomicAdd: {
        if ((node.operands.size() != 2 && node.operands.size() != 3) || node.results.size() != 1) {
          error(tag + ": expects 2 or 3 operands and 1 result");
          break;
        }
        const Type& pty = kernel_.value(node.operands[0]).type;
        const Type& vty = kernel_.value(node.operands[1]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!pty.is_ptr_elem()) {
          error(tag + ": address operand must be pointer or tile of pointers");
          break;
        }
        if (vty.is_ptr_elem() || vty.scalar_elem() != pty.ptr_elem().pointee || !same_shape(pty, vty) ||
            !vty.same_shape_and_elem(rty)) {
          error(tag + ": value and result must match pointer pointee dtype and shape");
        }
        if (node.operands.size() == 3) {
          const Type& mty = kernel_.value(node.operands[2]).type;
          if (!mty.is_bool_elem() || (!mty.is_scalar() && !same_shape(mty, vty))) {
            error(tag + ": mask must be bool with matching shape");
          }
        }
        break;
      }

      case OpKind::Dot: {
        if ((node.operands.size() != 2 && node.operands.size() != 3) || node.results.size() != 1 ||
            !std::holds_alternative<DotAttr>(node.attr)) {
          error(tag + ": expects 2 or 3 operands, 1 result, and DotAttr");
          break;
        }
        const auto& da = std::get<DotAttr>(node.attr);
        const Type& lty = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.operands[1]).type;
        const Type& out_ty = kernel_.value(node.results[0]).type;
        if (!lty.is_tile() || !rty.is_tile() || !out_ty.is_tile()) {
          error(tag + ": operands and result must be tiles");
          break;
        }
        if (lty.is_ptr_elem() || rty.is_ptr_elem() || out_ty.is_ptr_elem()) {
          error(tag + ": dot cannot operate on pointer tiles");
          break;
        }
        const auto& ls = lty.as_tile().shape;
        const auto& rs = rty.as_tile().shape;
        const auto& os = out_ty.as_tile().shape;
        if (ls.size() != rs.size() || os.size() != ls.size() || (ls.size() != 2 && ls.size() != 3)) {
          error(tag + ": expects 2D or 3D tile shapes");
          break;
        }
        const size_t off = ls.size() == 3 ? 1 : 0;
        if (ls.size() == 3 && (ls[0] != rs[0] || ls[0] != os[0])) {
          error(tag + ": batch dimension mismatch");
        }
        const int64_t m = da.transpose_a ? ls[off + 1] : ls[off];
        const int64_t k_lhs = da.transpose_a ? ls[off] : ls[off + 1];
        const int64_t k_rhs = da.transpose_b ? rs[off + 1] : rs[off];
        const int64_t n = da.transpose_b ? rs[off] : rs[off + 1];
        if (k_lhs != k_rhs) {
          error(tag + ": inner K dimension mismatch (" + std::to_string(k_lhs) + " vs " + std::to_string(k_rhs) + ")");
        }
        if (os[off] != m || os[off + 1] != n) {
          error(tag + ": result shape does not match [M, N]");
        }
        if (out_ty.scalar_elem() != da.acc_dtype) {
          error(tag + ": result dtype does not match DotAttr::acc_dtype");
        }
        if (node.operands.size() == 3) {
          const Type& aty = kernel_.value(node.operands[2]).type;
          if (!aty.same_shape_and_elem(out_ty)) {
            error(tag + ": accumulator type must match dot result type");
          }
        }
        break;
      }

      case OpKind::Reduce: {
        if (node.operands.size() != 1 || !std::holds_alternative<ReduceAttr>(node.attr)) {
          error(tag + ": expects 1 operand and ReduceAttr");
          break;
        }
        const auto& ra = std::get<ReduceAttr>(node.attr);
        const size_t expected_res = (ra.result_mode == ReduceResultMode::ValueAndIndex) ? 2 : 1;
        if (node.results.size() != expected_res) {
          error(tag + ": unexpected result count");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        if (!ity.is_tile() || ity.is_ptr_elem()) {
          error(tag + ": operand must be a non-pointer tile");
          break;
        }
        const auto& shape = ity.as_tile().shape;
        if (ra.axis < 0 || static_cast<size_t>(ra.axis) >= shape.size()) {
          error(tag + ": reduce axis out of range");
          break;
        }
        std::vector<int64_t> expected_shape;
        if (ra.keep_dims) {
          expected_shape = shape;
          expected_shape[static_cast<size_t>(ra.axis)] = 1;
        } else {
          for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int32_t>(i) != ra.axis) {
              expected_shape.push_back(shape[i]);
            }
          }
        }
        for (ValueId rid : node.results) {
          const Type& rty = kernel_.value(rid).type;
          if (expected_shape.empty()) {
            if (!rty.is_scalar()) {
              error(tag + ": expected scalar result for 1D reduction without keep_dims");
            }
          } else {
            if (!rty.is_tile() || rty.as_tile().shape != expected_shape) {
              error(tag + ": reduced result shape mismatch");
            }
          }
        }
        break;
      }

      case OpKind::Scan: {
        if (!std::holds_alternative<ScanAttr>(node.attr)) {
          error(tag + ": missing ScanAttr");
          break;
        }
        const auto& sa = std::get<ScanAttr>(node.attr);
        const size_t expected_ops = sa.has_mask ? 2 : 1;
        if (node.operands.size() != expected_ops || node.results.size() != 1) {
          error(tag + ": unexpected operand or result count");
          break;
        }
        const Type& ity = kernel_.value(node.operands[0]).type;
        const Type& rty = kernel_.value(node.results[0]).type;
        if (!ity.is_tile() || ity.is_ptr_elem() || !ity.same_shape_and_elem(rty)) {
          error(tag + ": input and result must be matching non-pointer tiles");
          break;
        }
        if (sa.axis < 0 || static_cast<size_t>(sa.axis) >= ity.as_tile().shape.size()) {
          error(tag + ": scan axis out of range");
        }
        if (sa.has_mask) {
          const Type& mty = kernel_.value(node.operands[1]).type;
          if (!mty.is_bool_elem() || (!mty.is_scalar() && !same_shape(mty, rty))) {
            error(tag + ": scan mask must be bool with matching shape");
          }
        }
        break;
      }

      case OpKind::ForLoop: {
        if (node.operands.size() < 3) {
          error(tag + ": requires at least 3 operands (lo, hi, step)");
          break;
        }
        const Type& lo_ty = kernel_.value(node.operands[0]).type;
        const Type& hi_ty = kernel_.value(node.operands[1]).type;
        const Type& step_ty = kernel_.value(node.operands[2]).type;
        if (!lo_ty.is_scalar() || !lo_ty.is_int_elem() || lo_ty != hi_ty || lo_ty != step_ty) {
          error(tag + ": lo, hi, and step must be matching scalar integers");
        }
        const size_t num_iter = node.operands.size() - 3;
        if (node.results.size() != num_iter) {
          error(tag + ": result count must match iter_init count");
        }
        if (node.regions.size() != 1) {
          error(tag + ": must have exactly 1 body region");
          break;
        }
        const BlockId body_id = node.regions[0];
        if (!valid_block_id(body_id)) {
          error(tag + ": invalid body region BlockId");
          break;
        }
        const Block& body = kernel_.block(body_id);
        if (body.parent_op != node.id) {
          error(tag + ": body block parent_op mismatch");
        }
        if (body.args.size() != 1 + num_iter) {
          error(tag + ": body block must have (1 + num_iter) args");
        } else {
          if (valid_value_id(body.args[0]) && kernel_.value(body.args[0]).type != lo_ty) {
            error(tag + ": induction variable type must match lo/hi/step");
          }
          for (size_t i = 0; i < num_iter; ++i) {
            const Type& init_ty = kernel_.value(node.operands[3 + i]).type;
            if (valid_value_id(body.args[1 + i]) &&
                !kernel_.value(body.args[1 + i]).type.same_shape_and_elem(init_ty)) {
              error(tag + ": iter_arg " + std::to_string(i) + " type mismatch with iter_init");
            }
            if (i < node.results.size() && !kernel_.value(node.results[i]).type.same_shape_and_elem(init_ty)) {
              error(tag + ": result " + std::to_string(i) + " type mismatch with iter_init");
            }
          }
        }
        verify_block(body_id, visible);
        if (!body.ops.empty() && valid_op_id(body.ops.back())) {
          const Op& term = kernel_.op(body.ops.back());
          if (term.kind == OpKind::Yield) {
            if (term.operands.size() != num_iter) {
              error(tag + ": body yield operand count (" + std::to_string(term.operands.size()) +
                    ") must match iter_arg count (" + std::to_string(num_iter) + ")");
            } else {
              for (size_t i = 0; i < num_iter && i < node.results.size(); ++i) {
                if (valid_value_id(term.operands[i]) &&
                    !kernel_.value(term.operands[i]).type.same_shape_and_elem(kernel_.value(node.results[i]).type)) {
                  error(tag + ": body yield operand " + std::to_string(i) + " type mismatch with result");
                }
              }
            }
          }
        }
        break;
      }

      case OpKind::If: {
        if (node.operands.size() != 1) {
          error(tag + ": expects exactly 1 condition operand");
          break;
        }
        const Type& cty = kernel_.value(node.operands[0]).type;
        if (!cty.is_scalar() || !cty.is_bool_elem()) {
          error(tag + ": condition must be scalar bool");
        }
        if (node.regions.empty() || node.regions.size() > 2) {
          error(tag + ": expects 1 or 2 regions (then, optional else)");
          break;
        }
        if (!node.results.empty() && node.regions.size() != 2) {
          error(tag + ": value-producing If must have an else region");
        }
        for (BlockId reg_id : node.regions) {
          if (!valid_block_id(reg_id)) {
            error(tag + ": invalid region BlockId");
            continue;
          }
          const Block& reg = kernel_.block(reg_id);
          if (reg.parent_op != node.id) {
            error(tag + ": region block parent_op mismatch");
          }
          if (!reg.args.empty()) {
            error(tag + ": If region blocks must not have block arguments");
          }
          verify_block(reg_id, visible);
          if (!reg.ops.empty() && valid_op_id(reg.ops.back())) {
            const Op& term = kernel_.op(reg.ops.back());
            if (term.kind == OpKind::Yield) {
              if (term.operands.size() != node.results.size()) {
                error(tag + ": region yield operand count must match If result count");
              } else {
                for (size_t i = 0; i < node.results.size(); ++i) {
                  if (valid_value_id(term.operands[i]) &&
                      !kernel_.value(term.operands[i]).type.same_shape_and_elem(kernel_.value(node.results[i]).type)) {
                    error(tag + ": region yield operand " + std::to_string(i) + " type mismatch with If result");
                  }
                }
              }
            }
          }
        }
        break;
      }

      case OpKind::Yield: {
        if (!node.results.empty()) {
          error(tag + ": Yield must have 0 results");
        }
        break;
      }

      case OpKind::Return: {
        if (!node.operands.empty() || !node.results.empty()) {
          error(tag + ": Return must have 0 operands and 0 results");
        }
        break;
      }

      case OpKind::ThreadgroupAlloc: {
        if (!std::holds_alternative<ThreadgroupAllocAttr>(node.attr)) {
          error(tag + ": missing ThreadgroupAllocAttr");
          break;
        }
        const auto& alloc = std::get<ThreadgroupAllocAttr>(node.attr);
        if (!is_positive_shape(alloc.shape) || alloc.stages == 0) {
          error(tag + ": invalid shape or stages");
          break;
        }
        uint64_t elems = 1;
        for (size_t i = 0; i + 1 < alloc.shape.size(); ++i) {
          elems *= static_cast<uint64_t>(alloc.shape[i]);
        }
        elems *= static_cast<uint64_t>(alloc.shape.back()) + alloc.pad_elems;
        total_tg_bytes_ += elems * dtype_bytes(alloc.elem) * alloc.stages;
        break;
      }

      case OpKind::OnlineSoftmax: {
        if (!std::holds_alternative<OnlineSoftmaxAttr>(node.attr)) {
          error(tag + ": missing OnlineSoftmaxAttr");
          break;
        }
        if (node.operands.size() != 4 || !node.results.empty()) {
          error(tag + ": expects 4 operands (x_ptr, out_ptr, num_rows, row_len) and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr()) {
          error(tag + ": operands 0 and 1 must be device pointers");
        }
        if (!kernel_.value(node.operands[2]).type.is_int_elem() ||
            !kernel_.value(node.operands[3]).type.is_int_elem()) {
          error(tag + ": num_rows and row_len must be integer scalars");
        }
        break;
      }

      case OpKind::WelfordReduce: {
        if (!std::holds_alternative<WelfordNormAttr>(node.attr)) {
          error(tag + ": missing WelfordNormAttr");
          break;
        }
        if (node.operands.size() < 3 || !node.results.empty()) {
          error(tag + ": expects >= 3 operands and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr()) {
          error(tag + ": operands 0 and 1 must be device pointers");
        }
        break;
      }

      case OpKind::Pool: {
        if (!std::holds_alternative<PoolAttr>(node.attr)) {
          error(tag + ": missing PoolAttr");
          break;
        }
        const auto& pa = std::get<PoolAttr>(node.attr);
        if (pa.ndim < 1 || pa.ndim > 3) {
          error(tag + ": ndim must be 1, 2, or 3");
          break;
        }
        const size_t expected_ops = 4 + 2 * static_cast<size_t>(pa.ndim);
        if (node.operands.size() != expected_ops || !node.results.empty()) {
          error(tag + ": expects " + std::to_string(expected_ops) + " operands and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr()) {
          error(tag + ": operands 0 and 1 must be device pointers");
        }
        break;
      }

      case OpKind::SimdScan: {
        if (!std::holds_alternative<ScanAttr>(node.attr)) {
          error(tag + ": missing ScanAttr");
          break;
        }
        const auto& sa = std::get<ScanAttr>(node.attr);
        const size_t expected_ops = sa.has_mask ? 5 : 4;
        if (node.operands.size() != expected_ops || !node.results.empty()) {
          error(tag + ": expects " + std::to_string(expected_ops) + " operands and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr()) {
          error(tag + ": operands 0 and 1 must be device pointers");
        }
        if (sa.has_mask && !kernel_.value(node.operands[2]).type.is_ptr()) {
          error(tag + ": mask operand must be a device pointer");
        }
        break;
      }

      case OpKind::ImplicitGemmConv: {
        if (!std::holds_alternative<ConvAttr>(node.attr)) {
          error(tag + ": missing ConvAttr");
          break;
        }
        const auto& ca = std::get<ConvAttr>(node.attr);
        if (ca.ndim < 1 || ca.ndim > 3) {
          error(tag + ": ndim must be 1, 2, or 3");
          break;
        }
        if (node.operands.size() < 6 || !node.results.empty()) {
          error(tag + ": expects >= 6 operands and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr() ||
            !kernel_.value(node.operands[2]).type.is_ptr()) {
          error(tag + ": operands 0, 1, 2 (x, w, out) must be device pointers");
        }
        break;
      }

      case OpKind::FusedLoss: {
        if (!std::holds_alternative<LossAttr>(node.attr)) {
          error(tag + ": missing LossAttr");
          break;
        }
        if (node.operands.size() < 4 || !node.results.empty()) {
          error(tag + ": expects >= 4 operands and 0 results");
          break;
        }
        if (!kernel_.value(node.operands[0]).type.is_ptr() || !kernel_.value(node.operands[1]).type.is_ptr()) {
          error(tag + ": operands 0 and 1 must be device pointers");
        }
        break;
      }

      case OpKind::FlashAttentionTile: {
        if (!std::holds_alternative<FlashAttnAttr>(node.attr)) {
          error(tag + ": missing FlashAttnAttr");
          break;
        }
        if (node.operands.size() != 8 || !node.results.empty()) {
          error(tag + ": expects 8 operands (q, k, v, out, BH, L_q, L_kv, D) and 0 results");
          break;
        }
        for (size_t i = 0; i < 4; ++i) {
          if (!kernel_.value(node.operands[i]).type.is_ptr()) {
            error(tag + ": operand " + std::to_string(i) + " must be a device pointer");
          }
        }
        break;
      }

      case OpKind::Barrier: {
        if (!node.operands.empty() || !node.results.empty() || !std::holds_alternative<BarrierAttr>(node.attr)) {
          error(tag + ": expects 0 operands, 0 results, and BarrierAttr");
        }
        break;
      }

      default:
        for (BlockId reg_id : node.regions) {
          verify_block(reg_id, visible);
        }
        break;
    }
    verify_op_hints(node.hints, tag);
  }

  const Kernel& kernel_;
  const TargetSpec& target_;
  VerifyStage stage_;
  VerifyResult& out_;
  std::unordered_set<BlockId> visited_blocks_;
  uint64_t total_tg_bytes_ = 0;
};

}  // namespace

std::string VerifyResult::summary() const {
  if (ok) {
    return "OK";
  }
  std::ostringstream os;
  for (size_t i = 0; i < errors.size(); ++i) {
    if (i > 0) os << "\n";
    os << errors[i];
  }
  return os.str();
}

VerifyResult IRVerifier::verify(const Kernel& kernel, const TargetSpec& target, VerifyStage stage) {
  VerifyResult res;
  KernelVerifier verifier(kernel, target, stage, res);
  verifier.run();
  return res;
}

VerifyResult IRVerifier::verify(const Module& mod, const TargetSpec& target, VerifyStage stage) {
  VerifyResult combined;
  std::unordered_set<std::string> names;
  for (const auto& k : mod.kernels) {
    if (!names.insert(k->name).second) {
      combined.ok = false;
      combined.errors.push_back("Duplicate kernel name: " + k->name);
    }
    VerifyResult kr = verify(*k, target, stage);
    if (!kr.ok) {
      combined.ok = false;
      for (auto& err : kr.errors) {
        combined.errors.push_back("kernel @" + k->name + ": " + std::move(err));
      }
    }
  }
  return combined;
}

}  // namespace m1triton::ir

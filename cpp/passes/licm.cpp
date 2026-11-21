#include "passes/licm.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

void collect_defined_in_block(const Kernel& k, BlockId blk_id, std::unordered_set<ValueId>& out) {
  const Block& blk = k.block(blk_id);
  for (ValueId arg : blk.args) {
    out.insert(arg);
  }
  for (OpId op_id : blk.ops) {
    const Op& node = k.op(op_id);
    if (node.erased) continue;
    for (ValueId rid : node.results) {
      out.insert(rid);
    }
    for (BlockId reg_id : node.regions) {
      collect_defined_in_block(k, reg_id, out);
    }
  }
}

bool is_safe_to_hoist(const Kernel& k, const Op& node) {
  if (!is_pure_op(node.kind) || !node.regions.empty()) {
    return false;
  }
  if (node.kind == OpKind::Binary && std::holds_alternative<BinaryAttr>(node.attr)) {
    const BinaryOp bop = std::get<BinaryAttr>(node.attr).op;
    if ((bop == BinaryOp::Div || bop == BinaryOp::Rem) && k.value(node.operands[0]).type.is_int_elem()) {
      const ValueId rhs = node.operands[1];
      const OpId def_id = k.value(rhs).def_op;
      if (def_id == kInvalidId || k.op(def_id).erased) return false;
      const Op* cop = &k.op(def_id);
      if (cop->kind == OpKind::Splat && !cop->operands.empty()) {
        const OpId inner = k.value(cop->operands[0]).def_op;
        if (inner == kInvalidId || k.op(inner).erased) return false;
        cop = &k.op(inner);
      }
      if (cop->kind != OpKind::Constant || !std::holds_alternative<ConstAttr>(cop->attr)) {
        return false;
      }
      const auto& ca = std::get<ConstAttr>(cop->attr);
      if (const auto* i = std::get_if<int64_t>(&ca.val)) return *i != 0;
      if (const auto* u = std::get_if<uint64_t>(&ca.val)) return *u != 0;
      return false;
    }
  }
  return true;
}

bool licm_block(Kernel& k, BlockId blk_id) {
  bool changed = false;
  const auto op_snapshot = k.block(blk_id).ops;

  for (OpId op_id : op_snapshot) {
    if (k.op(op_id).erased) continue;
    const OpKind kind = k.op(op_id).kind;

    if (kind == OpKind::If) {
      const auto regions = k.op(op_id).regions;
      for (BlockId reg_id : regions) {
        if (licm_block(k, reg_id)) {
          changed = true;
        }
      }
      continue;
    }

    if (kind != OpKind::ForLoop || k.op(op_id).regions.empty()) {
      continue;
    }

    const BlockId body_id = k.op(op_id).regions[0];
    if (licm_block(k, body_id)) {
      changed = true;
    }

    std::unordered_set<ValueId> defined_in_loop;
    collect_defined_in_block(k, body_id, defined_in_loop);

    bool hoisted_in_sweep = true;
    while (hoisted_in_sweep) {
      hoisted_in_sweep = false;
      const auto body_ops = k.block(body_id).ops;
      for (OpId inner_id : body_ops) {
        if (k.op(inner_id).erased) continue;
        const Op& inner = k.op(inner_id);
        if (!is_safe_to_hoist(k, inner)) continue;

        bool invariant = true;
        for (ValueId operand : inner.operands) {
          if (defined_in_loop.contains(operand)) {
            invariant = false;
            break;
          }
        }
        if (!invariant) continue;

        auto& b_ops = k.block(body_id).ops;
        std::erase(b_ops, inner_id);

        auto& p_ops = k.block(blk_id).ops;
        auto loop_pos = std::find(p_ops.begin(), p_ops.end(), op_id);
        p_ops.insert(loop_pos, inner_id);

        k.op(inner_id).parent_block = blk_id;
        for (ValueId rid : k.op(inner_id).results) {
          defined_in_loop.erase(rid);
        }

        hoisted_in_sweep = true;
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace

PassResult LICMPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  if (kernel.entry_block == ir::kInvalidId) {
    return res;
  }
  res.changed = licm_block(kernel, kernel.entry_block);
  return res;
}

}  // namespace m1triton::passes

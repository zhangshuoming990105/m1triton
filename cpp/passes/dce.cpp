#include "passes/dce.hpp"

#include <unordered_set>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

bool op_has_side_effect(const Kernel& k, OpId op_id) {
  const Op& node = k.op(op_id);
  if (node.erased) return false;
  if (node.kind == OpKind::ForLoop || node.kind == OpKind::If) {
    for (BlockId reg_id : node.regions) {
      for (OpId inner_id : k.block(reg_id).ops) {
        const Op& inner = k.op(inner_id);
        if (inner.erased || inner.kind == OpKind::Yield) continue;
        if (op_has_side_effect(k, inner_id)) {
          return true;
        }
      }
    }
    return false;
  }
  return !is_pure_op(node.kind);
}

class DCERunner {
 public:
  explicit DCERunner(Kernel& k) : k_(k) {}

  bool run() {
    if (k_.entry_block == kInvalidId) return false;
    seed_block(k_.entry_block);
    return sweep_block(k_.entry_block);
  }

 private:
  void mark_value_live(ValueId vid) {
    if (vid == kInvalidId || vid >= k_.values.size()) return;
    if (!live_values_.insert(vid).second) return;
    const Value& val = k_.value(vid);
    if (val.def_op != kInvalidId) {
      mark_op_live(val.def_op);
    } else if (val.def_block != kInvalidId) {
      const OpId parent = k_.block(val.def_block).parent_op;
      if (parent != kInvalidId) {
        mark_op_live(parent);
      }
    }
  }

  void mark_op_live(OpId op_id) {
    if (op_id == kInvalidId || op_id >= k_.ops.size()) return;
    const Op& node = k_.op(op_id);
    if (node.erased) return;
    if (!live_ops_.insert(op_id).second) return;

    for (ValueId operand : node.operands) {
      mark_value_live(operand);
    }

    if (node.kind == OpKind::ForLoop || node.kind == OpKind::If) {
      for (BlockId reg_id : node.regions) {
        seed_block(reg_id);
      }
    }
  }

  void seed_block(BlockId blk_id) {
    const Block& blk = k_.block(blk_id);
    for (OpId op_id : blk.ops) {
      const Op& node = k_.op(op_id);
      if (node.erased) continue;
      if (node.kind == OpKind::Yield || op_has_side_effect(k_, op_id)) {
        mark_op_live(op_id);
      }
    }
  }

  bool sweep_block(BlockId blk_id) {
    bool changed = false;
    const auto op_snapshot = k_.block(blk_id).ops;
    for (auto it = op_snapshot.rbegin(); it != op_snapshot.rend(); ++it) {
      const OpId op_id = *it;
      if (k_.op(op_id).erased) continue;
      if (!live_ops_.contains(op_id)) {
        k_.erase_op(op_id);
        changed = true;
        continue;
      }
      const auto regions = k_.op(op_id).regions;
      for (BlockId reg_id : regions) {
        if (sweep_block(reg_id)) {
          changed = true;
        }
      }
    }
    return changed;
  }

  Kernel& k_;
  std::unordered_set<OpId> live_ops_;
  std::unordered_set<ValueId> live_values_;
};

}  // namespace

PassResult DCEPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  DCERunner runner(kernel);
  res.changed = runner.run();
  return res;
}

}  // namespace m1triton::passes

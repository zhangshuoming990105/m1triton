#include "passes/epilogue_dag_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

bool is_fusable_producer(OpKind k) {
  switch (k) {
    case OpKind::Dot:
    case OpKind::ImplicitGemmConv:
    case OpKind::Pool:
    case OpKind::WelfordReduce:
    case OpKind::OnlineSoftmax:
      return true;
    default:
      return false;
  }
}

const Op* def_op_of(const Kernel& k, ValueId v) {
  if (v == kInvalidId) return nullptr;
  const Value& val = k.value(v);
  if (val.def_op == kInvalidId) return nullptr;
  const Op& op = k.op(val.def_op);
  return op.erased ? nullptr : &op;
}

std::optional<double> const_scalar_val(const Kernel& k, ValueId v) {
  const Op* op = def_op_of(k, v);
  if (op == nullptr) return std::nullopt;
  if (op->kind == OpKind::Splat || op->kind == OpKind::Broadcast || op->kind == OpKind::Cast) {
    if (!op->operands.empty()) return const_scalar_val(k, op->operands[0]);
    return std::nullopt;
  }
  if (op->kind != OpKind::Constant) return std::nullopt;
  const auto* c = std::get_if<ConstAttr>(&op->attr);
  if (c == nullptr) return std::nullopt;
  if (const auto* d = std::get_if<double>(&c->val)) return *d;
  if (const auto* i = std::get_if<int64_t>(&c->val)) return static_cast<double>(*i);
  if (const auto* u = std::get_if<uint64_t>(&c->val)) return static_cast<double>(*u);
  return std::nullopt;
}

// Find single direct non-erased consumer of `v` inside the same block.
std::optional<OpId> single_user_in_block(const Kernel& k, ValueId v, BlockId blk) {
  OpId found = kInvalidId;
  for (const auto& op_ptr : k.ops) {
    if (!op_ptr || op_ptr->erased) continue;
    for (ValueId operand : op_ptr->operands) {
      if (operand == v) {
        if (op_ptr->parent_block != blk || found != kInvalidId) return std::nullopt;
        found = op_ptr->id;
      }
    }
  }
  return found == kInvalidId ? std::nullopt : std::optional<OpId>(found);
}

// Match a straight-line elementwise post-Dot chain (AddConst/SubConst/MulConst/DivConst/ReLU/Sigmoid/Tanh/Exp/Log/Abs/Neg/Sqrt)
// feeding into a Store, lowering it into an EpilogueDAGAttr on the Dot/ForLoop producer.
bool fuse_post_dot_ssa_chain(Kernel& k) {
  bool any_changed = false;
  for (const auto& blk_ptr : k.blocks) {
    if (!blk_ptr) continue;
    const BlockId blk_id = blk_ptr->id;
    const std::vector<OpId> ops_snap = blk_ptr->ops;
    for (OpId op_id : ops_snap) {
      const Op& producer = k.op(op_id);
      if (producer.erased) continue;
      // Only fuse chains rooted directly at a 2D float tile result of Dot or ForLoop containing Dot.
      bool is_dot_root = (producer.kind == OpKind::Dot);
      if (!is_dot_root && producer.kind == OpKind::ForLoop && producer.results.size() == 1 &&
          !producer.regions.empty()) {
        const auto& body = k.block(producer.regions[0]);
        for (OpId bop : body.ops) {
          if (!k.op(bop).erased && k.op(bop).kind == OpKind::Dot) {
            is_dot_root = true;
            break;
          }
        }
      }
      if (!is_dot_root || producer.results.size() != 1) continue;
      const Type& root_ty = k.value(producer.results[0]).type;
      if (!root_ty.is_tile() || !root_ty.is_float_elem() || root_ty.as_tile().shape.size() != 2) continue;

      ValueId cur_val = producer.results[0];
      std::vector<EpilogueStep> steps;
      std::vector<OpId> fused_ops;

      while (true) {
        auto user_opt = single_user_in_block(k, cur_val, blk_id);
        if (!user_opt.has_value()) break;
        const Op& u = k.op(*user_opt);
        if (u.kind == OpKind::Store && u.operands.size() >= 2 && u.operands[1] == cur_val) {
          break;
        }
        std::optional<EpilogueStep> matched;
        if (u.kind == OpKind::Unary && u.operands.size() == 1 && u.operands[0] == cur_val) {
          const auto* ua = std::get_if<UnaryAttr>(&u.attr);
          if (ua != nullptr) {
            switch (ua->op) {
              case UnaryOp::Sigmoid: matched = EpilogueStep{EpilogueOpKind::Sigmoid}; break;
              case UnaryOp::Tanh: matched = EpilogueStep{EpilogueOpKind::Tanh}; break;
              case UnaryOp::Exp: matched = EpilogueStep{EpilogueOpKind::Exp}; break;
              case UnaryOp::Log: matched = EpilogueStep{EpilogueOpKind::Log}; break;
              case UnaryOp::Abs: matched = EpilogueStep{EpilogueOpKind::Abs}; break;
              case UnaryOp::Neg: matched = EpilogueStep{EpilogueOpKind::Neg}; break;
              case UnaryOp::Sqrt: matched = EpilogueStep{EpilogueOpKind::Sqrt}; break;
              default: break;
            }
          }
        } else if (u.kind == OpKind::Binary && u.operands.size() == 2) {
          const auto* ba = std::get_if<BinaryAttr>(&u.attr);
          if (ba != nullptr) {
            const bool lhs_is_cur = (u.operands[0] == cur_val);
            const bool rhs_is_cur = (u.operands[1] == cur_val);
            if (lhs_is_cur != rhs_is_cur) {
              const ValueId other = lhs_is_cur ? u.operands[1] : u.operands[0];
              if (auto cv = const_scalar_val(k, other)) {
                const float cf = static_cast<float>(*cv);
                if (ba->op == BinaryOp::Add) matched = EpilogueStep{EpilogueOpKind::AddConst, cf};
                else if (ba->op == BinaryOp::Mul) matched = EpilogueStep{EpilogueOpKind::MulConst, cf};
                else if (ba->op == BinaryOp::Sub && lhs_is_cur) matched = EpilogueStep{EpilogueOpKind::SubConst, cf};
                else if (ba->op == BinaryOp::Sub && rhs_is_cur) matched = EpilogueStep{EpilogueOpKind::RSubConst, cf};
                else if (ba->op == BinaryOp::Div && lhs_is_cur) matched = EpilogueStep{EpilogueOpKind::DivConst, cf};
                else if (ba->op == BinaryOp::Max && cf == 0.0f) matched = EpilogueStep{EpilogueOpKind::ReLU};
                else if (ba->op == BinaryOp::Max) matched = EpilogueStep{EpilogueOpKind::MaxConst, cf};
                else if (ba->op == BinaryOp::Min) matched = EpilogueStep{EpilogueOpKind::MinConst, cf};
              }
            }
          }
        }
        if (!matched.has_value() || u.results.size() != 1) {
          steps.clear();
          fused_ops.clear();
          break;
        }
        steps.push_back(*matched);
        fused_ops.push_back(u.id);
        cur_val = u.results[0];
      }

      if (steps.empty()) continue;
      // Only commit fusion if the chain terminates directly at a Store.
      auto final_user = single_user_in_block(k, cur_val, blk_id);
      if (!final_user.has_value() || k.op(*final_user).kind != OpKind::Store) continue;

      Op& prod_mut = k.op(op_id);
      if (!prod_mut.hints.epilogue_dag.has_value()) {
        prod_mut.hints.epilogue_dag = EpilogueDAGAttr{};
      }
      for (const auto& st : steps) {
        prod_mut.hints.epilogue_dag->steps.push_back(st);
      }
      if (!k.hints.epilogue_dag.has_value()) {
        k.hints.epilogue_dag = prod_mut.hints.epilogue_dag;
      }

      // Bypass fused ops by rewiring the Store to consume the producer result directly.
      k.replace_all_uses(cur_val, prod_mut.results[0]);
      for (auto it = fused_ops.rbegin(); it != fused_ops.rend(); ++it) {
        k.erase_op(*it);
      }
      any_changed = true;
    }
  }
  return any_changed;
}

}  // namespace

PassResult EpilogueDAGFusionPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  res.ok = true;
  res.changed = false;

  // Propagate kernel-level EpilogueDAG / FusedTail hints onto the primary compute op.
  if (kernel.hints.epilogue_dag.has_value() || kernel.hints.fused_tail.has_value()) {
    for (auto& op_ptr : kernel.ops) {
      if (!op_ptr || op_ptr->erased) continue;
      if (!is_fusable_producer(op_ptr->kind)) continue;
      if (kernel.hints.epilogue_dag.has_value() && !op_ptr->hints.epilogue_dag.has_value()) {
        op_ptr->hints.epilogue_dag = kernel.hints.epilogue_dag;
        res.changed = true;
      }
      if (kernel.hints.fused_tail.has_value() && !op_ptr->hints.fused_tail.has_value()) {
        op_ptr->hints.fused_tail = kernel.hints.fused_tail;
        res.changed = true;
      }
    }
  }

  if (fuse_post_dot_ssa_chain(kernel)) {
    res.changed = true;
  }

  return res;
}

}  // namespace m1triton::passes

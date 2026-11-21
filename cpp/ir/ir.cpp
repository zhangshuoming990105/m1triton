#include "ir/ir.hpp"

#include <algorithm>

namespace m1triton::ir {

ValueId Kernel::create_value(const Type& ty, OpId def_op, uint32_t res_idx, BlockId def_block, uint32_t arg_idx,
                             std::string debug_name) {
  auto val = std::make_unique<Value>();
  val->id = static_cast<ValueId>(values.size());
  val->type = ty;
  val->def_op = def_op;
  val->result_idx = res_idx;
  val->def_block = def_block;
  val->arg_idx = arg_idx;
  val->debug_name = std::move(debug_name);
  const ValueId id = val->id;
  values.push_back(std::move(val));
  return id;
}

BlockId Kernel::create_block(OpId parent_op) {
  auto blk = std::make_unique<Block>();
  blk->id = static_cast<BlockId>(blocks.size());
  blk->parent_op = parent_op;
  const BlockId id = blk->id;
  blocks.push_back(std::move(blk));
  return id;
}

OpId Kernel::create_op(OpKind kind, BlockId parent_block, std::vector<ValueId> operands,
                       const std::vector<Type>& result_types, OpPayload attr, OpHints hints) {
  auto node = std::make_unique<Op>();
  const OpId id = static_cast<OpId>(ops.size());
  node->id = id;
  node->kind = kind;
  node->parent_block = parent_block;
  node->operands = std::move(operands);
  node->attr = std::move(attr);
  node->hints = std::move(hints);
  node->erased = false;
  node->results.reserve(result_types.size());
  for (uint32_t i = 0; i < result_types.size(); ++i) {
    node->results.push_back(create_value(result_types[i], id, i, kInvalidId, 0));
  }
  ops.push_back(std::move(node));
  if (parent_block != kInvalidId) {
    blocks.at(parent_block)->ops.push_back(id);
  }
  return id;
}

void Kernel::replace_all_uses(ValueId old_val, ValueId new_val) {
  if (old_val == new_val) {
    return;
  }
  for (auto& node : ops) {
    if (node->erased) {
      continue;
    }
    for (ValueId& operand : node->operands) {
      if (operand == old_val) {
        operand = new_val;
      }
    }
  }
}

void Kernel::erase_op(OpId op_id) {
  Op& node = op(op_id);
  if (node.erased) {
    return;
  }
  node.erased = true;
  if (node.parent_block != kInvalidId) {
    auto& blk_ops = block(node.parent_block).ops;
    std::erase(blk_ops, op_id);
  }
  for (BlockId reg_id : node.regions) {
    auto inner_ops = block(reg_id).ops;
    for (OpId inner_id : inner_ops) {
      erase_op(inner_id);
    }
  }
}

Kernel& Module::add_kernel(std::string name) {
  auto k = std::make_unique<Kernel>();
  k->name = std::move(name);
  k->entry_block = k->create_block(kInvalidId);
  Kernel& ref = *k;
  kernels.push_back(std::move(k));
  return ref;
}

const Kernel* Module::find_kernel(std::string_view name) const {
  for (const auto& k : kernels) {
    if (k->name == name) {
      return k.get();
    }
  }
  return nullptr;
}

std::string_view op_kind_name(OpKind kind) {
  switch (kind) {
    case OpKind::Constant:
      return "const";
    case OpKind::ProgramId:
      return "program_id";
    case OpKind::NumPrograms:
      return "num_programs";
    case OpKind::Arange:
      return "arange";
    case OpKind::Splat:
      return "splat";
    case OpKind::ExpandDims:
      return "expand_dims";
    case OpKind::Broadcast:
      return "broadcast";
    case OpKind::Reshape:
      return "reshape";
    case OpKind::Permute:
      return "permute";
    case OpKind::Cast:
      return "cast";
    case OpKind::Unary:
      return "unary";
    case OpKind::Binary:
      return "binary";
    case OpKind::Select:
      return "select";
    case OpKind::PtrAdd:
      return "ptr_add";
    case OpKind::Load:
      return "load";
    case OpKind::Store:
      return "store";
    case OpKind::AtomicAdd:
      return "atomic_add";
    case OpKind::ForLoop:
      return "for_loop";
    case OpKind::If:
      return "if";
    case OpKind::Yield:
      return "yield";
    case OpKind::Return:
      return "return";
    case OpKind::Dot:
      return "dot";
    case OpKind::Reduce:
      return "reduce";
    case OpKind::Scan:
      return "scan";
    case OpKind::WelfordReduce:
      return "welford_reduce";
    case OpKind::WelfordCombine:
      return "welford_combine";
    case OpKind::OnlineSoftmax:
      return "online_softmax";
    case OpKind::OnlineSoftmaxStep:
      return "online_softmax_step";
    case OpKind::ImplicitGemmConv:
      return "implicit_gemm_conv";
    case OpKind::ImplicitGemmLoad:
      return "implicit_gemm_load";
    case OpKind::FlashAttentionTile:
      return "flash_attention_tile";
    case OpKind::ThreadgroupAlloc:
      return "threadgroup_alloc";
    case OpKind::ThreadgroupStage:
      return "threadgroup_stage";
    case OpKind::Barrier:
      return "barrier";
    case OpKind::SimdgroupMatmul:
      return "simdgroup_matmul";
    case OpKind::SimdReduce:
      return "simd_reduce";
    case OpKind::SimdScan:
      return "simd_scan";
  }
  return "const";
}

bool is_pure_op(OpKind kind) {
  switch (kind) {
    case OpKind::Load:
    case OpKind::Store:
    case OpKind::AtomicAdd:
    case OpKind::ForLoop:
    case OpKind::If:
    case OpKind::Yield:
    case OpKind::Return:
    case OpKind::ImplicitGemmLoad:
    case OpKind::ThreadgroupAlloc:
    case OpKind::ThreadgroupStage:
    case OpKind::Barrier:
      return false;
    default:
      return true;
  }
}

bool is_terminator_op(OpKind kind) {
  return kind == OpKind::Return || kind == OpKind::Yield;
}

}  // namespace m1triton::ir

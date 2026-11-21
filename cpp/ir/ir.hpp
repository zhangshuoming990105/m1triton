#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ir/attrs.hpp"
#include "ir/types.hpp"

namespace m1triton::ir {

enum class OpKind : uint16_t {
  Constant,
  ProgramId,
  NumPrograms,
  Arange,
  Splat,
  ExpandDims,
  Broadcast,
  Reshape,
  Permute,
  Cast,
  Unary,
  Binary,
  Select,
  PtrAdd,
  Load,
  Store,
  AtomicAdd,
  ForLoop,
  If,
  Yield,
  Return,
  Dot,
  Reduce,
  Scan,
  WelfordReduce,
  WelfordCombine,
  OnlineSoftmax,
  OnlineSoftmaxStep,
  ImplicitGemmConv,
  ImplicitGemmLoad,
  FlashAttentionTile,
  ThreadgroupAlloc,
  ThreadgroupStage,
  Barrier,
  SimdgroupMatmul,
  SimdReduce,
  SimdScan,
};

struct Value {
  ValueId id = kInvalidId;
  Type type;
  OpId def_op = kInvalidId;
  uint32_t result_idx = 0;
  BlockId def_block = kInvalidId;
  uint32_t arg_idx = 0;
  std::string debug_name;
};

struct Op {
  OpId id = kInvalidId;
  OpKind kind = OpKind::Constant;
  BlockId parent_block = kInvalidId;
  std::vector<ValueId> operands;
  std::vector<ValueId> results;
  std::vector<BlockId> regions;
  OpPayload attr = EmptyAttr{};
  OpHints hints;
  bool erased = false;
};

struct Block {
  BlockId id = kInvalidId;
  OpId parent_op = kInvalidId;
  std::vector<ValueId> args;
  std::vector<OpId> ops;
};

struct KernelParam {
  std::string name;
  Type type;
  bool is_constexpr = false;
  int64_t constexpr_val = 0;
  uint32_t divisibility = 1;
  bool writable = false;
};

struct Kernel {
  std::string name;
  std::vector<KernelParam> params;
  OpHints hints;
  BlockId entry_block = kInvalidId;
  std::vector<std::unique_ptr<Value>> values;
  std::vector<std::unique_ptr<Op>> ops;
  std::vector<std::unique_ptr<Block>> blocks;

  const Value& value(ValueId id) const { return *values.at(id); }
  Value& value(ValueId id) { return *values.at(id); }
  const Op& op(OpId id) const { return *ops.at(id); }
  Op& op(OpId id) { return *ops.at(id); }
  const Block& block(BlockId id) const { return *blocks.at(id); }
  Block& block(BlockId id) { return *blocks.at(id); }

  ValueId create_value(const Type& ty, OpId def_op, uint32_t res_idx, BlockId def_block, uint32_t arg_idx,
                       std::string debug_name = "");
  BlockId create_block(OpId parent_op = kInvalidId);
  OpId create_op(OpKind kind, BlockId parent_block, std::vector<ValueId> operands,
                 const std::vector<Type>& result_types, OpPayload attr = EmptyAttr{}, OpHints hints = {});
  void replace_all_uses(ValueId old_val, ValueId new_val);
  void erase_op(OpId op_id);
};

struct Module {
  std::vector<std::unique_ptr<Kernel>> kernels;

  Kernel& add_kernel(std::string name);
  const Kernel* find_kernel(std::string_view name) const;
};

std::string_view op_kind_name(OpKind kind);
bool is_pure_op(OpKind kind);
bool is_terminator_op(OpKind kind);

}  // namespace m1triton::ir

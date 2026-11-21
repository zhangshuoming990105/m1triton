#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ir/ir.hpp"

namespace m1triton::ir {

struct ForLoopHandle {
  OpId op_id = kInvalidId;
  BlockId body_block = kInvalidId;
  ValueId iv = kInvalidId;
  std::vector<ValueId> iter_args;
  std::vector<ValueId> results;
};

struct IfHandle {
  OpId op_id = kInvalidId;
  BlockId then_block = kInvalidId;
  BlockId else_block = kInvalidId;
  std::vector<ValueId> results;
};

class IRBuilder {
 public:
  explicit IRBuilder(Kernel& kernel);

  Kernel& kernel() { return kernel_; }
  BlockId current_block() const { return current_block_; }
  void set_insertion_block(BlockId block_id) { current_block_ = block_id; }

  ValueId add_param(std::string name, const Type& ty, bool is_constexpr = false, int64_t constexpr_val = 0,
                    uint32_t divisibility = 1, bool writable = false);

  ValueId emit_const_i32(int32_t v);
  ValueId emit_const_u32(uint32_t v);
  ValueId emit_const_i64(int64_t v);
  ValueId emit_const_f32(float v);
  ValueId emit_const_f16(float v);
  ValueId emit_const_bool(bool v);

  ValueId emit_program_id(uint32_t axis);
  ValueId emit_num_programs(uint32_t axis);
  ValueId emit_arange(int64_t start, int64_t end);
  ValueId emit_splat(ValueId scalar_val, std::vector<int64_t> shape);
  ValueId emit_expand_dims(ValueId val, uint32_t axis);
  ValueId emit_broadcast(ValueId val, std::vector<int64_t> target_shape);
  ValueId emit_reshape(ValueId val, std::vector<int64_t> target_shape);
  ValueId emit_permute(ValueId val, std::vector<uint32_t> perm);
  ValueId emit_cast(ValueId val, ScalarDType target_dtype);

  ValueId emit_unary(UnaryOp op, ValueId val);
  ValueId emit_binary(BinaryOp op, ValueId lhs, ValueId rhs);
  ValueId emit_select(ValueId cond, ValueId true_val, ValueId false_val);
  ValueId emit_ptr_add(ValueId ptr, ValueId offset);

  ValueId emit_load(ValueId ptrs, ValueId mask = kInvalidId, ValueId other = kInvalidId,
                    LoadStoreAttr attr = {});
  OpId emit_store(ValueId ptrs, ValueId value, ValueId mask = kInvalidId, LoadStoreAttr attr = {});
  ValueId emit_atomic_add(ValueId ptrs, ValueId value, ValueId mask = kInvalidId);

  ValueId emit_dot(ValueId lhs, ValueId rhs, ValueId acc = kInvalidId, DotAttr attr = {}, OpHints hints = {});
  ValueId emit_reduce(ValueId input, ReduceAttr attr);
  ValueId emit_scan(ValueId input, ScanAttr attr, ValueId mask = kInvalidId);

  ForLoopHandle begin_for_loop(ValueId lo, ValueId hi, ValueId step, const std::vector<ValueId>& iter_inits);
  void end_for_loop(const ForLoopHandle& handle, const std::vector<ValueId>& yield_vals, BlockId resume_block);

  IfHandle begin_if(ValueId cond, const std::vector<Type>& result_types, bool with_else = true);
  void emit_yield(const std::vector<ValueId>& vals);
  void emit_return();

 private:
  ValueId broadcast_pair_if_needed(ValueId val, const Type& target_shape_like);
  Type infer_broadcast_type(const Type& a, const Type& b) const;

  Kernel& kernel_;
  BlockId current_block_ = kInvalidId;
};

}  // namespace m1triton::ir

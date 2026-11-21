#include "ir/builder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace m1triton::ir {

namespace {

ScalarDType promote_scalar_dtype(ScalarDType a, ScalarDType b) {
  if (a == b) return a;
  if (a == ScalarDType::F32 || b == ScalarDType::F32) return ScalarDType::F32;
  if (a == ScalarDType::F16 || b == ScalarDType::F16) return ScalarDType::F16;
  if (a == ScalarDType::I64 || b == ScalarDType::I64) return ScalarDType::I64;
  if (a == ScalarDType::I32 || b == ScalarDType::I32) return ScalarDType::I32;
  if (a == ScalarDType::U32 || b == ScalarDType::U32) return ScalarDType::U32;
  return ScalarDType::Bool;
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
    const int64_t db = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
    if (da == db) {
      out[i] = da;
    } else if (da == 1) {
      out[i] = db;
    } else if (db == 1) {
      out[i] = da;
    } else {
      throw std::invalid_argument("Incompatible tile shapes for broadcasting");
    }
  }
  return out;
}

Type plain_type(const Type& ty) {
  if (ty.is_tile()) {
    return Type::tile(ty.as_tile().shape, ty.as_tile().elem);
  }
  return ty;
}

}  // namespace

IRBuilder::IRBuilder(Kernel& kernel) : kernel_(kernel) {
  if (kernel_.entry_block == kInvalidId) {
    kernel_.entry_block = kernel_.create_block(kInvalidId);
  }
  current_block_ = kernel_.entry_block;
}

ValueId IRBuilder::add_param(std::string name, const Type& ty, bool is_constexpr, int64_t constexpr_val,
                             uint32_t divisibility, bool writable) {
  auto& entry = kernel_.block(kernel_.entry_block);
  const uint32_t arg_idx = static_cast<uint32_t>(entry.args.size());
  kernel_.params.push_back(KernelParam{name, ty, is_constexpr, constexpr_val, divisibility, writable});
  const ValueId val_id = kernel_.create_value(ty, kInvalidId, 0, kernel_.entry_block, arg_idx, std::move(name));
  entry.args.push_back(val_id);
  return val_id;
}

ValueId IRBuilder::emit_const_i32(int32_t v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::I32)},
                                    ConstAttr{static_cast<int64_t>(v)});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_const_u32(uint32_t v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::U32)},
                                    ConstAttr{static_cast<uint64_t>(v)});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_const_i64(int64_t v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::I64)},
                                    ConstAttr{v});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_const_f32(float v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::F32)},
                                    ConstAttr{static_cast<double>(v)});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_const_f16(float v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::F16)},
                                    ConstAttr{static_cast<double>(v)});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_const_bool(bool v) {
  const OpId id = kernel_.create_op(OpKind::Constant, current_block_, {}, {Type::scalar(ScalarDType::Bool)},
                                    ConstAttr{v});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_program_id(uint32_t axis) {
  const OpId id =
      kernel_.create_op(OpKind::ProgramId, current_block_, {}, {Type::scalar(ScalarDType::I32)}, AxisAttr{axis});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_num_programs(uint32_t axis) {
  const OpId id =
      kernel_.create_op(OpKind::NumPrograms, current_block_, {}, {Type::scalar(ScalarDType::I32)}, AxisAttr{axis});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_arange(int64_t start, int64_t end) {
  const int64_t len = std::max<int64_t>(0, end - start);
  const OpId id = kernel_.create_op(OpKind::Arange, current_block_, {}, {Type::tile({len}, ScalarDType::I32)},
                                    ArangeAttr{start, end});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_splat(ValueId scalar_val, std::vector<int64_t> shape) {
  const Type ty = kernel_.value(scalar_val).type;
  if (ty.is_tile()) {
    if (ty.as_tile().shape == shape) {
      return scalar_val;
    }
    return emit_broadcast(scalar_val, std::move(shape));
  }
  const OpId id = kernel_.create_op(OpKind::Splat, current_block_, {scalar_val},
                                    {Type::tile(std::move(shape), ty.elem_type())});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_expand_dims(ValueId val, uint32_t axis) {
  const Type ty = kernel_.value(val).type;
  std::vector<int64_t> shape;
  if (ty.is_tile()) {
    shape = ty.as_tile().shape;
  }
  if (axis > shape.size()) {
    throw std::invalid_argument("ExpandDims axis out of range");
  }
  shape.insert(shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
  const OpId id = kernel_.create_op(OpKind::ExpandDims, current_block_, {val},
                                    {Type::tile(std::move(shape), ty.elem_type())}, AxisAttr{axis});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_broadcast(ValueId val, std::vector<int64_t> target_shape) {
  const Type ty = kernel_.value(val).type;
  if (!ty.is_tile()) {
    return emit_splat(val, std::move(target_shape));
  }
  if (ty.as_tile().shape == target_shape) {
    return val;
  }
  const OpId id = kernel_.create_op(OpKind::Broadcast, current_block_, {val},
                                    {Type::tile(std::move(target_shape), ty.elem_type())});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_reshape(ValueId val, std::vector<int64_t> target_shape) {
  const Type ty = kernel_.value(val).type;
  const OpId id = kernel_.create_op(OpKind::Reshape, current_block_, {val},
                                    {Type::tile(std::move(target_shape), ty.elem_type())});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_permute(ValueId val, std::vector<uint32_t> perm) {
  const Type ty = kernel_.value(val).type;
  if (!ty.is_tile()) {
    throw std::invalid_argument("Permute requires a tile operand");
  }
  const auto& old_shape = ty.as_tile().shape;
  std::vector<int64_t> new_shape;
  new_shape.reserve(perm.size());
  for (uint32_t idx : perm) {
    if (idx >= old_shape.size()) {
      throw std::invalid_argument("Permute axis out of range");
    }
    new_shape.push_back(old_shape[idx]);
  }
  const OpId id = kernel_.create_op(OpKind::Permute, current_block_, {val},
                                    {Type::tile(std::move(new_shape), ty.elem_type())}, PermuteAttr{std::move(perm)});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_cast(ValueId val, ScalarDType target_dtype) {
  const Type ty = kernel_.value(val).type;
  if (ty.is_ptr_elem()) {
    throw std::invalid_argument("Cast cannot be applied to pointer types");
  }
  if (ty.scalar_elem() == target_dtype) {
    return val;
  }
  const Type res_ty = ty.is_tile() ? Type::tile(ty.as_tile().shape, target_dtype) : Type::scalar(target_dtype);
  const OpId id = kernel_.create_op(OpKind::Cast, current_block_, {val}, {res_ty});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::broadcast_pair_if_needed(ValueId val, const Type& target_shape_like) {
  if (!target_shape_like.is_tile()) {
    return val;
  }
  const auto& target_shape = target_shape_like.as_tile().shape;
  const Type vty = kernel_.value(val).type;
  if (!vty.is_tile()) {
    return emit_splat(val, target_shape);
  }
  if (vty.as_tile().shape == target_shape) {
    return val;
  }
  return emit_broadcast(val, target_shape);
}

Type IRBuilder::infer_broadcast_type(const Type& a, const Type& b) const {
  ElemType elem;
  if (a.is_ptr_elem()) {
    elem = a.elem_type();
  } else if (b.is_ptr_elem()) {
    elem = b.elem_type();
  } else {
    elem = promote_scalar_dtype(a.scalar_elem(), b.scalar_elem());
  }

  if (a.is_tile() && b.is_tile()) {
    return Type::tile(broadcast_shapes(a.as_tile().shape, b.as_tile().shape), elem);
  }
  if (a.is_tile()) {
    return Type::tile(a.as_tile().shape, elem);
  }
  if (b.is_tile()) {
    return Type::tile(b.as_tile().shape, elem);
  }
  if (std::holds_alternative<PtrType>(elem)) {
    const auto p = std::get<PtrType>(elem);
    return Type::ptr(p.pointee, p.space);
  }
  return Type::scalar(std::get<ScalarDType>(elem));
}

ValueId IRBuilder::emit_unary(UnaryOp op, ValueId val) {
  const Type res_ty = plain_type(kernel_.value(val).type);
  const OpId id = kernel_.create_op(OpKind::Unary, current_block_, {val}, {res_ty}, UnaryAttr{op});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_binary(BinaryOp op, ValueId lhs, ValueId rhs) {
  const Type lty = kernel_.value(lhs).type;
  const Type rty = kernel_.value(rhs).type;
  const Type bcast_ty = infer_broadcast_type(lty, rty);

  lhs = broadcast_pair_if_needed(lhs, bcast_ty);
  rhs = broadcast_pair_if_needed(rhs, bcast_ty);

  if (!bcast_ty.is_ptr_elem()) {
    const ScalarDType target_dt = bcast_ty.scalar_elem();
    lhs = emit_cast(lhs, target_dt);
    rhs = emit_cast(rhs, target_dt);
  }

  Type res_ty = bcast_ty;
  if (is_comparison_op(op)) {
    res_ty = bcast_ty.is_tile() ? Type::tile(bcast_ty.as_tile().shape, ScalarDType::Bool)
                                : Type::scalar(ScalarDType::Bool);
  }

  const OpId id = kernel_.create_op(OpKind::Binary, current_block_, {lhs, rhs}, {res_ty}, BinaryAttr{op});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_select(ValueId cond, ValueId true_val, ValueId false_val) {
  const Type cty = kernel_.value(cond).type;
  const Type tty = kernel_.value(true_val).type;
  const Type fty = kernel_.value(false_val).type;

  const Type val_bcast = infer_broadcast_type(tty, fty);
  Type shape_like = val_bcast;
  if (cty.is_tile()) {
    if (val_bcast.is_tile()) {
      shape_like = Type::tile(broadcast_shapes(val_bcast.as_tile().shape, cty.as_tile().shape), val_bcast.elem_type());
    } else {
      shape_like = Type::tile(cty.as_tile().shape, val_bcast.elem_type());
    }
  }

  cond = broadcast_pair_if_needed(cond, shape_like);
  true_val = broadcast_pair_if_needed(true_val, shape_like);
  false_val = broadcast_pair_if_needed(false_val, shape_like);

  if (!shape_like.is_ptr_elem()) {
    const ScalarDType target_dt = shape_like.scalar_elem();
    true_val = emit_cast(true_val, target_dt);
    false_val = emit_cast(false_val, target_dt);
  }

  const OpId id = kernel_.create_op(OpKind::Select, current_block_, {cond, true_val, false_val}, {shape_like});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_ptr_add(ValueId ptr, ValueId offset) {
  const Type pty = kernel_.value(ptr).type;
  const Type oty = kernel_.value(offset).type;
  if (!pty.is_ptr_elem()) {
    throw std::invalid_argument("PtrAdd requires pointer or tile-of-pointer operand");
  }
  const PtrType ptr_elem = pty.ptr_elem();

  Type res_ty;
  if (pty.is_tile() && oty.is_tile()) {
    const auto shape = broadcast_shapes(pty.as_tile().shape, oty.as_tile().shape);
    res_ty = Type::tile(shape, ptr_elem);
    ptr = broadcast_pair_if_needed(ptr, res_ty);
    offset = broadcast_pair_if_needed(offset, res_ty);
  } else if (pty.is_tile()) {
    res_ty = Type::tile(pty.as_tile().shape, ptr_elem);
  } else if (oty.is_tile()) {
    res_ty = Type::tile(oty.as_tile().shape, ptr_elem);
  } else {
    res_ty = Type::ptr(ptr_elem.pointee, ptr_elem.space);
  }

  const OpId id = kernel_.create_op(OpKind::PtrAdd, current_block_, {ptr, offset}, {res_ty});
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_load(ValueId ptrs, ValueId mask, ValueId other, LoadStoreAttr attr) {
  const Type pty = kernel_.value(ptrs).type;
  if (!pty.is_ptr_elem()) {
    throw std::invalid_argument("Load requires pointer or tile-of-pointer operand");
  }
  const ScalarDType dt = pty.ptr_elem().pointee;
  const Type res_ty = pty.is_tile() ? Type::tile(pty.as_tile().shape, dt) : Type::scalar(dt);

  std::vector<ValueId> operands;
  operands.push_back(ptrs);

  if (mask != kInvalidId) {
    mask = broadcast_pair_if_needed(mask, res_ty);
    operands.push_back(mask);
    attr.has_mask = true;
  } else {
    attr.has_mask = false;
  }

  if (other != kInvalidId) {
    other = broadcast_pair_if_needed(other, res_ty);
    other = emit_cast(other, dt);
    operands.push_back(other);
    attr.has_other = true;
  } else {
    attr.has_other = false;
  }

  if (attr.alignment_bytes == 0) {
    attr.alignment_bytes = dtype_bytes(dt);
  }
  if (attr.vec_width == 0) {
    attr.vec_width = 1;
  }

  const OpId id = kernel_.create_op(OpKind::Load, current_block_, std::move(operands), {res_ty}, attr);
  return kernel_.op(id).results[0];
}

OpId IRBuilder::emit_store(ValueId ptrs, ValueId value, ValueId mask, LoadStoreAttr attr) {
  const Type pty = kernel_.value(ptrs).type;
  if (!pty.is_ptr_elem()) {
    throw std::invalid_argument("Store requires pointer or tile-of-pointer operand");
  }
  const ScalarDType dt = pty.ptr_elem().pointee;
  const Type val_target_ty = pty.is_tile() ? Type::tile(pty.as_tile().shape, dt) : Type::scalar(dt);

  value = broadcast_pair_if_needed(value, val_target_ty);
  value = emit_cast(value, dt);

  std::vector<ValueId> operands{ptrs, value};
  if (mask != kInvalidId) {
    mask = broadcast_pair_if_needed(mask, val_target_ty);
    operands.push_back(mask);
    attr.has_mask = true;
  } else {
    attr.has_mask = false;
  }
  attr.has_other = false;

  if (attr.alignment_bytes == 0) {
    attr.alignment_bytes = dtype_bytes(dt);
  }
  if (attr.vec_width == 0) {
    attr.vec_width = 1;
  }

  return kernel_.create_op(OpKind::Store, current_block_, std::move(operands), {}, attr);
}

ValueId IRBuilder::emit_atomic_add(ValueId ptrs, ValueId value, ValueId mask) {
  const Type pty = kernel_.value(ptrs).type;
  if (!pty.is_ptr_elem()) {
    throw std::invalid_argument("AtomicAdd requires pointer or tile-of-pointer operand");
  }
  const ScalarDType dt = pty.ptr_elem().pointee;
  const Type res_ty = pty.is_tile() ? Type::tile(pty.as_tile().shape, dt) : Type::scalar(dt);

  value = broadcast_pair_if_needed(value, res_ty);
  value = emit_cast(value, dt);

  LoadStoreAttr attr;
  attr.alignment_bytes = dtype_bytes(dt);
  std::vector<ValueId> operands{ptrs, value};
  if (mask != kInvalidId) {
    mask = broadcast_pair_if_needed(mask, res_ty);
    operands.push_back(mask);
    attr.has_mask = true;
  }

  const OpId id = kernel_.create_op(OpKind::AtomicAdd, current_block_, std::move(operands), {res_ty}, attr);
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_dot(ValueId lhs, ValueId rhs, ValueId acc, DotAttr attr, OpHints hints) {
  const Type lty = kernel_.value(lhs).type;
  const Type rty = kernel_.value(rhs).type;
  if (!lty.is_tile() || !rty.is_tile()) {
    throw std::invalid_argument("Dot requires tile operands");
  }
  const auto& ls = lty.as_tile().shape;
  const auto& rs = rty.as_tile().shape;
  if (ls.size() != rs.size() || (ls.size() != 2 && ls.size() != 3)) {
    throw std::invalid_argument("Dot requires 2D or 3D tile operands of matching rank");
  }

  std::vector<int64_t> out_shape;
  const size_t offset = ls.size() == 3 ? 1 : 0;
  if (ls.size() == 3) {
    out_shape.push_back(ls[0]);
  }
  const int64_t m = attr.transpose_a ? ls[offset + 1] : ls[offset];
  const int64_t n = attr.transpose_b ? rs[offset] : rs[offset + 1];
  out_shape.push_back(m);
  out_shape.push_back(n);

  const Type out_ty = Type::tile(std::move(out_shape), attr.acc_dtype, LayoutAttr{}, attr.out_struct);
  std::vector<ValueId> operands{lhs, rhs};
  if (acc != kInvalidId) {
    acc = broadcast_pair_if_needed(acc, out_ty);
    acc = emit_cast(acc, attr.acc_dtype);
    operands.push_back(acc);
  }

  const OpId id =
      kernel_.create_op(OpKind::Dot, current_block_, std::move(operands), {out_ty}, std::move(attr), std::move(hints));
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_reduce(ValueId input, ReduceAttr attr) {
  const Type ity = kernel_.value(input).type;
  if (!ity.is_tile() || ity.as_tile().shape.empty()) {
    throw std::invalid_argument("Reduce requires a non-empty tile operand");
  }
  const auto& shape = ity.as_tile().shape;
  const int32_t rank = static_cast<int32_t>(shape.size());
  int32_t ax = attr.axis < 0 ? rank + attr.axis : attr.axis;
  if (ax < 0 || ax >= rank) {
    throw std::invalid_argument("Reduce axis out of range");
  }
  attr.axis = ax;

  std::vector<int64_t> out_shape;
  if (attr.keep_dims) {
    out_shape = shape;
    out_shape[static_cast<size_t>(ax)] = 1;
  } else {
    out_shape.reserve(shape.size() - 1);
    for (int32_t i = 0; i < rank; ++i) {
      if (i != ax) {
        out_shape.push_back(shape[static_cast<size_t>(i)]);
      }
    }
  }

  const ScalarDType val_dt = ity.scalar_elem();
  const auto make_ty = [&](ScalarDType dt) {
    return out_shape.empty() ? Type::scalar(dt) : Type::tile(out_shape, dt);
  };

  std::vector<Type> result_types;
  if (attr.result_mode == ReduceResultMode::ValueAndIndex) {
    result_types.push_back(make_ty(val_dt));
    result_types.push_back(make_ty(ScalarDType::I32));
  } else if (attr.result_mode == ReduceResultMode::IndexOnly || attr.kind == ReduceKind::Argmax ||
             attr.kind == ReduceKind::Argmin) {
    result_types.push_back(make_ty(ScalarDType::I32));
  } else {
    result_types.push_back(make_ty(val_dt));
  }

  const OpId id = kernel_.create_op(OpKind::Reduce, current_block_, {input}, result_types, attr);
  return kernel_.op(id).results[0];
}

ValueId IRBuilder::emit_scan(ValueId input, ScanAttr attr, ValueId mask) {
  const Type ity = kernel_.value(input).type;
  if (!ity.is_tile() || ity.as_tile().shape.empty()) {
    throw std::invalid_argument("Scan requires a non-empty tile operand");
  }
  const int32_t rank = static_cast<int32_t>(ity.as_tile().shape.size());
  int32_t ax = attr.axis < 0 ? rank + attr.axis : attr.axis;
  if (ax < 0 || ax >= rank) {
    throw std::invalid_argument("Scan axis out of range");
  }
  attr.axis = ax;

  const Type res_ty = plain_type(ity);
  std::vector<ValueId> operands{input};
  if (mask != kInvalidId) {
    mask = broadcast_pair_if_needed(mask, res_ty);
    operands.push_back(mask);
    attr.has_mask = true;
  } else {
    attr.has_mask = false;
  }

  const OpId id = kernel_.create_op(OpKind::Scan, current_block_, std::move(operands), {res_ty}, attr);
  return kernel_.op(id).results[0];
}

OpId IRBuilder::emit_online_softmax(ValueId x_ptr, ValueId out_ptr, ValueId num_rows, ValueId row_len,
                                    OnlineSoftmaxAttr attr, OpHints hints) {
  return kernel_.create_op(OpKind::OnlineSoftmax, current_block_, {x_ptr, out_ptr, num_rows, row_len}, {},
                           std::move(attr), std::move(hints));
}

OpId IRBuilder::emit_welford_norm(std::vector<ValueId> operands, WelfordNormAttr attr, OpHints hints) {
  return kernel_.create_op(OpKind::WelfordReduce, current_block_, std::move(operands), {}, std::move(attr),
                           std::move(hints));
}

OpId IRBuilder::emit_pool(std::vector<ValueId> operands, PoolAttr attr, OpHints hints) {
  return kernel_.create_op(OpKind::Pool, current_block_, std::move(operands), {}, std::move(attr), std::move(hints));
}

OpId IRBuilder::emit_row_scan(ValueId x_ptr, ValueId out_ptr, ValueId num_rows, ValueId row_len, ScanAttr attr,
                              ValueId mask_ptr, OpHints hints) {
  std::vector<ValueId> operands{x_ptr, out_ptr};
  if (mask_ptr != kInvalidId) {
    operands.push_back(mask_ptr);
    attr.has_mask = true;
  } else {
    attr.has_mask = false;
  }
  operands.push_back(num_rows);
  operands.push_back(row_len);
  return kernel_.create_op(OpKind::SimdScan, current_block_, std::move(operands), {}, std::move(attr),
                           std::move(hints));
}

ForLoopHandle IRBuilder::begin_for_loop(ValueId lo, ValueId hi, ValueId step,
                                        const std::vector<ValueId>& iter_inits) {
  const Type lo_ty = kernel_.value(lo).type;
  if (lo_ty.is_scalar() && lo_ty.is_int_elem()) {
    hi = emit_cast(hi, lo_ty.as_scalar());
    step = emit_cast(step, lo_ty.as_scalar());
  }

  std::vector<ValueId> operands;
  operands.reserve(3 + iter_inits.size());
  operands.push_back(lo);
  operands.push_back(hi);
  operands.push_back(step);

  std::vector<Type> result_types;
  result_types.reserve(iter_inits.size());
  for (ValueId init : iter_inits) {
    operands.push_back(init);
    result_types.push_back(plain_type(kernel_.value(init).type));
  }

  const OpId op_id = kernel_.create_op(OpKind::ForLoop, current_block_, std::move(operands), result_types);
  const BlockId body_block = kernel_.create_block(op_id);
  kernel_.op(op_id).regions.push_back(body_block);

  ForLoopHandle handle;
  handle.op_id = op_id;
  handle.body_block = body_block;
  handle.results = kernel_.op(op_id).results;

  handle.iv = kernel_.create_value(lo_ty, kInvalidId, 0, body_block, 0, "iv");
  kernel_.block(body_block).args.push_back(handle.iv);

  handle.iter_args.reserve(iter_inits.size());
  for (size_t i = 0; i < iter_inits.size(); ++i) {
    const ValueId arg_val = kernel_.create_value(result_types[i], kInvalidId, 0, body_block,
                                                 static_cast<uint32_t>(i + 1), "iter_" + std::to_string(i));
    kernel_.block(body_block).args.push_back(arg_val);
    handle.iter_args.push_back(arg_val);
  }

  current_block_ = body_block;
  return handle;
}

void IRBuilder::end_for_loop(const ForLoopHandle& handle, const std::vector<ValueId>& yield_vals,
                             BlockId resume_block) {
  current_block_ = handle.body_block;
  emit_yield(yield_vals);
  if (resume_block != kInvalidId) {
    current_block_ = resume_block;
  } else {
    current_block_ = kernel_.op(handle.op_id).parent_block;
  }
}

IfHandle IRBuilder::begin_if(ValueId cond, const std::vector<Type>& result_types, bool with_else) {
  const OpId op_id = kernel_.create_op(OpKind::If, current_block_, {cond}, result_types);
  const BlockId then_block = kernel_.create_block(op_id);
  kernel_.op(op_id).regions.push_back(then_block);

  BlockId else_block = kInvalidId;
  if (with_else) {
    else_block = kernel_.create_block(op_id);
    kernel_.op(op_id).regions.push_back(else_block);
  }

  IfHandle handle;
  handle.op_id = op_id;
  handle.then_block = then_block;
  handle.else_block = else_block;
  handle.results = kernel_.op(op_id).results;

  current_block_ = then_block;
  return handle;
}

void IRBuilder::emit_yield(const std::vector<ValueId>& vals) {
  kernel_.create_op(OpKind::Yield, current_block_, vals, {});
}

void IRBuilder::emit_return() {
  kernel_.create_op(OpKind::Return, current_block_, {}, {});
}

}  // namespace m1triton::ir

#include "ir/attrs.hpp"

namespace m1triton::ir {

std::string_view unary_op_name(UnaryOp op) {
  switch (op) {
    case UnaryOp::Exp:
      return "exp";
    case UnaryOp::Log:
      return "log";
    case UnaryOp::Sqrt:
      return "sqrt";
    case UnaryOp::Rsqrt:
      return "rsqrt";
    case UnaryOp::Abs:
      return "abs";
    case UnaryOp::Neg:
      return "neg";
    case UnaryOp::Tanh:
      return "tanh";
    case UnaryOp::Erf:
      return "erf";
    case UnaryOp::Floor:
      return "floor";
    case UnaryOp::Ceil:
      return "ceil";
    case UnaryOp::Sin:
      return "sin";
    case UnaryOp::Cos:
      return "cos";
    case UnaryOp::Sigmoid:
      return "sigmoid";
    case UnaryOp::Not:
      return "not";
  }
  return "neg";
}

std::string_view binary_op_name(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return "add";
    case BinaryOp::Sub:
      return "sub";
    case BinaryOp::Mul:
      return "mul";
    case BinaryOp::Div:
      return "div";
    case BinaryOp::Rem:
      return "rem";
    case BinaryOp::Max:
      return "max";
    case BinaryOp::Min:
      return "min";
    case BinaryOp::Pow:
      return "pow";
    case BinaryOp::And:
      return "and";
    case BinaryOp::Or:
      return "or";
    case BinaryOp::Xor:
      return "xor";
    case BinaryOp::Shl:
      return "shl";
    case BinaryOp::Shr:
      return "shr";
    case BinaryOp::CmpEq:
      return "cmp_eq";
    case BinaryOp::CmpNe:
      return "cmp_ne";
    case BinaryOp::CmpLt:
      return "cmp_lt";
    case BinaryOp::CmpLe:
      return "cmp_le";
    case BinaryOp::CmpGt:
      return "cmp_gt";
    case BinaryOp::CmpGe:
      return "cmp_ge";
  }
  return "add";
}

bool is_comparison_op(BinaryOp op) {
  switch (op) {
    case BinaryOp::CmpEq:
    case BinaryOp::CmpNe:
    case BinaryOp::CmpLt:
    case BinaryOp::CmpLe:
    case BinaryOp::CmpGt:
    case BinaryOp::CmpGe:
      return true;
    default:
      return false;
  }
}

bool is_commutative_op(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
    case BinaryOp::Mul:
    case BinaryOp::Max:
    case BinaryOp::Min:
    case BinaryOp::And:
    case BinaryOp::Or:
    case BinaryOp::Xor:
    case BinaryOp::CmpEq:
    case BinaryOp::CmpNe:
      return true;
    default:
      return false;
  }
}

std::string_view epilogue_op_name(EpilogueOpKind op) {
  switch (op) {
    case EpilogueOpKind::AddConst: return "add_const";
    case EpilogueOpKind::SubConst: return "sub_const";
    case EpilogueOpKind::RSubConst: return "rsub_const";
    case EpilogueOpKind::MulConst: return "mul_const";
    case EpilogueOpKind::DivConst: return "div_const";
    case EpilogueOpKind::MinConst: return "min_const";
    case EpilogueOpKind::MaxConst: return "max_const";
    case EpilogueOpKind::Clamp: return "clamp";
    case EpilogueOpKind::AddBias: return "add_bias";
    case EpilogueOpKind::SubBias: return "sub_bias";
    case EpilogueOpKind::MulBias: return "mul_bias";
    case EpilogueOpKind::DivBias: return "div_bias";
    case EpilogueOpKind::MinBias: return "min_bias";
    case EpilogueOpKind::MaxBias: return "max_bias";
    case EpilogueOpKind::AddScalar: return "add_scalar";
    case EpilogueOpKind::SubScalar: return "sub_scalar";
    case EpilogueOpKind::MulScalar: return "mul_scalar";
    case EpilogueOpKind::DivScalar: return "div_scalar";
    case EpilogueOpKind::AddResidual: return "add_residual";
    case EpilogueOpKind::SubResidual: return "sub_residual";
    case EpilogueOpKind::MulResidual: return "mul_residual";
    case EpilogueOpKind::SaveReg: return "save_reg";
    case EpilogueOpKind::AddSaved: return "add_saved";
    case EpilogueOpKind::SubSaved: return "sub_saved";
    case EpilogueOpKind::MulSaved: return "mul_saved";
    case EpilogueOpKind::ReLU: return "relu";
    case EpilogueOpKind::LeakyReLU: return "leaky_relu";
    case EpilogueOpKind::Sigmoid: return "sigmoid";
    case EpilogueOpKind::Tanh: return "tanh";
    case EpilogueOpKind::GELU: return "gelu";
    case EpilogueOpKind::GELUTanh: return "gelu_tanh";
    case EpilogueOpKind::Swish: return "swish";
    case EpilogueOpKind::Mish: return "mish";
    case EpilogueOpKind::HardSwish: return "hardswish";
    case EpilogueOpKind::HardSigmoid: return "hardsigmoid";
    case EpilogueOpKind::HardTanh: return "hardtanh";
    case EpilogueOpKind::ELU: return "elu";
    case EpilogueOpKind::SELU: return "selu";
    case EpilogueOpKind::Softplus: return "softplus";
    case EpilogueOpKind::Softsign: return "softsign";
    case EpilogueOpKind::Log: return "log";
    case EpilogueOpKind::Exp: return "exp";
    case EpilogueOpKind::Abs: return "abs";
    case EpilogueOpKind::Neg: return "neg";
    case EpilogueOpKind::Sqrt: return "sqrt";
  }
  return "relu";
}

std::optional<EpilogueOpKind> parse_epilogue_op_name(std::string_view name) {
  if (name == "add_const") return EpilogueOpKind::AddConst;
  if (name == "sub_const") return EpilogueOpKind::SubConst;
  if (name == "rsub_const") return EpilogueOpKind::RSubConst;
  if (name == "mul_const") return EpilogueOpKind::MulConst;
  if (name == "div_const") return EpilogueOpKind::DivConst;
  if (name == "min_const") return EpilogueOpKind::MinConst;
  if (name == "max_const") return EpilogueOpKind::MaxConst;
  if (name == "clamp") return EpilogueOpKind::Clamp;
  if (name == "add_bias") return EpilogueOpKind::AddBias;
  if (name == "sub_bias") return EpilogueOpKind::SubBias;
  if (name == "mul_bias") return EpilogueOpKind::MulBias;
  if (name == "div_bias") return EpilogueOpKind::DivBias;
  if (name == "min_bias") return EpilogueOpKind::MinBias;
  if (name == "max_bias") return EpilogueOpKind::MaxBias;
  if (name == "add_scalar") return EpilogueOpKind::AddScalar;
  if (name == "sub_scalar") return EpilogueOpKind::SubScalar;
  if (name == "mul_scalar") return EpilogueOpKind::MulScalar;
  if (name == "div_scalar") return EpilogueOpKind::DivScalar;
  if (name == "add_residual") return EpilogueOpKind::AddResidual;
  if (name == "sub_residual") return EpilogueOpKind::SubResidual;
  if (name == "mul_residual") return EpilogueOpKind::MulResidual;
  if (name == "save_reg") return EpilogueOpKind::SaveReg;
  if (name == "add_saved") return EpilogueOpKind::AddSaved;
  if (name == "sub_saved") return EpilogueOpKind::SubSaved;
  if (name == "mul_saved") return EpilogueOpKind::MulSaved;
  if (name == "relu") return EpilogueOpKind::ReLU;
  if (name == "leaky_relu") return EpilogueOpKind::LeakyReLU;
  if (name == "sigmoid") return EpilogueOpKind::Sigmoid;
  if (name == "tanh") return EpilogueOpKind::Tanh;
  if (name == "gelu") return EpilogueOpKind::GELU;
  if (name == "gelu_tanh") return EpilogueOpKind::GELUTanh;
  if (name == "swish" || name == "silu") return EpilogueOpKind::Swish;
  if (name == "mish") return EpilogueOpKind::Mish;
  if (name == "hardswish") return EpilogueOpKind::HardSwish;
  if (name == "hardsigmoid") return EpilogueOpKind::HardSigmoid;
  if (name == "hardtanh") return EpilogueOpKind::HardTanh;
  if (name == "elu") return EpilogueOpKind::ELU;
  if (name == "selu") return EpilogueOpKind::SELU;
  if (name == "softplus") return EpilogueOpKind::Softplus;
  if (name == "softsign") return EpilogueOpKind::Softsign;
  if (name == "log") return EpilogueOpKind::Log;
  if (name == "exp") return EpilogueOpKind::Exp;
  if (name == "abs") return EpilogueOpKind::Abs;
  if (name == "neg") return EpilogueOpKind::Neg;
  if (name == "sqrt") return EpilogueOpKind::Sqrt;
  return std::nullopt;
}

bool epilogue_op_needs_ptr_arg(EpilogueOpKind op) {
  switch (op) {
    case EpilogueOpKind::AddBias:
    case EpilogueOpKind::SubBias:
    case EpilogueOpKind::MulBias:
    case EpilogueOpKind::DivBias:
    case EpilogueOpKind::MinBias:
    case EpilogueOpKind::MaxBias:
    case EpilogueOpKind::AddResidual:
    case EpilogueOpKind::SubResidual:
    case EpilogueOpKind::MulResidual:
      return true;
    default:
      return false;
  }
}

bool epilogue_op_needs_scalar_arg(EpilogueOpKind op) {
  switch (op) {
    case EpilogueOpKind::AddScalar:
    case EpilogueOpKind::SubScalar:
    case EpilogueOpKind::MulScalar:
    case EpilogueOpKind::DivScalar:
      return true;
    default:
      return false;
  }
}

bool epilogue_op_uses_reg_slot(EpilogueOpKind op) {
  switch (op) {
    case EpilogueOpKind::SaveReg:
    case EpilogueOpKind::AddSaved:
    case EpilogueOpKind::SubSaved:
    case EpilogueOpKind::MulSaved:
      return true;
    default:
      return false;
  }
}

std::string_view tail_kind_name(TailKind kind) {
  switch (kind) {
    case TailKind::None: return "none";
    case TailKind::RowSum: return "row_sum";
    case TailKind::RowMean: return "row_mean";
    case TailKind::RowMax: return "row_max";
    case TailKind::RowMin: return "row_min";
    case TailKind::RowLogSumExp: return "row_logsumexp";
    case TailKind::SpatialGlobalAvgPool: return "spatial_global_avg_pool";
    case TailKind::SpatialGlobalSum: return "spatial_global_sum";
    case TailKind::SpatialGlobalMax: return "spatial_global_max";
    case TailKind::Pool1DMax: return "pool1d_max";
    case TailKind::Pool1DAvg: return "pool1d_avg";
  }
  return "none";
}

std::optional<TailKind> parse_tail_kind_name(std::string_view name) {
  if (name == "none") return TailKind::None;
  if (name == "row_sum") return TailKind::RowSum;
  if (name == "row_mean") return TailKind::RowMean;
  if (name == "row_max") return TailKind::RowMax;
  if (name == "row_min") return TailKind::RowMin;
  if (name == "row_logsumexp") return TailKind::RowLogSumExp;
  if (name == "spatial_global_avg_pool") return TailKind::SpatialGlobalAvgPool;
  if (name == "spatial_global_sum") return TailKind::SpatialGlobalSum;
  if (name == "spatial_global_max") return TailKind::SpatialGlobalMax;
  if (name == "pool1d_max") return TailKind::Pool1DMax;
  if (name == "pool1d_avg") return TailKind::Pool1DAvg;
  return std::nullopt;
}

}  // namespace m1triton::ir

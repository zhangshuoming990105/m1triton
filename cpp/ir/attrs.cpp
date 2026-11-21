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

}  // namespace m1triton::ir

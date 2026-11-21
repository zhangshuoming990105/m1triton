#pragma once

#include <ostream>
#include <string>

#include "ir/ir.hpp"

namespace m1triton::ir {

class IRPrinter {
 public:
  static std::string print(const Kernel& kernel);
  static std::string print(const Module& mod);
  static void print(const Kernel& kernel, std::ostream& os);
  static void print(const Module& mod, std::ostream& os);
};

}  // namespace m1triton::ir

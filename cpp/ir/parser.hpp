#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "ir/ir.hpp"

namespace m1triton::ir {

struct ParseResult {
  std::unique_ptr<Module> module;
  bool ok = false;
  std::string error;
};

class IRParser {
 public:
  static ParseResult parse(std::string_view text);
};

}  // namespace m1triton::ir

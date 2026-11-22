#include "passes/algebraic_norm_fold.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace m1triton::passes {

namespace {

using namespace m1triton::ir;

bool is_exact_zero(float v) { return v == 0.0f; }
bool is_exact_one(float v) { return v == 1.0f; }

bool simplify_epilogue_dag(EpilogueDAGAttr& dag) {
  bool any_changed = false;
  for (int iter = 0; iter < 8; ++iter) {
    bool changed = false;
    auto& steps = dag.steps;
    std::vector<EpilogueStep> out;
    out.reserve(steps.size());

    for (size_t i = 0; i < steps.size(); ++i) {
      EpilogueStep cur = steps[i];

      // Normalize SubConst(c) -> AddConst(-c) and DivConst(d) -> MulConst(1/d) when finite & non-zero.
      if (cur.op == EpilogueOpKind::SubConst && std::isfinite(cur.param0)) {
        cur.op = EpilogueOpKind::AddConst;
        cur.param0 = -cur.param0;
        changed = true;
      } else if (cur.op == EpilogueOpKind::DivConst && cur.param0 != 0.0f && std::isfinite(1.0f / cur.param0)) {
        cur.op = EpilogueOpKind::MulConst;
        cur.param0 = 1.0f / cur.param0;
        changed = true;
      }

      // Drop exact algebraic identities.
      if (cur.op == EpilogueOpKind::AddConst && is_exact_zero(cur.param0)) {
        changed = true;
        continue;
      }
      if (cur.op == EpilogueOpKind::MulConst && is_exact_one(cur.param0)) {
        changed = true;
        continue;
      }

      // Fold SaveReg(s) + MulConst(a) + AddSaved(s) -> MulConst(a + 1) when slot s is dead after.
      if (cur.op == EpilogueOpKind::SaveReg && i + 2 < steps.size() &&
          steps[i + 1].op == EpilogueOpKind::MulConst &&
          steps[i + 2].op == EpilogueOpKind::AddSaved && steps[i + 2].arg_idx == cur.arg_idx) {
        bool reused = false;
        for (size_t j = i + 3; j < steps.size(); ++j) {
          if (epilogue_op_uses_reg_slot(steps[j].op) && steps[j].arg_idx == cur.arg_idx) {
            reused = (steps[j].op != EpilogueOpKind::SaveReg);
            break;
          }
        }
        if (!reused) {
          EpilogueStep folded{EpilogueOpKind::MulConst, steps[i + 1].param0 + 1.0f, 0.0f, -1};
          if (!is_exact_one(folded.param0)) out.push_back(folded);
          i += 2;
          changed = true;
          continue;
        }
      }

      // Fold SaveReg(s) + AddSaved(s) -> MulConst(2.0f) when slot s is dead after.
      if (cur.op == EpilogueOpKind::SaveReg && i + 1 < steps.size() &&
          steps[i + 1].op == EpilogueOpKind::AddSaved && steps[i + 1].arg_idx == cur.arg_idx) {
        bool reused = false;
        for (size_t j = i + 2; j < steps.size(); ++j) {
          if (epilogue_op_uses_reg_slot(steps[j].op) && steps[j].arg_idx == cur.arg_idx) {
            reused = (steps[j].op != EpilogueOpKind::SaveReg);
            break;
          }
        }
        if (!reused) {
          out.push_back(EpilogueStep{EpilogueOpKind::MulConst, 2.0f, 0.0f, -1});
          i += 1;
          changed = true;
          continue;
        }
      }

      // Combine with previous step in out.
      if (!out.empty()) {
        EpilogueStep& prev = out.back();
        if (prev.op == EpilogueOpKind::AddConst && cur.op == EpilogueOpKind::AddConst) {
          prev.param0 += cur.param0;
          if (is_exact_zero(prev.param0)) out.pop_back();
          changed = true;
          continue;
        }
        if (prev.op == EpilogueOpKind::MulConst && cur.op == EpilogueOpKind::MulConst) {
          prev.param0 *= cur.param0;
          if (is_exact_one(prev.param0)) out.pop_back();
          changed = true;
          continue;
        }
        if (prev.op == EpilogueOpKind::ReLU && cur.op == EpilogueOpKind::ReLU) {
          changed = true;
          continue;
        }
        if (prev.op == EpilogueOpKind::Clamp && cur.op == EpilogueOpKind::Clamp) {
          const float lo = std::max(prev.param0, cur.param0);
          const float hi = std::min(prev.param1, cur.param1);
          if (lo <= hi) {
            prev.param0 = lo;
            prev.param1 = hi;
            changed = true;
            continue;
          }
        }
      }

      out.push_back(cur);
    }

    if (changed) {
      steps = std::move(out);
      any_changed = true;
    } else {
      break;
    }
  }
  return any_changed;
}

}  // namespace

PassResult AlgebraicNormFoldingPass::run(ir::Kernel& kernel, const PassContext&) {
  PassResult res;
  res.ok = true;
  res.changed = false;

  if (kernel.hints.epilogue_dag.has_value()) {
    if (simplify_epilogue_dag(*kernel.hints.epilogue_dag)) {
      res.changed = true;
    }
    if (kernel.hints.epilogue_dag->steps.empty()) {
      kernel.hints.epilogue_dag.reset();
      res.changed = true;
    }
  }

  for (auto& op_ptr : kernel.ops) {
    if (!op_ptr || op_ptr->erased) continue;
    if (op_ptr->hints.epilogue_dag.has_value()) {
      if (simplify_epilogue_dag(*op_ptr->hints.epilogue_dag)) {
        res.changed = true;
      }
      if (op_ptr->hints.epilogue_dag->steps.empty()) {
        op_ptr->hints.epilogue_dag.reset();
        res.changed = true;
      }
    }
  }

  return res;
}

}  // namespace m1triton::passes

#pragma once

#include "llvm.h"

namespace polyregion::backend::details {

class CPUTargetSpecificHandler final : public TargetSpecificHandler {
  void witnessEntry(CodeGen &cg, llvm::Function &fn) override;
  ValPtr mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) override;
  ValPtr mkMathVal(CodeGen &cg, const Expr::MathOp &op) override;
};

} // namespace polyregion::backend::details
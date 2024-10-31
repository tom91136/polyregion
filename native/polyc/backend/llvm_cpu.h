#pragma once

#include "llvm.h"

namespace polyregion::backend::details {

class CPUTargetSpecificHandler final : public TargetSpecificHandler {
  void witnessFn(CodeGen &, llvm::Function &, const Function &source) override;
  ValPtr mkSpecVal(CodeGen &, const Expr::SpecOp &) override;
  ValPtr mkMathVal(CodeGen &, const Expr::MathOp &) override;
};

} // namespace polyregion::backend::details
#pragma once

#include "llvm.h"

namespace polyregion::backend::details {

class OpenCLTargetSpecificHandler final : public TargetSpecificHandler {
  void witnessEntry(CodeGen &, llvm::Function &) override;
  ValPtr mkSpecVal(CodeGen &, const Expr::SpecOp &) override;
  ValPtr mkMathVal(CodeGen &, const Expr::MathOp &) override;
};

} // namespace polyregion::backend::details
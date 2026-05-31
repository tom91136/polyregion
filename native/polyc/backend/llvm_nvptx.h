#pragma once

#include "llvm.h"

namespace polyregion::backend::details {

inline constexpr llvm::StringRef PolycDynSharedGlobal = "polyc_dyn_shared";

class NVPTXTargetSpecificHandler final : public TargetSpecificHandler {
  void witnessFn(CodeGen &, llvm::Function &, const Function &source) override;
  ValPtr mkSpecVal(CodeGen &, const Expr::SpecOp &) override;
  ValPtr mkMathVal(CodeGen &, const Expr::MathOp &) override;
  void postProcessModule(CodeGen &) override;
};

} // namespace polyregion::backend::details
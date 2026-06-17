#pragma once

#include "llvm.h"

namespace polyregion::backend::details {

class SPIRVOpenCLTargetSpecificHandler : public TargetSpecificHandler {
public:
  void witnessFn(CodeGen &, llvm::Function &, const Function &source) override;
  ValPtr mkSpecVal(CodeGen &, const Expr::SpecOp &) override;
  ValPtr mkMathVal(CodeGen &, const Expr::MathOp &) override;
};

class SPIRVVulkanTargetSpecificHandler final : public SPIRVOpenCLTargetSpecificHandler {
public:
  void witnessFn(CodeGen &, llvm::Function &, const Function &source) override;
  ValPtr mkSpecVal(CodeGen &, const Expr::SpecOp &) override;
  ValPtr mkMathVal(CodeGen &, const Expr::MathOp &) override;
  ValPtr isNaN(CodeGen &, llvm::Value *from) override;
};

} // namespace polyregion::backend::details
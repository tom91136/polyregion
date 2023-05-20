#pragma once
#include "llvm.h"

namespace polyregion::backend {

class NVPTXTargetSpecificHandler : public LLVMBackend::TargetSpecificHandler {
  void witnessEntry(LLVMBackend::AstTransformer &xform, llvm::Module &mod, llvm::Function &fn) override;
  ValPtr mkSpecVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::SpecOp &op) override;
  ValPtr mkMathVal(LLVMBackend::AstTransformer &xform, llvm::Function *fn, const Expr::MathOp &op) override;
};

} // namespace polyregion::backend
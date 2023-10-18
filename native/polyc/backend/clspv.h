#pragma once

#include "llvm/IR/Module.h"
namespace polyregion::backend::clspv {
int RunPassPipeline(llvm::Module &M, char OptimizationLevel, llvm::raw_svector_ostream *binaryStream);
}

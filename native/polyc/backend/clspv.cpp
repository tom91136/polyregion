#include "clspv.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

int polyregion::backend::clspv::RunPassPipeline(llvm::Module &M, char OptimizationLevel, llvm::raw_svector_ostream *binaryStream) {
  throw std::logic_error("Unimplemented");
}

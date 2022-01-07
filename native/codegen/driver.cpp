
#define ENABLE_BACKTRACES 0

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "polyregion.h"
#include "utils.hpp"

int main(int argc, char *argv[]) {

  polyregion_initialise();

  std::vector<uint8_t> xs = polyregion::readNStruct<uint8_t>("../ast.bin");

  polyregion_buffer buffer{xs.size(), xs.data()};
  polyregion_compile(&buffer);

  return 0;
}

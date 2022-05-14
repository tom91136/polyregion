#include "backend/llvmc.h"
#include "compiler.h"
#include "utils.hpp"
#include <iostream>

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char *argv[]) {

  polyregion::compiler::initialise();

//  std::vector<uint8_t> xs = polyregion::read_struct<uint8_t>("../ast.bin");

  using namespace llvm;

  auto ctx = std::make_unique<llvm::LLVMContext>();

  llvm::SMDiagnostic Err;
  auto modExt = llvm::parseIRFile("/home/tom/Nextcloud/vecAdd-cuda-nvptx64-nvidia-cuda-sm_61.ll", Err, *ctx, [&](llvm::StringRef DataLayoutTargetTriple) -> Optional<std::string>  {


    std::cout << DataLayoutTargetTriple.str() << std::endl;
    return {};
  });

  auto c = polyregion::backend::llvmc::compileModule(true, std::move(modExt), *ctx);


  std::cout << c << std::endl;


  std::cout << "Done" << std::endl;
  return EXIT_SUCCESS;
}

#include "polyregion.h"
#include "ast.h"
#include "codegen/llvm.h"
#include "codegen/opencl.h"
#include "json.hpp"

#include "generated/polyast_codec.h"

#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include <iostream>

std::atomic_bool init = false;

void polyregion_initialise() {
  if (!init) {
    init = true;

//    int argc = 0;
//     char *argv[]  = {};
//    llvm::InitLLVM X(argc, argv);
    std::cout << "Init LLVM..." << std::endl;

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();
  }
}

polyregion_program *polyregion_compile(polyregion_buffer *ast) {
  if (!init) {
    return nullptr;

  }

  std::cout << "Compile: " << std::endl;

  using nlohmann::json;
  using namespace polyregion;
  std::cout << "[polyregion-native] Len  : " << ast->size << std::endl;
  auto j = json::from_msgpack(ast->data, ast->data + ast->size);
  std::cout << "[polyregion-native] JSON :" << j << std::endl;
  auto x = polyast::function_json(j);
  std::cout << "[polyregion-native] AST  :" << x << std::endl;
  std::cout << "[polyregion-native] Repr :" << polyast::repr(x) << std::endl;

  codegen::OpenCLCodeGen oclGen;
  oclGen.run(x);
  codegen::LLVMCodeGen gen;
  gen.run(x);

  return nullptr;
}

void polyregion_release(polyregion_program *buffer) {
  if (buffer) {
    std::cout << "Release: " << std::endl;
  }
}
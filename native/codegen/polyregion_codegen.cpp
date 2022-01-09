#include <iostream>
#include <unordered_map>

#include "ast.h"
#include "backend/llvm.h"
#include "backend/opencl.h"
#include "generated/polyast_codec.h"
#include "json.hpp"
#include "llc.h"
#include "polyregion_codegen.h"

std::atomic_bool init = false;

void polyregion_initialise() {
  if (!init) {
    init = true;

    //    int argc = 0;
    //     char *argv[]  = {};
    //    llvm::InitLLVM X(argc, argv);
    std::cout << "Init LLVM..." << std::endl;

    //    llvm::InitializeNativeTarget();
    //    llvm::InitializeNativeTargetAsmPrinter();
    //
    //    llvm::InitializeAllTargets();
    //    llvm::InitializeAllTargetInfos();
    //    llvm::InitializeAllTargetMCs();
    //    llvm::InitializeAllDisassemblers();

    llc::setup();
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

  backend::OpenCL oclGen;
  oclGen.run(x);
  backend::LLVM gen;
  gen.run(x);

  return nullptr;
}

void polyregion_release(polyregion_program *buffer) {
  if (buffer) {
    std::cout << "Release: " << std::endl;
  }
}
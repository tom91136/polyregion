#include "backend/llvmc.h"
#include "compiler.h"
#include "utils.hpp"
#include <iostream>

#include "ast.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
//#include "lld"

int main(int argc, char *argv[]) {

  polyregion::compiler::initialise();

  //  std::vector<uint8_t> xs = polyregion::read_struct<uint8_t>("../ast.bin");

  auto ctx = std::make_unique<llvm::LLVMContext>();

  llvm::SMDiagnostic Err;

  // "/home/tom/Nextcloud/vecAdd-cuda-nvptx64-nvidia-cuda-sm_61.ll"
  //"/home/tom/Nextcloud/vecAdd-cuda-nvptx64-nvidia-cuda-sm_61.ll"
  //  auto modExt = llvm::parseIRFile("/home/tom/Nextcloud/vecAdd-hip-amdgcn-amd-amdhsa-gfx906.ll", Err, *ctx,
  //                                  [&](llvm::StringRef DataLayoutTargetTriple) -> llvm::Optional<std::string> {
  //                                    std::cout << DataLayoutTargetTriple.str() << std::endl;
  //                                    return {};
  //                                  });
  //
  //
  using namespace polyregion::polyast;
  using namespace polyregion::polyast::Stmt;
  using namespace polyregion::polyast::Term;
  using namespace polyregion::polyast::Expr;

  Function fn(
      Sym({"foo"}), {}, {}, {Named("xs", Type::Array(Type::Int())), Named("x", Type::Int())}, {}, Type::Unit(),
      {

          Var(Named("gid", Type::Int()), {NullaryIntrinsic(NullaryIntrinsicKind::GpuGlobalIdxX(), Type::Int())}),

          Var(Named("xs@gid", Type::Int()), //
              {Index(Select({}, Named("xs", Type::Array(Type::Int()))), Select({}, Named("gid", Type::Int())),
                     Type::Int())}),

          Var(Named("result", Type::Int()), //
              {
                  BinaryIntrinsic(Select({}, Named("xs@gid", Type::Int())), //
                                  Select({}, Named("gid", Type::Int())),    //
                                  BinaryIntrinsicKind::Add(), Type::Int())  //
              }),

          Var(Named("resultX2", Type::Int()), //
              {
                  BinaryIntrinsic(Select({}, Named("result", Type::Int())), //
                                  Select({}, Named("x", Type::Int())),      //
                                  BinaryIntrinsicKind::Mul(), Type::Int())  //
              }),

          Update(Select({}, Named("xs", Type::Array(Type::Int()))), Select({}, Named("gid", Type::Int())),
                 Select({}, Named("resultX2", Type::Int()))),
          Return(Alias(UnitConst())),
      });

  Program p(fn, {}, {});
  std::cout << repr(p) << std::endl;
  polyregion::compiler::Options opt{polyregion::compiler::Target::Object_LLVM_AMDGCN, "gfx906"};
  //  polyregion::compiler::Options opt{polyregion::compiler::Target::Object_LLVM_NVPTX64, "sm_61"};
  auto c = polyregion::compiler::compile(p, opt);
//  std::cout << c << std::endl;

  if (c.binary) {
    std::ofstream outfile("bin_" + opt.arch.value_or("no_arch") + ".so",
                          std::ios::out | std::ios::binary | std::ios::trunc);
    outfile.write(c.binary->data(), c.binary->size());
    outfile.close();
  }

  //  using namespace polyregion;
  //
  ////  auto triple = backend::llvmc::NVIDIA_NVPTX64;
  //  auto triple = llvm::Triple("amdgcn-amd-amdhsa");
  //
  //  backend::llvmc::TargetInfo info {
  //      .triple = triple,
  //      .target = backend::llvmc::targetFromTriple(triple),
  //      .cpu = {.uArch = "gfx906"}
  //  };
  //  auto c = polyregion::backend::llvmc::compileModule(info, true, std::move(modExt), *ctx);
  //
    std::cout << c << std::endl;

//  lld::elf

  std::cout << "Done!" << std::endl;
  return EXIT_SUCCESS;
}


#include <iostream>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Object/ObjectFile.h>
#include "llvm/Support/TargetSelect.h"
#include "lld/Common/Driver.h"

static llvm::ExitOnError ExitOnErr;

int main() {


//  lld::safeLldMain(0, {}, llvm::dbgs(), llvm::dbgs());
//ld -shared -o hello2.so hello.o
  lld::elf::link(  {"--shared","-o" ,"/home/tom/Desktop/hello3.so",  "/home/tom/Desktop/hello.o", },false,  llvm::dbgs(), llvm::dbgs());

//
//  using namespace llvm;
//  orc::LLJITBuilder builder = orc::LLJITBuilder();
//  llvm::InitializeNativeTarget();
//  auto jit = ExitOnErr(builder.create());
//
//  auto x = ExitOnErr(llvm::errorOrToExpected(MemoryBuffer::getFile("/home/tom/polyregion/native/obj.so")));
//  jit->addObjectFile(std::move(x));
//
//  JITEvaluatedSymbol symbol = ExitOnErr(jit->lookup("lambda"));
//  std::cout << "S="
//            << " " << symbol.getAddress() << " " << std::hex << symbol.getAddress() << std::endl;
}
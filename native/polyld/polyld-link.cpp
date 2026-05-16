#include "lld/Common/Driver.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

LLD_HAS_DRIVER(coff)

extern "C" llvm::PassPluginLibraryInfo llvmGetPassPluginInfo();

static void registerPolyreflect(llvm::PassBuilder &PB) { llvmGetPassPluginInfo().RegisterPassBuilderCallbacks(PB); }

int main(int argc, const char *argv[]) {
  llvm::InitLLVM init(argc, argv);
  llvm::lto::setStaticPluginRegister(&registerPolyreflect);
  const lld::DriverDef drivers[] = {{lld::WinLink, &lld::coff::link}};
  auto r = lld::lldMain({argv, argv + argc}, llvm::outs(), llvm::errs(), drivers);
  return r.retCode;
}

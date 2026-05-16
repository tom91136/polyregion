// Fused polyfc: argv[0] does not always match flang's DriverSuffix table, so set the flang
// driver mode explicitly via the friend-template private-access idiom.

#include <algorithm>
#include <memory>
#include <string>

#include <stdlib.h>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Options/Options.h"
#include "flang/Config/config.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "polyregion/env.h"

// XXX Template instantiation with a pointer-to-private-member bypasses access checks so the
// inner friend can call the private setDriverMode.
namespace polyfc_priv {
template <void (clang::driver::Driver::*MemPtr)(llvm::StringRef)> struct DriverSetModeAccessor {
  friend void invokeSetDriverMode(clang::driver::Driver &d, llvm::StringRef v) { (d.*MemPtr)(v); }
};
template struct DriverSetModeAccessor<&clang::driver::Driver::setDriverMode>;
void invokeSetDriverMode(clang::driver::Driver &d, llvm::StringRef v);
} // namespace polyfc_priv

extern int fc1_main(llvm::ArrayRef<const char *> argv, const char *argv0);

namespace {

std::unique_ptr<clang::DiagnosticOptions> createAndPopulateDiagOpts(llvm::ArrayRef<const char *> argv) {
  auto diagOpts = std::make_unique<clang::DiagnosticOptions>();
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = clang::getDriverOptTable().ParseArgs(argv.slice(1), missingArgIndex, missingArgCount,
                                                                      llvm::opt::Visibility(clang::options::FlangOption));
  (void)Fortran::frontend::parseDiagnosticArgs(*diagOpts, args);
  return diagOpts;
}

int executeFC1Tool(llvm::SmallVectorImpl<const char *> &argV) {
  llvm::StringRef tool = argV[1];
  if (tool == "-fc1") return fc1_main(llvm::ArrayRef(argV).slice(2), argV[0]);
  llvm::errs() << "error: unknown integrated tool '" << tool << "'. Valid tools include '-fc1'.\n";
  return 1;
}

} // namespace

int flang_main(int argc, const char **argv) {
  // XXX Use throwaway argv copies; InitLLVM rewrites argv via GetCommandLineArgumentsW on Windows.
  int initArgc = argc;
  const char **initArgv = argv;
  llvm::InitLLVM x(initArgc, initArgv);
  llvm::SmallVector<const char *, 256> args(argv, argv + argc);

  clang::driver::ParsedClangName targetandMode = clang::driver::ToolChain::getTargetAndModeFromProgramName(argv[0]);
  std::string driverPath = llvm::sys::fs::getMainExecutable(args[0], (void *)(intptr_t)&flang_main);

  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver saver(alloc);
  llvm::cl::ExpansionContext expCtx(saver.getAllocator(), llvm::cl::TokenizeGNUCommandLine);
  if (llvm::Error err = expCtx.expandResponseFiles(args)) llvm::errs() << toString(std::move(err)) << '\n';

  auto firstArg = std::find_if(args.begin() + 1, args.end(), [](const char *a) { return a != nullptr; });
  if (firstArg != args.end()) {
    if (llvm::StringRef(args[1]).starts_with("-cc1")) {
      llvm::errs() << "error: unknown integrated tool '" << args[1] << "'. Valid tools include '-fc1'.\n";
      return 1;
    }
    if (llvm::StringRef(args[1]).starts_with("-fc1")) return executeFC1Tool(args);
  }

  llvm::StringSet<> savedStrings;
  if (const char *overrideStr = ::getenv("FCC_OVERRIDE_OPTIONS"))
    clang::driver::applyOverrideOptions(args, overrideStr, savedStrings, "FCC_OVERRIDE_OPTIONS", &llvm::errs());

  std::unique_ptr<clang::DiagnosticOptions> diagOpts = createAndPopulateDiagOpts(args);
  auto *diagClient = new Fortran::frontend::TextDiagnosticPrinter(llvm::errs(), *diagOpts);
  diagClient->setPrefix(std::string(llvm::sys::path::stem(driverPath)));
  clang::DiagnosticsEngine diags(clang::DiagnosticIDs::create(), *diagOpts, diagClient);

  clang::driver::Driver theDriver(driverPath, llvm::sys::getDefaultTargetTriple(), diags, "flang LLVM compiler");
  theDriver.setTargetAndMode(targetandMode);
  polyfc_priv::invokeSetDriverMode(theDriver, "flang");
  theDriver.setPreferredLinker(FLANG_DEFAULT_LINKER);
#ifdef FLANG_RUNTIME_F128_MATH_LIB
  theDriver.setFlangF128MathLibrary(FLANG_RUNTIME_F128_MATH_LIB);
#endif
  std::unique_ptr<clang::driver::Compilation> c(theDriver.BuildCompilation(args));
  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4> failingCommands;

  std::string compilerOptsGathered;
  llvm::raw_string_ostream os(compilerOptsGathered);
  for (int i = 0; i < argc; ++i) {
    os << argv[i];
    if (i < argc - 1) os << ' ';
  }
  polyregion::env::put("FLANG_COMPILER_OPTIONS_STRING", compilerOptsGathered.c_str(), true);

  int res = theDriver.ExecuteCompilation(*c, failingCommands);
  for (const auto &p : failingCommands) {
    int commandRes = p.first;
    const clang::driver::Command *failingCommand = p.second;
    if (!res) res = commandRes;
    bool isCrash = commandRes < 0;
#ifdef _WIN32
    isCrash |= commandRes == 3;
#endif
    if (isCrash) {
      theDriver.generateCompilationDiagnostics(*c, *failingCommand);
      break;
    }
  }

  diags.getClient()->finish();
  return res;
}

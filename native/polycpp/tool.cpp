#include <cstdlib>
#include <iostream>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

#include "aspartame/vector.hpp"
#include "aspartame/string.hpp"
#include "aspartame/view.hpp"


#include "aspartame/fluent.hpp"

#include "frontend.h"
#include "rewriter.h"
#include "utils.hpp"
using namespace aspartame;

static std::variant<std::string, polyregion::polyast::Target> parse(std::string csv){



  csv ^ split(',') ^ map([](auto &x) -> std::variant<std::string, polyregion::polyast::Target>{

    auto parseTarget = [](const std::string& target){
      return
    };

    auto targetArchTuple = x ^ split(':');

    switch (targetArchTuple.size()){
      case 0:

      case 1:

      default:

        break;
    }


    return 0;

  });


}

static std::vector<std::string> mkDelimitedEnvPaths(const char *env, std::optional<std::string> leading) {
  std::vector<std::string> xs;
  if (auto line = std::getenv(env); line) {
    for (auto &path :  line ^ split(llvm::sys::EnvPathSeparator)) {
      if (leading) xs.push_back(*leading);
      xs.push_back(path);
    }
  }
  return xs;
};

int executeCC1(std::vector<std::string> &cc1Args, bool stdpar) {

  if (stdpar) {
    auto includes = mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem");
    cc1Args.insert(cc1Args.end(), includes.begin(), includes.end());

    cc1Args.insert(cc1Args.end(), {"-include", "polystl/polystl.hpp"});

    // -static-polyrt
    // -dynamic-polyrt
    // -no-polyrt
  }

  std::vector<const char *> cc1Args_ = cc1Args ^ map([]( auto &x){return x.c_str();});

//  std::vector<const char *> cc1Args_(cc1Args.size());
//  std::transform(cc1Args.begin(), cc1Args.end(), cc1Args_.begin(), [](auto &x) { return x.c_str(); });
  auto diagOptions = clang::CreateAndPopulateDiagOpts(cc1Args_);
  auto diagClient = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(), &*diagOptions);
  auto diag = clang::CompilerInstance::createDiagnostics(diagOptions.release(), diagClient.release(), true);

  clang::CompilerInstance CI;
  CI.setDiagnostics(diag.get());
  if (!clang::CompilerInvocation::CreateFromArgs(CI.getInvocation(), cc1Args_, *diag)) {
    return EXIT_FAILURE;
  }

  auto ct = clang::TargetInfo::CreateTargetInfo(CI.getDiagnostics(), CI.getInvocation().TargetOpts);
  CI.setTarget(ct);
  bool success;
  if (stdpar && !std::getenv("POLYCPP_NO_REWRITE")) {
    using namespace polyregion;
    polystl::ModifyASTAndEmitObjAction action([]() { return std::make_unique<polystl::OffloadRewriteConsumer>(); });
    success = CI.ExecuteAction(action);
    CI.getSourceManager().PrintStats();
  } else {
    clang::EmitObjAction action;
    success = CI.ExecuteAction(action);
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, const char *argv[]) {

  llvm::InitLLVM init(argc, argv);
  llvm::setBugReportMsg("PLEASE submit a bug report to TODO and include the crash backtrace, "
                        "preprocessed source, and associated run script.\n");

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  auto triple = llvm::sys::getDefaultTargetTriple();
  auto diagOptions = new clang::DiagnosticOptions();
  auto diagClient = new clang::TextDiagnosticPrinter(llvm::errs(), &*diagOptions);
  auto diagID = new clang::DiagnosticIDs();
  auto diags = clang::DiagnosticsEngine(diagID, diagOptions, diagClient);

  auto execPath = llvm::sys::fs::getMainExecutable(argv[0], nullptr);
  auto execParentDir = llvm::sys::path::parent_path(execPath).str();

  clang::driver::Driver driver(execPath, triple, diags, "PolyCpp compiler");
  driver.ResourceDir = (execParentDir + "/lib/clang/" + std::to_string(CLANG_VERSION_MAJOR));

  std::vector<const char *> args(argv, argv + argc);



  auto stdparIdx = args ^ index_of("-fstdpar");
  args | zip_with_index<decltype(stdparIdx)>() | filter([&](auto &, auto i){ return i == stdparIdx; });



  // sort out -fstdpar
  auto fstdparIt = std::remove_if(args.begin(), args.end(), [](auto x) { return x == std::string("-fstdpar"); });
  auto stdpar = fstdparIt != args.end();
  if (stdpar) args.erase(fstdparIt, args.end());

  // since the executable name won't be clang++ anymore, we manually set the mode to C++ by inserting the override after the executable name
  args.insert(std::next(args.begin()), "--driver-mode=g++");

  auto libs = mkDelimitedEnvPaths("POLYSTL_LIB", {});
  std::transform(libs.begin(), libs.end(), std::back_inserter(args), [](auto &x) { return x.c_str(); });

  std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(llvm::ArrayRef(args)));

  int returnCode = EXIT_SUCCESS;
  for (const auto &command : compilation->getJobs()) {
    const auto &cmdArgs = command.getArguments();
    if (command.getExecutable() == execPath &&                    // make sure the driver is actually calling us
        command.getCreator().getName() == std::string("clang") && // and that clang is the compiler
        cmdArgs[0] == std::string("-cc1")                         // and we're invoking the cc1 frontend
    ) {
      std::vector<std::string> actual;
      actual.insert(actual.begin(), std::next(cmdArgs.begin()), cmdArgs.end()); //  skip the first -cc1
      returnCode = executeCC1(actual, stdpar);
      if (returnCode != EXIT_SUCCESS) break;
    } else {
      const clang::driver::Command *failed{};
      if (auto code = compilation->ExecuteCommand(command, failed); code != EXIT_SUCCESS) {
        driver.generateCompilationDiagnostics(*compilation, *failed);
        returnCode = code;
      }
    }
  }
  diags.getClient()->finish();
  return returnCode;
}
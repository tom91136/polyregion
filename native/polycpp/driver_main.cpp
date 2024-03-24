#include <clang/CodeGen/BackendUtil.h>
#include <cstdlib>
#include <iostream>

#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/FrontendTool/Utils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

#include "aspartame/optional.hpp"
#include "aspartame/string.hpp"
#include "aspartame/variant.hpp"
#include "aspartame/vector.hpp"
#include "aspartame/view.hpp"

#include "driver_clang.h"
#include "driver_polyc.h"
#include "frontend.h"
#include "options.h"
#include "rewriter.h"

using namespace aspartame;

static std::vector<std::string> mkDelimitedEnvPaths(const char *env, std::optional<std::string> leading) {
  std::vector<std::string> xs;
  if (auto line = std::getenv(env); line) {
    for (auto &path : line ^ split(llvm::sys::EnvPathSeparator)) {
      if (leading) xs.push_back(*leading);
      xs.push_back(path);
    }
  }
  return xs;
};

int executeCC1(const std::string &executable,           //
               const std::string &polycppLibDir,        //
               const std::string &polycppIncludeDir,    //
               const llvm::opt::ArgStringList &cc1Args, //
               const std::optional<polyregion::polystl::StdParOptions> &stdpar) {
  auto noCC1ArgsBacking = cc1Args | drop(1)                            // skip the first cc1 arg
                          | map([](auto s) { return std::string(s); }) // own the strings as we need this to survive until CC1 finishes
                          | to_vector();                               //

  if (stdpar) {

    auto includes = mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem");
    noCC1ArgsBacking.insert(noCC1ArgsBacking.end(), includes.begin(), includes.end());

    noCC1ArgsBacking.insert(noCC1ArgsBacking.end(), {"-isystem", polycppIncludeDir});

    noCC1ArgsBacking.insert(noCC1ArgsBacking.end(), {"-include", "polystl/polystl.h"});
    noCC1ArgsBacking.insert(noCC1ArgsBacking.end(), {"-fpass-plugin=" + polycppLibDir + "/polystl-interpose.so"});
  }

  auto cc1Verbose = noCC1ArgsBacking ^ contains("-v");
  if (cc1Verbose || true) {
    // XXX dump the command; we handle this directly because CC1 is not longer a separate process
    (llvm::errs() << "\"<PolySTL c1>\" " << (noCC1ArgsBacking ^ mk_string(" ")) << "\n").flush();
  }

  return polyregion::polystl::useCI(
      noCC1ArgsBacking ^ map([](auto &x) { return x.c_str(); }), "polycpp", nullptr, [&](clang::CompilerInstance &CI) {
        if (CI.getFrontendOpts().ShowVersion) {
          llvm::cl::PrintVersionMessage();
          return EXIT_SUCCESS;
        }

        int code = 0;
        if (stdpar && !std::getenv("POLYCPP_NO_REWRITE")) {
          using namespace polyregion;

          if (!stdpar->quiet) {
            CI.getDiagnostics()
                .Report(CI.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Remark, "[PolySTL] Targeting architectures: %0"))
                .AddString(stdpar->targets ^
                           mk_string(", ", [](auto target, auto feature) { return std::string(to_string(target)) + "@" + feature; }));
          }

          auto action = std::make_unique<polystl::ModifyASTAndEmitObjAction>(clang::CreateFrontendAction(CI), [&]() {
            return std::make_unique<polystl::OffloadRewriteConsumer>(CI.getDiagnostics(),
                                                                     polystl::DriverContext{executable, *stdpar, cc1Verbose});
          });

          auto success = polyregion::polystl::executeFrontendAction(&CI, std::move(action));
          if (!success) {
            CI.getDiagnostics().Report(
                CI.getDiagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolySTL] Frontend pass did not succeed"));
          }
          code = success ? EXIT_SUCCESS : EXIT_FAILURE;
        } else {
          code = clang::ExecuteCompilerInvocation(&CI) ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        return code;
      });
}

[[maybe_unused]] void addrFn() { /* dummy symbol used for use with getMainExecutable */
}

int main(int argc, const char *argv[]) {

  if (argc >= 2 && argv[1] == std::string("--polyc")) {
    return polyregion::polyc(argc - 1, argv + 1);
  }

  auto diags = polyregion::polystl::initialiseAndCreateDiag(argc, argv,
                                                            "PLEASE submit a bug report to TODO and include the crash backtrace, "
                                                            "preprocessed source, and associated run script.\n");

  if (!diags) {
    return EXIT_FAILURE;
  }

  auto execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)(&addrFn));
  auto execParentDir = llvm::sys::path::parent_path(execPath).str();

  auto triple = llvm::sys::getDefaultTargetTriple();
  clang::driver::Driver driver(execPath, triple, *diags, "PolyCpp compiler");

  driver.ResourceDir = execParentDir + "/lib/clang/" + std::to_string(CLANG_VERSION_MAJOR);
  if (!llvm::sys::fs::is_directory(driver.ResourceDir)) {
    driver.getDiags().Report(diags->getCustomDiagID(
        clang::DiagnosticsEngine::Warning,
        "Clang resource directory (%0) missing, intrinsics and certain features that require bundled libraries (e.g ASan) will not work."))
        << driver.ResourceDir;
  }

  auto polycppResourceDir = execParentDir + "/lib/polycpp";
  auto polycppLibDir = polycppResourceDir + "/lib";
  auto polycppIncludeDir = polycppResourceDir + "/include";
  for (auto &[dir, kind] :
       {std::pair{polycppResourceDir, "resource"}, std::pair{polycppLibDir, "library"}, std::pair{polycppIncludeDir, "header"}}) {
    if (!llvm::sys::fs::is_directory(dir)) {
      driver.getDiags().Report(
          diags->getCustomDiagID(clang::DiagnosticsEngine::Warning, "Polycpp %0 directory (%1) missing, -fstdpar will not work."))
          << kind << dir;
    }
  }

  return polyregion::polystl::StdParOptions::stripAndParse({argv, argv + argc}) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               driver.getDiags().Report(
                   diags->getCustomDiagID(clang::DiagnosticsEngine::Fatal, "Errors (%0) while parsing the following arguments:\n%1"))
                   << errors.size() << (errors ^ mk_string("\n") ^ indent(2));
               return EXIT_FAILURE;
             },
             [&](const std::pair<std::vector<const char *>, std::optional<polyregion::polystl::StdParOptions>> &v) {
               auto [driverArgs, stdpar] = v;

               // since the executable name won't be clang++ anymore, we manually set the mode to C++ by inserting the override
               // after the executable name
               driverArgs.insert(std::next(driverArgs.begin()), "--driver-mode=g++");

               auto libs = mkDelimitedEnvPaths("POLYSTL_LIB", {});
               std::transform(libs.begin(), libs.end(), std::back_inserter(driverArgs), [](auto &x) { return x.c_str(); });

               std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(llvm::ArrayRef(driverArgs)));

               int returnCode = EXIT_SUCCESS;

               auto runCommand = [&](clang::driver::Command &command) {
                 const clang::driver::Command *failed{};
                 if (auto code = compilation->ExecuteCommand(command, failed); code != EXIT_SUCCESS) {
                   driver.generateCompilationDiagnostics(*compilation, *failed);
                   returnCode = code;
                 }
               };

               for (auto &command : compilation->getJobs()) {
                 const auto &cmdArgs = command.getArguments();
                 if (command.getExecutable() == execPath &&                    // make sure the driver is actually calling us
                     command.getCreator().getName() == std::string("clang") && // and that clang is the compiler
                     cmdArgs[0] == std::string("-cc1")                         // and we're invoking the cc1 frontend
                 ) {
                   returnCode = executeCC1(execPath, polycppLibDir, polycppIncludeDir, cmdArgs, stdpar);
                   if (returnCode != EXIT_SUCCESS) break;
                 } else if (stdpar && command.getCreator().isLinkJob()) {
                   auto argsWithExtraLib = cmdArgs;
                   switch (stdpar->rt) {
                     case polyregion::polystl::StdParOptions::LinkKind::Static: {
                       auto polyStlArchivePath = polycppLibDir + "/libpolystl-static.a";
                       // XXX file order matters and should be in reverse dependency order (object with the most dependencies on the left)
                       //   We want to insert our runtime at a position that's just after all the inputs but before the system dependencies.
                       //   So, for a set of input, we find the right most index and insert our library at that position.
                       auto rightMostIdx = //
                           (command.getInputInfos() | collect([&](auto input) {
                              std::string rhs = input.getFilename();
                              return argsWithExtraLib | index_where_maybe([&](auto lhs) { return lhs == rhs; });
                            }) |
                            to_vector()) ^
                           sort() ^ last_maybe() ^ get_or_else(argsWithExtraLib.size() - 1);
                       argsWithExtraLib.insert(argsWithExtraLib.begin() + rightMostIdx + 1, polyStlArchivePath.c_str());
                       if (!stdpar->noCompress) argsWithExtraLib.append({"--compress-debug-sections=zlib", "--gc-sections"});
                       command.replaceArguments(argsWithExtraLib);
                       runCommand(command);
                       break;
                     }
                     case polyregion::polystl::StdParOptions::LinkKind::Dynamic: {
                       auto linkFlag = "-L" + polycppLibDir;
                       argsWithExtraLib.append({linkFlag.c_str(), "-lpolystl", "-rpath", polycppLibDir.c_str(), "-rpath", "$ORIGIN"});
                       command.replaceArguments(argsWithExtraLib);
                       if (!stdpar->quiet) {
                         driver.getDiags().Report(diags->getCustomDiagID(
                             clang::DiagnosticsEngine::Remark,
                             "Dynamic linking of PolySTL runtime requested, if you would like to relocate your binary, "
                             "please copy %0 to the same directory as the executable (-rpath=$ORIGIN has been set for you)"))
                             << (polycppLibDir + "/libpolystl.so");
                       }
                       runCommand(command);
                       break;
                     }
                     case polyregion::polystl::StdParOptions::LinkKind::Disabled: runCommand(command); break;
                   }
                 } else runCommand(command);
               }
               diags->getClient()->finish();
               return returnCode;
             });
}
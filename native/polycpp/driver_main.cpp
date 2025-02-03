#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "aspartame/all.hpp"
#include "driver_polyc.h"
#include "fmt/core.h"
#include "polyfront/options_frontend.hpp"
#include "llvm/Support/Program.h"

using namespace aspartame;
using namespace polyregion::polyfront;

[[maybe_unused]] void addrFn() { /* dummy symbol used for use with getMainExecutable */ }

int main(int argc, const char *argv[]) {
  CliArgs args(std::vector(argv, argv + argc));
  if (args.has("--polyc", 1)) {
    return polyregion::polyc(argc - 1, argv + 1);
  }

  namespace fs = std::filesystem;

  fs::path execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrFn);
  fs::path execParentPath = execPath.parent_path();

  fs::path clangPath;

  if (auto driverArg = args.popValue("--driver")) clangPath = *driverArg;         // Explicit driver takes precedence
  else if (auto driverEnv = std::getenv("POLYCPP_DRIVER")) clangPath = driverEnv; // Then try environment vars
  else if (fs::path clangBin = execParentPath / "clang++";
           fs::exists(clangBin)) { // Finally, find the clang++ that's in the same dir as the current wrapper
    clangPath = clangBin;
  } else {
    std::cerr << fmt::format(
                     "[PolyCpp] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_clang++>`",
                     execPath.string())
              << std::endl;
    return EXIT_FAILURE;
  }

  return StdParOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               std::cerr << fmt::format("[PolyCpp] Unable to parse PolyCpp specific arguments:\n{}", (errors ^ mk_string("\n") ^ indent(2)))
                         << std::endl;
               return EXIT_FAILURE;
             },
             [&](const std::optional<StdParOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::initializer_list<std::string> &xs) { remaining.insert(remaining.end(), xs); };

               if (opts) {
                 auto includes = mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem", llvm::sys::EnvPathSeparator);
                 auto libs = mkDelimitedEnvPaths("POLYSTL_LIB", {}, llvm::sys::EnvPathSeparator);
                 remaining.insert(remaining.end(), includes.begin(), includes.end());
                 remaining.insert(remaining.end(), libs.begin(), libs.end());

                 const auto polycppResourcePath = execParentPath / "lib/polycpp";
                 const auto polycppIncludePath = polycppResourcePath / "include";
                 const auto polycppLibPath = polycppResourcePath / "lib";
                 const auto polycppReflectPlugin = polycppLibPath / fmt::format("polystl-reflect-plugin.{}", dynamicLibSuffix());
                 const auto polycppClangPlugin = polycppLibPath / fmt::format("polycpp-clang-plugin.{}", dynamicLibSuffix());
                 append({"-isystem", polycppIncludePath.string()});
                 append({"-include", "polystl/polystl.h"});
                 append({"-include", "rt-reflect/rt.hpp"});

                 const bool noRewrite = std::getenv("POLYCPP_NO_REWRITE") != nullptr;
                 if (!noRewrite) {
                   append({"-Xclang", "-load", "-Xclang", polycppClangPlugin.string()});
                   append({"-Xclang", "-add-plugin", "-Xclang", "polycpp"});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontExe, execPath.string())});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontVerbose, opts->verbose ? "1" : "0")});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontTargets, opts->targets)});
                   if (opts->interposeMalloc) append({fmt::format("-fpass-plugin={}", polycppReflectPlugin.string())});
                 }

                 const auto compileOnly = std::vector{"-c", "-S", "-E", "-M", "-MM", "-MD", "-fsyntax-only"} ^
                                          exists([&](auto &flag) { return args.has(flag); });
                 if (!compileOnly) {
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(),
                                        (polycppLibPath / fmt::format("libpolystl-static.{}", staticLibSuffix())).string());
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case StdParOptions::LinkKind::Dynamic: {
                       append({fmt::format("-L{}", polycppLibPath.string()), "-lpolystl", //
                               fmt::format("-Wl,-rpath,{}", polycppLibPath.string()), "-Wl,-rpath,$ORIGIN"});
                       if (opts->verbose) {
                         std::cerr
                             << fmt::format(
                                    "[PolyCpp] Dynamic linking of PolySTL runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)",
                                    (polycppLibPath / fmt::format("libpolystl.{}", dynamicLibSuffix())).string())
                             << std::endl;
                       }
                       break;
                     }
                     case StdParOptions::LinkKind::Disabled: break;
                   }
                 }
               }

               remaining[0] = "clang++";
               std::cout << ">>> " << (remaining ^ mk_string(" ")) << std::endl;

               return llvm::sys::ExecuteAndWait(clangPath.string(),
                                                remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
             });
}
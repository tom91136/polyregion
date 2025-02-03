#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "aspartame/all.hpp"
#include "driver_polyc.h"
#include "fmt/core.h"
#include "polyfront/options_frontend.hpp"
#include "polyregion/env.h"
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

  const fs::path execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrFn);
  const fs::path execParentPath = execPath.parent_path();

  fs::path flangPath;
  if (const auto driverArg = args.popValue("--driver")) flangPath = *driverArg;        // Explicit driver takes precedence
  else if (const auto driverEnv = std::getenv("POLYFC_DRIVER")) flangPath = driverEnv; // Then try environment vars
  else if (fs::path flangBin = execParentPath / "flang-new";
           fs::exists(flangBin)) { // Finally, find the clang++ that's in the same dir as the current wrapper
    flangPath = flangBin;
  } else {
    std::cerr << fmt::format(
                     "[PolyFC] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_clang++>`",
                     execPath.string())
              << std::endl;
    return EXIT_FAILURE;
  }

  return StdParOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               std::cerr << fmt::format("[PolyFC] Unable to parse PolyFC specific arguments:\n{}", (errors ^ mk_string("\n") ^ indent(2)))
                         << std::endl;
               return EXIT_FAILURE;
             },
             [&](const std::optional<StdParOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::initializer_list<std::string> &xs) { remaining.insert(remaining.end(), xs); };

               std::vector<std::pair<const char *, std::string>> envs;

               if (opts) {
                 auto includes = mkDelimitedEnvPaths("POLYDCO_INCLUDE", "-isystem", llvm::sys::EnvPathSeparator);
                 auto libs = mkDelimitedEnvPaths("POLYDCO_LIB", {}, llvm::sys::EnvPathSeparator);
                 remaining.insert(remaining.end(), includes.begin(), includes.end());
                 remaining.insert(remaining.end(), libs.begin(), libs.end());

                 const auto polyfcResourcePath = execParentPath / "lib/polyfc";
                 const auto polyfcLibPath = polyfcResourcePath / "lib";
                 const auto polyfcClangPlugin = polyfcLibPath / fmt::format("polyfc-flang-plugin.{}", dynamicLibSuffix());

                 if (const bool noRewrite = std::getenv("POLYFC_NO_REWRITE") != nullptr; !noRewrite) {
                   append({"-Xflang", "-load", "-Xflang", polyfcClangPlugin.string()});
                   append({"-Xflang", "-plugin", "-Xflang", "polyfc"});
                   envs.emplace_back(PolyfrontExe, execPath.string());
                   envs.emplace_back(PolyfrontVerbose, opts->verbose ? "1" : "0");
                   envs.emplace_back(PolyfrontTargets, opts->targets);
                 }

                 const auto compileOnly =
                     std::vector{"-c", "-S", "-E", "-M", "-fsyntax-only"} ^ exists([&](auto &flag) { return args.has(flag); });
                 if (!compileOnly) {
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), (polyfcLibPath / fmt::format("libpolydco-static.{}", staticLibSuffix())).string());
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case StdParOptions::LinkKind::Dynamic: {
                       append({fmt::format("-L{}", polyfcLibPath.string()), "-lpolydco", //
                               fmt::format("-Wl,-rpath,{}", polyfcLibPath.string()), "-Wl,-rpath,$ORIGIN"});
                       if (opts->verbose) {
                         std::cerr
                             << fmt::format(
                                    "[PolyFC] Dynamic linking of PolyDCO runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)",
                                    (polyfcLibPath / fmt::format("libpolydco.{}", dynamicLibSuffix())).string())
                             << std::endl;
                       }
                       break;
                     }
                     case StdParOptions::LinkKind::Disabled: break;
                   }
                 }
               }

               remaining[0] = "flang-new";
               for (auto [k, v] : envs)
                 polyregion::env::put(k, v.c_str(), true);
               return llvm::sys::ExecuteAndWait(flangPath.string(),
                                                remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
             });
}
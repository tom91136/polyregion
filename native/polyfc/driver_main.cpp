#include <cstdlib>
#include <iostream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#ifdef POLYREGION_FUSED_DRIVER
int flang_main(int argc, const char **argv);
#endif

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyfront/options_frontend.hpp"
#include "polyregion/env.h"

#include "driver_polyc.h"

using namespace aspartame;
using namespace polyregion::polyfront;

[[maybe_unused]] void addrFn() { /* dummy symbol used for use with getMainExecutable */ }

static std::string joinPath(llvm::StringRef a, llvm::StringRef b, llvm::StringRef c = {}, llvm::StringRef d = {}) {
  llvm::SmallString<256> p(a);
  if (!b.empty()) llvm::sys::path::append(p, b);
  if (!c.empty()) llvm::sys::path::append(p, c);
  if (!d.empty()) llvm::sys::path::append(p, d);
  return p.str().str();
}

static std::string executableName(llvm::StringRef name) {
#if defined(_WIN32)
  return name.str() + ".exe";
#else
  return name.str();
#endif
}

static std::string staticLibraryName(llvm::StringRef name) {
#if defined(_WIN32)
  return name.str() + ".lib";
#else
  return "lib" + name.str() + ".a";
#endif
}

int main(int argc, const char *argv[]) {
  CliArgs args(std::vector(argv, argv + argc));
  if (args.has("--polyc", 1)) {
    return polyregion::polyc(argc - 1, argv + 1);
  }

  const std::string execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrFn);
  const std::string execParentPath = llvm::sys::path::parent_path(execPath).str();

  std::string flangPath;
#ifdef POLYREGION_FUSED_DRIVER
  // XXX fused build has no external driver; --driver/POLYFC_DRIVER are accepted but ignored.
  (void)args.popValue("--driver");
#else
  if (const auto driverArg = args.popValue("--driver")) flangPath = *driverArg;
  else if (const auto driverEnv = std::getenv("POLYFC_DRIVER")) flangPath = driverEnv;
  else if (auto flangBin = joinPath(execParentPath, executableName("flang-new")); llvm::sys::fs::exists(flangBin)) {
    flangPath = flangBin;
  } else {
    std::cerr << fmt::format(
                     "[PolyFC] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_clang++>`",
                     execPath)
              << std::endl;
    return EXIT_FAILURE;
  }
#endif

  return StdParOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               std::cerr << fmt::format("[PolyFC] Unable to parse PolyFC specific arguments:\n{}", (errors ^ mk_string("\n") ^ indent(2)))
                         << std::endl;
               return EXIT_FAILURE;
             },
             [&](const std::optional<StdParOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::vector<std::string> &xs) { remaining.insert(remaining.end(), xs.begin(), xs.end()); };

               std::vector<std::pair<const char *, std::string>> envs;

               if (opts) {
                 auto includes = mkDelimitedEnvPaths("POLYDCO_INCLUDE", "-isystem", llvm::sys::EnvPathSeparator);
                 auto libs = mkDelimitedEnvPaths("POLYDCO_LIB", {}, llvm::sys::EnvPathSeparator);
                 remaining.insert(remaining.end(), includes.begin(), includes.end());
                 remaining.insert(remaining.end(), libs.begin(), libs.end());

                 std::string polyfcResourcePath = joinPath(execParentPath, "lib", "polyfc");
                 if (!llvm::sys::fs::exists(polyfcResourcePath)) polyfcResourcePath = joinPath(execParentPath, "..", "lib", "polyfc");
                 const auto polyfcLibPath = joinPath(polyfcResourcePath, "lib");
                 const auto polyreflectPlugin = joinPath(polyfcLibPath, fmt::format("polyreflect-plugin.{}", dynamicLibSuffix()));
                 const auto polyfcFlangPlugin = joinPath(polyfcLibPath, fmt::format("polyfc-flang-plugin.{}", dynamicLibSuffix()));

                 const auto debug = opts->verbose == StdParOptions::VerboseLevel::Debug;

                 if (const bool noRewrite = std::getenv("POLYFC_NO_REWRITE") != nullptr; !noRewrite) {
#ifndef POLYREGION_FUSED_DRIVER
                   append({"-Xflang", "-load", "-Xflang", polyfcFlangPlugin});
#endif
                   append({"-Xflang", "-plugin", "-Xflang", "polyfc"});
                   envs.emplace_back(PolyfrontExe, execPath);
                   envs.emplace_back(PolyfrontVerbose, debug ? "1" : "0");
                   envs.emplace_back(PolyfrontTargets, opts->targets);
                 }

                 const auto compileOnly =
                     std::vector{"-c", "-S", "-E", "-M", "-fsyntax-only"} ^ exists([&](auto &flag) { return args.has(flag); });
                 if (!compileOnly) {
                   switch (opts->mem) {
                     case StdParOptions::MemKind::Direct: break;
                     case StdParOptions::MemKind::Interpose:
                       append(Driver::clangPassPluginFlags(polyreflectPlugin, {fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                                                               "polyreflect-late=interpose"}));
                       break;
                     case StdParOptions::MemKind::Reflect:
                       append(Driver::enableLLDAndLTO(args));
                       append(Driver::lldPassPluginFlags(polyreflectPlugin, {fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                                                             "-polyreflect-late=ReflectMem"}));
                       break;
                   }
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), joinPath(polyfcLibPath, staticLibraryName("polydco-static")));
#if defined(_WIN32)
                       // flang Fortran sources cannot use #pragma comment(lib); inject the
                       // Windows system imports that polydco-static depends on directly.
                       append({"-Xlinker", "Version.lib", "-Xlinker", "psapi.lib", "-Xlinker", "ntdll.lib", "-Xlinker", "ws2_32.lib",
                               "-Xlinker", "ole32.lib", "-Xlinker", "shell32.lib", "-Xlinker", "advapi32.lib", "-Xlinker", "uuid.lib",
                               "-Xlinker", "delayimp.lib"});
                       // flang_rt uses __udivti3 (128-bit div); pull clang_rt.builtins from
                       // the resource dir since MSVC libucrt lacks it.
                       for (const auto &resRoot :
                            {joinPath(execParentPath, "..", "lib", "clang/22"), std::string(POLYFC_FUSED_DIST_DIR "/lib/clang/22")}) {
                         const auto builtins = joinPath(resRoot, "lib", "windows", "clang_rt.builtins-x86_64.lib");
                         if (llvm::sys::fs::exists(builtins)) {
                           append({"-Xlinker", builtins});
                           break;
                         }
                       }
#endif
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case StdParOptions::LinkKind::Dynamic: {
                       // XXX libpolydco's DT_NEEDED must precede user-supplied -lstdc++ or
                       // libhsa-runtime64 segfaults inside std::codecvt during dispatch.
                       auto flags = Driver::dynamicOriginLinkFlags(polyfcLibPath, "polydco");
                       remaining.insert(remaining.begin() + 1, flags.begin(), flags.end());
#if defined(__APPLE__)
                       // XXX clang aliases -lstdc++ to libc++ on macOS and stamps @rpath/libc++.1.dylib;
                       // bundle's libc++ sits in the dist's main lib/, not under polyfc/lib/.
                       append({fmt::format("-Wl,-rpath,{}", joinPath(execParentPath, "..", "lib"))});
#endif
                       if (opts->verbose == StdParOptions::VerboseLevel::Info) {
                         std::cerr
                             << fmt::format(
                                    "[PolyFC] Dynamic linking of PolyDCO runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)",
                                    joinPath(polyfcLibPath, fmt::format("libpolydco.{}", dynamicLibSuffix())))
                             << std::endl;
                       }
                       break;
                     }
                     case StdParOptions::LinkKind::Disabled: break;
                   }
                 }
               }

               for (auto [k, v] : envs)
                 polyregion::env::put(k, v.c_str(), true);
#ifdef POLYREGION_FUSED_DRIVER
               // Probe install-relative paths first so the binary works both installed and
               // from the build tree. XXX flang_rt sentinel hard-codes the MSVC triple;
               // non-Windows fused builds skip this path until parameterised.
               auto firstWith = [](llvm::StringRef sentinel, std::initializer_list<std::string> roots) -> std::string {
                 for (const auto &root : roots)
                   if (llvm::sys::fs::exists(joinPath(root, sentinel))) return root;
                 return {};
               };
               if (auto modDir = firstWith("iso_fortran_env.mod", {joinPath(execParentPath, "..", "include", "flang"),
                                                                   std::string(POLYFC_FUSED_LLVM_BUILD_DIR "/include/flang")});
                   !modDir.empty()) {
                 remaining.push_back("-fintrinsic-modules-path=" + modDir);
               }
               if (auto resDir =
                       firstWith("lib/x86_64-pc-windows-msvc/flang_rt.runtime.static.lib",
                                 {joinPath(execParentPath, "..", "lib", "clang/22"), std::string(POLYFC_FUSED_DIST_DIR "/lib/clang/22")});
                   !resDir.empty()) {
                 remaining.push_back("-resource-dir");
                 remaining.push_back(resDir);
               }
               remaining[0] = execPath;
               std::vector<const char *> rawArgs;
               rawArgs.reserve(remaining.size());
               for (auto &arg : remaining)
                 rawArgs.push_back(arg.c_str());
               return flang_main(static_cast<int>(rawArgs.size()), rawArgs.data());
#else
               return llvm::sys::ExecuteAndWait(flangPath, remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
#endif
             });
}

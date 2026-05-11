#include <cstdlib>
#include <iostream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyfront/options_frontend.hpp"

#include "driver_polyc.h"

using namespace aspartame;
using namespace polyregion::polyfront;

[[maybe_unused]] void addrFn() { /* dummy symbol used for use with getMainExecutable */ }

static std::string joinPath(llvm::StringRef a, llvm::StringRef b, llvm::StringRef c = {}, llvm::StringRef d = {}) {
  llvm::SmallString<256> p(a);
  // Skip empty components: llvm::sys::path::append unconditionally inserts a separator before
  // each non-null Twine, which would yield a trailing `/` for unused trailing args.
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

  std::string execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrFn);
  std::string execParentPath = llvm::sys::path::parent_path(execPath).str();

  std::string clangPath;

  if (auto driverArg = args.popValue("--driver")) clangPath = *driverArg;         // Explicit driver takes precedence
  else if (auto driverEnv = std::getenv("POLYCPP_DRIVER")) clangPath = driverEnv; // Then try environment vars
  else if (auto clangBin = joinPath(execParentPath, executableName("clang++"));
           llvm::sys::fs::exists(clangBin)) { // Finally, find the clang++ that's in the same dir as the current wrapper
    clangPath = clangBin;
  } else {
    std::cerr << fmt::format(
                     "[PolyCpp] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_clang++>`",
                     execPath)
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
               auto append = [&](const std::vector<std::string> &xs) { remaining ^= concat_inplace(xs); };

               if (opts) {
                 remaining ^= concat_inplace(mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem", llvm::sys::EnvPathSeparator));
                 remaining ^= concat_inplace(mkDelimitedEnvPaths("POLYSTL_LIB", {}, llvm::sys::EnvPathSeparator));

                 std::string polycppResourcePath = joinPath(execParentPath, "lib", "polycpp");
                 if (!llvm::sys::fs::exists(polycppResourcePath)) polycppResourcePath = joinPath(execParentPath, "..", "lib", "polycpp");
                 const auto polycppIncludePath = joinPath(polycppResourcePath, "include");
                 const auto polycppLibPath = joinPath(polycppResourcePath, "lib");
                 const auto polyreflectPlugin = joinPath(polycppLibPath, fmt::format("polyreflect-plugin.{}", dynamicLibSuffix()));
                 const auto polycppClangPlugin = joinPath(polycppLibPath, fmt::format("polycpp-clang-plugin.{}", dynamicLibSuffix()));
                 append({"-isystem", polycppIncludePath});
                 append({"-include", "polystl/polystl.h"});

                 const auto debug = opts->verbose == StdParOptions::VerboseLevel::Debug;
                 const bool noRewrite = std::getenv("POLYCPP_NO_REWRITE") != nullptr;
                 if (!noRewrite) {
                   append({"-Xclang", "-load", "-Xclang", polycppClangPlugin});
                   append({"-Xclang", "-add-plugin", "-Xclang", "polycpp"});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontExe, execPath)});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontVerbose, debug ? "1" : "0")});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontTargets, opts->targets)});
                 }

                 const auto compileOnly = std::vector{"-c", "-S", "-E", "-M", "-MM", "-MD", "-fsyntax-only"} ^
                                          exists([&](auto &flag) { return args.has(flag); });

                 switch (opts->mem) {
                   case StdParOptions::MemKind::Direct: break;
                   case StdParOptions::MemKind::Interpose:
                     append(Driver::clangPassPluginFlags(polyreflectPlugin, {fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                                                             "-polyreflect-late=Interpose"}));
                     break;
                   case StdParOptions::MemKind::Reflect:
                     append({"-include", "reflect-rt/rt.hpp"});
                     append(Driver::clangPassPluginFlags(
                         polyreflectPlugin,
                         {fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                          "-polyreflect-late=ProtectRT"})); // protect it here as it's an ODR error before LLD's plugin even runs
                     append(Driver::enableLLDAndLTO(args));
                     break;
                 }

                 if (!compileOnly) {
                   switch (opts->mem) {
                     case StdParOptions::MemKind::Direct: break;
                     case StdParOptions::MemKind::Interpose: break;
                     case StdParOptions::MemKind::Reflect:
                       append(
                           Driver::lldPassPluginFlags(polyreflectPlugin, {
                                                                             fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                                                             "-polyreflect-late=ReflectStack+ReflectMem",               //
                                                                         }));
                       break;
                   }
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), joinPath(polycppLibPath, staticLibraryName("polystl-static")));
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case StdParOptions::LinkKind::Dynamic: {
                       append(Driver::dynamicOriginLinkFlags(polycppLibPath, "polystl"));
                       if (opts->verbose == StdParOptions::VerboseLevel::Info) {
                         std::cerr
                             << fmt::format(
                                    "[PolyCpp] Dynamic linking of PolySTL runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)",
                                    joinPath(polycppLibPath, fmt::format("libpolystl.{}", dynamicLibSuffix())))
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

               return llvm::sys::ExecuteAndWait(clangPath, remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
             });
}

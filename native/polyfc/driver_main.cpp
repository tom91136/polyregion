#include <algorithm>
#include <cstdlib>

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#ifdef POLYREGION_FUSED_DRIVER
int flang_main(int argc, const char **argv);
#endif

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyfront/options_frontend.hpp"
#include "polyregion/conventions.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"

#include "driver_polyc.h"

using namespace aspartame;
using namespace polyregion::polyfront;
using polyregion::conventions::reflect::FlagLate;
using polyregion::conventions::reflect::FlagVerbose;
using polyregion::conventions::reflect::PassInterpose;
using polyregion::conventions::reflect::PassMem;
using polyregion::conventions::reflect::PassRecordAlloc;

int main(int argc, const char *argv[]) {
  CliArgs args(std::vector(argv, argv + argc));
  if (args.has("--polyc", 1)) {
    return polyregion::polyc(argc - 1, argv + 1);
  }

#if defined(_WIN32)
  // XXX per-process mspdbsrv endpoint; shared default deadlocks if a sibling link crashes
  // mid-PDB-write (LNK1318/LNK1201/STATUS_ACCESS_VIOLATION).
  polyregion::env::put("_MSPDBSRV_ENDPOINT_", fmt::format("polyfc-{}", llvm::sys::Process::getProcessId()).c_str(), true);

  // XXX derive arch / triple from LLVM's host triple; *TypeName drops the MSVC version suffix.
  const llvm::Triple hostTriple(llvm::sys::getDefaultTargetTriple());
  const auto windowsClangRtLib = fmt::format("clang_rt.builtins-{}.lib", hostTriple.getArchName().str());
  const auto windowsFlangRtSentinel =
      fmt::format("lib/{}-{}-{}-{}/flang_rt.runtime.static.lib", llvm::Triple::getArchTypeName(hostTriple.getArch()).str(),
                  llvm::Triple::getVendorTypeName(hostTriple.getVendor()).str(), llvm::Triple::getOSTypeName(hostTriple.getOS()).str(),
                  llvm::Triple::getEnvironmentTypeName(hostTriple.getEnvironment()).str());
#endif

  const std::string execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrAnchor);
  const std::string execParentPath = llvm::sys::path::parent_path(execPath).str();

  std::string flangPath;
#ifdef POLYREGION_FUSED_DRIVER
  // XXX fused build has no external driver; --driver/POLYFC_DRIVER are accepted but ignored.
  (void)args.popValue("--driver");
#else
  flangPath = resolveExternalDriver(args, polyregion::env::PolyfcDriver, "flang-new", execParentPath);
  if (flangPath.empty()) {
    fmt::print(stderr, "[PolyFC] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_flang-new>`\n",
               execPath);
    return EXIT_FAILURE;
  }
#endif

  return StdParOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               fmt::print(stderr, "[PolyFC] Unable to parse PolyFC specific arguments:\n{}\n", (errors ^ mk_string("\n") ^ indent(2)));
               return EXIT_FAILURE;
             },
             [&](const std::optional<StdParOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::vector<std::string> &xs) {
                 for (const auto &x : xs) {
#if defined(__APPLE__)
                   // Runtime, libc++, and JIT dependencies can share install
                   // locations; ld64.lld warns for repeated identical rpaths.
                   if (x.starts_with("-Wl,-rpath,") && std::find(remaining.begin(), remaining.end(), x) != remaining.end()) continue;
#endif
                   remaining.push_back(x);
                 }
               };

               std::vector<std::pair<const char *, std::string>> envs;

               if (opts) {
                 auto includes = mkDelimitedEnvPaths(polyregion::env::PolydcoInclude, "-isystem", llvm::sys::EnvPathSeparator);
                 auto libs = mkDelimitedEnvPaths(polyregion::env::PolydcoLib, {}, llvm::sys::EnvPathSeparator);
                 remaining ^= concat(includes);
                 remaining ^= concat(libs);

                 const auto polyfcLibPath = joinPath(resolveResourcePath(execParentPath, "polyfc"), "lib");
                 const auto polyreflectPlugin = joinPath(polyfcLibPath, fmt::format("polyreflect-plugin.{}", dynamicLibSuffix()));
                 const auto polyfcFlangPlugin = joinPath(polyfcLibPath, fmt::format("polyfc-flang-plugin.{}", dynamicLibSuffix()));

                 const auto debug = opts->verbose == StdParOptions::VerboseLevel::Debug;

                 if (const bool noRewrite = std::getenv(polyregion::env::PolyfcNoRewrite) != nullptr; !noRewrite) {
#ifndef POLYREGION_FUSED_DRIVER
                   append({"-Xflang", "-load", "-Xflang", polyfcFlangPlugin});
#endif
                   append({"-Xflang", "-plugin", "-Xflang", "polyfc"});
                   envs.emplace_back(PolyfrontExe, execPath);
                   envs.emplace_back(PolyfrontVerbose, debug ? "1" : "0");
                   envs.emplace_back(PolyfrontTargets, opts->targets);
                   if (opts->stackDepth) envs.emplace_back(PolyfrontStackDepth, std::to_string(*opts->stackDepth));
                   envs.emplace_back(PolyfrontJit, opts->jit != StdParOptions::LinkKind::Disabled ? "1" : "0");
                 }

                 const auto compileOnly =
                     std::vector{"-c", "-S", "-E", "-M", "-fsyntax-only"} ^ exists([&](auto &flag) { return args.has(flag); });
                 if (!compileOnly) {
                   // XXX flang has no clang -fplugin / -fpass-plugin equivalent; both Interpose and
                   // Reflect run at link time via LLD's LTO codegen. The Interpose pass rewrites
                   // malloc / operator new uses regardless of whether the rewrite happens per-TU or
                   // in the merged LTO module, so it works equally well as a late LTO pass.
                   const auto lateLldFlags = [&](const std::string &latePasses) {
#if defined(POLYREGION_FUSED_DRIVER) && defined(_WIN32)
                     // lld-link consumes /mllvm:VAL via -Xlinker; the polyreflect-* CL options come
                     // from polyld-link (polyreflect statically linked in), no pass-plugin load.
                     append({fmt::format("-B{}", execParentPath), "-fuse-ld=lld"});
                     append({"-Xlinker", fmt::format("/mllvm:-{}={}", FlagVerbose, debug ? "1" : "0"), //
                             "-Xlinker", fmt::format("/mllvm:-{}={}", FlagLate, latePasses)});
#else
                     // Flang diagnoses thin LTO as work-in-progress; full LTO is
                     // supported and still gives the late link pass one module.
                     append(Driver::enableLLDAndLTO(args, "full"));
                     append(Driver::lldPassPluginFlags(polyreflectPlugin, {fmt::format("-{}={}", FlagVerbose, debug ? "1" : "0"), //
                                                                           fmt::format("-{}={}", FlagLate, latePasses)}));
#endif
                   };
                   switch (opts->mem) {
                     case StdParOptions::MemKind::Direct: break;
                     case StdParOptions::MemKind::Interpose: lateLldFlags(PassInterpose); break;
                     case StdParOptions::MemKind::Reflect: lateLldFlags(fmt::format("{}+{}", PassRecordAlloc, PassMem)); break;
                   }
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), joinPath(polyfcLibPath, staticLibraryName("polydco-static")));
                       // polydco-static references _rt_record / _rt_release / _rt_reflect_p; link the
                       // shared polyreflect-rt so MSVC's dllimport thunks resolve.
                       append(Driver::dynamicOriginLinkFlags(polyfcLibPath, "polyreflect-rt"));
#if defined(_WIN32)
                       // flang Fortran sources cannot use #pragma comment(lib); inject the
                       // Windows system imports that polydco-static depends on directly.
                       append({"-Xlinker", "Version.lib", "-Xlinker", "psapi.lib", "-Xlinker", "ntdll.lib", "-Xlinker", "ws2_32.lib",
                               "-Xlinker", "ole32.lib", "-Xlinker", "shell32.lib", "-Xlinker", "advapi32.lib", "-Xlinker", "uuid.lib",
                               "-Xlinker", "delayimp.lib"});
                       // flang_rt uses __udivti3 (128-bit div); pull clang_rt.builtins from
                       // the resource dir since MSVC libucrt lacks it.
                       const auto clangResourceDirRel = fmt::format("clang/{}", LLVM_VERSION_MAJOR);
                       for (const auto &resRoot : {joinPath(execParentPath, "..", "lib", clangResourceDirRel),
                                                   joinPath(POLYFC_FUSED_DIST_DIR, "lib", clangResourceDirRel)}) {
                         const auto builtins = joinPath(resRoot, "lib", "windows", windowsClangRtLib);
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
                       append(appleDistLibcxxRpath(execParentPath));
                       if (opts->verbose == StdParOptions::VerboseLevel::Info) {
                         fmt::print(stderr,
                                    "[PolyFC] Dynamic linking of PolyDCO runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)\n",
                                    joinPath(polyfcLibPath, fmt::format("libpolydco.{}", dynamicLibSuffix())));
                       }
                       break;
                     }
                     case StdParOptions::LinkKind::Disabled: break;
                   }
                   const auto compilerLibPath = flangPath.empty() ? joinPath(execParentPath, "..", "lib")
                                                                  : joinPath(llvm::sys::path::parent_path(flangPath), "..", "lib");
                   append(jitCompilerLinkFlags(opts->jit, polyfcLibPath, compilerLibPath, /*needsCxxRuntime*/ true));
                   if (const char *t = std::getenv(polyregion::env::PolyfcLinkThreads); t && *t)
                     append({fmt::format("-Wl,--threads={}", t)});
                 }
               }

               for (auto [k, v] : envs)
                 polyregion::env::put(k, v.c_str(), true);
#ifdef POLYREGION_FUSED_DRIVER
               // Probe install-relative paths first so the binary works both installed and
               // from the build tree.
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
  #if defined(_WIN32)
               // XXX Windows-only: only the MSVC dist ships per-triple flang_rt.
               const auto flangResourceDirRel = fmt::format("clang/{}", LLVM_VERSION_MAJOR);
               if (auto resDir = firstWith(windowsFlangRtSentinel, {joinPath(execParentPath, "..", "lib", flangResourceDirRel),
                                                                    joinPath(POLYFC_FUSED_DIST_DIR, "lib", flangResourceDirRel)});
                   !resDir.empty()) {
                 remaining.push_back("-resource-dir");
                 remaining.push_back(resDir);
               }
  #endif
               remaining[0] = execPath;
               auto rawArgs = remaining ^ map([](auto &arg) { return arg.c_str(); });
               return flang_main(static_cast<int>(rawArgs.size()), rawArgs.data());
#else
               return llvm::sys::ExecuteAndWait(flangPath, remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
#endif
             });
}

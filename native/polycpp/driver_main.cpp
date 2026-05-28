#include <cstdlib>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#ifdef POLYREGION_FUSED_DRIVER
  #include "llvm/Support/LLVMDriver.h"
int clang_main(int Argc, char **Argv, const llvm::ToolContext &ToolContext);
#endif

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyfront/options_frontend.hpp"
#include "polyregion/env.h"

#include "driver_polyc.h"

using namespace aspartame;
using namespace polyregion::polyfront;

int main(int argc, const char *argv[]) {
#ifdef POLYREGION_FUSED_DRIVER
  // XXX clang_main requires InitLLVM for target/PassBuilder registration; use throwaway argv
  // copies so the Windows GetCommandLineArgumentsW rescan cannot clobber ours.
  int initArgc = argc;
  const char **initArgv = argv;
  llvm::InitLLVM initLLVM(initArgc, initArgv);
#endif
  CliArgs args(std::vector(argv, argv + argc));
  if (args.has("--polyc", 1)) {
    return polyregion::polyc(argc - 1, argv + 1);
  }

#if defined(_WIN32)
  // XXX per-process mspdbsrv endpoint; see polyfc/driver_main.cpp.
  polyregion::env::put("_MSPDBSRV_ENDPOINT_", fmt::format("polycpp-{}", llvm::sys::Process::getProcessId()).c_str(), true);
#endif

  std::string execPath = llvm::sys::fs::getMainExecutable(argv[0], (void *)&addrAnchor);
  std::string execParentPath = llvm::sys::path::parent_path(execPath).str();

  std::string clangPath;

#ifdef POLYREGION_FUSED_DRIVER
  // XXX fused build has no external driver; --driver/POLYCPP_DRIVER are accepted but ignored.
  (void)args.popValue("--driver");
#else
  clangPath = resolveExternalDriver(args, "POLYCPP_DRIVER", "clang++", execParentPath);
  if (clangPath.empty()) {
    fmt::print(stderr, "[PolyCpp] Cannot locate driver executable at {}, manually specify the driver with `--driver <path_to_clang++>`\n",
               execPath);
    return EXIT_FAILURE;
  }
#endif

  return StdParOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               fmt::print(stderr, "[PolyCpp] Unable to parse PolyCpp specific arguments:\n{}\n", (errors ^ mk_string("\n") ^ indent(2)));
               return EXIT_FAILURE;
             },
             [&](const std::optional<StdParOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::vector<std::string> &xs) { remaining ^= concat_inplace(xs); };

               if (opts) {
                 remaining ^= concat_inplace(mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem", llvm::sys::EnvPathSeparator));
                 remaining ^= concat_inplace(mkDelimitedEnvPaths("POLYSTL_LIB", {}, llvm::sys::EnvPathSeparator));

                 const auto polycppResourcePath = resolveResourcePath(execParentPath, "polycpp");
                 const auto polycppIncludePath = joinPath(polycppResourcePath, "include");
                 const auto polycppLibPath = joinPath(polycppResourcePath, "lib");
                 const auto polyreflectPlugin = joinPath(polycppLibPath, fmt::format("polyreflect-plugin.{}", dynamicLibSuffix()));
                 const auto polycppClangPlugin = joinPath(polycppLibPath, fmt::format("polycpp-clang-plugin.{}", dynamicLibSuffix()));
                 append({"-isystem", polycppIncludePath});
                 append({"-include", "polystl/polystl.h"});

                 const auto debug = opts->verbose == StdParOptions::VerboseLevel::Debug;
                 const bool noRewrite = std::getenv("POLYCPP_NO_REWRITE") != nullptr;
#ifndef POLYREGION_FUSED_DRIVER
                 // XXX non-fused: plugin only loaded when rewriting; fused needs it for polyreflect callbacks.
                 if (!noRewrite) append({"-Xclang", "-load", "-Xclang", polycppClangPlugin});
#endif
                 if (!noRewrite
#ifdef POLYREGION_FUSED_DRIVER
                     || opts->mem != StdParOptions::MemKind::Direct
#endif
                 ) {
                   append({"-Xclang", "-add-plugin", "-Xclang", "polycpp"});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontExe, execPath)});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontVerbose, debug ? "1" : "0")});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("{}={}", PolyfrontTargets, opts->targets)});
                 }

                 const auto compileOnly = std::vector{"-c", "-S", "-E", "-M", "-MM", "-MD", "-fsyntax-only"} ^
                                          exists([&](auto &flag) { return args.has(flag); });

                 const auto passPluginFlags = [&](const std::vector<std::string> &mllvmArgs) {
#ifdef POLYREGION_FUSED_DRIVER
                   return mllvmArgs ^ map([](auto &arg) { return fmt::format("-mllvm={}", arg); });
#else
                   return Driver::clangPassPluginFlags(polyreflectPlugin, mllvmArgs);
#endif
                 };
                 switch (opts->mem) {
                   case StdParOptions::MemKind::Direct: break;
                   case StdParOptions::MemKind::Interpose:
                     append(passPluginFlags({fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                             "-polyreflect-late=Interpose+ReflectStack"}));
                     break;
                   case StdParOptions::MemKind::Reflect:
                     append({"-include", "reflect-rt/rt.hpp"});
                     // XXX run ProtectRT per-TU; otherwise LLD's plugin would see the ODR error first.
                     append(passPluginFlags({fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"), //
                                             "-polyreflect-late=ProtectRT"}));
                     append(Driver::enableLLDAndLTO(args));
                     break;
                 }

                 if (!compileOnly) {
                   switch (opts->mem) {
                     case StdParOptions::MemKind::Direct: break;
                     case StdParOptions::MemKind::Interpose: break;
                     case StdParOptions::MemKind::Reflect:
#ifdef POLYREGION_FUSED_DRIVER
                       // XXX polyld-link is installed as ld.lld / lld-link.exe next to polycpp; -B
                       // routes `-fuse-ld=lld` (literal name required by clang's LTO check) to it.
                       append({fmt::format("-B{}", execParentPath), "-fuse-ld=lld"});
  #ifdef _WIN32
                       // XXX lld-link takes /mllvm:VAL, ld.lld takes -mllvm VAL; -Xlinker forwards either.
                       // lld-link has no --lto-newpm-passes equivalent, so keep the EP-callback path.
                       append({"-Xlinker", fmt::format("/mllvm:-polyreflect-verbose={}", debug ? "1" : "0"), "-Xlinker",
                               "/mllvm:-polyreflect-late=ReflectStack+ReflectMem"});
                       // XXX /INCLUDE: pulls polyreflect-rt's new/delete in ahead of vcruntime's.
                       for (auto sym : {"??2@YAPEAX_K@Z",
                                        "??2@YAPEAX_KW4align_val_t@std@@@Z",
                                        "??2@YAPEAX_KAEBUnothrow_t@std@@@Z",
                                        "??2@YAPEAX_KW4align_val_t@std@@AEBUnothrow_t@1@@Z",
                                        "??_U@YAPEAX_K@Z",
                                        "??_U@YAPEAX_KW4align_val_t@std@@@Z",
                                        "??_U@YAPEAX_KAEBUnothrow_t@std@@@Z",
                                        "??_U@YAPEAX_KW4align_val_t@std@@AEBUnothrow_t@1@@Z",
                                        "??3@YAXPEAX@Z",
                                        "??_V@YAXPEAX@Z",
                                        "??3@YAXPEAXW4align_val_t@std@@@Z",
                                        "??_V@YAXPEAXW4align_val_t@std@@@Z",
                                        "??3@YAXPEAX_K@Z",
                                        "??_V@YAXPEAX_K@Z",
                                        "??3@YAXPEAX_KW4align_val_t@std@@@Z",
                                        "??_V@YAXPEAX_KW4align_val_t@std@@@Z",
                                        "??3@YAXPEAXAEBUnothrow_t@std@@@Z",
                                        "??_V@YAXPEAXAEBUnothrow_t@std@@@Z",
                                        "??3@YAXPEAXW4align_val_t@std@@AEBUnothrow_t@1@@Z",
                                        "??_V@YAXPEAXW4align_val_t@std@@AEBUnothrow_t@1@@Z"}) {
                         append({"-Xlinker", fmt::format("/INCLUDE:{}", sym)});
                       }
  #else
                       // XXX At -O0 LLD's LTO codegen builds no optimisation pipeline at all, so
                       // EP-callback-registered passes never fire. Inject the late passes by name
                       // via --lto-newpm-passes so they run regardless of opt level. Comma in
                       // value uses -Xlinker to avoid `-Wl,...,...` arg splitting.
                       append({fmt::format("-Wl,-mllvm,-polyreflect-verbose={}", debug ? "1" : "0"), "-Xlinker",
                               "--lto-newpm-passes=polyreflect-stack,polyreflect-mem"});
  #endif
#else
                       append(Driver::lldPassPluginFlags(polyreflectPlugin, {
                                                                                fmt::format("-polyreflect-verbose={}", debug ? "1" : "0"),
                                                                                "-polyreflect-late=ReflectStack+ReflectMem",
                                                                            }));
                       // XXX -O0 LLD LTO skips EP callbacks; force the passes by name.
                       append({"-Xlinker", "--lto-newpm-passes=polyreflect-stack,polyreflect-mem"});
#endif
                       break;
                   }
                   switch (opts->rt) {
                     case StdParOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), joinPath(polycppLibPath, staticLibraryName("polystl-static")));
                       // XXX link libpolyreflect-rt so the per-TU malloc/free interposers resolve to
                       // its single `_rt_record` / `_rt_release` instance.
                       append(Driver::dynamicOriginLinkFlags(polycppLibPath, "polyreflect-rt"));
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case StdParOptions::LinkKind::Dynamic: {
                       append(Driver::dynamicOriginLinkFlags(polycppLibPath, "polystl"));
                       append(Driver::dynamicOriginLinkFlags(polycppLibPath, "polyreflect-rt"));
                       append(appleDistLibcxxRpath(execParentPath));
                       if (opts->verbose == StdParOptions::VerboseLevel::Info) {
                         fmt::print(stderr,
                                    "[PolyCpp] Dynamic linking of PolySTL runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)\n",
                                    joinPath(polycppLibPath, fmt::format("libpolystl.{}", dynamicLibSuffix())));
                       }
                       break;
                     }
                     case StdParOptions::LinkKind::Disabled: break;
                   }
                   if (const char *t = std::getenv("POLYCPP_LINK_THREADS"); t && *t) append({fmt::format("-Wl,--threads={}", t)});
                 }
               }

#ifdef POLYREGION_FUSED_DRIVER
               // XXX a basename ending in `cpp` puts clang into CPP (preprocess-only) mode; force
               // g++ mode via argv[1], except for -cc1* re-entrant invocations.
               remaining[0] = execPath;
               const bool isCC1 =
                   remaining.size() >= 2 && (remaining[1] == "-cc1" || remaining[1] == "-cc1as" || remaining[1] == "-cc1gen-reproducer");
               if (!isCC1) remaining.insert(remaining.begin() + 1, "--driver-mode=g++");
               std::vector<char *> rawArgs;
               rawArgs.reserve(remaining.size());
               for (auto &arg : remaining)
                 rawArgs.push_back(arg.data());
               llvm::ToolContext toolContext{execPath.c_str(), nullptr, false};
               return clang_main(static_cast<int>(rawArgs.size()), rawArgs.data(), toolContext);
#else
               remaining[0] = "clang++";
               fmt::print(">>> {}\n", remaining ^ mk_string(" "));
               return llvm::sys::ExecuteAndWait(clangPath, remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
#endif
             });
}

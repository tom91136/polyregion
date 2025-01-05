#include "fmt/core.h" // fmt/std.h requires RTTI
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "aspartame/all.hpp"
#include "driver_polyc.h"
#include "llvm/Support/Program.h"

using namespace aspartame;

struct CliArgs {
  std::vector<const char *> data;
  std::unordered_set<size_t> deleted{};
  explicit CliArgs(const std::vector<const char *> &data) : data(data) {}

  bool has(const std::string &flag) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() |
           exists([&](auto &chars) { return chars == flag; });
  }

  bool has(const std::string &flag, size_t pos) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) |
           exists([&](auto &chars, auto i) { return chars == flag && i == pos; });
  }

  std::optional<std::string> get(const std::string &flag) const {
    return data                                                        //
           | zip_with_index()                                          //
           | filter([&](auto v, auto i) { return !deleted.count(i); }) //
           | collect_first([&](auto &chars, auto i) -> std::optional<std::string> {
               if (const std::string arg = chars; arg == flag) {
                 if (i + 1 < data.size()) return data[i + 1];                                // -foo :: bar :: Nil
                 else return {};                                                             // -foo :: Nil
               } else if (arg ^ starts_with(flag + "=")) return arg.substr(flag.size() + 1); // -foo=bar
               return {};
             });
  }

  bool popBool(const std::string &flag) {
    auto oldSize = deleted.size();
    for (size_t i = 0; i < data.size(); ++i) {
      if (deleted.count(i)) continue;
      if (data[i] == flag) deleted.insert(i);
    }
    return oldSize != deleted.size();
  }

  std::optional<std::string> popValue(const std::string &flag) {
    std::optional<std::string> last;
    for (size_t i = 0; i < data.size(); ++i) {
      if (deleted.count(i)) continue;
      if (const std::string arg = data[i]; arg == flag && i + 1 < data.size()) { // -foo :: bar :: Nil
        deleted.insert(i);
        deleted.insert(i + 1);
        last = data[i + 1];
      } else if (arg ^ starts_with(flag + "=")) { // -foo=bar
        deleted.insert(i);
        last = arg.substr(flag.size() + 1);
      }
    }
    return last;
  }

  std::vector<const char *> remaining() const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() | to_vector();
  }
};

struct PolyCppOptions {

  enum class LinkKind : uint8_t { Static = 1, Dynamic, Disabled = 3 };

  static std::variant<std::string, LinkKind> parseLinkKind(const std::string &arg) {
    if (auto v = arg ^ to_lower(); v == "static") return LinkKind::Static;
    else if (v == "dynamic") return LinkKind::Dynamic;
    else if (v == "disabled") return LinkKind::Disabled;
    return "Unknown link kind `" + arg + "`";
  }

  bool verbose = false;
  bool noCompress = false;
  bool interposeMalloc = true;
  bool interposeAlloca = false;
  std::string targets{};
  LinkKind rt = LinkKind::Static;
  LinkKind jit = LinkKind::Disabled;

  static std::variant<std::vector<std::string>, std::optional<PolyCppOptions>> parse(CliArgs &args) {

    const std::string fStdParFlag = "-fstdpar";
    const std::string fStdParVerboseFlag = "-fstdpar-verbose";
    const std::string fStdParArchNoCompressFlag = "-fstdpar-no-compress";
    const std::string fStdParInterposeMallocFlag = "-fstdpar-interpose-malloc";
    const std::string fStdParInterposeAllocaFlag = "-fstdpar-interpose-alloca";
    const std::string fStdParArchFlag = "-fstdpar-arch";
    const std::string fStdParRtFlag = "-fstdpar-rt";
    const std::string fStdParJitFlag = "-fstdpar-jit";

    auto fStdPar = false, fStdParDependents = false;
    PolyCppOptions options;
    std::vector<std::string> errors;

    auto markError = [&](const std::string &prefix) { return [&](const std::string &x) { errors.push_back("\"" + prefix + "\": " + x); }; };

    if (args.popBool(fStdParFlag)) {
      fStdPar = true;
    }
    if (args.popBool(fStdParVerboseFlag)) {
      fStdParDependents = true;
      options.verbose = true;
    }
    if (args.popBool(fStdParArchNoCompressFlag)) {
      fStdParDependents = true;
      options.noCompress = true;
    }
    if (args.popBool(fStdParInterposeMallocFlag)) {
      fStdParDependents = true;
      options.interposeMalloc = true;
    }
    if (args.popBool(fStdParInterposeAllocaFlag)) {
      fStdParDependents = true;
      options.interposeAlloca = true;
    }
    if (auto arch = args.popValue(fStdParArchFlag)) {
      fStdParDependents = true;
      options.targets = *arch;
    }
    if (auto rt = args.popValue(fStdParRtFlag)) {
      fStdParDependents = true;
      parseLinkKind(*rt) ^ foreach_total(markError(fStdParRtFlag), [&](const LinkKind &x) { options.rt = x; });
    }
    if (auto jit = args.popValue(fStdParJitFlag)) {
      fStdParDependents = true;
      parseLinkKind(*jit) ^ foreach_total(markError(fStdParJitFlag), [&](const LinkKind &x) { options.jit = x; });
    }

    if (!fStdPar && fStdParDependents)
      errors.insert(errors.begin(), fStdParFlag + " not specified but StdPar dependent flags used, pleased add " + fStdParFlag);

    if (errors.empty()) return fStdPar ? std::optional{options} : std::nullopt;
    else return errors;
  }
};

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

  return PolyCppOptions::parse(args) ^
         fold_total(
             [&](const std::vector<std::string> &errors) {
               std::cerr << fmt::format("[PolyCpp] Unable to parse PolyCpp specific arguments:\n{}", (errors ^ mk_string("\n") ^ indent(2)))
                         << std::endl;
               return EXIT_FAILURE;
             },
             [&](const std::optional<PolyCppOptions> &opts) {
               auto remaining = args.remaining() ^ map([](auto &s) -> std::string { return s; });
               auto append = [&](const std::initializer_list<std::string> &xs) { remaining.insert(remaining.end(), xs); };

               if (opts) {
                 auto includes = mkDelimitedEnvPaths("POLYSTL_INCLUDE", "-isystem");
                 auto libs = mkDelimitedEnvPaths("POLYSTL_LIB", {});
                 remaining.insert(remaining.end(), includes.begin(), includes.end());
                 remaining.insert(remaining.end(), libs.begin(), libs.end());

                 auto polycppResourcePath = execParentPath / "lib/polycpp";
                 auto polycppIncludePath = polycppResourcePath / "include";
                 auto polycppLibPath = polycppResourcePath / "lib";
                 auto polycppReflectPlugin = polycppLibPath / "polystl-reflect-plugin.so";
                 auto polycppClangPlugin = polycppLibPath / "polycpp-clang-plugin.so";
                 append({"-isystem", polycppIncludePath.string()});
                 append({"-include", "polystl/polystl.h"});
                 append({"-include", "rt-reflect/rt.hpp"});

                 bool noRewrite = std::getenv("POLYCPP_NO_REWRITE") != nullptr;
                 if (!noRewrite) {
                   append({"-Xclang", "-load", "-Xclang", polycppClangPlugin.string()});
                   append({"-Xclang", "-add-plugin", "-Xclang", "polycpp"});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("exe={}", execPath.string())});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("verbose={}", opts->verbose ? "1" : "0")});
                   append({"-Xclang", "-plugin-arg-polycpp", "-Xclang", fmt::format("targets={}", opts->targets)});
                   if (opts->interposeMalloc) append({fmt::format("-fpass-plugin={}", polycppReflectPlugin.string())});
                 }

                 auto compileOnly = std::vector{"-c", "-S", "-E", "-M", "-MM", "-MD", "-fsyntax-only"} ^
                                    exists([&](auto &flag) { return args.has(flag); });
                 if (!compileOnly) {
                   switch (opts->rt) {
                     case PolyCppOptions::LinkKind::Static: {
                       remaining.insert(remaining.end(), (polycppLibPath / "libpolystl-static.a").string());
                       // if (!opts->noCompress) append({"-Wl,--compress-debug-sections=zlib,--gc-sections"});
                       break;
                     }
                     case PolyCppOptions::LinkKind::Dynamic: {
                       append({fmt::format("-L{}", polycppLibPath.string()), "-lpolystl", //
                               fmt::format("-Wl,-rpath,{}", polycppLibPath.string()), "-Wl,-rpath,$ORIGIN"});
                       if (opts->verbose) {
                         std::cerr
                             << fmt::format(
                                    "[PolyCpp] Dynamic linking of PolySTL runtime requested, if you would like to relocate your binary, "
                                    "please copy {} to the same directory as the executable (-rpath=$ORIGIN has been set for you)",
                                    (polycppLibPath / "libpolystl.so").string())
                             << std::endl;
                       }
                       break;
                     }
                     case PolyCppOptions::LinkKind::Disabled: break;
                   }
                 }
               }

               remaining[0] = "clang++";
               std::cout << ">>> " << (remaining ^ mk_string(" ")) << std::endl;

               return llvm::sys::ExecuteAndWait(clangPath.string(),
                                                remaining | map([](auto &x) -> llvm::StringRef { return x; }) | to_vector());
             });
}
#pragma once

#include <cstdlib>
#include <vector>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "nlohmann/json.hpp"

#include "polyregion/types.h"

#include "options.hpp"
#include "polyast_codec.h"

template <> struct std::hash<std::pair<polyregion::compiletime::Target, std::string>> {
  std::size_t operator()(const std::pair<polyregion::compiletime::Target, std::string> &p) const noexcept {
    const std::size_t h1 = std::hash<polyregion::compiletime::Target>{}(p.first);
    const std::size_t h2 = std::hash<std::string>{}(p.second);
    return h1 ^ h2 << 1;
  }
};

namespace polyregion::polyfront {

using namespace aspartame;

struct KernelObject {
  runtime::ModuleFormat format{};
  runtime::PlatformKind kind{};
  std::vector<std::string> features{};
  std::string moduleImage{};
};

struct KernelBundle {
  std::string moduleName{};
  std::vector<KernelObject> objects{};
  std::vector<std::pair<bool, polyast::StructLayout>> layouts{};
  std::string metadata;
};

struct Options {
  using Target = std::pair<compiletime::Target, std::string>;

  bool verbose = false;
  std::string executable;
  std::vector<Target> targets;

  static std::variant<std::vector<std::string>, Options>
  parseArgs(std::optional<std::string> maybeExe, std::optional<std::string> maybeVerbose, std::optional<std::string> maybeTargets) {
    Options opts;
    std::vector<std::string> errors;
    if (auto verbose = maybeVerbose) opts.verbose = *verbose == "1";
    if (auto exe = maybeExe) opts.executable = *exe;
    else errors.emplace_back("exe argument missing");
    if (auto targets = maybeTargets) {
      // archA@featureA:archB@featureB:...archN@featureN
      for (auto &rawArchAndFeaturesList : *targets ^ split(':')) {
        auto archAndFeatures = rawArchAndFeaturesList ^ split('@');
        if (archAndFeatures.size() != 2)
          errors.emplace_back("Missing or invalid placement of arch and feature separator '@' in " + rawArchAndFeaturesList);
        if (auto s = polyregion::compiletime::TargetSpec::findByName(archAndFeatures[0]); s) {
          for (auto &feature : archAndFeatures[1] ^ split(','))
            opts.targets.emplace_back(s->codegen, feature);
        } else errors.emplace_back("Unknown arch " + archAndFeatures[0]);
      }
      opts.targets = opts.targets ^ distinct();
    } else errors.emplace_back("target argument missing");
    if (errors.empty()) return opts;
    else return errors;
  }

  static std::variant<std::vector<std::string>, Options> parseArgs(const std::vector<std::string> &args) {
    auto parseSuffix = [&](const std::string &key) -> std::optional<std::string> {
      const auto prefix = key + "=";
      return args ^ collect_first([&](auto &arg) -> std::optional<std::string> {
               if (arg ^ starts_with(prefix)) return arg.substr(prefix.size());
               return {};
             });
    };
    return parseArgs(parseSuffix(PolyfrontExe), parseSuffix(PolyfrontVerbose), parseSuffix(PolyfrontTargets));
  }

  static std::variant<std::vector<std::string>, Options> parseArgsFromEnv() {
    auto readEnv = [&](const auto &key) -> std::optional<std::string> {
      if (auto env = std::getenv(key)) return env;
      else return {};
    };
    return parseArgs(readEnv(PolyfrontExe), readEnv(PolyfrontVerbose), readEnv(PolyfrontTargets));
  }
};

static std::variant<std::string, polyast::CompileResult> compileProgram(const polyfront::Options &opts,    //
                                                                        const polyast::Program &p,         //
                                                                        const compiletime::Target &target, //
                                                                        const std::string &arch) {
  auto data = polyast::hashed_program_to_msgpack(p);

  auto tempModel = [](llvm::StringRef leaf) {
    llvm::SmallString<128> path;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, path);
    llvm::sys::path::append(path, leaf);
    return path;
  };

  auto inputFileExp = llvm::sys::fs::TempFile::create(tempModel("polyfront-in-%%%%%%.msgpack"));
  if (!inputFileExp) return "Failed to create temp input file: " + llvm::toString(inputFileExp.takeError());
  auto inputFile = std::move(*inputFileExp);

  auto outputFileExp = llvm::sys::fs::TempFile::create(tempModel("polyfront-out-%%%%%%.msgpack"));
  if (!outputFileExp) {
    llvm::consumeError(inputFile.discard());
    return "Failed to create temp output file: " + llvm::toString(outputFileExp.takeError());
  }
  auto outputFile = std::move(*outputFileExp);

  auto cleanup = llvm::make_scope_exit([&] {
    llvm::consumeError(inputFile.discard());
    llvm::consumeError(outputFile.discard());
  });

  {
    llvm::raw_fd_ostream file(inputFile.FD, /*shouldClose=*/false);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
    file.flush();
  }

  const auto canonical = polyregion::compiletime::TargetSpec::findByCodegen(target);
  if (!canonical) return std::string("Unknown codegen target ordinal: ") + std::to_string(static_cast<int>(target));
  std::vector<llvm::StringRef> args{
      //
      "", "--polyc", inputFile.TmpName, "--out", outputFile.TmpName, "--target", std::string_view(canonical->canonical), "--arch", arch};

  if (opts.verbose) {
    (llvm::errs() << (args | prepend(opts.executable) | mk_string(" ", [](auto &s) { return s.data(); })) << "\n").flush();
  }

  if (int code = llvm::sys::ExecuteAndWait(opts.executable, args); code != 0)
    return "Non-zero exit code for task: " + (args ^ mk_string(" ", [](auto &s) { return s.str(); }));

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputFile.TmpName);

  if (auto Err = BufferOrErr.getError()) return "Failed to read output buffer: " + toString(llvm::errorCodeToError(Err));
  // The polycpp clang plugin is built with -fno-exceptions, so a throw from `from_msgpack(empty)`
  // would unwind into terminate(); return a string error here instead of letting it propagate.
  if ((*BufferOrErr)->getBufferSize() == 0)
    return "Empty output from polyc subprocess (exit code was zero, this is a polyc bug) for task: " +
           (args ^ mk_string(" ", [](auto &s) { return s.str(); }));
  const auto *begin = reinterpret_cast<const uint8_t *>((*BufferOrErr)->getBufferStart());
  const auto *end = begin + (*BufferOrErr)->getBufferSize();
  return polyast::compileresult_from_msgpack(begin, end);
}

} // namespace polyregion::polyfront

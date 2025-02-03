#pragma once

#include <cstdlib>
#include <vector>

#include "options.hpp"
#include "polyast_codec.h"

#include "aspartame/all.hpp"
#include "nlohmann/json.hpp"
#include "polyregion/types.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

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
        if (auto t = polyregion::compiletime::parseTarget(archAndFeatures[0]); t) {
          for (auto &feature : archAndFeatures[1] ^ split(','))
            opts.targets.emplace_back(*t, feature);
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
  auto data = nlohmann::json::to_msgpack(polyast::hashed_to_json(polyast::program_to_json(p)));

  llvm::SmallString<64> inputPath;
  auto inputCreateEC = llvm::sys::fs::createTemporaryFile("", "", inputPath);
  if (inputCreateEC) return "Failed to create temp input file: " + inputCreateEC.message();

  llvm::SmallString<64> outputPath;
  auto outputCreateEC = llvm::sys::fs::createTemporaryFile("", "", outputPath);
  if (outputCreateEC) return "Failed to create temp output file: " + outputCreateEC.message();

  std::error_code streamEC;
  llvm::raw_fd_ostream file(inputPath, streamEC, llvm::sys::fs::OF_None);
  if (streamEC) return "Failed to open file: " + streamEC.message();

  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  file.flush();

  std::vector<llvm::StringRef> args{//
                                    "",         "--polyc",         inputPath.str(), "--out", outputPath.str(),
                                    "--target", to_string(target), "--arch",        arch};

  if (opts.verbose) {
    (llvm::errs() << (args | prepend(opts.executable) | mk_string(" ", [](auto &s) { return s.data(); })) << "\n").flush();
  }

  if (int code = llvm::sys::ExecuteAndWait(opts.executable, args); code != 0)
    return "Non-zero exit code for task: " + (args ^ mk_string(" ", [](auto &s) { return s.str(); }));

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputPath);

  if (auto Err = BufferOrErr.getError()) return "Failed to read output buffer: " + toString(llvm::errorCodeToError(Err));
  else
    return polyast::compileresult_from_json(nlohmann::json::from_msgpack((*BufferOrErr)->getBufferStart(), (*BufferOrErr)->getBufferEnd()));
}

} // namespace polyregion::polyfront

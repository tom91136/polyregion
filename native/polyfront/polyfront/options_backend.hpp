#pragma once

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "aspartame/all.hpp"
#include "nlohmann/json.hpp"

#include "polyregion/conventions.h"
#include "polyregion/mirror_names.h"
#include "polyregion/types.h"

#include "options.hpp"
#include "pass_specs.hpp"
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

inline bool entryNeedsErrorBuffer(const polyast::CompileResult &r) {
  return r.entryArgs ^ exists([](auto &n) { return n.symbol == conventions::ErrorArg; });
}

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
  std::unordered_map<std::string, std::unordered_set<std::string>> readOnlyMembers{};
  std::string metadata;
  std::string hostMirrorBitcode{};
  std::string mirrorId{};
  bool asserts = false;
};

struct Options {
  using Target = std::pair<compiletime::Target, std::string>;

  bool verbose = false;
  std::string executable;
  std::vector<Target> targets;
  std::optional<int> stackDepth = {};

  static std::variant<std::vector<std::string>, Options> parseArgs(std::optional<std::string> maybeExe,
                                                                   std::optional<std::string> maybeVerbose,
                                                                   std::optional<std::string> maybeTargets,
                                                                   std::optional<std::string> maybeStackDepth = {}) {
    Options opts;
    std::vector<std::string> errors;
    if (auto verbose = maybeVerbose) opts.verbose = *verbose == "1";
    if (auto exe = maybeExe) opts.executable = *exe;
    else errors.emplace_back("exe argument missing");
    if (auto depth = maybeStackDepth) {
      if (auto n = parsePositiveInt(*depth)) opts.stackDepth = *n;
      else errors.emplace_back("invalid stack depth: " + *depth);
    }
    if (auto targets = maybeTargets) {
      for (auto &rawEntry : *targets ^ split(';')) {
        const auto bang = rawEntry.find('!');
        const auto rawArchAndFeaturesList = bang == std::string::npos ? rawEntry : rawEntry.substr(0, bang);
        auto archAndFeatures = rawArchAndFeaturesList ^ split('@');
        if (archAndFeatures.size() != 2) {
          errors.emplace_back("Missing or invalid placement of arch and feature separator '@' in " + rawArchAndFeaturesList);
          continue;
        }
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
    return parseArgs(parseSuffix(PolyfrontExe), parseSuffix(PolyfrontVerbose), parseSuffix(PolyfrontTargets),
                     parseSuffix(PolyfrontStackDepth));
  }

  static std::variant<std::vector<std::string>, Options> parseArgsFromEnv() {
    auto readEnv = [&](const auto &key) -> std::optional<std::string> {
      if (auto env = std::getenv(key)) return env;
      else return {};
    };
    return parseArgs(readEnv(PolyfrontExe), readEnv(PolyfrontVerbose), readEnv(PolyfrontTargets), readEnv(PolyfrontStackDepth));
  }
};

static std::variant<std::string, polyast::CompileResult> compileProgram(const polyfront::Options &opts,    //
                                                                        const polyast::Program &p,         //
                                                                        const compiletime::Target &target, //
                                                                        const std::string &arch,           //
                                                                        const std::vector<std::string> &extraArgs = {}) {
  auto data = polyast::hashed_program_to_msgpack(p);

  // Use createTemporaryFile (returns path, closes its FD) rather than fs::TempFile (keeps the FD
  // open + on Windows opens with FILE_FLAG_DELETE_ON_CLOSE). A held parent FD blocks the child
  // from opening via Windows file sharing; closing it deletes the file outright. Either way the
  // child's read_struct throws "Cannot open ..." into SEH 0xE06D7363.
  llvm::SmallString<128> inputPath, outputPath;
  if (auto ec = llvm::sys::fs::createTemporaryFile("polyfront-in", "msgpack", inputPath))
    return "Failed to create temp input file: " + ec.message();
  if (auto ec = llvm::sys::fs::createTemporaryFile("polyfront-out", "msgpack", outputPath)) {
    llvm::sys::fs::remove(inputPath);
    return "Failed to create temp output file: " + ec.message();
  }
  auto cleanup = llvm::scope_exit([&] {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
  });

  {
    std::error_code ec;
    llvm::raw_fd_ostream file(inputPath, ec, llvm::sys::fs::OF_None);
    if (ec) return "Failed to open temp input file for writing: " + ec.message();
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
    file.flush();
  }

  const auto canonical = polyregion::compiletime::TargetSpec::findByCodegen(target);
  if (!canonical) return std::string("Unknown codegen target ordinal: ") + std::to_string(static_cast<int>(target));
  std::vector<llvm::StringRef> args{
      //
      "", "--polyc", inputPath.str(), "--out", outputPath.str(), "--target", std::string_view(canonical->canonical), "--arch", arch};
  for (const auto &a : extraArgs)
    args.emplace_back(a);

  if (opts.verbose) {
    (llvm::errs() << (args | prepend(opts.executable) | mk_string(" ", [](auto &s) { return s.data(); })) << "\n").flush();
  }

  if (int code = llvm::sys::ExecuteAndWait(opts.executable, args); code != 0)
    return "Non-zero exit code for task: " + (args ^ mk_string(" ", [](auto &s) { return s.str(); }));

  auto BufferOrErr = llvm::MemoryBuffer::getFile(outputPath);

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

struct HostMirrorResult {
  std::optional<std::string> error;
  std::string bitcode;
};

static HostMirrorResult compileHostMirror(const polyfront::Options &opts,  //
                                          const polyast::Program &program, //
                                          const std::string &passSpec) {
  auto result = compileProgram(opts, program, compiletime::Target::Object_LLVM_HOST, "native", {"--host-mirroring", "--passes", passSpec});
  if (auto *err = std::get_if<std::string>(&result)) return {*err, {}};
  const auto &r = std::get<polyast::CompileResult>(result);
  if (r.binary) return {std::nullopt, std::string(reinterpret_cast<const char *>(r.binary->data()), r.binary->size())};
  return {std::nullopt, {}};
}

struct ManagedHostMirror {
  std::string bitcode;
  std::string mirrorId;
  std::optional<std::string> error;
};

static ManagedHostMirror compileManagedHostMirror(const Options &opts, const polyast::Program &program, const runtime::PlatformKind kind,
                                                  const std::string &moduleId) {
  using namespace aspartame;
  const auto managed = [](auto &t, auto &) { return runtime::targetPlatformKind(t) == runtime::PlatformKind::Managed; };
  if (kind != runtime::PlatformKind::Managed || !(opts.targets ^ exists(managed))) return {};
  // physical backends (cuda/hsa Compiletime mirror) consume this host prelude; binding-slot targets
  // marshal through the runtime arena and ignore it
  const auto id = mirror::idFor(moduleId);
  if (auto res = compileHostMirror(opts, program, passes::hostMirror(id)); res.error)
    return {.bitcode = {}, .mirrorId = {}, .error = res.error};
  else return {.bitcode = std::move(res.bitcode), .mirrorId = id, .error = std::nullopt};
}

} // namespace polyregion::polyfront

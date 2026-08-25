#include "polyfront/polyc_client.hpp"

#include <cstdlib>
#include <fstream>
#include <optional>
#include <string_view>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include "fmt/format.h"

#include "polyast_codec.h"

namespace polyregion::polyfront::package {
namespace {

constexpr auto PackageExecutableEnv = "POLYC_PACKAGE_EXECUTABLE";

std::string clientExecutable(const std::string &explicitPath) {
  if (!explicitPath.empty()) return explicitPath;
  if (const char *path = std::getenv(PackageExecutableEnv); path && *path) return path;
#ifdef POLYC_DEV_EXECUTABLE
  if (llvm::sys::fs::exists(POLYC_DEV_EXECUTABLE)) return POLYC_DEV_EXECUTABLE;
#endif
  if (const auto path = llvm::sys::findProgramByName("polyc")) return *path;
  return {};
}

std::vector<std::string> diagnosticLines(std::string diagnostic) {
  std::vector<std::string> errors;
  for (size_t offset = 0; offset <= diagnostic.size();) {
    const auto end = diagnostic.find('\n', offset);
    const auto line = diagnostic.substr(offset, end - offset);
    if (!line.empty()) errors.emplace_back(line);
    if (end == std::string::npos) break;
    offset = end + 1;
  }
  return errors;
}

bool writeFile(const llvm::StringRef path, const std::vector<uint8_t> &bytes, std::string &error) {
  std::ofstream stream(path.str(), std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (stream) return true;
  error = "cannot write polyc package request to " + path.str();
  return false;
}

template <typename T, typename Decode>
Checked<T> invoke(const std::vector<uint8_t> &request, const std::string &explicitExecutable, const std::vector<std::string> &operationArgs,
                  Decode decode) {
  const auto executable = clientExecutable(explicitExecutable);
  if (executable.empty()) return {{}, {"cannot locate polyc package compiler"}};

  llvm::SmallString<128> inputPath, outputPath, errorPath;
  if (const auto ec = llvm::sys::fs::createTemporaryFile("polyregion-package-in", "msgpack", inputPath))
    return {{}, {"cannot create polyc package input: " + ec.message()}};
  if (const auto ec = llvm::sys::fs::createTemporaryFile("polyregion-package-out", "msgpack", outputPath)) {
    llvm::sys::fs::remove(inputPath);
    return {{}, {"cannot create polyc package output: " + ec.message()}};
  }
  if (const auto ec = llvm::sys::fs::createTemporaryFile("polyregion-package-error", "txt", errorPath)) {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
    return {{}, {"cannot create polyc package diagnostic: " + ec.message()}};
  }
  const llvm::scope_exit cleanup([&] {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
    llvm::sys::fs::remove(errorPath);
  });
  std::string writeError;
  if (!writeFile(inputPath, request, writeError)) return {{}, {std::move(writeError)}};

  if (operationArgs.empty()) return {{}, {"polyc package operation is empty"}};
  std::vector<std::string> ownedArgs{"", "--polyc", "package", operationArgs.front()};
  ownedArgs.emplace_back(inputPath.str());
  ownedArgs.insert(ownedArgs.end(), operationArgs.begin() + 1, operationArgs.end());
  ownedArgs.emplace_back("--out=" + outputPath.str().str());
  std::vector<llvm::StringRef> args;
  args.reserve(ownedArgs.size());
  for (const auto &arg : ownedArgs)
    args.emplace_back(arg);
  std::string executionError;
  const int code =
      llvm::sys::ExecuteAndWait(executable, args, std::nullopt, {std::nullopt, std::nullopt, errorPath.str()}, 0, 0, &executionError);
  if (code != 0) {
    std::string diagnostic;
    if (const auto buffer = llvm::MemoryBuffer::getFile(errorPath)) diagnostic = (*buffer)->getBuffer().str();
    auto errors = diagnosticLines(std::move(diagnostic));
    if (!executionError.empty()) errors.emplace_back(std::move(executionError));
    if (errors.empty()) errors.emplace_back(fmt::format("polyc package compiler exited with code {}", code));
    return {{}, std::move(errors)};
  }
  const auto buffer = llvm::MemoryBuffer::getFile(outputPath);
  if (!buffer) return {{}, {"cannot read polyc package result"}};
  if ((*buffer)->getBufferSize() == 0) return {{}, {"polyc package compiler returned an empty successful result"}};
  const auto *begin = reinterpret_cast<const uint8_t *>((*buffer)->getBufferStart());
  const auto *end = begin + (*buffer)->getBufferSize();
  try {
    return {{decode(begin, end)}, {}};
  } catch (const std::exception &error) {
    return {{}, {std::string("cannot decode polyc package result: ") + error.what()}};
  }
}

} // namespace

Checked<polyast::PackageSymCompileResult>
PolycClient::compileSym(const polyast::PackageSymRequest &request, const std::string &executable, const compiletime::Target hostTarget,
                        const std::string &hostArch, const std::vector<std::pair<compiletime::Target, std::string>> &deviceTargets,
                        const std::optional<int> stackDepth) {
  const auto host = compiletime::TargetSpec::findByCodegen(hostTarget);
  if (!host) return {{}, {fmt::format("unknown package host target {}", static_cast<int>(hostTarget))}};
  std::vector<std::string> args{"compile-sym", "--target=" + std::string(host->canonical), "--arch=" + hostArch};
  for (const auto &[target, arch] : deviceTargets) {
    const auto device = compiletime::TargetSpec::findByCodegen(target);
    if (!device) return {{}, {fmt::format("unknown package device target {}", static_cast<int>(target))}};
    args.emplace_back("--device=" + std::string(device->canonical) + "@" + arch);
  }
  if (stackDepth) args.emplace_back("--stack-depth=" + std::to_string(*stackDepth));
  return invoke<polyast::PackageSymCompileResult>(
      polyast::packagesymrequest_to_msgpack(request), executable, args,
      [](const uint8_t *begin, const uint8_t *end) { return polyast::packagesymcompileresult_from_msgpack(begin, end); });
}

} // namespace polyregion::polyfront::package

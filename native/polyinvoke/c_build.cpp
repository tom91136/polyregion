#include <array>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
  #include <dlfcn.h>
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/cache.hpp"
#include "polyregion/env_keys.h"

#include "c_build.h"

using namespace aspartame;

namespace polyregion::invoke::object {

namespace {

constexpr auto CompilerBasename = "clang";
constexpr size_t MaxDistDepth = 4;

#if defined(__APPLE__)
constexpr auto SharedSuffix = ".dylib";
#else
constexpr auto SharedSuffix = ".so";
#endif

[[maybe_unused]] void anchor() {}

std::string moduleContaining(const void *address) {
#if defined(__linux__) || defined(__APPLE__)
  Dl_info info;
  if (dladdr(address, &info) && info.dli_fname) {
    llvm::SmallString<256> resolved;
    if (!llvm::sys::fs::real_path(info.dli_fname, resolved)) return resolved.str().str();
  }
#endif
  return {};
}

std::string compilerUnder(llvm::StringRef root) {
  llvm::SmallString<256> candidate(root);
  llvm::sys::path::append(candidate, "bin", CompilerBasename);
  return llvm::sys::fs::exists(candidate) ? candidate.str().str() : std::string{};
}

std::string compilerAbove(llvm::StringRef file) {
  if (file.empty()) return {};
  auto dir = llvm::sys::path::parent_path(file).str();
  for (size_t depth = 0; depth < MaxDistDepth && !dir.empty(); ++depth) {
    if (auto found = compilerUnder(dir); !found.empty()) return found;
    dir = llvm::sys::path::parent_path(dir).str();
  }
  return {};
}

std::string compilerFromBitcodeDir() {
  const char *dir = std::getenv(polyregion::env::PolyregionBitcodeDir);
  if (!dir || !*dir) return {};
  return compilerUnder(llvm::sys::path::parent_path(llvm::StringRef(dir).rtrim("/\\")));
}

std::string readFile(const std::string &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  return buffer ? (*buffer)->getBuffer().str() : std::string{};
}

std::string compilerStamp(const std::string &path) {
  llvm::sys::fs::file_status status;
  if (llvm::sys::fs::status(path, status)) return {};
  return fmt::format("{}:{}", status.getSize(), status.getLastModificationTime().time_since_epoch().count());
}

std::string builtinsFor(const std::string &compiler, const std::string &triple) {
  llvm::SmallString<256> archive(llvm::sys::path::parent_path(llvm::sys::path::parent_path(compiler)));
  llvm::sys::path::append(archive, "lib", "clang", std::to_string(LLVM_VERSION_MAJOR), "lib");
  llvm::sys::path::append(archive, triple, "libclang_rt.builtins.a");
  return llvm::sys::fs::exists(archive) ? archive.str().str() : std::string{};
}

} // namespace

CCompiler locateCCompiler() {
#ifdef _WIN32
  return {{}, "building C11 images is not supported on Windows"};
#else
  if (const char *explicitPath = std::getenv(polyregion::env::PolyinvokeCCompiler); explicitPath && *explicitPath)
    return llvm::sys::fs::exists(explicitPath)
               ? CCompiler{explicitPath, {}}
               : CCompiler{{}, fmt::format("{}={} does not exist", polyregion::env::PolyinvokeCCompiler, explicitPath)};
  const std::array<std::string, 3> found = {
      compilerFromBitcodeDir(),
      compilerAbove(moduleContaining(reinterpret_cast<const void *>(&anchor))),
      compilerAbove(llvm::sys::fs::getMainExecutable(nullptr, reinterpret_cast<void *>(&anchor))),
  };
  if (const auto first = found ^ find([](const auto &c) { return !c.empty(); })) return {*first, {}};
  return {{},
          fmt::format("no dist {} found beside the runtime; set {} or {}", CompilerBasename, polyregion::env::PolyinvokeCCompiler,
                      polyregion::env::PolyregionBitcodeDir)};
#endif
}

std::string buildSharedObject(const CCompiler &compiler, const std::string &source) {
  namespace fs = llvm::sys::fs;
  const auto triple = llvm::sys::getProcessTriple();
  const auto builtins = builtinsFor(compiler.path, triple);
  std::vector<std::string> flags = {"--target=" + triple, "-shared", "-fPIC", "-O2", "-w", "-nostdlib", "-fuse-ld=lld"};
#ifdef __APPLE__
  flags.emplace_back("-Wl,-undefined,dynamic_lookup");
  flags.emplace_back("-lSystem");
#endif

  const auto cached =
      cache::path("c11", {source, compiler.path, compilerStamp(compiler.path), flags ^ mk_string(" "), builtins}, SharedSuffix);
  if (const auto hit = cache::read(cached); !hit.empty()) return {hit.begin(), hit.end()};

  llvm::SmallString<128> dir;
  if (const auto ec = fs::createUniqueDirectory("polyinvoke-c11", dir))
    throw std::runtime_error(fmt::format("cannot create a build directory: {}", ec.message()));
  const auto cleanup = [&] { fs::remove_directories(dir); };
  const auto inDir = [&](const std::string &name) {
    llvm::SmallString<160> file(dir);
    llvm::sys::path::append(file, name);
    return file.str().str();
  };
  const auto sourcePath = inDir("module.c"), outputPath = inDir(std::string("module") + SharedSuffix), logPath = inDir("compiler.log");
  try {
    {
      std::error_code ec;
      llvm::raw_fd_ostream out(sourcePath, ec);
      if (ec) throw std::runtime_error(fmt::format("cannot write {}: {}", sourcePath, ec.message()));
      out << source;
    }

    std::vector<std::string> args = {compiler.path};
    args.insert(args.end(), flags.begin(), flags.end());
    for (const auto &arg : {std::string("-xc"), sourcePath, std::string("-o"), outputPath})
      args.push_back(arg);
    if (!builtins.empty()) {
      args.emplace_back("-xnone");
      args.push_back(builtins);
    }
    const auto argRefs = args ^ map([](const auto &a) { return llvm::StringRef(a); });
    const std::optional<llvm::StringRef> redirects[] = {std::nullopt, llvm::StringRef(logPath), llvm::StringRef(logPath)};
    std::string execError;
    const auto status = llvm::sys::ExecuteAndWait(compiler.path, argRefs, std::nullopt, redirects, 0, 0, &execError);
    if (status != 0)
      throw std::runtime_error(fmt::format("`{}` exited with {}{}\n{}", args ^ mk_string(" "), status,
                                           execError.empty() ? "" : fmt::format(" ({})", execError), readFile(logPath)));
    auto image = readFile(outputPath);
    if (image.empty()) throw std::runtime_error(fmt::format("`{}` produced no output\n{}", compiler.path, readFile(logPath)));
    cache::write(cached, reinterpret_cast<const uint8_t *>(image.data()), image.size());
    cleanup();
    return image;
  } catch (...) {
    cleanup();
    throw;
  }
}

} // namespace polyregion::invoke::object

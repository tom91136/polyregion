#pragma once

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"

#include "polyregion/env_keys.h"

#include "ast.h"
#include "polyast_codec.h"

namespace polyregion::polyfront::package {

using namespace aspartame;

template <typename T> struct Checked {
  std::optional<T> value;
  std::vector<std::string> errors;
  explicit operator bool() const { return value.has_value(); }
};

inline std::string symbol(const polyast::Sym &sym) { return sym.fqn ^ mk_string("."); }

inline constexpr auto PackagePathEnv = env::PolyfrontLibraryPath;
inline constexpr auto PackageName = "lib.polyast";

inline std::vector<std::string> splitPackageRoots(const std::string &value) {
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  const auto roots = value ^ split(separator);
  return roots ^ filter([](const auto &root) { return !root.empty(); });
}

inline std::vector<std::string> packageRoots() {
  const auto *value = std::getenv(PackagePathEnv);
  return value && *value ? splitPackageRoots(value) : std::vector<std::string>{};
}

inline bool safePathComponent(const std::string &value) {
  if (value.empty() || value == "." || value == ".." || value.back() == '.' || value.back() == ' ') return false;
  if (!(value ^ forall([](char x) {
          const auto byte = static_cast<unsigned char>(x);
          return byte >= 32 && byte != 127 && x != '/' && x != '\\' && x != '<' && x != '>' && x != ':' && x != '"' && x != '|' && x != '?'
                 && x != '*';
        })))
    return false;
  const auto basePrefix = value ^ take_while([](char x) { return x != '.'; });
  const auto base = basePrefix ^ to_upper();
  if (std::vector<std::string>{"CON", "PRN", "AUX", "NUL"} ^ contains(base)) return false;
  return !(base.size() == 4 && ((base ^ starts_with("COM")) || (base ^ starts_with("LPT"))) && base.back() >= '1' && base.back() <= '9');
}

inline Checked<polyast::Package> loadPackageFile(const std::string &path) {
  Checked<polyast::Package> out;
  auto buffer = llvm::MemoryBuffer::getFile(path);
#ifdef _WIN32
  for (size_t attempt = 1; !buffer && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    buffer = llvm::MemoryBuffer::getFile(path);
  }
#endif
  if (!buffer) {
    out.errors.emplace_back("cannot read package `" + path + "`");
    return out;
  }
  const auto bytes = (*buffer)->getBuffer();
  auto decoded = polyast::decodePackage(reinterpret_cast<const uint8_t *>(bytes.begin()), reinterpret_cast<const uint8_t *>(bytes.end()));
  if (const auto *error = std::get_if<std::string>(&decoded)) {
    out.errors.emplace_back("cannot decode package: " + *error);
    return out;
  }
  out.value = std::get<polyast::Package>(std::move(decoded));
  return out;
}

inline Checked<polyast::Package> loadPackage(const std::string &packageName, const std::vector<std::string> &roots = packageRoots()) {
  Checked<polyast::Package> out;
  if (!safePathComponent(packageName)) {
    out.errors.emplace_back("invalid package identity `" + packageName + "`");
    return out;
  }
  const auto find = [&] {
    return roots ^ collect([&](const auto &root) -> std::optional<std::string> {
             llvm::SmallString<256> path(root);
             llvm::sys::path::append(path, packageName, PackageName);
             return llvm::sys::fs::is_regular_file(path) ? std::optional(path.str().str()) : std::nullopt;
           });
  };
  auto found = find();
#ifdef _WIN32
  for (size_t attempt = 1; found.empty() && attempt < 50; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    found = find();
  }
#endif
  if (found.empty()) {
    out.errors.emplace_back("no package is available for library `" + packageName + "`");
    return out;
  }
  if (found.size() != 1) {
    out.errors.emplace_back("library `" + packageName + "` is ambiguous across " + std::to_string(found.size()) + " package roots");
    return out;
  }
  auto package = loadPackageFile(found.front());
  if (!package) return package;
  if (symbol(package.value->interface.name) != packageName) {
    out.errors.emplace_back("package identity differs: expected `" + packageName + "`, got `" + symbol(package.value->interface.name)
                            + "`");
    return out;
  }
  return package;
}

} // namespace polyregion::polyfront::package

#pragma once

#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"

namespace polyregion::polyfront::package {

using namespace aspartame;

inline Checked<polyast::Package> publishPackage(const polyast::Package &package, const std::string &root) {
  Checked<polyast::Package> out;
  const auto packageName = symbol(package.interface.name);
  if (!safePathComponent(packageName)) {
    out.errors.emplace_back("invalid package identity `" + packageName + "`");
    return out;
  }

  llvm::SmallString<256> directory(root);
  llvm::sys::path::append(directory, packageName);
  if (llvm::sys::fs::exists(directory) && llvm::sys::fs::is_symlink_file(directory)) {
    out.errors.emplace_back("package emission directory cannot be a symbolic link");
    return out;
  }
  if (const auto error = llvm::sys::fs::create_directories(directory)) {
    out.errors.emplace_back("cannot create package directory: " + error.message());
    return out;
  }

  llvm::SmallString<256> temporary, temporaryModel(directory);
  llvm::sys::path::append(temporaryModel, ".lib-%%%%%%.polyast");
  int temporaryFd = -1;
  if (const auto error = llvm::sys::fs::createUniqueFile(temporaryModel, temporaryFd, temporary)) {
    out.errors.emplace_back("cannot create temporary package: " + error.message());
    return out;
  }
  const auto fail = [&](std::string error) {
    out.errors.emplace_back(std::move(error));
    if (const auto cleanupError = llvm::sys::fs::remove(temporary, false))
      out.errors.emplace_back("cannot remove temporary package: " + cleanupError.message());
    return out;
  };

  const auto bytes = polyast::package_to_msgpack(package);
  llvm::raw_fd_ostream stream(temporaryFd, true);
  stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  stream.close();
  if (stream.has_error()) return fail("cannot write temporary package: " + stream.error().message());
  auto staged = loadPackageFile(temporary.str().str());
  if (!staged) return fail("cannot verify temporary package: " + (staged.errors ^ mk_string("; ")));
  if (*staged.value != package) return fail("temporary package differs after serialisation");
  llvm::SmallString<256> target(directory);
  llvm::sys::path::append(target, PackageName);
  if (const auto error = llvm::sys::fs::rename(temporary, target)) return fail("cannot emit package: " + error.message());
  out.value = std::move(*staged.value);
  return out;
}

} // namespace polyregion::polyfront::package

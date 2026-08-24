#include "polyfront/package_emit.hpp"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyregion/env.h"
#include "polyregion/env_keys.h"

#include "ast.h"
#include "polyast_codec.h"

namespace {

using namespace aspartame;
using namespace polyregion;

void packageEmitAnchor() {}

std::optional<std::vector<uint8_t>> read(const std::string &path) {
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    llvm::errs() << "cannot read `" << path << "`\n";
    return {};
  }
  const auto bytes = (*buffer)->getBuffer();
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

} // namespace

int main(int argc, char **argv) {
  if (!std::getenv(polyregion::env::PolypassPlugins)) {
    const auto executable = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&packageEmitAnchor));
    llvm::SmallString<256> executableDir(executable);
    llvm::sys::path::remove_filename(executableDir);
    llvm::SmallString<256> colocated(executableDir);
    llvm::sys::path::append(colocated, POLYPASS_DSO_BASENAME);
    llvm::SmallString<256> installed(executableDir);
    llvm::sys::path::append(installed, "..", "lib", POLYPASS_DSO_BASENAME);
    const std::vector<std::string> candidates = {colocated.str().str(), installed.str().str(), POLYPASS_DSO_DEV_PATH};
    for (const auto &candidate : candidates)
      if (llvm::sys::fs::exists(candidate)) {
        polyregion::env::put(polyregion::env::PolypassPlugins, candidate.c_str(), false);
        break;
      }
  }
  std::set<std::string> capabilities;
  int first = 1;
  while (first < argc && std::string_view(argv[first]).starts_with("--capability=")) {
    capabilities.emplace(std::string_view(argv[first]).substr(std::string_view("--capability=").size()));
    ++first;
  }
  if (argc - first < 3) {
    llvm::errs() << "usage: polypackage-emit [--capability=<name>]... <interface.polyast> <output-root> "
                    "<program.polyast>...\n";
    return 2;
  }
  try {
    const auto interfaceBytes = read(argv[first]);
    if (!interfaceBytes) return 3;
    const std::vector<std::string> paths(argv + first + 2, argv + argc);
    const auto programs = traverse(paths, [&](const auto &path) -> std::optional<polyast::Program> {
      const auto bytes = read(path);
      return bytes ^ map([](const auto &value) { return polyast::hashed_program_from_msgpack(value); });
    });
    if (!programs) return 3;
    const auto interface = polyast::interface_from_msgpack(*interfaceBytes);
    const auto request =
        polyast::PackageLinkRequest(interface, *programs, std::vector<std::string>(capabilities.begin(), capabilities.end()));
    const auto emitted = polyfront::package::linkAndPublish(request, argv[first + 1]);
    if (!emitted) {
      llvm::errs() << (emitted.errors | mk_string("\n")) << "\n";
      return 4;
    }
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << error.what() << "\n";
    return 5;
  }
}

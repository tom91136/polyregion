#include "js_cache.h"

#include <cstdlib>
#include <mutex>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "fmt/format.h"

#include "polyregion/env_keys.h"
#include "polyregion/io.hpp"

namespace polyregion::polypass {

namespace {

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

String cacheRoot() {
  if (const char *e = std::getenv(polyregion::env::PolyregionCacheDir); e && *e) return e;
  llvm::SmallString<256> dir;
  if (!path::cache_directory(dir)) path::system_temp_directory(/*ErasedOnReboot=*/true, dir);
  path::append(dir, "polyregion", "js");
  return dir.str().str();
}

String contentKey(std::string_view s) {
  const auto h = llvm::xxh3_64bits(llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(s.data()), s.size()));
  return fmt::format("{:016x}", h);
}

String cachePath(std::string_view engineTag, std::string_view source) {
  llvm::SmallString<256> p(cacheRoot());
  path::append(p, engineTag, contentKey(source) + ".bc");
  return p.str().str();
}

} // namespace

String hostArchTag() { return llvm::Triple(llvm::sys::getProcessTriple()).getArchName().str(); }

std::optional<Vector<uint8_t>> readJsCache(std::string_view engineTag, std::string_view source) {
  const auto p = cachePath(engineTag, source);
  if (!fs::exists(p)) return std::nullopt;
  auto buf = polyregion::read_struct<uint8_t>(p);
  if (buf.empty()) return std::nullopt;
  return buf;
}

void writeJsCache(std::string_view engineTag, std::string_view source, const uint8_t *data, size_t size) {
  if (size == 0) return;
  const auto finalPath = cachePath(engineTag, source);
  static std::once_flag warnOnce;
  const auto warnOnce_ = [&](std::string_view why, std::string detail = {}) {
    std::call_once(warnOnce, [&] {
      fmt::print(stderr, "polypass: js bytecode cache disabled ({}): {}{}{}\n", why, finalPath, detail.empty() ? "" : ": ", detail);
    });
  };
  if (auto ec = fs::create_directories(path::parent_path(finalPath)); ec) {
    warnOnce_("create_directories", ec.message());
    return;
  }

  // XXX TempFile::keep does the platform-correct atomic rename and registers RemoveFileOnSignal so SIGINT mid-write does not leak staging
  // files.
  llvm::SmallString<256> model(finalPath);
  model.append(".tmp.%%%%%%");
  auto tmp = fs::TempFile::create(model);
  if (!tmp) {
    warnOnce_("TempFile::create", llvm::toString(tmp.takeError()));
    return;
  }
  {
    llvm::raw_fd_ostream out(tmp->FD, /*shouldClose=*/false);
    out.write(reinterpret_cast<const char *>(data), size);
    out.flush();
    if (out.has_error()) {
      warnOnce_("write", "raw_fd_ostream error");
      llvm::consumeError(tmp->discard());
      return;
    }
  }
  if (auto err = tmp->keep(finalPath)) warnOnce_("TempFile::keep", llvm::toString(std::move(err)));
}

} // namespace polyregion::polypass

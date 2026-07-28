#include "js_cache.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "polyregion/cache.hpp"

namespace polyregion::polypass {

namespace {
String cachePath(const std::string_view engineTag, const std::string_view source) { return cache::path("js", {engineTag, source}, ".bc"); }
} // namespace

String hostArchTag() { return llvm::Triple(llvm::sys::getProcessTriple()).getArchName().str(); }

std::optional<Vector<uint8_t>> readJsCache(const std::string_view engineTag, const std::string_view source) {
  auto buf = cache::read(cachePath(engineTag, source));
  if (buf.empty()) return std::nullopt;
  return buf;
}

void writeJsCache(const std::string_view engineTag, const std::string_view source, const uint8_t *data, const size_t size) {
  cache::write(cachePath(engineTag, source), data, size);
}

} // namespace polyregion::polypass

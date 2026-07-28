#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include "fmt/format.h"

#include "polyregion/env_keys.h"

namespace polyregion::cache {

namespace detail {
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

constexpr std::array<char, 4> Magic{'P', 'O', 'L', 'Y'};
constexpr uint32_t Version = 1;

struct Header {
  std::array<char, 4> magic;
  uint32_t version;
  uint64_t digest;
};
static_assert(sizeof(Header) == 16);

inline uint64_t digestOf(const uint8_t *data, const size_t size) { return llvm::xxh3_64bits(llvm::ArrayRef(data, size)); }
inline uint64_t digestOf(const std::string_view s) { return digestOf(reinterpret_cast<const uint8_t *>(s.data()), s.size()); }

inline std::string root() {
  if (const char *e = std::getenv(env::PolyregionCacheDir)) {
    const std::string_view v(e);
    return v.empty() || v == "0" || v == "off" ? std::string{} : std::string(v);
  }
  llvm::SmallString<256> dir;
  if (!path::cache_directory(dir)) path::system_temp_directory(/*ErasedOnReboot=*/true, dir);
  path::append(dir, "polyregion");
  return dir.str().str();
}
} // namespace detail

inline std::string path(const std::string_view domain, const std::initializer_list<std::string_view> parts, const std::string_view ext) {
  const auto dir = detail::root();
  if (dir.empty()) return {};
  std::string key;
  key.reserve(parts.size() * 2 * sizeof(uint64_t));
  for (const auto part : parts) {
    const uint64_t framed[]{static_cast<uint64_t>(part.size()), detail::digestOf(part)};
    key.append(reinterpret_cast<const char *>(framed), sizeof(framed));
  }
  const auto h = llvm::xxh3_128bits(llvm::ArrayRef(reinterpret_cast<const uint8_t *>(key.data()), key.size()));
  llvm::SmallString<256> file(dir);
  detail::path::append(file, domain, fmt::format("{:016x}{:016x}{}", h.high64, h.low64, ext));
  return file.str().str();
}

inline std::vector<uint8_t> read(const std::string &p) {
  if (p.empty()) return {};
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto end = in.tellg();
  if (end <= static_cast<std::streamoff>(sizeof(detail::Header))) return {};
  detail::Header header = {};
  in.seekg(0);
  if (!in.read(reinterpret_cast<char *>(&header), sizeof(header))) return {};
  if (header.magic != detail::Magic || header.version != detail::Version) return {};
  std::vector<uint8_t> out(static_cast<size_t>(end) - sizeof(detail::Header));
  if (!in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()))) return {};
  if (detail::digestOf(out.data(), out.size()) != header.digest) return {};
  return out;
}

inline void write(const std::string &p, const uint8_t *data, const size_t size) {
  if (p.empty() || !data || size == 0) return;
  if (detail::fs::create_directories(detail::path::parent_path(p))) return;
  llvm::SmallString<256> model(p);
  model.append(".tmp-%%%%%%");
  auto tmp = detail::fs::TempFile::create(model);
  if (!tmp) return llvm::consumeError(tmp.takeError());
  const detail::Header header{detail::Magic, detail::Version, detail::digestOf(data, size)};
  {
    llvm::raw_fd_ostream out(tmp->FD, /*shouldClose=*/false);
    out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    out.write(reinterpret_cast<const char *>(data), size);
    out.flush();
    if (out.has_error()) return llvm::consumeError(tmp->discard());
  }
  if (auto err = tmp->keep(p)) llvm::consumeError(std::move(err));
}

inline void evict(const std::string &p) {
  if (!p.empty()) detail::fs::remove(p);
}

} // namespace polyregion::cache

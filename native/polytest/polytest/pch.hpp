#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/xxhash.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/io.hpp"

namespace polyregion::polytest::pch {

using namespace aspartame;

// a compile step can start from a PCH of what its command preloads plus `header`, the shared test header its source
// includes (empty when it has none). `command` is the step minus input, output and the per-test macros, so equal
// commands share one PCH; `macros` are the names of the macros that were left out
struct Spec {
  std::string header;
  std::vector<std::string> command;
  std::vector<std::string> macros;
};

struct Dep {
  std::string path;
  std::uint64_t size = 0;
  std::int64_t mtime = 0;
  bool operator==(const Dep &that) const { return path == that.path && size == that.size && mtime == that.mtime; }
};

struct Manifest {
  std::vector<std::string> key;
  std::string pch;
  std::vector<Dep> deps;
  std::vector<std::string> macroReads;
};

inline std::string digest(std::string_view s) {
  return fmt::format("{:016x}", llvm::xxh3_64bits(reinterpret_cast<const std::uint8_t *>(s.data()), s.size()));
}

inline std::optional<Dep> statDep(const std::string &path) {
  llvm::sys::fs::file_status status;
  if (llvm::sys::fs::status(path, status)) return {};
  return Dep{path, status.getSize(), status.getLastModificationTime().time_since_epoch().count()};
}

inline std::optional<std::string> headerPath(const std::string &source, const std::string &name) {
  std::ifstream in(source);
  for (std::string line; std::getline(in, line);) {
    const auto text = line ^ trim();
    if (!(text ^ starts_with("#include"))) continue;
    const auto target = text ^ drop(8) ^ trim();
    if (target.size() < 2 || target.front() != '"' || target.back() != '"') continue;
    const auto spelled = target.substr(1, target.size() - 2);
    if (spelled != name && !(spelled ^ ends_with("/" + name))) continue;
    llvm::SmallString<256> path(llvm::sys::path::parent_path(source));
    llvm::sys::path::append(path, spelled);
    return std::string(path);
  }
  return {};
}

inline bool isEnvBit(const std::string &token) { return token ^ contains('='); }

inline std::optional<Spec> specFor(const std::string &name, const std::string &source, const std::string &tmpl,
                                   const std::function<std::string(const std::string &)> &expand) {
  const auto raw = tmpl ^ words();
  Spec spec{name.empty() ? std::string{} : headerPath(source, name).value_or(std::string{}), {}, {}};
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto &token = raw[i];
    if (token == "{input}") continue;
    if (token == "-o" && i + 1 < raw.size() && raw[i + 1] == "{output}") {
      ++i;
      continue;
    }
    if ((token ^ contains_slice("{input}")) || (token ^ contains_slice("{output}"))) return {};
    if ((token ^ starts_with("-D")) || (token ^ starts_with("-U"))) {
      const auto macro = token.substr(2, token.find('=') == std::string::npos ? std::string::npos : token.find('=') - 2);
      if (macro.empty() || (macro ^ contains('{'))) return {};
      spec.macros.push_back(macro);
    } else
      spec.command ^= concat(expand(token) ^ words() ^ filter([](const auto &t) {
                               return !(t ^ starts_with("-l")) && !(t ^ starts_with("-L")) && !(t ^ starts_with("-Wl,"));
                             }));
  }
  const auto args = spec.command ^ drop_while(isEnvBit);
  const auto actions = std::set<std::string>{"-x", "-c", "-S", "-E", "-fsyntax-only", "--emit-ast", "-emit-ast", "--polyc", "-###"};
  const bool stdpar = args ^ exists([](const auto &t) { return t == "-fstdpar"; });
  const bool otherAction = args ^ exists([&](const auto &t) {
                             return actions.count(t) || (t ^ starts_with("-M")) || (t ^ starts_with("-fstdpar-emit-library"));
                           });
  if (args.empty() || !stdpar || otherAction) return {};
  return spec;
}

inline std::vector<std::string> parseDepfile(const std::string &text) {
  std::vector<std::string> tokens;
  std::string current;
  const auto flush = [&] {
    if (!current.empty()) tokens.push_back(std::move(current));
    current.clear();
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\\' && i + 1 < text.size() && (text[i + 1] == ' ' || text[i + 1] == '#')) current += text[++i];
    else if (c == '\\' && i + 1 < text.size() && (text[i + 1] == '\n' || text[i + 1] == '\r')) {
      flush();
      i += text[i + 1] == '\r' && i + 2 < text.size() && text[i + 2] == '\n' ? 2 : 1;
    } else if (std::isspace(static_cast<unsigned char>(c))) flush();
    else current += c;
  }
  flush();
  return tokens ^ drop(1);
}

inline std::set<std::string> identifiersOf(const std::vector<std::string> &paths) {
  std::set<std::string> identifiers;
  for (const auto &path : paths) {
    const auto text = polyregion::read_string(path);
    for (std::size_t i = 0; i < text.size();) {
      const auto head = static_cast<unsigned char>(text[i]);
      if (!std::isalpha(head) && head != '_') {
        ++i;
        continue;
      }
      std::size_t end = i + 1;
      while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_'))
        ++end;
      identifiers.emplace(text, i, end - i);
      i = end;
    }
  }
  return identifiers;
}

inline void writeManifest(const std::string &path, const Manifest &manifest, const std::set<std::string> &identifiers) {
  const auto tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    manifest.key | for_each([&](const auto &line) { out << "key " << line << '\n'; });
    out << "pch " << manifest.pch << '\n';
    manifest.deps | for_each([&](const auto &d) { out << "dep " << d.size << ' ' << d.mtime << ' ' << d.path << '\n'; });
    identifiers | for_each([&](const auto &id) { out << "ident " << id << '\n'; });
  }
  llvm::sys::fs::rename(tmp, path);
}

inline std::optional<Manifest> readManifest(const std::string &path, const std::vector<std::string> &macros) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  const std::set<std::string> wanted(macros.begin(), macros.end());
  Manifest manifest;
  for (std::string line; std::getline(in, line);) {
    if (line ^ starts_with("key ")) manifest.key.push_back(line.substr(4));
    else if (line ^ starts_with("pch ")) manifest.pch = line.substr(4);
    else if (line ^ starts_with("ident ")) {
      if (const auto id = line.substr(6); wanted.count(id)) manifest.macroReads.push_back(id);
    } else if (line ^ starts_with("dep ")) {
      const auto sizeEnd = line.find(' ', 4), mtimeEnd = line.find(' ', sizeEnd + 1);
      Dep dep;
      if (sizeEnd == std::string::npos || mtimeEnd == std::string::npos
          || llvm::StringRef(line).slice(4, sizeEnd).getAsInteger(10, dep.size)
          || llvm::StringRef(line).slice(sizeEnd + 1, mtimeEnd).getAsInteger(10, dep.mtime))
        return {};
      dep.path = line.substr(mtimeEnd + 1);
      manifest.deps.push_back(std::move(dep));
    }
  }
  return manifest.pch.empty() ? std::nullopt : std::optional{std::move(manifest)};
}

inline bool fresh(const Manifest &manifest, const std::vector<std::string> &key) {
  return manifest.key == key && manifest.deps ^ forall([](const auto &d) { return statDep(d.path) == std::optional{d}; });
}

// serialises builders across processes; the descriptor closing releases the lock if the holder dies
struct FileLock {
  int fd = -1;
  explicit FileLock(const std::string &path) {
    if (llvm::sys::fs::openFileForReadWrite(path, fd, llvm::sys::fs::CD_OpenAlways, llvm::sys::fs::OF_None)) fd = -1;
    else if (llvm::sys::fs::lockFile(fd)) llvm::sys::Process::SafelyCloseFileDescriptor(fd), fd = -1;
  }
  FileLock(const FileLock &) = delete;
  FileLock &operator=(const FileLock &) = delete;
  ~FileLock() {
    if (fd < 0) return;
    llvm::sys::fs::unlockFile(fd);
    llvm::sys::Process::SafelyCloseFileDescriptor(fd);
  }
};

} // namespace polyregion::polytest::pch

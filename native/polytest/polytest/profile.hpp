#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <climits>

  #include <unistd.h>
  #ifndef _POSIX_HOST_NAME_MAX
    #define _POSIX_HOST_NAME_MAX 255
  #endif
#endif

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "polyregion/types.h"

namespace polyregion::polytest {

inline std::optional<std::string> hostname() {
#ifdef _WIN32
  DWORD n = 0;
  GetComputerNameExA(ComputerNameDnsHostname, nullptr, &n); // probes required size, returns false
  std::string s(n, '\0');
  if (!GetComputerNameExA(ComputerNameDnsHostname, s.data(), &n)) return {};
  s.resize(n);
  return s;
#else
  std::string s(_POSIX_HOST_NAME_MAX + 1, '\0');
  if (gethostname(s.data(), s.size()) != 0) return {};
  s.resize(std::strlen(s.c_str()));
  return s;
#endif
}

// `<backend>@<uarch>` test targets, in order: $<envKey> (verbatim, colon-split),
// $POLYREGION_TEST_PROFILE.env, <hostname>.env, default.env, {}. Profile files concatenate every
// `<envKey>=...` line; `#` and blank lines are skipped.
inline std::vector<std::string> loadTestTargets(const std::string &profileDir, const char *envKey = "POLYREGION_TEST_TARGETS") {
  auto split = [](const std::string &s, std::vector<std::string> &out) {
    for (size_t i = 0, j = 0; i <= s.size(); ++i)
      if (i == s.size() || s[i] == ':') {
        if (i > j) out.emplace_back(s.substr(j, i - j));
        j = i + 1;
      }
  };
  if (auto v = std::getenv(envKey)) {
    std::vector<std::string> xs;
    split(v, xs);
    return xs;
  }
  auto readKey = [&](llvm::StringRef file) -> std::optional<std::vector<std::string>> {
    if (!llvm::sys::fs::exists(file)) return {};
    std::ifstream is(file.str());
    std::vector<std::string> targets;
    bool any = false;
    for (std::string line; std::getline(is, line);) {
      size_t s = 0;
      while (s < line.size() && (line[s] == ' ' || line[s] == '\t'))
        ++s;
      if (s == line.size() || line[s] == '#') continue;
      // Tolerate the shell-sourcable forms: `export K=v`, `K+=v`, and a leading `:` on appends.
      if (line.compare(s, 7, "export ") == 0) s += 7;
      const auto eq = line.find('=', s);
      if (eq == std::string::npos) continue;
      auto keyEnd = eq;
      if (keyEnd > s && line[keyEnd - 1] == '+') --keyEnd;
      if (line.compare(s, keyEnd - s, envKey, std::strlen(envKey)) != 0) continue;
      any = true;
      auto valStart = eq + 1;
      if (valStart < line.size() && line[valStart] == ':') ++valStart;
      split(line.substr(valStart), targets);
    }
    if (!any) return {};
    return targets;
  };
  auto profilePath = [&](llvm::StringRef name) {
    llvm::SmallString<128> p(profileDir);
    llvm::sys::path::append(p, name);
    return p;
  };
  if (auto v = std::getenv("POLYREGION_TEST_PROFILE"))
    if (auto t = readKey(profilePath(std::string(v) + ".env"))) return *t;
  if (auto h = hostname()) {
    if (const auto dot = h->find('.'); dot != std::string::npos) h->resize(dot);
    if (auto t = readKey(profilePath(*h + ".env"))) return *t;
  }
  if (auto t = readKey(profilePath("default.env"))) return *t;
  return {};
}

// A resolved `<backend>@<uarch>` test target: a TargetSpec from the canonical registry plus the
// `uarch` portion (e.g. `sm_89`, `gfx1036`, `x86-64-v3`). The uarch may be empty for backends
// that don't take an architecture (e.g. `cl@`, `vulkan@`).
struct ResolvedTarget {
  compiletime::TargetSpec spec;
  std::string arch;
  std::string canonical() const { return std::string(spec.canonical) + "@" + arch; }
};

inline std::optional<ResolvedTarget> resolveTestTarget(std::string_view token) {
  const auto at = token.find('@');
  const auto backendName = token.substr(0, at);
  const auto arch = at == std::string_view::npos ? std::string_view{} : token.substr(at + 1);
  if (auto s = compiletime::TargetSpec::findByName(backendName)) return ResolvedTarget{*s, std::string(arch)};
  return std::nullopt;
}

inline std::vector<ResolvedTarget> resolveTestTargets(const std::string &profileDir, const char *envKey = "POLYREGION_TEST_TARGETS") {
  std::vector<ResolvedTarget> out;
  for (const auto &t : loadTestTargets(profileDir, envKey))
    if (auto r = resolveTestTarget(t)) out.emplace_back(*r);
  return out;
}

} // namespace polyregion::polytest

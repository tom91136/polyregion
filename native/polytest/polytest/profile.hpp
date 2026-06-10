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

#include "aspartame/all.hpp"

#include "polyregion/env_keys.h"
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

// `[export ]KEY[+]=VALUE` -> (KEY, VALUE); nullopt for comments, blanks, and non-assignments
inline std::optional<std::pair<std::string, std::string>> parseEnvLine(const std::string &raw) {
  using namespace aspartame;
  const auto line = raw ^ trim_leading();
  if (line.empty() || line[0] == '#') return {};
  const auto body = line ^ starts_with("export ") ? line.substr(7) : line;
  const auto eq = body.find('=');
  if (eq == std::string::npos) return {};
  const auto keyEnd = eq > 0 && body[eq - 1] == '+' ? eq - 1 : eq;
  return std::pair{body.substr(0, keyEnd), body.substr(eq + 1)};
}

inline std::vector<std::string> fileLines(const std::string &file) {
  std::ifstream is(file);
  std::vector<std::string> out;
  for (std::string line; std::getline(is, line);)
    out.push_back(line);
  return out;
}

// candidate profile files, in precedence order: $POLYREGION_TEST_PROFILE.env, <hostname>.env, default.env
inline std::vector<std::string> profileCandidates(const std::string &profileDir) {
  auto profilePath = [&](const std::string &name) {
    llvm::SmallString<128> p(profileDir);
    llvm::sys::path::append(p, name);
    return std::string(p);
  };
  std::vector<std::string> out;
  if (const auto v = std::getenv(polyregion::env::PolyregionTestProfile)) out.push_back(profilePath(std::string(v) + ".env"));
  if (auto h = hostname()) {
    if (const auto dot = h->find('.'); dot != std::string::npos) h->resize(dot);
    out.push_back(profilePath(*h + ".env"));
  }
  out.push_back(profilePath("default.env"));
  return out;
}

inline std::vector<std::string> loadTestTargets(const std::string &profileDir,
                                                const char *envKey = polyregion::env::PolyregionTestTargets) {
  using namespace aspartame;
  const auto splitTargets = [](const std::string &v) {
    return v ^ split(';') ^ collect([](auto &piece) -> std::optional<std::string> {
             auto t = piece ^ trim();
             return t.empty() ? std::nullopt : std::optional{t};
           });
  };
  if (const auto v = std::getenv(envKey)) return splitTargets(v);
  const auto readKey = [&](const std::string &file) -> std::optional<std::vector<std::string>> {
    if (!llvm::sys::fs::exists(file)) return {};
    const auto vals = fileLines(file) ^ collect(parseEnvLine) ^ collect([&](auto &k, auto &v) { //
                        return k == envKey ? std::optional{v} : std::nullopt;
                      });
    if (vals.empty()) return {};
    return vals ^ flat_map([&](auto &v) { return splitTargets(v ^ starts_with(":") ? v.substr(1) : v); });
  };
  return profileCandidates(profileDir) ^ collect_first(readKey) ^ get_or_else(std::vector<std::string>{});
}

inline const std::vector<std::string> &loadProfileEnv(const std::string &profileDir) {
  using namespace aspartame;
  static const std::vector<std::string> cached = [&profileDir] {
    return profileCandidates(profileDir)                            //
           ^ find([](auto &f) { return llvm::sys::fs::exists(f); }) //
           ^ map([](auto &file) {
               return fileLines(file) ^ collect(parseEnvLine) ^ collect([](auto &k, auto &v) -> std::optional<std::string> {
                        return k == polyregion::env::PolyregionTestTargets || k == polyregion::env::PolyinvokeTestTargets
                                   ? std::nullopt
                                   : std::optional{k + "=" + v};
                      });
             }) ^
           get_or_else(std::vector<std::string>{});
  }();
  return cached;
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

inline std::vector<ResolvedTarget> resolveTestTargets(const std::string &profileDir,
                                                      const char *envKey = polyregion::env::PolyregionTestTargets) {
  using namespace aspartame;
  return loadTestTargets(profileDir, envKey) ^ collect([](auto &t) { return resolveTestTarget(t); });
}

} // namespace polyregion::polytest

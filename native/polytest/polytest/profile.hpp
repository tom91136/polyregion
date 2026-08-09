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
#include "polyregion/host.h"
#include "polyregion/types.h"

namespace polyregion::polytest {

using namespace aspartame;

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
  const auto line = raw ^ trim_leading();
  if (line.empty() || line[0] == '#') return {};
  const auto body = line ^ starts_with("export ") ? line ^ drop(7) : line;
  const auto assignment = body ^ split_once('=');
  if (!assignment) return {};
  auto [key, value] = *assignment;
  if (!key.empty() && key.back() == '+') key ^= drop_right(1);
  return std::pair{std::move(key), std::move(value)};
}

inline std::vector<std::string> fileLines(const std::string &file) {
  std::ifstream is(file);
  return istream_lines(is) | to_vector();
}

// candidate profile files, most-specific first: <profile>.<os>.env, <profile>.env, <hostname>.<os>.env, <hostname>.env, default.env
inline std::vector<std::string> profileCandidates(const std::string &profileDir) {
  const auto path = [&](const std::string &name) {
    llvm::SmallString<128> p(profileDir);
    llvm::sys::path::append(p, name);
    return std::string(p);
  };

  std::vector<std::string> bases; // the profile name, then the short hostname
  if (const auto v = std::getenv(polyregion::env::PolyregionTestProfile)) bases ^= append(v);
  if (auto h = hostname()) bases ^= append(*h ^ take_while([](char c) { return c != '.'; }));
  return bases                                                                                                                    //
         ^ flat_map([&](const auto &b) { return std::vector{path(b + "." + std::string(hostOs()) + ".env"), path(b + ".env")}; }) //
         ^ append(path("default.env"));
}

inline std::vector<std::string> loadTestTargets(const std::string &profileDir,
                                                const char *envKey = polyregion::env::PolyregionTestTargets) {
  const auto splitTargets = [](const std::string &v) {
    return v ^ split(';') ^ collect([](const auto &piece) -> std::optional<std::string> {
             auto t = piece ^ trim();
             return t.empty() ? std::nullopt : std::optional{t};
           });
  };
  if (const auto v = std::getenv(envKey)) return splitTargets(v);
  const auto readKey = [&](const std::string &file) -> std::optional<std::vector<std::string>> {
    if (!llvm::sys::fs::exists(file)) return {};
    const auto vals = fileLines(file)                               //
                      | collect(parseEnvLine)                       //
                      | collect([&](const auto &k, const auto &v) { //
                          return k == envKey ? std::optional{v} : std::nullopt;
                        }) //
                      | to_vector();
    if (vals.empty()) return {};
    return vals ^ flat_map([&](const auto &v) { return splitTargets(v ^ starts_with(":") ? v ^ drop(1) : v); });
  };
  return profileCandidates(profileDir) ^ collect_first(readKey) ^ get_or_else(std::vector<std::string>{});
}

inline const std::vector<std::string> &loadProfileEnv(const std::string &profileDir) {
  static const std::vector<std::string> cached = [&profileDir] {
    return profileCandidates(profileDir)                                  //
           ^ find([](const auto &f) { return llvm::sys::fs::exists(f); }) //
           ^ map([](const auto &file) {
               return fileLines(file)         //
                      | collect(parseEnvLine) //
                      | collect([](const auto &k, const auto &v) -> std::optional<std::string> {
                          return k == polyregion::env::PolyregionTestTargets || k == polyregion::env::PolyinvokeTestTargets
                                     ? std::nullopt
                                     : std::optional{k + "=" + v};
                        }) //
                      | to_vector();
             }) //
           ^ get_or_else(std::vector<std::string>{});
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
  const auto pair = token ^ split_once('@');
  const auto backendName = pair ? pair->first : token;
  const auto arch = pair ? pair->second : std::string_view{};
  if (auto s = compiletime::TargetSpec::findByName(backendName)) return ResolvedTarget{*s, std::string(arch)};
  return std::nullopt;
}

inline std::vector<ResolvedTarget> resolveTestTargets(const std::string &profileDir,
                                                      const char *envKey = polyregion::env::PolyregionTestTargets) {
  return loadTestTargets(profileDir, envKey) ^ collect([](const auto &t) { return resolveTestTarget(t); });
}

} // namespace polyregion::polytest

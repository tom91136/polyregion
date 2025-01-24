#pragma once

#include <cstdlib>

#include "aspartame/all.hpp"

namespace polyregion::polyfront {

using namespace aspartame;

struct CliArgs {

  std::vector<const char *> data;
  std::unordered_set<size_t> deleted{};
  explicit CliArgs(const std::vector<const char *> &data) : data(data) {}

  bool has(const std::string &flag) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() |
           exists([&](auto &chars) { return chars == flag; });
  }

  bool has(const std::string &flag, size_t pos) const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) |
           exists([&](auto &chars, auto i) { return chars == flag && i == pos; });
  }

  std::optional<std::string> get(const std::string &flag) const {
    return data                                                        //
           | zip_with_index()                                          //
           | filter([&](auto v, auto i) { return !deleted.count(i); }) //
           | collect_first([&](auto &chars, auto i) -> std::optional<std::string> {
               if (const std::string arg = chars; arg == flag) {
                 if (i + 1 < data.size()) return data[i + 1];                                // -foo :: bar :: Nil
                 else return {};                                                             // -foo :: Nil
               } else if (arg ^ starts_with(flag + "=")) return arg.substr(flag.size() + 1); // -foo=bar
               return {};
             });
  }

  bool popBool(const std::string &flag) {
    auto oldSize = deleted.size();
    for (size_t i = 0; i < data.size(); ++i) {
      if (deleted.count(i)) continue;
      if (data[i] == flag) deleted.insert(i);
    }
    return oldSize != deleted.size();
  }

  std::optional<std::string> popValue(const std::string &flag) {
    std::optional<std::string> last;
    for (size_t i = 0; i < data.size(); ++i) {
      if (deleted.count(i)) continue;
      if (const std::string arg = data[i]; arg == flag && i + 1 < data.size()) { // -foo :: bar :: Nil
        deleted.insert(i);
        deleted.insert(i + 1);
        last = data[i + 1];
      } else if (arg ^ starts_with(flag + "=")) { // -foo=bar
        deleted.insert(i);
        last = arg.substr(flag.size() + 1);
      }
    }
    return last;
  }

  std::vector<const char *> remaining() const {
    return data | zip_with_index() | filter([&](auto v, auto i) { return !deleted.count(i); }) | keys() | to_vector();
  }
};

struct StdParOptions {

  enum class LinkKind : uint8_t { Static = 1, Dynamic, Disabled = 3 };

  static std::variant<std::string, LinkKind> parseLinkKind(const std::string &arg) {
    if (auto v = arg ^ to_lower(); v == "static") return LinkKind::Static;
    else if (v == "dynamic") return LinkKind::Dynamic;
    else if (v == "disabled") return LinkKind::Disabled;
    return "Unknown link kind `" + arg + "`";
  }

  bool verbose = false;
  bool noCompress = false;
  bool interposeMalloc = true;
  bool interposeAlloca = false;
  std::string targets{};
  LinkKind rt = LinkKind::Static;
  LinkKind jit = LinkKind::Disabled;

  static std::variant<std::vector<std::string>, std::optional<StdParOptions>> parse(CliArgs &args) {
    const std::string fStdParFlag = "-fstdpar";
    const std::string fStdParVerboseFlag = "-fstdpar-verbose";
    const std::string fStdParArchNoCompressFlag = "-fstdpar-no-compress";
    const std::string fStdParInterposeMallocFlag = "-fstdpar-interpose-malloc";
    const std::string fStdParInterposeAllocaFlag = "-fstdpar-interpose-alloca";
    const std::string fStdParArchFlag = "-fstdpar-arch";
    const std::string fStdParRtFlag = "-fstdpar-rt";
    const std::string fStdParJitFlag = "-fstdpar-jit";

    auto fStdPar = false, fStdParDependents = false;
    StdParOptions options;
    std::vector<std::string> errors;

    auto markError = [&](const std::string &prefix) { return [&](const std::string &x) { errors.push_back("\"" + prefix + "\": " + x); }; };

    if (args.popBool(fStdParFlag)) {
      fStdPar = true;
    }
    if (args.popBool(fStdParVerboseFlag)) {
      fStdParDependents = true;
      options.verbose = true;
    }
    if (args.popBool(fStdParArchNoCompressFlag)) {
      fStdParDependents = true;
      options.noCompress = true;
    }
    if (args.popBool(fStdParInterposeMallocFlag)) {
      fStdParDependents = true;
      options.interposeMalloc = true;
    }
    if (args.popBool(fStdParInterposeAllocaFlag)) {
      fStdParDependents = true;
      options.interposeAlloca = true;
    }
    if (auto arch = args.popValue(fStdParArchFlag)) {
      fStdParDependents = true;
      options.targets = *arch;
    }
    if (auto rt = args.popValue(fStdParRtFlag)) {
      fStdParDependents = true;
      parseLinkKind(*rt) ^ foreach_total(markError(fStdParRtFlag), [&](const LinkKind &x) { options.rt = x; });
    }
    if (auto jit = args.popValue(fStdParJitFlag)) {
      fStdParDependents = true;
      parseLinkKind(*jit) ^ foreach_total(markError(fStdParJitFlag), [&](const LinkKind &x) { options.jit = x; });
    }

    if (!fStdPar && fStdParDependents)
      errors.insert(errors.begin(), fStdParFlag + " not specified but StdPar dependent flags used, pleased add " + fStdParFlag);

    if (errors.empty()) return fStdPar ? std::optional{options} : std::nullopt;
    else return errors;
  }
};

inline std::vector<std::string> mkDelimitedEnvPaths(const char *env,  std::optional<std::string> leading, const char separator) {
  std::vector<std::string> xs;
  if (auto line = std::getenv(env); line) {
    for (auto &path : line ^ split(separator)) {
      if (leading) xs.push_back(*leading);
      xs.push_back(path);
    }
  }
  return xs;
}

inline std::string dynamicLibSuffix() {
#if defined(__linux__)
  return "so";
#elif defined(__APPLE__)
  return "dylib";
#elif defined(_WIN32)
  return "dll";
#else
  #error "Unsupported platform"
#endif
}

inline std::string staticLibSuffix() {
#if defined(__linux__)
  return "a";
#elif defined(__APPLE__)
  return "a";
#elif defined(_WIN32)
  return "lib";
#else
  #error "Unsupported platform"
#endif
}

} // namespace polyregion::polyfront

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "polyregion/types.h"

namespace polyregion::polystl {

struct StdParOptions {
  enum class LinkKind : uint8_t { Static = 1, Dynamic, Disabled = 3 };
  using Targets = std::vector<std::pair<polyregion::compiletime::Target, std::string>>;

  bool quiet = false;
  bool noCompress = false;
  bool interposeMalloc = true;
  bool interposeAlloca = false;
  Targets targets{};
  LinkKind rt = LinkKind::Static;
  LinkKind jit = LinkKind::Disabled;

  static std::variant<                                                   //
      std::vector<std::string>,                                          //
      std::pair<std::vector<const char *>, std::optional<StdParOptions>> //
      >
  stripAndParse(const std::vector<const char *> &args);

private:
  static std::variant<std::string, LinkKind> parseLinkKind(const std::string &arg);
  static std::variant<std::string, StdParOptions::Targets> parseTargets(const std::string &arg);
};

struct DriverContext {
  std::string executable;
  StdParOptions opts;
  bool cc1Verbose;
};

} // namespace polyregion::polystl
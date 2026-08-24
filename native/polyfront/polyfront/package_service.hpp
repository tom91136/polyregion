#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ast.h"

namespace polyregion::polyfront::package {

template <typename T> struct ServiceResult {
  std::optional<T> value;
  std::vector<std::string> errors;
  explicit operator bool() const { return value.has_value(); }
};

class PackageService {
public:
  [[nodiscard]] static ServiceResult<polyast::Package> linkPackage(const polyast::PackageLinkRequest &request);
  [[nodiscard]] static ServiceResult<polyast::PackageSymResolvedProgram> resolveSym(const polyast::PackageSymRequest &request);
};

} // namespace polyregion::polyfront::package

#pragma once

#include <string>

namespace polyregion::invoke::object {

struct CCompiler {
  std::string path, unavailable;
  explicit operator bool() const { return !path.empty(); }
};

CCompiler locateCCompiler();

std::string buildSharedObject(const CCompiler &compiler, const std::string &source);

} // namespace polyregion::invoke::object

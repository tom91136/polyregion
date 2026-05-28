#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace polyregion::invoke::amd {

// e_ident[8] (EI_ABIVERSION) on an amdgcn ELF encodes the code object version: 2=COv4, 3=COv5, 4=COv6.
inline std::optional<int> amdgcn_code_object_version(const std::string &image) {
  if (image.size() < 9) return std::nullopt;
  const auto *e = reinterpret_cast<const uint8_t *>(image.data());
  if (!(e[0] == 0x7f && e[1] == 'E' && e[2] == 'L' && e[3] == 'F')) return std::nullopt;
  return e[8] + 2;
}

} // namespace polyregion::invoke::amd

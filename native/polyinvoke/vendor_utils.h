#pragma once

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace polyregion::invoke {

inline std::string normaliseVendor(std::string_view vendor) {
  std::string lower(vendor.size(), {});
  for (size_t i = 0; i < vendor.size(); ++i)
    lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(vendor[i])));
  const auto has = [&](std::string_view s) { return lower.find(s) != std::string::npos; };
  if (has("nvidia")) return "nvidia";
  if (has("intel")) return "intel";
  if (has("advanced micro devices") || has("amd") || has("radv") || has("radeon")) return "amd";
  if (has("llvmpipe")) return "llvmpipe";
  if (has("mesa") || has("rusticl")) return "mesa";
  if (has("apple")) return "apple";
  if (has("arm")) return "arm";
  if (has("qualcomm")) return "qualcomm";
  return "unknown";
}

namespace amd {


inline std::optional<int> amdgcn_code_object_version(const std::string &image) {
  // XXX e_ident[8] (EI_ABIVERSION) on an amdgcn ELF encodes the code object version: 2=COv4, 3=COv5, 4=COv6.
  if (image.size() < 9) return std::nullopt;
  const auto *e = reinterpret_cast<const uint8_t *>(image.data());
  if (!(e[0] == 0x7f && e[1] == 'E' && e[2] == 'L' && e[3] == 'F')) return std::nullopt;
  return e[8] + 2;
}

} // namespace amd

} // namespace polyregion::invoke

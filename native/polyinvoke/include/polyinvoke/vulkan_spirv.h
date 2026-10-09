#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "aspartame/all.hpp"
#include "spirv/unified1/spirv.hpp"

namespace polyregion::invoke::vulkan {

inline constexpr std::string_view ArenaViewStartMarker = "polyregion.arena-view-start=";

inline std::optional<size_t> spirvArenaViewStart(const std::vector<uint32_t> &words) {
  using namespace aspartame;
  constexpr std::string_view prefix = "#av";
  std::unordered_map<uint32_t, uint32_t> viewIndex, binding;
  for (size_t i = 5; i < words.size();) {
    const auto count = words[i] >> 16, op = words[i] & 0xFFFF;
    if (count == 0 || i + count > words.size()) break;
    if (op == spv::OpModuleProcessed && count >= 2) {
      const auto *text = reinterpret_cast<const char *>(&words[i + 1]);
      const std::string_view processed(text, strnlen(text, (count - 1) * sizeof(uint32_t)));
      if (processed ^ starts_with(ArenaViewStartMarker))
        return size_t(std::stoul(std::string(processed.substr(ArenaViewStartMarker.size()))));
    } else if (op == spv::OpName && count >= 3) {
      const auto *name = reinterpret_cast<const char *>(&words[i + 2]);
      const std::string_view text(name, strnlen(name, (count - 2) * sizeof(uint32_t)));
      const auto digits = text.substr(std::min(prefix.size(), text.size()));
      if ((text ^ starts_with(prefix)) && !digits.empty() && (digits ^ forall([](const char c) { return c >= '0' && c <= '9'; })))
        viewIndex[words[i + 1]] = uint32_t(std::stoul(std::string(digits)));
    } else if (op == spv::OpDecorate && count >= 4 && words[i + 2] == spv::DecorationBinding) binding[words[i + 1]] = words[i + 3];
    i += count;
  }
  return viewIndex //
         | collect([&](const uint32_t id, const uint32_t index) {
             return binding ^ get_maybe(id) ^ filter([&](const uint32_t at) { return at >= index; })
                    ^ map([&](const uint32_t at) { return size_t(at - index); });
           }) //
         | head_maybe();
}

} // namespace polyregion::invoke::vulkan

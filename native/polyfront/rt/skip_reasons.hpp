#pragma once

#include <algorithm>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"

namespace polyregion::polyrt {

using namespace aspartame;

class SkipReasons {
  using ByDevice = std::vector<std::pair<std::string, std::vector<std::string>>>;

  mutable std::mutex mutex;
  ByDevice lacking;
  ByDevice unloadable;
  ByDevice unprepared;
  bool deviceless = false;

  static void note(ByDevice &into, const std::string &device, std::string item) {
    auto it = std::find_if(into.begin(), into.end(), [&](const auto &entry) { return entry.first == device; });
    if (it == into.end()) it = into.emplace(into.end(), device, std::vector<std::string>{});
    if (!(it->second ^ contains(item))) it->second.push_back(std::move(item));
  }

  static std::string render(const ByDevice &from, std::string_view verb, std::string_view suffix) {
    return from ^ mk_string("; ", [&](const auto &entry) {
             return entry.first + " " + std::string(verb) + " " + (entry.second ^ mk_string(", ")) + std::string(suffix);
           });
  }

public:
  void lacks(const std::string &device, std::string feature) {
    const std::lock_guard lock(mutex);
    note(lacking, device, std::move(feature));
  }

  void cannotLoad(const std::string &device, std::string format) {
    const std::lock_guard lock(mutex);
    note(unloadable, device, std::move(format));
  }

  void noDevice() {
    const std::lock_guard lock(mutex);
    deviceless = true;
  }

  void cannotPrepare(const std::string &device, std::string stage) {
    const std::lock_guard lock(mutex);
    note(unprepared, device, std::move(stage));
  }

  std::string line() const {
    const std::lock_guard lock(mutex);
    constexpr std::string_view prefix = "polyrt: no compatible target: ";
    std::vector<std::string> parts;
    if (!lacking.empty()) parts.push_back(render(lacking, "lacks", ""));
    if (!unprepared.empty()) parts.push_back(render(unprepared, "cannot prepare", ""));
    if (parts.empty() && !unloadable.empty()) parts.push_back(render(unloadable, "cannot load", " images"));
    if (parts.empty()) parts.push_back(deviceless ? "no device" : "no image offered");
    return std::string(prefix) + (parts ^ mk_string("; "));
  }
};

} // namespace polyregion::polyrt

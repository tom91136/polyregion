#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"

#include "polyregion/enums.h"

namespace polyregion::polyrt {

using namespace aspartame;

constexpr std::string_view outcomeName(const runtime::LoadOutcome outcome) {
  switch (outcome) {
    case runtime::LoadOutcome::Loaded: return "loaded";
    case runtime::LoadOutcome::Format: return "format";
    case runtime::LoadOutcome::Lacks: return "lacks";
    case runtime::LoadOutcome::Unprepared: return "unprepared";
    case runtime::LoadOutcome::NoDevice: return "no-device";
  }
  return "";
}

class SkipReasons {
public:
  struct Device {
    std::string platform, name, kind, format;
  };

  struct Attempt {
    std::string module, kind, format;
    std::vector<std::string> features;
    std::string outcome;
    std::vector<std::string> detail;
  };

private:
  mutable std::mutex mutex;
  std::optional<Device> device;
  std::vector<Attempt> attempts;

  std::vector<std::string> detailOf(const runtime::LoadOutcome outcome) const {
    std::vector<std::string> out;
    for (const auto &a : attempts)
      if (a.outcome == outcomeName(outcome))
        for (const auto &d : a.detail)
          if (!(out ^ contains(d))) out.push_back(d);
    return out;
  }

  static std::string quote(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
      if (c == '"' || c == '\\') out += '\\', out += c;
      else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
      else out += c;
    }
    return out + "\"";
  }

  static std::string array(const std::vector<std::string> &xs) {
    return "[" + (xs ^ mk_string(",", [](const auto &x) { return quote(x); })) + "]";
  }

public:
  void attempt(const std::optional<Device> &on, Attempt a) {
    const std::scoped_lock lock(mutex);
    if (on) device = on;
    attempts.push_back(std::move(a));
  }

  bool lacksCapability() const {
    const std::scoped_lock lock(mutex);
    return attempts ^ exists([](const auto &a) { return a.outcome == outcomeName(runtime::LoadOutcome::Lacks); });
  }

  std::string line() const {
    const std::scoped_lock lock(mutex);
    constexpr std::string_view prefix = "polyrt: no compatible target: ";
    const auto name = device ? device->name : std::string{};
    const auto lacks = detailOf(runtime::LoadOutcome::Lacks);
    const auto unprepared = detailOf(runtime::LoadOutcome::Unprepared);
    const auto unloadable = detailOf(runtime::LoadOutcome::Format);
    const bool deviceless = attempts ^ exists([](const auto &a) { return a.outcome == outcomeName(runtime::LoadOutcome::NoDevice); });
    std::vector<std::string> parts;
    if (!lacks.empty()) parts.push_back(name + " lacks " + (lacks ^ mk_string(", ")));
    if (!unprepared.empty()) parts.push_back(name + " cannot prepare " + (unprepared ^ mk_string(", ")));
    if (parts.empty() && !unloadable.empty()) parts.push_back(name + " cannot load " + (unloadable ^ mk_string(", ")) + " images");
    if (parts.empty()) parts.push_back(deviceless ? "no device" : "no image offered");
    return std::string(prefix) + (parts ^ mk_string("; "));
  }

  std::string record() const {
    const std::scoped_lock lock(mutex);
    const auto dev = device ? "{\"platform\":" + quote(device->platform) + ",\"name\":" + quote(device->name)
                                  + ",\"kind\":" + quote(device->kind) + ",\"format\":" + quote(device->format) + "}"
                            : std::string("null");
    const auto items = attempts ^ mk_string(",", [](const auto &a) {
                         return "{\"module\":" + quote(a.module) + ",\"kind\":" + quote(a.kind) + ",\"format\":" + quote(a.format)
                                + ",\"features\":" + array(a.features) + ",\"outcome\":" + quote(a.outcome)
                                + ",\"detail\":" + array(a.detail) + "}";
                       });
    return "polyrt-skip: {\"device\":" + dev + ",\"attempts\":[" + items + "]}";
  }
};

} // namespace polyregion::polyrt

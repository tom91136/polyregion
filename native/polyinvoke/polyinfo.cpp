#include <cstdio>
#include <optional>
#include <string>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"
#include "polyregion/types.h"

#include "fire.hpp"

using namespace aspartame;
using namespace polyregion::invoke;

namespace {

void dumpDevice(Device &d, const size_t i, const std::string &indent) {
  fmt::print("{}Device #{}: {}\n", indent, i, d.name());
  fmt::print("{}  id:                   {}\n", indent, d.id());
  fmt::print("{}  moduleFormat:         {}\n", indent, magic_enum::enum_name(d.moduleFormat()));
  fmt::print("{}  physicalDevice:       {}\n", indent, d.physicalDevice().str());
  fmt::print("{}  sharedAddressSpace:   {}\n", indent, d.sharedAddressSpace());
  fmt::print("{}  pagingMode:           {}\n", indent, magic_enum::enum_name(d.pagingMode()));
  fmt::print("{}  singleEntryPerModule: {}\n", indent, d.singleEntryPerModule());
  fmt::print("{}  features:             {}\n", indent, d.features() | mk_string(", "));
  if (const auto props = d.properties(); !props.empty()) {
    fmt::print("{}  properties:\n", indent);
    props ^ for_each([&](const auto &k, const auto &v) { fmt::print("{}    {}: {}\n", indent, k, v); });
  }
}

std::string quote(const std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\', out += c;
    else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
    else out += c;
  }
  return out + "\"";
}

} // namespace

int fired_main( //
    bool list = fire::arg({"-l", "--list", "List backends and their devices by name only, like `clinfo -l`"}),
    bool json = fire::arg({"--json", "Print one JSON object per backend, devices with their features, one per line"}),
    fire::optional<std::string> select = fire::arg({"--select", "Only enumerate the backend a POLYRT_PLATFORM selector names"})) {

  std::optional<Backend> only;
  if (select) {
    const auto parsed = polyregion::compiletime::TargetSpec::parse(select.value());
    if (!parsed) return fmt::print(stderr, "polyinfo: '{}' names no backend\n", select.value()), EXIT_FAILURE;
    only = parsed->spec.runtime;
  }

  magic_enum::enum_values<Backend>() | zip_with_index<size_t>() | for_each([&](const auto &backend, const auto &backendIdx) {
    if (only && *only != backend) return;
    auto r = Platform::of(backend);
    if (json) {
      fmt::print("{{\"backend\":{}", quote(magic_enum::enum_name(backend)));
      if (const auto err = r ^ get_maybe<std::string>()) {
        fmt::print(",\"error\":{}}}\n", quote(*err));
        return;
      }
      auto platform = std::move(std::get<std::unique_ptr<Platform>>(r));
      const auto devices = platform->enumerate() ^ mk_string(",", [](const auto &d) {
                             return fmt::format("{{\"name\":{},\"format\":{},\"features\":[{}]}}", quote(d->name()),
                                                quote(magic_enum::enum_name(d->moduleFormat())),
                                                d->features() ^ mk_string(",", [](const auto &f) { return quote(f); }));
                           });
      fmt::print(",\"platform\":{},\"kind\":{},\"devices\":[{}]}}\n", quote(platform->name()),
                 quote(magic_enum::enum_name(platform->kind())), devices);
      return;
    }
    if (const auto err = r ^ get_maybe<std::string>()) {
      if (!list) fmt::print("Backend #{} {}: unavailable ({})\n", backendIdx, magic_enum::enum_name(backend), *err);
      return;
    }
    auto platform = std::move(std::get<std::unique_ptr<Platform>>(r));
    auto devices = platform->enumerate();

    if (list) {
      fmt::print("Backend #{}: {} [{}]\n", backendIdx, platform->name(), magic_enum::enum_name(platform->kind()));
      devices | zip_with_index() | for_each([&](const auto &device, const auto &i) {
        fmt::print(" {}-- Device #{}: {}\n", i + 1 == devices.size() ? "`" : "+", i, device->name());
      });
    } else {
      fmt::print("Backend #{}: {} [{}] ({} device(s))\n", backendIdx, platform->name(), magic_enum::enum_name(platform->kind()),
                 devices.size());
      if (const auto props = platform->properties(); !props.empty())
        props ^ for_each([](const auto &k, const auto &v) { fmt::print("  {}: {}\n", k, v); });
      devices | zip_with_index() | for_each([&](const auto &device, const auto &i) { dumpDevice(*device, i, "  "); });
    }
  });
  return EXIT_SUCCESS;
}

FIRE(fired_main, "polyinfo: enumerate polyregion backends, devices, features and properties (the device-selection surface). Use -l for a "
                 "clinfo-style listing.")

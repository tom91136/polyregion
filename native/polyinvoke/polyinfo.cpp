#include <cstdio>
#include <string>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"

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
  fmt::print("{}  singleEntryPerModule: {}\n", indent, d.singleEntryPerModule());
  fmt::print("{}  features:             {}\n", indent, d.features() | mk_string(", "));
  if (const auto props = d.properties(); !props.empty()) {
    fmt::print("{}  properties:\n", indent);
    for (auto &[k, v] : props)
      fmt::print("{}    {}: {}\n", indent, k, v);
  }
}

} // namespace

int fired_main( //
    bool list = fire::arg({"-l", "--list", "List backends and their devices by name only, like `clinfo -l`"})) {

  magic_enum::enum_values<Backend>() | zip_with_index<size_t>() | for_each([&](const auto backend, const auto backendIdx) {
    auto r = Platform::of(backend);
    if (const auto err = r ^ get_maybe<std::string>()) {
      if (!list) fmt::print("Backend #{} {}: unavailable ({})\n", backendIdx, magic_enum::enum_name(backend), *err);
      return;
    }
    auto platform = std::move(std::get<std::unique_ptr<Platform>>(r));
    auto devices = platform->enumerate();

    if (list) {
      fmt::print("Backend #{}: {} [{}]\n", backendIdx, platform->name(), magic_enum::enum_name(platform->kind()));
      for (size_t i = 0; i < devices.size(); ++i)
        fmt::print(" {}-- Device #{}: {}\n", i + 1 == devices.size() ? "`" : "+", i, devices[i]->name());
    } else {
      fmt::print("Backend #{}: {} [{}] ({} device(s))\n", backendIdx, platform->name(), magic_enum::enum_name(platform->kind()),
                 devices.size());
      if (const auto props = platform->properties(); !props.empty())
        for (auto &[k, v] : props)
          fmt::print("  {}: {}\n", k, v);
      for (size_t i = 0; i < devices.size(); ++i)
        dumpDevice(*devices[i], i, "  ");
    }
  });
  return EXIT_SUCCESS;
}

FIRE(fired_main, "polyinfo: enumerate polyregion backends, devices, features and properties (the device-selection surface). Use -l for a "
                 "clinfo-style listing.")

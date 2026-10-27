#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "llvm/Support/JSON.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyregion/types.h"

namespace polyregion::polytest::skip {

using namespace aspartame;

struct Device {
  std::string name, format;
  std::vector<std::string> features;
};

struct Inventory {
  std::string platform, kind, error;
  std::vector<Device> devices;
};

struct Image {
  std::string kind, format;
  std::vector<std::string> features;

  bool operator==(const Image &that) const { return kind == that.kind && format == that.format && features == that.features; }
};

using Manifest = std::map<std::string, std::vector<Image>>;

struct Attempt {
  std::string module, outcome;
  Image image;
  std::vector<std::string> detail;
};

struct Record {
  std::optional<std::pair<std::string, Device>> device;
  std::vector<Attempt> attempts;
};

struct Verdict {
  bool justified = false;
  std::string reason;
};

namespace detail {

inline std::vector<std::string> strings(const llvm::json::Array *xs) {
  std::vector<std::string> out;
  if (xs)
    for (const auto &x : *xs)
      if (const auto s = x.getAsString()) out.emplace_back(s->str());
  return out;
}

inline bool sameFeature(const std::string &a, const std::string &b) { return (a ^ to_lower()) == (b ^ to_lower()); }

inline bool hasFeature(const Device &d, const std::string &f) {
  return d.features ^ exists([&](const auto &x) { return sameFeature(x, f); });
}

inline Image imageOf(const llvm::json::Object &o) {
  return {o.getString("kind").value_or("").str(), o.getString("format").value_or("").str(), strings(o.getArray("features"))};
}

inline std::string show(const Image &i) { return fmt::format("{}/{}[{}]", i.kind, i.format, i.features ^ mk_string(",")); }

} // namespace detail

inline std::optional<Inventory> parseInventory(const std::string &text) {
  for (const auto &line : text ^ split('\n')) {
    auto parsed = llvm::json::parse(line);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      continue;
    }
    const auto *o = parsed->getAsObject();
    if (!o) continue;
    Inventory inv;
    inv.platform = o->getString("platform").value_or("").str();
    inv.kind = o->getString("kind").value_or("").str();
    inv.error = o->getString("error").value_or("").str();
    if (const auto *ds = o->getArray("devices"))
      for (const auto &d : *ds)
        if (const auto *dobj = d.getAsObject())
          inv.devices.push_back({dobj->getString("name").value_or("").str(), dobj->getString("format").value_or("").str(),
                                 detail::strings(dobj->getArray("features"))});
    return inv;
  }
  return std::nullopt;
}

inline Manifest parseManifest(const std::string &text) {
  Manifest out;
  for (const auto &line : text ^ split('\n')) {
    auto parsed = llvm::json::parse(line);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      continue;
    }
    const auto *o = parsed->getAsObject();
    const auto *images = o ? o->getArray("images") : nullptr;
    if (!images) continue;
    auto &into = out[o->getString("module").value_or("").str()];
    for (const auto &i : *images)
      if (const auto *iobj = i.getAsObject()) {
        const auto image = detail::imageOf(*iobj);
        if (!(into ^ contains(image))) into.push_back(image);
      }
  }
  return out;
}

inline std::optional<Record> parseRecord(const std::string &stderrText) {
  constexpr std::string_view prefix = "polyrt-skip: ";
  const auto line = (stderrText ^ split('\n')) ^ find_last([&](const auto &l) { return l ^ starts_with(prefix); });
  if (!line) return std::nullopt;
  auto parsed = llvm::json::parse(std::string_view(*line).substr(prefix.size()));
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }
  const auto *o = parsed->getAsObject();
  if (!o) return std::nullopt;
  Record rec;
  if (const auto *d = o->getObject("device"))
    rec.device = std::pair{d->getString("kind").value_or("").str(),
                           Device{d->getString("name").value_or("").str(), d->getString("format").value_or("").str(), {}}};
  if (const auto *as = o->getArray("attempts"))
    for (const auto &a : *as)
      if (const auto *aobj = a.getAsObject())
        rec.attempts.push_back({aobj->getString("module").value_or("").str(), aobj->getString("outcome").value_or("").str(),
                                detail::imageOf(*aobj), detail::strings(aobj->getArray("detail"))});
  return rec;
}

inline Verdict reject(std::string reason) { return {false, std::move(reason)}; }

inline Verdict judge(const std::string &selector, const std::optional<Inventory> &inventory, const Manifest &manifest,
                     const std::string &stderrText) {
  const auto parsed = compiletime::TargetSpec::parse(selector);
  if (!parsed) return reject(fmt::format("selector '{}' names no backend", selector));
  if (!inventory) return reject("polyinfo reported nothing for the selected backend");
  if (!inventory->error.empty()) return reject(fmt::format("selected backend is unavailable: {}", inventory->error));

  const auto &required = parsed->spec.requiredDeviceFeatures;
  const auto pattern = parsed->deviceGlob.empty() ? std::string("*") : parsed->deviceGlob;
  const auto eligible = inventory->devices ^ filter([&](const auto &d) {
                          return required ^ forall([&](const auto &f) { return detail::hasFeature(d, std::string(f)); });
                        });
  const auto matched = eligible ^ filter([&](const auto &d) { return d.name ^ glob_matches_ignore_case(pattern); });
  if (matched.size() != 1)
    return reject(fmt::format("selector '{}' matched {} of {} eligible device(s): [{}]", selector, matched.size(), eligible.size(),
                              eligible ^ mk_string(", ", [](const auto &d) { return d.name; })));
  const auto &device = matched.front();

  const auto record = parseRecord(stderrText);
  if (!record) return reject("exit 77 without a skip record");
  if (!record->device) return reject("runtime had no device");
  if (record->device->second.name != device.name)
    return reject(fmt::format("runtime used '{}' but the selector resolves to '{}'", record->device->second.name, device.name));

  const auto failing =
      record->attempts //
      ^ filter([&](const auto &a) {
          return !(record->attempts ^ exists([&](const auto &b) { return b.module == a.module && b.outcome == "loaded"; }));
        }) //
      ^ map([](const auto &a) { return a.module; }) ^ distinct();
  if (failing.empty()) return reject("no image offered");

  std::vector<std::string> notes;
  std::string unreachable;
  for (const auto &module : failing) {
    const auto found = manifest.find(module);
    if (found == manifest.end()) return reject(fmt::format("no compile-time image manifest for module '{}'", module));
    const auto &offered = found->second;
    const auto attempts = record->attempts ^ filter([&](const auto &a) { return a.module == module; });

    for (const auto &a : attempts)
      if (!(offered ^ contains(a.image)))
        return reject(fmt::format("module '{}': runtime tried image {} that the compiler did not emit", module, detail::show(a.image)));
    for (const auto &i : offered)
      if (i.kind == inventory->kind && !(attempts ^ exists([&](const auto &a) { return a.image == i; })))
        return reject(fmt::format("module '{}': runtime never tried emitted image {}", module, detail::show(i)));

    const auto candidates = offered ^ filter([&](const auto &i) { return i.kind == inventory->kind && i.format == device.format; });
    if (candidates.empty()) {
      if (unreachable.empty())
        unreachable = fmt::format("module '{}': no image for {}/{} device '{}' (offered: {})", module, inventory->kind, device.format,
                                  device.name, offered ^ mk_string(", ", [](const auto &i) { return detail::show(i); }));
      continue;
    }

    for (const auto &image : candidates) {
      std::vector<std::string> missing;
      for (const auto gated : runtime::DeviceGatedFeatures) {
        const std::string name(gated);
        if ((image.features ^ exists([&](const auto &f) { return detail::sameFeature(f, name); })) && !detail::hasFeature(device, name))
          missing.push_back(name);
      }
      if (missing.empty())
        return reject(
            fmt::format("module '{}': device '{}' has every feature image {} requires", module, device.name, detail::show(image)));
      const auto rejected = attempts ^ exists([&](const auto &a) {
                              return a.image == image && a.outcome == "lacks" && a.detail.size() == missing.size()
                                     && (a.detail ^ forall([&](const auto &d) { return missing ^ contains(d); }));
                            });
      if (!rejected)
        return reject(fmt::format("module '{}': runtime did not reject image {} for lacking {}", module, detail::show(image),
                                  missing ^ mk_string(", ")));
      notes.push_back(fmt::format("{} needs {}", image.format, missing ^ mk_string(", ")));
    }
  }
  if (notes.empty()) return reject(unreachable);
  return {true, fmt::format("polyrt: no compatible target: {} lacks a feature its image requires ({})", device.name,
                            notes ^ distinct() ^ mk_string("; "))};
}

} // namespace polyregion::polytest::skip

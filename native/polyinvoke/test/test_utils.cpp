#include "test_utils.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/env_keys.h"
#include "polyregion/llvm_utils.hpp"

#include "polytest/profile.hpp"

#ifndef POLYREGION_TEST_PROFILE_DIR
  #define POLYREGION_TEST_PROFILE_DIR ""
#endif

using namespace aspartame;
using polyregion::polytest::cases::Context;
using polyregion::polytest::cases::RequireFailed;
using polyregion::polytest::cases::SkipRequested;
using polyregion::polytest::cases::Task;

namespace polyregion::test_utils {

const DeviceSkip skipHasSpirv = [](invoke::Backend, invoke::Device &d) { return d.features() | contains("spirv_kernel"); };
const DeviceSkip skipNoSpirv = [](invoke::Backend, invoke::Device &d) { return !(d.features() | contains("spirv_kernel")); };
const DeviceSkip skipNoFp64 = [](invoke::Backend, invoke::Device &d) { return !(d.features() | contains("fp64")); };

ImageGroup findTestImage(const ImageGroups &archToImage, const invoke::Backend &backend, const std::vector<std::string> &features) {
  const auto sortedFeatures = features ^ sort();

  // XXX vulkan: every entry is a separate kernel module that must be loaded together (singleEntryPerModule).
  if (backend == invoke::Backend::Vulkan) {
    ImageGroup out;
    out.reserve(archToImage.size());
    for (auto &[module_, image] : archToImage)
      out.emplace_back(module_, std::string(image.begin(), image.end()));
    return out;
  }

  if (auto direct = sortedFeatures | collect_first([&](auto &feature) {
                      return archToImage ^ get_maybe(feature) //
                             ^ map([&](auto &x) { return ImageGroup{{feature, std::string(x.begin(), x.end())}}; });
                    }))
    return *direct;

  for (auto &[arch, image] : archToImage) {
#if defined(__x86_64__) || defined(_M_X64)
    auto archType = llvm::Triple::ArchType::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
    auto archType = llvm::Triple::ArchType::aarch64;
#elif defined(__arm__) || defined(_M_ARM)
    auto archType = llvm::Triple::ArchType::arm;
#else
  #error "Unsupported architecture"
#endif
    std::vector<std::string> required;
    llvm_shared::collectCPUFeatures(arch, archType, required);
    if (required ^ forall([&](auto &r) { return sortedFeatures ^ contains(r); })) return {{arch, std::string(image.begin(), image.end())}};
  }

  if (auto it = archToImage.find(""); it != archToImage.end()) return {{"", std::string(it->second.begin(), it->second.end())}};

  return {};
}

int runOnDevice(invoke::Backend backend, std::string_view deviceName, const ImageGroups &images, const DeviceRunner &runner) {
  Context ctx;

  std::optional<invoke::DeviceLock> lock;
  try {
    auto platform = invoke::Platform::maybe(backend);
    if (!platform) return 77;
    std::unique_ptr<invoke::Device> device;
    for (auto &d : platform->enumerate())
      if (d->name() == deviceName) {
        device = std::move(d);
        break;
      }
    if (!device) return 77;
    lock.emplace(device->physicalDevice());
    auto imageGroup = findTestImage(images, backend, device->features());
    if (imageGroup.empty()) return 77;
    runner(ctx, backend, *platform, *device, imageGroup);
  } catch (const SkipRequested &) {
    return 77;
  } catch (const RequireFailed &) {
    return 1;
  }
  return ctx.failed ? 1 : 0;
}

std::vector<Task> discoverMatrix(const std::vector<Suite> &suites) {
  const auto targets = polytest::resolveTestTargets(POLYREGION_TEST_PROFILE_DIR, env::PolyinvokeTestTargets);
  const auto profileAllows = [&targets](invoke::Backend b, invoke::Device &d) -> bool {
    if (targets.empty()) return true;
    const auto features = d.features();
    const auto hasFeature = [&](std::string_view f) {
      return std::any_of(features.begin(), features.end(), [&](auto &x) { return x == f; });
    };
    for (const auto &t : targets) {
      if (t.spec.runtime != b) continue;
      if (!std::all_of(t.spec.requiredDeviceFeatures.begin(), t.spec.requiredDeviceFeatures.end(), hasFeature)) continue;
      const bool archAsFeature = b != invoke::Backend::RelocatableObject && b != invoke::Backend::SharedObject;
      if (archAsFeature && !t.arch.empty() && t.arch != "*" && !hasFeature(t.arch)) continue;
      return true;
    }
    return false;
  };
  return suites                                                              //
         | flat_map([&](const Suite &s) -> std::vector<Task> {               //
             return s.backends                                               //
                    | flat_map([&](invoke::Backend b) -> std::vector<Task> { //
                        auto platform = invoke::Platform::maybe(b);
                        if (!platform) return {};
                        return platform->enumerate()                                                     //
                               | filter([&](auto &d) { return profileAllows(b, *d) && !s.skip(b, *d); }) //
                               | map([&, name = s.name, images = s.images, runner = s.runner](auto &d) -> Task {
                                   const std::string deviceName = d->name();
                                   return Task{
                                       fmt::format("{}-{}-{}", name, magic_enum::enum_name(b), polytest::cases::sanitiseId(deviceName)),
                                       (b == invoke::Backend::RelocatableObject || b == invoke::Backend::SharedObject) ? "" : "device",
                                       [b, deviceName, images, runner]() -> int { return runOnDevice(b, deviceName, *images, runner); }};
                                 }) //
                               | to_vector();
                      }) //
                    | to_vector();
           }) //
         | to_vector();
}

} // namespace polyregion::test_utils

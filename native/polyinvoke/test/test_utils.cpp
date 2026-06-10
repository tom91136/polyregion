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

  if (auto direct = sortedFeatures ^ collect_first([&](auto &feature) {
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

  if (const auto x = archToImage ^ get_maybe(std::string{})) return {{"", std::string(x->begin(), x->end())}};

  return {};
}

int runOnTarget(invoke::Backend backend, std::string_view arch, const std::vector<std::string> &requiredFeatures, const ImageGroups &images,
                const DeviceRunner &runner, const DeviceSkip &skip) {
  Context ctx;

  std::optional<invoke::DeviceLock> lock;
  try {
    auto platform = invoke::Platform::maybe(backend);
    if (!platform) return 77;
    const bool archAsFeature = backend != invoke::Backend::RelocatableObject && backend != invoke::Backend::SharedObject;
    std::unique_ptr<invoke::Device> device;
    for (auto &d : platform->enumerate()) {
      const auto features = d->features();
      const auto hasFeature = [&](const std::string_view f) { return features ^ exists([&](auto &x) { return x == f; }); };
      if (!(requiredFeatures ^ forall(hasFeature))) continue;
      if (archAsFeature && !arch.empty() && arch != "*" && !hasFeature(arch)) continue;
      if (skip(backend, *d)) continue;
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
  return suites ^ flat_map([&](const Suite &s) {
           return s.backends ^ flat_map([&](const invoke::Backend b) {
                    return targets ^ collect([&](auto &t) -> std::optional<Task> {
                             if (t.spec.runtime != b) return {};
                             const std::string arch = t.arch;
                             const std::vector<std::string> required(t.spec.requiredDeviceFeatures.begin(),
                                                                     t.spec.requiredDeviceFeatures.end());
                             // XXX Key on the canonical spec name, not the Backend enum: opencl1_1 and spirv64_kernel both map
                             // to Backend::OpenCL and would collide on the same arch (e.g. both `@intel`).
                             return Task{fmt::format("{}-{}-{}", s.name, std::string(t.spec.canonical),
                                                     polytest::cases::sanitiseId(arch.empty() ? "any" : arch)),
                                         (b == invoke::Backend::RelocatableObject || b == invoke::Backend::SharedObject) ? "" : "device",
                                         [b, arch, required, images = s.images, runner = s.runner, skip = s.skip]() -> int {
                                           return runOnTarget(b, arch, required, *images, runner, skip);
                                         }};
                           });
                  });
         });
}

} // namespace polyregion::test_utils

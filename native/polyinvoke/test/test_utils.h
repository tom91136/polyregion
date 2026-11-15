#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyinvoke/device_lock.h"
#include "polyinvoke/runtime.h"

#include "polytest/skip_proof.hpp"
#include "polytest/test_case.hpp"

namespace polyregion::test_utils {

using ImageGroups = std::unordered_map<std::string, std::vector<uint8_t>>;
using ImageGroup = std::vector<std::pair<std::string, std::string>>;

ImageGroup findTestImage(const ImageGroups &images, const invoke::Backend &backend, const std::vector<std::string> &features);

// a device that violates any need is skipped; the audit recomputes the violation from the device's features
using DeviceSkip = std::vector<polytest::skip::Need>;

inline DeviceSkip operator+(DeviceSkip lhs, const DeviceSkip &rhs) {
  lhs.insert(lhs.end(), rhs.begin(), rhs.end());
  return lhs;
}

inline void prependTidArg(invoke::Platform &platform, invoke::ArgBuffer &buffer) {
  if (platform.kind() == invoke::PlatformKind::HostThreaded) buffer.append(invoke::Type::IntS64, nullptr);
}

void checkHostCapabilities(polytest::cases::Context &ctx, invoke::Platform &platform, invoke::Device &device);

using DeviceRunner =
    std::function<void(polytest::cases::Context &, invoke::Backend, invoke::Platform &, invoke::Device &, const ImageGroup &)>;

inline const DeviceSkip skipHasSpirv{{"spirv_kernel", false, "the source variant targets devices without SPIR-V kernels"}};
inline const DeviceSkip skipNoSpirv{{"spirv_kernel", true, "the SPIR-V variant targets devices with SPIR-V kernels"}};
inline const DeviceSkip skipNoFp64{{"fp64", true}};

struct Suite {
  std::string name;
  std::shared_ptr<const ImageGroups> images;
  std::vector<invoke::Backend> backends;
  DeviceRunner runner;
  DeviceSkip skip;

  Suite(std::string name, const ImageGroups &images, std::vector<invoke::Backend> backends, DeviceRunner runner, DeviceSkip skip = {})
      : name(std::move(name)), images(std::make_shared<ImageGroups>(images)), backends(std::move(backends)), runner(std::move(runner)),
        skip(std::move(skip)) {}
};

int runOnTarget(invoke::Backend backend, std::string_view arch, const std::vector<std::string> &requiredFeatures, const ImageGroups &images,
                const DeviceRunner &runner, const DeviceSkip &skip);

std::vector<polytest::cases::Task> discoverMatrix(const std::vector<Suite> &suites);

} // namespace polyregion::test_utils

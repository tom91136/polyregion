#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyinvoke/device_lock.h"
#include "polyinvoke/runtime.h"

#include "polytest/test_case.hpp"

namespace polyregion::test_utils {

using ImageGroups = std::unordered_map<std::string, std::vector<uint8_t>>;
using ImageGroup = std::vector<std::pair<std::string, std::string>>;

ImageGroup findTestImage(const ImageGroups &images, const invoke::Backend &backend, const std::vector<std::string> &features);

using DeviceSkip = std::function<bool(invoke::Backend, invoke::Device &)>;

using DeviceRunner =
    std::function<void(polytest::cases::Context &, invoke::Backend, invoke::Platform &, invoke::Device &, const ImageGroup &)>;

extern const DeviceSkip skipHasSpirv;
extern const DeviceSkip skipNoSpirv;
extern const DeviceSkip skipNoFp64;

struct Suite {
  std::string name;
  std::shared_ptr<const ImageGroups> images;
  std::vector<invoke::Backend> backends;
  DeviceRunner runner;
  DeviceSkip skip;

  Suite(
      std::string name, const ImageGroups &images, std::vector<invoke::Backend> backends, DeviceRunner runner,
      DeviceSkip skip = [](invoke::Backend, invoke::Device &) { return false; })
      : name(std::move(name)), images(std::make_shared<ImageGroups>(images)), backends(std::move(backends)), runner(std::move(runner)),
        skip(std::move(skip)) {}
};

int runOnDevice(invoke::Backend backend, std::string_view deviceName, const ImageGroups &images, const DeviceRunner &runner);

std::vector<polytest::cases::Task> discoverMatrix(const std::vector<Suite> &suites);

} // namespace polyregion::test_utils

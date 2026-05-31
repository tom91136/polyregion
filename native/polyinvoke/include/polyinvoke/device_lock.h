#pragma once

#include <memory>
#include <string_view>

#include "polyregion/export.h"
#include "polyregion/types.h"

namespace polyregion::invoke {

// Cross-process exclusive lock keyed on the physical device, so backends sharing one GPU serialise.
// A no-op for host/CPU devices (PhysicalDevice::host()).
class POLYREGION_EXPORT DeviceLock {
public:
  explicit DeviceLock(const PhysicalDevice &device);
  ~DeviceLock();

  DeviceLock(DeviceLock &&) noexcept;
  DeviceLock &operator=(DeviceLock &&) noexcept;
  DeviceLock(const DeviceLock &) = delete;
  DeviceLock &operator=(const DeviceLock &) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace polyregion::invoke

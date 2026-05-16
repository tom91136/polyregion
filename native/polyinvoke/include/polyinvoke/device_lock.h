#pragma once

#include <memory>
#include <string_view>

#include "polyregion/export.h"
#include "polyregion/types.h"

namespace polyregion::invoke {

// When "1", polyrt acquires a cross-process lock on each unique device
inline constexpr const char *DeviceLockEnv = "POLYINVOKE_TEST_LOCK";

class POLYREGION_EXPORT DeviceLock {
public:
  DeviceLock(Backend backend, std::string_view deviceName);
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

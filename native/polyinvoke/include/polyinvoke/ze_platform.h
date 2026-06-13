#pragma once

#include "polyregion/compat.h"

#include "runtime.h"
#include "zeew.h"

namespace polyregion::invoke::ze {

class POLYREGION_EXPORT ZePlatform final : public Platform {
  ze_driver_handle_t driver;
  POLYREGION_EXPORT explicit ZePlatform(ze_driver_handle_t driver);

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~ZePlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace details {
using ZeModuleStore = detail::ModuleStore<ze_module_handle_t, ze_kernel_handle_t>;
}

class POLYREGION_EXPORT ZeDevice final : public Device {

  ze_driver_handle_t driver;
  ze_device_handle_t device;
  detail::LazyDroppable<ze_context_handle_t> context;
  std::string deviceName;
  details::ZeModuleStore store; // must be dropped before context (kernel/module owned by context)

public:
  POLYREGION_EXPORT explicit ZeDevice(ze_driver_handle_t driver, ze_device_handle_t device);
  POLYREGION_EXPORT ~ZeDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT PhysicalDevice physicalDevice() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT PagingMode pagingMode() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT uintptr_t mallocDevice(size_t size, Access access) override;
  POLYREGION_EXPORT void freeDevice(uintptr_t ptr) override;
  POLYREGION_EXPORT std::optional<void *> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void *ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT ZeDeviceQueue final : public DeviceQueue {

  detail::CountingLatch latch;
  details::ZeModuleStore &store;
  ze_command_list_handle_t cmdList = {};
  ze_event_pool_handle_t eventPool = {};
  ze_event_handle_t event = {};

  // Level Zero has no "host function append".  Each op signals a reusable event; we host-
  // sync on it then fire the callback inline.  Serialises submissions on this queue but
  // keeps the API contract and stays within v1.0 (no zeCommandListHostSynchronize needed).
  void waitAndFire(const MaybeCallback &cb);
  void appendCopyAndWait(void *dst, const void *src, size_t size, const MaybeCallback &cb);

public:
  POLYREGION_EXPORT ZeDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, ze_context_handle_t context,
                                  ze_device_handle_t device);
  POLYREGION_EXPORT ~ZeDeviceQueue() override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::ze

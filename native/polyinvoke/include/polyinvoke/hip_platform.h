#pragma once

#include "polyregion/compat.h"

#include "hipew.h"
#include "runtime.h"

namespace polyregion::invoke::hip {

class POLYREGION_EXPORT HipPlatform final : public Platform {
  POLYREGION_EXPORT explicit HipPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~HipPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace details {
using HipModuleStore = detail::ModuleStore<hipModule_t, hipFunction_t>;
}

class POLYREGION_EXPORT HipDevice final : public Device {

  hipDevice_t device = {};
  detail::LazyDroppable<hipCtx_t> context;
  std::string deviceName;
  details::HipModuleStore store; // store needs to be dropped before dropping device

public:
  explicit HipDevice(int ordinal);
  ~HipDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
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

class POLYREGION_EXPORT HipDeviceQueue final : public DeviceQueue {

  detail::CountingLatch latch;

  details::HipModuleStore &store;
  hipStream_t stream{};

  void enqueueCallback(const MaybeCallback &cb);

public:
  POLYREGION_EXPORT explicit HipDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store);
  POLYREGION_EXPORT ~HipDeviceQueue() override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::hip

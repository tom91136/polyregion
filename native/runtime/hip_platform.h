#pragma once

#include "hipew.h"
#include "runtime.h"

namespace polyregion::runtime::hip {

class POLYREGION_EXPORT HipPlatform : public Platform {
public:
  POLYREGION_EXPORT explicit HipPlatform();
  POLYREGION_EXPORT ~HipPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using HipModuleStore = detail::ModuleStore<hipModule_t, hipFunction_t>;
}

class POLYREGION_EXPORT HipDevice : public Device {

  hipDevice_t device = {};
  detail::LazyDroppable<hipCtx_t> context;
  std::string deviceName;
  HipModuleStore store; // store needs to be dropped before dropping device

public:
  explicit HipDevice(int ordinal);
  ~HipDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT bool leadingIndexArgument() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT uintptr_t malloc(size_t size, Access access) override;
  POLYREGION_EXPORT void free(uintptr_t ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class POLYREGION_EXPORT HipDeviceQueue : public DeviceQueue {

  detail::CountingLatch latch;

  HipModuleStore &store;
  hipStream_t stream{};

  void enqueueCallback(const MaybeCallback &cb);

public:
  POLYREGION_EXPORT explicit HipDeviceQueue(decltype(store) store);
  POLYREGION_EXPORT ~HipDeviceQueue() override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::hip

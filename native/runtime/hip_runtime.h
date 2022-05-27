#pragma once

#include "hipew.h"
#include "runtime.h"

namespace polyregion::runtime::hip {

class EXPORT HipRuntime : public Runtime {
public:
  EXPORT explicit HipRuntime();
  EXPORT ~HipRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using HipModuleStore = detail::ModuleStore<hipModule_t, hipFunction_t>;
}

class EXPORT HipDevice : public Device {

  hipDevice_t device = {};
  detail::LazyDroppable<hipCtx_t> context;
  std::string deviceName;
  HipModuleStore store; // store needs to be dropped before dropping device

public:
  EXPORT explicit HipDevice(int ordinal);
  EXPORT ~HipDevice() override;
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT HipDeviceQueue : public DeviceQueue {

  HipModuleStore &store;
  hipStream_t stream{};

  void enqueueCallback(const MaybeCallback &cb);

public:
  EXPORT explicit HipDeviceQueue(decltype(store) store);
  EXPORT ~HipDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::hip

#pragma once

#include "cuew.h"
#include "runtime.h"

namespace polyregion::runtime::cuda {

class EXPORT CudaPlatform : public Platform {
public:
  EXPORT explicit CudaPlatform();
  EXPORT ~CudaPlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using CudaModuleStore = detail::ModuleStore<CUmodule, CUfunction>;
}

class EXPORT CudaDevice : public Device {

  CUdevice device = {};
  detail::LazyDroppable<CUcontext> context;
  std::string deviceName;
  CudaModuleStore store;

public:
  EXPORT explicit CudaDevice(int ordinal);
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT bool sharedAddressSpace() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::string> features() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
  ~CudaDevice() override;
};

class EXPORT CudaDeviceQueue : public DeviceQueue {

  CudaModuleStore &store;
  CUstream stream{};

  void enqueueCallback(const MaybeCallback &cb);

public:
  EXPORT explicit CudaDeviceQueue(decltype(store) store);
  EXPORT ~CudaDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::cuda

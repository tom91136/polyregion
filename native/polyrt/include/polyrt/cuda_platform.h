#pragma once

#include "cuew.h"
#include "runtime.h"

namespace polyregion::runtime::cuda {

class POLYREGION_EXPORT CudaPlatform : public Platform {
  POLYREGION_EXPORT explicit CudaPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~CudaPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using CudaModuleStore = detail::ModuleStore<CUmodule, CUfunction>;
}

class POLYREGION_EXPORT CudaDevice : public Device {

  CUdevice device = {};
  detail::LazyDroppable<CUcontext> context;
  std::string deviceName;
  CudaModuleStore store;

public:
  explicit CudaDevice(int ordinal);
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
  POLYREGION_EXPORT std::optional<void*> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void* ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
  ~CudaDevice() override;
};

class POLYREGION_EXPORT CudaDeviceQueue : public DeviceQueue {

  detail::CountingLatch latch;

  CudaModuleStore &store;
  CUstream stream{};

  void enqueueCallback(const MaybeCallback &cb);

public:
  POLYREGION_EXPORT explicit CudaDeviceQueue(decltype(store) store);
  POLYREGION_EXPORT ~CudaDeviceQueue() override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::cuda

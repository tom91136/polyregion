#pragma once

#include "cuew.h"
#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::cuda {

class EXPORT CudaRuntime : public Runtime {
public:
  EXPORT explicit CudaRuntime();
  EXPORT ~CudaRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT CudaDevice : public Device {

  CUdevice device = {};
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  std::string deviceName;
  using LoadedModule = std::pair<CUmodule, std::unordered_map<std::string, CUfunction>>;
  std::unordered_map<std::string, LoadedModule> modules;

  void enqueueCallback(const std::optional<Callback> &cb);

public:
  EXPORT explicit CudaDevice(int ordinal);
  EXPORT ~CudaDevice() override;
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                       const std::optional<Callback> &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size,
                                       const std::optional<Callback> &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                 const std::optional<Callback> &cb) override;
};

} // namespace polyregion::runtime::cuda

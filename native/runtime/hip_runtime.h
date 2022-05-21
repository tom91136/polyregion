#pragma once

#include "hipew.h"
#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::hip {

class EXPORT HipRuntime : public Runtime {
public:
  EXPORT explicit HipRuntime();
  EXPORT ~HipRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT HipDevice : public Device {

  hipDevice_t device = {};
  hipCtx_t context = nullptr;
  hipStream_t stream = nullptr;
  std::string deviceName;
  using LoadedModule = std::pair<hipModule_t, std::unordered_map<std::string, hipFunction_t>> ;
  std::unordered_map<std::string, LoadedModule> modules;

  void enqueueCallback(const std::optional<Callback> &cb);

public:
  EXPORT explicit HipDevice(int ordinal);
  EXPORT ~HipDevice() override;
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                       const std::optional<Callback> &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size,
                                       const std::optional<Callback> &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                 const std::optional<Callback> &cb) override;
};

} // namespace polyregion::runtime::hip

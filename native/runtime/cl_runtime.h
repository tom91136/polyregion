#pragma once

#include "clew.h"
#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::cl {

class EXPORT ClRuntime : public Runtime {
public:
  EXPORT explicit ClRuntime();
  EXPORT ~ClRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT ClDevice : public Device {

  std::string deviceName;
  cl_device_id device = {};
  cl_context context = {};
  cl_command_queue queue = {};

  std::atomic_uintptr_t bufferCounter;
  std::unordered_map<uintptr_t, cl_mem> allocations;

  using LoadedModule = std::pair<cl_program, std::unordered_map<std::string, cl_kernel>>;
  std::unordered_map<std::string, LoadedModule> modules;

  void enqueueCallback(const std::optional<Callback> &cb, cl_event event);
  cl_mem queryMemObject(uintptr_t handle) const;

public:
  EXPORT explicit ClDevice(cl_device_id device);
  EXPORT ~ClDevice() override;
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

} // namespace polyregion::runtime::cl

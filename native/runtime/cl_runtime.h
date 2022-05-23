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

namespace {
using ClModuleStore = detail::ModuleStore<cl_program, cl_kernel>;
}

class EXPORT ClDevice : public Device {

  detail::LazyDroppable<cl_device_id> device;
  detail::LazyDroppable<cl_context> context;
  std::string deviceName;
  ClModuleStore store; // store needs to be dropped before dropping device

  std::atomic_uintptr_t bufferCounter;
  std::unordered_map<uintptr_t, cl_mem> allocations;

  cl_mem queryMemObject(uintptr_t handle) const;

public:
  EXPORT explicit ClDevice(cl_device_id device);
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT ClDeviceQueue : public DeviceQueue {

  ClModuleStore &store;
  cl_command_queue queue = {};
  std::function<cl_mem(uintptr_t)> queryMemObject;

  void enqueueCallback(const MaybeCallback &cb, cl_event event);

public:
  EXPORT ClDeviceQueue(decltype(store) store, decltype(queue) queue, decltype(queryMemObject) queryMemObject);
  EXPORT ~ClDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::cl

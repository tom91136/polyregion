#pragma once

#include "polyregion/compat.h"

#include "clew.h"
#include "runtime.h"

namespace polyregion::invoke::cl {

class POLYREGION_EXPORT ClPlatform final : public Platform {
  POLYREGION_EXPORT explicit ClPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~ClPlatform() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace details {
using ClModuleStore = detail::ModuleStore<cl_program, cl_kernel>;
}

class POLYREGION_EXPORT ClDevice final : public Device {

  detail::LazyDroppable<cl_device_id> device;
  detail::LazyDroppable<cl_context> context;
  std::string deviceName;
  details::ClModuleStore store; // store needs to be dropped before dropping device
  detail::MemoryObjects<cl_mem> memoryObjects;

public:
  explicit ClDevice(cl_device_id device);
  ~ClDevice() override;
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT uintptr_t mallocDevice(size_t size, Access access) override;
  POLYREGION_EXPORT std::optional<void *> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void *ptr) override;
  POLYREGION_EXPORT void freeDevice(uintptr_t ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT ClDeviceQueue final : public DeviceQueue {

  detail::CountingLatch latch;

  details::ClModuleStore &store;
  cl_command_queue queue = {};
  std::function<cl_mem(uintptr_t)> queryMemObject;

  void enqueueCallback(const MaybeCallback &cb, cl_event event);

public:
  POLYREGION_EXPORT ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                                  decltype(queryMemObject) queryMemObject);
  POLYREGION_EXPORT ~ClDeviceQueue() override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName,  //
                                            const std::string &symbol,      //
                                            const std::vector<Type> &types, //
                                            std::vector<std::byte> argData, //
                                            const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::cl

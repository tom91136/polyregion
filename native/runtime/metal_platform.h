#pragma once

#ifdef RUNTIME_ENABLE_METAL

  #define METALCPP_SYMBOL_VISIBILITY_HIDDEN
  #include "Metal/Metal.hpp"
  #include "runtime.h"

namespace polyregion::runtime::metal {

class EXPORT MetalPlatform : public Platform {
  NS::AutoreleasePool *pool;

public:
  EXPORT explicit MetalPlatform();
  EXPORT ~MetalPlatform() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using MetalModuleStore = detail::ModuleStore<MTL::Library *, MTL::ComputePipelineState *>;
}

class EXPORT MetalDevice : public Device {

  NS::AutoreleasePool *pool;
  MTL::Device *device;
  MetalModuleStore store;
  detail::MemoryObjects<MTL::Buffer *> memoryObjects;

public:
  explicit MetalDevice(decltype(device) device);
  ~MetalDevice() override;
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT bool sharedAddressSpace() override;
  EXPORT bool singleEntryPerModule() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::string> features() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT MetalDeviceQueue : public DeviceQueue {

  NS::AutoreleasePool *pool;
  MetalModuleStore &store;
  MTL::CommandQueue *queue;
  std::function<MTL::Buffer *(uintptr_t)> queryMemObject;

public:
  EXPORT MetalDeviceQueue(decltype(store) store, decltype(queue) queue, decltype(queryMemObject) queryMemObject);
  EXPORT ~MetalDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName,  //
                                 const std::string &symbol,      //
                                 const std::vector<Type> &types, //
                                 std::vector<std::byte> argData, //
                                 const Policy &policy, const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::metal

#endif
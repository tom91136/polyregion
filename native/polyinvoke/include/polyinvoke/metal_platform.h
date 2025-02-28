#pragma once

#ifdef RUNTIME_ENABLE_METAL

  #define METALCPP_SYMBOL_VISIBILITY_HIDDEN
  #include "Metal/Metal.hpp"
  #include "runtime.h"

namespace polyregion::invoke::metal {

class POLYREGION_EXPORT MetalPlatform final : public Platform {
  NS::AutoreleasePool *pool;
  POLYREGION_EXPORT explicit MetalPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~MetalPlatform() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using MetalModuleStore = detail::ModuleStore<MTL::Library *, MTL::ComputePipelineState *>;
}

class POLYREGION_EXPORT MetalDevice final : public Device {

  NS::AutoreleasePool *pool;
  MTL::Device *device;
  MetalModuleStore store;
  detail::MemoryObjects<MTL::Buffer *> memoryObjects;

public:
  explicit MetalDevice(decltype(device) device);
  ~MetalDevice() override;
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
  POLYREGION_EXPORT std::optional<void *> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void *ptr) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT MetalDeviceQueue final : public DeviceQueue {

  NS::AutoreleasePool *pool;
  MetalModuleStore &store;
  MTL::CommandQueue *queue;
  std::function<MTL::Buffer *(uintptr_t)> queryMemObject;

public:
  POLYREGION_EXPORT MetalDeviceQueue(decltype(store) store, decltype(queue) queue, decltype(queryMemObject) queryMemObject);
  POLYREGION_EXPORT ~MetalDeviceQueue() override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName,  //
                                            const std::string &symbol,      //
                                            const std::vector<Type> &types, //
                                            std::vector<std::byte> argData, //
                                            const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::metal

#endif
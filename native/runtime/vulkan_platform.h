#pragma once


#include "runtime.h"

#define VK_NO_PROTOTYPES
//#define VK_USE_PLATFORM_ANDROID_KHR 1
//#define KOMPUTE_OPT_USE_SPDLOG 1

//#include "kompute/Core.hpp"
//#include "kompute/Manager.hpp"
#include "json.hpp"
#include "vulkan/vulkan_raii.hpp"

namespace polyregion::runtime::vulkan {

class EXPORT VulkanPlatform : public Platform {

//  vk::DynamicLoader dl;
   vk::raii::Context context;
   vk::raii::Instance instance;

//  kp::Manager mgr;


public:
  EXPORT explicit VulkanPlatform();
  EXPORT ~VulkanPlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using VulkanModuleStore = detail::ModuleStore<int, int>;
}

class EXPORT VulkanDevice : public Device {
//
  size_t deviceId;
  std::string deviceName;
  vk::raii::Device device;
  vk::raii::CommandPool commandPool;
  vk::raii::CommandBuffers commandBuffers;

  VulkanModuleStore store;

public:
  explicit VulkanDevice(size_t queueId,size_t deviceId, const std::string &deviceName, vk::raii::Device device );
  int64_t id() override;
  EXPORT std::string name() override;
  EXPORT bool sharedAddressSpace() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::string> features() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
  ~VulkanDevice() override;
};

class EXPORT VulkanDeviceQueue : public DeviceQueue {

  detail::CountingLatch latch;

  VulkanModuleStore &store;

  void enqueueCallback(const MaybeCallback &cb);

public:
  EXPORT explicit VulkanDeviceQueue(decltype(store) store);
  EXPORT ~VulkanDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, std::vector<Type> types,
                                 std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::vulkan

#pragma once

#include "polyregion/compat.h"

#include "runtime.h"
#include <thread>

#define VK_NO_PROTOTYPES

#include "nlohmann/json.hpp"
#include "vulkan/vulkan_raii.hpp"

#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wcast-align"
  #pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_NOT_NULL
#define VMA_NULLABLE
#include "vk_mem_alloc.h"

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wno-everything
#endif

namespace polyregion::runtime::vulkan {

class POLYREGION_EXPORT VulkanPlatform : public Platform {
  std::vector<const char *> extensions;
  std::vector<const char *> layers;
  vk::raii::Context context;
  vk::raii::Instance instance;
  POLYREGION_EXPORT explicit VulkanPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~VulkanPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {

struct Resolved {
  std::shared_ptr<vk::raii::ShaderModule> shaderModule;
  vk::raii::DescriptorSetLayout dscLayout;
  vk::raii::DescriptorPool dscPool;
  vk::raii::DescriptorSet dscSet;
  vk::raii::PipelineCache pipeCache;
  vk::raii::PipelineLayout pipeLayout;

  vk::raii::CommandPool cmdPool;
  vk::raii::CommandBuffer cmdBuffer;

  Resolved(uint32_t computeQueueId,
           const std::shared_ptr<vk::raii::ShaderModule> &shaderModule, //
           const std::vector<vk::DescriptorSetLayoutBinding> &bindings, //
           const std::vector<vk::DescriptorPoolSize> &size,                                 //
           vk::raii::Device &ctx);
};

struct MemObject {
  vk::Buffer buffer;
  VmaAllocation allocation;
  void *mappedData;
  size_t size;
};

struct Enqueued {
  vk::raii::Pipeline pipeline;
  vk::raii::Fence fence;
  std::shared_ptr<MemObject> argObject;
};

using VulkanModuleStore = detail::ModuleStore<std::shared_ptr<vk::raii::ShaderModule>, Resolved>;
using VkMemObject = std::shared_ptr<MemObject>;
} // namespace

class POLYREGION_EXPORT VulkanDevice : public Device {
  //

  std::pair<uint32_t, size_t> computeQueueId;
  std::pair<uint32_t, size_t> transferQueueId;

  vk::raii::PhysicalDevice device;
  vk::raii::Device ctx;
  VmaAllocator allocator;

  //  std::shared_ptr<vk::raii::CommandPool> computeCmdPool;
  //  std::shared_ptr<vk::raii::CommandBuffer> computeCmdBuffer;

  std::shared_ptr<vk::raii::CommandPool> transferCmdPool;
  std::shared_ptr<vk::raii::CommandBuffer> transferCmdBuffer;

  std::atomic_size_t activeComputeQueues;
  std::atomic_size_t activeTransferQueues;

  VulkanModuleStore store;
  detail::MemoryObjects<VkMemObject> memoryObjects;

public:
  explicit VulkanDevice(vk::raii::Instance &instance,              //
                        decltype(computeQueueId) computeQueueId,   //
                        decltype(transferQueueId) transferQueueId, //
                        decltype(device) device_);
  int64_t id() override;
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
  ~VulkanDevice() override;
};

class POLYREGION_EXPORT VulkanDeviceQueue : public DeviceQueue {

  detail::CountingLatch latch;
  vk::raii::Device &ctx;
  VmaAllocator &allocator;
  vk::raii::Queue computeQueue;
  vk::raii::Queue transferQueue;

  VulkanModuleStore &store;
  std::function<VkMemObject(uintptr_t)> queryMemObject;
  detail::CountedStore<size_t, std::shared_ptr<Enqueued>> enqueuedStore;
  detail::BlockingQueue<std::function<void()>> callbackQueue;
  std::thread callbackThread;

  void enqueueCallback(const MaybeCallback &cb);

public:
  POLYREGION_EXPORT explicit VulkanDeviceQueue(decltype(ctx) ctx,                     //
                                    decltype(allocator) allocator,         //
                                    decltype(computeQueue) computeQueue,   //
                                    decltype(transferQueue) transferQueue, //
                                    decltype(store) store,                 //
                                    decltype(queryMemObject) queryMemObject);
  POLYREGION_EXPORT ~VulkanDeviceQueue() override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::vulkan

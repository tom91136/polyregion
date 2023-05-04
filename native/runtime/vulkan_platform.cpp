#include "vulkan_platform.h"
#include "utils.hpp"

using namespace polyregion::runtime;
using namespace polyregion::runtime::vulkan;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[Vulkan error] ";

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

VulkanPlatform::VulkanPlatform() : context(), instance(context, vk::InstanceCreateInfo())   {
  TRACE();
}
std::string VulkanPlatform::name() {
  TRACE();
  return "Vulkan";
}
std::vector<Property> VulkanPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> VulkanPlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> devices;
  for (const vk::raii::PhysicalDevice &dev : instance.enumeratePhysicalDevices()) {
    std::vector<vk::QueueFamilyProperties> queueProps = dev.getQueueFamilyProperties();
    if (auto computeQueueIter = std::find_if(queueProps.begin(), queueProps.end(),
                                             [](const auto &q) { return q.queueFlags & vk::QueueFlagBits::eCompute; });
        computeQueueIter != queueProps.end()) {
      size_t computeQueueIdx = std::distance(queueProps.begin(), computeQueueIter);
      float queuePriority = 0.0f;
      vk::DeviceQueueCreateInfo queueCreateInfo({}, computeQueueIdx, 1, &queuePriority);
      vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo);
      vk::raii::Device device = dev.createDevice(deviceCreateInfo);
      devices.push_back(std::make_unique<VulkanDevice>(computeQueueIdx,                //
                                                       dev.getProperties().deviceID,   //
                                                       dev.getProperties().deviceName, //
                                                       std::move(device))              //
      );
    }
  }

  return devices;
}

// ---

VulkanDevice::VulkanDevice(size_t queueId, size_t deviceId, const std::string &deviceName, vk::raii::Device _device )
    :
      deviceId(deviceId),                                                                                       //
      deviceName(deviceName) ,                                                                                   //
      device(std::move(_device)),                                                                                //
      commandPool(device, vk::CommandPoolCreateInfo({}, queueId)),                                              //
      commandBuffers(device, vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1)), //
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            TRACE();
            return 0;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            return 0;
          },
          [&](auto &&m) { TRACE(); }, [&](auto &&) { TRACE(); }) {
  TRACE();
}

int64_t VulkanDevice::id() {
  TRACE();
  return  deviceId;
}
std::string VulkanDevice::name() {
  TRACE();
  return  deviceName;
}
bool VulkanDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
std::vector<Property> VulkanDevice::properties() {
  TRACE();
  return {};
}
std::vector<std::string> VulkanDevice::features() {
  TRACE();
  return {};
}
void VulkanDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool VulkanDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t VulkanDevice::malloc(size_t size, Access) {
  TRACE();
  return 0;
}
void VulkanDevice::free(uintptr_t ptr) {
  TRACE();
}
std::unique_ptr<DeviceQueue> VulkanDevice::createQueue() {
  TRACE();
  return std::make_unique<VulkanDeviceQueue>(store);
}
VulkanDevice::~VulkanDevice() { TRACE(); }

// ---

VulkanDeviceQueue::VulkanDeviceQueue(decltype(store) store) : store(store) {
  TRACE();
}
VulkanDeviceQueue::~VulkanDeviceQueue() {
  TRACE();
}
void VulkanDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
}

void VulkanDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  enqueueCallback(cb);
}
void VulkanDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  enqueueCallback(cb);
}
void VulkanDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                           std::vector<Type> types, std::vector<std::byte> argData,
                                           const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  enqueueCallback(cb);
}

#undef CHECKED

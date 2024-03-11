#include "polyrt/vulkan_platform.h"
#include "utils.hpp"
#include <iostream>
#include <shared_mutex>
#include <utility>

#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wcast-align"
  #pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wno-everything
#endif

using namespace polyregion::runtime;
using namespace polyregion::runtime::vulkan;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[Vulkan error] ";

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// good to haves
static std::vector<const char *> commonExtensions = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};

#ifndef NDEBUG
static std::vector<const char *> extraLayers = {
    "VK_LAYER_KHRONOS_validation" //
};
static std::vector<const char *> extraExtensions = {
    VK_EXT_DEBUG_REPORT_EXTENSION_NAME, //
};
#else
static std::vector<const char *> extraLayers = {};
static std::vector<const char *> extraExtensions = {};
#endif

static vk::ApplicationInfo AppInfo(__FILE__, 1, nullptr, 0, VK_API_VERSION_1_1);

static vk::raii::Instance createInstance(std::vector<const char *> &extensions, //
                                         std::vector<const char *> &layers,     //
                                         const vk::raii::Context &ctx) {

  auto insertSupported = [](auto &from, auto &to, auto &available, auto f) {
    std::copy_if(from.begin(), from.end(), std::back_inserter(to), [&](auto ext) {
      return std::find_if(available.begin(), available.end(), [&](auto avail) { return std::string_view(f(avail)) == ext; }) !=
             available.end();
    });
  };

  auto supportedExtensions = ctx.enumerateInstanceExtensionProperties();
  auto supportedLayers = ctx.enumerateInstanceLayerProperties();
  insertSupported(commonExtensions, extensions, supportedExtensions, [](auto e) { return e.extensionName; });
  insertSupported(extraExtensions, extensions, supportedExtensions, [](auto e) { return e.extensionName; });
  insertSupported(extraLayers, layers, supportedLayers, [](auto e) { return e.layerName; });

  vk::InstanceCreateInfo info(vk::InstanceCreateFlags(), &AppInfo, //
                              layers.size(), layers.data(),        //
                              extensions.size(), extensions.data() //
  );

  return {ctx, info};
}

VulkanPlatform::VulkanPlatform() : context(), instance(createInstance(extensions, layers, context)) { TRACE(); }
std::string VulkanPlatform::name() {
  TRACE();
  return "Vulkan";
}
std::vector<Property> VulkanPlatform::properties() {
  TRACE();
  return {};
}

PlatformKind VulkanPlatform::kind() {
  TRACE();
  return PlatformKind::Managed;
}
ModuleFormat VulkanPlatform::moduleFormat() {
  TRACE();
  return ModuleFormat::SPIRV;
}

template <typename T, typename U, typename F> constexpr static auto transform_idx_if(U &from, F &&f) {
  std::vector<typename decltype(f(from[0], T(0)))::value_type> out;
  for (T i = 0; i < from.size(); ++i) {
    if (auto maybe = f(from[i], i); maybe) out.push_back(*maybe);
  }
  return out;
};

std::vector<std::unique_ptr<Device>> VulkanPlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> devices;
  for (const vk::raii::PhysicalDevice &dev : instance.enumeratePhysicalDevices()) {
    std::vector<vk::QueueFamilyProperties> queueProps = dev.getQueueFamilyProperties();

    auto computeQueueIds = transform_idx_if<uint32_t>(queueProps, [](auto &q, auto i) {
      return q.queueCount > 0 && q.queueFlags & vk::QueueFlagBits::eCompute ? std::optional{std::pair{i, q.queueCount}} : std::nullopt;
    });

    auto transferQueueIds = transform_idx_if<uint32_t>(queueProps, [](auto &q, auto i) {
      return q.queueCount > 0 && q.queueFlags & vk::QueueFlagBits::eTransfer ? std::optional{std::pair{i, q.queueCount}} : std::nullopt;
    });

    // XXX VK_QUEUE_COMPUTE_BIT implies VK_QUEUE_TRANSFER_BIT, see
    //   https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkQueueFlagBits.html
    if (!computeQueueIds.empty()) {
      // Typically, we get the following queues
      //  0 : eCompute + eTransfer + ...
      //  1: eTransfer + ...
      //  2: eTransfer + ...
      // The same queue can support both xfer and compute, so we use different queues for each if possible.
      // We first default to compute, then try to pick out a distinct transfer queue if we can find one.
      auto computeQueueId = computeQueueIds[0];
      auto transferQueueId = computeQueueId;
      for (auto xferId : transferQueueIds)
        if (xferId.first != computeQueueId.first) {
          transferQueueId = xferId;
          break;
        }
      float priority = 1.0f;
      std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
      queueCreateInfos.reserve(2);
      queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), computeQueueId.first, 1, &priority);
      if (computeQueueId != transferQueueId)
        queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), transferQueueId.first, 1, &priority);
      devices.push_back(std::make_unique<VulkanDevice>(instance, computeQueueId, transferQueueId, dev));
    }
  }

  return devices;
}

VmaAllocator createAllocator(vk::raii::Instance &instance, vk::raii::PhysicalDevice &dev, vk::raii::Device &ctx) {
  VmaVulkanFunctions vulkanFunctions = {};
  vulkanFunctions.vkGetInstanceProcAddr = instance.getDispatcher()->vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = instance.getDispatcher()->vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocatorCreateInfo = {};
  allocatorCreateInfo.vulkanApiVersion = AppInfo.apiVersion;
  allocatorCreateInfo.physicalDevice = *dev;
  allocatorCreateInfo.device = *ctx;
  allocatorCreateInfo.instance = *instance;
  allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
  VmaAllocator allocator;
  vmaCreateAllocator(&allocatorCreateInfo, &allocator);
  return allocator;
}

static vk::raii::Device createDevice(const vk::raii::PhysicalDevice &dev, uint32_t computeQueueId, uint32_t transferQueueId) {
  float priority = 1.0f;
  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  queueCreateInfos.reserve(2);
  queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), computeQueueId, 1, &priority);
  if (computeQueueId != transferQueueId) queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), transferQueueId, 1, &priority);

  auto f = dev.getFeatures();
  return dev.createDevice({{}, queueCreateInfos, {}, {}, &f});
}

// ---

Resolved::Resolved(uint32_t computeQueueId,
                   const std::shared_ptr<vk::raii::ShaderModule> &shaderModule, //
                   const std::vector<vk::DescriptorSetLayoutBinding> &bindings, //
                   const std::vector<vk::DescriptorPoolSize> &sizes,
                   vk::raii::Device &ctx)                       //
    : shaderModule(shaderModule),                               //
      dscLayout(ctx.createDescriptorSetLayout({{}, bindings})), //
      dscPool(ctx.createDescriptorPool(
          vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, sizes))), //
      dscSet(std::move(ctx.allocateDescriptorSets({*dscPool, *dscLayout})[0])), pipeCache({}),                                           //
      pipeLayout(ctx.createPipelineLayout(vk::PipelineLayoutCreateInfo{vk::PipelineLayoutCreateFlags(), 1, &*dscLayout, 0, nullptr})),

      cmdPool(ctx.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, computeQueueId})), //
      cmdBuffer(std::move(ctx.allocateCommandBuffers({*cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0]))

{}

VulkanDevice::VulkanDevice(vk::raii::Instance &instance,              //
                           decltype(computeQueueId) computeQueueId,   //
                           decltype(transferQueueId) transferQueueId, //
                           decltype(device) device_)
    : computeQueueId(computeQueueId),   //
      transferQueueId(transferQueueId), //
      device(std::move(device_)),       //
      ctx(createDevice(device, computeQueueId.first, transferQueueId.first)), allocator(createAllocator(instance, device, ctx)),
      transferCmdPool(std::make_shared<vk::raii::CommandPool>(
          ctx.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, transferQueueId.first}))), //
      transferCmdBuffer(std::make_shared<vk::raii::CommandBuffer>(
          std::move(ctx.allocateCommandBuffers({**transferCmdPool, vk::CommandBufferLevel::ePrimary, 1})[0]))),
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            TRACE();
            auto data = std::vector<uint32_t>((s.size() + 3) / 4, 0);
            std::copy(s.begin(), s.end(), reinterpret_cast<char *>(data.data()));
            return std::make_shared<vk::raii::ShaderModule>(
                ctx.createShaderModule({vk::ShaderModuleCreateFlags(), sizeof(uint32_t) * data.size(), data.data()}));
          },
          [this](auto &&m, auto &&name, auto &&types) {
            TRACE();
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            uint32_t bindingsId = 0;
            size_t storages = 0;
            size_t scalars = 0;
            for (auto tpe : types) {
              if (tpe == Type::Ptr) {
                bindings.emplace_back(bindingsId++, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
                storages++;
              } else if (tpe != Type::Void)
                scalars++;
            }
            if (scalars != 0) bindings.emplace_back(bindingsId, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute);

            std::vector<vk::DescriptorPoolSize> sizes;
            if (storages != 0) sizes.emplace_back(vk::DescriptorType::eStorageBuffer, storages);
            if (scalars != 0) sizes.emplace_back(vk::DescriptorType::eUniformBuffer, scalars);

            return Resolved(this->computeQueueId.first, m, bindings, sizes, ctx);
          }) {
  TRACE();
}

int64_t VulkanDevice::id() {
  TRACE();
  auto uuid = device.getProperties().pipelineCacheUUID;
  std::string data(uuid.begin(), uuid.end());
  return int64_t(std::hash<std::string>{}(data));
}
std::string VulkanDevice::name() {
  TRACE();
  return device.getProperties().deviceName;
}
bool VulkanDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
bool VulkanDevice::singleEntryPerModule() {
  TRACE();
  return true;
}
std::vector<Property> VulkanDevice::properties() {
  TRACE();
  const auto props = device.getProperties();

  auto apiVersion = props.apiVersion;

  return {
      {"apiVersion", std::to_string(VK_VERSION_MAJOR(apiVersion)) + "." +     //
                         std::to_string(VK_VERSION_MINOR(apiVersion)) + "." + //
                         std::to_string(VK_VERSION_PATCH(apiVersion))},       //
      {"driverVersion", std::to_string(props.driverVersion)},                 //
      {"vendorID", std::to_string(props.vendorID)},                           //
      {"deviceID", std::to_string(props.deviceID)},                           //
      {"deviceType", vk::to_string(props.deviceType)},                        //
      {"deviceName", props.deviceName},                                       //
      {"pipelineCacheUUID", std::to_string(id())},                            //
      //      { "limits",   vk::to_string(props.limits           ) },
      //      { "sparseProperties",   vk::to_string(props.sparseProperties ) },
  };
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

static MemObject allocate(VmaAllocator &allocator, size_t size, bool uniform) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = (uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo allocCreateInfo = {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer buffer{};
  VmaAllocation allocation{};
  VmaAllocationInfo allocInfo;
  vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, &allocInfo);
  return {vk::Buffer(buffer), allocation, allocInfo.pMappedData, size};
}

uintptr_t VulkanDevice::mallocDevice(size_t size, Access) {
  TRACE();
  return memoryObjects.malloc(std::make_shared<MemObject>(allocate(allocator, size, false)));
}
void VulkanDevice::freeDevice(uintptr_t ptr) {
  TRACE();
  if (auto obj = memoryObjects.query(ptr); obj) {
    vmaDestroyBuffer(allocator, VkBuffer((*obj)->buffer), (*obj)->allocation);
    memoryObjects.erase(ptr);
  } else
    throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
}

std::optional<void *> VulkanDevice::mallocShared(size_t size, Access access) {
  TRACE();
  return std::nullopt;
}
void VulkanDevice::freeShared(void *ptr) {
  TRACE();
  throw std::runtime_error("Unsupported");
}

std::unique_ptr<DeviceQueue> VulkanDevice::createQueue() {
  TRACE();
  return std::make_unique<VulkanDeviceQueue>(
      ctx, allocator, //
      ctx.getQueue(computeQueueId.first, activeComputeQueues++ % computeQueueId.second),
      ctx.getQueue(transferQueueId.first, activeTransferQueues++ % transferQueueId.second), store,
      [this](auto &&ptr) {
        if (auto mem = memoryObjects.query(ptr); mem) {
          return *mem;
        } else
          throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
      } //
  );
}
VulkanDevice::~VulkanDevice() {
  TRACE();
  vmaDestroyAllocator(allocator);
}

// ---

VulkanDeviceQueue::VulkanDeviceQueue(decltype(ctx) ctx, decltype(allocator) allocator,
                                     decltype(computeQueue) computeQueue,   //
                                     decltype(transferQueue) transferQueue, //
                                     decltype(store) store,                 //
                                     decltype(queryMemObject) queryMemObject)
    : ctx(ctx), allocator(allocator),                          //
      computeQueue(std::move(computeQueue)),                   //
      transferQueue(std::move(transferQueue)),                 //
      store(store), queryMemObject(std::move(queryMemObject)), //
      callbackQueue(), callbackThread([this]() {
        while (true) {
          auto [f, keepGoing] = callbackQueue.pop();
          //          std::cout << "Task! " << f.has_value() << " " << keepGoing << std::endl;
          if (f) (*f)();
          if (!keepGoing) break;
        }
        //        std::cout << "Thread stop" << std::endl;
      }) {
  TRACE();
  callbackThread.detach();
}
VulkanDeviceQueue::~VulkanDeviceQueue() {
  TRACE();
  callbackQueue.terminate();
}
void VulkanDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  callbackQueue.push([cb]() {
    if (cb) (*cb)();
  });
}

void VulkanDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  auto obj = queryMemObject(dst);
  std::memcpy(obj->mappedData, src, size);
  vmaInvalidateAllocation(allocator, obj->allocation, 0, size);
  if (cb) (*cb)();
}
void VulkanDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  auto obj = queryMemObject(src);
  std::memcpy(dst, obj->mappedData, size);
  vmaInvalidateAllocation(allocator, obj->allocation, 0, size);
  if (cb) (*cb)();
}
void VulkanDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                           std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void) throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");

  // pointers are uniforms sharing the same descriptor set with monotonically increasing binding
  // anything that's scalar goes into a struct and added as the last binding of the same descriptor set
  // See https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md#example-descriptor-set-mapping

  auto args = detail::argDataAsPointers(types, argData);

  std::vector<std::pair<vk::DescriptorBufferInfo, vk::DescriptorType>> infos;
  size_t argBufferSize = 0;

  {
    static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
    for (size_t i = 0; i < types.size() - 1; ++i) {
      auto rawPtr = args[i];
      auto tpe = types[i];
      if (tpe == Type::Scratch) continue;
      if (tpe != Type::Ptr) {
        argBufferSize += byteOfType(tpe);
      } else {
        uintptr_t ptr = {};
        std::memcpy(&ptr, rawPtr, byteOfType(Type::Ptr));
        const auto obj = queryMemObject(ptr);
        infos.emplace_back(vk::DescriptorBufferInfo{obj->buffer, 0, obj->size}, vk::DescriptorType::eStorageBuffer);
      }
    }
  }

  size_t scratchCount = 0;
  auto argObj = argBufferSize == 0 ? nullptr : std::make_shared<MemObject>(allocate(allocator, argBufferSize, true));
  if (argObj) {
    auto *argPtr = static_cast<std::byte *>(argObj->mappedData);
    for (size_t i = 0; i < types.size() - 1; ++i) {
      auto tpe = types[i];
      if (tpe == Type::Scratch) scratchCount++;
      if (tpe == Type::Ptr || tpe == Type::Scratch) continue;
      std::memcpy(argPtr, args[i], byteOfType(tpe));
      argPtr += byteOfType(tpe);
    }
    infos.emplace_back(vk::DescriptorBufferInfo{argObj->buffer, 0, argObj->size}, vk::DescriptorType::eUniformBuffer);
  }
  if (scratchCount > 1) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Only a single scratch buffer is supported, " + std::to_string(scratchCount) +
                           " requested.");
  }

  auto &fn = store.resolveFunction(moduleName, symbol, types);
  const auto [local, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  const auto global = Dim3{policy.global.x, policy.global.y, policy.global.z};
  auto specEntries = std::vector<vk::SpecializationMapEntry>{
      {0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
      {1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
      {2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
  };
  auto specValues = std::vector<uint32_t>{uint32_t(local.x), uint32_t(local.y), uint32_t(local.z)};
  if (scratchCount > 0) {
    specEntries.emplace_back(3, 3 * sizeof(uint32_t), sizeof(uint32_t));
    specValues.emplace_back(uint32_t(sharedMem));
  }

  vk::SpecializationInfo specInfo(specEntries.size(), specEntries.data(), //
                                  specValues.size() * sizeof(uint32_t), specValues.data());

  auto pipe = ctx.createComputePipeline( //
      fn.pipeCache,                      //
      vk::ComputePipelineCreateInfo(     //
          {},                            //
          {{}, vk::ShaderStageFlagBits::eCompute, **fn.shaderModule, symbol.c_str(), &specInfo},
          *fn.pipeLayout //
          )              //
  );

  std::vector<vk::WriteDescriptorSet> writeDsSets;
  {
    for (uint32_t i = 0; i < infos.size(); ++i)
      writeDsSets.emplace_back(*fn.dscSet, i, 0, 1, infos[i].second, nullptr, &infos[i].first);
  }

  ctx.updateDescriptorSets(writeDsSets, {});

  auto beginInfo = vk::CommandBufferBeginInfo();
  fn.cmdBuffer.begin(beginInfo);
  fn.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe);
  fn.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *fn.pipeLayout, 0, *fn.dscSet, {});

  fn.cmdBuffer.dispatch(global.x, global.y, global.z);
  fn.cmdBuffer.end();

  if (cb) {
    auto [key, enqueued] = enqueuedStore.store(
        std::make_shared<Enqueued>(Enqueued{std::move(pipe), ctx.createFence(vk::FenceCreateInfo()), std::move(argObj)}));

    computeQueue.submit(vk::SubmitInfo{{}, {}, *fn.cmdBuffer}, *(enqueued->fence));
    callbackQueue.push([this, &fn, cb, key = key]() {
      if (auto enqueued_ = enqueuedStore.find(key); enqueued_) {
        auto result = ctx.waitForFences({*(*enqueued_)->fence}, true, uint64_t(-1));
        fn.cmdBuffer.reset();
        fn.cmdPool.reset();
        enqueuedStore.erase(key);
        if (auto argObj = (*enqueued_)->argObject; argObj) {
          vmaDestroyBuffer(allocator, VkBuffer(argObj->buffer), argObj->allocation);
        }
      }
      (*cb)();
    });
  } else {
    auto fence = ctx.createFence(vk::FenceCreateInfo());
    computeQueue.submit(vk::SubmitInfo{{}, {}, *fn.cmdBuffer}, *fence);
    auto r = ctx.waitForFences({*fence}, true, uint64_t(-1));
    fn.cmdPool.reset();
    if (argObj) {
      vmaDestroyBuffer(allocator, VkBuffer(argObj->buffer), argObj->allocation);
    }
  }
}

#undef CHECKED

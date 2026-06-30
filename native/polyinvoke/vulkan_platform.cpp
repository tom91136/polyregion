#include "polyinvoke/vulkan_platform.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"
#include "spirv/unified1/spirv.hpp"

#ifndef _MSC_VER
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wcast-align"
  #pragma clang diagnostic ignored "-Wmissing-field-initializers"
#endif

#define VMA_IMPLEMENTATION
#include "vendor_utils.h"
#include "vk_mem_alloc.h"

#ifndef _MSC_VER
  #pragma clang diagnostic pop // -Wno-everything
#endif

using namespace polyregion::invoke;
using namespace polyregion::invoke::vulkan;

static constexpr const char *PREFIX = "Vulkan";

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// XXX lavapipe mis-vectorises a multi-subgroup work-group once the per-invocation private footprint is large
static constexpr uint32_t lavapipeMaxFunctionPrivateBytes = 768;

// total bytes of Function-storage OpVariables - the kernel's per-invocation private (stack) memory
static uint32_t spirvFunctionPrivateBytes(const std::vector<uint32_t> &w) {
  if (w.size() < 5) return 0;
  auto op = [](uint32_t x) { return uint16_t(x & 0xFFFF); };
  auto wc = [](uint32_t x) { return uint16_t(x >> 16); };
  struct Inst {
    const uint32_t *p;
    uint16_t n;
  };
  std::vector<Inst> insts;
  for (size_t i = 5; i < w.size();) {
    const uint16_t n = wc(w[i]);
    if (n == 0 || i + n > w.size()) break;
    insts.push_back({&w[i], n});
    i += n;
  }
  std::unordered_map<uint32_t, uint32_t> typeSize, constVal, ptrPointee;
  for (const auto &in : insts)
    if (op(in.p[0]) == spv::OpConstant && in.n >= 4) constVal[in.p[2]] = in.p[3];
  for (bool more = true; more;) {
    more = false;
    for (const auto &in : insts) {
      const uint32_t *p = in.p;
      const uint16_t n = in.n, o = op(p[0]);
      const uint32_t id = n > 1 ? p[1] : 0;
      if (!id || typeSize.count(id)) continue;
      auto put = [&](uint32_t v) { typeSize[id] = v, more = true; };
      if ((o == spv::OpTypeInt || o == spv::OpTypeFloat) && n >= 3) put(p[2] / 8);
      else if (o == spv::OpTypeBool) put(1);
      else if ((o == spv::OpTypeVector || o == spv::OpTypeMatrix) && typeSize.count(p[2])) put(typeSize[p[2]] * p[3]);
      else if (o == spv::OpTypeArray && typeSize.count(p[2]) && constVal.count(p[3])) put(typeSize[p[2]] * constVal[p[3]]);
      else if (o == spv::OpTypeStruct) {
        uint32_t s = 0;
        bool all = true;
        for (uint16_t m = 2; m < n; ++m)
          if (!typeSize.count(p[m])) {
            all = false;
            break;
          } else s += typeSize[p[m]];
        if (all) put(s);
      } else if (o == spv::OpTypePointer && n >= 4) {
        ptrPointee[id] = p[3];
        if (typeSize.count(p[3])) put(8);
      }
    }
  }
  uint32_t total = 0;
  for (const auto &in : insts) {
    const uint32_t *p = in.p;
    if (op(p[0]) == spv::OpVariable && in.n >= 4 && p[3] == spv::StorageClassFunction) {
      const auto it = ptrPointee.find(p[1]);
      if (it != ptrPointee.end() && typeSize.count(it->second)) total += typeSize[it->second];
    }
  }
  return total;
}

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

// SPIR-V 1.4 (the LLVM backend's GLCompute default) needs a Vulkan >= 1.2 app; lavapipe silently runs
// a no-op shader otherwise (writes never land), so declare 1.2 to stay in-spec everywhere
static vk::ApplicationInfo AppInfo(__FILE__, 1, nullptr, 0, VK_API_VERSION_1_2);

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
std::variant<std::string, std::unique_ptr<Platform>> VulkanPlatform::create() {
  try {
    return std::unique_ptr<Platform>(new VulkanPlatform());
  } catch (const std::exception &e) {
    return std::string("Vulkan: ") + e.what();
  }
}
VulkanPlatform::VulkanPlatform() : context(), instance(createInstance(extensions, layers, context)) { POLYINVOKE_TRACE(); }
std::string VulkanPlatform::name() {
  POLYINVOKE_TRACE();
  return "Vulkan";
}
std::vector<Property> VulkanPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}

PlatformKind VulkanPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat VulkanDevice::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::SPIRV_GLCompute;
}

template <typename T, typename U, typename F> constexpr static auto transform_idx_if(U &from, F &&f) {
  std::vector<typename decltype(f(from[0], T(0)))::value_type> out;
  for (T i = 0; i < from.size(); ++i) {
    if (auto maybe = f(from[i], i); maybe) out.push_back(*maybe);
  }
  return out;
};

std::vector<std::unique_ptr<Device>> VulkanPlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> devices;
  std::vector<vk::raii::PhysicalDevice> physicalDevices;
  try {
    physicalDevices = instance.enumeratePhysicalDevices();
  } catch (const vk::SystemError &) {
    return devices;
  }
  for (const vk::raii::PhysicalDevice &dev : physicalDevices) {
    std::vector<vk::QueueFamilyProperties> queueProps = dev.getQueueFamilyProperties();

    auto computeQueueIds = transform_idx_if<uint32_t>(queueProps, [](auto &q, auto i) {
      return q.queueCount > 0 && q.queueFlags & vk::QueueFlagBits::eCompute ? std::optional{std::pair{i, q.queueCount}} : std::nullopt;
    });

    auto transferQueueIds = transform_idx_if<uint32_t>(queueProps, [](auto &q, auto i) {
      return q.queueCount > 0 && q.queueFlags & vk::QueueFlagBits::eTransfer ? std::optional{std::pair{i, q.queueCount}} : std::nullopt;
    });

    // VK_QUEUE_COMPUTE_BIT implies VK_QUEUE_TRANSFER_BIT
    if (!computeQueueIds.empty()) {
      // default transfer to the compute queue, then prefer a distinct transfer queue if one exists
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

  auto features = dev.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features>();
  vk::DeviceCreateInfo info{{}, queueCreateInfos};
  info.pNext = &features.get<vk::PhysicalDeviceFeatures2>();
  return dev.createDevice(info);
}

// ---

details::Resolved::Resolved(uint32_t computeQueueId,
                            const std::shared_ptr<vk::raii::ShaderModule> &shaderModule, //
                            uint32_t maxWorkGroupX,                                      //
                            const std::vector<vk::DescriptorSetLayoutBinding> &bindings, //
                            const std::vector<vk::DescriptorPoolSize> &sizes,
                            const vk::raii::Device &ctx)        //
    : shaderModule(shaderModule), maxWorkGroupX(maxWorkGroupX), //
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
      deviceMaxAllocSize(device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMaintenance3Properties>()
                             .get<vk::PhysicalDeviceMaintenance3Properties>()
                             .maxMemoryAllocationSize),
      deviceMaxWorkGroupInvocations(device.getProperties().limits.maxComputeWorkGroupInvocations),
      lavapipe(std::string(device.getProperties().deviceName.data()).find("llvmpipe") != std::string::npos),
      subgroupSize(device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>()
                       .get<vk::PhysicalDeviceSubgroupProperties>()
                       .subgroupSize),
      transferCmdPool(std::make_shared<vk::raii::CommandPool>(
          ctx.createCommandPool({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, transferQueueId.first}))), //
      transferCmdBuffer(std::make_shared<vk::raii::CommandBuffer>(
          std::move(ctx.allocateCommandBuffers({**transferCmdPool, vk::CommandBufferLevel::ePrimary, 1})[0]))),
      store(
          PREFIX,
          [this](auto &&s) {
            POLYINVOKE_TRACE();
            auto data = std::vector<uint32_t>((s.size() + 3) / 4, 0);
            std::copy(s.begin(), s.end(), reinterpret_cast<char *>(data.data()));
            auto module = std::make_shared<vk::raii::ShaderModule>(
                ctx.createShaderModule({vk::ShaderModuleCreateFlags(), sizeof(uint32_t) * data.size(), data.data()}));
            uint32_t maxWgX = UINT32_MAX;
            if (lavapipe && spirvFunctionPrivateBytes(data) > lavapipeMaxFunctionPrivateBytes) maxWgX = subgroupSize;
            return details::LoadedModule{module, maxWgX};
          },
          [this](auto &&m, auto &&name, auto &&types) {
            POLYINVOKE_TRACE();
            std::vector<vk::DescriptorSetLayoutBinding> bindings;
            uint32_t bindingsId = 0;
            size_t storages = 0;
            size_t scalars = 0;
            for (auto tpe : types) {
              if (tpe == Type::Ptr) {
                bindings.emplace_back(bindingsId++, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
                storages++;
              } else if (tpe != Type::Void) scalars++;
            }
            if (scalars != 0) bindings.emplace_back(bindingsId, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute);

            std::vector<vk::DescriptorPoolSize> sizes;
            if (storages != 0) sizes.emplace_back(vk::DescriptorType::eStorageBuffer, storages);
            if (scalars != 0) sizes.emplace_back(vk::DescriptorType::eUniformBuffer, scalars);

            return details::Resolved(this->computeQueueId.first, m.module, m.maxWorkGroupX, bindings, sizes, ctx);
          }) {
  POLYINVOKE_TRACE();
}

int64_t VulkanDevice::id() {
  POLYINVOKE_TRACE();
  auto uuid = device.getProperties().pipelineCacheUUID;
  std::string data(uuid.begin(), uuid.end());
  return int64_t(std::hash<std::string>{}(data));
}
std::string VulkanDevice::name() {
  POLYINVOKE_TRACE();
  return device.getProperties().deviceName;
}
PhysicalDevice VulkanDevice::physicalDevice() {
  POLYINVOKE_TRACE();
  const auto exts = device.enumerateDeviceExtensionProperties();
  const bool hasPci = std::any_of(exts.begin(), exts.end(), [](const vk::ExtensionProperties &e) {
    return std::strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0;
  });
  if (hasPci) {
    const auto chain = device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
    const auto &pci = chain.template get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
    return PhysicalDevice::pci(pci.pciDomain, static_cast<uint8_t>(pci.pciBus), static_cast<uint8_t>(pci.pciDevice),
                               static_cast<uint8_t>(pci.pciFunction));
  }
  const auto chain = device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
  const auto &devUuid = chain.template get<vk::PhysicalDeviceIDProperties>().deviceUUID;
  std::array<uint8_t, 16> uuid{};
  std::memcpy(uuid.data(), devUuid.data(), uuid.size());
  return PhysicalDevice::uuid(uuid);
}
bool VulkanDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
PagingMode VulkanDevice::pagingMode() {
  POLYINVOKE_TRACE();
  return PagingMode::None; // no USM/managed-memory model
}
bool VulkanDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return true;
}
size_t VulkanDevice::maxThreadsPerBlock() {
  POLYINVOKE_TRACE();
  return deviceMaxWorkGroupInvocations;
}
std::vector<Property> VulkanDevice::properties() {
  POLYINVOKE_TRACE();
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
  };
}
std::vector<std::string> VulkanDevice::features() {
  POLYINVOKE_TRACE();
  std::vector<std::string> out{"vulkan", "spirv_glcompute", normaliseVendor(device.getProperties().deviceName)};
  const auto f2 = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features>();
  const auto &f = f2.get<vk::PhysicalDeviceFeatures2>().features;
  if (f.shaderFloat64) out.emplace_back("fp64");
  if (f.shaderInt64) out.emplace_back("int64");
  if (f.shaderInt16) out.emplace_back("int16");
  if (f2.get<vk::PhysicalDeviceShaderFloat16Int8Features>().shaderFloat16) out.emplace_back("fp16");
  out.emplace_back(fmt::format("paging:{}", magic_enum::enum_name(pagingMode())));
  return out;
}
void VulkanDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool VulkanDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}

static details::MemObject allocate(VmaAllocator &allocator, size_t size, bool uniform) {
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
  if (const VkResult r = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, &allocInfo); r != VK_SUCCESS)
    POLYINVOKE_FATAL(PREFIX, "vmaCreateBuffer failed for %zu bytes (VkResult %d); likely exceeds the device maxMemoryAllocationSize", size,
                     static_cast<int>(r));
  return {vk::Buffer(buffer), allocation, allocInfo.pMappedData, size};
}

uintptr_t VulkanDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  // XXX a buffer past maxMemoryAllocationSize is unaddressable (lavapipe's SSBO byte offset is 32-bit -> reads
  // past ~2GB silently return 0), but the alloc itself does not fail, so check explicitly here.
  // maxStorageBufferRange is not a reliable bound - lavapipe reports 128MB yet binds far larger ranges fine
  if (size > deviceMaxAllocSize)
    POLYINVOKE_FATAL(PREFIX,
                     "device buffer of %zu bytes exceeds maxMemoryAllocationSize (%zu); the marshalling arena is too large for this device",
                     size, deviceMaxAllocSize);
  return memoryObjects.malloc(std::make_shared<details::MemObject>(allocate(allocator, size, false)));
}
void VulkanDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  if (auto obj = memoryObjects.query(ptr); obj) {
    vmaDestroyBuffer(allocator, VkBuffer((*obj)->buffer), (*obj)->allocation);
    memoryObjects.erase(ptr);
  } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %lu", ptr);
}

std::optional<void *> VulkanDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return std::nullopt;
}
void VulkanDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  POLYINVOKE_FATAL(PREFIX, "Unsupported: %p", ptr);
}

std::unique_ptr<DeviceQueue> VulkanDevice::createQueue(const std::chrono::duration<int64_t> &) {
  POLYINVOKE_TRACE();
  return std::make_unique<VulkanDeviceQueue>(ctx, allocator, //
                                             ctx.getQueue(computeQueueId.first, activeComputeQueues++ % computeQueueId.second),
                                             ctx.getQueue(transferQueueId.first, activeTransferQueues++ % transferQueueId.second), store,
                                             [this](auto &&ptr) {
                                               if (auto mem = memoryObjects.query(ptr); mem) {
                                                 return *mem;
                                               } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %lu", ptr);
                                             } //
  );
}
VulkanDevice::~VulkanDevice() {
  POLYINVOKE_TRACE();
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
          if (f) (*f)();
          if (!keepGoing) break;
        }
      }) {
  POLYINVOKE_TRACE();
}
VulkanDeviceQueue::~VulkanDeviceQueue() {
  POLYINVOKE_TRACE();
  callbackQueue.terminate();
  if (callbackThread.joinable()) callbackThread.join();
}
void VulkanDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  callbackQueue.push([cb]() {
    if (cb) (*cb)();
  });
}
void VulkanDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                   const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  const auto dstObj = queryMemObject(dst);
  const auto srcObj = queryMemObject(src);
  std::memcpy(static_cast<char *>(dstObj->mappedData) + dstOffset, static_cast<char *>(srcObj->mappedData) + srcOffset, size);
  vmaInvalidateAllocation(allocator, dstObj->allocation, dstOffset, size);
  if (cb) (*cb)();
}
void VulkanDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  const auto dstObj = queryMemObject(dst);
  std::memcpy(static_cast<char *>(dstObj->mappedData) + dstOffset, src, size);
  vmaInvalidateAllocation(allocator, dstObj->allocation, dstOffset, size);
  if (cb) (*cb)();
}
void VulkanDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  const auto srcObj = queryMemObject(src);
  std::memcpy(dst, static_cast<char *>(srcObj->mappedData) + srcOffset, size);
  vmaInvalidateAllocation(allocator, srcObj->allocation, srcOffset, size);
  if (cb) (*cb)();
}
void VulkanDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                           std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void)
    POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", magic_enum::enum_name(types.back()).data());

  // pointers bind as storage descriptors in arg order; scalars share one trailing uniform block
  // (clspv OpenCLCOnVulkan descriptor-set-mapping)
  auto args = detail::argDataAsPointers(types, argData);

  static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
  // the scalar uniform block uses runtime::std140ScalarLayout - the SAME rule codegen pads with, so a
  // strict driver (lavapipe) reads each 64-bit member at its 8-aligned offset; host and device MUST agree
  std::vector<std::pair<vk::DescriptorBufferInfo, vk::DescriptorType>> infos;
  std::vector<size_t> scalarSizes;
  std::vector<const void *> scalarSrcs;
  size_t scratchCount = 0;
  for (size_t i = 0; i < types.size() - 1; ++i) {
    const auto tpe = types[i];
    if (tpe == Type::Scratch) scratchCount++;
    else if (tpe == Type::Ptr) {
      uintptr_t ptr = {};
      std::memcpy(&ptr, args[i], byteOfType(Type::Ptr));
      const auto obj = queryMemObject(ptr);
      infos.emplace_back(vk::DescriptorBufferInfo{obj->buffer, 0, obj->size}, vk::DescriptorType::eStorageBuffer);
    } else {
      scalarSizes.push_back(byteOfType(tpe));
      scalarSrcs.push_back(args[i]);
    }
  }
  const auto [scalarOffsets, argBufferSize] = polyregion::runtime::std140ScalarLayout(scalarSizes);

  auto argObj = argBufferSize == 0 ? nullptr : std::make_shared<details::MemObject>(allocate(allocator, argBufferSize, true));
  if (argObj) {
    auto *argBase = static_cast<std::byte *>(argObj->mappedData);
    for (size_t i = 0; i < scalarSizes.size(); ++i)
      std::memcpy(argBase + scalarOffsets[i], scalarSrcs[i], scalarSizes[i]);
    infos.emplace_back(vk::DescriptorBufferInfo{argObj->buffer, 0, argObj->size}, vk::DescriptorType::eUniformBuffer);
  }
  if (scratchCount > 1) POLYINVOKE_FATAL(PREFIX, "Only a single scratch buffer is supported, %zu requested", scratchCount);
  auto &fn = store.resolveFunction(moduleName, symbol, types);
  const auto [local, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  const auto global = Dim3{policy.global.x, policy.global.y, policy.global.z};
  auto specEntries = std::vector<vk::SpecializationMapEntry>{
      {0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
      {1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
      {2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
  };
  auto specValues = std::vector<uint32_t>{uint32_t(local.x), uint32_t(local.y), uint32_t(local.z)};
  specValues[0] = std::min(specValues[0], fn.maxWorkGroupX);
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
        std::make_shared<details::Enqueued>(details::Enqueued{std::move(pipe), ctx.createFence(vk::FenceCreateInfo()), std::move(argObj)}));

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
void VulkanDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  ctx.waitIdle();
}

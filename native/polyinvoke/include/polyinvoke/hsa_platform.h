#pragma once

#include <condition_variable>

#include "polyregion/compat.h"

#include "hsaew.h"
#include "runtime.h"

namespace polyregion::invoke::hsa {

class POLYREGION_EXPORT HsaPlatform final : public Platform {
  POLYREGION_EXPORT explicit HsaPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~HsaPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace details {
using SymbolArgOffsetTable = std::unordered_map<std::string, std::vector<size_t>>;
using HsaModuleStore = detail::ModuleStore<                               //
    std::pair<hsa_executable_t, SymbolArgOffsetTable>,                    //
    std::pair<hsa_executable_symbol_t, SymbolArgOffsetTable::mapped_type> //
    >;
} // namespace details

class POLYREGION_EXPORT HsaDeviceQueue;

class POLYREGION_EXPORT HsaDevice final : public Device {

  uint32_t queueSize;
  hsa_agent_t hostAgent;
  hsa_agent_t agent;
  std::string deviceName;

  hsa_region_t kernelArgRegion{};
  hsa_amd_memory_pool_t deviceGlobalRegion{};

  details::HsaModuleStore store; // store needs to be dropped before dropping device

  friend class HsaDeviceQueue;

public:
  POLYREGION_EXPORT HsaDevice(uint32_t queueSize, hsa_agent_t hostAgent, hsa_agent_t agent);
  POLYREGION_EXPORT ~HsaDevice() override;
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

class POLYREGION_EXPORT HsaDeviceQueue final : public DeviceQueue {

  detail::CountingLatch latch;

  HsaDevice &device;
  hsa_queue_t *queue;

public:
  POLYREGION_EXPORT explicit HsaDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(device) device, decltype(queue) queue);
  POLYREGION_EXPORT ~HsaDeviceQueue() override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

} // namespace polyregion::invoke::hsa

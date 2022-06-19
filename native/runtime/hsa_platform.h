#pragma once

#include "hsaew.h"
#include "runtime.h"

namespace polyregion::runtime::hsa {

class EXPORT HsaPlatform : public Platform {
public:
  EXPORT explicit HsaPlatform();
  EXPORT ~HsaPlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

namespace {
using HsaModuleStore = detail::ModuleStore<hsa_executable_t, hsa_executable_symbol_t>;
}

class EXPORT HsaDeviceQueue;

class EXPORT HsaDevice : public Device {

  uint32_t queueSize;
  hsa_agent_t hostAgent;
  hsa_agent_t agent;
  std::string deviceName;

  hsa_region_t kernelArgRegion{};
  hsa_amd_memory_pool_t deviceGlobalRegion{};

  HsaModuleStore store; // store needs to be dropped before dropping device

  friend class HsaDeviceQueue;

public:
  EXPORT HsaDevice(uint32_t queueSize, hsa_agent_t hostAgent, hsa_agent_t agent);
  EXPORT ~HsaDevice() override;
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT bool sharedAddressSpace() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::string> features() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT HsaDeviceQueue : public DeviceQueue {

  HsaDevice &device;
  hsa_queue_t *queue;
  hsa_signal_t kernelSignal{};
  hsa_signal_t hostToDeviceSignal{};
  hsa_signal_t deviceToHostSignal{};

  void enqueueCallback(hsa_signal_t &signal, const Callback &cb);

public:
  EXPORT explicit HsaDeviceQueue(decltype(device) device, decltype(queue) queue);
  EXPORT ~HsaDeviceQueue() override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::hsa

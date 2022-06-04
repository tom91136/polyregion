#pragma once

#include <atomic>

#include "dl.h"
#include "runtime.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"

namespace polyregion::runtime::object {

class EXPORT ObjectDevice : public Device {
public:
  EXPORT int64_t id() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT bool sharedAddressSpace() override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
};

class EXPORT ObjectDeviceQueue : public DeviceQueue {
public:
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const MaybeCallback &cb) override;
};
namespace {
using ObjectModules = std::unordered_map<std::string, std::unique_ptr<llvm::object::ObjectFile>>;
}

class EXPORT RelocatablePlatform : public Platform {
public:
  EXPORT explicit RelocatablePlatform();
  EXPORT ~RelocatablePlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT RelocatableDevice : public ObjectDevice, private llvm::SectionMemoryManager {
  ObjectModules objects = {};
  llvm::RuntimeDyld ld;
  uint64_t getSymbolAddress(const std::string &Name) override;

public:
  EXPORT RelocatableDevice();
  EXPORT std::string name() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT RelocatableDeviceQueue : public ObjectDeviceQueue {
  ObjectModules &objects;
  llvm::RuntimeDyld &ld;

public:
  RelocatableDeviceQueue(decltype(objects) objects, decltype(ld) ld);
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

namespace {
using LoadedModule = std::tuple<std::string, polyregion_dl_handle, std::unordered_map<std::string, void *>>;
using DynamicModules = std::unordered_map<std::string, LoadedModule>;
} // namespace
class EXPORT SharedPlatform : public Platform {
public:
  EXPORT explicit SharedPlatform();
  EXPORT ~SharedPlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT SharedDevice : public ObjectDevice {
  DynamicModules modules;

public:
  ~SharedDevice() override;
  EXPORT std::string name() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT SharedDeviceQueue : public ObjectDeviceQueue {
  DynamicModules &modules;

public:
  explicit SharedDeviceQueue(decltype(modules) modules);
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<void *> &args, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::object

#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include "dl.h"
#include "runtime.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"

namespace polyregion::runtime::object {

using WriteLock = std::unique_lock<std::shared_mutex>;
using ReadLock = std::shared_lock<std::shared_mutex>;

class EXPORT ObjectDevice : public Device {
public:
  EXPORT int64_t id() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::string> features() override;
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

class MemoryManager : public llvm::SectionMemoryManager {
public:
  MemoryManager();

private:
  uint64_t getSymbolAddress(const std::string &Name) override;
};

struct LoadedCodeObject {
  MemoryManager mm;
  llvm::RuntimeDyld ld;
  std::unique_ptr<llvm::object::ObjectFile> rawObject;
  explicit LoadedCodeObject(std::unique_ptr<llvm::object::ObjectFile> obj);
};

using ObjectModules = std::unordered_map<std::string, std::unique_ptr<LoadedCodeObject>>;

} // namespace

class EXPORT RelocatablePlatform : public Platform {
public:
  EXPORT explicit RelocatablePlatform();
  EXPORT ~RelocatablePlatform() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT RelocatableDevice : public ObjectDevice { //, private llvm::SectionMemoryManager {
  ObjectModules objects = {};
  std::shared_mutex lock;

public:
  RelocatableDevice();
  EXPORT std::string name() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT RelocatableDeviceQueue : public ObjectDeviceQueue {
  ObjectModules &objects;
  std::shared_mutex &lock;

public:
  RelocatableDeviceQueue(decltype(objects) objects, decltype(lock) lock);
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, std::vector<Type> types,
                                 std::vector<std::byte> argData, const Policy &policy,
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
  std::shared_mutex lock;

public:
  ~SharedDevice() override;
  EXPORT std::string name() override;
  EXPORT void loadModule(const std::string &name, const std::string &image) override;
  EXPORT bool moduleLoaded(const std::string &name) override;
  EXPORT std::unique_ptr<DeviceQueue> createQueue() override;
};

class EXPORT SharedDeviceQueue : public ObjectDeviceQueue {
  DynamicModules &modules;
  std::shared_mutex &lock;

public:
  explicit SharedDeviceQueue(decltype(modules) modules, decltype(lock) lock);
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, std::vector<Type> types,
                                 std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::object

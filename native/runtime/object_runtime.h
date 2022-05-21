#pragma once

#include "runtime.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"
#include <atomic>

namespace polyregion::runtime::object {

class EXPORT ObjectDevice : public Device {
public:
  EXPORT int64_t id() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                       const std::optional<Callback> &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size,
                                       const std::optional<Callback> &cb) override;
};

class EXPORT RelocatableObjectRuntime : public Runtime {
public:
  EXPORT explicit RelocatableObjectRuntime();
  EXPORT ~RelocatableObjectRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT RelocatableObjectDevice : public ObjectDevice, private llvm::SectionMemoryManager {
  std::unordered_map<std::string, std::unique_ptr<llvm::object::ObjectFile>> objects = {};
  llvm::RuntimeDyld ld;
  uint64_t getSymbolAddress(const std::string &Name) override;

public:
  RelocatableObjectDevice();
  std::string name() override;
  void loadModule(const std::string &name, const std::string &image) override;
  void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                          const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                          const std::optional<Callback> &cb) override;
};

class EXPORT SharedObjectRuntime : public Runtime {
public:
  EXPORT explicit SharedObjectRuntime();
  EXPORT ~SharedObjectRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT SharedObjectDevice : public ObjectDevice {
  using LoadedModule = std::tuple<std::string, void*, std::unordered_map<std::string, void *>>;
  std::unordered_map<std::string, LoadedModule> objects = {};

public:
  virtual ~SharedObjectDevice();
  std::string name() override;
  void loadModule(const std::string &name, const std::string &image) override;
  void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                          const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                          const std::optional<Callback> &cb) override;
};

} // namespace polyregion::runtime::object

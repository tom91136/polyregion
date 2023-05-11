#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include "dl.h"
#include "oneapi/tbb.h"
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
  EXPORT bool singleEntryPerModule() override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
};

class EXPORT ObjectDeviceQueue : public DeviceQueue {
protected:
  detail::CountingLatch latch;
  tbb::task_arena arena;
  tbb::task_group group;

  template <typename F> void threadedLaunch(size_t N, const MaybeCallback &cb, F f) {
    static std::atomic_size_t counter(0);
    static std::unordered_map<size_t, std::atomic_size_t> pending;
    static std::shared_mutex pendingLock;
    static detail::CountedCallbackHandler handler;

    //      auto cbHandle = cb ? handler.createHandle(*cb) : nullptr;

    auto id = counter++;
    WriteLock wPending(pendingLock);
    pending.emplace(id, N);
    for (size_t tid = 0; tid < N; ++tid) {
      std::thread([id, cb, f, tid]() {
        f(tid);
        WriteLock rwPending(pendingLock);
        if (auto it = pending.find(id); it != pending.end()) {
          if (--it->second == 0) {
            if (cb) (*cb)();
            pending.erase(id);
          }
        }
      }).detach();
    }

    //    auto id = counter++;
    //    WriteLock wPending(pendingLock);
    //    pending.emplace(id, N);
    //    arena.enqueue([N, id, f, cb, this]() {
    //      for (size_t tid = 0; tid < N; ++tid) {
    //        group.run([id, tid, f, cb]() {
    //          f(tid);
    //          WriteLock rwPending(pendingLock);
    //          if (auto it = pending.find(id); it != pending.end()) {
    //            if (--it->second == 0) {
    //              pending.erase(id);
    //              if (cb) (*cb)();
    //              //            detail::CountedCallbackHandler::consume(cbHandle);
    //            }
    //          }
    //        });
    //      }
    //    });
  }

public:
  EXPORT explicit ObjectDeviceQueue();
  EXPORT ~ObjectDeviceQueue() noexcept override;
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
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
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
  EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                 const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                 const MaybeCallback &cb) override;
};

} // namespace polyregion::runtime::object

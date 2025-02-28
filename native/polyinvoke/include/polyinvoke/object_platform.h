#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include "polyregion/compat.h"
#include "polyregion/dl.h"
#include "runtime.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"

namespace polyregion::invoke::object {

using WriteLock = std::unique_lock<std::shared_mutex>;
using ReadLock = std::shared_lock<std::shared_mutex>;

class POLYREGION_EXPORT ObjectDevice : public Device {
public:
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT bool singleEntryPerModule() override;
  POLYREGION_EXPORT uintptr_t mallocDevice(size_t size, Access access) override;
  POLYREGION_EXPORT void freeDevice(uintptr_t ptr) override;
  POLYREGION_EXPORT std::optional<void *> mallocShared(size_t size, Access access) override;
  POLYREGION_EXPORT void freeShared(void *ptr) override;
};

class POLYREGION_EXPORT ObjectDeviceQueue : public DeviceQueue {
protected:
  detail::CountingLatch latch;

  template <typename F> static void threadedLaunch(size_t N, const MaybeCallback &cb, F f) {
    static std::atomic_size_t counter(0);
    static std::unordered_map<size_t, std::atomic_size_t> pending;
    static std::shared_mutex pendingMutex;
    static detail::CountedCallbackHandler handler;

    //      auto cbHandle = cb ? handler.createHandle(*cb) : nullptr;

    auto id = counter++;
    WriteLock wPending(pendingMutex);
    pending.emplace(id, N);
    for (size_t tid = 0; tid < N; ++tid) {
      std::thread([id, cb, f, tid]() {
        f(tid);
        WriteLock rwPending(pendingMutex);
        if (const auto it = pending.find(id); it != pending.end()) {
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
  POLYREGION_EXPORT explicit ObjectDeviceQueue(const std::chrono::duration<int64_t> &timeout);
  POLYREGION_EXPORT ~ObjectDeviceQueue() noexcept override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

class MemoryManager final : public llvm::SectionMemoryManager {
public:
  MemoryManager();

private:
  uint64_t getSymbolAddress(const std::string &Name) override;
};

namespace details {

struct LoadedCodeObject {
  MemoryManager mm;
  llvm::RuntimeDyld ld;
  std::unique_ptr<llvm::object::ObjectFile> rawObject;
  explicit LoadedCodeObject(std::unique_ptr<llvm::object::ObjectFile> obj);
};

using ObjectModules = std::unordered_map<std::string, std::unique_ptr<LoadedCodeObject>>;

} // namespace details

class POLYREGION_EXPORT RelocatablePlatform final : public Platform {
  POLYREGION_EXPORT explicit RelocatablePlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~RelocatablePlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class POLYREGION_EXPORT RelocatableDevice final : public ObjectDevice { //, private llvm::SectionMemoryManager {
  details::ObjectModules objects = {};
  std::shared_mutex mutex;

public:
  RelocatableDevice();
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT RelocatableDeviceQueue final : public ObjectDeviceQueue {
  details::ObjectModules &objects;
  std::shared_mutex &mutex;

public:
  RelocatableDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(objects) objects, decltype(mutex) mutex);
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
};

namespace details {
using LoadedModule = std::tuple<std::string, polyregion_dl_handle, std::unordered_map<std::string, void *>>;
using DynamicModules = std::unordered_map<std::string, LoadedModule>;
} // namespace details
class POLYREGION_EXPORT SharedPlatform final : public Platform {
  POLYREGION_EXPORT explicit SharedPlatform();

public:
  POLYREGION_EXPORT static std::variant<std::string, std::unique_ptr<Platform>> create();
  POLYREGION_EXPORT ~SharedPlatform() override = default;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT PlatformKind kind() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class POLYREGION_EXPORT SharedDevice final : public ObjectDevice {
  details::DynamicModules modules;
  std::shared_mutex mutex;

public:
  ~SharedDevice() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT SharedDeviceQueue final : public ObjectDeviceQueue {

  details::DynamicModules &modules;
  std::shared_mutex &mutex;

public:
  explicit SharedDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(modules) modules, decltype(mutex) mutex);
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
};

} // namespace polyregion::invoke::object

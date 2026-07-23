#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

#include "polyregion/compat.h"
#include "polyregion/dl.h"

#include "runtime.h"

namespace polyregion::invoke::object {

using WriteLock = std::unique_lock<std::shared_mutex>;
using ReadLock = std::shared_lock<std::shared_mutex>;

class POLYREGION_EXPORT ObjectDevice : public Device {
public:
  POLYREGION_EXPORT int64_t id() override;
  POLYREGION_EXPORT PhysicalDevice physicalDevice() override;
  POLYREGION_EXPORT ModuleFormat moduleFormat() override;
  POLYREGION_EXPORT std::vector<Property> properties() override;
  POLYREGION_EXPORT std::vector<std::string> features() override;
  POLYREGION_EXPORT bool sharedAddressSpace() override;
  POLYREGION_EXPORT PagingMode pagingMode() override;
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
    if (N == 0) {
      if (cb) (*cb)();
      return;
    }
    const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t workers = std::min(N, hw);
    const size_t chunk = (N + workers - 1) / workers;
    auto pending = std::make_shared<std::atomic_size_t>(workers);
    for (size_t w = 0; w < workers; ++w) {
      const size_t begin = w * chunk;
      const size_t end = std::min(N, begin + chunk);
      std::thread([begin, end, cb, f, pending]() {
        for (size_t tid = begin; tid < end; ++tid)
          f(tid);
        if (--*pending == 0 && cb) (*cb)();
      }).detach();
    }
  }

public:
  POLYREGION_EXPORT explicit ObjectDeviceQueue(const std::chrono::duration<int64_t> &timeout);
  POLYREGION_EXPORT ~ObjectDeviceQueue() noexcept override;
  POLYREGION_EXPORT void enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                    const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes,
                                                  const MaybeCallback &cb) override;
  POLYREGION_EXPORT void enqueueWaitBlocking() override;
};

namespace details {

struct LoadedCodeObject {
  llvm::orc::JITDylib *jd; // borrowed; owned by the ExecutionSession
  std::unordered_map<std::string, uint64_t> symbolCache;
  std::shared_mutex symbolCacheMutex;
  explicit LoadedCodeObject(llvm::orc::JITDylib &jd) : jd(&jd) {}
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
  POLYREGION_EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class POLYREGION_EXPORT RelocatableDevice final : public ObjectDevice {
  std::unique_ptr<llvm::orc::ExecutionSession> es;
  std::unique_ptr<llvm::orc::ObjectLayer> ol;
  std::unique_ptr<llvm::jitlink::JITLinkMemoryManager> jitMemMgr;
  llvm::orc::JITDylib *processJD = nullptr; // borrowed; owned by es
  char globalPrefix = '\0';
  details::ObjectModules objects = {};
  std::shared_mutex mutex;

public:
  RelocatableDevice();
  ~RelocatableDevice() override;
  POLYREGION_EXPORT std::string name() override;
  POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) override;
  POLYREGION_EXPORT bool moduleLoaded(const std::string &name) override;
  POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue(const std::chrono::duration<int64_t> &timeout) override;
};

class POLYREGION_EXPORT RelocatableDeviceQueue final : public ObjectDeviceQueue {
  details::ObjectModules &objects;
  std::shared_mutex &mutex;
  llvm::orc::ExecutionSession &es;
  char globalPrefix;

public:
  RelocatableDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(objects) objects, decltype(mutex) mutex,
                         llvm::orc::ExecutionSession &es, char globalPrefix);
  POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                            std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) override;
};

namespace details {
using LoadedModule = std::tuple<std::string, polyregion_dl_handle, std::unordered_map<std::string, uint64_t>>;
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

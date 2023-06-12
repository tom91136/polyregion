#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "export.h"
#include "types.h"

#ifdef TRACE
  #error Trace already defined
#else

  #if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
  #endif

//  #define TRACE() fprintf(stderr, "[TRACE] %s:%d (this=%p) %s\n", __FILE__, __LINE__, (void *)this,
//  __PRETTY_FUNCTION__)
  #define TRACE()

#endif

namespace polyregion::runtime {

using TypedPointer = std::pair<Type, void *>;

using Property = std::pair<std::string, std::string>;
using Callback = std::function<void()>;
using MaybeCallback = std::optional<Callback>;

struct ArgBuffer {
  std::vector<Type> types;
  std::vector<std::byte> data;
  ArgBuffer(std::initializer_list<TypedPointer> args = {});
  void append(Type tpe, void *ptr);
  void append(const ArgBuffer &that);
  void append(std::initializer_list<TypedPointer> args);
  void prepend(Type tpe, void *ptr);
  void prepend(const ArgBuffer &that);
  void prepend(std::initializer_list<TypedPointer> args);
};

// non-public APIs
namespace detail {

template <typename T,                            //
          typename Lift = std::function<T()>,    //
          typename Drop = std::function<void(T)> //
          >
class LazyDroppable {
  std::optional<T> value = {};
  Lift lift;
  Drop drop;
  static_assert(std::is_invocable_r<T, Lift>());
  static_assert(std::is_invocable_r<void, Drop, T &>());

public:
  constexpr explicit LazyDroppable(Lift lift, Drop drop) : lift(lift), drop(drop) {}
  LazyDroppable &operator=(const T &t) {
    value = t;
    return *this;
  }
  T &operator->() {
    if (value) return *value;
    else {
      value = lift();
      return *value;
    }
  }

  T &operator*() { return operator->(); }

  void touch() { operator->(); }
  virtual ~LazyDroppable() {
    if (value) drop(*value);
  }
};

std::string allocateAndTruncate(const std::function<void(char *, size_t)> &f, size_t length = 512);
std::vector<void *> argDataAsPointers(const std::vector<Type> &types, std::vector<std::byte> &argData);

class CountingLatch {
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic_long pending{};

  class Token {
    CountingLatch &latch;

  public:
    explicit Token(CountingLatch &latch);
    virtual ~Token();
  };

public:
  virtual ~CountingLatch();
  std::shared_ptr<Token> acquire();
};

template <typename T> class BlockingQueue {
  std::condition_variable condition;
  std::mutex mutex;
  std::queue<T> queue;
  bool shutdown = false;

public:
  void terminate() {
    std::unique_lock lock(mutex);
    shutdown = true;
    condition.notify_all();
  }

  void push(T item) {
    std::unique_lock lock(mutex);
    queue.push(std::move(item));
    condition.notify_one();
  }

  std::pair<std::optional<T>, bool> pop() {
    std::unique_lock lock(mutex);
    while (true) {
      if (queue.empty()) {
        if (shutdown) return {{}, false};
      } else
        break;
      condition.wait(lock);
    }
    T item(std::move(queue.front()));
    queue.pop();
    return {{item}, true};
  }
};

// XXX We're storing the callbacks statically to extend lifetime because the callback behaviour on different runtimes
// is not predictable, some may transfer control back even after destruction of all related context.
class CountedCallbackHandler {
  std::atomic_uintptr_t eventCounter = 0;
  std::unordered_map<uintptr_t, Callback> callbacks;
  std::mutex lock;

public:
  ~CountedCallbackHandler();
  void *createHandle(const Callback &cb);
  void consume(void *data);
};

template <typename K, typename V> class CountedStore {
  std::shared_mutex mutex;
  std::atomic<K> counter;
  std::unordered_map<K, V> allocations;

public:
  std::pair<K, V &> store(V value) {
    std::unique_lock writeLock(mutex);
    while (true) {
      auto id = this->counter++;
      if (auto it = allocations.find(id); it != allocations.end()) continue;
      else {
        auto inserted = allocations.emplace_hint(it, id, std::move(value));
        return {id, inserted->second};
      }
    }
  }

  std::optional<V> find(K key) {
    std::shared_lock readLock(mutex);
    if (auto it = allocations.find(key); it != allocations.end()) return std::optional{it->second};
    return {};
  }

  bool erase(K key) {
    std::unique_lock writeLock(mutex);
    return allocations.erase(key) == 1;
  }
};

template <typename T> class MemoryObjects {
  CountedStore<uintptr_t, T> store;

public:
  uintptr_t malloc(T t) { return store.store(t).first; }
  std::optional<T> query(uintptr_t ptr) { return store.find(ptr); }
  void erase(uintptr_t ptr) { store.erase(ptr); }
};

template <typename M, typename F> class ModuleStore {

  struct LoadedModule {
    M first;
    std::unordered_map<std::string, F> second;
  };
  //  using LoadedModule = std::pair<M, std::unordered_map<std::string, F>>;
  std::unordered_map<std::string, LoadedModule> modules = {};

  std::string errorPrefix;
  std::function<M(const std::string &)> load;
  std::function<F(M, const std::string &, const std::vector<Type> &)> resolve;
  std::function<void(M)> dropModule;
  std::function<void(F)> dropFunction;

public:
  ModuleStore(decltype(errorPrefix) errorPrefix,           //
              const decltype(load) &load,                  //
              const decltype(resolve) &resolve,            //
              const decltype(dropModule) &dropModule = {}, //
              const decltype(dropFunction) &dropFunction = {})
      : errorPrefix(std::move(errorPrefix)), //
        load(load), resolve(resolve), dropModule(dropModule), dropFunction(dropFunction) {

    static_assert(std::is_move_constructible<F>::value == std::is_move_constructible<M>::value,
                  "move constructible mismatch between Module (M) and Func (F)");
  }

  ~ModuleStore() {
    for (auto &[moduleName, loaded] : modules) {
      auto &[m, fns] = loaded;
      if (dropFunction) {
        for (auto &[fnName, fn] : fns)
          dropFunction(std::move(fn));
      }
      if (dropModule) dropModule(std::move(m));
    }
  }

  void loadModule(const std::string &name, const std::string &image) {
    if (auto it = modules.find(name); it != modules.end()) {
      throw std::logic_error(std::string(errorPrefix) + "Module named `" + name + "` was already loaded");
    } else {
      modules.emplace_hint(it, name, LoadedModule{load(image), {}});
    }
  }

  bool moduleLoaded(const std::string &name) { return modules.find(name) != modules.end(); }

  F &resolveFunction(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types) {
    auto moduleIt = modules.find(moduleName);
    if (moduleIt == modules.end()) throw std::logic_error(errorPrefix + "No module named `" + moduleName + "` was loaded");
    auto &[m, fnTable] = moduleIt->second;
    if (auto it = fnTable.find(symbol); it != fnTable.end()) return it->second;
    else {
      auto inserted = fnTable.emplace_hint(it, symbol, resolve(m, symbol, types));
      return inserted->second;
    }
  }
};

} // namespace detail

POLYREGION_EXPORT void init();

struct POLYREGION_EXPORT Dim3 {
  POLYREGION_EXPORT size_t x, y, z;
  [[nodiscard]] std::array<size_t, 3> sizes() const { return {x, y, z}; }
  constexpr Dim3(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {
    if (x < 1) throw std::logic_error("x < 1");
    if (y < 1) throw std::logic_error("y < 1");
    if (z < 1) throw std::logic_error("z < 1");
  }
  constexpr Dim3() : Dim3(1, 1, 1) {}
  friend std::ostream &operator<<(std::ostream &os, const Dim3 &dim3);
};

struct POLYREGION_EXPORT Policy {
  POLYREGION_EXPORT Dim3 global{};
  POLYREGION_EXPORT std::optional<std::pair<Dim3, size_t>> local{};
};

enum class Access : uint8_t { RW = 1, RO, WO };

constexpr std::optional<Access> POLYREGION_EXPORT fromUnderlying(uint8_t v) {
  auto x = static_cast<Access>(v);
  switch (x) {
    case Access::RW:
    case Access::RO:
    case Access::WO: return x;
    default: return {};
  }
}

enum class POLYREGION_EXPORT Backend {
  CUDA,
  HIP,
  HSA,
  OpenCL,
  Vulkan,
  Metal,
  SHARED_OBJ,
  RELOCATABLE_OBJ,
};

constexpr std::string_view POLYREGION_EXPORT nameOfBackend(const Backend &b) {
  switch (b) {
    case Backend::CUDA: return "CUDA";
    case Backend::HIP: return "HIP";
    case Backend::HSA: return "HSA";
    case Backend::OpenCL: return "OpenCL";
    case Backend::Vulkan: return "Vulkan";
    case Backend::Metal: return "Metal";
    case Backend::SHARED_OBJ: return "SHARED_OBJ";
    case Backend::RELOCATABLE_OBJ: return "RELOCATABLE_OBJ";
  }
}

struct POLYREGION_EXPORT DeviceQueue {

public:
  virtual POLYREGION_EXPORT ~DeviceQueue() = default;
  virtual POLYREGION_EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t bytes, const MaybeCallback &cb) = 0;

  template <typename T>
  POLYREGION_EXPORT void enqueueHostToDeviceAsyncTyped(const T *src, uintptr_t dst, size_t count, const MaybeCallback &cb = {}) {
    static_assert(sizeof(T) != 0);
    enqueueHostToDeviceAsync(src, dst, count * sizeof(T), cb);
  };

  virtual POLYREGION_EXPORT void enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) = 0;

  template <typename T>
  POLYREGION_EXPORT void enqueueDeviceToHostAsyncTyped(uintptr_t src, T *dst, size_t count, const MaybeCallback &cb = {}) {
    static_assert(sizeof(T) != 0);
    enqueueDeviceToHostAsync(src, dst, count * sizeof(T), cb);
  };

  virtual POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                                    const std::vector<Type> &types, std::vector<std::byte> argData, const Policy &policy,
                                                    const MaybeCallback &cb) = 0;

  virtual POLYREGION_EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const ArgBuffer &buffer,
                                                    const Policy &policy, const MaybeCallback &cb) {
    enqueueInvokeAsync(moduleName, symbol, buffer.types, buffer.data, policy, cb);
  };
};

struct POLYREGION_EXPORT Device {
public:
  virtual POLYREGION_EXPORT ~Device() = default;
  [[nodiscard]] virtual POLYREGION_EXPORT int64_t id() = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT std::string name() = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT bool sharedAddressSpace() = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT bool singleEntryPerModule() = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT std::vector<Property> properties() = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT std::vector<std::string> features() = 0;
  virtual POLYREGION_EXPORT void loadModule(const std::string &name, const std::string &image) = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT bool moduleLoaded(const std::string &name) = 0;
  [[nodiscard]] virtual POLYREGION_EXPORT uintptr_t malloc(size_t size, Access access) = 0;

  template <typename T> [[nodiscard]] POLYREGION_EXPORT uintptr_t mallocTyped(size_t count, Access access) {
    static_assert(sizeof(T) != 0);
    return malloc(count * sizeof(T), access);
  };

  virtual POLYREGION_EXPORT void free(uintptr_t ptr) = 0;
  template <typename... T> POLYREGION_EXPORT void freeAll(T... ptrs) {
    ([&]() { free(ptrs); }(), ...);
  };
  [[nodiscard]] virtual POLYREGION_EXPORT std::unique_ptr<DeviceQueue> createQueue() = 0;
};

class Platform {
public:
  virtual POLYREGION_EXPORT ~Platform() = default;
  [[nodiscard]] POLYREGION_EXPORT virtual std::string name() = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::vector<Property> properties() = 0;
  [[nodiscard]] POLYREGION_EXPORT virtual std::vector<std::unique_ptr<Device>> enumerate() = 0;
  static std::unique_ptr<Platform> of(const Backend &b);
};

} // namespace polyregion::runtime
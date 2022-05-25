#pragma once

#include "export.h"
#include "types.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace polyregion::runtime {

using TypedPointer = std::pair<Type, void *>;

struct Data;

EXPORT void init();

EXPORT void run();

using Property = std::pair<std::string, std::string>;
using Callback = std::function<void()>;
using MaybeCallback = std::optional<Callback>;

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
  explicit LazyDroppable(Lift lift, Drop drop) : lift(lift), drop(drop) {}
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

std::vector<void *> pointers(const std::vector<TypedPointer> &args);

std::string allocateAndTruncate(const std::function<void(char *, size_t)> &f, size_t length = 512);

class CountedCallbackHandler {
  using Storage = std::unordered_map<uint64_t, Callback>;
  using EntryPtr = std::add_pointer_t<Storage::value_type>;

public:
  static void *createHandle(const Callback &cb);
  static void consume(void *data);
};

template <typename M, typename F> class ModuleStore {

  using LoadedModule = std::pair<M, std::unordered_map<std::string, F>>;
  std::unordered_map<std::string, LoadedModule> modules = {};

  std::string errorPrefix;
  std::function<M(const std::string &)> load;
  std::function<F(M, const std::string &)> resolve;
  std::function<void(M)> dropModule;
  std::function<void(F)> dropFunction;

public:
  ModuleStore(
      decltype(errorPrefix) errorPrefix,                //
      const decltype(load) &load,                       //
      const decltype(resolve) &resolve,                 //
      const decltype(dropModule) &dropModule = []() {}, //
      const decltype(dropFunction) &dropFunction = []() {})
      : errorPrefix(std::move(errorPrefix)), //
        load(load), resolve(resolve), dropModule(dropModule), dropFunction(dropFunction) {}

  ~ModuleStore() {
    printf("~ModuleStore\n");
    for (auto &[moduleName, loaded] : modules) {
      auto &[m, fns] = loaded;
      for (auto &[fnName, fn] : fns) {
        dropFunction(fn);
      }
      printf("Drop module %s\n", moduleName.c_str());
      dropModule(m);
    }
  }

  void loadModule(const std::string &name, const std::string &image) {
    if (auto it = modules.find(name); it != modules.end()) {
      throw std::logic_error(std::string(errorPrefix) + "Module named " + name + " was already loaded");
    } else {
      modules.emplace_hint(it, name, LoadedModule{load(image), {}});
    }
  }

  F resolveFunction(const std::string &moduleName, const std::string &symbol) {
    auto moduleIt = modules.find(moduleName);
    if (moduleIt == modules.end())
      throw std::logic_error(errorPrefix + "No module named " + moduleName + " was loaded");
    auto &[m, fnTable] = moduleIt->second;
    if (auto it = fnTable.find(symbol); it != fnTable.end()) return it->second;
    else {
      auto fn = resolve(m, symbol);
      //      CHECKED(hipModuleGetFunction(&fn, m, symbol.c_str()));
      fnTable.emplace_hint(it, symbol, fn);
      return fn;
    }
  }
};

} // namespace detail

struct EXPORT Dim3 {
  EXPORT size_t x = 1, y = 1, z = 1;
  [[nodiscard]] std::array<size_t, 3> sizes() const { return {x, y, z}; }
};

struct EXPORT Policy {
  EXPORT Dim3 global;
  EXPORT std::optional<Dim3> local;
};

enum class Access : uint8_t { RW = 1, RO, WO };

std::optional<Access> fromUnderlying(uint8_t v);

struct EXPORT DeviceQueue {

public:
  virtual EXPORT ~DeviceQueue() = default;
  virtual EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                               const MaybeCallback &cb) = 0;
  virtual EXPORT void enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) = 0;
  virtual EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                         const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                         const MaybeCallback &cb) = 0;
};

struct EXPORT Device {
public:
  virtual EXPORT ~Device() = default;
  [[nodiscard]] virtual EXPORT int64_t id() = 0;
  [[nodiscard]] virtual EXPORT std::string name() = 0;
  [[nodiscard]] virtual EXPORT std::vector<Property> properties() = 0;
  virtual EXPORT void loadModule(const std::string &name, const std::string &image) = 0;
  [[nodiscard]] virtual EXPORT uintptr_t malloc(size_t size, Access access) = 0;
  virtual EXPORT void free(uintptr_t ptr) = 0;
  [[nodiscard]] virtual EXPORT std::unique_ptr<DeviceQueue> createQueue() = 0;
};

class Runtime {
public:
  virtual EXPORT ~Runtime() = default;
  [[nodiscard]] EXPORT virtual std::string name() = 0;
  [[nodiscard]] EXPORT virtual std::vector<Property> properties() = 0;
  [[nodiscard]] EXPORT virtual std::vector<std::unique_ptr<Device>> enumerate() = 0;
};

} // namespace polyregion::runtime
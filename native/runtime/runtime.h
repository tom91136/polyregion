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

// non-public APIs
namespace detail {
class CountedCallbackHandler {
  using Storage = std::unordered_map<uint64_t, Callback>;
  using EntryPtr = std::add_pointer_t<Storage::value_type>;

public:
  static void *createHandle(const Callback &cb);
  static void consume(void *data);
};
} // namespace detail

struct EXPORT Dim {
  EXPORT size_t x = 1, y = 1, z = 1;
  [[nodiscard]] std::array<size_t, 3> sizes() const { return {x, y, z}; }
};

struct EXPORT Policy {
  EXPORT Dim global;
  EXPORT std::optional<Dim> local;
};

enum class Access { RO, WO, RW };

struct EXPORT Device {

public:
  virtual EXPORT ~Device() = default;

  [[nodiscard]] virtual EXPORT int64_t id() = 0;
  [[nodiscard]] virtual EXPORT std::string name() = 0;
  [[nodiscard]] virtual EXPORT std::vector<Property> properties() = 0;

  virtual EXPORT void loadModule(const std::string &name, const std::string &image) = 0;

  virtual EXPORT uintptr_t malloc(size_t size, Access access) = 0;
  virtual EXPORT void free(uintptr_t ptr) = 0;
  virtual EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                               const std::optional<Callback> &cb) = 0;
  virtual EXPORT void enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size,
                                               const std::optional<Callback> &cb) = 0;
  virtual EXPORT void enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                         const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                         const std::optional<Callback> &cb) = 0;
};

class Runtime {
public:
  virtual EXPORT ~Runtime() = default;
  [[nodiscard]] EXPORT virtual std::string name() = 0;
  [[nodiscard]] EXPORT virtual std::vector<Property> properties() = 0;
  [[nodiscard]] EXPORT virtual std::vector<std::unique_ptr<Device>> enumerate() = 0;
};

} // namespace polyregion::runtime
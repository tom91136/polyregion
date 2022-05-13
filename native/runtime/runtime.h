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

EXPORT static void *_malloc(size_t size);

using Property = std::pair<std::string, std::string>;
using Callback = std::function<void()>;

// non-public API
class CountedCallbackHandler {
  std::atomic_uint64_t eventCounter;
  std::unordered_map<uint64_t, Callback> callbacks;
  using EntryPtr = std::add_pointer_t<decltype(callbacks)::value_type>;

public:
  void *createHandle(const Callback &cb);
  void clear();
  static void consume(void *data);
};

struct EXPORT Dim {
  EXPORT size_t x, y, z;
  [[nodiscard]] std::array<size_t, 3> sizes() const { return {x, y, z}; }
};

enum class Access { RO, WO, RW };

struct Memory;

struct EXPORT Device {

public:
  virtual EXPORT ~Device() = default;

  [[nodiscard]] virtual EXPORT int64_t id() = 0;
  [[nodiscard]] virtual EXPORT std::string name() = 0;
  [[nodiscard]] virtual EXPORT std::vector<Property> properties() = 0;

  virtual EXPORT void loadKernel(const std::string &image) = 0;

  virtual EXPORT uintptr_t malloc(size_t size, Access access) = 0;
  virtual EXPORT void free(uintptr_t ptr) = 0;
  virtual EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                               const std::function<void()> &cb) = 0;
  virtual EXPORT void enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size,
                                               const std::function<void()> &cb) = 0;
  virtual EXPORT void enqueueKernelAsync(const std::string &name, //
                                         std::vector<TypedPointer> args,
                                         Dim gridDim,  //
                                         Dim blockDim, //
                                         const Callback &cb) = 0;
};

class Runtime {
public:
  virtual EXPORT ~Runtime() = default;
  [[nodiscard]] EXPORT virtual std::string name() = 0;
  [[nodiscard]] EXPORT virtual std::vector<Property> properties() = 0;
  [[nodiscard]] EXPORT virtual std::vector<std::unique_ptr<Device>> enumerate() = 0;
};

class EXPORT Object {
  std::unique_ptr<Data> data;

public:
  virtual ~Object();
  EXPORT explicit Object(const std::vector<uint8_t> &object);
  [[nodiscard]] EXPORT std::vector<std::pair<std::string, uint64_t>> enumerate() const;
  EXPORT void invoke(const std::string &symbol,                  //
                     const std::function<void *(size_t)> &alloc, //
                     const std::vector<TypedPointer> &args,      //
                     TypedPointer rtn                            //
  ) const;
};

} // namespace polyregion::runtime
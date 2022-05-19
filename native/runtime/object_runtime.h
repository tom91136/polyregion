#pragma once

#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::object {

class EXPORT ObjectRuntime : public Runtime {
public:
  EXPORT explicit ObjectRuntime();
  EXPORT ~ObjectRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT ObjectDevice : public Device {

public:
  EXPORT int64_t id() override;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT void loadKernel(const std::string &image) override;
  EXPORT uintptr_t malloc(size_t size, Access access) override;
  EXPORT void free(uintptr_t ptr) override;
  EXPORT void enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const Callback &cb) override;
  EXPORT void enqueueDeviceToHostAsync(uintptr_t stc, void *dst, size_t size, const Callback &cb) override;
  EXPORT void enqueueKernelAsync(const std::string &name, std::vector<TypedPointer> args, Dim gridDim, Dim blockDim,
                                 const Callback &cb) override;
};

} // namespace polyregion::runtime::object

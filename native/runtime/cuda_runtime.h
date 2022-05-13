#pragma once

#include "cuew.h"
#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::cuda {

class EXPORT CudaRuntime : public Runtime {
public:
  EXPORT explicit CudaRuntime();
  EXPORT ~CudaRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT CudaDevice : public Device {

  CUdevice device = {};
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUmodule module = nullptr;
  std::string deviceName;
  CountedCallbackHandler handler;
  std::unordered_map<std::string, CUfunction> symbols;



  void enqueueCallback(const Callback &cb);

public:
  EXPORT explicit CudaDevice(int ordinal);
  EXPORT ~CudaDevice() override;
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

} // namespace polyregion::runtime::cuda

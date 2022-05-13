#pragma once

#include "clew.h"
#include "runtime.h"
#include <atomic>

namespace polyregion::runtime::cl {

class EXPORT ClRuntime : public Runtime {
public:
  EXPORT explicit ClRuntime();
  EXPORT ~ClRuntime() override = default;
  EXPORT std::string name() override;
  EXPORT std::vector<Property> properties() override;
  EXPORT std::vector<std::unique_ptr<Device>> enumerate() override;
};

class EXPORT ClDevice : public Device {

  std::string deviceName;
  cl_device_id device = {};
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;

  std::atomic_uintptr_t bufferCounter;
  std::unordered_map<uintptr_t, cl_mem> allocations;
  std::unordered_map<std::string, cl_kernel> kernels;

  CountedCallbackHandler handler;

  void enqueueCallback(const Callback &cb, cl_event event);
  cl_mem queryMemObject(uintptr_t handle) const;

public:
  EXPORT explicit ClDevice(cl_device_id device);
  EXPORT ~ClDevice() override;
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
  void releaseProgram();
};

} // namespace polyregion::runtime::cl

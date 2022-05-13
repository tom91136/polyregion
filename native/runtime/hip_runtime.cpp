
#include "hip_runtime.h"
#include <iostream>

using namespace polyregion::runtime;
using namespace polyregion::runtime::hip;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static void checked(hipError_t result, const std::string &file, int line) {
  if (result != hipSuccess) {
    throw std::logic_error(std::string("[HIP error] " + file + ":" + std::to_string(line) + ": ") +
                           hipewErrorString(result));
  }
}

HipRuntime::HipRuntime() {
  if (hipewInit(HIPEW_INIT_HIP) != HIPEW_SUCCESS) {
    throw std::logic_error("HIPEW initialisation failed, no HIP driver present?");
  }
  CHECKED(hipInit(0));
}

std::string HipRuntime::name() { return "HIP"; }

std::vector<Property> HipRuntime::properties() { return {}; }

std::vector<std::unique_ptr<Device>> HipRuntime::enumerate() {
  int count = 0;

  CHECKED(hipGetDeviceCount(&count));
  std::vector<std::unique_ptr<Device>> devices(count);
  for (int i = 0; i < count; i++) {
    devices[i] = std::make_unique<HipDevice>(i);
  }
  return devices;
}

void HipDevice::enqueueCallback(const Callback &cb) {
  CHECKED(hipStreamAddCallback(
      stream,
      [](hipStream_t s, hipError_t e, void *data) {
        CHECKED(e);
        CountedCallbackHandler::consume(data);
      },
      handler.createHandle(cb), 0));
}

HipDevice::HipDevice(int ordinal) {
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName = std::string(512, '\0');
  CHECKED(hipDeviceGetName(deviceName.data(), static_cast<int>(deviceName.length() - 1), device));
  deviceName.erase(deviceName.find('\0'));
  CHECKED(hipDevicePrimaryCtxRetain(&context, device));
  CHECKED(hipCtxPushCurrent(context));
  CHECKED(hipStreamCreate(&stream));
}

HipDevice::~HipDevice() {
  if (module) {
    CHECKED(hipModuleUnload(module));
  }
  CHECKED(hipDevicePrimaryCtxRelease(device));
  handler.clear();
}

int64_t HipDevice::id() { return device; }

std::string HipDevice::name() { return deviceName; }

std::vector<Property> HipDevice::properties() { return {}; }

void HipDevice::loadKernel(const std::string &image) {
  if (module) {
    symbols.clear();
    CHECKED(hipModuleUnload(module));
  }

  CHECKED(hipModuleLoad(&module, "vecAdd-hip-amdgcn-amd-amdhsa-gfx906.bc"));
//  CHECKED(hipModuleLoadData(&module, image.data()));
}

uintptr_t HipDevice::malloc(size_t size, Access access) {
  hipDeviceptr_t ptr = {};
  CHECKED(hipMalloc(&ptr, size));
  return ptr;
}

void HipDevice::free(uintptr_t ptr) { CHECKED(hipFree(ptr)); }

void HipDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const Callback &cb) {
  CHECKED(hipMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void HipDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const Callback &cb) {
  CHECKED(hipMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void HipDevice::enqueueKernelAsync(const std::string &name, std::vector<TypedPointer> args, Dim gridDim, Dim blockDim,
                                   const std::function<void()> &cb) {
  if (!module) throw std::logic_error("No module loaded");

  auto it = symbols.find(name);
  hipFunction_t fn = nullptr;
  if (!(it == symbols.end())) fn = it->second;
  else {
    CHECKED(hipModuleGetFunction(&fn, module, name.c_str()));
    symbols.emplace(name, fn);
  }

  std::vector<void *> ptrs(args.size());
  for (size_t i = 0; i < args.size(); ++i)
    ptrs[i] = args[i].second;

  int sharedMem = 0;
  CHECKED(hipModuleLaunchKernel(fn,                                 //
                                gridDim.x, gridDim.y, gridDim.z,    //
                                blockDim.x, blockDim.y, blockDim.z, //
                                sharedMem,                          //
                                stream, ptrs.data(),                //
                                nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

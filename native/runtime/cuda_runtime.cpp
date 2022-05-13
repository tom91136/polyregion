
#include "cuda_runtime.h"
#include <iostream>

using namespace polyregion::runtime;
using namespace polyregion::runtime::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static void checked(CUresult result, const std::string &file, int line) {
  if (result != CUDA_SUCCESS) {
    throw std::logic_error(std::string("[CUDA error] " + file + ":" + std::to_string(line) + ": ") +
                           cuewErrorString(result));
  }
}

CudaRuntime::CudaRuntime() {
  if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
    throw std::logic_error("CUEW initialisation failed, no CUDA driver present?");
  }
  CHECKED(cuInit(0));
}

std::string CudaRuntime::name() { return "CUDA"; }

std::vector<Property> CudaRuntime::properties() { return {}; }

std::vector<std::unique_ptr<Device>> CudaRuntime::enumerate() {
  int count = 0;
  CHECKED(cuDeviceGetCount(&count));
  std::vector<std::unique_ptr<Device>> devices(count);
  for (int i = 0; i < count; i++) {
    devices[i] = std::make_unique<CudaDevice>(i);
  }
  return devices;
}

void CudaDevice::enqueueCallback(const Callback &cb) {

  if (cuLaunchHostFunc) { // >= CUDA 10
    CHECKED(cuLaunchHostFunc(stream, CountedCallbackHandler::consume, handler.createHandle(cb)));
  } else {
    CHECKED(cuStreamAddCallback(
        stream,
        [](CUstream s, CUresult e, void *data) {
          CHECKED(e);
          CountedCallbackHandler::consume(data);
        },
        handler.createHandle(cb), 0));
  }
}

CudaDevice::CudaDevice(int ordinal) {
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName = std::string(512, '\0');
  CHECKED(cuDeviceGetName(deviceName.data(), static_cast<int>(deviceName.length() - 1), device));
  deviceName.erase(deviceName.find('\0'));
  CHECKED(cuDevicePrimaryCtxRetain(&context, device));
  CHECKED(cuCtxPushCurrent(context));
  CHECKED(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}

CudaDevice::~CudaDevice() {
  if (module) {
    CHECKED(cuModuleUnload(module));
  }
  CHECKED(cuDevicePrimaryCtxRelease(device));
  handler.clear();
}

int64_t CudaDevice::id() { return device; }

std::string CudaDevice::name() { return deviceName; }

std::vector<Property> CudaDevice::properties() { return {}; }

void CudaDevice::loadKernel(const std::string &image) {
  if (module) {
    symbols.clear();
    CHECKED(cuModuleUnload(module));
  }
  CHECKED(cuModuleLoadData(&module, image.data()));
}

uintptr_t CudaDevice::malloc(size_t size, Access access) {
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}

void CudaDevice::free(uintptr_t ptr) { CHECKED(cuMemFree(ptr)); }

void CudaDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const Callback &cb) {
  CHECKED(cuMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void CudaDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const Callback &cb) {
  CHECKED(cuMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void CudaDevice::enqueueKernelAsync(const std::string &name, std::vector<TypedPointer> args, Dim gridDim, Dim blockDim,
                                    const std::function<void()> &cb) {
  if (!module) throw std::logic_error("No module loaded");

  auto it = symbols.find(name);
  CUfunction fn = nullptr;
  if (!(it == symbols.end())) fn = it->second;
  else {
    CHECKED(cuModuleGetFunction(&fn, module, name.c_str()));
    symbols.emplace(name, fn);
  }

  std::vector<void *> ptrs(args.size());
  for (size_t i = 0; i < args.size(); ++i)
    ptrs[i] = args[i].second;

  int sharedMem = 0;
  CHECKED(cuLaunchKernel(fn,                                 //
                         gridDim.x, gridDim.y, gridDim.z,    //
                         blockDim.x, blockDim.y, blockDim.z, //
                         sharedMem,                          //
                         stream, ptrs.data(),                //
                         nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

#include "cuda_runtime.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[CUDA error] ";

static void checked(CUresult result, const std::string &file, int line) {
  if (result != CUDA_SUCCESS) {
    throw std::logic_error(std::string(ERROR_PREFIX + file + ":" + std::to_string(line) + ": ") +
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

void CudaDevice::enqueueCallback(const std::optional<Callback> &cb) {
  if (!cb) return;

  if (cuLaunchHostFunc) { // >= CUDA 10
    CHECKED(cuLaunchHostFunc(
        stream, [](void *data) { return detail::CountedCallbackHandler::consume(data); },
        detail::CountedCallbackHandler::createHandle(*cb)));
  } else {
    CHECKED(cuStreamAddCallback(
        stream,
        [](CUstream s, CUresult e, void *data) {
          CHECKED(e);
          detail::CountedCallbackHandler::consume(data);
        },
        detail::CountedCallbackHandler::createHandle(*cb), 0));
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
  for (auto &[_, p] : modules)
    CHECKED(cuModuleUnload(p.first));
  CHECKED(cuDevicePrimaryCtxRelease(device));
}

int64_t CudaDevice::id() { return device; }

std::string CudaDevice::name() { return deviceName; }

std::vector<Property> CudaDevice::properties() { return {}; }

void CudaDevice::loadModule(const std::string &name, const std::string &image) {
  if (auto it = modules.find(name); it != modules.end()) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    CUmodule module;
    CHECKED(cuModuleLoadData(&module, image.data()));
    modules.emplace_hint(it, name, LoadedModule{module, {}});
  }
}

uintptr_t CudaDevice::malloc(size_t size, Access access) {
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}

void CudaDevice::free(uintptr_t ptr) { CHECKED(cuMemFree(ptr)); }

void CudaDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                          const std::optional<Callback> &cb) {
  CHECKED(cuMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void CudaDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const std::optional<Callback> &cb) {
  CHECKED(cuMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void CudaDevice::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                    const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                    const std::optional<Callback> &cb) {
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[m, fnTable] = moduleIt->second;
  CUfunction fn = nullptr;
  if (auto it = fnTable.find(symbol); it != fnTable.end()) fn = it->second;
  else {
    CHECKED(cuModuleGetFunction(&fn, m, symbol.c_str()));
    fnTable.emplace_hint(it, symbol, fn);
  }
  std::vector<void *> ptrs(args.size());
  for (size_t i = 0; i < args.size(); ++i)
    ptrs[i] = args[i].second;
  auto grid = policy.global;
  auto block = policy.local.value_or(Dim{});

  int sharedMem = 0;
  CHECKED(cuLaunchKernel(fn,                        //
                         grid.x, grid.y, grid.z,    //
                         block.x, block.y, block.z, //
                         sharedMem,                 //
                         stream, ptrs.data(),       //
                         nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

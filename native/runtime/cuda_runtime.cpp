#include "cuda_runtime.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__);

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
  for (int i = 0; i < count; i++)
    devices[i] = std::make_unique<CudaDevice>(i);
  return devices;
}

// ---

CudaDevice::CudaDevice(int ordinal)
    : context(
          [&]() {
            CUcontext c;
            CHECKED(cuDevicePrimaryCtxRetain(&c, device));
            CHECKED(cuCtxPushCurrent(c));
            return c;
          },
          [&](auto) {
            if (device) CHECKED(cuDevicePrimaryCtxRelease(device));
          }),
      store(
          ERROR_PREFIX,
          [&](auto &&s) {
            context.touch();
            CUmodule module;
            CHECKED(cuModuleLoadData(&module, s.data()));
            return module;
          },
          [&](auto &&m, auto &&name) {
            context.touch();
            CUfunction fn;
            CHECKED(cuModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [](auto &&m) { CHECKED(cuModuleUnload(m)); }, [](auto &&f) {}) {
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(cuDeviceGetName(data, static_cast<int>(length), device)); });
}
int64_t CudaDevice::id() { return device; }
std::string CudaDevice::name() { return deviceName; }
std::vector<Property> CudaDevice::properties() { return {}; }
void CudaDevice::loadModule(const std::string &name, const std::string &image) { store.loadModule(name, image); }
uintptr_t CudaDevice::malloc(size_t size, Access access) {
  context.touch();
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}
void CudaDevice::free(uintptr_t ptr) {
  context.touch();
  CHECKED(cuMemFree(ptr));
}
std::unique_ptr<DeviceQueue> CudaDevice::createQueue() { return std::make_unique<CudaDeviceQueue>(store); }

// ---

CudaDeviceQueue::CudaDeviceQueue(decltype(store) store) : store(store) {
  CHECKED(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}
CudaDeviceQueue::~CudaDeviceQueue() { CHECKED(cuStreamDestroy(stream)); }
void CudaDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
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

void CudaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  CHECKED(cuMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  CHECKED(cuMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                         const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                         const MaybeCallback &cb) {
  if (rtn.first != Type::Void) throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  auto fn = store.resolveFunction(moduleName, symbol);
  auto ptrs = detail::pointers(args);
  auto grid = policy.global;
  auto block = policy.local.value_or(Dim3{});
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

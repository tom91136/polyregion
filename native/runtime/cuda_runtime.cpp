#include "cuda_runtime.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[CUDA error] ";

static void checked(CUresult result, const char *file, int line) {
  if (result != CUDA_SUCCESS) {
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           cuewErrorString(result));
  }
}

CudaRuntime::CudaRuntime() {
  TRACE();
  if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
    throw std::logic_error("CUEW initialisation failed, no CUDA driver present?");
  }
  CHECKED(cuInit(0));
}
std::string CudaRuntime::name() {
  TRACE();
  return "CUDA";
}
std::vector<Property> CudaRuntime::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> CudaRuntime::enumerate() {
  TRACE();
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
          [this]() {
            TRACE();
            CUcontext c;
            CHECKED(cuDevicePrimaryCtxRetain(&c, device));
            CHECKED(cuCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            TRACE();
            if (device) CHECKED(cuDevicePrimaryCtxRelease(device));
          }),
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            TRACE();
            context.touch();
            CUmodule module;
            CHECKED(cuModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            context.touch();
            CUfunction fn;
            CHECKED(cuModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            TRACE();
            CHECKED(cuModuleUnload(m));
          },
          [&](auto &&f) { TRACE(); }) {
  TRACE();
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(cuDeviceGetName(data, static_cast<int>(length), device)); });
}

int64_t CudaDevice::id() {
  TRACE();
  return device;
}
std::string CudaDevice::name() {
  TRACE();
  return deviceName;
}
std::vector<Property> CudaDevice::properties() {
  TRACE();
  return {};
}
void CudaDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
uintptr_t CudaDevice::malloc(size_t size, Access access) {
  TRACE();
  context.touch();
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}
void CudaDevice::free(uintptr_t ptr) {
  TRACE();
  context.touch();
  CHECKED(cuMemFree(ptr));
}

std::unique_ptr<DeviceQueue> CudaDevice::createQueue() {
  TRACE();
  return std::make_unique<CudaDeviceQueue>(store);
}
CudaDevice::~CudaDevice() { TRACE(); }

// ---

CudaDeviceQueue::CudaDeviceQueue(decltype(store) store) : store(store) {
  TRACE();
  CHECKED(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}
CudaDeviceQueue::~CudaDeviceQueue() {
  TRACE();
  CHECKED(cuStreamDestroy(stream));
}
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
  TRACE();
  CHECKED(cuMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  CHECKED(cuMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                         const std::vector<Type> &types, std::vector<void *> &args,
                                         const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  auto fn = store.resolveFunction(moduleName, symbol);
  auto grid = policy.global;
  auto block = policy.local.value_or(Dim3{});
  int sharedMem = 0;
  CHECKED(cuLaunchKernel(fn,                        //
                         grid.x, grid.y, grid.z,    //
                         block.x, block.y, block.z, //
                         sharedMem,                 //
                         stream, args.data(),       //
                         nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

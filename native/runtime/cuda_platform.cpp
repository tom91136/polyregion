#include "cuda_platform.h"
#include "utils.hpp"

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

CudaPlatform::CudaPlatform() {
  TRACE();
  if (auto result = cuewInit(CUEW_INIT_CUDA); result != CUEW_SUCCESS) {
    throw std::logic_error("CUEW initialisation failed (" + std::to_string(result) + ") ,no CUDA driver present?");
  }
  CHECKED(cuInit(0));
}
std::string CudaPlatform::name() {
  TRACE();
  return "CUDA";
}
std::vector<Property> CudaPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> CudaPlatform::enumerate() {
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
          [&](auto &&) { TRACE(); }) {
  TRACE();
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(cuDeviceGetName(data, int_cast<int>(length), device)); });
}

int64_t CudaDevice::id() {
  TRACE();
  return device;
}
std::string CudaDevice::name() {
  TRACE();
  return deviceName;
}
bool CudaDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
std::vector<Property> CudaDevice::properties() {
  TRACE();
  return {};
}
std::vector<std::string> CudaDevice::features() {
  TRACE();
  int ccMajor = 0, ccMinor = 0;
  CHECKED(cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CHECKED(cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  return {"sm_" + std::to_string((ccMajor * 10) + ccMinor)};
}
void CudaDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool CudaDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t CudaDevice::malloc(size_t size, Access) {
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
  context.touch();
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
  auto f = [cb, token = latch.acquire()]() {
    if (cb) (*cb)();
  };

  if (cuLaunchHostFunc && false) { // >= CUDA 10
    // FIXME cuLaunchHostFunc does not retain errors from previous launches, use the deprecated cuStreamAddCallback
    //  for now. See https://stackoverflow.com/a/58173486
    CHECKED(cuLaunchHostFunc(
        stream, [](void *data) { return detail::CountedCallbackHandler::consume(data); },
        detail::CountedCallbackHandler::createHandle(f)));
  } else {
    CHECKED(cuStreamAddCallback(
        stream,
        [](CUstream, CUresult e, void *data) {
          CHECKED(e);
          detail::CountedCallbackHandler::consume(data);
        },
        detail::CountedCallbackHandler::createHandle(f), 0));
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
                                         std::vector<Type> types, std::vector<std::byte> argData, const Policy &policy,
                                         const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  auto fn = store.resolveFunction(moduleName, symbol);
  auto grid = policy.global;
  auto [block, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  auto args = detail::argDataAsPointers(types, argData);
  CHECKED(cuLaunchKernel(fn,                        //
                         grid.x, grid.y, grid.z,    //
                         block.x, block.y, block.z, //
                         sharedMem,                 //
                         stream, args.data(),       //
                         nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

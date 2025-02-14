#include "polyinvoke/cuda_platform.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr auto PREFIX = "CUDA";

static void checked(CUresult result, const char *file, int line) {
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    auto name = "(unknown)", desc = "(unknown)";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    POLYINVOKE_FATAL(PREFIX, "%s:%d: %s(%u): %s", file, line, name, result, desc);
  }
}

std::variant<std::string, std::unique_ptr<Platform>> CudaPlatform::create() {
  if (const auto result = cuewInit(CUEW_INIT_CUDA); result != CUEW_SUCCESS)
    return "CUEW initialisation failed (" + std::to_string(result) + "), no CUDA driver present?";
  if (const auto result = cuInit(0); result != CUDA_SUCCESS) {
    auto name = "(unknown)", desc = "(unknown)";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    return std::string(name) + ": " + std::string(desc);
  }
  return std::unique_ptr<Platform>(new CudaPlatform());
}

CudaPlatform::CudaPlatform() { POLYINVOKE_TRACE(); }
std::string CudaPlatform::name() {
  POLYINVOKE_TRACE();
  return "CUDA";
}
std::vector<Property> CudaPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind CudaPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat CudaPlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::PTX;
}
std::vector<std::unique_ptr<Device>> CudaPlatform::enumerate() {
  POLYINVOKE_TRACE();
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
            POLYINVOKE_TRACE();
            CUcontext c;
            CHECKED(cuDevicePrimaryCtxRetain(&c, device));
            CHECKED(cuCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            POLYINVOKE_TRACE();
            if (device) CHECKED(cuDevicePrimaryCtxRelease(device));
          }),
      store(
          PREFIX,
          [this](auto &&s) {
            POLYINVOKE_TRACE();
            context.touch();
            CUmodule module;
            CHECKED(cuModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name, auto) {
            POLYINVOKE_TRACE();
            context.touch();
            CUfunction fn;
            CHECKED(cuModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            POLYINVOKE_TRACE();
            CHECKED(cuModuleUnload(m));
          },
          [&](auto &&) { POLYINVOKE_TRACE(); }) {
  POLYINVOKE_TRACE();
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName =
      detail::allocateAndTruncate([&](auto &&data, auto &&length) { CHECKED(cuDeviceGetName(data, static_cast<int>(length), device)); });
}

int64_t CudaDevice::id() {
  POLYINVOKE_TRACE();
  return device;
}
std::string CudaDevice::name() {
  POLYINVOKE_TRACE();
  return deviceName;
}
bool CudaDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
bool CudaDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> CudaDevice::properties() {
  POLYINVOKE_TRACE();
  return {};
}
std::vector<std::string> CudaDevice::features() {
  POLYINVOKE_TRACE();
  int ccMajor = 0, ccMinor = 0;
  CHECKED(cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CHECKED(cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  return {"sm_" + std::to_string((ccMajor * 10) + ccMinor)};
}
void CudaDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool CudaDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t CudaDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}
void CudaDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(cuMemFree(ptr));
}
std::optional<void *> CudaDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  CUdeviceptr ptr = {};
  CHECKED(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));

  return reinterpret_cast<void *>(ptr);
}
void CudaDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}
std::unique_ptr<DeviceQueue> CudaDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  context.touch();
  return std::make_unique<CudaDeviceQueue>(timeout, store);
}
CudaDevice::~CudaDevice() { POLYINVOKE_TRACE(); }

// ---

CudaDeviceQueue::CudaDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store) : latch(timeout), store(store) {
  POLYINVOKE_TRACE();
  CHECKED(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}
CudaDeviceQueue::~CudaDeviceQueue() {
  POLYINVOKE_TRACE();
  CHECKED(cuStreamDestroy(stream));
}
void CudaDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  if (!cb) return;
  auto f = [cb, token = latch.acquire()]() {
    if (cb) (*cb)();
  };
  static detail::CountedCallbackHandler handler;
  if (cuLaunchHostFunc && false) { // >= CUDA 10
    // FIXME cuLaunchHostFunc does not retain errors from previous launches, use the deprecated cuStreamAddCallback
    //  for now. See https://stackoverflow.com/a/58173486
    CHECKED(cuLaunchHostFunc(stream, [](void *data) { return handler.consume(data); }, handler.createHandle(f)));
  } else {
    CHECKED(cuStreamAddCallback(
        stream,
        [](CUstream, CUresult e, void *data) {
          CHECKED(e);
          handler.consume(data);
        },
        handler.createHandle(f), 0));
  }
}

void CudaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(cuMemcpyHtoDAsync(dst + dstOffset, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(cuMemcpyDtoHAsync(dst, src + srcOffset, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                         std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void) POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", to_string(types.back()).data());
  auto fn = store.resolveFunction(moduleName, symbol, types);
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
void CudaDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  CHECKED(cuStreamSynchronize(stream));
}

#undef CHECKED

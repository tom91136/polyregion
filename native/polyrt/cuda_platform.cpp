#include "polyrt/cuda_platform.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::cuda;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *PREFIX = "CUDA";

static void checked(CUresult result, const char *file, int line) {
  if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
    POLYRT_FATAL(PREFIX, "%s:%d: %s (code=%u)", file, line, cuewErrorString(result), result);
  }
}

std::variant<std::string, std::unique_ptr<Platform>> CudaPlatform::create() {
  if (auto result = cuewInit(CUEW_INIT_CUDA); result != CUEW_SUCCESS)
    return "CUEW initialisation failed (" + std::to_string(result) + "), no CUDA driver present?";
  if (auto result = cuInit(0); result != CUDA_SUCCESS) return cuewErrorString(result);
  return std::unique_ptr<Platform>(new CudaPlatform());
}

CudaPlatform::CudaPlatform() { POLYRT_TRACE(); }
std::string CudaPlatform::name() {
  POLYRT_TRACE();
  return "CUDA";
}
std::vector<Property> CudaPlatform::properties() {
  POLYRT_TRACE();
  return {};
}
PlatformKind CudaPlatform::kind() {
  POLYRT_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat CudaPlatform::moduleFormat() {
  POLYRT_TRACE();
  return ModuleFormat::PTX;
}
std::vector<std::unique_ptr<Device>> CudaPlatform::enumerate() {
  POLYRT_TRACE();
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
            POLYRT_TRACE();
            CUcontext c;
            CHECKED(cuDevicePrimaryCtxRetain(&c, device));
            CHECKED(cuCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            POLYRT_TRACE();
            if (device) CHECKED(cuDevicePrimaryCtxRelease(device));
          }),
      store(
          PREFIX,
          [this](auto &&s) {
            POLYRT_TRACE();
            context.touch();
            CUmodule module;
            CHECKED(cuModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name, auto) {
            POLYRT_TRACE();
            context.touch();
            CUfunction fn;
            CHECKED(cuModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            POLYRT_TRACE();
            CHECKED(cuModuleUnload(m));
          },
          [&](auto &&) { POLYRT_TRACE(); }) {
  POLYRT_TRACE();
  CHECKED(cuDeviceGet(&device, ordinal));
  deviceName =
      detail::allocateAndTruncate([&](auto &&data, auto &&length) { CHECKED(cuDeviceGetName(data, static_cast<int>(length), device)); });
}

int64_t CudaDevice::id() {
  POLYRT_TRACE();
  return device;
}
std::string CudaDevice::name() {
  POLYRT_TRACE();
  return deviceName;
}
bool CudaDevice::sharedAddressSpace() {
  POLYRT_TRACE();
  return false;
}
bool CudaDevice::singleEntryPerModule() {
  POLYRT_TRACE();
  return false;
}
std::vector<Property> CudaDevice::properties() {
  POLYRT_TRACE();
  return {};
}
std::vector<std::string> CudaDevice::features() {
  POLYRT_TRACE();
  int ccMajor = 0, ccMinor = 0;
  CHECKED(cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CHECKED(cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  return {"sm_" + std::to_string((ccMajor * 10) + ccMinor)};
}
void CudaDevice::loadModule(const std::string &name, const std::string &image) {
  POLYRT_TRACE();
  store.loadModule(name, image);
}
bool CudaDevice::moduleLoaded(const std::string &name) {
  POLYRT_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t CudaDevice::mallocDevice(size_t size, Access) {
  POLYRT_TRACE();
  context.touch();
  if (size == 0) POLYRT_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  CUdeviceptr ptr = {};
  CHECKED(cuMemAlloc(&ptr, size));
  return ptr;
}
void CudaDevice::freeDevice(uintptr_t ptr) {
  POLYRT_TRACE();
  context.touch();
  CHECKED(cuMemFree(ptr));
}
std::optional<void *> CudaDevice::mallocShared(size_t size, Access access) {
  POLYRT_TRACE();
  context.touch();
  if (size == 0) POLYRT_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  CUdeviceptr ptr = {};
  CHECKED(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));

  return reinterpret_cast<void *>(ptr);
}
void CudaDevice::freeShared(void *ptr) {
  POLYRT_TRACE();
  context.touch();
  CHECKED(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}
std::unique_ptr<DeviceQueue> CudaDevice::createQueue() {
  POLYRT_TRACE();
  context.touch();
  return std::make_unique<CudaDeviceQueue>(store);
}
CudaDevice::~CudaDevice() { POLYRT_TRACE(); }

// ---

CudaDeviceQueue::CudaDeviceQueue(decltype(store) store) : store(store) {
  POLYRT_TRACE();
  CHECKED(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
}
CudaDeviceQueue::~CudaDeviceQueue() {
  POLYRT_TRACE();
  CHECKED(cuStreamDestroy(stream));
}
void CudaDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
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

void CudaDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  POLYRT_TRACE();
  CHECKED(cuMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  POLYRT_TRACE();
  CHECKED(cuMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void CudaDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                         std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYRT_TRACE();
  if (types.back() != Type::Void) POLYRT_FATAL(PREFIX, "Non-void return type not supported: %s", to_string(types.back()).data());
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

#undef CHECKED

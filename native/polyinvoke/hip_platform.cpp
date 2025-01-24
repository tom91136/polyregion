#include "polyinvoke/hip_platform.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::hip;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *PREFIX = "HIP";

static void checked(hipError_t result, const char *file, int line) {
  if (result != hipSuccess) {

    POLYINVOKE_FATAL(PREFIX, "%s:%d: %s (code=%u)", file, line, hipewErrorString(result), result);
  }
}

std::variant<std::string, std::unique_ptr<Platform>> HipPlatform::create() {
  if (auto result = hipewInit(HIPEW_INIT_HIP); result != HIPEW_SUCCESS) {
    std::string description;
    switch (result) {
      case HIPEW_ERROR_OPEN_FAILED: return "HIPEW initialisation failed: error opening amdhip64 dynamic library, no HIP driver present?";
      case HIPEW_ERROR_ATEXIT_FAILED: return "HIPEW initialisation failed: error setting up atexit() handler!";
      case HIPEW_ERROR_OLD_DRIVER: // see https://developer.blender.org/D13324
        return "HIPEW initialisation failed: driver version too old, requires AMD Radeon Pro 21.Q4 driver or newer";
      default: return "HIPEW initialisation failed: Unknown error (" + std::to_string(result) + ")";
    }
  }
  if (auto result = hipInit(0); result != hipSuccess) return hipewErrorString(result);
  return std::unique_ptr<Platform>(new HipPlatform());
}

HipPlatform::HipPlatform() { POLYINVOKE_TRACE(); }
std::string HipPlatform::name() {
  POLYINVOKE_TRACE();
  return "HIP";
}
std::vector<Property> HipPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind HipPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
ModuleFormat HipPlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::HSACO;
}
std::vector<std::unique_ptr<Device>> HipPlatform::enumerate() {
  POLYINVOKE_TRACE();
  int count = 0;
  CHECKED(hipGetDeviceCount(&count));
  std::vector<std::unique_ptr<Device>> devices(count);
  for (int i = 0; i < count; i++)
    devices[i] = std::make_unique<HipDevice>(i);
  return devices;
}

// ---

HipDevice::HipDevice(int ordinal)
    : context(
          [this]() {
            POLYINVOKE_TRACE();
            hipCtx_t c;
            CHECKED(hipDevicePrimaryCtxRetain(&c, device));
            CHECKED(hipCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            POLYINVOKE_TRACE();
            if (device) CHECKED(hipDevicePrimaryCtxRelease(device));
          }),
      store(
          PREFIX,
          [this](auto &&s) {
            POLYINVOKE_TRACE();
            context.touch();
            hipModule_t module;
            CHECKED(hipModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name, auto) {
            POLYINVOKE_TRACE();
            context.touch();
            hipFunction_t fn;
            CHECKED(hipModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            POLYINVOKE_TRACE();
            CHECKED(hipModuleUnload(m));
          },
          [&](auto &&) { POLYINVOKE_TRACE(); }) {
  POLYINVOKE_TRACE();
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName =
      detail::allocateAndTruncate([&](auto &&data, auto &&length) { CHECKED(hipDeviceGetName(data, static_cast<int>(length), device)); });
}

int64_t HipDevice::id() {
  POLYINVOKE_TRACE();
  return device;
}
std::string HipDevice::name() {
  POLYINVOKE_TRACE();
  return deviceName;
}
bool HipDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
bool HipDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> HipDevice::properties() {
  // XXX Do not use hipDeviceGetAttribute and hipDeviceAttribute_t. AMD "reordered" the enums in ROCm 5.0
  // See https://docs.amd.com/bundle/ROCm_Release_Notes_v5.0/page/Breaking_Changes.html
  POLYINVOKE_TRACE();
  return {};
}
std::vector<std::string> HipDevice::features() {
  POLYINVOKE_TRACE();
  // XXX Do not use hipDeviceGetAttribute and hipDeviceAttribute_t. AMD "reordered" the enums in ROCm 5.0
  // See https://docs.amd.com/bundle/ROCm_Release_Notes_v5.0/page/Breaking_Changes.html
  hipDeviceProp_t prop;
  CHECKED(hipGetDeviceProperties(&prop, device));
  return {"gfx" + std::to_string(prop.gcnArch)};
}
void HipDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool HipDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t HipDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  hipDeviceptr_t ptr = {};
  CHECKED(hipMalloc(&ptr, size));
  return ptr;
}
void HipDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(hipFree(ptr));
}
std::optional<void *> HipDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (size == 0) POLYINVOKE_FATAL(PREFIX, "Cannot malloc size of %ld", size);
  hipDeviceptr_t ptr = {};
  CHECKED(hipMallocManaged(&ptr, size, hipMemAttachGlobal));
  return reinterpret_cast<void *>(ptr);
}
void HipDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(hipFree(reinterpret_cast<hipDeviceptr_t>(ptr)));
}
std::unique_ptr<DeviceQueue> HipDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  context.touch();
  return std::make_unique<HipDeviceQueue>(timeout, store);
}
HipDevice::~HipDevice() { POLYINVOKE_TRACE(); }

// ---

HipDeviceQueue::HipDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store) : latch(timeout), store(store) {
  POLYINVOKE_TRACE();
  CHECKED(hipStreamCreate(&stream));
}
HipDeviceQueue::~HipDeviceQueue() {
  POLYINVOKE_TRACE();
  auto result = hipStreamDestroy(stream);
  if (result == hipError_t::hipErrorContextIsDestroyed) return;
  CHECKED(result);
}
void HipDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  if (!cb) return;
  POLYINVOKE_TRACE();
  static detail::CountedCallbackHandler handler;
  CHECKED(hipStreamAddCallback(
      stream,
      [](hipStream_t, hipError_t e, void *data) {
        CHECKED(e);
        handler.consume(data);
      },
      handler.createHandle([cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      }),
      0));
}

void HipDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(hipMemcpyHtoDAsync(dst + dstOffset, src, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(hipMemcpyDtoHAsync(dst, src + srcOffset, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                        std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void) POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", to_string(types.back()).data());
  auto fn = store.resolveFunction(moduleName, symbol, types);
  auto grid = policy.global;
  auto [block, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  auto args = detail::argDataAsPointers(types, argData);
  CHECKED(hipModuleLaunchKernel(fn,                        //
                                grid.x, grid.y, grid.z,    //
                                block.x, block.y, block.z, //
                                sharedMem,                 //
                                stream, args.data(),       //
                                nullptr));
// XXX HIP is available on Win32 but submitting hangs unless we "touch" the stream to get to dispatch...
#ifdef _WIN32
  hipStreamQuery(stream);
#endif
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  CHECKED(hipStreamSynchronize(stream));
}

#undef CHECKED

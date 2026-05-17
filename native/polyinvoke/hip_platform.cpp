#include "polyinvoke/hip_platform.h"

#include "magic_enum/magic_enum.hpp"

#include "dl_util.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::hip;

static constexpr auto PREFIX = "HIP";

#define CHECKED(f__)                                                                                                                       \
  do {                                                                                                                                     \
    hipError_t result__ = (f__);                                                                                                           \
    if (result__ != hipSuccess) {                                                                                                          \
      POLYINVOKE_FATAL(PREFIX, "%s (code=%u): %s", hipGetErrorName(result__), result__, hipGetErrorString(result__));                      \
    }                                                                                                                                      \
  } while (0)

std::variant<std::string, std::unique_ptr<Platform>> HipPlatform::create() {
#ifdef _WIN32
  void *lib = dl::open_first({"amdhip64.dll"});
#elif defined(__APPLE__)
  void *lib = nullptr;
#else
  void *lib = dl::open_first({"libamdhip64.so.6", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so"});
#endif
  if (!lib) return "HIP: failed to open amdhip64 dynamic library, no HIP driver present?";
  hipew_hip_resolve(dl::lookup, lib);
  if (const auto result = hipInit(0); result != hipSuccess) {
    return std::string(hipGetErrorName(result)) + ": " + std::string(hipGetErrorString(result));
  }
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
ModuleFormat HipDevice::moduleFormat() {
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
  return {"hip", "amd", "gfx" + std::to_string(prop.gcnArch)};
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
  return reinterpret_cast<uintptr_t>(ptr);
}
void HipDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  CHECKED(hipFree(reinterpret_cast<hipDeviceptr_t>(ptr)));
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
void HipDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst + dstOffset), reinterpret_cast<hipDeviceptr_t>(src + srcOffset), size,
                             stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst + dstOffset), const_cast<void *>(src), size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src + srcOffset), size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                        std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void)
    POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", magic_enum::enum_name(types.back()).data());
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

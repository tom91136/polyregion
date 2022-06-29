#include "hip_platform.h"
#include "utils.hpp"

using namespace polyregion::runtime;
using namespace polyregion::runtime::hip;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[HIP error] ";

static void checked(hipError_t result, const char *file, int line) {
  if (result != hipSuccess) {
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           hipewErrorString(result));
  }
}

HipPlatform::HipPlatform() {
  TRACE();
  if (auto result = hipewInit(HIPEW_INIT_HIP); result != HIPEW_SUCCESS) {
    std::string description;
    switch (result) {
      case HIPEW_ERROR_OPEN_FAILED: description = "Error opening amdhip64 dynamic library, no HIP driver present?";
      case HIPEW_ERROR_ATEXIT_FAILED: description = "Error setting up atexit() handler!";
      case HIPEW_ERROR_OLD_DRIVER: // see https://developer.blender.org/D13324
        description = "Driver version too old, requires AMD Radeon Pro 21.Q4 driver or newer";
      default: description = "Unknown error(" + std::to_string(result) + ")";
    }
    throw std::logic_error("HIPEW initialisation failed:" + description);
  }
  CHECKED(hipInit(0));
}
std::string HipPlatform::name() {
  TRACE();
  return "HIP";
}
std::vector<Property> HipPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> HipPlatform::enumerate() {
  TRACE();
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
            TRACE();
            hipCtx_t c;
            CHECKED(hipDevicePrimaryCtxRetain(&c, device));
            CHECKED(hipCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            TRACE();
            if (device) CHECKED(hipDevicePrimaryCtxRelease(device));
          }),
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            TRACE();
            context.touch();
            hipModule_t module;
            CHECKED(hipModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            context.touch();
            hipFunction_t fn;
            CHECKED(hipModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            TRACE();
            CHECKED(hipModuleUnload(m));
          },
          [&](auto &&) { TRACE(); }) {
  TRACE();
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(hipDeviceGetName(data, int_cast<int>(length), device)); });
}

int64_t HipDevice::id() {
  TRACE();
  return device;
}
std::string HipDevice::name() {
  TRACE();
  return deviceName;
}
bool HipDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
std::vector<Property> HipDevice::properties() {
  // XXX Do not use hipDeviceGetAttribute and hipDeviceAttribute_t. AMD "reordered" the enums in ROCm 5.0
  // See https://docs.amd.com/bundle/ROCm_Release_Notes_v5.0/page/Breaking_Changes.html
  TRACE();
  return {};
}
std::vector<std::string> HipDevice::features() {
  TRACE();
  // XXX Do not use hipDeviceGetAttribute and hipDeviceAttribute_t. AMD "reordered" the enums in ROCm 5.0
  // See https://docs.amd.com/bundle/ROCm_Release_Notes_v5.0/page/Breaking_Changes.html
  hipDeviceProp_t prop;
  CHECKED(hipGetDeviceProperties(&prop, device));
  return {"gfx" + std::to_string(prop.gcnArch)};
}
void HipDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool HipDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t HipDevice::malloc(size_t size, Access) {
  TRACE();
  context.touch();
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  hipDeviceptr_t ptr = {};
  CHECKED(hipMalloc(&ptr, size));
  return ptr;
}
void HipDevice::free(uintptr_t ptr) {
  TRACE();
  context.touch();
  CHECKED(hipFree(ptr));
}
std::unique_ptr<DeviceQueue> HipDevice::createQueue() {
  TRACE();
  context.touch();
  return std::make_unique<HipDeviceQueue>(store);
}
HipDevice::~HipDevice() { TRACE(); }

// ---

HipDeviceQueue::HipDeviceQueue(decltype(store) store) : store(store) {
  TRACE();
  CHECKED(hipStreamCreate(&stream));
}
HipDeviceQueue::~HipDeviceQueue() {
  TRACE();
  CHECKED(hipStreamDestroy(stream));
}
void HipDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  TRACE();
  if (!cb) return;
  CHECKED(hipStreamAddCallback(
      stream,
      [](hipStream_t, hipError_t e, void *data) {
        CHECKED(e);
        detail::CountedCallbackHandler::consume(data);
      },
      detail::CountedCallbackHandler::createHandle(*cb), 0));
}

void HipDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  CHECKED(hipMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  CHECKED(hipMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                        std::vector<Type> types, std::vector<std::byte> argData, const Policy &policy,
                                        const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  auto fn = store.resolveFunction(moduleName, symbol);
  auto grid = policy.global;
  auto block = policy.local.value_or(Dim3{});
  int sharedMem = 0;
  auto args = detail::argDataAsPointers(types, argData);
  CHECKED(hipModuleLaunchKernel(fn,                        //
                                grid.x, grid.y, grid.z,    //
                                block.x, block.y, block.z, //
                                sharedMem,                 //
                                stream, args.data(),       //
                                nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

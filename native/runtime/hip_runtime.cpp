#include "hip_runtime.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::hip;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[HIP error] ";

static void checked(hipError_t result, const std::string &file, int line) {
  if (result != hipSuccess) {
    throw std::logic_error(std::string(ERROR_PREFIX + file + ":" + std::to_string(line) + ": ") +
                           hipewErrorString(result));
  }
}

HipRuntime::HipRuntime() {
  if (hipewInit(HIPEW_INIT_HIP) != HIPEW_SUCCESS) {
    throw std::logic_error("HIPEW initialisation failed, no HIP driver present?");
  }
  CHECKED(hipInit(0));
}
std::string HipRuntime::name() { return "HIP"; }
std::vector<Property> HipRuntime::properties() { return {}; }
std::vector<std::unique_ptr<Device>> HipRuntime::enumerate() {
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
            hipCtx_t c;
            CHECKED(hipDevicePrimaryCtxRetain(&c, device));
            CHECKED(hipCtxPushCurrent(c));
            return c;
          },
          [this](auto) {
            if (device) CHECKED(hipDevicePrimaryCtxRelease(device));
          }),
      store(
          ERROR_PREFIX,
          [this](auto &&s) {
            hipModule_t module;
            context.touch();
            CHECKED(hipModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name) {
            hipFunction_t fn;
            context.touch();
            CHECKED(hipModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [](auto &&m) { CHECKED(hipModuleUnload(m)); }, [](auto &&f) {}) {
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(hipDeviceGetName(data, static_cast<int>(length), device)); });
}
HipDevice::~HipDevice() { CHECKED(hipDevicePrimaryCtxRelease(device)); }
int64_t HipDevice::id() { return device; }
std::string HipDevice::name() { return deviceName; }
std::vector<Property> HipDevice::properties() { return {}; }
void HipDevice::loadModule(const std::string &name, const std::string &image) { store.loadModule(name, image); }
uintptr_t HipDevice::malloc(size_t size, Access access) {
  context.touch();
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  hipDeviceptr_t ptr = {};
  CHECKED(hipMalloc(&ptr, size));
  return ptr;
}
void HipDevice::free(uintptr_t ptr) {
  context.touch();
  CHECKED(hipFree(ptr));
}
std::unique_ptr<DeviceQueue> HipDevice::createQueue() { return std::make_unique<HipDeviceQueue>(store); }

// ---

HipDeviceQueue::HipDeviceQueue(decltype(store) store) : store(store) { CHECKED(hipStreamCreate(&stream)); }
HipDeviceQueue::~HipDeviceQueue() { CHECKED(hipStreamDestroy(stream)); }
void HipDeviceQueue::enqueueCallback(const MaybeCallback &cb) {
  if (!cb) return;
  CHECKED(hipStreamAddCallback(
      stream,
      [](hipStream_t s, hipError_t e, void *data) {
        CHECKED(e);
        detail::CountedCallbackHandler::consume(data);
      },
      detail::CountedCallbackHandler::createHandle(*cb), 0));
}
void HipDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  CHECKED(hipMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  CHECKED(hipMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}
void HipDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                        const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                        const MaybeCallback &cb) {
  if (rtn.first != Type::Void) throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported");
  auto fn = store.resolveFunction(moduleName, symbol);
  auto ptrs = detail::pointers(args);
  auto grid = policy.global;
  auto block = policy.local.value_or(Dim3{});
  int sharedMem = 0;
  CHECKED(hipModuleLaunchKernel(fn,                        //
                                grid.x, grid.y, grid.z,    //
                                block.x, block.y, block.z, //
                                sharedMem,                 //
                                stream, ptrs.data(),       //
                                nullptr));
  enqueueCallback(cb);
}

#undef CHECKED

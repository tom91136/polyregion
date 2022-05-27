#include "hip_runtime.h"

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

HipRuntime::HipRuntime() {
  TRACE();
  if (hipewInit(HIPEW_INIT_HIP) != HIPEW_SUCCESS) {
    throw std::logic_error("HIPEW initialisation failed, no HIP driver present?");
  }
  CHECKED(hipInit(0));
}
std::string HipRuntime::name() {
  TRACE();
  return "HIP";
}
std::vector<Property> HipRuntime::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> HipRuntime::enumerate() {
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
            hipModule_t module;
            context.touch();
            CHECKED(hipModuleLoadData(&module, s.data()));
            return module;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            hipFunction_t fn;
            context.touch();
            CHECKED(hipModuleGetFunction(&fn, m, name.c_str()));
            return fn;
          },
          [&](auto &&m) {
            TRACE();
            CHECKED(hipModuleUnload(m));
          },
          [&](auto &&f) { TRACE(); }) {
  TRACE();
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName = detail::allocateAndTruncate(
      [&](auto &&data, auto &&length) { CHECKED(hipDeviceGetName(data, static_cast<int>(length), device)); });
}
HipDevice::~HipDevice() {
  TRACE();
  CHECKED(hipDevicePrimaryCtxRelease(device));
}
int64_t HipDevice::id() {
  TRACE();
  return device;
}
std::string HipDevice::name() {
  TRACE();
  return deviceName;
}
std::vector<Property> HipDevice::properties() {
  TRACE();
  return {};
}
void HipDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
uintptr_t HipDevice::malloc(size_t size, Access access) {
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
  return std::make_unique<HipDeviceQueue>(store);
}

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
      [](hipStream_t s, hipError_t e, void *data) {
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
                                        const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                        const MaybeCallback &cb) {
  TRACE();
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

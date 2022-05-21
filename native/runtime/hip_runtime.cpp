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

void HipDevice::enqueueCallback(const std::optional<Callback> &cb) {
  if (!cb) return;
  CHECKED(hipStreamAddCallback(
      stream,
      [](hipStream_t s, hipError_t e, void *data) {
        CHECKED(e);
        detail::CountedCallbackHandler::consume(data);
      },
      detail::CountedCallbackHandler::createHandle(*cb), 0));
}

HipDevice::HipDevice(int ordinal) {
  CHECKED(hipDeviceGet(&device, ordinal));
  deviceName = std::string(512, '\0');
  CHECKED(hipDeviceGetName(deviceName.data(), static_cast<int>(deviceName.length() - 1), device));
  deviceName.erase(deviceName.find('\0'));
  CHECKED(hipDevicePrimaryCtxRetain(&context, device));
  CHECKED(hipCtxPushCurrent(context));
  CHECKED(hipStreamCreate(&stream));
}

HipDevice::~HipDevice() {
  for (auto &[_, p] : modules)
    CHECKED(hipModuleUnload(p.first));
  CHECKED(hipDevicePrimaryCtxRelease(device));
}

int64_t HipDevice::id() { return device; }

std::string HipDevice::name() { return deviceName; }

std::vector<Property> HipDevice::properties() { return {}; }

void HipDevice::loadModule(const std::string &name, const std::string &image) {
  if (auto it = modules.find(name); it != modules.end()) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    hipModule_t module;
    CHECKED(hipModuleLoadData(&module, image.data()));
    modules.emplace_hint(it, name, LoadedModule{module, {}});
  }
}

uintptr_t HipDevice::malloc(size_t size, Access access) {
  if (size == 0) throw std::logic_error(std::string(ERROR_PREFIX) + "Cannot malloc size of 0");
  hipDeviceptr_t ptr = {};
  CHECKED(hipMalloc(&ptr, size));
  return ptr;
}

void HipDevice::free(uintptr_t ptr) { CHECKED(hipFree(ptr)); }

void HipDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                         const std::optional<Callback> &cb) {
  CHECKED(hipMemcpyHtoDAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void HipDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const std::optional<Callback> &cb) {
  CHECKED(hipMemcpyDtoHAsync(dst, src, size, stream));
  enqueueCallback(cb);
}

void HipDevice::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                   const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                   const std::optional<Callback> &cb) {

  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[m, fnTable] = moduleIt->second;
  hipFunction_t fn = nullptr;
  if (auto it = fnTable.find(symbol); it != fnTable.end()) fn = it->second;
  else {
    CHECKED(hipModuleGetFunction(&fn, m, symbol.c_str()));
    fnTable.emplace_hint(it, symbol, fn);
  }
  std::vector<void *> ptrs(args.size());
  for (size_t i = 0; i < args.size(); ++i)
    ptrs[i] = args[i].second;

  auto grid = policy.global;
  auto block = policy.local.value_or(Dim{});

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

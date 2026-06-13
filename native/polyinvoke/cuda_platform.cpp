#include "polyinvoke/cuda_platform.h"

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "dl_util.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cuda;

static constexpr auto PREFIX = "CUDA";

#define CHECKED(f__)                                                                                                                       \
  do {                                                                                                                                     \
    CUresult result__ = (f__);                                                                                                             \
    if (result__ != CUDA_SUCCESS && result__ != CUDA_ERROR_DEINITIALIZED) {                                                                \
      const char *name__ = "(unknown)", *desc__ = "(unknown)";                                                                             \
      cuGetErrorName(result__, &name__);                                                                                                   \
      cuGetErrorString(result__, &desc__);                                                                                                 \
      POLYINVOKE_FATAL(PREFIX, "%s (code=%u): %s; callsite: `%s`", name__, result__, desc__, #f__);                                        \
    }                                                                                                                                      \
  } while (0)

#define CHECKED_EXTRA(f__, fmt__, ...)                                                                                                     \
  do {                                                                                                                                     \
    CUresult result__ = (f__);                                                                                                             \
    if (result__ != CUDA_SUCCESS && result__ != CUDA_ERROR_DEINITIALIZED) {                                                                \
      const char *name__ = "(unknown)", *desc__ = "(unknown)";                                                                             \
      cuGetErrorName(result__, &name__);                                                                                                   \
      cuGetErrorString(result__, &desc__);                                                                                                 \
      POLYINVOKE_FATAL(PREFIX, "%s (code=%u) %s; callsite: `" fmt__ "`", name__, result__, desc__, __VA_ARGS__);                           \
    }                                                                                                                                      \
  } while (0)

std::variant<std::string, std::unique_ptr<Platform>> CudaPlatform::create() {
#ifdef _WIN32
  void *lib = dl::open_first({"nvcuda.dll"});
#elif defined(__APPLE__)
  void *lib = dl::open_first({"/usr/local/cuda/lib/libcuda.dylib"});
#else
  void *lib = dl::open_first({"libcuda.so.1", "libcuda.so"});
#endif
  if (!lib) return "CUDA: failed to open libcuda dynamic library, no CUDA driver present?";
  cuew_cuda_resolve(dl::lookup, lib);
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
ModuleFormat CudaDevice::moduleFormat() {
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
            CHECKED_EXTRA(cuModuleGetFunction(&fn, m, name.c_str()), //
                          "cuModuleGetFunction(module=%p, name=%s)", static_cast<void *>(m), name.c_str());
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
PhysicalDevice CudaDevice::physicalDevice() {
  POLYINVOKE_TRACE();
  int domain = 0, bus = 0, slot = 0;
  CHECKED(cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device));
  CHECKED(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
  CHECKED(cuDeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
  return PhysicalDevice::pci(static_cast<uint32_t>(domain), static_cast<uint8_t>(bus), static_cast<uint8_t>(slot), 0);
}
bool CudaDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
PagingMode CudaDevice::pagingMode() {
  POLYINVOKE_TRACE();
  int pageable = 0;
  cuDeviceGetAttribute(&pageable, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device);
  if (pageable) return PagingMode::System;
  int managed = 0;
  cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device);
  return managed ? PagingMode::Managed : PagingMode::None;
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
  std::vector<std::string> out{"cuda", "nvidia", "sm_" + std::to_string((ccMajor * 10) + ccMinor), "fp64", "int64"};
  if (ccMajor > 5 || (ccMajor == 5 && ccMinor >= 3)) out.emplace_back("fp16");
  const auto paging = pagingMode();
  out.emplace_back(fmt::format("paging:{}", magic_enum::enum_name(paging)));
  if (paging == PagingMode::System) out.emplace_back("hmm");
  return out;
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
#ifdef _WIN32
  // XXX WDDM: cuMemAllocManaged + sync is not sufficient for safe CPU read-back (pages fault
  // with STATUS_IN_PAGE_ERROR). Use pinned host memory which is permanently mapped on both
  // sides via UVA.
  void *ptr = nullptr;
  CHECKED(cuMemAllocHost(&ptr, size));
  return ptr;
#else
  CUdeviceptr ptr = {};
  CHECKED(cuMemAllocManaged(&ptr, size, CU_MEM_ATTACH_GLOBAL));
  return reinterpret_cast<void *>(ptr);
#endif
}
void CudaDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
#ifdef _WIN32
  CHECKED(cuMemFreeHost(ptr));
#else
  CHECKED(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
#endif
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
  // FIXME cuLaunchHostFunc does not retain errors from previous launches, use the deprecated cuStreamAddCallback
  //  for now. See https://stackoverflow.com/a/58173486
  CHECKED(cuStreamAddCallback(
      stream,
      [](CUstream, CUresult e, void *data) {
        CHECKED(e);
        detail::CountedCallbackHandler::instance().consume(data);
      },
      detail::CountedCallbackHandler::instance().createHandle(f), 0));
}
void CudaDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                 const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  CHECKED(cuMemcpyDtoDAsync(dst + dstOffset, src + srcOffset, size, stream));
  enqueueCallback(cb);
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
  if (types.back() != Type::Void)
    POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported: %s", magic_enum::enum_name(types.back()).data());
  auto fn = store.resolveFunction(moduleName, symbol, types);
  auto grid = policy.global;
  auto [block, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  auto args = detail::argDataAsPointers(types, argData);
  // XXX `Type::Scratch` slot must be 0 (start of dynamic shared); kernel signature keeps the
  // slot, the OpenCL kernarg ABI expects this value.
  static const uint64_t scratchPlaceholder = 0;
  size_t out = 0;
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i] == Type::Void) continue;
    args[out++] = types[i] == Type::Scratch ? const_cast<void *>(static_cast<const void *>(&scratchPlaceholder)) : args[i];
  }
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
#ifdef _WIN32
  // XXX WDDM: host read-back of managed memory needs cuCtxSynchronize; stream sync alone
  // leaves pages un-migrated and faults with STATUS_IN_PAGE_ERROR.
  CHECKED(cuCtxSynchronize());
#endif
}

#undef CHECKED

#include "polyinvoke/cl_platform.h"

#include <cstring>
#include <iostream>
#include <thread>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

#include "magic_enum/magic_enum.hpp"

#include "polyregion/env.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cl;

static constexpr const char *PREFIX = "OpenCL";

#define CHECKED(f__)                                                                                                                       \
  do {                                                                                                                                     \
    cl_int result__ = (f__);                                                                                                               \
    if (result__ != CL_SUCCESS) {                                                                                                          \
      POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", __FILE__, __LINE__, clewErrorString(result__));                                                \
    }                                                                                                                                      \
  } while (0)

#define OUT_ERR err__
#define OUT_CHECKED(f__)                                                                                                                   \
  ([&]() {                                                                                                                                 \
    cl_int result__ = CL_SUCCESS;                                                                                                          \
    auto OUT_ERR = &result__;                                                                                                              \
    auto x__ = (f__);                                                                                                                      \
    if (result__ == CL_SUCCESS) return x__;                                                                                                \
    POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", __FILE__, __LINE__, clewErrorString(result__));                                                  \
  })()

static std::string queryDeviceInfo(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  CHECKED(clGetDeviceInfo(device, info, 0, nullptr, &size));
  std::string data(size - 1, '\0'); // -1 as clGetDeviceInfo returns the length+1 for \0
  CHECKED(clGetDeviceInfo(device, info, size, data.data(), nullptr));
  return data;
}

namespace {
constexpr cl_uint CL_DEVICE_SVM_CAPABILITIES_ = 0x1053;
constexpr cl_bitfield CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_ = 1 << 0;
constexpr cl_bitfield CL_DEVICE_SVM_FINE_GRAIN_BUFFER_ = 1 << 1;
constexpr cl_bitfield CL_MEM_SVM_FINE_GRAIN_BUFFER_ = 1 << 10;
constexpr cl_uint CL_KERNEL_EXEC_INFO_SVM_PTRS_ = 0x11B6;

bool deviceSupportsIL(cl_device_id device) {
  size_t size = 0;
  if (clGetDeviceInfo(device, /*CL_DEVICE_IL_VERSION=*/0x105B, 0, nullptr, &size) == CL_SUCCESS && size > 1) return true;
  return queryDeviceInfo(device, CL_DEVICE_EXTENSIONS).find("cl_khr_il_program") != std::string::npos;
}

details::SVMFns resolveSVM(cl_platform_id /*platform*/, cl_device_id device) {
  details::SVMFns fns{};
  cl_bitfield caps = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES_, sizeof(caps), &caps, nullptr) != CL_SUCCESS) return fns;
  if (!(caps & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_ | CL_DEVICE_SVM_FINE_GRAIN_BUFFER_))) return fns;
  static void *clHandle = []() -> void * {
#ifdef _WIN32
    if (HMODULE h = GetModuleHandleA("OpenCL.dll")) return reinterpret_cast<void *>(h);
    return reinterpret_cast<void *>(LoadLibraryA("OpenCL.dll"));
#else
    void *h = nullptr;
  #ifdef RTLD_NOLOAD
    h = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_NOLOAD);
  #endif
    if (!h) h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY);
    return h;
#endif
  }();
  static auto resolve = [](const char *name) -> void * {
#ifdef _WIN32
    if (!clHandle) return nullptr;
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(clHandle), name));
#else
    if (clHandle) return dlsym(clHandle, name);
  #ifdef RTLD_DEFAULT
    return dlsym(RTLD_DEFAULT, name);
  #else
    return nullptr;
  #endif
#endif
  };
  fns.alloc = reinterpret_cast<details::ClSVMAlloc_fn>(resolve("clSVMAlloc"));
  fns.free = reinterpret_cast<details::ClSVMFree_fn>(resolve("clSVMFree"));
  fns.memcpy = reinterpret_cast<details::ClEnqueueSVMMemcpy_fn>(resolve("clEnqueueSVMMemcpy"));
  fns.setKernelArg = reinterpret_cast<details::ClSetKernelArgSVMPointer_fn>(resolve("clSetKernelArgSVMPointer"));
  fns.setKernelExecInfo = reinterpret_cast<details::ClSetKernelExecInfo_fn>(resolve("clSetKernelExecInfo"));
  fns.map = reinterpret_cast<details::ClEnqueueSVMMap_fn>(resolve("clEnqueueSVMMap"));
  fns.unmap = reinterpret_cast<details::ClEnqueueSVMUnmap_fn>(resolve("clEnqueueSVMUnmap"));
  fns.memFlags = (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER_) ? CL_MEM_SVM_FINE_GRAIN_BUFFER_ : 0;
  if (!fns) return {};
  return fns;
}
} // namespace

std::variant<std::string, std::unique_ptr<Platform>> ClPlatform::create() {
  // XXX FP64 is emulated on Intel Arc and needs to be enabled via environment variable
  // we set it unless it's already defined with some other value
  env::put("OverrideDefaultFP64Settings", "1", false);
  env::put("IGC_EnableDPEmulation", "1", false);
  switch (auto result = clewInit(); result) {
    case CLEW_SUCCESS: break;
    case CLEW_ERROR_OPEN_FAILED: return "CLEW: failed to open the dynamic library";
    case CLEW_ERROR_ATEXIT_FAILED: return "CLEW: cannot queue atexit for closing the dynamic library";
    default: return "Unknown error: " + std::to_string(result);
  }
  return std::unique_ptr<Platform>(new ClPlatform());
}
ClPlatform::ClPlatform() { POLYINVOKE_TRACE(); }
std::string ClPlatform::name() {
  POLYINVOKE_TRACE();
  return "OpenCL";
}
std::vector<Property> ClPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind ClPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::Managed;
}
std::vector<std::unique_ptr<Device>> ClPlatform::enumerate() {
  POLYINVOKE_TRACE();
  cl_uint numPlatforms = 0;
  if (const auto r = clGetPlatformIDs(0, nullptr, &numPlatforms); r == -1001 || numPlatforms == 0) return {};
  else CHECKED(r);
  std::vector<cl_platform_id> platforms(numPlatforms);
  CHECKED(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));
  std::vector<std::unique_ptr<Device>> clDevices;
  // XXX OpenCL CPU ICDs duplicate the SharedObject/RelocatableObject paths and tend to
  // misbehave; restrict to GPU/accelerator devices only.
  const cl_device_type kAcceleratorMask = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR;
  for (const auto &platform : platforms) {
    cl_uint numDevices = 0;
    if (const auto deviceIdResult = clGetDeviceIDs(platform, kAcceleratorMask, 0, nullptr, &numDevices);
        deviceIdResult == CL_DEVICE_NOT_FOUND) {
      continue; // no device
    } else CHECKED(deviceIdResult);

    std::vector<cl_device_id> devices(numDevices);
    CHECKED(clGetDeviceIDs(platform, kAcceleratorMask, numDevices, devices.data(), nullptr));
    auto ilFn =
        reinterpret_cast<details::ClCreateProgramWithIL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithIL"));
    if (!ilFn)
      ilFn = reinterpret_cast<details::ClCreateProgramWithIL_fn>(
          clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithILKHR"));
    for (auto &device : devices) {
      auto svm = resolveSVM(platform, device);
      clDevices.push_back(std::make_unique<ClDevice>(device, ModuleFormat::Source, nullptr, svm));
      if (ilFn && deviceSupportsIL(device)) clDevices.push_back(std::make_unique<ClDevice>(device, ModuleFormat::SPIRV_Kernel, ilFn, svm));
    }
  }
  return clDevices;
}
ClPlatform::~ClPlatform() { POLYINVOKE_TRACE(); }

// ---

ClDevice::ClDevice(cl_device_id device, ModuleFormat format, details::ClCreateProgramWithIL_fn ilCreateFn, details::SVMFns svm)
    : device(
          [&, device]() {
            POLYINVOKE_TRACE();
            // XXX clReleaseDevice appears to crash various CL implementation regardless of version, skip retain as well
            //            if (__clewRetainDevice && __clewReleaseDevice) { // clRetainDevice requires OpenCL >= 1.2
            //              CHECKED(__clewRetainDevice(device));
            //            }
            return device;
          },
          [&](auto &&) {
            POLYINVOKE_TRACE();
            // XXX see above
            //            if (__clewRetainDevice && __clewReleaseDevice) // clReleaseDevice requires OpenCL >= 1.2
            //              CHECKED(__clewReleaseDevice(device));
          }),
      context(
          [this]() {
            POLYINVOKE_TRACE();
            return OUT_CHECKED(clCreateContext(nullptr, 1, &*this->device, nullptr, nullptr, OUT_ERR));
          },
          [&](auto &&c) {
            POLYINVOKE_TRACE();
            CHECKED(clReleaseContext(c));
          }),
      deviceName(queryDeviceInfo(device, CL_DEVICE_NAME)), format(format), ilCreateFn(ilCreateFn), svm(svm),
      svmTracker(svm ? std::make_shared<details::SVMTracker>() : nullptr),
      store(
          PREFIX,
          [this](auto &&image) {
            POLYINVOKE_TRACE();
            auto imageData = image.data();
            auto imageLen = image.size();
            cl_program program;
            if (this->format == ModuleFormat::SPIRV_Kernel) {
              program = OUT_CHECKED(this->ilCreateFn(*context, imageData, imageLen, OUT_ERR));
            } else {
              program = OUT_CHECKED(clCreateProgramWithSource(*context, 1, &imageData, &imageLen, OUT_ERR));
            }
            const std::string compilerArgs = "";
            cl_int result = clBuildProgram(program, 1, &*this->device, compilerArgs.data(), nullptr, nullptr);
            if (result != CL_SUCCESS) {
              size_t len;
              CHECKED(clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len));
              std::string buildLog(len, '\0');
              CHECKED(clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, len, buildLog.data(), nullptr));
              auto compilerMessage = std::string(clewErrorString(result));
              POLYINVOKE_FATAL(PREFIX, "Program failed to compile with: %s\nDiagnostics:\n%s\n===Program source===:\n%s\n", //
                               compilerMessage.c_str(), buildLog.c_str(), image.c_str());
            }
            POLYINVOKE_TRACE();
            return program;
          },
          [this](auto &&m, auto &&name, auto) {
            POLYINVOKE_TRACE();
            context.touch();
            POLYINVOKE_TRACE();
            return OUT_CHECKED(clCreateKernel(m, name.c_str(), OUT_ERR));
          },
          [&](auto &&m) {
            POLYINVOKE_TRACE();
            CHECKED(clReleaseProgram(m));
          },
          [&](auto &&f) {
            POLYINVOKE_TRACE();
            CHECKED(clReleaseKernel(f));
          }) {
  POLYINVOKE_TRACE();
}

int64_t ClDevice::id() {
  POLYINVOKE_TRACE();
  return reinterpret_cast<int64_t>(*device);
}
std::string ClDevice::name() {
  POLYINVOKE_TRACE();
  return deviceName;
}
ModuleFormat ClDevice::moduleFormat() {
  POLYINVOKE_TRACE();
  return format;
}
bool ClDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return false;
}
bool ClDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> ClDevice::properties() {
  POLYINVOKE_TRACE();
  return {
      {"CL_DEVICE_PROFILE", queryDeviceInfo(*device, CL_DEVICE_PROFILE)},
      {"CL_DRIVER_VERSION", queryDeviceInfo(*device, CL_DRIVER_VERSION)},
      {"CL_DEVICE_VENDOR", queryDeviceInfo(*device, CL_DEVICE_VENDOR)},
      {"CL_DEVICE_VERSION", queryDeviceInfo(*device, CL_DEVICE_VERSION)},
      {"CL_DEVICE_EXTENSIONS", queryDeviceInfo(*device, CL_DEVICE_EXTENSIONS)},
  };
}
// Stable vendor token from CL_DEVICE_VENDOR's free-form string; values match what test profiles
// write as the @hint (`opencl@nvidia` etc.).
static std::string normaliseVendor(std::string vendor) {
  std::string lower(vendor.size(), {});
  for (size_t i = 0; i < vendor.size(); ++i)
    lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(vendor[i])));
  if (lower.find("nvidia") != std::string::npos) return "nvidia";
  if (lower.find("intel") != std::string::npos) return "intel";
  if (lower.find("advanced micro devices") != std::string::npos || lower.find("amd") != std::string::npos) return "amd";
  if (lower.find("mesa") != std::string::npos || lower.find("rusticl") != std::string::npos) return "mesa";
  if (lower.find("apple") != std::string::npos) return "apple";
  if (lower.find("arm") != std::string::npos) return "arm";
  if (lower.find("qualcomm") != std::string::npos) return "qualcomm";
  return "unknown";
}

std::vector<std::string> ClDevice::features() {
  POLYINVOKE_TRACE();
  if (cachedFeatures) return *cachedFeatures;
  std::vector<std::string> out{"opencl"};
  out.push_back(normaliseVendor(queryDeviceInfo(*device, CL_DEVICE_VENDOR)));
  out.emplace_back(format == ModuleFormat::SPIRV_Kernel ? "spirv_kernel" : "source");
  const auto exts = queryDeviceInfo(*device, CL_DEVICE_EXTENSIONS);
  const auto hasExt = [&](std::string_view e) { return exts.find(e) != std::string::npos; };
  if (hasExt("cl_khr_fp64")) out.emplace_back("fp64");
  if (hasExt("cl_khr_fp16")) out.emplace_back("fp16");
  if (hasExt("cl_khr_int64_base_atomics")) out.emplace_back("int64");
  cachedFeatures = out;
  return out;
}
void ClDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  store.loadModule(name, image);
}
bool ClDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return store.moduleLoaded(name);
}
uintptr_t ClDevice::mallocDevice(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (svm) {
    void *p = svm.alloc(*context, /*CL_MEM_READ_WRITE*/ 1 << 0 | svm.memFlags, size, 0);
    if (!p) POLYINVOKE_FATAL(PREFIX, "clSVMAlloc failed for %zu bytes", size);
    if (svmTracker) {
      std::lock_guard lock(svmTracker->mtx);
      svmTracker->entries.emplace(p, details::SVMTracker::Entry{size, /*mappedForHost*/ false});
    }
    return reinterpret_cast<uintptr_t>(p);
  }
  cl_mem_flags flags = {};
  switch (access) {
    case Access::RO: flags = CL_MEM_READ_ONLY; break;
    case Access::WO: flags = CL_MEM_WRITE_ONLY; break;
    case Access::RW:
    default: flags = CL_MEM_READ_WRITE; break;
  }
  return memoryObjects.malloc(OUT_CHECKED(clCreateBuffer(*context, flags, size, nullptr, OUT_ERR)));
}

void ClDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();

  if (svm) {
    if (svmTracker) {
      std::lock_guard lock(svmTracker->mtx);
      svmTracker->entries.erase(reinterpret_cast<void *>(ptr));
    }
    svm.free(*context, reinterpret_cast<void *>(ptr));
    return;
  }
  if (auto mem = memoryObjects.query(ptr); mem) {
    CHECKED(clReleaseMemObject(*mem));
    memoryObjects.erase(ptr);
  } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %ld", ptr);
}
std::optional<void *> ClDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (!svm) return std::nullopt;
  void *p = svm.alloc(*context, /*CL_MEM_READ_WRITE*/ 1 << 0 | svm.memFlags, size, 0);
  if (!p) return std::nullopt;
  if (svmTracker) {
    std::lock_guard lock(svmTracker->mtx);
    svmTracker->entries.emplace(p, details::SVMTracker::Entry{size, /*mappedForHost*/ false});
  }
  return p;
}
void ClDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  if (!svm) POLYINVOKE_FATAL(PREFIX, "Unsupported: %p", ptr);
  if (svmTracker) {
    std::lock_guard lock(svmTracker->mtx);
    svmTracker->entries.erase(ptr);
  }
  svm.free(*context, ptr);
}
std::unique_ptr<DeviceQueue> ClDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<ClDeviceQueue>(
      timeout, store, OUT_CHECKED(clCreateCommandQueue(*context, *device, 0, OUT_ERR)),
      [this](auto &&ptr) {
        if (auto mem = memoryObjects.query(ptr); mem) {
          return *mem;
        } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %ld", ptr);
      },
      svm, svmTracker);
}
ClDevice::~ClDevice() { POLYINVOKE_TRACE(); }

// ---

ClDeviceQueue::ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                             decltype(queryMemObject) queryMemObject, details::SVMFns svm,
                             std::shared_ptr<details::SVMTracker> svmTracker)
    : latch(timeout), store(store), queue(queue), queryMemObject(std::move(queryMemObject)), svm(svm), svmTracker(std::move(svmTracker)) {
  POLYINVOKE_TRACE();
}
ClDeviceQueue::~ClDeviceQueue() {
  POLYINVOKE_TRACE();
  CHECKED(clReleaseCommandQueue(queue));
}
void ClDeviceQueue::unmapAllSvmForDevice() {
  // Fine-grain SVM is auto-coherent; coarse-grain needs explicit map/unmap. If the impl didn't
  // expose map/unmap symbols there's nothing safe we can do, so silently fall through.
  if (!svmTracker || !svm.unmap || !svm.coarseGrain()) return;
  std::lock_guard lock(svmTracker->mtx);
  for (auto &[ptr, entry] : svmTracker->entries) {
    if (!entry.mappedForHost) continue;
    CHECKED(svm.unmap(queue, ptr, 0, nullptr, nullptr));
    entry.mappedForHost = false;
  }
}
void ClDeviceQueue::mapAllSvmForHost() {
  if (!svmTracker || !svm.map || !svm.coarseGrain()) return;
  std::lock_guard lock(svmTracker->mtx);
  for (auto &[ptr, entry] : svmTracker->entries) {
    if (entry.mappedForHost) continue;
    // CL_MAP_READ | CL_MAP_WRITE = 0x1 | 0x2 -- blocking so the caller can read immediately.
    CHECKED(svm.map(queue, CL_TRUE, 0x3, ptr, entry.size, 0, nullptr, nullptr));
    entry.mappedForHost = true;
  }
}
void ClDeviceQueue::enqueueCallback(const MaybeCallback &cb, cl_event event) {
  POLYINVOKE_TRACE();
  if (!cb) return;
  // SVM transfer paths use blocking memcpy and don't produce an event; the operation has
  // already completed synchronously by the time we get here, so invoke the callback directly
  // (clSetEventCallback would return CL_INVALID_EVENT on a null event).
  if (!event) {
    (*cb)();
    return;
  }
  static detail::CountedCallbackHandler handler;
  CHECKED(clSetEventCallback(
      event, CL_COMPLETE,
      [](cl_event e, cl_int status, void *data) {
        CHECKED(clReleaseEvent(e));
        CHECKED(status);
        handler.consume(data);
      },
      handler.createHandle([cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      })));
  CHECKED(clFlush(queue));
}
void ClDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                               const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (svm) {
    // XXX Coarse-grain SVMMemcpy is UB if either region is host-mapped; unmap first.
    unmapAllSvmForDevice();
    auto *srcP = reinterpret_cast<const char *>(src) + srcOffset;
    auto *dstP = reinterpret_cast<char *>(dst) + dstOffset;
    // XXX must be blocking on the host side as the src needs to be valid until the copy completes
    CHECKED(svm.memcpy(queue, CL_TRUE, dstP, srcP, size, 0, nullptr, nullptr));
  } else {
    CHECKED(clEnqueueCopyBuffer(queue, queryMemObject(src), queryMemObject(dst), srcOffset, dstOffset, size, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!src) POLYINVOKE_FATAL(PREFIX, "Source pointer is NULL, destination=%lu", dst);
  if (svm) {
    unmapAllSvmForDevice();
    auto *dstP = reinterpret_cast<char *>(dst) + dstOffset;
    CHECKED(svm.memcpy(queue, CL_TRUE, dstP, src, size, 0, nullptr, nullptr));
  } else {
    CHECKED(clEnqueueWriteBuffer(queue, queryMemObject(dst), CL_FALSE, dstOffset, size, src, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!dst) POLYINVOKE_FATAL(PREFIX, "Destination pointer is NULL, source=%lu", src);
  if (svm) {
    unmapAllSvmForDevice();
    auto *srcP = reinterpret_cast<const char *>(src) + srcOffset;
    CHECKED(svm.memcpy(queue, CL_TRUE, dst, srcP, size, 0, nullptr, nullptr));
  } else {
    CHECKED(clEnqueueReadBuffer(queue, queryMemObject(src), CL_FALSE, srcOffset, size, dst, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                       std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void)
    POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported, was %s", magic_enum::enum_name(types.back()).data());
  auto kernel = store.resolveFunction(moduleName, symbol, types);
  auto toSize = [](Type t) -> size_t {
    switch (t) {
      case Type::Ptr: return sizeof(cl_mem);
      case Type::Void: POLYINVOKE_FATAL(PREFIX, "Illegal argument type: %s", magic_enum::enum_name(t).data());
      default: return byteOfType(t);
    }
  };

  const auto args = detail::argDataAsPointers(types, argData);
  const auto [local, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  const auto global = Dim3{policy.global.x * local.x, policy.global.y * local.y, policy.global.z * local.z};

  // last arg is the return, void assertion should have been done before this
  for (cl_uint i = 0; i < types.size() - 1; ++i) {
    const auto rawPtr = args[i];
    switch (const auto tpe = types[i]) {
      case Type::Ptr: {
        static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
        uintptr_t ptr = {};
        std::memcpy(&ptr, rawPtr, byteOfType(Type::Ptr));
        if (svm) {
          CHECKED(svm.setKernelArg(kernel, i, reinterpret_cast<void *>(ptr)));
        } else {
          cl_mem mem = queryMemObject(ptr);
          CHECKED(clSetKernelArg(kernel, i, toSize(tpe), &mem));
        }
      } break;
      case Type::Scratch: {
        CHECKED(clSetKernelArg(kernel, i, sharedMem, nullptr));
        break;
      }
      default: {
        CHECKED(clSetKernelArg(kernel, i, toSize(tpe), rawPtr));
        break;
      }
    }
  }
  if (svm) {
    // Indirect SVM allocs (e.g. one captured inside a marshalled lambda struct) must be
    // declared via CL_KERNEL_EXEC_INFO_SVM_PTRS or the runtime treats them as unused by the
    // kernel and skips coherency for them.
    std::vector<void *> svmPtrs;
    svmPtrs.reserve(types.size() + (svmTracker ? svmTracker->entries.size() : 0));
    for (cl_uint i = 0; i < types.size() - 1; ++i) {
      if (types[i] != Type::Ptr) continue;
      uintptr_t ptr = {};
      std::memcpy(&ptr, args[i], byteOfType(Type::Ptr));
      // NVIDIA OpenCL rejects clSetKernelExecInfo on a null entry with CL_INVALID_VALUE.
      if (ptr) svmPtrs.push_back(reinterpret_cast<void *>(ptr));
    }
    if (svmTracker) {
      std::lock_guard lock(svmTracker->mtx);
      for (auto &[ptr, _] : svmTracker->entries) svmPtrs.push_back(ptr);
    }
    if (!svmPtrs.empty())
      CHECKED(svm.setKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS_, svmPtrs.size() * sizeof(void *), svmPtrs.data()));
  }

  POLYINVOKE_TRACE();
  unmapAllSvmForDevice();
  cl_event event = {};
  CHECKED(clEnqueueNDRangeKernel(queue, kernel,         //
                                 3,                     //
                                 nullptr,               //
                                 global.sizes().data(), //
                                 local.sizes().data(),  //
                                 0, nullptr, &event));
  enqueueCallback(cb, event);
  CHECKED(clFlush(queue));
}
void ClDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  cl_event event = {};
  CHECKED(clEnqueueBarrierWithWaitList(queue, 0, nullptr, &event));
  CHECKED(clWaitForEvents(1, &event));
  mapAllSvmForHost();
}

#undef CHECKED
#undef OUT_CHECKED

#include "polyinvoke/cl_platform.h"

#include <cstring>
#include <thread>

#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/env.h"
#include "polyregion/env_keys.h"

#include "dl_util.h"
#include "vendor_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cl;

static constexpr const char *PREFIX = "OpenCL";

static const char *clewErrorString(cl_int error) {
  static const char *strings[] = {
      "CL_SUCCESS",
      "CL_DEVICE_NOT_FOUND",
      "CL_DEVICE_NOT_AVAILABLE",
      "CL_COMPILER_NOT_AVAILABLE",
      "CL_MEM_OBJECT_ALLOCATION_FAILURE",
      "CL_OUT_OF_RESOURCES",
      "CL_OUT_OF_HOST_MEMORY",
      "CL_PROFILING_INFO_NOT_AVAILABLE",
      "CL_MEM_COPY_OVERLAP",
      "CL_IMAGE_FORMAT_MISMATCH",
      "CL_IMAGE_FORMAT_NOT_SUPPORTED",
      "CL_BUILD_PROGRAM_FAILURE",
      "CL_MAP_FAILURE",
      "CL_MISALIGNED_SUB_BUFFER_OFFSET",
      "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
      "CL_COMPILE_PROGRAM_FAILURE",
      "CL_LINKER_NOT_AVAILABLE",
      "CL_LINK_PROGRAM_FAILURE",
      "CL_DEVICE_PARTITION_FAILED",
      "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "CL_INVALID_VALUE",
      "CL_INVALID_DEVICE_TYPE",
      "CL_INVALID_PLATFORM",
      "CL_INVALID_DEVICE",
      "CL_INVALID_CONTEXT",
      "CL_INVALID_QUEUE_PROPERTIES",
      "CL_INVALID_COMMAND_QUEUE",
      "CL_INVALID_HOST_PTR",
      "CL_INVALID_MEM_OBJECT",
      "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
      "CL_INVALID_IMAGE_SIZE",
      "CL_INVALID_SAMPLER",
      "CL_INVALID_BINARY",
      "CL_INVALID_BUILD_OPTIONS",
      "CL_INVALID_PROGRAM",
      "CL_INVALID_PROGRAM_EXECUTABLE",
      "CL_INVALID_KERNEL_NAME",
      "CL_INVALID_KERNEL_DEFINITION",
      "CL_INVALID_KERNEL",
      "CL_INVALID_ARG_INDEX",
      "CL_INVALID_ARG_VALUE",
      "CL_INVALID_ARG_SIZE",
      "CL_INVALID_KERNEL_ARGS",
      "CL_INVALID_WORK_DIMENSION",
      "CL_INVALID_WORK_GROUP_SIZE",
      "CL_INVALID_WORK_ITEM_SIZE",
      "CL_INVALID_GLOBAL_OFFSET",
      "CL_INVALID_EVENT_WAIT_LIST",
      "CL_INVALID_EVENT",
      "CL_INVALID_OPERATION",
      "CL_INVALID_GL_OBJECT",
      "CL_INVALID_BUFFER_SIZE",
      "CL_INVALID_MIP_LEVEL",
      "CL_INVALID_GLOBAL_WORK_SIZE",
      "CL_INVALID_PROPERTY",
      "CL_INVALID_IMAGE_DESCRIPTOR",
      "CL_INVALID_COMPILER_OPTIONS",
      "CL_INVALID_LINKER_OPTIONS",
      "CL_INVALID_DEVICE_PARTITION_COUNT",
  };
  static const int num_errors = sizeof(strings) / sizeof(strings[0]);
  if (error == -1001) return "CL_PLATFORM_NOT_FOUND_KHR";
  if (error > 0 || -error >= num_errors) return "Unknown OpenCL error";
  return strings[-error];
}

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

static std::string queryPlatformInfo(cl_platform_id platform, cl_platform_info info) {
  size_t size = 0;
  CHECKED(clGetPlatformInfo(platform, info, 0, nullptr, &size));
  std::string data(size - 1, '\0');
  CHECKED(clGetPlatformInfo(platform, info, size, data.data(), nullptr));
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

// memflags to OR into clSVMAlloc (0 = coarse-grain, FINE_GRAIN otherwise); nullopt = fall back to cl_mem
std::optional<cl_bitfield> resolveSVM(cl_device_id device, const std::string &platformName) {
  if (const char *off = std::getenv(polyregion::env::PolyinvokeDisableSvm); off && *off && *off != '0') return std::nullopt;
  // XXX rusticl advertises SVM caps but indirect SVM access faults; force the buffer path
  if (platformName.find("rusticl") != std::string::npos) return std::nullopt;
  cl_bitfield caps = 0;
  if (clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES_, sizeof(caps), &caps, nullptr) != CL_SUCCESS) return std::nullopt;
  if (!(caps & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_ | CL_DEVICE_SVM_FINE_GRAIN_BUFFER_))) return std::nullopt;
  if (!clSVMAlloc || !clSVMFree || !clEnqueueSVMMemcpy || !clSetKernelArgSVMPointer || !clSetKernelExecInfo) return std::nullopt;
  return (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER_) ? cl_bitfield(CL_MEM_SVM_FINE_GRAIN_BUFFER_) : cl_bitfield(0);
}
} // namespace

std::variant<std::string, std::unique_ptr<Platform>> ClPlatform::create() {
  // XXX FP64 is emulated on Intel Arc and needs to be enabled via environment variable
  // we set it unless it's already defined with some other value
  env::put("OverrideDefaultFP64Settings", "1", false);
  env::put("IGC_EnableDPEmulation", "1", false);
#ifdef _WIN32
  void *lib = dl::open_first({"OpenCL.dll"});
#elif defined(__APPLE__)
  void *lib = dl::open_first({"/Library/Frameworks/OpenCL.framework/OpenCL", "/System/Library/Frameworks/OpenCL.framework/OpenCL"});
#else
  void *lib = dl::open_first({"libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.0", "libOpenCL.so.2"});
#endif
  if (!lib) return "OpenCL: failed to open libOpenCL dynamic library";
  clew_cl_resolve(dl::lookup, lib);
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
  const cl_device_type AcceleratorMask = CL_DEVICE_TYPE_ALL;
  for (const auto &platform : platforms) {
    cl_uint numDevices = 0;
    if (const auto deviceIdResult = clGetDeviceIDs(platform, AcceleratorMask, 0, nullptr, &numDevices);
        deviceIdResult == CL_DEVICE_NOT_FOUND) {
      continue;
    } else CHECKED(deviceIdResult);

    std::vector<cl_device_id> devices(numDevices);
    CHECKED(clGetDeviceIDs(platform, AcceleratorMask, numDevices, devices.data(), nullptr));
    const auto platformName = queryPlatformInfo(platform, CL_PLATFORM_NAME);
    auto ilFn =
        reinterpret_cast<details::ClCreateProgramWithIL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithIL"));
    if (!ilFn)
      ilFn = reinterpret_cast<details::ClCreateProgramWithIL_fn>(
          clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithILKHR"));
    for (auto &device : devices) {
      auto svm = resolveSVM(device, platformName);
      clDevices.push_back(std::make_unique<ClDevice>(device, ModuleFormat::Source, nullptr, svm, platformName));
      if (ilFn && deviceSupportsIL(device))
        clDevices.push_back(std::make_unique<ClDevice>(device, ModuleFormat::SPIRV_Kernel, ilFn, svm, platformName));
    }
  }
  return clDevices;
}
ClPlatform::~ClPlatform() { POLYINVOKE_TRACE(); }

// ---

static DeviceQuirks resolveQuirks(const std::string &deviceName) {
  const bool llvmpipe = deviceName.find("llvmpipe") != std::string::npos;
  return DeviceQuirks{/*nativeTrig*/ llvmpipe, /*overReadSlackBytes*/ llvmpipe ? size_t{4096} : size_t{0}};
}

ClDevice::ClDevice(cl_device_id device, ModuleFormat format, details::ClCreateProgramWithIL_fn ilCreateFn, std::optional<cl_bitfield> svm,
                   const std::string &platformName)
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
      // XXX kernel-format suffix disambiguates source vs SPIRV_Kernel instances over the same cl_device_id.
      deviceName(queryDeviceInfo(device, CL_DEVICE_NAME) + (format == ModuleFormat::SPIRV_Kernel ? " [SPIR-V]" : " [source]")),
      format(format), ilCreateFn(ilCreateFn), svm(svm), svmTracker(svm ? std::make_shared<details::SVMTracker>() : nullptr),
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
            // XXX llvmpipe libclc crashes its JIT on precise sin/cos/tan range-reduction; route POLY_* trig to
            // native_ for that device only (source #ifdef; SPIR-V is pre-compiled)
            const std::string compilerArgs =
                (this->format != ModuleFormat::SPIRV_Kernel && this->quirks.nativeTrig) ? "-DPOLY_NATIVE_TRIG" : "";
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
PhysicalDevice ClDevice::physicalDevice() {
  POLYINVOKE_TRACE();
  // XXX CPU OpenCL stacks (llvmpipe, pocl) report host scheme (needsLock()==false) so they run lock-free in parallel
  if (cl_device_type dtype = 0;
      clGetDeviceInfo(*device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr) == CL_SUCCESS && (dtype & CL_DEVICE_TYPE_CPU))
    return PhysicalDevice::host();
  // XXX clew doesn't have cl_khr_pci_bus_info/cl_khr_device_uuid tokens; use their published values.
  // we do PCI first then UUID fallback
  constexpr cl_device_info PCI_BUS_INFO_KHR = 0x410F, UUID_KHR = 0x106A;
  struct PciBusInfoKHR {
    cl_uint domain, bus, device, function;
  } pci{};
  if (clGetDeviceInfo(*device, PCI_BUS_INFO_KHR, sizeof(pci), &pci, nullptr) == CL_SUCCESS)
    return PhysicalDevice::pci(pci.domain, static_cast<uint8_t>(pci.bus), static_cast<uint8_t>(pci.device),
                               static_cast<uint8_t>(pci.function));
  // XXX AMD runtimes predate cl_khr_pci_bus_info; without the BDF from cl_amd_device_topology the
  // APU device gets a synthetic key and never serialises with HIP/HSA on the same device
  if (queryDeviceInfo(*device, CL_DEVICE_EXTENSIONS).find("cl_amd_device_attribute_query") != std::string::npos) {
    constexpr cl_device_info TOPOLOGY_AMD = 0x4037;
    constexpr cl_uint TOPOLOGY_TYPE_PCIE_AMD = 1;
    struct { // cl_device_topology_amd::pcie; no padding possible, all members align <= 4
      cl_uint type;
      cl_char unused[17];
      cl_char bus, dev, function;
    } topo{};
    static_assert(sizeof(topo) == 24, "must match cl_device_topology_amd");
    if (clGetDeviceInfo(*device, TOPOLOGY_AMD, sizeof(topo), &topo, nullptr) == CL_SUCCESS && topo.type == TOPOLOGY_TYPE_PCIE_AMD)
      return PhysicalDevice::pci(0, static_cast<uint8_t>(topo.bus), static_cast<uint8_t>(topo.dev), static_cast<uint8_t>(topo.function));
  }
  std::array<uint8_t, 16> uuid{};
  if (clGetDeviceInfo(*device, UUID_KHR, uuid.size(), uuid.data(), nullptr) == CL_SUCCESS) return PhysicalDevice::uuid(uuid);
  return PhysicalDevice::synthetic(Backend::OpenCL, static_cast<int64_t>(std::hash<std::string>{}(deviceName)));
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
// Track every SVM alloc (device-only and shared) so it participates in CL_KERNEL_EXEC_INFO_SVM_PTRS
// indirect-SVM declarations and coarse-grain map/unmap; without tracking, reductions through
// pointer-in-pointer args silently zero partials on Intel/NVIDIA.
void ClDevice::trackSvm(void *p, size_t size) {
  if (!svmTracker) return;
  std::lock_guard lock(svmTracker->mtx);
  svmTracker->entries.emplace(p, details::SVMTracker::Entry{size, /*mappedForHost*/ false});
}
void ClDevice::untrackSvm(void *p) {
  if (!svmTracker) return;
  std::lock_guard lock(svmTracker->mtx);
  svmTracker->entries.erase(p);
}

uintptr_t ClDevice::mallocDevice(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  if (svm) {
    void *p = clSVMAlloc(*context, /*CL_MEM_READ_WRITE*/ 1 << 0 | *svm, size, 0);
    if (!p) POLYINVOKE_FATAL(PREFIX, "clSVMAlloc failed for %zu bytes", size);
    trackSvm(p, size);
    return reinterpret_cast<uintptr_t>(p);
  }
  cl_mem_flags flags = {};
  switch (access) {
    case Access::RO: flags = CL_MEM_READ_ONLY; break;
    case Access::WO: flags = CL_MEM_WRITE_ONLY; break;
    case Access::RW:
    default: flags = CL_MEM_READ_WRITE; break;
  }
  // llvmpipe doesn't predicate inactive SIMD remainder lanes, so a non-SIMD-multiple trip count reads
  // past the buffer (SIGSEGV into dirty host heap); over-allocate zeroed slack to absorb the over-read
  if (const size_t slack = quirks.overReadSlackBytes; slack > 0) {
    std::vector<char> zeros(size + slack, 0);
    return memoryObjects.malloc(OUT_CHECKED(clCreateBuffer(*context, flags | CL_MEM_COPY_HOST_PTR, size + slack, zeros.data(), OUT_ERR)));
  }
  return memoryObjects.malloc(OUT_CHECKED(clCreateBuffer(*context, flags, size, nullptr, OUT_ERR)));
}

void ClDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  context.touch();

  if (svm) {
    untrackSvm(reinterpret_cast<void *>(ptr));
    clSVMFree(*context, reinterpret_cast<void *>(ptr));
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
  if (!svm || *svm == 0) return std::nullopt;
  void *p = clSVMAlloc(*context, /*CL_MEM_READ_WRITE*/ 1 << 0 | *svm, size, 0);
  if (!p) return std::nullopt;
  trackSvm(p, size);
  return p;
}
void ClDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  if (!svm) POLYINVOKE_FATAL(PREFIX, "Unsupported: %p", ptr);
  untrackSvm(ptr);
  clSVMFree(*context, ptr);
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
                             decltype(queryMemObject) queryMemObject, std::optional<cl_bitfield> svm,
                             std::shared_ptr<details::SVMTracker> svmTracker)
    : latch(timeout), store(store), queue(queue), queryMemObject(std::move(queryMemObject)), svm(svm), svmTracker(std::move(svmTracker)) {
  POLYINVOKE_TRACE();
}
ClDeviceQueue::~ClDeviceQueue() {
  POLYINVOKE_TRACE();
  CHECKED(clReleaseCommandQueue(queue));
}
void ClDeviceQueue::unmapAllSvmForDevice() {
  // fine-grain SVM is auto-coherent; coarse-grain needs explicit map/unmap (skip if the impl lacks them)
  if (!svmTracker || !clEnqueueSVMUnmap || *svm != 0) return;
  std::lock_guard lock(svmTracker->mtx);
  for (auto &[ptr, entry] : svmTracker->entries) {
    if (!entry.mappedForHost) continue;
    CHECKED(clEnqueueSVMUnmap(queue, ptr, 0, nullptr, nullptr));
    entry.mappedForHost = false;
  }
}
void ClDeviceQueue::mapAllSvmForHost() {
  if (!svmTracker || !clEnqueueSVMMap || *svm != 0) return;
  std::lock_guard lock(svmTracker->mtx);
  for (auto &[ptr, entry] : svmTracker->entries) {
    if (entry.mappedForHost) continue;
    // 0x3 = CL_MAP_READ | CL_MAP_WRITE; blocking so the caller can read immediately
    CHECKED(clEnqueueSVMMap(queue, CL_TRUE, 0x3, ptr, entry.size, 0, nullptr, nullptr));
    entry.mappedForHost = true;
  }
}
void ClDeviceQueue::enqueueCallback(const MaybeCallback &cb, cl_event event) {
  POLYINVOKE_TRACE();
  if (!cb) return;
  // SVM paths use blocking memcpy with no event (already complete); invoke cb directly, clSetEventCallback
  // would return CL_INVALID_EVENT on a null event
  if (!event) {
    (*cb)();
    return;
  }
  CHECKED(clSetEventCallback(
      event, CL_COMPLETE,
      [](cl_event e, cl_int status, void *data) {
        CHECKED(clReleaseEvent(e));
        CHECKED(status);
        detail::CountedCallbackHandler::instance().consume(data);
      },
      detail::CountedCallbackHandler::instance().createHandle([cb, token = latch.acquire()]() {
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
    // blocking: src must stay valid until the copy completes
    CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dstP, srcP, size, 0, nullptr, nullptr));
  } else {
    CHECKED(clEnqueueCopyBuffer(queue, queryMemObject(src), queryMemObject(dst), srcOffset, dstOffset, size, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!src) POLYINVOKE_FATAL(PREFIX, "Source pointer is NULL, destination=%lu", dst);
  if (size == 0) return enqueueCallback(cb, {});
  if (svm) {
    unmapAllSvmForDevice();
    auto *dstP = reinterpret_cast<char *>(dst) + dstOffset;
    CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dstP, src, size, 0, nullptr, nullptr));
  } else {
    CHECKED(clEnqueueWriteBuffer(queue, queryMemObject(dst), CL_FALSE, dstOffset, size, src, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!dst) POLYINVOKE_FATAL(PREFIX, "Destination pointer is NULL, source=%lu", src);
  // XXX zero-byte is a no-op; ReadBuffer/SVMMemcpy reject size 0 with CL_INVALID_VALUE (an -O3 reflect can size a result to 0)
  if (size == 0) return enqueueCallback(cb, {});
  if (svm) {
    auto *srcP = reinterpret_cast<char *>(src) + srcOffset;
    if (*svm == 0 && clEnqueueSVMMap && clEnqueueSVMUnmap) {
      // Intel NEO SVMMemcpy reads stale coarse-grain SVM after a kernel write; Map/Unmap forces the flush
      CHECKED(clEnqueueSVMMap(queue, CL_TRUE, /*CL_MAP_READ*/ 0x1, srcP, size, 0, nullptr, nullptr));
      std::memcpy(dst, srcP, size);
      CHECKED(clEnqueueSVMUnmap(queue, srcP, 0, nullptr, nullptr));
    } else {
      unmapAllSvmForDevice();
      CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dst, srcP, size, 0, nullptr, nullptr));
    }
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

  for (cl_uint i = 0; i < types.size() - 1; ++i) {
    const auto rawPtr = args[i];
    switch (const auto tpe = types[i]) {
      case Type::Ptr: {
        static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
        uintptr_t ptr = {};
        std::memcpy(&ptr, rawPtr, byteOfType(Type::Ptr));
        if (svm) {
          CHECKED(clSetKernelArgSVMPointer(kernel, i, reinterpret_cast<void *>(ptr)));
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
    // indirect SVM allocs need CL_KERNEL_EXEC_INFO_SVM_PTRS or the driver skips coherency; some drivers
    // reject the batched call but accept it per-pointer, so on CL_INVALID_VALUE retry per-pointer
    std::vector<void *> allSvmPtrs;
    allSvmPtrs.reserve(types.size() + (svmTracker ? svmTracker->entries.size() : 0));
    for (cl_uint i = 0; i < types.size() - 1; ++i) {
      if (types[i] != Type::Ptr) continue;
      uintptr_t ptr = {};
      std::memcpy(&ptr, args[i], byteOfType(Type::Ptr));
      // NVIDIA OpenCL rejects clSetKernelExecInfo on a null entry with CL_INVALID_VALUE.
      if (ptr) allSvmPtrs.push_back(reinterpret_cast<void *>(ptr));
    }
    if (svmTracker) {
      std::lock_guard lock(svmTracker->mtx);
      for (auto &[ptr, _] : svmTracker->entries)
        allSvmPtrs.push_back(ptr);
    }
    auto declare = [&](void *const *ptrs, size_t n) {
      return n == 0 ? CL_SUCCESS : clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS_, n * sizeof(void *), ptrs);
    };
    cl_int rc = declare(allSvmPtrs.data(), allSvmPtrs.size());
    if (rc == CL_INVALID_VALUE) {
      size_t rejected = 0;
      for (void *ptr : allSvmPtrs) {
        cl_int rcOne = declare(&ptr, 1);
        if (rcOne == CL_INVALID_VALUE) ++rejected;
        else if (rcOne != CL_SUCCESS) CHECKED(rcOne);
      }
      if (rejected > 0)
        fmt::print(stderr, "[OpenCL] WARN: setKernelExecInfo rejected {} / {} SVM pointers (likely indirect-SVM not supported)\n", rejected,
                   allSvmPtrs.size());
      rc = CL_SUCCESS;
    }
    if (rc != CL_SUCCESS) CHECKED(rc);
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

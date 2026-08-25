#include "polyinvoke/cl_platform.h"

#include <cinttypes>
#include <cstring>
#include <limits>
#include <thread>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"
#include "spirv/unified1/OpenCL.std.h"
#include "spirv/unified1/spirv.hpp"

#include "polyinvoke/module_cache.h"
#include "polyregion/env.h"
#include "polyregion/env_keys.h"

#include "dl_util.h"
#include "vendor_utils.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cl;
namespace cl_details = polyregion::invoke::cl::details;

static constexpr const char *PREFIX = "OpenCL";
static constexpr cl_ulong NullPointerOffset = std::numeric_limits<cl_ulong>::max();

auto cl_details::SVMTracker::owner(uintptr_t ptr) {
  const auto next = entries.upper_bound(ptr);
  if (next == entries.begin()) return entries.end();
  const auto candidate = std::prev(next);
  return ptr - candidate->first < candidate->second.size ? candidate : entries.end();
}

auto cl_details::SVMTracker::owner(uintptr_t ptr) const {
  const auto next = entries.upper_bound(ptr);
  if (next == entries.begin()) return entries.end();
  const auto candidate = std::prev(next);
  return ptr - candidate->first < candidate->second.size ? candidate : entries.end();
}

void cl_details::SVMTracker::track(void *ptr, size_t size) {
  std::lock_guard lock(mutex);
  const auto base = reinterpret_cast<uintptr_t>(ptr);
  const auto inherited = leakedHostMaps.erase(base) != 0 && !freeReleasesMap;
  entries.insert_or_assign(base, Entry{size, inherited ? Ownership::InheritedHost : Ownership::Device});
}

void cl_details::SVMTracker::untrack(void *ptr) {
  std::lock_guard lock(mutex);
  const auto base = reinterpret_cast<uintptr_t>(ptr);
  const auto it = entries.find(base);
  if (it == entries.end()) return;
  if (it->second.ownership != Ownership::Device) leakedHostMaps.insert(base);
  entries.erase(it);
}

std::optional<cl_int> cl_details::SVMTracker::mapForHost(void *ptr, const Map &map) {
  std::lock_guard lock(mutex);
  const auto it = owner(reinterpret_cast<uintptr_t>(ptr));
  if (it == entries.end()) return {};
  if (it->second.ownership == Ownership::Device) {
    const auto result = map(reinterpret_cast<void *>(it->first), it->second.size);
    if (result != CL_SUCCESS) return result;
    it->second.ownership = Ownership::Host;
  }
  return CL_SUCCESS;
}

cl_int cl_details::SVMTracker::mapAllForHost(const Map &map) {
  std::lock_guard lock(mutex);
  for (auto &[base, entry] : entries) {
    if (entry.ownership != Ownership::Device) continue;
    if (const auto result = map(reinterpret_cast<void *>(base), entry.size); result != CL_SUCCESS) return result;
    entry.ownership = Ownership::Host;
  }
  return CL_SUCCESS;
}

cl_int cl_details::SVMTracker::unmapAllForDevice(const Unmap &unmap) {
  std::lock_guard lock(mutex);
  for (auto &[base, entry] : entries) {
    if (entry.ownership == Ownership::Device) continue;
    if (const auto result = unmap(reinterpret_cast<void *>(base)); result != CL_SUCCESS) {
      if (result != CL_INVALID_VALUE || entry.ownership != Ownership::InheritedHost) return result;
      freeReleasesMap = true;
    }
    entry.ownership = Ownership::Device;
  }
  return CL_SUCCESS;
}

std::vector<void *> cl_details::SVMTracker::pointers() const {
  std::lock_guard lock(mutex);
  return entries | aspartame::keys() | aspartame::map([](uintptr_t base) { return reinterpret_cast<void *>(base); })
         | aspartame::to_vector();
}

std::optional<cl_details::SVMTracker::Ownership> cl_details::SVMTracker::ownership(void *ptr) const {
  std::lock_guard lock(mutex);
  const auto it = owner(reinterpret_cast<uintptr_t>(ptr));
  return it == entries.end() ? std::nullopt : std::optional{it->second.ownership};
}

bool cl_details::SVMTracker::freeReleasesHostMap() const {
  std::lock_guard lock(mutex);
  return freeReleasesMap;
}

std::string cl_details::errorString(cl_int error) {
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
  const auto index = -static_cast<int64_t>(error);
  if (index >= 0 && index < num_errors && strings[index][0] != '\0') return strings[index];
  return fmt::format("unknown OpenCL error ({})", error);
}

cl_details::LaunchDimensions cl_details::launchDimensions(const Dim3 &groups, const Dim3 &local) {
  return {Dim3{groups.x * local.x, groups.y * local.y, groups.z * local.z}, local};
}

std::optional<cl_details::LaunchDimensions> cl_details::retryLaunchDimensions(cl_int error, const Dim3 &groups, const Dim3 &local,
                                                                              size_t kernelMax) {
  if (error != CL_INVALID_WORK_GROUP_SIZE || local.x <= 1 || kernelMax == 0) return {};
  const auto yz = local.y * local.z;
  return std::optional{kernelMax / yz} | aspartame::filter([&](size_t x) { return yz <= kernelMax && x > 0 && x < local.x; })
         | aspartame::map([&](size_t x) { return launchDimensions(groups, Dim3{x, local.y, local.z}); });
}

#define CHECKED(f__)                                                                                                                       \
  do {                                                                                                                                     \
    cl_int result__ = (f__);                                                                                                               \
    if (result__ != CL_SUCCESS) {                                                                                                          \
      const auto message__ = cl_details::errorString(result__);                                                                            \
      POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", __FILE__, __LINE__, message__.c_str());                                                        \
    }                                                                                                                                      \
  } while (0)

#define OUT_ERR err__
#define OUT_CHECKED(f__)                                                                                                                   \
  ([&]() {                                                                                                                                 \
    cl_int result__ = CL_SUCCESS;                                                                                                          \
    auto OUT_ERR = &result__;                                                                                                              \
    auto x__ = (f__);                                                                                                                      \
    if (result__ == CL_SUCCESS) return x__;                                                                                                \
    const auto message__ = cl_details::errorString(result__);                                                                              \
    POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", __FILE__, __LINE__, message__.c_str());                                                          \
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

std::string programIdentity(cl_device_id device, const std::string &deviceName, const std::string &compilerArgs) {
  return fmt::format("{}|{}|{}|{}|{}", deviceName, queryDeviceInfo(device, CL_DEVICE_VENDOR), queryDeviceInfo(device, CL_DEVICE_VERSION),
                     queryDeviceInfo(device, CL_DRIVER_VERSION), compilerArgs);
}

bool deviceSupportsIL(cl_device_id device) {
  size_t size = 0;
  if (clGetDeviceInfo(device, /*CL_DEVICE_IL_VERSION=*/0x105B, 0, nullptr, &size) == CL_SUCCESS && size > 1) return true;
  return queryDeviceInfo(device, CL_DEVICE_EXTENSIONS) ^ aspartame::contains_slice("cl_khr_il_program");
}

// memflags to OR into clSVMAlloc (0 = coarse-grain, FINE_GRAIN otherwise); nullopt = fall back to cl_mem
std::optional<cl_bitfield> resolveSVM(cl_device_id device, const std::string &platformName) {
  if (const char *off = std::getenv(polyregion::env::PolyinvokeDisableSvm); off && *off && *off != '0') return std::nullopt;
  // XXX rusticl advertises SVM caps but indirect SVM access faults; force the buffer path
  if (platformName ^ aspartame::contains_slice("rusticl")) return std::nullopt;
  // gfx1036 (Raphael) / gfx1037 (Mendocino) - the minimal 2-CU RDNA2 desktop/low-power iGPUs - silently
  // corrupt fine-grain SVM under concurrent oversubscription (validated: cl_mem clean, fine-grain SVM
  // ~12/30 stale-read mismatches; gfx1103/gfx1034 and matched oversubscription+clock are unaffected). The
  // defect is specific to this 2-CU RDNA part, so gate on RDNA (gfx>=1000) + <=2 CU and fall back to plain
  // cl_mem buffers; low-CU Vega (gfx9xx) APUs are a different arch and stay on the SVM path
  if (const std::string name = queryDeviceInfo(device, CL_DEVICE_NAME); name ^ aspartame::starts_with("gfx")) {
    cl_uint cus = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    if (std::strtol(name.c_str() + 3, nullptr, 10) >= 1000 && cus <= 2) return std::nullopt;
  }
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
  // XXX Windows searches PATH last
  const char *oclLib = std::getenv(polyregion::env::PolyinvokeOpenclLib);
  void *lib = (oclLib && *oclLib) ? dl::open_first({oclLib}) : nullptr;
  if (!lib) {
#ifdef _WIN32
    lib = dl::open_first({"OpenCL.dll"});
#elif defined(__APPLE__)
    lib = dl::open_first({"libOpenCL.dylib", "libOpenCL.1.dylib", "/Library/Frameworks/OpenCL.framework/OpenCL",
                          "/System/Library/Frameworks/OpenCL.framework/OpenCL"});
#else
    lib = dl::open_first({"libOpenCL.so.1", "libOpenCL.so", "libOpenCL.so.0", "libOpenCL.so.2"});
#endif
  }
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
        reinterpret_cast<cl_details::ClCreateProgramWithIL_fn>(clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithIL"));
    if (!ilFn)
      ilFn = reinterpret_cast<cl_details::ClCreateProgramWithIL_fn>(
          clGetExtensionFunctionAddressForPlatform(platform, "clCreateProgramWithILKHR"));
    if (!ilFn) ilFn = reinterpret_cast<cl_details::ClCreateProgramWithIL_fn>(clCreateProgramWithIL);
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
  const bool llvmpipe = deviceName ^ aspartame::contains_slice("llvmpipe");
  const bool rusticl = deviceName ^ aspartame::contains_slice("rusticl");
  return DeviceQuirks{/*nativeTrig*/ rusticl || llvmpipe, /*overReadPad*/ llvmpipe ? size_t{4096} : size_t{0}};
}

// Rusticl's libclc JIT fails to link precise sin/cos/tan range-reduction, while llvmpipe historically crashed there;
// rewrite the OpenCL.std trig extinsts to their native_ variants (a same-size swap), dual to -DPOLY_NATIVE_TRIG.
static std::string patchSpirvNativeTrig(const char *data, size_t len) {
  std::string out(data, len);
  if (len < 5 * sizeof(uint32_t) || len % sizeof(uint32_t) != 0) return out;
  auto *w = reinterpret_cast<uint32_t *>(out.data());
  const size_t n = len / sizeof(uint32_t);
  auto opcode = [](uint32_t inst) { return static_cast<uint16_t>(inst & 0xFFFF); };
  auto wcount = [](uint32_t inst) { return static_cast<uint16_t>(inst >> 16); };
  uint32_t openclSet = 0;
  for (size_t i = 5; i < n;) {
    const uint16_t wc = wcount(w[i]);
    if (wc == 0 || i + wc > n) return out;
    if (opcode(w[i]) == spv::OpExtInstImport && wc >= 3
        && std::strncmp(reinterpret_cast<const char *>(&w[i + 2]), "OpenCL.std", sizeof("OpenCL.std")) == 0)
      openclSet = w[i + 1];
    i += wc;
  }
  if (!openclSet) return out;
  for (size_t i = 5; i < n;) {
    const uint16_t wc = wcount(w[i]);
    if (opcode(w[i]) == spv::OpExtInst && wc >= 5 && w[i + 3] == openclSet) {
      switch (w[i + 4]) {
        case OpenCLLIB::Sin: w[i + 4] = OpenCLLIB::Native_sin; break;
        case OpenCLLIB::Cos: w[i + 4] = OpenCLLIB::Native_cos; break;
        case OpenCLLIB::Tan: w[i + 4] = OpenCLLIB::Native_tan; break;
        default: break;
      }
    }
    i += wc;
  }
  return out;
}

ClDevice::ClDevice(cl_device_id device, ModuleFormat format, cl_details::ClCreateProgramWithIL_fn ilCreateFn,
                   std::optional<cl_bitfield> svm, const std::string &platformName)
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
      // <cl_platform>:<cl_device> [<format>] - prefix selects the ICD, suffix the source/SPIR-V instance
      deviceName(platformName + ":" + queryDeviceInfo(device, CL_DEVICE_NAME)
                 + (format == ModuleFormat::SPIRV_Kernel ? " [SPIR-V]" : " [source]")),
      quirks(resolveQuirks(deviceName)), format(format), ilCreateFn(ilCreateFn), svm(svm),
      svmTracker(svm ? std::make_shared<cl_details::SVMTracker>() : nullptr),
      store(
          PREFIX,
          [this](auto &&image) {
            POLYINVOKE_TRACE();
            // XXX Rusticl libclc fails to link precise sin/cos/tan range-reduction, while llvmpipe historically
            // crashed there; route POLY_* trig to native_ via this #ifdef or patchSpirvNativeTrig above.
            const std::string compilerArgs =
                (this->format != ModuleFormat::SPIRV_Kernel && this->quirks.nativeTrig) ? "-DPOLY_NATIVE_TRIG" : "";
            const auto build = [&](cl_program p) { return clBuildProgram(p, 1, &*this->device, compilerArgs.c_str(), nullptr, nullptr); };
            const auto cachePath = detail::moduleCachePath("opencl", programIdentity(*this->device, this->deviceName, compilerArgs), image);
            cl_program program = {};
            if (const auto binary = cache::read(cachePath); !binary.empty()) {
              const size_t binaryLen = binary.size();
              const unsigned char *binaryData = binary.data();
              cl_int binaryStatus = CL_SUCCESS, createStatus = CL_SUCCESS;
              program = clCreateProgramWithBinary(*context, 1, &*this->device, &binaryLen, &binaryData, &binaryStatus, &createStatus);
              if (!program || createStatus != CL_SUCCESS || binaryStatus != CL_SUCCESS || build(program) != CL_SUCCESS) {
                if (program) CHECKED(clReleaseProgram(program));
                program = {};
                cache::evict(cachePath);
              }
            }
            if (!program) {
              auto imageData = image.data();
              auto imageLen = image.size();
              if (this->format == ModuleFormat::SPIRV_Kernel) {
                const std::string spv = this->quirks.nativeTrig ? patchSpirvNativeTrig(imageData, imageLen) : std::string{};
                const char *il = this->quirks.nativeTrig ? spv.data() : imageData;
                program = OUT_CHECKED(this->ilCreateFn(*context, il, this->quirks.nativeTrig ? spv.size() : imageLen, OUT_ERR));
              } else {
                program = OUT_CHECKED(clCreateProgramWithSource(*context, 1, &imageData, &imageLen, OUT_ERR));
              }
              if (const cl_int result = build(program); result != CL_SUCCESS) {
                size_t len;
                CHECKED(clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len));
                std::string buildLog(len, '\0');
                CHECKED(clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, len, buildLog.data(), nullptr));
                auto compilerMessage = cl_details::errorString(result);
                POLYINVOKE_FATAL(PREFIX, "Program failed to compile with: %s\nDiagnostics:\n%s\n===Program source===:\n%s\n", //
                                 compilerMessage.c_str(), buildLog.c_str(), image.c_str());
              }
              size_t binaryLen = 0;
              if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binaryLen), &binaryLen, nullptr) == CL_SUCCESS && binaryLen) {
                std::vector<unsigned char> binary(binaryLen);
                unsigned char *binaries[] = {binary.data()};
                if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binaries), binaries, nullptr) == CL_SUCCESS)
                  cache::write(cachePath, binary.data(), binary.size());
              }
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
  if (queryDeviceInfo(*device, CL_DEVICE_EXTENSIONS) ^ aspartame::contains_slice("cl_amd_device_attribute_query")) {
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
PagingMode ClDevice::pagingMode() {
  POLYINVOKE_TRACE();
  // our OpenCL path binds capture pointers as clSVMAlloc buffers, not system pointers, so it tops out
  // at Managed even on a fine-grain-system device; nullopt svm => no shared memory at all
  return svm ? PagingMode::Managed : PagingMode::None;
}
bool ClDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
size_t ClDevice::maxThreadsPerBlock() {
  POLYINVOKE_TRACE();
  size_t v = 0;
  return clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(v), &v, nullptr) == CL_SUCCESS && v ? v : 1024;
}
size_t ClDevice::localMemoryBytes() {
  cl_ulong value = 0;
  CHECKED(clGetDeviceInfo(*device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(value), &value, nullptr));
  return static_cast<size_t>(value);
}
size_t ClDevice::globalMemoryBytes() {
  cl_ulong value = 0;
  CHECKED(clGetDeviceInfo(*device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(value), &value, nullptr));
  return static_cast<size_t>(value);
}
size_t ClDevice::computeUnits() {
  cl_uint value = 0;
  CHECKED(clGetDeviceInfo(*device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, nullptr));
  return static_cast<size_t>(value);
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
  const auto hasExt = [&](std::string_view e) { return exts ^ aspartame::contains_slice(e); };
  if (hasExt("cl_khr_fp64")) out.emplace_back("fp64");
  if (hasExt("cl_khr_fp16")) out.emplace_back("fp16");
  // XXX int64 is behind cles_khr_int64 for embedded profiles
  if (queryDeviceInfo(*device, CL_DEVICE_PROFILE) == "FULL_PROFILE" || hasExt("cles_khr_int64")) out.emplace_back("int64");
  out.emplace_back(fmt::format("paging:{}", magic_enum::enum_name(pagingMode())));
  if (quirks.overReadPad) out.emplace_back(fmt::format("{}:{}", OverReadPadFeature, quirks.overReadPad));
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
void ClDevice::trackSvm(void *p, size_t size) {
  if (svmTracker) svmTracker->track(p, size);
}
void ClDevice::untrackSvm(void *p) {
  if (svmTracker) svmTracker->untrack(p);
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
  if (const size_t slack = quirks.overReadPad; slack > 0) {
    std::vector<char> zeros(size + slack, 0);
    return memoryObjects.malloc(size + slack,
                                OUT_CHECKED(clCreateBuffer(*context, flags | CL_MEM_COPY_HOST_PTR, size + slack, zeros.data(), OUT_ERR)));
  }
  return memoryObjects.malloc(size, OUT_CHECKED(clCreateBuffer(*context, flags, size, nullptr, OUT_ERR)));
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
    if (mem->offset != 0) POLYINVOKE_FATAL(PREFIX, "Illegal free of %" PRIuPTR ", %zu bytes into its allocation", ptr, mem->offset);
    CHECKED(clReleaseMemObject(mem->value));
    memoryObjects.erase(ptr);
  } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %" PRIuPTR, ptr);
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
  cl_uint alignBits = 0;
  CHECKED(clGetDeviceInfo(*device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(alignBits), &alignBits, nullptr));
  return std::make_unique<ClDeviceQueue>(
      timeout, store, OUT_CHECKED(clCreateCommandQueue(*context, *device, 0, OUT_ERR)),
      [this](auto &&ptr) -> detail::MemoryObjects<cl_mem>::Resolved {
        if (auto mem = memoryObjects.query(ptr); mem) {
          return *mem;
        } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %" PRIuPTR, ptr);
      },
      format, alignBits / 8, deviceName, svm, svmTracker);
}
ClDevice::~ClDevice() { POLYINVOKE_TRACE(); }

// ---

ClDeviceQueue::ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                             decltype(queryMemObject) queryMemObject, ModuleFormat format, size_t memBaseAddrAlign, std::string deviceName,
                             std::optional<cl_bitfield> svm, std::shared_ptr<cl_details::SVMTracker> svmTracker)
    : latch(timeout), store(store), queue(queue), queryMemObject(std::move(queryMemObject)), format(format),
      memBaseAddrAlign(memBaseAddrAlign), deviceName(std::move(deviceName)), svm(svm), svmTracker(std::move(svmTracker)) {
  POLYINVOKE_TRACE();
  CHECKED(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr));
}
void *ClDeviceQueue::ensureNullArgStub() {
  if (!nullArgStub && svm) {
    nullArgStub = clSVMAlloc(context, /*CL_MEM_READ_WRITE*/ 1 << 0 | *svm, 256, 0);
    if (!nullArgStub) POLYINVOKE_FATAL(PREFIX, "clSVMAlloc failed for the null-argument stub (%zu bytes)", size_t(256));
  }
  return nullArgStub;
}
cl_mem ClDeviceQueue::ensureNullArgBuffer() {
  if (!nullArgBuffer) nullArgBuffer = OUT_CHECKED(clCreateBuffer(context, CL_MEM_READ_WRITE, 256, nullptr, OUT_ERR));
  return nullArgBuffer;
}
ClDeviceQueue::~ClDeviceQueue() {
  POLYINVOKE_TRACE();
  // Internal argument storage must outlive every command that can reference it. clFinish also makes
  // teardown safe when the caller drops a queue immediately after an asynchronous launch.
  CHECKED(clFinish(queue));
  (void)latch.waitAll();
  if (nullArgStub) clSVMFree(context, nullArgStub);
  if (nullArgBuffer) CHECKED(clReleaseMemObject(nullArgBuffer));
  CHECKED(clReleaseCommandQueue(queue));
}
bool ClDeviceQueue::mapSvmForHost(void *ptr) {
  if (!svmTracker || !clEnqueueSVMMap || *svm != 0) return false;
  const auto result = svmTracker->mapForHost(
      ptr, [&](void *base, size_t size) { return clEnqueueSVMMap(queue, CL_TRUE, /*CL_MAP_READ*/ 0x1, base, size, 0, nullptr, nullptr); });
  if (!result) return false;
  CHECKED(*result);
  return true;
}
void ClDeviceQueue::unmapAllSvmForDevice() {
  if (!svmTracker || !clEnqueueSVMUnmap || *svm != 0) return;
  CHECKED(svmTracker->unmapAllForDevice([&](void *ptr) { return clEnqueueSVMUnmap(queue, ptr, 0, nullptr, nullptr); }));
}
void ClDeviceQueue::mapAllSvmForHost() {
  if (!svmTracker || !clEnqueueSVMMap || *svm != 0) return;
  CHECKED(svmTracker->mapAllForHost([&](void *ptr, size_t size) {
    return clEnqueueSVMMap(queue, CL_TRUE, /*CL_MAP_READ | CL_MAP_WRITE*/ 0x3, ptr, size, 0, nullptr, nullptr);
  }));
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
    auto *srcP = reinterpret_cast<char *>(src) + srcOffset;
    auto *dstP = reinterpret_cast<char *>(dst) + dstOffset;
    if (*svm == 0 && clEnqueueSVMMap && clEnqueueSVMUnmap) {
      std::vector<char> staging(size);
      if (mapSvmForHost(srcP)) std::memcpy(staging.data(), srcP, size);
      else {
        unmapAllSvmForDevice();
        CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, staging.data(), srcP, size, 0, nullptr, nullptr));
      }
      unmapAllSvmForDevice();
      CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dstP, staging.data(), size, 0, nullptr, nullptr));
    } else {
      unmapAllSvmForDevice();
      CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dstP, srcP, size, 0, nullptr, nullptr));
    }
  } else {
    const auto srcMem = detail::MemoryObjects<cl_mem>::subrange(queryMemObject(src), srcOffset, size);
    const auto dstMem = detail::MemoryObjects<cl_mem>::subrange(queryMemObject(dst), dstOffset, size);
    if (!srcMem) POLYINVOKE_FATAL(PREFIX, "Source range exceeds memory object: %" PRIuPTR "+%zu (%zu bytes)", src, srcOffset, size);
    if (!dstMem) POLYINVOKE_FATAL(PREFIX, "Destination range exceeds memory object: %" PRIuPTR "+%zu (%zu bytes)", dst, dstOffset, size);
    CHECKED(clEnqueueCopyBuffer(queue, srcMem->value, dstMem->value, srcMem->offset, dstMem->offset, size, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!src) POLYINVOKE_FATAL(PREFIX, "Source pointer is NULL, destination=%" PRIuPTR, dst);
  if (size == 0) return enqueueCallback(cb, {});
  if (svm) {
    unmapAllSvmForDevice();
    auto *dstP = reinterpret_cast<char *>(dst) + dstOffset;
    CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dstP, src, size, 0, nullptr, nullptr));
  } else {
    const auto mem = detail::MemoryObjects<cl_mem>::subrange(queryMemObject(dst), dstOffset, size);
    if (!mem) POLYINVOKE_FATAL(PREFIX, "Destination range exceeds memory object: %" PRIuPTR "+%zu (%zu bytes)", dst, dstOffset, size);
    CHECKED(clEnqueueWriteBuffer(queue, mem->value, CL_FALSE, mem->offset, size, src, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!dst) POLYINVOKE_FATAL(PREFIX, "Destination pointer is NULL, source=%" PRIuPTR, src);
  // XXX zero-byte is a no-op; ReadBuffer/SVMMemcpy reject size 0 with CL_INVALID_VALUE (an -O3 reflect can size a result to 0)
  if (size == 0) return enqueueCallback(cb, {});
  if (svm) {
    auto *srcP = reinterpret_cast<char *>(src) + srcOffset;
    if (mapSvmForHost(srcP)) {
      std::memcpy(dst, srcP, size);
    } else {
      unmapAllSvmForDevice();
      CHECKED(clEnqueueSVMMemcpy(queue, CL_TRUE, dst, srcP, size, 0, nullptr, nullptr));
    }
  } else {
    const auto mem = detail::MemoryObjects<cl_mem>::subrange(queryMemObject(src), srcOffset, size);
    if (!mem) POLYINVOKE_FATAL(PREFIX, "Source range exceeds memory object: %" PRIuPTR "+%zu (%zu bytes)", src, srcOffset, size);
    CHECKED(clEnqueueReadBuffer(queue, mem->value, CL_FALSE, mem->offset, size, dst, 0, nullptr, &event));
  }
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                       std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  const bool trace = std::getenv(polyregion::env::PolyinvokeTrace) != nullptr;
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
  std::vector<cl_mem> subBuffers;
  const cl_uint logicalArgCount = static_cast<cl_uint>(types.size() - 1);
  const cl_uint pointerArgCount =
      static_cast<cl_uint>(types | aspartame::take(logicalArgCount) | aspartame::count([](const Type type) { return type == Type::Ptr; }));
  cl_uint physicalArgCount = logicalArgCount;
  bool expandedPointerAbi = false;
  if (format == ModuleFormat::Source) {
    CHECKED(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(physicalArgCount), &physicalArgCount, nullptr));
    const cl_uint expandedArgCount = logicalArgCount + pointerArgCount;
    if (physicalArgCount == expandedArgCount) expandedPointerAbi = true;
    else if (physicalArgCount != logicalArgCount)
      POLYINVOKE_FATAL(PREFIX, "OpenCL source kernel `%s` has %u physical args; expected %u legacy args or %u owner+offset ABI args",
                       symbol.c_str(), physicalArgCount, logicalArgCount, expandedArgCount);
  }
  if (trace)
    fmt::print(stderr,
               "[OpenCL launch] {} device={} groups={}x{}x{} local={}x{}x{} shared={} logical_args={} physical_args={} expanded={}\n",
               symbol, deviceName, policy.global.x, policy.global.y, policy.global.z, local.x, local.y, local.z, sharedMem, logicalArgCount,
               physicalArgCount, expandedPointerAbi);

  cl_uint physicalIdx = 0;
  for (cl_uint logicalIdx = 0; logicalIdx < logicalArgCount; ++logicalIdx) {
    const auto rawPtr = args[logicalIdx];
    switch (const auto tpe = types[logicalIdx]) {
      case Type::Ptr: {
        static_assert(byteOfType(Type::Ptr) == sizeof(uintptr_t));
        uintptr_t ptr = {};
        std::memcpy(&ptr, rawPtr, byteOfType(Type::Ptr));
        if (svm) {
          void *value = reinterpret_cast<void *>(ptr);
          if (!value && expandedPointerAbi) value = ensureNullArgStub();
          CHECKED(clSetKernelArgSVMPointer(kernel, physicalIdx++, value));
          if (expandedPointerAbi) {
            // Some OpenCL implementations reject a null physical pointer argument. Expanded source
            // kernels reconstruct logical null from this reserved offset and never observe the stub.
            const cl_ulong byteOffset = ptr ? 0 : NullPointerOffset;
            CHECKED(clSetKernelArg(kernel, physicalIdx++, sizeof(byteOffset), &byteOffset));
          }
        } else {
          cl_mem mem = {};
          size_t offset = 0, remaining = 0;
          if (!ptr && expandedPointerAbi) mem = ensureNullArgBuffer();
          else {
            if (ptr) {
              const auto resolved = queryMemObject(ptr);
              mem = resolved.value;
              offset = resolved.offset;
              remaining = resolved.remaining;
            }
          }
          if (trace)
            fmt::print(stderr, "  ptr[{}] logical=0x{:x} owner={} offset={} remaining={} physical={}{}\n", logicalIdx, ptr,
                       static_cast<const void *>(mem), offset, remaining, physicalIdx, expandedPointerAbi ? "+offset" : "");
          if (!expandedPointerAbi && offset != 0) {
            if (remaining == 0) POLYINVOKE_FATAL(PREFIX, "Interior pointer %" PRIuPTR " is at the end of its allocation", ptr);
            if (memBaseAddrAlign != 0 && offset % memBaseAddrAlign != 0)
              POLYINVOKE_FATAL(PREFIX, "Interior pointer %" PRIuPTR " is %zu bytes into its allocation, not aligned to %zu bytes on %s",
                               ptr, offset, memBaseAddrAlign, deviceName.c_str());
            cl_buffer_region region{offset, remaining};
            mem = OUT_CHECKED(clCreateSubBuffer(mem, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, OUT_ERR));
            subBuffers.push_back(mem);
          }
          CHECKED(clSetKernelArg(kernel, physicalIdx++, toSize(tpe), &mem));
          if (expandedPointerAbi) {
            const cl_ulong byteOffset = ptr ? static_cast<cl_ulong>(offset) : NullPointerOffset;
            CHECKED(clSetKernelArg(kernel, physicalIdx++, sizeof(byteOffset), &byteOffset));
          }
        }
      } break;
      case Type::Scratch: {
        CHECKED(clSetKernelArg(kernel, physicalIdx++, sharedMem, nullptr));
        break;
      }
      default: {
        CHECKED(clSetKernelArg(kernel, physicalIdx++, toSize(tpe), rawPtr));
        break;
      }
    }
  }
  if (physicalIdx != physicalArgCount)
    POLYINVOKE_FATAL(PREFIX, "OpenCL kernel `%s` bound %u physical args but its ABI reports %u", symbol.c_str(), physicalIdx,
                     physicalArgCount);
  if (svm) {
    // indirect SVM allocs need CL_KERNEL_EXEC_INFO_SVM_PTRS or the driver skips coherency; some drivers
    // reject the batched call but accept it per-pointer, so on CL_INVALID_VALUE retry per-pointer
    std::vector<void *> allSvmPtrs;
    const auto tracked = svmTracker ? svmTracker->pointers() : std::vector<void *>{};
    allSvmPtrs.reserve(types.size() + tracked.size());
    for (cl_uint i = 0; i < types.size() - 1; ++i) {
      if (types[i] != Type::Ptr) continue;
      uintptr_t ptr = {};
      std::memcpy(&ptr, args[i], byteOfType(Type::Ptr));
      // NVIDIA OpenCL rejects null pointer args and null SVM declarations. Expanded source kernels recover
      // logical null from NullPointerOffset, so declaring the physical stub does not alter program semantics.
      if (ptr) allSvmPtrs.push_back(reinterpret_cast<void *>(ptr));
      else if (expandedPointerAbi)
        if (void *stub = ensureNullArgStub()) allSvmPtrs.push_back(stub);
    }
    allSvmPtrs.insert(allSvmPtrs.end(), tracked.begin(), tracked.end());
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
  const auto enqueue = [&](const cl_details::LaunchDimensions &dimensions) {
    return clEnqueueNDRangeKernel(queue, kernel,                    //
                                  3,                                //
                                  nullptr,                          //
                                  dimensions.global.sizes().data(), //
                                  dimensions.local.sizes().data(),  //
                                  0, nullptr, &event);
  };
  auto dimensions = cl_details::launchDimensions(policy.global, local);
  cl_int result = enqueue(dimensions);
  if (result == CL_INVALID_WORK_GROUP_SIZE) {
    cl_device_id device = {};
    size_t kernelMax = 0;
    if (clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr) == CL_SUCCESS
        && clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMax), &kernelMax, nullptr) == CL_SUCCESS) {
      if (const auto retry = cl_details::retryLaunchDimensions(result, policy.global, local, kernelMax)) result = enqueue(*retry);
    }
  }
  CHECKED(result);
  for (const auto subBuffer : subBuffers)
    CHECKED(clReleaseMemObject(subBuffer));
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

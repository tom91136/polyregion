#include <cstring>
#include <iostream>
#include <thread>

#include "polyinvoke/cl_platform.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::cl;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *PREFIX = "OpenCL";

#define OUT_ERR __err
#define OUT_CHECKED(f) checked([&](auto &&OUT_ERR) { return (f); }, __FILE__, __LINE__)

static void checked(cl_int result, const char *file, int line) {
  if (result != CL_SUCCESS) POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", file, line, clewErrorString(result));
}

template <typename F> static auto checked(F &&f, const char *file, int line) {
  cl_int result = CL_SUCCESS;
  auto y = f(&result);
  if (result == CL_SUCCESS) return y;
  else POLYINVOKE_FATAL(PREFIX, "%s:%d: %s", file, line, clewErrorString(result));
}

static std::string queryDeviceInfo(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  CHECKED(clGetDeviceInfo(device, info, 0, nullptr, &size));
  std::string data(size - 1, '\0'); // -1 as clGetDeviceInfo returns the length+1 for \0
  CHECKED(clGetDeviceInfo(device, info, size, data.data(), nullptr));
  return data;
}
std::variant<std::string, std::unique_ptr<Platform>> ClPlatform::create() {
  switch (auto result = clewInit(); result) {
    case CLEW_SUCCESS: break;
    case CLEW_ERROR_OPEN_FAILED: return "CLEW: failed to open the dynamic library";
    case CLEW_ERROR_ATEXIT_FAILED: return "CLEW: cannot queue atexit for closing the dynamic library";
    default: return "Unknown error: " + std::to_string(result);
  }
  return std::unique_ptr<Platform>(new ClPlatform());
}
ClPlatform::ClPlatform() {
  POLYINVOKE_TRACE();
  // XXX FP64 is emulated on Intel Arc and needs to be enabled via environment variable
  // we set it unless it's already defined with some other value
  setenv("OverrideDefaultFP64Settings", "1", false);
}
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
ModuleFormat ClPlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::Source;
}
std::vector<std::unique_ptr<Device>> ClPlatform::enumerate() {
  POLYINVOKE_TRACE();
  cl_uint numPlatforms = 0;
  CHECKED(clGetPlatformIDs(0, nullptr, &numPlatforms));
  std::vector<cl_platform_id> platforms(numPlatforms);
  CHECKED(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));
  std::vector<std::unique_ptr<Device>> clDevices;
  for (const auto &platform : platforms) {
    cl_uint numDevices = 0;
    if (const auto deviceIdResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        deviceIdResult == CL_DEVICE_NOT_FOUND) {
      continue; // no device
    } else CHECKED(deviceIdResult);

    std::vector<cl_device_id> devices(numDevices);
    CHECKED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr));
    for (auto &device : devices)
      clDevices.push_back(std::make_unique<ClDevice>(device));
  }
  return clDevices;
}
ClPlatform::~ClPlatform() { POLYINVOKE_TRACE(); }

// ---

ClDevice::ClDevice(cl_device_id device)
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
      deviceName(queryDeviceInfo(device, CL_DEVICE_NAME)),
      store(
          PREFIX,
          [this](auto &&image) {
            POLYINVOKE_TRACE();
            auto imageData = image.data();
            auto imageLen = image.size();
            auto program = OUT_CHECKED(clCreateProgramWithSource(*context, 1, &imageData, &imageLen, OUT_ERR));
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
  return {};
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

  if (auto mem = memoryObjects.query(ptr); mem) {
    CHECKED(clReleaseMemObject(*mem));
    memoryObjects.erase(ptr);
  } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %ld", ptr);
}
std::optional<void *> ClDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  context.touch();
  return std::nullopt;
}
void ClDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  context.touch();
  POLYINVOKE_FATAL(PREFIX, "Unsupported: %p", ptr);
}
std::unique_ptr<DeviceQueue> ClDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<ClDeviceQueue>(timeout, store, OUT_CHECKED(clCreateCommandQueue(*context, *device, 0, OUT_ERR)),
                                         [this](auto &&ptr) {
                                           if (auto mem = memoryObjects.query(ptr); mem) {
                                             return *mem;
                                           } else POLYINVOKE_FATAL(PREFIX, "Illegal memory object: %ld", ptr);
                                         });
}
ClDevice::~ClDevice() { POLYINVOKE_TRACE(); }

// ---

ClDeviceQueue::ClDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(store) store, decltype(queue) queue,
                             decltype(queryMemObject) queryMemObject)
    : latch(timeout), store(store), queue(queue), queryMemObject(std::move(queryMemObject)) {
  POLYINVOKE_TRACE();
}
ClDeviceQueue::~ClDeviceQueue() {
  POLYINVOKE_TRACE();
  CHECKED(clReleaseCommandQueue(queue));
}
void ClDeviceQueue::enqueueCallback(const MaybeCallback &cb, cl_event event) {
  POLYINVOKE_TRACE();
  if (!cb) return;
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
void ClDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!src) POLYINVOKE_FATAL(PREFIX, "Source pointer is NULL, destination=%lu", dst);
  CHECKED(clEnqueueWriteBuffer(queue, queryMemObject(dst), CL_FALSE, dstOffset, size, src, 0, nullptr, &event));
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  cl_event event = {};
  if (!dst) POLYINVOKE_FATAL(PREFIX, "Destination pointer is NULL, source=%lu", src);
  CHECKED(clEnqueueReadBuffer(queue, queryMemObject(src), CL_FALSE, srcOffset, size, dst, 0, nullptr, &event));
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                       std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  if (types.back() != Type::Void) POLYINVOKE_FATAL(PREFIX, "Non-void return type not supported, was %s", to_string(types.back()).data());
  auto kernel = store.resolveFunction(moduleName, symbol, types);
  auto toSize = [](Type t) -> size_t {
    switch (t) {
      case Type::Ptr: return sizeof(cl_mem);
      case Type::Void: POLYINVOKE_FATAL(PREFIX, "Illegal argument type: %s", to_string(t).data());
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
        cl_mem mem = queryMemObject(ptr);
        CHECKED(clSetKernelArg(kernel, i, toSize(tpe), &mem));
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

  POLYINVOKE_TRACE();
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
}

#undef CHECKED
#undef OUT_CHECKED

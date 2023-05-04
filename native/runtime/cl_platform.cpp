
#include "cl_platform.h"
#include <chrono>
#include <thread>
using namespace polyregion::runtime;
using namespace polyregion::runtime::cl;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[OpenCL error] ";

#define OUT_ERR __err
#define OUT_CHECKED(f) checked([&](auto &&OUT_ERR) { return (f); }, __FILE__, __LINE__)

static void checked(cl_int result, const char *file, int line) {
  if (result != CL_SUCCESS) {
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           clewErrorString(result));
  }
}

template <typename F> static auto checked(F &&f, const char *file, int line) {
  cl_int result = CL_SUCCESS;
  auto y = f(&result);
  if (result == CL_SUCCESS) return y;
  else {
    throw std::logic_error(std::string(ERROR_PREFIX) + file + ":" + std::to_string(line) + ": " +
                           clewErrorString(result));
  }
}

static std::string queryDeviceInfo(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  CHECKED(clGetDeviceInfo(device, info, 0, nullptr, &size));
  std::string data(size - 1, '\0'); // -1 as clGetDeviceInfo returns the length+1 for \0
  CHECKED(clGetDeviceInfo(device, info, size, data.data(), nullptr));
  return data;
}

ClPlatform::ClPlatform() {
  TRACE();
  if (clewInit() != CLEW_SUCCESS) {
    throw std::logic_error("CLEW initialisation failed, no OpenCL library present?");
  }
}
std::string ClPlatform::name() {
  TRACE();
  return "OpenCL";
}
std::vector<Property> ClPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> ClPlatform::enumerate() {
  TRACE();
  cl_uint numPlatforms = 0;
  CHECKED(clGetPlatformIDs(0, nullptr, &numPlatforms));
  std::vector<cl_platform_id> platforms(numPlatforms);
  CHECKED(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));
  std::vector<std::unique_ptr<Device>> clDevices;
  for (auto &platform : platforms) {
    cl_uint numDevices = 0;
    auto deviceIdResult = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (deviceIdResult == CL_DEVICE_NOT_FOUND) {
      continue; // no device
    } else
      CHECKED(deviceIdResult);

    std::vector<cl_device_id> devices(numDevices);
    CHECKED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr));
    for (auto &device : devices)
      clDevices.push_back(std::make_unique<ClDevice>(device));
  }
  return clDevices;
}
ClPlatform::~ClPlatform() { TRACE(); }

// ---

ClDevice::ClDevice(cl_device_id device)
    : device(
          [&, device]() {
            TRACE();
            // XXX clReleaseDevice appears to crash various CL implementation regardless of version, skip retain as well
            //            if (__clewRetainDevice && __clewReleaseDevice) { // clRetainDevice requires OpenCL >= 1.2
            //              CHECKED(__clewRetainDevice(device));
            //            }
            return device;
          },
          [&](auto &&device) {
            TRACE();
            // XXX see above
            //            if (__clewRetainDevice && __clewReleaseDevice) // clReleaseDevice requires OpenCL >= 1.2
            //              CHECKED(__clewReleaseDevice(device));
          }),
      context(
          [this]() {
            TRACE();
            return OUT_CHECKED(clCreateContext(nullptr, 1, &(*this->device), nullptr, nullptr, OUT_ERR));
          },
          [&](auto &&c) {
            TRACE();
            CHECKED(clReleaseContext(c));
          }),
      deviceName(queryDeviceInfo(device, CL_DEVICE_NAME)),
      store(
          ERROR_PREFIX,
          [this](auto &&image) {
            TRACE();
            auto imageData = image.data();
            auto imageLen = image.size();
            auto program = OUT_CHECKED(clCreateProgramWithSource(*context, 1, &imageData, &imageLen, OUT_ERR));
            const std::string compilerArgs = "-Werror";
            cl_int result = clBuildProgram(program, 1, &(*this->device), compilerArgs.data(), nullptr, nullptr);
            if (result != CL_SUCCESS) {
              size_t len;
              CHECKED(clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len));
              std::string buildLog(len, '\0');
              CHECKED(
                  clGetProgramBuildInfo(program, *this->device, CL_PROGRAM_BUILD_LOG, len, buildLog.data(), nullptr));
              auto compilerMessage = std::string(clewErrorString(result));
              const static auto P = std::string(ERROR_PREFIX);
              throw std::logic_error(std::string("Program failed to compile with ") + compilerMessage +
                                     std::string(")\n") +                                                   //
                                     std::string("Diagnostics:\n") + buildLog + "\n" +                      //
                                     std::string("Program source:=====\n") + std::string(image) + "\n=====" //
              );                                                                                            //
            }
            TRACE();
            return program;
          },
          [this](auto &&m, auto &&name) {
            TRACE();
            context.touch();
            TRACE();
            return OUT_CHECKED(clCreateKernel(m, name.c_str(), OUT_ERR));
          },
          [&](auto &&m) {
            TRACE();
            CHECKED(clReleaseProgram(m));
          },
          [&](auto &&f) {
            TRACE();
            CHECKED(clReleaseKernel(f));
          }),
      bufferCounter(), allocations() {
  TRACE();
}
cl_mem ClDevice::queryMemObject(uintptr_t ptr) {
  std::shared_lock readLock(mutex);
  if (auto it = allocations.find(ptr); it != allocations.end()) return it->second;
  else
    throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
}
int64_t ClDevice::id() {
  TRACE();
  return reinterpret_cast<int64_t>(*device);
}
std::string ClDevice::name() {
  TRACE();
  return deviceName;
}
bool ClDevice::sharedAddressSpace() {
  TRACE();
  return false;
}
std::vector<Property> ClDevice::properties() {
  TRACE();
  return {
      {"CL_DEVICE_PROFILE", queryDeviceInfo(*device, CL_DEVICE_PROFILE)},
      {"CL_DRIVER_VERSION", queryDeviceInfo(*device, CL_DRIVER_VERSION)},
      {"CL_DEVICE_VENDOR", queryDeviceInfo(*device, CL_DEVICE_VENDOR)},
      {"CL_DEVICE_VERSION", queryDeviceInfo(*device, CL_DEVICE_VERSION)},
      {"CL_DEVICE_EXTENSIONS", queryDeviceInfo(*device, CL_DEVICE_EXTENSIONS)},
  };
}
std::vector<std::string> ClDevice::features() {
  TRACE();
  return {};
}
void ClDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  store.loadModule(name, image);
}
bool ClDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return store.moduleLoaded(name);
}
uintptr_t ClDevice::malloc(size_t size, Access access) {
  TRACE();
  context.touch();
  cl_mem_flags flags = {};
  switch (access) {
    case Access::RO: flags = CL_MEM_READ_ONLY; break;
    case Access::WO: flags = CL_MEM_WRITE_ONLY; break;
    case Access::RW:
    default: flags = CL_MEM_READ_WRITE; break;
  }
  std::unique_lock writeLock(mutex);
  while (true) {
    auto id = this->bufferCounter++;
    if (auto it = allocations.find(id); it != allocations.end()) continue;
    else {
      allocations.emplace_hint(it, id, OUT_CHECKED(clCreateBuffer(*context, flags, size, nullptr, OUT_ERR)));
      return id;
    }
  }
}
void ClDevice::free(uintptr_t ptr) {
  TRACE();
  context.touch();
  CHECKED(clReleaseMemObject(queryMemObject(ptr)));
  std::unique_lock writeLock(mutex);
  allocations.erase(ptr);
}
std::unique_ptr<DeviceQueue> ClDevice::createQueue() {
  TRACE();
  return std::make_unique<ClDeviceQueue>(store, OUT_CHECKED(clCreateCommandQueue(*context, *device, 0, OUT_ERR)),
                                         [this](auto &&ptr) { return queryMemObject(ptr); });
}
ClDevice::~ClDevice() { TRACE(); }

// ---

ClDeviceQueue::ClDeviceQueue(decltype(store) store, decltype(queue) queue, decltype(queryMemObject) queryMemObject)
    : store(store), queue(queue), queryMemObject(std::move(queryMemObject)) {
  TRACE();
}
ClDeviceQueue::~ClDeviceQueue() {
  TRACE();
  CHECKED(clReleaseCommandQueue(queue));
}
void ClDeviceQueue::enqueueCallback(const MaybeCallback &cb, cl_event event) {
  TRACE();
  if (!cb) return;
  CHECKED(clSetEventCallback(
      event, CL_COMPLETE,
      [](cl_event e, cl_int status, void *data) {
        CHECKED(clReleaseEvent(e));
        CHECKED(status);
        detail::CountedCallbackHandler::consume(data);
      },
      detail::CountedCallbackHandler::createHandle([cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      })));
  CHECKED(clFlush(queue));
}
void ClDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  cl_event event = {};
  if (!src) throw std::logic_error("Source pointer is NULL");
  CHECKED(clEnqueueWriteBuffer(queue, queryMemObject(dst), CL_FALSE, 0, size, src, 0, nullptr, &event));
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  cl_event event = {};
  if (!dst) throw std::logic_error("Destination pointer is NULL");
  CHECKED(clEnqueueReadBuffer(queue, queryMemObject(src), CL_FALSE, 0, size, dst, 0, nullptr, &event));
  enqueueCallback(cb, event);
}
void ClDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                       std::vector<Type> types, std::vector<std::byte> argData, const Policy &policy,
                                       const MaybeCallback &cb) {
  TRACE();
  if (types.back() != Type::Void)
    throw std::logic_error(std::string(ERROR_PREFIX) + "Non-void return type not supported, was " +
                           runtime::typeName(types.back()));
  auto kernel = store.resolveFunction(moduleName, symbol);
  auto toSize = [](Type t) -> size_t {
    switch (t) {
      case Type::Ptr: return sizeof(cl_mem);
      case Type::Void: throw std::logic_error("Illegal argument type: void");
      default: return ::byteOfType(t);
    }
  };

  auto args = detail::argDataAsPointers(types, argData);
  auto [local, sharedMem] = policy.local.value_or(std::pair{Dim3{}, 0});
  auto global = Dim3{policy.global.x * local.x, policy.global.y * local.y, policy.global.z * local.z};

  // last arg is the return, void assertion should have been done before this
  for (cl_uint i = 0; i < types.size() - 1; ++i) {
    auto rawPtr = args[i];
    auto tpe = types[i];
    switch (tpe) {
      case Type::Ptr: {
        cl_mem mem = queryMemObject(*static_cast<uintptr_t *>(rawPtr));
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

  TRACE();
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

#undef CHECKED
#undef OUT_CHECKED

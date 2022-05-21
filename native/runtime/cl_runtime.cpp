
#include "cl_runtime.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::cl;

#define CHECKED(f) checked((f), __FILE__, __LINE__)

static constexpr const char *ERROR_PREFIX = "[OpenCL error] ";

#define OUT_ERR __err
#define OUT_CHECKED(f) checked([&](auto &&OUT_ERR) { return (f); }, __FILE__, __LINE__)

static void checked(cl_int result, const std::string &file, int line) {
  if (result != CL_SUCCESS) {
    throw std::logic_error(std::string(ERROR_PREFIX + file + ":" + std::to_string(line) + ": ") +
                           clewErrorString(result));
  }
}

template <typename F> static auto checked(F &&f, const std::string &file, int line) {
  cl_int err = CL_SUCCESS;
  auto y = f(&err);
  if (err == CL_SUCCESS) return y;
  else {
    throw std::logic_error(std::string(ERROR_PREFIX + file + ":" + std::to_string(line) + ": ") + clewErrorString(err));
  }
}

static std::string queryDeviceInfo(cl_device_id device, cl_device_info info) {
  size_t size = 0;
  CHECKED(clGetDeviceInfo(device, info, 0, nullptr, &size));
  std::string data(size - 1, '\0'); // -1 as clGetDeviceInfo returns the length+1 for \0
  CHECKED(clGetDeviceInfo(device, info, size, data.data(), nullptr));
  return data;
}

ClRuntime::ClRuntime() {
  if (clewInit() != CLEW_SUCCESS) {
    throw std::logic_error("CLEW initialisation failed, no OpenCL library present?");
  }
}

std::string ClRuntime::name() { return "OpenCL"; }

std::vector<Property> ClRuntime::properties() { return {}; }

std::vector<std::unique_ptr<Device>> ClRuntime::enumerate() {
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

void ClDevice::enqueueCallback(const std::optional<Callback> &cb, cl_event event) {
  if (!cb) return;
  CHECKED(clSetEventCallback(
      event, CL_COMPLETE,
      [](cl_event e, cl_int status, void *data) {
        CHECKED(status);
        detail::CountedCallbackHandler::consume(data);
      },
      detail::CountedCallbackHandler::createHandle(*cb)));
  CHECKED(clFlush(queue));
}

cl_mem ClDevice::queryMemObject(uintptr_t ptr) const {
  if (auto it = allocations.find(ptr); it != allocations.end()) return it->second;
  else
    throw std::logic_error(std::string(ERROR_PREFIX) + "Illegal memory object: " + std::to_string(ptr));
}

ClDevice::ClDevice(cl_device_id device) : device(device) {
  if (__clewRetainDevice) // clRetainDevice requires OpenCL >= 1.2
    CHECKED(__clewRetainDevice(device));
  context = OUT_CHECKED(clCreateContext(nullptr, 1, &device, nullptr, nullptr, OUT_ERR));
  queue = OUT_CHECKED(clCreateCommandQueue(context, device, 0, OUT_ERR));
  deviceName = queryDeviceInfo(device, CL_DEVICE_NAME);
}

ClDevice::~ClDevice() {
  CHECKED(clFinish(queue));
  for (auto &[_, lm] : modules) {
    auto &[p, fnTable] = lm;
    for (auto [name, fn] : fnTable)
      CHECKED(clReleaseKernel(fn));
    CHECKED(clReleaseProgram(p));
  }
  modules.clear();
  CHECKED(clReleaseCommandQueue(queue));
  CHECKED(clReleaseContext(context));
  if (__clewReleaseDevice) // clReleaseDevice requires OpenCL >= 1.2
    CHECKED(__clewReleaseDevice(device));
}

int64_t ClDevice::id() { return reinterpret_cast<int64_t>(device); }

std::string ClDevice::name() { return deviceName; }

std::vector<Property> ClDevice::properties() { return {}; }

void ClDevice::loadModule(const std::string &name, const std::string &image) {
  if (auto it = modules.find(name); it != modules.end()) {
    throw std::logic_error(std::string(ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    auto imageData = image.data();
    auto imageLen = image.size();
    auto program = OUT_CHECKED(clCreateProgramWithSource(context, 1, &imageData, &imageLen, OUT_ERR));
    const std::string compilerArgs = "-Werror";
    cl_int result = clBuildProgram(program, 1, &device, compilerArgs.data(), nullptr, nullptr);
    if (result != CL_SUCCESS) {
      size_t len;
      CHECKED(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len));
      std::string buildLog(len, '\0');
      CHECKED(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buildLog.data(), nullptr));
      auto compilerMessage = std::string(clewErrorString(result));
      throw std::logic_error(std::string(ERROR_PREFIX) + "Program failed to compile with " + compilerMessage +
                             ")\n" +                                                          //
                             std::string(ERROR_PREFIX) + "Program source:\n" + image + "\n" + //
                             std::string(ERROR_PREFIX) + "Diagnostics:\n" + buildLog);        //
    }
    modules.emplace_hint(it, name, LoadedModule{program, {}});
  }
}

uintptr_t ClDevice::malloc(size_t size, Access access) {
  cl_mem_flags flags = {};
  switch (access) {
    case Access::RO: flags = CL_MEM_READ_ONLY; break;
    case Access::WO: flags = CL_MEM_WRITE_ONLY; break;
    case Access::RW:
    default: flags = CL_MEM_READ_WRITE; break;
  }
  auto id = this->bufferCounter++;
  allocations[id] = OUT_CHECKED(clCreateBuffer(context, flags, size, nullptr, OUT_ERR));
  return id;
}

void ClDevice::free(uintptr_t ptr) { CHECKED(clReleaseMemObject(queryMemObject(ptr))); }

void ClDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                        const std::optional<Callback> &cb) {
  cl_event event = {};
  CHECKED(clEnqueueWriteBuffer(queue, queryMemObject(dst), CL_FALSE, 0, size, src, 0, nullptr, &event));
  enqueueCallback(cb, event);
}

void ClDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const std::optional<Callback> &cb) {
  cl_event event = {};
  CHECKED(clEnqueueReadBuffer(queue, queryMemObject(src), CL_FALSE, 0, size, dst, 0, nullptr, &event));
  enqueueCallback(cb, event);
}

void ClDevice::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                  const std::vector<TypedPointer> &args, TypedPointer rtn, const Policy &policy,
                                  const std::optional<Callback> &cb) {

  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[m, fnTable] = moduleIt->second;
  cl_kernel kernel = nullptr;
  if (auto it = fnTable.find(symbol); it != fnTable.end()) kernel = it->second;
  else {
    auto &&m0 = m; // XXX we need this crap because OUT_CHECKED internally uses a lambda (can't use bindings in those)
    kernel = OUT_CHECKED(clCreateKernel(m0, symbol.c_str(), OUT_ERR));
    fnTable.emplace_hint(it, symbol, kernel);
  }

  auto toSize = [](Type t) -> size_t {
    switch (t) {
      case Type::Bool8: // fallthrough
      case Type::Byte8: return 8 / 8;
      case Type::CharU16: // fallthrough
      case Type::Short16: return 16 / 8;
      case Type::Int32: return 32 / 8;
      case Type::Long64: return 64 / 8;
      case Type::Float32: return 32 / 8;
      case Type::Double64: return 64 / 8;
      case Type::Ptr: return sizeof(cl_mem);
      case Type::Void: throw std::logic_error("Illegal argument type: void");
    }
  };

  for (cl_uint i = 0; i < args.size(); ++i) {
    auto [tpe, rawPtr] = args[i];
    if (tpe == Type::Ptr) {
      cl_mem mem = queryMemObject(*static_cast<uintptr_t *>(rawPtr));
      CHECKED(clSetKernelArg(kernel, i, toSize(tpe), &mem));
    } else
      CHECKED(clSetKernelArg(kernel, i, toSize(tpe), rawPtr));
  }

  auto global = policy.global;
  auto local = policy.local.value_or(Dim{});

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

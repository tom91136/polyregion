#pragma once

#include "../runtime/runtime.h"
#include "concurrency_utils.hpp"
#include <algorithm>
#include <execution>

namespace polystl {


using namespace polyregion::runtime;

static const std::unordered_map<std::string, Backend> NameToBackend = {
    {"ptx", Backend::CUDA},             //
    {"cuda", Backend::CUDA},            //
    {"amdgpu", Backend::HIP},           //
    {"hip", Backend::HIP},              //
    {"hsa", Backend::HSA},              //
    {"opencl", Backend::OpenCL},        //
    {"ocl", Backend::OpenCL},           //
    {"cl", Backend::OpenCL},            //
    {"vulkan", Backend::Vulkan},        //
    {"vk", Backend::Vulkan},            //
    {"metal", Backend::Metal},          //
    {"mtl", Backend::Metal},            //
    {"apple", Backend::Metal},          //
    {"host_so", Backend::SHARED_OBJ},   //
    {"host", Backend::RELOCATABLE_OBJ}, //
};

std::unique_ptr<polyregion::runtime::Platform> createPlatform() {
  if (auto env = std::getenv("POLY_PLATFORM"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    if (auto it = NameToBackend.find(name); it != NameToBackend.end()) return Platform::of(it->second);
  }
  return {};
}

std::unique_ptr<polyregion::runtime::Device> selectDevice(polyregion::runtime::Platform &p) {
  if (auto env = std::getenv("POLY_DEVICE"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    auto devices = p.enumerate();
    size_t index = std::strtol(name.c_str(), nullptr, 10);
    if (errno == 0 && (index - 1) > devices.size()) { // we got a number, check inbounds and select device
      return std::move(devices[index]);
    } else if (auto matching = // or do a substring match
               std::find_if(devices.begin(), devices.end(),
                            [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
               matching != devices.end()) {
      return std::move(*matching);
    }
  }
  return {};
}

template <class _ExecutionPolicy, //
          class _ForwardIterator1,
          class _UnaryOperation>
void for_each(_ExecutionPolicy &&__exec, //
              _ForwardIterator1 __first, //
              _ForwardIterator1 __last,  //
              _UnaryOperation __op) {

  using namespace polyregion::runtime;

  const std::string &image = __op.__kernelImage;
  const std::string &name = __op.__uniqueName;
  const ArgBuffer &buffer = __op.__argBuffer;

  static auto thePlatform = createPlatform();
  static auto theDevice = thePlatform ? selectDevice(*thePlatform) : std::unique_ptr<polyregion::runtime::Device>{};
  static auto theQueue = theDevice ? theDevice->createQueue() : std::unique_ptr<polyregion::runtime::DeviceQueue>{};

  if (theDevice && theQueue) {
    if (!theDevice->moduleLoaded(name)) {
      theDevice->loadModule(name, image);
    }
    size_t N = std::distance(__first, __last);
    polyregion::concurrency_utils::waitAll([&](auto cb) {
      theQueue->enqueueInvokeAsync(name, "kernel", buffer, Policy{Dim3{N, 1, 1}, {}}, [&]() {
        fprintf(stderr, "Module %s completed\n", name.c_str());
        cb();
      });
    });
    fprintf(stderr, "Done\n");
  } else {
    fprintf(stderr, "Host fallback\n");
    std::for_each(__exec, __first, __last, __op);
  }
}

} // namespace polystl

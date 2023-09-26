#pragma once

#include "../runtime/runtime.h"
#include "concurrency_utils.hpp"
#include <cstring>

namespace polystl {

namespace {
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
static std::unique_ptr<polyregion::runtime::Platform> createPlatform() {
  if (auto env = std::getenv("POLY_PLATFORM"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    if (auto it = NameToBackend.find(name); it != NameToBackend.end()) return Platform::of(it->second);
  }
  return {};
}

static std::unique_ptr<polyregion::runtime::Device> selectDevice(polyregion::runtime::Platform &p) {
  if (auto env = std::getenv("POLY_DEVICE"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    auto devices = p.enumerate();
    errno = 0;
    size_t index = std::strtol(name.c_str(), nullptr, 10);
    if (errno == 0 && index < devices.size()) { // we got a number, check inbounds and select device
      return std::move(devices.at(index));
    } else if (auto matching = // or do a substring match
               std::find_if(devices.begin(), devices.end(),
                            [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
               matching != devices.end()) {
      return std::move(*matching);
    }
  }
  return {};
}

} // namespace

template <typename F>
void __polyregion_offload_dispatch__(size_t global,              //
                                     size_t local,               //
                                     size_t localMemBytes,       //
                                     F __f, const char *__kernelName, const unsigned char *__kernelImageBytes, size_t __kernelImageSize) {

  using namespace polyregion::runtime;

  char argData[sizeof(F)];
  std::memcpy(argData, (&__f), sizeof(F));
  const ArgBuffer buffer{{Type::Ptr, argData}};

  static auto thePlatform = createPlatform();
  static auto theDevice = thePlatform ? selectDevice(*thePlatform) : std::unique_ptr<polyregion::runtime::Device>{};
  static auto theQueue = theDevice ? theDevice->createQueue() : std::unique_ptr<polyregion::runtime::DeviceQueue>{};

  if (theDevice && theQueue) {
    if (!theDevice->moduleLoaded(__kernelName)) {
      theDevice->loadModule(__kernelName, std::string(__kernelImageBytes, __kernelImageBytes + __kernelImageSize));
    }

    // [](int n) { __op(__first[x]);  }

    polyregion::concurrency_utils::waitAll([&](auto cb) {
      theQueue->enqueueInvokeAsync(
          __kernelName, "kernel", buffer,
          Policy{                    //
                 Dim3{global, 1, 1}, //
                 local > 0 ? std::optional{std::pair<Dim3, size_t>{Dim3{local, 0, 0}, localMemBytes}} : std::nullopt},
          [&]() {
            fprintf(stderr, "Module %s completed\n", __kernelName);
            cb();
          });
    });
    fprintf(stderr, "Done\n");
  } else {
    fprintf(stderr, "Host fallback\n");
//    for (size_t i = 0; i < global; ++i) {
//      __fallback(i);
//    }
  }
}

} // namespace polystl

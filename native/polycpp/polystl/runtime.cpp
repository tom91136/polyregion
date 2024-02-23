#pragma once

#include "polyrt/runtime.h"
#include "concurrency_utils.hpp"
#include "polystl/runtime.h"

using namespace polyregion::runtime;

// -fstdpar

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

std::unique_ptr<polyregion::runtime::Platform> polystl::createPlatform() {
  if (auto env = std::getenv("POLY_PLATFORM"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    if (auto it = NameToBackend.find(name); it != NameToBackend.end()) return Platform::of(it->second);
  }
  return {};
}

std::unique_ptr<polyregion::runtime::Device> polystl::selectDevice(polyregion::runtime::Platform &p) {
  auto devices = p.enumerate();
  if (auto env = std::getenv("POLY_DEVICE"); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
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
  } else if (!devices.empty())
    return std::move(devices[0]);
  return {};
}

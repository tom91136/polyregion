#include "polyrt/runtime.h"
#include "memory.hpp"
#include "polyregion/concurrency_utils.hpp"
#include "polystl/polystl.h"
#include "rt-reflect/rt_reflect.hpp"

#include <algorithm>

constexpr auto PlatformSelectorEnv = "POLYSTL_PLATFORM";
constexpr auto DeviceSelectorEnv = "POLYSTL_DEVICE";

using namespace polyregion::runtime;

std::unique_ptr<Platform> polyregion::polystl::currentPlatform{};
std::unique_ptr<Device> polyregion::polystl::currentDevice{};
std::unique_ptr<DeviceQueue> polyregion::polystl::currentQueue{};

static polyregion::polystl::SynchronisedAllocation allocations(
    [](const void *ptr) -> polyregion::polystl::PtrQuery {
      if (const auto meta = polyregion::rt_reflect::_rt_reflect_p(ptr); meta.type != polyregion::rt_reflect::Type::Unknown) {
        return polyregion::polystl::PtrQuery{.sizeInBytes = meta.size, .offsetInBytes = meta.offset};
      }
      POLYSTL_LOG("Local: Failed to query %p", ptr);
      return polyregion::polystl::PtrQuery{0, 0};
    },
    /*remoteAlloc*/
    [](const size_t size) { //
      const auto p = polyregion::polystl::currentDevice->mallocDevice(size, Access::RW);
      POLYSTL_LOG("                               Remote 0x%jx = malloc(%ld)", p, size);
      return p;
    },
    /*remoteRead*/
    [](void *dst, const uintptr_t src, const size_t srcOffset, const size_t size) {
      POLYSTL_LOG("Local %p <|[%4ld]- Remote [%p + %4ld]", dst, size, reinterpret_cast<void *>(src), srcOffset);
      polyregion::polystl::currentQueue->enqueueDeviceToHostAsync(src, srcOffset, dst, size, {});
    },
    /*remoteWrite*/
    [](const void *src, const uintptr_t dst, const size_t dstOffset, const size_t size) {
      POLYSTL_LOG("Local %p -[%4ld]|> Remote [%p + %4ld]", src, size, reinterpret_cast<void *>(dst), dstOffset);
      polyregion::polystl::currentQueue->enqueueHostToDeviceAsync(src, dst, dstOffset, size, {});
    },
    /*remoteRelease*/
    [](const uintptr_t remotePtr) {
      POLYSTL_LOG("                               Remote free(0x%jx)", remotePtr);
      polyregion::polystl::currentDevice->freeDevice(remotePtr);
    });

void polyregion::polystl::initialise() {
  const auto setupBackend = [](const Backend backend) {
    if (auto errorOrPlatform = Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
      POLYSTL_LOG("Backend %s failed to initialise: %s", to_string(backend).data(), std::get<std::string>(errorOrPlatform).c_str());
    } else currentPlatform = std::move(std::get<std::unique_ptr<Platform>>(errorOrPlatform));
  };

  const auto selectDevice = [](Platform &p) {
    auto devices = p.enumerate();
    if (const auto env = std::getenv(DeviceSelectorEnv); env) {
      std::string name(env);
      std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
      errno = 0; // strtol to avoid exceptions
      if (const size_t index = std::strtol(name.c_str(), nullptr, 10);
          errno == 0 && index < devices.size()) { // we got a number, check inbounds and select device
        currentDevice = std::move(devices.at(index));
      } else if (const auto matching = // or do a substring match
                 std::find_if(devices.begin(), devices.end(),
                              [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
                 matching != devices.end()) {
        currentDevice = std::move(*matching);
      }
    } else if (!devices.empty()) currentDevice = std::move(devices[0]);
  };

  const auto selectPlatform = [&]() {
    const static std::unordered_map<std::string, Backend> NameToBackend = {
        {"host", Backend::RelocatableObject}, //
        {"host_so", Backend::SharedObject},   //

        {"ptx", Backend::CUDA},  //
        {"cuda", Backend::CUDA}, //

        {"amdgpu", Backend::HIP}, //
        {"hip", Backend::HIP},    //
        {"hsa", Backend::HSA},    //

        {"opencl", Backend::OpenCL}, //
        {"ocl", Backend::OpenCL},    //
        {"cl", Backend::OpenCL},     //

        {"vulkan", Backend::Vulkan}, //
        {"vk", Backend::Vulkan},     //

        {"metal", Backend::Metal}, //
        {"mtl", Backend::Metal},   //
        {"apple", Backend::Metal}, //
    };

    if (const auto env = std::getenv(PlatformSelectorEnv); env) {
      std::string name(env);
      std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
      if (const size_t pos = name.find('@'); pos != std::string::npos) name = name.substr(0, pos);
      if (auto it = NameToBackend.find(name); it != NameToBackend.end()) setupBackend(it->second);
      else {
        POLYSTL_LOG("Backend %s is not a supported value for %s; options are %s={", env, PlatformSelectorEnv, PlatformSelectorEnv);
        size_t i = 0;
        for (auto &[k, _] : NameToBackend)
          std::fprintf(stderr, "%s%s", k.c_str(), i++ < NameToBackend.size() - 1 ? "|" : "");
        std::fprintf(stderr, "}\n");
      }
    } else {
      POLYSTL_LOG("Backend selector %s is not set: using default host platform", PlatformSelectorEnv);
      setupBackend(Backend::RelocatableObject);
    }
  };

  if (!currentPlatform) {
    POLYSTL_LOG("Initialising backends... (addr=%p)", (void *)&initialise);
    selectPlatform();
    if (currentPlatform) selectDevice(*currentPlatform);
    if (currentDevice) currentQueue = currentDevice->createQueue(std::chrono::seconds(10));
    if (currentPlatform) {
      POLYSTL_LOG("- Platform: %s [%s, %s] Device: %s",
                  currentPlatform->name().c_str(),           //
                  to_string(currentPlatform->kind()).data(), //
                  to_string(currentPlatform->moduleFormat()).data(), currentDevice->name().c_str());
    }
  }
}

bool polyregion::polystl::loadKernelObject(const char *moduleName, const KernelObject &object) {
  initialise();
  if (!currentPlatform || !currentDevice || !currentQueue) {
    POLYSTL_LOG("No device/queue in %s", __func__);
    return false;
  }

  if (currentPlatform->kind() != object.kind || currentPlatform->moduleFormat() != object.format) {
    POLYSTL_LOG("Skipping incompatible image: %s [%s] (targeting %s [%s])",
                to_string(object.kind).data(), //
                to_string(object.format).data(),
                to_string(currentPlatform->kind()).data(), //
                to_string(currentPlatform->moduleFormat()).data());
    return false;
  }

  POLYSTL_LOG("Found compatible image: %s [%s] (targeting %s [%s])",
              to_string(object.kind).data(), //
              to_string(object.format).data(),
              to_string(currentPlatform->kind()).data(), //
              to_string(currentPlatform->moduleFormat()).data());
  if (!currentDevice->moduleLoaded(moduleName)) //
    currentDevice->loadModule(moduleName, std::string(object.image, object.image + object.imageLength));
  return true;
}

void polyregion::polystl::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  POLYSTL_LOG("<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  concurrency_utils::waitAll([&](auto &cb) {
    ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
    currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, Policy{Dim3{global, 1, 1}}, [&]() {
      POLYSTL_LOG("<%s:%s:%zu> Unlatched", __func__, moduleId, global);
      cb();
    });
  });
  POLYSTL_LOG("<%s:%s:%zu> Done", __func__, moduleId, global);
}

void polyregion::polystl::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes, const TypeLayout *layout,
                                          void *functorData, const char *moduleId) {
  POLYSTL_LOG("<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);
  // auto functorDevicePtr = currentDevice->mallocDevice(functorDataSize, Access::RW);
  // concurrency_utils::waitAll(
  //     [&](auto &cb) { currentQueue->enqueueHostToDeviceAsync(functorData, functorDevicePtr, 0, functorDataSize, [&]() { cb(); }); });

  auto functorDevicePtr = allocations.syncLocalToRemote(functorData, *layout);

  // concurrency_utils::waitAll([&](auto &cb) {
  auto buffer = //
      localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorDevicePtr}, {Type::Void, {}}}
                        : ArgBuffer{{Type::Ptr, &functorDevicePtr}, {Type::Void, {}}};

  currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, //
                                   Policy{                    //
                                          Dim3{global, 1, 1}, //
                                          local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt},
                                   {});
  // });
  allocations.syncRemoteToLocal(functorData);
  currentQueue->enqueueWaitBlocking();
  // currentDevice->freeDevice(functorDevicePtr);
  POLYSTL_LOG("<%s:%s:%zu> Done", __func__, moduleId, global);
}

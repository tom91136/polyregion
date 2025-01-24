#include "polyregion/concurrency_utils.hpp"
#include "polyrt/rt.h"

#include <algorithm>

#ifdef POLYRT_LOG
  #error Trace already defined
#else

  #define POLYRT_LOG(fmt, ...) std::fprintf(stderr, "[PolyRT] " fmt "\n", __VA_ARGS__)
//  #define POLYRT_LOG(fmt, ...)
#endif

constexpr auto PlatformSelectorEnv = "POLYRT_PLATFORM";
constexpr auto DeviceSelectorEnv = "POLYRT_DEVICE";

using namespace polyregion::invoke;

std::unique_ptr<Platform> polyregion::polyrt::currentPlatform{};
std::unique_ptr<Device> polyregion::polyrt::currentDevice{};
std::unique_ptr<DeviceQueue> polyregion::polyrt::currentQueue{};

void polyregion::polyrt::initialise() {
  const auto setupBackend = [](const Backend backend) {
    if (auto errorOrPlatform = Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
      POLYRT_LOG("Backend %s failed to initialise: %s", to_string(backend).data(), std::get<std::string>(errorOrPlatform).c_str());
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
        POLYRT_LOG("Backend %s is not a supported value for %s; options are %s={", env, PlatformSelectorEnv, PlatformSelectorEnv);
        size_t i = 0;
        for (auto &[k, _] : NameToBackend)
          std::fprintf(stderr, "%s%s", k.c_str(), i++ < NameToBackend.size() - 1 ? "|" : "");
        std::fprintf(stderr, "}\n");
      }
    } else {
      POLYRT_LOG("Backend selector %s is not set: using default host platform", PlatformSelectorEnv);
      setupBackend(Backend::RelocatableObject);
    }
  };

  if (!currentPlatform) {
    POLYRT_LOG("Initialising backends... (addr=%p)", (void *)&initialise);
    selectPlatform();
    if (currentPlatform) selectDevice(*currentPlatform);
    if (currentDevice) currentQueue = currentDevice->createQueue(std::chrono::seconds(10));
    if (currentPlatform) {
      POLYRT_LOG("- Platform: %s [%s, %s] Device: %s",
                 currentPlatform->name().c_str(),           //
                 to_string(currentPlatform->kind()).data(), //
                 to_string(currentPlatform->moduleFormat()).data(), currentDevice->name().c_str());
    }
  }
}

bool polyregion::polyrt::loadKernelObject(const char *moduleName, const KernelObject &object) {
  initialise();
  if (!currentPlatform || !currentDevice || !currentQueue) {
    POLYRT_LOG("No device/queue in %s", __func__);
    return false;
  }

  if (currentPlatform->kind() != object.kind || currentPlatform->moduleFormat() != object.format) {
    POLYRT_LOG("Skipping incompatible image: %s [%s] (targeting %s [%s])",
               to_string(object.kind).data(), //
               to_string(object.format).data(),
               to_string(currentPlatform->kind()).data(), //
               to_string(currentPlatform->moduleFormat()).data());
    return false;
  }

  POLYRT_LOG("Found compatible image: %s [%s] (targeting %s [%s])",
             to_string(object.kind).data(), //
             to_string(object.format).data(),
             to_string(currentPlatform->kind()).data(), //
             to_string(currentPlatform->moduleFormat()).data());
  if (!currentDevice->moduleLoaded(moduleName)) //
    currentDevice->loadModule(moduleName, std::string(object.image, object.image + object.imageLength));
  return true;
}

void polyregion::polyrt::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  POLYRT_LOG("<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  concurrency_utils::waitAll([&](auto &cb) {
    ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
    currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, Policy{Dim3{global, 1, 1}}, [&]() {
      POLYRT_LOG("<%s:%s:%zu> Unlatched", __func__, moduleId, global);
      cb();
    });
  });
  POLYRT_LOG("<%s:%s:%zu> Done", __func__, moduleId, global);
}

void polyregion::polyrt::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes, void *functorData,
                                         const char *moduleId) {
  POLYRT_LOG("<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);

  const auto buffer = localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorData}, {Type::Void, {}}}
                                        : ArgBuffer{{Type::Ptr, &functorData}, {Type::Void, {}}};

  currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, //
                                   Policy{                    //
                                          Dim3{global, 1, 1}, //
                                          local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt},
                                   {});

  POLYRT_LOG("<%s:%s:%zu> Submitted", __func__, moduleId, global);
}

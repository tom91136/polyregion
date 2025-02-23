#include <algorithm>
#include <cstdarg>

#include "polyregion/concurrency_utils.hpp"
#include "polyrt/rt.h"

constexpr auto PlatformSelectorEnv = "POLYRT_PLATFORM";
constexpr auto DeviceSelectorEnv = "POLYRT_DEVICE";
constexpr auto HostFallbackEnv = "POLYRT_HOST_FALLBACK";
constexpr auto DebugEnv = "POLYRT_DEBUG";

using namespace polyregion::invoke;
using polyregion::polyrt::DebugLevel;

std::unique_ptr<Platform> polyregion::polyrt::currentPlatform{};
std::unique_ptr<Device> polyregion::polyrt::currentDevice{};
std::unique_ptr<DeviceQueue> polyregion::polyrt::currentQueue{};

static void setupBackend(const Backend backend) {
  if (auto errorOrPlatform = Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
    log(DebugLevel::None, "Backend %s failed to initialise: %s", to_string(backend).data(), std::get<std::string>(errorOrPlatform).c_str());
  } else polyregion::polyrt::currentPlatform = std::move(std::get<std::unique_ptr<Platform>>(errorOrPlatform));
}

static void selectDevice(Platform &p) {
  auto devices = p.enumerate();
  if (const auto env = std::getenv(DeviceSelectorEnv); env) {
    std::string name(env);
    std::transform(name.begin(), name.end(), name.begin(), [](auto &c) { return std::tolower(c); });
    errno = 0; // strtol to avoid exceptions
    if (const size_t index = std::strtol(name.c_str(), nullptr, 10);
        errno == 0 && index < devices.size()) { // we got a number, check inbounds and select device
      polyregion::polyrt::currentDevice = std::move(devices.at(index));
    } else if (const auto matching = // or do a substring match
               std::find_if(devices.begin(), devices.end(),
                            [&name](const auto &device) { return device->name().find(name) != std::string::npos; });
               matching != devices.end()) {
      polyregion::polyrt::currentDevice = std::move(*matching);
    }
  } else if (!devices.empty()) polyregion::polyrt::currentDevice = std::move(devices[0]);
};

static void selectPlatform() {
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
      log(DebugLevel::None, "Backend %s is not a supported value for %s; options are %s={", env, PlatformSelectorEnv, PlatformSelectorEnv);
      size_t i = 0;
      for (auto &[k, _] : NameToBackend)
        std::fprintf(stderr, "%s%s", k.c_str(), i++ < NameToBackend.size() - 1 ? "|" : "");
      std::fprintf(stderr, "}\n");
    }
  } else {
    log(DebugLevel::Debug, "Backend selector %s is not set: using default host platform", PlatformSelectorEnv);
    setupBackend(Backend::RelocatableObject);
  }
}

static std::optional<size_t> parseIntNoExcept(const char *str) {
  errno = 0; // strtol to avoid exceptions
  const size_t value = std::strtol(str, nullptr, 10);
  return errno == 0 ? std::optional{value} : std::nullopt;
}

void polyregion::polyrt::initialise() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (!currentPlatform) {
      log(DebugLevel::Info, "Initialising backends... (addr=%p)", (void *)&initialise);
      selectPlatform();
      if (currentPlatform) selectDevice(*currentPlatform);
      if (currentDevice) currentQueue = currentDevice->createQueue(std::chrono::seconds(10));
      if (currentPlatform) {
        log(DebugLevel::Info, "- Platform: %s [%s, %s] Device: %s",
            currentPlatform->name().c_str(),           //
            to_string(currentPlatform->kind()).data(), //
            to_string(currentPlatform->moduleFormat()).data(), currentDevice->name().c_str());
        std::string row;
        auto features = currentDevice->features();
        for (auto it = features.begin(); it != features.end(); ++it) {
          row += (row.empty() ? "" : ", ") + *it;
          if ((std::distance(features.begin(), it) + 1) % 10 == 0) {
            log(DebugLevel::Info, "  - %s", row.c_str());
            row.clear();
          }
        }
        if (!row.empty()) log(DebugLevel::Info, "  - %s", row.c_str());
      }
    }
  });
}

bool polyregion::polyrt::hostFallback() {
  static bool fallback = []() {
    if (const auto env = std::getenv(HostFallbackEnv); env) {
      if (const auto v = parseIntNoExcept(env); v && *v == 0) {
        log(DebugLevel::Debug, "<%s> No compatible backend and host fallback disabled, returning...", __func__);
        return false;
      }
    }
    return true; // The default is to use host fallback
  }();
  return fallback;
}

polyregion::polyrt::DebugLevel polyregion::polyrt::debugLevel() {
  static DebugLevel level = []() {
    if (const auto env = std::getenv(DebugEnv); env) {
      if (const auto v = parseIntNoExcept(env)) {
        if (*v < static_cast<std::underlying_type_t<DebugLevel>>(DebugLevel::Trace)) {
          return static_cast<DebugLevel>(*v);
        }
      }
    }
    return DebugLevel::None;
  }();
  return level;
}

void polyregion::polyrt::log(const DebugLevel level, const char *fmt, ...) {
  if (debugLevel() < level) return;
  va_list args;
  va_start(args, fmt);
  std::fprintf(stderr, "[PolyRT] ");
  std::vfprintf(stderr, fmt, args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
  va_end(args);
}

bool polyregion::polyrt::loadKernelObject(const char *moduleName, const KernelObject &object) {
  initialise();
  if (!currentPlatform || !currentDevice || !currentQueue) {
    log(DebugLevel::Info, "No device/queue in %s", __func__);
    return false;
  }

  if (currentPlatform->kind() != object.kind || currentPlatform->moduleFormat() != object.format) {
    log(DebugLevel::Debug, "Skipping incompatible image: %s [%s] (targeting %s [%s])",
        to_string(object.kind).data(), //
        to_string(object.format).data(),
        to_string(currentPlatform->kind()).data(), //
        to_string(currentPlatform->moduleFormat()).data());
    return false;
  }

  log(DebugLevel::Debug, "Found compatible image: %s [%s] (targeting %s [%s])",
      to_string(object.kind).data(), //
      to_string(object.format).data(),
      to_string(currentPlatform->kind()).data(), //
      to_string(currentPlatform->moduleFormat()).data());

  //  TODO need to check if object.feature is a strict subset of device feature

  if (!currentDevice->moduleLoaded(moduleName)) //
    currentDevice->loadModule(moduleName, std::string(object.image, object.image + object.imageLength));
  return true;
}

void polyregion::polyrt::dispatchHostThreaded(const size_t global, void *functorData, const char *moduleId) {
  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch hostthread", __func__, moduleId, global);
  concurrency_utils::waitAll([&](auto &cb) {
    ArgBuffer buffer{{Type::IntS64, nullptr}, {Type::Ptr, &functorData}, {Type::Void, nullptr}};
    currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, Policy{Dim3{global, 1, 1}}, [&]() {
      log(DebugLevel::Debug, "<%s:%s:%zu> Unlatched", __func__, moduleId, global);
      cb();
    });
  });
  log(DebugLevel::Debug, "<%s:%s:%zu> Done", __func__, moduleId, global);
}

void polyregion::polyrt::dispatchManaged(const size_t global, const size_t local, const size_t localMemBytes, void *functorData,
                                         const char *moduleId) {
  log(DebugLevel::Debug, "<%s:%s:%zu> Dispatch managed, arg=%p bytes", __func__, moduleId, global, functorData);

  const auto buffer = localMemBytes > 0 ? ArgBuffer{{Type::Scratch, {}}, {Type::Ptr, &functorData}, {Type::Void, {}}}
                                        : ArgBuffer{{Type::Ptr, &functorData}, {Type::Void, {}}};

  currentQueue->enqueueInvokeAsync(moduleId, "_main", buffer, //
                                   Policy{                    //
                                          Dim3{global, 1, 1}, //
                                          local > 0 ? std::optional{std::pair{Dim3{local, 1, 1}, localMemBytes}} : std::nullopt},
                                   {});

  log(DebugLevel::Debug, "<%s:%s:%zu> Submitted", __func__, moduleId, global);
}

static void *sharedAlloc(const size_t size) {
  if (const auto p = polyregion::polyrt::currentDevice->mallocShared(size, Access::RW)) return *p;
  log(DebugLevel::None, "Device %s does not support shared allocation, aborting...\n", polyregion::polyrt::currentDevice->name().c_str());
  std::abort();
}

static void sharedFree(void *p) { polyregion::polyrt::currentDevice->freeShared(p); }

extern "C" {

POLYREGION_EXPORT void *polyrt_usm_malloc(const size_t size) {
  polyregion::polyrt::initialise();
  return sharedAlloc(size);
}

POLYREGION_EXPORT void *polyrt_usm_calloc(const size_t nmemb, const size_t size) {
  polyregion::polyrt::initialise();
  return sharedAlloc(nmemb * size);
}

POLYREGION_EXPORT void *polyrt_usm_realloc(void *ptr, const size_t size) {
  polyregion::polyrt::initialise();
  const auto p = sharedAlloc(size);
  std::memcpy(p, ptr, size);
  sharedFree(ptr);
  return p;
}

POLYREGION_EXPORT void *polyrt_usm_memalign(const size_t /*alignment*/, const size_t size) {
  polyregion::polyrt::initialise();
  return sharedAlloc(size);
}

POLYREGION_EXPORT void *polyrt_usm_aligned_alloc(size_t /*alignment*/, const size_t size) {
  polyregion::polyrt::initialise();
  return sharedAlloc(size);
}

POLYREGION_EXPORT void polyrt_usm_free(void *ptr) {
  polyregion::polyrt::initialise();
  sharedFree(ptr);
}

POLYREGION_EXPORT void *polyrt_usm_operator_new(const size_t size) {
  polyregion::polyrt::initialise();
  return sharedAlloc(size);
}

POLYREGION_EXPORT void polyrt_usm_operator_delete(void *ptr) {
  polyregion::polyrt::initialise();
  sharedFree(ptr);
}

POLYREGION_EXPORT void polyrt_usm_operator_delete_sized(void *ptr, size_t /*size*/) {
  polyregion::polyrt::initialise();
  sharedFree(ptr);
}
}
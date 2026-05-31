#include <algorithm>
#include <cstdarg>
#include <mutex>
#include <unordered_set>

#include "aspartame/all.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"
#include "polyregion/env_keys.h"
#include "polyregion/types.h"
#include "polyrt/rt.h"

// XXX __RT_IMPL defines polyreflect-rt singletons in this TU so SMA's localReflect can resolve
// captured pointers. On Windows the HashMap allocator routes through HeapAlloc (rt_protected.hpp)
// to avoid recursing back into polyrt_usm_* via InterposePass.
#include "reflect-rt/rt.hpp"

using namespace polyregion::invoke;
using namespace aspartame;
using polyregion::polyrt::DebugLevel;

std::unique_ptr<Platform> polyregion::polyrt::currentPlatform{};
std::unique_ptr<Device> polyregion::polyrt::currentDevice{};
std::unique_ptr<DeviceQueue> polyregion::polyrt::currentQueue{};

static std::optional<size_t> parseIntNoExcept(const char *str) {
  errno = 0;
  char *end = nullptr;
  const size_t value = std::strtol(str, &end, 10);
  // strtol returns 0 on "no conversion" without setting errno; reject empty / trailing garbage.
  if (errno != 0 || end == str || *end != '\0') return std::nullopt;
  return value;
}

// Preserved for diagnostics; the DeviceLock keys on the device's PhysicalDevice, not the backend.
static std::optional<Backend> selectedBackend;

static void setupBackend(const Backend backend) {
  auto errorOrPlatform = Platform::of(backend);
  if (const auto err = errorOrPlatform ^ get_maybe<std::string>())
    log(DebugLevel::None, "Backend %s failed to initialise: %s", magic_enum::enum_name(backend).data(), err->c_str());
  else {
    polyregion::polyrt::currentPlatform = std::move(std::get<std::unique_ptr<Platform>>(errorOrPlatform));
    selectedBackend = backend;
  }
}

namespace {
// Selection is by feature equality, never by name substring -- names are for display.
bool deviceMatchesHint(Device &d, const std::string &hint) {
  if (hint.empty()) return true;
  const auto needle = hint ^ to_lower();
  for (const auto &f : d.features())
    if ((f ^ to_lower()) == needle) return true;
  return false;
}

bool deviceMatchesRequired(Device &d, const std::vector<std::string_view> &requiredFeatures) {
  if (requiredFeatures.empty()) return true;
  const auto features = d.features();
  for (const auto &req : requiredFeatures) {
    bool found = false;
    for (const auto &f : features) {
      if (f.size() != req.size()) continue;
      bool eq = true;
      for (size_t i = 0; i < f.size(); ++i) {
        char x = f[i], y = req[i];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y + 32);
        if (x != y) {
          eq = false;
          break;
        }
      }
      if (eq) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}
} // namespace

static void selectDevice(Platform &p, const std::vector<std::string_view> &requiredFeatures, const std::string &hint, bool strict) {
  auto devices = p.enumerate();
  std::vector<std::unique_ptr<Device>> eligible;
  eligible.reserve(devices.size());
  for (auto &d : devices)
    if (deviceMatchesRequired(*d, requiredFeatures)) eligible.push_back(std::move(d));

  if (!hint.empty())
    if (const auto index = parseIntNoExcept(hint.c_str()); index && *index < eligible.size()) {
      polyregion::polyrt::currentDevice = std::move(eligible.at(*index));
      return;
    }
  for (auto &d : eligible)
    if (deviceMatchesHint(*d, hint)) {
      polyregion::polyrt::currentDevice = std::move(d);
      return;
    }
  if (!strict && !eligible.empty()) polyregion::polyrt::currentDevice = std::move(eligible.front());
}

static std::optional<polyregion::compiletime::TargetSpec::ParsedRef> selectPlatform() {
  std::optional<std::string> envValue;
  if (const auto env = std::getenv(polyregion::env::PolyrtPlatform)) envValue = env;
  if (!envValue) {
    log(DebugLevel::Debug, "Backend selector %s is not set: using default host platform", polyregion::env::PolyrtPlatform);
    setupBackend(Backend::RelocatableObject);
    return std::nullopt;
  }
  auto parsed = polyregion::compiletime::TargetSpec::parse(*envValue);
  if (!parsed) {
    log(DebugLevel::None, "Backend %s is not a supported value for %s", envValue->c_str(), polyregion::env::PolyrtPlatform);
    return std::nullopt;
  }
  setupBackend(parsed->spec.runtime);
  return parsed;
}

void polyregion::polyrt::initialise() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (!currentPlatform) {
      log(DebugLevel::Info, "Initialising backends... (addr=%p)", (void *)&initialise);
      auto parsed = selectPlatform();
      if (currentPlatform) {
        const auto requiredFeatures = parsed ? parsed->spec.requiredDeviceFeatures : std::vector<std::string_view>{};
        const auto hint = parsed ? parsed->hint : std::string{};
        // An explicit POLYRT_DEVICE override is strict: no fallback to devices[0].
        bool strict = false;
        std::string effectiveHint = hint;
        if (const auto env = std::getenv(polyregion::env::PolyrtDevice)) {
          effectiveHint = env;
          strict = true;
        }
        selectDevice(*currentPlatform, requiredFeatures, effectiveHint, strict);
      }
      // Test-only: cross-process lock so ctest -j workers do not race on the same device.
      // Held for process lifetime; the file lock auto-releases on exit.
      if (currentDevice && selectedBackend) {
        if (const auto env = std::getenv(polyregion::env::PolyinvokeTestLock); env && env[0] == '1') {
          static std::optional<polyregion::invoke::DeviceLock> currentDeviceLock;
          const auto physical = currentDevice->physicalDevice();
          // No-op for host/CPU devices; GPU backends sharing one physical device serialise.
          log(DebugLevel::Info, "<%s> Acquiring DeviceLock for (%s, %s)", __func__, magic_enum::enum_name(*selectedBackend).data(),
              physical.str().c_str());
          currentDeviceLock.emplace(physical);
        }
      }
      if (currentDevice) currentQueue = currentDevice->createQueue(std::chrono::seconds(10));
      // XXX HIP/CUDA/HSA runtimes don't survive explicit teardown during __cxa_finalize. their globals are being destroyed concurrently and
      // the destroy-stream call SIGSEGVs. Just leak it as program is terminating anyway.
      std::atexit([] {
        (void)currentQueue.release();
        (void)currentDevice.release();
        (void)currentPlatform.release();
      });
      if (currentPlatform) {
        log(DebugLevel::Info, "- Platform: %s [%s] Device: %s [%s]",
            currentPlatform->name().c_str(),                          //
            magic_enum::enum_name(currentPlatform->kind()).data(),    //
            currentDevice ? currentDevice->name().c_str() : "(none)", //
            currentDevice ? magic_enum::enum_name(currentDevice->moduleFormat()).data() : "(no device)");
        if (currentDevice)
          currentDevice->features() ^ grouped(10) ^
              for_each([](const auto &chunk) { log(DebugLevel::Info, "  - %s", (chunk ^ mk_string(", ")).c_str()); });
      }
    }
  });
}

void polyregion::polyrt::noCompatibleKernelExit(const char *site) {
  std::fprintf(stderr, "[PolyRT] %s: no kernel object matched any enumerated device, exiting 77 (skip)\n", site);
  std::fflush(stderr);
  std::_Exit(77);
}

bool polyregion::polyrt::hostFallback() {
  static bool fallback = []() {
    if (const auto env = std::getenv(polyregion::env::PolyrtHostFallback); env) {
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
    if (const auto env = std::getenv(polyregion::env::PolyrtDebug); env) {
      if (const auto v = parseIntNoExcept(env)) {
        if (*v <= static_cast<std::underlying_type_t<DebugLevel>>(DebugLevel::Trace)) {
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

  if (currentPlatform->kind() != object.kind || currentDevice->moduleFormat() != object.format) {
    log(DebugLevel::Debug, "Skipping incompatible image: %s [%s] (targeting %s [%s])",
        magic_enum::enum_name(object.kind).data(),   //
        magic_enum::enum_name(object.format).data(), //
        magic_enum::enum_name(currentPlatform->kind()).data(), magic_enum::enum_name(currentDevice->moduleFormat()).data());
    return false;
  }

  log(DebugLevel::Debug, "Found compatible image: %s [%s] (targeting %s [%s])",
      magic_enum::enum_name(object.kind).data(),   //
      magic_enum::enum_name(object.format).data(), //
      magic_enum::enum_name(currentPlatform->kind()).data(), magic_enum::enum_name(currentDevice->moduleFormat()).data());

  const auto deviceFeatures = currentDevice->features();
  for (size_t i = 0; i < object.featureCount; ++i) {
    std::string_view req(object.features[i]);
    if (req != "fp64" && req != "fp16" && req != "int64") continue;
    bool found = false;
    for (auto &f : deviceFeatures) {
      if (f.size() != req.size()) continue;
      bool eq = true;
      for (size_t j = 0; j < f.size(); ++j) {
        char x = f[j], y = req[j];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y + 32);
        if (x != y) {
          eq = false;
          break;
        }
      }
      if (eq) {
        found = true;
        break;
      }
    }
    if (!found) {
      log(DebugLevel::Debug, "Device %s lacks required feature `%s` for module `%s`; skipping", currentDevice->name().c_str(),
          std::string(req).c_str(), moduleName);
      return false;
    }
  }

  if (!currentDevice->moduleLoaded(moduleName)) {
    if (auto dumpDir = std::getenv(polyregion::env::PolyrtDumpKernel)) {
      static int counter = 0;
      auto path = std::string(dumpDir) + "/kernel_" + std::to_string(counter++) + ".o";
      if (FILE *f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(object.image, 1, object.imageLength, f);
        std::fclose(f);
      }
    }
    currentDevice->loadModule(moduleName, std::string(object.image, object.image + object.imageLength));
  }
  return true;
}

// XXX InterposePass routes every free/delete here, but not every pointer was allocated by us
// (pre-init static ctors, foreign runtimes, untouched TUs). Track our own allocations so
// foreign pointers fall through instead of being passed to the backend free.
//
// XXX Intentionally leaked: the SMA destructor at shutdown re-enters this allocator via delete
// callbacks; function-local statics would already be destroyed.
static std::mutex &usmAllocSetMutex() {
  static auto *m = new std::mutex();
  return *m;
}
static std::unordered_set<void *> &usmAllocSet() {
  static auto *s = new std::unordered_set<void *>();
  return *s;
}

static void *sharedAllocTracked(const size_t size, const polyregion::rt_reflect::Type recordType) {
  void *p = nullptr;
  if (polyregion::polyrt::currentDevice) {
    if (const auto shared = polyregion::polyrt::currentDevice->mallocShared(size, Access::RW)) p = *shared;
  }
  if (!p) p = std::malloc(size);
  if (p) {
    std::lock_guard<std::mutex> g(usmAllocSetMutex());
    usmAllocSet().insert(p);
    // Recording lets SMA's localReflect size the pointer when walking captured fields;
    // safe because polyreflect-rt's allocator bypasses InterposePass.
    polyregion::rt_reflect::_rt_record(p, size, recordType);
  }
  return p;
}

static void sharedFreeTracked(void *p, const polyregion::rt_reflect::Type releaseType) {
  if (!p) return;
  bool tracked = false;
  {
    std::lock_guard<std::mutex> g(usmAllocSetMutex());
    tracked = usmAllocSet().erase(p) > 0;
  }
  polyregion::rt_reflect::_rt_release(p, releaseType);
  if (tracked && polyregion::polyrt::currentDevice) {
    polyregion::polyrt::currentDevice->freeShared(p);
    return;
  }
  if (tracked) {
    std::free(p);
    return;
  }
  // Untracked: free spliced by InterposePass but the alloc was foreign (pre-init, foreign CRT,
  // uninstrumented TU). Leak rather than risk freeing on the wrong heap.
}

POLYREGION_EXPORT extern "C" void *polyrt_usm_malloc(const size_t size) {
  polyregion::polyrt::initialise();
  const auto p = sharedAllocTracked(size, polyregion::rt_reflect::Type::HeapMalloc);
  log(DebugLevel::Debug, "%p = polyrt_usm_malloc(%zu)", p, size);
  return p;
}

POLYREGION_EXPORT extern "C" void *polyrt_usm_aligned_alloc(size_t /*alignment*/, const size_t size) {
  polyregion::polyrt::initialise();
  const auto p = sharedAllocTracked(size, polyregion::rt_reflect::Type::HeapAlignedAlloc);
  log(DebugLevel::Debug, "%p = polyrt_usm_aligned_alloc(%zu)", p, size);
  return p;
}

POLYREGION_EXPORT extern "C" void polyrt_usm_free(void *ptr) {
  polyregion::polyrt::initialise();
  log(DebugLevel::Debug, "polyrt_usm_free(%p)", ptr);
  sharedFreeTracked(ptr, polyregion::rt_reflect::Type::HeapFree);
}

POLYREGION_EXPORT extern "C" void *polyrt_usm_operator_new(const size_t size) {
  polyregion::polyrt::initialise();
  const auto p = sharedAllocTracked(size, polyregion::rt_reflect::Type::HeapCXXNew);
  log(DebugLevel::Debug, "%p = polyrt_usm_operator_new(%zu)", p, size);
  return p;
}

POLYREGION_EXPORT extern "C" void polyrt_usm_operator_delete(void *ptr) {
  polyregion::polyrt::initialise();
  log(DebugLevel::Debug, "polyrt_usm_operator_delete(%p)", ptr);
  sharedFreeTracked(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
}

POLYREGION_EXPORT extern "C" void polyrt_usm_operator_delete_sized(void *ptr, size_t /*size*/) {
  polyregion::polyrt::initialise();
  log(DebugLevel::Debug, "polyrt_usm_operator_delete_sized(%p)", ptr);
  sharedFreeTracked(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
}

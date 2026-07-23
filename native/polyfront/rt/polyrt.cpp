#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32) && !defined(__APPLE__)
  #include <link.h>
#elif defined(_WIN32)
  #include <windows.h>
#elif defined(__APPLE__)
  #include <cstring>

  #include <mach-o/dyld.h>
  #include <mach-o/loader.h>
#endif

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/device_lock.h"
#include "polyregion/concurrency_utils.hpp"
#include "polyregion/dl.h"
#include "polyregion/env_keys.h"
#include "polyregion/polyc_jit_symbols.h"
#include "polyregion/types.h"
#include "polyrt/rt.h"

#include "jit_policy.hpp"

// XXX __RT_IMPL defines polyreflect-rt singletons in this TU so SMA's localReflect can resolve
// captured pointers. On Windows the HashMap allocator routes through HeapAlloc (rt_protected.hpp)
// to avoid recursing back into polyrt_usm_* via InterposePass.
// __RT_NO_GLOBAL_NEW: the global operator new override belongs in user executables, not this DSO.
#define __RT_NO_GLOBAL_NEW
#include "reflect-rt/rt.hpp"

using namespace polyregion::invoke;
using namespace aspartame;
using polyregion::polyrt::DebugLevel;

std::unique_ptr<Platform> polyregion::polyrt::currentPlatform{};
std::unique_ptr<Device> polyregion::polyrt::currentDevice{};
std::unique_ptr<DeviceQueue> polyregion::polyrt::currentQueue{};

namespace {
struct RoSegment {
  uintptr_t base;
  size_t size;
};
#if !defined(_WIN32) && !defined(__APPLE__)
POLYREGION_RT_PROTECT int collectRoSegment(struct dl_phdr_info *info, size_t, void *data) {
  if (info->dlpi_name && info->dlpi_name[0] != '\0') return 0;
  auto &out = *static_cast<std::vector<RoSegment> *>(data);
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const auto &ph = info->dlpi_phdr[i];
    if (ph.p_type != PT_LOAD) continue;
    if ((ph.p_flags & PF_W) != 0) continue;
    if ((ph.p_flags & PF_X) != 0) continue;
    if (ph.p_memsz == 0) continue;
    out.push_back({info->dlpi_addr + ph.p_vaddr, ph.p_memsz});
  }
  return 0;
}
#elif defined(__APPLE__)
// const data lives in __TEXT (r-x) and __DATA_CONST (name-matched since its initprot keeps W for dyld fixups)
POLYREGION_RT_PROTECT void collectRoSegmentsMachO(std::vector<RoSegment> &out) {
  const auto *hdr = reinterpret_cast<const mach_header_64 *>(_dyld_get_image_header(0));
  if (!hdr || hdr->magic != MH_MAGIC_64) return;
  const auto slide = _dyld_get_image_vmaddr_slide(0);
  const char *cursor = reinterpret_cast<const char *>(hdr) + sizeof(mach_header_64);
  for (uint32_t i = 0; i < hdr->ncmds; ++i) {
    const auto *lc = reinterpret_cast<const load_command *>(cursor);
    if (lc->cmd == LC_SEGMENT_64) {
      const auto *seg = reinterpret_cast<const segment_command_64 *>(cursor);
      const bool readOnly = (seg->initprot & VM_PROT_READ) && !(seg->initprot & VM_PROT_WRITE);
      const bool dataConst = std::strncmp(seg->segname, "__DATA_CONST", sizeof(seg->segname)) == 0;
      const bool linkEdit = std::strncmp(seg->segname, SEG_LINKEDIT, sizeof(seg->segname)) == 0;
      if ((readOnly || dataConst) && !linkEdit && seg->vmsize != 0)
        out.push_back({static_cast<uintptr_t>(seg->vmaddr) + static_cast<uintptr_t>(slide), static_cast<size_t>(seg->vmsize)});
    }
    cursor += lc->cmdsize;
  }
}
#else
POLYREGION_RT_PROTECT void collectRoSegmentsPE(std::vector<RoSegment> &out) {
  auto *base = reinterpret_cast<unsigned char *>(::GetModuleHandleW(nullptr));
  if (!base) return;
  const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
  const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE) return;
  const auto *section = IMAGE_FIRST_SECTION(nt);
  for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
    const DWORD chars = section->Characteristics;
    if ((chars & IMAGE_SCN_MEM_READ) == 0) continue;
    if ((chars & IMAGE_SCN_MEM_WRITE) != 0) continue;
    if ((chars & IMAGE_SCN_MEM_EXECUTE) != 0) continue;
    if (section->Misc.VirtualSize == 0) continue;
    out.push_back({reinterpret_cast<uintptr_t>(base + section->VirtualAddress), section->Misc.VirtualSize});
  }
}
#endif
POLYREGION_RT_PROTECT std::vector<RoSegment> roSegments;
POLYREGION_RT_PROTECT std::once_flag roSegmentsOnce;
} // namespace

void polyregion::polyrt::ensureRoSegmentsRecorded() {
  std::call_once(roSegmentsOnce, [] {
#if !defined(_WIN32) && !defined(__APPLE__)
    dl_iterate_phdr(collectRoSegment, &roSegments);
#elif defined(__APPLE__)
    collectRoSegmentsMachO(roSegments);
#else
    collectRoSegmentsPE(roSegments);
#endif
    for (const auto &s : roSegments)
      polyregion::rt_reflect::_rt_record(reinterpret_cast<void *>(s.base), s.size, polyregion::rt_reflect::Type::StaticRodata);
  });
}

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

bool hasFeature(Device &d, const std::string_view token) {
  const auto needle = std::string(token) ^ to_lower();
  return d.features() ^ exists([&](const std::string &f) { return (f ^ to_lower()) == needle; });
}

bool globMatch(const std::string_view pat, const std::string_view s) {
  const auto lc = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; };
  size_t pi = 0, si = 0, star = std::string_view::npos, mark = 0;
  while (si < s.size()) {
    if (pi < pat.size() && pat[pi] == '*') star = pi++, mark = si;
    else if (pi < pat.size() && (pat[pi] == '?' || lc(pat[pi]) == lc(s[si]))) ++pi, ++si;
    else if (star != std::string_view::npos) pi = star + 1, si = ++mark;
    else return false;
  }
  while (pi < pat.size() && pat[pi] == '*')
    ++pi;
  return pi == pat.size();
}
} // namespace

static void selectDevice(Platform &p, const std::vector<std::string_view> &requiredFeatures, const std::string &glob, bool strict) {
  auto devices = p.enumerate();
  auto eligible = devices                                    //
                  | map([](auto &d) { return std::ref(d); }) //
                  | filter([&](auto rw) {                    //
                      return requiredFeatures ^ forall([&](auto &r) { return hasFeature(*rw.get(), r); });
                    }) //
                  | to_vector();

  const std::string pattern = glob.empty() ? "*" : glob;
  auto matched = eligible                                                                //
                 | filter([&](auto rw) { return globMatch(pattern, rw.get()->name()); }) //
                 | to_vector();
  const auto names = [](auto &xs) { return xs | map([](auto rw) { return rw.get()->name(); }) | mk_string(", "); };

  if (matched.empty()) {
    if (strict || !eligible.empty()) {
      log(DebugLevel::None, "Selector '%s' matched none of the %zu eligible device(s): [%s]%s", pattern.c_str(), eligible.size(),
          names(eligible).c_str(), strict ? " (strict)" : "; refusing to run on a different device");
      std::fflush(stderr);
      std::abort();
    }
    return;
  }
  if (matched.size() > 1) {
    if (strict) {
      log(DebugLevel::None, "Selector '%s' is ambiguous (strict): matched %zu device(s): [%s]; tighten the glob", pattern.c_str(),
          matched.size(), names(matched).c_str());
      std::fflush(stderr);
      std::abort();
    }
    log(DebugLevel::Info, "Selector '%s' matched %zu device(s): [%s]; using the first", pattern.c_str(), matched.size(),
        names(matched).c_str());
  }
  polyregion::polyrt::currentDevice = std::move(matched.front().get());
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
        std::string glob = parsed ? parsed->deviceGlob : std::string{};
        if (const auto env = std::getenv(polyregion::env::PolyrtDevice)) glob = env;
        bool strict = false;
        if (const auto env = std::getenv(polyregion::env::PolyrtStrictSelect); env && env[0] && env[0] != '0') strict = true;
        selectDevice(*currentPlatform, requiredFeatures, glob, strict);
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
      if (currentDevice) {
        auto timeout = std::chrono::seconds(10);
        if (const auto env = std::getenv(polyregion::env::PolyrtQueueTimeoutSec); env && env[0]) {
          if (const auto secs = std::strtol(env, nullptr, 10); secs > 0) timeout = std::chrono::seconds(secs);
        }
        currentQueue = currentDevice->createQueue(timeout);
      }
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

void polyregion::polyrt::skipExit(const char *reason) {
  std::fprintf(stderr, "[PolyRT] %s, exiting 77 (skip)\n", reason);
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

struct JitAbi {
  polyregion::polyc_jit::abi::CompileFn compile = nullptr;
  polyregion::polyc_jit::abi::LastErrorFn lastError = nullptr;
  polyregion::polyc_jit::abi::FreeFn free = nullptr;
};

// Prefer linked symbols; fall back to POLYRT_JIT_LIB or the platform default.
static const JitAbi &resolveJitAbi() {
  static JitAbi abi = []() -> JitAbi {
    namespace a = polyregion::polyc_jit::abi;
    const auto from = [](polyregion_dl_handle h) {
      return JitAbi{reinterpret_cast<a::CompileFn>(polyregion_dl_find(h, a::Compile)),
                    reinterpret_cast<a::LastErrorFn>(polyregion_dl_find(h, a::LastError)),
                    reinterpret_cast<a::FreeFn>(polyregion_dl_find(h, a::Free))};
    };
#if defined(_WIN32)
    if (auto r = from(GetModuleHandleW(nullptr)); r.compile && r.free) return r;
#else
    if (auto r = from(RTLD_DEFAULT); r.compile && r.free) return r;
#endif
#if defined(_WIN32)
    constexpr auto defaultJitLib = "polyc.dll";
#elif defined(__APPLE__)
    constexpr auto defaultJitLib = "libpolyc.dylib";
#else
    constexpr auto defaultJitLib = "libpolyc.so";
#endif
    return std::array<const char *, 2>{std::getenv(polyregion::env::PolyrtJitLib), defaultJitLib} |
           collect_first([&](const char *name) -> std::optional<JitAbi> {
             if (!name || !name[0]) return std::nullopt;
             if (auto h = polyregion_dl_open(name))
               if (auto r = from(h); r.compile && r.free) return r;
             return std::nullopt;
           }) |
           get_or_else(JitAbi{});
  }();
  return abi;
}

// Specialise read-only scalar leaves; data borrows the capture.
static void collectSpecScalars(const unsigned char *base, const polyregion::runtime::TypeLayout *layout, const std::string &prefix,
                               int depth, std::deque<std::string> &names, std::vector<polyc_jit_spec_const_t> &out) {
  if (!base || !layout || depth > 4) return;
  view(layout->members, layout->memberCount ? layout->members + layout->memberCount : layout->members) | for_each([&](const auto &m) {
    if (!m.type) return;
    const auto name = prefix.empty() ? std::string(m.name) : prefix + "." + m.name;
    if (m.ptrIndirection == 0) {
      if (polyregion::runtime::isSet(m.type->attrs, polyregion::runtime::LayoutAttrs::Primitive)) {
        if (m.readOnly && m.sizeInBytes > 0 && m.sizeInBytes <= 8) {
          names.push_back(name);
          out.push_back({names.back().c_str(), m.type->name, base + m.offsetInBytes, m.sizeInBytes});
        }
      } else if (m.type->memberCount > 0) collectSpecScalars(base + m.offsetInBytes, m.type, name, depth + 1, names, out);
    } else if (m.ptrIndirection == 1 && m.type->memberCount > 0)
      collectSpecScalars(*reinterpret_cast<const unsigned char *const *>(base + m.offsetInBytes), m.type, name, depth + 1, names, out);
  });
}

struct SpecConsts {
  std::deque<std::string> names; // stable storage borrowed by values[].field
  std::vector<polyc_jit_spec_const_t> values;
};

static SpecConsts collectSpecConsts(const void *capture, const polyregion::runtime::TypeLayout *layout) {
  SpecConsts out;
  collectSpecScalars(static_cast<const unsigned char *>(capture), layout, {}, 0, out.names, out.values);
  return out;
}

static uint64_t hashSpecs(const std::vector<polyc_jit_spec_const_t> &specs) {
  const auto add = [](uint64_t h, const void *p, size_t n) {
    const auto *begin = static_cast<const unsigned char *>(p);
    return view(begin, n ? begin + n : begin) |
           fold_left(h, [](const uint64_t acc, const unsigned char x) { return (acc ^ x) * 1099511628211ull; });
  };
  return specs | fold_left(uint64_t{1469598103934665603ull}, [&](uint64_t h, const auto &s) {
           const auto fieldLen = std::strlen(s.field);
           const auto reprLen = std::strlen(s.repr);
           h = add(h, &fieldLen, sizeof(fieldLen));
           h = add(h, s.field, fieldLen);
           h = add(h, &reprLen, sizeof(reprLen));
           h = add(h, s.repr, reprLen);
           h = add(h, &s.dataLen, sizeof(s.dataLen));
           return add(h, s.data, s.dataLen);
         });
}

static bool envEnabled(const char *key) {
  const char *value = std::getenv(key);
  if (!value || !value[0]) return false;
  const std::string_view v(value);
  return v != "0" && v != "off" && v != "false";
}

static size_t envSize(const char *key, const size_t fallback) {
  const char *value = std::getenv(key);
  if (!value || !value[0]) return fallback;
  size_t result = 0;
  const auto [end, error] = std::from_chars(value, value + std::strlen(value), result);
  return error == std::errc{} && !*end ? result : fallback;
}

static polyregion::polyrt::AdaptiveJitPolicy &jitPolicy() {
  static auto *policy = new polyregion::polyrt::AdaptiveJitPolicy(envSize(polyregion::env::PolyrtJitSpecialiseHot, 3),
                                                                  envSize(polyregion::env::PolyrtJitSpecialiseLimit, 8));
  return *policy;
}

static std::mutex &jitPolicyMutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static bool jitCompileObject(const char *moduleName, const polyregion::runtime::KernelObject &object,
                             const std::vector<polyc_jit_spec_const_t> &specialise, std::string &out) {
  const auto &abi = resolveJitAbi();
  if (!abi.compile || !abi.free) {
    polyregion::polyrt::log(DebugLevel::None, "JIT program for `%s` but no compiler available; link with -fstdpar-jit or set %s",
                            moduleName, polyregion::env::PolyrtJitLib);
    return false;
  }
  uint8_t *image = nullptr;
  size_t imageLen = 0;
  const auto start = std::chrono::steady_clock::now();
  const auto rc = abi.compile(object.program, object.programLength, static_cast<uint32_t>(object.target), object.arch, object.pipelineSpec,
                              static_cast<uint32_t>(object.opt), specialise.empty() ? nullptr : specialise.data(), specialise.size(),
                              &image, &imageLen);
  if (rc != POLYC_JIT_OK || !image) {
    const char *msg = abi.lastError ? abi.lastError() : nullptr;
    polyregion::polyrt::log(DebugLevel::None, "JIT compile failed for `%s`: %s", moduleName, msg ? msg : "(no message)");
    if (image && abi.free) abi.free(image);
    return false;
  }
  out.assign(reinterpret_cast<const char *>(image), imageLen);
  abi.free(image);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  polyregion::polyrt::log(DebugLevel::Debug, "JIT compiled `%s` for %s [%s] (%zu const) in %lldms -> %zu bytes", moduleName,
                          magic_enum::enum_name(object.target).data(), object.arch ? object.arch : "", specialise.size(),
                          static_cast<long long>(ms), imageLen);
  return true;
}

bool polyregion::polyrt::loadKernelObject(const char *moduleName, const KernelObject &object, const void *capture,
                                          const TypeLayout *interfaceLayout, std::string *loadedModuleName) {
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

  for (size_t i = 0; i < object.featureCount; ++i) {
    std::string_view req(object.features[i]);
    if (req != "fp64" && req != "fp16" && req != "int64") continue;
    if (!hasFeature(*currentDevice, req)) {
      log(DebugLevel::Debug, "Device %s lacks required feature `%s` for module `%s`; skipping", currentDevice->name().c_str(),
          std::string(req).c_str(), moduleName);
      return false;
    }
  }

  SpecConsts specStorage;
  std::string effectiveModuleName(moduleName);
  if (object.programLength > 0 && object.program && envEnabled(polyregion::env::PolyrtJitSpecialise)) {
    specStorage = collectSpecConsts(capture, interfaceLayout);
    if (!specStorage.values.empty()) {
      const uint64_t specKey = hashSpecs(specStorage.values);
      const auto policyKey = fmt::format("{}|{}|{}|{}|{}", moduleName, currentDevice->name(), magic_enum::enum_name(object.target),
                                         object.arch ? object.arch : "", object.pipelineSpec ? object.pipelineSpec : "");
      polyregion::polyrt::JitPolicyChoice choice{};
      {
        std::lock_guard lock(jitPolicyMutex());
        choice = jitPolicy().select(policyKey, specKey);
      }
      if (choice.specialise) {
        effectiveModuleName += fmt::format("@jit-{:016x}", specKey);
        if (choice.admitted)
          log(DebugLevel::Debug, "JIT admitted hot specialization `%s` as `%s`", moduleName, effectiveModuleName.c_str());
      } else {
        specStorage.values.clear();
      }
    }
  }
  if (loadedModuleName) *loadedModuleName = effectiveModuleName;

  if (!currentDevice->moduleLoaded(effectiveModuleName)) {
    std::string image;
    if (object.imageLength > 0 && object.image) {
      image.assign(reinterpret_cast<const char *>(object.image), object.imageLength);
    } else if (object.programLength > 0 && object.program) {
      if (!jitCompileObject(effectiveModuleName.c_str(), object, specStorage.values, image)) return false;
    } else {
      log(DebugLevel::Debug, "Object for `%s` carries neither a prebuilt image nor a JIT program", moduleName);
      return false;
    }
    if (auto dumpDir = std::getenv(polyregion::env::PolyrtDumpKernel)) {
      static int counter = 0;
      auto path = std::string(dumpDir) + "/kernel_" + std::to_string(counter++) + ".o";
      if (FILE *f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(image.data(), 1, image.size(), f);
        std::fclose(f);
      }
    }
    currentDevice->loadModule(effectiveModuleName, image);
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

POLYREGION_EXPORT extern "C" void *polyrt_record_malloc(const size_t size) {
  void *p = __RT_ALTERNATIVE(malloc)(size);
  if (p) polyregion::rt_reflect::_rt_record(p, size, polyregion::rt_reflect::Type::HeapMalloc);
  return p;
}

POLYREGION_EXPORT extern "C" void polyrt_record_free(void *ptr) {
  if (!ptr) return;
  polyregion::rt_reflect::_rt_release(ptr, polyregion::rt_reflect::Type::HeapFree);
  __RT_ALTERNATIVE(free)(ptr);
}

POLYREGION_EXPORT extern "C" void *polyrt_record_aligned_alloc(const size_t alignment, const size_t size) {
  void *p = __RT_ALTERNATIVE(memalign)(alignment, size);
  if (p) polyregion::rt_reflect::_rt_record(p, size, polyregion::rt_reflect::Type::HeapAlignedAlloc);
  return p;
}

POLYREGION_EXPORT extern "C" void *polyrt_record_operator_new(const size_t size) noexcept(false) {
  void *p = __RT_ALTERNATIVE(malloc)(size);
  if (!p) {
#if __cpp_exceptions == 199711
    throw std::bad_alloc{};
#else
    std::abort();
#endif
  }
  polyregion::rt_reflect::_rt_record(p, size, polyregion::rt_reflect::Type::HeapCXXNew);
  return p;
}

POLYREGION_EXPORT extern "C" void polyrt_record_operator_delete(void *ptr) {
  if (!ptr) return;
  polyregion::rt_reflect::_rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}

POLYREGION_EXPORT extern "C" void polyrt_record_operator_delete_sized(void *ptr, size_t /*size*/) {
  if (!ptr) return;
  polyregion::rt_reflect::_rt_release(ptr, polyregion::rt_reflect::Type::HeapCXXDelete);
  __RT_ALTERNATIVE(free)(ptr);
}

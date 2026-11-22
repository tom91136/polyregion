#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "magic_enum/magic_enum.hpp"

#include "polyinvoke/runtime.h"
#include "polyregion/enums.h"
#include "polyregion/env_keys.h"
#include "polyregion/types.h"
#include "polyrt/mem.hpp"
#include "polyrt/rt.h"
#include "polyrt/sma_runtime.hpp"

#include "generated/context.h"
#include "jni_utils.h"

#ifndef _WIN32
  #include <dlfcn.h>
#endif

using namespace polyregion;
namespace rt = ::invoke;
using namespace generated::registry;

static constexpr const char *EX = "polyregion/jvm/runtime/PolyregionRuntimeException";

static JavaVM *CurrentVM = nullptr;

namespace {

struct HostBlocks {
  std::mutex mutex;
  std::map<uintptr_t, size_t> blocks;
};

HostBlocks &hostBlocks() {
  static auto *blocks = new HostBlocks();
  return *blocks;
}

auto &hostAllocations() {
  static auto *allocations = new polyrt::SynchronisedMemAllocation(
      [](const void *ptr) -> polyrt::PtrQuery {
        auto &host = hostBlocks();
        const std::scoped_lock lock(host.mutex);
        const auto address = reinterpret_cast<uintptr_t>(ptr);
        const auto upper = host.blocks.upper_bound(address);
        if (upper == host.blocks.begin()) return {0, 0};
        const auto &[base, size] = *std::prev(upper);
        return address - base <= size ? polyrt::PtrQuery{.sizeInBytes = size, .offsetInBytes = address - base} : polyrt::PtrQuery{0, 0};
      },
      polyrt::sma::stdRemoteAlloc(), polyrt::sma::stdRemoteRead(), polyrt::sma::stdRemoteWrite(), polyrt::sma::stdRemoteRelease());
  return *allocations;
}

polyrt::sma::CaptureMarshal &hostCaptureMarshal() {
  static auto *marshal = new polyrt::sma::CaptureMarshalOf<std::remove_reference_t<decltype(hostAllocations())>>(hostAllocations());
  return *marshal;
}

} // namespace

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "OnLoad polyrt\n");
  CurrentVM = vm;
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
#ifndef _WIN32
  // host programs loaded by polyinvoke resolve the polyrt entry points from the global scope
  static const bool promoted = [] {
    if (Dl_info info{}; !dladdr(reinterpret_cast<void *>(&JNI_OnLoad), &info) || !info.dli_fname
                        || !dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD)) {
      fprintf(stderr, "polyrt-JNI: cannot promote the library to global scope: %s\n", dlerror());
      return false;
    }
    return true;
  }();
  if (!promoted) return JNI_ERR;
#endif
  if (!Context::registerMethods(env)) return JNI_ERR;
  polyregion::polyrt::sma::registerCaptureMarshal(&hostCaptureMarshal());
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "OnUnload polyrt\n");
  polyregion::polyrt::sma::registerCaptureMarshal(nullptr);
  if (JNIEnv *env = getEnv(vm); env) Context::unregisterMethods(env);
}

namespace {

struct RaisedAssert {
  bool raised = false;
  uint32_t code = 0;
  std::string message;
};
thread_local RaisedAssert raisedAssert;

std::string codeName(const uint32_t code) {
  std::string name;
  for (int shift = 0; shift < 32; shift += 8) {
    const auto c = static_cast<char>((code >> shift) & 0xff);
    if (c < 0x20 || c > 0x7e) {
      char hex[11];
      snprintf(hex, sizeof(hex), "0x%08x", code);
      return hex;
    }
    name.push_back(c);
  }
  return name;
}

jclass loadThroughContextLoader(JNIEnv *env, const std::string &name) {
  jclass result = nullptr;
  [&] {
    auto *threadClass = env->FindClass("java/lang/Thread");
    auto *loaderClass = env->FindClass("java/lang/ClassLoader");
    if (!threadClass || !loaderClass) return;
    auto *current = env->GetStaticMethodID(threadClass, "currentThread", "()Ljava/lang/Thread;");
    auto *contextLoader = env->GetMethodID(threadClass, "getContextClassLoader", "()Ljava/lang/ClassLoader;");
    auto *load = env->GetMethodID(loaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
    if (!current || !contextLoader || !load) return;
    auto *thread = env->CallStaticObjectMethod(threadClass, current);
    if (!thread || env->ExceptionCheck()) return;
    auto *loader = env->CallObjectMethod(thread, contextLoader);
    if (!loader || env->ExceptionCheck()) return;
    auto *jname = env->NewStringUTF(name.c_str());
    if (!jname) return;
    result = static_cast<jclass>(env->CallObjectMethod(loader, load, jname));
  }();
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return nullptr;
  }
  return result;
}

void throwJvmException(JNIEnv *env, const std::string &raised) {
  const auto split = raised.find(": ");
  const auto cls = (split == std::string::npos ? raised : raised.substr(0, split));
  const auto message = split == std::string::npos ? std::string{} : raised.substr(split + 2);
  auto *exClass = loadThroughContextLoader(env, cls);
  if (!exClass) {
    auto internal = cls;
    std::replace(internal.begin(), internal.end(), '.', '/');
    exClass = env->FindClass(internal.c_str());
    if (env->ExceptionCheck()) env->ExceptionClear();
  }
  auto *throwable = env->FindClass("java/lang/Throwable");
  if (exClass && throwable && env->IsAssignableFrom(exClass, throwable) && env->ThrowNew(exClass, message.c_str()) == JNI_OK) return;
  if (env->ExceptionCheck()) env->ExceptionClear();
  throwGeneric(env, EX, "Device raised " + raised);
}

void raiseAssert(JNIEnv *env) {
  const auto raised = std::exchange(raisedAssert, {});
  if (!raised.raised) return;
  if (raised.code == static_cast<uint32_t>(polyregion::invoke::AssertCode::Assert))
    throwGeneric(env, "java/lang/AssertionError", raised.message);
  else if (raised.code == static_cast<uint32_t>(polyregion::invoke::AssertCode::Arithmetic))
    throwGeneric(env, "java/lang/ArithmeticException", raised.message);
  else if (raised.code == static_cast<uint32_t>(polyregion::invoke::AssertCode::JvmException)) throwJvmException(env, raised.message);
  else if (raised.code == static_cast<uint32_t>(polyregion::invoke::AssertCode::Runtime))
    throwGeneric(env, EX, raised.message.empty() ? "unknown runtime failure" : raised.message);
  else throwGeneric(env, EX, "Device assertion failed (code " + codeName(raised.code) + "): " + raised.message);
}

void *toHandle(const jlong handle) { return reinterpret_cast<void *>(static_cast<intptr_t>(handle)); }
jlong fromHandle(void *handle) { return static_cast<jlong>(reinterpret_cast<intptr_t>(handle)); }

void check(const bool ok) {
  if (!ok) throw std::runtime_error(polyrt_error_message());
}

struct HostRuntime {
  std::unique_ptr<rt::Platform> platform;
  std::unique_ptr<rt::Device> device;
  std::unique_ptr<rt::DeviceQueue> queue;
  std::mutex mutex;
};

HostRuntime &hostRuntime() {
  static auto *host = [] {
    auto runtime = std::make_unique<HostRuntime>();
    auto platform = rt::Platform::of(rt::Backend::RelocatableObject);
    if (std::holds_alternative<std::string>(platform))
      throw std::runtime_error("The relocatable object backend failed to initialise: " + std::get<std::string>(platform));
    runtime->platform = std::move(std::get<std::unique_ptr<rt::Platform>>(platform));
    auto devices = runtime->platform->enumerate();
    if (devices.empty()) throw std::runtime_error("The relocatable object backend has no device");
    runtime->device = std::move(devices.front());
    runtime->queue = runtime->device->createQueue(std::chrono::seconds(600));
    return runtime.release();
  }();
  return *host;
}

struct Transaction {
  void *context;
  explicit Transaction(void *context) : context(context) { check(polyrt_context_acquire(context)); }
  Transaction(const Transaction &) = delete;
  Transaction &operator=(const Transaction &) = delete;
  void end() {
    if (const auto ended = std::exchange(context, nullptr); ended) check(polyrt_context_release(ended));
  }
  std::string fail() {
    if (const auto ended = std::exchange(context, nullptr); ended && !polyrt_context_release_failed(ended)) return polyrt_error_message();
    return {};
  }
  ~Transaction() { (void)fail(); }
};

struct HostFailures {
  const bool previous = polyrt_raise_host_failures(true);
  HostFailures() = default;
  HostFailures(const HostFailures &) = delete;
  HostFailures &operator=(const HostFailures &) = delete;
  ~HostFailures() { polyrt_raise_host_failures(previous); }
};

struct ImageSet {
  std::vector<std::string> names;
  std::vector<int32_t> kinds, formats;
  std::vector<std::vector<std::string>> features;
  std::vector<std::vector<const char *>> featurePointers;
  std::vector<std::vector<jbyte>> images;
};

ImageSet readImages(JNIEnv *env, jobjectArray names, jbyteArray targets, jobjectArray features, jobjectArray images) {
  ImageSet set;
  const auto count = env->GetArrayLength(names);
  const auto targetValues = fromJni<jbyte>(env, targets);
  if (static_cast<jsize>(targetValues.size()) != count || env->GetArrayLength(features) != count || env->GetArrayLength(images) != count)
    throw std::runtime_error("The image names, targets, features and images differ in length");
  for (jsize i = 0; i < count; ++i) {
    auto *name = reinterpret_cast<jstring>(env->GetObjectArrayElement(names, i));
    set.names.emplace_back(fromJni(env, name));
    env->DeleteLocalRef(name);

    const auto target =
        magic_enum::enum_cast<compiletime::Target>(static_cast<std::underlying_type_t<compiletime::Target>>(targetValues[i]));
    const auto format = target ? runtime::moduleFormatOf(*target) : std::nullopt;
    if (!target || !format) throw std::runtime_error("Unknown target value: " + std::to_string(targetValues[i]));
    set.kinds.emplace_back(static_cast<int32_t>(runtime::targetPlatformKind(*target)));
    set.formats.emplace_back(static_cast<int32_t>(*format));

    auto *featureArray = reinterpret_cast<jobjectArray>(env->GetObjectArrayElement(features, i));
    auto &strings = set.features.emplace_back();
    for (jsize j = 0, n = env->GetArrayLength(featureArray); j < n; ++j) {
      auto *feature = reinterpret_cast<jstring>(env->GetObjectArrayElement(featureArray, j));
      strings.emplace_back(fromJni(env, feature));
      env->DeleteLocalRef(feature);
    }
    env->DeleteLocalRef(featureArray);

    auto *image = reinterpret_cast<jbyteArray>(env->GetObjectArrayElement(images, i));
    set.images.emplace_back(fromJni<jbyte>(env, image));
    env->DeleteLocalRef(image);
  }
  for (const auto &strings : set.features) {
    auto &pointers = set.featurePointers.emplace_back();
    for (const auto &s : strings)
      pointers.emplace_back(s.c_str());
  }
  return set;
}

void loadImages(void *context, ImageSet &set) {
  for (size_t i = 0; i < set.names.size(); ++i) {
    const auto &name = set.names[i];
    bool settled = false;
    for (size_t j = 0; j < set.names.size() && !settled; ++j)
      if (set.names[j] == name)
        settled =
            polyrt_remote_load(context, name.c_str(), set.kinds[j], set.formats[j], set.features[j].size(), set.featurePointers[j].data(),
                               set.images[j].size(), reinterpret_cast<const uint8_t *>(set.images[j].data()));
    if (!settled) throw std::runtime_error("No image of module `" + name + "` is compatible with the context's device");
  }
}

void runCompletion(void *user) {
  auto *completion = static_cast<jobject>(user);
  JNIEnv *env = nullptr;
  const bool attach = CurrentVM->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_EDETACHED;
  if (attach && CurrentVM->AttachCurrentThreadAsDaemon(reinterpret_cast<void **>(&env), nullptr) != JNI_OK) {
    fprintf(stderr, "polyrt-JNI: cannot attach a completion thread to the JVM, aborting\n");
    std::abort();
  }
  if (auto *run = env->GetMethodID(env->GetObjectClass(completion), "run", "()V"); run) env->CallVoidMethod(completion, run);
  if (env->ExceptionCheck()) env->ExceptionDescribe();
  env->DeleteGlobalRef(completion);
  if (attach) CurrentVM->DetachCurrentThread();
}

void copyAsync(JNIEnv *env, const jlong context, const uintptr_t dst, const uintptr_t src, const jlong bytes, const int32_t direction,
               jobject completion) {
  if (bytes < 0) throw std::runtime_error("Cannot copy " + std::to_string(bytes) + " bytes");
  if (!completion) throw std::runtime_error("The completion is null");
  auto *global = env->NewGlobalRef(completion);
  if (!global) throw std::runtime_error("Cannot create a global reference for the completion");
  if (!polyrt_remote_memcpy_async(toHandle(context), dst, src, static_cast<size_t>(bytes), direction, &runCompletion, global)) {
    env->DeleteGlobalRef(global);
    check(false);
  }
}

std::vector<rt::Type> readTypes(const std::vector<jbyte> &raw) {
  std::vector<rt::Type> types;
  for (const auto t : raw)
    types.emplace_back(static_cast<rt::Type>(t));
  return types;
}

} // namespace

POLYREGION_EXPORT extern "C" void __polyregion_builtin_assert(const uint32_t code, const char *message) { // NOLINT(*-reserved-identifier)
  raisedAssert = {.raised = true, .code = code, .message = message ? message : ""};
}

jint Context::platformCount0(JNIEnv *env, jclass) {
  return wrapBackend(env, EX, [&] {
    int32_t count = 0;
    check(polyrt_platform_count(&count));
    return static_cast<jint>(count);
  });
}

jlongArray Context::platformInfo0(JNIEnv *env, jclass, jint platform) {
  return wrapBackend(env, EX, [&] {
    PolyrtPlatformInfo info{};
    check(polyrt_platform_info(platform, &info));
    const jlong values[] = {info.backend, info.kind, info.deviceCount};
    auto *array = env->NewLongArray(3);
    env->SetLongArrayRegion(array, 0, 3, values);
    return array;
  });
}

jstring Context::platformName0(JNIEnv *env, jclass, jint platform) {
  return wrapBackend(env, EX, [&] {
    PolyrtPlatformInfo info{};
    check(polyrt_platform_info(platform, &info));
    return toJni(env, std::string(info.name, strnlen(info.name, sizeof(info.name))));
  });
}

jlongArray Context::deviceInfo0(JNIEnv *env, jclass, jint platform, jint device) {
  return wrapBackend(env, EX, [&] {
    PolyrtDeviceInfo info{};
    check(polyrt_device_info(platform, device, &info));
    const jlong values[] = {info.backend,
                            info.platformKind,
                            info.moduleFormat,
                            info.deviceKind,
                            info.cudaArchitectureMajor,
                            info.cudaArchitectureMinor,
                            info.systemPaging,
                            static_cast<jlong>(info.globalMemoryBytes),
                            static_cast<jlong>(info.computeUnits)};
    auto *array = env->NewLongArray(9);
    env->SetLongArrayRegion(array, 0, 9, values);
    return array;
  });
}

jstring Context::deviceName0(JNIEnv *env, jclass, jint platform, jint device) {
  return wrapBackend(env, EX, [&] {
    PolyrtDeviceInfo info{};
    check(polyrt_device_info(platform, device, &info));
    return toJni(env, std::string(info.name, strnlen(info.name, sizeof(info.name))));
  });
}

jlong Context::create0(JNIEnv *env, jclass, jint platform, jint device) {
  return wrapBackend(env, EX, [&] {
    void *context = nullptr;
    check(polyrt_context_create(platform, device, &context));
    return fromHandle(context);
  });
}

void Context::destroy0(JNIEnv *env, jclass, jlong context) {
  wrapBackend(env, EX, [&] { check(polyrt_context_destroy(toHandle(context))); });
}

jboolean Context::libjsigLoaded0(JNIEnv *, jclass) { return static_cast<jboolean>(libjsigLoaded()); }

jlong Context::current0(JNIEnv *env, jclass) {
  return wrapException(env, EX, [&] { return fromHandle(polyrt_context_current()); });
}

jlong Context::malloc0(JNIEnv *env, jclass, jlong context, jlong bytes) {
  return wrapException(env, EX, [&] {
    uintptr_t pointer = 0;
    check(polyrt_remote_malloc(toHandle(context), static_cast<size_t>(bytes), &pointer));
    return static_cast<jlong>(pointer);
  });
}

void Context::free0(JNIEnv *env, jclass, jlong context, jlong pointer) {
  wrapException(env, EX, [&] { check(polyrt_remote_free(toHandle(context), static_cast<uintptr_t>(pointer))); });
}

void Context::copyToDevice0(JNIEnv *env, jclass, jlong context, jobject src, jlong dst, jlong bytes) {
  wrapException(env, EX, [&] {
    const auto *local = env->GetDirectBufferAddress(src);
    if (!local) throw std::runtime_error("The source is not a direct buffer");
    if (bytes < 0 || bytes > env->GetDirectBufferCapacity(src))
      throw std::runtime_error("Cannot copy " + std::to_string(bytes) + " bytes from a buffer of "
                               + std::to_string(env->GetDirectBufferCapacity(src)));
    check(polyrt_remote_memcpy(toHandle(context), static_cast<uintptr_t>(dst), reinterpret_cast<uintptr_t>(local),
                               static_cast<size_t>(bytes), 0));
  });
}

void Context::copyFromDevice0(JNIEnv *env, jclass, jlong context, jlong src, jobject dst, jlong bytes) {
  wrapException(env, EX, [&] {
    auto *local = env->GetDirectBufferAddress(dst);
    if (!local) throw std::runtime_error("The destination is not a direct buffer");
    if (bytes < 0 || bytes > env->GetDirectBufferCapacity(dst))
      throw std::runtime_error("Cannot copy " + std::to_string(bytes) + " bytes into a buffer of "
                               + std::to_string(env->GetDirectBufferCapacity(dst)));
    check(polyrt_remote_memcpy(toHandle(context), reinterpret_cast<uintptr_t>(local), static_cast<uintptr_t>(src),
                               static_cast<size_t>(bytes), 1));
  });
}

void Context::copyToDeviceAsync0(JNIEnv *env, jclass, jlong context, jobject src, jlong dst, jlong bytes, jobject completion) {
  wrapException(env, EX, [&] {
    const auto *local = env->GetDirectBufferAddress(src);
    if (!local) throw std::runtime_error("The source is not a direct buffer");
    if (bytes < 0 || bytes > env->GetDirectBufferCapacity(src))
      throw std::runtime_error("Cannot copy " + std::to_string(bytes) + " bytes from a buffer of "
                               + std::to_string(env->GetDirectBufferCapacity(src)));
    copyAsync(env, context, static_cast<uintptr_t>(dst), reinterpret_cast<uintptr_t>(local), bytes, 0, completion);
  });
}

void Context::copyFromDeviceAsync0(JNIEnv *env, jclass, jlong context, jlong src, jobject dst, jlong bytes, jobject completion) {
  wrapException(env, EX, [&] {
    auto *local = env->GetDirectBufferAddress(dst);
    if (!local) throw std::runtime_error("The destination is not a direct buffer");
    if (bytes < 0 || bytes > env->GetDirectBufferCapacity(dst))
      throw std::runtime_error("Cannot copy " + std::to_string(bytes) + " bytes into a buffer of "
                               + std::to_string(env->GetDirectBufferCapacity(dst)));
    copyAsync(env, context, reinterpret_cast<uintptr_t>(local), static_cast<uintptr_t>(src), bytes, 1, completion);
  });
}

jlong Context::address0(JNIEnv *env, jclass, jobject buffer) {
  return wrapException(env, EX, [&] {
    const auto *address = env->GetDirectBufferAddress(buffer);
    if (!address) throw std::runtime_error("The buffer is not a direct buffer");
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(address));
  });
}

void Context::recordHost0(JNIEnv *env, jclass, jlong address, jlong bytes) {
  wrapException(env, EX, [&] {
    auto &host = hostBlocks();
    const std::scoped_lock lock(host.mutex);
    host.blocks[static_cast<uintptr_t>(address)] = static_cast<size_t>(bytes);
  });
}

void Context::releaseHost0(JNIEnv *env, jclass, jlong address) {
  wrapException(env, EX, [&] {
    {
      auto &host = hostBlocks();
      const std::scoped_lock lock(host.mutex);
      host.blocks.erase(static_cast<uintptr_t>(address));
    }
    hostAllocations().disassociate(reinterpret_cast<const void *>(address), /*releaseRemote*/ false);
  });
}

jboolean Context::sharedAddressSpace0(JNIEnv *env, jclass, jlong context) {
  return wrapException(env, EX, [&] { return static_cast<jboolean>(polyrt_device_shared_address_space(toHandle(context))); });
}

jlongArray Context::stats0(JNIEnv *env, jclass, jlong context) {
  return wrapException(env, EX, [&] {
    PolyrtStats stats{};
    check(polyrt_stats(toHandle(context), &stats));
    const jlong values[] = {static_cast<jlong>(stats.launches), static_cast<jlong>(stats.h2dCopies), static_cast<jlong>(stats.d2hCopies),
                            static_cast<jlong>(stats.remoteMallocs), static_cast<jlong>(stats.remoteAllocationsLive)};
    auto *array = env->NewLongArray(5);
    env->SetLongArrayRegion(array, 0, 5, values);
    return array;
  });
}

jint Context::select0(JNIEnv *env, jclass, jlong context, //
                      jobjectArray imageNames, jbyteArray imageTargets, jobjectArray imageFeatures, jobjectArray images) {
  return wrapBackend(env, EX, [&] {
    auto set = readImages(env, imageNames, imageTargets, imageFeatures, images);
    auto *handle = toHandle(context);
    Transaction transaction(handle);
    using polyregion::runtime::LoadOutcome;
    std::optional<jint> selected;
    int32_t worst = static_cast<int32_t>(LoadOutcome::Format);
    for (size_t i = 0; i < set.names.size() && !selected; ++i) {
      const auto outcome = polyrt_remote_load_outcome(handle, set.names[i].c_str(), set.kinds[i], set.formats[i], set.features[i].size(),
                                                      set.featurePointers[i].data(), set.images[i].size(),
                                                      reinterpret_cast<const uint8_t *>(set.images[i].data()));
      if (outcome == static_cast<int32_t>(LoadOutcome::Loaded)) selected = static_cast<jint>(i);
      else worst = std::max(worst, outcome);
    }
    transaction.end();
    return selected.value_or(-worst);
  });
}

void Context::launch0(JNIEnv *env, jclass, jlong context,                                                                //
                      jobjectArray imageNames, jbyteArray imageTargets, jobjectArray imageFeatures, jobjectArray images, //
                      jstring moduleName, jstring kernel, jbyteArray argTypes, jbyteArray argData, jlongArray dims) {
  wrapBackend(env, EX, [&] {
    auto set = readImages(env, imageNames, imageTargets, imageFeatures, images);
    const auto module = fromJni(env, moduleName);
    const auto entry = fromJni(env, kernel);
    const auto types = readTypes(fromJni<jbyte>(env, argTypes));
    auto data = fromJni<jbyte>(env, argData);
    const auto launch = fromJni(env, dims);
    if (launch.size() != 7) throw std::runtime_error("Expected 7 launch dimensions, got " + std::to_string(launch.size()));
    auto bytes = std::vector<std::byte>(reinterpret_cast<const std::byte *>(data.data()),
                                        reinterpret_cast<const std::byte *>(data.data()) + data.size());
    const auto pointers = rt::detail::argDataAsPointers(types, bytes);
    std::vector<uint8_t> rawTypes;
    for (const auto t : types)
      rawTypes.emplace_back(static_cast<uint8_t>(t));

    auto *handle = toHandle(context);
    Transaction transaction(handle);
    loadImages(handle, set);
    raisedAssert = {};
    check(polyrt_remote_launch(handle, module.c_str(), entry.c_str(), launch[0], launch[1], launch[2], launch[3], launch[4], launch[5],
                               launch[6], rawTypes.size(), rawTypes.data(), pointers.data()));
    transaction.end();
    raiseAssert(env);
  });
}

void Context::loadHostProgram0(JNIEnv *env, jclass, jstring name, jbyteArray image) {
  wrapBackend(env, EX, [&] {
    auto &host = hostRuntime();
    const auto moduleName = fromJni(env, name);
    const auto data = fromJni<jbyte>(env, image);
    const std::scoped_lock lock(host.mutex);
    if (host.device->moduleLoaded(moduleName)) throw std::runtime_error("Host program `" + moduleName + "` is already loaded");
    host.device->loadModule(moduleName, std::string(reinterpret_cast<const char *>(data.data()), data.size()));
  });
}

jboolean Context::hostProgramLoaded0(JNIEnv *env, jclass, jstring name) {
  return wrapBackend(env, EX, [&] {
    auto &host = hostRuntime();
    const std::scoped_lock lock(host.mutex);
    return static_cast<jboolean>(host.device->moduleLoaded(fromJni(env, name)));
  });
}

void Context::callHost0(JNIEnv *env, jclass, jlong context,                                                                //
                        jobjectArray imageNames, jbyteArray imageTargets, jobjectArray imageFeatures, jobjectArray images, //
                        jstring programName, jstring entry, jbyteArray argTypes, jbyteArray argData) {
  wrapBackend(env, EX, [&] {
    auto set = readImages(env, imageNames, imageTargets, imageFeatures, images);
    const auto program = fromJni(env, programName);
    const auto symbol = fromJni(env, entry);
    auto types = readTypes(fromJni<jbyte>(env, argTypes));
    const auto data = fromJni<jbyte>(env, argData);

    auto *handle = toHandle(context);
    types.insert(types.begin(), rt::Type::Ptr);
    types.emplace_back(rt::Type::Void);
    std::vector<std::byte> bytes(sizeof(void *));
    std::memcpy(bytes.data(), &handle, sizeof(void *));
    bytes.insert(bytes.end(), reinterpret_cast<const std::byte *>(data.data()),
                 reinterpret_cast<const std::byte *>(data.data()) + data.size());

    auto &host = hostRuntime();
    {
      const std::scoped_lock lock(host.mutex);
      if (!host.device->moduleLoaded(program)) throw std::runtime_error("Host program `" + program + "` is not loaded");
    }
    Transaction transaction(handle);
    loadImages(handle, set);
    raisedAssert = {};
    const HostFailures raising;
    if (!host.queue->invokeEntryBlocking(program, symbol, types, std::move(bytes)))
      throw std::runtime_error("Host program `" + program + "` has no entry `" + symbol + "`");
    if (raisedAssert.raised) {
      if (const auto released = transaction.fail(); !released.empty()) raisedAssert.message += "; release after failure: " + released;
    } else transaction.end();
    raiseAssert(env);
  });
}

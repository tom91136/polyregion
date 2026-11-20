#include "polyinvoke/runtime.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "aspartame/optional.hpp"
#include "aspartame/vector.hpp"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/env_keys.h"

#include "generated/mirror.h"
#include "generated/platform.h"
#include "generated/platforms.h"
#include "jni_utils.h"

using namespace aspartame;
using namespace polyregion;
namespace rt = ::invoke;
namespace gen = ::generated;
using namespace gen::registry;

static constexpr const char *EX = "polyregion/jvm/runtime/PolyregionRuntimeException";

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "OnLoad runtime\n");
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  rt::init();
  if (!Platforms::registerMethods(env) || !Platform::registerMethods(env)) return JNI_ERR;
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  if (const char *d = std::getenv(env::PolyregionDebug); d && *d) fprintf(stderr, "OnUnload runtime\n");
  JNIEnv *env = getEnv(vm);
  gen::Platform::drop(env);
  gen::Property::drop(env);
  gen::Device::drop(env);
  gen::String::drop(env);
  Platforms::unregisterMethods(env);
  Platform::unregisterMethods(env);
}

static std::atomic<jlong> peerCounter = 0;
static std::unordered_map<jlong, std::shared_ptr<rt::Platform>> platforms;
static std::unordered_map<jlong, std::shared_ptr<rt::Device>> devices;
static std::unordered_map<jlong, std::vector<std::unique_ptr<std::string>>> deviceModuleImages;
static std::mutex lock;

template <typename T, typename U> static auto emplaceRef(std::unordered_map<jlong, std::shared_ptr<T>> &storage, std::shared_ptr<U> x) {
  auto i = peerCounter++;
  const std::scoped_lock l(lock);
  auto &&[it, _] = storage.emplace(i, x);
  return *it;
}

template <typename T>
static std::shared_ptr<T> findRef(JNIEnv *env, std::unordered_map<jlong, std::shared_ptr<T>> &storage, jlong nativePeer) {
  if (const auto it = storage.find(nativePeer); it != storage.end()) return it->second;
  else return throwGeneric(env, EX, "Cannot find native peer (" + std::to_string(nativePeer) + ") ");
}

static jobjectArray toJni(JNIEnv *env, const std::vector<rt::Property> &xs) {
  const auto &property = gen::Property::of(env);
  return toJni(env, xs, property.clazz, [&](const auto &x) { return property(env, toJni(env, x.first), toJni(env, x.second)).instance; });
}

void Platforms::deleteAllPeers0(JNIEnv *, jclass) {
  const std::scoped_lock l(lock);
  devices.clear();
  deviceModuleImages.clear();
  platforms.clear();
}

void Platform::deletePlatformPeer0(JNIEnv *, jclass, jlong nativePeer) {
  const std::scoped_lock l(lock);
  platforms.erase(nativePeer);
}
void Platform::deleteDevicePeer0(JNIEnv *, jclass, jlong nativePeer) {
  const std::scoped_lock l(lock);
  devices.erase(nativePeer);
  deviceModuleImages.erase(nativePeer);
}

jlongArray Platforms::pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers) {
  const jsize n = env->GetArrayLength(buffers);
  std::vector<jlong> ptrs(n);
  for (jsize i = 0; i < n; ++i) {
    auto *buf = env->GetObjectArrayElement(buffers, i);
    const void *ptr = env->GetDirectBufferAddress(buf);
    env->DeleteLocalRef(buf);
    if (!ptr) return throwGeneric(env, EX, "Object at " + std::to_string(i) + " is either not a direct Buffer or not a Buffer at all.");
    ptrs[i] = reinterpret_cast<jlong>(ptr);
  }
  auto *array = env->NewLongArray(n);
  env->SetLongArrayRegion(array, 0, n, ptrs.data());
  return array;
}

jlong Platforms::pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer) {
  if (const void *ptr = env->GetDirectBufferAddress(buffer); ptr) return reinterpret_cast<jlong>(ptr);
  else return throwGeneric<jlong>(env, EX, "Object is either not a direct Buffer or not a Buffer at all.");
}

static jobject toJni(JNIEnv *env, rt::Backend backend) {
  return wrapBackend(env, EX, [&] {
    if (auto errorOrPlatform = rt::Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
      throw std::runtime_error("Backend " + std::string(magic_enum::enum_name(backend))
                               + " failed to initialise: " + std::get<std::string>(errorOrPlatform));
    } else {
      auto [peer, platform] =
          emplaceRef(platforms, std::shared_ptr<rt::Platform>(std::move(std::get<std::unique_ptr<rt::Platform>>(errorOrPlatform))));
      return gen::Platform::of(env)(env, peer, toJni(env, platform->name())).instance;
    }
  });
}

jobject Platforms::CUDA0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::CUDA); }
jobject Platforms::HIP0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::HIP); }
jobject Platforms::HSA0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::HSA); }
jobject Platforms::OpenCL0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::OpenCL); }
jobject Platforms::LevelZero0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::LevelZero); }
jobject Platforms::Vulkan0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::Vulkan); }
// NOLINTNEXTLINE(misc-const-correctness)
jobject Platforms::Metal0([[maybe_unused]] JNIEnv *env, jclass) {
#ifdef RUNTIME_ENABLE_METAL
  return toJni(env, rt::Backend::Metal);
#else

  return nullptr;
#endif
}
jobject Platforms::Relocatable0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::RelocatableObject); }
jobject Platforms::Dynamic0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::SharedObject); }
jobject Platforms::directBufferFromPointer0(JNIEnv *env, jclass, jlong ptr, jlong size) {
  return env->NewDirectByteBuffer(reinterpret_cast<void *>(ptr), size);
}
jobjectArray Platform::runtimeProperties0(JNIEnv *env, jclass, jlong nativePeer) {
  return toJni(env, findRef(env, platforms, nativePeer)->properties());
}
jobjectArray Platform::devices0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapBackend(env, EX, [&] {
    auto xs = findRef(env, platforms, nativePeer)->enumerate();
    const auto &deviceMirror = gen::Device::of(env);
    return toJni(env, xs, deviceMirror.clazz, [&](auto &device) {
      auto [peer, d] = emplaceRef(devices, std::shared_ptr(std::move(device)));
      return deviceMirror(env, peer, d->id(), toJni(env, d->name()), d->sharedAddressSpace()).instance;
    });
  });
}

jobjectArray Platform::deviceProperties0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&] { return toJni(env, findRef(env, devices, nativePeer)->properties()); });
}

jobjectArray Platform::deviceFeatures0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&] {
    auto xs = findRef(env, devices, nativePeer)->features();
    return toJni(env, xs, gen::String::of(env).clazz, [&](const auto &x) { return toJni(env, x); });
  });
}

void Platform::loadModule0(JNIEnv *env, jclass, jlong nativePeer, jstring name, jbyteArray image) {

  wrapBackend(env, EX, [&] {
    const auto dev = findRef(env, devices, nativePeer);
    // We need to hold on to our copied image here before passing it on, this is later destroyed with the device.
    auto data = std::make_unique<std::string>(env->GetArrayLength(image), '\0');
    env->GetByteArrayRegion(image, 0, env->GetArrayLength(image), reinterpret_cast<jbyte *>(data->data()));
    const std::scoped_lock l(lock);
    auto &&[it, _] = deviceModuleImages.try_emplace(nativePeer);
    it->second.push_back(std::move(data));
    dev->loadModule(fromJni(env, name), *it->second.back());
  });
}

jboolean Platform::moduleLoaded0(JNIEnv *env, jclass, jlong nativePeer, jstring name) {
  return wrapException(env, EX, [&] { return findRef(env, devices, nativePeer)->moduleLoaded(fromJni(env, name)); });
}

jlong Platform::malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size, jbyte access) {
  if (auto a = rt::from_underlying<rt::Access>(access); a) {
    return wrapException(env, EX, [&] { return static_cast<jlong>(findRef(env, devices, nativePeer)->mallocDevice(size, *a)); });
  } else return throwGeneric<jlong>(env, EX, "Illegal access type " + std::to_string(access));
}

void Platform::free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle) {
  wrapException(env, EX, [&] { findRef(env, devices, nativePeer)->freeDevice(handle); });
}

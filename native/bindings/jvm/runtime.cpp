#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "aspartame/optional.hpp"
#include "aspartame/vector.hpp"
#include "generated/mirror.h"
#include "generated/platform.h"
#include "generated/platforms.h"
#include "jni_utils.h"
#include "polyinvoke/runtime.h"

using namespace polyregion;
namespace rt = ::invoke;
namespace gen = ::generated;
using namespace gen::registry;

static constexpr const char *EX = "polyregion/jvm/runtime/PolyregionRuntimeException";

static JavaVM *CurrentVM;

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  fprintf(stderr, "OnLoad runtime\n");
  CurrentVM = vm;
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  rt::init();
  Platforms::registerMethods(env);
  Platform::registerMethods(env);
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  fprintf(stderr, "OnUnload runtime\n");
  JNIEnv *env = getEnv(vm);
  gen::Platform::drop(env);
  gen::Property::drop(env);
  gen::Device::drop(env);
  gen::Queue::drop(env);
  gen::Runnable::drop(env);
  gen::Policy::drop(env);
  gen::Dim3::drop(env);
  gen::String::drop(env);
  Platforms::unregisterMethods(env);
  Platform::unregisterMethods(env);
}

static std::atomic<jlong> peerCounter = 0;
static std::unordered_map<jlong, std::shared_ptr<rt::Platform>> platforms;
static std::unordered_map<jlong, std::shared_ptr<rt::Device>> devices;
static std::unordered_map<jlong, std::vector<std::unique_ptr<std::string>>> deviceModuleImages;
static std::unordered_map<jlong, std::shared_ptr<rt::DeviceQueue>> deviceQueues;
static std::mutex lock;

template <typename T, typename U> static auto emplaceRef(std::unordered_map<jlong, std::shared_ptr<T>> &storage, std::shared_ptr<U> x) {
  auto i = peerCounter++;
  std::lock_guard l(lock);
  auto &&[it, _] = storage.emplace(i, x);
  return *it;
}

template <typename T>
static std::shared_ptr<T> findRef(JNIEnv *env, std::unordered_map<jlong, std::shared_ptr<T>> &storage, jlong nativePeer) {
  if (auto it = storage.find(nativePeer); it != storage.end()) return std::reinterpret_pointer_cast<T>(it->second);
  else return throwGeneric(env, EX, "Cannot find native peer (" + std::to_string(nativePeer) + ") ");
}

static jobjectArray toJni(JNIEnv *env, const std::vector<rt::Property> &xs) {
  return toJni(env, xs, gen::Property::of(env).clazz,
               [&](auto &x) { return gen::Property::of(env)(env, toJni(env, x.first), toJni(env, x.second)).instance; });
}

void Platforms::deleteAllPeers0(JNIEnv *env, jclass) {
  std::lock_guard l(lock);
  deviceQueues.clear();
  devices.clear();
  deviceModuleImages.clear();
  platforms.clear();
}

void Platform::deletePlatformPeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  platforms.erase(nativePeer);
}
void Platform::deleteQueuePeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  deviceQueues.erase(nativePeer);
}
void Platform::deleteDevicePeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  devices.erase(nativePeer);
  deviceModuleImages.erase(nativePeer);
}

jlongArray Platforms::pointerOfDirectBuffers0(JNIEnv *env, jclass, jobjectArray buffers) {
  jsize n = env->GetArrayLength(buffers);
  auto array = env->NewLongArray(n);
  auto ptrs = env->GetLongArrayElements(array, nullptr);
  for (jsize i = 0; i < n; ++i) {
    if (auto ptr = env->GetDirectBufferAddress(env->GetObjectArrayElement(buffers, i)); ptr) ptrs[i] = reinterpret_cast<jlong>(ptr);
    else return throwGeneric(env, EX, "Object at " + std::to_string(i) + " is either not a direct Buffer or not a Buffer at all.");
  }
  env->ReleaseLongArrayElements(array, ptrs, 0);
  return array;
}

jlong Platforms::pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer) {
  if (auto ptr = env->GetDirectBufferAddress(buffer); ptr) return reinterpret_cast<jlong>(ptr);
  else return throwGeneric<jlong>(env, EX, "Object is either not a direct Buffer or not a Buffer at all.");
}

static jobject toJni(JNIEnv *env, rt::Backend backend) {
  return wrapException(env, EX, [&]() {
    if (auto errorOrPlatform = rt::Platform::of(backend); std::holds_alternative<std::string>(errorOrPlatform)) {
      throw std::runtime_error("Backend " + std::string(to_string(backend)) +
                               " failed to initialise: " + std::get<std::string>(errorOrPlatform));
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
jobject Platforms::Vulkan0(JNIEnv *env, jclass) { return toJni(env, rt::Backend::Vulkan); }
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
  return wrapException(env, EX, [&]() {
    auto xs = findRef(env, platforms, nativePeer)->enumerate();
    return toJni(env, xs, gen::Device::of(env).clazz, [&](auto &device) {
      auto [peer, d] = emplaceRef(devices, std::shared_ptr(std::move(device)));
      return gen::Device::of(env)(env, peer, d->id(), toJni(env, d->name()), d->sharedAddressSpace()).instance;
    });
  });
}

jobjectArray Platform::deviceProperties0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&]() { return toJni(env, findRef(env, devices, nativePeer)->properties()); });
}

jobjectArray Platform::deviceFeatures0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&]() {
    auto xs = findRef(env, devices, nativePeer)->features();
    return toJni(env, xs, gen::String::of(env).clazz, [&](auto &x) { return toJni(env, x); });
  });
}

void Platform::loadModule0(JNIEnv *env, jclass, jlong nativePeer, jstring name, jbyteArray image) {

  wrapException(env, EX, [&]() {
    auto dev = findRef(env, devices, nativePeer);
    // We need to hold on to our copied image here before passing it on, this is later destroyed with the device.
    auto data = std::make_unique<std::string>(env->GetArrayLength(image), '\0');
    env->GetByteArrayRegion(image, 0, env->GetArrayLength(image), reinterpret_cast<jbyte *>(data->data()));
    std::lock_guard l(lock);
    auto &&[it, _] = deviceModuleImages.try_emplace(nativePeer, decltype(deviceModuleImages)::mapped_type{});
    it->second.push_back(std::move(data));
    dev->loadModule(fromJni(env, name), *it->second.back());
  });
}

jboolean Platform::moduleLoaded0(JNIEnv *env, jclass, jlong nativePeer, jstring name) {
  return wrapException(env, EX, [&]() { return findRef(env, devices, nativePeer)->moduleLoaded(fromJni(env, name)); });
}

jlong Platform::malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size, jbyte access) {
  if (auto a = rt::from_underlying<rt::Access>(access); a) {
    return wrapException(env, EX, [&]() { return static_cast<jlong>(findRef(env, devices, nativePeer)->mallocDevice(size, *a)); });
  } else return throwGeneric<jlong>(env, EX, "Illegal access type " + std::to_string(access));
}

void Platform::free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle) {
  wrapException(env, EX, [&]() { findRef(env, devices, nativePeer)->freeDevice(static_cast<jlong>(handle)); });
}

jobject Platform::createQueue0(JNIEnv *env, jclass, jlong nativePeer, jobject device, jlong timeoutMillis) {
  return wrapException(env, EX, [&]() {
    auto queue = findRef(env, devices, nativePeer)
                     ->createQueue(std::chrono::duration_cast<std::chrono::duration<int64_t>>(std::chrono::milliseconds(timeoutMillis)));
    auto [peer, _] = emplaceRef(deviceQueues, std::shared_ptr(std::move(queue)));
    return gen::Queue::of(env)(env, peer, device).instance;
  });
}

static rt::MaybeCallback fromJni(JNIEnv *env, jobject cb) {
  return !cb ? std::nullopt : std::make_optional([cbRef = env->NewGlobalRef(cb)]() {
    JNIEnv *attachedEnv{};
    if (CurrentVM->AttachCurrentThread(reinterpret_cast<void **>(&attachedEnv), nullptr) != JNI_OK) {
      // Can't attach here so just send the error to stderr.
      fprintf(stderr, "Unable to attach thread <%zx> to JVM from a callback passed to enqueueInvokeAsync\n",
              std::hash<std::thread::id>()(std::this_thread::get_id()));
    }
    if (!cbRef) throwGeneric(attachedEnv, EX, "Unable to retrieve reference to the callback passed to enqueueInvokeAsync");
    else {
      fprintf(stderr, "JNI thread attached\n");
      gen::Runnable::of(attachedEnv).wrap(attachedEnv, cbRef).run(attachedEnv);
      if (attachedEnv->ExceptionCheck()) attachedEnv->ExceptionClear();
      attachedEnv->DeleteGlobalRef(cbRef);
      gen::Runnable::drop(attachedEnv);
    }

    CurrentVM->DetachCurrentThread();
  });
}

void Platform::enqueueHostToDeviceAsync0(JNIEnv *env, jclass, //
                                         jlong nativePeer,    //
                                         jobject src, jlong dst, jint size, jobject cb) {
  auto srcPtr = env->GetDirectBufferAddress(src);
  if (!srcPtr) throwGeneric(env, EX, "The source ByteBuffer is not backed by an direct allocation.");

  return wrapException(env, EX,
                       [&]() { findRef(env, deviceQueues, nativePeer)->enqueueHostToDeviceAsync(srcPtr, dst, 0, size, fromJni(env, cb)); });
}
void Platform::enqueueDeviceToHostAsync0(JNIEnv *env, jclass, //
                                         jlong nativePeer,    //
                                         jlong src, jobject dst, jint size, jobject cb) {
  auto dstPtr = env->GetDirectBufferAddress(dst);
  if (!dstPtr) throwGeneric(env, EX, "The destination ByteBuffer is not backed by an direct allocation.");
  return wrapException(env, EX,
                       [&]() { findRef(env, deviceQueues, nativePeer)->enqueueDeviceToHostAsync(src, 0, dstPtr, size, fromJni(env, cb)); });
}

static rt::Dim3 fromJni(JNIEnv *env, const generated::Dim3::Instance &d3) {
  return {static_cast<size_t>(d3.x(env)), static_cast<size_t>(d3.y(env)), static_cast<size_t>(d3.z(env))};
}

void Platform::enqueueInvokeAsync0(JNIEnv *env, jclass, jlong nativePeer, //
                                   jstring moduleName, jstring symbol,    //
                                   jbyteArray argTypes,                   //
                                   jbyteArray argData,                    //
                                   jobject policy, jobject cb) {

  auto argCount = env->GetArrayLength(argTypes);
  if (argCount == 0) {
    throwGeneric(env, EX, "empty argTypes, expecting at >= 1 for return type");
    return;
  }

  wrapException(env, EX, [&]() {
    static_assert(sizeof(jbyte) == sizeof(std::byte));
    static_assert(sizeof(jbyte) == sizeof(rt::Type));

    using namespace aspartame;
    auto argTs = fromJni<jbyte>(env, argTypes) ^ map([](auto &t) { return static_cast<rt::Type>(t); });
    auto argPs = fromJni<jbyte>(env, argData) ^ map([](auto &t) { return static_cast<std::byte>(t); });

    auto p = gen::Policy::of(env).wrap(env, policy);
    auto global = fromJni(env, p.global(env, gen::Dim3::of(env)));
    auto localMemoryBytes = p.localMemoryBytes(env);
    auto local = p.local(env, gen::Dim3::of(env)).map<rt::Dim3>([&](auto x) { return fromJni(env, x); });
    if (!local && localMemoryBytes) {
      throwGeneric(env, EX, "Launch configured with no local dims but with local memory");
    }

    findRef(env, deviceQueues, nativePeer)
        ->enqueueInvokeAsync(fromJni(env, moduleName), fromJni(env, symbol), argTs, argPs,
                             rt::Policy{global, {(local ^ map([&](auto &&d) { return std::make_pair(d, localMemoryBytes); }))}},
                             fromJni(env, cb));

    if (argTs[argCount - 1] == rt::Type::Ptr) {
      // we got four possible cases when a function return pointers:
      //  1. Pointer to one of the argument   (LUT[ptr]==Some) => passthrough
      //  2. Pointer to malloc'd memory       (LUT[ptr]==Some) => passthrough
      //  3. Pointer within a malloc'd region (LUT[ptr]==None) => copy
      //  4. Pointer to stack allocated data  (LUT[ptr]==None) => undefined, should not happen

      auto args = argTs ^ mk_string(",", [](auto &tpe) { return std::string(to_string(tpe)); });
      throwGeneric(env, EX, "Returning pointers is unimplemented, args (return at the end): " + args);
      //      std::unordered_map<void *, std::pair<jobject, jsize>> allocations;
      //
      //      // save all argPs in the alloc LUT first so that we can identify them later
      //      for (jsize i = 0; i < env->GetArrayLength(argPtrs); ++i) {
      //        auto buffer = env->GetObjectArrayElement(argPtrs, i);
      //        allocations.emplace(env->GetDirectBufferAddress(buffer),
      //                            std::make_pair(buffer, env->GetDirectBufferCapacity(buffer)));
      //      }
      //
      //      //    // make our allocator store it as well
      //      //    auto allocator = [&](size_t size) {
      //      //      //      std::cerr << "[runtime][obj rtn] Allocating " << size << " bytes" << std::endl;
      //      //      auto buffer = NIOBuffer(env, allocDirect(env, jint(size)));
      //      //      allocations[buffer.ptr] = buffer;
      //      //      return buffer.ptr;
      //      //    };
      //
      //      if (auto r = allocations.find(rtnPtrStore); r != allocations.end()) {
      //        auto buffer = r->second; // we found the original allocation, passthrough-return
      //
      //          if (env->ExceptionCheck()) {
      //            env->Throw(env->ExceptionOccurred());
      //          }
      //          return buffer.first;
      //
      //      } else {
      //        if (rtnBytes < 0) {
      //          throwGeneric(env, "Bad size (" + std::to_string(rtnBytes) + ") for copy buffer");
      //          return nullptr;
      //        } else {
      //          auto buffer = allocDirect(env, rtnBytes);
      //          std::memcpy(env->GetDirectBufferAddress(buffer), rtnData, rtnBytes);
      //          return buffer;
      //        }
      //      }
    }
  });
}

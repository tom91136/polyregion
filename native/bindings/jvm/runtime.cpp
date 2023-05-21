#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cl_platform.h"
#include "cuda_platform.h"
#include "generated/mirror.h"
#include "generated/platform.h"
#include "generated/platforms.h"
#include "hip_platform.h"
#include "hsa_platform.h"
#include "vulkan_platform.h"
#include "metal_platform.h"
#include "jni_utils.h"
#include "object_platform.h"
#include "utils.hpp"

using namespace polyregion;
namespace rt = ::runtime;
namespace gen = ::generated;
using namespace gen::registry;

static constexpr const char *EX = "polyregion/jvm/runtime/PolyregionRuntimeException";

static_assert(polyregion::to_underlying(rt::Type::Void) == Platforms::TYPE_VOID);
static_assert(polyregion::to_underlying(rt::Type::Bool8) == Platforms::TYPE_BOOL);
static_assert(polyregion::to_underlying(rt::Type::Byte8) == Platforms::TYPE_BYTE);
static_assert(polyregion::to_underlying(rt::Type::CharU16) == Platforms::TYPE_CHAR);
static_assert(polyregion::to_underlying(rt::Type::Short16) == Platforms::TYPE_SHORT);
static_assert(polyregion::to_underlying(rt::Type::Int32) == Platforms::TYPE_INT);
static_assert(polyregion::to_underlying(rt::Type::Long64) == Platforms::TYPE_LONG);
static_assert(polyregion::to_underlying(rt::Type::Float32) == Platforms::TYPE_FLOAT);
static_assert(polyregion::to_underlying(rt::Type::Double64) == Platforms::TYPE_DOUBLE);
static_assert(polyregion::to_underlying(rt::Type::Ptr) == Platforms::TYPE_PTR);

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

template <typename T, typename U>
static auto emplaceRef(std::unordered_map<jlong, std::shared_ptr<T>> &storage, std::shared_ptr<U> x) {
  auto i = peerCounter++;
  std::lock_guard l(lock);
  auto &&[it, _] = storage.emplace(i, x);
  return *it;
}

template <typename T>
static std::shared_ptr<T> findRef(JNIEnv *env, std::unordered_map<jlong, std::shared_ptr<T>> &storage,
                                  jlong nativePeer) {
  if (auto it = storage.find(nativePeer); it != storage.end()) return reinterpret_pointer_cast<T>(it->second);
  else
    return throwGeneric(env, EX, "Cannot find native peer (" + std::to_string(nativePeer) + ") ");
}

static jobjectArray toJni(JNIEnv *env, const std::vector<rt::Property> &xs) {
  return toJni(env, xs, gen::Property::of(env).clazz, [&](auto &x) {
    return gen::Property::of(env)(env, toJni(env, x.first), toJni(env, x.second)).instance;
  });
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
    if (auto ptr = env->GetDirectBufferAddress(env->GetObjectArrayElement(buffers, i)); ptr)
      ptrs[i] = reinterpret_cast<jlong>(ptr);
    else
      return throwGeneric(env, EX,
                          "Object at " + std::to_string(i) + " is either not a direct Buffer or not a Buffer at all.");
  }
  env->ReleaseLongArrayElements(array, ptrs, 0);
  return array;
}

jlong Platforms::pointerOfDirectBuffer0(JNIEnv *env, jclass, jobject buffer) {
  if (auto ptr = env->GetDirectBufferAddress(buffer); ptr) return reinterpret_cast<jlong>(ptr);
  else
    return throwGeneric<jlong>(env, EX, "Object is either not a direct Buffer or not a Buffer at all.");
}

template <typename R> static jobject toJni(JNIEnv *env) {
  return wrapException(env, EX, [&]() {
    auto [peer, platform] = emplaceRef(platforms, std::make_shared<R>());
    return gen::Platform::of(env)(env, peer, toJni(env, platform->name())).instance;
  });
}

jobject Platforms::CUDA0(JNIEnv *env, jclass) { return toJni<rt::cuda::CudaPlatform>(env); }
jobject Platforms::HIP0(JNIEnv *env, jclass) { return toJni<rt::hip::HipPlatform>(env); }
jobject Platforms::HSA0(JNIEnv *env, jclass) { return toJni<rt::hsa::HsaPlatform>(env); }
jobject Platforms::OpenCL0(JNIEnv *env, jclass) { return toJni<rt::cl::ClPlatform>(env); }
jobject Platforms::Vulkan0(JNIEnv *env, jclass) { return toJni<rt::vulkan::VulkanPlatform>(env); }
jobject Platforms::Metal0(JNIEnv *env, jclass) {
#ifdef RUNTIME_ENABLE_METAL
  return toJni<rt::metal::MetalPlatform>(env);
#else
  return nullptr;
#endif
}
jobject Platforms::Relocatable0(JNIEnv *env, jclass) { return toJni<rt::object::RelocatablePlatform>(env); }
jobject Platforms::Dynamic0(JNIEnv *env, jclass) { return toJni<rt::object::SharedPlatform>(env); }
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
  if (auto a = rt::fromUnderlying(access); a) {
    return wrapException(env, EX,
                         [&]() { return static_cast<jlong>(findRef(env, devices, nativePeer)->malloc(size, *a)); });
  } else
    return throwGeneric<jlong>(env, EX, "Illegal access type " + std::to_string(access));
}

void Platform::free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle) {
  wrapException(env, EX, [&]() { findRef(env, devices, nativePeer)->free(static_cast<jlong>(handle)); });
}

jobject Platform::createQueue0(JNIEnv *env, jclass, jlong nativePeer, jobject device) {
  return wrapException(env, EX, [&]() {
    auto queue = findRef(env, devices, nativePeer)->createQueue();
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
    if (!cbRef)
      throwGeneric(attachedEnv, EX, "Unable to retrieve reference to the callback passed to enqueueInvokeAsync");
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

  return wrapException(env, EX, [&]() {
    findRef(env, deviceQueues, nativePeer)->enqueueHostToDeviceAsync(srcPtr, dst, size, fromJni(env, cb));
  });
}
void Platform::enqueueDeviceToHostAsync0(JNIEnv *env, jclass, //
                                         jlong nativePeer,    //
                                         jlong src, jobject dst, jint size, jobject cb) {
  auto dstPtr = env->GetDirectBufferAddress(dst);
  if (!dstPtr) throwGeneric(env, EX, "The destination ByteBuffer is not backed by an direct allocation.");
  return wrapException(env, EX, [&]() {
    findRef(env, deviceQueues, nativePeer)->enqueueDeviceToHostAsync(src, dstPtr, size, fromJni(env, cb));
  });
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

    auto argTs =
        map_vec<jbyte, rt::Type>(fromJni<jbyte>(env, argTypes), [](auto &t) { return static_cast<rt::Type>(t); });
    auto argPs =
        map_vec<jbyte, std::byte>(fromJni<jbyte>(env, argData), [](auto &t) { return static_cast<std::byte>(t); });

    auto p = gen::Policy::of(env).wrap(env, policy);
    auto global = fromJni(env, p.global(env, gen::Dim3::of(env)));
    auto localMemoryBytes = p.localMemoryBytes(env);
    auto local = p.local(env, gen::Dim3::of(env)).map<rt::Dim3>([&](auto x) { return fromJni(env, x); });
    if (!local && localMemoryBytes) {
      throwGeneric(env, EX, "Launch configured with no local dims but with local memory");
    }

    findRef(env, deviceQueues, nativePeer)
        ->enqueueInvokeAsync(
            fromJni(env, moduleName), fromJni(env, symbol), argTs, argPs,
            rt::Policy{global, {map_opt(local, [&](auto &&d) { return std::make_pair(d, localMemoryBytes); })}},
            fromJni(env, cb));

    if (argTs[argCount - 1] == rt::Type::Ptr) {
      // we got four possible cases when a function return pointers:
      //  1. Pointer to one of the argument   (LUT[ptr]==Some) => passthrough
      //  2. Pointer to malloc'd memory       (LUT[ptr]==Some) => passthrough
      //  3. Pointer within a malloc'd region (LUT[ptr]==None) => copy
      //  4. Pointer to stack allocated data  (LUT[ptr]==None) => undefined, should not happen

      auto args = mk_string<rt::Type>(
          argTs, [](auto &tpe) { return typeName(tpe); }, ",");
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

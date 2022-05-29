#include <cassert>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cl_runtime.h"
#include "cuda_runtime.h"
#include "hip_runtime.h"
#include "jni_utils.h"
#include "mirror.h"
#include "object_runtime.h"
#include "polyregion_jvm_runtime_Runtime.h"
#include "runtime.h"
#include "utils.hpp"
using namespace polyregion;
namespace rt = ::runtime;
namespace gen = ::generated;

static constexpr const char *EX = "polyregion/PolyregionRuntimeException";

static_assert(polyregion::to_underlying(rt::Type::Void) == polyregion_jvm_runtime_Runtime_TYPE_1VOID);
static_assert(polyregion::to_underlying(rt::Type::Bool8) == polyregion_jvm_runtime_Runtime_TYPE_1BOOL);
static_assert(polyregion::to_underlying(rt::Type::Byte8) == polyregion_jvm_runtime_Runtime_TYPE_1BYTE);
static_assert(polyregion::to_underlying(rt::Type::CharU16) == polyregion_jvm_runtime_Runtime_TYPE_1CHAR);
static_assert(polyregion::to_underlying(rt::Type::Short16) == polyregion_jvm_runtime_Runtime_TYPE_1SHORT);
static_assert(polyregion::to_underlying(rt::Type::Int32) == polyregion_jvm_runtime_Runtime_TYPE_1INT);
static_assert(polyregion::to_underlying(rt::Type::Long64) == polyregion_jvm_runtime_Runtime_TYPE_1LONG);
static_assert(polyregion::to_underlying(rt::Type::Float32) == polyregion_jvm_runtime_Runtime_TYPE_1FLOAT);
static_assert(polyregion::to_underlying(rt::Type::Double64) == polyregion_jvm_runtime_Runtime_TYPE_1DOUBLE);
static_assert(polyregion::to_underlying(rt::Type::Ptr) == polyregion_jvm_runtime_Runtime_TYPE_1PTR);

static JavaVM *CurrentVM;

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *) {
  CurrentVM = vm;
  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;
  rt::init();
  return JNI_VERSION_1_1;
}

[[maybe_unused]] void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = getEnv(vm);
  gen::Runtime::drop(env);
  gen::Property::drop(env);
  gen::Device::drop(env);
  gen::Queue::drop(env);
  gen::Runnable::drop(env);
  gen::Policy::drop(env);
  gen::Dim3::drop(env);
  gen::String::drop(env);
}

static std::atomic<jlong> peerCounter = 0;
static std::unordered_map<jlong, std::shared_ptr<rt::Runtime>> Runtimes;
static std::unordered_map<jlong, std::shared_ptr<rt::Device>> Devices;
static std::unordered_map<jlong, std::vector<std::unique_ptr<std::string>>> DeviceModuleImages;
static std::unordered_map<jlong, std::shared_ptr<rt::DeviceQueue>> DeviceQueues;
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

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_deleteAllPeer0(JNIEnv *env, jclass) {
  std::lock_guard l(lock);
  DeviceQueues.clear();
  Devices.clear();
  DeviceModuleImages.clear();
  Runtimes.clear();
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_deleteRuntimePeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  Runtimes.erase(nativePeer);
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_deleteQueuePeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  DeviceQueues.erase(nativePeer);
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_deleteDevicePeer0(JNIEnv *env, jclass, jlong nativePeer) {
  std::lock_guard l(lock);
  Devices.erase(nativePeer);
  DeviceModuleImages.erase(nativePeer);
}

template <typename R> static jobject toJni(JNIEnv *env) {
  return wrapException(env, EX, [&]() {
    auto [peer, runtime] = emplaceRef(Runtimes, std::make_shared<R>());
    return gen::Runtime::of(env)(env, peer, toJni(env, runtime->name())).instance;
  });
}

[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_CUDA0(JNIEnv *env, jclass) {
  return toJni<rt::cuda::CudaRuntime>(env);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_HIP0(JNIEnv *env, jclass) {
  return toJni<rt::hip::HipRuntime>(env);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_OpenCL0(JNIEnv *env, jclass) {
  return toJni<rt::cl::ClRuntime>(env);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_Relocatable0(JNIEnv *env, jclass) {
  return toJni<rt::object::RelocatableRuntime>(env);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_Dynamic0(JNIEnv *env, jclass) {
  return toJni<rt::object::SharedRuntime>(env);
}
[[maybe_unused]] jobjectArray Java_polyregion_jvm_runtime_Runtime_runtimeProperties0(JNIEnv *env, jclass,
                                                                                     jlong nativePeer) {
  return toJni(env, findRef(env, Runtimes, nativePeer)->properties());
}
[[maybe_unused]] jobjectArray Java_polyregion_jvm_runtime_Runtime_devices0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&]() {
    auto devices = findRef(env, Runtimes, nativePeer)->enumerate();
    return toJni(env, devices, gen::Device::of(env).clazz, [&](auto &device) {
      auto [peer, d] = emplaceRef(Devices, std::shared_ptr(std::move(device)));
      return gen::Device::of(env)(env, peer, d->id(), toJni(env, d->name())).instance;
    });
  });
}

[[maybe_unused]] jobjectArray Java_polyregion_jvm_runtime_Runtime_deviceProperties0(JNIEnv *env, jclass,
                                                                                    jlong nativePeer) {
  return wrapException(env, EX, [&]() { return toJni(env, findRef(env, Devices, nativePeer)->properties()); });
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_loadModule0(JNIEnv *env, jclass, jlong nativePeer,
                                                                       jstring name, jbyteArray image) {

  wrapException(env, EX, [&]() {
    auto dev = findRef(env, Devices, nativePeer);
    // We need to hold on to our copied image here before passing it on, this is later destroyed with the device.
    auto data = std::make_unique<std::string>(env->GetArrayLength(image), '\0');
    env->GetByteArrayRegion(image, 0, env->GetArrayLength(image), reinterpret_cast<jbyte *>(data->data()));
    std::lock_guard l(lock);
    auto &&[it, _] = DeviceModuleImages.try_emplace(nativePeer, decltype(DeviceModuleImages)::mapped_type{});
    it->second.push_back(std::move(data));
    dev->loadModule(fromJni(env, name), *it->second.back());
  });
}

[[maybe_unused]] jlong Java_polyregion_jvm_runtime_Runtime_malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size,
                                                                    jbyte access) {
  if (auto a = rt::fromUnderlying(access); a) {
    return wrapException(env, EX,
                         [&]() { return static_cast<jlong>(findRef(env, Devices, nativePeer)->malloc(size, *a)); });
  } else
    return throwGeneric<jlong>(env, EX, "Illegal access type " + std::to_string(access));
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle) {
  wrapException(env, EX, [&]() { findRef(env, Devices, nativePeer)->free(static_cast<jlong>(handle)); });
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtime_createQueue0(JNIEnv *env, jclass, jlong nativePeer) {
  return wrapException(env, EX, [&]() {
    auto queue = findRef(env, Devices, nativePeer)->createQueue();
    auto [peer, _] = emplaceRef(DeviceQueues, std::shared_ptr(std::move(queue)));
    return gen::Queue::of(env)(env, peer).instance;
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
      gen::Runnable::of(attachedEnv).wrap(attachedEnv, cbRef).run(attachedEnv);
      if (attachedEnv->ExceptionCheck()) attachedEnv->ExceptionClear();
      attachedEnv->DeleteGlobalRef(cbRef);
      gen::Runnable::drop(attachedEnv);
    }

    CurrentVM->DetachCurrentThread();
  });
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_enqueueHostToDeviceAsync0(JNIEnv *env, jclass, //
                                                                                     jlong nativePeer,    //
                                                                                     jobject src, jlong dst, jint size,
                                                                                     jobject cb) {
  auto srcPtr = env->GetDirectBufferAddress(src);
  if (!srcPtr) throwGeneric(env, EX, "The source ByteBuffer is not backed by an direct allocation.");

  return wrapException(env, EX, [&]() {
    findRef(env, DeviceQueues, nativePeer)->enqueueHostToDeviceAsync(srcPtr, dst, size, fromJni(env, cb));
  });
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_enqueueDeviceToHostAsync0(JNIEnv *env, jclass, //
                                                                                     jlong nativePeer,    //
                                                                                     jlong src, jobject dst, jint size,
                                                                                     jobject cb) {
  auto dstPtr = env->GetDirectBufferAddress(dst);
  if (!dstPtr) throwGeneric(env, EX, "The destination ByteBuffer is not backed by an direct allocation.");
  return wrapException(env, EX, [&]() {
    findRef(env, DeviceQueues, nativePeer)->enqueueDeviceToHostAsync(src, dstPtr, size, fromJni(env, cb));
  });
}

static rt::Dim3 fromJni(JNIEnv *env, const generated::Dim3::Instance &d3) {
  return {static_cast<size_t>(d3.x(env)), static_cast<size_t>(d3.y(env)), static_cast<size_t>(d3.z(env))};
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtime_enqueueInvokeAsync0(JNIEnv *env, jclass, jlong nativePeer, //
                                                                               jstring moduleName, jstring symbol,    //
                                                                               jbyteArray argTypes,                   //
                                                                               jbyteArray argData,                    //
                                                                               jobject policy, jobject cb) {

  JNIEnv *e2;
  CurrentVM->AttachCurrentThread(reinterpret_cast<void **>(&e2), nullptr);

  auto argCount = env->GetArrayLength(argTypes);
  if (argCount == 0) {
    throwGeneric(env, EX, "empty argTypes, expecting at >= 1 for return type");
    return;
  }

  wrapException(env, EX, [&]() {
    auto argTs = fromJni<jbyte>(env, argTypes);
    auto argPs = fromJni<jbyte>(env, argData);
    auto argsPtr = argPs.data();
    std::vector<void *> argsPtrStore(argCount);
    std::vector<rt::Type> argTpeStore(argCount);
    for (jsize i = 0; i < argCount; ++i) {
      auto tpe = static_cast<rt::Type>(argTs[i]);
      argsPtrStore[i] = tpe == rt::Type::Void ? nullptr : argsPtr;
      argTpeStore[i] = tpe;
      argsPtr += rt::byteOfType(tpe);
    }

    // XXX we MUST hold on to the return pointer in the same block as invoke even if we don't use it
    void *rtnPtrStore = {};
    auto p = gen::Policy::of(env).wrap(env, policy);
    auto global = fromJni(env, p.global(env, gen::Dim3::of(env)));
    auto local = p.local(env, gen::Dim3::of(env)).map<rt::Dim3>([&](auto x) { return fromJni(env, x); });

    findRef(env, DeviceQueues, nativePeer)
        ->enqueueInvokeAsync(fromJni(env, moduleName), fromJni(env, symbol), argTpeStore, argsPtrStore, {global, local},
                             fromJni(env, cb));

    if (argTpeStore[argPs.size() - 1] == rt::Type::Ptr) {
      // we got four possible cases when a function return pointers:
      //  1. Pointer to one of the argument   (LUT[ptr]==Some) => passthrough
      //  2. Pointer to malloc'd memory       (LUT[ptr]==Some) => passthrough
      //  3. Pointer within a malloc'd region (LUT[ptr]==None) => copy
      //  4. Pointer to stack allocated data  (LUT[ptr]==None) => undefined, should not happen

      throwGeneric(env, EX, "Returning pointers is unimplemented");
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

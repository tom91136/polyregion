#include <cassert>
#include <string>
#include <utility>
#include <vector>

#include "cl_runtime.h"
#include "cuda_runtime.h"
#include "hip_runtime.h"
#include "jni_utils.h"
#include "mirror.h"
#include "object_runtime.h"
#include "polyregion_jvm_runtime_Runtimes.h"
#include "runtime.h"
#include "utils.hpp"

using namespace polyregion;
namespace rt = ::runtime;

[[noreturn]] static void throwGeneric(JNIEnv *env, const std::string &message) {
  throwGeneric("polyregion/PolyregionRuntimeException", env, message);
}

static jclass ByteBuffer;
static jmethodID ByteBuffer_allocateDirect;

static jclass bindGlobalClassRef(JNIEnv *env, const std::string &name) {
  jclass localClsRef = env->FindClass(name.c_str());
  auto ref = (jclass)env->NewGlobalRef(localClsRef);
  env->DeleteLocalRef(localClsRef);
  return ref;
}

static std::unique_ptr<generated::Runtime> Runtime;
static std::unique_ptr<generated::Property> Property;
static std::unique_ptr<generated::Device> Device;
static std::unique_ptr<generated::Queue> DeviceQueue;
static std::unique_ptr<generated::Runnable> Runnable;
static std::unique_ptr<generated::Policy> Policy;
static std::unique_ptr<generated::Dim3> Dim3;

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *) {

  polyregion::runtime::init();

  JNIEnv *env = getEnv(vm);
  if (!env) return JNI_ERR;

  Runtime = std::make_unique<generated::Runtime>(env);
  Property = std::make_unique<generated::Property>(env);
  Device = std::make_unique<generated::Device>(env);
  DeviceQueue = std::make_unique<generated::Queue>(env);
  Runnable = std::make_unique<generated::Runnable>(env);
  Policy = std::make_unique<generated::Policy>(env);
  Dim3 = std::make_unique<generated::Dim3>(env);

  // pin the class as global ref so that method-ref remains usable
  ByteBuffer = bindGlobalClassRef(env, "java/nio/ByteBuffer");
  if (!ByteBuffer) env->FatalError("ByteBuffer not found!");

  ByteBuffer_allocateDirect = env->GetStaticMethodID(ByteBuffer, "allocateDirect", "(I)Ljava/nio/ByteBuffer;");
  if (!ByteBuffer_allocateDirect) env->FatalError("ByteBuffer.allocateDirect not found!");

  return JNI_VERSION_1_1;
}

[[maybe_unused]] void JNI_OnUnload(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_1);
  env->DeleteGlobalRef(ByteBuffer);
}

struct NIOBuffer {
  jobject buffer;
  void *ptr;
  size_t sizeInBytes;
  NIOBuffer() = default;
  NIOBuffer(JNIEnv *env, jobject buffer)
      : buffer(buffer), ptr(env->GetDirectBufferAddress(buffer)), sizeInBytes(env->GetDirectBufferCapacity(buffer)) {}
};

static jobject allocDirect(JNIEnv *env, jint bytes) {
  auto buffer = env->CallStaticObjectMethod(ByteBuffer, ByteBuffer_allocateDirect, bytes);
  if (env->ExceptionCheck()) {
    env->Throw(env->ExceptionOccurred());
  }
  if (!buffer) {
    throwGeneric(env, "JNI ByteBuffer allocation of " + std::to_string(bytes) + " bytes returned NULL");
    return nullptr;
  }
  return buffer;
}

static std::atomic<jlong> peerCounter = 0;
static std::unordered_map<uint64_t, std::shared_ptr<void>> values;

template <typename T> static auto emplaceRef(std::shared_ptr<T> x) {
  auto i = peerCounter++;
  values.emplace(i, x);
  return std::make_pair(i, x);
}

// template <typename T, typename... Args> static auto emplaceRef(Args &&...args) {
//   return emplaceRef(std::make_shared<T>(std::forward<Args>(args)...));
// }

template <typename T> static std::shared_ptr<T> findRef(JNIEnv *env, jlong nativePeer) {
  if (auto it = values.find(nativePeer); it == values.end()) {
    throwGeneric(env, "Cannot find native peer (" + std::to_string(nativePeer) + ") ");
    return {};
  } else
    return reinterpret_pointer_cast<T>(it->second);
}

static jobjectArray toJni(JNIEnv *env, const std::vector<runtime::Property> &xs) {
  return toJni(env, xs, Property->clazz,
               [&](auto &&x) { return (*Property)(env, toJni(env, x.first), toJni(env, x.second)).instance; });
}

template <typename R> static jobject toJni(JNIEnv *env) {
  try {
    auto [peer, runtime] = emplaceRef<R>(std::make_shared<R>());
    return (*Runtime)(env, peer, toJni(env, runtime->name()), toJni(env, runtime->properties())).instance;
  } catch (const std::exception &e) {
    throwGeneric(env, e.what());
    return nullptr;
  }
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_deletePeer(JNIEnv *env, jclass, jlong value) {
  values.erase(value);
}

[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_CUDA(JNIEnv *e, jclass) {
  return toJni<rt::cuda::CudaRuntime>(e);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_HIP(JNIEnv *e, jclass) {
  return toJni<rt::hip::HipRuntime>(e);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_OpenCL(JNIEnv *e, jclass) {
  return toJni<rt::cl::ClRuntime>(e);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_Relocatable(JNIEnv *e, jclass) {
  return toJni<rt::object::RelocatableRuntime>(e);
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_Dynamic(JNIEnv *e, jclass) {
  return toJni<rt::object::SharedRuntime>(e);
}
[[maybe_unused]] jobjectArray Java_polyregion_jvm_runtime_Runtimes_devices0(JNIEnv *env, jclass, jlong nativePeer) {
  auto devices = findRef<rt::Runtime>(env, nativePeer)->enumerate();
  return toJni(env, devices, Device->clazz, [&](auto &&device) {
    auto [peer, d] = emplaceRef<rt::Device>(std::move(device));
    return (*Device)(env, peer, d->id(), toJni(env, d->name()), toJni(env, d->properties())).instance;
  });
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_loadModule0(JNIEnv *env, jclass, jlong nativePeer,
                                                                       jstring name, jbyteArray image) {
  auto imageData = env->GetByteArrayElements(image, nullptr);
  if (!imageData) throwGeneric(env, "Cannot read object byte[]");
  findRef<rt::Device>(env, nativePeer)
      ->loadModule(fromJni(env, name), std::string(imageData, imageData + env->GetArrayLength(image)));
  env->ReleaseByteArrayElements(image, imageData, JNI_ABORT);
}
[[maybe_unused]] jlong Java_polyregion_jvm_runtime_Runtimes_malloc0(JNIEnv *env, jclass, jlong nativePeer, jlong size,
                                                                    jbyte access) {
  if (auto a = rt::fromUnderlying(access); a) {
    return static_cast<jlong>(findRef<rt::Device>(env, nativePeer)->malloc(size, *a));
  } else {
    throwGeneric(env, "Illegal access type " + std::to_string(access));
    return 0;
  }
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_free0(JNIEnv *env, jclass, jlong nativePeer, jlong handle) {
  findRef<rt::Device>(env, nativePeer)->free(static_cast<jlong>(handle));
}
[[maybe_unused]] jobject Java_polyregion_jvm_runtime_Runtimes_createQueue0(JNIEnv *env, jclass, jlong nativePeer) {
  auto queue = findRef<rt::Device>(env, nativePeer)->createQueue();
  auto [peer, _] = emplaceRef<rt::DeviceQueue>(std::move(queue));
  return (*DeviceQueue)(env, peer).instance;
}

static rt::MaybeCallback fromJni(JNIEnv *env, jobject cb) {
  return !cb ? std::nullopt : std::make_optional([&]() {
    if (cb) Runnable->wrap(env, cb).run(env);
  });
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_enqueueHostToDeviceAsync0(JNIEnv *env, jclass, //
                                                                                     jlong nativePeer,    //
                                                                                     jobject src, jlong dst, jint size,
                                                                                     jobject cb) {
  findRef<rt::DeviceQueue>(env, nativePeer)
      ->enqueueHostToDeviceAsync(env->GetDirectBufferAddress(src), dst, size, fromJni(env, cb));
}
[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_enqueueDeviceToHostAsync0(JNIEnv *env, jclass, //
                                                                                     jlong nativePeer,    //
                                                                                     jlong src, jobject dst, jint size,
                                                                                     jobject cb) {
  findRef<rt::DeviceQueue>(env, nativePeer)
      ->enqueueDeviceToHostAsync(src, env->GetDirectBufferAddress(dst), size, fromJni(env, cb));
}

static rt::Dim3 fromJni(JNIEnv *env, const generated::Dim3::Instance &d3) {
  return {static_cast<size_t>(d3.x(env)), static_cast<size_t>(d3.y(env)), static_cast<size_t>(d3.z(env))};
}

[[maybe_unused]] void Java_polyregion_jvm_runtime_Runtimes_enqueueInvokeAsync0(JNIEnv *env, jclass, jlong nativePeer, //
                                                                               jstring moduleName, jstring symbol,    //
                                                                               jbyteArray argTypes,                   //
                                                                               jobjectArray argPtrs,                  //
                                                                               jbyte rtnType,                         //
                                                                               jobject rtnPtr,                        //
                                                                               jobject policy, jobject cb) {

  if (env->GetArrayLength(argTypes) != env->GetArrayLength(argPtrs)) {
    throwGeneric(env, "argPtrs size !=  argTypes size");
    return;
  }

  auto bind = [&](jobject argBuffer, jbyte argTy, void **argPtrStore) -> runtime::TypedPointer {
    *argPtrStore = env->GetDirectBufferAddress(argBuffer);
    auto tpe = static_cast<runtime::Type>(argTy);
    switch (tpe) {
      case runtime::Type::Bool8:
      case runtime::Type::Byte8:
      case runtime::Type::CharU16:
      case runtime::Type::Short16:
      case runtime::Type::Int32:
      case runtime::Type::Long64:
      case runtime::Type::Float32:
      case runtime::Type::Double64:
      case runtime::Type::Void: {
        return std::make_pair(tpe, *argPtrStore); // XXX no indirection needed, dereference now
      }
      case runtime::Type::Ptr: {
        return std::make_pair(tpe, argPtrStore); // XXX pointer indirection, don't dereference
      }
      default: throwGeneric(env, "Unimplemented parameter type " + std::to_string(to_underlying(tpe)));
    }
  };

  try {
    std::vector<runtime::TypedPointer> args(env->GetArrayLength(argPtrs));

    // XXX we MUST hold on to the vector of pointers in the same block as invoke even if we don't use it
    std::vector<void *> argsPtrStore(args.size());
    auto argTs = fromJni<jbyte>(env, argTypes);
    for (jsize i = 0; i < jsize(args.size()); ++i)
      bind(env->GetObjectArrayElement(argPtrs, i), argTs[i], &argsPtrStore[i]);

    // XXX we MUST hold on to the return pointer in the same block as invoke even if we don't use it
    void *rtnPtrStore = {};
    runtime::TypedPointer rtn = bind(rtnPtr, rtnType, &rtnPtrStore);

    auto p = Policy->wrap(env, policy);
    auto global = fromJni(env, p.local(env, *Dim3));
    auto local = p.local(env, *Dim3).map<rt::Dim3>([&](auto x) { return fromJni(env, x); });

    findRef<rt::DeviceQueue>(env, nativePeer)
        ->enqueueInvokeAsync(fromJni(env, moduleName), fromJni(env, symbol), args, rtn, {global, local},
                             fromJni(env, cb));

  } catch (const std::exception &e) {
    throwGeneric(env, e.what());
  }
}

template <typename R>
static R invokeGeneric(JNIEnv *env, jbyteArray object, jbyteArray argTypes, jobjectArray argPtrs,
                       const std::function<R(const rt::Object &, const std::vector<runtime::TypedPointer> &)> &f) {
  auto objData = env->GetByteArrayElements(object, nullptr);
  if (!objData) {
    throwGeneric(env, "Cannot read object byte[]");
    return {};
  }

  if (env->GetArrayLength(argTypes) != env->GetArrayLength(argPtrs)) {
    throwGeneric(env, "argPtrs size !=  argTypes size");
    return {};
  }

  try {
    auto data = reinterpret_cast<const uint8_t *>(objData);
    std::vector<uint8_t> bytes(data, data + env->GetArrayLength(object));

    runtime::Object obj(bytes);

    std::vector<runtime::TypedPointer> params(env->GetArrayLength(argPtrs));
    // XXX we MUST hold on to the vector of pointers in the same block as invoke even if we don't use it
    std::vector<void *> pointers(params.size());

    auto argTypes_ = env->GetByteArrayElements(argTypes, nullptr);
    for (jint i = 0; i < env->GetArrayLength(argPtrs); ++i) {
      pointers[i] = env->GetDirectBufferAddress(env->GetObjectArrayElement(argPtrs, i));
      params[i].first = static_cast<runtime::Type>(argTypes_[i]);
      switch (params[i].first) {
        case runtime::Type::Bool8:
        case runtime::Type::Byte8:
        case runtime::Type::CharU16:
        case runtime::Type::Short16:
        case runtime::Type::Int32:
        case runtime::Type::Long64:
        case runtime::Type::Float32:
        case runtime::Type::Double64:
        case runtime::Type::Void: {
          params[i].second = pointers[i]; // XXX no indirection
          break;
        }
        case runtime::Type::Ptr: {
          params[i].second = &pointers[i]; // XXX pointer indirection
          break;
        }
        default: throwGeneric(env, "Unimplemented parameter type " + std::to_string(to_underlying(params[i].first)));
      }
    }
    env->ReleaseByteArrayElements(argTypes, argTypes_, JNI_ABORT);

    auto result = f(obj, params);
    env->ReleaseByteArrayElements(object, objData, JNI_ABORT);
    return result;
  } catch (const std::exception &e) {
    throwGeneric(env, e.what());
    return {};
  }
}

template <typename R, runtime::Type RT>
static R invokePrimitive(JNIEnv *env, //
                         jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokeGeneric<R>(env, object, argTypes, argPtrs, [&](auto &obj, auto &params) {
    R rtnData{};
    runtime::TypedPointer rtn{RT, &rtnData};

    auto allocator = [&](size_t size) {
      //      std::cerr << "[runtime][primitive] Allocating " << size << "bytes" << std::endl;
      //      auto buffer = allocDirect(env, jint(size));
      //      auto ptr = env->GetDirectBufferAddress(buffer);
      auto ptr = malloc(size);
      return ptr;
    };

    obj.invoke(copyString(env, symbol), allocator, params, rtn);
    return rtnData;
  });
}

static jobject invokeObject(JNIEnv *env, //
                            jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs,
                            jint rtnBytes) {
  return invokeGeneric<jobject>(env, object, argTypes, argPtrs, [&](auto &obj, auto &params) -> jobject {
    // we got four possible cases when a function return pointers:
    //  1. Pointer to one of the argument   (LUT[ptr]==Some) => passthrough
    //  2. Pointer to malloc'd memory       (LUT[ptr]==Some) => passthrough
    //  3. Pointer within a malloc'd region (LUT[ptr]==None) => copy
    //  4. Pointer to stack allocated data  (LUT[ptr]==None) => undefined, should not happen

    std::unordered_map<void *, NIOBuffer> allocations;

    // save all args in the alloc LUT first so that we can identify them later
    for (jint i = 0; i < env->GetArrayLength(argPtrs); ++i) {
      auto buffer = NIOBuffer(env, env->GetObjectArrayElement(argPtrs, i));
      allocations[buffer.ptr] = buffer;
    }

    // make our allocator store it as well
    auto allocator = [&](size_t size) {
      //      std::cerr << "[runtime][obj rtn] Allocating " << size << " bytes" << std::endl;
      auto buffer = NIOBuffer(env, allocDirect(env, jint(size)));
      allocations[buffer.ptr] = buffer;
      return buffer.ptr;
    };

    void *rtnData{};
    runtime::TypedPointer rtn{runtime::Type::Ptr, &rtnData};
    obj.invoke(copyString(env, symbol), allocator, params, rtn);

    if (auto r = allocations.find(rtnData); r != allocations.end()) {
      auto buffer = r->second; // we found the original allocation, passthrough-return
      if (rtnBytes != -1) {
        throwGeneric(env, "Bad size (" + std::to_string(rtnBytes) + ") for passthrough buffer of " +
                              std::to_string(buffer.sizeInBytes) + " bytes; use -1 for passthrough");
        return nullptr;
      } else {
        if (env->ExceptionCheck()) {
          env->Throw(env->ExceptionOccurred());
        }
        return buffer.buffer;
      }

    } else {
      if (rtnBytes < 0) {
        throwGeneric(env, "Bad size (" + std::to_string(rtnBytes) + ") for copy buffer");
        return nullptr;
      } else {
        auto buffer = allocDirect(env, rtnBytes);
        std::memcpy(env->GetDirectBufferAddress(buffer), rtnData, rtnBytes);
        return buffer;
      }
    }
  });
}

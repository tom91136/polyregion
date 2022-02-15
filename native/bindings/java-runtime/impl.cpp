#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "polyregion_PolyregionRuntime.h"
#include "runtime.h"
#include "utils.hpp"

using namespace polyregion;
static void throwGeneric(JNIEnv *env, const std::string &message) {
  if (auto exClass = env->FindClass("polyregion/PolyregionRuntimeException"); exClass) {
    env->ThrowNew(exClass, message.c_str());
  }
}

static jclass ByteBuffer;
static jmethodID ByteBuffer_allocateDirect;

static jclass bindGlobalClassRef(JNIEnv *env, const std::string &name) {
  jclass localClsRef;
  localClsRef = env->FindClass(name.c_str());
  auto ref = (jclass)env->NewGlobalRef(localClsRef);
  env->DeleteLocalRef(localClsRef);
  return ref;
}

[[maybe_unused]] jint JNI_OnLoad(JavaVM *vm, void *reserved) {

  polyregion::runtime::init();

  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_1) != JNI_OK) {
    return JNI_ERR;
  }

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

static std::string copyString(JNIEnv *env, jstring str) {
  auto data = env->GetStringUTFChars(str, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(str, data);
  return out;
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
  }
  return buffer;
}

template <typename R>
static R invokeGeneric(JNIEnv *env, jbyteArray object, jbyteArray argTypes, jobjectArray argPtrs,
                       const std::function<R(const runtime::Object &, const std::vector<runtime::TypedPointer> &)> &f) {
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
      case runtime::Type::Bool:
      case runtime::Type::Byte:
      case runtime::Type::Char:
      case runtime::Type::Short:
      case runtime::Type::Int:
      case runtime::Type::Long:
      case runtime::Type::Float:
      case runtime::Type::Double:
      case runtime::Type::Void: {
        params[i].second = pointers[i]; // XXX no indirection
        break;
      }
      case runtime::Type::Ptr: {
        params[i].second = &pointers[i]; // XXX pointer indirection
        break;
      }
      default:
        throwGeneric(env, "Unimplemented parameter type " + std::to_string(to_underlying(params[i].first)));
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
      std::cerr << "[runtime][primitive] Allocating " << size << "bytes" << std::endl;
      auto buffer = allocDirect(env, jint(size));
      auto ptr = env->GetDirectBufferAddress(buffer);
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
      std::cerr << "[runtime][obj rtn] Allocating " << size << "bytes" << std::endl;
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

[[maybe_unused]] JNIEXPORT void JNICALL Java_polyregion_PolyregionRuntime_invoke( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  invokePrimitive<std::nullptr_t, runtime::Type::Void>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jboolean JNICALL Java_polyregion_PolyregionRuntime_invokeBool( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jboolean, runtime::Type::Byte>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jbyte JNICALL Java_polyregion_PolyregionRuntime_invokeByte( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jbyte, runtime::Type::Byte>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jchar JNICALL Java_polyregion_PolyregionRuntime_invokeChar( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jchar, runtime::Type::Char>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jshort JNICALL Java_polyregion_PolyregionRuntime_invokeShort( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jshort, runtime::Type::Short>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jint JNICALL Java_polyregion_PolyregionRuntime_invokeInt( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jint, runtime::Type::Int>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jlong JNICALL Java_polyregion_PolyregionRuntime_invokeLong( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jlong, runtime::Type::Long>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jfloat JNICALL Java_polyregion_PolyregionRuntime_invokeFloat( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jfloat, runtime::Type::Float>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jdouble JNICALL Java_polyregion_PolyregionRuntime_invokeDouble( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jdouble, runtime::Type::Double>(env, object, symbol, argTypes, argPtrs);
}
[[maybe_unused]] JNIEXPORT jobject JNICALL Java_polyregion_PolyregionRuntime_invokeObject( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs, jint rtnBytes) {
  return invokeObject(env, object, symbol, argTypes, argPtrs, rtnBytes);
}
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

jclass ByteBuffer;
jmethodID ByteBuffer_allocateDirect;

jint JNI_OnLoad(JavaVM *vm, void *reserved) {

  polyregion::runtime::init();

  JNIEnv *env;
  vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_1);

  ByteBuffer = env->FindClass("java/nio/ByteBuffer");
  if (!ByteBuffer) env->FatalError("ByteBuffer not found!");
  ByteBuffer_allocateDirect = env->GetStaticMethodID(ByteBuffer, "allocateDirect", "(I)Ljava/nio/ByteBuffer;");
  if (!ByteBuffer_allocateDirect) env->FatalError("ByteBuffer.allocateDirect not found!");

  return JNI_VERSION_1_1;
}

static std::string copyString(JNIEnv *env, jstring str) {
  auto data = env->GetStringUTFChars(str, nullptr);
  std::string out(data);
  env->ReleaseStringUTFChars(str, data);
  return out;
}

using Buffer = std::pair<void *, jobject>;

static std::pair<std::vector<runtime::TypedPointer>, std::vector<Buffer>> bindArgs( //
    JNIEnv *env, jbyteArray argTypes, jobjectArray argPtrs) {
  std::vector<runtime::TypedPointer> params(env->GetArrayLength(argPtrs));
  std::vector<std::pair<void *, jobject>> pointers(env->GetArrayLength(argPtrs));
  auto argTypes_ = env->GetByteArrayElements(argTypes, nullptr);
  for (jint i = 0; i < env->GetArrayLength(argPtrs); ++i) {
    params[i].first = static_cast<runtime::Type>(argTypes_[i]);
    auto buffer = env->GetObjectArrayElement(argPtrs, i);
    pointers[i] = {env->GetDirectBufferAddress(buffer), buffer};
    if (!pointers[i].first) {
      throwGeneric(env, "Unable to retrieve direct buffer address");
      return {};
    }
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
      params[i].second = pointers[i].first; // XXX no indirection
      break;
    }
    case runtime::Type::Ptr: {
      params[i].second = &pointers[i].first; // XXX pointer indirection
      break;
    }
    default:
      throwGeneric(env, "Unimplemented parameter type " + std::to_string(to_underlying(params[i].first)));
    }
  }
  env->ReleaseByteArrayElements(argTypes, argTypes_, JNI_ABORT);
  return {params, pointers};
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

    // XXX we MUST hold on to the vector of pointers in the same block as invoke even if we don't use it
    auto [params, _] = bindArgs(env, argTypes, argPtrs);

    return f(obj, params);

    env->ReleaseByteArrayElements(object, objData, JNI_ABORT);
  } catch (const std::exception &e) {
    throwGeneric(env, e.what());
    return {};
  }
}

template <typename R, runtime::Type RT>
R invokePrimitive(JNIEnv *env, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokeGeneric<R>(env, object, argTypes, argPtrs, [&](auto &obj, auto &params) {
    R rtnData{};
    runtime::TypedPointer rtn{RT, &rtnData};
    std::unordered_map<void *, jobject> allocations;
    auto allocator = [&](size_t size) {
      std::cerr << "[runtime] Allocating " << size << "bytes @ ";
      auto buffer = env->CallStaticObjectMethod(ByteBuffer, ByteBuffer_allocateDirect, size);
      auto ptr = env->GetDirectBufferAddress(buffer);
      allocations[ptr] = buffer;
      std::cerr << ptr << std::endl;
      return ptr;
    };

    obj.invoke(copyString(env, symbol), allocator, params, rtn);
    return rtnData;
  });
}

//TODO
template <typename R, runtime::Type RT>
jobject invokeObject(JNIEnv *env, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokeGeneric<R>(env, object, argTypes, argPtrs, [&](auto &obj, auto &params) {
    void *rtnData{};
    runtime::TypedPointer rtn{RT, &rtnData};

    // we got four possible cases when a function return pointers:
    //  1. Pointer to one of the argument   => passthrough
    //  2. Pointer to malloc'd memory       => passthrough
    //  3. Pointer within a malloc'd region => copy
    //  4. Pointer to stack allocated data  => copy

    std::unordered_map<void *, jobject> allocations;

    auto allocator = [&](size_t size) {
      std::cerr << "[runtime] Allocating " << size << "bytes @ ";
      auto buffer = env->CallStaticObjectMethod(ByteBuffer, ByteBuffer_allocateDirect, size);
      auto ptr = env->GetDirectBufferAddress(buffer);
      allocations[ptr] = buffer;
      std::cerr << ptr << std::endl;
      return ptr;
    };

    //
    obj.invoke(copyString(env, symbol), allocator, params, rtn);

    return rtnData;
  });
}

JNIEXPORT void JNICALL Java_polyregion_PolyregionRuntime_invoke( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  invokePrimitive<std::nullptr_t, runtime::Type::Void>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jint JNICALL Java_polyregion_PolyregionRuntime_invokeInt( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jint, runtime::Type::Int>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jlong JNICALL Java_polyregion_PolyregionRuntime_invokeLong( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jlong, runtime::Type::Long>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jfloat JNICALL Java_polyregion_PolyregionRuntime_invokeFloat( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jfloat, runtime::Type::Float>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jdouble JNICALL Java_polyregion_PolyregionRuntime_invokeDouble( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jdouble, runtime::Type::Double>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jshort JNICALL Java_polyregion_PolyregionRuntime_invokeShort( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jshort, runtime::Type::Short>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jchar JNICALL Java_polyregion_PolyregionRuntime_invokeChar( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jchar, runtime::Type::Char>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jbyte JNICALL Java_polyregion_PolyregionRuntime_invokeByte( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs) {
  return invokePrimitive<jbyte, runtime::Type::Byte>(env, object, symbol, argTypes, argPtrs);
}
JNIEXPORT jobject JNICALL Java_polyregion_PolyregionRuntime_invokeObject( //
    JNIEnv *env, jclass, jbyteArray object, jstring symbol, jbyteArray argTypes, jobjectArray argPtrs, jint rtnBytes) {

  return nullptr;
}
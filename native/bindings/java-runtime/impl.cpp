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

void Java_polyregion_PolyregionRuntime_invoke(JNIEnv *env, jclass thisCls,                  //
                                              jbyteArray object, jstring symbol,            //
                                              jbyte returnType, jobject returnPtr,          //
                                              jbyteArray paramTypes, jobjectArray paramPtrs //
) {

  auto objData = env->GetByteArrayElements(object, nullptr);
  if (!objData) {
    throwGeneric(env, "Cannot read object byte[]");
    return;
  }

  if (env->GetArrayLength(paramTypes) != env->GetArrayLength(paramPtrs)) {
    throwGeneric(env, "paramPtrs size !=  paramTypes size");
    return;
  }

  auto sym = env->GetStringUTFChars(symbol, nullptr);
  try {
    auto data = reinterpret_cast<const uint8_t *>(objData);
    std::vector<uint8_t> bytes(data, data + env->GetArrayLength(object));
    runtime::Object obj(bytes);
    runtime::TypedPointer rtn{static_cast<runtime::Type>(returnType), env->GetDirectBufferAddress(returnPtr)};
    std::vector<runtime::TypedPointer> params(env->GetArrayLength(paramPtrs));
    std::vector<void *> pointers(env->GetArrayLength(paramPtrs));
    auto paramTypes_ = env->GetByteArrayElements(paramTypes, nullptr);
    for (jint i = 0; i < env->GetArrayLength(paramPtrs); ++i) {
      params[i].first = static_cast<runtime::Type>(paramTypes_[i]);
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
        pointers[i] = env->GetDirectBufferAddress(env->GetObjectArrayElement(paramPtrs, i));
        if (!pointers[i]) {
          throwGeneric(env, "Unable to retrieve direct buffer address");
          return;
        }
        params[i].second = pointers[i]; // XXX no indirection
        break;
      }
      case runtime::Type::Ptr: {
        pointers[i] = env->GetDirectBufferAddress(env->GetObjectArrayElement(paramPtrs, i));
        if (!pointers[i]) {
          throwGeneric(env, "Unable to retrieve direct buffer address");
          return;
        }
        params[i].second = &pointers[i]; // XXX pointer indirection
        break;
      }
      default:
        throwGeneric(env, "Unimplemented parameter type " + std::to_string(to_underlying(params[i].first)));
      }
    }
    env->ReleaseByteArrayElements(paramTypes, paramTypes_, JNI_ABORT);

    obj.invoke(sym, params, rtn);
  } catch (const std::exception &e) {
    throwGeneric(env, e.what());
  }
  env->ReleaseStringUTFChars(symbol, sym);
  env->ReleaseByteArrayElements(object, objData, JNI_ABORT);
}
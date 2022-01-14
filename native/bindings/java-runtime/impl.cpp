#include <iostream>
#include <string>
#include <vector>

#include "polyregion_PolyregionRuntime.h"
#include "polyregion_runtime.h"

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

  auto obj = polyregion_load_object(reinterpret_cast<const uint8_t *>(objData), env->GetArrayLength(object));

  if (!obj->object) {
    throwGeneric(env, std::string(obj->message));
  } else {

    auto sym = env->GetStringUTFChars(symbol, nullptr);

    polyregion_data rtn{static_cast<polyregion_type>(returnType), env->GetDirectBufferAddress(returnPtr)};
    std::vector<polyregion_data> params(env->GetArrayLength(paramPtrs));

    std::vector<void*> pointer1 (env->GetArrayLength(paramPtrs));

    auto paramTypes_ = env->GetByteArrayElements(paramTypes, nullptr);
    for (jint i = 0; i < env->GetArrayLength(paramPtrs); ++i) {
      params[i].type = static_cast<polyregion_type>(paramTypes_[i]);
      switch (params[i].type) {
      case Bool:
      case Byte:
      case Char:
      case Short:
      case Int:
      case Long:
      case Float:
      case Double:
      case Void:
        pointer1[i] = env->GetObjectArrayElement(paramPtrs, i);
        params[i].ptr = &pointer1[i];
        break;
      case Ptr: {
        std::cout << "PRE" << std::endl;
        pointer1[i] = env->GetDirectBufferAddress(env->GetObjectArrayElement(paramPtrs, i));
        params[i].ptr = &pointer1[i];
        break;
      }
      }

      std::cout << "GO" << std::endl;

      //      params[i].ptr = env->GetObjectArrayElement(paramPtrs, i);
    }
    env->ReleaseByteArrayElements(paramTypes, paramTypes_, JNI_ABORT);
    std::cout << "INV" << std::endl;

    auto error = polyregion_invoke(obj->object, sym, params.data(), env->GetArrayLength(paramPtrs), &rtn);
    std::cout << "exit" << std::endl;

    if (error) {
      throwGeneric(env, error);
    }

    env->ReleaseStringUTFChars(symbol, sym);
  }
  polyregion_release_object(obj);
  env->ReleaseByteArrayElements(object, objData, JNI_ABORT);
}
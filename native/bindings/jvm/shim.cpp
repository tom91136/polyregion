#include "polyregion/dl.h"
#include "generated/mirror.h"
#include "generated/natives.h"
#include "jni_utils.h"

using namespace polyregion::generated::registry;

static constexpr const char *EX = "polyregion/jvm/PolyregionLoaderException";

static std::vector<jobject> files;

static JavaVM *CurrentVM = {};

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  fprintf(stdout, "Shim JNI_OnLoad\n");
  files.clear(); // In case OnUnload didn't finish normally
  CurrentVM = vm;

  JNIEnv *env = getEnv(vm);
  Natives::registerMethods(env);
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  fprintf(stdout, "Shim JNI_OnUnload\n");
  JNIEnv *env = getEnv(vm);
  Natives::unregisterMethods(env);

  if (!files.empty()) {
    fprintf(stdout, "Registered files: %ld\n", files.size());
    auto File = polyregion::generated::File::of(env);
    for (auto &f : files) {
      File.wrap(env, f).delete_(env);
      if (env->ExceptionCheck()) env->ExceptionClear();
      env->DeleteGlobalRef(f);
    }
    files.clear();
  }
  CurrentVM = nullptr;
}

void Natives::registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file) {
  files.push_back(env->NewGlobalRef(file));
}

static std::string resolveDlError() {
  auto message = polyregion_dl_error();
  return message ? std::string(message) : "(no message reported)";
}

jlong Natives::dynamicLibraryLoad0(JNIEnv *env, jclass, jstring name) {
  auto str = fromJni(env, name);
  if (auto dylib = polyregion_dl_open(str.c_str()); !dylib) {
    throwGeneric(env, EX, "Cannot load library `" + str + "` :" + resolveDlError());
    return {};
  } else {
    void *f = polyregion_dl_find(dylib, "JNI_OnLoad");
    if (f) {
      ((jint(*)(JavaVM *, void *))(f))(CurrentVM, nullptr);
    }
    return reinterpret_cast<jlong>(dylib);
  }
}

void Natives::dynamicLibraryRelease0(JNIEnv *env, jclass, jlong handle) {
  auto typedHandle = reinterpret_cast<polyregion_dl_handle>(handle);
  void *f = polyregion_dl_find(typedHandle, "JNI_OnUnload");
  if (f) {
    ((void (*)(JavaVM *, void *))(f))(CurrentVM, nullptr);
  }
  if (auto code = polyregion_dl_close(typedHandle); code != 0) {
    throwGeneric(env, EX, "Cannot unload module:" + resolveDlError());
  }
}

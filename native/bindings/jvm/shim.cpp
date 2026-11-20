#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "polyregion/dl.h"
#include "polyregion/env_keys.h"

#include "generated/mirror.h"
#include "generated/natives.h"
#include "jni_utils.h"

using namespace polyregion::generated::registry;

static constexpr const char *EX = "polyregion/jvm/PolyregionLoaderException";

static std::vector<jobject> files;

static JavaVM *CurrentVM = {};

struct Binding {
  polyregion_dl_handle handle;
  jweak owner;
  std::size_t holders;
};

static std::mutex dlMutex;
static std::vector<Binding> bindings;

static std::vector<Binding>::iterator findBinding(JNIEnv *env, polyregion_dl_handle handle, jclass owner) {
  std::erase_if(bindings, [&](const Binding &b) {
    if (!env->IsSameObject(b.owner, nullptr)) return false;
    env->DeleteWeakGlobalRef(b.owner);
    return true;
  });
  return std::find_if(bindings.begin(), bindings.end(),
                      [&](const Binding &b) { return b.handle == handle && env->IsSameObject(b.owner, owner); });
}

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  if (const char *d = std::getenv(polyregion::env::PolyregionDebug); d && *d) fprintf(stdout, "Shim JNI_OnLoad\n");
  CurrentVM = vm;

  JNIEnv *env = getEnv(vm);
  Natives::registerMethods(env);
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  const bool debug = [] {
    const char *d = std::getenv(polyregion::env::PolyregionDebug);
    return d && *d;
  }();
  if (debug) fprintf(stdout, "Shim JNI_OnUnload\n");
  JNIEnv *env = getEnv(vm);

  if (!files.empty()) {
    if (debug) fprintf(stdout, "Registered files: %zu\n", files.size());
    auto File = polyregion::generated::File::of(env);
    for (auto &f : files) {
      File.wrap(env, f).delete_(env);
      if (env->ExceptionCheck()) env->ExceptionClear();
      env->DeleteGlobalRef(f);
    }
    files.clear();
  }
}

void Natives::registerFilesToDropOnUnload0(JNIEnv *env, jclass, jobject file) { files.push_back(env->NewGlobalRef(file)); }

static std::string resolveDlError() {
  const char *message = polyregion_dl_error();
  return message ? std::string(message) : "(no message reported)";
}

jlong Natives::dynamicLibraryLoad0(JNIEnv *env, jclass self, jstring name) {
  const auto str = fromJni(env, name);
  if (auto *dylib = polyregion_dl_open(str.c_str()); !dylib) {
    throwGeneric(env, EX, "Cannot load library `" + str + "` :" + resolveDlError());
    return {};
  } else {
    const std::scoped_lock lock(dlMutex);
    if (const auto it = findBinding(env, dylib, self); it != bindings.end()) {
      it->holders++;
    } else {
      bindings.push_back({dylib, env->NewWeakGlobalRef(self), 1});
      if (auto *f = polyregion_dl_find(dylib, "JNI_OnLoad")) {
        reinterpret_cast<jint (*)(JavaVM *, void *)>(f)(CurrentVM, nullptr);
      }
    }
    return reinterpret_cast<jlong>(dylib);
  }
}

void Natives::dynamicLibraryRelease0(JNIEnv *env, jclass self, jlong handle) {
  auto *typedHandle = reinterpret_cast<polyregion_dl_handle>(handle);
  {
    const std::scoped_lock lock(dlMutex);
    if (const auto it = findBinding(env, typedHandle, self); it != bindings.end() && --it->holders == 0) {
      if (auto *f = polyregion_dl_find(typedHandle, "JNI_OnUnload")) {
        reinterpret_cast<void (*)(JavaVM *, void *)>(f)(CurrentVM, nullptr);
      }
      env->DeleteWeakGlobalRef(it->owner);
      bindings.erase(it);
    }
  }
  if (const auto code = polyregion_dl_close(typedHandle); code != 0) {
    throwGeneric(env, EX, "Cannot unload module:" + resolveDlError());
  }
}

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

static JavaVM *CurrentVM = {};

struct DroppedFile {
  jweak owner;
  jobject file;
};

static std::mutex filesMutex;
static std::vector<DroppedFile> files;

enum class State { Loading, Ready, Unloading };

struct Binding {
  polyregion_dl_handle handle;
  jweak owner;
  std::size_t holders;
  State state;
};

static std::mutex dlMutex;
static std::condition_variable dlChanged;
static std::vector<std::shared_ptr<Binding>> bindings;

static bool isDead(JNIEnv *env, jweak ref) { return env->IsSameObject(ref, nullptr); }

static std::vector<polyregion_dl_handle> purgeDead(JNIEnv *env) {
  std::vector<polyregion_dl_handle> holds;
  std::erase_if(bindings, [&](const std::shared_ptr<Binding> &b) {
    if (b->state != State::Ready || !isDead(env, b->owner)) return false;
    env->DeleteWeakGlobalRef(b->owner);
    holds.insert(holds.end(), b->holders, b->handle);
    return true;
  });
  return holds;
}

static std::shared_ptr<Binding> findBinding(JNIEnv *env, polyregion_dl_handle handle, jclass owner) {
  const auto it = std::find_if(bindings.begin(), bindings.end(),
                               [&](const auto &b) { return b->handle == handle && env->IsSameObject(b->owner, owner); });
  return it == bindings.end() ? nullptr : *it;
}

static void eraseBinding(JNIEnv *env, const std::shared_ptr<Binding> &binding) {
  std::erase(bindings, binding);
  env->DeleteWeakGlobalRef(binding->owner);
}

[[maybe_unused]] JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
  if (const char *d = std::getenv(polyregion::env::PolyregionDebug); d && *d) fprintf(stdout, "Shim JNI_OnLoad\n");
  CurrentVM = vm;

  JNIEnv *env = getEnv(vm);
  if (!env || !Natives::registerMethods(env)) return JNI_ERR;
  return JNI_VERSION_1_1;
}

[[maybe_unused]] JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *) {
  const bool debug = [] {
    const char *d = std::getenv(polyregion::env::PolyregionDebug);
    return d && *d;
  }();
  if (debug) fprintf(stdout, "Shim JNI_OnUnload\n");
  JNIEnv *env = getEnv(vm);

  std::vector<DroppedFile> dropped;
  {
    const std::scoped_lock lock(filesMutex);
    std::erase_if(files, [&](const DroppedFile &f) {
      if (!isDead(env, f.owner)) return false;
      dropped.push_back(f);
      return true;
    });
  }
  if (debug) fprintf(stdout, "Registered files: %zu\n", dropped.size());
  if (dropped.empty()) return;
  auto File = polyregion::generated::File::of(env);
  for (auto &f : dropped) {
    File.wrap(env, f.file).delete_(env);
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteGlobalRef(f.file);
    env->DeleteWeakGlobalRef(f.owner);
  }
}

void Natives::registerFilesToDropOnUnload0(JNIEnv *env, jclass self, jobject file) {
  const std::scoped_lock lock(filesMutex);
  files.push_back({env->NewWeakGlobalRef(self), env->NewGlobalRef(file)});
}

static std::string resolveDlError() {
  const char *message = polyregion_dl_error();
  return message ? std::string(message) : "(no message reported)";
}

jlong Natives::dynamicLibraryLoad0(JNIEnv *env, jclass self, jstring name) {
  const auto str = fromJni(env, name);
  auto *dylib = polyregion_dl_open(str.c_str());
  if (!dylib) {
    throwGeneric(env, EX, "Cannot load library `" + str + "` :" + resolveDlError());
    return {};
  }

  std::vector<polyregion_dl_handle> deadHolds;
  std::shared_ptr<Binding> loading;
  {
    std::unique_lock lock(dlMutex);
    deadHolds = purgeDead(env);
    while (!loading) {
      if (auto binding = findBinding(env, dylib, self); !binding) {
        loading = std::make_shared<Binding>(Binding{dylib, env->NewWeakGlobalRef(self), 1, State::Loading});
        bindings.push_back(loading);
      } else if (binding->state == State::Ready) {
        binding->holders++;
        break;
      } else {
        dlChanged.wait(lock);
      }
    }
  }
  for (auto *hold : deadHolds)
    polyregion_dl_close(hold);

  if (loading) {
    auto *onLoad = polyregion_dl_find(dylib, "JNI_OnLoad");
    const bool failed = onLoad && reinterpret_cast<jint (*)(JavaVM *, void *)>(onLoad)(CurrentVM, nullptr) == JNI_ERR;
    {
      const std::scoped_lock lock(dlMutex);
      if (failed) eraseBinding(env, loading);
      else loading->state = State::Ready;
    }
    dlChanged.notify_all();
    if (failed) {
      polyregion_dl_close(dylib);
      if (!env->ExceptionCheck()) throwGeneric(env, EX, "Library `" + str + "` failed to initialise");
      return {};
    }
  }
  return reinterpret_cast<jlong>(dylib);
}

void Natives::dynamicLibraryRelease0(JNIEnv *env, jclass self, jlong handle) {
  auto *typedHandle = reinterpret_cast<polyregion_dl_handle>(handle);
  std::shared_ptr<Binding> unloading;
  {
    const std::scoped_lock lock(dlMutex);
    const auto binding = findBinding(env, typedHandle, self);
    if (!binding || binding->state != State::Ready) return;
    if (--binding->holders == 0) {
      binding->state = State::Unloading;
      unloading = binding;
    }
  }
  if (unloading) {
    if (auto *f = polyregion_dl_find(typedHandle, "JNI_OnUnload")) {
      reinterpret_cast<void (*)(JavaVM *, void *)>(f)(CurrentVM, nullptr);
    }
    {
      const std::scoped_lock lock(dlMutex);
      eraseBinding(env, unloading);
    }
    dlChanged.notify_all();
  }
  if (const auto code = polyregion_dl_close(typedHandle); code != 0) {
    throwGeneric(env, EX, "Cannot unload module:" + resolveDlError());
  }
}

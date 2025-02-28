#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "polyregion/compat.h"

#include "polyinvoke/object_platform.h"
#include "polyregion/llvm_utils.hpp"

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/TargetParser/Host.h"

// XXX Make sure this goes last as libffi pollutes the global namespace with macros
#include "ffi_wrapped.h"

using namespace polyregion::invoke;
using namespace polyregion::invoke::object;

static void ffiInvoke(const char *prefix, uint64_t symbolAddress, const std::vector<Type> &types, std::vector<void *> &args) {
  const auto toFFITpe = [prefix](const Type &tpe) -> ffi_type * {
    switch (tpe) {
      case Type::Bool1:
      case Type::IntS8: return &ffi_type_sint8;
      case Type::IntU16: return &ffi_type_uint16;
      case Type::IntS16: return &ffi_type_sint16;
      case Type::IntS32: return &ffi_type_sint32;
      case Type::IntS64: return &ffi_type_sint64;
      case Type::Float32: return &ffi_type_float;
      case Type::Float64: return &ffi_type_double;
      case Type::Ptr: return &ffi_type_pointer;
      case Type::Void: return &ffi_type_void;
      default: POLYINVOKE_FATAL(prefix, "Illegal ffi type: %s", to_string(tpe).data());
    }
  };
  if (types.size() != args.size()) POLYINVOKE_FATAL(prefix, "types size (%zu) != args size (%zu)", types.size(), args.size());

  std::vector<ffi_type *> ffiTypes(args.size());
  for (size_t i = 0; i < args.size(); i++)
    ffiTypes[i] = toFFITpe(types[i]);
  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size() - 1, ffiTypes.back(), ffiTypes.data());
  switch (status) {
    case FFI_OK: ffi_call(&cif, FFI_FN(symbolAddress), args.back(), args.data()); break;
    case FFI_BAD_TYPEDEF: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: FFI_BAD_TYPEDEF (%d)", status);
    case FFI_BAD_ABI: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: FFI_BAD_ABI (%d)", status);
    default: POLYINVOKE_FATAL(prefix, "ffi_prep_cif: unknown error (%d)", status);
  }
}

int64_t ObjectDevice::id() {
  POLYINVOKE_TRACE();
  return 0;
}
bool ObjectDevice::sharedAddressSpace() {
  POLYINVOKE_TRACE();
  return true;
}
bool ObjectDevice::singleEntryPerModule() {
  POLYINVOKE_TRACE();
  return false;
}
std::vector<Property> ObjectDevice::properties() {
  POLYINVOKE_TRACE();
  return {};
}
std::vector<std::string> ObjectDevice::features() {
  POLYINVOKE_TRACE();

  std::vector<std::string> features;
  for (auto &F : llvm::sys::getHostCPUFeatures()) {
    if (F.second) features.push_back(F.first().str());
  }

  polyregion::llvm_shared::collectCPUFeatures(llvm::sys::getHostCPUName().str(),
                                              llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch(), features);

  return features;
}
uintptr_t ObjectDevice::mallocDevice(size_t size, Access) {
  POLYINVOKE_TRACE();
  return reinterpret_cast<uintptr_t>(std::malloc(size));
}
void ObjectDevice::freeDevice(uintptr_t ptr) {
  POLYINVOKE_TRACE();
  std::free(reinterpret_cast<void *>(ptr));
}

std::optional<void *> ObjectDevice::mallocShared(size_t size, Access access) {
  POLYINVOKE_TRACE();
  return std::malloc(size);
}
void ObjectDevice::freeShared(void *ptr) {
  POLYINVOKE_TRACE();
  std::free(ptr);
}

// ---

void ObjectDeviceQueue::enqueueDeviceToDeviceAsync(uintptr_t src, size_t srcOffset, uintptr_t dst, size_t dstOffset, size_t size,
                                                   const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(reinterpret_cast<void *>(dst + dstOffset), reinterpret_cast<void *>(src + srcOffset), size);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t dstOffset, size_t size, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(reinterpret_cast<void *>(dst + dstOffset), src, size);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, size_t srcOffset, void *dst, size_t bytes, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  std::memcpy(dst, reinterpret_cast<void *>(src + srcOffset), bytes);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueWaitBlocking() {
  POLYINVOKE_TRACE();
  latch.waitAll();
}
ObjectDeviceQueue::ObjectDeviceQueue(const std::chrono::duration<int64_t> &timeout) : latch(timeout) { POLYINVOKE_TRACE(); }

ObjectDeviceQueue::~ObjectDeviceQueue() noexcept { POLYINVOKE_TRACE(); }
std::variant<std::string, std::unique_ptr<Platform>> RelocatablePlatform::create() {
  return std::unique_ptr<Platform>(new RelocatablePlatform());
}
RelocatablePlatform::RelocatablePlatform() { POLYINVOKE_TRACE(); }
std::string RelocatablePlatform::name() {
  POLYINVOKE_TRACE();
  return "CPU (RelocatableObject)";
}
std::vector<Property> RelocatablePlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind RelocatablePlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::HostThreaded;
}
ModuleFormat RelocatablePlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::Object;
}
std::vector<std::unique_ptr<Device>> RelocatablePlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
  return xs;
}

static constexpr const char *RELOBJ_PREFIX = "RelocatableObject";

RelocatableDevice::RelocatableDevice() { POLYINVOKE_TRACE(); }

std::string RelocatableDevice::name() {
  POLYINVOKE_TRACE();
  return "RelocatableObjectDevice(llvm::RuntimeDyld)";
}

MemoryManager::MemoryManager() : SectionMemoryManager(nullptr) {}
details::LoadedCodeObject::LoadedCodeObject(std::unique_ptr<llvm::object::ObjectFile> obj) : ld(mm, mm), rawObject(std::move(obj)) {
  ld.loadObject(*this->rawObject);
}

void RelocatableDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  // polyregion::abort
  WriteLock rw(mutex);
  if (auto it = objects.find(name); it != objects.end()) {
    POLYINVOKE_FATAL(RELOBJ_PREFIX, "Module named %s was already loaded", name.c_str());
  } else {
    if (auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(llvm::StringRef(image), ""));
        auto e = object.takeError()) {
      POLYINVOKE_FATAL(RELOBJ_PREFIX, "Cannot load module: %s", toString(std::move(e)).c_str());
    } else {
      const auto inserted = objects.emplace_hint(it, name, std::make_unique<details::LoadedCodeObject>(std::move(*object)));
      if (inserted->second->ld.finalizeWithMemoryManagerLocking(); inserted->second->ld.hasError()) {
        POLYINVOKE_FATAL(RELOBJ_PREFIX, "Module `%s` failed to finalise for execution: %s", name.c_str(),
                         inserted->second->ld.getErrorString().data());
      }
    }
  }
}

bool RelocatableDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  ReadLock r(mutex);
  return objects.find(name) != objects.end();
}
std::unique_ptr<DeviceQueue> RelocatableDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<RelocatableDeviceQueue>(timeout, objects, mutex);
}

RelocatableDeviceQueue::RelocatableDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(objects) objects,
                                               decltype(mutex) mutex)
    : ObjectDeviceQueue(timeout), objects(objects), mutex(mutex) {
  POLYINVOKE_TRACE();
}

void *malloc_(size_t size) {
  auto p = std::malloc(size);
  fprintf(stderr, "kernel malloc(%zu) = %p\n", size, p);
  return p;
}

uint64_t MemoryManager::getSymbolAddress(const std::string &Name) {
  return Name == "malloc" ? reinterpret_cast<uint64_t>(&malloc_) : llvm::RTDyldMemoryManager::getSymbolAddress(Name);
}

template <typename F> static void threadedLaunch(detail::CountedCallbackHandler &handler, size_t N, const MaybeCallback &cb, F f) {
  static std::atomic_size_t counter(0);
  static std::unordered_map<size_t, std::atomic_size_t> pending;
  static std::shared_mutex pendingLock;

  //  auto cbHandle = cb ? handler.createHandle(*cb) : nullptr;

  //  auto id = counter++;
  //  WriteLock wPending(pendingLock);
  //  pending.emplace(id, N);
  //  for (size_t tid = 0; tid < N; ++tid) {
  //    std::thread([id, cb,  f, tid, &handler]() {
  //      f(tid);
  //        WriteLock rwPending(pendingLock);
  //        if (auto it = pending.find(id); it != pending.end()) {
  //          if (--it->second == 0) {
  //              if(cb) (*cb)();
  ////            handler.consume(cbHandle);
  //            pending.erase(id);
  //          }
  //        }
  //    }).detach();
  //  }

  auto id = counter++;
  WriteLock wPending(pendingLock);
  pending.emplace(id, N);
  for (size_t tid = 0; tid < N; ++tid) {
    // arena.enqueue([id, tid, f, cb]() {
    f(tid);
    WriteLock rwPending(pendingLock);
    if (auto it = pending.find(id); it != pending.end()) {
      if (--it->second == 0) {
        pending.erase(id);
        if (cb) (*cb)();
        //            detail::CountedCallbackHandler::consume(cbHandle);
      }
    }
    // });
  }
}

void validatePolicyAndArgs(const char *prefix, std::vector<Type> types, const Policy &policy) {
  if (auto scratchCount = std::count(types.begin(), types.end(), Type::Scratch); scratchCount != 0)
    POLYINVOKE_FATAL(prefix, "Scratch types are not supported on the CPU, found %td arg(s)", scratchCount);
  if (policy.global.y != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Y > 1 is not supported: %zu", policy.global.y);
  if (policy.global.z != 1) POLYINVOKE_FATAL(prefix, "Policy dimension Z > 1 is not supported: %zu", policy.global.z);
  if (policy.local) POLYINVOKE_FATAL(prefix, "Policy local dimension is not supported: size=%zu", policy.local->second);
  if (types[0] != Type::IntS64)
    POLYINVOKE_FATAL(prefix, "Expecting first argument as index (%s), but was %s", to_string(Type::IntS64).data(),
                     to_string(types[0]).data());
}
void RelocatableDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                                std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  validatePolicyAndArgs(RELOBJ_PREFIX, types, policy);

  ReadLock r(mutex);
  const auto moduleIt = objects.find(moduleName);
  if (moduleIt == objects.end()) POLYINVOKE_FATAL(RELOBJ_PREFIX, "No module named %s was loaded", moduleName.c_str());
  auto &[_, obj] = *moduleIt;
  const auto fnName = (obj->rawObject->isMachO() || obj->rawObject->isMachOUniversalBinary()) ? std::string("_") + symbol : symbol;
  const auto sym = obj->ld.getSymbol(fnName);
  if (!sym) POLYINVOKE_FATAL(RELOBJ_PREFIX, "Symbol `%s` not found in the given object", fnName.c_str());
  this->threadedLaunch(
      policy.global.x,
      [cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      },
      [sym, types, argData](size_t tid) {
        auto argData_ = argData;
        auto argPtrs = detail::argDataAsPointers(types, argData_);
        if (types[0] != Type::IntS64) {
          POLYINVOKE_FATAL(RELOBJ_PREFIX, "Expecting first argument as index: %s, but was %s", to_string(Type::IntS64).data(),
                           to_string(types[0]).data());
        }
        auto _tid = int64_t(tid);
        argPtrs[0] = &_tid;
        ffiInvoke(RELOBJ_PREFIX, sym.getAddress(), types, argPtrs);
      });
}

static constexpr const char *SHOBJ_PREFIX = "SharedObject";

std::variant<std::string, std::unique_ptr<Platform>> SharedPlatform::create() { return std::unique_ptr<Platform>(new SharedPlatform()); }

SharedPlatform::SharedPlatform() { POLYINVOKE_TRACE(); }
std::string SharedPlatform::name() {
  POLYINVOKE_TRACE();
  return "CPU (SharedObject)";
}
std::vector<Property> SharedPlatform::properties() {
  POLYINVOKE_TRACE();
  return {};
}
PlatformKind SharedPlatform::kind() {
  POLYINVOKE_TRACE();
  return PlatformKind::HostThreaded;
}
ModuleFormat SharedPlatform::moduleFormat() {
  POLYINVOKE_TRACE();
  return ModuleFormat::Object;
}
std::vector<std::unique_ptr<Device>> SharedPlatform::enumerate() {
  POLYINVOKE_TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<SharedDevice>();
  return xs;
}

SharedDevice::~SharedDevice() {
  POLYINVOKE_TRACE();
  for (auto &[_, m] : modules) {
    auto &[path, handle, symbols] = m;
    if (auto code = polyregion_dl_close(handle); code != 0) {
      std::fprintf(stderr, "%s Cannot unload module, code %d: %s\n", SHOBJ_PREFIX, code, polyregion_dl_error());
    }
  }
}
std::string SharedDevice::name() {
  POLYINVOKE_TRACE();
  return "SharedObjectDevice(dlopen/dlsym)";
}
void SharedDevice::loadModule(const std::string &name, const std::string &image) {
  POLYINVOKE_TRACE();
  if (auto it = modules.find(name); it != modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "Module named %s was already loaded", name.c_str());
  else {

    // TODO implement Linux: https://x-c3ll.github.io/posts/fileless-memfd_create/
    // TODO implement Windows: https://github.com/fancycode/MemoryModule

    // dlopen must open from a file :(
    auto tmpPath = std::tmpnam(nullptr);
    if (!tmpPath) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Unable to buffer image to file for %s, tmpfile creation failed: cannot synthesise temp path",
                       name.c_str());
    }
    std::FILE *objectFile = std::fopen(tmpPath, "wb");
    if (!objectFile) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Unable to buffer image to file for %s, tmpfile creation failed: %s", name.c_str(),
                       std::strerror(errno));
    }
    std::fwrite(image.data(), image.size(), 1, objectFile);
    std::fflush(objectFile);
    std::fclose(objectFile);
    static std::vector<std::string> tmpImagePaths;
    tmpImagePaths.emplace_back(tmpPath);
    static std::mutex cleanupMutex;
    static auto cleanUp = []() {
      std::unique_lock<std::mutex> lock(cleanupMutex);
      for (auto &path : tmpImagePaths) {
        if (std::remove(path.c_str()) != 0) {
          fprintf(stderr, "[%s] Warning: cannot remove temporary image file %s\n", SHOBJ_PREFIX, path.c_str());
        }
      }
      tmpImagePaths.clear();
    };
    std::atexit(cleanUp);
    std::set_terminate(cleanUp);

    if (auto dylib = polyregion_dl_open(tmpPath); !dylib) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load module: %s", polyregion_dl_error());
    } else modules.emplace_hint(it, name, details::LoadedModule{image, dylib, {}});
  }
}
bool SharedDevice::moduleLoaded(const std::string &name) {
  POLYINVOKE_TRACE();
  return modules.find(name) != modules.end();
}
std::unique_ptr<DeviceQueue> SharedDevice::createQueue(const std::chrono::duration<int64_t> &timeout) {
  POLYINVOKE_TRACE();
  return std::make_unique<SharedDeviceQueue>(timeout, modules, mutex);
}

SharedDeviceQueue::SharedDeviceQueue(const std::chrono::duration<int64_t> &timeout, decltype(modules) modules, decltype(mutex) mutex)
    : ObjectDeviceQueue(timeout), modules(modules), mutex(mutex) {
  POLYINVOKE_TRACE();
}
void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol, const std::vector<Type> &types,
                                           std::vector<std::byte> argData, const Policy &policy, const MaybeCallback &cb) {
  POLYINVOKE_TRACE();
  validatePolicyAndArgs(SHOBJ_PREFIX, types, policy);

  ReadLock r(mutex);
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end()) POLYINVOKE_FATAL(SHOBJ_PREFIX, "No module named %s was loaded", moduleName.c_str());

  auto &[image, handle, symbolTable] = moduleIt->second;

  void *address = nullptr;
  if (const auto it = symbolTable.find(symbol); it != symbolTable.end()) address = it->second;
  else {
    address = polyregion_dl_find(handle, symbol.c_str());
    auto err = polyregion_dl_error();
    if (err) {
      POLYINVOKE_FATAL(SHOBJ_PREFIX, "Cannot load symbol %s from module %s (%zd bytes): %s", //
                       symbol.c_str(), moduleName.c_str(), image.size(), err);
    }
    symbolTable.emplace_hint(it, symbol, address);
  }

  this->threadedLaunch(
      policy.global.x,
      [cb, token = latch.acquire()]() {
        if (cb) (*cb)();
      },
      [address, types, argData](size_t tid) {
        auto argData_ = argData;
        auto argPtrs = detail::argDataAsPointers(types, argData_);
        auto _tid = int64_t(tid);
        argPtrs[0] = &_tid;
        ffiInvoke(SHOBJ_PREFIX, reinterpret_cast<uint64_t>(address), types, argPtrs);
      });
  POLYINVOKE_TRACE();
}

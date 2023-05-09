#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "ffi_wrapped.h"
#include "object_platform.h"
#include "utils.hpp"

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"

#include "llvm_utils.hpp"

using namespace polyregion::runtime;
using namespace polyregion::runtime::object;

static void invoke(uint64_t symbolAddress, const std::vector<Type> &types, std::vector<void *> &args) {
  const auto toFFITpe = [](const Type &tpe) -> ffi_type * {
    switch (tpe) {
      case polyregion::runtime::Type::Bool8:
      case polyregion::runtime::Type::Byte8: return &ffi_type_sint8;
      case polyregion::runtime::Type::CharU16: return &ffi_type_uint16;
      case polyregion::runtime::Type::Short16: return &ffi_type_sint16;
      case polyregion::runtime::Type::Int32: return &ffi_type_sint32;
      case polyregion::runtime::Type::Long64: return &ffi_type_sint64;
      case polyregion::runtime::Type::Float32: return &ffi_type_float;
      case polyregion::runtime::Type::Double64: return &ffi_type_double;
      case polyregion::runtime::Type::Ptr: return &ffi_type_pointer;
      case polyregion::runtime::Type::Void: return &ffi_type_void;
      default: throw std::logic_error("Illegal ffi type " + std::to_string(polyregion::to_underlying(tpe)));
    }
  };
  if (types.size() != args.size()) throw std::logic_error("types size  != args size");

  std::vector<ffi_type *> ffiTypes(args.size());
  for (size_t i = 0; i < args.size(); i++)
    ffiTypes[i] = toFFITpe(types[i]);
  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size() - 1, ffiTypes.back(), ffiTypes.data());
  switch (status) {
    case FFI_OK: ffi_call(&cif, FFI_FN(symbolAddress), args.back(), args.data()); break;
    case FFI_BAD_TYPEDEF: throw std::logic_error("ffi_prep_cif: FFI_BAD_TYPEDEF");
    case FFI_BAD_ABI: throw std::logic_error("ffi_prep_cif: FFI_BAD_ABI");
    default: throw std::logic_error("ffi_prep_cif: unknown error (" + std::to_string(status) + ")");
  }
}

int64_t ObjectDevice::id() {
  TRACE();
  return 0;
}
bool ObjectDevice::sharedAddressSpace() {
  TRACE();
  return true;
}
bool ObjectDevice::singleEntryPerModule() {
  TRACE();
  return false;
}
std::vector<Property> ObjectDevice::properties() {
  TRACE();
  return {};
}
std::vector<std::string> ObjectDevice::features() {
  TRACE();

  std::vector<std::string> features;

  llvm::StringMap<bool> Features;
  llvm::sys::getHostCPUFeatures(Features);
  for (auto &F : Features) {
    if (F.second) features.push_back(F.first().str());
  }

  polyregion::llvm_shared::collectCPUFeatures(llvm::sys::getHostCPUName().str(),
                                              llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch(), features);

  return features;
}
uintptr_t ObjectDevice::malloc(size_t size, Access) {
  TRACE();
  return reinterpret_cast<uintptr_t>(std::malloc(size));
}
void ObjectDevice::free(uintptr_t ptr) {
  TRACE();
  std::free(reinterpret_cast<void *>(ptr));
}

// ---

void ObjectDeviceQueue::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  std::memcpy(reinterpret_cast<void *>(dst), src, size);
  if (cb) (*cb)();
}
void ObjectDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  std::memcpy(dst, reinterpret_cast<void *>(src), size);
  if (cb) (*cb)();
}

RelocatablePlatform::RelocatablePlatform() { TRACE(); }
std::string RelocatablePlatform::name() {
  TRACE();
  return "CPU (RelocatableObject)";
}
std::vector<Property> RelocatablePlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> RelocatablePlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
  return xs;
}

static constexpr const char *RELOBJ_ERROR_PREFIX = "[RelocatableObject error] ";

RelocatableDevice::RelocatableDevice() { TRACE(); }

std::string RelocatableDevice::name() {
  TRACE();
  return "RelocatableObjectDevice(llvm::RuntimeDyld)";
}

MemoryManager::MemoryManager() : SectionMemoryManager(nullptr) {}
LoadedCodeObject::LoadedCodeObject(std::unique_ptr<llvm::object::ObjectFile> obj)
    : ld(mm, mm), rawObject(std::move(obj)) {
  ld.loadObject(*this->rawObject);
}

void RelocatableDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  WriteLock rw(lock);
  if (auto it = objects.find(name); it != objects.end()) {
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    if (auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(llvm::StringRef(image), ""));
        auto e = object.takeError()) {
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Cannot load module: " + toString(std::move(e)));
    } else {
      auto inserted = objects.emplace_hint(it, name, std::make_unique<LoadedCodeObject>(std::move(*object)));
      if (inserted->second->ld.finalizeWithMemoryManagerLocking(); inserted->second->ld.hasError()) {
        throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Module `" + name +
                               "` failed to finalise for execution: " + inserted->second->ld.getErrorString().str());
      }
    }
  }
}

bool RelocatableDevice::moduleLoaded(const std::string &name) {
  TRACE();
  ReadLock r(lock);
  return objects.find(name) != objects.end();
}
std::unique_ptr<DeviceQueue> RelocatableDevice::createQueue() {
  TRACE();
  return std::make_unique<RelocatableDeviceQueue>(objects, lock);
}

RelocatableDeviceQueue::RelocatableDeviceQueue(decltype(objects) objects, decltype(lock) lock)
    : objects(objects), lock(lock) {
  TRACE();
}

void *malloc_(size_t size) {
  auto p = std::malloc(size);
  fprintf(stderr, "kernel malloc(%zu) = %p\n", size, p);
  return p;
}

uint64_t MemoryManager::getSymbolAddress(const std::string &Name) {
  return Name == "malloc" ? (uint64_t)&malloc_ : llvm::RTDyldMemoryManager::getSymbolAddress(Name);
}

template <typename F> static void threadedLaunch(size_t N, const MaybeCallback &cb, F f) {
  static std::atomic_size_t counter(0);
  static std::unordered_map<size_t, std::atomic_size_t> pending;
  static std::shared_mutex pendingLock;

  auto cbHandle = cb ? detail::CountedCallbackHandler::createHandle(*cb) : nullptr;

  auto id = counter++;
  WriteLock wPending(pendingLock);
  pending.emplace(id, N);

  for (size_t tid = 0; tid < N; ++tid) {
    std::thread([id, cbHandle, f, tid]() {
      f(tid);
      if (cbHandle) {
        WriteLock rwPending(pendingLock);
        if (auto it = pending.find(id); it != pending.end()) {
          if (--it->second == 0) {
            detail::CountedCallbackHandler::consume(cbHandle);
            pending.erase(id);
          }
        }
      }
    }).detach();
  }
}

void validatePolicyAndArgs(const char *prefix, std::vector<Type> types, const Policy &policy) {
  if (auto scratchCount = std::count(types.begin(), types.end(), Type::Scratch); scratchCount != 0) {
    throw std::logic_error(std::string(prefix) + "Scratch types are not supported on the CPU, found" +
                           std::to_string(scratchCount) + " arg(s)");
  }
  if (policy.global.y != 1) {
    throw std::logic_error(std::string(prefix) + "Policy dimension Y > 1 is not supported");
  }
  if (policy.global.z != 1) {
    throw std::logic_error(std::string(prefix) + "Policy dimension Z > 1 is not supported");
  }
  if (policy.local) {
    throw std::logic_error(std::string(prefix) + "Policy local dimension is not supported");
  }
  if (types[0] != Type::Long64) {
    throw std::logic_error(std::string(prefix) + "Expecting first argument as index: " + typeName(Type::Long64) +
                           ", but was " + typeName(types[0]));
  }
}

void RelocatableDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                                const std::vector<Type> &types, std::vector<std::byte> argData,
                                                const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  validatePolicyAndArgs(RELOBJ_ERROR_PREFIX, types, policy);

  ReadLock r(lock);
  const auto moduleIt = objects.find(moduleName);
  if (moduleIt == objects.end())
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[_, obj] = *moduleIt;
  auto fnName =
      (obj->rawObject->isMachO() || obj->rawObject->isMachOUniversalBinary()) ? std::string("_") + symbol : symbol;
  auto sym = obj->ld.getSymbol(fnName);
  if (!sym) {
    auto table = obj->ld.getSymbolTable();
    std::vector<std::string> symbols;
    symbols.reserve(table.size());
    for (auto &[k, v] : table)
      symbols.emplace_back("[`" + k.str() + "`@" + polyregion::hex(v.getAddress()) + "]");
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Symbol `" + std::string(fnName) +
                           "` not found in the given object, available symbols (" + std::to_string(table.size()) +
                           ") = " +
                           polyregion::mk_string<std::string>(
                               symbols, [](auto &x) { return x; }, ","));
  }

  threadedLaunch(policy.global.x, cb, [sym, types, argData](size_t tid) {
    auto argData_ = argData;
    auto argPtrs = detail::argDataAsPointers(types, argData_);
    if (types[0] != Type::Long64) {
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Expecting first argument as index: " +
                             typeName(Type::Long64) + ", but was " + typeName(types[0]));
    }
    auto _tid = int64_t(tid);
    argPtrs[0] = &_tid;
    invoke(sym.getAddress(), types, argPtrs);
  });
}

static constexpr const char *SHOBJ_ERROR_PREFIX = "[SharedObject error] ";
SharedPlatform::SharedPlatform() { TRACE(); }
std::string SharedPlatform::name() {
  TRACE();
  return "CPU (SharedObject)";
}
std::vector<Property> SharedPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> SharedPlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<SharedDevice>();
  return xs;
}

SharedDevice::~SharedDevice() {
  TRACE();
  for (auto &[_, m] : modules) {
    auto &[path, handle, symbols] = m;
    if (auto code = polyregion_dl_close(handle); code != 0) {
      std::fprintf(stderr, "%s Cannot unload module, code %d: %s\n", SHOBJ_ERROR_PREFIX, code, polyregion_dl_error());
    }
  }
}
std::string SharedDevice::name() {
  TRACE();
  return "SharedObjectDevice(dlopen/dlsym)";
}
void SharedDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  if (auto it = modules.find(name); it != modules.end()) {
    throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {

    // TODO implement Linux: https://x-c3ll.github.io/posts/fileless-memfd_create/
    // TODO implement Windows: https://github.com/fancycode/MemoryModule

    // dlopen must open from a file :(
    auto tmpPath = std::tmpnam(nullptr);
    if (!tmpPath) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) +
                             "Unable to buffer image to file, tmpfile creation failed: cannot synthesise temp path");
    }
    std::FILE *objectFile = std::fopen(tmpPath, "wb");
    if (!objectFile) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) +
                             "Unable to buffer image to file, tmpfile creation failed: " + std::strerror(errno));
    }
    std::fwrite(image.data(), image.size(), 1, objectFile);
    std::fflush(objectFile);
    std::fclose(objectFile);
    static std::vector<std::string> tmpImagePaths;
    tmpImagePaths.emplace_back(tmpPath);
    static std::mutex mutex;
    static auto cleanUp = []() {
      std::unique_lock<std::mutex> lock(mutex);
      for (auto &path : tmpImagePaths) {
        if (std::remove(path.c_str()) != 0) {
          fprintf(stderr, "Warning: cannot remove temporary image file %s\n", path.c_str());
        }
      }
      tmpImagePaths.clear();
    };
    std::atexit(cleanUp);
    std::set_terminate(cleanUp);

    if (auto dylib = polyregion_dl_open(tmpPath); !dylib) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) +
                             "Cannot load module: " + std::string(polyregion_dl_error()));
    } else
      modules.emplace_hint(it, name, LoadedModule{image, dylib, {}});
  }
}
bool SharedDevice::moduleLoaded(const std::string &name) {
  TRACE();
  return modules.find(name) != modules.end();
}
std::unique_ptr<DeviceQueue> SharedDevice::createQueue() {
  TRACE();
  return std::make_unique<SharedDeviceQueue>(modules, lock);
}

SharedDeviceQueue::SharedDeviceQueue(decltype(modules) modules, decltype(lock) lock) : modules(modules), lock(lock) {
  TRACE();
}
void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                           const std::vector<Type> &types, std::vector<std::byte> argData,
                                           const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  validatePolicyAndArgs(SHOBJ_ERROR_PREFIX, types, policy);

  ReadLock r(lock);
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[image, handle, symbolTable] = moduleIt->second;

  void *address = nullptr;
  if (auto it = symbolTable.find(symbol); it != symbolTable.end()) address = it->second;
  else {
    address = polyregion_dl_find(handle, symbol.c_str());
    auto err = polyregion_dl_error();
    if (err) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "Cannot load symbol " + symbol + " from module " +
                             moduleName + " (" + std::to_string(image.size()) + " bytes): " + std::string(err));
    }
    symbolTable.emplace_hint(it, symbol, address);
  }

  threadedLaunch(policy.global.x, cb, [address, types, argData](size_t tid) {
    auto argData_ = argData;
    auto argPtrs = detail::argDataAsPointers(types, argData_);
    auto _tid = int64_t(tid);
    argPtrs[0] = &_tid;
    invoke(reinterpret_cast<uint64_t>(address), types, argPtrs);
  });
  TRACE();
}

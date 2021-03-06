#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "cpuinfo.h"
#include "ffi.h"
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
void RelocatableDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
  WriteLock rw(lock);
  if (auto it = objects.find(name); it != objects.end()) {
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    if (auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(llvm::StringRef(image), ""));
        auto e = object.takeError()) {
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Cannot load module: " + toString(std::move(e)));
    } else
      objects.emplace_hint(it, name, std::move(*object));
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

MemoryManager::MemoryManager() : SectionMemoryManager(nullptr) {}
uint64_t MemoryManager::getSymbolAddress(const std::string &Name) {
  return Name == "malloc" ? (uint64_t)&std::malloc : llvm::RTDyldMemoryManager::getSymbolAddress(Name);
}

void RelocatableDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                                std::vector<Type> types, std::vector<std::byte> argData,
                                                const Policy &policy, const MaybeCallback &cb) {
  TRACE();

  RelocatableDevice::ReadLock r(lock);
  const auto moduleIt = objects.find(moduleName);
  if (moduleIt == objects.end())
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  std::thread([symbol, types, argData, cb, &obj = moduleIt->second]() {
    MemoryManager mm;
    llvm::RuntimeDyld ld(mm, mm);

    ld.loadObject(*obj);
    auto fnName = (obj->isMachO() || obj->isMachOUniversalBinary()) ? std::string("_") + symbol : symbol;

    if (auto sym = ld.getSymbol(fnName); !sym) {
      auto table = ld.getSymbolTable();
      std::vector<std::string> symbols;
      symbols.reserve(table.size());
      for (auto &[k, v] : table)
        symbols.emplace_back("[`" + k.str() + "`@" + polyregion::hex(v.getAddress()) + "]");
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Symbol `" + std::string(fnName) +
                             "` not found in the given object, available symbols (" + std::to_string(table.size()) +
                             ") = " +
                             polyregion::mk_string<std::string>(
                                 symbols, [](auto &x) { return x; }, ","));
    } else {

      if (ld.finalizeWithMemoryManagerLocking(); ld.hasError()) {
        throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Symbol `" + std::string(symbol) +
                               "` failed to finalise for execution: " + ld.getErrorString().str());
      }

      auto argData_ = argData;
      auto argPtrs = detail::argDataAsPointers(types, argData_);
      invoke(sym.getAddress(), types, argPtrs);
      if (cb) (*cb)();
    }
  }).detach();
}

static constexpr const char *SHOBJ_ERROR_PREFIX = "[RelocatableObject error] ";
SharedPlatform::SharedPlatform() { TRACE(); }
std::string SharedPlatform::name() {
  TRACE();
  return "CPU (SharedObjectR)";
}
std::vector<Property> SharedPlatform::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> SharedPlatform::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
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
    if (auto dylib = polyregion_dl_open(image.c_str()); !dylib) {
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
  return std::make_unique<SharedDeviceQueue>(modules);
}

SharedDeviceQueue::SharedDeviceQueue(decltype(modules) modules) : modules(modules) { TRACE(); }
void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                           std::vector<Type> types, std::vector<std::byte> argData,
                                           const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[path, handle, symbolTable] = moduleIt->second;

  void *address = nullptr;
  if (auto it = symbolTable.find(symbol); it != symbolTable.end()) address = it->second;
  else {
    address = polyregion_dl_find(handle, moduleName.c_str());
    auto err = polyregion_dl_error();
    if (err) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "Cannot load symbol " + symbol + " from module " +
                             moduleName + " (" + path + "): " + std::string(err));
    }
    symbolTable.emplace_hint(it, symbol, address);
  }
  auto args = detail::argDataAsPointers(types, argData);
  invoke(reinterpret_cast<uint64_t>(address), types, args);
  if (cb) (*cb)();
}

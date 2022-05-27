#include <system_error>
#include <utility>

#include "ffi.h"
#include "object_runtime.h"
#include "utils.hpp"

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"

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
std::vector<Property> ObjectDevice::properties() {
  TRACE();
  return {};
}
uintptr_t ObjectDevice::malloc(size_t size, Access access) {
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
  if (cb) (*cb)(); // no-op for CPUs
}
void ObjectDeviceQueue::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const MaybeCallback &cb) {
  TRACE();
  std::memcpy(dst, reinterpret_cast<void *>(src), size);
  if (cb) (*cb)(); // no-op for CPUs
}

RelocatableRuntime::RelocatableRuntime() { TRACE(); }
std::string RelocatableRuntime::name() {
  TRACE();
  return "CPU (RelocatableObject)";
}
std::vector<Property> RelocatableRuntime::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> RelocatableRuntime::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
  return xs;
}

static constexpr const char *RELOBJ_ERROR_PREFIX = "[RelocatableObject error] ";
RelocatableDevice::RelocatableDevice() : llvm::SectionMemoryManager(nullptr), ld(*this, *this) { TRACE(); }
uint64_t RelocatableDevice::getSymbolAddress(const std::string &Name) {
  auto self = this;
  thread_local static std::function<void *(size_t)> threadLocalMallocFn = [&self](size_t size) {
    return self ? reinterpret_cast<void *>(self->malloc(size, Access::RW)) : nullptr;
  };
  return Name == "malloc" ? (uint64_t)&threadLocalMallocFn : llvm::RTDyldMemoryManager::getSymbolAddress(Name);
}
std::string RelocatableDevice::name() {
  TRACE();
  return "RelocatableObjectDevice(llvm::RuntimeDyld)";
}
void RelocatableDevice::loadModule(const std::string &name, const std::string &image) {
  TRACE();
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
std::unique_ptr<DeviceQueue> RelocatableDevice::createQueue() {
  TRACE();
  return std::make_unique<RelocatableDeviceQueue>(objects, ld);
}

RelocatableDeviceQueue::RelocatableDeviceQueue(decltype(objects) objects, decltype(ld) ld) : objects(objects), ld(ld) {}
void RelocatableDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                                const std::vector<Type> &types, std::vector<void *> &args,
                                                const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  auto moduleIt = objects.find(moduleName);
  if (moduleIt == objects.end())
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &obj = moduleIt->second;
  ld.loadObject(*obj);
  auto fnName = (obj->isMachO() || obj->isMachOUniversalBinary()) ? std::string("_") + symbol : symbol;

  if (auto sym = ld.getSymbol(fnName); !sym) {
    auto table = ld.getSymbolTable();
    auto symbols = polyregion::mk_string2<llvm::StringRef, llvm::JITEvaluatedSymbol>(
        table, [](auto &x) { return "[`" + x.first.str() + "`@" + polyregion::hex(x.second.getAddress()) + "]"; }, ",");
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Symbol `" + std::string(fnName) +
                           "` not found in the given object, available symbols (" + std::to_string(table.size()) +
                           ") = " + symbols);
  } else {
    if (ld.finalizeWithMemoryManagerLocking(); ld.hasError()) {
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Symbol `" + std::string(symbol) +
                             "` failed to finalise for execution: " + ld.getErrorString().str());
    }
    invoke(sym.getAddress(), types, args);
    if (cb) (*cb)();
  }
}

static constexpr const char *SHOBJ_ERROR_PREFIX = "[RelocatableObject error] ";
SharedRuntime::SharedRuntime() { TRACE(); }
std::string SharedRuntime::name() {
  TRACE();
  return "CPU (SharedObjectR)";
}
std::vector<Property> SharedRuntime::properties() {
  TRACE();
  return {};
}
std::vector<std::unique_ptr<Device>> SharedRuntime::enumerate() {
  TRACE();
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableDevice>();
  return xs;
}
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <windows.h>

  #define dynamic_library_open(path) LoadLibraryA(path)
  #define dynamic_library_error() std::system_category().message(::GetLastError())
  #define dynamic_library_close(lib) FreeLibrary(lib)
  #define dynamic_library_find(lib, symbol) GetProcAddress(lib, symbol)
#else
  #include <dlfcn.h>
//
  #define dynamic_library_open(path) dlopen(path, RTLD_NOW)
  #define dynamic_library_error() dlerror()
  #define dynamic_library_close(lib) dlclose(lib)
  #define dynamic_library_find(lib, symbol) dlsym(lib, symbol)
#endif
SharedDevice::~SharedDevice() {
  TRACE();
  for (auto &[_, m] : modules) {
    auto &[path, handle, symbols] = m;
    if (auto code = dynamic_library_close(handle); code != 0) {
      std::fprintf(stderr, "%s Cannot unload module, code %d: %s\n", SHOBJ_ERROR_PREFIX, code, dynamic_library_error());
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
    if (auto dylib = dynamic_library_open(image.c_str()); !dylib) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) +
                             "Cannot load module: " + std::string(dynamic_library_error()));
    } else
      modules.emplace_hint(it, name, LoadedModule{image, dylib, {}});
  }
}
std::unique_ptr<DeviceQueue> SharedDevice::createQueue() {
  TRACE();
  return std::make_unique<SharedDeviceQueue>(modules);
}

SharedDeviceQueue::SharedDeviceQueue(decltype(modules) modules) : modules(modules) { TRACE(); }
void SharedDeviceQueue::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                           const std::vector<Type> &types, std::vector<void *> &args,
                                           const Policy &policy, const MaybeCallback &cb) {
  TRACE();
  auto moduleIt = modules.find(moduleName);
  if (moduleIt == modules.end())
    throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "No module named " + moduleName + " was loaded");

  auto &[path, handle, symbolTable] = moduleIt->second;

  void *address = nullptr;
  if (auto it = symbolTable.find(symbol); it != symbolTable.end()) address = it->second;
  else {
    address = dynamic_library_find(handle, moduleName.c_str());
    auto err = dynamic_library_error();
    if (err) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "Cannot load symbol " + symbol + " from module " +
                             moduleName + " (" + path + "): " + std::string(err));
    }
    symbolTable.emplace_hint(it, symbol, address);
  }
  invoke(reinterpret_cast<uint64_t>(address), types, args);
  if (cb) (*cb)();
}
#undef dynamic_library_open
#undef dynamic_library_close
#undef dynamic_library_find
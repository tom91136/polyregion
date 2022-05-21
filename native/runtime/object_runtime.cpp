#include <system_error>

#include "ffi.h"
#include "object_runtime.h"
#include "utils.hpp"

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace polyregion::runtime;
using namespace polyregion::runtime::object;

static void invoke(uint64_t symbolAddress, const std::vector<TypedPointer> &args, TypedPointer rtn) {

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
      default: return nullptr;
    }
  };

  auto rtnFFIType = toFFITpe(rtn.first);
  if (!rtnFFIType) {
    throw std::logic_error("Illegal return type " + std::to_string(polyregion::to_underlying(rtn.first)));
  }

  std::vector<void *> argPointers(args.size());
  std::vector<ffi_type *> argsFFIType(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    argPointers[i] = args[i].second;
    argsFFIType[i] = toFFITpe(args[i].first);
    if (!argsFFIType[i])
      throw std::logic_error("Illegal parameter type on arg " + std::to_string(i) + ": " +
                             std::to_string(polyregion::to_underlying(args[i].first)));
  }

  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, args.size(), rtnFFIType, argsFFIType.data());
  switch (status) {
    case FFI_OK: ffi_call(&cif, FFI_FN(symbolAddress), rtn.second, argPointers.data()); break;
    case FFI_BAD_TYPEDEF: throw std::logic_error("ffi_prep_cif: FFI_BAD_TYPEDEF");
    case FFI_BAD_ABI: throw std::logic_error("ffi_prep_cif: FFI_BAD_ABI");
    default: throw std::logic_error("ffi_prep_cif: unknown error (" + std::to_string(status) + ")");
  }
}

int64_t ObjectDevice::id() { return 0; }
std::vector<Property> ObjectDevice::properties() { return {}; }
uintptr_t ObjectDevice::malloc(size_t size, Access access) { return reinterpret_cast<uintptr_t>(std::malloc(size)); }
void ObjectDevice::free(uintptr_t ptr) { std::free(reinterpret_cast<void *>(ptr)); }
void ObjectDevice::enqueueHostToDeviceAsync(const void *src, uintptr_t dst, size_t size,
                                            const std::optional<Callback> &cb) {
  // no-op for CPUs
  if (cb) (*cb)();
}
void ObjectDevice::enqueueDeviceToHostAsync(uintptr_t src, void *dst, size_t size, const std::optional<Callback> &cb) {
  // no-op for CPUs
  if (cb) (*cb)();
}

RelocatableObjectRuntime::RelocatableObjectRuntime() = default;
std::string RelocatableObjectRuntime::name() { return "CPU (RelocatableObject)"; }
std::vector<Property> RelocatableObjectRuntime::properties() { return {}; }
std::vector<std::unique_ptr<Device>> RelocatableObjectRuntime::enumerate() {
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableObjectDevice>();
  return xs;
}

static constexpr const char *RELOBJ_ERROR_PREFIX = "[RelocatableObject error] ";
RelocatableObjectDevice::RelocatableObjectDevice() : SectionMemoryManager(nullptr), ld(*this, *this) {}
uint64_t RelocatableObjectDevice::getSymbolAddress(const std::string &Name) {
  auto self = this;
  thread_local static std::function<void *(size_t)> threadLocalMallocFn = [&self](size_t size) {
    return self ? reinterpret_cast<void *>(self->malloc(size, Access::RW)) : nullptr;
  };
  return Name == "malloc" ? (uint64_t)&threadLocalMallocFn : RTDyldMemoryManager::getSymbolAddress(Name);
}
std::string RelocatableObjectDevice::name() { return "RelocatableObjectDevice(llvm::RuntimeDyld)"; }
void RelocatableObjectDevice::loadModule(const std::string &name, const std::string &image) {
  if (auto it = objects.find(name); it != objects.end()) {
    throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {

    auto ref = llvm::MemoryBufferRef{llvm::StringRef(image), ""};
    printf("n=%d  %d\n", ref.getBufferSize(), image.length());
    if (auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef{llvm::StringRef(image), ""});
        auto e = object.takeError()) {
      throw std::logic_error(std::string(RELOBJ_ERROR_PREFIX) + "Cannot load module: " + toString(std::move(e)));
    } else
      objects.emplace_hint(it, name, std::move(*object));
  }
}
void RelocatableObjectDevice::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                                 const std::vector<TypedPointer> &args, TypedPointer rtn,
                                                 const Policy &policy, const std::optional<Callback> &cb) {
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
    invoke(sym.getAddress(), args, rtn);
    if (cb) (*cb)();
  }
}

static constexpr const char *SHOBJ_ERROR_PREFIX = "[RelocatableObject error] ";

SharedObjectRuntime::SharedObjectRuntime() = default;
std::string SharedObjectRuntime::name() { return "CPU (SharedObjectR)"; }
std::vector<Property> SharedObjectRuntime::properties() { return {}; }
std::vector<std::unique_ptr<Device>> SharedObjectRuntime::enumerate() {
  std::vector<std::unique_ptr<Device>> xs(1);
  xs[0] = std::make_unique<RelocatableObjectDevice>();
  return xs;
}

std::string SharedObjectDevice::name() { return "SharedObjectDevice(dlopen/dlsym)"; }

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

void SharedObjectDevice::loadModule(const std::string &name, const std::string &image) {
  if (auto it = objects.find(name); it != objects.end()) {
    throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) + "Module named " + name + " was already loaded");
  } else {
    if (auto dylib = dynamic_library_open(image.c_str()); !dylib) {
      throw std::logic_error(std::string(SHOBJ_ERROR_PREFIX) +
                             "Cannot load module: " + std::string(dynamic_library_error()));
    } else
      objects.emplace_hint(it, name, LoadedModule{image, dylib, {}});
  }
}
void SharedObjectDevice::enqueueInvokeAsync(const std::string &moduleName, const std::string &symbol,
                                            const std::vector<TypedPointer> &args, TypedPointer rtn,
                                            const Policy &policy, const std::optional<Callback> &cb) {
  auto moduleIt = objects.find(moduleName);
  if (moduleIt == objects.end())
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
  invoke(reinterpret_cast<uint64_t>(address), args, rtn);
  if (cb) (*cb)();
}

SharedObjectDevice::~SharedObjectDevice() {

  for (auto &[_, m] : objects) {
    auto &[path, handle, symbols] = m;
    if (auto code = dynamic_library_close(handle); code != 0) {
      std::fprintf(stderr, "%s Cannot unload module, code %d: %s\n", SHOBJ_ERROR_PREFIX, code, dynamic_library_error());
    }
  }
}

#undef dynamic_library_open
#undef dynamic_library_close
#undef dynamic_library_find
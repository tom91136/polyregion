#include "runtime.h"

#include "ffi.h"
#include "libm.h"
#include "utils.hpp"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

using namespace polyregion;

struct runtime::Data {
  std::unique_ptr<llvm::object::ObjectFile> file;
  explicit Data(std::unique_ptr<llvm::object::ObjectFile> file) : file(std::move(file)) {}
};

polyregion::runtime::Object::~Object() = default;

polyregion::runtime::Object::Object(const std::vector<uint8_t> &object) {
  llvm::MemoryBufferRef buffer(llvm::StringRef(reinterpret_cast<const char *>(object.data()), object.size()), "");
  auto objfile = llvm::object::ObjectFile::createObjectFile(buffer);
  if (auto e = objfile.takeError()) {
    throw std::logic_error(toString(std::move(e)));
  } else {
    this->data = std::make_unique<runtime::Data>(std::move(*objfile));
  }
}

std::vector<std::pair<std::string, uint64_t>> polyregion::runtime::Object::enumerate() {
  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*data->file);
  auto table = ld.getSymbolTable();
  std::vector<std::pair<std::string, uint64_t>> result(table.size());
  std::transform(table.begin(), table.end(), result.begin(),
                 [](auto &p) { return std::make_pair(p.first, p.second.getAddress()); });
  return result;
}
void polyregion::runtime::Object::invoke(const std::string &symbol, const std::vector<TypedPointer> &args,
                                         runtime::TypedPointer rtn) {

  static_assert(sizeof(uint8_t) == sizeof(char));

  const auto toFFITpe = [&](const runtime::Type &tpe) -> ffi_type * {
    switch (tpe) {
    case Type::Bool:
      return &ffi_type_sint8;
    case Type::Byte:
      return &ffi_type_sint8;
    case Type::Char:
      return &ffi_type_uint8;
    case Type::Short:
      return &ffi_type_sint16;
    case Type::Int:
      return &ffi_type_sint32;
    case Type::Long:
      return &ffi_type_sint64;
    case Type::Float:
      return &ffi_type_float;
    case Type::Double:
      return &ffi_type_double;
    case Type::Ptr:
      return &ffi_type_pointer;
    case Type::Void:
      return &ffi_type_void;
    default:
      return nullptr;
    }
  };

  polyregion::libm::exportAll();

  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*data->file);

  auto sym = ld.getSymbol(symbol);

  if (!sym && (data->file->isMachO() || data->file->isMachOUniversalBinary())) {
    // Mach-O has a leading underscore for all exported symbols, we try again with that prepended
    sym = ld.getSymbol(std::string("_") + symbol);
  }

  if (!sym) {
    auto table = ld.getSymbolTable();
    auto symbols = polyregion::mk_string2<llvm::StringRef, llvm::JITEvaluatedSymbol>(
        table, [](auto &x) { return "[`" + x.first.str() + "`@" + polyregion::hex(x.second.getAddress()) + "]"; }, ",");
    throw std::logic_error("Symbol `" + std::string(symbol) + "` not found in the given object, available symbols (" +
                           std::to_string(table.size()) + ") = " + symbols);
  }

  ld.finalizeWithMemoryManagerLocking();

  if (ld.hasError()) {
    throw std::logic_error("Symbol `" + std::string(symbol) +
                           "` failed to finalise for execution: " + ld.getErrorString().str());
  }

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
  case FFI_OK:
    ffi_call(&cif, FFI_FN(sym.getAddress()), rtn.second, argPointers.data());
    break;
  case FFI_BAD_TYPEDEF:
    throw std::logic_error("ffi_prep_cif: FFI_BAD_TYPEDEF");
  case FFI_BAD_ABI:
    throw std::logic_error("ffi_prep_cif: FFI_BAD_ABI");
  default:
    throw std::logic_error("ffi_prep_cif: unknown error (" + std::to_string(status) + ")");
  }
}

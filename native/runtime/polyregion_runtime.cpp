#include "polyregion_runtime.h"
#include "ffi.h"
#include "utils.hpp"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

static constexpr char *error(const std::string &s) { return polyregion::new_str(s); }

struct polyregion_object {
  std::unique_ptr<llvm::object::ObjectFile> file;
  polyregion_object() = delete;
  polyregion_object(const polyregion_object &other) = delete;
  explicit polyregion_object(std::unique_ptr<llvm::object::ObjectFile> file) : file(std::move(file)) {}
};

void polyregion_release_object(polyregion_object_ref *ref) {
  if (ref) {
    polyregion::free_str(ref->message);
    delete ref->object;
    delete ref;
  }
}

polyregion_object_ref *polyregion_load_object(const uint8_t *object, size_t object_size) {
  llvm::MemoryBufferRef buffer(llvm::StringRef(reinterpret_cast<const char *>(object), object_size), "");
  auto ref = new polyregion_object_ref{};
  auto obj = llvm::object::ObjectFile::createObjectFile(buffer);
  if (auto e = obj.takeError()) {
    ref->message = error(toString(std::move(e)));
    return ref;
  }
  ref->object = new polyregion_object{std::move(*obj)};
  return ref;
}

void polyregion_release_enumerate(polyregion_symbol_table *table) {
  if (table) {
    for (size_t i = 0; i < table->size; ++i) {
      polyregion::free_str(table->symbols[i].name);
    }
    delete[] table->symbols;
    delete table;
  }
}

polyregion_symbol_table *polyregion_enumerate(const polyregion_object *object) {
  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*object->file);
  auto table = ld.getSymbolTable();

  auto xs = new polyregion_symbol[table.size()];

  std::transform(table.begin(), table.end(), xs, [](auto &p) {
    // copy name here as the symbol table is deleted with dyld
    return polyregion_symbol{polyregion::new_str(p.first.str()), p.second.getAddress()};
  });
  return new polyregion_symbol_table{xs, table.size()};
}

void polyregion_release_invoke(char *err) { polyregion::free_str(err); }

char *polyregion_invoke(const polyregion_object *object,
                        const char *symbol,                        //
                        const polyregion_data *args, size_t nargs, //
                        polyregion_data *rtn                       //
) {

  static_assert(sizeof(uint8_t) == sizeof(char));

  const auto toFFITpe = [&](const polyregion_type &tpe) -> ffi_type * {
    switch (tpe) {
    case Bool:
      return &ffi_type_sint8;
    case Byte:
      return &ffi_type_sint8;
    case Char:
      return &ffi_type_uint8;
    case Short:
      return &ffi_type_sint16;
    case Int:
      return &ffi_type_sint32;
    case Long:
      return &ffi_type_sint64;
    case Float:
      return &ffi_type_float;
    case Double:
      return &ffi_type_double;
    case Ptr:
      return &ffi_type_pointer;
    case Void:
      return &ffi_type_void;
    default:
      return nullptr;
    }
  };

  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(*object->file);

  auto sym = ld.getSymbol(symbol);

  if (!sym && (object->file->isMachO() || object->file->isMachOUniversalBinary())) {
    // Mach-O has a leading underscore for all exported symbols, we try again with that prepended
    sym = ld.getSymbol(std::string("_") + symbol);
  }

  if (!sym) {
    auto table = ld.getSymbolTable();
    auto symbols = polyregion::mk_string2<llvm::StringRef, llvm::JITEvaluatedSymbol>(
        table, [](auto &x) { return "[`" + x.first.str() + "`@" + polyregion::hex(x.second.getAddress()) + "]"; }, ",");
    return error("Symbol `" + std::string(symbol) + "` not found in the given object, available symbols (" +
                 std::to_string(table.size()) + ") = " + symbols);
  }

  ld.finalizeWithMemoryManagerLocking();

  if (ld.hasError()) {
    return error("Symbol `" + std::string(symbol) + "` failed to finalise for execution: " + ld.getErrorString().str());
  }

  auto rtnFFIType = toFFITpe(rtn->type);
  if (!rtnFFIType) {
    return error("Illegal return type " + std::to_string(rtn->type));
  }

  std::vector<void *> argPointers(nargs);
  std::vector<ffi_type *> argsFFIType(nargs);
  for (size_t i = 0; i < nargs; i++) {
    argPointers[i] = args[i].ptr;
    argsFFIType[i] = toFFITpe(args[i].type);
    if (!argsFFIType[i])
      return error("Illegal parameter type on arg " + std::to_string(i) + ": " + std::to_string(args[i].type));
  }

  ffi_cif cif{};
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI, nargs, rtnFFIType, argsFFIType.data());
  switch (status) {
  case FFI_OK:
    ffi_call(&cif, FFI_FN(sym.getAddress()), rtn->ptr, argPointers.data());
    return nullptr;
  case FFI_BAD_TYPEDEF:
    return error("ffi_prep_cif: FFI_BAD_TYPEDEF");
  case FFI_BAD_ABI:
    return error("ffi_prep_cif: FFI_BAD_ABI");
  default:
    return error("ffi_prep_cif: unknown error (" + std::to_string(status) + ")");
  }
}

#include <bitset>
#include <fstream>
#include <iostream>

#include "ffi.h"
#include "runtime.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

void polyregion_consume_error(char *err) {
  if (err) std::free(err);
}

char *polyregion_invoke(const uint8_t *object, size_t object_size, //
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
      return &ffi_type_sint8;
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

  const auto error = [](const std::string &s) -> char * { return strdup(s.c_str()); };

  llvm::MemoryBufferRef buffer(llvm::StringRef(reinterpret_cast<const char *>(object), object_size), "");

  auto obj = llvm::object::ObjectFile::createObjectFile(buffer);
  if (auto e = obj.takeError()) {
    return error(toString(std::move(e)));
  }

  llvm::SectionMemoryManager mm;
  llvm::RuntimeDyld ld(mm, mm);
  ld.loadObject(**obj);

  auto sym = ld.getSymbol(symbol);
  if (!sym) {
    return error("Symbol `" + std::string(symbol) + "` not found in the given object of size " +
                 std::to_string(object_size));
  }

  ld.finalizeWithMemoryManagerLocking();

  if (ld.hasError()) {
    return error("Symbol `" + std::string(symbol) + "` failed to finalise for execution: " + ld.getErrorString().str());
  }

  std::cout << "[" << symbol << "] = " << sym.getAddress() << std::endl;

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

int main(int argc, char*argv[]) {

  std::string path = "/home/tom/polyregion/native/the_obj2.o";
  //  std::string path = "/home/tom/Desktop/prime.o";
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<uint8_t> xs(len / sizeof(uint8_t));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();

  int u;
  polyregion_data rtn{.type = polyregion_type::Void, .ptr = &u};

  std::vector<float> data = {1.1, 2.1, 3.1};
  auto ptr = data.data();
  polyregion_data arg1{.type = polyregion_type::Ptr, .ptr = &ptr}; // XXX pointer to T, so ** for pointers

  auto err = polyregion_invoke(xs.data(), xs.size(), "lambda", &arg1, 1, &rtn);

  //  int exp = 0;
  //  int in = 99;
  //  polyregion_data arg1{.type = polyregion_type::Int, .ptr = &in};
  //  polyregion_data rtn{.type = polyregion_type::Int, .ptr = &exp};
  //  polyregion_invoke(xs.data(), xs.size(), "lambda", &arg1, 1, &rtn, &err);

  auto mk = [](float f) {
    long long unsigned int f_as_int = 0;
    std::memcpy(&f_as_int, &f, sizeof(float));            // 2.
    std::bitset<8 * sizeof(float)> f_as_bitset{f_as_int}; // 3.
    return f_as_bitset;
  };

  if (err) {
    std::cerr << "Err:" << err << std::endl;
  } else {

    std::bitset<32> act1 = mk(data[0]);
    std::bitset<32> act2 = mk(data[1]);

    float d = 1;
    std::bitset<32> exp = mk(d);
    std::cout << exp << '\n';
    std::cout << act1 << '\n';
    std::cout << act2 << '\n';

    std::cout << "r=" << data[0] << std::endl;
    std::cout << "r=" << data[1] << std::endl;
    std::cout << "r=" << data[2] << std::endl;
    std::cout << "OK" << err << std::endl;
  }
  polyregion_consume_error(err);
}
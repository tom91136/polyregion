#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>

namespace polyregion::runtime {

enum class EXPORT Type : uint8_t {
  Void = 1,
  Bool8,
  Byte8,
  CharU16,
  Short16,
  Int32,
  Long64,
  Float32,
  Double64,
  Ptr,
};

static constexpr size_t byteOfType(Type t) {
  switch (t) {
    case Type::Void: return 0;
    case Type::Bool8:
    case Type::Byte8: return 8 / 8;
    case Type::CharU16:
    case Type::Short16: return 16 / 8;
    case Type::Int32: return 32 / 8;
    case Type::Long64: return 64 / 8;
    case Type::Float32: return 32 / 8;
    case Type::Double64: return 64 / 8;
    case Type::Ptr: static_assert(sizeof(ptrdiff_t) == 8); return 64 / 8;
  }
}

static constexpr const char * typeName(Type t) {
  switch (t) {
    case Type::Void: return "Void";
    case Type::Bool8: return "Bool8";
    case Type::Byte8: return "Byte8";
    case Type::CharU16: return "Char";
    case Type::Short16: return "Short16";
    case Type::Int32: return "Int32";
    case Type::Long64: return "Long64";
    case Type::Float32: return "Float32";
    case Type::Double64: return "Double64";
    case Type::Ptr:return "Ptr";
  }
}

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime
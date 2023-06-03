#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace polyregion::runtime {

enum class POLYREGION_EXPORT Type : uint8_t {
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
  Scratch,
};

static constexpr POLYREGION_EXPORT size_t byteOfType(Type t) {
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
    case Type::Scratch:
    case Type::Ptr: return sizeof(void *);
  }
}

static constexpr POLYREGION_EXPORT const char *typeName(Type t) {
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
    case Type::Ptr: return "Ptr";
    case Type::Scratch: return "Scratch";
  }
}

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime
#pragma once

#include <cstdint>
#include <utility>

namespace polyregion::runtime {

enum class EXPORT Type : uint8_t {
  Void = 1,
  Bool8 ,
  Byte8,
  CharU16,
  Short16,
  Int32,
  Long64,
  Float32,
  Double64,
  Ptr,
};

using TypedPointer = std::pair<Type, void *>;

} // namespace polyregion::runtime
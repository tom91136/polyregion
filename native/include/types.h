#pragma once

#include <cstdint>

namespace polyregion::runtime {

enum class EXPORT Type : uint8_t {
  Bool = 1,
  Byte,
  Char,
  Short,
  Int,
  Long,
  Float,
  Double,
  Ptr,
  Void,
};

}
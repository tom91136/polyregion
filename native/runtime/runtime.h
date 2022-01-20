#pragma once

#include "export.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

using TypedPointer = std::pair<Type, void *>;

struct Data;

class EXPORT Object {
  std::unique_ptr<Data> data;

public:
  virtual ~Object();
  EXPORT explicit Object(const std::vector<uint8_t> &object);
  EXPORT std::vector<std::pair<std::string, uint64_t>> enumerate();
  EXPORT void invoke(const std::string &symbol, const std::vector<TypedPointer> &args, TypedPointer rtn);
};

} // namespace polyregion::runtime
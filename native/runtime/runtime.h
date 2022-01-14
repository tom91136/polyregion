#pragma once

#include "export.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace polyregion::runtime {

enum class EXPORT Type {
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

struct ObjectRef;

using TypedPointer = std::pair<Type, void *>;

EXPORT std::unique_ptr<ObjectRef> loadObject(const std::vector<uint8_t> &object); //

EXPORT std::vector<std::pair<std::string, uint64_t>> enumerate(const ObjectRef &object); //

EXPORT std::optional<std::string> invoke(const ObjectRef &object,        //
                                         const std::string &symbol,      //
                                         const std::vector<TypedPointer> &args, //
                                         TypedPointer rtn                //

);

} // namespace polyregion::runtime
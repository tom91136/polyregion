#pragma once

#include "export.h"
#include "types.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace polyregion::runtime {

using TypedPointer = std::pair<Type, void *>;

struct Data;

EXPORT void init();

EXPORT void run();

EXPORT static void *_malloc(size_t size);


class EXPORT Object {
  std::unique_ptr<Data> data;

public:
  virtual ~Object();
  EXPORT explicit Object(const std::vector<uint8_t> &object);
  [[nodiscard]] EXPORT std::vector<std::pair<std::string, uint64_t>> enumerate() const;
  EXPORT void invoke(const std::string &symbol,                  //
                     const std::function<void *(size_t)> &alloc, //
                     const std::vector<TypedPointer> &args,      //
                     TypedPointer rtn                            //
  ) const;
};

} // namespace polyregion::runtime
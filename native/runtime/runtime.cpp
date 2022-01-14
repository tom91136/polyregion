#include "runtime.h"

using namespace polyregion;
struct runtime::ObjectRef {};

std::unique_ptr<runtime::ObjectRef> runtime::loadObject(const std::vector<uint8_t> &object) {
  return std::unique_ptr<ObjectRef>();
}
std::vector<std::pair<std::string, uint64_t>> runtime::enumerate(const runtime::ObjectRef &object) {
  return std::vector<std::pair<std::string, uint64_t>>();
}
std::optional<std::string> runtime::invoke(const runtime::ObjectRef &object, const std::string &symbol,
                                           const std::vector<TypedPointer> &args, runtime::TypedPointer rtn) {
  return std::optional<std::string>();
}

#include "polyregion_runtime.h"
#include "runtime.h"
#include "utils.hpp"

using namespace polyregion;

static_assert(                                //
    std::is_same_v<                           //
        decltype(polyregion_type::ordinal),   //
        std::underlying_type_t<runtime::Type> //
        >);

const polyregion_type POLYREGION_BOOL = {to_underlying(runtime::Type::Bool)};
const polyregion_type POLYREGION_BYTE = {to_underlying(runtime::Type::Byte)};
const polyregion_type POLYREGION_CHAR = {to_underlying(runtime::Type::Char)};
const polyregion_type POLYREGION_SHORT = {to_underlying(runtime::Type::Short)};
const polyregion_type POLYREGION_INT = {to_underlying(runtime::Type::Int)};
const polyregion_type POLYREGION_LONG = {to_underlying(runtime::Type::Long)};
const polyregion_type POLYREGION_FLOAT = {to_underlying(runtime::Type::Float)};
const polyregion_type POLYREGION_DOUBLE = {to_underlying(runtime::Type::Double)};
const polyregion_type POLYREGION_PTR = {to_underlying(runtime::Type::Ptr)};
const polyregion_type POLYREGION_VOID = {to_underlying(runtime::Type::Void)};

struct polyregion_object {
  std::unique_ptr<runtime::Object> data;
  explicit polyregion_object(std::unique_ptr<runtime::Object> data) : data(std::move(data)) {}
};

void polyregion_release_object(polyregion_object_ref *ref) {
  if (ref) {
    polyregion::free_str(ref->message);
    delete ref->object;
    delete ref;
  }
}

polyregion_object_ref *polyregion_load_object(const uint8_t *object, size_t object_size) {
  auto ref = new polyregion_object_ref{};
  try {
    ref->object =
        new polyregion_object(std::make_unique<runtime::Object>(std::vector<uint8_t>(object, object + object_size)));
  } catch (const std::exception &e) {
    ref->message = new_str(e.what());
  }
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
  auto table = object->data->enumerate();
  auto xs = new polyregion_symbol[table.size()];
  std::transform(table.begin(), table.end(), xs, [](auto &p) {
    // copy name here as the symbol table is deleted with dyld
    return polyregion_symbol{polyregion::new_str(p.first), p.second};
  });
  return new polyregion_symbol_table{xs, table.size()};
}

void polyregion_release_invoke(char *err) { polyregion::free_str(err); }

char *polyregion_invoke(const polyregion_object *object,
                        const char *symbol,                        //
                        const polyregion_data *args, size_t nargs, //
                        polyregion_data *rtn                       //
) {

  auto toTyped = [](const auto &data) -> runtime::TypedPointer {
    return std::make_pair(static_cast<runtime::Type>(data.type.ordinal), data.ptr);
  };

  std::vector<runtime::TypedPointer> typedArgs(nargs);
  std::transform(args, args + nargs, typedArgs.begin(), toTyped);

  try {
    object->data->invoke(symbol, typedArgs, toTyped(*rtn));
    return nullptr;
  } catch (const std::exception &e) {
    return new_str(e.what());
  }
}

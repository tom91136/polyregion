#pragma once

#include <cstddef>
#include <cstdint>

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum EXPORT {
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
} polyregion_type;

typedef struct EXPORT {
  polyregion_type type;
  void *ptr;
} polyregion_data;

typedef struct EXPORT {
  char *name;
  uint64_t address;
} polyregion_symbol;

typedef struct EXPORT {
  polyregion_symbol *symbols;
  size_t size;
} polyregion_symbol_table;

struct EXPORT polyregion_object;
typedef struct EXPORT {
  polyregion_object *object;
  char *message;
} polyregion_object_ref;

EXPORT void polyregion_release_object(polyregion_object_ref *object);                            //
EXPORT polyregion_object_ref *polyregion_load_object(const uint8_t *object, size_t object_size); //

EXPORT void polyregion_release_enumerate(polyregion_symbol_table *symbols);            //
EXPORT polyregion_symbol_table *polyregion_enumerate(const polyregion_object *object); //

EXPORT void polyregion_release_invoke(char *err);
EXPORT char *polyregion_invoke(const polyregion_object *object,
                               const char *symbol,                        //
                               const polyregion_data *args, size_t nargs, //
                               polyregion_data *rtn                       //

);

#ifdef __cplusplus
}
#endif

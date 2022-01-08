#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  Bool,
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

typedef struct {
  polyregion_type type;
  void *ptr;
} polyregion_data;

typedef struct {
  char *name;
  uint64_t address;
} polyregion_symbol;

typedef struct {
  polyregion_symbol *symbols;
  size_t size;
} polyregion_symbol_table;

struct polyregion_object;
struct polyregion_object_ref {
  polyregion_object *object;
    char *message;
};

void polyregion_release_object(polyregion_object_ref *object);                            //
polyregion_object_ref *polyregion_load_object(const uint8_t *object, size_t object_size); //

void polyregion_release_enumerate(polyregion_symbol_table *symbols);            //
polyregion_symbol_table *polyregion_enumerate(const polyregion_object *object); //

void polyregion_release_invoke(char *err);
char *polyregion_invoke(const polyregion_object *object,
                        const char *symbol,                        //
                        const polyregion_data *args, size_t nargs, //
                        polyregion_data *rtn                       //

);

#ifdef __cplusplus
}
#endif

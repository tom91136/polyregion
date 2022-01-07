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

void polyregion_consume_error(char *err);


char *polyregion_invoke(const uint8_t *object, size_t object_size, //
                        const char *symbol,                        //
                        const polyregion_data *args, size_t nargs, //
                        polyregion_data *rtn                       //

);


#ifdef __cplusplus
}
#endif

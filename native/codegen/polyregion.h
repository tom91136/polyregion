#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { Invalid = 0, LLVM, OpenCL } Backend;

typedef struct {
  uint8_t *data;
  size_t size;
} polyregion_buffer;

typedef struct {
  polyregion_buffer program;
  char *disassembly;
} polyregion_program;

void polyregion_initialise();

polyregion_program *polyregion_compile(polyregion_buffer *polyast);

void polyregion_release(polyregion_program *buffer);

#ifdef __cplusplus
}
#endif
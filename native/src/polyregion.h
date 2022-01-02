#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { Invalid = 0, LLVM, OpenCL } Backend;

typedef struct {
  size_t size;
  uint8_t *data;
} polyregion_buffer;

typedef struct {
  polyregion_buffer program;
  char *disassembly;
} polyregion_program;

void polyregion_initialise();

polyregion_program *polyregion_compile(polyregion_buffer *polyast_proto);

void polyregion_release(polyregion_program *buffer);

#ifdef __cplusplus
}
#endif
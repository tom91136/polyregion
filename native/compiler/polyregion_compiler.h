#pragma once

#include <cstddef>
#include <cstdint>

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EXPORT {
  uint8_t ordinal;
} polyregion_backend;

EXPORT extern const polyregion_backend POLYREGION_BACKEND_LLVM;
EXPORT extern const polyregion_backend POLYREGION_BACKEND_OPENCL;
EXPORT extern const polyregion_backend POLYREGION_BACKEND_CUDA;

typedef struct EXPORT {
  uint8_t *data;
  size_t size;
} polyregion_buffer;

typedef struct EXPORT {
  int64_t epochMillis;
  char *name;
  int64_t elapsedNanos;
} polyregion_event;

typedef struct EXPORT {
  polyregion_buffer program;
  char *disassembly;
  char *messages;
  polyregion_event *elapsed;
  size_t elapsed_size;
} polyregion_compilation;

EXPORT void polyregion_initialise();

EXPORT polyregion_compilation *polyregion_compile( //
    const polyregion_buffer *ast,                  //
    bool emitDisassembly,                          //
    polyregion_backend backend                     //
);

EXPORT void polyregion_release_compile(polyregion_compilation *buffer);

#ifdef __cplusplus
}
#endif
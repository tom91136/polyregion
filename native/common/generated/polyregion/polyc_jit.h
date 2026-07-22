// AUTO-GENERATED from PolyAST.PolyJitAbi via polyregion.ast.CodeGen. DO NOT EDIT.

#ifndef POLYREGION_POLYC_JIT_H
#define POLYREGION_POLYC_JIT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(POLYC_JIT_BUILD)
  #define POLYC_JIT_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
  #define POLYC_JIT_EXPORT
#else
  #define POLYC_JIT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define POLYC_JIT_ABI_VERSION 2u

typedef enum polyc_jit_status { POLYC_JIT_OK = 0, POLYC_JIT_FAILED = 1 } polyc_jit_status_t;

typedef struct polyc_jit_spec_const {
  const char *field;
  const char *repr;
  const uint8_t *data;
  size_t dataLen;
} polyc_jit_spec_const_t;

/**
 * Compile a msgpack Program. Free the result with polyc_jit_free.
 */
POLYC_JIT_EXPORT polyc_jit_status_t polyc_jit_compile(const uint8_t *program, size_t programLen, uint32_t target, const char *arch,
                                                      const char *pipelineSpec, uint32_t opt, const polyc_jit_spec_const_t *specialise,
                                                      size_t specialiseLen, uint8_t **out, size_t *outLen);

/**
 * NUL-terminated diagnostic for the most recent non-Ok status; valid until the next polyc_jit_compile call, NULL when none.
 */
POLYC_JIT_EXPORT const char *polyc_jit_last_error(void);

/**
 * Release a buffer returned by polyc_jit_compile.
 */
POLYC_JIT_EXPORT void polyc_jit_free(void *ptr);

typedef polyc_jit_status_t (*polyc_jit_compile_fn)(const uint8_t *, size_t, uint32_t, const char *, const char *, uint32_t,
                                                   const polyc_jit_spec_const_t *, size_t, uint8_t **, size_t *);
typedef const char *(*polyc_jit_last_error_fn)(void);
typedef void (*polyc_jit_free_fn)(void *);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // POLYREGION_POLYC_JIT_H

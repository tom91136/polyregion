// AUTO-GENERATED from PolyAST.PolyPassAbi via polyregion.ast.CodeGen. DO NOT EDIT.

#ifndef POLYREGION_POLYPASS_H
#define POLYREGION_POLYPASS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(POLYPASS_BUILD)
  #define POLYPASS_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
  #define POLYPASS_EXPORT
#else
  #define POLYPASS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define POLYPASS_ABI_VERSION 1u
#define POLYPASS_ENV_PLUGINS "POLYPASS_PLUGINS"

typedef enum polypass_status {
  POLYPASS_OK = 0,
  POLYPASS_ALLOC_FAILED = 1,
  POLYPASS_PIPELINE_ERROR = 2,
  POLYPASS_UNKNOWN_PASS = 3,
  POLYPASS_ABI_MISMATCH = 4
} polypass_status_t;

/**
 * ABI version the plugin was built against; polyc refuses mismatched plugins.
 */
POLYPASS_EXPORT uint32_t polypass_abi_version(void);

/**
 * Number of passes this plugin contributes.
 */
POLYPASS_EXPORT size_t polypass_pass_count(void);

/**
 * Bare identifier of the i-th pass (e.g. "FullOpt"). Process-lifetime; NULL if i out of range.
 */
POLYPASS_EXPORT const char * polypass_pass_name(size_t i);

/**
 * Optional human-readable description of the i-th pass; may return NULL or "".
 */
POLYPASS_EXPORT const char * polypass_pass_descr(size_t i);

/**
 * Run the NULL-terminated `steps` list against `in` (msgpack Program); steps share in-process state. On Ok, *out is a malloc'd PassRunResult; caller frees via polypass_free.
 */
POLYPASS_EXPORT polypass_status_t polypass_run_passes(const char *const * steps, const uint8_t * in, size_t inLen, uint8_t ** out, size_t * outLen);

/**
 * NUL-terminated diagnostic for the most recent non-Ok status. Valid until the next polypass_run_passes call; NULL when no error is set.
 */
POLYPASS_EXPORT const char * polypass_last_error(void);

/**
 * Release a buffer returned by polypass_run_passes.
 */
POLYPASS_EXPORT void polypass_free(void * ptr);

typedef uint32_t (*polypass_abi_version_fn)(void);
typedef size_t (*polypass_pass_count_fn)(void);
typedef const char * (*polypass_pass_name_fn)(size_t);
typedef const char * (*polypass_pass_descr_fn)(size_t);
typedef polypass_status_t (*polypass_run_passes_fn)(const char *const *, const uint8_t *, size_t, uint8_t **, size_t *);
typedef const char * (*polypass_last_error_fn)(void);
typedef void (*polypass_free_fn)(void *);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // POLYREGION_POLYPASS_H

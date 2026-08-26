// AUTO-GENERATED from PolyAST.PolyPackageAbi via polyregion.ast.CodeGen. DO NOT EDIT.

#ifndef POLYREGION_POLYPACKAGE_H
#define POLYREGION_POLYPACKAGE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(POLYPACKAGE_BUILD)
  #define POLYPACKAGE_EXPORT __declspec(dllexport)
#elif defined(_WIN32)
  #define POLYPACKAGE_EXPORT
#else
  #define POLYPACKAGE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define POLYPACKAGE_ABI_VERSION 2u

typedef enum polypackage_status {
  POLYPACKAGE_OK = 0,
  POLYPACKAGE_ALLOC_FAILED = 1,
  POLYPACKAGE_INVALID = 2,
  POLYPACKAGE_ABI_MISMATCH = 3
} polypackage_status_t;

/**
 * ABI version of the non-composable package service.
 */
POLYPACKAGE_EXPORT uint32_t polypackage_abi_version(void);

/**
 * Link a versioned PackageLinkRequest into a Package encoded with the package-service wire schema.
 */
POLYPACKAGE_EXPORT polypackage_status_t polypackage_link_package(const uint8_t *request, size_t requestLen, uint8_t **out, size_t *outLen);

/**
 * Link a versioned ProgramLinkRequest into a Program encoded with the package-service wire schema.
 */
POLYPACKAGE_EXPORT polypackage_status_t polypackage_link_program(const uint8_t *request, size_t requestLen, uint8_t **out, size_t *outLen);

/**
 * NUL-terminated diagnostic for the most recent failed package operation; NULL when no error is set.
 */
POLYPACKAGE_EXPORT const char *polypackage_last_error(void);

/**
 * Release a buffer returned by a package operation.
 */
POLYPACKAGE_EXPORT void polypackage_free(void *ptr);

typedef uint32_t (*polypackage_abi_version_fn)(void);
typedef polypackage_status_t (*polypackage_link_package_fn)(const uint8_t *, size_t, uint8_t **, size_t *);
typedef polypackage_status_t (*polypackage_link_program_fn)(const uint8_t *, size_t, uint8_t **, size_t *);
typedef const char *(*polypackage_last_error_fn)(void);
typedef void (*polypackage_free_fn)(void *);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // POLYREGION_POLYPACKAGE_H

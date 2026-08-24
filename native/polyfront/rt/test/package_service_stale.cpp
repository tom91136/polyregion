#include <cstdlib>

#include "polyregion/polypackage.h"

extern "C" {

uint32_t polypackage_abi_version() { return 0; }

polypackage_status_t polypackage_link_package(const uint8_t *, size_t, uint8_t **, size_t *) { return POLYPACKAGE_ABI_MISMATCH; }

polypackage_status_t polypackage_resolve_sym(const uint8_t *, size_t, uint8_t **, size_t *) { return POLYPACKAGE_ABI_MISMATCH; }

const char *polypackage_last_error() { return "stale package service"; }

void polypackage_free(void *ptr) { std::free(ptr); }

} // extern "C"

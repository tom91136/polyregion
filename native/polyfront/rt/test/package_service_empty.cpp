#include <cstdlib>

#include "polyregion/polypackage.h"

extern "C" {

uint32_t polypackage_abi_version() { return POLYPACKAGE_ABI_VERSION; }

polypackage_status_t polypackage_link_package(const uint8_t *, size_t, uint8_t **out, size_t *outSize) {
  *out = nullptr;
  *outSize = 0;
  return POLYPACKAGE_OK;
}

polypackage_status_t polypackage_resolve_sym(const uint8_t *, size_t, uint8_t **out, size_t *outSize) {
  *out = nullptr;
  *outSize = 0;
  return POLYPACKAGE_OK;
}

const char *polypackage_last_error() { return nullptr; }

void polypackage_free(void *ptr) { std::free(ptr); }

} // extern "C"

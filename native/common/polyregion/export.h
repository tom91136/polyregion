#pragma once

#include "compat.h"

#if _MSC_VER
  #define POLYREGION_EXPORT
#else
  #define POLYREGION_EXPORT __attribute__((visibility("default")))
#endif

#pragma once

// Aggregator: malloc/free interposers (per-TU by design) + declarations of `_rt_*`. The
// definitions live in libpolyreflect-rt; see rt_impl.cpp.
#include "rt_memory.hpp"
#include "rt_reflect.hpp"

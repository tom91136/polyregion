#pragma once

#include "ffi.h"

// When libffi is configured for X86, the header defines the macro X86/X86_64, this breaks anything with this name (e.g.
// LLVM's X86 namespace)

#undef X86_64
#undef X86

// Same thing for ARM/aarch64

#undef ARM64
#undef ARM

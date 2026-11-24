#pragma once

// LLVM and lld are uninstrumented, so their inline Allocate never unpoisons slabs that an instrumented TU poisons here.
// Keep ASan instrumentation on our code; drop only this manual poisoning from the shared allocator definitions.
#include <llvm/Support/Compiler.h>

#define __asan_poison_memory_region(p, size) ((void)0)
#define __asan_unpoison_memory_region(p, size) ((void)0)
#include_next <llvm/Support/Allocator.h>
#undef __asan_poison_memory_region
#undef __asan_unpoison_memory_region
